"""Self-describing figures: images that carry their own plotting code and data.

The idea (kept from the original ``plot_utils.py``) is simple: right after you
save a figure, the decorator stashes everything needed to reproduce it -- the
function source, its keyword arguments, and some provenance -- inside the image
file itself.  Months later you can recover the data without remembering which
notebook or script produced the picture.

What this rewrite fixes
-----------------------
* The original always read ``img.info['exif'][6:]``.  That 6-byte ``Exif\\x00\\x00``
  prefix is a **JPEG** detail; PNG stores metadata differently, so the payload
  was silently mis-read for PNGs.  We now use PNG ``iTXt`` chunks for PNG (no
  re-encoding, lossless) and EXIF ``UserComment`` for JPEG/WEBP, and fall back
  to a ``.meta`` sidecar when a format cannot carry metadata.
* The old backup file was deleted *before* the new one was written, so a failed
  save destroyed the figure.  Embedding now writes to a temporary file and
  atomically replaces the original.
* ``tmp-<hash(file)>./imgs/a.png`` produced invalid paths when re-rendering.
  Re-rendering now writes a proper unique file next to the source image.
* ``hash()`` is salted per process; the payload is now checksummed with SHA-256
  and tagged with a magic header + format version.
* Arguments that cannot be pickled no longer crash the plot: their values are
  replaced by ``repr`` and reported in ``bundle.warnings``.
* ``retrieve_plot`` executes code embedded in the file, which is a real code
  execution risk; it now requires ``allow_execute=True`` explicitly.

Dependencies: ``Pillow`` (required), ``dill`` (optional, enables pickling of
lambdas/closures).  Read-only helpers (`retrieve_data`, `retrieve_code`,
`unpack`) never execute embedded code.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import logging
import os
import pickle
import struct
import textwrap
import zlib
from dataclasses import dataclass, field
from datetime import timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping

from .naming import short_id, unique_path, utcnow

__all__ = [
    "PlotBundle",
    "packplot",
    "unpack",
    "retrieve_data",
    "retrieve_code",
    "retrieve_plot",
    "embed_metadata",
    "extract_metadata",
]

#: metadata key used in PNG text chunks
KEY = "dlt:plot"
#: magic bytes at the start of every payload
MAGIC = b"DLTP"
#: payload format version
FORMAT_VERSION = 1
#: EXIF tag for UserComment
_EXIF_USER_COMMENT = 0x9286
#: conservative ceiling for a JPEG APP1 EXIF segment
_JPEG_EXIF_LIMIT = 60_000
#: payloads larger than this are stored in a sidecar instead
_MAX_EMBED_BYTES = _JPEG_EXIF_LIMIT

_LOG = logging.getLogger("dlt.plot_utils")


@dataclass
class PlotBundle:
    """Everything needed to understand/reproduce one figure."""

    code: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    module: str | None = None
    file: str | None = None
    created_at: str = field(default_factory=lambda: utcnow().isoformat())
    warnings: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def render(self, **overrides: Any) -> Any:
        """Unsafe: execute ``code`` and call the plotting function."""
        return _render(self, **overrides)


# --------------------------------------------------------------------------- #
# serialisation
# --------------------------------------------------------------------------- #
def _dumps(bundle: PlotBundle) -> tuple[bytes, bytes]:
    """Return ``(payload, pickle_lib_tag)`` where tag is b'p' or b'd'."""
    try:
        return pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL), b"p"
    except Exception:
        try:
            import dill

            return dill.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL), b"d"
        except Exception as exc:
            raise TypeError(f"could not serialise plotting metadata: {exc}") from exc


def _loads(raw: bytes, tag: bytes) -> PlotBundle:
    if tag == b"d":
        try:
            import dill

            return dill.loads(raw)
        except ImportError:  # pragma: no cover
            pass
    return pickle.loads(raw)


def _pack(bundle: PlotBundle, *, compress: bool = True) -> bytes:
    raw, tag = _dumps(bundle)
    body = zlib.compress(raw, 9) if compress else raw
    digest = hashlib.sha256(raw).digest()[:16]
    header = MAGIC + bytes([FORMAT_VERSION, tag[0], 1 if compress else 0])
    header += struct.pack("<I", len(raw)) + digest
    return header + body


def _unpack(blob: bytes) -> PlotBundle:
    if len(blob) < len(MAGIC) + 23 or not blob.startswith(MAGIC):
        raise ValueError("not a dlt plot payload (bad magic)")
    offset = len(MAGIC)
    version = blob[offset]
    tag = bytes([blob[offset + 1]])
    compressed = bool(blob[offset + 2])
    raw_len = struct.unpack("<I", blob[offset + 3 : offset + 7])[0]
    digest = blob[offset + 7 : offset + 23]
    body = blob[offset + 23 :]
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported payload version {version}")
    try:
        raw = zlib.decompress(body) if compressed else body
    except Exception as exc:
        raise ValueError(f"plot payload is corrupted: {exc}") from exc
    if len(raw) != raw_len or hashlib.sha256(raw).digest()[:16] != digest:
        raise ValueError("plot payload is corrupted (checksum mismatch)")
    try:
        bundle = _loads(raw, tag)
    except Exception as exc:
        raise ValueError(f"could not deserialize plot payload: {exc}") from exc
    if not isinstance(bundle, PlotBundle):  # pragma: no cover - defensive
        raise ValueError("unexpected payload type")
    return bundle


# --------------------------------------------------------------------------- #
# embedding into image files
# --------------------------------------------------------------------------- #
def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta")


def _supports_embedding(path: Path) -> bool:
    """True when we can store metadata inside the image container itself."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            return (img.format or "").upper() in ("PNG", "JPEG", "JPG", "MPO", "WEBP")
    except Exception:
        return False


def embed_metadata(path: str | os.PathLike[str], blob: bytes) -> str:
    """Store ``blob`` inside the image at ``path``; returns the method used.

    Methods: ``"png-itxt"``, ``"jpeg-exif"``, ``"webp-exif"`` or ``"sidecar"``.
    The original image is replaced atomically.
    """
    from PIL import Image, PngImagePlugin

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    tmp = path.with_name(path.name + f".dlt-tmp-{short_id(6)}")
    try:
        if len(blob) > _MAX_EMBED_BYTES and path.suffix.lower() not in (".png",):
            _LOG.warning(
                "payload for %s is %d bytes (> %d); writing %s.meta instead",
                path,
                len(blob),
                _MAX_EMBED_BYTES,
                path.name,
            )
            _sidecar_path(path).write_bytes(blob)
            return "sidecar"

        with Image.open(path) as img:
            fmt = (img.format or "").upper()
            img.load()
            if fmt == "PNG":
                info = PngImagePlugin.PngInfo()
                info.add_itxt(KEY, base64.b64encode(blob).decode("ascii"))
                img.save(tmp, format="PNG", pnginfo=info)
                method = "png-itxt"
            elif fmt in ("JPEG", "JPG", "MPO"):
                exif = img.getexif()
                exif[_EXIF_USER_COMMENT] = blob
                img.save(
                    tmp,
                    format="JPEG",
                    exif=exif.tobytes(),
                    quality="keep",
                    subsampling="keep",
                )
                method = "jpeg-exif"
            elif fmt == "WEBP":
                exif = img.getexif()
                exif[_EXIF_USER_COMMENT] = blob
                img.save(tmp, format="WEBP", exif=exif.tobytes(), quality=False)
                method = "webp-exif"
            else:
                _sidecar_path(path).write_bytes(blob)
                return "sidecar"
        os.replace(tmp, path)
        return method
    except Exception:
        # Last resort: never lose the payload, and never corrupt the image.
        _LOG.warning("could not embed metadata in %s; writing sidecar instead", path, exc_info=True)
        _sidecar_path(path).write_bytes(blob)
        return "sidecar"
    finally:
        if tmp.exists():
            try:
                os.remove(tmp)
            except OSError:
                pass


def extract_metadata(path: str | os.PathLike[str]) -> bytes:
    """Return the raw payload stored in/next to ``path`` (without unpickling)."""
    from PIL import Image

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with Image.open(path) as img:
        img.load()
        text = img.info.get(KEY)
        if text:
            try:
                return base64.b64decode(text)
            except Exception:
                pass
        try:
            value = img.getexif().get(_EXIF_USER_COMMENT)
        except Exception:
            value = None
        if isinstance(value, (bytes, bytearray)) and bytes(value).startswith(MAGIC):
            return bytes(value)
        # Some Pillow versions expose the raw EXIF blob instead.
        raw_exif = img.info.get("exif")
        if isinstance(raw_exif, (bytes, bytearray)):
            idx = bytes(raw_exif).find(MAGIC)
            if idx >= 0:
                return bytes(raw_exif)[idx:]

    sidecar = _sidecar_path(path)
    if sidecar.is_file():
        return sidecar.read_bytes()
    raise ValueError(f"no dlt plot metadata found in {path}")


# --------------------------------------------------------------------------- #
# the decorator
# --------------------------------------------------------------------------- #
def _clean_source(source: str) -> str:
    source = textwrap.dedent(source)
    lines = source.splitlines()
    index = 0
    while index < len(lines) and (not lines[index].strip() or lines[index].lstrip().startswith("@")):
        index += 1
    return "\n".join(lines[index:])


def _make_picklable(args: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Replace unpicklable argument values and report what was lost."""
    clean: dict[str, Any] = {}
    warnings: list[str] = []
    for key, value in args.items():
        try:
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            clean[key] = value
        except Exception:
            try:
                import dill

                dill.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                clean[key] = value
            except Exception:
                clean[key] = f"<unpicklable {type(value).__name__}>"
                warnings.append(f"argument {key!r} was not picklable; stored a placeholder")
    return clean, warnings


def packplot(
    func: Callable[..., Any] | None = None,
    *,
    extra: Mapping[str, Any] | None = None,
) -> Callable[..., Any]:
    """Decorator: after the wrapped function saves ``filename``, embed its recipe.

    Usable as ``@packplot`` or ``@packplot()``.  The wrapped function **must**
    accept a ``filename`` argument (positional or keyword) whose value is the
    image it writes.
    """

    def decorate(plotf: Callable[..., Any]) -> Callable[..., Any]:
        import inspect

        signature = inspect.signature(plotf)

        @wraps(plotf)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            if "filename" not in bound.arguments:
                raise TypeError(
                    f"{plotf.__name__}() must accept a 'filename' argument for @packplot"
                )
            filename = Path(os.fspath(bound.arguments["filename"]))
            bound.arguments["filename"] = filename

            # Keep matplotlib's font cache inside the project (./cache) so the
            # first import inside the plotting function does not litter $HOME.
            os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), "cache", "matplotlib"))

            result = plotf(*args, **kwargs)

            if not filename.is_file():
                raise FileNotFoundError(
                    f"{plotf.__name__} did not create {filename}; "
                    "@packplot needs the plotted file to be saved before returning"
                )

            try:
                code = _clean_source(inspect.getsource(plotf))
            except (OSError, TypeError):
                code = ""
            try:
                source_file = inspect.getsourcefile(plotf)
            except TypeError:
                source_file = None
            clean_args, warnings = _make_picklable(dict(bound.arguments))
            if not code:
                warnings.append("source code could not be captured (interactive session?)")

            bundle = PlotBundle(
                code=code,
                name=plotf.__name__,
                args=clean_args,
                module=getattr(plotf, "__module__", None),
                file=source_file,
                warnings=warnings,
                extra=dict(extra or {}),
            )
            if not _supports_embedding(filename):
                bundle.warnings.append(
                    f"{filename.suffix or 'this format'} cannot embed metadata; "
                    f"wrote {filename.name}.meta instead"
                )
            payload = _pack(bundle)
            embed_metadata(filename, payload)
            return result

        return wrapper

    return decorate(func) if func is not None else decorate


# --------------------------------------------------------------------------- #
# reading back
# --------------------------------------------------------------------------- #
def unpack(path: str | os.PathLike[str]) -> PlotBundle:
    """Return the :class:`PlotBundle` embedded in ``path`` (no code execution)."""
    return _unpack(extract_metadata(path))


def retrieve_data(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return the keyword arguments that produced the figure."""
    return unpack(path).args


def retrieve_code(path: str | os.PathLike[str]) -> str:
    """Return the (dedented) source of the plotting function."""
    return unpack(path).code


def _common_namespace() -> dict[str, Any]:
    """Best-effort globals so re-rendered code can resolve common imports."""
    namespace: dict[str, Any] = {"__name__": "__dlt_retrieve__"}
    for alias, module in (
        ("np", "numpy"),
        ("plt", "matplotlib.pyplot"),
        ("Image", "PIL.Image"),
        ("os", "os"),
        ("math", "math"),
        ("random", "random"),
        ("torch", "torch"),
    ):
        try:
            namespace[alias] = importlib.import_module(module)
        except Exception:
            continue
    return namespace


def _render(bundle: PlotBundle, *, save_to: str | os.PathLike[str] | None = None, **overrides: Any) -> Path:
    if not bundle.code:
        raise ValueError("no source code stored in this figure")
    # Keep matplotlib's cache inside the project (./cache) and render headless.
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), "cache", "matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    namespace = _common_namespace()
    exec(compile(bundle.code, f"<embedded:{bundle.name}>", "exec"), namespace)
    function = namespace.get(bundle.name)
    if not callable(function):
        raise ValueError(f"embedded code did not define a callable named {bundle.name!r}")

    args = dict(bundle.args)
    args.update(overrides)
    original = args.get("filename")
    if save_to is not None:
        target = Path(save_to)
        target.parent.mkdir(parents=True, exist_ok=True)
    elif original is not None:
        original = Path(os.fspath(original))
        target = unique_path(original.with_name(f"rendered-{original.stem}-{short_id(4)}{original.suffix}"))
    else:
        target = unique_path(Path("tmp") / f"rendered-{bundle.name}-{short_id(4)}.png")
    args["filename"] = target

    function(**args)
    try:
        import matplotlib.pyplot as plt

        plt.close("all")
    except Exception:
        pass
    return target


def retrieve_plot(
    path: str | os.PathLike[str],
    *,
    save_to: str | os.PathLike[str] | None = None,
    allow_execute: bool = False,
    **overrides: Any,
) -> Path:
    """Re-run the plotting code embedded in ``path``.

    .. warning::
        This **executes arbitrary Python** taken from the image file.  Only do
        it for files you created/trust, hence ``allow_execute=True`` is
        required.  Use :func:`retrieve_code` / :func:`retrieve_data` to inspect
        a file safely first.
    """
    if not allow_execute:
        raise PermissionError(
            "retrieve_plot executes code stored in the image. "
            "Inspect it with retrieve_code()/retrieve_data(), then pass allow_execute=True."
        )
    return _render(unpack(path), save_to=save_to, **overrides)


if __name__ == "__main__":  # pragma: no cover
    import matplotlib

    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), "cache", "matplotlib"))
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    @packplot
    def demo(x, y, filename):
        plt.figure()
        plt.plot(x, y)
        plt.savefig(filename)
        plt.close()

    out = Path("tmp") / f"packplot-demo-{short_id(4)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    demo(np.arange(10), np.arange(10) ** 2, out)
    bundle = unpack(out)
    print("stored:", out, "| method ok | args:", sorted(bundle.args))
    print("re-rendered:", retrieve_plot(out, allow_execute=True))
