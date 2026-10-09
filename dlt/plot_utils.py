"""Self-describing figures: images that carry their own plotting code and data.

The idea (kept from the original ``plot_utils.py``) is simple: right after you
save a figure, the decorator stashes everything needed to reproduce it -- the
function source, its keyword arguments, and some provenance -- inside the image
file itself.  Months later you can recover the data without remembering which
notebook or script produced the picture.

Payload format (version 2)
--------------------------
Metadata is stored as **size-bounded JSON** (optionally zlib-compressed), not as
a pickle.  This is deliberate:

* a pickled payload can contain *anything* (a whole dataset, an open file, a
  secret), and unpickling untrusted bytes executes code, so even the "safe"
  readers were not actually safe;
* JSON lets us measure and cap the size, redact credentials, and decode with a
  fixed whitelist instead of arbitrary class loading.

Only JSON-friendly values are stored.  NumPy arrays and torch tensors are
stored losslessly (dtype + shape + base64 bytes) **when small enough**; anything
larger, or of an unsupported type (modules, file handles, models, functions,
recursive containers, ...) is replaced by a compact *summary* plus a warning,
so a plot call can never silently embed a multi-GB object.

What this guards against
------------------------
* **Big arguments / big files**: per-argument (``max_arg_bytes``, default
  256 KiB) and per-payload (``max_payload_bytes``, default 2 MiB) budgets;
  oversized values become summaries and the largest arguments are dropped
  (recorded in ``warnings``) until the payload fits.
* **Non-image output**: if the decorated function does not save a readable
  image, :func:`packplot` raises instead of leaving metadata next to a random
  file.  Formats that cannot carry text metadata (TIFF, BMP, ...) still get a
  ``.meta`` sidecar.
* **Huge strings / deeply nested / recursive containers**: length, item-count
  and recursion limits.
* **Secrets in arguments**: keys that look like tokens/passwords are redacted.
* **Zip bombs**: the decompressed size is capped and verified.
* **Symlinks**: metadata is written to the resolved target, never replacing the
  symlink itself.
* **Code execution on read**: :func:`unpack`, :func:`retrieve_data` and
  :func:`retrieve_code` only parse JSON; only :func:`retrieve_plot` runs the
  embedded code, and only with ``allow_execute=True``.

Dependencies: ``Pillow`` only.
"""

from __future__ import annotations

import base64
import dataclasses
import enum
import hashlib
import importlib
import json
import logging
import os
import struct
import sys
import textwrap
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable

from .naming import short_id, unique_path, utcnow

__all__ = [
    "EncodeLimits",
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
#: current payload format (JSON); version 1 was pickle-based and is refused by default
FORMAT_VERSION = 2
#: EXIF tag for UserComment
_EXIF_USER_COMMENT = 0x9286
#: conservative ceiling for a single JPEG APP1 EXIF segment
_JPEG_EXIF_LIMIT = 60_000
#: hard ceiling for a payload embedded into *any* image
_MAX_EMBED_BYTES = 8 * 1024 * 1024
#: refuse to decompress more than this (zip-bomb guard)
_MAX_DECOMPRESSED = 64 * 1024 * 1024
#: formats that can carry metadata inside the file itself
_EMBEDDABLE = {"PNG", "JPEG", "JPG", "MPO", "WEBP"}
#: argument names that look like credentials and should never be embedded
_SENSITIVE = (
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "APIKEY",
    "API_KEY",
    "CREDENTIAL",
    "PRIVATE",
    "AUTH",
    "SESSION",
    "COOKIE",
)


@dataclass
class EncodeLimits:
    """Size limits applied while turning arguments into storable JSON."""

    #: max encoded size of a single argument before it is summarised
    max_arg_bytes: int = 256 * 1024
    #: max encoded size of the whole argument mapping
    max_payload_bytes: int = 2 * 1024 * 1024
    #: max number of elements kept from any container / array
    max_items: int = 10_000
    #: max bytes of any single buffer / string before truncation
    max_bytes: int = 256 * 1024
    #: max characters kept from any string
    max_string: int = 100_000
    #: max nesting depth
    max_depth: int = 32
    #: max characters of captured source code
    max_code: int = 200_000
    #: redact argument names that look like credentials
    redact_sensitive: bool = True


_LOG = logging.getLogger("dlt.plot_utils")


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _module(name: str) -> Any:
    """Return an already-imported module, or ``None`` (never imports)."""
    return sys.modules.get(name)


def _maybe_import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception:  # pragma: no cover - optional dependency
        return None


def _json_size(obj: Any) -> int:
    try:
        return len(json.dumps(obj, ensure_ascii=False, default=repr).encode("utf-8"))
    except Exception:
        return 1 << 60


def _short_repr(value: Any, limit: int = 200) -> str:
    try:
        text = repr(value)
    except Exception:
        text = f"<unrepresentable {type(value).__name__}>"
    return text if len(text) <= limit else text[:limit] + "..."


def _is_sensitive(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    upper = key.upper()
    return any(marker in upper for marker in _SENSITIVE)


def _tag_of(item: Any) -> str | None:
    return item.get("__dlt_type__") if isinstance(item, dict) else None


def _safe_len(value: Any) -> int | None:
    try:
        return len(value)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# encoding (Python -> storable JSON)
# --------------------------------------------------------------------------- #
def _array_summary(array: Any, numpy: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "__dlt_type__": "summary",
        "type": "ndarray",
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
    }
    try:
        if array.size and array.dtype.kind in "biuf":
            info.update(
                min=float(numpy.min(array)),
                max=float(numpy.max(array)),
                mean=float(numpy.mean(array)),
            )
        elif array.size and array.dtype.kind in "US":
            info["head"] = array.reshape(-1)[:5].tolist()
    except Exception:
        pass
    try:
        info["sha256"] = hashlib.sha256(
            numpy.ascontiguousarray(array).tobytes()
        ).hexdigest()[:16]
    except Exception:
        pass
    return info


def _summary(value: Any) -> dict[str, Any]:
    numpy = _module("numpy")
    if numpy is not None and isinstance(value, numpy.ndarray):
        return _array_summary(value, numpy)
    return {
        "__dlt_type__": "summary",
        "type": type(value).__name__,
        "len": _safe_len(value),
        "repr": _short_repr(value),
    }


def _encode_array(array: Any, numpy: Any, limits: EncodeLimits, warnings: list[str], label: str) -> Any:
    if array.dtype == object or array.dtype.kind in "OU":
        if array.size > limits.max_items:
            warnings.append(f"{label}: object/string array is too large; stored a summary")
            return _array_summary(array, numpy)
        return _encode(array.tolist(), limits, warnings, depth=1, label=label)
    if array.size > limits.max_items or array.nbytes > limits.max_bytes:
        warnings.append(
            f"{label}: array shape={list(array.shape)} dtype={array.dtype} exceeds the "
            "embedding budget; stored a summary"
        )
        return _array_summary(array, numpy)
    contiguous = numpy.ascontiguousarray(array)
    return {
        "__dlt_type__": "ndarray",
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "b64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _encode_tensor(tensor: Any, torch: Any, limits: EncodeLimits, warnings: list[str], label: str) -> Any:
    dtype = str(tensor.dtype).replace("torch.", "")
    numel = int(tensor.numel())
    too_big = numel > limits.max_items or numel * tensor.element_size() > limits.max_bytes
    if too_big or dtype in ("bfloat16",) or tensor.is_complex():
        warnings.append(f"{label}: tensor shape={list(tensor.shape)} dtype={dtype} stored as summary")
        info: dict[str, Any] = {
            "__dlt_type__": "summary",
            "type": "tensor",
            "dtype": dtype,
            "shape": list(tensor.shape),
            "numel": numel,
        }
        try:
            flat = tensor.detach().float().cpu().reshape(-1)
            if flat.numel():
                info.update(min=float(flat.min()), max=float(flat.max()), mean=float(flat.mean()))
        except Exception:
            pass
        return info
    array = tensor.detach().cpu().numpy()
    try:
        numpy = numpy_module()
    except Exception:  # pragma: no cover - torch without numpy is unusual
        return _summary(tensor)
    contiguous = numpy.ascontiguousarray(array)
    return {
        "__dlt_type__": "tensor",
        "dtype": dtype,
        "shape": list(tensor.shape),
        "b64": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def numpy_module() -> Any:  # pragma: no cover - tiny indirection for tests
    import numpy

    return numpy


def _encode_sequence(
    value: Any,
    limits: EncodeLimits,
    warnings: list[str],
    depth: int,
    label: str,
    seen: set[int],
) -> Any:
    if id(value) in seen:
        warnings.append(f"{label}: recursive container detected; stored a placeholder")
        return {"__dlt_type__": "recursive"}
    seen = seen | {id(value)}
    items = list(value)
    truncated = len(items) > limits.max_items
    if truncated:
        warnings.append(f"{label}: {len(items)} items exceed max_items={limits.max_items}")
    encoded = [
        _encode(item, limits, warnings, depth=depth + 1, label=f"{label}[{index}]", seen=seen)
        for index, item in enumerate(items[: limits.max_items])
    ]
    if isinstance(value, tuple):
        return {"__dlt_type__": "tuple", "items": encoded}
    if isinstance(value, (set, frozenset)):
        return {"__dlt_type__": "set", "items": encoded}
    if truncated:
        return {"__dlt_type__": "list", "truncated": True, "len": len(items), "items": encoded}
    return encoded


def _encode_mapping(
    value: Mapping,
    limits: EncodeLimits,
    warnings: list[str],
    depth: int,
    label: str,
    seen: set[int],
) -> Any:
    if id(value) in seen:
        warnings.append(f"{label}: recursive container detected; stored a placeholder")
        return {"__dlt_type__": "recursive"}
    seen = seen | {id(value)}
    keys = list(value.keys())
    truncated = len(keys) > limits.max_items
    if truncated:
        warnings.append(f"{label}: {len(keys)} keys exceed max_items={limits.max_items}")
    keys = keys[: limits.max_items]
    simple = not truncated and all(isinstance(k, str) and k != "__dlt_type__" for k in keys)
    if simple:
        out: dict[str, Any] = {}
        for key in keys:
            if limits.redact_sensitive and _is_sensitive(key):
                out[key] = "<redacted>"
                warnings.append(f"{label}.{key}: sensitive-looking key redacted")
            else:
                out[key] = _encode(
                    value[key], limits, warnings, depth=depth + 1, label=f"{label}.{key}", seen=seen
                )
        return out
    items = []
    for key in keys:
        if limits.redact_sensitive and _is_sensitive(key):
            items.append(
                [
                    _encode(key, limits, warnings, depth=depth + 1, label=label, seen=seen),
                    "<redacted>",
                ]
            )
            warnings.append(f"{label}.{key}: sensitive-looking key redacted")
        else:
            items.append(
                [
                    _encode(key, limits, warnings, depth=depth + 1, label=label, seen=seen),
                    _encode(
                        value[key], limits, warnings, depth=depth + 1, label=label, seen=seen
                    ),
                ]
            )
    out = {"__dlt_type__": "dict", "items": items}
    if truncated:
        out["truncated"] = True
        out["len"] = len(list(value.keys()))
    return out


def _encode(
    value: Any,
    limits: EncodeLimits,
    warnings: list[str],
    *,
    depth: int = 0,
    label: str = "value",
    seen: set[int] | None = None,
) -> Any:
    """Turn ``value`` into JSON-safe data, honouring the size limits."""
    seen = seen or set()
    if depth > limits.max_depth:
        warnings.append(f"{label}: deeper than max_depth={limits.max_depth}; stored a summary")
        return _summary(value)

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > limits.max_string or len(value.encode("utf-8", "replace")) > limits.max_bytes:
            warnings.append(f"{label}: string of {len(value)} chars truncated")
            return {
                "__dlt_type__": "str",
                "truncated": True,
                "len": len(value),
                "sha256": hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:16],
                "head": value[: limits.max_string],
            }
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        if len(data) > limits.max_bytes:
            warnings.append(f"{label}: {len(data)} bytes exceed max_bytes; stored a digest only")
            return {
                "__dlt_type__": "bytes",
                "truncated": True,
                "len": len(data),
                "sha256": hashlib.sha256(data).hexdigest()[:16],
            }
        return {"__dlt_type__": "bytes", "b64": base64.b64encode(data).decode("ascii")}
    if isinstance(value, os.PathLike):
        return {"__dlt_type__": "path", "value": os.fspath(value)}
    if isinstance(value, datetime):
        return {"__dlt_type__": "datetime", "kind": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"__dlt_type__": "datetime", "kind": "date", "value": value.isoformat()}
    if isinstance(value, enum.Enum):
        return _encode(value.value, limits, warnings, depth=depth + 1, label=label, seen=seen)

    numpy = _module("numpy")
    if numpy is not None:
        if isinstance(value, numpy.generic):
            try:
                return value.item()
            except Exception:
                return _summary(value)
        if isinstance(value, numpy.ndarray):
            return _encode_array(value, numpy, limits, warnings, label)

    torch = _module("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        return _encode_tensor(value, torch, limits, warnings, label)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _encode(
            dataclasses.asdict(value), limits, warnings, depth=depth + 1, label=label, seen=seen
        )
    if isinstance(value, Mapping):
        return _encode_mapping(value, limits, warnings, depth, label, seen)
    if isinstance(value, (list, tuple, set, frozenset)):
        return _encode_sequence(value, limits, warnings, depth, label, seen)

    warnings.append(f"{label}: type {type(value).__name__} is not storable; stored its repr")
    return {
        "__dlt_type__": "unsupported",
        "type": type(value).__name__,
        "repr": _short_repr(value),
    }


def _encode_args(args: Mapping[str, Any], limits: EncodeLimits, warnings: list[str]) -> dict[str, Any]:
    """Encode each argument and enforce per-argument / total size budgets."""
    encoded: dict[str, Any] = {}
    for raw_key, value in args.items():
        key = str(raw_key)
        if limits.redact_sensitive and _is_sensitive(key):
            encoded[key] = "<redacted>"
            warnings.append(f"argument {key!r} looks sensitive; value redacted")
            continue
        item = _encode(value, limits, warnings, label=key)
        if _json_size(item) > limits.max_arg_bytes:
            warnings.append(
                f"argument {key!r} exceeds max_arg_bytes={limits.max_arg_bytes}; stored a summary"
            )
            item = _summary(value)
        encoded[key] = item

    # Drop the largest arguments until the whole payload fits the budget.
    for _ in range(len(encoded)):
        if _json_size(encoded) <= limits.max_payload_bytes:
            break
        largest = max(encoded, key=lambda k: _json_size(encoded[k]))
        if _tag_of(encoded[largest]) == "omitted":
            break
        warnings.append(
            f"argument {largest!r} omitted: payload budget "
            f"({limits.max_payload_bytes} bytes) exceeded"
        )
        encoded[largest] = {"__dlt_type__": "omitted", "reason": "payload budget exceeded"}
    return encoded


# --------------------------------------------------------------------------- #
# decoding (storable JSON -> Python)
# --------------------------------------------------------------------------- #
def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get("__dlt_type__")
    if tag is None:
        return {key: _decode(item) for key, item in value.items()}
    if tag == "bytes":
        if "b64" in value:
            return base64.b64decode(value["b64"])
        return {key: item for key, item in value.items()}
    if tag == "str":
        if value.get("truncated"):
            return {key: item for key, item in value.items()}
        return value.get("head", "")
    if tag == "path":
        return Path(value["value"])
    if tag == "datetime":
        try:
            parsed = datetime.fromisoformat(value["value"])
            return parsed.date() if value.get("kind") == "date" else parsed
        except Exception:
            return value.get("value")
    if tag in ("ndarray", "tensor"):
        numpy = _maybe_import("numpy")
        if numpy is None:  # pragma: no cover - numpy is needed to write these
            return value
        array = numpy.frombuffer(base64.b64decode(value["b64"]), dtype=numpy.dtype(value["dtype"]))
        array = array.reshape(tuple(value["shape"])).copy()
        if tag == "ndarray":
            return array
        torch = _maybe_import("torch")
        return torch.from_numpy(array) if torch is not None else array
    if tag in ("list", "tuple", "set"):
        items = [_decode(item) for item in value.get("items", [])]
        return {"list": list, "tuple": tuple, "set": set}[tag](items)
    if tag == "dict":
        return {_decode(k): _decode(v) for k, v in value.get("items", [])}
    # summary / unsupported / omitted / recursive stay as plain dicts so the
    # caller can see *why* a value is not available.
    return {key: item for key, item in value.items()}


# --------------------------------------------------------------------------- #
# bundle + wire format
# --------------------------------------------------------------------------- #
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


def _pack(bundle: PlotBundle, limits: EncodeLimits | None = None) -> bytes:
    """Serialise a bundle to a size-bounded, checksummed payload."""
    limits = limits or EncodeLimits()
    warnings = list(bundle.warnings)
    code = bundle.code
    if len(code) > limits.max_code:
        warnings.append(f"source code truncated to {limits.max_code} characters")
        code = code[: limits.max_code]
    document = {
        "name": bundle.name,
        "code": code,
        "args": _encode_args(bundle.args, limits, warnings),
        "extra": _encode(dict(bundle.extra), limits, warnings, label="extra") if bundle.extra else {},
        "module": bundle.module,
        "file": bundle.file,
        "created_at": bundle.created_at,
        "warnings": warnings,
    }
    # Keep the recorded warnings in sync for callers that inspect the bundle.
    bundle.warnings = warnings
    raw = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    body = zlib.compress(raw, 9)
    digest = hashlib.sha256(raw).digest()[:16]
    header = MAGIC + bytes([FORMAT_VERSION, ord("j"), 1]) + struct.pack("<I", len(raw)) + digest
    return header + body


def _legacy_loads(raw: bytes, tag: bytes) -> PlotBundle:  # pragma: no cover - legacy only
    if tag == b"d":
        dill = _maybe_import("dill")
        if dill is not None:
            return dill.loads(raw)
    import pickle

    return pickle.loads(raw)


def _unpack(blob: bytes, *, allow_pickle: bool = False) -> PlotBundle:
    if len(blob) < len(MAGIC) + 23 or not blob.startswith(MAGIC):
        raise ValueError("not a dlt plot payload (bad magic)")
    offset = len(MAGIC)
    version = blob[offset]
    tag = bytes([blob[offset + 1]])
    compressed = bool(blob[offset + 2])
    raw_len = struct.unpack("<I", blob[offset + 3 : offset + 7])[0]
    digest = blob[offset + 7 : offset + 23]
    body = blob[offset + 23 :]

    if version == 1:
        if not allow_pickle:
            raise ValueError(
                "this is a legacy (pickle) dlt payload. Unpickling can execute arbitrary "
                "code, so pass allow_pickle=True only for files you trust."
            )
        raw = zlib.decompress(body) if compressed else body
        return _legacy_loads(raw, tag)
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported payload version {version}")
    if compressed:
        if raw_len > _MAX_DECOMPRESSED:
            raise ValueError(
                f"refusing to decompress {raw_len} bytes (limit {_MAX_DECOMPRESSED})"
            )
        decompressor = zlib.decompressobj()
        try:
            raw = decompressor.decompress(body, _MAX_DECOMPRESSED)
        except Exception as exc:
            raise ValueError(f"plot payload is corrupted: {exc}") from exc
        if decompressor.unconsumed_tail or len(raw) > raw_len:
            raise ValueError("plot payload expands beyond its declared size")
    else:
        raw = body
    if len(raw) != raw_len or hashlib.sha256(raw).digest()[:16] != digest:
        raise ValueError("plot payload is corrupted (checksum mismatch)")
    try:
        document = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"could not parse plot payload: {exc}") from exc
    return PlotBundle(
        code=document.get("code", ""),
        name=document.get("name", ""),
        args=_decode(document.get("args", {})),
        module=document.get("module"),
        file=document.get("file"),
        created_at=document.get("created_at", ""),
        warnings=list(document.get("warnings", [])),
        extra=_decode(document.get("extra", {})),
    )


# --------------------------------------------------------------------------- #
# embedding into image files
# --------------------------------------------------------------------------- #
def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".meta")


def _image_format(path: Path) -> str | None:
    """Return the Pillow format name, or ``None`` if it is not a readable image."""
    try:
        from PIL import Image

        with Image.open(path) as img:
            return (img.format or "").upper() or None
    except Exception:
        return None


def embed_metadata(
    path: str | os.PathLike[str],
    blob: bytes,
    *,
    max_bytes: int = _MAX_EMBED_BYTES,
) -> str:
    """Store ``blob`` inside the image at ``path``; returns the method used.

    Methods: ``"png-itxt"``, ``"jpeg-exif"``, ``"webp-exif"`` or ``"sidecar"``.
    Raises ``ValueError`` for files that are not readable images, so metadata is
    never attached to a non-image file.
    """
    from PIL import Image, PngImagePlugin

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    fmt = _image_format(path)
    if fmt is None:
        raise ValueError(f"{path} is not a readable image; refusing to attach metadata")
    # Follow symlinks so we modify the target instead of replacing the link.
    real = Path(os.path.realpath(path))
    if fmt not in _EMBEDDABLE or len(blob) > max_bytes:
        if len(blob) > max_bytes:
            _LOG.warning(
                "payload for %s is %d bytes (> %d); writing %s.meta instead",
                real,
                len(blob),
                max_bytes,
                real.name,
            )
        _sidecar_path(real).write_bytes(blob)
        return "sidecar"

    tmp = real.with_name(real.name + f".dlt-tmp-{short_id(6)}")
    try:
        with Image.open(real) as img:
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
                    tmp, format="JPEG", exif=exif.tobytes(), quality="keep", subsampling="keep"
                )
                method = "jpeg-exif"
            else:  # WEBP
                exif = img.getexif()
                exif[_EXIF_USER_COMMENT] = blob
                img.save(tmp, format="WEBP", exif=exif.tobytes(), quality=False)
                method = "webp-exif"
        os.replace(tmp, real)
        return method
    except Exception:
        # Never lose the payload, and never corrupt the image.
        _LOG.warning("could not embed metadata in %s; writing sidecar instead", real, exc_info=True)
        _sidecar_path(real).write_bytes(blob)
        return "sidecar"
    finally:
        if tmp.exists():
            try:
                os.remove(tmp)
            except OSError:
                pass


def extract_metadata(path: str | os.PathLike[str]) -> bytes:
    """Return the raw payload stored in/next to ``path`` (without parsing it)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        from PIL import Image

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
            raw_exif = img.info.get("exif")
            if isinstance(raw_exif, (bytes, bytearray)):
                index = bytes(raw_exif).find(MAGIC)
                if index >= 0:
                    return bytes(raw_exif)[index:]
    except Exception:
        pass
    sidecar = _sidecar_path(path)
    if sidecar.is_file():
        return sidecar.read_bytes()
    # embed_metadata() writes next to the resolved target, so also look there.
    real_sidecar = _sidecar_path(Path(os.path.realpath(path)))
    if real_sidecar.is_file():
        return real_sidecar.read_bytes()
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


def packplot(
    func: Callable[..., Any] | None = None,
    *,
    extra: Mapping[str, Any] | None = None,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] = (),
    limits: EncodeLimits | None = None,
) -> Callable[..., Any]:
    """Decorator: after the wrapped function saves ``filename``, embed its recipe.

    Usable as ``@packplot`` or ``@packplot()``.  The wrapped function **must**
    accept a ``filename`` argument whose value is the image it writes; if that
    file is not a readable image, the call raises (nothing is modified).

    Parameters
    ----------
    extra:
        Extra JSON-friendly provenance stored with the figure.
    include / exclude:
        Argument names to keep/drop before embedding (``exclude`` wins).
    limits:
        An :class:`EncodeLimits` to tune the size/security budgets.
    """

    def decorate(plotf: Callable[..., Any]) -> Callable[..., Any]:
        import inspect

        signature = inspect.signature(plotf)
        include_set = set(include) if include is not None else None
        exclude_set = set(exclude)
        active_limits = limits or EncodeLimits()

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

            # Keep matplotlib's font cache inside the project (./cache).
            os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), "cache", "matplotlib"))

            result = plotf(*args, **kwargs)

            if not filename.is_file():
                raise FileNotFoundError(
                    f"{plotf.__name__} did not create {filename}; "
                    "@packplot needs the plotted file to be saved before returning"
                )
            if _image_format(filename) is None:
                raise ValueError(
                    f"{plotf.__name__} did not save a readable image to {filename}; "
                    "@packplot only annotates image files and left the file untouched"
                )

            try:
                code = _clean_source(inspect.getsource(plotf))
            except (OSError, TypeError):
                code = ""
            try:
                source_file = inspect.getsourcefile(plotf)
            except TypeError:
                source_file = None

            selected: dict[str, Any] = {}
            for key, value in bound.arguments.items():
                if include_set is not None and key not in include_set:
                    continue
                if key in exclude_set:
                    continue
                selected[key] = value

            bundle = PlotBundle(
                code=code,
                name=plotf.__name__,
                args=selected,
                module=getattr(plotf, "__module__", None),
                file=source_file,
                warnings=[] if code else ["source code could not be captured"],
                extra=dict(extra or {}),
            )
            if _image_format(filename) not in _EMBEDDABLE:
                bundle.warnings.append(
                    f"{filename.suffix or 'this format'} cannot embed metadata; "
                    f"wrote {filename.name}.meta instead"
                )
            if not code:
                _LOG.warning("source of %s could not be captured; figure will not be re-renderable", plotf.__name__)
            payload = _pack(bundle, active_limits)
            embed_metadata(filename, payload)
            for warning in bundle.warnings:
                _LOG.warning("%s: %s", filename.name, warning)
            return result

        return wrapper

    return decorate(func) if func is not None else decorate


# --------------------------------------------------------------------------- #
# reading back
# --------------------------------------------------------------------------- #
def unpack(path: str | os.PathLike[str], *, allow_pickle: bool = False) -> PlotBundle:
    """Return the :class:`PlotBundle` embedded in ``path`` (no code execution).

    Legacy version-1 payloads are refused unless ``allow_pickle=True``.
    """
    return _unpack(extract_metadata(path), allow_pickle=allow_pickle)


def retrieve_data(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return the arguments that produced the figure."""
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
        imported = _maybe_import(module)
        if imported is not None:
            namespace[alias] = imported
    return namespace


def _render(
    bundle: PlotBundle, *, save_to: str | os.PathLike[str] | None = None, **overrides: Any
) -> Path:
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
        target = unique_path(
            original.with_name(f"rendered-{original.stem}-{short_id(4)}{original.suffix}")
        )
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
        a file safely first (they never execute the payload).
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
    print("stored:", out, "| args:", sorted(bundle.args), "| warnings:", bundle.warnings)
    print("re-rendered:", retrieve_plot(out, allow_execute=True))
