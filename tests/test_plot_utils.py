from __future__ import annotations

import hashlib
import pickle
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
matplotlib = pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from dlt.plot_utils import (  # noqa: E402
    EncodeLimits,
    PlotBundle,
    extract_metadata,
    packplot,
    retrieve_code,
    retrieve_data,
    retrieve_plot,
    unpack,
)


@packplot
def line_plot(x, y, filename, title="line"):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.savefig(filename)
    plt.close()
    return "done"


@packplot()
def scatter_plot(x, y, filename):
    plt.figure()
    plt.scatter(x, y)
    plt.savefig(filename)
    plt.close()


@packplot
def no_filename(x):
    return x


def _blank_image(path: Path, fmt: str | None = None) -> None:
    Image.new("RGB", (20, 20), (200, 200, 200)).save(path, format=fmt)


# --------------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------------- #
def test_png_roundtrip_stores_code_and_args(tmp_path: Path) -> None:
    target = tmp_path / "line.png"
    assert line_plot(np.arange(5), np.arange(5) ** 2, target, title="hello") == "done"

    bundle = unpack(target)
    assert bundle.name == "line_plot"
    assert bundle.args["title"] == "hello"
    assert np.array_equal(bundle.args["x"], np.arange(5))
    assert "def line_plot" in bundle.code
    assert bundle.warnings == []


def test_no_argument_decorator_form(tmp_path: Path) -> None:
    target = tmp_path / "scatter.png"
    scatter_plot(np.arange(3), np.arange(3), target)
    assert unpack(target).name == "scatter_plot"


def test_retrieve_helpers(tmp_path: Path) -> None:
    target = tmp_path / "line.png"
    line_plot(np.arange(4), np.arange(4), target, title="t")
    assert retrieve_data(target)["title"] == "t"
    assert retrieve_code(target).lstrip().startswith("def line_plot")


def test_retrieve_plot_requires_opt_in_then_renders(tmp_path: Path) -> None:
    target = tmp_path / "line.png"
    line_plot(np.arange(4), np.arange(4), target)
    with pytest.raises(PermissionError):
        retrieve_plot(target)
    rendered = retrieve_plot(target, allow_execute=True)
    assert rendered.is_file() and rendered != target


def test_retrieve_plot_custom_output(tmp_path: Path) -> None:
    target = tmp_path / "line.png"
    line_plot(np.arange(4), np.arange(4), target)
    out = tmp_path / "sub" / "redraw.png"
    rendered = retrieve_plot(target, save_to=out, allow_execute=True)
    assert rendered == out and out.is_file()


def test_sidecar_for_formats_without_embedding(tmp_path: Path) -> None:
    target = tmp_path / "line.tiff"
    line_plot(np.arange(4), np.arange(4), target)
    assert (tmp_path / "line.tiff.meta").is_file()
    assert any("cannot embed metadata" in w for w in unpack(target).warnings)


# --------------------------------------------------------------------------- #
# extreme cases: big / non-image / sensitive / recursive / exotic
# --------------------------------------------------------------------------- #
@packplot
def writes_text(filename):
    Path(filename).write_text("not an image")
    return "ok"


def test_non_image_output_is_refused_and_untouched(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    with pytest.raises(ValueError, match="readable image"):
        writes_text(target)
    assert target.read_text() == "not an image"
    assert not (tmp_path / "out.txt.meta").exists()


@packplot
def big_plot(data, filename):
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.savefig(filename)
    plt.close()


def test_large_array_is_summarised_not_embedded(tmp_path: Path) -> None:
    target = tmp_path / "big.png"
    big_plot(np.zeros(5_000_000, dtype=np.float32), target)  # 20 MB
    bundle = unpack(target)
    assert bundle.args["data"]["__dlt_type__"] == "summary"
    assert bundle.args["data"]["nbytes"] == 5_000_000 * 4
    assert any("exceeds the embedding budget" in w for w in bundle.warnings)
    assert target.stat().st_size < 200_000  # the image stays small


@packplot
def big_bytes_plot(blob, filename):
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.savefig(filename)
    plt.close()


def test_large_bytes_are_digested_not_embedded(tmp_path: Path) -> None:
    target = tmp_path / "bytes.png"
    big_bytes_plot(b"x" * 2_000_000, target)
    encoded = unpack(target).args["blob"]
    assert encoded["__dlt_type__"] == "bytes" and encoded["truncated"] is True
    assert encoded["len"] == 2_000_000
    assert "b64" not in encoded


@packplot(limits=EncodeLimits(max_arg_bytes=10_000, max_payload_bytes=3_000, max_string=10_000))
def multi_arg_plot(a, b, c, d, filename):
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.savefig(filename)
    plt.close()


def test_payload_budget_omits_largest_arguments(tmp_path: Path) -> None:
    target = tmp_path / "multi.png"
    multi_arg_plot("x" * 900, "y" * 900, "z" * 900, "w" * 900, target)
    bundle = unpack(target)
    omitted = [v for v in bundle.args.values() if isinstance(v, dict) and v.get("__dlt_type__") == "omitted"]
    assert omitted
    assert any("omitted" in w for w in bundle.warnings)


@packplot
def secret_plot(token, api_key, filename):
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.savefig(filename)
    plt.close()


def test_sensitive_arguments_are_redacted(tmp_path: Path) -> None:
    target = tmp_path / "secret.png"
    secret_plot("super-secret", "also-secret", target)
    bundle = unpack(target)
    assert bundle.args["token"] == "<redacted>"
    assert bundle.args["api_key"] == "<redacted>"
    assert "super-secret" not in target.read_bytes().decode("latin-1")


@packplot
def recursive_plot(data, filename):
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.savefig(filename)
    plt.close()


def test_recursive_container_does_not_blow_up(tmp_path: Path) -> None:
    data = [1, 2]
    data.append(data)
    target = tmp_path / "recursive.png"
    recursive_plot(data, target)
    bundle = unpack(target)
    assert bundle.args["data"][0] == 1
    assert any(
        isinstance(item, dict) and item.get("__dlt_type__") == "recursive"
        for item in bundle.args["data"]
    )


def test_unsupported_object_becomes_summary(tmp_path: Path) -> None:
    class Weird:
        pass

    target = tmp_path / "line.png"
    line_plot(np.arange(3), np.arange(3), target, title=Weird())
    bundle = unpack(target)
    assert bundle.args["title"]["__dlt_type__"] in ("unsupported", "summary")
    assert bundle.args["title"]["type"] == "Weird"
    assert any("title" in w for w in bundle.warnings)


def test_include_and_exclude_select_arguments(tmp_path: Path) -> None:
    @packplot(exclude=("secret",))
    def selective(public, secret, filename):
        plt.figure()
        plt.plot([0, 1], [0, 1])
        plt.savefig(filename)
        plt.close()

    target = tmp_path / "sel.png"
    selective("shown", "hidden", target)
    args = unpack(target).args
    assert args["public"] == "shown"
    assert "secret" not in args


@packplot
def mapping_plot(mapping, blob, path, filename):
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.savefig(filename)
    plt.close()


def test_dict_keys_bytes_and_path_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "map.png"
    mapping_plot({1: "a", (2, 3): "b"}, b"\x00\x01\x02", Path("some/file.txt"), target)
    args = unpack(target).args
    assert args["mapping"] == {1: "a", (2, 3): "b"}
    assert args["blob"] == b"\x00\x01\x02"
    assert args["path"] == Path("some/file.txt")


@packplot(limits=EncodeLimits(max_items=3))
def long_list_plot(items, filename):
    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.savefig(filename)
    plt.close()


def test_long_container_is_truncated(tmp_path: Path) -> None:
    target = tmp_path / "long.png"
    long_list_plot(list(range(100)), target)
    bundle = unpack(target)
    assert bundle.args["items"] == [0, 1, 2]
    assert any("max_items" in w for w in bundle.warnings)


def test_unpack_does_not_execute_embedded_code(tmp_path: Path) -> None:
    sentinel = tmp_path / "pwned.txt"
    evil = PlotBundle(
        code=f"open({str(sentinel)!r}, 'w').write('boom')",
        name="f",
        args={},
    )
    target = tmp_path / "evil.png"
    _blank_image(target)
    from dlt.plot_utils import _pack, embed_metadata

    embed_metadata(target, _pack(evil))

    bundle = unpack(target)  # must parse JSON only
    assert "open(" in bundle.code
    assert not sentinel.exists()


def test_legacy_pickle_payload_is_refused_by_default(tmp_path: Path) -> None:
    target = tmp_path / "legacy.png"
    _blank_image(target)
    from dlt.plot_utils import MAGIC, _EXIF_USER_COMMENT, _unpack, embed_metadata

    raw = pickle.dumps(PlotBundle(code="def f(): pass", name="f"))
    body = zlib.compress(raw)
    digest = hashlib.sha256(raw).digest()[:16]
    blob = MAGIC + bytes([1, ord("p"), 1]) + struct.pack("<I", len(raw)) + digest + body

    with pytest.raises(ValueError, match="legacy"):
        _unpack(blob)
    # and it can be read back explicitly if the user trusts the file
    bundle = _unpack(blob, allow_pickle=True)
    assert bundle.name == "f"


def test_symlink_target_is_updated_without_breaking_the_link(tmp_path: Path) -> None:
    target = tmp_path / "real.png"
    link = tmp_path / "link.png"
    _blank_image(target)
    try:
        link.symlink_to(target)
    except OSError:  # pragma: no cover - filesystem without symlinks
        pytest.skip("symlinks unavailable")

    line_plot(np.arange(3), np.arange(3), link)
    assert link.is_symlink()
    assert unpack(link).name == "line_plot"
    assert unpack(target).name == "line_plot"


# --------------------------------------------------------------------------- #
# errors / corruption
# --------------------------------------------------------------------------- #
def test_missing_filename_argument_is_reported() -> None:
    with pytest.raises(TypeError):
        no_filename(1)


def test_function_that_does_not_save_is_reported(tmp_path: Path) -> None:
    @packplot
    def forgetful(x, filename):
        return x

    with pytest.raises(FileNotFoundError):
        forgetful(1, tmp_path / "never.png")


def test_extract_without_metadata_raises(tmp_path: Path) -> None:
    plain = tmp_path / "plain.png"
    _blank_image(plain)
    with pytest.raises(ValueError):
        extract_metadata(plain)


def test_payload_is_corruption_checked() -> None:
    from dlt.plot_utils import _pack, _unpack

    blob = bytearray(_pack(PlotBundle(code="def f(): pass", name="f")))
    blob[-1] ^= 0xFF
    with pytest.raises(ValueError):
        _unpack(bytes(blob))


def test_zip_bomb_is_refused() -> None:
    from dlt.plot_utils import MAGIC, _unpack

    raw = b"a" * 1_000_000
    body = zlib.compress(raw, 9)
    digest = hashlib.sha256(raw).digest()[:16]
    # claim a huge decompressed size to trip the guard
    blob = MAGIC + bytes([2, ord("j"), 1]) + struct.pack("<I", 10**9) + digest + body
    with pytest.raises(ValueError, match="decompress"):
        _unpack(blob)
