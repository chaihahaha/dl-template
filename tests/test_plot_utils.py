from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
matplotlib = pytest.importorskip("matplotlib")

import matplotlib.pyplot as plt  # noqa: E402

from dlt.plot_utils import (  # noqa: E402
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


def test_png_roundtrip_stores_code_and_args(tmp_path: Path) -> None:
    target = tmp_path / "line.png"
    assert line_plot(np.arange(5), np.arange(5) ** 2, target, title="hello") == "done"

    bundle = unpack(target)
    assert bundle.name == "line_plot"
    assert bundle.args["title"] == "hello"
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


def test_unpicklable_argument_is_placeholder(tmp_path: Path) -> None:
    class Weird:
        def __reduce__(self):
            raise TypeError("nope")

    target = tmp_path / "line.png"
    line_plot(np.arange(3), np.arange(3), target, title=Weird())
    bundle = unpack(target)
    assert bundle.args["title"] == "<unpicklable Weird>"
    assert any("title" in w for w in bundle.warnings)


def test_missing_filename_argument_is_reported(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        no_filename(1)


def test_function_that_does_not_save_is_reported(tmp_path: Path) -> None:
    @packplot
    def forgetful(x, filename):
        return x

    with pytest.raises(FileNotFoundError):
        forgetful(1, tmp_path / "never.png")


def test_extract_without_metadata_raises(tmp_path: Path) -> None:
    from PIL import Image

    plain = tmp_path / "plain.png"
    Image.new("RGB", (5, 5)).save(plain)
    with pytest.raises(ValueError):
        extract_metadata(plain)


def test_payload_is_corruption_checked(tmp_path: Path) -> None:
    target = tmp_path / "line.png"
    line_plot(np.arange(3), np.arange(3), target)
    from dlt.plot_utils import _pack, _unpack
    from dlt.plot_utils import PlotBundle

    blob = bytearray(_pack(PlotBundle(code="def f(): pass", name="f")))
    blob[-1] ^= 0xFF
    with pytest.raises(ValueError):
        _unpack(bytes(blob))
