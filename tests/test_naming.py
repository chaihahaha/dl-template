from __future__ import annotations

from pathlib import Path

import pytest

from dlt.naming import (
    ExperimentNamer,
    auto_name,
    short_id,
    slugify,
    timestamp,
    unique_path,
)


def test_timestamp_is_sortable_utc() -> None:
    value = timestamp()
    assert len(value) == 16 and value.endswith("Z") and "T" in value
    assert value.startswith("20")


def test_slugify_keeps_unicode_and_collapses_punctuation() -> None:
    assert slugify("ResNet 50 / CIFAR-10") == "ResNet-50-CIFAR-10"
    assert slugify("  实验 一  ") == "实验-一"
    assert slugify("") == "run"
    assert len(slugify("x" * 200, max_length=10)) == 10


def test_short_id_length() -> None:
    assert len(short_id(8)) == 8
    assert short_id(8) != short_id(8)


def test_unique_path_counter(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    assert unique_path(target) == target
    target.write_text("x")
    first = unique_path(target)
    assert first.name == "out_001.txt"
    first.write_text("y")
    assert unique_path(target).name == "out_002.txt"


def test_unique_path_timestamp_strategy(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("x")
    other = unique_path(target, strategy="timestamp")
    assert other != target and other.suffix == ".txt" and other.exists() is False


def test_unique_path_create_is_atomic(tmp_path: Path) -> None:
    target = tmp_path / "reserved.txt"
    created = unique_path(target, create=True)
    assert created.exists()
    again = unique_path(target, create=True)
    assert again != created and again.exists()


def test_unique_path_rejects_unknown_strategy(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        unique_path(tmp_path / "x", strategy="nope")


def test_auto_name_has_directory_extension_and_uniqueness(tmp_path: Path) -> None:
    first = auto_name("loss curve", ext=".png", directory=tmp_path)
    assert first.parent == tmp_path and first.suffix == ".png"
    first.write_bytes(b"")
    second = auto_name("loss curve", ext="png", directory=tmp_path)
    assert second != first


def test_experiment_namer_creates_standard_layout(tmp_path: Path) -> None:
    namer = ExperimentNamer(tmp_path / "runs", name="My Exp")
    run = namer.new_run(config={"lr": 1e-3, "optimizer": "adam"})
    assert run.run_dir.is_dir()
    for sub in ("logs", "ckpts", "figures", "results"):
        assert (run.run_dir / sub).is_dir()
    assert (run.run_dir / "config.yaml").is_file()
    assert "My-Exp" in run.run_dir.name


def test_experiment_namer_config_shorthand_and_latest(tmp_path: Path) -> None:
    namer = ExperimentNamer(tmp_path / "runs", name="x")
    first = namer.new_run({"a": 1})  # positional mapping == config
    assert first.name == "x"
    second = namer.new_run(config={"b": 2}, run_id="deadbeef")
    assert second.run_id == "deadbeef"
    latest = namer.latest_run()
    assert latest is not None and latest.name >= first.run_dir.name


def test_run_paths_file_is_unique(tmp_path: Path) -> None:
    run = ExperimentNamer(tmp_path / "runs", name="x").new_run()
    a = run.file("curve", ext=".png", subdir="figures")
    a.write_bytes(b"")
    b = run.file("curve", ext=".png", subdir="figures")
    assert a != b and b.name == "curve_001.png"
