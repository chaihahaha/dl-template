from __future__ import annotations

from pathlib import Path

from dlt import atomic_write_bytes, atomic_write_text


def test_atomic_write_text_creates_and_replaces(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "a.txt"
    atomic_write_text(target, "hello")
    assert target.read_text() == "hello"
    atomic_write_text(target, "world")
    assert target.read_text() == "world"


def test_atomic_write_leaves_no_temp_files(tmp_path: Path) -> None:
    target = tmp_path / "a.bin"
    atomic_write_bytes(target, b"x" * 100)
    leftovers = [p for p in tmp_path.iterdir() if p.name != "a.bin"]
    assert leftovers == []
    assert target.read_bytes() == b"x" * 100
