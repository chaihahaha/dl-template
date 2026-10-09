from __future__ import annotations

import builtins
from pathlib import Path

import pytest

import dlt.never_overwrite as nw
from dlt.never_overwrite import BackupPolicy
from dlt.yaml_logging import YamlLogger


@pytest.fixture(autouse=True)
def _clean_state():
    """Ensure each test starts and ends with the unpatched ``open``."""
    nw.uninstall()
    yield
    nw.uninstall()


def backups_of(directory: Path, stem: str) -> list[Path]:
    return sorted(directory.glob(f"{stem}.bak*"))


def test_install_and_uninstall_restore_open() -> None:
    original = builtins.open
    nw.install()
    assert builtins.open is nw.safe_open
    nw.uninstall()
    assert builtins.open is original


def test_truncating_write_backs_up_old_content(tmp_path: Path) -> None:
    nw.install()
    target = tmp_path / "data.txt"
    target.write_text("important")
    with open(target, "w") as handle:
        handle.write("new")
    assert target.read_text() == "new"
    kept = backups_of(tmp_path, "data")
    assert len(kept) == 1
    assert kept[0].read_text() == "important"
    assert kept[0].suffix == ".txt"


def test_append_and_exclusive_modes_are_not_backed_up(tmp_path: Path) -> None:
    nw.install()
    target = tmp_path / "data.txt"
    target.write_text("a")
    with open(target, "a") as handle:
        handle.write("b")
    assert backups_of(tmp_path, "data") == []
    with open(tmp_path / "fresh.txt", "x") as handle:
        handle.write("new")
    assert backups_of(tmp_path, "fresh") == []


def test_positional_open_arguments_are_forwarded(tmp_path: Path) -> None:
    nw.install()
    target = tmp_path / "pos.txt"
    with open(target, "w", 1, "utf-8") as handle:  # buffering + encoding positionally
        handle.write("ok")
    assert target.read_text() == "ok"


def test_pathlib_writes_are_protected_via_io_open(tmp_path: Path) -> None:
    nw.install()
    target = tmp_path / "p.txt"
    target.write_text("first")
    target.write_text("second")
    assert target.read_text() == "second"
    assert len(backups_of(tmp_path, "p")) == 1


def test_keep_policy_prunes_old_backups(tmp_path: Path) -> None:
    nw.install(policy=BackupPolicy(keep=2))
    target = tmp_path / "k.txt"
    for i in range(8):
        with open(target, "w") as handle:
            handle.write(str(i))
    assert len(backups_of(tmp_path, "k")) == 2


def test_empty_files_are_not_backed_up(tmp_path: Path) -> None:
    nw.install(policy=BackupPolicy(skip_empty=True))
    target = tmp_path / "empty.txt"
    target.write_text("")
    with open(target, "w") as handle:
        handle.write("data")
    assert backups_of(tmp_path, "empty") == []


def test_symlinks_are_left_alone(tmp_path: Path) -> None:
    nw.install()
    real = (tmp_path / "real.txt").resolve()
    real.write_text("real")
    link = tmp_path / "link.txt"
    link.symlink_to(real)
    with open(link, "w") as handle:
        handle.write("through-link")
    assert link.is_symlink()
    assert real.read_text() == "through-link"


def test_disabled_policy_does_not_touch_files(tmp_path: Path) -> None:
    nw.install(policy=BackupPolicy(enabled=False))
    target = tmp_path / "d.txt"
    target.write_text("old")
    with open(target, "w") as handle:
        handle.write("new")
    assert backups_of(tmp_path, "d") == []
    assert target.read_text() == "new"


def test_max_total_bytes_bounds_backup_size(tmp_path: Path) -> None:
    nw.install(policy=BackupPolicy(max_total_bytes=50))
    target = tmp_path / "m.bin"
    for _ in range(6):
        with open(target, "wb") as handle:
            handle.write(b"x" * 40)
    kept = backups_of(tmp_path, "m")
    assert len(kept) == 1  # 40-byte versions, 50-byte budget -> only the newest


def test_min_free_bytes_prunes_when_disk_is_low(tmp_path: Path) -> None:
    # An absurdly high threshold makes the guard fire on every backup.
    nw.install(policy=BackupPolicy(min_free_bytes=10**18))
    target = tmp_path / "cap.bin"
    for _ in range(5):
        with open(target, "wb") as handle:
            handle.write(b"y" * 10)
    # The guard frees previous backups before creating a new one, so the count
    # stays tiny instead of growing once per write.
    assert len(backups_of(tmp_path, "cap")) <= 1


def test_backups_of_other_files_are_never_pruned_by_keep(tmp_path: Path) -> None:
    nw.install(policy=BackupPolicy(keep=1))
    other = tmp_path / "other.txt"
    other.write_text("other-old")
    with open(other, "w") as handle:
        handle.write("other-new")
    target = tmp_path / "target.txt"
    for i in range(3):
        with open(target, "w") as handle:
            handle.write(str(i))
    assert len(backups_of(tmp_path, "other")) == 1
    assert len(backups_of(tmp_path, "target")) == 1


def test_torch_save_is_protected(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import torch

    original_save = torch.save
    nw.install()
    try:
        nw.protect_torch()
        assert getattr(torch.save, "_dlt_protected", False) is True
        target = tmp_path / "m.pt"
        target.write_bytes(b"old-checkpoint")
        torch.save({"w": torch.zeros(2)}, target)
        kept = backups_of(tmp_path, "m")
        assert len(kept) == 1 and kept[0].read_bytes() == b"old-checkpoint"
    finally:
        torch.save = original_save


def test_logger_saves_do_not_create_backups(tmp_path: Path) -> None:
    """The whole point of atomic saves: no .bak storm from periodic logging."""
    nw.install()
    log = YamlLogger("no-backup", logdir=tmp_path, save_every=1)
    for i in range(5):
        log.log(loss=float(i))
    log.close()
    assert backups_of(tmp_path, log.path.name) == []
    assert [p for p in tmp_path.iterdir() if p.suffix == ".bak" or ".bak-" in p.name] == []
