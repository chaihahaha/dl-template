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


# --------------------------------------------------------------------------- #
# opt-in usage: importing must not patch anything
# --------------------------------------------------------------------------- #
def test_import_does_not_patch_open() -> None:
    assert nw.is_installed() is False
    assert builtins.open is not nw.safe_open


def test_protect_decorator_scopes_and_restores_open(tmp_path: Path) -> None:
    original = builtins.open
    target = tmp_path / "a.txt"
    target.write_text("first")

    @nw.protect
    def rewrite(path: Path) -> bool:
        with open(path, "w") as handle:
            handle.write("second")
        return builtins.open is nw.safe_open

    assert rewrite(target) is True  # patched only inside the call
    assert target.read_text() == "second"
    assert len(backups_of(tmp_path, "a")) == 1
    assert builtins.open is original  # restored afterwards


def test_protect_scope_restores_on_exception(tmp_path: Path) -> None:
    original = builtins.open
    with pytest.raises(RuntimeError):
        with nw.protect_scope():
            assert builtins.open is nw.safe_open
            raise RuntimeError("boom")
    assert builtins.open is original


def test_protect_nesting_restores_only_at_the_end(tmp_path: Path) -> None:
    original = builtins.open

    @nw.protect
    def inner() -> bool:
        return builtins.open is nw.safe_open

    @nw.protect
    def outer() -> bool:
        inside_outer = builtins.open is nw.safe_open
        result = inner()
        return inside_outer and result and builtins.open is nw.safe_open

    assert outer() is True
    assert builtins.open is original


def test_protect_only_ignores_unrelated_files(tmp_path: Path) -> None:
    protected_dir = tmp_path / "runs"
    other_dir = tmp_path / "other"
    protected_dir.mkdir()
    other_dir.mkdir()
    protected_file = protected_dir / "x.txt"
    other_file = other_dir / "y.txt"
    protected_file.write_text("1")
    other_file.write_text("1")

    @nw.protect(only=[protected_dir])
    def rewrite_both() -> None:
        with open(protected_file, "w") as handle:
            handle.write("2")
        with open(other_file, "w") as handle:
            handle.write("2")

    rewrite_both()
    assert len(backups_of(protected_dir, "x")) == 1
    assert backups_of(other_dir, "y") == []  # untouched by the scope


def test_protect_paths_needs_no_patch(tmp_path: Path) -> None:
    target = tmp_path / "ckpt.bin"
    target.write_bytes(b"old")

    @nw.protect_paths("path")
    def save(state: bytes, path: Path) -> None:
        assert builtins.open is not nw.safe_open  # no monkey patch involved
        with open(path, "wb") as handle:
            handle.write(state)

    save(b"new", path=target)
    assert target.read_bytes() == b"new"
    assert len(backups_of(tmp_path, "ckpt")) == 1


def test_protect_paths_accepts_a_collection(tmp_path: Path) -> None:
    targets = [tmp_path / "a.txt", tmp_path / "b.txt"]
    for item in targets:
        item.write_text("old")

    @nw.protect_paths("paths")
    def save_all(paths) -> None:
        for item in paths:
            item.write_text("new")

    save_all(targets)
    assert all(item.read_text() == "new" for item in targets)
    assert len(backups_of(tmp_path, "a")) == 1
    assert len(backups_of(tmp_path, "b")) == 1


def test_protect_paths_reports_unknown_argument() -> None:
    @nw.protect_paths("nope")
    def save(x) -> None:
        pass

    with pytest.raises(TypeError):
        save(1)
