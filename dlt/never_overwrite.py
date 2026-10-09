"""Never lose data to an accidental ``open(path, "w")``.

``import dlt.never_overwrite`` (or the top-level ``import never_overwrite``
shim) installs a drop-in replacement for :func:`open`, :func:`io.open` and
therefore :meth:`pathlib.Path.open` / :meth:`~pathlib.Path.write_text`.  When a
file that already exists is opened in a truncating mode (``"w"`` / ``"w+"``),
the old content is **renamed to a timestamped backup in the same directory**
before the truncation happens.  The returned file object is the normal one, so
the change is transparent to the rest of your code::

    import never_overwrite
    with open("result.csv", "w") as fh:     # result.csv -> result.bak-...-ab12.csv
        fh.write("new data")

Design notes / fixed bugs compared with the original implementation
-------------------------------------------------------------------
* The old version only created a backup when the root filesystem had more than
  16 GiB free, and otherwise **silently overwrote the file**.  Renaming within
  one directory costs no extra space, so we now always back up and rely on
  :func:`os.replace` (atomic, same-filesystem).
* ``open(file, mode, *args)`` extras (buffering, encoding, ...) are forwarded
  correctly; the original signature dropped them and raised ``TypeError``.
* File descriptors (``open(fd, "w")``) and non-path objects are passed through.
* Symlinks are not moved (that would change what the link points at); the
  backup step is skipped for them.
* ``mode="x"`` already refuses to overwrite, and ``mode="a"`` never truncates,
  so neither needs a backup.
* Backups are named ``<stem>.bak-<utc-timestamp>-<id><ext>`` and can optionally
  be capped with ``keep=N``.  Every action is logged to the
  ``dlt.never_overwrite`` logger.
* Repeatedly rewriting one large file (e.g. a checkpoint every epoch) used to
  grow the backup set without bound.  ``keep``, ``max_total_bytes`` and a
  ``min_free_bytes`` disk-space guard (default 512 MiB) now bound how much disk
  backups can consume; the guard only removes files matching the backup naming
  scheme, oldest first, and logs a warning.

This module intentionally has **no third-party dependencies**.
"""

from __future__ import annotations

import builtins
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable

from .naming import short_id, timestamp, unique_path

__all__ = [
    "BackupPolicy",
    "backup",
    "safe_open",
    "protect_torch",
    "install",
    "uninstall",
    "is_installed",
]

_ORIGINAL_OPEN: Callable[..., Any] = builtins.open
_ORIGINAL_IO_OPEN: Callable[..., Any] | None = None
try:  # ``io.open`` is the same object at import time; keep a handle for restore
    import io as _io

    _ORIGINAL_IO_OPEN = _io.open
except Exception:  # pragma: no cover
    _io = None  # type: ignore[assignment]

_LOCK = threading.RLock()
_INSTALLED = False
_POLICY: "BackupPolicy | None" = None


@dataclass
class BackupPolicy:
    """Knobs for the backup behaviour."""

    enabled: bool = True
    #: string inserted before the extension, e.g. ``model.bak-<ts>-<id>.pt``
    suffix: str = ".bak"
    #: keep at most this many backups per file (``None`` = keep them all)
    keep: int | None = None
    #: an empty file carries no data, so there is nothing to back up
    skip_empty: bool = True
    #: copy backups into this directory instead of next to the original
    backup_dir: str | os.PathLike[str] | None = None
    #: log every backup at INFO (otherwise DEBUG)
    verbose: bool = False
    #: when the filesystem holding the backups has less than this much free
    #: space, prune the *oldest* backups until there is room again.  This is a
    #: safety valve so that repeatedly rewriting one big file cannot fill the
    #: disk.  ``None`` disables the check.
    min_free_bytes: int | None = 512 * 1024 * 1024
    #: hard ceiling on the total size of backups of a single file (oldest are
    #: removed first).  ``None`` = no ceiling.
    max_total_bytes: int | None = None

    def normalized(self) -> "BackupPolicy":
        if self.keep is not None and self.keep < 0:
            raise ValueError("keep must be >= 0 or None")
        return self


def _log(policy: BackupPolicy, msg: str) -> None:
    import logging

    logging.getLogger("dlt.never_overwrite").log(
        logging.INFO if policy.verbose else logging.DEBUG, msg
    )


def _warn(msg: str) -> None:
    import logging

    logging.getLogger("dlt.never_overwrite").warning(msg)


def _backup_name(path: str, policy: BackupPolicy) -> str:
    stem, ext = os.path.splitext(path)
    base = os.path.basename(stem)
    name = f"{base}{policy.suffix}-{timestamp()}-{short_id(4)}{ext}"
    if policy.backup_dir is not None:
        return os.path.join(os.fspath(policy.backup_dir), name)
    return os.path.join(os.path.dirname(path), name)


def _backup_directory(path: str, policy: BackupPolicy) -> str:
    if policy.backup_dir is not None:
        return os.fspath(policy.backup_dir)
    return os.path.dirname(path) or "."


def _list_backups(directory: str, policy: BackupPolicy, prefix: str | None = None) -> list[str]:
    """All files in ``directory`` that look like one of our backups."""
    marker = f"{policy.suffix}-"
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    found = []
    for name in names:
        if marker not in name:
            continue
        if prefix is not None and not name.startswith(prefix):
            continue
        found.append(os.path.join(directory, name))
    return found


def _enforce_free_space(directory: str, policy: BackupPolicy) -> None:
    """Prune our oldest backups if the disk is running low.

    Only files matching the backup naming scheme are removed, oldest first, and
    every removal is logged.  This directly answers "can never_overwrite fill
    the disk?" -- repeated rewrites of a large checkpoint no longer can.
    """
    if not policy.min_free_bytes:
        return
    import shutil

    try:
        free = shutil.disk_usage(directory).free
    except OSError:
        return
    if free >= policy.min_free_bytes:
        return

    candidates = sorted(_list_backups(directory, policy), key=os.path.getmtime)
    if not candidates:
        return
    _warn(
        "only %.1f MiB free on the filesystem holding %s; "
        "pruning oldest backups (min_free_bytes=%d)"
        % (free / 2**20, directory, policy.min_free_bytes)
    )
    for stale in candidates:
        if free >= policy.min_free_bytes:
            break
        try:
            size = os.path.getsize(stale)
            os.remove(stale)
            free += size
            _warn(f"pruned old backup {stale} to free disk space")
        except OSError:
            continue


def _prune_backups(path: str, policy: BackupPolicy) -> None:
    if policy.keep is None and not policy.max_total_bytes:
        return
    stem, ext = os.path.splitext(os.path.basename(path))
    directory = _backup_directory(path, policy)
    entries = [
        entry
        for entry in _list_backups(directory, policy, prefix=f"{stem}{policy.suffix}-")
        if entry.endswith(ext)
    ]
    entries.sort(key=os.path.getmtime, reverse=True)  # newest first

    if policy.keep is not None:
        doomed, entries = entries[policy.keep :], entries[: policy.keep]
        for stale in doomed:
            _remove_backup(stale, policy)

    if policy.max_total_bytes:
        total = 0
        for entry in entries:
            try:
                total += os.path.getsize(entry)
            except OSError:
                continue
        for entry in reversed(entries):  # oldest first
            if total <= policy.max_total_bytes:
                break
            try:
                size = os.path.getsize(entry)
            except OSError:
                continue
            if _remove_backup(entry, policy):
                total -= size


def _remove_backup(path: str, policy: BackupPolicy) -> bool:
    try:
        os.remove(path)
        _log(policy, f"pruned old backup {path}")
        return True
    except OSError:
        return False


def backup(
    path: str | os.PathLike[str],
    *,
    policy: BackupPolicy | None = None,
) -> str | None:
    """Move ``path`` aside to a timestamped backup and return the new path.

    Returns ``None`` when nothing was done (missing file, empty file, disabled
    policy, ...).  Safe to call repeatedly.
    """
    policy = (policy or _POLICY or BackupPolicy()).normalized()
    if not policy.enabled:
        return None
    path = os.fspath(path)
    # ``isfile`` follows symlinks; moving a symlink would relocate the link
    # rather than the data, so leave symlinks alone.
    if os.path.islink(path):
        _log(policy, f"skipping backup of symlink {path}")
        return None
    if not os.path.isfile(path):
        return None
    try:
        if policy.skip_empty and os.path.getsize(path) == 0:
            return None
    except OSError:
        return None

    target = unique_path(_backup_name(path, policy))
    directory = _backup_directory(path, policy)
    if policy.backup_dir is not None:
        os.makedirs(directory, exist_ok=True)
    # Free space *before* we need it, so a nearly-full disk does not tip over
    # because of the backups we are about to create.
    _enforce_free_space(directory, policy)
    try:
        os.replace(path, target)
    except OSError:
        # Cross-device move (only possible with a custom backup_dir).
        import shutil

        shutil.move(path, target)
    _log(policy, f"backed up {path} -> {target}")
    _prune_backups(path, policy)
    return target


def safe_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
    """Drop-in :func:`open` that backs up truncating writes."""
    policy = (_POLICY or BackupPolicy()).normalized()
    if (
        policy.enabled
        and isinstance(mode, str)
        and "w" in mode
        and not isinstance(file, int)
        and file is not None
        and isinstance(file, (str, bytes, os.PathLike))
    ):
        path = os.fsdecode(os.fspath(file))
        with _LOCK:
            backup(path, policy=policy)
    return _ORIGINAL_OPEN(file, mode, *args, **kwargs)


def protect_torch() -> Any:
    """Back up the destination of ``torch.save(obj, path)``.

    Modern PyTorch writes checkpoints through a C++ file writer that never
    touches Python's :func:`open`, so patching ``open`` alone does **not**
    protect ``torch.save``.  Calling this (or :func:`install`, which calls it
    automatically when torch is already imported) wraps ``torch.save`` so the
    existing checkpoint is backed up first.
    """
    import functools

    torch = __import__("torch")
    save = torch.save
    if getattr(save, "_dlt_protected", False):
        return save

    @functools.wraps(save)
    def protected_save(obj: Any, f: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(f, (str, bytes, os.PathLike)):
            with _LOCK:
                backup(os.fsdecode(os.fspath(f)))
        return save(obj, f, *args, **kwargs)

    protected_save._dlt_protected = True  # type: ignore[attr-defined]
    torch.save = protected_save
    return protected_save


def _auto_protect_optional() -> None:
    """Patch optional libraries that bypass ``open``, if they are loaded."""
    import sys

    if "torch" in sys.modules:
        try:
            protect_torch()
        except Exception:  # pragma: no cover - best effort
            pass


def install(
    *,
    policy: BackupPolicy | None = None,
    also_io: bool = True,
) -> BackupPolicy:
    """Install the patched ``open``.  Idempotent; returns the active policy."""
    global _INSTALLED, _POLICY
    with _LOCK:
        if policy is not None:
            _POLICY = policy.normalized()
        elif _POLICY is None:
            _POLICY = BackupPolicy()
        if not _INSTALLED:
            builtins.open = safe_open
            if also_io and _io is not None:
                _io.open = safe_open
            _INSTALLED = True
        _auto_protect_optional()
    return _POLICY


def uninstall() -> None:
    """Restore the original ``open`` and reset the policy (mainly for tests)."""
    global _INSTALLED, _POLICY
    with _LOCK:
        if builtins.open is safe_open:
            builtins.open = _ORIGINAL_OPEN
        if _io is not None and _io.open is safe_open and _ORIGINAL_IO_OPEN is not None:
            _io.open = _ORIGINAL_IO_OPEN
        _INSTALLED = False
        _POLICY = None


def is_installed() -> bool:
    return _INSTALLED


def _env_enabled() -> bool:
    value = os.environ.get("DLT_NEVER_OVERWRITE", os.environ.get("DLT_NO_OVERWRITE", "1"))
    return value.strip().lower() not in ("0", "false", "off", "no", "")


# Importing this module is the documented way to enable protection, so do it
# automatically unless the user opted out with DLT_NEVER_OVERWRITE=0.
if _env_enabled():
    _keep = os.environ.get("DLT_NEVER_OVERWRITE_KEEP")
    _min_free_mb = os.environ.get("DLT_NEVER_OVERWRITE_MIN_FREE_MB")
    _max_total_mb = os.environ.get("DLT_NEVER_OVERWRITE_MAX_TOTAL_MB")
    install(
        policy=BackupPolicy(
            keep=int(_keep) if _keep and _keep.isdigit() else None,
            min_free_bytes=(
                int(_min_free_mb) * 1024 * 1024
                if _min_free_mb is not None and _min_free_mb.lstrip("-").isdigit()
                else BackupPolicy.min_free_bytes
            ),
            max_total_bytes=(
                int(_max_total_mb) * 1024 * 1024
                if _max_total_mb is not None and _max_total_mb.isdigit()
                else None
            ),
            verbose=os.environ.get("DLT_NEVER_OVERWRITE_VERBOSE", "0").lower()
            in ("1", "true", "on", "yes"),
        )
    )
