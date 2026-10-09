"""Never lose data to an accidental ``open(path, "w")`` -- without patching the
whole process.

The old version patched :func:`builtins.open` **at import time**, so every
library in the process (and every unrelated ``write_text``) silently produced
``.bak-*`` files.  Importing this module now has **no side effects**: normal
Python I/O stays untouched.  Protection is opt-in, at the granularity you
choose.

Three ways to use it
--------------------
1. Back up specific files, no monkey-patching at all (recommended)::

       from dlt.never_overwrite import protect_paths

       @protect_paths("path")           # backs up the file before the call
       def save_checkpoint(state, path):
           torch.save(state, path)

   or simply call ``backup("result.csv")`` before writing.

2. Protect a whole function/block with a *scoped* replacement for ``open``.
   It is installed on entry and the pristine ``open`` is restored on exit, so
   nothing outside the scope is affected::

       from dlt.never_overwrite import protect

       @protect                              # any overwrite inside is backed up
       def export(result, path):
           ...

       @protect(only=["runs/exp1"])          # only writes under this directory
       def train(...):
           ...

3. Explicitly opt into the global patch (what the module used to do on import)::

       import dlt.never_overwrite as nw
       nw.install(policy=nw.BackupPolicy(keep=5))
       ...
       nw.uninstall()

   ``DLT_NEVER_OVERWRITE=1`` also turns the global install on at import time.

Design notes / fixed bugs
-------------------------
* ``os.replace`` within one directory is atomic and costs no extra space, so we
  always back up (the original only backed up when the disk had > 16 GiB free,
  and otherwise **silently overwrote** the file).
* ``open(file, mode, *args)`` extras (buffering, encoding, ...) are forwarded;
  file descriptors and non-path objects pass through untouched.
* Symlinks are skipped (moving a link would change what it points at), and
  ``mode="x"`` / ``mode="a"`` need no backup.
* Backups are ``<stem>.bak-<utc-timestamp>-<id><ext>``.  ``keep``,
  ``max_total_bytes`` and a ``min_free_bytes`` disk-space guard (default
  512 MiB) bound how much disk they can consume.
* ``protect`` only affects writes made while it runs; packages doing unrelated
  I/O outside the scope (or to files not matching ``only``) use normal ``open``.

No third-party dependencies.
"""

from __future__ import annotations

import builtins
import contextlib
import functools
import inspect
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .naming import short_id, timestamp, unique_path

__all__ = [
    "BackupPolicy",
    "backup",
    "safe_open",
    "protect",
    "protect_scope",
    "protect_paths",
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
#: stack of active ``protect`` scopes: (policy, path matcher or None)
_SCOPE_STACK: list[tuple["BackupPolicy", Callable[[str], bool] | None]] = []


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


def _build_matcher(only: Iterable[Any] | None) -> Callable[[str], bool] | None:
    """Turn ``only`` into a predicate over absolute paths.

    Entries may be file paths (exact match), directories (prefix match) or
    callables taking the absolute path.
    """
    if only is None:
        return None
    resolved: list[tuple[str, Any]] = []
    for item in only:
        if callable(item):
            resolved.append(("call", item))
        else:
            absolute = os.path.abspath(os.fspath(item))
            resolved.append(("dir", absolute) if os.path.isdir(absolute) else ("file", absolute))

    def match(path: str) -> bool:
        absolute = os.path.abspath(path)
        for kind, value in resolved:
            if kind == "call":
                try:
                    if value(absolute):
                        return True
                except Exception:
                    continue
            elif kind == "dir":
                if absolute == value or absolute.startswith(value + os.sep):
                    return True
            elif absolute == value:
                return True
        return False

    return match


def _active_policy() -> BackupPolicy:
    """Policy for an explicit :func:`backup` call.

    Prefers the innermost ``protect`` scope, then a global ``install``, and
    otherwise an ordinary enabled policy (calling ``backup`` directly means the
    caller *wants* a backup).  :func:`safe_open` does not use this helper: it
    checks the scope/install state itself and stays inert when neither is set.
    """
    with _LOCK:
        if _SCOPE_STACK:
            return _SCOPE_STACK[-1][0]
        if _INSTALLED and _POLICY is not None:
            return _POLICY
        return BackupPolicy()


def backup(
    path: str | os.PathLike[str],
    *,
    policy: BackupPolicy | None = None,
) -> str | None:
    """Move ``path`` aside to a timestamped backup and return the new path.

    Returns ``None`` when nothing was done (missing file, empty file, disabled
    policy, ...).  Safe to call repeatedly.
    """
    policy = (policy or _active_policy()).normalized()
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
    """Drop-in :func:`open` that backs up truncating writes.

    Only active inside a :func:`protect_scope` / :func:`protect` or after an
    explicit :func:`install`; other callers get the original ``open``.
    """
    if (
        isinstance(mode, str)
        and "w" in mode
        and not isinstance(file, int)
        and file is not None
        and isinstance(file, (str, bytes, os.PathLike))
    ):
        path = os.fsdecode(os.fspath(file))
        with _LOCK:
            if _SCOPE_STACK:
                policy, matcher = _SCOPE_STACK[-1]
            elif _INSTALLED:
                policy, matcher = (_POLICY or BackupPolicy()), None
            else:
                policy, matcher = BackupPolicy(enabled=False), None
        if policy.enabled and (matcher is None or matcher(path)):
            with _LOCK:
                backup(path, policy=policy)
    return _ORIGINAL_OPEN(file, mode, *args, **kwargs)


@contextlib.contextmanager
def protect_scope(
    *,
    policy: BackupPolicy | None = None,
    only: Iterable[Any] | None = None,
    also_io: bool = True,
) -> Any:
    """Context manager that replaces ``open`` only for its own duration.

    Parameters
    ----------
    policy:
        Backup policy used inside the scope.
    only:
        Restrict protection to these files/directories (or a predicate).  This
        is the safest way to make sure third-party code doing unrelated I/O is
        never touched.
    also_io:
        Also replace ``io.open`` (covers ``pathlib`` writes).
    """
    entry = ((policy or BackupPolicy()).normalized(), _build_matcher(only))
    with _LOCK:
        first = not _SCOPE_STACK
        _SCOPE_STACK.append(entry)
        if first and not _INSTALLED:
            builtins.open = safe_open
            if also_io and _io is not None:
                _io.open = safe_open
    try:
        yield
    finally:
        with _LOCK:
            _SCOPE_STACK.pop()
            if not _SCOPE_STACK and not _INSTALLED:
                if builtins.open is safe_open:
                    builtins.open = _ORIGINAL_OPEN
                if _io is not None and _io.open is safe_open and _ORIGINAL_IO_OPEN is not None:
                    _io.open = _ORIGINAL_IO_OPEN


def protect(
    func: Callable[..., Any] | None = None,
    *,
    policy: BackupPolicy | None = None,
    only: Iterable[Any] | None = None,
    also_io: bool = True,
) -> Callable[..., Any]:
    """Decorator form of :func:`protect_scope`.

    Works as ``@protect`` or ``@protect(policy=..., only=...)``.  The pristine
    ``open`` is restored when the call returns, including on exceptions.
    """

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with protect_scope(policy=policy, only=only, also_io=also_io):
                return function(*args, **kwargs)

        return wrapper

    return decorate(func) if func is not None else decorate


def protect_paths(*param_names: str, policy: BackupPolicy | None = None) -> Callable[..., Any]:
    """Decorator: back up path-valued arguments before calling the function.

    This does **not** patch anything -- it simply moves the named arguments
    aside first, then lets the function write normally.  Ideal for checkpoint /
    export helpers::

        @protect_paths("path")
        def save_checkpoint(state, path):
            torch.save(state, path)

    An argument may also be a list/tuple of paths.  ``None`` entries are
    ignored.
    """
    if not param_names:
        raise TypeError("protect_paths() needs at least one argument name")

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(function)

        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            for name in param_names:
                if name not in bound.arguments:
                    raise TypeError(
                        f"{function.__name__}() has no argument {name!r}; "
                        "check @protect_paths(...)"
                    )
                value = bound.arguments[name]
                paths = (
                    list(value)
                    if isinstance(value, (list, tuple, set, frozenset))
                    else [value]
                )
                for candidate in paths:
                    if candidate is None or not isinstance(
                        candidate, (str, bytes, os.PathLike)
                    ):
                        continue
                    with _LOCK:
                        backup(os.fsdecode(os.fspath(candidate)), policy=policy)
            return function(*args, **kwargs)

        return wrapper

    return decorate


def protect_torch() -> Any:
    """Back up the destination of ``torch.save(obj, path)`` (opt-in global patch).

    Modern PyTorch writes checkpoints through a C++ file writer that never
    touches Python's :func:`open`, so :func:`protect` alone does not cover it.
    Prefer :func:`protect_paths` on your own save function; use this only if you
    want every ``torch.save`` in the process protected.
    """
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
    """Explicitly patch ``open`` for the rest of the process.

    Importing the module no longer does this.  Returns the active policy.
    """
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
    """Undo :func:`install` (and any running scope) and restore ``open``."""
    global _INSTALLED, _POLICY
    with _LOCK:
        _INSTALLED = False
        _POLICY = None
        if not _SCOPE_STACK:
            if builtins.open is safe_open:
                builtins.open = _ORIGINAL_OPEN
            if _io is not None and _io.open is safe_open and _ORIGINAL_IO_OPEN is not None:
                _io.open = _ORIGINAL_IO_OPEN


def is_installed() -> bool:
    return _INSTALLED


def _env_wants_install() -> bool:
    value = os.environ.get("DLT_NEVER_OVERWRITE")
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on")


def _policy_from_env() -> BackupPolicy:
    keep = os.environ.get("DLT_NEVER_OVERWRITE_KEEP")
    min_free_mb = os.environ.get("DLT_NEVER_OVERWRITE_MIN_FREE_MB")
    max_total_mb = os.environ.get("DLT_NEVER_OVERWRITE_MAX_TOTAL_MB")
    return BackupPolicy(
        keep=int(keep) if keep and keep.isdigit() else None,
        min_free_bytes=(
            int(min_free_mb) * 1024 * 1024
            if min_free_mb is not None and min_free_mb.lstrip("-").isdigit()
            else BackupPolicy.min_free_bytes
        ),
        max_total_bytes=(
            int(max_total_mb) * 1024 * 1024
            if max_total_mb is not None and max_total_mb.isdigit()
            else None
        ),
        verbose=os.environ.get("DLT_NEVER_OVERWRITE_VERBOSE", "0").lower()
        in ("1", "true", "on", "yes"),
    )


# Importing is a no-op by default.  Set DLT_NEVER_OVERWRITE=1 (or call install())
# if you really want the whole process patched.
if _env_wants_install():
    install(policy=_policy_from_env())
