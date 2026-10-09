"""Atomic file-write helpers.

The whole point of this module is that a crash (or ``Ctrl+C``, or a full disk)
must never leave a half-written file behind.  We therefore always write to a
temporary file in the *same directory* and then :func:`os.replace` it onto the
destination.  ``os.replace`` is atomic on POSIX and on Windows, so a reader
either sees the old file or the new file, never a truncated one.

A useful side effect: because the temporary file is brand new and the final
step uses ``os.replace`` (not ``open(path, "w")``), these helpers do **not**
trigger the backup logic in :mod:`dlt.never_overwrite`.  Log files and
checkpoints therefore do not accumulate thousands of ``.bak`` copies.
"""

from __future__ import annotations

import os
import tempfile
from typing import Union

__all__ = ["atomic_write_bytes", "atomic_write_text", "FileLike"]

FileLike = Union[str, "os.PathLike[str]"]


def _fsync_dir(directory: str) -> None:
    """Best-effort fsync of a directory so the rename itself is durable.

    Not every platform/filesystem supports opening a directory, hence the
    broad ``except``: durability is a bonus, correctness never depends on it.
    """
    if os.name == "nt":  # opening a directory is not allowed on Windows
        return
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(
    path: FileLike,
    data: bytes,
    *,
    fsync: bool = True,
    make_dirs: bool = True,
    mode: int | None = None,
) -> str:
    """Write ``data`` to ``path`` atomically.

    Parameters
    ----------
    path:
        Destination.  Parent directories are created when ``make_dirs``.
    fsync:
        Flush file contents (and the directory entry) to disk before
        returning.  Slower but survives power loss.
    mode:
        Optional permission bits for the final file (e.g. ``0o600`` for
        sensitive logs).  By default the temporary file's mode is used.
    """
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    if make_dirs:
        os.makedirs(directory, exist_ok=True)

    fd, tmp = tempfile.mkstemp(
        prefix="." + os.path.basename(path) + ".",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        # Never leak the temporary file, but never mask the original error.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    if fsync:
        _fsync_dir(directory)
    return path


def atomic_write_text(
    path: FileLike,
    text: str,
    *,
    encoding: str = "utf-8",
    errors: str = "strict",
    fsync: bool = True,
    make_dirs: bool = True,
    mode: int | None = None,
) -> str:
    """Text flavour of :func:`atomic_write_bytes`."""
    return atomic_write_bytes(
        path,
        text.encode(encoding, errors),
        fsync=fsync,
        make_dirs=make_dirs,
        mode=mode,
    )
