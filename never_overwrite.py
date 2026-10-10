"""Backwards-compatible shim for the old top-level ``never_overwrite`` module.

.. important::
   Importing this module **no longer patches** Python's ``open``.  The old
   import-time monkey patch affected every package in the process, so it is now
   opt-in.  Use one of these instead:

   * ``@protect_paths("path")`` -- back up specific file arguments (no patch);
   * ``@protect`` / ``with protect_scope(...)`` -- scoped patch, restored on exit;
   * ``install()`` -- explicit, process-wide patch.

Set ``DLT_NEVER_OVERWRITE=1`` before importing if you really want the global
patch applied automatically.
"""

from dlt.never_overwrite import (  # noqa: F401
    BackupPolicy,
    backup,
    install,
    is_installed,
    protect,
    protect_numpy,
    protect_paths,
    protect_safetensors,
    protect_scope,
    protect_torch,
    safe_open,
    uninstall,
)

__all__ = [
    "BackupPolicy",
    "backup",
    "install",
    "is_installed",
    "protect",
    "protect_numpy",
    "protect_paths",
    "protect_safetensors",
    "protect_scope",
    "protect_torch",
    "safe_open",
    "uninstall",
]
