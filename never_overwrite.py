"""Backwards-compatible shim.

The original template told you to put ``import never_overwrite`` on the first
line of your training script.  That still works: importing this module enables
the protection implemented in :mod:`dlt.never_overwrite`.

New code can equivalently write::

    import dlt.never_overwrite   # noqa: F401  (import for the side effect)

Set ``DLT_NEVER_OVERWRITE=0`` to import without patching ``open``.
"""

from dlt.never_overwrite import (  # noqa: F401
    BackupPolicy,
    backup,
    install,
    is_installed,
    protect_torch,
    safe_open,
    uninstall,
)

__all__ = [
    "BackupPolicy",
    "backup",
    "install",
    "is_installed",
    "protect_torch",
    "safe_open",
    "uninstall",
]
