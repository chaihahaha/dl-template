"""Shared pytest configuration.

* Keep pytest's temporary directories inside the project (``tmp/pytest``)
  instead of ``/tmp``, per the repository's AGENTS.md rules.
* Render matplotlib headless, with its font cache inside ``./cache``.
* Disable the auto-install of :mod:`dlt.never_overwrite` at import time so the
  test-suite can install/uninstall it explicitly and stay isolated.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

os.environ.setdefault("DLT_NEVER_OVERWRITE", "0")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache" / "matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:  # matplotlib is an optional test dependency
    import matplotlib  # noqa: E402

    matplotlib.use("Agg")
except ImportError:  # pragma: no cover
    pass
