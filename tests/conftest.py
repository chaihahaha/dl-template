"""Shared pytest configuration.

* Keep pytest's temporary directories inside the project (``tmp/pytest``)
  instead of ``/tmp``, per the repository's AGENTS.md rules.
* Render matplotlib headless, with its font cache inside ``./cache``.
* Keep :mod:`dlt.never_overwrite` inert; importing it no longer patches ``open``,
  and each test installs/uninstalls explicitly for isolation.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Belt and braces: `import never_overwrite` must not patch open in the tests.
os.environ["DLT_NEVER_OVERWRITE"] = "0"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache" / "matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:  # matplotlib is an optional test dependency
    import matplotlib  # noqa: E402

    matplotlib.use("Agg")
except ImportError:  # pragma: no cover
    pass
