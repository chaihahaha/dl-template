"""dlt - a minimalist deep-learning experiment template.

Importing this package has **no side effects** and only requires the standard
library plus PyYAML for logging.  Heavy helpers are imported lazily:

* :mod:`dlt.naming`            - automatic, collision-free run/file naming
* :mod:`dlt.yaml_logging`      - crash-safe YAML experiment logger
* :mod:`dlt.never_overwrite`   - backup-on-truncate ``open`` (import to enable)
* :mod:`dlt.plot_utils`        - images that carry their own plotting code
  (needs Pillow, and ``dill``/``matplotlib`` optionally)

Example
-------
>>> from dlt import ExperimentNamer, YamlLogger
>>> run = ExperimentNamer("runs", name="my-exp").new_run({"lr": 1e-3})  # doctest: +SKIP
>>> with YamlLogger("my-exp", run=run, save_every=10) as log:          # doctest: +SKIP
...     log.log(loss=0.5, acc=0.9)
"""

from __future__ import annotations

from ._atomic import atomic_write_bytes, atomic_write_text
from .naming import (
    ExperimentNamer,
    RunPaths,
    auto_name,
    short_id,
    slugify,
    timestamp,
    unique_path,
    utcnow,
)
from .yaml_logging import (
    YamlLogger,
    install_excepthook,
    latest_log,
    load_log,
    load_logs,
    yaml_dump,
    yaml_load,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # atomic io
    "atomic_write_bytes",
    "atomic_write_text",
    # naming
    "ExperimentNamer",
    "RunPaths",
    "auto_name",
    "short_id",
    "slugify",
    "timestamp",
    "unique_path",
    "utcnow",
    # logging
    "YamlLogger",
    "install_excepthook",
    "latest_log",
    "load_log",
    "load_logs",
    "yaml_dump",
    "yaml_load",
    # lazy
    "packplot",
    "retrieve_plot",
    "retrieve_data",
    "retrieve_code",
]

_LAZY_PLOT = {
    "packplot",
    "retrieve_plot",
    "retrieve_data",
    "retrieve_code",
    "unpack",
    "PlotBundle",
    "embed_metadata",
    "extract_metadata",
}


def __getattr__(name: str):
    """Lazily expose :mod:`dlt.plot_utils` so Pillow is only needed when used."""
    if name in _LAZY_PLOT:
        from . import plot_utils

        return getattr(plot_utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
