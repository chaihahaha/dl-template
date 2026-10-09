"""Backwards-compatible shim for the old top-level ``yaml_logging`` module.

The old file was a snippet with undefined names and import-time side effects
(it dumped ``os.environ`` and dropped into ``pdb`` at exit).  Use the new,
importable API instead::

    from dlt import YamlLogger

    with YamlLogger("my-exp", config={"lr": 1e-3}) as log:
        log.log(loss=0.1)
"""

from dlt.yaml_logging import (  # noqa: F401
    YamlLogger,
    filter_env,
    install_excepthook,
    latest_log,
    load_log,
    load_logs,
    yaml_dump,
    yaml_load,
)

__all__ = [
    "YamlLogger",
    "filter_env",
    "install_excepthook",
    "latest_log",
    "load_log",
    "load_logs",
    "yaml_dump",
    "yaml_load",
]
