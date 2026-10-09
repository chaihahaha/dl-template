"""Backwards-compatible shim for the old top-level ``plot_utils`` module.

Prefer ``from dlt.plot_utils import packplot, retrieve_plot, retrieve_data`` or
``from dlt import packplot``.  This file only re-exports the new API.
"""

from dlt.plot_utils import (  # noqa: F401
    KEY,
    PlotBundle,
    embed_metadata,
    extract_metadata,
    packplot,
    retrieve_code,
    retrieve_data,
    retrieve_plot,
    unpack,
)

__all__ = [
    "KEY",
    "PlotBundle",
    "embed_metadata",
    "extract_metadata",
    "packplot",
    "retrieve_code",
    "retrieve_data",
    "retrieve_plot",
    "unpack",
]
