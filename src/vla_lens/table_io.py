"""Shared table I/O helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def read_optional_parquet(path: Path, *, context: str) -> pd.DataFrame:
    """Read a parquet table, returning an empty frame when an optional table is absent.

    Corrupt or unreadable tables are still optional for dashboard payload assembly,
    but they should be visible during debugging and data-trust checks.
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        LOGGER.warning("Failed to read %s parquet table at %s", context, path, exc_info=True)
        return pd.DataFrame()
