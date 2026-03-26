"""Small file and logging helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def write_records_parquet(records: list[dict[str, Any]], path: str | Path) -> Path:
    output_path = ensure_parent(path)
    pd.DataFrame.from_records(records).to_parquet(output_path, index=False)
    return output_path

