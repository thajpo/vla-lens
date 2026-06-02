"""Cheap filesystem signatures for dataset-level server caches."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from vla_lens.traces import TraceBundle


def _dataset_signature(root: Path) -> tuple[int, int]:
    """Cheap cache key for dataset-level metadata endpoints."""
    if (root / TraceBundle.MANIFEST).exists():
        paths = [root / TraceBundle.MANIFEST, root / TraceBundle.ARTIFACT_INDEX]
        trace_count = 1
    else:
        paths = [
            root / "vla_lens" / "tables" / "index_manifest.json",
            root / "vla_lens" / "tables" / "episode_index.parquet",
            root / "vla_lens" / "tables" / "model_site_index.parquet",
            root / "vla_lens" / "tables" / "artifact_index.parquet",
            root / "vla_lens" / "tables" / "probe_predictions.parquet",
            *_workbench_signature_paths(root),
        ]
        trace_count = _dataset_trace_count_hint(root)
    existing = [path for path in paths if path.exists()]
    latest_mtime = max((path.stat().st_mtime_ns for path in existing), default=0)
    return trace_count, latest_mtime


def _lerobot_signature_paths(root: Path) -> list[Path]:
    """Return source paths used by index builders and older diagnostics.

    Normal dashboard cache signatures intentionally do not call this helper so
    serving an indexed dataset does not recurse through all episode bundles.
    """
    paths: list[Path] = []
    for pattern in (
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.jsonl",
        "meta/tasks.parquet",
        "meta/episodes/**/*.parquet",
        "vla_lens/overlay.json",
        "vla_lens/tables/*.parquet",
        "*/meta/info.json",
        "*/meta/stats.json",
        "*/meta/tasks.jsonl",
        "*/meta/tasks.parquet",
        "*/meta/episodes/**/*.parquet",
        "*/vla_lens/overlay.json",
        "*/vla_lens/tables/*.parquet",
        "**/vla_lens/episodes/*/manifest.json",
    ):
        paths.extend(root.glob(pattern))
    return paths


def _workbench_signature_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory in (root / "workbench", root / "vla_lens" / "workbench"):
        if directory.exists():
            paths.extend(directory.rglob("*.json"))
    return paths


def _dataset_trace_count_hint(root: Path) -> int:
    refs_paths = _unique_paths(
        [
            root / "vla_lens" / "tables" / "episode_refs.parquet",
            *list(root.glob("**/vla_lens/tables/episode_refs.parquet")),
        ]
    )
    refs_count = 0
    for path in refs_paths:
        if not path.exists():
            continue
        try:
            refs_count += int(len(pd.read_parquet(path, columns=["trace_id"])))
        except Exception:
            continue
    if refs_count:
        return refs_count
    episode_plan = root / "episode_plan.csv"
    if episode_plan.exists():
        try:
            with episode_plan.open("r", encoding="utf-8") as handle:
                return max(0, sum(1 for _line in handle) - 1)
        except OSError:
            return 0
    return 0


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out
