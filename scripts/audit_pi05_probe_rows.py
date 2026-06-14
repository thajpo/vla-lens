"""Audit PI0.5 feature-cache row grain and attached object-flow labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import zarr

from vla_lens.probes.workflow_prepare import _attach_episode_metadata
from vla_lens.traces import TraceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Trace dataset root")
    parser.add_argument(
        "--cache-key",
        default=None,
        help="Feature cache key under .vla_cache/features. Defaults to the largest cache.",
    )
    parser.add_argument("--top", type=int, default=12, help="Number of value-count rows to print")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = TraceDataset.open(args.root)
    cache_dir = _feature_cache_dir(args.root, args.cache_key)
    rows = pd.read_parquet(cache_dir / "rows.parquet")
    x_path = cache_dir / "X.zarr"
    x_shape = tuple(zarr.open_array(str(x_path), mode="r").shape) if x_path.exists() else ()
    attached = _attach_episode_metadata(rows.copy(), dataset)

    print(f"cache_key={cache_dir.name}")
    print(f"feature_cache={cache_dir}")
    print(f"X_shape={x_shape}")
    print(f"rows={len(rows)}")
    print(f"columns={list(rows.columns)}")
    print(f"unique_episodes={_nunique(rows, 'trace_id')}")
    print(f"unique_policy_calls={_unique_policy_calls(rows)}")
    print(f"unique_layers={_nunique(rows, 'layer')}")
    print(f"rows_per_layer={_counts(rows, 'layer')}")
    print(f"rows_per_activation={_counts(rows, 'activation')}")
    print(f"rows_per_generation_step={_counts(rows, 'generation_step')}")
    print(f"policy_call_index_counts={_counts(rows, 'policy_call_index')}")
    print(f"avg_rows_per_episode={len(rows) / max(1, _nunique(rows, 'trace_id')):.3f}")
    print(
        "avg_rows_per_episode_policy_call="
        f"{len(rows) / max(1, _episode_policy_call_count(rows)):.3f}"
    )
    for column in [
        "split",
        "task_phase",
        "next_manipulated_object",
        "active_manipulated_object",
        "active_receptacle_object",
        "is_pre_contact",
        "is_pre_motion",
    ]:
        if column in attached:
            print(f"{column}_counts={_counts(attached, column, top=args.top)}")


def _feature_cache_dir(root: Path, cache_key: str | None) -> Path:
    root = Path(root)
    features_dir = root / ".vla_cache" / "features"
    if cache_key:
        path = features_dir / cache_key
        if not (path / "rows.parquet").exists():
            raise FileNotFoundError(path / "rows.parquet")
        return path
    candidates = [
        path
        for path in features_dir.iterdir()
        if path.is_dir() and (path / "rows.parquet").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No feature caches under {features_dir}")
    return max(candidates, key=lambda path: len(pd.read_parquet(path / "rows.parquet")))


def _counts(frame: pd.DataFrame, column: str, *, top: int | None = None) -> dict[str, int]:
    if column not in frame:
        return {}
    counts = frame[column].fillna("<NA>").astype(str).value_counts(dropna=False)
    if top is not None:
        counts = counts.head(top)
    return {str(key): int(value) for key, value in counts.items()}


def _nunique(frame: pd.DataFrame, column: str) -> int:
    return int(frame[column].nunique()) if column in frame else 0


def _unique_policy_calls(frame: pd.DataFrame) -> int:
    if not {"trace_id", "policy_call_index"}.issubset(frame.columns):
        return 0
    return int(len(frame[["trace_id", "policy_call_index"]].drop_duplicates()))


def _episode_policy_call_count(frame: pd.DataFrame) -> int:
    if not {"trace_id", "policy_call_index"}.issubset(frame.columns):
        return 0
    return int(len(frame[["trace_id", "policy_call_index"]].drop_duplicates()))


if __name__ == "__main__":
    main()
