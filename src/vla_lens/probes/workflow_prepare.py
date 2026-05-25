"""Probe row metadata, filtering, and split helpers."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.artifacts import LensArtifact
from vla_lens.probes.workflow_types import INTERACTION_METRICS_ARTIFACT_TYPE
from vla_lens.traces import TraceDataset


def _attach_episode_metadata(rows: pd.DataFrame, dataset: TraceDataset) -> pd.DataFrame:
    episode_index = dataset.episode_index.copy()
    if episode_index.empty:
        return rows.copy()
    episode_index = _merge_probe_split_sidecar(episode_index, dataset)
    episode_index = _merge_interaction_metrics(episode_index, dataset)
    duplicate_columns = [
        column
        for column in episode_index.columns
        if column in rows.columns and column not in {"trace_id"}
    ]
    episode_index = episode_index.drop(columns=duplicate_columns)
    return rows.merge(episode_index, on="trace_id", how="left")


def _merge_probe_split_sidecar(
    episode_index: pd.DataFrame,
    dataset: TraceDataset,
) -> pd.DataFrame:
    """Attach capture-plan split metadata when a PI0.5 sidecar is present."""
    split_path = dataset.root / "probe_splits.csv"
    if not split_path.exists() or "trace_id" not in episode_index:
        return episode_index
    split_frame = pd.read_csv(split_path)
    if split_frame.empty or "trace_id" not in split_frame:
        return episode_index

    split_frame = split_frame.drop_duplicates(subset=["trace_id"], keep="last")
    wanted = [
        column
        for column in [
            "trace_id",
            "benchmark",
            "split",
            "seed",
            "capture_profile",
            "capture_design",
            "trace_variant",
            "counterfactual_group_id",
            "counterfactual_role",
            "counterfactual_type",
            "pair_index",
            "paired_trace_id",
            "changed_fields",
            "matched_fields",
            "target_object_id",
            "counterfactual_target_object_id",
        ]
        if column in split_frame
    ]
    split_frame = split_frame[wanted].copy()
    duplicate_columns = [
        column
        for column in split_frame.columns
        if column in episode_index.columns and column != "trace_id"
    ]
    if duplicate_columns:
        split_frame = split_frame.rename(
            columns={column: f"split_sidecar_{column}" for column in duplicate_columns}
        )
    merged = episode_index.merge(split_frame, on="trace_id", how="left")
    if "benchmark" not in merged and "env_id" in merged:
        merged["benchmark"] = merged["env_id"]
    return merged


def _merge_interaction_metrics(
    episode_index: pd.DataFrame,
    dataset: TraceDataset,
) -> pd.DataFrame:
    labels = _latest_interaction_labels(dataset)
    if labels.empty or "trace_id" not in labels or "trace_id" not in episode_index:
        return episode_index
    labels = labels.drop_duplicates(subset=["trace_id"], keep="last")
    labels = _add_derived_interaction_label_columns(labels, dataset)
    merged = episode_index.merge(labels, on="trace_id", how="left", suffixes=("", "__derived"))
    for column in list(merged.columns):
        if not column.endswith("__derived"):
            continue
        base = column.removesuffix("__derived")
        derived = merged.pop(column)
        if base in merged:
            missing = merged[base].isna() | (merged[base].astype(str) == "")
            merged.loc[missing, base] = derived.loc[missing]
            conflict = (
                (~missing)
                & derived.notna()
                & (derived.astype(str) != merged[base].astype(str))
            )
            if bool(conflict.any()):
                merged[f"derived_{base}"] = derived
        else:
            merged[base] = derived
    return merged


def _add_derived_interaction_label_columns(
    labels: pd.DataFrame,
    dataset: TraceDataset,
) -> pd.DataFrame:
    labels = labels.copy()
    for column in [
        "primary_target_object",
        "first_moved_object",
        "first_lifted_object",
        "first_contacted_object",
    ]:
        if column in labels:
            labels[f"{column}_base"] = labels[column].map(_base_object_name)
    if {"first_contacted_object", "target_objects"}.issubset(labels.columns):
        labels["first_contacted_is_target"] = [
            _object_in_targets(obj, targets)
            for obj, targets in zip(
                labels["first_contacted_object"],
                labels["target_objects"],
                strict=False,
            )
        ]
    object_metrics = _latest_interaction_object_metrics(dataset)
    if not object_metrics.empty and "trace_id" in object_metrics:
        target_rows = object_metrics.loc[
            object_metrics.get("is_target_object", pd.Series(False, index=object_metrics.index))
            .fillna(False)
            .astype(bool)
        ]
        if not target_rows.empty:
            flags = (
                target_rows.groupby("trace_id", dropna=False)[["moved", "lifted", "contacted"]]
                .any()
                .rename(
                    columns={
                        "moved": "target_moved",
                        "lifted": "target_lifted",
                        "contacted": "target_contacted",
                    }
                )
                .reset_index()
            )
            labels = labels.merge(flags, on="trace_id", how="left")
    for column in [
        "target_moved",
        "target_lifted",
        "target_contacted",
        "first_contacted_is_target",
    ]:
        if column in labels:
            labels[column] = labels[column].where(labels[column].notna(), False).astype(bool)
    return labels


def _latest_interaction_object_metrics(dataset: TraceDataset) -> pd.DataFrame:
    artifact = _latest_interaction_artifact(dataset)
    if artifact is None:
        return pd.DataFrame()
    outputs = dict(artifact.method.get("outputs") or {})
    path = outputs.get("object_metrics")
    if not path:
        return pd.DataFrame()
    table_path = dataset.root / str(path)
    if not table_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(table_path)


def _latest_interaction_labels(dataset: TraceDataset) -> pd.DataFrame:
    artifact = _latest_interaction_artifact(dataset)
    if artifact is None:
        return pd.DataFrame()
    outputs = dict(artifact.method.get("outputs") or {})
    path = outputs.get("episode_labels")
    if not path:
        return pd.DataFrame()
    table_path = dataset.root / str(path)
    if table_path.exists():
        return pd.read_parquet(table_path)
    return pd.DataFrame()


def _latest_interaction_artifact(dataset: TraceDataset) -> LensArtifact | None:
    table = dataset.artifact_index
    if table.empty or "artifact_type" not in table:
        return None
    matches = table.loc[
        table["artifact_type"].astype(str) == INTERACTION_METRICS_ARTIFACT_TYPE
    ].copy()
    if matches.empty:
        return None
    matches = matches.sort_values("created_utc", ascending=False, na_position="last")
    for artifact_id in matches["artifact_id"].astype(str):
        try:
            return dataset.load_artifact(artifact_id)
        except (FileNotFoundError, KeyError):
            continue
    return None


def _base_object_name(value: Any) -> str:
    if value is None or isinstance(value, (list, tuple, dict, set)):
        text = ""
    else:
        try:
            text = "" if pd.isna(value) else str(value)
        except (TypeError, ValueError):
            text = str(value)
    parts = text.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return text


def _object_in_targets(obj: Any, target_objects: Any) -> bool:
    obj_base = _base_object_name(obj)
    targets = _json_load_list(target_objects)
    return bool(obj_base) and any(obj_base == _base_object_name(target) for target in targets)


def _json_load_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _ensure_split(
    rows: pd.DataFrame,
    split_column: str,
    *,
    train_value: str,
    test_value: str,
    split_kind: str,
) -> pd.DataFrame:
    rows = rows.copy()
    if split_column in rows and rows[split_column].notna().any():
        return rows

    traces = sorted(str(value) for value in rows["trace_id"].dropna().unique())
    if len(traces) <= 1:
        rows[split_column] = train_value
        return rows
    test_traces = _test_traces(rows, traces, split_kind)
    rows[split_column] = np.where(
        rows["trace_id"].astype(str).isin(test_traces),
        test_value,
        train_value,
    )
    return rows


def _apply_row_filters(
    X: np.ndarray,
    rows: pd.DataFrame,
    row_filter: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    filters = _normalize_row_filters(row_filter)
    if not filters:
        return X, rows, {"filters": [], "input_rows": int(len(rows)), "output_rows": int(len(rows))}
    mask = pd.Series(True, index=rows.index)
    applied: list[dict[str, Any]] = []
    for spec in filters:
        column = str(spec.get("column") or "")
        if column not in rows:
            raise KeyError(f"Row filter column {column!r} is not present in probe rows")
        before = int(mask.sum())
        next_mask = _row_filter_mask(rows[column], spec)
        mask &= next_mask
        applied.append(
            {
                **dict(spec),
                "input_rows": before,
                "output_rows": int(mask.sum()),
            }
        )
    kept = mask.to_numpy(dtype=bool)
    return (
        X[kept],
        rows.loc[mask].reset_index(drop=True),
        {"filters": applied, "input_rows": int(len(rows)), "output_rows": int(kept.sum())},
    )


def _normalize_row_filters(
    row_filter: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
) -> list[Mapping[str, Any]]:
    if row_filter is None:
        return []
    if isinstance(row_filter, Mapping):
        if "all" in row_filter and isinstance(row_filter["all"], Sequence):
            return [dict(item) for item in row_filter["all"]]
        return [dict(row_filter)]
    return [dict(item) for item in row_filter]


def _row_filter_mask(values: pd.Series, spec: Mapping[str, Any]) -> pd.Series:
    op = str(spec.get("op") or spec.get("operator") or "==")
    if op in {"notna", "present"}:
        return values.notna() & (values.astype(str) != "")
    if op in {"isna", "missing"}:
        return values.isna() | (values.astype(str) == "")
    if op == "truthy":
        return values.fillna(False).astype(bool)
    if op == "falsy":
        return ~values.fillna(False).astype(bool)
    expected = spec.get("value")
    if op in {"==", "="}:
        return values.map(_coerce_filter_value) == _coerce_filter_value(expected)
    if op in {"!=", "ne"}:
        return values.map(_coerce_filter_value) != _coerce_filter_value(expected)
    if op == "in":
        allowed = {_coerce_filter_value(item) for item in _filter_values(spec, expected)}
        return values.map(_coerce_filter_value).isin(allowed)
    if op in {"notin", "not_in"}:
        blocked = {_coerce_filter_value(item) for item in _filter_values(spec, expected)}
        return ~values.map(_coerce_filter_value).isin(blocked)
    if op in {">", ">=", "<", "<="}:
        numeric = pd.to_numeric(values, errors="coerce")
        threshold = float(expected)
        if op == ">":
            return numeric > threshold
        if op == ">=":
            return numeric >= threshold
        if op == "<":
            return numeric < threshold
        return numeric <= threshold
    raise ValueError(f"Unknown row filter operator: {op!r}")


def _filter_values(spec: Mapping[str, Any], fallback: Any) -> list[Any]:
    values = spec.get("values", fallback)
    if values is None:
        return []
    if isinstance(values, str) or not isinstance(values, Sequence):
        return [values]
    return list(values)


def _coerce_filter_value(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"none", "null", "nan"}:
            return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _apply_missing_policy(
    X: np.ndarray,
    rows: pd.DataFrame,
    target_name: str,
    *,
    policy: str,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    missing = rows[target_name].isna() | (rows[target_name].astype(str) == "")
    missing_count = int(missing.sum())
    summary = {
        "policy": policy,
        "missing_rows": missing_count,
        "input_rows": int(len(rows)),
        "output_rows": int(len(rows) - missing_count if policy == "drop" else len(rows)),
    }
    if missing_count == 0:
        return X, rows, summary
    if policy == "drop":
        kept = (~missing).to_numpy(dtype=bool)
        return X[kept], rows.loc[~missing].reset_index(drop=True), summary
    if policy == "skip_probe":
        raise ValueError(
            f"Probe skipped because target {target_name!r} has {missing_count} missing rows"
        )
    raise ValueError(
        f"Target {target_name!r} has {missing_count} missing rows; "
        "set target.missing_policy to 'drop' or 'skip_probe' for sparse targets."
    )


def _test_traces(rows: pd.DataFrame, traces: list[str], split_kind: str) -> set[str]:
    test_count = max(1, int(round(len(traces) * 0.2)))
    if split_kind in {"random_episode", "episode", "auto"}:
        return set(traces[-test_count:])
    column_by_kind = {
        "heldout_benchmark": "benchmark",
        "heldout_env": "env_id",
        "heldout_task": "task_id",
        "heldout_object": "target_object",
        "heldout_target_object": "target_object",
    }
    column = column_by_kind.get(split_kind)
    if column is None:
        if split_kind.startswith("heldout_"):
            column = split_kind.removeprefix("heldout_")
        else:
            return set(traces[-test_count:])
    if column not in rows or rows[column].dropna().empty:
        return set(traces[-test_count:])
    trace_groups = (
        rows[["trace_id", column]]
        .dropna()
        .drop_duplicates()
        .sort_values(["trace_id", column])
        .groupby(column)["trace_id"]
        .apply(lambda values: sorted(str(value) for value in values.unique()))
    )
    groups = list(trace_groups.items())
    if not groups:
        return set(traces[-test_count:])
    target = max(1, int(round(len(traces) * 0.2)))
    chosen: set[str] = set()
    for _, group_traces in sorted(groups, key=lambda item: (len(item[1]), str(item[0]))):
        if not chosen or len(chosen) < target:
            chosen.update(group_traces)
    if len(chosen) >= len(traces):
        return set(traces[-test_count:])
    return chosen
