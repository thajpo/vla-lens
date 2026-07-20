"""Probe row metadata, filtering, and split helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.artifacts import LensArtifact
from vla_lens.probes.workflow_types import (
    INTERACTION_METRICS_ARTIFACT_TYPE,
    OBJECT_FLOW_ARTIFACT_TYPE,
    POLICY_CALL_LABELS_ARTIFACT_TYPE,
)
from vla_lens.traces import TraceDataset


def _attach_episode_metadata(rows: pd.DataFrame, dataset: TraceDataset) -> pd.DataFrame:
    episode_index = dataset.episode_index.copy()
    if episode_index.empty:
        return _merge_object_flow_timestep_labels(rows.copy(), dataset)
    episode_index = _merge_probe_split_sidecar(episode_index, dataset)
    episode_index = _merge_interaction_metrics(episode_index, dataset)
    duplicate_columns = [
        column
        for column in episode_index.columns
        if column in rows.columns and column not in {"trace_id"}
    ]
    episode_index = episode_index.drop(columns=duplicate_columns)
    merged = rows.merge(episode_index, on="trace_id", how="left")
    merged = _merge_object_flow_timestep_labels(merged, dataset)
    merged = _merge_policy_call_labels(merged, dataset)
    merged = _add_target_role_columns(merged)
    return _add_temporal_target_event_columns(merged)


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
            onset_columns = {
                "movement_onset_timestep": "target_first_motion_timestep",
                "lift_onset_timestep": "target_first_lift_timestep",
                "contact_onset_timestep": "target_first_contact_timestep",
            }
            available_onsets = [
                column for column in onset_columns if column in target_rows.columns
            ]
            if available_onsets:
                onset_frame = target_rows[["trace_id", *available_onsets]].copy()
                for column in available_onsets:
                    onset_frame[column] = pd.to_numeric(onset_frame[column], errors="coerce")
                onsets = (
                    onset_frame.groupby("trace_id", dropna=False)[available_onsets]
                    .min()
                    .rename(columns=onset_columns)
                    .reset_index()
                )
                labels = labels.merge(onsets, on="trace_id", how="left")
    for column in [
        "target_moved",
        "target_lifted",
        "target_contacted",
        "first_contacted_is_target",
    ]:
        if column in labels:
            labels[column] = labels[column].where(labels[column].notna(), False).astype(bool)
    return labels


def _add_temporal_target_event_columns(rows: pd.DataFrame) -> pd.DataFrame:
    """Add policy-call-local target event labels from per-episode target onsets."""
    event_columns = {
        "contact": "target_first_contact_timestep",
        "motion": "target_first_motion_timestep",
        "lift": "target_first_lift_timestep",
    }
    if not any(column in rows for column in event_columns.values()):
        return rows

    rows = rows.copy()
    base = _first_numeric_column(
        rows,
        [
            "policy_call_label_timestep",
            "observation_timestep",
            "timestep",
            "env_timestep_start",
        ],
    )
    span = _policy_call_span(rows)
    for event_name, column in event_columns.items():
        if column not in rows:
            continue
        event_time = pd.to_numeric(rows[column], errors="coerce")
        future = event_time.notna() & base.notna() & (event_time > base)
        rows[f"target_{event_name}_in_future"] = future
        for horizon in [1, 2]:
            rows[f"target_{event_name}_within_{horizon}_policy_calls"] = (
                future & (event_time <= base + (span * horizon))
            )
    return rows


def _add_target_role_columns(rows: pd.DataFrame) -> pd.DataFrame:
    if "target_objects" not in rows:
        return rows
    object_columns = [
        "next_manipulated_object",
        "active_manipulated_object",
        "current_contact_object",
        "current_moved_object",
        "current_lifted_object",
    ]
    available = [column for column in object_columns if column in rows]
    if not available:
        return rows
    rows = rows.copy()
    for column in available:
        present_column = f"{column.removesuffix('_object')}_present"
        rows[present_column] = rows[column].map(_object_present)
        role_column = f"{column.removesuffix('_object')}_is_target"
        rows[role_column] = [
            _object_in_targets(obj, targets)
            for obj, targets in zip(rows[column], rows["target_objects"], strict=False)
        ]
        if "primary_target_object" in rows:
            primary_role_column = f"{column.removesuffix('_object')}_is_primary_target"
            rows[primary_role_column] = [
                _object_matches(obj, target)
                for obj, target in zip(
                    rows[column],
                    rows["primary_target_object"],
                    strict=False,
                )
            ]
    return rows


def _first_numeric_column(rows: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    out = pd.Series(np.nan, index=rows.index, dtype="float64")
    for column in columns:
        if column not in rows:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        missing = out.isna()
        out.loc[missing] = values.loc[missing]
    return out


def _policy_call_span(rows: pd.DataFrame) -> pd.Series:
    span = pd.Series(1.0, index=rows.index, dtype="float64")
    if {"env_timestep_start", "env_timestep_end"}.issubset(rows.columns):
        start = pd.to_numeric(rows["env_timestep_start"], errors="coerce")
        end = pd.to_numeric(rows["env_timestep_end"], errors="coerce")
        derived = end - start + 1
        valid = derived.notna() & (derived > 0)
        span.loc[valid] = derived.loc[valid]
    return span


def _latest_interaction_object_metrics(dataset: TraceDataset) -> pd.DataFrame:
    artifact = _latest_interaction_artifact(dataset)
    if artifact is None:
        return pd.DataFrame()
    outputs = dict(artifact.method.get("outputs") or {})
    path = outputs.get("object_metrics")
    if not path:
        return pd.DataFrame()
    table_path = _artifact_output_path(dataset, str(path))
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
    table_path = _artifact_output_path(dataset, str(path))
    if table_path.exists():
        return pd.read_parquet(table_path)
    return pd.DataFrame()


def _merge_object_flow_timestep_labels(
    rows: pd.DataFrame,
    dataset: TraceDataset,
) -> pd.DataFrame:
    labels = _latest_object_flow_timestep_labels(dataset)
    keys = ["trace_id", "timestep"]
    if (
        labels.empty
        or any(key not in labels for key in keys)
        or any(key not in rows for key in keys)
    ):
        return rows
    label_columns = [
        column
        for column in [
            "trace_id",
            "timestep",
            "current_contact_object",
            "current_moved_object",
            "current_lifted_object",
            "next_contact_object",
            "next_moved_object",
            "next_lifted_object",
            "next_manipulated_object",
            "active_manipulated_object",
            "active_receptacle_object",
            "active_flow_step_index",
            "next_flow_step_index",
            "task_phase",
        ]
        if column in labels
    ]
    labels = labels[label_columns].drop_duplicates(subset=keys, keep="last")
    merged = rows.merge(labels, on=keys, how="left", suffixes=("", "__object_flow"))
    for column in list(merged.columns):
        if not column.endswith("__object_flow"):
            continue
        base = column.removesuffix("__object_flow")
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
                merged[f"object_flow_{base}"] = derived
        else:
            merged[base] = derived
    return merged


def _latest_object_flow_timestep_labels(dataset: TraceDataset) -> pd.DataFrame:
    artifact = latest_loadable_artifact(dataset, OBJECT_FLOW_ARTIFACT_TYPE)
    if artifact is None:
        return pd.DataFrame()
    outputs = dict(artifact.method.get("outputs") or {})
    path = outputs.get("timestep_labels")
    if not path:
        return pd.DataFrame()
    table_path = _artifact_output_path(dataset, str(path))
    if table_path.exists():
        return pd.read_parquet(table_path)
    return pd.DataFrame()


def _latest_object_roles(dataset: TraceDataset) -> pd.DataFrame:
    artifact = latest_loadable_artifact(dataset, OBJECT_FLOW_ARTIFACT_TYPE)
    if artifact is None:
        return pd.DataFrame()
    outputs = dict(artifact.method.get("outputs") or {})
    path = outputs.get("object_roles")
    if not path:
        return pd.DataFrame()
    table_path = _artifact_output_path(dataset, str(path))
    if table_path.exists():
        return pd.read_parquet(table_path)
    return pd.DataFrame()


def _filter_role_rows(rows: pd.DataFrame, role_filter: Mapping[str, Any]) -> pd.DataFrame:
    mask = pd.Series(True, index=rows.index)
    for column, expected in role_filter.items():
        if column not in rows:
            mask &= False
            continue
        if isinstance(expected, bool):
            mask &= rows[column].fillna(False).astype(bool) == expected
        elif isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
            mask &= rows[column].astype(str).isin([str(value) for value in expected])
        else:
            mask &= rows[column].astype(str) == str(expected)
    return rows.loc[mask]


def _merge_policy_call_labels(
    rows: pd.DataFrame,
    dataset: TraceDataset,
) -> pd.DataFrame:
    labels = _latest_policy_call_labels(dataset)
    keys = ["trace_id", "policy_call_index"]
    if (
        labels.empty
        or any(key not in labels for key in keys)
        or any(key not in rows for key in keys)
    ):
        return rows
    label_columns = [
        column
        for column in [
            "trace_id",
            "policy_call_index",
            "policy_call_id",
            "observation_timestep",
            "env_timestep_start",
            "env_timestep_end",
            "policy_call_label_timestep",
            "task_phase",
            "next_manipulated_object",
            "active_manipulated_object",
            "active_receptacle_object",
            "current_contact_object",
            "current_moved_object",
            "current_lifted_object",
            "next_flow_step_index",
            "active_flow_step_index",
            "next_object_flow_step_index",
            "first_contact_time_next_object",
            "first_motion_time_next_object",
            "first_lift_time_next_object",
            "is_pre_contact",
            "is_pre_motion",
            "is_pre_lift",
            "candidate_objects",
            "visible_candidate_objects",
            "visible_candidate_count",
        ]
        if column in labels
    ]
    labels = labels[label_columns].drop_duplicates(subset=keys, keep="last")
    return _merge_prefer_existing(rows, labels, on=keys, suffix="__policy_call")


def _latest_policy_call_labels(dataset: TraceDataset) -> pd.DataFrame:
    artifact = latest_loadable_artifact(dataset, POLICY_CALL_LABELS_ARTIFACT_TYPE)
    if artifact is None:
        return pd.DataFrame()
    outputs = dict(artifact.method.get("outputs") or {})
    path = outputs.get("policy_call_labels")
    if not path:
        return pd.DataFrame()
    table_path = _artifact_output_path(dataset, str(path))
    if table_path.exists():
        return pd.read_parquet(table_path)
    return pd.DataFrame()


def _merge_prefer_existing(
    rows: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    on: Sequence[str],
    suffix: str,
) -> pd.DataFrame:
    merged = rows.merge(labels, on=list(on), how="left", suffixes=("", suffix))
    for column in list(merged.columns):
        if not column.endswith(suffix):
            continue
        base = column.removesuffix(suffix)
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
                merged[f"{suffix.strip('_')}_{base}"] = derived
        else:
            merged[base] = derived
    return merged


def _artifact_output_path(dataset: TraceDataset, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    dataset_path = dataset.root / path
    if dataset_path.exists():
        return dataset_path
    return dataset._dataset_artifact_root() / path


def _latest_interaction_artifact(dataset: TraceDataset) -> LensArtifact | None:
    return latest_loadable_artifact(dataset, INTERACTION_METRICS_ARTIFACT_TYPE)


def latest_loadable_artifact(
    dataset: TraceDataset,
    artifact_type: str,
) -> LensArtifact | None:
    table = dataset.artifact_index
    if table.empty or "artifact_type" not in table:
        return None
    matches = table.loc[
        table["artifact_type"].astype(str) == artifact_type
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


def _object_matches(left: Any, right: Any) -> bool:
    left_base = _base_object_name(left)
    right_base = _base_object_name(right)
    return bool(left_base) and left_base == right_base


def _object_present(value: Any) -> bool:
    return bool(_base_object_name(value))


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


def _apply_row_expansion(
    X: np.ndarray,
    rows: pd.DataFrame,
    dataset: TraceDataset,
    row_expand: Mapping[str, Any] | None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    if not row_expand:
        return (
            X,
            rows,
            {"kind": "none", "input_rows": int(len(rows)), "output_rows": int(len(rows))},
        )
    kind = str(row_expand.get("kind") or row_expand.get("source") or "")
    if kind not in {"object_roles", "scene_objects"}:
        raise ValueError(f"Unknown row expansion kind: {kind!r}")
    roles = _latest_object_roles(dataset)
    if roles.empty:
        raise ValueError("Object-role row expansion requested, but no object_roles table exists.")
    if "trace_id" not in roles or "trace_id" not in rows:
        raise ValueError("Object-role row expansion requires trace_id in rows and object_roles.")

    role_rows = roles.copy()
    object_kind = row_expand.get("object_kind", "object")
    if object_kind is not None and "object_kind" in role_rows:
        role_rows = role_rows.loc[role_rows["object_kind"].astype(str) == str(object_kind)]
    role_filter = row_expand.get("role_filter")
    if isinstance(role_filter, Mapping):
        role_rows = _filter_role_rows(role_rows, role_filter)

    role_columns = [
        column
        for column in [
            "trace_id",
            "object_index",
            "object_name",
            "object_base_name",
            "object_kind",
            "prompt_mentioned",
            "role_manipulated",
            "role_receptacle",
            "role_fixture",
            "role_distractor",
            "observed_contacted",
            "observed_moved",
            "observed_lifted",
            "max_displacement",
            "max_xy_displacement",
            "max_z_delta",
        ]
        if column in role_rows
    ]
    role_rows = role_rows[role_columns].drop_duplicates(
        subset=[column for column in ["trace_id", "object_name"] if column in role_columns],
        keep="last",
    )
    prefix = str(row_expand.get("prefix") or "probe_object")
    renamed = {
        column: f"{prefix}_{column.removeprefix('object_')}"
        for column in role_rows.columns
        if column != "trace_id"
    }
    role_rows = role_rows.rename(columns=renamed)
    merged = rows.reset_index(names="__source_row_index").merge(
        role_rows,
        on="trace_id",
        how="inner",
    )
    if merged.empty:
        raise ValueError("Object-role row expansion produced no rows after joining by trace_id.")

    source_indices = merged.pop("__source_row_index").to_numpy(dtype=np.int64)
    expanded_X = X[source_indices]
    summary = {
        "kind": kind,
        "input_rows": int(len(rows)),
        "output_rows": int(len(merged)),
        "objects": int(len(role_rows)),
        "object_kind": object_kind,
        "role_filter": dict(role_filter) if isinstance(role_filter, Mapping) else None,
        "prefix": prefix,
    }
    return expanded_X, merged.reset_index(drop=True), summary


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
