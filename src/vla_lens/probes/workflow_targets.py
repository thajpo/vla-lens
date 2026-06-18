"""Probe target resolution helpers."""

from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np
import pandas as pd

from vla_lens.traces import TraceBundle, TraceDataset


def _normalize_target_spec(target: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(target, Mapping):
        spec = dict(target)
    else:
        spec = {"kind": str(target)}
    if "name" not in spec:
        spec["name"] = spec.get("kind") or spec.get("source") or "target"
    if "source" not in spec:
        spec["source"] = "row"
        spec["column"] = spec["name"]
    return spec


def _target_name(target_spec: Mapping[str, Any]) -> str:
    return str(target_spec.get("name") or target_spec.get("kind") or target_spec.get("source"))


def _resolve_probe_target(
    dataset: TraceDataset,
    rows: pd.DataFrame,
    target_spec: Mapping[str, Any],
) -> pd.DataFrame:
    target_name = _target_name(target_spec)
    source = str(target_spec.get("source") or "row")
    column = str(target_spec.get("column") or target_name)
    if source == "row" and column in rows:
        rows = rows.copy()
        rows[target_name] = _apply_target_transform(rows[column], target_spec)
        return rows
    if source == "row":
        raise KeyError(f"Probe target column '{column}' is not present in selected rows")

    resolved = rows.apply(
        lambda row: _resolve_target_value_or_missing(dataset, row, target_spec),
        axis=1,
    )
    out = rows.copy()
    out[target_name] = _apply_target_transform(resolved, target_spec)
    missing = int(out[target_name].isna().sum())
    if missing:
        policy = str(target_spec.get("missing_policy") or "error")
        if policy in {"drop", "skip_probe"}:
            return out
        raise ValueError(
            f"Target {target_name!r} could not be resolved for {missing} selected rows. "
            "Narrow the selector or choose a target with complete coverage."
        )
    return out


def _resolve_target_value_or_missing(
    dataset: TraceDataset,
    row: pd.Series,
    target_spec: Mapping[str, Any],
) -> Any:
    try:
        return _resolve_target_value(dataset, row, target_spec)
    except KeyError:
        policy = str(target_spec.get("missing_policy") or "error")
        if policy in {"drop", "skip_probe"}:
            return None
        raise


def _apply_target_transform(values: pd.Series, target_spec: Mapping[str, Any]) -> pd.Series:
    transform = target_spec.get("transform") or {"kind": "identity"}
    if isinstance(transform, str):
        transform = {"kind": transform}
    if not isinstance(transform, Mapping):
        raise TypeError(f"Target transform must be a mapping or string, got {type(transform)!r}")
    kind = str(transform.get("kind") or "identity")
    if kind == "identity":
        return values
    if kind == "threshold":
        threshold = transform.get("value", transform.get("threshold"))
        if threshold is None:
            raise ValueError("threshold target transform requires 'value' or 'threshold'")
        operator = str(transform.get("operator") or ">")
        numeric = pd.to_numeric(values)
        threshold_value = float(threshold)
        if operator == ">":
            return numeric > threshold_value
        if operator == ">=":
            return numeric >= threshold_value
        if operator == "<":
            return numeric < threshold_value
        if operator == "<=":
            return numeric <= threshold_value
        if operator in {"==", "="}:
            return numeric == threshold_value
        if operator == "!=":
            return numeric != threshold_value
        raise ValueError(f"Unknown threshold operator: {operator!r}")
    raise ValueError(f"Unknown target transform kind: {kind!r}")


def _resolve_target_value(
    dataset: TraceDataset,
    row: pd.Series,
    target_spec: Mapping[str, Any],
) -> Any:
    bundle = dataset.bundle(str(row["trace_id"]))
    source = str(target_spec.get("source") or "row")
    timestep = _target_timestep(row, target_spec)
    if source.startswith("evaluation."):
        metric = source.split(".", 1)[1]
        return _evaluation_value(bundle, metric, timestep, target_spec)
    if source == "evaluation":
        metric = str(target_spec.get("metric_name") or target_spec.get("metric") or "")
        return _evaluation_value(bundle, metric, timestep, target_spec)
    if source == "table":
        table = _bundle_table(bundle, str(target_spec["table"]))
        value_column = str(target_spec.get("column") or target_spec.get("value_column"))
        return _table_timestep_value(table, timestep, value_column)
    if source == "array":
        return _array_target_value(bundle, str(target_spec["array_id"]), timestep, target_spec, row)
    if source in {"executed_actions", "action"}:
        spec = {**dict(target_spec), "selector": {"action_dim": target_spec.get("action_dim", 0)}}
        array_id = "action" if _array_has_name(bundle, "action") else "executed_actions"
        return _array_target_value(bundle, array_id, timestep, spec, row)
    if source == "robot_state":
        array_id = _robot_array_id(bundle, str(target_spec.get("field") or ""))
        return _array_target_value(bundle, array_id, timestep, target_spec, row)
    if source == "scene_state":
        row_target_spec = {**dict(target_spec), "_row": row}
        array_id, object_index = _scene_array_id_and_object(bundle, row_target_spec)
        selector = {**dict(target_spec.get("selector") or {}), "object": object_index}
        value = _array_target_value(
            bundle,
            array_id,
            timestep,
            {**row_target_spec, "selector": selector},
            row,
        )
        relative_to = target_spec.get("relative_to")
        if relative_to is None:
            return value
        reference_spec = _relative_reference_spec(target_spec, relative_to)
        reference = _array_target_value(
            bundle,
            str(reference_spec["array_id"]),
            timestep,
            reference_spec,
            row,
        )
        return float(value) - float(reference)
    if source.startswith("array."):
        return _array_target_value(
            bundle,
            source.split(".", 1)[1],
            timestep,
            target_spec,
            row,
        )
    return _array_target_value(bundle, source, timestep, target_spec, row)


def _target_timestep(row: pd.Series, target_spec: Mapping[str, Any]) -> int:
    alignment = (
        target_spec.get("alignment") if isinstance(target_spec.get("alignment"), Mapping) else {}
    )
    offset = int(alignment.get("offset", target_spec.get("offset", 0)) or 0)
    base = row.get("timestep")
    if base is None or pd.isna(base):
        base = row.get("target_timestep", 0)
    return max(0, int(base) + offset)


def _evaluation_value(
    bundle: TraceBundle,
    metric_name: str,
    timestep: int,
    target_spec: Mapping[str, Any],
) -> Any:
    table = bundle.evaluation
    if table.empty or "metric_name" not in table:
        return None
    rows = table.loc[table["metric_name"].astype(str) == metric_name].copy()
    if rows.empty:
        return None
    if "timestep" in rows:
        exact = rows.loc[rows["timestep"].astype(int) == timestep]
        if not exact.empty:
            rows = exact
        else:
            prior = rows.loc[rows["timestep"].astype(int) <= timestep]
            if not prior.empty:
                rows = prior
            elif _allow_future_evaluation_label(target_spec):
                rows = rows
            else:
                return None
    value_column = str(target_spec.get("value_column") or "metric_value")
    if value_column not in rows:
        value_column = "passed" if "passed" in rows else rows.columns[-1]
    return rows.iloc[-1].get(value_column)


def _allow_future_evaluation_label(target_spec: Mapping[str, Any]) -> bool:
    alignment = (
        target_spec.get("alignment") if isinstance(target_spec.get("alignment"), Mapping) else {}
    )
    kind = str(alignment.get("kind") or target_spec.get("alignment") or "")
    return kind in {"episode_final", "final", "episode_outcome"}


def _bundle_table(bundle: TraceBundle, table_name: str) -> pd.DataFrame:
    tables = {
        "timesteps": bundle.timesteps,
        "policy_calls": bundle.policy_calls,
        "robot_state": bundle.robot_state,
        "scene_state": bundle.scene_state,
        "camera_state": bundle.camera_state,
        "evaluation": bundle.evaluation,
    }
    if table_name not in tables:
        raise KeyError(f"Unknown target table {table_name!r}")
    return tables[table_name]


def _table_timestep_value(table: pd.DataFrame, timestep: int, column: str) -> Any:
    if table.empty or column not in table:
        return None
    rows = table
    if "timestep" in table:
        rows = table.loc[table["timestep"].astype(int) == timestep]
        if rows.empty:
            return None
    return rows.iloc[-1].get(column)


def _array_target_value(
    bundle: TraceBundle,
    array_id: str,
    timestep: int,
    target_spec: Mapping[str, Any],
    row: pd.Series,
) -> Any:
    array = np.asarray(bundle.array(array_id, mmap=True))
    axes = _array_axes(bundle, array_id)
    value = array
    remaining_axes = list(axes)
    if "timestep" in remaining_axes:
        axis = remaining_axes.index("timestep")
        value = np.take(value, _axis_index(int(value.shape[axis]), timestep), axis=axis)
        remaining_axes.pop(axis)
    if "policy_call" in remaining_axes:
        policy_call_value = _target_axis_value(
            target_spec,
            "policy_call_index",
            fallback=target_spec.get(
                "policy_call",
                row.get("policy_call_index", row.get("policy_call")),
            ),
        )
        if _is_missing_scalar(policy_call_value):
            policy_call_value = _target_axis_value(
                target_spec,
                "policy_call",
                fallback=row.get("policy_call_index", row.get("policy_call")),
            )
        if _is_missing_scalar(policy_call_value):
            policy_call_value = 0
        policy_call = int(policy_call_value)
        axis = remaining_axes.index("policy_call")
        value = np.take(value, _axis_index(int(value.shape[axis]), policy_call), axis=axis)
        remaining_axes.pop(axis)
    if "generation_step" in remaining_axes:
        generation_step_value = _target_axis_value(
            target_spec,
            "generation_step",
            fallback=row.get("generation_step"),
        )
        if _is_missing_scalar(generation_step_value):
            generation_step_value = 0
        axis = remaining_axes.index("generation_step")
        step = _generation_step_index(int(value.shape[axis]), generation_step_value)
        value = np.take(value, step, axis=axis)
        remaining_axes.pop(axis)
    selector = dict(target_spec.get("selector") or {})
    selector.update(
        {
            key: target_spec[key]
            for key in ["component", "dim", "action_dim", "object", "horizon"]
            if key in target_spec
        }
    )
    for axis_name, selected in list(selector.items()):
        canonical = _canonical_axis_name(str(axis_name), remaining_axes)
        if canonical not in remaining_axes:
            continue
        axis = remaining_axes.index(canonical)
        index = _axis_selector_index(bundle, array_id, canonical, selected)
        value = np.take(value, _axis_index(int(value.shape[axis]), index), axis=axis)
        remaining_axes.pop(axis)
    flat = np.asarray(value).reshape(-1)
    if flat.size != 1:
        raise ValueError(
            f"Target {target_spec!r} resolved to vector shape {np.asarray(value).shape}; "
            "select a component/dim/object so the current probe suite gets a scalar y."
        )
    return flat[0].item()


def _target_axis_value(
    target_spec: Mapping[str, Any],
    key: str,
    *,
    fallback: Any,
) -> Any:
    alignment = (
        target_spec.get("alignment") if isinstance(target_spec.get("alignment"), Mapping) else {}
    )
    return alignment.get(key, target_spec.get(key, fallback))


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _axis_index(count: int, index: int) -> int:
    if index < 0:
        return max(0, count + index)
    return min(index, max(0, count - 1))


def _generation_step_index(count: int, generation_step: Any) -> int:
    if str(generation_step) == "final":
        return max(0, count - 1)
    return _axis_index(count, int(generation_step))


def _array_axes(bundle: TraceBundle, array_id: str) -> list[str]:
    table = bundle.array_index
    matches = table.loc[table["name"].astype(str) == array_id] if not table.empty else table
    if matches.empty:
        matches = bundle.model_sites.loc[bundle.model_sites["name"].astype(str) == array_id]
    if matches.empty:
        return []
    return list(json.loads(str(matches.iloc[0].get("axes") or "[]")))


def _array_has_name(bundle: TraceBundle, array_id: str) -> bool:
    table = bundle.array_index
    return not table.empty and "name" in table and array_id in set(table["name"].astype(str))


def _robot_array_id(bundle: TraceBundle, field_name: str) -> str:
    table = bundle.robot_state
    if table.empty:
        raise KeyError("robot_state table is empty")
    rows = table.loc[table.get("field_name", "").astype(str) == field_name]
    if rows.empty:
        raise KeyError(f"robot_state field {field_name!r} is unavailable")
    return str(rows.iloc[0].get("array_id"))


def _scene_array_id_and_object(
    bundle: TraceBundle,
    target_spec: Mapping[str, Any],
) -> tuple[str, int]:
    table = bundle.scene_state
    if table.empty:
        raise KeyError("scene_state table is empty")
    selector = dict(target_spec.get("selector") or {})
    object_name = _scene_object_selector_value(target_spec, selector)
    rows = table
    if object_name is not None and not isinstance(object_name, int):
        names = _scene_object_names(rows)
        object_ids = rows.get("object_id", pd.Series(dtype=object)).astype(str)
        rows = rows.loc[(names == str(object_name)) | (object_ids == str(object_name))]
    if rows.empty:
        raise KeyError(f"scene object {object_name!r} is unavailable")
    field = str(target_spec.get("field") or "pos")
    array_column = {
        "pos": "pos_array_id",
        "position": "pos_array_id",
        "quat": "quat_array_id",
        "quaternion": "quat_array_id",
        "pose": "pose_array_id",
        "joints": "joints_array_id",
    }.get(field, f"{field}_array_id")
    row = rows.iloc[0]
    if array_column not in row or pd.isna(row.get(array_column)):
        raise KeyError(f"scene_state field {field!r} has no linked array")
    object_index = (
        int(object_name) if isinstance(object_name, int) else int(row.get("object_index", 0))
    )
    return str(row[array_column]), object_index


def _scene_object_selector_value(
    target_spec: Mapping[str, Any],
    selector: Mapping[str, Any],
) -> Any:
    object_name = selector.get("object") or target_spec.get("object")
    object_column = selector.get("object_column") or target_spec.get("object_column")
    if object_column is None:
        return object_name
    row = target_spec.get("_row")
    if row is None:
        return object_name
    value = row.get(str(object_column)) if hasattr(row, "get") else None
    if _is_missing_scalar(value):
        if object_name is None:
            raise KeyError(f"Scene object row column {object_column!r} is missing")
        return object_name
    return value


def _scene_object_names(rows: pd.DataFrame) -> pd.Series:
    out = pd.Series("", index=rows.index, dtype=object)
    for column in ["object_name", "name", "object_id"]:
        if column not in rows:
            continue
        values = rows[column]
        missing = out.astype(str) == ""
        out.loc[missing] = values.loc[missing]
    return out.astype(str)


def _relative_reference_spec(
    target_spec: Mapping[str, Any],
    relative_to: Any,
) -> dict[str, Any]:
    if isinstance(relative_to, Mapping):
        reference = dict(relative_to)
    else:
        reference = {"array_id": str(relative_to)}
    reference.setdefault("source", "array")
    if "array_id" not in reference:
        raise ValueError("relative_to target reference requires an array_id")
    selector = dict(reference.get("selector") or {})
    if "component" not in selector and "component" in target_spec:
        selector["component"] = target_spec["component"]
    if "component" not in selector:
        target_selector = target_spec.get("selector")
        if isinstance(target_selector, Mapping) and "component" in target_selector:
            selector["component"] = target_selector["component"]
    reference["selector"] = selector
    return reference


def _canonical_axis_name(axis_name: str, axes: list[str]) -> str:
    aliases = {
        "component": [
            "component",
            "pose_component",
            "gripper_component",
            "joint",
            "action_dim",
            "xyz",
        ],
        "dim": ["action_dim", "component", "pose_component", "xyz"],
        "action_dim": ["action_dim"],
        "object": ["object"],
        "horizon": ["horizon", "action_horizon"],
    }
    for candidate in aliases.get(axis_name, [axis_name]):
        if candidate in axes:
            return candidate
    return axis_name


def _axis_selector_index(
    bundle: TraceBundle,
    array_id: str,
    axis_name: str,
    selected: Any,
) -> int:
    if isinstance(selected, (int, np.integer)):
        return int(selected)
    text = str(selected)
    component_names = {
        "x": 0,
        "y": 1,
        "z": 2,
        "qx": 3,
        "qy": 4,
        "qz": 5,
        "qw": 6,
        "roll": 3,
        "pitch": 4,
        "yaw": 5,
        "gripper": 6,
    }
    if axis_name in {"component", "pose_component", "gripper_component", "action_dim", "xyz"}:
        metadata = _array_metadata(bundle, array_id)
        action_names = metadata.get("action_dim_names") or metadata.get("action_names")
        if isinstance(action_names, list) and text in action_names:
            return int(action_names.index(text))
        if text in component_names:
            return component_names[text]
    return int(text)


def _array_metadata(bundle: TraceBundle, array_id: str) -> dict[str, Any]:
    table = bundle.array_index
    matches = table.loc[table["name"].astype(str) == array_id] if not table.empty else table
    if matches.empty:
        return {}
    try:
        return dict(json.loads(str(matches.iloc[0].get("metadata") or "{}")))
    except Exception:
        return {}
