"""Probe training workflow that saves reusable VLA-lens artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
import yaml

from vla_lens.artifacts import LensArtifact, make_artifact_id
from vla_lens.probes.suite import run_probe_suite
from vla_lens.selectors import ActivationQuery
from vla_lens.traces import TraceBundle, TraceDataset

PROBE_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SavedProbeSuite:
    artifact: LensArtifact
    results: pd.DataFrame
    rows: pd.DataFrame


DEFAULT_PROBE_SPEC: dict[str, Any] = {
    "name": "Outcome probe over expert action features",
    "target": {"kind": "outcome"},
    "features": {
        "module": "pi05.expert.layers.*",
        "tensor_type": "hidden_mean",
        "token_kind": "action",
        "layers": None,
        "timesteps": "all",
        "generation_step": None,
        "reduction": "mean",
    },
    "split": {"kind": "heldout_benchmark"},
    "baseline": ["majority_class", "benchmark", "target_object"],
    "sweep": "layer",
}


def train_probe_artifact(
    dataset: TraceDataset,
    *,
    name: str,
    selector: ActivationQuery,
    target: str | Mapping[str, Any] = "outcome",
    split_kind: str = "random_episode",
    split_column: str = "split",
    train_value: str = "train",
    test_value: str = "test",
    metadata_baseline_columns: Sequence[str] = (),
    sweep: str = "layer",
) -> SavedProbeSuite:
    """Train simple probes from an activation selector and save a ``LensArtifact``."""
    feature_matrix = dataset.select_model_sites(selector).materialize(cache=True)
    X, rows = feature_matrix.X, feature_matrix.rows
    if rows.empty or X.shape[0] == 0:
        raise ValueError(f"Probe selector matched no activation rows: {selector.to_dict()}")
    rows = _attach_episode_metadata(rows, dataset)
    target_spec = _normalize_target_spec(target)
    target_name = _target_name(target_spec)
    rows = _resolve_probe_target(dataset, rows, target_spec)
    rows = _ensure_split(
        rows,
        split_column,
        train_value=train_value,
        test_value=test_value,
        split_kind=split_kind,
    )
    if target_name not in rows:
        raise KeyError(f"Probe target '{target_name}' is not present in selected rows")
    if X.shape[0] != len(rows):
        raise ValueError(
            f"Feature rows mismatch: X has {X.shape[0]} rows, metadata has {len(rows)}"
        )

    results = _run_sweep(
        X=X,
        rows=rows,
        target=target_name,
        split_column=split_column,
        train_value=train_value,
        test_value=test_value,
        metadata_baseline_columns=[
            column for column in metadata_baseline_columns if column in rows.columns
        ],
        sweep=sweep,
        target_kind=str(_probe_target(target_name, rows, target_spec=target_spec)["kind"]),
    )
    if results.empty:
        raise ValueError(
            "No probe result could be trained. Check that train/test rows exist "
            "and the training split has at least two target values."
        )

    artifact_id = make_artifact_id(name, "probe_suite")
    prediction_records = _prediction_frame(results)
    model_arrays, model_state_summary = _best_model_arrays(results)
    outputs = {
        "metrics": str(Path("artifacts") / artifact_id / "metrics.json"),
        "predictions": str(Path("artifacts") / artifact_id / "predictions.parquet"),
        "weights": str(Path("artifacts") / artifact_id / "weights.zarr")
        if "weights" in model_arrays
        else None,
        "bias": str(Path("artifacts") / artifact_id / "bias.zarr")
        if "bias" in model_arrays
        else None,
        "normalizer_feature_mean": str(Path("artifacts") / artifact_id / "feature_mean.zarr")
        if "feature_mean" in model_arrays
        else None,
        "normalizer_feature_scale": str(Path("artifacts") / artifact_id / "feature_scale.zarr")
        if "feature_scale" in model_arrays
        else None,
    }
    method = {
        "workflow": "train_probe_artifact",
        "probe_artifact_schema_version": PROBE_ARTIFACT_SCHEMA_VERSION,
        "lineage": _probe_lineage(random_seed=None),
        "source": _probe_source(dataset, rows),
        "input": _probe_input(selector, rows, X, feature_matrix.cache_key),
        "target": _probe_target(target_name, rows, target_spec=target_spec),
        "examples": _probe_examples(rows, target=target_name, split_column=split_column),
        "split": _probe_split(
            rows,
            split_kind=split_kind,
            split_column=split_column,
            train_value=train_value,
            test_value=test_value,
        ),
        "normalization": {
            "method": "standardize",
            "feature_centering": True,
            "feature_scaling": True,
            "target_centering": False,
            "target_scaling": False,
            "fit_split": train_value,
            "weights_space": "normalized_feature_space",
        },
        "probe": {
            "type": _primary_probe_type(results),
            "library": "sklearn",
            "library_version": sklearn.__version__,
            "hyperparams": _probe_hyperparams(results),
            "trained_on_split": train_value,
            "weights_space": "normalized_feature_space",
            "best_model_state": model_state_summary,
        },
        "evaluation": {
            "primary_split": test_value,
            "primary_metric": _primary_metric(results),
            "grain": "row",
            "aggregation": "over_rows",
            "metric_definitions": _metric_definitions(results),
        },
        "prediction_retention": {
            "mode": "row_level_test",
            "splits": [test_value],
            "row_count": int(len(prediction_records)),
        },
        "outputs": {key: value for key, value in outputs.items() if value is not None},
        "split_kind": split_kind,
        "split_column": split_column,
        "train_value": train_value,
        "test_value": test_value,
        "metadata_baseline_columns": [
            column for column in metadata_baseline_columns if column in rows.columns
        ],
        "sweep": sweep,
    }
    metrics = _probe_metrics(results, rows, target=target_name)
    metrics["probe_artifact_schema_version"] = PROBE_ARTIFACT_SCHEMA_VERSION
    metrics["prediction_row_count"] = int(len(prediction_records))
    metrics["feature_matrix_fingerprint"] = _array_fingerprint(X)
    artifact = LensArtifact(
        artifact_id=artifact_id,
        artifact_type="probe_suite",
        name=name,
        group_id="probe_suites",
        scope="dataset",
        selector=selector.to_dict(),
        method=method,
        metrics=metrics,
        display={
            "kind": "probe_suite",
            "results": _records(results),
            "best_result_details": _best_result_details(results),
            "target": target_name,
            "split_summary": _split_summary(rows, split_column),
            "target_distribution": _value_counts(rows[target_name]),
            "baseline_columns": [
                column for column in metadata_baseline_columns if column in rows.columns
            ],
            "interpretation_notes": _probe_notes(
                rows,
                target_name,
                [column for column in metadata_baseline_columns if column in rows.columns],
            ),
            "row_count": int(len(rows)),
            "sample_count": int(X.shape[0]),
            "feature_dim": int(X.shape[1]) if X.ndim == 2 else None,
            "source_columns": sorted(str(column) for column in rows.columns),
            "data_quality": _probe_data_quality(
                rows,
                target=target_name,
                split_column=split_column,
                metadata_baseline_columns=[
                    column for column in metadata_baseline_columns if column in rows.columns
                ],
            ),
        },
        tags=("probe", target_name),
        source_trace_ids=tuple(sorted(str(value) for value in rows["trace_id"].dropna().unique())),
    )
    saved = dataset.save_artifact(artifact, arrays=model_arrays)
    artifact_dir = _artifact_dir(dataset, saved)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prediction_records.to_parquet(artifact_dir / "predictions.parquet", index=False)
    (artifact_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return SavedProbeSuite(artifact=saved, results=results, rows=rows)


def train_probe_artifact_from_spec(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
) -> SavedProbeSuite:
    """Train a probe artifact from a YAML/JSON-compatible spec."""
    normalized = normalize_probe_spec(spec)
    features = normalized["features"]
    selector = ActivationQuery(
        episodes=dict(features.get("episodes") or {}),
        name=features.get("name"),
        module=features.get("module"),
        layers=features.get("layers"),
        tensor_type=features.get("tensor_type"),
        token_kind=features.get("token_kind"),
        timesteps=features.get("timesteps", "all"),
        generation_step=features.get("generation_step"),
        reduce_tokens=features.get("reduction", "mean"),
    )
    return train_probe_artifact(
        dataset,
        name=str(normalized["name"]),
        selector=selector,
        target=normalized["target"],
        split_kind=str(normalized["split"]["kind"]),
        split_column=str(normalized["split"].get("column", "split")),
        train_value=str(normalized["split"].get("train_value", "train")),
        test_value=str(normalized["split"].get("test_value", "test")),
        metadata_baseline_columns=baseline_columns(normalized.get("baseline", [])),
        sweep=str(normalized.get("sweep", "layer")),
    )


def normalize_probe_spec(spec: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a complete probe spec with conservative defaults."""
    merged = _deep_merge(DEFAULT_PROBE_SPEC, dict(spec or {}))
    target = merged.get("target")
    if isinstance(target, str):
        merged["target"] = {"kind": target}
    features = merged.setdefault("features", {})
    if "reduce_tokens" in features and "reduction" not in features:
        features["reduction"] = features.pop("reduce_tokens")
    split = merged.get("split")
    if isinstance(split, str):
        merged["split"] = {"kind": split}
    merged.setdefault("baseline", [])
    return merged


def load_probe_spec(path: str | Path) -> dict[str, Any]:
    """Load a probe spec from YAML. Use ``-`` for stdin."""
    if str(path) == "-":
        import sys

        payload = yaml.safe_load(sys.stdin.read())
    else:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError("Probe spec is empty")
    if not isinstance(payload, Mapping):
        raise TypeError("Probe spec must be a mapping")
    return normalize_probe_spec(payload)


def dump_probe_spec(spec: Mapping[str, Any]) -> str:
    return yaml.safe_dump(normalize_probe_spec(spec), sort_keys=False)


def baseline_columns(items: Sequence[Any]) -> list[str]:
    columns: list[str] = []
    aliases = {
        "majority_class": None,
        "majority": None,
        "benchmark": "benchmark",
        "benchmark_only": "benchmark",
        "task": "task_id",
        "task_id": "task_id",
        "target_object": "target_object",
        "object": "target_object",
        "env": "env_id",
        "env_id": "env_id",
    }
    for item in items:
        value = str(item).strip()
        column = aliases.get(value, value)
        if column and column not in columns:
            columns.append(column)
    return columns


def _attach_episode_metadata(rows: pd.DataFrame, dataset: TraceDataset) -> pd.DataFrame:
    episode_index = dataset.episode_index.copy()
    if episode_index.empty:
        return rows.copy()
    duplicate_columns = [
        column
        for column in episode_index.columns
        if column in rows.columns and column not in {"trace_id"}
    ]
    episode_index = episode_index.drop(columns=duplicate_columns)
    return rows.merge(episode_index, on="trace_id", how="left")


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


def _run_sweep(
    *,
    X: np.ndarray,
    rows: pd.DataFrame,
    target: str,
    split_column: str,
    train_value: str,
    test_value: str,
    metadata_baseline_columns: list[str],
    sweep: str,
    target_kind: str,
) -> pd.DataFrame:
    if sweep == "none":
        return run_probe_suite(
            rows,
            {"selected model_sites": X},
            [target],
            split_column=split_column,
            train_value=train_value,
            test_value=test_value,
            metadata_baseline_columns=metadata_baseline_columns,
            target_kinds={target: target_kind},
        )

    if sweep not in rows:
        raise KeyError(f"Sweep column '{sweep}' is not present in selected rows")
    frames: list[pd.DataFrame] = []
    for value, group in rows.groupby(sweep, dropna=False, sort=True):
        index = group.index.to_numpy()
        result = run_probe_suite(
            group.reset_index(drop=True),
            {f"{sweep} {value}": X[index]},
            [target],
            split_column=split_column,
            train_value=train_value,
            test_value=test_value,
            metadata_baseline_columns=metadata_baseline_columns,
            target_kinds={target: target_kind},
        )
        if result.empty:
            continue
        result.insert(0, "sweep", sweep)
        result.insert(1, "sweep_value", _json_scalar(value))
        frames.append(result)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


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
        lambda row: _resolve_target_value(dataset, row, target_spec),
        axis=1,
    )
    out = rows.copy()
    out[target_name] = _apply_target_transform(resolved, target_spec)
    missing = int(out[target_name].isna().sum())
    if missing:
        raise ValueError(
            f"Target {target_name!r} could not be resolved for {missing} selected rows. "
            "Narrow the selector or choose a target with complete coverage."
        )
    return out


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
        return _array_target_value(bundle, "executed_actions", timestep, spec, row)
    if source == "robot_state":
        array_id = _robot_array_id(bundle, str(target_spec.get("field") or ""))
        return _array_target_value(bundle, array_id, timestep, target_spec, row)
    if source == "scene_state":
        array_id, object_index = _scene_array_id_and_object(bundle, target_spec)
        selector = {**dict(target_spec.get("selector") or {}), "object": object_index}
        return _array_target_value(
            bundle,
            array_id,
            timestep,
            {**dict(target_spec), "selector": selector},
            row,
        )
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
        value = np.take(value, min(timestep, value.shape[axis] - 1), axis=axis)
        remaining_axes.pop(axis)
    if "policy_call" in remaining_axes:
        policy_call_value = target_spec.get(
            "policy_call_index",
            row.get("policy_call_index", row.get("policy_call")),
        )
        if policy_call_value is None or pd.isna(policy_call_value):
            policy_call_value = 0
        policy_call = int(policy_call_value)
        axis = remaining_axes.index("policy_call")
        value = np.take(value, min(policy_call, value.shape[axis] - 1), axis=axis)
        remaining_axes.pop(axis)
    if "generation_step" in remaining_axes:
        generation_step_value = target_spec.get("generation_step", row.get("generation_step"))
        if generation_step_value is None or pd.isna(generation_step_value):
            generation_step_value = 0
        axis = remaining_axes.index("generation_step")
        step = int(generation_step_value)
        value = np.take(value, min(step, value.shape[axis] - 1), axis=axis)
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
        value = np.take(value, min(index, value.shape[axis] - 1), axis=axis)
        remaining_axes.pop(axis)
    flat = np.asarray(value).reshape(-1)
    if flat.size != 1:
        raise ValueError(
            f"Target {target_spec!r} resolved to vector shape {np.asarray(value).shape}; "
            "select a component/dim/object so the current probe suite gets a scalar y."
        )
    return flat[0].item()


def _array_axes(bundle: TraceBundle, array_id: str) -> list[str]:
    table = bundle.array_index
    matches = table.loc[table["name"].astype(str) == array_id] if not table.empty else table
    if matches.empty:
        matches = bundle.model_sites.loc[bundle.model_sites["name"].astype(str) == array_id]
    if matches.empty:
        return []
    return list(json.loads(str(matches.iloc[0].get("axes") or "[]")))


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
    object_name = selector.get("object") or target_spec.get("object")
    rows = table
    if object_name is not None and not isinstance(object_name, int):
        names = rows.get("name", pd.Series(dtype=object)).astype(str)
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


def _canonical_axis_name(axis_name: str, axes: list[str]) -> str:
    aliases = {
        "component": ["component", "pose_component", "gripper_component", "joint", "action_dim"],
        "dim": ["action_dim", "component", "pose_component"],
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
    if axis_name in {"component", "pose_component", "gripper_component", "action_dim"}:
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


def _probe_lineage(*, random_seed: int | None) -> dict[str, Any]:
    return {
        "probe_artifact_schema_version": PROBE_ARTIFACT_SCHEMA_VERSION,
        "code_commit": _git_commit(),
        "random_seed": random_seed,
        "library_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }


def _probe_source(dataset: TraceDataset, rows: pd.DataFrame) -> dict[str, Any]:
    trace_ids = tuple(sorted(str(value) for value in rows["trace_id"].dropna().unique()))
    source_episodes: list[dict[str, Any]] = []
    trace_fingerprints = []
    schema_versions = set()
    for trace_id in trace_ids:
        bundle = dataset.bundle(trace_id)
        fingerprints = dict(bundle.fingerprints or {})
        if not fingerprints:
            fingerprints = {
                "trace_fingerprint": _bundle_fingerprint(bundle),
                "trajectory_fingerprint": None,
                "context_fingerprint": None,
                "trace_schema_fingerprint": None,
            }
        trace_fingerprint = str(fingerprints.get("trace_fingerprint") or "")
        trace_fingerprints.append(trace_fingerprint)
        schema_versions.add(str(bundle.manifest.schema_version))
        split = None
        if "split" in rows:
            split_values = sorted(
                str(value)
                for value in rows.loc[rows["trace_id"].astype(str) == trace_id, "split"]
                .dropna()
                .unique()
            )
            split = split_values[0] if len(split_values) == 1 else split_values
        source_episodes.append(
            {
                "trace_id": bundle.manifest.trace_id,
                "episode_id": bundle.manifest.episode_id,
                "split": split,
                "trajectory_fingerprint": fingerprints.get("trajectory_fingerprint"),
                "context_fingerprint": fingerprints.get("context_fingerprint"),
                "trace_schema_fingerprint": fingerprints.get("trace_schema_fingerprint"),
                "trace_fingerprint": trace_fingerprint,
            }
        )
    return {
        "source_traces": list(trace_ids),
        "source_episodes": source_episodes,
        "source_trace_fingerprints": trace_fingerprints,
        "source_collection_fingerprint": _hash_json(trace_fingerprints),
        "vlatrace_schema_versions": sorted(schema_versions),
    }


def _probe_input(
    selector: ActivationQuery,
    rows: pd.DataFrame,
    X: np.ndarray,
    cache_key: str,
) -> dict[str, Any]:
    model_site_column = rows.get("model_site_id", rows["activation"])
    model_sites = sorted(str(value) for value in model_site_column.dropna().unique())
    token_spaces = sorted(
        str(value)
        for value in rows.get("token_space_id", pd.Series(dtype=object)).dropna().unique()
    )
    axes = sorted(
        str(value) for value in rows.get("axes", pd.Series(dtype=object)).dropna().unique()
    )
    dtypes = sorted(
        str(value) for value in rows.get("dtype", pd.Series(dtype=object)).dropna().unique()
    )
    return {
        "selector": selector.to_dict(),
        "trace_arrays": model_sites,
        "model_site_ids": model_sites,
        "axes": axes,
        "dtype": dtypes[0] if len(dtypes) == 1 else dtypes,
        "token_space_ids": token_spaces,
        "selection": {
            "timesteps": selector.to_dict().get("timesteps"),
            "policy_calls": selector.to_dict().get("policy_calls"),
            "generation_step": selector.generation_step,
            "token_kind": selector.token_kind,
        },
        "pooling": selector.reduce_tokens,
        "feature_transform": "identity",
        "feature_shape": [int(item) for item in X.shape],
        "feature_dim": int(X.shape[1]) if X.ndim == 2 else None,
        "feature_matrix_cache_key": cache_key,
        "feature_matrix_fingerprint": _array_fingerprint(X),
    }


def _probe_target(
    target: str,
    rows: pd.DataFrame,
    *,
    target_spec: Mapping[str, Any],
) -> dict[str, Any]:
    kind = "classification"
    if target in rows:
        values = rows[target].dropna().to_numpy()
        if values.size and not _target_looks_categorical(values):
            kind = "regression"
    source = target_spec.get("source") or target
    if source == "row" and target == "outcome":
        source = "manifest.outcome"
    declared_kind = str(target_spec.get("kind") or "")
    target_kind = declared_kind if declared_kind in {"regression", "classification"} else kind
    return {
        "name": target,
        "kind": target_kind,
        "source": source,
        "selector": target_spec.get("selector"),
        "alignment": target_spec.get("alignment")
        or {
            "kind": "same_selected_activation_row",
            "offset": 0,
        },
        "transform": target_spec.get("transform") or {"kind": "identity"},
        "resolved_column": target,
        "target_fingerprint": _series_fingerprint(rows[target]) if target in rows else None,
    }


def _probe_examples(
    rows: pd.DataFrame,
    *,
    target: str,
    split_column: str,
) -> dict[str, Any]:
    count_by_split = (
        _value_counts(rows[split_column]) if split_column in rows else {"all": int(len(rows))}
    )
    return {
        "unit": "selected_activation_row",
        "row_construction": (
            "one row per selected trace/model_site/sample after selector reduction"
        ),
        "filters": ["exclude_empty_feature_rows", "exclude_nonfinite_feature_rows"],
        "missing_target_policy": "error",
        "count": int(len(rows)),
        "count_by_split": count_by_split,
        "example_id_definition": [
            "trace_id",
            "policy_call_index",
            "generation_step",
            "token_space_id",
            "token_index",
            "model_site_id",
            target,
        ],
        "row_index_fingerprint": _rows_fingerprint(rows),
    }


def _probe_split(
    rows: pd.DataFrame,
    *,
    split_kind: str,
    split_column: str,
    train_value: str,
    test_value: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "method": "grouped" if split_kind != "random_row" else "random_row",
        "kind": split_kind,
        "group_key": "trace_id" if split_kind != "random_row" else None,
        "column": split_column,
        "train_value": train_value,
        "test_value": test_value,
        "leakage_risk": "high" if split_kind == "random_row" else "controlled_by_group",
    }
    if split_column in rows and "trace_id" in rows:
        split_rows = rows[[split_column, "trace_id"]].drop_duplicates()
        for value in [train_value, test_value]:
            trace_rows = split_rows.loc[
                split_rows[split_column].astype(str) == value,
                "trace_id",
            ]
            out[f"{value}_traces"] = sorted(str(item) for item in trace_rows.dropna().unique())
    return out


def _prediction_frame(results: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for result_index, result in results.iterrows():
        for record in result.get("prediction_records") or []:
            row = dict(record)
            row["result_index"] = int(result_index)
            row["feature"] = str(result.get("feature"))
            row["probe_type"] = str(result.get("probe_type"))
            row["sweep"] = _json_scalar(result.get("sweep"))
            row["sweep_value"] = _json_scalar(result.get("sweep_value"))
            records.append(row)
    return pd.DataFrame.from_records(records)


def _best_model_arrays(results: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if results.empty or "model_state" not in results:
        return {}, {}
    delta = results["score"] - results["baseline_score"]
    best_idx = int(delta.idxmax())
    state = results.loc[best_idx].get("model_state")
    if not isinstance(state, Mapping):
        return {}, {}
    arrays: dict[str, np.ndarray] = {}
    for key in ["weights", "bias", "feature_mean", "feature_scale"]:
        value = state.get(key)
        if isinstance(value, np.ndarray) and value.size:
            arrays[key] = value.astype(np.float32, copy=False)
    summary = {
        "feature": str(results.loc[best_idx].get("feature")),
        "sweep_value": _json_scalar(results.loc[best_idx].get("sweep_value")),
        "probe_type": state.get("probe_type"),
        "weights_space": state.get("weights_space"),
        "classes": list(state.get("classes") or []),
        "array_shapes": {key: [int(item) for item in value.shape] for key, value in arrays.items()},
    }
    return arrays, summary


def _primary_probe_type(results: pd.DataFrame) -> str:
    if results.empty or "probe_type" not in results:
        return "unknown"
    values = sorted(str(value) for value in results["probe_type"].dropna().unique())
    return values[0] if len(values) == 1 else "mixed"


def _probe_hyperparams(results: pd.DataFrame) -> dict[str, Any]:
    probe_type = _primary_probe_type(results)
    if probe_type == "classification":
        return {"model": "LogisticRegression", "max_iter": 1000}
    if probe_type == "regression":
        return {"model": "Ridge", "alpha": 1.0}
    return {}


def _primary_metric(results: pd.DataFrame) -> str:
    probe_type = _primary_probe_type(results)
    return "accuracy" if probe_type == "classification" else "r2"


def _metric_definitions(results: pd.DataFrame) -> dict[str, str]:
    probe_type = _primary_probe_type(results)
    if probe_type == "classification":
        return {
            "score": "accuracy_score over retained test rows",
            "baseline_score": "max of majority-class baseline and configured metadata baselines",
            "delta": "score - baseline_score",
        }
    return {
        "score": (
            "sklearn.metrics.r2_score over retained test rows; "
            "negative MAE fallback when R2 is not finite"
        ),
        "baseline_score": "train-target-mean baseline with the same metric as score",
        "delta": "score - baseline_score",
        "error": "prediction_value - target_value",
    }


def _probe_metrics(results: pd.DataFrame, rows: pd.DataFrame, *, target: str) -> Mapping[str, Any]:
    delta = results["score"] - results["baseline_score"]
    best_idx = int(delta.idxmax())
    best = results.loc[best_idx]
    return {
        "target": target,
        "result_count": int(len(results)),
        "sample_count": int(len(rows)),
        "source_episode_count": int(rows["trace_id"].nunique()),
        "best_score": float(best["score"]),
        "best_baseline": float(best["baseline_score"]),
        "best_delta": float(delta.loc[best_idx]),
        "best_feature": str(best["feature"]),
        "best_sweep_value": _json_scalar(best.get("sweep_value")),
        "target_distribution": _value_counts(rows[target]),
    }


def _best_result_details(results: pd.DataFrame) -> Mapping[str, Any]:
    if results.empty or "details" not in results:
        return {}
    delta = results["score"] - results["baseline_score"]
    best_idx = int(delta.idxmax())
    details = results.loc[best_idx].get("details")
    if not isinstance(details, Mapping):
        return {}
    return {
        "feature": str(results.loc[best_idx].get("feature")),
        "sweep_value": _json_scalar(results.loc[best_idx].get("sweep_value")),
        "details": dict(details),
    }


def _split_summary(rows: pd.DataFrame, split_column: str) -> dict[str, Any]:
    if split_column not in rows:
        return {}
    out: dict[str, Any] = {"column": split_column, "values": _value_counts(rows[split_column])}
    if "trace_id" in rows:
        by_split = rows[[split_column, "trace_id"]].drop_duplicates()
        out["episodes"] = _value_counts(by_split[split_column])
    return out


def _value_counts(values: pd.Series) -> dict[str, int]:
    counts = values.astype(str).value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def _probe_notes(
    rows: pd.DataFrame,
    target: str,
    metadata_baseline_columns: Sequence[str],
) -> list[str]:
    notes: list[str] = []
    if target == "outcome":
        notes.append("Outcome probes are correlational; they do not prove a failure mechanism.")
        if "task_id" in rows:
            task_outcomes = rows[["trace_id", "task_id", target]].drop_duplicates()
            mixed = int((task_outcomes.groupby("task_id")[target].nunique() > 1).sum())
            if mixed == 0:
                notes.append(
                    "No task has both success and failure examples, so task identity can be "
                    "confused with behavior."
                )
        if metadata_baseline_columns:
            notes.append(
                "Compare probe score against metadata baselines before interpreting model "
                "features as behavior-relevant."
            )
    return notes


def _probe_data_quality(
    rows: pd.DataFrame,
    *,
    target: str,
    split_column: str,
    metadata_baseline_columns: Sequence[str],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if split_column in rows:
        split_by_episode = rows[[split_column, "trace_id"]].drop_duplicates()
        checks.append(
            {
                "name": "split_episodes",
                "status": "ok" if split_by_episode[split_column].nunique() > 1 else "weak",
                "detail": _value_counts(split_by_episode[split_column]),
            }
        )
    if target in rows:
        checks.append(
            {
                "name": "target_balance",
                "status": "ok" if rows[target].astype(str).nunique() > 1 else "weak",
                "detail": _value_counts(rows[target]),
            }
        )
    if target == "outcome" and "task_id" in rows:
        task_outcomes = rows[["trace_id", "task_id", target]].drop_duplicates()
        mixed = int((task_outcomes.groupby("task_id")[target].nunique() > 1).sum())
        checks.append(
            {
                "name": "within_task_outcomes",
                "status": "ok" if mixed else "confounded",
                "detail": {
                    "tasks_with_both_outcomes": mixed,
                    "unique_tasks": int(task_outcomes["task_id"].nunique()),
                },
            }
        )
    if metadata_baseline_columns:
        checks.append(
            {
                "name": "metadata_baselines",
                "status": "ok",
                "detail": list(metadata_baseline_columns),
            }
        )
    return checks


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    hidden_columns = {"prediction_records", "model_state"}
    return [
        {
            str(key): _json_scalar(value)
            for key, value in record.items()
            if str(key) not in hidden_columns
        }
        for record in frame.to_dict("records")
    ]


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return _json_scalar(value.item())
    if isinstance(value, np.ndarray):
        return {
            "array_shape": [int(item) for item in value.shape],
            "array_dtype": str(value.dtype),
        }
    if isinstance(value, Mapping):
        return {str(key): _json_scalar(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_scalar(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _artifact_dir(dataset: TraceDataset, artifact: LensArtifact) -> Path:
    if artifact.scope == "dataset" and not (dataset.root / TraceBundle.MANIFEST).exists():
        return dataset.root / "artifacts" / artifact.artifact_id
    if len(dataset.bundles) == 1:
        return dataset.bundles[0].path / "artifacts" / artifact.artifact_id
    return dataset.root / "artifacts" / artifact.artifact_id


def _target_looks_categorical(values: np.ndarray) -> bool:
    array = np.asarray(values)
    if array.dtype == bool or array.dtype.kind in {"O", "U", "S", "b"}:
        return True
    return len(np.unique(array[~pd.isna(array)])) <= 20


def _array_fingerprint(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.shape).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.view(np.uint8))
    return f"sha256:{digest.hexdigest()}"


def _rows_fingerprint(rows: pd.DataFrame) -> str:
    columns = [
        column
        for column in [
            "trace_id",
            "episode_id",
            "timestep",
            "policy_call_index",
            "generation_step",
            "token_space_id",
            "token_index",
            "model_site_id",
            "activation",
        ]
        if column in rows
    ]
    return _hash_json(rows[columns].astype(str).to_dict("records") if columns else [])


def _series_fingerprint(values: pd.Series) -> str:
    payload = {
        "name": str(values.name),
        "dtype": str(values.dtype),
        "values": [_json_scalar(value) for value in values.tolist()],
    }
    return _hash_json(payload)


def _bundle_fingerprint(bundle: Any) -> str:
    payload = {
        "trace_id": bundle.manifest.trace_id,
        "episode_id": bundle.manifest.episode_id,
        "schema_version": bundle.manifest.schema_version,
        "length": bundle.manifest.length,
        "model_sites": (
            bundle.model_sites[["name", "shape", "dtype", "axes"]].to_dict("records")
            if not bundle.model_sites.empty
            else []
        ),
    }
    return _hash_json(payload)


def _hash_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None
