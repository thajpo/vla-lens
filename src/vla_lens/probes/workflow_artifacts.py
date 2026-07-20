"""Probe artifact payload, metric, and fingerprint helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score, f1_score

from vla_lens.artifacts import LensArtifact
from vla_lens.probes.workflow_types import PROBE_ARTIFACT_SCHEMA_VERSION
from vla_lens.selectors import ActivationQuery
from vla_lens.traces import TraceDataset


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
        "trace_schema_versions": sorted(schema_versions),
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
        "feature_matrix_bytes": int(X.nbytes),
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
    eval_values: Sequence[str],
    selection_value: str,
) -> dict[str, Any]:
    eval_values = list(dict.fromkeys(str(value) for value in eval_values))
    out: dict[str, Any] = {
        "method": "grouped" if split_kind != "random_row" else "random_row",
        "kind": split_kind,
        "group_key": "trace_id" if split_kind != "random_row" else None,
        "column": split_column,
        "train_value": train_value,
        "test_value": test_value,
        "eval_values": eval_values,
        "selection_value": selection_value,
        "leakage_risk": "high" if split_kind == "random_row" else "controlled_by_group",
    }
    if split_column in rows and "trace_id" in rows:
        split_rows = rows[[split_column, "trace_id"]].drop_duplicates()
        for value in dict.fromkeys([train_value, test_value, selection_value, *eval_values]):
            trace_rows = split_rows.loc[
                split_rows[split_column].astype(str) == value,
                "trace_id",
            ]
            out[f"{value}_traces"] = sorted(str(item) for item in trace_rows.dropna().unique())
    return out


def _prediction_frame(
    results: pd.DataFrame,
    *,
    record_column: str = "prediction_records",
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for result_index, result in results.iterrows():
        for record in result.get(record_column) or []:
            row = dict(record)
            row["result_index"] = int(result_index)
            row["feature"] = str(result.get("feature"))
            row["probe_type"] = str(result.get("probe_type"))
            row["model"] = str(result.get("model", "linear"))
            row["eval_split"] = str(result.get("split_value", row.get("split")))
            row["primary_metric"] = str(result.get("primary_metric", "score"))
            row["sweep"] = _json_scalar(result.get("sweep"))
            row["sweep_value"] = _json_scalar(result.get("sweep_value"))
            records.append(row)
    return pd.DataFrame.from_records(records)


def _best_model_arrays(
    results: pd.DataFrame,
    *,
    selection_value: str | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if results.empty or "model_state" not in results:
        return {}, {}
    best_idx = _best_result_index(results, selection_value=selection_value, prefer_model="linear")
    state = results.loc[best_idx].get("model_state")
    if not isinstance(state, Mapping):
        return {}, {}
    arrays: dict[str, np.ndarray] = {}
    for key in ["weights", "bias", "feature_mean", "feature_scale"]:
        value = state.get(key)
        if isinstance(value, np.ndarray) and value.size:
            arrays[key] = value.copy()
    for state_key, array_prefix in [
        ("layer_weights", "layer_weights"),
        ("layer_biases", "layer_biases"),
    ]:
        for index, value in enumerate(state.get(state_key) or []):
            if isinstance(value, np.ndarray) and value.size:
                arrays[f"{array_prefix}_{index}"] = value.copy()
    summary = {
        "feature": str(results.loc[best_idx].get("feature")),
        "sweep_value": _json_scalar(results.loc[best_idx].get("sweep_value")),
        "model": str(results.loc[best_idx].get("model", state.get("model", "linear"))),
        "split_value": str(results.loc[best_idx].get("split_value", selection_value)),
        "primary_metric": str(results.loc[best_idx].get("primary_metric", "score")),
        "probe_type": state.get("probe_type"),
        "weights_space": state.get("weights_space"),
        "classes": list(state.get("classes") or []),
        "activation": state.get("activation"),
        "out_activation": state.get("out_activation"),
        "array_shapes": {key: [int(item) for item in value.shape] for key, value in arrays.items()},
    }
    return arrays, summary


def _best_result_index(
    results: pd.DataFrame,
    *,
    selection_value: str | None = None,
    prefer_model: str | None = "linear",
) -> int:
    candidates = results
    if prefer_model and "model" in candidates:
        model_rows = candidates.loc[candidates["model"].astype(str) == prefer_model]
        if not model_rows.empty:
            candidates = model_rows
    if selection_value and "split_value" in candidates:
        split_rows = candidates.loc[candidates["split_value"].astype(str) == str(selection_value)]
        if not split_rows.empty:
            candidates = split_rows
    delta = candidates["score"] - candidates["baseline_score"]
    return int(delta.idxmax())


def _primary_probe_type(results: pd.DataFrame) -> str:
    if results.empty or "probe_type" not in results:
        return "unknown"
    values = sorted(str(value) for value in results["probe_type"].dropna().unique())
    return values[0] if len(values) == 1 else "mixed"


def _probe_hyperparams(results: pd.DataFrame) -> dict[str, Any]:
    probe_type = _primary_probe_type(results)
    models = sorted(str(value) for value in results.get("model", pd.Series(["linear"])).unique())
    if probe_type == "classification":
        params: dict[str, Any] = {}
        if "linear" in models:
            params["linear"] = {
                "model": "LogisticRegression",
                "max_iter": 1000,
                "class_weight": "balanced",
            }
        if "mlp" in models:
            params["mlp"] = {
                "model": "MLPClassifier",
                "hidden_layer_sizes": [64],
                "alpha": 1e-4,
                "max_iter": 300,
                "random_state": 0,
            }
        return params
    if probe_type == "regression":
        params = {}
        if "linear" in models:
            params["linear"] = {"model": "Ridge", "alpha": 1.0}
        if "mlp" in models:
            params["mlp"] = {
                "model": "MLPRegressor",
                "hidden_layer_sizes": [64],
                "alpha": 1e-4,
                "max_iter": 300,
                "random_state": 0,
            }
        return params
    return {}


def _primary_metric(results: pd.DataFrame) -> str:
    if "primary_metric" in results:
        values = sorted(str(value) for value in results["primary_metric"].dropna().unique())
        if values:
            return values[0] if len(values) == 1 else "mixed"
    probe_type = _primary_probe_type(results)
    return "balanced_accuracy" if probe_type == "classification" else "negative_mae"


def _metric_definitions(results: pd.DataFrame) -> dict[str, str]:
    probe_type = _primary_probe_type(results)
    if probe_type == "classification":
        return {
            "score": "balanced_accuracy_score over retained evaluation rows",
            "baseline_score": "max of majority-class baseline and configured metadata baselines",
            "delta": "score - baseline_score",
            "accuracy": "raw fraction of correct predictions",
            "macro_f1": "unweighted mean F1 over classes",
            "log_loss": "cross-entropy over model probabilities when available",
        }
    return {
        "score": (
            "negative mean absolute error over retained evaluation rows; higher is better"
        ),
        "baseline_score": "negative MAE for the train-target-mean baseline",
        "delta": "score - baseline_score",
        "r2": "sklearn.metrics.r2_score, stored as a secondary diagnostic",
        "error": "prediction_value - target_value",
    }


def _probe_metrics(
    results: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    target: str,
    selection_value: str | None,
) -> Mapping[str, Any]:
    best_idx = _best_result_index(results, selection_value=selection_value, prefer_model="linear")
    delta = results["score"] - results["baseline_score"]
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
        "best_model": str(best.get("model", "linear")),
        "best_eval_split": str(best.get("split_value", selection_value)),
        "best_primary_metric": str(best.get("primary_metric", "score")),
        "target_distribution": _value_counts(rows[target]),
    }


def _best_result_details(
    results: pd.DataFrame,
    *,
    selection_value: str | None = None,
) -> Mapping[str, Any]:
    if results.empty or "details" not in results:
        return {}
    best_idx = _best_result_index(results, selection_value=selection_value, prefer_model="linear")
    details = results.loc[best_idx].get("details")
    if not isinstance(details, Mapping):
        return {}
    return {
        "feature": str(results.loc[best_idx].get("feature")),
        "sweep_value": _json_scalar(results.loc[best_idx].get("sweep_value")),
        "model": str(results.loc[best_idx].get("model", "linear")),
        "eval_split": str(results.loc[best_idx].get("split_value", selection_value)),
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


def _per_split_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "split" not in predictions:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for split, group in predictions.groupby("split", dropna=False, sort=True):
        records.append({"split": str(split), **_prediction_metric_record(group)})
    return pd.DataFrame.from_records(records)


def _per_group_metrics(
    predictions: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    if predictions.empty or "trace_id" not in predictions:
        return pd.DataFrame()
    episode_rows = rows[["trace_id", *[c for c in group_columns if c in rows]]].drop_duplicates(
        subset=["trace_id"]
    )
    joined = predictions.merge(episode_rows, on="trace_id", how="left")
    records: list[dict[str, Any]] = []
    for column in group_columns:
        if column not in joined:
            continue
        for value, group in joined.groupby(column, dropna=False, sort=True):
            records.append(
                {
                    "group_column": column,
                    "group_value": str(value),
                    **_prediction_metric_record(group),
                }
            )
    return pd.DataFrame.from_records(records)


def _null_metrics(predictions: pd.DataFrame, *, runs: int = 20) -> pd.DataFrame:
    if predictions.empty or "target_kind" not in predictions:
        return pd.DataFrame()
    if str(predictions["target_kind"].dropna().iloc[0]) != "classification":
        return pd.DataFrame()
    actual = predictions["actual"].astype(str).to_numpy()
    predicted = predictions["predicted"].astype(str).to_numpy()
    if len(actual) == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(0)
    records = []
    for run in range(runs):
        shuffled = actual.copy()
        rng.shuffle(shuffled)
        records.append(
            {
                "null_kind": "label_shuffle_predictions_fixed",
                "run": run,
                "score": float(balanced_accuracy_score(shuffled, predicted)),
                "metric": "balanced_accuracy",
                "row_count": int(len(actual)),
            }
        )
    return pd.DataFrame.from_records(records)


def _prediction_metric_record(group: pd.DataFrame) -> dict[str, Any]:
    if group.empty:
        return {"row_count": 0, "score": np.nan, "metric": "unknown"}
    kind = str(group.get("target_kind", pd.Series(["classification"])).dropna().iloc[0])
    if kind == "classification" and "correct" in group:
        actual = group["actual"].astype(str)
        predicted = group["predicted"].astype(str)
        return {
            "row_count": int(len(group)),
            "score": float(balanced_accuracy_score(actual, predicted)),
            "accuracy": float(group["correct"].astype(bool).mean()),
            "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
            "metric": "balanced_accuracy",
        }
    if "error" in group:
        error = pd.to_numeric(group["error"], errors="coerce")
        return {
            "row_count": int(len(group)),
            "score": float(-error.abs().mean()),
            "metric": "negative_mae",
        }
    return {"row_count": int(len(group)), "score": np.nan, "metric": kind}


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


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    hidden_columns = {"all_prediction_records", "prediction_records", "model_state"}
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
    if artifact.scope == "dataset":
        if artifact.path:
            return (dataset._dataset_artifact_root() / artifact.path).parent
        return dataset._dataset_artifact_root() / "artifacts" / artifact.artifact_id
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
