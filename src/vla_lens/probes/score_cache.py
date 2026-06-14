"""Refreshable probe score caches for dataset browsing.

The probe artifact is the frozen training/evaluation record.  This module
re-applies the artifact's saved linear probe to the current dataset so newly
added compatible episodes can appear in dataset ranking/readout UIs without
changing the original heldout metrics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.artifacts import LensArtifact
from vla_lens.probes.suite import _prediction_join_keys
from vla_lens.probes.workflow_artifacts import _array_fingerprint
from vla_lens.probes.workflow_prepare import (
    _apply_row_filters,
    _attach_episode_metadata,
    _ensure_split,
)
from vla_lens.probes.workflow_targets import (
    _normalize_target_spec,
    _resolve_probe_target,
    _target_name,
)
from vla_lens.selectors import ActivationQuery
from vla_lens.table_io import read_optional_parquet
from vla_lens.traces import TraceDataset

PROBE_SCORE_CACHE_DIR = Path("tables") / "probe_score_cache"
PROBE_SCORE_CACHE_VERSION = "0.1.0"


@dataclass(frozen=True, slots=True)
class ProbeScoreCacheResult:
    artifact_id: str
    path: Path
    manifest_path: Path
    row_count: int
    trace_count: int
    labeled_row_count: int
    unlabeled_row_count: int
    target_name: str
    feature: str
    model: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        payload["manifest_path"] = str(self.manifest_path)
        return payload


def refresh_all_probe_score_caches(
    dataset: TraceDataset,
    *,
    artifact_ids: Sequence[str] | None = None,
) -> list[ProbeScoreCacheResult]:
    """Refresh score caches for all or selected dataset-level probe artifacts."""
    ids = list(artifact_ids or _probe_artifact_ids(dataset))
    results: list[ProbeScoreCacheResult] = []
    for artifact_id in ids:
        results.append(refresh_probe_score_cache(dataset, artifact_id))
    return results


def refresh_probe_score_cache(
    dataset: TraceDataset,
    artifact_id: str,
) -> ProbeScoreCacheResult:
    """Write a mutable score table for one frozen probe artifact."""
    artifact = dataset.load_artifact(artifact_id)
    if artifact.artifact_type != "probe_suite":
        raise ValueError(f"Artifact {artifact_id!r} is not a probe_suite artifact")

    selector = _artifact_selector(artifact)
    feature_matrix = dataset.select_model_sites(selector).materialize(cache=True)
    X = np.asarray(feature_matrix.X, dtype=np.float32)
    rows = feature_matrix.rows
    if rows.empty or X.shape[0] == 0:
        raise ValueError(f"Probe artifact {artifact_id!r} selector matched no rows")

    rows = _attach_episode_metadata(rows, dataset)
    X, rows, _ = _apply_row_filters(X, rows, _artifact_row_filters(artifact))
    rows = _ensure_split(
        rows,
        _artifact_split_column(artifact),
        train_value=_artifact_split_value(artifact, "train_value", "train"),
        test_value=_artifact_split_value(artifact, "test_value", "test"),
        split_kind=str(artifact.method.get("split_kind") or "random_episode"),
    )

    target_spec = _artifact_target_spec(artifact, rows)
    target_name = _target_name(target_spec)
    rows = _attach_optional_targets(dataset, rows, target_spec, target_name)

    best_state = _best_model_state(artifact)
    X, rows = _filter_best_sweep_rows(X, rows, artifact, best_state)
    model = _linear_probe_model(dataset, artifact, best_state)
    predictions = _score_rows(X, rows, artifact, target_name, model, best_state)

    path = probe_score_cache_path(dataset, artifact.artifact_id)
    manifest_path = probe_score_cache_manifest_path(dataset, artifact.artifact_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(path, index=False)
    manifest = _score_cache_manifest(
        dataset,
        artifact,
        predictions,
        selector=selector,
        target_name=target_name,
        best_state=best_state,
        feature_fingerprint=_array_fingerprint(X),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    labeled = int(predictions["actual"].notna().sum()) if "actual" in predictions else 0
    return ProbeScoreCacheResult(
        artifact_id=artifact.artifact_id,
        path=path,
        manifest_path=manifest_path,
        row_count=int(len(predictions)),
        trace_count=int(predictions["trace_id"].astype(str).nunique())
        if "trace_id" in predictions and not predictions.empty
        else 0,
        labeled_row_count=labeled,
        unlabeled_row_count=int(len(predictions) - labeled),
        target_name=target_name,
        feature=str(best_state.get("feature") or ""),
        model=str(best_state.get("model") or "linear"),
    )


def read_probe_score_cache(dataset: TraceDataset, artifact_id: str) -> pd.DataFrame:
    """Return the refreshable score cache table for an artifact, if present."""
    return read_optional_parquet(
        probe_score_cache_path(dataset, artifact_id),
        context=f"probe score cache {artifact_id}",
    )


def probe_score_cache_path(dataset: TraceDataset, artifact_id: str) -> Path:
    return dataset._dataset_artifact_root() / PROBE_SCORE_CACHE_DIR / f"{artifact_id}.parquet"


def probe_score_cache_manifest_path(dataset: TraceDataset, artifact_id: str) -> Path:
    return dataset._dataset_artifact_root() / PROBE_SCORE_CACHE_DIR / f"{artifact_id}.json"


def _probe_artifact_ids(dataset: TraceDataset) -> list[str]:
    table = dataset.artifact_index
    if table.empty or "artifact_type" not in table or "artifact_id" not in table:
        return []
    rows = table.loc[table["artifact_type"].astype(str) == "probe_suite"]
    return [str(value) for value in rows["artifact_id"].dropna()]


def _artifact_selector(artifact: LensArtifact) -> ActivationQuery:
    method_input = artifact.method.get("input") if isinstance(artifact.method, Mapping) else None
    selector_payload = (
        method_input.get("selector")
        if isinstance(method_input, Mapping) and isinstance(method_input.get("selector"), Mapping)
        else artifact.selector
    )
    payload = dict(selector_payload or {})
    return ActivationQuery(
        episodes=dict(payload.get("episodes") or {}),
        name=_optional_str(payload.get("name")),
        module=_optional_str(payload.get("module")),
        layers=payload.get("layers"),
        tensor_type=_optional_str(payload.get("tensor_type")),
        token_kind=_optional_str(payload.get("token_kind")),
        timesteps=payload.get("timesteps", "all"),
        policy_calls=payload.get("policy_calls", "all"),
        generation_step=payload.get("generation_step"),
        reduce_tokens=str(payload.get("reduce_tokens") or payload.get("reduction") or "mean"),
        dtype=str(payload.get("dtype") or "float32"),
    )


def _artifact_row_filters(artifact: LensArtifact) -> list[Mapping[str, Any]]:
    row_filter = artifact.method.get("row_filter") if isinstance(artifact.method, Mapping) else None
    if isinstance(row_filter, Mapping):
        filters = row_filter.get("filters")
        if isinstance(filters, Sequence) and not isinstance(filters, str):
            return [dict(item) for item in filters if isinstance(item, Mapping)]
    return []


def _artifact_split_column(artifact: LensArtifact) -> str:
    split = artifact.method.get("split") if isinstance(artifact.method, Mapping) else None
    if isinstance(split, Mapping):
        return str(split.get("column") or artifact.method.get("split_column") or "split")
    return str(artifact.method.get("split_column") or "split")


def _artifact_split_value(artifact: LensArtifact, key: str, fallback: str) -> str:
    split = artifact.method.get("split") if isinstance(artifact.method, Mapping) else None
    if isinstance(split, Mapping) and split.get(key) is not None:
        return str(split[key])
    return str(artifact.method.get(key) or fallback)


def _artifact_target_spec(artifact: LensArtifact, rows: pd.DataFrame) -> dict[str, Any]:
    target = artifact.method.get("target") if isinstance(artifact.method, Mapping) else None
    if isinstance(target, Mapping):
        target_name = str(target.get("name") or target.get("resolved_column") or "target")
        if target_name in rows:
            return _normalize_target_spec(
                {
                    "name": target_name,
                    "kind": target.get("kind") or target_name,
                    "source": "row",
                    "column": target_name,
                    "transform": target.get("transform") or {"kind": "identity"},
                }
            )
        return _normalize_target_spec(
            {
                "name": target_name,
                "kind": target.get("kind") or target_name,
                "source": target.get("source") or "row",
                "column": target.get("resolved_column") or target_name,
                "selector": target.get("selector"),
                "alignment": target.get("alignment"),
                "transform": target.get("transform") or {"kind": "identity"},
            }
        )
    target_name = str(artifact.metrics.get("target") or artifact.display.get("target") or "target")
    return _normalize_target_spec({"name": target_name, "source": "row", "column": target_name})


def _attach_optional_targets(
    dataset: TraceDataset,
    rows: pd.DataFrame,
    target_spec: Mapping[str, Any],
    target_name: str,
) -> pd.DataFrame:
    try:
        return _resolve_probe_target(dataset, rows, target_spec)
    except (KeyError, ValueError):
        out = rows.copy()
        if target_name not in out:
            out[target_name] = pd.Series([None] * len(out), dtype=object)
        return out


def _best_model_state(artifact: LensArtifact) -> Mapping[str, Any]:
    probe = artifact.method.get("probe") if isinstance(artifact.method, Mapping) else None
    state = probe.get("best_model_state") if isinstance(probe, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError(f"Probe artifact {artifact.artifact_id!r} has no best_model_state")
    model = str(state.get("model") or "linear")
    if model != "linear":
        raise ValueError(
            f"Probe score refresh only supports saved linear probes, got model={model!r}"
        )
    return state


def _filter_best_sweep_rows(
    X: np.ndarray,
    rows: pd.DataFrame,
    artifact: LensArtifact,
    best_state: Mapping[str, Any],
) -> tuple[np.ndarray, pd.DataFrame]:
    sweep_columns = _normalize_sweep_columns(artifact.method.get("sweep"))
    if not sweep_columns:
        return X, rows.reset_index(drop=True)
    sweep_value = _sweep_value_from_best_state(sweep_columns, best_state)
    if len(sweep_columns) == 1:
        expected = _normalized_value(sweep_value)
        column = sweep_columns[0]
        if column not in rows:
            raise KeyError(f"Probe score refresh missing sweep column {column!r}")
        mask = rows[column].map(_normalized_value) == expected
    else:
        if not isinstance(sweep_value, Mapping):
            raise ValueError("Multi-column probe sweep requires mapping sweep_value")
        mask = pd.Series(True, index=rows.index)
        for column in sweep_columns:
            if column not in rows:
                raise KeyError(f"Probe score refresh missing sweep column {column!r}")
            mask &= rows[column].map(_normalized_value) == _normalized_value(
                sweep_value.get(column)
            )
    kept = mask.to_numpy(dtype=bool)
    return X[kept], rows.loc[mask].reset_index(drop=True)


def _sweep_value_from_best_state(
    sweep_columns: Sequence[str],
    best_state: Mapping[str, Any],
) -> Any:
    sweep_value = best_state.get("sweep_value")
    if sweep_value is not None:
        return sweep_value
    feature = str(best_state.get("feature") or "")
    if len(sweep_columns) == 1:
        column = sweep_columns[0]
        prefix = f"{column} "
        if feature.startswith(prefix):
            return feature[len(prefix) :]
        prefix = f"{column}="
        if feature.startswith(prefix):
            return feature[len(prefix) :]
        return sweep_value
    parsed = _parse_feature_sweep_mapping(feature)
    if parsed:
        return {column: parsed.get(column) for column in sweep_columns}
    return sweep_value


def _parse_feature_sweep_mapping(feature: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in feature.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        if key:
            out[key] = value.strip()
    return out


def _linear_probe_model(
    dataset: TraceDataset,
    artifact: LensArtifact,
    best_state: Mapping[str, Any],
) -> dict[str, Any]:
    required = ["weights", "bias", "feature_mean", "feature_scale"]
    missing = [name for name in required if name not in artifact.arrays]
    if missing:
        raise ValueError(
            f"Probe artifact {artifact.artifact_id!r} cannot be refreshed; "
            f"missing arrays: {', '.join(missing)}"
        )
    weights = np.asarray(dataset.load_artifact_array(artifact, "weights"), dtype=np.float32)
    bias = np.asarray(dataset.load_artifact_array(artifact, "bias"), dtype=np.float32).reshape(-1)
    mean = np.asarray(dataset.load_artifact_array(artifact, "feature_mean"), dtype=np.float32)
    scale = np.asarray(dataset.load_artifact_array(artifact, "feature_scale"), dtype=np.float32)
    scale = np.where(scale == 0, 1.0, scale)
    return {
        "weights": weights,
        "bias": bias,
        "feature_mean": mean,
        "feature_scale": scale,
        "classes": [str(value) for value in best_state.get("classes") or []],
        "probe_type": str(best_state.get("probe_type") or "classification"),
    }


def _score_rows(
    X: np.ndarray,
    rows: pd.DataFrame,
    artifact: LensArtifact,
    target_name: str,
    model: Mapping[str, Any],
    best_state: Mapping[str, Any],
) -> pd.DataFrame:
    if X.shape[0] != len(rows):
        raise ValueError(f"Score row mismatch: X has {X.shape[0]} rows, metadata has {len(rows)}")
    mean = np.asarray(model["feature_mean"], dtype=np.float32)
    scale = np.asarray(model["feature_scale"], dtype=np.float32)
    if X.shape[1] != mean.shape[0]:
        raise ValueError(
            "Probe feature dimension mismatch: "
            f"selected rows have {X.shape[1]} features, saved model expects {mean.shape[0]}"
        )
    normalized = (X.astype(np.float32, copy=False) - mean) / scale
    weights = np.asarray(model["weights"], dtype=np.float32)
    bias = np.asarray(model["bias"], dtype=np.float32)
    logits = normalized @ weights.T + bias.reshape(1, -1)
    target_kind = str(model.get("probe_type") or "classification")
    if target_kind == "classification":
        predicted, confidence = _classification_predictions(logits, model)
    else:
        values = logits.reshape(len(rows), -1)[:, 0]
        predicted = [_optional_float(value) for value in values]
        confidence = [None] * len(rows)

    split_column = _artifact_split_column(artifact)
    records: list[dict[str, Any]] = []
    for index, row in rows.iterrows():
        actual = _target_value(row.get(target_name))
        prediction = predicted[index]
        correct = _correct_value(actual, prediction)
        join = _prediction_join_keys(
            row,
            target=target_name,
            split_value=None,
            split_column=split_column,
        )
        records.append(
            {
                **join,
                "target_kind": target_kind,
                "target_dim": 0,
                "target_value": actual,
                "prediction_value": prediction,
                "trace_id": str(row.get("trace_id", "")),
                "episode_id": _optional_str(row.get("episode_id")),
                "timestep": _optional_int(row.get("timestep")),
                "actual": actual,
                "predicted": prediction,
                "correct": correct,
                "confidence": confidence[index],
                "prediction_kind": "class_label"
                if target_kind == "classification"
                else "continuous_value",
                "model": str(best_state.get("model") or "linear"),
                "feature": str(best_state.get("feature") or ""),
                "eval_split": str(best_state.get("split_value") or ""),
                "primary_metric": str(best_state.get("primary_metric") or ""),
                "score_source": "probe_score_cache",
                "artifact_id": artifact.artifact_id,
            }
        )
    return pd.DataFrame.from_records(records)


def _classification_predictions(
    logits: np.ndarray,
    model: Mapping[str, Any],
) -> tuple[list[str], list[float | None]]:
    classes = [str(value) for value in model.get("classes") or []]
    if logits.shape[1] == 1:
        if len(classes) < 2:
            classes = ["0", "1"]
        probabilities = _sigmoid(logits[:, 0])
        predicted = [classes[1] if value > 0.5 else classes[0] for value in probabilities]
        confidence = [float(max(value, 1.0 - value)) for value in probabilities]
        return predicted, confidence
    if len(classes) != logits.shape[1]:
        classes = [str(index) for index in range(logits.shape[1])]
    probabilities = _softmax(logits)
    indices = np.argmax(probabilities, axis=1)
    predicted = [classes[int(index)] for index in indices]
    confidence = [float(probabilities[row_index, index]) for row_index, index in enumerate(indices)]
    return predicted, confidence


def _score_cache_manifest(
    dataset: TraceDataset,
    artifact: LensArtifact,
    predictions: pd.DataFrame,
    *,
    selector: ActivationQuery,
    target_name: str,
    best_state: Mapping[str, Any],
    feature_fingerprint: str,
) -> dict[str, Any]:
    created_utc = datetime.now(UTC).isoformat()
    return {
        "schema_version": PROBE_SCORE_CACHE_VERSION,
        "artifact_id": artifact.artifact_id,
        "artifact_created_utc": artifact.created_utc,
        "target_name": target_name,
        "row_count": int(len(predictions)),
        "trace_count": int(predictions["trace_id"].astype(str).nunique())
        if "trace_id" in predictions and not predictions.empty
        else 0,
        "selector": selector.to_dict(),
        "best_model_state": {
            "feature": best_state.get("feature"),
            "sweep_value": best_state.get("sweep_value"),
            "model": best_state.get("model"),
            "probe_type": best_state.get("probe_type"),
            "split_value": best_state.get("split_value"),
            "primary_metric": best_state.get("primary_metric"),
        },
        "feature_matrix_fingerprint": feature_fingerprint,
        "dataset_root": str(dataset.root),
        "created_utc": created_utc,
        "updated_utc": created_utc,
        "semantics": {
            "kind": "mutable_probe_score_cache",
            "metrics_frozen": True,
            "correctness_requires_label": True,
        },
    }


def _normalize_sweep_columns(value: Any) -> list[str]:
    if isinstance(value, str):
        if value in {"", "none", "null"}:
            return []
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item) not in {"", "none", "null"}]
    return []


def _normalized_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            numeric = float(stripped)
        except ValueError:
            return stripped
        if numeric.is_integer():
            return str(int(numeric))
        return stripped
    if isinstance(value, (Mapping, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _target_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, str)):
        return str(value)
    if isinstance(value, (int, float)):
        return _optional_float(value) if isinstance(value, float) else str(value)
    return str(value)


def _correct_value(actual: Any, predicted: Any) -> bool | None:
    if actual is None or predicted is None:
        return None
    return str(actual) == str(predicted)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    out = float(value)
    return out if np.isfinite(out) else None
