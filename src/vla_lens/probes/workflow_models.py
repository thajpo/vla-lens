"""Fitted-model contracts and bounded candidate-readout retention."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.probes.workflow_artifacts import (
    _array_fingerprint,
    _readout_id,
    _readout_key,
    _result_model_arrays,
)


def _probe_model_contract(
    model_arrays: Mapping[str, np.ndarray],
    model_state: Mapping[str, Any],
    *,
    hyperparameters: Mapping[str, Any],
    feature_matrix: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    model_name = str(model_state.get("model") or "")
    probe_type = str(model_state.get("probe_type") or "unknown")
    numeric_precision = {name: str(np.asarray(value).dtype) for name, value in model_arrays.items()}
    common = {
        "probe_type": probe_type,
        "feature_dim": int(np.asarray(model_arrays["feature_mean"]).size),
        "classes": list(model_state.get("classes") or []),
        "array_fingerprints": {
            name: _array_fingerprint(value) for name, value in model_arrays.items()
        },
        "hyperparameters": dict(hyperparameters.get(model_name) or {}),
        "selected_readout": {
            "feature": model_state.get("feature"),
            "sweep_value": model_state.get("sweep_value"),
            "model": model_name,
            "selection_split": model_state.get("split_value"),
            "primary_metric": model_state.get("primary_metric"),
        },
        "numeric_precision": numeric_precision,
        "prediction_tolerance": (
            _floating_replay_tolerance(
                model_arrays,
                feature_matrix=feature_matrix,
                predictions=predictions,
            )
            if probe_type == "regression"
            else None
        ),
    }
    if model_name == "linear":
        required = {"weights", "bias", "feature_mean", "feature_scale"}
        missing = sorted(required - set(model_arrays))
        if missing:
            raise ValueError(f"Selected linear probe is missing fitted arrays: {missing}")
        return {
            **common,
            "format": "standardized_linear_v1",
            "array_names": {name: name for name in sorted(required)},
        }
    if model_name == "mlp":
        weight_names = sorted(
            (name for name in model_arrays if name.startswith("layer_weights_")),
            key=lambda value: int(value.rsplit("_", 1)[1]),
        )
        bias_names = sorted(
            (name for name in model_arrays if name.startswith("layer_biases_")),
            key=lambda value: int(value.rsplit("_", 1)[1]),
        )
        if not weight_names or len(weight_names) != len(bias_names):
            raise ValueError("Selected MLP probe is missing fitted layer arrays")
        return {
            **common,
            "format": "standardized_mlp_v1",
            "activation": model_state.get("activation"),
            "out_activation": model_state.get("out_activation"),
            "array_names": {
                "feature_mean": "feature_mean",
                "feature_scale": "feature_scale",
                "layer_weights": weight_names,
                "layer_biases": bias_names,
            },
        }
    raise ValueError(f"Selected probe model {model_name!r} cannot be saved for reuse")


def _retained_readout_contracts(
    results: pd.DataFrame,
    *,
    selection_value: str,
    feature_matrix: np.ndarray,
    hyperparameters: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    selection_rows = results.loc[
        results["split_value"].astype(str) == str(selection_value)
    ].copy()
    selection_rows["delta"] = selection_rows["score"] - selection_rows["baseline_score"]
    retained_rows = (
        selection_rows.sort_values("delta", ascending=False)
        .groupby("model", sort=True, as_index=False)
        .head(1)
    )
    readouts: list[dict[str, Any]] = []
    saved_arrays: dict[str, np.ndarray] = {}
    for _, row in retained_rows.iterrows():
        arrays, state = _result_model_arrays(row)
        records = list(row.get("all_prediction_records") or [])
        indices = np.asarray(
            [int(record["prepared_row_index"]) for record in records],
            dtype=np.int64,
        )
        candidate_features = np.asarray(feature_matrix[indices])
        candidate_predictions = np.asarray(
            [record.get("prediction_value") for record in records]
        )
        contract = _probe_model_contract(
            arrays,
            state,
            hyperparameters=hyperparameters,
            feature_matrix=candidate_features,
            predictions=candidate_predictions,
        )
        readout_id = _readout_id(_readout_key(row))
        prefix = f"{readout_id}__"
        renamed = {f"{prefix}{name}": value for name, value in arrays.items()}
        saved_arrays.update(renamed)
        readouts.append(
            {
                "readout_id": readout_id,
                "capabilities": {"use": True, "replay": False},
                "selection_score": float(row["score"]),
                "selection_baseline_score": float(row["baseline_score"]),
                "selection_delta": float(row["delta"]),
                "model": _prefixed_model_contract(contract, prefix=prefix),
            }
        )
    return readouts, saved_arrays


def _prefixed_model_contract(model: Mapping[str, Any], *, prefix: str) -> dict[str, Any]:
    out = dict(model)
    array_names = dict(out["array_names"])
    for key, value in list(array_names.items()):
        if isinstance(value, str):
            array_names[key] = f"{prefix}{value}"
        elif isinstance(value, Sequence):
            array_names[key] = [f"{prefix}{name}" for name in value]
    out["array_names"] = array_names
    out["array_fingerprints"] = {
        f"{prefix}{name}": fingerprint
        for name, fingerprint in dict(out["array_fingerprints"]).items()
    }
    out["numeric_precision"] = {
        f"{prefix}{name}": dtype
        for name, dtype in dict(out.get("numeric_precision") or {}).items()
    }
    return out


def _floating_replay_tolerance(
    model_arrays: Mapping[str, np.ndarray],
    *,
    feature_matrix: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, Any]:
    arrays = [np.asarray(value) for value in model_arrays.values()]
    arrays.append(np.asarray(feature_matrix))
    floating_dtypes = [value.dtype for value in arrays if np.issubdtype(value.dtype, np.floating)]
    if not floating_dtypes:
        return {
            "absolute": 0.0,
            "relative": 0.0,
            "least_precise_dtype": None,
            "operation_count": 0,
            "prediction_scale": 0.0,
            "estimated_relative_error": 0.0,
            "maximum_relative_error": 0.0,
        }
    least_precise_dtype = max(floating_dtypes, key=lambda dtype: float(np.finfo(dtype).eps))
    epsilon = float(np.finfo(least_precise_dtype).eps)
    feature_dim = int(feature_matrix.shape[1]) if feature_matrix.ndim == 2 else 0
    layer_operation_count = sum(
        int(np.asarray(value).shape[0])
        for name, value in model_arrays.items()
        if name.startswith("layer_weights_") and np.asarray(value).ndim > 1
    )
    operation_count = max(1, layer_operation_count or feature_dim)
    estimated_relative_error = epsilon * operation_count * 4.0
    maximum_relative_error = 1e-4
    total_relative_error = min(estimated_relative_error, maximum_relative_error)
    component_tolerance = total_relative_error / 2.0
    numeric_predictions = pd.to_numeric(pd.Series(predictions), errors="coerce").to_numpy(
        dtype=np.float64
    )
    finite_predictions = numeric_predictions[np.isfinite(numeric_predictions)]
    prediction_scale = (
        float(np.max(np.abs(finite_predictions))) if len(finite_predictions) else 0.0
    )
    return {
        "absolute": component_tolerance * max(1.0, prediction_scale),
        "relative": component_tolerance,
        "least_precise_dtype": str(least_precise_dtype),
        "operation_count": operation_count,
        "prediction_scale": prediction_scale,
        "estimated_relative_error": estimated_relative_error,
        "maximum_relative_error": maximum_relative_error,
    }
