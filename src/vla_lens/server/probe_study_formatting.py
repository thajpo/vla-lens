"""Probe-study identifiers, display copy, and scalar normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from vla_lens.server.common import _json_parse
from vla_lens.traces import TraceDataset


def _load_artifact_or_record(
    dataset: TraceDataset,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_id = str(record.get("artifact_id") or "")
    try:
        artifact = dataset.load_artifact(artifact_id)
        return artifact.to_dict()
    except Exception:
        return _artifact_record_payload(record)


def _artifact_record_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(key): _clean_scalar(value) for key, value in record.items()}
    for key in ["selector", "method", "metrics", "arrays", "display", "tags", "source_trace_ids"]:
        payload[key] = _json_parse(payload.get(key))
    return payload


def _mapping(value: Any) -> dict[str, Any]:
    parsed = _json_parse(value)
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _readout_id(target: str, layer: Any, split: str, status: str) -> str:
    layer_value = layer if layer is not None else "all"
    split_value = split or "all"
    return f"{target or 'target'}|layer:{layer_value}|split:{split_value}|{status}"


def _trained_probe_id(target: str, layer: Any, split: str) -> str:
    return "-".join(
        [
            _target_code(target),
            f"L{_id_piece(layer, fallback='ALL')}",
            _id_piece(split, fallback="NOSPLIT"),
        ]
    )


def _target_code(target: str) -> str:
    labels = {
        "active_manipulated_object": "AMO",
        "active_receptacle_object": "ARO",
        "next_manipulated_object": "NMO",
        "task_phase": "TPH",
    }
    if target in labels:
        return labels[target]
    pieces = [piece for piece in str(target or "").split("_") if piece]
    if len(pieces) >= 2:
        return "".join(piece[0].upper() for piece in pieces[:4])
    return _id_piece(target, fallback="TARGET")[:10]


def _id_piece(value: Any, *, fallback: str) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = "" if value is None else str(value).strip().upper()
    text = "".join(char if char.isalnum() else "-" for char in text)
    text = "-".join(piece for piece in text.split("-") if piece)
    return text or fallback


def _split_category(split: str) -> str | None:
    text = str(split or "").lower()
    if not text:
        return None
    if text == "train" or text.startswith("train"):
        return "train"
    if "val" in text or "validation" in text:
        return "validation"
    if "test" in text:
        return "test"
    return text


def _question_label(target: str) -> str:
    if target == "next_manipulated_object":
        return "Which object will the robot manipulate next before contact?"
    if target == "active_manipulated_object":
        return "Which object is the robot currently manipulating?"
    if target == "active_receptacle_object":
        return "Which receptacle is active in the current interaction?"
    if target == "task_phase":
        return "Which object-centric phase is the robot in?"
    return target.replace("_", " ") if target else "Probe study"


def _prediction_label(target: str) -> str:
    labels = {
        "next_manipulated_object": "Next manipulated object",
        "active_manipulated_object": "Active manipulated object",
        "active_receptacle_object": "Active receptacle",
        "task_phase": "Task phase",
    }
    return labels.get(target, target.replace("_", " ").strip().capitalize() if target else "Target")


def _input_label(selector: Mapping[str, Any], method: Mapping[str, Any]) -> str:
    site = selector.get("site") or selector.get("model_site")
    token_space = selector.get("token_space")
    layers = selector.get("layers")
    if not site and isinstance(method.get("feature_cache"), Mapping):
        feature_cache = method["feature_cache"]
        site = feature_cache.get("site") or feature_cache.get("model_site")
        token_space = feature_cache.get("token_space")
        layers = feature_cache.get("layers") or layers
    pieces = ["Expert hidden states"]
    if site:
        pieces.append(str(site))
    if layers:
        if isinstance(layers, list):
            pieces.append(f"layers {', '.join(str(item) for item in layers)}")
        else:
            pieces.append(f"layers {layers}")
    if token_space:
        pieces.append(str(token_space))
    return " / ".join(pieces)


def _output_label(summary: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    class_count = _clean_int(summary.get("class_count") or metrics.get("class_count"))
    if class_count:
        return f"{class_count} object classes"
    return "Class label"


def _objective_label(
    method: Mapping[str, Any],
    summary: Mapping[str, Any] | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> str:
    training = _training_summary(method, summary or {}, metrics or {})
    if training.get("objective"):
        return str(training["objective"])
    probe = method.get("probe")
    if isinstance(probe, Mapping):
        model = probe.get("model") or probe.get("classifier")
        if model:
            return str(model).replace("_", " ")
    model = method.get("model") or method.get("classifier")
    return str(model).replace("_", " ") if model else "Linear readout"


def _training_summary(
    method: Mapping[str, Any],
    summary: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    probe = _mapping(method.get("probe"))
    target = _mapping(method.get("target"))
    evaluation = _mapping(method.get("evaluation"))
    normalization = _mapping(method.get("normalization"))
    best_state = _mapping(probe.get("best_model_state"))
    hyperparams = _mapping(probe.get("hyperparams"))

    primary_model = str(
        probe.get("primary_model")
        or best_state.get("model")
        or method.get("model")
        or method.get("classifier")
        or ""
    )
    target_type = str(
        target.get("kind")
        or probe.get("type")
        or best_state.get("probe_type")
        or ""
    )
    class_count = _clean_int(summary.get("class_count") or metrics.get("class_count"))
    objective = _training_objective_label(primary_model, target_type, class_count)
    params = _mapping(hyperparams.get(primary_model)) or _mapping(
        hyperparams.get(primary_model.lower())
    )
    estimator = str(params.get("model") or primary_model or "").replace("_", " ")
    preprocessing = _normalization_label(normalization)
    hyperparameter_lines = _hyperparameter_lines(params)

    split = _mapping(method.get("split"))
    trained_on = _clean_scalar(split.get("train_value") or probe.get("trained_on_split"))
    selected_on = _clean_scalar(
        split.get("selection_value")
        or evaluation.get("selection_split")
        or method.get("selection_value")
    )
    metric = _clean_scalar(evaluation.get("primary_metric") or evaluation.get("metric"))

    payload = {
        "objective": objective,
        "target_type": _clean_scalar(target_type.replace("_", " ") if target_type else None),
        "estimator": _clean_scalar(estimator),
        "library": _clean_scalar(probe.get("library")),
        "preprocessing": preprocessing,
        "hyperparameters": hyperparameter_lines,
        "trained_on": trained_on,
        "selected_on": selected_on,
        "metric": _clean_scalar(str(metric).replace("_", " ") if metric else None),
    }
    return {key: value for key, value in payload.items() if value not in (None, "", [])}


def _training_objective_label(model: str, target_type: str, class_count: int | None) -> str | None:
    model_key = model.lower()
    target_key = target_type.lower()
    regression = "regression" in target_key or target_key == "continuous"
    classification = "classification" in target_key or bool(class_count)
    if "linear" in model_key:
        if regression:
            return "Ridge regression"
        if classification:
            return (
                "Multiclass logistic regression"
                if class_count and class_count > 2
                else "Binary logistic regression"
            )
    if "mlp" in model_key:
        if regression:
            return "MLP regression"
        if classification:
            return (
                "Multiclass MLP classifier"
                if class_count and class_count > 2
                else "Binary MLP classifier"
            )
    if model:
        if regression:
            return f"{model.replace('_', ' ')} regression"
        if classification:
            return f"{model.replace('_', ' ')} classification"
    if regression:
        return "Regression probe"
    if classification:
        return "Classification probe"
    return None


def _normalization_label(normalization: Mapping[str, Any]) -> str | None:
    method = str(normalization.get("method") or "")
    if method == "standardize":
        return "standardized X"
    return method.replace("_", " ") if method else None


def _hyperparameter_lines(params: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in ["class_weight", "alpha", "max_iter", "hidden_layer_sizes", "random_state"]:
        value = params.get(key)
        if value is None:
            continue
        clean = _clean_scalar(value)
        if isinstance(clean, list):
            clean = ", ".join(str(item) for item in clean)
        if key == "class_weight" and clean == "balanced":
            lines.append("class-balanced")
        else:
            lines.append(f"{key.replace('_', ' ')} {clean}")
    return lines


def _clean_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _clean_scalar(value) for key, value in row.items()}


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return _clean_scalar(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        return value if value == value and value not in {float("inf"), float("-inf")} else None
    if isinstance(value, Path):
        return str(value)
    return value


def _clean_float(value: Any) -> float | None:
    value = _clean_scalar(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in {float("inf"), float("-inf")} else None


def _clean_int(value: Any) -> int | None:
    number = _clean_float(value)
    return int(number) if number is not None else None


def _clean_bool(value: Any) -> bool | None:
    value = _clean_scalar(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    return None


def _optional_text(value: Any) -> str:
    value = _clean_scalar(value)
    return "" if value is None else str(value)
