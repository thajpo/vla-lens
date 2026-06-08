"""Probe metadata helpers for EpisodeLensView payloads."""

from __future__ import annotations

from typing import Any, Mapping

from vla_lens.artifacts import LensArtifact
from vla_lens.probes.score_cache import _artifact_selector
from vla_lens.server.common import _jsonable
from vla_lens.server.episode_lens_common import (
    _first_present,
    _human_label,
    _optional_float,
    _optional_int,
)


def _probe_best_model_state(artifact: LensArtifact) -> dict[str, Any]:
    probe = artifact.method.get("probe") if isinstance(artifact.method, Mapping) else {}
    state = probe.get("best_model_state") if isinstance(probe, Mapping) else {}
    return dict(state) if isinstance(state, Mapping) else {}


def _probe_spec_payload(
    artifact: LensArtifact,
    best_state: Mapping[str, Any],
) -> dict[str, Any]:
    method = artifact.method if isinstance(artifact.method, Mapping) else {}
    method_input = method.get("input") if isinstance(method.get("input"), Mapping) else {}
    method_target = method.get("target") if isinstance(method.get("target"), Mapping) else {}
    probe = method.get("probe") if isinstance(method.get("probe"), Mapping) else {}
    evaluation = method.get("evaluation") if isinstance(method.get("evaluation"), Mapping) else {}
    target = _first_present(
        method_target.get("name"),
        method_target.get("resolved_column"),
        artifact.metrics.get("target"),
        artifact.display.get("target"),
    )
    classes = [str(item) for item in best_state.get("classes") or []]
    output = (
        f"{classes[0]} / {classes[1]}"
        if len(classes) == 2
        else str(method_target.get("kind") or best_state.get("probe_type") or "prediction")
    )
    return {
        "prediction": _human_label(target or "prediction"),
        "input": _probe_input_label(method_input),
        "output": output,
        "objective": _probe_objective_label(probe, best_state),
        "metric": _human_label(
            _first_present(
                evaluation.get("primary_metric"),
                artifact.metrics.get("best_primary_metric"),
                best_state.get("primary_metric"),
            )
            or "score"
        ),
    }


def _probe_identity_payload(
    artifact: Mapping[str, Any],
    artifact_object: LensArtifact,
    best_state: Mapping[str, Any],
) -> dict[str, Any]:
    spec = _probe_spec_payload(artifact_object, best_state)
    classes = [str(item) for item in best_state.get("classes") or []]
    metric_value = _optional_float(artifact_object.metrics.get("best_score"))
    return {
        "suite_id": str(artifact.get("artifact_id") or artifact_object.artifact_id),
        "probe_id": str(artifact.get("artifact_id") or artifact_object.artifact_id),
        "probe_key": None,
        "probe_name": str(artifact.get("name") or artifact_object.name),
        "prediction_target": spec.get("prediction"),
        "input_space": spec.get("input"),
        "output_space": spec.get("output"),
        "objective": spec.get("objective"),
        "trained_layers": _probe_trained_layers(artifact_object),
        "trained_token_kinds": _probe_trained_token_kinds(artifact_object),
        "trained_policy_call_scope": _probe_policy_call_scope(artifact_object),
        "training_spec": _probe_training_spec_payload(artifact_object, best_state),
        "classification": _classification_payload(best_state),
        "metric": {
            "name": spec.get("metric"),
            "value": metric_value,
            "split": _first_present(
                artifact_object.metrics.get("best_eval_split"),
                best_state.get("split_value"),
            ),
        },
        "classes": classes,
    }


def _probe_training_spec_payload(
    artifact: LensArtifact,
    best_state: Mapping[str, Any],
) -> dict[str, Any]:
    selector = _artifact_selector(artifact)
    method = artifact.method if isinstance(artifact.method, Mapping) else {}
    method_target = method.get("target") if isinstance(method.get("target"), Mapping) else {}
    split = method.get("split") if isinstance(method.get("split"), Mapping) else {}
    return {
        "model": str(best_state.get("model") or "linear"),
        "probe_type": str(best_state.get("probe_type") or "classification"),
        "target_column": _first_present(
            method_target.get("resolved_column"),
            method_target.get("name"),
            artifact.metrics.get("target"),
        ),
        "feature_space": _probe_input_label(
            method.get("input") if isinstance(method.get("input"), Mapping) else {}
        ),
        "layers": _probe_trained_layers(artifact),
        "policy_calls": _jsonable(_probe_policy_calls_payload(selector.policy_calls)),
        "token_kinds": _probe_trained_token_kinds(artifact),
        "split_column": split.get("column") if isinstance(split, Mapping) else None,
    }


def _classification_payload(best_state: Mapping[str, Any]) -> dict[str, Any]:
    classes = [str(item) for item in best_state.get("classes") or []]
    probe_type = str(best_state.get("probe_type") or "unknown")
    if probe_type != "classification":
        return {"kind": "regression" if probe_type == "regression" else "unknown"}
    if len(classes) == 2:
        return {
            "kind": "binary",
            "negative_class": classes[0],
            "positive_class": classes[1],
            "logit_class": classes[1],
        }
    return {"kind": "multiclass", "classes": classes}


def _probe_trained_layers(artifact: LensArtifact) -> list[int]:
    selector = _artifact_selector(artifact)
    if selector.layers is not None:
        return [int(layer) for layer in selector.layers]
    return []


def _probe_trained_token_kinds(artifact: LensArtifact) -> list[str]:
    selector = _artifact_selector(artifact)
    return [selector.token_kind] if selector.token_kind else []


def _probe_policy_call_scope(artifact: LensArtifact) -> str:
    selector = _artifact_selector(artifact)
    if selector.policy_calls == "all":
        return "all"
    if selector.policy_calls is None or selector.policy_calls == "":
        return "unknown"
    return "selected"


def _probe_policy_calls_payload(value: Any) -> str | list[int] | dict[str, int] | None:
    if value == "all":
        return "all"
    if value is None or value == "":
        return "unknown"
    if isinstance(value, range):
        return {"start": int(value.start), "end": int(value.stop - 1)}
    if isinstance(value, (list, tuple, set)):
        out = [_optional_int(item) for item in value]
        parsed = [item for item in out if item is not None]
        return sorted(set(parsed)) if parsed else "unknown"
    parsed = _optional_int(value)
    if parsed is not None:
        return [parsed]
    return str(value)


def _probe_input_label(method_input: Mapping[str, Any]) -> str:
    selector = (
        method_input.get("selector") if isinstance(method_input.get("selector"), Mapping) else {}
    )
    tensor = _first_present(selector.get("tensor_type"), "hidden states")
    token = selector.get("token_kind")
    if token:
        return _human_label(f"{token} {tensor}")
    return _human_label(tensor)


def _probe_objective_label(probe: Mapping[str, Any], best_state: Mapping[str, Any]) -> str:
    model = str(best_state.get("model") or probe.get("primary_model") or "linear")
    probe_type = str(best_state.get("probe_type") or probe.get("type") or "classification")
    if model == "linear" and probe_type == "classification":
        return "Logistic regression"
    if model == "linear" and probe_type == "regression":
        return "Ridge regression"
    if model == "mlp":
        return "MLP"
    return _human_label(f"{model} {probe_type}")
