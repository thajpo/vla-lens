"""Probe representation choices shown before training.

The generic probe trainer currently consumes one two-dimensional feature matrix:
one row per example and one pooled feature vector per row. Richer representation
choices may be supported by the saved capture while still requiring a dedicated
runner. Keeping those states separate prevents a requested tokenwise or object-
conditioned experiment from silently becoming a mean-pooled probe.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import pandas as pd

from vla_lens.traces import TraceDataset

DEFAULT_REPRESENTATION_KIND = "mean_pool"
GENERIC_PROBE_REPRESENTATION_KINDS = frozenset({DEFAULT_REPRESENTATION_KIND})

_ALIASES = {
    "mean": "mean_pool",
    "mean_pooled": "mean_pool",
    "layer_mix": "learned_layer_mix",
    "learned_mix": "learned_layer_mix",
    "tokens": "tokenwise",
    "object_query": "object_conditioned",
    "slots": "set_decoder",
    "slot_decoder": "set_decoder",
}


def normalize_representation_spec(value: Any) -> dict[str, Any]:
    """Normalize a probe representation request without claiming runner support."""

    if value is None or value == "":
        return {"kind": DEFAULT_REPRESENTATION_KIND}
    if isinstance(value, str):
        parsed: dict[str, Any] = {"kind": value}
    elif isinstance(value, Mapping):
        parsed = dict(value)
    else:
        raise TypeError("Probe representation must be a string or mapping")
    kind = str(parsed.get("kind") or DEFAULT_REPRESENTATION_KIND).strip().lower()
    parsed["kind"] = _ALIASES.get(kind, kind)
    return parsed


def require_generic_probe_representation(value: Any) -> dict[str, Any]:
    """Reject richer choices instead of silently training the pooled fallback."""

    normalized = normalize_representation_spec(value)
    kind = str(normalized["kind"])
    if kind not in GENERIC_PROBE_REPRESENTATION_KINDS:
        raise ValueError(
            f"Representation {kind!r} is not supported by the generic probe trainer. "
            "Run probe preflight to review its data requirements and use or add the "
            "specialized runner named in representation_options."
        )
    return normalized


def probe_representation_options(
    dataset: TraceDataset,
    feature_rows: pd.DataFrame,
    selected: Any = None,
) -> dict[str, Any]:
    """Describe meaningful representation choices for the selected capture rows."""

    selected_spec = normalize_representation_spec(selected)
    capabilities = _representation_capabilities(dataset, feature_rows)
    options = [
        {
            "kind": "mean_pool",
            "label": "Average tokens",
            "question": "Is the signal broadly available in this token group?",
            "status": "ready",
            "runner": "generic_probe",
            "keeps": "one feature vector per example",
            "loses": "which token carried the signal",
            "reason": "The generic linear/logistic probe trainer supports this now.",
        },
        _specialized_option(
            kind="learned_layer_mix",
            label="Learn a layer mixture",
            question="Is the signal distributed across model depth?",
            runner="layer_mix_probe",
            keeps="matching token positions while learning layer weights",
            loses="a simple one-layer localization claim",
            data_ready=capabilities["aligned_token_layers"],
            ready_reason=("Multiple captured layers share a token space and can be aligned."),
            blocked_reason=("Select at least two token-bearing layers with the same token space."),
        ),
        _specialized_option(
            kind="tokenwise",
            label="Keep tokens separate",
            question="Which tokens contain the decodable information?",
            runner="tokenwise_probe",
            keeps="token identity and token position",
            loses="the simplicity of one vector per example",
            data_ready=capabilities["token_axis"],
            ready_reason="The source activation arrays retain a token axis.",
            blocked_reason="The selected source arrays do not retain a token axis.",
        ),
        _specialized_option(
            kind="object_conditioned",
            label="Ask about one object",
            question="Where and how is a specified object represented?",
            runner="object_conditioned_probe",
            keeps="tokens plus an explicit object query",
            loses="a query-free whole-scene output",
            data_ready=(capabilities["token_axis"] and capabilities["object_labels"]),
            ready_reason=("Token arrays and per-object role/pose labels are both available."),
            blocked_reason=("This needs token arrays plus per-object identity and pose labels."),
        ),
        _specialized_option(
            kind="set_decoder",
            label="Predict an object set",
            question="Can the representation produce an unordered set of scene objects?",
            runner="set_decoder_probe",
            keeps="multiple object identities and positions without fixed output slots",
            loses="the low-capacity interpretation of a linear probe",
            data_ready=(capabilities["token_axis"] and capabilities["scene_object_sets"]),
            ready_reason=(
                "Token arrays and complete scene-object identity/pose sets are available."
            ),
            blocked_reason=("This needs token arrays and complete per-scene object-set labels."),
        ),
    ]
    known = {str(option["kind"]) for option in options}
    if str(selected_spec["kind"]) not in known:
        options.append(
            {
                "kind": str(selected_spec["kind"]),
                "label": str(selected_spec["kind"]),
                "question": "Custom representation request",
                "status": "unknown",
                "runner": None,
                "keeps": None,
                "loses": None,
                "reason": "No VLA Lens capability contract exists for this choice.",
            }
        )
    return {
        "selected": selected_spec,
        "capabilities": capabilities,
        "options": options,
    }


def _specialized_option(
    *,
    kind: str,
    label: str,
    question: str,
    runner: str,
    keeps: str,
    loses: str,
    data_ready: bool,
    ready_reason: str,
    blocked_reason: str,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "label": label,
        "question": question,
        "status": "data_ready" if data_ready else "blocked",
        "runner": runner,
        "keeps": keeps,
        "loses": loses,
        "reason": (
            f"{ready_reason} A specialized runner is still required."
            if data_ready
            else blocked_reason
        ),
    }


def _representation_capabilities(
    dataset: TraceDataset,
    feature_rows: pd.DataFrame,
) -> dict[str, Any]:
    axes = feature_rows.get("axes", pd.Series("", index=feature_rows.index)).map(_parse_axes)
    token_axis = bool(axes.map(lambda value: "token" in value).any())
    layers = pd.to_numeric(
        feature_rows.get("layer", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    token_spaces = set(
        str(value)
        for value in feature_rows.get("token_space_id", pd.Series(dtype=object)).dropna()
        if str(value)
    )
    artifact_index = dataset.artifact_index
    artifact_types = set(
        str(value)
        for value in artifact_index.get("artifact_type", pd.Series(dtype=object)).dropna()
    )
    object_labels = "pi05_object_flow" in artifact_types
    selected_trace_ids = set(
        str(value) for value in feature_rows.get("trace_id", pd.Series(dtype=object)).dropna()
    )
    selected_bundles = [
        bundle
        for bundle in dataset.bundles
        if not selected_trace_ids or str(bundle.manifest.trace_id) in selected_trace_ids
    ]
    scene_positions = bool(selected_bundles) and all(
        not bundle.array_index.empty
        and "name" in bundle.array_index
        and bool((bundle.array_index["name"].astype(str) == "scene_object_pos").any())
        for bundle in selected_bundles
    )
    return {
        "token_axis": token_axis,
        "captured_layer_count": int(layers.nunique()),
        "token_space_count": int(len(token_spaces)),
        "aligned_token_layers": bool(
            token_axis and layers.nunique() >= 2 and len(token_spaces) == 1
        ),
        "object_labels": object_labels,
        "scene_positions": scene_positions,
        "scene_object_sets": bool(object_labels and scene_positions),
    }


def _parse_axes(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []
