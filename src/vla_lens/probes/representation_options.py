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

from vla_lens.selectors import ActivationQuery
from vla_lens.traces import TraceDataset

DEFAULT_REPRESENTATION_KIND = "mean_pool"
_REDUCTION_KIND = {"mean": "mean_pool", "flat": "flat_tokens", "none": "vector"}
GENERIC_PROBE_REPRESENTATION_KINDS = frozenset(_REDUCTION_KIND.values())

_ALIASES = {
    "mean": "mean_pool",
    "mean_pooled": "mean_pool",
    "pooled": "mean_pool",
    "flat": "flat_tokens",
    "flatten": "flat_tokens",
    "layer_mix": "learned_layer_mix",
    "learned_mix": "learned_layer_mix",
    "tokens": "tokenwise",
    "object_query": "object_conditioned",
    "slots": "set_decoder",
    "slot_decoder": "set_decoder",
}


def normalize_representation_spec(
    value: Any,
    *,
    reduction: str = "mean",
) -> dict[str, Any]:
    """Normalize a probe representation request without claiming runner support."""

    inferred = _REDUCTION_KIND.get(str(reduction))
    if inferred is None:
        raise ValueError(f"Unknown token reduction {reduction!r}")
    if value is None or value == "":
        return {"kind": inferred, "inferred_from_reduction": True}
    if isinstance(value, str):
        parsed: dict[str, Any] = {"kind": value}
    elif isinstance(value, Mapping):
        parsed = dict(value)
    else:
        raise TypeError("Probe representation must be a string or mapping")
    kind = str(parsed.get("kind") or inferred).strip().lower()
    parsed["kind"] = _ALIASES.get(kind, kind)
    parsed.setdefault("inferred_from_reduction", False)
    return parsed


def representation_kind_for_token_reduction(value: Any) -> str:
    reduction = str(value or "mean")
    if reduction not in _REDUCTION_KIND:
        raise ValueError(f"Unknown token reduction {reduction!r}")
    return _REDUCTION_KIND[reduction]


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


def require_generic_representation(
    spec: Mapping[str, Any],
    *,
    reduction: str,
) -> None:
    """Require the named representation to match the constructed feature matrix."""

    kind = str(spec.get("kind"))
    expected = representation_kind_for_token_reduction(reduction)
    if kind not in GENERIC_PROBE_REPRESENTATION_KINDS:
        raise ValueError(
            f"Representation {kind!r} needs a specialized probe runner; the generic "
            "trainer will not silently replace it with pooled features."
        )
    if kind != expected:
        raise ValueError(
            f"Representation {kind!r} conflicts with features.reduction={reduction!r}; "
            f"that reduction constructs {expected!r}."
        )


def probe_representation_options(
    dataset: TraceDataset,
    feature_rows: pd.DataFrame,
    selected: Any = None,
    *,
    selector: ActivationQuery | None = None,
) -> dict[str, Any]:
    """Describe meaningful representation choices for the selected capture rows."""

    selected_spec = normalize_representation_spec(selected)
    capabilities = _representation_capabilities(dataset, feature_rows, selector=selector)
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
        {
            "kind": "flat_tokens",
            "label": "Keep token positions by flattening",
            "question": "Does preserving token position reveal information pooling hides?",
            "status": "ready" if capabilities["token_axis"] else "blocked",
            "runner": "generic_probe",
            "keeps": "token positions as separate feature columns",
            "loses": "a shared readout that can move across token positions",
            "reason": (
                "The generic trainer can flatten the retained token axis."
                if capabilities["token_axis"]
                else "The selected arrays do not retain a token axis."
            ),
        },
        {
            "kind": "vector",
            "label": "Use the captured vector",
            "question": "Is the signal available in an already vector-shaped activation?",
            "status": "ready" if not capabilities["token_axis"] else "blocked",
            "runner": "generic_probe",
            "keeps": "the selected vector without token pooling",
            "loses": "token structure when the source is token-shaped",
            "reason": (
                "The selected source is already vector-shaped."
                if not capabilities["token_axis"]
                else "Token-shaped sources need mean or flat token handling."
            ),
        },
        _specialized_option(
            kind="learned_layer_mix",
            label="Learn a layer mixture",
            question="Is the signal distributed across model depth?",
            runner="token_scene_probe",
            runnable_scope="whole-scene object identity and XYZ targets",
            keeps="matching token positions while learning layer weights",
            loses="a simple one-layer localization claim",
            data_ready=capabilities["aligned_token_layers"],
            ready_reason=("Multiple captured layers share a token space and can be aligned."),
            blocked_reason=capabilities["token_topology_reason"],
        ),
        _specialized_option(
            kind="tokenwise",
            label="Keep tokens separate",
            question="Which tokens contain the decodable information?",
            runner="token_scene_probe",
            runnable_scope="whole-scene object identity and XYZ targets",
            keeps="token identity and token position",
            loses="the simplicity of one vector per example",
            data_ready=capabilities["token_topology_ready"],
            ready_reason="The source activation arrays retain a token axis.",
            blocked_reason=capabilities["token_topology_reason"],
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


def representation_options(
    feature_rows: pd.DataFrame,
    *,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    """Return lightweight options when dataset-level capability checks are unavailable."""

    axes = feature_rows.get("axes", pd.Series(dtype=object)).map(_parse_axes)
    has_token_axis = bool(axes.map(lambda value: "token" in value).any())
    layer_count = int(
        pd.to_numeric(feature_rows.get("layer", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .nunique()
    )
    return {
        "selected": dict(selected),
        "options": [
            _simple_option("mean_pool", "Average tokens", "ready", "One vector per example."),
            _simple_option(
                "flat_tokens",
                "Keep token positions by flattening",
                "ready" if has_token_axis else "blocked",
                (
                    "Token positions become separate feature columns."
                    if has_token_axis
                    else "The selected arrays do not retain a token axis."
                ),
            ),
            _simple_option(
                "learned_layer_mix",
                "Learn a layer mixture",
                "data_ready" if has_token_axis and layer_count > 1 else "blocked",
                "Needs an aligned multi-layer runner and validation-only mixture fitting.",
            ),
            _simple_option(
                "tokenwise",
                "Shared token-level readout",
                "data_ready" if has_token_axis else "blocked",
                "Needs a runner that treats tokens as examples.",
            ),
            _simple_option(
                "object_conditioned",
                "Ask about one object",
                "data_ready" if has_token_axis else "blocked",
                "Needs an explicit object query joined to token-preserving features.",
            ),
            _simple_option(
                "set_decoder",
                "Predict an unordered object set",
                "data_ready" if has_token_axis else "blocked",
                "Needs a structured target and matching-based evaluation.",
            ),
        ],
    }


def _simple_option(kind: str, label: str, status: str, reason: str) -> dict[str, str]:
    return {"kind": kind, "label": label, "status": status, "reason": reason}


def _specialized_option(
    *,
    kind: str,
    label: str,
    question: str,
    runner: str,
    runnable_scope: str | None = None,
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
        "status": (
            "ready" if data_ready and runnable_scope else "data_ready" if data_ready else "blocked"
        ),
        "runner": runner,
        "runnable_scope": runnable_scope,
        "keeps": keeps,
        "loses": loses,
        "reason": (
            (
                f"{ready_reason} The {runner} runner supports {runnable_scope}."
                if runnable_scope
                else f"{ready_reason} A specialized runner is still required."
            )
            if data_ready
            else blocked_reason
        ),
    }


def _representation_capabilities(
    dataset: TraceDataset,
    feature_rows: pd.DataFrame,
    *,
    selector: ActivationQuery | None,
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
    topology_ready, topology_reason = _token_topology_readiness(dataset, selector)
    return {
        "token_axis": token_axis,
        "captured_layer_count": int(layers.nunique()),
        "token_space_count": int(len(token_spaces)),
        "aligned_token_layers": bool(
            token_axis
            and layers.nunique() >= 2
            and len(token_spaces) == 1
            and topology_ready
        ),
        "token_topology_ready": bool(token_axis and topology_ready),
        "token_topology_reason": topology_reason,
        "object_labels": object_labels,
        "scene_positions": scene_positions,
        "scene_object_sets": bool(object_labels and scene_positions),
    }


def _token_topology_readiness(
    dataset: TraceDataset,
    selector: ActivationQuery | None,
) -> tuple[bool, str]:
    if selector is None:
        return True, "The selected source arrays do not retain a compatible token topology."
    from vla_lens.probes.token_representations import (
        _common_token_topology,
        _require_complete_source_traces,
        _selected_layers,
        _source_rows,
        _token_site_rows,
    )

    try:
        sites = _token_site_rows(dataset.select_model_sites(selector)._matching_model_sites())
        if selector.generation_step is None and sites["axes"].astype(str).str.contains(
            '"generation_step"', regex=False
        ).any():
            raise ValueError(
                "Token-preserving studies require an explicit generation_step when the "
                "captured arrays retain that axis"
            )
        layers = _selected_layers(sites, selector.layers)
        rows, source_sites = _source_rows(dataset, sites, layers, selector)
        _require_complete_source_traces(sites, rows)
        if rows.empty:
            raise ValueError("Token selector produced no complete cross-layer scene rows")
        _common_token_topology(dataset, rows, source_sites, selector.token_kind)
    except (KeyError, TypeError, ValueError) as error:
        return False, str(error)
    return True, "Selected traces share the token topology required by the token scene runner."


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
