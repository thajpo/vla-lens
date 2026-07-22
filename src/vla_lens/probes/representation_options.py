"""Explicit representation contracts for generic probe training."""

from __future__ import annotations

import json
from typing import Any, Mapping

import pandas as pd

_ALIASES = {
    "mean": "mean_pool",
    "pooled": "mean_pool",
    "flat": "flat_tokens",
    "flatten": "flat_tokens",
    "tokens": "tokenwise",
    "object_query": "object_conditioned",
    "slots": "set_decoder",
}
_REDUCTION_KIND = {"mean": "mean_pool", "flat": "flat_tokens", "none": "vector"}
_GENERIC_KINDS = frozenset(_REDUCTION_KIND.values())


def normalize_representation_spec(
    value: Any,
    *,
    reduction: str,
) -> dict[str, Any]:
    inferred = _REDUCTION_KIND.get(str(reduction))
    if inferred is None:
        raise ValueError(f"Unknown token reduction {reduction!r}")
    if value is None or value == "":
        return {"kind": inferred, "inferred_from_reduction": True}
    parsed = {"kind": value} if isinstance(value, str) else dict(value)
    kind = _ALIASES.get(str(parsed.get("kind") or inferred).lower(), str(parsed.get("kind")))
    parsed["kind"] = kind
    parsed["inferred_from_reduction"] = False
    return parsed


def require_generic_representation(spec: Mapping[str, Any], *, reduction: str) -> None:
    kind = str(spec.get("kind"))
    expected = _REDUCTION_KIND[str(reduction)]
    if kind not in _GENERIC_KINDS:
        raise ValueError(
            f"Representation {kind!r} needs a specialized probe runner; the generic "
            "trainer will not silently replace it with pooled features."
        )
    if kind != expected:
        raise ValueError(
            f"Representation {kind!r} conflicts with features.reduction={reduction!r}; "
            f"that reduction constructs {expected!r}."
        )


def representation_options(
    feature_rows: pd.DataFrame,
    *,
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    axes = feature_rows.get("axes", pd.Series(dtype=object)).map(_axes)
    has_token_axis = bool(axes.map(lambda value: "token" in value).any())
    layer_count = int(
        pd.to_numeric(feature_rows.get("layer", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .nunique()
    )
    return {
        "selected": dict(selected),
        "options": [
            _option("mean_pool", "Average tokens", "ready", "One vector per example."),
            _option(
                "flat_tokens",
                "Keep token positions by flattening",
                "ready" if has_token_axis else "blocked",
                (
                    "Token positions become separate feature columns; dimensionality can be large."
                    if has_token_axis
                    else "The selected tensors do not retain a token axis."
                ),
            ),
            _option(
                "learned_layer_mix",
                "Learn a layer mixture",
                "data_ready" if has_token_axis and layer_count > 1 else "blocked",
                "Needs an aligned multi-layer runner and validation-only mixture fitting.",
            ),
            _option(
                "tokenwise",
                "Shared token-level readout",
                "data_ready" if has_token_axis else "blocked",
                "Needs a runner that treats tokens as examples instead of anonymous columns.",
            ),
            _option(
                "object_conditioned",
                "Ask about one object",
                "data_ready" if has_token_axis else "blocked",
                "Needs an explicit object query joined to token-preserving features.",
            ),
            _option(
                "set_decoder",
                "Predict an unordered object set",
                "data_ready" if has_token_axis else "blocked",
                "Needs a structured target and matching-based evaluation.",
            ),
        ],
    }


def _option(kind: str, label: str, status: str, reason: str) -> dict[str, str]:
    return {"kind": kind, "label": label, "status": status, "reason": reason}


def _axes(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []
