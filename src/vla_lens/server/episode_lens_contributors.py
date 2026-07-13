"""Probe contributor ranking helpers for EpisodeLensView."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from vla_lens.artifacts import LensArtifact
from vla_lens.probes.score_cache import _artifact_selector, _filter_best_sweep_rows
from vla_lens.server.episode_lens_common import _first_present, _optional_int
from vla_lens.traces import TraceDataset


def _probe_site_readout(
    dataset: TraceDataset,
    artifact: LensArtifact,
    best_state: Mapping[str, Any],
    trace_id: str,
    resolved_selection: Mapping[str, Any],
    *,
    ranking_mode: str,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, str]]]:
    unavailable = _empty_probe_site_readout(resolved_selection, ranking_mode, top_k)
    callouts: list[dict[str, str]] = []
    try:
        X, rows = _selected_probe_feature_rows(dataset, artifact, best_state, trace_id)
    except Exception as exc:
        reason = f"Selected activation vector is unavailable: {exc}"
        unavailable["unavailable_reason"] = reason
        callouts.append({"severity": "warning", "text": reason})
        return unavailable, dict(resolved_selection), callouts
    if rows.empty or X.size == 0:
        reason = "Selected activation vector is unavailable for this episode."
        unavailable["unavailable_reason"] = reason
        callouts.append({"severity": "warning", "text": reason})
        return unavailable, dict(resolved_selection), callouts
    position, row = _choose_feature_row(rows, resolved_selection)
    if position is None or row is None:
        reason = "No activation row matches this probe selection."
        unavailable["unavailable_reason"] = reason
        callouts.append({"severity": "warning", "text": reason})
        return unavailable, dict(resolved_selection), callouts
    vector = np.asarray(X[int(position)], dtype=np.float32).reshape(-1)
    actual_site = str(
        _first_present(
            row.get("activation"), row.get("model_site_id"), resolved_selection.get("model_site_id")
        )
        or ""
    )
    actual_layer = _optional_int(row.get("layer"))
    actual_policy_call = _optional_int(row.get("policy_call_index"))
    actual_timestep = _optional_int(row.get("timestep"))
    actual_token_space = _first_present(
        row.get("token_space_id"),
        row.get("token_space"),
        resolved_selection.get("token_space"),
        resolved_selection.get("token_space_id"),
    )
    actual_selection = {
        **dict(resolved_selection),
        "trace_id": trace_id,
        "timestep": actual_timestep,
        "policy_call_index": actual_policy_call,
        "model_site_id": actual_site,
        "token_space": actual_token_space,
        "token_space_id": actual_token_space,
        "layer": actual_layer,
        "mode": "features",
    }
    raw_rows = _raw_activation_rows(vector, top_k)
    readout = {
        **_empty_probe_site_readout(actual_selection, ranking_mode, top_k),
        "available": True,
        "unavailable_reason": None,
        "model_site_id": actual_site,
        "token_space_id": actual_token_space,
        "layer": actual_layer,
        "policy_call_index": actual_policy_call,
        "timestep": actual_timestep,
        "raw_activation_ranking_available": True,
        "raw_activation_ranking": raw_rows,
        "total_features": int(vector.shape[0]),
    }
    contributor_payload, reason = _linear_probe_contributors(
        dataset,
        artifact,
        best_state,
        vector,
        actual_site,
        actual_layer,
        actual_policy_call,
        top_k,
    )
    if reason:
        readout["probe_contribution_ranking_available"] = False
        readout["feature_contributors_available"] = False
        readout["feature_contributors_unavailable_reason"] = reason
        readout["unavailable_reason"] = None
        callouts.append({"severity": "warning", "text": reason})
    else:
        readout.update(contributor_payload)
        first = (contributor_payload.get("feature_contributors") or [None])[0]
        if isinstance(first, Mapping):
            readout["default_feature"] = _optional_int(first.get("feature"))
            actual_selection["feature"] = readout["default_feature"]
    return readout, actual_selection, callouts


def _empty_probe_site_readout(
    selection: Mapping[str, Any],
    ranking_mode: str,
    top_k: int,
) -> dict[str, Any]:
    return {
        "available": False,
        "unavailable_reason": None,
        "model_site_id": selection.get("model_site_id"),
        "token_space_id": _first_present(
            selection.get("token_space_id"), selection.get("token_space")
        ),
        "layer": selection.get("layer"),
        "policy_call_index": selection.get("policy_call_index"),
        "timestep": selection.get("timestep"),
        "ranking_basis": "linear_logit_contribution"
        if ranking_mode == "probe_contribution"
        else "raw_activation",
        "ranking_order": "absolute_value",
        "top_k": top_k,
        "total_features": None,
        "units": "logit" if ranking_mode == "probe_contribution" else "activation",
        "normalization": "training_standardizer",
        "site_readout_available": False,
        "feature_contributors_available": False,
        "probe_contribution_ranking_available": False,
        "raw_activation_ranking_available": False,
        "temporal_readout_available": False,
        "intervention_seed_available": False,
        "feature_contributors": [],
        "raw_activation_ranking": [],
        "logit_reconstruction": None,
    }


def _selected_probe_feature_rows(
    dataset: TraceDataset,
    artifact: LensArtifact,
    best_state: Mapping[str, Any],
    trace_id: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    selector = _artifact_selector(artifact)
    selector = replace(selector, episodes={**dict(selector.episodes), "trace_id": trace_id})
    X, rows = dataset.select_model_sites(selector).to_matrix(cache=False)
    if not rows.empty and best_state:
        X, rows = _filter_best_sweep_rows(X, rows, artifact, best_state)
    return np.asarray(X, dtype=np.float32), rows.reset_index(drop=True)


def _choose_feature_row(
    rows: pd.DataFrame,
    selection: Mapping[str, Any],
) -> tuple[int | None, Mapping[str, Any] | None]:
    if rows.empty:
        return None, None
    candidates = rows.copy()
    candidates["_position"] = np.arange(len(candidates))
    requested_site = str(selection.get("model_site_id") or "")
    if requested_site:
        site_mask = candidates.get("activation", pd.Series("", index=candidates.index)).astype(
            str
        ).eq(requested_site) | candidates.get(
            "model_site_id", pd.Series("", index=candidates.index)
        ).astype(str).eq(requested_site)
        site_rows = candidates.loc[site_mask]
        if not site_rows.empty:
            candidates = site_rows
    for column, key in (("policy_call_index", "policy_call_index"), ("timestep", "timestep")):
        value = selection.get(key)
        if value is None or column not in candidates:
            continue
        numeric = pd.to_numeric(candidates[column], errors="coerce")
        matching = candidates.loc[numeric == int(value)]
        if not matching.empty:
            candidates = matching
    row = candidates.iloc[0].to_dict()
    return int(row.pop("_position")), row


def _linear_probe_contributors(
    dataset: TraceDataset,
    artifact: LensArtifact,
    best_state: Mapping[str, Any],
    vector: np.ndarray,
    model_site_id: str,
    layer: int | None,
    policy_call_index: int | None,
    top_k: int,
) -> tuple[dict[str, Any], str | None]:
    if str(best_state.get("model") or "linear") != "linear":
        return {}, "Probe contributor ranking is unavailable for nonlinear probe models."
    required = ["weights", "bias", "feature_mean", "feature_scale"]
    missing = [name for name in required if name not in artifact.arrays]
    if missing:
        return (
            {},
            f"Probe contributor ranking is unavailable; missing arrays: {', '.join(missing)}.",
        )
    try:
        weights = np.asarray(dataset.load_artifact_array(artifact, "weights"), dtype=np.float32)
        bias = np.asarray(dataset.load_artifact_array(artifact, "bias"), dtype=np.float32).reshape(
            -1
        )
        mean = np.asarray(
            dataset.load_artifact_array(artifact, "feature_mean"), dtype=np.float32
        ).reshape(-1)
        scale = np.asarray(
            dataset.load_artifact_array(artifact, "feature_scale"), dtype=np.float32
        ).reshape(-1)
    except Exception as exc:
        return {}, f"Probe contributor ranking is unavailable; could not load probe arrays: {exc}."
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    probe_type = str(best_state.get("probe_type") or "classification")
    classes = [str(item) for item in best_state.get("classes") or []]
    if probe_type == "classification" and weights.shape[0] != 1:
        return (
            {},
            (
                "Multiclass probe contributor ranking is unavailable until class "
                "orientation is explicit."
            ),
        )
    if (
        weights.shape[1] != vector.shape[0]
        or mean.shape[0] != vector.shape[0]
        or scale.shape[0] != vector.shape[0]
    ):
        return {}, (
            "Probe contributor ranking is unavailable; selected activation shape does not "
            "match the saved probe feature order."
        )
    if not (
        np.isfinite(vector).all()
        and np.isfinite(weights).all()
        and np.isfinite(mean).all()
        and np.isfinite(scale).all()
    ):
        return (
            {},
            (
                "Probe contributor ranking is unavailable; probe arrays or activations "
                "contain non-finite values."
            ),
        )
    if np.any(scale == 0):
        return (
            {},
            "Probe contributor ranking is unavailable; saved feature scale contains zero values.",
        )
    feature_order = _optional_feature_order(dataset, artifact)
    if feature_order is not None and not np.array_equal(
        feature_order, np.arange(vector.shape[0], dtype=feature_order.dtype)
    ):
        return (
            {},
            (
                "Probe contributor ranking is unavailable; saved feature order does not "
                "match activation order."
            ),
        )
    normalized = (vector.astype(np.float32, copy=False) - mean) / scale
    weight = weights.reshape(weights.shape[0], -1)[0]
    contribution = normalized * weight
    order = np.argsort(-np.abs(contribution))[:top_k]
    raw_order = np.argsort(-np.abs(vector))[:top_k]
    positive_class = classes[1] if len(classes) == 2 else None
    negative_class = classes[0] if len(classes) == 2 else None
    rows = []
    for rank, feature_index in enumerate(order, start=1):
        value = float(contribution[int(feature_index)])
        if probe_type == "regression":
            supports_class = None
            opposes_class = None
            sign_label = "raises prediction" if value >= 0 else "lowers prediction"
        elif value >= 0:
            supports_class = positive_class
            opposes_class = negative_class
            sign_label = (
                f"supports {positive_class}" if positive_class is not None else "positive logit"
            )
        else:
            supports_class = negative_class
            opposes_class = positive_class
            sign_label = (
                f"supports {negative_class}" if negative_class is not None else "negative logit"
            )
        rows.append(
            {
                "feature": int(feature_index),
                "weight": float(weight[int(feature_index)]),
                "activation": float(vector[int(feature_index)]),
                "normalized_activation": float(normalized[int(feature_index)]),
                "contribution": value,
                "abs_contribution": abs(value),
                "rank": rank,
                "abs_rank": rank,
                "raw_activation_abs_rank": _rank_in_order(int(feature_index), raw_order),
                "direction": "positive" if value > 0 else "negative" if value < 0 else "neutral",
                "sign_label": sign_label,
                "supports_class": supports_class,
                "opposes_class": opposes_class,
                "label": sign_label,
                "overlay_available": True,
                "feature_ref": {"model_site_id": model_site_id, "feature": int(feature_index)},
            }
        )
    total_sum = float(np.sum(contribution))
    reconstructed = float(total_sum + float(bias[0] if bias.size else 0.0))
    return {
        "available": True,
        "site_readout_available": True,
        "feature_contributors_available": True,
        "probe_contribution_ranking_available": True,
        "ranking_basis": "linear_logit_contribution",
        "ranking_order": "absolute_value",
        "units": "logit",
        "normalization": "training_standardizer",
        "feature_ordering": "selector_flattened_channel_order",
        "model_site_id": model_site_id,
        "layer": layer,
        "policy_call_index": policy_call_index,
        "feature_contributors": rows,
        "logit_reconstruction": {
            "bias": float(bias[0] if bias.size else 0.0),
            "shown_contribution_sum": float(np.sum(contribution[order])),
            "total_contribution_sum": total_sum,
            "reconstructed_logit": reconstructed,
            "model_logit": None,
            "residual": None,
        },
    }, None


def _optional_feature_order(dataset: TraceDataset, artifact: LensArtifact) -> np.ndarray | None:
    if "feature_order" not in artifact.arrays:
        return None
    try:
        return np.asarray(dataset.load_artifact_array(artifact, "feature_order")).reshape(-1)
    except Exception:
        return None


def _raw_activation_rows(vector: np.ndarray, top_k: int) -> list[dict[str, Any]]:
    order = np.argsort(-np.abs(vector))[:top_k]
    return [
        {
            "feature": int(index),
            "activation": float(vector[int(index)]),
            "abs_activation": float(abs(vector[int(index)])),
            "rank": rank,
            "feature_ref": {"feature": int(index)},
        }
        for rank, index in enumerate(order, start=1)
    ]


def _rank_in_order(index: int, order: np.ndarray) -> int | None:
    matches = np.where(order == index)[0]
    return int(matches[0] + 1) if matches.size else None
