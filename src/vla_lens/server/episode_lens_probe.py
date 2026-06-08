"""Probe-suite EpisodeLensView payload assembly."""

from __future__ import annotations

import fnmatch
from typing import Any, Mapping

from vla_lens.artifacts import LensArtifact
from vla_lens.probes.score_cache import _artifact_selector
from vla_lens.server.common import _jsonable
from vla_lens.server.episode_lens_common import (
    _episode_lens_episode_payload,
    _first_present,
    _layer_for_site,
    _layer_from_feature_name,
    _lens_payload,
    _optional_float,
    _optional_int,
    _ranking_mode,
    _record_correct,
    _short_site_label,
    _site_for_best_state,
    _site_from_feature_name,
    _site_label,
    _site_name_for_id,
    _top_k,
)
from vla_lens.server.episode_lens_contributors import _probe_site_readout
from vla_lens.server.episode_lens_probe_metadata import (
    _probe_best_model_state,
    _probe_identity_payload,
    _probe_spec_payload,
)
from vla_lens.server.indexed_probes import indexed_episode_probes_payload
from vla_lens.traces import TraceDataset


def _unavailable_episode_lens_view(
    dataset: TraceDataset,
    artifact: Mapping[str, Any],
    family: Mapping[str, Any],
    trace_id: str,
    current_selection: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": "episode_lens_view.v1",
        "family": str(artifact.get("artifact_type") or family.get("artifact_type") or "unknown"),
        "available": False,
        "unavailable_reason": reason,
        "lens": _lens_payload(artifact, family),
        "episode": _episode_lens_episode_payload(dataset, trace_id),
        "current_selection": dict(current_selection),
        "resolved_selection": dict(current_selection),
        "recommended_selection": None,
        "readout": None,
        "annotations": {
            "pipeline": [],
            "timeline": [],
            "overlays": [],
            "callouts": [
                {"severity": "warning", "text": reason},
            ],
        },
        "inspector": {
            "default_mode": "features",
            "default_ranking_id": None,
            "pipeline_marks": [],
            "timeline_marks": [],
            "overlay_marks": [],
            "rankings": [],
            "callouts": [
                {"severity": "warning", "text": reason, "applies_to": "source_scope"},
            ],
        },
        "view": {},
        "actions": [
            {
                "kind": "open_artifact_debug",
                "label": "Open artifact debug",
                "enabled": True,
            }
        ],
    }

def _probe_suite_episode_lens_view(
    dataset: TraceDataset,
    artifact: Mapping[str, Any],
    artifact_object: LensArtifact,
    family: Mapping[str, Any],
    trace_id: str,
    current_selection: Mapping[str, Any],
    query: Mapping[str, list[str]],
) -> dict[str, Any]:
    # v0 invariant: one probe_suite artifact represents one active probe target/head.
    probes = indexed_episode_probes_payload(dataset.root, {"trace_id": [trace_id]})
    probe = next(
        (
            item
            for item in probes.get("probes", [])
            if str(item.get("artifact_id") or "") == str(artifact.get("artifact_id") or "")
        ),
        None,
    )
    if probe is None:
        reason = "This probe artifact has no indexed readout for the selected trace."
        return _unavailable_episode_lens_view(
            dataset,
            artifact,
            family,
            trace_id,
            current_selection,
            reason,
        )

    best_state = _probe_best_model_state(artifact_object)
    summary = (
        probe.get("episode_summary") if isinstance(probe.get("episode_summary"), Mapping) else {}
    )
    best_row = summary.get("best_row") if isinstance(summary.get("best_row"), Mapping) else {}
    readout = _probe_lens_readout(probe, summary, best_row)
    source_scope = _probe_source_scope(dataset, artifact_object, trace_id, best_row, best_state)
    recommended_selection = _probe_recommended_selection(
        trace_id,
        source_scope,
        best_row,
        best_state,
    )
    resolved_selection, selection_callouts = _resolve_probe_lens_selection(
        current_selection,
        recommended_selection,
        source_scope,
    )
    ranking_mode = _ranking_mode(query)
    top_k = _top_k(query)
    site_readout, resolved_selection, contributor_callouts = _probe_site_readout(
        dataset,
        artifact_object,
        best_state,
        trace_id,
        resolved_selection,
        ranking_mode=ranking_mode,
        top_k=top_k,
    )
    source_scope = _mark_source_scope_selection(
        source_scope, resolved_selection, recommended_selection
    )
    if (
        isinstance(site_readout.get("default_feature"), int)
        and resolved_selection.get("feature") is None
    ):
        resolved_selection = {**resolved_selection, "feature": site_readout["default_feature"]}
    temporal_readout = _probe_temporal_readout(probe)
    annotations = _probe_lens_annotations(
        source_scope,
        readout,
        resolved_selection,
        selection_callouts + contributor_callouts,
        temporal_readout,
    )
    action_feature = _first_present(
        resolved_selection.get("feature"),
        site_readout.get("default_feature"),
    )
    actions = _probe_lens_actions(
        artifact,
        trace_id,
        recommended_selection,
        resolved_selection,
        action_feature,
        site_readout,
    )
    available = bool(probe.get("available"))
    reason = "" if available else "This probe artifact is not scored for the selected trace."
    return _jsonable(
        {
            "schema_version": "episode_lens_view.v1",
            "family": "probe_suite",
            "available": available,
            "unavailable_reason": None if available else reason,
            "lens": _lens_payload(
                artifact, family, spec=_probe_spec_payload(artifact_object, best_state)
            ),
            "episode": _episode_lens_episode_payload(dataset, trace_id),
            "current_selection": dict(current_selection),
            "resolved_selection": resolved_selection,
            "recommended_selection": recommended_selection,
            "readout": readout,
            "annotations": annotations,
            "inspector": _probe_lens_inspector(
                annotations,
                site_readout,
                recommended_selection,
                ranking_mode,
            ),
            "view": {
                "probe": _probe_identity_payload(artifact, artifact_object, best_state),
                "source_scope": source_scope,
                "episode_readout": readout,
                "site_readout": site_readout,
                "temporal_readout": temporal_readout,
            },
            "actions": actions,
        }
    )

def _probe_lens_readout(
    probe: Mapping[str, Any],
    summary: Mapping[str, Any],
    best_row: Mapping[str, Any],
) -> dict[str, Any]:
    predicted = summary.get("predicted")
    actual = summary.get("actual")
    confidence = _optional_float(summary.get("confidence"))
    correct = _record_correct(summary.get("correct"))
    available = bool(probe.get("available"))
    return {
        "predicted": predicted,
        "actual": actual,
        "confidence": confidence,
        "score": _optional_float(summary.get("correct_rate")),
        "correct": correct,
        "split": _first_present(best_row.get("split"), best_row.get("eval_split"), "unknown"),
        "verdict": _probe_verdict(available, predicted, actual, correct, confidence),
        "policy_call_index": _optional_int(best_row.get("policy_call_index")),
        "timestep": _optional_int(best_row.get("timestep")),
        "model_site_id": _first_present(
            best_row.get("activation"), best_row.get("model_site_id"), best_row.get("model")
        ),
    }

def _probe_verdict(
    available: bool,
    predicted: Any,
    actual: Any,
    correct: bool | None,
    confidence: float | None,
) -> str:
    if not available:
        return "unscored"
    if correct is True:
        return "correct"
    if correct is False:
        if confidence is not None and confidence >= 0.9:
            return "high_conf_wrong"
        return "wrong"
    if predicted not in {None, ""} and actual not in {None, ""}:
        return "ambiguous"
    return "unknown"

def _probe_source_scope(
    dataset: TraceDataset,
    artifact: LensArtifact,
    trace_id: str,
    best_row: Mapping[str, Any],
    best_state: Mapping[str, Any],
) -> dict[str, Any]:
    records = _probe_source_site_records(dataset, artifact, trace_id)
    best_state_site = _site_for_best_state(records, best_state)
    default_site = _first_present(
        best_state_site,
        best_row.get("activation"),
        best_row.get("model_site_id"),
        best_row.get("model"),
    )
    if default_site:
        default_site = _site_name_for_id(records, str(default_site)) or str(default_site)
    if not default_site and records:
        default_site = str(records[0].get("name") or records[0].get("model_site_id") or "")
    default_layer = _first_present(
        _optional_int(best_row.get("layer")),
        _layer_for_site(records, default_site),
        _layer_from_feature_name(str(best_state.get("feature") or "")),
    )
    sites = []
    for record in records:
        site_name = str(record.get("name") or record.get("model_site_id") or "")
        layer = _optional_int(record.get("layer"))
        is_default = bool(default_site and site_name == default_site)
        sites.append(
            {
                "model_site_id": site_name,
                "site_id": _first_present(record.get("site_id"), site_name),
                "layer": layer,
                "tensor_type": _first_present(record.get("tensor_type"), "unknown"),
                "token_kind": record.get("token_kind"),
                "token_space_id": record.get("token_space_id"),
                "trained": True,
                "available": True,
                "selected": False,
                "default": is_default,
                "best": is_default,
                "policy_calls": [],
                "label": _site_label(site_name, layer, record),
                "short_label": f"L{layer}" if layer is not None else _short_site_label(site_name),
            }
        )
    return {
        "default_policy_call_index": _optional_int(best_row.get("policy_call_index")),
        "default_model_site_id": default_site,
        "default_layer": _optional_int(default_layer),
        "default_feature": None,
        "sites": sites,
    }

def _probe_source_site_records(
    dataset: TraceDataset,
    artifact: LensArtifact,
    trace_id: str,
) -> list[dict[str, Any]]:
    selector = _artifact_selector(artifact)
    index = dataset.model_site_index
    if index.empty:
        return []
    rows = index.loc[index["trace_id"].astype(str) == str(trace_id)]
    if selector.name is not None and "name" in rows:
        rows = rows.loc[
            rows["name"].astype(str).map(lambda value: fnmatch.fnmatchcase(value, selector.name))
        ]
    if selector.module is not None and "module" in rows:
        rows = rows.loc[
            rows["module"]
            .astype(str)
            .map(lambda value: fnmatch.fnmatchcase(value, selector.module))
        ]
    if selector.layers is not None and "layer" in rows:
        layers = {int(layer) for layer in selector.layers}
        rows = rows.loc[rows["layer"].map(_optional_int).isin(layers)]
    if selector.tensor_type is not None and "tensor_type" in rows:
        rows = rows.loc[rows["tensor_type"].astype(str) == selector.tensor_type]
    if selector.token_kind is not None and "token_kind" in rows:
        token_column = rows["token_kind"]
        rows = rows.loc[token_column.isna() | (token_column.astype(str) == selector.token_kind)]
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in rows.to_dict("records"):
        site_name = str(record.get("name") or record.get("site_id") or "")
        if not site_name or site_name in seen:
            continue
        seen.add(site_name)
        records.append(record)
    return records

def _probe_recommended_selection(
    trace_id: str,
    source_scope: Mapping[str, Any],
    best_row: Mapping[str, Any],
    best_state: Mapping[str, Any],
) -> dict[str, Any] | None:
    sites = [
        site for site in source_scope.get("sites", [])
        if isinstance(site, Mapping)
    ]
    best_state_site = _site_for_best_state(sites, best_state)
    model_site_id = _first_present(
        best_state_site,
        source_scope.get("default_model_site_id"),
        best_row.get("activation"),
        best_row.get("model_site_id"),
        best_row.get("model"),
        _site_from_feature_name(str(best_state.get("feature") or "")),
    )
    if not model_site_id:
        return None
    return {
        "trace_id": trace_id,
        "timestep": _optional_int(best_row.get("timestep")),
        "policy_call_index": _optional_int(best_row.get("policy_call_index")),
        "model_site_id": model_site_id,
        "layer": _first_present(
            source_scope.get("default_layer"),
            _optional_int(best_row.get("layer")),
            _layer_from_feature_name(str(best_state.get("feature") or "")),
        ),
        "feature": None,
        "mode": "features",
    }

def _resolve_probe_lens_selection(
    current_selection: Mapping[str, Any],
    recommended_selection: Mapping[str, Any] | None,
    source_scope: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    callouts: list[dict[str, str]] = []
    available_sites = {
        str(site.get("model_site_id") or "")
        for site in source_scope.get("sites", [])
        if isinstance(site, Mapping) and site.get("available")
    }
    requested_site = str(current_selection.get("model_site_id") or "")
    resolved = {
        **dict(recommended_selection or {}),
        **{key: value for key, value in current_selection.items() if value not in {None, ""}},
    }
    if requested_site and requested_site not in available_sites:
        fallback = (recommended_selection or {}).get("model_site_id")
        if fallback:
            resolved["model_site_id"] = fallback
            callouts.append(
                {
                    "severity": "warning",
                    "text": (
                        "Requested site is outside this probe input; using the probe "
                        "default site."
                    ),
                }
            )
    if not resolved.get("model_site_id") and recommended_selection:
        resolved["model_site_id"] = recommended_selection.get("model_site_id")
    resolved.setdefault("trace_id", current_selection.get("trace_id"))
    resolved.setdefault("mode", "features")
    return resolved, callouts

def _mark_source_scope_selection(
    source_scope: Mapping[str, Any],
    resolved_selection: Mapping[str, Any],
    recommended_selection: Mapping[str, Any] | None,
) -> dict[str, Any]:
    selected_site = str(resolved_selection.get("model_site_id") or "")
    default_site = str((recommended_selection or {}).get("model_site_id") or "")
    sites = []
    for site in source_scope.get("sites", []):
        if not isinstance(site, Mapping):
            continue
        site_name = str(site.get("model_site_id") or "")
        sites.append(
            {
                **dict(site),
                "selected": bool(site_name and site_name == selected_site),
                "default": bool(site_name and site_name == default_site),
                "best": bool(site_name and site_name == default_site),
            }
        )
    return {**dict(source_scope), "sites": sites}

def _probe_temporal_readout(probe: Mapping[str, Any]) -> dict[str, Any]:
    rows = probe.get("rows") if isinstance(probe.get("rows"), list) else []
    out = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        confidence = _optional_float(row.get("confidence"))
        correct = _record_correct(row.get("correct"))
        out.append(
            {
                "timestep": _optional_int(row.get("timestep")),
                "policy_call_index": _optional_int(row.get("policy_call_index")),
                "model_site_id": _first_present(
                    row.get("activation"), row.get("model_site_id"), row.get("model")
                ),
                "predicted": _first_present(row.get("predicted"), row.get("prediction_value")),
                "confidence": confidence,
                "logit": _optional_float(row.get("logit")),
                "probability": _optional_float(row.get("probability")),
                "correct": correct,
                "verdict": _probe_verdict(
                    True, row.get("predicted"), row.get("actual"), correct, confidence
                ),
            }
        )
    return {
        "available": bool(out),
        "temporal_readout_available": bool(out),
        "unavailable_reason": None if out else "No temporal probe rows are available.",
        "rows": out,
    }

def _probe_lens_annotations(
    source_scope: Mapping[str, Any],
    readout: Mapping[str, Any],
    resolved_selection: Mapping[str, Any],
    callouts: list[dict[str, str]],
    temporal_readout: Mapping[str, Any],
) -> dict[str, Any]:
    verdict = str(readout.get("verdict") or "unknown")
    pipeline = []
    for site in source_scope.get("sites", []):
        if not isinstance(site, Mapping):
            continue
        pipeline.append(
            {
                "model_site_id": site.get("model_site_id"),
                "layer": site.get("layer"),
                "role": "source" if site.get("trained") else "unavailable",
                "trained": bool(site.get("trained")),
                "available": bool(site.get("available")),
                "selected": bool(site.get("selected")),
                "default": bool(site.get("default")),
                "severity": verdict if site.get("selected") else "neutral",
                "label": site.get("label"),
                "short_label": site.get("short_label"),
            }
        )
    timeline = _probe_timeline_annotations(temporal_readout, resolved_selection, readout, verdict)
    return {
        "pipeline": pipeline,
        "timeline": timeline,
        "overlays": [],
        "callouts": callouts,
    }

def _probe_timeline_annotations(
    temporal_readout: Mapping[str, Any],
    resolved_selection: Mapping[str, Any],
    readout: Mapping[str, Any],
    fallback_verdict: str,
) -> list[dict[str, Any]]:
    rows = temporal_readout.get("rows") if isinstance(temporal_readout.get("rows"), list) else []
    resolved_call = _optional_int(resolved_selection.get("policy_call_index"))
    resolved_timestep = _optional_int(resolved_selection.get("timestep"))
    timeline: list[dict[str, Any]] = []
    seen: set[tuple[int | None, int | None]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        call_index = _optional_int(row.get("policy_call_index"))
        timestep = _optional_int(row.get("timestep"))
        if call_index is None and timestep is None:
            continue
        key = (call_index, timestep)
        if key in seen:
            continue
        seen.add(key)
        row_verdict = str(row.get("verdict") or fallback_verdict)
        selected = (
            call_index is not None
            and resolved_call is not None
            and int(call_index) == int(resolved_call)
        ) or (
            timestep is not None
            and resolved_timestep is not None
            and int(timestep) == int(resolved_timestep)
        )
        timeline.append(
            {
                "policy_call_index": call_index,
                "timestep": timestep,
                "kind": "prediction",
                "value": row.get("confidence"),
                "verdict": row_verdict,
                "label": "Selected probe call" if selected else "Probe call",
                "selected": selected,
            }
        )
    if timeline:
        return sorted(
            timeline,
            key=lambda item: (
                item.get("timestep") is None,
                item.get("timestep") if item.get("timestep") is not None else 10**9,
                item.get("policy_call_index")
                if item.get("policy_call_index") is not None
                else 10**9,
            ),
        )
    if resolved_selection.get("policy_call_index") is not None:
        return [
            {
                "policy_call_index": resolved_selection.get("policy_call_index"),
                "timestep": resolved_selection.get("timestep"),
                "kind": "prediction",
                "value": readout.get("confidence"),
                "verdict": fallback_verdict,
                "label": "Selected probe call",
                "selected": True,
            }
        ]
    return []

def _probe_lens_inspector(
    annotations: Mapping[str, Any],
    site_readout: Mapping[str, Any],
    recommended_selection: Mapping[str, Any] | None,
    ranking_mode: str,
) -> dict[str, Any]:
    rankings = [
        {
            "id": "probe_contributors",
            "label": "Probe contributors",
            "kind": "feature_ranking",
            "available": bool(site_readout.get("probe_contribution_ranking_available")),
            "unavailable_reason": site_readout.get("feature_contributors_unavailable_reason"),
            "basis": "linear_logit_contribution",
            "normalization": site_readout.get("normalization"),
            "units": site_readout.get("units") if ranking_mode == "probe_contribution" else "logit",
            "default": ranking_mode == "probe_contribution",
            "model_site_id": site_readout.get("model_site_id"),
            "policy_call_index": site_readout.get("policy_call_index"),
            "row_count": len(site_readout.get("feature_contributors") or []),
        },
        {
            "id": "raw_activations",
            "label": "Raw activations",
            "kind": "feature_ranking",
            "available": bool(site_readout.get("raw_activation_ranking_available")),
            "unavailable_reason": None
            if site_readout.get("raw_activation_ranking_available")
            else site_readout.get("unavailable_reason"),
            "basis": "raw_activation",
            "normalization": None,
            "units": "activation",
            "default": ranking_mode == "raw_activation",
            "model_site_id": site_readout.get("model_site_id"),
            "policy_call_index": site_readout.get("policy_call_index"),
            "row_count": len(site_readout.get("raw_activation_ranking") or []),
        },
    ]
    return {
        "default_mode": "features",
        "default_ranking_id": (
            "probe_contributors" if ranking_mode == "probe_contribution" else "raw_activations"
        ),
        "recommended_selection": recommended_selection,
        "pipeline_marks": list(annotations.get("pipeline") or []),
        "timeline_marks": list(annotations.get("timeline") or []),
        "overlay_marks": list(annotations.get("overlays") or []),
        "rankings": rankings,
        "callouts": [
            {
                **dict(callout),
                "applies_to": callout.get("applies_to") or "selection",
            }
            for callout in annotations.get("callouts") or []
            if isinstance(callout, Mapping)
        ],
    }

def _probe_lens_actions(
    artifact: Mapping[str, Any],
    trace_id: str,
    recommended_selection: Mapping[str, Any] | None,
    resolved_selection: Mapping[str, Any],
    feature: Any,
    site_readout: Mapping[str, Any],
) -> list[dict[str, Any]]:
    policy_call = _optional_int(resolved_selection.get("policy_call_index"))
    model_site = str(resolved_selection.get("model_site_id") or "")
    seed_enabled = policy_call is not None and bool(model_site)
    seed = None
    if seed_enabled:
        seed = {
            "trace_id": trace_id,
            "artifact_id": str(artifact.get("artifact_id") or ""),
            "family": "probe_suite",
            "probe_id": str(artifact.get("artifact_id") or ""),
            "policy_call_index": policy_call,
            "timestep": _optional_int(resolved_selection.get("timestep")),
            "model_site_id": model_site,
            "layer": _optional_int(resolved_selection.get("layer")),
            "feature": _optional_int(feature),
            "suggested_operator": "ablate",
        }
    return [
        {
            "kind": "jump_to_lens_default",
            "label": "Use probe site",
            "enabled": recommended_selection is not None,
            "selection": recommended_selection,
        },
        {
            "kind": "send_to_intervention",
            "label": "Seed intervention",
            "enabled": seed_enabled,
            "seed": seed,
            "unavailable_reason": None
            if seed_enabled
            else "Policy call or model site is unavailable.",
        },
        {
            "kind": "compare_raw_activations",
            "label": "Compare raw activations",
            "enabled": bool(site_readout.get("raw_activation_ranking_available")),
        },
        {
            "kind": "open_artifact_debug",
            "label": "Open artifact debug",
            "enabled": True,
        },
    ]
