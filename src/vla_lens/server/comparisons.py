"""Observational comparison response helpers."""

from __future__ import annotations

import re
from typing import Any, Mapping

from vla_lens.server.common import (
    _dedupe_reasons,
    _is_missing_scalar,
    _json_scalar,
    _jsonable,
    _metadata_text,
    _query_int_value,
    _query_one,
    _record_bool,
    _record_float,
    _record_text,
)
from vla_lens.traces import TraceBundle, TraceDataset


def _observational_comparisons_payload(
    dataset: TraceDataset,
    query: Mapping[str, list[str]],
) -> dict[str, Any]:
    from vla_lens.server.probes import _probe_index_artifact_payload

    trace_id = _query_one(dict(query), "trace_id")
    probe_id = (query.get("probe_id") or query.get("probe") or [""])[0]
    limit = _query_int_value(query, "limit", 6)
    source = dataset.bundle(trace_id)
    probe = _probe_index_artifact_payload(dataset, probe_id) if probe_id else None
    source_probe = _probe_trace_record(probe, trace_id)
    candidates: list[dict[str, Any]] = []
    for candidate in dataset.bundles:
        if candidate.manifest.trace_id == trace_id:
            continue
        candidate_probe = _probe_trace_record(probe, candidate.manifest.trace_id)
        score, reasons, metrics = _observational_candidate_score(
            source,
            candidate,
            source_probe,
            candidate_probe,
            has_probe=probe is not None,
        )
        candidates.append(
            {
                "trace_id": candidate.manifest.trace_id,
                "score": round(score, 3),
                "reasons": reasons,
                "episode": _comparison_episode_payload(candidate),
                "probe": _jsonable(candidate_probe) if candidate_probe else None,
                "metrics": metrics,
                "contract": {
                    "source_trace_id": trace_id,
                    "comparison_trace_id": candidate.manifest.trace_id,
                    "method": "nearest_neighbor_existing_trace",
                    "causal": False,
                    "requires_live_intervention": False,
                },
            }
        )
    candidates.sort(key=lambda item: (-float(item["score"]), str(item["trace_id"])))
    candidates = candidates[: max(1, min(limit, 24))]
    return {
        "artifact_type": "observational_counterfactual_comparison",
        "artifact_id": _observational_comparison_artifact_id(trace_id, probe_id),
        "name": "Observational comparison candidates",
        "causal": False,
        "comparison_kind": "nearest_neighbor_existing_trace",
        "source_trace_id": trace_id,
        "probe_id": probe_id or None,
        "probe_name": probe.get("name") if probe else None,
        "source": {
            "episode": _comparison_episode_payload(source),
            "probe": _jsonable(source_probe) if source_probe else None,
        },
        "candidates": candidates,
        "total_candidates": max(0, len(dataset.bundles) - 1),
        "limit": limit,
        "notes": (
            "Existing traces only. This is a comparison queue for inspection, "
            "not evidence that an activation change caused the behavior change."
        ),
    }


def _observational_comparison_artifact_id(trace_id: str, probe_id: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{trace_id}.{probe_id or 'episode'}").strip("_")
    return f"observational_comparison.{suffix[:160]}"


def _probe_trace_record(
    probe: Mapping[str, Any] | None,
    trace_id: str,
) -> Mapping[str, Any] | None:
    if not probe:
        return None
    by_trace = probe.get("by_trace")
    if not isinstance(by_trace, Mapping):
        return None
    record = by_trace.get(trace_id)
    return record if isinstance(record, Mapping) else None


def _comparison_episode_payload(bundle: TraceBundle) -> dict[str, Any]:
    metadata = dict(bundle.manifest.metadata or {})
    compact_metadata = {
        key: _json_scalar(metadata.get(key))
        for key in (
            "benchmark",
            "capture_profile",
            "seed",
            "split",
            "suite",
            "target_object",
            "task_name",
        )
        if key in metadata and not _is_missing_scalar(metadata.get(key))
    }
    return {
        "trace_id": bundle.manifest.trace_id,
        "episode_id": bundle.manifest.episode_id,
        "task_id": bundle.manifest.task_id,
        "prompt": bundle.manifest.prompt,
        "model_id": bundle.manifest.model_id,
        "env_id": bundle.manifest.env_id,
        "outcome": bundle.manifest.outcome,
        "length": bundle.manifest.length,
        "metadata": compact_metadata,
    }


def _observational_candidate_score(
    source: TraceBundle,
    candidate: TraceBundle,
    source_probe: Mapping[str, Any] | None,
    candidate_probe: Mapping[str, Any] | None,
    *,
    has_probe: bool,
) -> tuple[float, list[str], dict[str, Any]]:
    source_manifest = source.manifest
    candidate_manifest = candidate.manifest
    source_metadata = dict(source_manifest.metadata or {})
    candidate_metadata = dict(candidate_manifest.metadata or {})
    reasons: list[str] = []
    score = 0.0

    same_task = bool(
        source_manifest.task_id and source_manifest.task_id == candidate_manifest.task_id
    )
    same_prompt = bool(
        source_manifest.prompt and source_manifest.prompt == candidate_manifest.prompt
    )
    same_env = bool(source_manifest.env_id and source_manifest.env_id == candidate_manifest.env_id)
    same_model = bool(
        source_manifest.model_id and source_manifest.model_id == candidate_manifest.model_id
    )
    source_target = _metadata_text(source_metadata, "target_object")
    candidate_target = _metadata_text(candidate_metadata, "target_object")
    same_target_object = bool(source_target and source_target == candidate_target)
    different_outcome = bool(
        source_manifest.outcome
        and candidate_manifest.outcome
        and source_manifest.outcome != candidate_manifest.outcome
    )
    length_delta = int(candidate_manifest.length) - int(source_manifest.length)

    if same_task:
        score += 220
        reasons.append("same task")
    if same_prompt:
        score += 60
    if same_target_object:
        score += 70
        reasons.append("same target")
    if same_env:
        score += 20
    if same_model:
        score += 20
    if different_outcome:
        score += 180
        reasons.append("different outcome")
    else:
        reasons.append("same outcome")
    score -= min(90, abs(length_delta) * 2.0)

    source_correct = _record_bool(source_probe, "correct")
    candidate_correct = _record_bool(candidate_probe, "correct")
    source_confidence = _record_float(source_probe, "confidence")
    candidate_confidence = _record_float(candidate_probe, "confidence")
    source_split = _record_text(source_probe, "split_category")
    candidate_split = _record_text(candidate_probe, "split_category")
    confidence_delta = (
        None
        if source_confidence is None or candidate_confidence is None
        else round(candidate_confidence - source_confidence, 4)
    )

    if has_probe:
        if candidate_probe and candidate_probe.get("available"):
            score += 80
            reasons.append("probe scored")
        else:
            score -= 120
            reasons.append("probe unscored")
        if candidate_split in {"test", "validation"}:
            score += 130 if candidate_split == "test" else 95
            reasons.append(f"{candidate_split} probe record")
        elif candidate_split == "train":
            score -= 180
            reasons.append("training-set probe record")
        if source_correct is not None and candidate_correct is not None:
            if source_correct != candidate_correct:
                score += 150
                reasons.append("probe result differs")
            elif candidate_correct is False:
                score += 95
                reasons.append("probe also misses")
            else:
                score -= 35
                reasons.append("probe also correct")
        if candidate_confidence is not None:
            score += min(40, max(0.0, candidate_confidence) * 24)
        if confidence_delta is not None and abs(confidence_delta) >= 0.2:
            score += min(45, abs(confidence_delta) * 60)
            reasons.append("confidence shift")

    metrics = {
        "same_task": same_task,
        "same_prompt": same_prompt,
        "same_target_object": same_target_object,
        "different_outcome": different_outcome,
        "length_delta": length_delta,
        "source_outcome": source_manifest.outcome,
        "candidate_outcome": candidate_manifest.outcome,
        "source_probe_correct": source_correct,
        "candidate_probe_correct": candidate_correct,
        "source_split_category": source_split or None,
        "candidate_split_category": candidate_split or None,
        "source_confidence": source_confidence,
        "candidate_confidence": candidate_confidence,
        "confidence_delta": confidence_delta,
    }
    return score, _dedupe_reasons(reasons), _jsonable(metrics)
