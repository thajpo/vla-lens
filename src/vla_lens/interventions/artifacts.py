"""LensArtifact indexing helpers for intervention evidence records."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from vla_lens.artifacts import LensArtifact
from vla_lens.interventions.results import InterventionRun, InterventionSweep
from vla_lens.interventions.serialization import jsonable

ARTIFACT_TYPE = "intervention_run"
SWEEP_ARTIFACT_TYPE = "intervention_sweep"


def intervention_run_to_lens_artifact(run: InterventionRun) -> LensArtifact:
    """Create an artifact-browser shell for a canonical intervention run."""
    request = jsonable(run.request)
    operator = _mapping(request.get("operator"))
    schedule = _mapping(request.get("schedule"))
    outcome = _mapping(request.get("outcome"))
    selector = {
        "workbench_run_id": run.run_id,
        "context": _context_summary(run),
        "target": _target_summary(run),
    }
    method = {
        "workbench_run_id": run.run_id,
        "operator": operator,
        "schedule": schedule,
        "output_refs": list(run.outputs),
        "request_hash": _request_hash(request),
    }
    display = {
        "kind": "intervention_card",
        "workbench_run_id": run.run_id,
        "title": run.title,
        "status": run.status,
        "claim": jsonable(run.claim),
        "context": selector["context"],
        "target": selector["target"],
        "intervention": {
            "operator": operator.get("operator"),
            "strength": operator.get("strength"),
            "strengths": operator.get("strengths", ()),
            "schedule": schedule,
        },
        "outcome": {
            "kind": outcome.get("kind"),
            "basis": outcome.get("basis", ()),
            "summary": _outcome_summary(run),
        },
        "controls": _control_summary(run),
        "output_refs": list(run.outputs),
        "provenance": {
            "schema_version": run.schema_version,
            "created_utc": run.created_utc,
            "dataset_fingerprint": run.context.dataset_fingerprint,
            "source_artifact_id": run.target.source_artifact_id,
        },
    }
    return LensArtifact.create(
        artifact_type=ARTIFACT_TYPE,
        name=run.title or f"Intervention run {run.run_id}",
        group_id=run.run_id,
        scope="dataset",
        selector=selector,
        method=method,
        metrics=_compact_metrics(run),
        arrays={},
        display=display,
        tags=_tags(run, operator, outcome),
        source_trace_ids=_source_trace_ids(run),
    )


def intervention_sweep_to_lens_artifact(sweep: InterventionSweep) -> LensArtifact:
    """Create an artifact-browser shell for a sweep aggregate."""
    selector = {
        "sweep_id": sweep.sweep_id,
        "run_ids": list(sweep.run_ids),
        "cohort": jsonable(sweep.cohort),
    }
    aggregate_metrics = {
        aggregate.metric: aggregate.to_dict() for aggregate in sweep.aggregate_outcomes
    }
    display = {
        "kind": "intervention_sweep_card",
        "sweep_id": sweep.sweep_id,
        "run_ids": list(sweep.run_ids),
        "axes": jsonable(sweep.axes),
        "cohort": jsonable(sweep.cohort),
        "summary": jsonable(sweep.summary),
        "claim": {
            "claim_strength": _summary_claim_labels(sweep.summary),
        },
        "controls": [jsonable(control) for control in sweep.controls],
        "aggregate_outcomes": aggregate_metrics,
        "provenance": {
            "schema_version": sweep.schema_version,
            "created_utc": sweep.provenance.get("created_utc"),
        },
    }
    return LensArtifact.create(
        artifact_type=SWEEP_ARTIFACT_TYPE,
        name=f"Intervention sweep {sweep.sweep_id}",
        group_id=sweep.sweep_id,
        scope="dataset",
        selector=selector,
        method={
            "axes": jsonable(sweep.axes),
            "controls": [jsonable(control) for control in sweep.controls],
            "run_count": len(sweep.run_ids),
        },
        metrics=_sweep_metrics(sweep, aggregate_metrics),
        display=display,
        tags=_sweep_tags(sweep),
        source_trace_ids=_sweep_source_trace_ids(sweep),
    )


def _context_summary(run: InterventionRun) -> dict[str, Any]:
    return {
        "dataset_id": run.context.dataset_id,
        "dataset_root_id": run.context.dataset_root_id,
        "dataset_fingerprint": run.context.dataset_fingerprint,
        "trace_id": run.context.trace_id,
        "episode_id": run.context.episode_id,
        "policy_call_index": run.context.policy_call_index,
        "timestep": run.context.timestep,
        "frame_index": run.context.frame_index,
        "instruction": run.context.instruction,
        "task": run.context.task,
    }


def _target_summary(run: InterventionRun) -> dict[str, Any]:
    return {
        "kind": run.target.kind,
        "source_artifact_id": run.target.source_artifact_id,
        "source_artifact_type": run.target.source_artifact_type,
        "model_site": run.target.model_site,
        "site_id": run.target.site_id,
        "layer": run.target.layer,
        "tensor_type": run.target.tensor_type,
        "token_space": run.target.token_space,
        "reduction": run.target.reduction,
    }


def _compact_metrics(run: InterventionRun) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "status": run.status,
        "trial_count": len(run.trials),
        "outcome_count": len(run.outcomes),
        "control_count": len(run.controls),
    }
    for index, outcome in enumerate(run.outcomes):
        outcome_metrics = _mapping(outcome).get("metrics")
        if isinstance(outcome_metrics, Mapping):
            metrics[f"outcome_{index}"] = jsonable(outcome_metrics)
    for index, control in enumerate(run.controls):
        control_metrics = _mapping(control).get("metrics")
        if isinstance(control_metrics, Mapping):
            metrics[f"control_{index}"] = jsonable(control_metrics)
    return metrics


def _outcome_summary(run: InterventionRun) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for outcome in run.outcomes:
        payload = _mapping(outcome)
        summaries.append(
            {
                "kind": payload.get("kind"),
                "basis": payload.get("basis"),
                "horizon": payload.get("horizon"),
                "metrics": payload.get("metrics", {}),
                "delta_ref": payload.get("delta_ref"),
            }
        )
    return summaries


def _control_summary(run: InterventionRun) -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []
    for control in run.controls:
        payload = _mapping(control)
        controls.append(
            {
                "control_kind": payload.get("control_kind") or payload.get("kind"),
                "status": payload.get("status"),
                "trial_ids": payload.get("trial_ids", ()),
                "metrics": payload.get("metrics", {}),
            }
        )
    return controls


def _tags(
    run: InterventionRun,
    operator: Mapping[str, Any],
    outcome: Mapping[str, Any],
) -> tuple[str, ...]:
    tags = {
        ARTIFACT_TYPE,
        run.status,
        run.target.kind,
    }
    if operator.get("operator"):
        tags.add(str(operator["operator"]))
    if outcome.get("kind"):
        tags.add(str(outcome["kind"]))
    for label in _claim_labels(run.claim):
        tags.add(label)
    return tuple(sorted(tag for tag in tags if tag))


def _source_trace_ids(run: InterventionRun) -> tuple[str, ...]:
    trace_ids = []
    if run.context.trace_id:
        trace_ids.append(run.context.trace_id)
    request = _mapping(run.request)
    for key in ("donor", "recipient"):
        value = request.get(key)
        if isinstance(value, Mapping):
            trace_id = value.get("trace_id")
            if trace_id:
                trace_ids.append(str(trace_id))
    return tuple(dict.fromkeys(trace_ids))


def _request_hash(request: Mapping[str, Any]) -> str:
    encoded = json.dumps(jsonable(request), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _claim_labels(claim: Mapping[str, Any]) -> tuple[str, ...]:
    labels = claim.get("claim_strength", claim.get("claim_strengths", ()))
    if isinstance(labels, str):
        return (labels,)
    if isinstance(labels, (list, tuple, set, frozenset)):
        return tuple(str(label) for label in labels if label)
    return ()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sweep_metrics(
    sweep: InterventionSweep,
    aggregate_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "run_count": len(sweep.run_ids),
        "failure_count": sweep.summary.get("failure_count", 0),
        "coverage": sweep.summary.get("coverage"),
        "aggregate_outcomes": jsonable(aggregate_metrics),
    }


def _sweep_tags(sweep: InterventionSweep) -> tuple[str, ...]:
    tags = {SWEEP_ARTIFACT_TYPE, "sweep"}
    if sweep.cohort:
        tags.add("cohort")
    for label in _summary_claim_labels(sweep.summary):
        tags.add(label)
    return tuple(sorted(tag for tag in tags if tag))


def _summary_claim_labels(summary: Mapping[str, Any]) -> tuple[str, ...]:
    labels = summary.get("claim_labels", ())
    if isinstance(labels, str):
        return (labels,)
    if isinstance(labels, (list, tuple, set, frozenset)):
        return tuple(str(label) for label in labels if label)
    return ()


def _sweep_source_trace_ids(sweep: InterventionSweep) -> tuple[str, ...]:
    cohort = _mapping(sweep.cohort)
    members = _mapping(cohort.get("members"))
    trace_ids = members.get("trace_id", ())
    if isinstance(trace_ids, str):
        return (trace_ids,)
    if isinstance(trace_ids, (list, tuple, set, frozenset)):
        return tuple(dict.fromkeys(str(trace_id) for trace_id in trace_ids if trace_id))
    provenance_trace_ids = sweep.provenance.get("source_trace_ids", ())
    if isinstance(provenance_trace_ids, str):
        return (provenance_trace_ids,)
    if isinstance(provenance_trace_ids, (list, tuple, set, frozenset)):
        return tuple(
            dict.fromkeys(str(trace_id) for trace_id in provenance_trace_ids if trace_id)
        )
    return ()
