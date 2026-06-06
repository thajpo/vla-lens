"""Runtime-free helpers for intervention sweeps and cohort studies."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import sqrt
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from vla_lens.interventions.results import (
    AggregateOutcomeResult,
    CohortInterventionRequest,
    InterventionRun,
    InterventionStudy,
    InterventionSweep,
    SweepAxis,
)
from vla_lens.interventions.serialization import jsonable, utc_now_iso


def promote_run_to_sweep(
    run: InterventionRun,
    *,
    sweep_id: str | None = None,
    axes: Sequence[SweepAxis] | None = None,
    cohort: Mapping[str, Any] | None = None,
    controls: Sequence[Mapping[str, Any]] = (),
    provenance: Mapping[str, Any] | None = None,
) -> InterventionSweep:
    """Promote one existing run into a one-run sweep summary without changing it."""
    return build_intervention_sweep(
        sweep_id=sweep_id or f"sweep-{run.run_id}",
        runs=(run,),
        axes=tuple(axes) if axes is not None else _axes_from_run(run),
        cohort=cohort or {},
        controls=controls,
        provenance={
            "source": "promote_run_to_sweep",
            "base_run_id": run.run_id,
            "created_utc": utc_now_iso(),
            **(provenance or {}),
        },
    )


def build_intervention_sweep(
    *,
    sweep_id: str,
    runs: Sequence[InterventionRun],
    axes: Sequence[SweepAxis] | Mapping[str, Any] | None = None,
    cohort: Mapping[str, Any] | None = None,
    controls: Sequence[Mapping[str, Any]] = (),
    provenance: Mapping[str, Any] | None = None,
) -> InterventionSweep:
    """Build an aggregate sweep summary from already materialized runs."""
    if not runs:
        raise ValueError("sweep runs are required")
    aggregate_outcomes = _aggregate_all_outcome_metrics(runs)
    normalized_axes = _normalize_axes(axes, runs)
    controls_tuple = tuple(jsonable(control) for control in controls)
    evidence_controls = controls_tuple + tuple(
        control for run in runs for control in run.controls
    )
    cohort_payload = jsonable(cohort or {})
    summary = _sweep_summary(
        runs,
        aggregates=aggregate_outcomes,
        controls=evidence_controls,
        cohort=cohort_payload,
    )
    return InterventionSweep(
        sweep_id=sweep_id,
        run_ids=tuple(run.run_id for run in runs),
        axes=normalized_axes,
        aggregate_outcomes=aggregate_outcomes,
        controls=controls_tuple,
        cohort=cohort_payload,
        summary=summary,
        provenance={
            "schema_kind": "vla_lens.intervention_sweep",
            "created_utc": utc_now_iso(),
            **(provenance or {}),
        },
    )


def build_intervention_study(
    *,
    study_id: str,
    sweeps: Sequence[InterventionSweep],
    requests: Sequence[CohortInterventionRequest] = (),
    cohort: Mapping[str, Any] | None = None,
    controls: Sequence[Mapping[str, Any]] = (),
    provenance: Mapping[str, Any] | None = None,
) -> InterventionStudy:
    """Build a study shell that references sweeps, requests, controls, and a cohort."""
    if not sweeps:
        raise ValueError("study sweeps are required")
    aggregate_outcomes = _aggregate_sweep_aggregates(sweeps)
    all_run_ids = tuple(dict.fromkeys(run_id for sweep in sweeps for run_id in sweep.run_ids))
    controls_tuple = tuple(jsonable(control) for control in controls)
    if not controls_tuple:
        controls_tuple = tuple(
            control for sweep in sweeps for control in sweep.controls
        )
    cohort_payload = jsonable(cohort or _first_nonempty(sweep.cohort for sweep in sweeps) or {})
    summary = {
        "sweep_count": len(sweeps),
        "run_count": len(all_run_ids),
        "request_count": len(requests),
        "control_count": len(controls_tuple),
        "claim_labels": _claim_labels_from_summary(
            sweep.summary for sweep in sweeps
        ),
        "aggregate_metrics": {
            aggregate.metric: aggregate.to_dict() for aggregate in aggregate_outcomes
        },
    }
    return InterventionStudy(
        study_id=study_id,
        sweep_ids=tuple(sweep.sweep_id for sweep in sweeps),
        run_ids=all_run_ids,
        request_ids=tuple(request.request_id for request in requests),
        cohort=cohort_payload,
        controls=controls_tuple,
        aggregate_outcomes=aggregate_outcomes,
        summary=summary,
        provenance={
            "schema_kind": "vla_lens.intervention_study",
            "created_utc": utc_now_iso(),
            **(provenance or {}),
        },
    )


def aggregate_outcome_metric(
    runs: Sequence[InterventionRun],
    metric: str,
    *,
    axis_values: Sequence[float] | None = None,
) -> AggregateOutcomeResult:
    """Aggregate one numeric outcome metric across runs."""
    values = _numeric_values_for_metric(runs, metric)
    count = len(values)
    coverage = count / len(runs) if runs else 0.0
    if not values:
        return AggregateOutcomeResult(metric=metric, count=0, coverage=coverage)
    return AggregateOutcomeResult(
        metric=metric,
        count=count,
        mean=sum(values) / count,
        median=float(median(values)),
        minimum=min(values),
        maximum=max(values),
        std=_std(values),
        coverage=coverage,
        monotonicity=_monotonicity(axis_values or _strength_axis_values(runs), values),
        values=tuple(values),
    )


def _normalize_axes(
    axes: Sequence[SweepAxis] | Mapping[str, Any] | None,
    runs: Sequence[InterventionRun],
) -> tuple[SweepAxis, ...] | Mapping[str, Any]:
    if axes is None:
        return _axes_from_runs(runs)
    if isinstance(axes, Mapping):
        return axes
    return tuple(axes)


def _axes_from_run(run: InterventionRun) -> tuple[SweepAxis, ...]:
    return _axes_from_runs((run,))


def _axes_from_runs(runs: Sequence[InterventionRun]) -> tuple[SweepAxis, ...]:
    strengths = tuple(_unique(_run_strength(run) for run in runs if _run_strength(run) is not None))
    policy_calls = tuple(
        _unique(
            run.context.policy_call_index
            for run in runs
            if run.context.policy_call_index is not None
        )
    )
    source_artifacts = tuple(
        _unique(
            run.target.source_artifact_id
            for run in runs
            if run.target.source_artifact_id
        )
    )
    axes: list[SweepAxis] = []
    if strengths:
        axes.append(
            SweepAxis(
                name="strength",
                values=strengths,
                path=("request", "operator", "strength"),
            )
        )
    if policy_calls:
        axes.append(
            SweepAxis(
                name="policy_call",
                values=policy_calls,
                source="context",
                path=("context", "policy_call_index"),
            )
        )
    if source_artifacts:
        axes.append(
            SweepAxis(
                name="target_source_artifact",
                values=source_artifacts,
                source="target",
                path=("target", "source_artifact_id"),
            )
        )
    return tuple(axes)


def _sweep_summary(
    runs: Sequence[InterventionRun],
    *,
    aggregates: Sequence[AggregateOutcomeResult],
    controls: Sequence[Mapping[str, Any]],
    cohort: Mapping[str, Any],
) -> dict[str, Any]:
    status_counts = Counter(run.status for run in runs)
    trace_ids = tuple(_unique(run.context.trace_id for run in runs if run.context.trace_id))
    claim_labels = _claim_labels_for_runs(runs, controls=controls, cohort=cohort)
    return {
        "run_count": len(runs),
        "status_counts": dict(status_counts),
        "failure_count": status_counts.get("failed", 0),
        "coverage": _execution_coverage(runs),
        "trace_count": len(trace_ids),
        "claim_labels": claim_labels,
        "aggregate_metrics": {
            aggregate.metric: aggregate.to_dict() for aggregate in aggregates
        },
    }


def _aggregate_all_outcome_metrics(
    runs: Sequence[InterventionRun],
) -> tuple[AggregateOutcomeResult, ...]:
    metrics = sorted(
        {
            key
            for run in runs
            for outcome in run.outcomes
            for key, value in _mapping(outcome).get("metrics", {}).items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
    )
    return tuple(aggregate_outcome_metric(runs, metric) for metric in metrics)


def _aggregate_sweep_aggregates(
    sweeps: Sequence[InterventionSweep],
) -> tuple[AggregateOutcomeResult, ...]:
    values_by_metric: dict[str, list[float]] = defaultdict(list)
    for sweep in sweeps:
        for aggregate in sweep.aggregate_outcomes:
            values_by_metric[aggregate.metric].extend(aggregate.values)
    aggregates = []
    for metric, values in sorted(values_by_metric.items()):
        aggregates.append(
            AggregateOutcomeResult(
                metric=metric,
                count=len(values),
                mean=sum(values) / len(values) if values else None,
                median=float(median(values)) if values else None,
                minimum=min(values) if values else None,
                maximum=max(values) if values else None,
                std=_std(values) if values else None,
                coverage=1.0 if values else 0.0,
                values=tuple(values),
            )
        )
    return tuple(aggregates)


def _numeric_values_for_metric(runs: Sequence[InterventionRun], metric: str) -> list[float]:
    values: list[float] = []
    for run in runs:
        run_values = []
        for outcome in run.outcomes:
            metrics = _mapping(outcome).get("metrics", {})
            value = metrics.get(metric) if isinstance(metrics, Mapping) else None
            if isinstance(value, int | float) and not isinstance(value, bool):
                run_values.append(float(value))
        if run_values:
            values.append(sum(run_values) / len(run_values))
    return values


def _claim_labels_for_runs(
    runs: Sequence[InterventionRun],
    *,
    controls: Sequence[Mapping[str, Any]],
    cohort: Mapping[str, Any],
) -> list[str]:
    labels: set[str] = {"observation"}
    outcome_kinds = {
        str(_mapping(outcome).get("kind"))
        for run in runs
        for outcome in run.outcomes
        if _mapping(outcome).get("kind")
    }
    if "action" in outcome_kinds:
        labels.add("action_level")
    if "rollout" in outcome_kinds:
        labels.add("behavioral")
    if _has_successful_control(controls):
        labels.add("specific")
    trace_count = len(_unique(run.context.trace_id for run in runs if run.context.trace_id))
    if cohort and trace_count > 1 and any(_has_claim(run, "causal_local") for run in runs):
        labels.add("causal_cohort")
    return sorted(labels)


def _claim_labels_from_summary(summaries: Iterable[Mapping[str, Any]]) -> list[str]:
    labels: set[str] = set()
    for summary in summaries:
        value = summary.get("claim_labels", ())
        if isinstance(value, str):
            labels.add(value)
        elif isinstance(value, Sequence):
            labels.update(str(item) for item in value)
    return sorted(labels)


def _has_successful_control(controls: Sequence[Mapping[str, Any]]) -> bool:
    return any(str(control.get("status")) in {"ok", "partial"} for control in controls)


def _has_claim(run: InterventionRun, label: str) -> bool:
    claim_labels = run.claim.get("claim_strength", run.claim.get("claim_strengths", ()))
    if isinstance(claim_labels, str):
        return claim_labels == label
    if isinstance(claim_labels, Sequence):
        return label in {str(item) for item in claim_labels}
    return False


def _execution_coverage(runs: Sequence[InterventionRun]) -> float:
    runnable = sum(1 for run in runs if run.status in {"ok", "partial"})
    return runnable / len(runs) if runs else 0.0


def _run_strength(run: InterventionRun) -> float | None:
    operator = _mapping(run.request.get("operator"))
    value = operator.get("strength")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _strength_axis_values(runs: Sequence[InterventionRun]) -> tuple[float, ...]:
    strengths = [_run_strength(run) for run in runs]
    return tuple(value for value in strengths if value is not None)


def _monotonicity(axis_values: Sequence[float], values: Sequence[float]) -> str | None:
    if len(axis_values) != len(values) or len(values) < 3:
        return None
    ordered = [value for _, value in sorted(zip(axis_values, values, strict=True))]
    deltas = [right - left for left, right in zip(ordered, ordered[1:], strict=False)]
    if all(delta >= 0 for delta in deltas):
        return "nondecreasing"
    if all(delta <= 0 for delta in deltas):
        return "nonincreasing"
    return "mixed"


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = sum(values) / len(values)
    return sqrt(sum((value - mean_value) ** 2 for value in values) / len(values))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_nonempty(values: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for value in values:
        if value:
            return value
    return None


def _unique(values: Iterable[Any]) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(value for value in values if value is not None))
