from __future__ import annotations

import json
from dataclasses import replace

from vla_lens.interventions import (
    ActionOutcomeResult,
    CohortInterventionRequest,
    ContextSpec,
    InterventionRun,
    RuntimePreflightResult,
    SweepAxis,
    TargetSpec,
    aggregate_outcome_metric,
    build_intervention_study,
    build_intervention_sweep,
    promote_run_to_sweep,
)


def _run(
    run_id: str = "run-a",
    *,
    strength: float = 1.0,
    trace_id: str = "trace-a",
    status: str = "ok",
    metric: float = 0.2,
    claim_strength: tuple[str, ...] = ("causal_local",),
) -> InterventionRun:
    return InterventionRun(
        run_id=run_id,
        title=f"Run {run_id}",
        status=status,
        context=ContextSpec(
            dataset_id="demo",
            dataset_fingerprint="fingerprint-1",
            trace_id=trace_id,
            episode_id=f"episode-{trace_id}",
            policy_call_index=7,
        ),
        target=TargetSpec(
            kind="probe_direction",
            source_artifact_id="probe-gripper-close",
            source_artifact_type="probe_suite",
            model_site="pi05.expert.layers.12.hidden_tokens",
            token_space="action",
        ),
        request={
            "operator": {"operator": "add_direction", "strength": strength},
            "schedule": {"policy_calls": [7], "tokens": "action"},
            "outcome": {"kind": "action", "basis": ["raw"]},
        },
        preflight=RuntimePreflightResult(status=status),
        outcomes=(
            ActionOutcomeResult(
                basis="raw",
                horizon="full_chunk",
                baseline_trial_id="trial_noop",
                intervention_trial_id="trial_intervention",
                metrics={"raw_delta_norm": metric, "side_effect_score": metric / 10},
            ).to_dict(),
        ),
        claim={"claim_strength": list(claim_strength)},
        provenance={"dataset_fingerprint": "fingerprint-1"},
        created_utc="2026-06-06T00:00:00+00:00",
    )


def test_promote_run_to_sweep_preserves_single_run_meaning():
    run = _run(strength=0.75, metric=0.15)
    before = run.to_dict()

    sweep = promote_run_to_sweep(run, provenance={"test_created": True})

    assert run.to_dict() == before
    assert sweep.run_ids == (run.run_id,)
    assert sweep.provenance["base_run_id"] == run.run_id
    assert [axis.name for axis in sweep.axes] == [
        "strength",
        "policy_call",
        "target_source_artifact",
    ]
    assert sweep.axes[0].values == (0.75,)
    assert sweep.summary["run_count"] == 1


def test_sweep_aggregates_metrics_and_gates_claim_labels():
    runs = (
        _run("run-low", strength=-1.0, trace_id="trace-1", metric=0.1),
        _run("run-mid", strength=0.0, trace_id="trace-2", metric=0.2),
        _run("run-high", strength=1.0, trace_id="trace-3", metric=0.3),
    )
    cohort = {"cohort_id": "heldout", "members": {"trace_id": ["trace-1", "trace-2", "trace-3"]}}
    controls = ({"control_kind": "random_direction", "status": "ok", "metrics": {}},)

    sweep = build_intervention_sweep(
        sweep_id="sweep-strength",
        runs=runs,
        cohort=cohort,
        controls=controls,
    )

    aggregate = sweep.summary["aggregate_metrics"]["raw_delta_norm"]
    assert aggregate["count"] == 3
    assert round(aggregate["mean"], 6) == 0.2
    assert aggregate["median"] == 0.2
    assert aggregate["monotonicity"] == "nondecreasing"
    assert sweep.summary["coverage"] == 1.0
    assert sweep.summary["failure_count"] == 0
    assert {
        "action_level",
        "causal_cohort",
        "observation",
        "specific",
    } <= set(sweep.summary["claim_labels"])
    assert "behavioral" not in sweep.summary["claim_labels"]


def test_sweep_does_not_add_cohort_or_specific_claims_without_evidence():
    sweep = build_intervention_sweep(
        sweep_id="sweep-local",
        runs=(_run("run-local", trace_id="trace-1"),),
    )

    assert "action_level" in sweep.summary["claim_labels"]
    assert "causal_cohort" not in sweep.summary["claim_labels"]
    assert "specific" not in sweep.summary["claim_labels"]


def test_aggregate_outcome_metric_reports_partial_coverage():
    runs = (
        _run("run-a", metric=0.4),
        _run("run-b", metric=0.6),
        _run("run-c", claim_strength=("observation",), metric=0.0),
    )
    runs = (runs[0], runs[1], replace(runs[2], outcomes=()))

    aggregate = aggregate_outcome_metric(runs, "raw_delta_norm")

    assert aggregate.count == 2
    assert aggregate.mean == 0.5
    assert aggregate.coverage == 2 / 3


def test_study_references_sweeps_requests_controls_and_cohort():
    axis = SweepAxis(
        name="strength",
        values=(-1.0, 0.0, 1.0),
        path=("request", "operator", "strength"),
    )
    request = CohortInterventionRequest(
        request_id="request-heldout",
        base_run_id="run-low",
        cohort={"cohort_id": "heldout"},
        axes=(axis,),
        controls=({"control_kind": "random_direction"},),
    )
    sweep = build_intervention_sweep(
        sweep_id="sweep-strength",
        runs=(_run("run-low", strength=-1.0), _run("run-high", strength=1.0)),
        axes=(axis,),
        cohort={"cohort_id": "heldout"},
        controls=({"control_kind": "random_direction", "status": "ok"},),
    )

    loaded_request = CohortInterventionRequest.from_dict(
        json.loads(json.dumps(request.to_dict()))
    )
    study = build_intervention_study(
        study_id="study-heldout",
        sweeps=(sweep,),
        requests=(loaded_request,),
    )
    loaded_study = type(study).from_dict(json.loads(json.dumps(study.to_dict())))

    assert loaded_request == request
    assert loaded_study == study
    assert study.sweep_ids == ("sweep-strength",)
    assert study.request_ids == ("request-heldout",)
    assert study.run_ids == ("run-low", "run-high")
    assert study.cohort["cohort_id"] == "heldout"
    assert study.controls[0]["control_kind"] == "random_direction"
