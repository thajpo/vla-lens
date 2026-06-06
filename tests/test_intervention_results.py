from __future__ import annotations

import json

import pytest

from vla_lens.interventions import (
    ActionOutcomeResult,
    ContextSpec,
    ControlResult,
    InterventionRun,
    InterventionStudy,
    InterventionSweep,
    InterventionTrial,
    RuntimePreflightResult,
    TargetSpec,
)


def _context(**overrides):
    payload = {
        "dataset_id": "demo",
        "dataset_fingerprint": "fingerprint-1",
        "trace_id": "trace-1",
        "episode_id": "episode-1",
        "policy_call_index": 7,
        "instruction": "close the gripper",
    }
    payload.update(overrides)
    return ContextSpec(**payload)


def _target() -> TargetSpec:
    return TargetSpec(
        kind="probe_direction",
        source_artifact_id="probe-gripper-close",
        source_artifact_type="probe_suite",
        model_site="pi05.expert.layers.12.hidden_tokens",
        token_space="action",
        reduction="mean",
        representation={"kind": "vector", "array_ref": "artifact://probe/coef"},
    )


def _run(**overrides) -> InterventionRun:
    stored = InterventionTrial(
        trial_id="trial_stored_original",
        trial_kind="stored_original",
        outputs={"action_ref": "array://stored_original_action"},
        status="inspected_only",
    )
    preflight = RuntimePreflightResult(
        status="inspected_only",
        missing_capabilities=("model_runtime",),
        capability_status={"policy_call_exists": True, "model_runtime_available": False},
    )
    outcome = ActionOutcomeResult(
        basis="raw",
        horizon="full_chunk",
        baseline_trial_id="trial_stored_original",
        intervention_trial_id="trial_stored_original",
        action_ref_baseline="array://stored_original_action",
        metrics={"raw_delta_norm": 0.0},
    )
    payload = {
        "run_id": "run-probe-direction-1",
        "title": "Probe direction inspected record",
        "status": "inspected_only",
        "context": _context(),
        "target": _target(),
        "request": {
            "operator": {"operator": "add_direction", "strength": 1.0},
            "schedule": {"policy_calls": [7], "tokens": "action"},
            "outcome": {"kind": "action", "basis": ["raw"]},
        },
        "preflight": preflight,
        "trials": (stored,),
        "outcomes": (outcome.to_dict(),),
        "controls": (),
        "outputs": ("array://stored_original_action",),
        "display": {"summary": "inspected-only record"},
        "claim": {
            "claim_strength": ["observation"],
            "limitations": ["no live runtime trial"],
        },
        "provenance": {
            "dataset_fingerprint": "fingerprint-1",
            "source_artifact_id": "probe-gripper-close",
        },
        "created_utc": "2026-06-06T00:00:00+00:00",
    }
    payload.update(overrides)
    return InterventionRun(**payload)


def test_intervention_trial_and_action_outcome_roundtrip():
    trial = InterventionTrial(
        trial_id="trial_intervention",
        trial_kind="intervention",
        strength=1.5,
        outputs={"action_ref": "array://intervened"},
        metrics={"delta_norm": 0.25},
        warnings=("single policy call",),
    )
    outcome = ActionOutcomeResult(
        basis="raw",
        horizon={"start": 0, "stop": 8},
        baseline_trial_id="trial_noop",
        intervention_trial_id="trial_intervention",
        action_ref_baseline="array://noop",
        action_ref_intervened="array://intervened",
        delta_ref="array://delta",
        metrics={"raw_delta_norm": 0.25},
    )

    loaded_trial = InterventionTrial.from_dict(json.loads(json.dumps(trial.to_dict())))
    loaded_outcome = ActionOutcomeResult.from_dict(json.loads(json.dumps(outcome.to_dict())))

    assert loaded_trial == trial
    assert loaded_outcome == outcome


def test_intervention_run_json_roundtrip():
    run = _run()

    loaded = InterventionRun.from_dict(json.loads(json.dumps(run.to_dict())))

    assert loaded == run
    assert loaded.schema_version == "0.1.0"
    assert loaded.status == "inspected_only"
    assert loaded.claim["claim_strength"] == ["observation"]


def test_intervention_run_required_identity_fields():
    with pytest.raises(ValueError, match="dataset_id or dataset_root_id"):
        _run(context=_context(dataset_id=None, dataset_root_id=None))

    with pytest.raises(ValueError, match="dataset_fingerprint"):
        _run(context=_context(dataset_fingerprint=None))

    with pytest.raises(ValueError, match="trace_id"):
        _run(context=_context(trace_id=None))

    with pytest.raises(ValueError, match="policy_call_index"):
        _run(context=_context(policy_call_index=None))


def test_intervention_run_keeps_status_separate_from_claim_strength():
    run = _run(
        status="ok",
        claim={
            "claim_strength": ["observation"],
            "limitations": ["runtime succeeded but no controls were run"],
        },
    )

    payload = run.to_dict()

    assert payload["status"] == "ok"
    assert payload["claim"]["claim_strength"] == ["observation"]
    assert "causal" not in payload["status"]


def test_sweep_and_study_shells_roundtrip():
    sweep = InterventionSweep(
        sweep_id="sweep-1",
        run_ids=("run-a", "run-b"),
        axes={"strength": [-1.0, 1.0]},
        summary={"count": 2},
    )
    study = InterventionStudy(
        study_id="study-1",
        sweep_ids=("sweep-1",),
        run_ids=("run-a", "run-b"),
        cohort={"task": "close gripper"},
        summary={"mean_effect": 0.1},
    )
    control = ControlResult(
        control_kind="random_direction",
        status="partial",
        trial_ids=("trial-control",),
        warnings=("runtime unavailable",),
    )

    loaded_sweep = InterventionSweep.from_dict(json.loads(json.dumps(sweep.to_dict())))
    loaded_study = InterventionStudy.from_dict(json.loads(json.dumps(study.to_dict())))
    loaded_control = ControlResult.from_dict(json.loads(json.dumps(control.to_dict())))

    assert loaded_sweep == sweep
    assert loaded_study == study
    assert loaded_control == control
