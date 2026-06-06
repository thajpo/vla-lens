from __future__ import annotations

import json

import pytest

from vla_lens import create_synthetic_trace_dataset
from vla_lens.interventions import (
    ActionOutcomeResult,
    ContextSpec,
    InterventionRun,
    InterventionTrial,
    RuntimePreflightResult,
    TargetSpec,
)
from vla_lens.workbench import list_analysis_runs, save_intervention_run
from vla_lens.workbench.schema import InterventionRunSpec


def _target(**overrides):
    payload = {
        "kind": "probe_direction",
        "source_artifact_id": "probe-gripper-close",
        "source_artifact_type": "probe_suite",
        "model_site": "pi05.expert.layers.12.hidden_tokens",
        "token_space": "action",
        "reduction": "mean",
        "representation": {"kind": "vector", "array_ref": "artifact://probe/coef"},
    }
    payload.update(overrides)
    return TargetSpec(**payload)


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


def _run(*, causal: bool = False) -> InterventionRun:
    stored = InterventionTrial(
        trial_id="trial_stored_original",
        trial_kind="stored_original",
        status="inspected_only",
        outputs={"action_ref": "array://stored-original"},
    )
    trials = [stored]
    outcomes = []
    claim = {"claim_strength": ["observation"], "limitations": ["no live runtime trial"]}
    status = "inspected_only"
    if causal:
        intervention = InterventionTrial(
            trial_id="trial_intervention",
            trial_kind="intervention",
            status="ok",
            outputs={"action_ref": "array://intervened"},
        )
        trials.append(intervention)
        outcomes.append(
            ActionOutcomeResult(
                basis="raw",
                horizon="full_chunk",
                baseline_trial_id="trial_stored_original",
                intervention_trial_id="trial_intervention",
                action_ref_baseline="array://stored-original",
                action_ref_intervened="array://intervened",
                delta_ref="array://delta",
                metrics={"raw_delta_norm": 0.2},
            ).to_dict()
        )
        claim = {"claim_strength": ["causal_local"], "limitations": ["single policy call"]}
        status = "ok"
    return InterventionRun(
        run_id="run-causal" if causal else "run-saved-only",
        title="Probe direction run",
        status=status,
        context=_context(),
        target=_target(),
        request={
            "operator": {"operator": "add_direction", "strength": 1.0},
            "schedule": {"policy_calls": [7], "tokens": "action"},
            "outcome": {"kind": "action", "basis": ["raw"]},
        },
        preflight=RuntimePreflightResult(status=status),
        trials=tuple(trials),
        outcomes=tuple(outcomes),
        outputs=("array://stored-original", "array://delta") if causal else ("array://stored-original",),
        display={"summary": "probe direction"},
        claim=claim,
        provenance={
            "schema_kind": "vla_lens.intervention_run",
            "schema_version": "0.1.0",
            "dataset_id": "demo",
            "dataset_root_id": None,
            "dataset_fingerprint": "fingerprint-1",
            "trace_id": "trace-1",
            "episode_id": "episode-1",
            "policy_call_index": 7,
            "source_artifact_id": "probe-gripper-close",
            "created_utc": "2026-06-06T00:00:00+00:00",
        },
        created_utc="2026-06-06T00:00:00+00:00",
    )


def test_typed_run_to_workbench_spec_roundtrip():
    run = _run(causal=True)
    spec = run.to_workbench_spec()
    payload = json.loads(json.dumps(spec.to_dict()))
    loaded_spec = InterventionRunSpec.from_dict(payload)
    loaded_run = InterventionRun.from_workbench_spec(loaded_spec)

    assert spec.intervention_type == "intervention_record"
    assert loaded_run == run
    assert loaded_spec.provenance["schema_kind"] == "vla_lens.intervention_run"
    assert loaded_spec.provenance["dataset_fingerprint"] == "fingerprint-1"


def test_saved_only_record_not_causal(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    save_intervention_run(dataset, _run(causal=False).to_workbench_spec())

    analysis_run = next(
        run for run in list_analysis_runs(dataset) if run.run_id == "run-saved-only"
    )

    assert analysis_run.provenance["causal_evidence"] is False
    assert analysis_run.inputs["intervention_type"] == "intervention_record"


def test_executed_record_causality_is_interpreted_from_typed_fields(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    saved = save_intervention_run(dataset, _run(causal=True).to_workbench_spec())

    analysis_run = next(run for run in list_analysis_runs(dataset) if run.run_id == "run-causal")

    assert saved.intervention_type == "intervention_record"
    assert analysis_run.inputs["intervention_type"] == "intervention_record"
    assert analysis_run.provenance["causal_evidence"] is True


def test_workbench_spec_rejects_unknown_intervention_type():
    with pytest.raises(ValueError, match="intervention_type is required"):
        InterventionRunSpec.from_dict({"run_id": "missing_type", "target": {}})

    with pytest.raises(ValueError, match="Unsupported intervention_type"):
        InterventionRunSpec.from_dict(
            {
                "run_id": "unknown_type",
                "intervention_type": "old_shell_alias",
                "target": {},
            }
        )


def test_artifact_source_validation_remains_available_through_typed_shell():
    with pytest.raises(ValueError, match="source_artifact_id"):
        _target(source_artifact_id=None)

    manual = TargetSpec(kind="manual", model_site="pi05.expert.layers.12.hidden_tokens")

    assert manual.source_artifact_id is None


def test_workbench_spec_outputs_refs_not_inline_arrays():
    spec = _run(causal=True).to_workbench_spec()

    assert spec.outputs == ("array://stored-original", "array://delta")
    assert "values" not in spec.to_dict()
