from __future__ import annotations

from vla_lens import TraceDataset, create_synthetic_trace_dataset
from vla_lens.interventions import (
    ActionOutcomeResult,
    ContextSpec,
    InterventionRun,
    InterventionTrial,
    RuntimePreflightResult,
    TargetSpec,
    intervention_run_to_lens_artifact,
)
from vla_lens.workbench import save_intervention_run


def _run() -> InterventionRun:
    return InterventionRun(
        run_id="run-artifact-index",
        title="Gripper-close direction action change",
        status="ok",
        context=ContextSpec(
            dataset_id="demo",
            dataset_fingerprint="fingerprint-1",
            trace_id="trace-1",
            episode_id="episode-1",
            policy_call_index=7,
            instruction="close the gripper",
        ),
        target=TargetSpec(
            kind="probe_direction",
            source_artifact_id="probe-gripper-close",
            source_artifact_type="probe_suite",
            model_site="pi05.expert.layers.12.hidden_tokens",
            layer=12,
            token_space="action",
            reduction="mean",
            representation={"kind": "vector", "array_ref": "artifact://probe/coef"},
        ),
        request={
            "operator": {"operator": "add_direction", "strength": 1.0},
            "schedule": {"policy_calls": [7], "tokens": "action"},
            "outcome": {"kind": "action", "basis": ["raw", "gripper"]},
        },
        preflight=RuntimePreflightResult(status="ok"),
        trials=(
            InterventionTrial(
                trial_id="trial_stored_original",
                trial_kind="stored_original",
                outputs={"action_ref": "array://stored-original"},
            ),
            InterventionTrial(
                trial_id="trial_intervention",
                trial_kind="intervention",
                outputs={"action_ref": "array://intervened"},
            ),
        ),
        outcomes=(
            ActionOutcomeResult(
                basis="raw",
                horizon="full_chunk",
                baseline_trial_id="trial_stored_original",
                intervention_trial_id="trial_intervention",
                action_ref_baseline="array://stored-original",
                action_ref_intervened="array://intervened",
                delta_ref="array://delta",
                metrics={"raw_delta_norm": 0.2, "gripper_mean_delta": 0.1},
            ).to_dict(),
        ),
        outputs=("array://stored-original", "array://intervened", "array://delta"),
        display={"summary": "Adding the probe direction changed the action chunk."},
        claim={"claim_strength": ["causal_local", "action_level"]},
        provenance={
            "schema_kind": "vla_lens.intervention_run",
            "schema_version": "0.1.0",
            "dataset_id": "demo",
            "dataset_fingerprint": "fingerprint-1",
            "trace_id": "trace-1",
            "episode_id": "episode-1",
            "policy_call_index": 7,
            "source_artifact_id": "probe-gripper-close",
        },
        created_utc="2026-06-06T00:00:00+00:00",
    )


def test_intervention_run_to_lens_artifact():
    artifact = intervention_run_to_lens_artifact(_run())

    assert artifact.artifact_type == "intervention_run"
    assert artifact.group_id == "run-artifact-index"
    assert artifact.selector["workbench_run_id"] == "run-artifact-index"
    assert artifact.selector["context"]["trace_id"] == "trace-1"
    assert artifact.selector["target"]["kind"] == "probe_direction"
    assert artifact.method["operator"]["operator"] == "add_direction"
    assert len(artifact.method["request_hash"]) == 16
    assert artifact.metrics["outcome_0"]["raw_delta_norm"] == 0.2
    assert artifact.arrays == {}
    assert artifact.display["kind"] == "intervention_card"
    assert artifact.display["workbench_run_id"] == "run-artifact-index"
    assert artifact.display["output_refs"] == [
        "array://stored-original",
        "array://intervened",
        "array://delta",
    ]
    assert {"intervention_run", "ok", "add_direction", "action", "causal_local"} <= set(
        artifact.tags
    )
    assert artifact.source_trace_ids == ("trace-1",)


def test_artifact_display_summary_has_context_target_outcome():
    display = intervention_run_to_lens_artifact(_run()).display

    assert display["context"]["policy_call_index"] == 7
    assert display["target"]["model_site"] == "pi05.expert.layers.12.hidden_tokens"
    assert display["outcome"]["kind"] == "action"
    assert display["outcome"]["summary"][0]["delta_ref"] == "array://delta"
    assert display["claim"]["claim_strength"] == ["causal_local", "action_level"]


def test_saved_typed_run_creates_listable_lens_artifact(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    run = _run()
    save_intervention_run(dataset, run.to_workbench_spec())
    saved = dataset.save_artifact(intervention_run_to_lens_artifact(run))
    reopened = TraceDataset.open(dataset.root)

    records = reopened.artifact_index.to_dict("records")
    intervention_records = [
        record for record in records if record["artifact_type"] == "intervention_run"
    ]
    artifact = reopened.load_artifact(saved.artifact_id)

    assert len(intervention_records) == 1
    assert intervention_records[0]["artifact_id"] == saved.artifact_id
    assert artifact.display["workbench_run_id"] == run.run_id
    assert artifact.selector["workbench_run_id"] == run.run_id
