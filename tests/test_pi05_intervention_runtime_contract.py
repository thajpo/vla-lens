from __future__ import annotations

import subprocess
import sys
from typing import Any, Mapping

import numpy as np

from vla_lens.interventions import RuntimeTrialOutput
from vla_lens.pi05.intervention_preflight import pi05_intervention_preflight
from vla_lens.pi05.intervention_runtime import run_pi05_intervention
from vla_lens.synthetic import create_synthetic_trace_dataset
from vla_lens.traces import TraceDataset
from vla_lens.workbench import list_intervention_runs


class FakePI05Executor:
    def __init__(self, base_action: np.ndarray, *, claim_eligible: bool = True):
        self.base_action = np.asarray(base_action, dtype=np.float32)
        self.claim_eligible = claim_eligible

    def run_noop(self, request: Mapping[str, Any]) -> RuntimeTrialOutput:
        del request
        return RuntimeTrialOutput(
            trial_id="trial_noop",
            trial_kind="noop_rerun",
            action_chunk=self.base_action.copy(),
            runtime={"executor": "fake_pi05", "claim_eligible": self.claim_eligible},
        )

    def run_intervention(self, request: Mapping[str, Any]) -> RuntimeTrialOutput:
        del request
        action = self.base_action.copy()
        action[:, 6] += 0.5
        action[:, 0] += 0.25
        return RuntimeTrialOutput(
            trial_id="trial_intervention",
            trial_kind="intervention",
            action_chunk=action,
            metrics={"strength": 1.0},
            runtime={"executor": "fake_pi05"},
        )

    def run_control(
        self,
        request: Mapping[str, Any],
        *,
        control_kind: str,
    ) -> RuntimeTrialOutput:
        del request
        action = self.base_action.copy()
        action[:, :] += 0.05
        return RuntimeTrialOutput(
            trial_id=f"trial_{control_kind}",
            trial_kind=control_kind,
            control_kind="random_direction",
            action_chunk=action,
            runtime={"executor": "fake_pi05"},
        )


def _probe_artifact_id(dataset: TraceDataset) -> str:
    rows = dataset.artifact_index
    matches = rows.loc[rows["artifact_type"].astype(str) == "probe_suite"]
    assert not matches.empty
    return str(matches.iloc[0]["artifact_id"])


def _request(dataset: TraceDataset) -> dict[str, Any]:
    trace_id = dataset.bundles[0].manifest.trace_id
    return {
        "run_id": "pi05-runtime-contract",
        "runtime_adapter": "pi05",
        "title": "PI0.5 fake runtime contract",
        "target": {
            "kind": "probe_direction",
            "source_artifact_id": _probe_artifact_id(dataset),
            "source_artifact_type": "probe_suite",
            "model_id": "openpi/pi05-test",
            "model_family": "pi05",
            "model_site": "action_head.layers.0.resid",
            "token_space": "synthetic.action_suffix",
            "metadata": {"intended_basis": "gripper"},
        },
        "baseline": {
            "context": {
                "trace_id": trace_id,
                "policy_call_index": 0,
            }
        },
        "intervention": {
            "request": {
                "operator": {"operator": "add_direction", "strength": 1.0},
                "schedule": {"policy_calls": [0], "tokens": "action"},
                "outcome": {
                    "kind": "action",
                    "basis": ["raw", "gripper"],
                    "intended_basis": "gripper",
                },
                "controls": [{"kind": "random_direction"}],
            }
        },
    }


def test_pi05_preflight_replaces_runtime_availability_without_heavy_imports(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=8)
    request = _request(dataset)

    unavailable = pi05_intervention_preflight(dataset, request, runtime_available=False).to_dict()
    available = pi05_intervention_preflight(dataset, request, runtime_available=True).to_dict()

    assert unavailable["status"] == "inspected_only"
    assert unavailable["capability_status"]["model_runtime_available"] is False
    assert available["status"] == "ok"
    assert available["capability_status"]["model_runtime_available"] is True
    assert available["runtime_resolution"]["adapter"] == "pi05"


def test_pi05_runtime_contract_writes_saved_intervention_run_and_artifact(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=8)
    stored = np.asarray(dataset.bundles[0].action_chunks(mmap=True)[0], dtype=np.float32)

    result = run_pi05_intervention(
        dataset,
        _request(dataset),
        executor=FakePI05Executor(stored),
    )
    reopened = TraceDataset.open(dataset.root)
    saved_runs = list_intervention_runs(reopened)
    artifact = reopened.load_artifact(str(result.artifact_id))
    intervened = reopened.load_artifact_array(artifact, "intervened")

    assert result.run.status == "ok"
    assert result.run.runtime_resolution is not None
    assert result.run.runtime_resolution.adapter == "pi05"
    assert {trial.trial_kind for trial in result.run.trials} >= {
        "stored_original",
        "noop_rerun",
        "intervention",
        "random_direction_control",
    }
    assert result.run.outcomes
    assert result.run.outcomes[0]["metrics"]["side_effect_score"] > 0.0
    assert result.run.controls[0]["control_kind"] == "random_direction"
    trials = {trial.trial_kind: trial for trial in result.run.trials}
    assert trials["stored_original"].outputs["action_ref"] == "stored_original"
    assert trials["noop_rerun"].outputs["action_ref"] == "noop"
    assert trials["intervention"].outputs["action_ref"] == "intervened"
    assert trials["random_direction_control"].outputs["action_ref"] == (
        "control_random_direction_control"
    )
    assert all(
        trial.outputs["action_ref"] in result.arrays
        for trial in result.run.trials
    )
    assert [run.run_id for run in saved_runs] == ["pi05-runtime-contract"]
    saved_action_refs = {
        trial["outputs"]["action_ref"]
        for trial in saved_runs[0].readouts["trials"]
    }
    assert saved_action_refs <= set(saved_runs[0].outputs)
    assert artifact.artifact_type == "intervention_run"
    assert set(artifact.arrays) == set(result.arrays)
    assert intervened.shape == stored.shape


def test_pi05_runtime_does_not_assign_causal_claim_to_engineering_hook_smoke(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=8)
    stored = np.asarray(dataset.bundles[0].action_chunks(mmap=True)[0], dtype=np.float32)

    result = run_pi05_intervention(
        dataset,
        _request(dataset),
        executor=FakePI05Executor(stored, claim_eligible=False),
        save=False,
    )

    assert result.run.status == "ok"
    assert result.run.claim == {"claim_strength": []}


def test_pi05_runtime_requires_replay_and_all_specificity_controls_for_method_eligibility(
    tmp_path,
):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=8)
    stored = np.asarray(dataset.bundles[0].action_chunks(mmap=True)[0], dtype=np.float32)
    request = _request(dataset)
    request["intervention"]["request"]["controls"] = [
        {"kind": "random_direction"},
        {"kind": "wrong_feature"},
        {"kind": "wrong_token"},
    ]

    class ArtifactExecutor(FakePI05Executor):
        def run_intervention(self, payload):
            output = super().run_intervention(payload)
            return RuntimeTrialOutput(
                trial_id=output.trial_id,
                trial_kind=output.trial_kind,
                action_chunk=output.action_chunk,
                metrics=output.metrics,
                runtime={
                    "purpose": "artifact_probe_direction",
                    "claim_eligible": True,
                },
            )

        def run_control(self, payload, *, control_kind):
            del payload
            resolved = {
                "random_direction_control": "matched_random",
                "wrong_feature": "wrong_identity",
                "wrong_token": "wrong_roi",
            }[control_kind]
            return RuntimeTrialOutput(
                trial_id=f"trial_{resolved}",
                trial_kind={
                    "matched_random": "random_direction_control",
                    "wrong_identity": "control",
                    "wrong_roi": "wrong_token_control",
                }[resolved],
                control_kind=resolved,
                action_chunk=self.base_action + 0.05,
                runtime={"claim_eligible": True},
            )

    eligible = run_pi05_intervention(
        dataset,
        request,
        executor=ArtifactExecutor(stored),
        save=False,
        claim_gate={"passed": True, "thresholds": {"max_noop_l2": 0.0}},
    )
    replay_blocked = run_pi05_intervention(
        dataset,
        request,
        executor=ArtifactExecutor(stored),
        save=False,
        claim_gate={"passed": False},
    )

    assert eligible.run.claim["method_eligible"] is True
    assert eligible.run.claim["scientific_verdict"] == "not_evaluated_from_execution_alone"
    assert eligible.run.claim["claim_strength"] == ["causal_local", "action_level"]
    assert replay_blocked.run.claim["method_eligible"] is False
    assert replay_blocked.run.claim["claim_strength"] == []
    assert len(eligible.run.display["specificity_summary"]["controls"]) == 3


def test_pi05_intervention_runtime_import_does_not_load_heavy_dependencies():
    code = """
import sys
import vla_lens.interventions.runtime
import vla_lens.pi05.intervention_preflight
import vla_lens.pi05.intervention_runtime
banned = {"torch", "lerobot", "libero", "robosuite"}
loaded = sorted(name for name in banned if name in sys.modules)
if loaded:
    raise SystemExit("loaded heavy modules: " + ", ".join(loaded))
"""

    subprocess.run([sys.executable, "-c", code], check=True)
