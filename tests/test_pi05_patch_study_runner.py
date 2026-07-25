from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from vla_lens.interventions import RuntimePreflightResult, RuntimeTrialOutput
from vla_lens.pi05.patch_study_runner import parse_args, run_patch_study_job
from vla_lens.synthetic import create_synthetic_trace_dataset
from vla_lens.traces import TraceDataset


class FakePatchExecutor:
    def __init__(self, base_action: np.ndarray, runtime: Any):
        self.base_action = np.asarray(base_action, dtype=np.float32)
        self.runtime = runtime
        self.replay_inputs = SimpleNamespace(
            stored_action_chunk=self.base_action,
            initial_noise_exactness="exact",
        )
        self.noop_calls = 0
        self.intervention_calls = 0
        self.primed_layers: list[int] = []
        self.primed_sites: list[str] = []

    def run_noop(self, request: Mapping[str, Any]) -> RuntimeTrialOutput:
        del request
        self.noop_calls += 1
        return RuntimeTrialOutput(
            trial_id="trial_noop",
            trial_kind="noop_rerun",
            action_chunk=self.base_action,
            runtime={"shared_noise_ref": "flow_initial_noise[0]"},
        )

    def prime_donor_cache(self, layers):
        self.primed_layers = list(layers)
        return self.base_action + 1.0

    def prime_donor_sites(self, model_sites):
        self.primed_sites = list(model_sites)
        return self.base_action + 1.0

    def run_intervention(self, request: Mapping[str, Any]) -> RuntimeTrialOutput:
        del request
        self.intervention_calls += 1
        return RuntimeTrialOutput(
            trial_id="trial_source_patch",
            trial_kind="intervention",
            action_chunk=self.base_action + 0.6,
            array_outputs={"donor_shared_noise": self.base_action + 1.0},
            runtime={
                "purpose": "donor_source_patch",
                "claim_eligible": True,
                "hook_calls": 1,
                "shared_noise_ref": "flow_initial_noise[0]",
                "recipient_token_indices": [1, 2],
                "token_mapping_sha256": "mapping",
                "pair_compatibility": {
                    "different_trace": True,
                    "model_id": True,
                    "prompt": True,
                    "benchmark": True,
                    "task_id": True,
                    "observation_shape": True,
                    "stored_action_shape": True,
                    "noise_shape": True,
                },
            },
        )

    def run_control(
        self,
        request: Mapping[str, Any],
        *,
        control_kind: str,
    ) -> RuntimeTrialOutput:
        del request
        offset = 0.1 if control_kind == "wrong_region" else 0.05
        return RuntimeTrialOutput(
            trial_id=f"trial_{control_kind}",
            trial_kind="source_patch_control",
            control_kind=control_kind,
            action_chunk=self.base_action + offset,
            runtime={
                "claim_eligible": True,
                "hook_calls": 1,
                "shared_noise_ref": "flow_initial_noise[0]",
                "recipient_token_indices": [3, 4],
            },
        )

    def close(self) -> None:
        return None


def _dataset(path) -> TraceDataset:
    dataset = create_synthetic_trace_dataset(path, num_episodes=2, timesteps=8)
    manifests = sorted((path / "vla_lens" / "episodes").glob("*/manifest.json"))
    recipient = json.loads(manifests[0].read_text(encoding="utf-8"))
    donor = json.loads(manifests[1].read_text(encoding="utf-8"))
    donor["prompt"] = recipient["prompt"]
    donor["task_id"] = recipient["task_id"]
    donor["model_id"] = recipient["model_id"]
    manifests[1].write_text(json.dumps(donor), encoding="utf-8")
    return TraceDataset.open(dataset.root)


def _job(dataset: TraceDataset) -> dict:
    recipient, donor = dataset.bundles
    pair_id = "pair-1"
    return {
        "study": {
            "study_id": "rq020-runner-test",
            "question": "Where does patching transfer the donor action?",
            "hypothesis": "One layer transfers the donor-directed action change.",
            "pair_ids": [pair_id],
            "sites": [{"layer": 0}, {"layer": 4}],
            "controls": ["wrong_region", "random_matched_norm"],
            "shared_noise_refs": ["flow_initial_noise[0]"],
            "axes": {
                "token_regions": ["target"],
                "wrong_region": "wrong",
            },
        },
        "pairs": [
            {
                "pair_id": pair_id,
                "recipe": {
                    "kind": "pose_exchange",
                    "target_object": "caddy",
                    "distractor_object": "mug",
                    "changed_variables": ["caddy.pose", "mug.pose"],
                    "held_fixed": {"prompt": True, "robot": True},
                },
                "recipient": {
                    "trace": {"trace_id": recipient.manifest.trace_id},
                    "policy_call": {
                        "trace_id": recipient.manifest.trace_id,
                        "policy_call_index": 0,
                    },
                },
                "donor": {
                    "trace": {"trace_id": donor.manifest.trace_id},
                    "policy_call": {
                        "trace_id": donor.manifest.trace_id,
                        "policy_call_index": 0,
                    },
                },
                "validation": {
                    "token_regions": {
                        "target": [1, 2],
                        "wrong": [3, 4],
                    }
                },
            }
        ],
        "request_template": {
            "runtime_adapter": "pi05",
            "target": {
                "kind": "activation_slice",
                "model_family": "pi05",
                "model_site": "placeholder",
                "token_space": "pi05.prefix",
            },
            "intervention": {
                "request": {
                    "operator": {
                        "operator": "source_patch",
                        "strength": 1.0,
                        "parameters": {"mode": "donor_source_patch"},
                    },
                    "schedule": {
                        "policy_calls": [0],
                        "generation_steps": "all",
                        "tokens": "target_tokens",
                    },
                    "controls": [
                        {"kind": "random_matched_norm", "parameters": {"seed": 7}}
                    ],
                    "outcome": {"kind": "action", "basis": ["raw"]},
                }
            },
        },
    }


def test_patch_study_runner_reuses_model_noop_and_donor_cache_then_resumes(
    tmp_path,
    monkeypatch,
):
    dataset = _dataset(tmp_path / "dataset")
    job_path = tmp_path / "study.json"
    job_path.write_text(json.dumps(_job(dataset)), encoding="utf-8")
    output_dir = tmp_path / "study-output"
    factory_calls = []
    executors = []

    def always_ok(*args, **kwargs):
        del args, kwargs
        return RuntimePreflightResult(status="ok", target_resolution={})

    monkeypatch.setattr(
        "vla_lens.pi05.patch_study_runner.pi05_intervention_preflight", always_ok
    )
    monkeypatch.setattr(
        "vla_lens.pi05.intervention_runtime.pi05_intervention_preflight", always_ok
    )

    def factory(dataset_arg, payload, *, runtime=None, **kwargs):
        del payload, kwargs
        factory_calls.append(runtime)
        resolved_runtime = runtime or object()
        context = _job(dataset_arg)["pairs"][0]["recipient"]["trace"]["trace_id"]
        base = np.asarray(dataset_arg.bundle(context).action_chunks()[0], dtype=np.float32)
        executor = FakePatchExecutor(base, resolved_runtime)
        executors.append(executor)
        return executor

    args = parse_args(
        [
            str(dataset.root),
            "--study",
            str(job_path),
            "--output-dir",
            str(output_dir),
            "--run-study",
            "--max-noop-l2",
            "0",
            "--max-noop-max-abs",
            "0",
            "--no-workbench",
        ]
    )

    report, exit_code = run_patch_study_job(args, executor_factory=factory)

    assert exit_code == 0
    assert report["status"] == "completed"
    assert report["completed_trial_count"] == 2
    assert factory_calls == [None]
    assert executors[0].noop_calls == 3
    assert executors[0].intervention_calls == 2
    assert executors[0].primed_layers == []
    assert executors[0].primed_sites == [
        "pi05.vlm.layers.0.prefix.hidden_tokens",
        "pi05.vlm.layers.4.prefix.hidden_tokens",
    ]
    assert (output_dir / "actions.zarr").exists()
    assert (output_dir / "artifact.json").exists()

    factory_calls.clear()
    resumed, resumed_exit = run_patch_study_job(args, executor_factory=factory)
    assert resumed_exit == 0
    assert resumed["completed_trial_count"] == 2
    assert factory_calls == []


def test_patch_study_runner_inspection_never_constructs_hardware_executor(
    tmp_path,
    monkeypatch,
):
    dataset = _dataset(tmp_path / "dataset")
    job_path = tmp_path / "study.json"
    job_path.write_text(json.dumps(_job(dataset)), encoding="utf-8")

    monkeypatch.setattr(
        "vla_lens.pi05.patch_study_runner.pi05_intervention_preflight",
        lambda *args, **kwargs: RuntimePreflightResult(status="ok"),
    )

    report, exit_code = run_patch_study_job(
        parse_args([str(dataset.root), "--study", str(job_path)]),
        executor_factory=lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("inspection must not load hardware")
        ),
    )

    assert exit_code == 0
    assert report["status"] == "inspected"
    assert report["planned_trial_count"] == 2
