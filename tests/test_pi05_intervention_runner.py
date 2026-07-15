from __future__ import annotations

import json
from typing import Any, Mapping

import numpy as np
import pytest

from vla_lens.interventions import RuntimeTrialOutput
from vla_lens.pi05.intervention_runner import parse_args, run_job
from vla_lens.synthetic import create_synthetic_trace_dataset
from vla_lens.traces import TraceDataset


class FakeRunnerExecutor:
    def __init__(self, base_action: np.ndarray, *, noop_offset: float = 0.0):
        self.base_action = np.asarray(base_action, dtype=np.float32)
        self.noop_offset = float(noop_offset)
        self.noop_calls = 0
        self.intervention_calls = 0
        self.closed = False

    def run_noop(self, request: Mapping[str, Any]) -> RuntimeTrialOutput:
        del request
        self.noop_calls += 1
        return RuntimeTrialOutput(
            trial_id="trial_noop",
            trial_kind="noop_rerun",
            action_chunk=self.base_action + self.noop_offset,
        )

    def run_intervention(self, request: Mapping[str, Any]) -> RuntimeTrialOutput:
        del request
        self.intervention_calls += 1
        return RuntimeTrialOutput(
            trial_id="trial_intervention",
            trial_kind="intervention",
            action_chunk=self.base_action + 0.1,
            runtime={"purpose": "hook_wiring_smoke", "claim_eligible": False},
        )

    def run_control(
        self,
        request: Mapping[str, Any],
        *,
        control_kind: str,
    ) -> RuntimeTrialOutput:
        del request
        return RuntimeTrialOutput(
            trial_id=f"trial_{control_kind}",
            trial_kind=control_kind,
            control_kind=control_kind,
            action_chunk=self.base_action,
            runtime={"purpose": "hook_wiring_smoke", "claim_eligible": False},
        )

    def close(self) -> None:
        self.closed = True


def _write_request(dataset, path) -> None:
    bundle = dataset.bundles[0]
    payload = {
        "runtime_adapter": "pi05",
        "target": {
            "kind": "manual",
            "model_family": "pi05",
            "model_site": "action_head.layers.0.resid",
            "token_space": "synthetic.action_suffix",
        },
        "baseline": {
            "context": {
                "trace_id": bundle.manifest.trace_id,
                "policy_call_index": 0,
            }
        },
        "intervention": {
            "request": {
                "operator": {
                    "operator": "add_direction",
                    "strength": 0.01,
                    "parameters": {"mode": "synthetic_hook_smoke", "dimension": 0},
                },
                "schedule": {
                    "policy_calls": [0],
                    "generation_steps": "all",
                    "tokens": "action",
                },
                "outcome": {"kind": "action", "basis": ["raw"]},
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _pi05ish_synthetic_dataset(path) -> TraceDataset:
    dataset = create_synthetic_trace_dataset(path, num_episodes=1, timesteps=8)
    manifest_path = path / "vla_lens" / "episodes" / "episode_000000" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task_id"] = "0"
    manifest["env_id"] = "libero_object"
    manifest["metadata"]["environment"] = {
        "benchmark": "libero_object",
        "task_id": 0,
        "seed": 0,
        "obs_size": 256,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return TraceDataset.open(dataset.root)


def _args(dataset, request_path, output_path, *extra: str):
    return parse_args(
        [
            str(dataset.root),
            "--request",
            str(request_path),
            "--output",
            str(output_path),
            *extra,
        ]
    )


def test_intervention_runner_dry_run_stays_runtime_free_and_writes_report(tmp_path):
    dataset = _pi05ish_synthetic_dataset(tmp_path / "demo")
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "report.json"
    _write_request(dataset, request_path)

    def forbidden_factory(*args, **kwargs):
        del args, kwargs
        raise AssertionError("dry-run must not load the PI0.5 executor")

    report, exit_code = run_job(
        _args(dataset, request_path, output_path, "--dry-run"),
        executor_factory=forbidden_factory,
    )

    assert exit_code == 0
    assert report["status"] == "inspected"
    assert report["preflight"]["status"] == "inspected_only"
    assert report["replay_inputs"]["initial_noise"]["exactness"] == "exact"
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "inspected"


def test_intervention_runner_measures_repeated_noop_without_intervening(tmp_path):
    dataset = _pi05ish_synthetic_dataset(tmp_path / "demo")
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "report.json"
    _write_request(dataset, request_path)
    stored = np.asarray(dataset.bundles[0].action_chunks()[0], dtype=np.float32)
    executor = FakeRunnerExecutor(stored)

    report, exit_code = run_job(
        _args(dataset, request_path, output_path, "--noop-repeats", "3"),
        executor_factory=lambda *args, **kwargs: executor,
    )

    assert exit_code == 0
    assert report["status"] == "replay_measured"
    assert report["noop_replay"]["deterministic_across_repeats"] is True
    assert all(
        trial["delta_from_stored"]["exact_match"]
        for trial in report["noop_replay"]["trials"]
    )
    assert executor.noop_calls == 3
    assert executor.intervention_calls == 0
    assert executor.closed is True


def test_intervention_runner_blocks_hook_when_noop_exceeds_tolerance(tmp_path):
    dataset = _pi05ish_synthetic_dataset(tmp_path / "demo")
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "report.json"
    _write_request(dataset, request_path)
    stored = np.asarray(dataset.bundles[0].action_chunks()[0], dtype=np.float32)
    executor = FakeRunnerExecutor(stored, noop_offset=0.1)

    report, exit_code = run_job(
        _args(
            dataset,
            request_path,
            output_path,
            "--run-intervention",
            "--max-noop-l2",
            "0",
            "--max-noop-max-abs",
            "0",
        ),
        executor_factory=lambda *args, **kwargs: executor,
    )

    assert exit_code == 3
    assert report["status"] == "blocked_by_replay_gate"
    assert report["intervention_gate"]["passed"] is False
    assert executor.intervention_calls == 0


def test_intervention_runner_rejects_non_finite_tolerance(tmp_path):
    dataset = _pi05ish_synthetic_dataset(tmp_path / "demo")
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "report.json"
    _write_request(dataset, request_path)

    with pytest.raises(ValueError, match="finite non-negative"):
        run_job(
            _args(dataset, request_path, output_path, "--max-noop-l2", "nan"),
        )


def test_intervention_runner_rejects_non_finite_noop_action(tmp_path):
    dataset = _pi05ish_synthetic_dataset(tmp_path / "demo")
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "report.json"
    _write_request(dataset, request_path)
    stored = np.asarray(dataset.bundles[0].action_chunks()[0], dtype=np.float32)
    executor = FakeRunnerExecutor(stored, noop_offset=np.nan)

    with pytest.raises(ValueError, match="No-op replay action chunk"):
        run_job(
            _args(dataset, request_path, output_path),
            executor_factory=lambda *args, **kwargs: executor,
        )

    assert executor.intervention_calls == 0
    assert executor.closed is True


def test_intervention_runner_runs_non_claiming_hook_after_exact_replay(tmp_path):
    dataset = _pi05ish_synthetic_dataset(tmp_path / "demo")
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "report.json"
    _write_request(dataset, request_path)
    stored = np.asarray(dataset.bundles[0].action_chunks()[0], dtype=np.float32)
    executor = FakeRunnerExecutor(stored)

    report, exit_code = run_job(
        _args(
            dataset,
            request_path,
            output_path,
            "--run-intervention",
            "--max-noop-l2",
            "0",
            "--max-noop-max-abs",
            "0",
            "--no-save",
        ),
        executor_factory=lambda *args, **kwargs: executor,
    )

    assert exit_code == 0
    assert report["status"] == "completed"
    assert report["claim_eligible"] is False
    assert report["intervention_result"]["run"]["claim"] == {"claim_strength": []}
    assert executor.intervention_calls == 1
