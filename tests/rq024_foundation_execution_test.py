from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import vla_lens.rq024_foundation_execution as execution
from vla_lens.pi05.batch_capture import ExactTraceOutput
from vla_lens.research_io import file_sha256

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_read_only_plan_resolves_all_72_hash_bound_wrapper_commands():
    plan = execution.load_foundation_plan()
    payload = execution.plan_payload(plan)

    assert payload["read_only"] is True
    assert payload["resolved_trial_count"] == 72
    assert payload["selected_trial_count"] == 72
    assert len({trial.row.trial_id for trial in plan.trials}) == 72
    assert all(len(trial.command.expected_trace_ids) == 1 for trial in plan.trials)
    assert all(
        trial.execution_command[0].endswith("scripts/pi05_batch_capture.sh")
        and "--validate-exact" in trial.execution_command
        and "--trial-id" in trial.execution_command
        and "--skip-plan-write" in trial.execution_command
        and "--force" not in trial.execution_command
        for trial in plan.trials
    )


def test_plan_selection_uses_ledger_completion_and_retry_history():
    plan = execution.load_foundation_plan()
    first, second = plan.trials[:2]
    state = SimpleNamespace(
        closed_attempts={
            "completed": {
                "completed": True,
                "started": {"trial_id": first.row.trial_id},
            },
            "failed": {
                "completed": False,
                "started": {"trial_id": second.row.trial_id},
            },
        },
        open_attempts={},
    )

    remaining = execution.select_trials(plan, state, max_trials=1)

    assert remaining[0].row.trial_id == second.row.trial_id
    assert execution._next_ordinal(state, second.row.trial_id) == 2
    assert execution.select_trials(plan, state, trial_id=first.row.trial_id) == ()


def test_resolved_plan_is_written_once_and_reused(tmp_path, monkeypatch):
    monkeypatch.delenv("VLA_LENS_CAPTURE_ENV_RECEIPT", raising=False)
    plan = replace(execution.load_foundation_plan(), output_root=tmp_path / "output")

    execution._ensure_resolved_plan(plan)
    first_hashes = {
        path.name: file_sha256(path) for path in plan.output_root.iterdir() if path.is_file()
    }
    execution._ensure_resolved_plan(plan)
    second_hashes = {
        path.name: file_sha256(path) for path in plan.output_root.iterdir() if path.is_file()
    }

    assert first_hashes == second_hashes
    assert len(json.loads((plan.output_root / "episode_plan.json").read_text())) == 72


def test_execution_appends_started_before_launch_and_completed_after_validation(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "output"
    event_root = tmp_path / "events"
    plan = replace(execution.load_foundation_plan(), output_root=output_root)
    trial = plan.trials[0]
    state = _authorized_state(plan)
    check = SimpleNamespace(state=state, valid=True, issues=())
    operations: list[str] = []
    sequence = 1

    def append_event(root, program, **kwargs):
        nonlocal sequence
        operations.append(f"append:{kwargs['event_type']}")
        sequence += 1
        path = root / f"{sequence:06d}-{kwargs['event_id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "sequence": sequence,
                    "event_id": kwargs["event_id"],
                    "payload": kwargs["payload"],
                }
            ),
            encoding="utf-8",
        )
        return path, "sha256:" + f"{sequence:064x}"

    def run_trial(command, repo_root):
        operations.append("run")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    def validate(command, row, *, expected_runtime):
        operations.append("validate")
        return ExactTraceOutput(
            trace_id=row.expected_trace_id,
            terminal_status="rollout_success",
            model_calls=2,
            action_generations=2,
            simulator_steps=10,
            output_bytes=100,
            files=(),
        )

    monkeypatch.setattr(execution, "_authorized_ledger", lambda *args, **kwargs: check)
    monkeypatch.setattr(execution, "_verified_ledger", lambda *args: check)
    monkeypatch.setattr(execution, "append_research_event", append_event)
    monkeypatch.setattr(execution, "_run_trial", run_trial)
    monkeypatch.setattr(execution, "validate_exact_trace_output", validate)

    completed = execution.execute_foundation(
        plan,
        event_root=event_root,
        actor_id="test-runner",
        repo_root=tmp_path,
        trial_id=trial.row.trial_id,
    )

    assert completed == [trial.row.trial_id]
    assert operations == [
        "append:trial_attempt_started",
        "run",
        "validate",
        "append:trial_attempt_completed",
    ]


def test_execution_records_validation_failure_without_model_or_simulator_work(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "output"
    event_root = tmp_path / "events"
    plan = replace(execution.load_foundation_plan(), output_root=output_root)
    trial = plan.trials[0]
    state = _authorized_state(plan)
    check = SimpleNamespace(state=state, valid=True, issues=())
    event_types: list[str] = []
    sequence = 1

    def append_event(root, program, **kwargs):
        nonlocal sequence
        event_types.append(kwargs["event_type"])
        sequence += 1
        path = root / f"{sequence:06d}-{kwargs['event_id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "sequence": sequence,
                    "event_id": kwargs["event_id"],
                    "payload": kwargs["payload"],
                }
            ),
            encoding="utf-8",
        )
        return path, "sha256:" + f"{sequence:064x}"

    monkeypatch.setattr(execution, "_authorized_ledger", lambda *args, **kwargs: check)
    monkeypatch.setattr(execution, "_verified_ledger", lambda *args: check)
    monkeypatch.setattr(execution, "append_research_event", append_event)
    monkeypatch.setattr(
        execution,
        "_run_trial",
        lambda command, repo_root: subprocess.CompletedProcess(command, 0, "", ""),
    )
    monkeypatch.setattr(
        execution,
        "validate_exact_trace_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("wrong trace")),
    )

    completed = execution.execute_foundation(
        plan,
        event_root=event_root,
        actor_id="test-runner",
        repo_root=tmp_path,
        trial_id=trial.row.trial_id,
    )

    assert completed == []
    assert event_types == ["trial_attempt_started", "trial_attempt_failed"]


def test_force_is_explicitly_rejected():
    with pytest.raises(SystemExit, match="--force is forbidden"):
        execution.main(["--run", "--event-root", "unused", "--force"])


def _authorized_state(plan):
    return SimpleNamespace(
        locked={
            execution.STUDY_ID: {
                "event_id": "foundation-lock",
                "event_sha256": "sha256:" + "a" * 64,
                "reservation_id": "foundation-reservation",
                "child_plan_fingerprint": plan.child_fingerprint,
                "lock_receipt_ref": {"content_fingerprint": "sha256:" + "b" * 64},
            }
        },
        events_by_id={"foundation-lock": {"sequence": 1, "event_id": "foundation-lock"}},
        event_hashes={"foundation-lock": "sha256:" + "a" * 64},
        open_attempts={},
        closed_attempts={},
    )
