"""Append-only execution driver for the locked RQ-024 FOUNDATION trial plan."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import re
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from vla_lens.pi05.batch_capture import (
    CaptureCommand,
    EpisodePlanRow,
    ExactTraceOutput,
    _capture_commands,
    _expected_trace_exists,
    _load_config,
    _read_episode_plan,
    _validate_batch_config,
    _validate_episode_rows,
    _write_plan_files,
    validate_exact_trace_output,
)
from vla_lens.pi05.runtime_identity import canonical_sha256
from vla_lens.research_child import child_plan_fingerprint, load_research_child
from vla_lens.research_events import append_research_event, verify_research_event_ledger
from vla_lens.research_io import (
    canonical_research_fingerprint,
    file_sha256,
    load_research_mapping,
    write_bytes_create_only,
)
from vla_lens.research_plan import load_research_plan, research_plan_fingerprint
from vla_lens.research_state import BUDGET_FIELDS, campaign_status, child_authorization_issues

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROGRAM = Path("configs/campaigns/rq024_controlled_scene_to_behavior.yaml")
DEFAULT_CHILD = Path("configs/campaigns/rq024/foundation-r1/child.yaml")
STUDY_ID = "FOUNDATION"


@dataclass(frozen=True, slots=True)
class FoundationTrial:
    row: EpisodePlanRow
    record: Mapping[str, str]
    command: CaptureCommand
    execution_command: tuple[str, ...]
    row_fingerprint: str
    seed_fingerprint: str


@dataclass(frozen=True, slots=True)
class FoundationPlan:
    program: Mapping[str, Any]
    child: Mapping[str, Any]
    config: Mapping[str, Any]
    child_fingerprint: str
    runtime_fingerprint: str
    expected_runtime: Mapping[str, Any]
    output_root: Path
    trials: tuple[FoundationTrial, ...]


def load_foundation_plan(
    *,
    repo_root: Path = REPO_ROOT,
    program_path: Path = DEFAULT_PROGRAM,
    child_path: Path = DEFAULT_CHILD,
) -> FoundationPlan:
    """Load and hash every locked execution input without writing files."""

    repo_root = repo_root.resolve()
    program_file = _repo_path(repo_root, program_path)
    child_file = _repo_path(repo_root, child_path)
    program = load_research_plan(program_file)
    child = load_research_child(child_file)
    child_root = child_file.parent

    if child.get("child_plan_id") != "rq024-foundation-r1":
        raise ValueError("driver only accepts child rq024-foundation-r1")
    if child.get("study", {}).get("id") != STUDY_ID:
        raise ValueError("driver only accepts the FOUNDATION study")
    if child.get("program", {}).get("path") != program_path.as_posix():
        raise ValueError("child names a different parent program path")
    if child.get("program", {}).get("fingerprint") != research_plan_fingerprint(program):
        raise ValueError("parent program bytes differ from the locked child")

    trial_ref = child["trials"]["manifest"]
    trial_path = _repo_path(repo_root, Path(str(trial_ref["path"])))
    _require_file_hash(trial_path, str(trial_ref["sha256"]), "trial manifest")
    rows = _read_episode_plan(trial_path, exact=True)
    expected_count = int(child["trials"]["expected_count"])
    if len(rows) != expected_count or expected_count != 72:
        raise ValueError(f"locked FOUNDATION plan must contain 72 rows, found {len(rows)}")
    _validate_episode_rows(rows, exact=True)
    records = _csv_records(trial_path)

    config_ref = child["runtime"]["runner"]["config"]
    config_path = _repo_path(repo_root, Path(str(config_ref["path"])))
    _require_file_hash(config_path, str(config_ref["sha256"]), "runner config")
    config = _load_config(config_path)
    _validate_batch_config(config, exact=True, has_episode_plan=True)
    _validate_runner_contract(child, config, program_path, trial_path.relative_to(repo_root))

    environment_ref = child["runtime"]["environment"]["package_receipt"]
    environment_path = _repo_path(repo_root, Path(str(environment_ref["path"])))
    _require_file_hash(environment_path, str(environment_ref["sha256"]), "environment receipt")
    environment = load_research_mapping(environment_path)
    _validate_environment(child, environment)

    checkpoint = load_research_mapping(child_root / "checkpoint.json")
    _validate_checkpoint(child, checkpoint)
    expected_runtime = _expected_runtime(child, environment)
    output_root = Path(str(config["output_root"])).expanduser()
    commands = _capture_commands(config, output_root, rows, exact=True)
    by_trial = {command.expected_trace_ids[0]: command for command in commands}
    if len(commands) != expected_count or any(
        len(command.expected_trace_ids) != 1 for command in commands
    ):
        raise ValueError("exact FOUNDATION config must resolve to one command per trial row")

    trials: list[FoundationTrial] = []
    for row, record in zip(rows, records, strict=True):
        command = by_trial.get(row.expected_trace_id)
        if command is None:
            raise ValueError(f"no exact capture command resolved for trial {row.trial_id}")
        trials.append(
            FoundationTrial(
                row=row,
                record=record,
                command=command,
                execution_command=(
                    str(repo_root / str(child["runtime"]["runner"]["entrypoint"])),
                    *[str(value) for value in child["runtime"]["runner"]["argv"]],
                    "--trial-id",
                    row.trial_id,
                    "--skip-plan-write",
                ),
                row_fingerprint=canonical_research_fingerprint(record),
                seed_fingerprint=_seed_fingerprint(record),
            )
        )
    return FoundationPlan(
        program=program,
        child=child,
        config=config,
        child_fingerprint=child_plan_fingerprint(child),
        runtime_fingerprint=canonical_research_fingerprint(config),
        expected_runtime=expected_runtime,
        output_root=output_root,
        trials=tuple(trials),
    )


def execute_foundation(
    plan: FoundationPlan,
    *,
    event_root: Path,
    actor_id: str,
    repo_root: Path = REPO_ROOT,
    trial_id: str | None = None,
    max_trials: int | None = None,
) -> list[str]:
    """Execute selected incomplete rows while preserving one event per transition."""

    repo_root = repo_root.resolve()
    event_root = event_root.resolve()
    _require_within_repo(event_root, repo_root, "event root")
    if max_trials is not None and max_trials < 0:
        raise ValueError("--max-trials must be non-negative")

    _authorized_ledger(plan, event_root, repo_root, allow_open=True)
    with _executor_lock(event_root):
        check = _authorized_ledger(plan, event_root, repo_root, allow_open=True)
        _reconcile_open_attempt(plan, check.state, event_root, repo_root, actor_id)
        check = _authorized_ledger(plan, event_root, repo_root)
        _ensure_resolved_plan(plan)
        selected = select_trials(
            plan,
            check.state,
            trial_id=trial_id,
            max_trials=max_trials,
        )
        completed: list[str] = []
        for trial in selected:
            check = _authorized_ledger(plan, event_root, repo_root)
            if _expected_trace_exists(trial.command.output_root, trial.row.expected_trace_id):
                raise ValueError(
                    f"refusing unledgered existing output for {trial.row.trial_id}"
                )
            ordinal = _next_ordinal(check.state, trial.row.trial_id)
            attempt_id = f"{trial.row.trial_id}-a{ordinal}"
            requested = _requested_budget(plan.child)
            start_payload = {
                "child_lock_event": _state_event_ref(
                    check.state, check.state.locked[STUDY_ID]["event_id"]
                ),
                "reservation_id": check.state.locked[STUDY_ID]["reservation_id"],
                "trial_id": trial.row.trial_id,
                "attempt_id": attempt_id,
                "ordinal": ordinal,
                "trial_manifest_row_fingerprint": trial.row_fingerprint,
                "runtime_config_fingerprint": plan.runtime_fingerprint,
                "seed_bundle_fingerprint": trial.seed_fingerprint,
                "requested_budget": requested,
            }
            subject_fingerprint = canonical_research_fingerprint(start_payload)
            start_path, start_hash = append_research_event(
                event_root,
                plan.program,
                event_id=f"{attempt_id}-started",
                event_type="trial_attempt_started",
                actor_id=actor_id,
                subject_id=attempt_id,
                subject_fingerprint=subject_fingerprint,
                payload=start_payload,
                repo_root=repo_root,
                verify_artifacts=True,
            )
            start_event = load_research_mapping(start_path)
            start_ref = {
                "sequence": start_event["sequence"],
                "event_id": start_event["event_id"],
                "event_sha256": start_hash,
            }
            receipt_created_utc = str(start_event.get("created_utc") or attempt_id)
            result = _run_trial(trial.execution_command, repo_root)
            try:
                output = validate_exact_trace_output(
                    trial.command,
                    trial.row,
                    expected_runtime=plan.expected_runtime,
                )
            except (OSError, ValueError, KeyError) as exc:
                _append_failed_attempt(
                    plan,
                    event_root,
                    repo_root,
                    actor_id,
                    attempt_id,
                    subject_fingerprint,
                    start_ref,
                    result,
                    error=exc,
                )
                continue
            if result.returncode != 0:
                _append_failed_attempt(
                    plan,
                    event_root,
                    repo_root,
                    actor_id,
                    attempt_id,
                    subject_fingerprint,
                    start_ref,
                    result,
                    error=RuntimeError(f"capture subprocess exited {result.returncode}"),
                    output=output,
                )
                continue
            _append_completed_attempt(
                plan,
                event_root,
                repo_root,
                actor_id,
                attempt_id,
                subject_fingerprint,
                start_ref,
                trial,
                output,
                receipt_created_utc,
            )
            completed.append(trial.row.trial_id)
    return completed


def select_trials(
    plan: FoundationPlan,
    state: Any | None,
    *,
    trial_id: str | None = None,
    max_trials: int | None = None,
    remaining: bool = True,
) -> tuple[FoundationTrial, ...]:
    known = {trial.row.trial_id for trial in plan.trials}
    if trial_id is not None and trial_id not in known:
        raise ValueError(f"unknown FOUNDATION trial id: {trial_id}")
    completed = _completed_trial_ids(state) if state is not None else set()
    selected = [
        trial
        for trial in plan.trials
        if (trial_id is None or trial.row.trial_id == trial_id)
        and (not remaining or trial.row.trial_id not in completed)
    ]
    if max_trials is not None:
        if max_trials < 0:
            raise ValueError("--max-trials must be non-negative")
        selected = selected[:max_trials]
    return tuple(selected)


def plan_payload(
    plan: FoundationPlan,
    *,
    state: Any | None = None,
    trial_id: str | None = None,
    max_trials: int | None = None,
    remaining: bool = False,
) -> Mapping[str, Any]:
    selected = select_trials(
        plan,
        state,
        trial_id=trial_id,
        max_trials=max_trials,
        remaining=remaining,
    )
    return {
        "child_plan_id": plan.child["child_plan_id"],
        "child_fingerprint": plan.child_fingerprint,
        "runtime_config_fingerprint": plan.runtime_fingerprint,
        "resolved_trial_count": len(plan.trials),
        "selected_trial_count": len(selected),
        "read_only": True,
        "trials": [
            {
                "trial_id": trial.row.trial_id,
                "trial_manifest_row_fingerprint": trial.row_fingerprint,
                "seed_bundle_fingerprint": trial.seed_fingerprint,
                "expected_trace_id": trial.row.expected_trace_id,
                "command": list(trial.execution_command),
            }
            for trial in selected
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="Print the resolved plan; write nothing.")
    mode.add_argument(
        "--run", action="store_true", help="Execute reducer-authorized hardware trials."
    )
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--child", type=Path, default=DEFAULT_CHILD)
    parser.add_argument("--event-root", type=Path)
    parser.add_argument("--actor-id", default="rq024-foundation-driver")
    parser.add_argument("--trial-id")
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--remaining", action="store_true")
    parser.add_argument("--force", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.force:
        raise SystemExit("--force is forbidden for append-only FOUNDATION execution")
    plan = load_foundation_plan(program_path=args.program, child_path=args.child)
    if args.plan:
        state = None
        if args.remaining:
            if args.event_root is None:
                raise SystemExit("--remaining requires --event-root")
            state = _verified_ledger(plan, args.event_root.resolve(), REPO_ROOT).state
        print(
            json.dumps(
                plan_payload(
                    plan,
                    state=state,
                    trial_id=args.trial_id,
                    max_trials=args.max_trials,
                    remaining=args.remaining,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.event_root is None:
        raise SystemExit("--run requires --event-root")
    completed = execute_foundation(
        plan,
        event_root=args.event_root,
        actor_id=args.actor_id,
        trial_id=args.trial_id,
        max_trials=args.max_trials,
    )
    print(json.dumps({"completed_trial_ids": completed}, indent=2, sort_keys=True))


def _verified_ledger(plan: FoundationPlan, event_root: Path, repo_root: Path):
    check = verify_research_event_ledger(
        event_root,
        plan.program,
        repo_root=repo_root,
        verify_artifacts=True,
    )
    if not check.valid:
        codes = ", ".join(issue.code for issue in check.issues)
        raise ValueError(f"campaign event ledger is invalid: {codes}")
    return check


def _authorized_ledger(
    plan: FoundationPlan,
    event_root: Path,
    repo_root: Path,
    *,
    allow_open: bool = False,
):
    check = _verified_ledger(plan, event_root, repo_root)
    issues = child_authorization_issues(plan.child, check.state, plan.program)
    if issues:
        raise PermissionError(
            "locked child is not authorized: " + ", ".join(issue.code for issue in issues)
        )
    status = campaign_status(check.state, plan.program)
    if (
        status.get("hardware_authorized") is not True
        or status.get("next_action", {}).get("study_id") != STUDY_ID
        or status.get("next_action", {}).get("action_id")
        not in {"run_or_analyze_next_locked_trial", "finish_open_attempt"}
    ):
        raise PermissionError("reducer-derived status does not authorize FOUNDATION hardware")
    locked = check.state.locked.get(STUDY_ID) or {}
    if locked.get("child_plan_fingerprint") != plan.child_fingerprint:
        raise PermissionError("ledger locks different child bytes")
    _validate_attempt_history(plan, check.state)
    has_open_attempt = any(
        attempt.get("reservation_id") == locked.get("reservation_id")
        for attempt in check.state.open_attempts.values()
    )
    if has_open_attempt and not allow_open:
        raise PermissionError("an open FOUNDATION attempt must be reconciled before execution")
    return check


def _validate_attempt_history(plan: FoundationPlan, state: Any) -> None:
    by_id = {trial.row.trial_id: trial for trial in plan.trials}
    reservation_id = (state.locked.get(STUDY_ID) or {}).get("reservation_id")
    completed: set[str] = set()
    attempts = [
        *((attempt.get("started", {}), attempt) for attempt in state.closed_attempts.values()),
        *((attempt, None) for attempt in state.open_attempts.values()),
    ]
    for started, closed in attempts:
        if started.get("reservation_id") != reservation_id:
            continue
        trial = by_id.get(str(started.get("trial_id") or ""))
        if trial is None:
            raise ValueError("ledger contains an unknown FOUNDATION trial attempt")
        expected = {
            "trial_manifest_row_fingerprint": trial.row_fingerprint,
            "runtime_config_fingerprint": plan.runtime_fingerprint,
            "seed_bundle_fingerprint": trial.seed_fingerprint,
        }
        if any(started.get(name) != value for name, value in expected.items()):
            raise ValueError(f"ledger attempt fingerprint drift for {trial.row.trial_id}")
        if closed is not None and closed.get("completed"):
            if trial.row.trial_id in completed:
                raise ValueError(f"multiple completed attempts exist for {trial.row.trial_id}")
            completed.add(trial.row.trial_id)


def _completed_trial_ids(state: Any | None) -> set[str]:
    if state is None:
        return set()
    return {
        str(attempt.get("started", {}).get("trial_id"))
        for attempt in state.closed_attempts.values()
        if attempt.get("completed") is True
    }


def _next_ordinal(state: Any, trial_id: str) -> int:
    attempts = [
        attempt
        for attempt in (*state.open_attempts.values(), *state.closed_attempts.values())
        if attempt.get("trial_id") == trial_id
        or attempt.get("started", {}).get("trial_id") == trial_id
    ]
    return len(attempts) + 1


def _reconcile_open_attempt(
    plan: FoundationPlan,
    state: Any,
    event_root: Path,
    repo_root: Path,
    actor_id: str,
) -> None:
    reservation_id = state.locked[STUDY_ID]["reservation_id"]
    open_attempts = [
        attempt
        for attempt in state.open_attempts.values()
        if attempt.get("reservation_id") == reservation_id
    ]
    if not open_attempts:
        return
    if len(open_attempts) != 1:
        raise ValueError("ledger contains multiple open FOUNDATION attempts")
    opened = open_attempts[0]
    trial = next(
        (item for item in plan.trials if item.row.trial_id == opened.get("trial_id")),
        None,
    )
    if trial is None:
        raise ValueError("open attempt names an unknown FOUNDATION trial")
    start_event = state.events_by_id[opened["event_id"]]
    start_ref = _state_event_ref(state, opened["event_id"])
    subject_fingerprint = canonical_research_fingerprint(start_event["payload"])
    result = subprocess.CompletedProcess((), 125, "", "executor resumed an open attempt")
    try:
        output = validate_exact_trace_output(
            trial.command,
            trial.row,
            expected_runtime=plan.expected_runtime,
        )
    except (OSError, ValueError, KeyError) as exc:
        _append_failed_attempt(
            plan,
            event_root,
            repo_root,
            actor_id,
            str(opened["attempt_id"]),
            subject_fingerprint,
            start_ref,
            result,
            error=exc,
        )
        return
    _append_completed_attempt(
        plan,
        event_root,
        repo_root,
        actor_id,
        str(opened["attempt_id"]),
        subject_fingerprint,
        start_ref,
        trial,
        output,
        str(start_event.get("created_utc") or opened["attempt_id"]),
    )


def _run_trial(command: Sequence[str], repo_root: Path) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return subprocess.CompletedProcess(command, 127, "", str(exc))


def _append_completed_attempt(
    plan: FoundationPlan,
    event_root: Path,
    repo_root: Path,
    actor_id: str,
    attempt_id: str,
    subject_fingerprint: str,
    start_ref: Mapping[str, Any],
    trial: FoundationTrial,
    output: ExactTraceOutput,
    receipt_created_utc: str,
) -> None:
    actual_budget = _actual_budget(output)
    output_manifest = {
        "schema_version": 1,
        "kind": "vla_lens.trial_output_manifest",
        "attempt_id": attempt_id,
        "trial_id": trial.row.trial_id,
        "trace_id": output.trace_id,
        "trial_manifest_row_fingerprint": trial.row_fingerprint,
        "runtime_config_fingerprint": plan.runtime_fingerprint,
        "seed_bundle_fingerprint": trial.seed_fingerprint,
        "terminal_status": output.terminal_status,
        "actual_budget": actual_budget,
        "output_bytes": output.output_bytes,
        "files": list(output.files),
    }
    output_ref = _write_artifact(
        event_root,
        repo_root,
        Path("artifacts") / attempt_id / "output-manifest.json",
        output_manifest,
        artifact_id=f"{attempt_id}-output",
        artifact_type="trial_output_manifest",
    )
    receipt = {
        "schema_version": 1,
        "kind": "vla_lens.trial_runtime_receipt",
        "attempt_id": attempt_id,
        "trial_id": trial.row.trial_id,
        "child_lock_fingerprint": _lock_fingerprint(plan, event_root, repo_root),
        "runtime_config_fingerprint": plan.runtime_fingerprint,
        "seed_bundle_fingerprint": trial.seed_fingerprint,
        "runtime_check_status": "pass",
        "terminal_status": output.terminal_status,
        "output_refs": [output_ref],
        "created_utc": receipt_created_utc,
    }
    receipt_ref = _write_artifact(
        event_root,
        repo_root,
        Path("artifacts") / attempt_id / "runtime-receipt.json",
        receipt,
        artifact_id=f"{attempt_id}-runtime",
        artifact_type="trial_runtime_receipt",
    )
    append_research_event(
        event_root,
        plan.program,
        event_id=f"{attempt_id}-completed",
        event_type="trial_attempt_completed",
        actor_id=actor_id,
        subject_id=attempt_id,
        subject_fingerprint=subject_fingerprint,
        payload={
            "start_event": dict(start_ref),
            "terminal_status": output.terminal_status,
            "output_refs": [output_ref],
            "actual_budget": actual_budget,
            "runtime_receipt_ref": receipt_ref,
        },
        repo_root=repo_root,
        verify_artifacts=True,
    )


def _append_failed_attempt(
    plan: FoundationPlan,
    event_root: Path,
    repo_root: Path,
    actor_id: str,
    attempt_id: str,
    subject_fingerprint: str,
    start_ref: Mapping[str, Any],
    result: subprocess.CompletedProcess[str],
    *,
    error: Exception,
    output: ExactTraceOutput | None = None,
) -> None:
    output = output or _observed_output(result)
    log_ref = _write_artifact(
        event_root,
        repo_root,
        Path("artifacts") / attempt_id / "failure.json",
        {
            "schema_version": 1,
            "kind": "vla_lens.trial_attempt_log",
            "attempt_id": attempt_id,
            "returncode": result.returncode,
            "error": f"{type(error).__name__}: {error}",
            "stdout": result.stdout,
            "stderr": result.stderr,
        },
        artifact_id=f"{attempt_id}-failure",
        artifact_type="trial_attempt_log",
        reuse_existing=True,
    )
    append_research_event(
        event_root,
        plan.program,
        event_id=f"{attempt_id}-failed",
        event_type="trial_attempt_failed",
        actor_id=actor_id,
        subject_id=attempt_id,
        subject_fingerprint=subject_fingerprint,
        payload={
            "start_event": dict(start_ref),
            "failure_stage": (
                "output_validation" if result.returncode == 0 else "capture_subprocess"
            ),
            "error_code": "exact_output_invalid" if result.returncode == 0 else "capture_failed",
            "retryable": True,
            "log_refs": [log_ref],
            "actual_budget": _actual_budget(output),
        },
        repo_root=repo_root,
        verify_artifacts=True,
    )


def _write_artifact(
    event_root: Path,
    repo_root: Path,
    relative_to_event_root: Path,
    payload: Mapping[str, Any],
    *,
    artifact_id: str,
    artifact_type: str,
    reuse_existing: bool = False,
) -> Mapping[str, Any]:
    path = event_root / relative_to_event_root
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.exists() and reuse_existing:
        existing = load_research_mapping(path)
        if existing.get("attempt_id") != payload.get("attempt_id"):
            raise ValueError(f"existing attempt artifact has the wrong identity: {path}")
    else:
        write_bytes_create_only(path, content)
    return {
        "id": artifact_id,
        "type": artifact_type,
        "root_id": "repo",
        "path": path.relative_to(repo_root).as_posix(),
        "sha256": file_sha256(path),
    }


def _actual_budget(output: ExactTraceOutput | None) -> Mapping[str, int | float]:
    if output is None:
        return {name: 0 for name in BUDGET_FIELDS}
    return {
        "model_calls": output.model_calls,
        "action_generations": output.action_generations,
        "full_rollouts": 1,
        "simulator_steps": output.simulator_steps,
        "probe_fits": 0,
        "persistent_gb": output.output_bytes / 1_000_000_000,
        "ephemeral_gb": 0,
    }


def _observed_output(result: subprocess.CompletedProcess[str]) -> ExactTraceOutput | None:
    matches = re.findall(r"(\S+) steps=(\d+) calls=(\d+) success=(?:True|False)", result.stdout)
    if not matches:
        return None
    trace_id, steps, calls = matches[-1]
    return ExactTraceOutput(
        trace_id=trace_id,
        terminal_status="rollout_behavior_failure",
        model_calls=int(calls),
        action_generations=int(calls),
        simulator_steps=int(steps),
        output_bytes=0,
        files=(),
    )


def _requested_budget(child: Mapping[str, Any]) -> Mapping[str, int | float]:
    budget = child["budget"]
    count = int(child["trials"]["expected_count"])
    return {
        "model_calls": int(budget["max_model_calls"]) // count,
        "action_generations": int(budget["max_action_generations"]) // count,
        "full_rollouts": 1,
        "simulator_steps": int(budget["max_simulator_steps"]) // count,
        "probe_fits": 0,
        "persistent_gb": float(budget["max_persistent_gb"]) / count,
        "ephemeral_gb": float(budget["max_ephemeral_gb"]) / count,
    }


def _ensure_resolved_plan(plan: FoundationPlan) -> None:
    paths = (
        plan.output_root / "episode_plan.csv",
        plan.output_root / "episode_plan.json",
        plan.output_root / "probe_splits.csv",
        plan.output_root / "capture_config.resolved.json",
    )
    existing = [path.exists() for path in paths]
    if any(existing):
        if not all(existing):
            raise ValueError("resolved plan files are incomplete; refusing to overwrite them")
        rows = _read_episode_plan(paths[0], exact=True)
        if [row.trial_id for row in rows] != [trial.row.trial_id for trial in plan.trials]:
            raise ValueError("existing resolved plan differs from the locked 72-row plan")
        observed_config = json.loads(paths[3].read_text(encoding="utf-8"))
        if canonical_research_fingerprint(observed_config) != plan.runtime_fingerprint:
            raise ValueError("existing resolved runner config differs from the lock")
        return
    _write_plan_files(
        plan.output_root,
        config=plan.config,
        rows=[trial.row for trial in plan.trials],
    )


@contextmanager
def _executor_lock(event_root: Path) -> Iterator[None]:
    event_root.mkdir(parents=True, exist_ok=True)
    lock_path = event_root / ".foundation-executor.lock"
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another FOUNDATION hardware executor is active") from None
        yield


def _state_event_ref(state: Any, event_id: str) -> Mapping[str, Any]:
    event = state.events_by_id[event_id]
    return {
        "sequence": event["sequence"],
        "event_id": event_id,
        "event_sha256": state.event_hashes[event_id],
    }


def _lock_fingerprint(plan: FoundationPlan, event_root: Path, repo_root: Path) -> str:
    check = _verified_ledger(plan, event_root, repo_root)
    return str(check.state.locked[STUDY_ID]["lock_receipt_ref"]["content_fingerprint"])


def _validate_runner_contract(
    child: Mapping[str, Any],
    config: Mapping[str, Any],
    program_path: Path,
    trial_path: Path,
) -> None:
    runner = child["runtime"]["runner"]
    expected_argv = [
        "--backend",
        str(child["runtime"]["environment"]["backend"]),
        "--config",
        str(runner["config"]["path"]),
        "--episode-plan",
        trial_path.as_posix(),
        "--validate-exact",
        "--run",
    ]
    if (
        runner.get("entrypoint") != "scripts/pi05_batch_capture.sh"
        or runner.get("argv") != expected_argv
    ):
        raise ValueError("runner entrypoint or argv differs from the locked FOUNDATION contract")
    locked_output_root = str(child["output"]["root"] + "/" + child["output"]["namespace"])
    if config.get("output_root") != locked_output_root:
        raise ValueError("runner output_root differs from the locked child namespace")
    if config.get("model_id") != child["runtime"]["model"]["repo_id"]:
        raise ValueError("runner model differs from the locked checkpoint")
    if config.get("model_revision") != child["runtime"]["model"]["revision"]:
        raise ValueError("runner revision differs from the locked checkpoint")
    if child["program"]["path"] != program_path.as_posix():
        raise ValueError("driver program path differs from the locked runner parent")


def _validate_environment(child: Mapping[str, Any], environment: Mapping[str, Any]) -> None:
    locked = child["runtime"]["environment"]
    if environment.get("status") != "pass" or environment.get("backend") != locked["backend"]:
        raise ValueError("environment receipt does not pass for the locked backend")
    contract = load_research_mapping(
        REPO_ROOT / "configs/campaigns/rq024/foundation-r1/runtime_contract.json"
    )
    for name in ("camera", "controller", "preprocessor", "postprocessor"):
        if contract.get("components", {}).get(f"{name}_config_sha256") != locked.get(
            f"{name}_config_sha256"
        ):
            raise ValueError(f"locked {name} config differs from the runtime contract")


def _validate_checkpoint(child: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> None:
    locked = child["runtime"]["model"]
    expected = {
        "repo_id": locked["repo_id"],
        "requested_revision": locked["revision"],
        "resolved_revision": locked["revision"],
        "snapshot_manifest_sha256": locked["snapshot_manifest_sha256"],
    }
    if checkpoint.get("kind") != "vla_lens.pi05_checkpoint_snapshot_receipt" or any(
        checkpoint.get(name) != value for name, value in expected.items()
    ):
        raise ValueError("checkpoint receipt differs from the locked model snapshot")


def _expected_runtime(
    child: Mapping[str, Any], environment: Mapping[str, Any]
) -> Mapping[str, Any]:
    model = child["runtime"]["model"]
    runtime = child["runtime"]["environment"]
    return {
        "model_id": model["repo_id"],
        "model_revision": model["revision"],
        "snapshot_manifest_sha256": model["snapshot_manifest_sha256"],
        "camera_config_sha256": runtime["camera_config_sha256"],
        "controller_config_sha256": runtime["controller_config_sha256"],
        "preprocessor_config_sha256": runtime["preprocessor_config_sha256"],
        "postprocessor_config_sha256": runtime["postprocessor_config_sha256"],
        "capture_environment_sha256": canonical_sha256(environment),
    }


def _seed_fingerprint(record: Mapping[str, str]) -> str:
    payload = {
        domain: {
            "identity": record[f"{domain}_seed_identity"],
            "seed": int(record[f"{domain}_seed"]),
        }
        for domain in ("layout", "reset", "environment", "policy", "flow_noise")
    }
    return canonical_research_fingerprint(payload)


def _csv_records(path: Path) -> list[Mapping[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _require_file_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError(f"{label} is missing or differs from its locked hash: {path}")


def _repo_path(repo_root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    _require_within_repo(resolved, repo_root, "locked input")
    return resolved


def _require_within_repo(path: Path, repo_root: Path, label: str) -> None:
    try:
        path.relative_to(repo_root)
    except ValueError:
        raise ValueError(f"{label} must be inside the trusted repository root") from None


if __name__ == "__main__":
    main()
