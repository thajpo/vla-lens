"""Append-only execution driver for the locked RQ-024 FOUNDATION trial plan."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import re
import subprocess
import time
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
LOCKED_INPUT_HASHES = {
    "child.yaml": "sha256:c7852748a162a7f5076995068c2730732629887edcb8a06788b7f5def3f90e6c",
    "trials.csv": "sha256:abbeb22023c892f545e8ff4ecc0d606b266e81529be5e2204646a8d492c8dae8",
    "checkpoint.json": "sha256:7359dc789a3237f5c7baeb22a560ef9fc991ef9a3c207082a354835a1073a80f",
    "environment.json": "sha256:003ad6d756482160be1e894b92794404f758670d9218c4a3d4e127dc007f5271",
    "runtime_contract.json": (
        "sha256:c40c0053526af4c8f316638d5050d1ab2443234a39a18066e0c964a29f620bb2"
    ),
    "capture.yaml": "sha256:81affbde1fe67ced3189fd1e3377a2c51ee3cf6d2053094b32bfb317417a1a21",
}


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


def _load_foundation_contract(
    repo_root: Path = REPO_ROOT,
    *,
    program_path: Path = DEFAULT_PROGRAM,
    child_path: Path = DEFAULT_CHILD,
) -> FoundationPlan:
    """Load and hash every locked execution input without writing files."""

    repo_root = repo_root.resolve()
    _require_locked_input_hashes(repo_root)
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
    _validate_environment(child, environment, repo_root)

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


def load_foundation_plan(
    repo_root: Path = REPO_ROOT,
    *,
    program_path: Path = DEFAULT_PROGRAM,
    child_path: Path = DEFAULT_CHILD,
) -> FoundationPlan:
    """Load the immutable, execution-ready 72-row plan without writing files."""

    return _load_foundation_contract(
        repo_root,
        program_path=program_path,
        child_path=child_path,
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
    with foundation_hardware_lock():
        check = _authorized_ledger(plan, event_root, repo_root, allow_open=True)
        _reconcile_open_attempt(plan, check.state, event_root, repo_root, actor_id)
        check = _authorized_ledger(plan, event_root, repo_root)
        _ensure_resolved_plan(plan)
        selected = _select_contract_trials(
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
            try:
                result = _run_trial(
                    trial.execution_command,
                    repo_root,
                    output_root=trial.command.output_root,
                )
            except Exception as exc:
                actual_budget = _measured_budget_or_block(result_or_error=exc)
                _append_failed_attempt(
                    plan,
                    event_root,
                    repo_root,
                    actor_id,
                    attempt_id,
                    subject_fingerprint,
                    start_ref,
                    subprocess.CompletedProcess(trial.execution_command, 1, "", str(exc)),
                    error=exc,
                    actual_budget=actual_budget,
                )
                continue
            if result.returncode != 0:
                actual_budget = _measured_budget_or_block(result_or_error=result)
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
                    actual_budget=actual_budget,
                )
                continue
            try:
                output = validate_exact_trace_output(
                    trial.command,
                    trial.row,
                    expected_runtime=plan.expected_runtime,
                )
            except Exception as exc:
                actual_budget = _measured_budget_or_block(result_or_error=result)
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
                    actual_budget=actual_budget,
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


def _select_contract_trials(
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
    selected = _select_contract_trials(
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


def select_trials(
    plan: FoundationPlan,
    state: Any | None,
    *,
    trial_id: str | None = None,
    max_trials: int | None = None,
    remaining: bool = True,
) -> tuple[FoundationTrial, ...]:
    """Select trials in manifest order using reducer state as the sole history."""

    return _select_contract_trials(
        plan,
        state,
        trial_id=trial_id,
        max_trials=max_trials,
        remaining=remaining,
    )


def persist_resolved_plan_once(output_root: Path, plan: Mapping[str, Any]) -> Path:
    """Create one complete resolved-plan document and never replace it."""

    path = Path(output_root) / "rq024_foundation_resolved_plan.json"
    content = (json.dumps(plan, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_bytes_create_only(path, content)
    return path


class RuntimeAccountingBlocker(RuntimeError):
    """Raised when a started attempt has no honest actual-usage evidence."""


def execute_attempt(
    trial: Mapping[str, Any],
    *,
    append_event: Any,
    launch: Any,
    validate_output: Any,
) -> dict[str, Any]:
    """Execute one append-only attempt with injectable I/O boundaries."""

    start_payload = {
        name: trial[name]
        for name in (
            "child_lock_event",
            "reservation_id",
            "trial_id",
            "attempt_id",
            "ordinal",
            "trial_manifest_row_fingerprint",
            "runtime_config_fingerprint",
            "seed_bundle_fingerprint",
            "requested_budget",
        )
    }
    start_ref = append_event("trial_attempt_started", start_payload)
    try:
        result = launch(trial["command"])
    except OSError as exc:
        actual_budget = {name: 0 for name in BUDGET_FIELDS}
        append_event(
            "trial_attempt_failed",
            _failed_attempt_payload(
                start_ref,
                actual_budget,
                failure_stage="subprocess_launch",
                error_code="subprocess_not_started",
            ),
        )
        raise RuntimeError(f"capture process could not start: {exc}") from exc
    except Exception as exc:
        actual_budget = _measured_budget_or_block(result_or_error=exc)
        append_event(
            "trial_attempt_failed",
            _failed_attempt_payload(
                start_ref,
                actual_budget,
                failure_stage="capture_subprocess",
                error_code=type(exc).__name__.lower(),
            ),
        )
        raise

    measured_budget = getattr(result, "measured_actual_budget", None)
    if int(result.returncode) != 0:
        actual_budget = _measured_budget_or_block(result_or_error=result)
        append_event(
            "trial_attempt_failed",
            _failed_attempt_payload(
                start_ref,
                actual_budget,
                failure_stage="capture_subprocess",
                error_code=f"capture_exit_{result.returncode}",
            ),
        )
        raise RuntimeError(f"capture process exited with return code {result.returncode}")

    try:
        output = validate_output(trial)
    except Exception as exc:
        actual_budget = _measured_budget_or_block(result_or_error=result)
        append_event(
            "trial_attempt_failed",
            _failed_attempt_payload(
                start_ref,
                actual_budget,
                failure_stage="output_validation",
                error_code=type(exc).__name__.lower(),
            ),
        )
        raise

    try:
        if not isinstance(output, Mapping):
            raise ValueError("exact output validator must return a mapping")
        terminal_status = str(output.get("terminal_status") or "")
        if terminal_status not in {"rollout_success", "rollout_behavior_failure"}:
            raise ValueError(f"invalid exact trial terminal status: {terminal_status!r}")
        actual_budget = _validate_actual_budget(output.get("actual_budget"), "validated output")
        if measured_budget is not None:
            _validate_actual_budget(measured_budget, "capture process")
        output_refs = output.get("output_refs")
        runtime_receipt_ref = output.get("runtime_receipt_ref")
        if not isinstance(output_refs, list) or not output_refs or not isinstance(
            runtime_receipt_ref, Mapping
        ):
            raise ValueError("exact output must provide output and runtime receipt references")
        output_bytes = output.get("output_bytes")
        if (
            isinstance(output_bytes, bool)
            or not isinstance(output_bytes, int)
            or output_bytes < 0
        ):
            raise ValueError("exact output must report nonnegative output bytes")
        completed_payload = {
            "start_event": start_ref,
            "terminal_status": terminal_status,
            "output_refs": output_refs,
            "actual_budget": actual_budget,
            "runtime_receipt_ref": runtime_receipt_ref,
        }
    except Exception as exc:
        evidence = output.get("actual_budget") if isinstance(output, Mapping) else None
        evidence = evidence or measured_budget
        actual_budget = _validate_actual_budget(evidence, "failed output validation")
        append_event(
            "trial_attempt_failed",
            _failed_attempt_payload(
                start_ref,
                actual_budget,
                failure_stage="output_validation",
                error_code=type(exc).__name__.lower(),
            ),
        )
        raise
    append_event("trial_attempt_completed", completed_payload)
    return {
        "terminal_event_type": "trial_attempt_completed",
        "terminal_status": terminal_status,
        "actual_budget": actual_budget,
        "output_bytes": output_bytes,
    }


def _failed_attempt_payload(
    start_ref: Mapping[str, Any],
    actual_budget: Mapping[str, int | float],
    *,
    failure_stage: str,
    error_code: str,
) -> dict[str, Any]:
    return {
        "start_event": start_ref,
        "failure_stage": failure_stage,
        "error_code": error_code,
        "retryable": True,
        "log_refs": [],
        "actual_budget": dict(actual_budget),
    }


def _measured_budget_or_block(*, result_or_error: Any) -> dict[str, int | float]:
    measured = getattr(result_or_error, "measured_actual_budget", None)
    if measured is None:
        raise RuntimeAccountingBlocker(
            "started attempt cannot be closed honestly: runtime evidence does not provide "
            "actual model calls, generations, rollouts, simulator steps, and bytes"
        )
    return _validate_actual_budget(measured, "runtime evidence")


def _validate_actual_budget(value: Any, source: str) -> dict[str, int | float]:
    if not isinstance(value, Mapping) or set(value) != set(BUDGET_FIELDS):
        raise RuntimeAccountingBlocker(f"{source} has incomplete actual resource accounting")
    budget: dict[str, int | float] = {}
    for name in BUDGET_FIELDS:
        number = value[name]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or number < 0:
            raise RuntimeAccountingBlocker(f"{source} has invalid actual budget field {name}")
        budget[name] = number
    return budget


@contextmanager
def foundation_hardware_lock(
    _caller_path: Path | None = None,
    *,
    timeout_s: float = 0.0,
) -> Iterator[Path]:
    """Lock the fixed FOUNDATION child/output namespace across all event roots."""

    identity = "rq024-foundation-r1|/mnt/new-volume/vla-lens/rq024/rq024-foundation-r1"
    digest = hashlib.sha256(identity.encode("ascii")).hexdigest()
    lock_root = Path("/tmp/vla-lens-hardware-locks")
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / f"{digest}.lock"
    deadline = time.monotonic() + max(0.0, timeout_s)
    with lock_path.open("a+b") as lock:
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("FOUNDATION hardware lock is already held") from None
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        try:
            yield lock_path
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="Print the resolved plan; write nothing.")
    mode.add_argument(
        "--run", action="store_true", help="Execute reducer-authorized hardware trials."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--child", type=Path, default=DEFAULT_CHILD)
    parser.add_argument("--event-root", type=Path)
    parser.add_argument("--actor-id", default="rq024-foundation-driver")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--trial-id")
    selection.add_argument("--max-trials", type=int)
    parser.add_argument("--remaining", action="store_true")
    parser.add_argument("--force", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.force:
        raise SystemExit("--force is forbidden for append-only FOUNDATION execution")
    repo_root = args.repo_root.resolve()
    plan = load_foundation_plan(
        repo_root,
        program_path=args.program,
        child_path=args.child,
    )
    if args.plan:
        state = None
        if args.remaining:
            if args.event_root is None:
                raise SystemExit("--remaining requires --event-root")
            state = _verified_ledger(plan, args.event_root.resolve(), repo_root).state
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
        return 0
    if args.event_root is None:
        raise SystemExit("--run requires --event-root")
    completed = execute_foundation(
        plan,
        event_root=args.event_root,
        actor_id=args.actor_id,
        repo_root=repo_root,
        trial_id=args.trial_id,
        max_trials=args.max_trials,
    )
    print(json.dumps({"completed_trial_ids": completed}, indent=2, sort_keys=True))
    return 0


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
    try:
        output = validate_exact_trace_output(
            trial.command,
            trial.row,
            expected_runtime=plan.expected_runtime,
        )
    except Exception as exc:
        raise RuntimeAccountingBlocker(
            "open attempt has neither exact output nor preserved actual-usage evidence; "
            "refusing fabricated failure accounting"
        ) from exc
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


def _run_trial(
    command: Sequence[str],
    repo_root: Path,
    *,
    output_root: Path,
) -> subprocess.CompletedProcess[str]:
    try:
        before = _output_snapshot(output_root)
    except OSError as exc:
        exc.measured_actual_budget = {name: 0 for name in BUDGET_FIELDS}
        raise
    try:
        result = subprocess.run(
            list(command),
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        result = subprocess.CompletedProcess(command, 127, "", str(exc))
        result.measured_actual_budget = {name: 0 for name in BUDGET_FIELDS}
        return result
    output_bytes = _changed_output_bytes(before, _output_snapshot(output_root))
    observed = _observed_output(result, output_bytes=output_bytes)
    if observed is not None:
        result.measured_actual_budget = _actual_budget(observed)
    return result


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
    actual_budget: Mapping[str, int | float],
) -> None:
    actual_budget = _validate_actual_budget(actual_budget, "failed attempt")
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
            "actual_budget": actual_budget,
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


def _actual_budget(output: ExactTraceOutput) -> Mapping[str, int | float]:
    return {
        "model_calls": output.model_calls,
        "action_generations": output.action_generations,
        "full_rollouts": 1,
        "simulator_steps": output.simulator_steps,
        "probe_fits": 0,
        "persistent_gb": output.output_bytes / (1024**3),
        "ephemeral_gb": 0,
    }


def _observed_output(
    result: subprocess.CompletedProcess[str],
    *,
    output_bytes: int,
) -> ExactTraceOutput | None:
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
        output_bytes=output_bytes,
        files=(),
    )


def _output_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def _changed_output_bytes(
    before: Mapping[str, tuple[int, int]],
    after: Mapping[str, tuple[int, int]],
) -> int:
    return sum(
        size
        for path, (size, modified) in after.items()
        if before.get(path) != (size, modified)
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


def _validate_environment(
    child: Mapping[str, Any],
    environment: Mapping[str, Any],
    repo_root: Path,
) -> None:
    locked = child["runtime"]["environment"]
    if environment.get("status") != "pass" or environment.get("backend") != locked["backend"]:
        raise ValueError("environment receipt does not pass for the locked backend")
    contract = load_research_mapping(
        repo_root / "configs/campaigns/rq024/foundation-r1/runtime_contract.json"
    )
    for name in ("camera", "controller", "preprocessor", "postprocessor"):
        if contract.get("components", {}).get(f"{name}_config_sha256") != locked.get(
            f"{name}_config_sha256"
        ):
            raise ValueError(f"locked {name} config differs from the runtime contract")


def _validate_checkpoint(child: Mapping[str, Any], checkpoint: Mapping[str, Any]) -> None:
    locked = child["runtime"]["model"]
    expected_fields = {
        "schema_version",
        "kind",
        "repo_id",
        "requested_revision",
        "resolved_revision",
        "snapshot_path",
        "snapshot_manifest_sha256",
        "files",
    }
    if set(checkpoint) != expected_fields or checkpoint.get("schema_version") != 1:
        raise ValueError("checkpoint receipt schema or fields changed")
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
    snapshot_path = Path(str(checkpoint.get("snapshot_path") or ""))
    if not snapshot_path.is_absolute() or snapshot_path.name != locked["revision"]:
        raise ValueError("checkpoint receipt snapshot path does not bind the locked revision")
    files = checkpoint.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("checkpoint receipt files manifest is empty or invalid")
    observed_paths: set[str] = set()
    for index, item in enumerate(files):
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256", "size"}:
            raise ValueError(f"checkpoint receipt file {index} schema changed")
        path = str(item.get("path") or "")
        pure = Path(path)
        if (
            not path
            or pure.is_absolute()
            or ".." in pure.parts
            or path in observed_paths
            or not _is_sha256(item.get("sha256"))
            or isinstance(item.get("size"), bool)
            or not isinstance(item.get("size"), int)
            or int(item["size"]) < 0
        ):
            raise ValueError(f"checkpoint receipt file {index} hash, size, or path is invalid")
        observed_paths.add(path)
    if [str(item["path"]) for item in files] != sorted(observed_paths):
        raise ValueError("checkpoint receipt files are not in canonical path order")
    if canonical_sha256(files) != checkpoint["snapshot_manifest_sha256"]:
        raise ValueError("checkpoint receipt file hashes and sizes do not match its bound manifest")


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


def _require_locked_input_hashes(repo_root: Path) -> None:
    root = repo_root / "configs/campaigns/rq024/foundation-r1"
    for name, expected in LOCKED_INPUT_HASHES.items():
        _require_file_hash(root / name, expected, f"locked {name}")


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", text))


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
