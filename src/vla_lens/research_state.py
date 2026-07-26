"""Deterministic campaign state derived from typed, hash-chained events.

The event log is the authority.  Human prose, result-card verdicts, and
``study_advanced`` events never activate work by themselves.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from vla_lens.research_io import (
    canonical_research_fingerprint,
    file_sha256,
    load_research_mapping,
)
from vla_lens.research_plan import research_plan_fingerprint

SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
POSITIVE_VERDICTS = frozenset(
    {"gate_passed", "exploratory_positive", "confirmed_positive", "design_supported"}
)
NEGATIVE_VERDICTS = frozenset(
    {"gate_failed", "exploratory_negative", "confirmed_negative", "design_not_supported"}
)
OUTCOME_BY_VERDICT = {
    **{verdict: "positive" for verdict in POSITIVE_VERDICTS},
    **{verdict: "negative" for verdict in NEGATIVE_VERDICTS},
    "inconclusive": "inconclusive",
    "not_applicable": "not_applicable",
    "invalid": "invalid",
}
BASE_LOCK_AUDITS = frozenset({"schema", "design", "runner", "budget"})
FINAL_AUDITS = frozenset({"result"})
BUDGET_FIELDS = (
    "model_calls",
    "action_generations",
    "full_rollouts",
    "simulator_steps",
    "probe_fits",
    "persistent_gb",
    "ephemeral_gb",
)
CUMULATIVE_BUDGET_FIELDS = BUDGET_FIELDS[:6]
COUNT_BUDGET_FIELDS = BUDGET_FIELDS[:5]

ARTIFACT_REF_FIELDS = frozenset({"id", "type", "root_id", "path", "sha256"})
DOCUMENT_REF_FIELDS = frozenset({*ARTIFACT_REF_FIELDS, "content_fingerprint"})
EVENT_REF_FIELDS = frozenset({"sequence", "event_id", "event_sha256"})

PAYLOAD_FIELDS: Mapping[str, frozenset[str]] = {
    "program_locked": frozenset({"program_ref", "schema_check_ref", "manifest_commit"}),
    "child_prepared": frozenset(
        {"child_ref", "study_id", "study_fingerprint", "study_instance_id"}
    ),
    "budget_reserved": frozenset(
        {
            "reservation_id",
            "child_ref",
            "study_id",
            "study_instance_id",
            "hardware",
            "budget",
            "output_namespace",
        }
    ),
    "audit_completed": frozenset(
        {
            "audit_id",
            "audit_type",
            "subject_kind",
            "subject_fingerprint",
            "auditor_id",
            "verdict",
            "report_ref",
            "checks",
            "unresolved_errors",
        }
    ),
    "child_locked": frozenset(
        {
            "child_ref",
            "study_id",
            "study_fingerprint",
            "study_instance_id",
            "lock_receipt_ref",
            "reservation_id",
            "predecessor_result_events",
            "audit_events",
            "prior_ledger_tip",
        }
    ),
    "execution_authorized": frozenset(
        {
            "child_lock_event",
            "reservation_id",
            "authorization_ref",
            "prior_ledger_tip",
            "runtime_check_required",
        }
    ),
    "pool_accessed": frozenset(
        {
            "child_lock_event",
            "family_pool",
            "namespace",
            "access_mode",
            "exposure_record_ref",
            "data_refs",
        }
    ),
    "trial_attempt_started": frozenset(
        {
            "child_lock_event",
            "reservation_id",
            "trial_id",
            "attempt_id",
            "ordinal",
            "trial_manifest_row_fingerprint",
            "runtime_config_fingerprint",
            "seed_bundle_fingerprint",
            "requested_budget",
        }
    ),
    "trial_attempt_completed": frozenset(
        {"start_event", "terminal_status", "output_refs", "actual_budget", "runtime_receipt_ref"}
    ),
    "trial_attempt_failed": frozenset(
        {"start_event", "failure_stage", "error_code", "retryable", "log_refs", "actual_budget"}
    ),
    "deviation_recorded": frozenset(
        {"target_event", "category", "disposition", "reason", "evidence_refs"}
    ),
    "result_recorded": frozenset(
        {
            "result_ref",
            "child_lock_event",
            "reservation_id",
            "analysis_ref",
            "authorization_ref",
            "attempt_ledger_ref",
            "budget_record_ref",
            "audit_report_ref",
            "audit_events",
            "attempt_range",
            "trial_accounting",
            "budget_used",
            "supersedes_result_events",
            "outcome",
            "verdict",
        }
    ),
    "budget_released": frozenset({"reservation_id", "closing_event", "final_budget", "reason"}),
    "study_advanced": frozenset(
        {"result_event", "outcome", "program_action", "newly_eligible_studies"}
    ),
    "study_superseded": frozenset({"old_result_event", "new_result_event", "program_rule_id"}),
    "blocker_recorded": frozenset({"code", "scope", "message", "evidence_refs"}),
}


@dataclass(frozen=True, slots=True)
class CampaignStateIssue:
    """One illegal or ambiguous state transition."""

    code: str
    message: str
    sequence: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "sequence": self.sequence,
            "details": dict(self.details),
        }


@dataclass(slots=True)
class CampaignState:
    """Mutable reducer state; serialize with :meth:`to_dict` for receipts."""

    program_locked: bool = False
    prepared: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    reservations: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    locked: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    authorizations: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    audits: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    pool_access: dict[str, list[Mapping[str, Any]]] = field(default_factory=dict)
    open_attempts: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    closed_attempts: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    recorded_results: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    terminal_results: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    released_reservations: set[str] = field(default_factory=set)
    spent: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in BUDGET_FIELDS})
    events_by_id: dict[str, Mapping[str, Any]] = field(default_factory=dict)
    event_hashes: dict[str, str] = field(default_factory=dict)
    ledger_tip: str | None = None

    def to_dict(self, program: Mapping[str, Any] | None = None) -> dict[str, Any]:
        active_reservation_ids = [
            reservation_id
            for reservation_id in self.reservations
            if reservation_id not in self.released_reservations
        ]
        payload = {
            "program_locked": self.program_locked,
            "ledger_tip": self.ledger_tip,
            "prepared_studies": sorted(self.prepared),
            "locked_studies": sorted(self.locked),
            "authorized_studies": sorted(self.authorizations),
            "active_reservations": sorted(active_reservation_ids),
            "terminal_results": {
                study_id: {
                    "result_card_id": value.get("result_card_id"),
                    "result_fingerprint": value.get("result_fingerprint"),
                    "result_event_id": value.get("result_event_id"),
                    "result_event_sha256": value.get("result_event_sha256"),
                    "outcome": value.get("outcome"),
                    "verdict": value.get("verdict"),
                }
                for study_id, value in sorted(self.terminal_results.items())
            },
            "open_attempt_ids": sorted(self.open_attempts),
            "budget": (
                _budget_status(self, program)
                if program is not None
                else {
                    "used": dict(self.spent),
                    "reserved": _sum_budgets(
                        self.reservations[reservation_id]["budget"]
                        for reservation_id in active_reservation_ids
                    ),
                }
            ),
        }
        if program is not None:
            payload["status"] = campaign_status(self, program)
        payload["state_fingerprint"] = canonical_research_fingerprint(payload)
        return payload


@dataclass(frozen=True, slots=True)
class CampaignStateCheck:
    state: CampaignState
    issues: tuple[CampaignStateIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues


def reduce_campaign_events(
    event_documents: Sequence[tuple[Mapping[str, Any], str]],
    program: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    verify_artifacts: bool = False,
) -> CampaignStateCheck:
    """Replay already envelope-validated events and reject illegal transitions."""

    state = CampaignState()
    issues: list[CampaignStateIssue] = []
    root = None if repo_root is None else Path(repo_root).resolve()
    for event, event_hash in event_documents:
        sequence = _positive_int(event.get("sequence"))
        event_type = str(event.get("event_type") or "")
        payload = event.get("payload")
        before = len(issues)
        if not isinstance(payload, Mapping):
            issues.append(
                _issue("invalid_event_payload", "Event payload must be a mapping", sequence)
            )
            continue
        expected = PAYLOAD_FIELDS.get(event_type)
        if expected is None:
            issues.append(_issue("unknown_event_type", "Event type has no state rule", sequence))
            continue
        _exact_fields(payload, expected, "payload", sequence, issues)
        if len(issues) != before:
            continue
        _apply_event(
            state,
            event,
            event_hash,
            program,
            root=root,
            verify_artifacts=verify_artifacts,
            issues=issues,
        )
        if len(issues) == before:
            event_id = str(event["event_id"])
            state.events_by_id[event_id] = event
            state.event_hashes[event_id] = event_hash
            state.ledger_tip = event_hash
    return CampaignStateCheck(state=state, issues=tuple(issues))


def campaign_status(state: CampaignState, program: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact, deterministic answer to "what may an agent do next?"""

    if not state.program_locked:
        return {
            "lifecycle": "preparing",
            "phase": "program",
            "active_gate_id": "PROGRAM_LOCK",
            "hardware_authorized": False,
            "next_action": {
                "action_id": "lock_program",
                "study_id": None,
                "authority": "automatic",
                "execution_class": "repo_prep",
                "authorized": True,
                "reason_code": "program_event_missing",
                "unmet_requirements": [
                    "checked_program_snapshot",
                    "program_manifest_commit",
                    "program_lock_event",
                ],
                "completion_predicates": ["valid_program_locked_event_is_ledger_event_1"],
            },
            "runnable": [],
            "blocked": _blocked_studies(state, program),
            "budget": _budget_status(state, program),
        }
    active = [
        reservation
        for reservation_id, reservation in state.reservations.items()
        if reservation_id not in state.released_reservations
    ]
    if active:
        current = active[0]
        study_id = str(current["study_id"])
        locked = study_id in state.locked
        execution_authorized = study_id in state.authorizations
        accessed = study_id in state.pool_access
        recorded = study_id in state.recorded_results
        has_open_attempt = any(
            _get(attempt, "reservation_id") == current["reservation_id"]
            for attempt in state.open_attempts.values()
        )
        if recorded:
            action_id = "release_budget"
            execution_class = "repo_prep"
            authorized = True
            hardware_authorized = False
            reason_code = "validated_result_waiting_for_release"
        elif has_open_attempt:
            action_id = "finish_open_attempt"
            execution_class = "hardware" if current["hardware"] else "normal_compute"
            authorized = locked and execution_authorized and accessed
            hardware_authorized = bool(current["hardware"] and authorized)
            reason_code = "attempt_is_open"
        elif locked and execution_authorized and accessed:
            action_id = "run_or_analyze_next_locked_trial"
            execution_class = "hardware" if current["hardware"] else "normal_compute"
            authorized = True
            hardware_authorized = bool(current["hardware"])
            reason_code = "locked_child_has_pool_access"
        elif locked and execution_authorized:
            action_id = "record_first_permitted_pool_access"
            execution_class = "read_only"
            required_gate = str(
                _get(_study(program, study_id) or {}, "data_scope.requires_gate") or "none"
            )
            authorized = _gate_is_open(required_gate, state, program)
            hardware_authorized = False
            reason_code = "child_locked_pool_access_not_recorded"
        elif locked:
            action_id = "run_full_preflight_and_record_authorization"
            execution_class = "repo_prep"
            authorized = True
            hardware_authorized = False
            reason_code = "child_locked_execution_not_authorized"
        else:
            action_id = "finish_child_lock"
            execution_class = "repo_prep"
            authorized = True
            hardware_authorized = False
            reason_code = "active_reservation_waiting_for_lock"
        return {
            "lifecycle": "analyzing"
            if recorded
            else "running"
            if locked and execution_authorized and accessed
            else "preparing",
            "phase": _study(program, study_id).get("phase") if _study(program, study_id) else None,
            "active_gate_id": _get(_study(program, study_id) or {}, "data_scope.requires_gate"),
            "hardware_authorized": hardware_authorized,
            "next_action": {
                "action_id": action_id,
                "study_id": study_id,
                "authority": "automatic",
                "execution_class": execution_class,
                "authorized": authorized,
                "reason_code": reason_code,
                "unmet_requirements": _active_unmet_requirements(
                    locked=locked,
                    execution_authorized=execution_authorized,
                    accessed=accessed,
                    recorded=recorded,
                    has_open_attempt=has_open_attempt,
                ),
                "completion_predicates": _active_completion_predicates(action_id),
            },
            "runnable": (
                [study_id]
                if authorized and locked and execution_authorized and accessed and not recorded
                else []
            ),
            "blocked": _blocked_studies(state, program),
            "budget": _budget_status(state, program),
        }
    eligible = [
        str(study.get("id"))
        for study in _records(program.get("studies"))
        if str(study.get("id")) not in state.terminal_results and _entry_is_satisfied(study, state)
    ]
    revisions = [
        study_id
        for study_id, result in state.terminal_results.items()
        if result.get("outcome") == "invalid"
    ]
    if revisions:
        next_study = sorted(revisions)[0]
        return {
            "lifecycle": "preparing",
            "phase": _study(program, next_study).get("phase"),
            "active_gate_id": "CHILD_REVISION",
            "hardware_authorized": False,
            "next_action": {
                "action_id": "revise_child",
                "study_id": next_study,
                "authority": "automatic",
                "execution_class": "repo_prep",
                "authorized": True,
                "reason_code": "invalid_result_requires_revision",
                "unmet_requirements": [
                    "new_child_revision",
                    "new_study_instance_id",
                    "new_parent_owned_audits",
                ],
                "completion_predicates": ["replacement_child_lock_event_accepted"],
            },
            "runnable": [],
            "blocked": _blocked_studies(state, program),
            "budget": _budget_status(state, program),
        }
    next_study = eligible[0] if eligible else None
    return {
        "lifecycle": "preparing" if next_study else "terminal",
        "phase": _study(program, next_study).get("phase") if next_study else "program",
        "active_gate_id": "CHILD_LOCK" if next_study else "PROGRAM_TERMINAL",
        "hardware_authorized": False,
        "next_action": {
            "action_id": "prepare_child" if next_study else "summarize_program",
            "study_id": next_study,
            "authority": "automatic",
            "execution_class": "repo_prep" if next_study else "read_only",
            "authorized": bool(next_study),
            "reason_code": "entry_conditions_satisfied" if next_study else "no_runnable_study",
            "unmet_requirements": (
                [
                    "resolved_child_inputs",
                    "parent_owned_lock_audits",
                    "active_budget_reservation",
                    "child_lock_receipt",
                ]
                if next_study
                else []
            ),
            "completion_predicates": (
                ["child_lock_event_accepted_by_reducer"]
                if next_study
                else ["all_activated_branches_have_terminal_results"]
            ),
        },
        "runnable": [],
        "blocked": _blocked_studies(state, program),
        "budget": _budget_status(state, program),
    }


def child_authorization_issues(
    child: Mapping[str, Any], state: CampaignState, program: Mapping[str, Any]
) -> tuple[CampaignStateIssue, ...]:
    """Check a child against reducer-derived entry, lock, gate, and predecessor state."""

    issues: list[CampaignStateIssue] = []
    study_id = str(_get(child, "study.id") or "")
    study = _study(program, study_id)
    child_id = str(child.get("child_plan_id") or "")
    child_fingerprint = canonical_research_fingerprint(child)
    if not state.program_locked:
        issues.append(_issue("program_not_locked", "Program has no valid lock event"))
    if study is None:
        issues.append(_issue("unknown_child_study", "Child study is absent from the program"))
        return tuple(issues)
    if not _entry_is_satisfied(study, state):
        issues.append(_issue("study_entry_not_satisfied", "Study predecessors are not satisfied"))
    locked = state.locked.get(study_id)
    if not locked or locked.get("child_plan_id") != child_id:
        issues.append(_issue("child_not_locked_in_ledger", "No matching child-lock event exists"))
    elif locked.get("child_plan_fingerprint") != child_fingerprint:
        issues.append(
            _issue("ledger_child_hash_mismatch", "Ledger lock names different child bytes")
        )
    reservation_id = None if not locked else str(locked.get("reservation_id") or "")
    if not reservation_id or reservation_id in state.released_reservations:
        issues.append(_issue("child_budget_not_active", "Child has no active budget reservation"))
    required_gate = str(_get(study, "data_scope.requires_gate") or "none")
    if not _gate_is_open(required_gate, state, program):
        issues.append(
            _issue("execution_gate_closed", "Child data-access gate is closed", gate=required_gate)
        )
    expected_predecessors = {
        result["result_event_sha256"]
        for dependency in _dependency_ids(study)
        if (result := state.terminal_results.get(dependency)) is not None
    }
    observed_predecessors = {
        str(item.get("event_sha256")) for item in _records(child.get("predecessor_result_events"))
    }
    if observed_predecessors != expected_predecessors:
        issues.append(
            _issue(
                "predecessor_event_mismatch",
                "Child predecessor events do not match reducer-derived terminal results",
                expected=sorted(expected_predecessors),
                observed=sorted(observed_predecessors),
            )
        )
    return tuple(issues)


def _apply_event(
    state: CampaignState,
    event: Mapping[str, Any],
    event_hash: str,
    program: Mapping[str, Any],
    *,
    root: Path | None,
    verify_artifacts: bool,
    issues: list[CampaignStateIssue],
) -> None:
    event_type = str(event["event_type"])
    payload = event["payload"]
    sequence = _positive_int(event.get("sequence"))
    if event_type != "program_locked" and not state.program_locked:
        issues.append(
            _issue("program_lock_required", "Program lock must be the first event", sequence)
        )
        return
    handler = globals()[f"_apply_{event_type}"]
    handler(
        state,
        event,
        event_hash,
        payload,
        program,
        root,
        verify_artifacts,
        issues,
    )


def _apply_program_locked(state, event, event_hash, payload, program, root, verify, issues):
    sequence = _positive_int(event.get("sequence"))
    if state.program_locked or sequence != 1:
        issues.append(
            _issue("duplicate_or_late_program_lock", "Program lock must be event one", sequence)
        )
        return
    _document_ref(payload["program_ref"], "program_ref", sequence, root, verify, issues)
    _artifact_ref(payload["schema_check_ref"], "schema_check_ref", sequence, root, verify, issues)
    if payload["program_ref"].get("content_fingerprint") != research_plan_fingerprint(program):
        issues.append(
            _issue("program_ref_mismatch", "Program document fingerprint differs", sequence)
        )
    if event.get("subject_fingerprint") != research_plan_fingerprint(program):
        issues.append(_issue("program_subject_mismatch", "Program lock subject differs", sequence))
    if not COMMIT_RE.fullmatch(str(payload.get("manifest_commit") or "")):
        issues.append(
            _issue("invalid_manifest_commit", "Program lock needs a full Git commit", sequence)
        )
    if not any(issue.sequence == sequence for issue in issues):
        state.program_locked = True


def _apply_child_prepared(state, event, event_hash, payload, program, root, verify, issues):
    sequence = _positive_int(event.get("sequence"))
    study = _study(program, str(payload["study_id"]))
    _document_ref(payload["child_ref"], "child_ref", sequence, root, verify, issues)
    if event.get("subject_fingerprint") != payload["child_ref"].get("content_fingerprint"):
        issues.append(
            _issue("prepared_child_subject_mismatch", "Event subject differs from child", sequence)
        )
    if study is None or payload["study_fingerprint"] != canonical_research_fingerprint(study or {}):
        issues.append(
            _issue("prepared_study_mismatch", "Prepared child names an unknown study", sequence)
        )
    instance = str(payload["study_instance_id"])
    if instance in state.prepared:
        issues.append(
            _issue("duplicate_child_preparation", "Study instance is already prepared", sequence)
        )
    if not any(issue.sequence == sequence for issue in issues):
        state.prepared[instance] = {
            **payload,
            "event_id": event["event_id"],
            "event_sha256": event_hash,
        }


def _apply_budget_reserved(state, event, event_hash, payload, program, root, verify, issues):
    sequence = _positive_int(event.get("sequence"))
    study = _study(program, str(payload["study_id"]))
    _document_ref(payload["child_ref"], "child_ref", sequence, root, verify, issues)
    if event.get("subject_fingerprint") != payload["child_ref"].get("content_fingerprint"):
        issues.append(
            _issue("reservation_child_subject_mismatch", "Reservation subject differs", sequence)
        )
    _budget(payload["budget"], "budget", sequence, issues)
    _safe_relative(payload["output_namespace"], "output_namespace", sequence, issues)
    if study is None or not _entry_is_satisfied(study or {}, state):
        issues.append(
            _issue("reserve_before_entry", "Study entry conditions are not satisfied", sequence)
        )
    reservation_id = str(payload["reservation_id"])
    if reservation_id in state.reservations:
        issues.append(_issue("duplicate_reservation", "Reservation ID already exists", sequence))
    prepared = state.prepared.get(str(payload["study_instance_id"]))
    if (
        prepared is None
        or prepared.get("study_id") != payload["study_id"]
        or _get(prepared, "child_ref.content_fingerprint")
        != _get(payload, "child_ref.content_fingerprint")
    ):
        issues.append(
            _issue(
                "reservation_without_matching_preparation",
                "Budget reservation must match an accepted child preparation",
                sequence,
            )
        )
    _check_study_budget_cap(payload["budget"], study, sequence, issues)
    active = [key for key in state.reservations if key not in state.released_reservations]
    hardware_active = sum(bool(state.reservations[key].get("hardware")) for key in active)
    hardware_limit = int(_get(program, "program_budget.max_concurrent_hardware_children") or 0)
    if payload["hardware"] and hardware_active >= hardware_limit:
        issues.append(
            _issue("hardware_concurrency_exceeded", "Hardware child limit is full", sequence)
        )
    _check_budget_available(state, payload["budget"], program, sequence, issues)
    if not any(issue.sequence == sequence for issue in issues):
        state.reservations[reservation_id] = {
            **payload,
            "event_id": event["event_id"],
            "event_sha256": event_hash,
        }


def _apply_audit_completed(state, event, event_hash, payload, program, root, verify, issues):
    sequence = _positive_int(event.get("sequence"))
    _artifact_ref(payload["report_ref"], "report_ref", sequence, root, verify, issues)
    if event.get("subject_fingerprint") != payload.get("subject_fingerprint"):
        issues.append(_issue("audit_subject_mismatch", "Audit event subject differs", sequence))
    audit_id = str(payload["audit_id"])
    if audit_id in state.audits:
        issues.append(_issue("duplicate_audit_id", "Audit ID already exists", sequence))
    if payload["verdict"] not in {"pass", "warn", "fail"}:
        issues.append(_issue("invalid_audit_verdict", "Audit verdict is invalid", sequence))
    if payload["subject_kind"] not in {
        "research_program",
        "child_plan",
        "child_execution",
        "analysis_package",
        "result_card",
    }:
        issues.append(
            _issue("invalid_audit_subject_kind", "Audit subject kind is invalid", sequence)
        )
    if not str(payload["audit_type"] or "") or not str(payload["auditor_id"] or ""):
        issues.append(
            _issue("invalid_audit_identity", "Audit type and auditor are required", sequence)
        )
    if not _is_sequence(payload["checks"]) or not _is_sequence(payload["unresolved_errors"]):
        issues.append(
            _issue("invalid_audit_shape", "Audit checks and errors must be lists", sequence)
        )
    if payload["verdict"] == "pass" and payload["unresolved_errors"]:
        issues.append(
            _issue("passing_audit_has_errors", "Passing audit lists unresolved errors", sequence)
        )
    for index, check in enumerate(_sequence(payload["checks"])):
        if (
            not isinstance(check, Mapping)
            or set(check) != {"id", "status"}
            or not str(check.get("id") or "")
            or check.get("status") not in {"pass", "warn", "fail"}
        ):
            issues.append(
                _issue(
                    "invalid_audit_check",
                    "Audit checks need exactly a stable ID and pass, warn, or fail status",
                    sequence,
                    index=index,
                )
            )
    if payload["audit_type"] == "result" and {
        str(item.get("id")) for item in _records(payload["checks"])
    } != {"execution", "calculation", "claim"}:
        issues.append(
            _issue(
                "result_audit_checks_missing",
                "Result audit must check execution, calculation, and claim",
                sequence,
            )
        )
    if verify and root is not None and not any(issue.sequence == sequence for issue in issues):
        from vla_lens.research_audit import ResearchAuditError, validate_research_audit_report

        report = _load_mapping_artifact(payload["report_ref"], root, sequence, issues)
        if report is not None:
            try:
                validate_research_audit_report(
                    report,
                    audit_id=str(payload["audit_id"]),
                    audit_type=str(payload["audit_type"]),
                    subject_kind=str(payload["subject_kind"]),
                    subject_fingerprint=str(payload["subject_fingerprint"]),
                    auditor_id=str(payload["auditor_id"]),
                    verdict=str(payload["verdict"]),
                    checks=payload["checks"],
                    unresolved_errors=payload["unresolved_errors"],
                )
            except ResearchAuditError as exc:
                issues.append(_issue(exc.code, exc.message, sequence, **dict(exc.details)))
            _verify_audit_evidence(report, root, sequence, issues)
    if not any(issue.sequence == sequence for issue in issues):
        state.audits[audit_id] = {
            **payload,
            "event_id": event["event_id"],
            "event_sha256": event_hash,
        }


def _apply_child_locked(state, event, event_hash, payload, program, root, verify, issues):
    sequence = _positive_int(event.get("sequence"))
    study_id = str(payload["study_id"])
    study = _study(program, study_id)
    _document_ref(payload["child_ref"], "child_ref", sequence, root, verify, issues)
    if event.get("subject_fingerprint") != payload["child_ref"].get("content_fingerprint"):
        issues.append(_issue("lock_child_subject_mismatch", "Lock event subject differs", sequence))
    _document_ref(payload["lock_receipt_ref"], "lock_receipt_ref", sequence, root, verify, issues)
    if study is None or payload["study_fingerprint"] != canonical_research_fingerprint(study or {}):
        issues.append(_issue("locked_study_mismatch", "Lock names an unknown study", sequence))
    if study_id in state.locked:
        issues.append(_issue("study_already_locked", "Study already has a locked child", sequence))
    reservation = state.reservations.get(str(payload["reservation_id"]))
    if (
        reservation is None
        or reservation.get("study_instance_id") != payload["study_instance_id"]
        or reservation.get("study_id") != study_id
        or _get(reservation, "child_ref.content_fingerprint")
        != _get(payload, "child_ref.content_fingerprint")
    ):
        issues.append(
            _issue("lock_reservation_mismatch", "Child lock lacks its reservation", sequence)
        )
    if payload["prior_ledger_tip"] != event.get("previous_event_sha256"):
        issues.append(
            _issue("lock_tip_mismatch", "Child lock does not bind the prior ledger tip", sequence)
        )
    _check_entry_event_refs(payload["predecessor_result_events"], study, state, sequence, issues)
    required = BASE_LOCK_AUDITS | {
        str(item) for item in _sequence((study or {}).get("required_audits"))
    }
    observed = _audit_types_from_refs(
        payload["audit_events"],
        state,
        sequence,
        issues,
        expected_subject_fingerprint=str(payload["child_ref"]["content_fingerprint"]),
        expected_subject_kind="child_plan",
    )
    if not required <= observed:
        issues.append(
            _issue(
                "required_lock_audits_missing",
                "Child lock lacks parent-owned passing audits",
                sequence,
                missing=sorted(required - observed),
            )
        )
    locked_metadata: dict[str, Any] = {}
    if not any(issue.sequence == sequence for issue in issues):
        if verify and root is not None:
            from vla_lens.research_child import check_research_child, check_research_child_lock

            child_document = _load_document(payload["child_ref"], root, sequence, issues)
            lock_document = _load_document(payload["lock_receipt_ref"], root, sequence, issues)
            if child_document is not None and lock_document is not None:
                if (
                    _get(child_document, "study.id") != study_id
                    or _get(child_document, "study.fingerprint") != payload["study_fingerprint"]
                    or lock_document.get("study_id") != study_id
                    or lock_document.get("study_fingerprint") != payload["study_fingerprint"]
                ):
                    issues.append(
                        _issue(
                            "lock_event_child_study_mismatch",
                            "Event labels, loaded child, and lock receipt name different studies",
                            sequence,
                        )
                    )
                locked_metadata = {
                    "valid_trial_statuses": list(
                        _sequence(_get(child_document, "completion.valid_trial_statuses"))
                    ),
                    "required_artifact_types": list(
                        _sequence(_get(child_document, "output.required_artifact_types"))
                    ),
                    "expected_trial_count": _get(child_document, "trials.expected_count"),
                }
                child_check = check_research_child(
                    child_document,
                    program,
                    repo_root=root,
                    verify_files=True,
                )
                lock_check = check_research_child_lock(
                    lock_document,
                    child_document,
                    program,
                    repo_root=root,
                    verify_files=True,
                )
                for contract_issue in (*child_check.issues, *lock_check.issues):
                    issues.append(
                        _issue(
                            f"locked_{contract_issue.code}",
                            contract_issue.message,
                            sequence,
                            **dict(contract_issue.details),
                        )
                    )
                if lock_document.get("reservation_id") != payload["reservation_id"]:
                    issues.append(
                        _issue(
                            "lock_receipt_reservation_mismatch",
                            "Lock receipt and event use different reservations",
                            sequence,
                        )
                    )
                if lock_document.get("prior_ledger_tip") != payload["prior_ledger_tip"]:
                    issues.append(
                        _issue(
                            "lock_receipt_tip_mismatch",
                            "Lock receipt and event bind different ledger tips",
                            sequence,
                        )
                    )
                _check_lock_audit_event_artifacts(
                    payload["audit_events"], lock_document, state, sequence, issues
                )
                if not _reservation_matches_child_budget(reservation, child_document.get("budget")):
                    issues.append(
                        _issue(
                            "reservation_child_budget_mismatch",
                            "Reserved resources differ from the locked child budget",
                            sequence,
                        )
                    )
                resolved_namespace = str(_get(child_document, "output.namespace") or "").replace(
                    "{child_fingerprint}",
                    canonical_research_fingerprint(child_document).removeprefix("sha256:"),
                )
                if (
                    reservation is not None
                    and reservation.get("output_namespace") != resolved_namespace
                ):
                    issues.append(
                        _issue(
                            "reservation_output_namespace_mismatch",
                            "Reserved output namespace differs from the locked child",
                            sequence,
                        )
                    )
        if any(issue.sequence == sequence for issue in issues):
            return
        child = payload["child_ref"]
        state.locked[study_id] = {
            **payload,
            **locked_metadata,
            "child_plan_id": child["id"],
            "child_plan_fingerprint": child["content_fingerprint"],
            "event_id": event["event_id"],
            "event_sha256": event_hash,
        }


def _apply_execution_authorized(state, event, event_hash, payload, program, root, verify, issues):
    sequence = _positive_int(event.get("sequence"))
    _artifact_ref(payload["authorization_ref"], "authorization_ref", sequence, root, verify, issues)
    if payload["authorization_ref"].get("type") != "child_authorization_receipt":
        issues.append(
            _issue(
                "invalid_authorization_artifact_type",
                "Execution authorization must cite a child authorization receipt",
                sequence,
            )
        )
    lock_event = _resolve_event_ref(payload["child_lock_event"], state, sequence, issues)
    if lock_event is None or lock_event.get("event_type") != "child_locked":
        issues.append(
            _issue(
                "authorization_without_child_lock",
                "Execution authorization must cite an accepted child lock",
                sequence,
            )
        )
        return
    study_id = str(_get(lock_event, "payload.study_id") or "")
    lock_fingerprint = str(_get(lock_event, "payload.lock_receipt_ref.content_fingerprint") or "")
    if event.get("subject_fingerprint") != lock_fingerprint:
        issues.append(
            _issue(
                "authorization_subject_mismatch",
                "Authorization event subject differs from the child lock",
                sequence,
            )
        )
    if payload["reservation_id"] != _get(lock_event, "payload.reservation_id"):
        issues.append(
            _issue(
                "authorization_reservation_mismatch",
                "Authorization and child lock use different reservations",
                sequence,
            )
        )
    if payload["prior_ledger_tip"] != event.get("previous_event_sha256"):
        issues.append(
            _issue(
                "authorization_tip_mismatch",
                "Authorization does not bind the immediately preceding ledger state",
                sequence,
            )
        )
    if payload["runtime_check_required"] is not True:
        issues.append(
            _issue(
                "runtime_check_not_required",
                "PI0.5 runtime verification must remain required at process launch",
                sequence,
            )
        )
    if study_id in state.authorizations:
        issues.append(
            _issue("duplicate_execution_authorization", "Child is already authorized", sequence)
        )
    if verify and root is not None and not any(issue.sequence == sequence for issue in issues):
        snapshot = _load_mapping_artifact(payload["authorization_ref"], root, sequence, issues)
        if snapshot is not None:
            _check_authorization_snapshot(
                snapshot,
                state,
                program,
                lock_event,
                payload,
                root,
                sequence,
                issues,
            )
    if not any(issue.sequence == sequence for issue in issues):
        state.authorizations[study_id] = {
            **payload,
            "event_id": event["event_id"],
            "event_sha256": event_hash,
        }


def _apply_pool_accessed(state, event, event_hash, payload, program, root, verify, issues):
    sequence = _positive_int(event.get("sequence"))
    lock_event = _resolve_event_ref(payload["child_lock_event"], state, sequence, issues)
    _artifact_ref(
        payload["exposure_record_ref"], "exposure_record_ref", sequence, root, verify, issues
    )
    _artifact_refs(payload["data_refs"], "data_refs", sequence, root, verify, issues)
    if lock_event is None or lock_event.get("event_type") != "child_locked":
        issues.append(
            _issue("pool_access_without_lock", "Pool access needs a child-lock event", sequence)
        )
        return
    study_id = str(_get(lock_event, "payload.study_id") or "")
    study = _study(program, study_id)
    lock_fingerprint = str(_get(lock_event, "payload.lock_receipt_ref.content_fingerprint") or "")
    if event.get("subject_fingerprint") != lock_fingerprint:
        issues.append(
            _issue(
                "pool_access_subject_mismatch",
                "Data-access event subject differs from the child lock",
                sequence,
            )
        )
    if study_id not in state.authorizations:
        issues.append(
            _issue(
                "pool_access_without_execution_authorization",
                "Full child preflight must be recorded before data access",
                sequence,
            )
        )
    if payload["family_pool"] != _get(study or {}, "data_scope.family_pool"):
        issues.append(_issue("wrong_family_pool", "Pool access disagrees with the study", sequence))
    if payload["namespace"] not in _sequence(_get(study or {}, "data_scope.read_namespaces")):
        issues.append(
            _issue("unauthorized_namespace", "Study cannot read this namespace", sequence)
        )
    if (
        payload["access_mode"] == "selection"
        and _get(study or {}, "data_scope.selection_allowed") is not True
    ):
        issues.append(
            _issue("selection_forbidden", "Selection is forbidden for this pool", sequence)
        )
    if payload["access_mode"] not in {"baseline", "measurement", "selection"}:
        issues.append(_issue("invalid_access_mode", "Data-access mode is invalid", sequence))
    if not payload["data_refs"]:
        issues.append(
            _issue("empty_data_access", "Data access must name the exact opened bytes", sequence)
        )
    gate = str(_get(study or {}, "data_scope.requires_gate") or "none")
    if not _gate_is_open(gate, state, program):
        issues.append(
            _issue("pool_gate_closed", "Pool-access gate is not open", sequence, gate=gate)
        )
    if verify and root is not None and not any(issue.sequence == sequence for issue in issues):
        record = _load_mapping_artifact(payload["exposure_record_ref"], root, sequence, issues)
        if record is not None:
            _check_exposure_record(record, payload, lock_fingerprint, sequence, issues)
    if not any(issue.sequence == sequence for issue in issues):
        state.pool_access.setdefault(study_id, []).append(
            {
                **payload,
                "event_id": event["event_id"],
                "event_sha256": event_hash,
            }
        )


def _apply_trial_attempt_started(state, event, event_hash, payload, program, root, verify, issues):
    sequence = _positive_int(event.get("sequence"))
    lock_event = _resolve_event_ref(payload["child_lock_event"], state, sequence, issues)
    _budget(payload["requested_budget"], "requested_budget", sequence, issues)
    if lock_event is None or lock_event.get("event_type") != "child_locked":
        issues.append(_issue("attempt_without_lock", "Trial attempt needs a child lock", sequence))
    study_id = str(_get(lock_event or {}, "payload.study_id") or "")
    if study_id not in state.authorizations:
        issues.append(
            _issue(
                "attempt_without_execution_authorization",
                "Trial attempt requires a reducer-validated preflight receipt",
                sequence,
            )
        )
    if study_id not in state.pool_access:
        issues.append(
            _issue(
                "attempt_before_pool_access", "Trial started before recorded data access", sequence
            )
        )
    reservation = state.reservations.get(str(payload["reservation_id"]))
    if (
        reservation is None
        or reservation.get("study_id") != study_id
        or _get(lock_event or {}, "payload.reservation_id") != payload["reservation_id"]
    ):
        issues.append(
            _issue("attempt_reservation_mismatch", "Attempt lacks the child reservation", sequence)
        )
    attempt_id = str(payload["attempt_id"])
    if attempt_id in state.open_attempts or attempt_id in state.closed_attempts:
        issues.append(_issue("duplicate_attempt_id", "Attempt ID already exists", sequence))
    if _positive_int(payload["ordinal"]) is None:
        issues.append(
            _issue("invalid_attempt_ordinal", "Attempt ordinal must be positive", sequence)
        )
    prior_trial_attempts = [
        attempt
        for attempt in (*state.open_attempts.values(), *state.closed_attempts.values())
        if (
            _get(attempt, "reservation_id") == payload["reservation_id"]
            and _get(attempt, "trial_id") == payload["trial_id"]
        )
        or (
            _get(attempt, "started.reservation_id") == payload["reservation_id"]
            and _get(attempt, "started.trial_id") == payload["trial_id"]
        )
    ]
    if any(
        _get(attempt, "trial_id") == payload["trial_id"]
        and _get(attempt, "reservation_id") == payload["reservation_id"]
        for attempt in state.open_attempts.values()
    ):
        issues.append(
            _issue(
                "parallel_attempt_for_trial",
                "A trial may have only one open attempt",
                sequence,
            )
        )
    if payload["ordinal"] != len(prior_trial_attempts) + 1:
        issues.append(
            _issue(
                "attempt_ordinal_mismatch",
                "Attempt ordinal must follow the preserved trial history",
                sequence,
                expected=len(prior_trial_attempts) + 1,
            )
        )
    if reservation is not None:
        _check_attempt_budget_available(
            state, reservation, payload["requested_budget"], sequence, issues
        )
    for field_name in (
        "trial_manifest_row_fingerprint",
        "runtime_config_fingerprint",
        "seed_bundle_fingerprint",
    ):
        if not _sha256(payload[field_name]):
            issues.append(
                _issue(
                    "invalid_attempt_fingerprint",
                    "Attempt fingerprint is invalid",
                    sequence,
                    field=field_name,
                )
            )
    if not any(issue.sequence == sequence for issue in issues):
        state.open_attempts[attempt_id] = {
            **payload,
            "event_sequence": sequence,
            "event_id": event["event_id"],
            "event_sha256": event_hash,
        }


def _apply_trial_attempt_completed(
    state, event, event_hash, payload, program, root, verify, issues
):
    _close_attempt(state, event, event_hash, payload, root, verify, issues, completed=True)


def _apply_trial_attempt_failed(state, event, event_hash, payload, program, root, verify, issues):
    _close_attempt(state, event, event_hash, payload, root, verify, issues, completed=False)


def _apply_deviation_recorded(state, event, event_hash, payload, program, root, verify, issues):
    sequence = _positive_int(event.get("sequence"))
    _resolve_event_ref(payload["target_event"], state, sequence, issues)
    _artifact_refs(payload["evidence_refs"], "evidence_refs", sequence, root, verify, issues)
    if payload["category"] not in {"technical", "protocol"} or payload["disposition"] not in {
        "retry",
        "exclude_trial",
        "invalidate_child",
        "continue",
    }:
        issues.append(
            _issue("invalid_deviation", "Deviation category or disposition is invalid", sequence)
        )


def _apply_result_recorded(state, event, event_hash, payload, program, root, verify, issues):
    sequence = _positive_int(event.get("sequence"))
    result_ref = payload["result_ref"]
    _document_ref(result_ref, "result_ref", sequence, root, verify, issues)
    if event.get("subject_fingerprint") != result_ref.get("content_fingerprint"):
        issues.append(_issue("result_subject_mismatch", "Result event subject differs", sequence))
    _artifact_ref(payload["analysis_ref"], "analysis_ref", sequence, root, verify, issues)
    _artifact_ref(payload["authorization_ref"], "authorization_ref", sequence, root, verify, issues)
    _artifact_ref(
        payload["attempt_ledger_ref"], "attempt_ledger_ref", sequence, root, verify, issues
    )
    _artifact_ref(payload["budget_record_ref"], "budget_record_ref", sequence, root, verify, issues)
    _artifact_ref(payload["audit_report_ref"], "audit_report_ref", sequence, root, verify, issues)
    lock_event = _resolve_event_ref(payload["child_lock_event"], state, sequence, issues)
    _event_refs(payload["supersedes_result_events"], state, sequence, issues)
    if lock_event is None or lock_event.get("event_type") != "child_locked":
        issues.append(_issue("result_without_child_lock", "Result lacks its child lock", sequence))
        return
    study_id = str(_get(lock_event, "payload.study_id") or "")
    prior_terminal = state.terminal_results.get(study_id)
    expected_superseded: set[str] = set()
    if prior_terminal is not None and prior_terminal.get("outcome") == "invalid":
        expected_superseded.add(str(prior_terminal["result_event_sha256"]))
    study = _study(program, study_id)
    if payload["outcome"] == "negative" and str((study or {}).get("kind")) in {
        "behavior_confirmation",
        "semantic_confirmation",
        "causal_confirmation",
    }:
        source_id = str((study or {}).get("source_claim_study") or "")
        if not source_id:
            positive_dependencies = _sequence(
                _get(study or {}, "entry_conditions.requires_any_positive")
            )
            source_id = str(positive_dependencies[0]) if len(positive_dependencies) == 1 else ""
        source = state.terminal_results.get(source_id)
        if source is not None:
            expected_superseded.add(str(source["result_event_sha256"]))
    observed_superseded = {
        str(item.get("event_sha256")) for item in _records(payload["supersedes_result_events"])
    }
    if observed_superseded != expected_superseded:
        issues.append(
            _issue(
                "result_supersession_mismatch",
                "Result supersession does not match the reducer's invalid predecessor",
                sequence,
                expected=sorted(expected_superseded),
                observed=sorted(observed_superseded),
            )
        )
    authorization = state.authorizations.get(study_id)
    if (
        authorization is None
        or authorization.get("authorization_ref") != payload["authorization_ref"]
    ):
        issues.append(
            _issue(
                "result_authorization_mismatch",
                "Result must cite the pre-execution authorization receipt",
                sequence,
            )
        )
    if study_id in state.recorded_results:
        issues.append(
            _issue("duplicate_study_result", "Study already has a recorded result", sequence)
        )
    child_open_attempts = [
        attempt
        for attempt in state.open_attempts.values()
        if attempt.get("reservation_id") == payload["reservation_id"]
    ]
    if child_open_attempts:
        issues.append(
            _issue(
                "result_with_open_attempts",
                "Result cannot close while this child has open attempts",
                sequence,
            )
        )
    reservation_id = str(payload["reservation_id"])
    if (
        reservation_id not in state.reservations
        or reservation_id in state.released_reservations
        or _get(lock_event, "payload.reservation_id") != reservation_id
    ):
        issues.append(
            _issue("result_reservation_mismatch", "Result lacks an active reservation", sequence)
        )
    _budget(payload["budget_used"], "budget_used", sequence, issues)
    _trial_accounting(payload["trial_accounting"], sequence, issues)
    attempt_range = payload["attempt_range"]
    expected_range_fields = {"first_sequence", "last_sequence", "ledger_tip_before_result"}
    if not isinstance(attempt_range, Mapping) or set(attempt_range) != expected_range_fields:
        issues.append(
            _issue(
                "invalid_attempt_range",
                "Result attempt range has the wrong fields",
                sequence,
            )
        )
    elif (
        _positive_int(attempt_range["first_sequence"]) is None
        or _positive_int(attempt_range["last_sequence"]) is None
        or attempt_range["first_sequence"] > attempt_range["last_sequence"]
        or attempt_range["ledger_tip_before_result"] != event.get("previous_event_sha256")
    ):
        issues.append(
            _issue(
                "invalid_attempt_range",
                "Result attempt range or prior ledger tip is invalid",
                sequence,
            )
        )
    child_attempts = [
        attempt
        for attempt in state.closed_attempts.values()
        if _get(attempt, "started.reservation_id") == reservation_id
    ]
    accounting = payload["trial_accounting"]
    if child_attempts and isinstance(attempt_range, Mapping):
        expected_first = min(
            int(_get(attempt, "started.event_sequence")) for attempt in child_attempts
        )
        expected_last = max(int(attempt["event_sequence"]) for attempt in child_attempts)
        if (
            attempt_range.get("first_sequence") != expected_first
            or attempt_range.get("last_sequence") != expected_last
        ):
            issues.append(
                _issue(
                    "attempt_range_mismatch",
                    "Result attempt range differs from this child's attempt events",
                    sequence,
                    expected_first=expected_first,
                    expected_last=expected_last,
                )
            )
    elif not child_attempts:
        issues.append(
            _issue(
                "result_without_attempts",
                "A result requires at least one recorded trial attempt",
                sequence,
            )
        )
    if len(child_attempts) != accounting["attempts"]:
        issues.append(
            _issue(
                "attempt_count_mismatch",
                "Result attempt count differs from the event ledger",
                sequence,
                observed=len(child_attempts),
                declared=accounting["attempts"],
            )
        )
    by_trial: dict[str, list[Mapping[str, Any]]] = {}
    for attempt in child_attempts:
        by_trial.setdefault(str(_get(attempt, "started.trial_id") or ""), []).append(attempt)
    completed_trials = sum(
        any(bool(item["completed"]) for item in items) for items in by_trial.values()
    )
    failed_trials = sum(
        not any(bool(item["completed"]) for item in items) for items in by_trial.values()
    )
    if (
        completed_trials != accounting["completed"]
        or failed_trials != accounting["technical_failed"]
    ):
        issues.append(
            _issue(
                "trial_status_mismatch",
                "Result trial statuses differ from terminal attempt events",
                sequence,
                completed=completed_trials,
                technical_failed=failed_trials,
            )
        )
    actual_budget = {name: 0.0 for name in BUDGET_FIELDS}
    for attempt in child_attempts:
        for name in BUDGET_FIELDS:
            actual_budget[name] += float(attempt["actual_budget"][name])
    if any(float(payload["budget_used"][name]) != actual_budget[name] for name in BUDGET_FIELDS):
        issues.append(
            _issue(
                "result_budget_mismatch",
                "Result resource use differs from attempt events",
                sequence,
                observed=actual_budget,
            )
        )
    outcome = str(payload["outcome"])
    verdict = str(payload["verdict"])
    if OUTCOME_BY_VERDICT.get(verdict) != outcome:
        issues.append(_issue("result_outcome_mismatch", "Verdict and outcome disagree", sequence))
    observed_audits = _audit_types_from_refs(
        payload["audit_events"],
        state,
        sequence,
        issues,
        expected_subject_fingerprint=str(
            _get(lock_event, "payload.lock_receipt_ref.content_fingerprint") or ""
        ),
        expected_subject_kind="child_execution",
        expected_report_ref=payload["audit_report_ref"],
    )
    if outcome != "invalid" and not FINAL_AUDITS <= observed_audits:
        issues.append(
            _issue(
                "final_audits_missing",
                "Promotable result lacks final audits",
                sequence,
                missing=sorted(FINAL_AUDITS - observed_audits),
            )
        )
    if verify and root is not None and not any(issue.sequence == sequence for issue in issues):
        from vla_lens.research_summary import (
            ResearchResultCardError,
            validate_research_result_card,
        )

        child_ref = _get(lock_event, "payload.child_ref")
        lock_receipt_ref = _get(lock_event, "payload.lock_receipt_ref")
        card = _load_document(result_ref, root, sequence, issues)
        child = _load_document(child_ref, root, sequence, issues)
        child_lock = _load_document(lock_receipt_ref, root, sequence, issues)
        analysis = _load_mapping_artifact(payload["analysis_ref"], root, sequence, issues)
        if all(item is not None for item in (card, child, child_lock, analysis)):
            if card["ledger_tip_before_result"] != attempt_range["ledger_tip_before_result"]:
                issues.append(
                    _issue(
                        "result_card_ledger_tip_mismatch",
                        "Result card and result event bind different prior tips",
                        sequence,
                    )
                )
            if card["attempt_event_range"] != {
                "first_sequence": attempt_range["first_sequence"],
                "last_sequence": attempt_range["last_sequence"],
            }:
                issues.append(
                    _issue(
                        "result_card_attempt_range_mismatch",
                        "Result card and result event bind different attempt ranges",
                        sequence,
                    )
                )
            try:
                validate_research_result_card(
                    card,
                    program=program,
                    child_plan=child,
                    child_lock=child_lock,
                    analysis_package=analysis,
                    lock_receipt_sha256=str(lock_receipt_ref["sha256"]),
                    audit_report_sha256=str(payload["audit_report_ref"]["sha256"]),
                    analysis_package_sha256=str(payload["analysis_ref"]["sha256"]),
                    authorization_receipt_sha256=str(payload["authorization_ref"]["sha256"]),
                    attempt_ledger_sha256=str(payload["attempt_ledger_ref"]["sha256"]),
                    budget_record_sha256=str(payload["budget_record_ref"]["sha256"]),
                )
            except ResearchResultCardError as exc:
                issues.append(
                    _issue(
                        "result_card_validation_failed",
                        str(exc),
                        sequence,
                    )
                )
            if card["verdict"] != verdict or card["decision"]["derived_outcome"] != outcome:
                issues.append(
                    _issue(
                        "result_event_card_mismatch",
                        "Event verdict or outcome differs from the validated card",
                        sequence,
                    )
                )
    if not any(issue.sequence == sequence for issue in issues):
        record = {
            "study_id": study_id,
            "result_card_id": result_ref["id"],
            "result_fingerprint": result_ref["content_fingerprint"],
            "result_event_id": event["event_id"],
            "result_event_sha256": event_hash,
            "outcome": outcome,
            "verdict": verdict,
            "reservation_id": reservation_id,
        }
        state.recorded_results[study_id] = record


def _apply_budget_released(state, event, event_hash, payload, program, root, verify, issues):
    sequence = _positive_int(event.get("sequence"))
    reservation_id = str(payload["reservation_id"])
    reservation = state.reservations.get(reservation_id)
    closing = _resolve_event_ref(payload["closing_event"], state, sequence, issues)
    _budget(payload["final_budget"], "final_budget", sequence, issues)
    if reservation is None or reservation_id in state.released_reservations:
        issues.append(
            _issue("invalid_budget_release", "Reservation is absent or already released", sequence)
        )
    if closing is None or closing.get("event_type") != "result_recorded":
        issues.append(
            _issue("release_without_result", "Budget release must cite a result", sequence)
        )
    elif _get(closing, "payload.reservation_id") != reservation_id or payload[
        "final_budget"
    ] != _get(closing, "payload.budget_used"):
        issues.append(
            _issue(
                "release_result_mismatch",
                "Budget release must match the cited result and its exact resource use",
                sequence,
            )
        )
    if payload["reason"] not in {"result", "blocked", "superseded"}:
        issues.append(
            _issue("invalid_release_reason", "Budget release reason is invalid", sequence)
        )
    if reservation is not None:
        for name in BUDGET_FIELDS:
            if float(payload["final_budget"][name]) > float(reservation["budget"][name]):
                issues.append(
                    _issue(
                        "budget_use_exceeds_reservation",
                        "Actual use exceeds reservation",
                        sequence,
                        field=name,
                    )
                )
    if not any(issue.sequence == sequence for issue in issues):
        state.released_reservations.add(reservation_id)
        for name in BUDGET_FIELDS:
            state.spent[name] += float(payload["final_budget"][name])
        result = next(
            (
                value
                for value in state.recorded_results.values()
                if value["reservation_id"] == reservation_id
            ),
            None,
        )
        if result is not None:
            state.terminal_results[str(result["study_id"])] = result
            if result["outcome"] == "invalid":
                study_id = str(result["study_id"])
                state.locked.pop(study_id, None)
                state.authorizations.pop(study_id, None)
                state.pool_access.pop(study_id, None)
                state.recorded_results.pop(study_id, None)


def _apply_study_advanced(state, event, event_hash, payload, program, root, verify, issues):
    sequence = _positive_int(event.get("sequence"))
    result_event = _resolve_event_ref(payload["result_event"], state, sequence, issues)
    if result_event is None or result_event.get("event_type") != "result_recorded":
        issues.append(
            _issue("advance_without_result", "Study advancement must cite a result", sequence)
        )
        return
    study_id = str(_study_id_for_result_event(result_event, state) or "")
    terminal = state.terminal_results.get(study_id)
    if terminal is None:
        issues.append(
            _issue("advance_before_budget_release", "Study is not terminal yet", sequence)
        )
        return
    expected_action = _get(
        _study(program, study_id) or {}, f"outcome_actions.{terminal['outcome']}"
    )
    expected_new = _newly_eligible_studies(state, program)
    if payload["outcome"] != terminal["outcome"] or payload["program_action"] != expected_action:
        issues.append(
            _issue("derived_advance_mismatch", "Advancement disagrees with reducer state", sequence)
        )
    if sorted(payload["newly_eligible_studies"]) != expected_new:
        issues.append(
            _issue("derived_eligibility_mismatch", "Advancement invents eligible studies", sequence)
        )


def _apply_study_superseded(state, event, event_hash, payload, program, root, verify, issues):
    sequence = _positive_int(event.get("sequence"))
    old = _resolve_event_ref(payload["old_result_event"], state, sequence, issues)
    new = _resolve_event_ref(payload["new_result_event"], state, sequence, issues)
    if (
        old is None
        or new is None
        or old.get("event_type") != "result_recorded"
        or new.get("event_type") != "result_recorded"
    ):
        issues.append(
            _issue("invalid_supersession", "Supersession needs two result events", sequence)
        )


def _apply_blocker_recorded(state, event, event_hash, payload, program, root, verify, issues):
    sequence = _positive_int(event.get("sequence"))
    _artifact_refs(payload["evidence_refs"], "evidence_refs", sequence, root, verify, issues)
    if not payload["code"] or not payload["message"]:
        issues.append(
            _issue("invalid_blocker", "Blocker needs a stable code and message", sequence)
        )


def _close_attempt(state, event, event_hash, payload, root, verify, issues, *, completed):
    sequence = _positive_int(event.get("sequence"))
    start = _resolve_event_ref(payload["start_event"], state, sequence, issues)
    refs_field = "output_refs" if completed else "log_refs"
    _artifact_refs(payload[refs_field], refs_field, sequence, root, verify, issues)
    if completed:
        _artifact_ref(
            payload["runtime_receipt_ref"], "runtime_receipt_ref", sequence, root, verify, issues
        )
        if payload["runtime_receipt_ref"].get("type") != "trial_runtime_receipt":
            issues.append(
                _issue(
                    "invalid_runtime_receipt_type",
                    "Completed attempt must cite a typed trial runtime receipt",
                    sequence,
                )
            )
    _budget(payload["actual_budget"], "actual_budget", sequence, issues)
    attempt_id = str(_get(start or {}, "payload.attempt_id") or "")
    if not attempt_id or attempt_id not in state.open_attempts:
        issues.append(
            _issue(
                "attempt_close_mismatch", "Terminal event does not close an open attempt", sequence
            )
        )
    if start is not None:
        opened = state.open_attempts.get(attempt_id)
        reservation = state.reservations.get(str(_get(start, "payload.reservation_id") or ""))
        if opened is not None and reservation is not None:
            study_id = str(reservation.get("study_id") or "")
            locked = state.locked.get(study_id) or {}
            if completed and payload["terminal_status"] not in locked.get(
                "valid_trial_statuses", []
            ):
                issues.append(
                    _issue(
                        "invalid_trial_terminal_status",
                        "Completed attempt status is not allowed by the locked child",
                        sequence,
                        observed=payload["terminal_status"],
                    )
                )
            if completed and not payload["output_refs"]:
                issues.append(
                    _issue(
                        "completed_attempt_without_outputs",
                        "Completed attempt must preserve at least one output or output manifest",
                        sequence,
                    )
                )
            if completed and verify and root is not None:
                receipt = _load_mapping_artifact(
                    payload["runtime_receipt_ref"], root, sequence, issues
                )
                if receipt is not None:
                    _check_runtime_receipt(
                        receipt,
                        opened,
                        locked,
                        payload,
                        sequence,
                        issues,
                    )
            _check_attempt_actual_budget(
                state,
                reservation,
                opened,
                payload["actual_budget"],
                sequence,
                issues,
            )
    if not any(issue.sequence == sequence for issue in issues):
        opened = state.open_attempts.pop(attempt_id)
        state.closed_attempts[attempt_id] = {
            **payload,
            "started": opened,
            "completed": completed,
            "event_sequence": sequence,
            "event_id": event["event_id"],
            "event_sha256": event_hash,
        }


def _check_runtime_receipt(receipt, opened, locked, terminal_payload, sequence, issues):
    fields = {
        "schema_version",
        "kind",
        "attempt_id",
        "trial_id",
        "child_lock_fingerprint",
        "runtime_config_fingerprint",
        "seed_bundle_fingerprint",
        "runtime_check_status",
        "terminal_status",
        "output_refs",
        "created_utc",
    }
    expected = {
        "schema_version": 1,
        "kind": "vla_lens.trial_runtime_receipt",
        "attempt_id": opened["attempt_id"],
        "trial_id": opened["trial_id"],
        "child_lock_fingerprint": _get(locked, "lock_receipt_ref.content_fingerprint"),
        "runtime_config_fingerprint": opened["runtime_config_fingerprint"],
        "seed_bundle_fingerprint": opened["seed_bundle_fingerprint"],
        "runtime_check_status": "pass",
        "terminal_status": terminal_payload["terminal_status"],
        "output_refs": terminal_payload["output_refs"],
    }
    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != fields
        or any(receipt.get(name) != value for name, value in expected.items())
        or not str(receipt.get("created_utc") or "")
    ):
        issues.append(
            _issue(
                "invalid_trial_runtime_receipt",
                "Runtime receipt bytes do not match the locked attempt and outputs",
                sequence,
            )
        )


def _check_entry_event_refs(refs, study, state, sequence, issues):
    if study is None:
        return
    expected = {
        state.terminal_results[dependency]["result_event_sha256"]
        for dependency in _dependency_ids(study)
        if dependency in state.terminal_results
    }
    observed: set[str] = set()
    for ref in _sequence(refs):
        event = _resolve_event_ref(ref, state, sequence, issues)
        if event is not None and event.get("event_type") == "result_recorded":
            observed.add(str(ref["event_sha256"]))
    if observed != expected:
        issues.append(
            _issue(
                "predecessor_event_mismatch",
                "Lock predecessor events disagree with state",
                sequence,
                expected=sorted(expected),
                observed=sorted(observed),
            )
        )


def _audit_types_from_refs(
    refs,
    state,
    sequence,
    issues,
    *,
    expected_subject_fingerprint=None,
    expected_subject_kind=None,
    expected_report_ref=None,
):
    observed: set[str] = set()
    for ref in _sequence(refs):
        event = _resolve_event_ref(ref, state, sequence, issues)
        if event is None or event.get("event_type") != "audit_completed":
            issues.append(
                _issue("invalid_audit_event_ref", "Expected an audit event reference", sequence)
            )
            continue
        payload = event["payload"]
        if payload.get("verdict") != "pass" or payload.get("unresolved_errors"):
            issues.append(
                _issue(
                    "referenced_audit_not_clean", "Referenced audit did not pass cleanly", sequence
                )
            )
        if (
            expected_subject_fingerprint is not None
            and payload.get("subject_fingerprint") != expected_subject_fingerprint
        ):
            issues.append(
                _issue(
                    "audit_subject_mismatch",
                    "Referenced audit reviewed a different immutable subject",
                    sequence,
                    audit_id=payload.get("audit_id"),
                )
            )
        if (
            expected_subject_kind is not None
            and payload.get("subject_kind") != expected_subject_kind
        ):
            issues.append(
                _issue(
                    "audit_subject_kind_mismatch",
                    "Referenced audit reviewed the wrong subject kind",
                    sequence,
                    audit_id=payload.get("audit_id"),
                )
            )
        if expected_report_ref is not None and payload.get("report_ref") != expected_report_ref:
            issues.append(
                _issue(
                    "audit_report_mismatch",
                    "Referenced audit event names a different report artifact",
                    sequence,
                    audit_id=payload.get("audit_id"),
                )
            )
        observed.add(str(payload.get("audit_type")))
    return observed


def _check_lock_audit_event_artifacts(refs, lock_document, state, sequence, issues):
    locked_by_type = {
        str(item.get("audit_type")): item.get("artifact")
        for item in _records(lock_document.get("audits"))
    }
    for ref in _sequence(refs):
        event = state.events_by_id.get(str(_get(ref, "event_id") or ""))
        if event is None:
            continue
        payload = event["payload"]
        event_report = payload.get("report_ref")
        locked_report = locked_by_type.get(str(payload.get("audit_type")))
        if not isinstance(event_report, Mapping) or not isinstance(locked_report, Mapping):
            continue
        comparable_event = {
            name: event_report.get(name) for name in ("id", "type", "path", "sha256")
        }
        if comparable_event != dict(locked_report):
            issues.append(
                _issue(
                    "lock_audit_artifact_mismatch",
                    "Audit event and child lock name different report bytes",
                    sequence,
                    audit_type=payload.get("audit_type"),
                )
            )


def _check_authorization_snapshot(
    snapshot, state, program, lock_event, payload, root, sequence, issues
):
    required = {
        "schema_version",
        "created_utc",
        "program_check",
        "child_check",
        "lock_check",
        "campaign_ledger_check",
        "git_lock_check",
        "storage_check",
        "output_freshness_check",
        "authorized_to_start_child",
        "limits",
        "snapshot_payload_fingerprint",
    }
    if not isinstance(snapshot, Mapping) or not required <= set(snapshot):
        issues.append(
            _issue(
                "invalid_authorization_snapshot",
                "Authorization receipt is missing required checks",
                sequence,
                missing=sorted(
                    required - set(snapshot) if isinstance(snapshot, Mapping) else required
                ),
            )
        )
        return
    unsigned = dict(snapshot)
    declared_fingerprint = unsigned.pop("snapshot_payload_fingerprint")
    if canonical_research_fingerprint(unsigned) != declared_fingerprint:
        issues.append(
            _issue(
                "authorization_snapshot_fingerprint_mismatch",
                "Authorization receipt fingerprint does not match its contents",
                sequence,
            )
        )
    expected_program = research_plan_fingerprint(program)
    expected_child = _get(lock_event, "payload.child_ref.content_fingerprint")
    expected_lock = _get(lock_event, "payload.lock_receipt_ref.content_fingerprint")
    exact_checks = {
        "authorized_to_start_child": snapshot.get("authorized_to_start_child") is True,
        "program": _get(snapshot, "program_check.valid") is True
        and _get(snapshot, "program_check.fingerprint") == expected_program,
        "child": _get(snapshot, "child_check.valid") is True
        and _get(snapshot, "child_check.files_verified") is True
        and _get(snapshot, "child_check.fingerprint") == expected_child,
        "lock": _get(snapshot, "lock_check.valid") is True
        and _get(snapshot, "lock_check.audit_files_verified") is True
        and _get(snapshot, "lock_check.fingerprint") == expected_lock,
        "git": _get(snapshot, "git_lock_check.valid") is True,
        "storage": _get(snapshot, "storage_check.valid") is True,
        "output_freshness": _get(snapshot, "output_freshness_check.valid") is True
        and _get(snapshot, "output_freshness_check.claimed") is True,
        "ledger": _get(snapshot, "campaign_ledger_check.valid") is True
        and _get(snapshot, "campaign_ledger_check.last_event_sha256") == payload["prior_ledger_tip"]
        and _get(snapshot, "campaign_ledger_check.state") == state.to_dict(),
        "runtime_boundary": _get(snapshot, "limits.capture_wrapper_runtime_check_still_required")
        is True,
    }
    failed = sorted(name for name, passed in exact_checks.items() if not passed)
    if failed:
        issues.append(
            _issue(
                "authorization_checks_failed",
                "Authorization receipt is stale, incomplete, or not for this child",
                sequence,
                failed=failed,
            )
        )
    child = _load_document(_get(lock_event, "payload.child_ref"), root, sequence, issues)
    if child is not None:
        _check_output_claim(snapshot["output_freshness_check"], child, sequence, issues)


def _check_output_claim(output_check, child, sequence, issues):
    child_fingerprint = canonical_research_fingerprint(child)
    namespace = str(_get(child, "output.namespace") or "").replace(
        "{child_fingerprint}", child_fingerprint.removeprefix("sha256:")
    )
    expected_destination = Path(str(_get(child, "output.root") or "")) / namespace
    marker_ref = output_check.get("claim_marker")
    expected_marker = expected_destination / ".vla-lens-output-claim.json"
    if (
        output_check.get("destination") != str(expected_destination)
        or not isinstance(marker_ref, Mapping)
        or marker_ref.get("path") != str(expected_marker)
        or not _sha256(marker_ref.get("sha256"))
        or _absolute_path_has_symlink(expected_marker)
        or not expected_marker.is_file()
        or file_sha256(expected_marker) != marker_ref.get("sha256")
    ):
        issues.append(
            _issue(
                "output_claim_verification_failed",
                "Output claim directory or marker does not match the locked child",
                sequence,
            )
        )
        return
    try:
        marker = load_research_mapping(expected_marker)
    except (OSError, ValueError):
        marker = None
    expected_fields = {
        "schema_version",
        "kind",
        "child_plan_fingerprint",
        "destination",
        "created_utc",
    }
    if (
        not isinstance(marker, Mapping)
        or set(marker) != expected_fields
        or marker.get("schema_version") != 1
        or marker.get("kind") != "vla_lens.output_claim"
        or marker.get("child_plan_fingerprint") != child_fingerprint
        or marker.get("destination") != str(expected_destination)
    ):
        issues.append(
            _issue(
                "invalid_output_claim_marker",
                "Output claim marker bytes are invalid or name another child",
                sequence,
            )
        )


def _check_exposure_record(record, payload, lock_fingerprint, sequence, issues):
    fields = {
        "schema_version",
        "kind",
        "access_id",
        "child_lock_fingerprint",
        "family_pool",
        "namespace",
        "access_mode",
        "accessed_utc",
        "data_refs",
    }
    expected = {
        "schema_version": 1,
        "kind": "vla_lens.research_data_access",
        "child_lock_fingerprint": lock_fingerprint,
        "family_pool": payload["family_pool"],
        "namespace": payload["namespace"],
        "access_mode": payload["access_mode"],
        "data_refs": payload["data_refs"],
    }
    if (
        not isinstance(record, Mapping)
        or set(record) != fields
        or any(record.get(name) != value for name, value in expected.items())
        or not str(record.get("access_id") or "")
        or not str(record.get("accessed_utc") or "")
    ):
        issues.append(
            _issue(
                "invalid_data_access_record",
                "Data-access record bytes do not match the event and child lock",
                sequence,
            )
        )


def _absolute_path_has_symlink(path: Path) -> bool:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _resolve_event_ref(ref, state, sequence, issues):
    if not isinstance(ref, Mapping) or set(ref) != EVENT_REF_FIELDS:
        issues.append(_issue("invalid_event_ref", "Event reference has the wrong shape", sequence))
        return None
    event = state.events_by_id.get(str(ref.get("event_id")))
    if event is None:
        issues.append(
            _issue("unknown_event_ref", "Event reference points forward or is absent", sequence)
        )
        return None
    if event.get("sequence") != ref.get("sequence") or state.event_hashes[
        str(ref["event_id"])
    ] != ref.get("event_sha256"):
        issues.append(
            _issue("event_ref_mismatch", "Event reference identity does not match", sequence)
        )
        return None
    return event


def _event_refs(refs, state, sequence, issues):
    if not _is_sequence(refs):
        issues.append(_issue("invalid_event_refs", "Event references must be a list", sequence))
        return
    for ref in refs:
        _resolve_event_ref(ref, state, sequence, issues)


def _entry_is_satisfied(study: Mapping[str, Any], state: CampaignState) -> bool:
    required = {
        str(item) for item in _sequence(_get(study, "entry_conditions.requires_all_completed"))
    }
    positive = {
        str(item) for item in _sequence(_get(study, "entry_conditions.requires_any_positive"))
    }
    if not required <= set(state.terminal_results):
        return False
    if any(state.terminal_results[item]["outcome"] == "invalid" for item in required):
        return False
    return not positive or any(
        item in state.terminal_results and state.terminal_results[item]["outcome"] == "positive"
        for item in positive
    )


def _gate_is_open(gate_id: str, state: CampaignState, program: Mapping[str, Any]) -> bool:
    if gate_id == "none":
        return True
    gate = _get(program, f"execution_gates.{gate_id}")
    if not isinstance(gate, Mapping):
        return False
    for field_name in (
        "all_terminal",
        "all_activated_terminal",
        "all_activated_discovery_terminal",
    ):
        for study_id in _sequence(gate.get(field_name)):
            study = _study(program, str(study_id))
            activated = field_name == "all_terminal" or (
                study is not None and _entry_branch_activated(study, state)
            )
            if activated and str(study_id) not in state.terminal_results:
                return False
    for lock in _records(gate.get("lock_before_next_pool_access")):
        source = state.terminal_results.get(str(lock.get("source_study_id")))
        if source is not None and source.get("outcome") == str(lock.get("source_outcome")):
            if str(lock.get("child_study_id")) not in state.locked:
                return False
    return True


def _entry_branch_activated(study: Mapping[str, Any], state: CampaignState) -> bool:
    positive = {
        str(item) for item in _sequence(_get(study, "entry_conditions.requires_any_positive"))
    }
    return not positive or any(
        item in state.terminal_results and state.terminal_results[item]["outcome"] == "positive"
        for item in positive
    )


def _check_budget_available(state, requested, program, sequence, issues):
    active = [
        value for key, value in state.reservations.items() if key not in state.released_reservations
    ]
    mapping = {
        "model_calls": "max_model_calls",
        "action_generations": "max_action_generations",
        "full_rollouts": "max_full_rollouts",
        "simulator_steps": "max_simulator_steps",
        "probe_fits": "max_probe_fits",
        "persistent_gb": "max_persistent_gb",
    }
    for field_name, program_name in mapping.items():
        reserved = sum(float(item["budget"][field_name]) for item in active)
        total = state.spent[field_name] + reserved + float(requested[field_name])
        cap = float(_get(program, f"program_budget.{program_name}") or 0)
        if total > cap:
            issues.append(
                _issue(
                    "program_budget_exceeded",
                    "Reservation exceeds program budget",
                    sequence,
                    field=field_name,
                    requested_total=total,
                    cap=cap,
                )
            )
    ephemeral = sum(float(item["budget"]["ephemeral_gb"]) for item in active) + float(
        requested["ephemeral_gb"]
    )
    if ephemeral > float(_get(program, "program_budget.max_ephemeral_gb") or 0):
        issues.append(
            _issue(
                "ephemeral_budget_exceeded",
                "Active ephemeral storage exceeds program cap",
                sequence,
            )
        )


def _check_attempt_budget_available(state, reservation, requested, sequence, issues):
    reservation_id = str(reservation["reservation_id"])
    committed = _attempt_budget_commitment(state, reservation_id)
    for name in BUDGET_FIELDS:
        total = committed[name] + float(requested[name])
        cap = float(reservation["budget"][name])
        if total > cap:
            issues.append(
                _issue(
                    "attempt_budget_exceeds_reservation",
                    "Attempt request exceeds the child's remaining reservation",
                    sequence,
                    field=name,
                    requested_total=total,
                    cap=cap,
                )
            )


def _check_attempt_actual_budget(state, reservation, opened, actual, sequence, issues):
    reservation_id = str(reservation["reservation_id"])
    committed = _attempt_budget_commitment(
        state,
        reservation_id,
        exclude_open_attempt_id=str(opened["attempt_id"]),
    )
    for name in BUDGET_FIELDS:
        if float(actual[name]) > float(opened["requested_budget"][name]):
            issues.append(
                _issue(
                    "actual_budget_exceeds_attempt_request",
                    "Attempt used more than its preauthorized request",
                    sequence,
                    field=name,
                    actual=actual[name],
                    requested=opened["requested_budget"][name],
                )
            )
        total = committed[name] + float(actual[name])
        cap = float(reservation["budget"][name])
        if total > cap:
            issues.append(
                _issue(
                    "actual_budget_exceeds_reservation",
                    "Recorded attempt use exceeds the child's reservation",
                    sequence,
                    field=name,
                    actual_total=total,
                    cap=cap,
                )
            )


def _attempt_budget_commitment(state, reservation_id, *, exclude_open_attempt_id=None):
    total = {name: 0.0 for name in BUDGET_FIELDS}
    for attempt_id, attempt in state.open_attempts.items():
        if attempt_id == exclude_open_attempt_id or attempt.get("reservation_id") != reservation_id:
            continue
        for name in BUDGET_FIELDS:
            total[name] += float(attempt["requested_budget"][name])
    for attempt in state.closed_attempts.values():
        if _get(attempt, "started.reservation_id") != reservation_id:
            continue
        for name in BUDGET_FIELDS:
            total[name] += float(attempt["actual_budget"][name])
    return total


def _check_study_budget_cap(requested, study, sequence, issues):
    if study is None:
        return
    cap_fields = {
        "model_calls": "max_model_calls",
        "action_generations": "max_action_generations",
        "full_rollouts": "max_full_rollouts",
        "simulator_steps": "max_simulator_steps",
        "probe_fits": "max_probe_fits",
        "persistent_gb": "max_additional_persistent_gb",
        "ephemeral_gb": "max_ephemeral_gb",
    }
    for name, cap_name in cap_fields.items():
        cap = float(_get(study, f"budget.{cap_name}") or 0)
        if float(requested[name]) > cap:
            issues.append(
                _issue(
                    "study_budget_exceeded",
                    "Reservation exceeds the study budget",
                    sequence,
                    field=name,
                    requested=requested[name],
                    cap=cap,
                )
            )


def _reservation_matches_child_budget(reservation, child_budget):
    if reservation is None or not isinstance(child_budget, Mapping):
        return False
    budget = reservation["budget"]
    expected = {
        "max_model_calls": budget["model_calls"],
        "max_action_generations": budget["action_generations"],
        "max_full_rollouts": budget["full_rollouts"],
        "max_simulator_steps": budget["simulator_steps"],
        "max_probe_fits": budget["probe_fits"],
        "max_persistent_gb": budget["persistent_gb"],
        "max_ephemeral_gb": budget["ephemeral_gb"],
    }
    return all(
        float(child_budget.get(name, -1)) == float(value) for name, value in expected.items()
    )


def _artifact_ref(value, label, sequence, root, verify, issues):
    if not isinstance(value, Mapping) or set(value) != ARTIFACT_REF_FIELDS:
        issues.append(
            _issue(
                "invalid_artifact_ref",
                "Artifact reference has the wrong shape",
                sequence,
                field=label,
            )
        )
        return
    if value.get("root_id") != "repo":
        issues.append(
            _issue(
                "unresolved_artifact_root",
                "Only the trusted repo root is supported",
                sequence,
                field=label,
            )
        )
    _safe_relative(value.get("path"), label, sequence, issues)
    if not _sha256(value.get("sha256")):
        issues.append(
            _issue(
                "invalid_artifact_hash",
                "Artifact reference has an invalid hash",
                sequence,
                field=label,
            )
        )
    if verify and root is not None and not any(issue.sequence == sequence for issue in issues):
        target = _resolve_repo_path(root, str(value["path"]))
        if target is None or _path_has_symlink(root, str(value["path"])) or not target.is_file():
            issues.append(
                _issue(
                    "artifact_missing_or_unsafe",
                    "Referenced artifact is missing, outside the repo, or symlinked",
                    sequence,
                    field=label,
                )
            )
        elif file_sha256(target) != value["sha256"]:
            issues.append(
                _issue(
                    "artifact_hash_mismatch",
                    "Referenced artifact bytes differ",
                    sequence,
                    field=label,
                )
            )


def _document_ref(value, label, sequence, root, verify, issues):
    if not isinstance(value, Mapping) or set(value) != DOCUMENT_REF_FIELDS:
        issues.append(
            _issue(
                "invalid_document_ref",
                "Document reference has the wrong shape",
                sequence,
                field=label,
            )
        )
        return
    artifact = {key: value[key] for key in ARTIFACT_REF_FIELDS}
    _artifact_ref(artifact, label, sequence, root, verify, issues)
    if not _sha256(value.get("content_fingerprint")):
        issues.append(
            _issue(
                "invalid_content_fingerprint",
                "Document fingerprint is invalid",
                sequence,
                field=label,
            )
        )
    if verify and root is not None and not any(issue.sequence == sequence for issue in issues):
        target = _resolve_repo_path(root, str(value["path"]))
        try:
            observed = (
                canonical_research_fingerprint(load_research_mapping(target)) if target else None
            )
        except (OSError, ValueError):
            observed = None
        if observed != value["content_fingerprint"]:
            issues.append(
                _issue(
                    "document_fingerprint_mismatch",
                    "Parsed document content differs",
                    sequence,
                    field=label,
                )
            )


def _load_document(value: Mapping[str, Any], root: Path, sequence: int, issues):
    target = _resolve_repo_path(root, str(value.get("path") or ""))
    if target is None:
        return None
    try:
        return load_research_mapping(target)
    except (OSError, ValueError) as exc:
        issues.append(
            _issue(
                "document_load_failed",
                "Referenced document cannot be loaded strictly",
                sequence,
                path=str(target),
                error=str(exc),
            )
        )
        return None


def _load_mapping_artifact(value: Mapping[str, Any], root: Path, sequence: int, issues):
    target = _resolve_repo_path(root, str(value.get("path") or ""))
    if target is None:
        return None
    try:
        return load_research_mapping(target)
    except (OSError, ValueError) as exc:
        issues.append(
            _issue(
                "mapping_artifact_load_failed",
                "Referenced analysis document cannot be loaded strictly",
                sequence,
                path=str(target),
                error=str(exc),
            )
        )
        return None


def _verify_audit_evidence(report, root, sequence, issues):
    for check in _records(report.get("checks")):
        for reference in _records(check.get("evidence_refs")):
            relative = str(reference.get("path") or "")
            target = _resolve_repo_path(root, relative)
            if (
                target is None
                or _path_has_symlink(root, relative)
                or not target.is_file()
                or file_sha256(target) != reference.get("sha256")
            ):
                issues.append(
                    _issue(
                        "audit_evidence_verification_failed",
                        "Audit evidence is missing, outside the repo, symlinked, "
                        "or hash-mismatched",
                        sequence,
                        check_id=check.get("id"),
                        path=relative,
                    )
                )


def _artifact_refs(values, label, sequence, root, verify, issues):
    if not _is_sequence(values):
        issues.append(
            _issue(
                "invalid_artifact_refs", "Artifact references must be a list", sequence, field=label
            )
        )
        return
    for index, value in enumerate(values):
        _artifact_ref(value, f"{label}[{index}]", sequence, root, verify, issues)


def _budget(value, label, sequence, issues):
    if not isinstance(value, Mapping) or set(value) != set(BUDGET_FIELDS):
        issues.append(
            _issue("invalid_budget_shape", "Budget has the wrong fields", sequence, field=label)
        )
        return
    for name in BUDGET_FIELDS:
        number = _number(value[name])
        if (
            number is None
            or number < 0
            or (name in COUNT_BUDGET_FIELDS and not number.is_integer())
        ):
            issues.append(
                _issue(
                    "invalid_budget_value",
                    "Budget values must be finite and nonnegative",
                    sequence,
                    field=f"{label}.{name}",
                )
            )


def _trial_accounting(value, sequence, issues):
    fields = {"expected", "completed", "technical_failed", "excluded", "attempts"}
    if not isinstance(value, Mapping) or set(value) != fields:
        issues.append(
            _issue("invalid_trial_accounting", "Trial accounting has wrong fields", sequence)
        )
        return
    counts = {name: _nonnegative_int(value[name]) for name in fields}
    if any(item is None for item in counts.values()):
        issues.append(
            _issue(
                "invalid_trial_accounting", "Trial counts must be nonnegative integers", sequence
            )
        )
        return
    if counts["completed"] + counts["technical_failed"] + counts["excluded"] != counts["expected"]:
        issues.append(
            _issue(
                "trial_accounting_mismatch", "Terminal trial counts must equal expected", sequence
            )
        )


def _safe_relative(value, label, sequence, issues):
    text = str(value or "")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts or "." in path.parts or "\\" in text:
        issues.append(
            _issue(
                "unsafe_relative_path",
                "Path must be a normalized safe relative path",
                sequence,
                field=label,
                path=text,
            )
        )


def _resolve_repo_path(root: Path, relative: str) -> Path | None:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _path_has_symlink(root: Path, relative: str) -> bool:
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _exact_fields(value, allowed, label, sequence, issues):
    observed = set(value)
    if observed != set(allowed):
        issues.append(
            _issue(
                "event_payload_fields_mismatch",
                "Event payload fields must match exactly",
                sequence,
                field=label,
                missing=sorted(set(allowed) - observed),
                unknown=sorted(observed - set(allowed)),
            )
        )


def _study(program, study_id):
    if study_id is None:
        return None
    for study in _records(program.get("studies")):
        if study.get("id") == study_id:
            return study
    return None


def _dependency_ids(study):
    return set(
        dict.fromkeys(
            [
                *[
                    str(item)
                    for item in _sequence(_get(study, "entry_conditions.requires_all_completed"))
                ],
                *[
                    str(item)
                    for item in _sequence(_get(study, "entry_conditions.requires_any_positive"))
                ],
            ]
        )
    )


def _blocked_studies(state, program):
    blocked = []
    for study in _records(program.get("studies")):
        study_id = str(study.get("id"))
        if study_id in state.terminal_results or _entry_is_satisfied(study, state):
            continue
        required = {
            str(item) for item in _sequence(_get(study, "entry_conditions.requires_all_completed"))
        }
        positive = {
            str(item) for item in _sequence(_get(study, "entry_conditions.requires_any_positive"))
        }
        missing = sorted(_dependency_ids(study) - set(state.terminal_results))
        invalid = sorted(
            dependency
            for dependency in required
            if _get(state.terminal_results.get(dependency) or {}, "outcome") == "invalid"
        )
        positive_terminal = positive <= set(state.terminal_results)
        has_positive = any(
            _get(state.terminal_results.get(dependency) or {}, "outcome") == "positive"
            for dependency in positive
        )
        if invalid:
            reason_code = "invalid_required_predecessor"
        elif positive and positive_terminal and not has_positive:
            reason_code = "no_positive_predecessor"
        else:
            reason_code = "waiting_for_predecessor"
        blocked.append(
            {
                "study_id": study_id,
                "reason_code": reason_code,
                "missing": missing,
                "invalid": invalid,
            }
        )
    return blocked


def _active_unmet_requirements(
    *, locked, execution_authorized, accessed, recorded, has_open_attempt
):
    if recorded:
        return ["budget_release_event"]
    if has_open_attempt:
        return ["terminal_attempt_event"]
    if locked and execution_authorized and accessed:
        return ["remaining_trial_or_validated_result"]
    if locked and not execution_authorized:
        return ["full_child_preflight_receipt", "execution_authorization_event"]
    if locked:
        return ["first_permitted_pool_access_event"]
    return [
        "passing_parent_owned_audits",
        "verified_child_lock_receipt",
        "child_lock_event",
    ]


def _active_completion_predicates(action_id):
    return {
        "release_budget": ["budget_release_event_accepted_and_result_becomes_terminal"],
        "finish_open_attempt": ["attempt_has_exactly_one_terminal_event"],
        "run_or_analyze_next_locked_trial": ["attempt_recorded_or_typed_result_validated"],
        "run_full_preflight_and_record_authorization": [
            "execution_authorization_event_accepted_by_reducer"
        ],
        "record_first_permitted_pool_access": ["pool_access_event_accepted_by_reducer"],
        "finish_child_lock": ["child_lock_event_accepted_by_reducer"],
    }[action_id]


def _sum_budgets(values):
    total = {name: 0.0 for name in BUDGET_FIELDS}
    for value in values:
        for name in BUDGET_FIELDS:
            total[name] += float(value[name])
    return total


def _budget_status(state, program):
    active = [
        reservation
        for reservation_id, reservation in state.reservations.items()
        if reservation_id not in state.released_reservations
    ]
    reserved = _sum_budgets(reservation["budget"] for reservation in active)
    cap_fields = {
        "model_calls": "max_model_calls",
        "action_generations": "max_action_generations",
        "full_rollouts": "max_full_rollouts",
        "simulator_steps": "max_simulator_steps",
        "probe_fits": "max_probe_fits",
        "persistent_gb": "max_persistent_gb",
        "ephemeral_gb": "max_ephemeral_gb",
    }
    caps = {
        name: float(_get(program, f"program_budget.{program_name}") or 0)
        for name, program_name in cap_fields.items()
    }
    remaining = {
        name: max(
            0.0,
            cap - reserved[name] - (0.0 if name == "ephemeral_gb" else state.spent[name]),
        )
        for name, cap in caps.items()
    }
    hardware_limit = int(_get(program, "program_budget.max_concurrent_hardware_children") or 0)
    return {
        "used": dict(state.spent),
        "reserved": reserved,
        "remaining": remaining,
        "active_hardware_children": sum(
            bool(reservation.get("hardware")) for reservation in active
        ),
        "hardware_child_limit": hardware_limit,
    }


def _newly_eligible_studies(state, program):
    return sorted(
        str(study.get("id"))
        for study in _records(program.get("studies"))
        if str(study.get("id")) not in state.terminal_results and _entry_is_satisfied(study, state)
    )


def _study_id_for_result_event(event, state):
    lock_ref = _get(event, "payload.child_lock_event")
    if not isinstance(lock_ref, Mapping):
        return None
    lock_event = state.events_by_id.get(str(lock_ref.get("event_id")))
    return _get(lock_event or {}, "payload.study_id")


def _get(payload, path):
    value = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _is_sequence(value):
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _sequence(value):
    return value if _is_sequence(value) else ()


def _records(value):
    return [item for item in _sequence(value) if isinstance(item, Mapping)]


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive_int(value):
    number = _number(value)
    return int(number) if number is not None and number > 0 and number.is_integer() else None


def _nonnegative_int(value):
    number = _number(value)
    return int(number) if number is not None and number >= 0 and number.is_integer() else None


def _sha256(value):
    return bool(SHA256_RE.fullmatch(str(value or "")))


def _issue(code, message, sequence=None, **details):
    return CampaignStateIssue(code=code, message=message, sequence=sequence, details=details)


__all__ = [
    "ARTIFACT_REF_FIELDS",
    "BUDGET_FIELDS",
    "CampaignState",
    "CampaignStateCheck",
    "CampaignStateIssue",
    "DOCUMENT_REF_FIELDS",
    "EVENT_REF_FIELDS",
    "PAYLOAD_FIELDS",
    "campaign_status",
    "child_authorization_issues",
    "reduce_campaign_events",
]
