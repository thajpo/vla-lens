"""Append-only, hash-chained events for autonomous research campaigns."""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from vla_lens.research_io import (
    canonical_research_fingerprint,
    file_sha256,
    load_research_mapping,
    write_bytes_create_only,
)
from vla_lens.research_plan import research_plan_fingerprint
from vla_lens.research_state import CampaignState, campaign_status, reduce_campaign_events

EVENT_SCHEMA_VERSION = 1
EVENT_TYPES = frozenset(
    {
        "program_locked",
        "child_prepared",
        "child_locked",
        "execution_authorized",
        "budget_reserved",
        "budget_released",
        "pool_accessed",
        "trial_attempt_started",
        "trial_attempt_completed",
        "trial_attempt_failed",
        "deviation_recorded",
        "audit_completed",
        "result_recorded",
        "study_advanced",
        "study_superseded",
        "blocker_recorded",
    }
)
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")
EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "program_id",
        "program_fingerprint",
        "sequence",
        "event_id",
        "event_type",
        "created_utc",
        "actor_id",
        "subject_id",
        "subject_fingerprint",
        "previous_event_sha256",
        "payload",
    }
)


@dataclass(frozen=True, slots=True)
class EventLedgerIssue:
    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": dict(self.details)}


@dataclass(frozen=True, slots=True)
class EventLedgerCheck:
    event_count: int
    last_event_sha256: str | None
    type_counts: Mapping[str, int]
    state: CampaignState
    issues: tuple[EventLedgerIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "research_campaign_event_ledger_check",
            "valid": self.valid,
            "event_count": self.event_count,
            "last_event_sha256": self.last_event_sha256,
            "type_counts": dict(self.type_counts),
            "state": self.state.to_dict() if not self.issues else None,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def append_research_event(
    root: str | Path,
    program: Mapping[str, Any],
    *,
    event_id: str,
    event_type: str,
    actor_id: str,
    subject_id: str,
    subject_fingerprint: str,
    payload: Mapping[str, Any],
    created_utc: str | None = None,
    repo_root: str | Path | None = None,
    verify_artifacts: bool = False,
) -> tuple[Path, str]:
    """Append one immutable event while serializing concurrent agents."""

    if not ID_RE.fullmatch(event_id):
        raise ValueError(
            "event_id must use 3-128 lowercase letters, digits, dots, dashes, or underscores"
        )
    if event_type not in EVENT_TYPES:
        raise ValueError(f"Unknown event type {event_type!r}")
    if not actor_id or not subject_id:
        raise ValueError("actor_id and subject_id are required")
    if not _sha256(subject_fingerprint):
        raise ValueError("subject_fingerprint must be a full sha256")
    canonical_research_fingerprint(payload)
    event_root = Path(root)
    event_root.mkdir(parents=True, exist_ok=True)
    lock_path = event_root / ".append.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing = _event_paths(event_root)
        check = verify_research_event_ledger(
            event_root,
            program,
            repo_root=repo_root,
            verify_artifacts=verify_artifacts,
        )
        if not check.valid:
            raise ValueError("Refusing to append to an invalid campaign event ledger")
        if any(path.name.endswith(f"-{event_id}.json") for path in existing):
            raise FileExistsError(f"Event ID already exists: {event_id}")
        sequence = len(existing) + 1
        event = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "program_id": program.get("program_id"),
            "program_fingerprint": research_plan_fingerprint(program),
            "sequence": sequence,
            "event_id": event_id,
            "event_type": event_type,
            "created_utc": created_utc or datetime.now(UTC).isoformat(),
            "actor_id": actor_id,
            "subject_id": subject_id,
            "subject_fingerprint": subject_fingerprint,
            "previous_event_sha256": check.last_event_sha256,
            "payload": payload,
        }
        content = _event_bytes(event)
        documents = _load_event_documents(existing)
        candidate_hash = _bytes_sha256(content)
        state_check = reduce_campaign_events(
            [*documents, (event, candidate_hash)],
            program,
            repo_root=repo_root,
            verify_artifacts=verify_artifacts,
        )
        if not state_check.valid:
            summary = ", ".join(issue.code for issue in state_check.issues[-3:])
            raise ValueError(f"Refusing illegal campaign transition: {summary}")
        destination = event_root / f"{sequence:06d}-{event_id}.json"
        write_bytes_create_only(destination, content)
        return destination, file_sha256(destination)


def verify_research_event_ledger(
    root: str | Path,
    program: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    verify_artifacts: bool = False,
) -> EventLedgerCheck:
    """Verify the hash chain, typed payloads, and every derived state transition."""

    event_root = Path(root)
    paths = _event_paths(event_root) if event_root.exists() else []
    issues: list[EventLedgerIssue] = []
    previous: str | None = None
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    expected_program = research_plan_fingerprint(program)
    documents: list[tuple[Mapping[str, Any], str]] = []
    for expected_sequence, path in enumerate(paths, start=1):
        try:
            event = load_research_mapping(path)
        except (OSError, ValueError) as exc:
            issues.append(
                _issue(
                    "invalid_event_file", "Event cannot be loaded", path=str(path), error=str(exc)
                )
            )
            continue
        unknown = sorted(set(event) - EVENT_FIELDS)
        missing = sorted(EVENT_FIELDS - set(event))
        if unknown or missing:
            issues.append(
                _issue(
                    "event_envelope_fields_mismatch",
                    "Event envelope fields must match exactly",
                    path=str(path),
                    unknown=unknown,
                    missing=missing,
                )
            )
        if event.get("schema_version") != EVENT_SCHEMA_VERSION:
            issues.append(
                _issue("invalid_event_schema", "Event schema is unsupported", path=str(path))
            )
        if event.get("sequence") != expected_sequence:
            issues.append(
                _issue(
                    "event_sequence_gap",
                    "Event sequence is not contiguous",
                    path=str(path),
                    expected=expected_sequence,
                    observed=event.get("sequence"),
                )
            )
        prefix = f"{expected_sequence:06d}-"
        if not path.name.startswith(prefix):
            issues.append(
                _issue(
                    "event_filename_mismatch",
                    "Event filename disagrees with sequence",
                    path=str(path),
                )
            )
        event_id = str(event.get("event_id") or "")
        if event_id in seen_ids:
            issues.append(
                _issue("duplicate_event_id", "Event ID appears more than once", event_id=event_id)
            )
        seen_ids.add(event_id)
        event_type = str(event.get("event_type") or "")
        if event_type not in EVENT_TYPES:
            issues.append(
                _issue("unknown_event_type", "Event type is not recognized", event_type=event_type)
            )
        counts[event_type] += 1
        if (
            event.get("program_id") != program.get("program_id")
            or event.get("program_fingerprint") != expected_program
        ):
            issues.append(
                _issue("event_program_mismatch", "Event belongs to another program", path=str(path))
            )
        if event.get("previous_event_sha256") != previous:
            issues.append(
                _issue("event_chain_broken", "Event prior hash does not match", path=str(path))
            )
        if not _sha256(event.get("subject_fingerprint")):
            issues.append(
                _issue(
                    "invalid_event_subject_hash", "Event subject hash is invalid", path=str(path)
                )
            )
        previous = file_sha256(path)
        documents.append((event, previous))
    state_check = reduce_campaign_events(
        documents,
        program,
        repo_root=repo_root,
        verify_artifacts=verify_artifacts,
    )
    issues.extend(
        _issue(
            item.code,
            item.message,
            sequence=item.sequence,
            **dict(item.details),
        )
        for item in state_check.issues
    )
    return EventLedgerCheck(
        event_count=len(paths),
        last_event_sha256=previous,
        type_counts=dict(sorted(counts.items())),
        state=state_check.state,
        issues=tuple(issues),
    )


def format_event_ledger_markdown(
    check: EventLedgerCheck, program: Mapping[str, Any] | None = None
) -> str:
    """Render a compact audit status for agents and humans."""

    lines = [
        "# Research campaign event ledger",
        "",
        f"- Integrity: `{'VALID' if check.valid else 'INVALID'}`",
        f"- Events: {check.event_count}",
        f"- Chain tip: `{check.last_event_sha256}`",
        "",
        "## Event counts",
        "",
    ]
    if check.type_counts:
        lines.extend(f"- `{name}`: {count}" for name, count in check.type_counts.items())
    else:
        lines.append("- No events yet.")
    lines.extend(["", "## Integrity issues", ""])
    if check.issues:
        lines.extend(f"- `{issue.code}`: {issue.message}" for issue in check.issues)
    else:
        lines.append("- None.")
    if program is not None and check.valid:
        status = campaign_status(check.state, program)
        next_action = status["next_action"]
        lines.extend(
            [
                "",
                "## Derived state",
                "",
                f"- Lifecycle: `{status['lifecycle']}`",
                f"- Phase: `{status['phase']}`",
                f"- Active gate: `{status['active_gate_id']}`",
                f"- Hardware authorized: `{str(status['hardware_authorized']).lower()}`",
                f"- Next action: `{next_action['action_id']}`",
                f"- Next study: `{next_action['study_id']}`",
                f"- Reason: `{next_action['reason_code']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _event_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*.json") if path.is_file())


def _event_bytes(event: Mapping[str, Any]) -> bytes:
    return (json.dumps(event, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _bytes_sha256(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _load_event_documents(paths: list[Path]) -> list[tuple[Mapping[str, Any], str]]:
    return [(load_research_mapping(path), file_sha256(path)) for path in paths]


def _sha256(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _issue(code: str, message: str, **details: Any) -> EventLedgerIssue:
    return EventLedgerIssue(code=code, message=message, details=details)


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "EVENT_TYPES",
    "EventLedgerCheck",
    "EventLedgerIssue",
    "append_research_event",
    "format_event_ledger_markdown",
    "verify_research_event_ledger",
]
