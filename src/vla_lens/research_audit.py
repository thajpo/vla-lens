"""Strict audit-report contracts for autonomous research campaigns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

AUDIT_REPORT_SCHEMA_VERSION = 1
AUDIT_REPORT_KIND = "vla_lens.research_audit"
AUDIT_REPORT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "audit_id",
        "audit_type",
        "subject_kind",
        "subject_fingerprint",
        "auditor_id",
        "created_utc",
        "verdict",
        "checks",
        "unresolved_errors",
    }
)
AUDIT_CHECK_FIELDS = frozenset({"id", "status", "evidence_refs"})
ARTIFACT_FIELDS = frozenset({"id", "type", "path", "sha256"})
VERDICTS = frozenset({"pass", "warn", "fail"})
SUBJECT_KINDS = frozenset(
    {
        "research_program",
        "child_plan",
        "child_execution",
        "analysis_package",
        "result_card",
    }
)


@dataclass(frozen=True, slots=True)
class ResearchAuditError(ValueError):
    code: str
    message: str
    details: Mapping[str, Any]

    def __str__(self) -> str:
        return self.message


def validate_research_audit_report(
    report: Mapping[str, Any],
    *,
    audit_id: str | None = None,
    audit_type: str | None = None,
    subject_kind: str | None = None,
    subject_fingerprint: str | None = None,
    auditor_id: str | None = None,
    verdict: str | None = None,
    checks: Sequence[Mapping[str, Any]] | None = None,
    unresolved_errors: Sequence[str] | None = None,
) -> None:
    """Validate report bytes and optionally bind them to duplicated envelope fields."""

    if set(report) != AUDIT_REPORT_FIELDS:
        _raise(
            "audit_report_fields_mismatch",
            "Audit report fields must match exactly",
            missing=sorted(AUDIT_REPORT_FIELDS - set(report)),
            unknown=sorted(set(report) - AUDIT_REPORT_FIELDS),
        )
    if (
        report["schema_version"] != AUDIT_REPORT_SCHEMA_VERSION
        or report["kind"] != AUDIT_REPORT_KIND
    ):
        _raise("invalid_audit_report_schema", "Audit report schema is unsupported")
    if report["subject_kind"] not in SUBJECT_KINDS:
        _raise("invalid_audit_subject_kind", "Audit report subject kind is invalid")
    if report["verdict"] not in VERDICTS:
        _raise("invalid_audit_verdict", "Audit report verdict is invalid")
    for name in ("audit_id", "audit_type", "subject_fingerprint", "auditor_id", "created_utc"):
        if not str(report[name] or ""):
            _raise("missing_audit_identity", "Audit report identity is incomplete", field=name)
    if not _is_sequence(report["checks"]) or not report["checks"]:
        _raise("invalid_audit_checks", "Audit report needs at least one typed check")
    for index, check in enumerate(report["checks"]):
        if not isinstance(check, Mapping) or set(check) != AUDIT_CHECK_FIELDS:
            _raise(
                "invalid_audit_check",
                "Audit checks need exactly id, status, and evidence_refs",
                index=index,
            )
        if not str(check["id"] or "") or check["status"] not in VERDICTS:
            _raise("invalid_audit_check", "Audit check identity or status is invalid", index=index)
        if not _is_sequence(check["evidence_refs"]):
            _raise(
                "invalid_audit_evidence", "Audit check evidence_refs must be a list", index=index
            )
        if report["verdict"] == "pass" and not check["evidence_refs"]:
            _raise(
                "missing_audit_evidence",
                "Every passing audit check needs at least one immutable evidence reference",
                index=index,
            )
        for ref_index, reference in enumerate(check["evidence_refs"]):
            _validate_artifact_ref(reference, index=index, ref_index=ref_index)
    if not _is_sequence(report["unresolved_errors"]):
        _raise("invalid_audit_errors", "Audit unresolved_errors must be a list")
    if any(not isinstance(item, str) or not item for item in report["unresolved_errors"]):
        _raise("invalid_audit_errors", "Audit errors must be nonempty strings")
    if report["verdict"] == "pass" and report["unresolved_errors"]:
        _raise("passing_audit_has_errors", "A passing audit cannot have unresolved errors")
    if report["verdict"] == "pass" and any(item["status"] != "pass" for item in report["checks"]):
        _raise(
            "passing_audit_has_failed_checks",
            "A passing audit requires every declared check to pass",
        )
    if report["audit_type"] == "result" and {str(item["id"]) for item in report["checks"]} != {
        "execution",
        "calculation",
        "claim",
    }:
        _raise(
            "result_audit_checks_missing",
            "Result audit must check execution, calculation, and claim",
        )
    expected = {
        "audit_id": audit_id,
        "audit_type": audit_type,
        "subject_kind": subject_kind,
        "subject_fingerprint": subject_fingerprint,
        "auditor_id": auditor_id,
        "verdict": verdict,
    }
    for name, value in expected.items():
        if value is not None and report[name] != value:
            _raise(
                "audit_report_binding_mismatch",
                "Audit report bytes disagree with their envelope",
                field=name,
                expected=value,
                observed=report[name],
            )
    if checks is not None:
        summaries = [{"id": item["id"], "status": item["status"]} for item in report["checks"]]
        if summaries != list(checks):
            _raise(
                "audit_report_checks_mismatch",
                "Audit report checks disagree with their event summary",
            )
    if unresolved_errors is not None and list(report["unresolved_errors"]) != list(
        unresolved_errors
    ):
        _raise(
            "audit_report_errors_mismatch",
            "Audit report errors disagree with their event summary",
        )


def _validate_artifact_ref(value: Any, **details: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != ARTIFACT_FIELDS:
        _raise("invalid_audit_evidence", "Audit evidence reference has the wrong fields", **details)
    path = PurePosixPath(str(value["path"] or ""))
    if not str(value["id"] or "") or not str(value["type"] or ""):
        _raise("invalid_audit_evidence", "Audit evidence identity is missing", **details)
    if not str(value["sha256"] or "").startswith("sha256:") or len(str(value["sha256"])) != 71:
        _raise("invalid_audit_evidence", "Audit evidence hash is invalid", **details)
    if not str(value["path"] or "") or path.is_absolute() or ".." in path.parts:
        _raise("invalid_audit_evidence", "Audit evidence path must be repo-relative", **details)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _raise(code: str, message: str, **details: Any) -> None:
    raise ResearchAuditError(code, message, details)


__all__ = [
    "AUDIT_REPORT_KIND",
    "AUDIT_REPORT_SCHEMA_VERSION",
    "ResearchAuditError",
    "validate_research_audit_report",
]
