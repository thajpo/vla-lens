"""Validated, deterministic human summaries for research study results."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from vla_lens.research_analysis import (
    ResearchAnalysisError,
    decision_value_by_id,
    metric_by_id,
    validate_research_analysis,
)
from vla_lens.research_child import (
    check_research_child,
    check_research_child_lock,
    child_plan_fingerprint,
    study_fingerprint,
)
from vla_lens.research_io import canonical_research_fingerprint
from vla_lens.research_plan import check_research_plan, research_plan_fingerprint

RESULT_CARD_SCHEMA_VERSION = 2
RESULT_KINDS = frozenset({"preparation_gate", "effect_estimate", "design_decision"})
VERDICTS = frozenset(
    {
        "invalid",
        "inconclusive",
        "not_applicable",
        "gate_failed",
        "gate_passed",
        "exploratory_negative",
        "exploratory_positive",
        "confirmed_negative",
        "confirmed_positive",
        "design_not_supported",
        "design_supported",
    }
)
CLAIM_TYPES = frozenset(
    {
        "preparation",
        "behavioral_response",
        "semantic_decodability",
        "internal_association",
        "local_causal_effect",
        "closed_loop_causal_effect",
        "reliability_design",
    }
)
BEHAVIOR_LEVELS = frozenset({"none", "action_chunk", "open_loop_trajectory", "closed_loop_rollout"})
CONFIRMATION_STATUSES = frozenset(
    {"preparation", "discovery", "prospective_confirmation", "design"}
)
AUDIT_STATUSES = frozenset({"pass", "warn", "fail"})
OUTCOMES = frozenset({"positive", "negative", "inconclusive", "not_applicable", "invalid"})
COMMON_RESULT_FIELDS = frozenset(
    {
        "schema_version",
        "result_card_id",
        "result_kind",
        "program_id",
        "program_fingerprint",
        "study_id",
        "study_fingerprint",
        "child_plan_id",
        "child_plan_fingerprint",
        "child_lock_id",
        "child_lock_fingerprint",
        "reservation_id",
        "ledger_tip_before_result",
        "attempt_event_range",
        "authorization_receipt",
        "attempt_ledger",
        "budget_record",
        "question",
        "one_sentence_answer",
        "verdict",
        "claim_type",
        "behavior_level",
        "confirmation_status",
        "predecessor_result_fingerprints",
        "supersedes_result_fingerprints",
        "what_changed",
        "held_fixed",
        "trial_manifest",
        "analysis_package",
        "trial_accounting",
        "supported_conclusion",
        "forbidden_conclusions",
        "strongest_surviving_alternative",
        "artifact_refs",
        "metric_ids",
        "decision",
        "audit",
    }
)
IDENTITY_REF_FIELDS = frozenset({"id", "sha256"})
ATTEMPT_RANGE_FIELDS = frozenset({"first_sequence", "last_sequence"})
TRIAL_ACCOUNTING_FIELDS = frozenset(
    {"expected", "completed", "technical_failed", "excluded", "attempts"}
)
DECISION_FIELDS = frozenset({"evaluated_gates", "derived_outcome", "next_action"})
AUDIT_FIELDS = frozenset(
    {
        "status",
        "report_id",
        "report_sha256",
        "auditor_id",
        "subject_child_fingerprint",
        "subject_child_lock_fingerprint",
        "subject_trial_manifest_sha256",
        "subject_analysis_package_sha256",
        "checks",
        "unresolved_errors",
    }
)

CLAIM_TYPE_BY_STUDY_KIND = {
    "preparation": "preparation",
    "behavior_discovery": "behavioral_response",
    "behavior_confirmation": "behavioral_response",
    "semantic_readout": "semantic_decodability",
    "semantic_confirmation": "semantic_decodability",
    "internal_discovery": "internal_association",
    "causal_discovery": "local_causal_effect",
    "causal_confirmation": "local_causal_effect",
    "behavior_intervention": "closed_loop_causal_effect",
    "reliability_design": "reliability_design",
}
BEHAVIOR_LEVEL_BY_STUDY_KIND = {
    "preparation": "closed_loop_rollout",
    "behavior_discovery": "closed_loop_rollout",
    "behavior_confirmation": "closed_loop_rollout",
    "semantic_readout": "none",
    "semantic_confirmation": "none",
    "internal_discovery": "action_chunk",
    "causal_discovery": "action_chunk",
    "causal_confirmation": "action_chunk",
    "behavior_intervention": "closed_loop_rollout",
    "reliability_design": "none",
}


class ResearchResultCardError(ValueError):
    """Raised when a result card cannot support autonomous advancement."""


def research_result_fingerprint(card: Mapping[str, Any]) -> str:
    """Return the external content ID used by later child plans."""

    return canonical_research_fingerprint(card)


def validate_research_result_card(
    card: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
    child_plan: Mapping[str, Any],
    child_lock: Mapping[str, Any],
    analysis_package: Mapping[str, Any],
    lock_receipt_sha256: str,
    audit_report_sha256: str,
    analysis_package_sha256: str,
    authorization_receipt_sha256: str,
    attempt_ledger_sha256: str,
    budget_record_sha256: str,
) -> None:
    """Reject cards that are incomplete, unbound, or internally inconsistent."""

    variant_fields = {
        "preparation_gate": {"gate_result"},
        "effect_estimate": {"primary_result", "strongest_control"},
        "design_decision": {"design_result"},
    }.get(str(card.get("result_kind") or ""), set())
    _exact_fields(card, COMMON_RESULT_FIELDS | variant_fields, "result card")
    for field in (
        "authorization_receipt",
        "attempt_ledger",
        "budget_record",
        "trial_manifest",
        "analysis_package",
    ):
        _exact_mapping(_get(card, field), IDENTITY_REF_FIELDS, field)
    _exact_mapping(_get(card, "trial_accounting"), TRIAL_ACCOUNTING_FIELDS, "trial_accounting")
    _exact_mapping(_get(card, "decision"), DECISION_FIELDS, "decision")
    _exact_mapping(_get(card, "audit"), AUDIT_FIELDS, "audit")
    _exact_mapping(_get(card, "attempt_event_range"), ATTEMPT_RANGE_FIELDS, "attempt_event_range")

    required = (
        "schema_version",
        "result_card_id",
        "result_kind",
        "program_id",
        "program_fingerprint",
        "study_id",
        "study_fingerprint",
        "child_plan_id",
        "child_plan_fingerprint",
        "child_lock_id",
        "child_lock_fingerprint",
        "reservation_id",
        "ledger_tip_before_result",
        "attempt_event_range.first_sequence",
        "attempt_event_range.last_sequence",
        "authorization_receipt.id",
        "authorization_receipt.sha256",
        "attempt_ledger.id",
        "attempt_ledger.sha256",
        "budget_record.id",
        "budget_record.sha256",
        "question",
        "one_sentence_answer",
        "verdict",
        "claim_type",
        "behavior_level",
        "confirmation_status",
        "predecessor_result_fingerprints",
        "supersedes_result_fingerprints",
        "what_changed",
        "held_fixed",
        "trial_manifest.id",
        "trial_manifest.sha256",
        "analysis_package.id",
        "analysis_package.sha256",
        "trial_accounting.expected",
        "trial_accounting.completed",
        "trial_accounting.technical_failed",
        "trial_accounting.excluded",
        "trial_accounting.attempts",
        "supported_conclusion",
        "forbidden_conclusions",
        "strongest_surviving_alternative",
        "artifact_refs",
        "metric_ids",
        "decision.evaluated_gates",
        "decision.derived_outcome",
        "decision.next_action",
        "audit.status",
        "audit.report_id",
        "audit.report_sha256",
        "audit.auditor_id",
        "audit.subject_child_fingerprint",
        "audit.subject_child_lock_fingerprint",
        "audit.subject_trial_manifest_sha256",
        "audit.subject_analysis_package_sha256",
        "audit.checks",
    )
    missing = [path for path in required if not _has_path(card, path)]
    if not _has_path(card, "audit.unresolved_errors"):
        missing.append("audit.unresolved_errors")
    if missing:
        raise ResearchResultCardError(f"Result card is missing fields: {missing}")
    if card.get("schema_version") != RESULT_CARD_SCHEMA_VERSION:
        raise ResearchResultCardError("Unsupported result-card schema")
    _enum(card, "result_kind", RESULT_KINDS)
    _enum(card, "verdict", VERDICTS)
    _enum(card, "claim_type", CLAIM_TYPES)
    _enum(card, "behavior_level", BEHAVIOR_LEVELS)
    _enum(card, "confirmation_status", CONFIRMATION_STATUSES)
    _enum(card, "audit.status", AUDIT_STATUSES)
    _enum(card, "decision.derived_outcome", OUTCOMES)
    for path in (
        "predecessor_result_fingerprints",
        "supersedes_result_fingerprints",
        "what_changed",
        "held_fixed",
        "forbidden_conclusions",
        "artifact_refs",
        "metric_ids",
        "decision.evaluated_gates",
        "audit.unresolved_errors",
    ):
        if not _is_sequence(_get(card, path)):
            raise ResearchResultCardError(f"Result-card field {path!r} must be a list")

    program_check = check_research_plan(program)
    if not program_check.valid:
        raise ResearchResultCardError("Supplied research program is invalid")
    child_check = check_research_child(child_plan, program)
    if not child_check.valid:
        raise ResearchResultCardError("Supplied child plan is invalid")
    lock_check = check_research_child_lock(child_lock, child_plan, program)
    if not lock_check.valid:
        raise ResearchResultCardError("Supplied child lock is invalid")
    study = _study(program, str(card["study_id"]))
    if study is None:
        raise ResearchResultCardError("Result-card study does not exist in the program")
    _validate_bindings(card, program, child_plan, study)
    _validate_artifacts(card)
    actual_external = {
        "authorization_receipt": authorization_receipt_sha256,
        "attempt_ledger": attempt_ledger_sha256,
        "budget_record": budget_record_sha256,
    }
    for field, observed_hash in actual_external.items():
        if observed_hash != _get(card, f"{field}.sha256"):
            raise ResearchResultCardError(f"Supplied {field} bytes do not match the result card")
        if observed_hash != analysis_package[f"{field}_sha256"]:
            raise ResearchResultCardError(
                f"Analysis {field} hash differs from the card and supplied bytes"
            )
    _validate_trials(card)
    if card["study_fingerprint"] != study_fingerprint(study):
        raise ResearchResultCardError("Result card does not match the exact study definition")
    expected_lock_fingerprint = canonical_research_fingerprint(child_lock)
    if (
        card["child_lock_id"] != child_lock.get("lock_id")
        or card["child_lock_fingerprint"] != expected_lock_fingerprint
    ):
        raise ResearchResultCardError("Result card does not match the supplied child lock")
    if card["reservation_id"] != child_lock.get("reservation_id"):
        raise ResearchResultCardError("Result card reservation differs from the child lock")
    if not _sha256(card["ledger_tip_before_result"]):
        raise ResearchResultCardError("Result card needs the exact prior event-chain tip")
    first_sequence = _positive_int(
        _get(card, "attempt_event_range.first_sequence"), "attempt range first sequence"
    )
    last_sequence = _positive_int(
        _get(card, "attempt_event_range.last_sequence"), "attempt range last sequence"
    )
    if first_sequence > last_sequence:
        raise ResearchResultCardError("Result attempt-event range is reversed")
    try:
        evaluated_gates, outcome = validate_research_analysis(
            analysis_package,
            program=program,
            child_plan=child_plan,
            child_lock_fingerprint=expected_lock_fingerprint,
        )
    except ResearchAnalysisError as exc:
        raise ResearchResultCardError(str(exc)) from exc
    if card["trial_accounting"] != analysis_package["trial_accounting"]:
        raise ResearchResultCardError("Result trial accounting differs from the analysis package")
    if card["decision"]["evaluated_gates"] != [gate.to_dict() for gate in evaluated_gates]:
        raise ResearchResultCardError("Result decision gates differ from the locked analysis")
    if card["decision"]["derived_outcome"] != outcome:
        raise ResearchResultCardError("Declared outcome does not match the structured checks")
    expected_verdict = _verdict_for(card, outcome)
    if card["verdict"] != expected_verdict:
        raise ResearchResultCardError(
            f"Verdict must be derived as {expected_verdict!r}, got {card['verdict']!r}"
        )
    expected_action = _get(study, f"outcome_actions.{outcome}")
    if _get(card, "decision.next_action") != expected_action:
        raise ResearchResultCardError("Next action does not match the program outcome action")
    _validate_audit(
        card,
        child_plan,
        outcome,
        child_lock=child_lock,
        lock_receipt_sha256=lock_receipt_sha256,
        audit_report_sha256=audit_report_sha256,
        analysis_package_sha256=analysis_package_sha256,
    )

    result_kind = str(card["result_kind"])
    if result_kind == "effect_estimate":
        _validate_effect_result(card, child_plan, analysis_package)
    elif result_kind == "preparation_gate":
        _validate_preparation_gate(card, analysis_package)
    else:
        _validate_design_decision(card)


def format_research_result_markdown(
    card: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
    child_plan: Mapping[str, Any],
    child_lock: Mapping[str, Any],
    analysis_package: Mapping[str, Any],
    lock_receipt_sha256: str,
    audit_report_sha256: str,
    analysis_package_sha256: str,
    authorization_receipt_sha256: str,
    attempt_ledger_sha256: str,
    budget_record_sha256: str,
) -> str:
    """Render the fixed human summary only after full contract validation."""

    validate_research_result_card(
        card,
        program=program,
        child_plan=child_plan,
        child_lock=child_lock,
        analysis_package=analysis_package,
        lock_receipt_sha256=lock_receipt_sha256,
        audit_report_sha256=audit_report_sha256,
        analysis_package_sha256=analysis_package_sha256,
        authorization_receipt_sha256=authorization_receipt_sha256,
        attempt_ledger_sha256=attempt_ledger_sha256,
        budget_record_sha256=budget_record_sha256,
    )
    audit = dict(card["audit"])
    trials = dict(card["trial_accounting"])
    lines = [
        f"# Result: {card['study_id']}",
        "",
        str(card["one_sentence_answer"]),
        "",
        f"- Question: {card['question']}",
        f"- Verdict: `{card['verdict']}`",
        f"- Claim: `{card['claim_type']}` / `{card['behavior_level']}`",
        f"- Stage: `{card['confirmation_status']}`",
        f"- Audit: `{audit['status']}` (`{audit['report_id']}`)",
        f"- Result ID: `{card['result_card_id']}` / `{research_result_fingerprint(card)}`",
        f"- Program: `{card['program_id']}` / `{card['program_fingerprint']}`",
        f"- Child: `{card['child_plan_id']}` / `{card['child_plan_fingerprint']}`",
        "",
        "## Comparison",
        "",
        "Changed: " + _joined(card["what_changed"]),
        "",
        "Held fixed: " + _joined(card["held_fixed"]),
        "",
    ]
    if card["result_kind"] == "effect_estimate":
        lines.extend(_effect_lines(card))
    elif card["result_kind"] == "preparation_gate":
        lines.extend(_preparation_lines(card))
    else:
        lines.extend(_design_lines(card))
    lines.extend(
        [
            "",
            f"Trials: {trials['completed']} completed, {trials['technical_failed']} "
            f"technical failures, {trials['excluded']} excluded, {trials['expected']} "
            f"expected; {trials['attempts']} append-only attempts.",
            "",
            "## Interpretation",
            "",
            f"Supports: {card['supported_conclusion']}",
            "",
            "Does not support: " + _joined(card["forbidden_conclusions"]),
            "",
            f"Strongest remaining alternative: {card['strongest_surviving_alternative']}",
            "",
            f"Next action: `{card['decision']['next_action']}`",
            "",
            "## Evidence",
            "",
            "Artifacts: " + _artifact_summary(card["artifact_refs"]),
            "",
            "Metrics: " + _code_joined(card["metric_ids"]),
        ]
    )
    if audit.get("unresolved_errors"):
        lines.extend(["", "Unresolved audit errors: " + _joined(audit["unresolved_errors"])])
    return "\n".join(lines) + "\n"


def _validate_bindings(
    card: Mapping[str, Any],
    program: Mapping[str, Any],
    child: Mapping[str, Any],
    study: Mapping[str, Any],
) -> None:
    expected = {
        "program_id": program.get("program_id"),
        "program_fingerprint": research_plan_fingerprint(program),
        "child_plan_id": child.get("child_plan_id"),
        "child_plan_fingerprint": child_plan_fingerprint(child),
        "study_id": _get(child, "study.id"),
        "question": study.get("question"),
        "result_kind": _get(child, "claim.result_kind"),
    }
    for field, value in expected.items():
        if card.get(field) != value:
            raise ResearchResultCardError(f"Result card does not match locked field {field!r}")
    kind = str(study.get("kind"))
    if card["claim_type"] != CLAIM_TYPE_BY_STUDY_KIND.get(kind):
        raise ResearchResultCardError("Claim type is incompatible with the study kind")
    if card["behavior_level"] != BEHAVIOR_LEVEL_BY_STUDY_KIND.get(kind):
        raise ResearchResultCardError("Behavior level is incompatible with the study kind")
    expected_status = _confirmation_status(kind)
    if card["confirmation_status"] != expected_status:
        raise ResearchResultCardError("Confirmation status is incompatible with the study kind")
    child_predecessors = {
        str(item.get("event_sha256"))
        for item in _sequence(child.get("predecessor_result_events"))
        if isinstance(item, Mapping)
    }
    if set(str(item) for item in card["predecessor_result_fingerprints"]) != child_predecessors:
        raise ResearchResultCardError("Result predecessor hashes do not match the child plan")
    if set(card["forbidden_conclusions"]) < set(_get(child, "claim.forbidden_conclusions")):
        raise ResearchResultCardError("Result card drops a forbidden conclusion")
    if card["supported_conclusion"] not in set(_get(child, "claim.allowed_conclusions")):
        raise ResearchResultCardError("Supported conclusion is not authorized by the child")
    if card["trial_manifest"] != {
        "id": _get(child, "trials.manifest.id"),
        "sha256": _get(child, "trials.manifest.sha256"),
    }:
        raise ResearchResultCardError("Result trial manifest does not match the child plan")


def _validate_artifacts(card: Mapping[str, Any]) -> None:
    refs = card["artifact_refs"]
    if not refs:
        raise ResearchResultCardError("Result card needs typed evidence artifacts")
    ids: set[str] = set()
    for reference in refs:
        if not isinstance(reference, Mapping):
            raise ResearchResultCardError("Artifact references must be mappings")
        required = {"id", "type", "uri", "sha256"}
        if set(reference) != required:
            raise ResearchResultCardError(
                "Artifact references require exactly id, type, uri, sha256"
            )
        if not _sha256(reference["sha256"]):
            raise ResearchResultCardError("Artifact reference has an invalid sha256")
        if reference["id"] in ids:
            raise ResearchResultCardError("Artifact IDs must be unique")
        ids.add(str(reference["id"]))
    for field in (
        "trial_manifest",
        "analysis_package",
        "authorization_receipt",
        "attempt_ledger",
        "budget_record",
    ):
        if not _sha256(_get(card, f"{field}.sha256")):
            raise ResearchResultCardError(f"{field} needs a full sha256")
    by_id = {str(reference["id"]): reference for reference in refs}
    for field in (
        "analysis_package",
        "authorization_receipt",
        "attempt_ledger",
        "budget_record",
        "audit",
    ):
        identifier = str(_get(card, f"{field}.id") or _get(card, f"{field}.report_id") or "")
        expected_hash = _get(card, f"{field}.sha256") or _get(card, f"{field}.report_sha256")
        if identifier not in by_id or by_id[identifier]["sha256"] != expected_hash:
            raise ResearchResultCardError(f"{field} must have a matching typed artifact reference")
    evidence_ids = {
        str(check["evidence_artifact_id"])
        for check in _get(card, "decision.evaluated_gates")
        if isinstance(check, Mapping)
    }
    if not evidence_ids <= set(by_id):
        raise ResearchResultCardError("Decision checks cite unknown evidence artifacts")


def _validate_trials(card: Mapping[str, Any]) -> None:
    counts = {
        name: _nonnegative_int(_get(card, f"trial_accounting.{name}"), name)
        for name in ("expected", "completed", "technical_failed", "excluded", "attempts")
    }
    if counts["completed"] + counts["technical_failed"] + counts["excluded"] != counts["expected"]:
        raise ResearchResultCardError("Completed, failed, and excluded trials must equal expected")
    if counts["attempts"] < counts["completed"] + counts["technical_failed"]:
        raise ResearchResultCardError("Attempt count cannot hide completed or failed attempts")


def _derive_outcome(card: Mapping[str, Any]) -> str:
    integrity = _check_values(_get(card, "decision.integrity_checks"), require_nonempty=True)
    applicability = _check_values(_get(card, "decision.applicability_checks"))
    positive = _check_values(_get(card, "decision.positive_checks"), require_nonempty=True)
    negative = _check_values(_get(card, "decision.negative_checks"), require_nonempty=True)
    if not all(integrity):
        return "invalid"
    if applicability and not all(applicability):
        return "not_applicable"
    positive_pass = all(positive)
    negative_pass = all(negative)
    if positive_pass and negative_pass:
        raise ResearchResultCardError("Positive and negative gates cannot both pass")
    if positive_pass:
        return "positive"
    if negative_pass:
        return "negative"
    return "inconclusive"


def _check_values(value: Any, *, require_nonempty: bool = False) -> list[bool]:
    if not _is_sequence(value) or (require_nonempty and not value):
        raise ResearchResultCardError("Decision checks are missing")
    results: list[bool] = []
    for check in value:
        if not isinstance(check, Mapping) or set(check) != {
            "id",
            "passed",
            "evidence_artifact_id",
        }:
            raise ResearchResultCardError(
                "Decision checks require exactly id, passed, and evidence_artifact_id"
            )
        if not isinstance(check["passed"], bool):
            raise ResearchResultCardError("Decision-check passed values must be booleans")
        results.append(check["passed"])
    return results


def _verdict_for(card: Mapping[str, Any], outcome: str) -> str:
    if outcome in {"invalid", "inconclusive", "not_applicable"}:
        return outcome
    if card["result_kind"] == "preparation_gate":
        return "gate_passed" if outcome == "positive" else "gate_failed"
    if card["result_kind"] == "design_decision":
        return "design_supported" if outcome == "positive" else "design_not_supported"
    prefix = (
        "confirmed" if card["confirmation_status"] == "prospective_confirmation" else "exploratory"
    )
    return f"{prefix}_{outcome}"


def _validate_audit(
    card: Mapping[str, Any],
    child: Mapping[str, Any],
    outcome: str,
    *,
    child_lock: Mapping[str, Any],
    lock_receipt_sha256: str,
    audit_report_sha256: str,
    analysis_package_sha256: str,
) -> None:
    audit = card["audit"]
    unresolved = list(audit["unresolved_errors"])
    if outcome != "invalid" and (audit["status"] != "pass" or unresolved):
        raise ResearchResultCardError("A promotable result requires a clean passing audit")
    if audit["status"] == "pass" and unresolved:
        raise ResearchResultCardError("A passing audit cannot list unresolved errors")
    expected = {
        "subject_child_fingerprint": child_plan_fingerprint(child),
        "subject_child_lock_fingerprint": canonical_research_fingerprint(child_lock),
        "subject_trial_manifest_sha256": _get(card, "trial_manifest.sha256"),
        "subject_analysis_package_sha256": _get(card, "analysis_package.sha256"),
    }
    for field, value in expected.items():
        if audit.get(field) != value:
            raise ResearchResultCardError(f"Audit is not bound to {field}")
    if audit.get("auditor_id") == child.get("prepared_by"):
        raise ResearchResultCardError("Result auditor must differ from the child preparer")
    if not _sha256(audit.get("report_sha256")):
        raise ResearchResultCardError("Audit report needs a full sha256")
    checks = audit.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != {
        "execution",
        "calculation",
        "claim",
    }:
        raise ResearchResultCardError("Audit checks must cover execution, calculation, and claim")
    if outcome != "invalid" and set(checks.values()) != {"pass"}:
        raise ResearchResultCardError("Every required result audit must pass")
    if audit_report_sha256 != audit["report_sha256"]:
        raise ResearchResultCardError("Supplied audit-report bytes do not match the result card")
    if analysis_package_sha256 != _get(card, "analysis_package.sha256"):
        raise ResearchResultCardError(
            "Supplied analysis-package bytes do not match the result card"
        )
    lock_id = str(child_lock.get("lock_id") or "")
    refs = {str(item["id"]): item for item in card["artifact_refs"]}
    if lock_id not in refs or refs[lock_id]["sha256"] != lock_receipt_sha256:
        raise ResearchResultCardError("Supplied lock-receipt bytes lack a matching artifact")


def _validate_effect_result(
    card: Mapping[str, Any], child: Mapping[str, Any], analysis: Mapping[str, Any]
) -> None:
    _exact_mapping(
        card.get("primary_result"),
        frozenset(
            {
                "metric_id",
                "estimate",
                "unit",
                "null_value",
                "minimum_useful_effect",
                "interval",
                "independent_units",
            }
        ),
        "primary_result",
    )
    _exact_mapping(
        card.get("strongest_control"),
        frozenset({"metric_id", "name", "estimate", "unit", "interval", "source_artifact_id"}),
        "strongest_control",
    )
    required = (
        "primary_result.metric_id",
        "primary_result.estimate",
        "primary_result.unit",
        "primary_result.null_value",
        "primary_result.minimum_useful_effect",
        "primary_result.interval.low",
        "primary_result.interval.high",
        "primary_result.interval.method",
        "primary_result.interval.level",
        "primary_result.interval.grouping_unit",
        "primary_result.interval.replicates",
        "primary_result.interval.seed",
        "primary_result.interval.source_artifact_id",
        "primary_result.independent_units.task_families",
        "primary_result.independent_units.scene_clusters",
        "primary_result.independent_units.noise_repeats",
        "primary_result.independent_units.rollouts",
        "strongest_control.metric_id",
        "strongest_control.name",
        "strongest_control.estimate",
        "strongest_control.unit",
        "strongest_control.interval.low",
        "strongest_control.interval.high",
        "strongest_control.interval.method",
        "strongest_control.interval.level",
        "strongest_control.interval.grouping_unit",
        "strongest_control.interval.replicates",
        "strongest_control.interval.seed",
        "strongest_control.source_artifact_id",
    )
    missing = [path for path in required if not _has_path(card, path)]
    if missing:
        raise ResearchResultCardError(f"Effect result is missing fields: {missing}")
    primary = card["primary_result"]
    interval = primary["interval"]
    _exact_mapping(
        interval,
        frozenset(
            {
                "low",
                "high",
                "method",
                "level",
                "grouping_unit",
                "replicates",
                "seed",
                "source_artifact_id",
            }
        ),
        "primary_result.interval",
    )
    _exact_mapping(
        primary["independent_units"],
        frozenset({"task_families", "scene_clusters", "noise_repeats", "rollouts"}),
        "primary_result.independent_units",
    )
    estimate = _number(primary["estimate"], "primary estimate")
    low = _number(interval["low"], "interval low")
    high = _number(interval["high"], "interval high")
    if not low <= estimate <= high:
        raise ResearchResultCardError("Primary estimate must lie inside its interval")
    level = _number(interval["level"], "interval level")
    if not 0 < level < 1:
        raise ResearchResultCardError("Interval level must lie between zero and one")
    _positive_int(interval["replicates"], "interval replicates")
    _nonnegative_int(interval["seed"], "interval seed")
    if primary["metric_id"] != _get(child, "measurement.primary.metric_id"):
        raise ResearchResultCardError("Primary metric does not match the child")
    if primary["unit"] != _get(child, "measurement.primary.unit"):
        raise ResearchResultCardError("Primary unit does not match the child")
    if primary["minimum_useful_effect"] != _get(child, "measurement.primary.minimum_useful_effect"):
        raise ResearchResultCardError("Minimum useful effect does not match the child")
    control = card["strongest_control"]
    _exact_mapping(
        control["interval"],
        frozenset({"low", "high", "method", "level", "grouping_unit", "replicates", "seed"}),
        "strongest_control.interval",
    )
    control_estimate = _number(control["estimate"], "control estimate")
    control_low = _number(control["interval"]["low"], "control interval low")
    control_high = _number(control["interval"]["high"], "control interval high")
    if not control_low <= control_estimate <= control_high:
        raise ResearchResultCardError("Control estimate must lie inside its interval")
    if control["unit"] != primary["unit"]:
        raise ResearchResultCardError("Strongest control must use the primary-result unit")
    if control["metric_id"] != _get(child, "measurement.strongest_control_metric_id"):
        raise ResearchResultCardError("Strongest control metric does not match the child")
    metrics = {str(item) for item in card["metric_ids"]}
    if primary["metric_id"] not in metrics or control["metric_id"] not in metrics:
        raise ResearchResultCardError("Primary and control metrics must appear in metric_ids")
    artifact_ids = {str(item["id"]) for item in card["artifact_refs"]}
    if (
        interval["source_artifact_id"] not in artifact_ids
        or control["source_artifact_id"] not in artifact_ids
    ):
        raise ResearchResultCardError("Metric intervals must cite typed evidence artifacts")
    primary_analysis = metric_by_id(analysis, str(primary["metric_id"]))
    control_analysis = metric_by_id(analysis, str(control["metric_id"]))
    for result, source, label in (
        (primary, primary_analysis, "primary"),
        (control, control_analysis, "control"),
    ):
        for field in ("metric_id", "estimate", "unit"):
            if result[field] != source[field]:
                raise ResearchResultCardError(
                    f"Result {label} field {field!r} differs from the analysis package"
                )
    primary_interval = {
        key: value for key, value in primary["interval"].items() if key != "source_artifact_id"
    }
    if (
        primary_interval != primary_analysis["interval"]
        or primary["interval"]["source_artifact_id"] != primary_analysis["source_artifact_id"]
    ):
        raise ResearchResultCardError("Primary interval differs from the analysis package")
    if (
        control["interval"] != control_analysis["interval"]
        or control["source_artifact_id"] != control_analysis["source_artifact_id"]
    ):
        raise ResearchResultCardError("Control interval differs from the analysis package")
    if primary["independent_units"] != primary_analysis["independent_units"]:
        raise ResearchResultCardError(
            "Result evidence-unit counts differ from the analysis package"
        )
    counts = {
        name: _nonnegative_int(primary["independent_units"][name], name)
        for name in ("task_families", "scene_clusters", "noise_repeats", "rollouts")
    }
    if card["verdict"] not in {"invalid", "inconclusive", "not_applicable"} and (
        counts["task_families"] <= 0 or counts["scene_clusters"] <= 0
    ):
        raise ResearchResultCardError("Scientific effects require positive independent-unit counts")
    if card["behavior_level"] == "closed_loop_rollout" and counts["rollouts"] <= 0:
        raise ResearchResultCardError("Closed-loop evidence requires positive rollout count")


def _validate_preparation_gate(card: Mapping[str, Any], analysis: Mapping[str, Any]) -> None:
    _exact_mapping(
        card.get("gate_result"),
        frozenset(
            {
                "discovery_eligible",
                "discovery_total",
                "confirmation_eligible",
                "confirmation_total",
                "gate_passed",
                "selection_rule",
                "diagnostics",
            }
        ),
        "gate_result",
    )
    required = (
        "gate_result.discovery_eligible",
        "gate_result.discovery_total",
        "gate_result.confirmation_eligible",
        "gate_result.confirmation_total",
        "gate_result.gate_passed",
        "gate_result.selection_rule",
        "gate_result.diagnostics",
    )
    missing = [path for path in required if not _has_path(card, path)]
    if missing:
        raise ResearchResultCardError(f"Preparation result is missing fields: {missing}")
    discovery = _nonnegative_int(_get(card, "gate_result.discovery_eligible"), "discovery eligible")
    discovery_total = _positive_int(_get(card, "gate_result.discovery_total"), "discovery total")
    confirmation = _nonnegative_int(
        _get(card, "gate_result.confirmation_eligible"), "confirmation eligible"
    )
    confirmation_total = _positive_int(
        _get(card, "gate_result.confirmation_total"), "confirmation total"
    )
    if discovery > discovery_total or confirmation > confirmation_total:
        raise ResearchResultCardError("Eligible family counts cannot exceed pool totals")
    passed = discovery >= 6 and confirmation >= 6
    if _get(card, "gate_result.gate_passed") is not passed:
        raise ResearchResultCardError(
            "Preparation gate flag disagrees with the locked six-and-six rule"
        )
    if card["verdict"] == "gate_passed" and not passed:
        raise ResearchResultCardError("Preparation verdict disagrees with its gate counts")
    expected_values = {
        "discovery_eligible": "discovery_eligible_count",
        "confirmation_eligible": "confirmation_eligible_count",
        "discovery_total": "discovery_total_count",
        "confirmation_total": "confirmation_total_count",
    }
    for card_field, value_id in expected_values.items():
        observed = decision_value_by_id(analysis, value_id)["value"]
        if _get(card, f"gate_result.{card_field}") != observed:
            raise ResearchResultCardError(
                f"Preparation count {card_field!r} differs from the analysis package"
            )


def _validate_design_decision(card: Mapping[str, Any]) -> None:
    if not _has_path(card, "design_result.decision") or not _has_path(
        card, "design_result.supporting_counts"
    ):
        raise ResearchResultCardError("Design result needs a decision and supporting counts")


def _effect_lines(card: Mapping[str, Any]) -> list[str]:
    primary = card["primary_result"]
    interval = primary["interval"]
    units = primary["independent_units"]
    control = card["strongest_control"]
    return [
        "## Primary result",
        "",
        f"`{primary['metric_id']}` = {_display(primary['estimate'])} {primary['unit']} "
        f"({100 * float(interval['level']):.1f}% {interval['method']} interval "
        f"{_display(interval['low'])} to {_display(interval['high'])}, grouped by "
        f"{interval['grouping_unit']}).",
        "",
        f"Strongest control: `{control['metric_id']}` ({control['name']}) = "
        f"{_display(control['estimate'])} {control['unit']}.",
        "",
        f"Independent evidence: {units['task_families']} task-object families and "
        f"{units['scene_clusters']} scene clusters. Nested noise/rollout counts: "
        f"{units['noise_repeats']} / {units['rollouts']}.",
    ]


def _preparation_lines(card: Mapping[str, Any]) -> list[str]:
    gate = card["gate_result"]
    return [
        "## Preparation gate",
        "",
        f"Discovery pool: {gate['discovery_eligible']} of {gate['discovery_total']} eligible. "
        f"Confirmation pool: {gate['confirmation_eligible']} of "
        f"{gate['confirmation_total']} eligible.",
        "",
        f"Six-and-six gate passed: `{str(gate['gate_passed']).lower()}`.",
        "",
        f"Selection rule: {gate['selection_rule']}",
        "",
        "Diagnostics: " + _joined(gate["diagnostics"]),
    ]


def _design_lines(card: Mapping[str, Any]) -> list[str]:
    result = card["design_result"]
    return [
        "## Design decision",
        "",
        str(result["decision"]),
        "",
        "Supporting counts: " + str(result["supporting_counts"]),
    ]


def _confirmation_status(study_kind: str) -> str:
    if study_kind == "preparation":
        return "preparation"
    if study_kind == "reliability_design":
        return "design"
    if study_kind in {"behavior_confirmation", "semantic_confirmation", "causal_confirmation"}:
        return "prospective_confirmation"
    return "discovery"


def _study(program: Mapping[str, Any], study_id: str) -> Mapping[str, Any] | None:
    for study in _sequence(program.get("studies")):
        if isinstance(study, Mapping) and study.get("id") == study_id:
            return study
    return None


def _enum(card: Mapping[str, Any], path: str, allowed: frozenset[str]) -> None:
    value = str(_get(card, path) or "")
    if value not in allowed:
        raise ResearchResultCardError(f"Field {path!r} must be one of {sorted(allowed)}")


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ResearchResultCardError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ResearchResultCardError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ResearchResultCardError(f"{label} must be finite")
    return parsed


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ResearchResultCardError(f"{label} must be a nonnegative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ResearchResultCardError(f"{label} must be a nonnegative integer") from exc
    if parsed < 0 or parsed != float(value):
        raise ResearchResultCardError(f"{label} must be a nonnegative integer")
    return parsed


def _positive_int(value: Any, label: str) -> int:
    parsed = _nonnegative_int(value, label)
    if parsed <= 0:
        raise ResearchResultCardError(f"{label} must be positive")
    return parsed


def _sha256(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _get(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _has_path(payload: Mapping[str, Any], path: str) -> bool:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return False
        value = value[part]
    return True


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _sequence(value: Any) -> Sequence[Any]:
    return value if _is_sequence(value) else ()


def _joined(value: Any) -> str:
    return "; ".join(str(item) for item in value) if _is_sequence(value) else str(value)


def _code_joined(value: Any) -> str:
    return ", ".join(f"`{item}`" for item in value) if _is_sequence(value) else f"`{value}`"


def _artifact_summary(value: Any) -> str:
    if not _is_sequence(value):
        return str(value)
    return ", ".join(f"`{item['id']}` ({item['type']})" for item in value)


def _display(value: Any) -> str:
    return f"{float(value):.6g}"


def _exact_fields(
    value: Mapping[str, Any], expected: frozenset[str] | set[str], label: str
) -> None:
    observed = set(value)
    if observed != set(expected):
        raise ResearchResultCardError(
            f"{label} fields differ: missing={sorted(set(expected) - observed)}, "
            f"unknown={sorted(observed - set(expected))}"
        )


def _exact_mapping(value: Any, expected: frozenset[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise ResearchResultCardError(f"{label} must be a mapping")
    _exact_fields(value, expected, label)


__all__ = [
    "AUDIT_STATUSES",
    "BEHAVIOR_LEVELS",
    "CLAIM_TYPES",
    "CONFIRMATION_STATUSES",
    "RESULT_CARD_SCHEMA_VERSION",
    "RESULT_KINDS",
    "VERDICTS",
    "ResearchResultCardError",
    "format_research_result_markdown",
    "research_result_fingerprint",
    "validate_research_result_card",
]
