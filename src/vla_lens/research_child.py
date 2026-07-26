"""Contracts for one locked, executable child of a research program."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from vla_lens.research_audit import ResearchAuditError, validate_research_audit_report
from vla_lens.research_io import (
    StrictResearchDataError,
    canonical_research_fingerprint,
    file_sha256,
    load_research_mapping,
)
from vla_lens.research_plan import research_plan_fingerprint
from vla_lens.research_state import (
    BASE_LOCK_AUDITS,
    EVENT_REF_FIELDS,
    CampaignState,
    child_authorization_issues,
)

CHILD_SCHEMA_VERSION = 1
CHILD_KIND = "vla_lens.research_child"
LOCK_SCHEMA_VERSION = 1
LOCK_KIND = "vla_lens.research_child_lock"
RESULT_KINDS = frozenset({"preparation_gate", "effect_estimate", "design_decision"})
VERDICTS = frozenset(
    {
        "exploratory_positive",
        "exploratory_negative",
        "confirmed_positive",
        "confirmed_negative",
        "gate_passed",
        "gate_failed",
        "design_supported",
        "design_not_supported",
        "inconclusive",
        "not_applicable",
        "invalid",
    }
)
POSITIVE_VERDICTS = frozenset(
    {"exploratory_positive", "confirmed_positive", "gate_passed", "design_supported"}
)
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "child_plan_id",
        "revision",
        "prepared_by",
        "program",
        "study",
        "predecessor_result_events",
        "claim",
        "cohort",
        "trials",
        "measurement",
        "decision",
        "runtime",
        "budget",
        "output",
        "completion",
        "protocol_lock",
    }
)
ARTIFACT_FIELDS = frozenset({"id", "type", "path", "sha256"})
LOCK_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "lock_id",
        "program_id",
        "program_fingerprint",
        "study_id",
        "study_fingerprint",
        "child_plan_id",
        "child_plan_fingerprint",
        "manifest_commit",
        "locked_utc",
        "prepared_by",
        "reservation_id",
        "prior_ledger_tip",
        "audits",
    }
)
BUDGET_FIELDS = frozenset(
    {
        "max_model_calls",
        "max_action_generations",
        "max_full_rollouts",
        "max_simulator_steps",
        "max_probe_fits",
        "max_persistent_gb",
        "max_ephemeral_gb",
        "min_free_space_gb",
    }
)
NESTED_FIELDS: Mapping[str, frozenset[str]] = {
    "program": frozenset({"path", "program_id", "fingerprint"}),
    "study": frozenset({"id", "fingerprint", "phase"}),
    "claim": frozenset({"result_kind", "question", "allowed_conclusions", "forbidden_conclusions"}),
    "cohort": frozenset(
        {
            "family_pool",
            "pool_phase",
            "requires_gate",
            "read_namespaces",
            "write_namespace",
            "selection_allowed",
            "manifest",
            "exposure_log",
        }
    ),
    "trials": frozenset(
        {
            "manifest",
            "expected_count",
            "stable_id_fields",
            "seed_domains",
            "expected_independent_units",
        }
    ),
    "trials.expected_independent_units": frozenset(
        {"task_families", "scene_clusters", "noise_repeats", "rollouts"}
    ),
    "measurement": frozenset({"primary", "controls", "strongest_control_metric_id", "inference"}),
    "measurement.primary": frozenset(
        {
            "metric_id",
            "formula",
            "implementation_id",
            "unit",
            "direction",
            "minimum_useful_effect",
        }
    ),
    "measurement.inference": frozenset({"method", "level", "grouping_unit", "replicates", "seed"}),
    "decision": frozenset(
        {
            "gate_components",
            "positive_combiner",
            "negative_combiner",
            "inconclusive_rule",
            "invalid_conditions",
        }
    ),
    "runtime": frozenset({"model", "environment", "code", "runner"}),
    "runtime.model": frozenset({"repo_id", "revision", "snapshot_manifest_sha256"}),
    "runtime.environment": frozenset(
        {
            "backend",
            "package_receipt",
            "camera_config_sha256",
            "controller_config_sha256",
            "preprocessor_config_sha256",
            "postprocessor_config_sha256",
        }
    ),
    "runtime.code": frozenset({"implementation_commit", "source_tree_sha256"}),
    "runtime.runner": frozenset({"entrypoint", "argv", "config"}),
    "budget": BUDGET_FIELDS,
    "output": frozenset({"root", "namespace", "attempt_ledger", "required_artifact_types"}),
    "completion": frozenset(
        {"valid_trial_statuses", "technical_retry_rule", "resume_identity_fields"}
    ),
    "protocol_lock": frozenset({"required_lock_fields", "locked_choices"}),
}
GATE_COMPONENT_FIELDS = frozenset(
    {"id", "role", "value_id", "operator", "threshold", "unit", "evidence_artifact_type"}
)
GATE_ROLES = frozenset({"integrity", "applicability", "positive", "negative"})
GATE_OPERATORS = frozenset(
    {"greater", "greater_than_or_equal", "less", "less_than_or_equal", "equal", "not_equal"}
)


@dataclass(frozen=True, slots=True)
class ContractIssue:
    """One machine-readable child-contract problem."""

    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": dict(self.details)}


@dataclass(frozen=True, slots=True)
class ChildPlanCheck:
    """Structural and optional file-integrity result for one child plan."""

    fingerprint: str | None
    issues: tuple[ContractIssue, ...]
    files_verified: bool

    @property
    def valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "research_child_plan_check",
            "valid": self.valid,
            "fingerprint": self.fingerprint,
            "files_verified": self.files_verified,
            "execution_ready": False,
            "issues": [issue.to_dict() for issue in self.issues],
            "limits": {
                "git_lock_checked": False,
                "independent_audit_checked": False,
                "capture_runtime_checked": False,
            },
        }


@dataclass(frozen=True, slots=True)
class ChildLockCheck:
    """Validation result for the separate, non-self-referential lock receipt."""

    fingerprint: str | None
    issues: tuple[ContractIssue, ...]
    audit_files_verified: bool

    @property
    def valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "research_child_lock_check",
            "valid": self.valid,
            "fingerprint": self.fingerprint,
            "audit_files_verified": self.audit_files_verified,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def load_research_child(path: str | Path) -> Mapping[str, Any]:
    """Load a strict child-plan mapping."""

    return load_research_mapping(path)


def child_plan_fingerprint(payload: Mapping[str, Any]) -> str:
    """Identify the immutable child without embedding its own hash."""

    return canonical_research_fingerprint(payload)


def study_fingerprint(study: Mapping[str, Any]) -> str:
    """Bind a child to the exact study definition inside its program."""

    return canonical_research_fingerprint(study)


def check_research_child(
    child: Mapping[str, Any],
    program: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    verify_files: bool = False,
    campaign_state: CampaignState | None = None,
) -> ChildPlanCheck:
    """Check child identity, inheritance, predecessor evidence, and file references."""

    issues: list[ContractIssue] = []
    _reject_unknown_keys(child, TOP_LEVEL_FIELDS, "$", issues)
    for nested_path, allowed_fields in NESTED_FIELDS.items():
        _reject_nested_keys(child, nested_path, allowed_fields, issues)
    required = (
        "schema_version",
        "kind",
        "child_plan_id",
        "revision",
        "prepared_by",
        "program.path",
        "program.program_id",
        "program.fingerprint",
        "study.id",
        "study.fingerprint",
        "study.phase",
        "predecessor_result_events",
        "claim.result_kind",
        "claim.question",
        "claim.allowed_conclusions",
        "claim.forbidden_conclusions",
        "cohort.family_pool",
        "cohort.pool_phase",
        "cohort.requires_gate",
        "cohort.read_namespaces",
        "cohort.write_namespace",
        "cohort.selection_allowed",
        "cohort.manifest",
        "cohort.exposure_log",
        "trials.manifest",
        "trials.expected_count",
        "trials.stable_id_fields",
        "trials.seed_domains",
        "trials.expected_independent_units.task_families",
        "trials.expected_independent_units.scene_clusters",
        "trials.expected_independent_units.noise_repeats",
        "trials.expected_independent_units.rollouts",
        "measurement.primary.metric_id",
        "measurement.primary.formula",
        "measurement.primary.implementation_id",
        "measurement.primary.unit",
        "measurement.primary.direction",
        "measurement.primary.minimum_useful_effect",
        "measurement.controls",
        "measurement.strongest_control_metric_id",
        "measurement.inference.method",
        "measurement.inference.level",
        "measurement.inference.grouping_unit",
        "measurement.inference.replicates",
        "measurement.inference.seed",
        "decision.gate_components",
        "decision.positive_combiner",
        "decision.negative_combiner",
        "decision.inconclusive_rule",
        "decision.invalid_conditions",
        "runtime.model.repo_id",
        "runtime.model.revision",
        "runtime.model.snapshot_manifest_sha256",
        "runtime.environment.backend",
        "runtime.environment.package_receipt",
        "runtime.environment.camera_config_sha256",
        "runtime.environment.controller_config_sha256",
        "runtime.environment.preprocessor_config_sha256",
        "runtime.environment.postprocessor_config_sha256",
        "runtime.code.implementation_commit",
        "runtime.code.source_tree_sha256",
        "runtime.runner.entrypoint",
        "runtime.runner.argv",
        "runtime.runner.config",
        "output.root",
        "output.namespace",
        "output.attempt_ledger",
        "output.required_artifact_types",
        "completion.valid_trial_statuses",
        "completion.technical_retry_rule",
        "completion.resume_identity_fields",
        "protocol_lock.required_lock_fields",
        "protocol_lock.locked_choices",
    )
    required += tuple(f"budget.{name}" for name in BUDGET_FIELDS)
    allow_empty = {"predecessor_result_events", "protocol_lock.locked_choices"}
    missing = [
        path
        for path in required
        if not (_has_path(child, path) if path in allow_empty else _present(child, path))
    ]
    if missing:
        issues.append(_issue("child_missing_fields", "Child plan is incomplete", missing=missing))

    if child.get("schema_version") != CHILD_SCHEMA_VERSION:
        issues.append(_issue("invalid_child_schema", "Unsupported child-plan schema"))
    if child.get("kind") != CHILD_KIND:
        issues.append(_issue("invalid_child_kind", "Child plan has the wrong kind"))
    if str(_get(child, "claim.result_kind") or "") not in RESULT_KINDS:
        issues.append(_issue("invalid_result_kind", "Child result kind is not recognized"))
    for path in (
        "predecessor_result_events",
        "claim.allowed_conclusions",
        "claim.forbidden_conclusions",
        "cohort.read_namespaces",
        "trials.stable_id_fields",
        "trials.seed_domains",
        "measurement.controls",
        "decision.gate_components",
        "decision.invalid_conditions",
        "runtime.runner.argv",
        "output.required_artifact_types",
        "completion.valid_trial_statuses",
        "completion.resume_identity_fields",
        "protocol_lock.required_lock_fields",
    ):
        if not _is_sequence(_get(child, path)):
            issues.append(_issue("invalid_child_list", "Child field must be a list", path=path))

    study = _find_study(program, str(_get(child, "study.id") or ""))
    if _get(child, "program.program_id") != program.get("program_id"):
        issues.append(_issue("program_id_mismatch", "Child does not match the program ID"))
    try:
        expected_program = research_plan_fingerprint(program)
    except StrictResearchDataError as exc:
        issues.append(
            _issue("invalid_parent_program", "Parent program is not canonical", error=str(exc))
        )
        expected_program = None
    if _get(child, "program.fingerprint") != expected_program:
        issues.append(
            _issue("program_fingerprint_mismatch", "Child does not match the program hash")
        )
    if study is None:
        issues.append(_issue("unknown_child_study", "Child study does not exist in the program"))
    else:
        _check_study_inheritance(child, study, issues)
        _check_predecessors(child, study, issues)
        _check_budget_inheritance(child, study, program, issues)
        _check_protocol_lock(child, study, program, issues)

    artifact_refs = {
        "cohort.manifest": _get(child, "cohort.manifest"),
        "cohort.exposure_log": _get(child, "cohort.exposure_log"),
        "trials.manifest": _get(child, "trials.manifest"),
        "runtime.environment.package_receipt": _get(child, "runtime.environment.package_receipt"),
        "runtime.runner.config": _get(child, "runtime.runner.config"),
    }
    for path, reference in artifact_refs.items():
        _check_artifact_reference(reference, path, issues)
    files_ok = False
    if verify_files:
        if repo_root is None:
            issues.append(_issue("missing_repo_root", "File verification requires repo_root"))
        else:
            files_ok = _verify_artifact_files(artifact_refs, Path(repo_root), issues)

    _check_hashes(child, issues)
    _check_child_numbers(child, issues)
    _check_decision_components(child, issues)
    _check_output_paths(child, program, issues)
    if campaign_state is not None:
        for state_issue in child_authorization_issues(child, campaign_state, program):
            issues.append(
                _issue(state_issue.code, state_issue.message, **dict(state_issue.details))
            )
        if study is not None and verify_files and repo_root is not None:
            _check_confirmation_protocol(
                child,
                study,
                campaign_state,
                Path(repo_root),
                issues,
            )
    try:
        fingerprint = child_plan_fingerprint(child)
    except StrictResearchDataError as exc:
        issues.append(_issue("noncanonical_child", "Child plan cannot be hashed", error=str(exc)))
        fingerprint = None
    return ChildPlanCheck(
        fingerprint=fingerprint,
        issues=tuple(issues),
        files_verified=verify_files and files_ok,
    )


def check_research_child_lock(
    receipt: Mapping[str, Any],
    child: Mapping[str, Any],
    program: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    verify_files: bool = False,
) -> ChildLockCheck:
    """Check a lock receipt created after the child plan and independent audits."""

    issues: list[ContractIssue] = []
    _reject_unknown_keys(receipt, LOCK_FIELDS, "$", issues)
    required = (
        "schema_version",
        "kind",
        "lock_id",
        "program_id",
        "program_fingerprint",
        "study_id",
        "study_fingerprint",
        "child_plan_id",
        "child_plan_fingerprint",
        "manifest_commit",
        "locked_utc",
        "prepared_by",
        "reservation_id",
        "prior_ledger_tip",
        "audits",
    )
    missing = [path for path in required if not _present(receipt, path)]
    if missing:
        issues.append(
            _issue("lock_missing_fields", "Child lock receipt is incomplete", missing=missing)
        )
    if receipt.get("schema_version") != LOCK_SCHEMA_VERSION or receipt.get("kind") != LOCK_KIND:
        issues.append(_issue("invalid_lock_schema", "Child lock receipt has an unsupported schema"))
    if receipt.get("program_id") != program.get("program_id"):
        issues.append(_issue("lock_program_mismatch", "Lock receipt has the wrong program ID"))
    if receipt.get("program_fingerprint") != research_plan_fingerprint(program):
        issues.append(
            _issue("lock_program_hash_mismatch", "Lock receipt has the wrong program hash")
        )
    child_fingerprint = child_plan_fingerprint(child)
    if receipt.get("child_plan_id") != child.get("child_plan_id"):
        issues.append(_issue("lock_child_mismatch", "Lock receipt has the wrong child ID"))
    if receipt.get("prepared_by") != child.get("prepared_by"):
        issues.append(_issue("lock_preparer_mismatch", "Lock receipt has the wrong preparer"))
    if receipt.get("child_plan_fingerprint") != child_fingerprint:
        issues.append(_issue("lock_child_hash_mismatch", "Lock receipt has the wrong child hash"))
    study = _find_study(program, str(_get(child, "study.id") or ""))
    if study is not None:
        if receipt.get("study_id") != study.get("id"):
            issues.append(_issue("lock_study_mismatch", "Lock receipt has the wrong study ID"))
        if receipt.get("study_fingerprint") != study_fingerprint(study):
            issues.append(
                _issue("lock_study_hash_mismatch", "Lock receipt has the wrong study hash")
            )
    if not COMMIT_RE.fullmatch(str(receipt.get("manifest_commit") or "")):
        issues.append(
            _issue("invalid_manifest_commit", "Lock receipt needs a full Git commit hash")
        )
    if not str(receipt.get("reservation_id") or ""):
        issues.append(_issue("missing_lock_reservation", "Lock must bind a budget reservation"))
    if not SHA256_RE.fullmatch(str(receipt.get("prior_ledger_tip") or "")):
        issues.append(_issue("invalid_lock_ledger_tip", "Lock must bind the prior event hash"))

    audits = receipt.get("audits")
    audit_files_ok = False
    if not _is_sequence(audits) or not audits:
        issues.append(_issue("missing_lock_audits", "Lock receipt needs independent audits"))
    else:
        seen_types: set[str] = set()
        references: dict[str, Any] = {}
        for index, audit in enumerate(audits):
            if not isinstance(audit, Mapping):
                issues.append(
                    _issue("invalid_lock_audit", "Audit entries must be mappings", index=index)
                )
                continue
            _reject_unknown_keys(
                audit,
                frozenset(
                    {
                        "audit_type",
                        "auditor_id",
                        "verdict",
                        "subject_child_fingerprint",
                        "artifact",
                    }
                ),
                f"audits[{index}]",
                issues,
            )
            for field_name in (
                "audit_type",
                "auditor_id",
                "verdict",
                "subject_child_fingerprint",
                "artifact",
            ):
                if not _present(audit, field_name):
                    issues.append(
                        _issue(
                            "invalid_lock_audit",
                            "Audit entry is incomplete",
                            index=index,
                            field=field_name,
                        )
                    )
            audit_type = str(audit.get("audit_type") or "")
            seen_types.add(audit_type)
            if audit.get("verdict") != "pass":
                issues.append(
                    _issue("lock_audit_failed", "Every lock audit must pass", audit_type=audit_type)
                )
            if audit.get("auditor_id") == receipt.get("prepared_by"):
                issues.append(
                    _issue("audit_not_independent", "Lock auditor must differ from the preparer")
                )
            if audit.get("subject_child_fingerprint") != child_fingerprint:
                issues.append(_issue("audit_subject_mismatch", "Audit is bound to another child"))
            reference = audit.get("artifact")
            references[f"audits[{index}].artifact"] = reference
            _check_artifact_reference(reference, f"audits[{index}].artifact", issues)
        required_audits = set(BASE_LOCK_AUDITS)
        if study is not None:
            required_audits.update(str(item) for item in _sequence(study.get("required_audits")))
        missing_audits = sorted(required_audits - seen_types)
        if missing_audits:
            issues.append(
                _issue(
                    "required_audits_missing",
                    "Lock lacks required audit types",
                    missing=missing_audits,
                )
            )
        if verify_files:
            if repo_root is None:
                issues.append(_issue("missing_repo_root", "Audit verification requires repo_root"))
            else:
                before_audit_files = len(issues)
                audit_files_ok = _verify_artifact_files(references, Path(repo_root), issues)
                _verify_lock_audit_reports(audits, Path(repo_root), issues)
                audit_files_ok = audit_files_ok and len(issues) == before_audit_files
    try:
        fingerprint = canonical_research_fingerprint(receipt)
    except StrictResearchDataError as exc:
        issues.append(_issue("noncanonical_lock", "Lock receipt cannot be hashed", error=str(exc)))
        fingerprint = None
    return ChildLockCheck(
        fingerprint=fingerprint,
        issues=tuple(issues),
        audit_files_verified=verify_files and audit_files_ok,
    )


def _check_study_inheritance(
    child: Mapping[str, Any], study: Mapping[str, Any], issues: list[ContractIssue]
) -> None:
    if _get(child, "study.fingerprint") != study_fingerprint(study):
        issues.append(
            _issue("study_fingerprint_mismatch", "Child does not match its study definition")
        )
    inherited_pairs = [
        ("study.phase", "phase"),
        ("cohort.family_pool", "data_scope.family_pool"),
        ("cohort.pool_phase", "data_scope.pool_phase"),
        ("cohort.requires_gate", "data_scope.requires_gate"),
        ("cohort.read_namespaces", "data_scope.read_namespaces"),
        ("cohort.write_namespace", "data_scope.write_namespace"),
        ("cohort.selection_allowed", "data_scope.selection_allowed"),
        ("measurement.primary.metric_id", "primary_claim.metric_id"),
        ("measurement.primary.unit", "primary_claim.unit"),
        ("measurement.primary.direction", "primary_claim.direction"),
        ("output.required_artifact_types", "required_outputs"),
    ]
    if str(study.get("kind")) not in {
        "behavior_confirmation",
        "semantic_confirmation",
        "causal_confirmation",
    }:
        inherited_pairs.append(("measurement.controls", "controls"))
    for child_path, study_path in inherited_pairs:
        if _get(child, child_path) != _get(study, study_path):
            issues.append(
                _issue(
                    "child_weakens_study",
                    "Child field does not exactly inherit the program study",
                    child_path=child_path,
                    study_path=study_path,
                )
            )
    if _get(child, "claim.question") != study.get("question"):
        issues.append(_issue("child_question_mismatch", "Child question must match its study"))
    if list(_sequence(_get(child, "claim.allowed_conclusions"))) != list(
        _sequence(study.get("allowed_conclusions"))
    ):
        issues.append(
            _issue("child_claim_mismatch", "Allowed conclusions must exactly match the study")
        )
    if list(_sequence(_get(child, "claim.forbidden_conclusions"))) != list(
        _sequence(study.get("forbidden_conclusions"))
    ):
        issues.append(
            _issue("child_claim_mismatch", "Forbidden conclusions must exactly match the study")
        )
    if str(study.get("kind")) not in {
        "behavior_confirmation",
        "semantic_confirmation",
        "causal_confirmation",
    } and _get(child, "measurement.primary.formula") != _get(study, "primary_claim.definition"):
        issues.append(
            _issue(
                "child_formula_mismatch",
                "Discovery and preparation formulas must exactly match the program",
            )
        )


def _check_predecessors(
    child: Mapping[str, Any], study: Mapping[str, Any], issues: list[ContractIssue]
) -> None:
    records = _sequence(child.get("predecessor_result_events"))
    required_count = len(
        {
            *[
                str(item)
                for item in _sequence(_get(study, "entry_conditions.requires_all_completed"))
            ],
            *[
                str(item)
                for item in _sequence(_get(study, "entry_conditions.requires_any_positive"))
            ],
        }
    )
    if len(records) != required_count:
        issues.append(
            _issue(
                "predecessor_event_count_mismatch",
                "Child must cite one exact result event per predecessor study",
                expected=required_count,
                observed=len(records),
            )
        )
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != EVENT_REF_FIELDS:
            issues.append(
                _issue(
                    "invalid_predecessor_event_ref",
                    "Predecessors must be exact ledger event references",
                    index=index,
                )
            )
            continue
        event_id = str(record.get("event_id"))
        if event_id in seen:
            issues.append(
                _issue("duplicate_predecessor_event", "A predecessor event appears twice")
            )
        seen.add(event_id)
        if not SHA256_RE.fullmatch(str(record.get("event_sha256") or "")):
            issues.append(_issue("invalid_predecessor_hash", "Predecessor event needs a sha256"))
        if _nonnegative_int(record.get("sequence")) in {None, 0}:
            issues.append(
                _issue("invalid_predecessor_sequence", "Predecessor sequence must be positive")
            )


def _check_budget_inheritance(
    child: Mapping[str, Any],
    study: Mapping[str, Any],
    program: Mapping[str, Any],
    issues: list[ContractIssue],
) -> None:
    parent_names = {
        "max_model_calls": "max_model_calls",
        "max_action_generations": "max_action_generations",
        "max_full_rollouts": "max_full_rollouts",
        "max_simulator_steps": "max_simulator_steps",
        "max_probe_fits": "max_probe_fits",
        "max_persistent_gb": "max_additional_persistent_gb",
        "max_ephemeral_gb": "max_ephemeral_gb",
    }
    for child_name, study_name in parent_names.items():
        observed = _number(_get(child, f"budget.{child_name}"))
        maximum = _number(_get(study, f"budget.{study_name}"))
        if observed is not None and maximum is not None and observed > maximum:
            issues.append(
                _issue(
                    "child_budget_exceeds_study",
                    "Child budget exceeds its program study cap",
                    field=child_name,
                    observed=observed,
                    maximum=maximum,
                )
            )
    free = _number(_get(child, "budget.min_free_space_gb"))
    program_free = _number(_get(program, "program_budget.min_free_space_gb"))
    if free is not None and program_free is not None and free < program_free:
        issues.append(
            _issue("child_weakens_free_space", "Child lowers the program free-space floor")
        )


def _check_protocol_lock(
    child: Mapping[str, Any],
    study: Mapping[str, Any],
    program: Mapping[str, Any],
    issues: list[ContractIssue],
) -> None:
    defaults = [
        str(item)
        for item in _sequence(_get(program, "child_contract_defaults.required_lock_fields"))
    ]
    additions = [
        str(item) for item in _sequence(_get(study, "child_contract.additional_lock_fields"))
    ]
    expected = list(dict.fromkeys([*defaults, *additions]))
    observed = list(_sequence(_get(child, "protocol_lock.required_lock_fields")))
    if observed != expected:
        issues.append(
            _issue(
                "protocol_lock_fields_mismatch",
                "Child must inherit every parent-owned lock field in order",
                missing=sorted(set(expected) - set(observed)),
                extra=sorted(set(observed) - set(expected)),
            )
        )
    choices = _get(child, "protocol_lock.locked_choices")
    if not isinstance(choices, Mapping):
        issues.append(
            _issue("invalid_locked_choices", "Study-specific locked choices must be a mapping")
        )
        return
    if set(choices) != set(additions):
        issues.append(
            _issue(
                "locked_choice_fields_mismatch",
                "Locked choices must resolve each study-specific lock field exactly once",
                missing=sorted(set(additions) - set(choices)),
                extra=sorted(set(choices) - set(additions)),
            )
        )
    empty = sorted(
        str(key)
        for key, value in choices.items()
        if value is None or value == "" or value == [] or value == {}
    )
    if empty:
        issues.append(
            _issue(
                "unresolved_locked_choices",
                "Every study-specific lock choice needs a concrete value or artifact ID",
                fields=empty,
            )
        )


def _check_decision_components(child: Mapping[str, Any], issues: list[ContractIssue]) -> None:
    components = _sequence(_get(child, "decision.gate_components"))
    roles: set[str] = set()
    ids: set[str] = set()
    value_ids: set[str] = set()
    for index, component in enumerate(components):
        if not isinstance(component, Mapping) or set(component) != GATE_COMPONENT_FIELDS:
            issues.append(
                _issue(
                    "invalid_gate_component",
                    "Decision gates must use the exact typed component schema",
                    index=index,
                )
            )
            continue
        component_id = str(component.get("id") or "")
        value_id = str(component.get("value_id") or "")
        role = str(component.get("role") or "")
        if not component_id or component_id in ids or not value_id or value_id in value_ids:
            issues.append(
                _issue(
                    "duplicate_or_empty_gate_component",
                    "Decision gate IDs and analysis value IDs must be unique and nonempty",
                    index=index,
                )
            )
        ids.add(component_id)
        value_ids.add(value_id)
        roles.add(role)
        if role not in GATE_ROLES:
            issues.append(
                _issue("invalid_gate_role", "Decision gate role is not recognized", index=index)
            )
        if component.get("operator") not in GATE_OPERATORS:
            issues.append(
                _issue(
                    "invalid_gate_operator",
                    "Decision gate operator is not recognized",
                    index=index,
                )
            )
        if _number(component.get("threshold")) is None or not str(component.get("unit") or ""):
            issues.append(
                _issue(
                    "invalid_gate_threshold",
                    "Decision gates need a finite threshold and explicit unit",
                    index=index,
                )
            )
    if not {"integrity", "positive", "negative"} <= roles:
        issues.append(
            _issue(
                "decision_roles_missing",
                "Decision needs integrity, positive, and negative gates",
            )
        )
    if (
        _get(child, "decision.positive_combiner") != "all"
        or _get(child, "decision.negative_combiner") != "all"
    ):
        issues.append(
            _issue("unsupported_gate_combiner", "This contract currently requires all-gates logic")
        )


def _check_output_paths(
    child: Mapping[str, Any], program: Mapping[str, Any], issues: list[ContractIssue]
) -> None:
    root = Path(str(_get(child, "output.root") or "."))
    if not root.is_absolute() or root == Path(root.anchor):
        issues.append(
            _issue(
                "unsafe_output_root",
                "Output root must be a specific absolute directory, never a filesystem root",
            )
        )
    allowed_roots = {
        Path(str(item)).resolve()
        for item in _sequence(_get(program, "protocol_defaults.allowed_output_roots"))
        if str(item)
    }
    if root.resolve() not in allowed_roots:
        issues.append(
            _issue(
                "untrusted_output_root",
                "Output root is not one of the program's explicit storage roots",
                observed=str(root),
                allowed=sorted(str(item) for item in allowed_roots),
            )
        )
    if _path_has_symlink(Path(root.anchor), str(root).removeprefix(root.anchor)):
        issues.append(
            _issue(
                "symlinked_output_root",
                "Output root may not traverse a symlink",
                observed=str(root),
            )
        )
    for dotted in ("output.namespace", "output.attempt_ledger"):
        value = str(_get(child, dotted) or "")
        relative = Path(value)
        if not value or relative.is_absolute() or ".." in relative.parts or value in {".", "./"}:
            issues.append(
                _issue(
                    "unsafe_output_path",
                    "Output namespace paths must be safe and relative to the output root",
                    path=dotted,
                )
            )
    program_id = str(program.get("program_id") or "")
    if program_id and program_id.split("-")[0] not in str(_get(child, "output.namespace") or ""):
        issues.append(
            _issue(
                "output_namespace_not_program_scoped",
                "Output namespace must visibly belong to this research program",
            )
        )


def _check_confirmation_protocol(
    child: Mapping[str, Any],
    study: Mapping[str, Any],
    state: CampaignState,
    repo_root: Path,
    issues: list[ContractIssue],
) -> None:
    if str(study.get("kind")) not in {
        "behavior_confirmation",
        "semantic_confirmation",
        "causal_confirmation",
    }:
        return
    source_id = str(study.get("source_claim_study") or "")
    if not source_id:
        dependencies = [
            str(item) for item in _sequence(_get(study, "entry_conditions.requires_any_positive"))
        ]
        source_id = dependencies[0] if len(dependencies) == 1 else ""
    terminal = state.terminal_results.get(source_id)
    result_event = state.events_by_id.get(str((terminal or {}).get("result_event_id") or ""))
    lock_ref = _get(result_event or {}, "payload.child_lock_event")
    lock_event = (
        state.events_by_id.get(str(lock_ref.get("event_id") or ""))
        if isinstance(lock_ref, Mapping)
        else None
    )
    source_ref = _get(lock_event or {}, "payload.child_ref")
    if not isinstance(source_ref, Mapping):
        issues.append(
            _issue(
                "confirmation_source_child_missing",
                "Confirmation cannot resolve its actual source child from the ledger",
            )
        )
        return
    path = (repo_root / str(source_ref.get("path") or "")).resolve()
    try:
        path.relative_to(repo_root.resolve())
        source = load_research_mapping(path)
    except (OSError, ValueError):
        issues.append(
            _issue(
                "confirmation_source_child_unreadable",
                "Confirmation source child bytes cannot be loaded",
            )
        )
        return
    frozen_paths = (
        "measurement.primary.metric_id",
        "measurement.primary.formula",
        "measurement.primary.implementation_id",
        "measurement.primary.unit",
        "measurement.primary.direction",
        "measurement.primary.minimum_useful_effect",
        "measurement.controls",
        "measurement.strongest_control_metric_id",
        "measurement.inference",
        "decision",
        "runtime.model",
        "runtime.environment",
        "runtime.code",
        "runtime.runner.entrypoint",
        "runtime.runner.config.sha256",
        "trials.stable_id_fields",
        "trials.seed_domains",
    )
    changed = [
        path_name for path_name in frozen_paths if _get(child, path_name) != _get(source, path_name)
    ]
    if changed:
        issues.append(
            _issue(
                "confirmation_protocol_drift",
                "Confirmation changed fields frozen by its actual discovery child",
                fields=changed,
                source_study_id=source_id,
            )
        )
    if _scientific_runner_argv(child) != _scientific_runner_argv(source):
        issues.append(
            _issue(
                "confirmation_runner_drift",
                "Confirmation changed runner arguments other than the cohort config path",
                source_study_id=source_id,
            )
        )
    expected_protocol = _scientific_protocol_fingerprint(source)
    if (
        _get(child, "protocol_lock.locked_choices.source_scientific_protocol_fingerprint")
        != expected_protocol
    ):
        issues.append(
            _issue(
                "confirmation_source_protocol_hash_mismatch",
                "Confirmation does not bind the actual source child's scientific protocol",
                expected=expected_protocol,
                source_study_id=source_id,
            )
        )


def _scientific_runner_argv(child: Mapping[str, Any]) -> list[Any]:
    argv = list(_sequence(_get(child, "runtime.runner.argv")))
    normalized: list[Any] = []
    skip_config_value = False
    for item in argv:
        if skip_config_value:
            normalized.append("<confirmation-specific-config>")
            skip_config_value = False
            continue
        normalized.append(item)
        if item == "--config":
            skip_config_value = True
    return normalized


def _scientific_protocol_fingerprint(child: Mapping[str, Any]) -> str:
    return canonical_research_fingerprint(
        {
            "measurement": child.get("measurement"),
            "decision": child.get("decision"),
            "model": _get(child, "runtime.model"),
            "environment": _get(child, "runtime.environment"),
            "code": _get(child, "runtime.code"),
            "runner_entrypoint": _get(child, "runtime.runner.entrypoint"),
            "runner_argv": _scientific_runner_argv(child),
            "stable_id_fields": _get(child, "trials.stable_id_fields"),
            "seed_domains": _get(child, "trials.seed_domains"),
            "source_locked_choices": _get(child, "protocol_lock.locked_choices"),
        }
    )


def _check_artifact_reference(reference: Any, path: str, issues: list[ContractIssue]) -> None:
    if not isinstance(reference, Mapping):
        issues.append(
            _issue("invalid_artifact_reference", "Artifact reference must be a mapping", path=path)
        )
        return
    _reject_unknown_keys(reference, ARTIFACT_FIELDS, path, issues)
    missing = sorted(ARTIFACT_FIELDS - set(reference))
    if missing:
        issues.append(
            _issue(
                "invalid_artifact_reference",
                "Artifact reference is incomplete",
                path=path,
                missing=missing,
            )
        )
    if not SHA256_RE.fullmatch(str(reference.get("sha256") or "")):
        issues.append(
            _issue("invalid_artifact_hash", "Artifact reference needs a full sha256", path=path)
        )
    artifact_path = Path(str(reference.get("path") or "."))
    if artifact_path.is_absolute() or ".." in artifact_path.parts:
        issues.append(
            _issue("unsafe_artifact_path", "Locked input paths must be repo-relative", path=path)
        )


def _verify_artifact_files(
    references: Mapping[str, Any], repo_root: Path, issues: list[ContractIssue]
) -> bool:
    start = len(issues)
    for label, reference in references.items():
        if not isinstance(reference, Mapping):
            continue
        relative = str(reference.get("path") or "")
        path = (repo_root / relative).resolve()
        try:
            path.relative_to(repo_root.resolve())
        except ValueError:
            issues.append(
                _issue(
                    "artifact_path_escape",
                    "Locked artifact resolves outside the repository",
                    label=label,
                    path=str(path),
                )
            )
            continue
        if _path_has_symlink(repo_root, relative):
            issues.append(
                _issue(
                    "artifact_path_symlink",
                    "Locked artifact paths may not contain symlinks",
                    label=label,
                    path=relative,
                )
            )
            continue
        if not path.is_file():
            issues.append(
                _issue(
                    "artifact_file_missing",
                    "Locked artifact file is missing",
                    label=label,
                    path=str(path),
                )
            )
            continue
        if file_sha256(path) != reference.get("sha256"):
            issues.append(
                _issue(
                    "artifact_file_hash_mismatch",
                    "Locked artifact bytes do not match",
                    label=label,
                    path=str(path),
                )
            )
    return len(issues) == start


def _verify_lock_audit_reports(audits, repo_root: Path, issues: list[ContractIssue]) -> None:
    records = [item for item in _sequence(audits) if isinstance(item, Mapping)]
    for index, audit in enumerate(records):
        reference = audit.get("artifact")
        if not isinstance(reference, Mapping):
            continue
        path = (repo_root / str(reference.get("path") or "")).resolve()
        if not path.is_file():
            continue
        try:
            report = load_research_mapping(path)
            validate_research_audit_report(
                report,
                audit_type=str(audit.get("audit_type") or ""),
                subject_kind="child_plan",
                subject_fingerprint=str(audit.get("subject_child_fingerprint") or ""),
                auditor_id=str(audit.get("auditor_id") or ""),
                verdict=str(audit.get("verdict") or ""),
            )
        except (OSError, ValueError, ResearchAuditError) as exc:
            code = exc.code if isinstance(exc, ResearchAuditError) else "invalid_audit_report"
            issues.append(
                _issue(
                    code,
                    "Locked audit report bytes are invalid or do not match the lock",
                    index=index,
                    error=str(exc),
                )
            )


def _path_has_symlink(root: Path, relative: str) -> bool:
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _check_hashes(child: Mapping[str, Any], issues: list[ContractIssue]) -> None:
    paths = (
        "program.fingerprint",
        "study.fingerprint",
        "runtime.model.snapshot_manifest_sha256",
        "runtime.environment.camera_config_sha256",
        "runtime.environment.controller_config_sha256",
        "runtime.environment.preprocessor_config_sha256",
        "runtime.environment.postprocessor_config_sha256",
        "runtime.code.source_tree_sha256",
    )
    for path in paths:
        if not SHA256_RE.fullmatch(str(_get(child, path) or "")):
            issues.append(
                _issue("invalid_child_hash", "Child identity field needs a full sha256", path=path)
            )
    if not COMMIT_RE.fullmatch(str(_get(child, "runtime.code.implementation_commit") or "")):
        issues.append(
            _issue(
                "invalid_implementation_commit",
                "Implementation commit must be 40 lowercase hex characters",
            )
        )


def _check_child_numbers(child: Mapping[str, Any], issues: list[ContractIssue]) -> None:
    expected = _nonnegative_int(_get(child, "trials.expected_count"))
    if expected is None or expected <= 0:
        issues.append(_issue("invalid_trial_count", "Child needs a positive exact trial count"))
    if (
        _get(child, "claim.result_kind") == "preparation_gate"
        and expected is not None
        and expected > (_nonnegative_int(_get(child, "budget.max_full_rollouts")) or 0)
    ):
        issues.append(
            _issue(
                "preparation_trials_exceed_rollout_budget",
                "Preparation trial count exceeds its full-rollout cap",
            )
        )
    level = _number(_get(child, "measurement.inference.level"))
    if level is None or not 0 < level < 1:
        issues.append(
            _issue(
                "invalid_interval_level", "Inference level must lie strictly between zero and one"
            )
        )
    replicates = _nonnegative_int(_get(child, "measurement.inference.replicates"))
    if replicates is None or replicates <= 0:
        issues.append(
            _issue("invalid_inference_replicates", "Inference needs a positive replicate count")
        )
    if _nonnegative_int(_get(child, "measurement.inference.seed")) is None:
        issues.append(
            _issue("invalid_inference_seed", "Inference seed must be a nonnegative integer")
        )
    for name in ("task_families", "scene_clusters", "noise_repeats", "rollouts"):
        if _nonnegative_int(_get(child, f"trials.expected_independent_units.{name}")) is None:
            issues.append(
                _issue(
                    "invalid_expected_independent_units",
                    "Expected evidence-unit counts must be nonnegative integers",
                    field=name,
                )
            )
    for name in BUDGET_FIELDS:
        value = _number(_get(child, f"budget.{name}"))
        if value is None or value < 0:
            issues.append(
                _issue(
                    "invalid_child_budget",
                    "Child budget must be finite and nonnegative",
                    field=name,
                )
            )


def _reject_unknown_keys(
    value: Mapping[str, Any], allowed: frozenset[str], path: str, issues: list[ContractIssue]
) -> None:
    unknown = sorted(str(key) for key in set(value) - allowed)
    if unknown:
        issues.append(
            _issue(
                "unknown_contract_fields",
                "Contract contains unknown fields",
                path=path,
                fields=unknown,
            )
        )


def _reject_nested_keys(
    payload: Mapping[str, Any],
    path: str,
    allowed: frozenset[str],
    issues: list[ContractIssue],
) -> None:
    value = _get(payload, path)
    if not isinstance(value, Mapping):
        issues.append(
            _issue(
                "invalid_contract_mapping",
                "Contract section must be a mapping",
                path=path,
            )
        )
        return
    _reject_unknown_keys(value, allowed, path, issues)


def _find_study(program: Mapping[str, Any], study_id: str) -> Mapping[str, Any] | None:
    for study in _sequence(program.get("studies")):
        if isinstance(study, Mapping) and study.get("id") == study_id:
            return study
    return None


def _get(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _present(payload: Mapping[str, Any], path: str) -> bool:
    value = _get(payload, path)
    return value is not None and value != "" and value != {} and value != []


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


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and abs(parsed) != float("inf") else None


def _nonnegative_int(value: Any) -> int | None:
    parsed = _number(value)
    if parsed is None or parsed < 0 or not parsed.is_integer():
        return None
    return int(parsed)


def _issue(code: str, message: str, **details: Any) -> ContractIssue:
    return ContractIssue(code=code, message=message, details=details)


__all__ = [
    "CHILD_KIND",
    "CHILD_SCHEMA_VERSION",
    "LOCK_KIND",
    "LOCK_SCHEMA_VERSION",
    "ChildLockCheck",
    "ChildPlanCheck",
    "ContractIssue",
    "check_research_child",
    "check_research_child_lock",
    "child_plan_fingerprint",
    "load_research_child",
    "study_fingerprint",
]
