"""Contracts for one locked, executable child of a research program."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from vla_lens.research_io import (
    StrictResearchDataError,
    canonical_research_fingerprint,
    file_sha256,
    load_research_mapping,
)
from vla_lens.research_plan import research_plan_fingerprint

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
        "predecessor_results",
        "claim",
        "cohort",
        "trials",
        "measurement",
        "decision",
        "runtime",
        "budget",
        "output",
        "completion",
        "required_audits",
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
        "first_output_absent",
        "audits",
    }
)
BUDGET_FIELDS = frozenset(
    {
        "max_model_calls",
        "max_action_generations",
        "max_full_rollouts",
        "max_simulator_steps",
        "max_persistent_gb",
        "max_ephemeral_gb",
        "min_free_space_gb",
    }
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
) -> ChildPlanCheck:
    """Check child identity, inheritance, predecessor evidence, and file references."""

    issues: list[ContractIssue] = []
    _reject_unknown_keys(child, TOP_LEVEL_FIELDS, "$", issues)
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
        "predecessor_results",
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
        "measurement.primary.metric_id",
        "measurement.primary.formula",
        "measurement.primary.implementation_id",
        "measurement.primary.unit",
        "measurement.primary.direction",
        "measurement.primary.minimum_useful_effect",
        "measurement.strongest_control_metric_id",
        "measurement.inference.method",
        "measurement.inference.level",
        "measurement.inference.grouping_unit",
        "measurement.inference.replicates",
        "measurement.inference.seed",
        "decision.gate_components",
        "decision.positive_combiner",
        "decision.negative_rule",
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
        "required_audits",
    )
    required += tuple(f"budget.{name}" for name in BUDGET_FIELDS)
    allow_empty = {"predecessor_results"}
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
        "predecessor_results",
        "claim.allowed_conclusions",
        "claim.forbidden_conclusions",
        "cohort.read_namespaces",
        "trials.stable_id_fields",
        "trials.seed_domains",
        "decision.gate_components",
        "decision.invalid_conditions",
        "runtime.runner.argv",
        "output.required_artifact_types",
        "completion.valid_trial_statuses",
        "completion.resume_identity_fields",
        "required_audits",
    ):
        if not _is_sequence(_get(child, path)):
            issues.append(_issue("invalid_child_list", "Child field must be a list", path=path))

    study = _find_study(program, str(_get(child, "study.id") or ""))
    if child.get("program", {}).get("program_id") != program.get("program_id"):
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
        "first_output_absent",
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
    if receipt.get("first_output_absent") is not True:
        issues.append(_issue("outputs_predate_lock", "Lock must state that no output existed yet"))

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
        required_audits = {str(item) for item in _sequence(child.get("required_audits"))}
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
                audit_files_ok = _verify_artifact_files(references, Path(repo_root), issues)
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
    for child_path, study_path in (
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
    ):
        if _get(child, child_path) != _get(study, study_path):
            issues.append(
                _issue(
                    "child_weakens_study",
                    "Child field does not exactly inherit the program study",
                    child_path=child_path,
                    study_path=study_path,
                )
            )
    if child.get("claim", {}).get("question") != study.get("question"):
        issues.append(_issue("child_question_mismatch", "Child question must match its study"))
    if set(_sequence(_get(child, "claim.allowed_conclusions"))) - set(
        _sequence(study.get("allowed_conclusions"))
    ):
        issues.append(_issue("child_claim_expansion", "Child adds an unauthorized conclusion"))
    if not set(_sequence(study.get("forbidden_conclusions"))) <= set(
        _sequence(_get(child, "claim.forbidden_conclusions"))
    ):
        issues.append(_issue("child_drops_forbidden_claim", "Child drops a forbidden conclusion"))


def _check_predecessors(
    child: Mapping[str, Any], study: Mapping[str, Any], issues: list[ContractIssue]
) -> None:
    records = _sequence(child.get("predecessor_results"))
    by_study: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            issues.append(
                _issue("invalid_predecessor", "Predecessors must be mappings", index=index)
            )
            continue
        required = {"study_id", "result_card_id", "fingerprint", "verdict"}
        missing = sorted(required - set(record))
        if missing:
            issues.append(
                _issue(
                    "invalid_predecessor", "Predecessor is incomplete", index=index, missing=missing
                )
            )
            continue
        study_id = str(record.get("study_id"))
        if study_id in by_study:
            issues.append(
                _issue(
                    "duplicate_predecessor", "A predecessor study appears twice", study_id=study_id
                )
            )
        by_study[study_id] = record
        if str(record.get("verdict")) not in VERDICTS:
            issues.append(
                _issue("invalid_predecessor_verdict", "Predecessor verdict is not recognized")
            )
        if not SHA256_RE.fullmatch(str(record.get("fingerprint") or "")):
            issues.append(
                _issue("invalid_predecessor_hash", "Predecessor needs a sha256 fingerprint")
            )
    required_all = {
        str(item) for item in _sequence(_get(study, "entry_conditions.requires_all_completed"))
    }
    required_positive = {
        str(item) for item in _sequence(_get(study, "entry_conditions.requires_any_positive"))
    }
    if not required_all <= set(by_study):
        issues.append(
            _issue(
                "missing_predecessor_results",
                "Child lacks completed predecessor evidence",
                missing=sorted(required_all - set(by_study)),
            )
        )
    if required_positive and not any(
        study_id in by_study and by_study[study_id].get("verdict") in POSITIVE_VERDICTS
        for study_id in required_positive
    ):
        issues.append(
            _issue(
                "positive_predecessor_missing", "Child entry gate lacks a positive source result"
            )
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
        path = repo_root / str(reference.get("path") or "")
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
