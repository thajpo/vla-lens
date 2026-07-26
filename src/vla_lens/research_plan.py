"""Schema checks and compact rendering for agent-run research programs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from vla_lens.research_io import (
    StrictResearchDataError,
    canonical_research_fingerprint,
    load_research_mapping,
)

RESEARCH_PLAN_SCHEMA_VERSION = 1
RESEARCH_PLAN_KIND = "autonomous_research_program"
STUDY_KINDS = frozenset(
    {
        "preparation",
        "behavior_discovery",
        "behavior_confirmation",
        "semantic_readout",
        "semantic_confirmation",
        "internal_discovery",
        "causal_discovery",
        "causal_confirmation",
        "behavior_intervention",
        "reliability_design",
    }
)
CONFIRMATION_KINDS = frozenset(
    {"behavior_confirmation", "semantic_confirmation", "causal_confirmation"}
)
OUTCOME_ACTIONS = frozenset(
    {
        "reevaluate_program",
        "stop_branch",
        "revise_child",
        "supersede_source_and_reevaluate",
        "create_separate_program",
    }
)
BRANCH_MODES = frozenset({"single", "one_per_positive_source", "inherit_source_branch"})
FAMILY_POOLS = frozenset({"none", "discovery", "confirmation", "both_baseline_only", "inherited"})
STUDY_PHASES = frozenset(
    {
        "foundation",
        "discovery",
        "behavior_confirmation",
        "mechanism_discovery",
        "mechanism_confirmation",
        "rollout",
    }
)
SOURCE_BOUND_KINDS = frozenset(
    {
        "internal_discovery",
        "causal_discovery",
        "semantic_confirmation",
        "causal_confirmation",
        "behavior_intervention",
    }
)
CONFIRMATION_LOCK_FIELDS = frozenset(
    {
        "prospective_cohort_hash",
        "metric_id_and_formula",
        "minimum_useful_effect",
        "controls",
        "analysis_code_commit",
        "exact_model_checkpoint_hash",
        "runner_config_hash",
        "precision_or_bounded_replication_justification",
        "small_sample_family_level_inference",
    }
)

PROGRAM_BUDGET_FIELDS = (
    "max_persistent_gb",
    "max_ephemeral_gb",
    "min_free_space_gb",
    "max_full_rollouts",
    "max_action_generations",
    "max_model_calls",
    "max_simulator_steps",
    "max_concurrent_hardware_children",
)
STUDY_BUDGET_FIELDS = (
    "max_instances",
    "max_model_calls",
    "max_action_generations",
    "max_full_rollouts",
    "max_simulator_steps",
    "max_additional_persistent_gb",
    "max_ephemeral_gb",
)
SUMMED_BUDGETS = {
    "max_model_calls": "max_model_calls",
    "max_action_generations": "max_action_generations",
    "max_full_rollouts": "max_full_rollouts",
    "max_simulator_steps": "max_simulator_steps",
    "max_additional_persistent_gb": "max_persistent_gb",
}


@dataclass(frozen=True, slots=True)
class ResearchPlanIssue:
    """One machine-readable schema problem."""

    code: str
    message: str
    severity: str = "error"
    path: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "path": self.path,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class ResearchPlanCheck:
    """Structural validation result for one immutable program plan."""

    fingerprint: str | None
    summary: Mapping[str, Any] = field(default_factory=dict)
    issues: tuple[ResearchPlanIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> tuple[ResearchPlanIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ResearchPlanIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": "research_program_schema_check",
            "valid": self.valid,
            "fingerprint": self.fingerprint,
            "summary": dict(self.summary),
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "issues": [issue.to_dict() for issue in self.issues],
        }


def load_research_plan(path: str | Path) -> Mapping[str, Any]:
    """Load one YAML plan and require a mapping at the root."""

    return load_research_mapping(path)


def research_plan_fingerprint(payload: Mapping[str, Any]) -> str:
    """Identify the immutable program without mixing in execution state."""

    return canonical_research_fingerprint(payload)


def check_research_plan_file(path: str | Path) -> ResearchPlanCheck:
    """Load and structurally check a plan without writing anything."""

    plan_path = Path(path)
    if not plan_path.exists():
        return _load_error(plan_path, "missing_research_plan", "Research program plan is missing")
    try:
        payload = load_research_plan(plan_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return _load_error(
            plan_path,
            "invalid_research_plan_yaml",
            "Could not load the research program plan",
            error=str(exc),
        )
    return check_research_plan(payload, path=plan_path)


def check_research_plan(
    payload: Mapping[str, Any], *, path: str | Path | None = None
) -> ResearchPlanCheck:
    """Check shape and cross-field consistency; do not claim execution readiness."""

    plan_path = None if path is None else str(path)
    issues: list[ResearchPlanIssue] = []
    required_paths = (
        "schema_version",
        "kind",
        "program_id",
        "revision",
        "title",
        "control_issue",
        "question",
        "decision_unlocked",
        "research_question_ids",
        "protocol_defaults.version",
        "protocol_defaults.documentation",
        "decision_protocol.completion_outcomes",
        "decision_protocol.entry_rule",
        "decision_protocol.branch_rule",
        "execution_gates.discovery_sealed",
        "execution_gates.behavior_results_sealed",
        "execution_gates.phase_two_locked",
        "execution_gates.program_terminal",
        "synthesis_rules",
        "scope.model",
        "scope.environment",
        "scope.freshness",
        "hypotheses",
        "population.primary_independent_unit",
        "population.nested_units",
        "population.candidate_pool.max_families",
        "population.pool_assignment.discovery_candidates",
        "population.pool_assignment.confirmation_candidates",
        "population.required_eligible.discovery",
        "population.required_eligible.confirmation",
        "studies",
        "child_contract_defaults.required_lock_fields",
        "child_contract_defaults.required_evidence",
        "program_budget.max_persistent_gb",
        "program_budget.max_ephemeral_gb",
        "program_budget.min_free_space_gb",
        "program_budget.max_full_rollouts",
        "program_budget.max_action_generations",
        "program_budget.max_model_calls",
        "program_budget.max_simulator_steps",
        "program_budget.max_concurrent_hardware_children",
        "program_budget.check_before_every_hardware_child",
        "authority.automatic",
        "authority.requires_human",
        "program_stop_rules",
    )
    missing = [item for item in required_paths if _missing(payload, item)]
    if missing:
        issues.append(
            _issue(
                "research_plan_missing_fields",
                "Research program plan is missing required fields",
                plan_path,
                missing=missing,
            )
        )

    if payload.get("schema_version") != RESEARCH_PLAN_SCHEMA_VERSION:
        issues.append(
            _issue(
                "unsupported_research_plan_schema",
                "Research program plan has an unsupported schema version",
                plan_path,
                expected=RESEARCH_PLAN_SCHEMA_VERSION,
                observed=payload.get("schema_version"),
            )
        )
    if payload.get("kind") != RESEARCH_PLAN_KIND:
        issues.append(
            _issue(
                "invalid_research_plan_kind",
                "Research program plan has the wrong kind",
                plan_path,
                expected=RESEARCH_PLAN_KIND,
                observed=payload.get("kind"),
            )
        )

    _require_sequence_paths(
        payload,
        (
            "research_question_ids",
            "population.nested_units",
            "decision_protocol.completion_outcomes",
            "synthesis_rules",
            "child_contract_defaults.required_lock_fields",
            "child_contract_defaults.required_evidence",
            "authority.automatic",
            "authority.requires_human",
            "program_stop_rules",
        ),
        issues,
        plan_path,
    )
    _check_hypotheses(payload.get("hypotheses"), issues, plan_path)
    _check_population(payload, issues, plan_path)
    studies = _check_studies(payload, issues, plan_path)
    _check_budgets(payload, studies, issues, plan_path)

    try:
        fingerprint = research_plan_fingerprint(payload)
    except StrictResearchDataError as exc:
        issues.append(
            _issue(
                "noncanonical_research_plan",
                "Research program contains values that cannot be hashed safely",
                plan_path,
                error=str(exc),
            )
        )
        fingerprint = None
    study_ids = [str(study.get("id") or "") for study in studies]
    return ResearchPlanCheck(
        fingerprint=fingerprint,
        summary={
            "path": plan_path,
            "program_id": payload.get("program_id"),
            "revision": payload.get("revision"),
            "question": payload.get("question"),
            "hypothesis_count": len(_records(payload.get("hypotheses"))),
            "study_count": len(studies),
            "study_ids": study_ids,
            "child_job_manifests_required": len(studies),
            "execution_readiness_evaluated": False,
        },
        issues=tuple(issues),
    )


def format_research_plan_markdown(payload: Mapping[str, Any], check: ResearchPlanCheck) -> str:
    """Render the program in a compact form that does not imply it is runnable."""

    lines = [
        f"# {payload.get('title') or payload.get('program_id') or 'Research program'}",
        "",
        f"- Program: `{payload.get('program_id')}` revision `{payload.get('revision')}`",
        f"- Plan fingerprint: `{check.fingerprint}`",
        f"- Plan schema: `{'VALID' if check.valid else 'INVALID'}` "
        f"({len(check.errors)} errors, {len(check.warnings)} warnings)",
        "- Execution readiness: `NOT EVALUATED` — each study requires a separate "
        "locked job manifest and runner preflight.",
        "",
        "## Question",
        "",
        str(payload.get("question") or "-"),
        "",
        "## Decision this unlocks",
        "",
        str(payload.get("decision_unlocked") or "-"),
        "",
        "## Competing explanations",
        "",
    ]
    for hypothesis in _records(payload.get("hypotheses")):
        lines.append(f"- `{hypothesis.get('id')}` — {hypothesis.get('statement')}")
    lines.extend(["", "## Independently locked studies", ""])
    for study in _records(payload.get("studies")):
        required = ", ".join(
            str(item) for item in _sequence(_get(study, "entry_conditions.requires_all_completed"))
        )
        positive = ", ".join(
            str(item) for item in _sequence(_get(study, "entry_conditions.requires_any_positive"))
        )
        entry_parts = []
        if required:
            entry_parts.append(f"after {required} completes")
        if positive:
            entry_parts.append(f"after any positive: {positive}")
        entry_text = f"; {'; '.join(entry_parts)}" if entry_parts else "; program root"
        lines.append(f"- `{study.get('id')}` {study.get('title')}{entry_text}")
        lines.append(
            f"  Data boundary: `{_get(study, 'data_scope.family_pool')}` / "
            f"`{_get(study, 'data_scope.exposure_class')}`"
        )
        lines.append(f"  Decision gate: {study.get('advance_gate')}")
    lines.extend(["", "## Schema issues", ""])
    if not check.issues:
        lines.append("- None.")
    else:
        for issue in check.issues:
            lines.append(f"- {issue.severity.upper()} `{issue.code}`: {issue.message}")
    return "\n".join(lines) + "\n"


def _check_hypotheses(value: Any, issues: list[ResearchPlanIssue], path: str | None) -> None:
    hypotheses = _records(value)
    if not hypotheses:
        issues.append(_issue("invalid_hypotheses", "Hypotheses must be a list of mappings", path))
        return
    seen: set[str] = set()
    for index, hypothesis in enumerate(hypotheses):
        hypothesis_id = str(hypothesis.get("id") or "")
        if not hypothesis_id or _missing(hypothesis, "statement"):
            issues.append(
                _issue(
                    "incomplete_hypothesis",
                    "Each hypothesis needs a unique ID and statement",
                    path,
                    index=index,
                )
            )
        if not _nonempty_sequence(hypothesis.get("distinguishing_predictions")):
            issues.append(
                _issue(
                    "invalid_hypothesis_predictions",
                    "Each hypothesis needs a list of distinguishing predictions",
                    path,
                    hypothesis_id=hypothesis_id,
                )
            )
        if hypothesis_id in seen:
            issues.append(
                _issue(
                    "duplicate_hypothesis_id",
                    "Hypothesis IDs must be unique",
                    path,
                    hypothesis_id=hypothesis_id,
                )
            )
        seen.add(hypothesis_id)


def _check_population(
    payload: Mapping[str, Any], issues: list[ResearchPlanIssue], path: str | None
) -> None:
    values = {
        "max_families": _number(_get(payload, "population.candidate_pool.max_families")),
        "discovery_candidates": _number(
            _get(payload, "population.pool_assignment.discovery_candidates")
        ),
        "confirmation_candidates": _number(
            _get(payload, "population.pool_assignment.confirmation_candidates")
        ),
        "required_discovery": _number(_get(payload, "population.required_eligible.discovery")),
        "required_confirmation": _number(
            _get(payload, "population.required_eligible.confirmation")
        ),
    }
    if any(value is None or value <= 0 or not value.is_integer() for value in values.values()):
        issues.append(
            _issue(
                "invalid_population_counts",
                "Population counts must be positive integers",
                path,
                observed=values,
            )
        )
        return
    assert all(value is not None for value in values.values())
    if values["max_families"] < values["discovery_candidates"] + values["confirmation_candidates"]:
        issues.append(
            _issue(
                "candidate_pool_too_small",
                "Candidate pool cannot fill both preassigned pools",
                path,
                observed=values,
            )
        )
    if (
        values["discovery_candidates"] < values["required_discovery"]
        or values["confirmation_candidates"] < values["required_confirmation"]
    ):
        issues.append(
            _issue(
                "eligible_pool_arithmetic_invalid",
                "Each candidate pool must be large enough for its eligibility gate",
                path,
                observed=values,
            )
        )


def _check_studies(
    payload: Mapping[str, Any], issues: list[ResearchPlanIssue], path: str | None
) -> list[Mapping[str, Any]]:
    studies = _records(payload.get("studies"))
    if not studies:
        issues.append(_issue("invalid_studies", "Studies must be a list of mappings", path))
        return []
    required_fields = (
        "id",
        "title",
        "kind",
        "phase",
        "question",
        "entry_conditions.requires_all_completed",
        "entry_conditions.requires_any_positive",
        "entry_conditions.branch_mode",
        "entry_conditions.source_priority",
        "data_scope.family_pool",
        "data_scope.pool_phase",
        "data_scope.requires_gate",
        "data_scope.read_namespaces",
        "data_scope.write_namespace",
        "data_scope.selection_allowed",
        "data_scope.exposure_class",
        "entry_gate",
        "actions",
        "primary_claim.metric_id",
        "primary_claim.definition",
        "primary_claim.unit",
        "primary_claim.direction",
        "independent_unit",
        "controls",
        "child_contract.additional_lock_fields",
        "required_outputs",
        "advance_gate",
        "stop_rules",
        "outcome_actions.positive",
        "outcome_actions.negative",
        "outcome_actions.inconclusive",
        "outcome_actions.not_applicable",
        "outcome_actions.invalid",
        "allowed_conclusions",
        "forbidden_conclusions",
        "required_audits",
    )
    required_fields += tuple(f"budget.{name}" for name in STUDY_BUDGET_FIELDS)
    list_fields = (
        "entry_conditions.requires_all_completed",
        "entry_conditions.requires_any_positive",
        "entry_conditions.source_priority",
        "data_scope.read_namespaces",
        "actions",
        "controls",
        "child_contract.additional_lock_fields",
        "required_outputs",
        "stop_rules",
        "allowed_conclusions",
        "forbidden_conclusions",
        "required_audits",
    )
    seen: set[str] = set()
    dependencies: dict[str, list[str]] = {}
    for index, study in enumerate(studies):
        study_id = str(study.get("id") or "")
        missing = []
        for field_name in required_fields:
            present = (
                _has_path(study, field_name)
                if field_name.startswith("entry_conditions.")
                else not _missing(study, field_name)
            )
            if not present:
                missing.append(field_name)
        if missing:
            issues.append(
                _issue(
                    "study_missing_fields",
                    "A child study is missing required planning fields",
                    path,
                    study=study_id or index,
                    missing=missing,
                )
            )
        for field_name in list_fields:
            value = _get(study, field_name)
            allow_empty = field_name.startswith("entry_conditions.")
            if not _is_sequence(value) or (not allow_empty and not value):
                issues.append(
                    _issue(
                        "invalid_study_list",
                        "Study collection fields must use YAML lists",
                        path,
                        study=study_id or index,
                        field=field_name,
                    )
                )
        if study_id in seen or not study_id:
            issues.append(
                _issue(
                    "invalid_study_id",
                    "Study IDs must be non-empty and unique",
                    path,
                    study=study_id,
                )
            )
        seen.add(study_id)
        completed = [
            str(item) for item in _sequence(_get(study, "entry_conditions.requires_all_completed"))
        ]
        positive = [
            str(item) for item in _sequence(_get(study, "entry_conditions.requires_any_positive"))
        ]
        dependencies[study_id] = list(dict.fromkeys([*completed, *positive]))
        _check_entry_conditions(study, completed, positive, issues, path)
        kind = str(study.get("kind") or "")
        if kind not in STUDY_KINDS:
            issues.append(
                _issue(
                    "invalid_study_kind",
                    "Study kind is not recognized",
                    path,
                    study=study_id,
                    observed=kind,
                    allowed=sorted(STUDY_KINDS),
                )
            )
        phase = str(study.get("phase") or "")
        if phase not in STUDY_PHASES:
            issues.append(
                _issue(
                    "invalid_study_phase",
                    "Study phase is not recognized",
                    path,
                    study=study_id,
                    observed=phase,
                    allowed=sorted(STUDY_PHASES),
                )
            )
        if kind in SOURCE_BOUND_KINDS:
            source = str(study.get("source_claim_study") or "")
            if not source or source not in positive or source not in completed:
                issues.append(
                    _issue(
                        "invalid_source_claim_binding",
                        "Source-bound studies must name their one positive predecessor "
                        "as the source claim",
                        path,
                        study=study_id,
                        source=source,
                        completed=completed,
                        positive=positive,
                    )
                )
        if kind in CONFIRMATION_KINDS:
            defaults = set(
                str(item)
                for item in _sequence(_get(payload, "child_contract_defaults.required_lock_fields"))
            )
            additions = set(
                str(item)
                for item in _sequence(_get(study, "child_contract.additional_lock_fields"))
            )
            missing_locks = sorted(CONFIRMATION_LOCK_FIELDS - defaults - additions)
            if missing_locks:
                issues.append(
                    _issue(
                        "confirmation_lock_incomplete",
                        "Confirmation child contract is missing frozen fields",
                        path,
                        study=study_id,
                        missing=missing_locks,
                    )
                )
    known = set(dependencies)
    for study_id, required in dependencies.items():
        for dependency in required:
            if dependency not in known:
                issues.append(
                    _issue(
                        "unknown_study_dependency",
                        "Study depends on an unknown study",
                        path,
                        study=study_id,
                        dependency=dependency,
                    )
                )
    if _has_cycle(dependencies):
        issues.append(_issue("study_dependency_cycle", "Study dependencies contain a cycle", path))
    _check_outcome_actions(studies, issues, path)
    _check_reachability(studies, known, issues, path)
    _check_execution_gates(payload, studies, known, issues, path)
    _check_synthesis_rules(payload, issues, path)
    return studies


def _check_entry_conditions(
    study: Mapping[str, Any],
    completed: Sequence[str],
    positive: Sequence[str],
    issues: list[ResearchPlanIssue],
    path: str | None,
) -> None:
    study_id = str(study.get("id") or "")
    mode = str(_get(study, "entry_conditions.branch_mode") or "")
    priority = [str(item) for item in _sequence(_get(study, "entry_conditions.source_priority"))]
    if mode not in BRANCH_MODES:
        issues.append(
            _issue(
                "invalid_branch_mode",
                "Study entry conditions use an unknown branch mode",
                path,
                study=study_id,
                observed=mode,
                allowed=sorted(BRANCH_MODES),
            )
        )
    if len(priority) != len(set(priority)):
        issues.append(
            _issue(
                "duplicate_source_priority",
                "Source-priority entries must be unique",
                path,
                study=study_id,
            )
        )
    if mode == "one_per_positive_source" and (not positive or set(priority) != set(positive)):
        issues.append(
            _issue(
                "invalid_fork_sources",
                "A one-per-positive-source study must order every positive source exactly once",
                path,
                study=study_id,
                positive=list(positive),
                source_priority=priority,
            )
        )
    if mode == "inherit_source_branch" and len(positive) != 1:
        issues.append(
            _issue(
                "invalid_inherited_branch",
                "An inherited branch must name exactly one positive parent study template",
                path,
                study=study_id,
                positive=list(positive),
            )
        )
    if mode == "single" and priority:
        issues.append(
            _issue(
                "unexpected_source_priority",
                "A single study must not declare source-priority branches",
                path,
                study=study_id,
            )
        )
    family_pool = str(_get(study, "data_scope.family_pool") or "")
    if family_pool not in FAMILY_POOLS:
        issues.append(
            _issue(
                "invalid_family_pool",
                "Study data scope names an unknown family pool",
                path,
                study=study_id,
                observed=family_pool,
                allowed=sorted(FAMILY_POOLS),
            )
        )
    kind = str(study.get("kind") or "")
    if kind in CONFIRMATION_KINDS and family_pool != "confirmation":
        issues.append(
            _issue(
                "confirmation_uses_wrong_pool",
                "Confirmation studies must use only the confirmation family pool",
                path,
                study=study_id,
                observed=family_pool,
            )
        )
    selection_allowed = _get(study, "data_scope.selection_allowed")
    if not isinstance(selection_allowed, bool):
        issues.append(
            _issue(
                "invalid_selection_permission",
                "Study data scope must explicitly allow or forbid selection",
                path,
                study=study_id,
                observed=selection_allowed,
            )
        )
    if family_pool == "confirmation" and selection_allowed is not False:
        issues.append(
            _issue(
                "confirmation_selection_allowed",
                "Selection is forbidden whenever a study reads confirmation families",
                path,
                study=study_id,
            )
        )
    write_namespace = str(_get(study, "data_scope.write_namespace") or "")
    if not write_namespace:
        issues.append(
            _issue(
                "missing_write_namespace",
                "Every study must write to one named evidence namespace",
                path,
                study=study_id,
            )
        )
    if kind in {"internal_discovery", "causal_discovery"} and family_pool != "discovery":
        issues.append(
            _issue(
                "discovery_uses_wrong_pool",
                "Internal and causal discovery must use only discovery families",
                path,
                study=study_id,
                observed=family_pool,
            )
        )
    if kind == "semantic_readout" and family_pool != "discovery":
        issues.append(
            _issue(
                "semantic_confirmation_leak",
                "The program's semantic readout is discovery-only to protect causal confirmation",
                path,
                study=study_id,
                observed=family_pool,
            )
        )
    if completed and study_id in completed:
        issues.append(
            _issue(
                "self_study_dependency",
                "A study cannot require itself to complete",
                path,
                study=study_id,
            )
        )


def _check_execution_gates(
    payload: Mapping[str, Any],
    studies: Sequence[Mapping[str, Any]],
    known: set[str],
    issues: list[ResearchPlanIssue],
    path: str | None,
) -> None:
    gates = payload.get("execution_gates")
    if not isinstance(gates, Mapping) or not gates:
        issues.append(_issue("invalid_execution_gates", "Execution gates must be a mapping", path))
        return
    allowed_gates = {"none", *[str(key) for key in gates]}
    namespaces: list[str] = []
    for study in studies:
        study_id = str(study.get("id") or "")
        required_gate = str(_get(study, "data_scope.requires_gate") or "")
        if required_gate not in allowed_gates:
            issues.append(
                _issue(
                    "unknown_required_gate",
                    "Study data access depends on an unknown execution gate",
                    path,
                    study=study_id,
                    gate=required_gate,
                )
            )
        namespaces.append(str(_get(study, "data_scope.write_namespace") or ""))
    duplicates = sorted(
        namespace for namespace in set(namespaces) if namespace and namespaces.count(namespace) > 1
    )
    if duplicates:
        issues.append(
            _issue(
                "duplicate_write_namespace",
                "Study output namespaces must be unique",
                path,
                namespaces=duplicates,
            )
        )
    for gate_id, gate in gates.items():
        if not isinstance(gate, Mapping) or _missing(gate, "rule"):
            issues.append(
                _issue(
                    "invalid_execution_gate",
                    "Every execution gate needs a mapping and human-readable rule",
                    path,
                    gate=gate_id,
                )
            )
            continue
        for list_name in (
            "all_terminal",
            "all_activated_terminal",
            "all_activated_discovery_terminal",
        ):
            if list_name not in gate:
                continue
            values = _sequence(gate.get(list_name))
            if not values or any(str(item) not in known for item in values):
                issues.append(
                    _issue(
                        "invalid_gate_study_list",
                        "Execution gate references missing or unknown studies",
                        path,
                        gate=gate_id,
                        field=list_name,
                    )
                )
        locks = gate.get("lock_before_next_pool_access")
        if locks is None:
            continue
        if not _is_sequence(locks) or not locks:
            issues.append(
                _issue(
                    "invalid_gate_locks",
                    "Gate lock conditions must be a non-empty list",
                    path,
                    gate=gate_id,
                )
            )
            continue
        for lock in locks:
            if not isinstance(lock, Mapping) or set(lock) != {
                "child_study_id",
                "source_study_id",
                "source_outcome",
            }:
                issues.append(
                    _issue(
                        "invalid_gate_lock",
                        "Gate lock entries need child, source, and source outcome",
                        path,
                        gate=gate_id,
                    )
                )
                continue
            child_id = str(lock["child_study_id"])
            source_id = str(lock["source_study_id"])
            if (
                child_id not in known
                or source_id not in known
                or lock["source_outcome"] != "positive"
            ):
                issues.append(
                    _issue(
                        "invalid_gate_lock_reference",
                        "Gate lock references an unknown study or unsupported outcome",
                        path,
                        gate=gate_id,
                        child=child_id,
                        source=source_id,
                    )
                )


def _check_synthesis_rules(
    payload: Mapping[str, Any], issues: list[ResearchPlanIssue], path: str | None
) -> None:
    rules = _records(payload.get("synthesis_rules"))
    if not rules:
        issues.append(_issue("invalid_synthesis_rules", "Synthesis rules must be mappings", path))
        return
    seen: set[str] = set()
    for rule in rules:
        rule_id = str(rule.get("id") or "")
        if not rule_id or _missing(rule, "when") or _missing(rule, "interpretation"):
            issues.append(
                _issue(
                    "incomplete_synthesis_rule",
                    "Every synthesis rule needs an ID, condition, and interpretation",
                    path,
                )
            )
        if rule_id in seen:
            issues.append(
                _issue(
                    "duplicate_synthesis_rule",
                    "Synthesis-rule IDs must be unique",
                    path,
                    rule_id=rule_id,
                )
            )
        seen.add(rule_id)


def _check_outcome_actions(
    studies: Sequence[Mapping[str, Any]],
    issues: list[ResearchPlanIssue],
    path: str | None,
) -> None:
    for study in studies:
        study_id = str(study.get("id") or "")
        for outcome in (
            "positive",
            "negative",
            "inconclusive",
            "not_applicable",
            "invalid",
        ):
            action = str(_get(study, f"outcome_actions.{outcome}") or "")
            if action not in OUTCOME_ACTIONS:
                issues.append(
                    _issue(
                        "invalid_outcome_action",
                        "Every scientific outcome needs one recognized program action",
                        path,
                        study=study_id,
                        outcome=outcome,
                        observed=action,
                        allowed=sorted(OUTCOME_ACTIONS),
                    )
                )


def _check_reachability(
    studies: Sequence[Mapping[str, Any]],
    known: set[str],
    issues: list[ResearchPlanIssue],
    path: str | None,
) -> None:
    reachable: set[str] = set()
    changed = True
    while changed:
        changed = False
        for study in studies:
            study_id = str(study.get("id") or "")
            if study_id in reachable:
                continue
            completed = set(
                str(item)
                for item in _sequence(_get(study, "entry_conditions.requires_all_completed"))
            )
            positive = set(
                str(item)
                for item in _sequence(_get(study, "entry_conditions.requires_any_positive"))
            )
            if completed <= reachable and (not positive or bool(positive & reachable)):
                reachable.add(study_id)
                changed = True
    unreachable = sorted(known - reachable)
    if unreachable:
        issues.append(
            _issue(
                "unreachable_studies",
                "Some studies cannot be reached through their declared entry conditions",
                path,
                studies=unreachable,
            )
        )


def _check_budgets(
    payload: Mapping[str, Any],
    studies: Sequence[Mapping[str, Any]],
    issues: list[ResearchPlanIssue],
    path: str | None,
) -> None:
    program_values: dict[str, float] = {}
    for name in PROGRAM_BUDGET_FIELDS:
        field_name = f"program_budget.{name}"
        value = _number(_get(payload, field_name))
        if (
            value is None
            or value <= 0
            or (
                name.startswith("max_")
                and name.endswith(("rollouts", "generations", "calls", "steps", "children"))
                and not value.is_integer()
            )
        ):
            issues.append(
                _issue(
                    "invalid_program_budget",
                    "Program budgets must be positive finite values with integer count caps",
                    path,
                    field=field_name,
                    observed=_get(payload, field_name),
                )
            )
        else:
            program_values[name] = value
    if _get(payload, "program_budget.check_before_every_hardware_child") is not True:
        issues.append(
            _issue(
                "hardware_budget_check_disabled",
                "The program must check cumulative budget and free space before "
                "every hardware child",
                path,
            )
        )
    totals = {name: 0.0 for name in SUMMED_BUDGETS}
    for study in studies:
        observed: dict[str, float] = {}
        for name in STUDY_BUDGET_FIELDS:
            raw = _get(study, f"budget.{name}")
            value = _number(raw)
            count_field = name in {
                "max_instances",
                "max_model_calls",
                "max_action_generations",
                "max_full_rollouts",
                "max_simulator_steps",
            }
            invalid = value is None or value < 0 or (count_field and not value.is_integer())
            if name == "max_instances" and value == 0:
                invalid = True
            if invalid:
                issues.append(
                    _issue(
                        "invalid_study_budget",
                        "Every study budget needs finite nonnegative caps and at least "
                        "one instance",
                        path,
                        study=study.get("id"),
                        field=name,
                        observed=raw,
                    )
                )
            else:
                observed[name] = value
        instances = observed.get("max_instances", 0)
        for child_name in SUMMED_BUDGETS:
            totals[child_name] += observed.get(child_name, 0) * instances
        ephemeral = observed.get("max_ephemeral_gb")
        if (
            ephemeral is not None
            and "max_ephemeral_gb" in program_values
            and ephemeral > program_values["max_ephemeral_gb"]
        ):
            issues.append(
                _issue(
                    "study_ephemeral_storage_exceeds_program_budget",
                    "A child ephemeral-storage cap exceeds the whole-program working-space cap",
                    path,
                    study=study.get("id"),
                    observed=ephemeral,
                    program_max=program_values["max_ephemeral_gb"],
                )
            )
    for child_name, program_name in SUMMED_BUDGETS.items():
        program_max = program_values.get(program_name)
        if program_max is not None and totals[child_name] > program_max:
            issues.append(
                _issue(
                    "study_caps_exceed_program_budget",
                    "Worst-case study caps exceed a cumulative program cap",
                    path,
                    dimension=program_name,
                    study_total=totals[child_name],
                    program_max=program_max,
                )
            )


def _require_sequence_paths(
    payload: Mapping[str, Any],
    paths: Sequence[str],
    issues: list[ResearchPlanIssue],
    plan_path: str | None,
) -> None:
    for field_name in paths:
        if not _nonempty_sequence(_get(payload, field_name)):
            issues.append(
                _issue(
                    "invalid_research_plan_list",
                    "Research program collection fields must use non-empty YAML lists",
                    plan_path,
                    field=field_name,
                )
            )


def _has_cycle(dependencies: Mapping[str, Sequence[str]]) -> bool:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for dependency in dependencies.get(node, ()):
            if dependency in dependencies and visit(dependency):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in dependencies)


def _load_error(path: Path, code: str, message: str, **details: Any) -> ResearchPlanCheck:
    return ResearchPlanCheck(
        fingerprint=None,
        summary={"path": str(path)},
        issues=(ResearchPlanIssue(code=code, message=message, path=str(path), details=details),),
    )


def _records(value: Any) -> list[Mapping[str, Any]]:
    if not _is_sequence(value) or not all(isinstance(item, Mapping) for item in value):
        return []
    return list(value)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _nonempty_sequence(value: Any) -> bool:
    return bool(value) and _is_sequence(value)


def _sequence(value: Any) -> Sequence[Any]:
    return value if _is_sequence(value) else ()


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


def _missing(payload: Mapping[str, Any], path: str) -> bool:
    value = _get(payload, path)
    return value is None or value == "" or value == [] or value == {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _issue(code: str, message: str, path: str | None, **details: Any) -> ResearchPlanIssue:
    return ResearchPlanIssue(code=code, message=message, path=path, details=details)


__all__ = [
    "BRANCH_MODES",
    "CONFIRMATION_KINDS",
    "CONFIRMATION_LOCK_FIELDS",
    "OUTCOME_ACTIONS",
    "RESEARCH_PLAN_KIND",
    "RESEARCH_PLAN_SCHEMA_VERSION",
    "STUDY_KINDS",
    "ResearchPlanCheck",
    "ResearchPlanIssue",
    "check_research_plan",
    "check_research_plan_file",
    "format_research_plan_markdown",
    "load_research_plan",
    "research_plan_fingerprint",
]
