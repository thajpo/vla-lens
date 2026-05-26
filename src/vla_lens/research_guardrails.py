"""Research guardrails for config, dataset, and probe-claim validation.

These helpers are intentionally dependency-light and side-effect free. They do
not run PI0.5 capture, load hardware models, or write artifacts; they only read
local configs, sidecars, and dataset metadata.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import yaml

from vla_lens.artifacts import LensArtifact
from vla_lens.capture.lerobot_v3 import LEROBOT_INFO_PATH, validate_lerobot_v3_dataset
from vla_lens.traces import TraceDataset
from vla_lens.validation import CAPTURE_PROFILE_REQUIREMENTS, validate_trace_dataset

AUDIT_CAPTURE_PROFILES = frozenset({"audit_sampled", "audit_windowed", "audit_full"})
RUNTIME_CONFIG_FIELDS = ("python_executable", "pythonpath", "device", "dtype")
CLAIM_LEVELS = (
    "integration_smoke",
    "decodable",
    "candidate_mechanism",
    "causal_intervention",
)


@dataclass(frozen=True, slots=True)
class GuardrailIssue:
    """One guardrail issue."""

    code: str
    message: str
    severity: str = "error"
    path: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True, slots=True)
class GuardrailReport:
    """A side-effect-free validation report."""

    name: str
    summary: Mapping[str, Any] = field(default_factory=dict)
    issues: tuple[GuardrailIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def errors(self) -> tuple[GuardrailIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[GuardrailIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "valid": self.valid,
            "summary": dict(self.summary),
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True, slots=True)
class ProbeClaimGateResult:
    """Claim-level classification for a saved probe artifact."""

    classified_level: str | None
    claimed_level: str | None
    valid: bool
    issues: tuple[GuardrailIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "classified_level": self.classified_level,
            "claimed_level": self.claimed_level,
            "valid": self.valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def lint_research_configs(
    root: str | Path = ".",
    *,
    config_globs: Sequence[str] = ("configs/*.yaml", "configs/probes/*.yaml"),
    episode_plans: Sequence[str | Path] = (),
    audit_contracts: Sequence[str | Path] = (),
) -> GuardrailReport:
    """Lint research configs and optional episode-plan/audit-contract files."""

    repo_root = Path(root)
    episode_plan_paths = tuple(episode_plans)
    audit_contract_paths = tuple(audit_contracts)
    issues: list[GuardrailIssue] = []
    config_paths: list[Path] = []
    for pattern in config_globs:
        config_paths.extend(sorted(repo_root.glob(pattern)))
    seen: set[Path] = set()
    for path in config_paths:
        if path in seen:
            continue
        seen.add(path)
        issues.extend(_lint_yaml_config(path, repo_root=repo_root))
    for plan_path in episode_plan_paths:
        issues.extend(lint_episode_plan(plan_path).issues)
    for contract_path in audit_contract_paths:
        issues.extend(validate_audit_capture_contract_file(contract_path).issues)

    return GuardrailReport(
        name="research_config_guardrails",
        summary={
            "config_files": len(seen),
            "episode_plans": len(episode_plan_paths),
            "audit_contracts": len(audit_contract_paths),
        },
        issues=tuple(issues),
    )


def lint_episode_plan(
    path: str | Path,
    *,
    max_broad_audit_rows: int = 20,
) -> GuardrailReport:
    """Lint one episode-plan CSV without writing capture outputs."""

    plan_path = Path(path)
    issues: list[GuardrailIssue] = []
    if not plan_path.exists():
        return GuardrailReport(
            name="episode_plan_guardrails",
            summary={"path": str(plan_path), "rows": 0},
            issues=(_issue("missing_episode_plan", "Episode plan does not exist", plan_path),),
        )

    with plan_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"dataset_id", "benchmark", "task_id", "seed", "split", "capture_profile"}
        missing = sorted(required - fieldnames)
        if missing:
            issues.append(
                _issue(
                    "episode_plan_missing_columns",
                    "Episode plan is missing required columns",
                    plan_path,
                    details={"missing": missing},
                )
            )
        rows = list(reader)

    profile_counts = _counts(row.get("capture_profile") for row in rows)
    split_counts = _counts(row.get("split") for row in rows)
    dataset_ids = sorted(
        {str(row.get("dataset_id") or "") for row in rows if row.get("dataset_id")}
    )
    broad_plan = any(_looks_broad_1000(value) for value in [plan_path.name, *dataset_ids])
    audit_rows = [
        row for row in rows if str(row.get("capture_profile") or "") in AUDIT_CAPTURE_PROFILES
    ]
    if audit_rows and (broad_plan or len(audit_rows) > max_broad_audit_rows):
        issues.append(
            _issue(
                "broad_episode_plan_audit_profile",
                "Broad episode plans must not request audit capture profiles",
                plan_path,
                details={
                    "audit_rows": len(audit_rows),
                    "row_count": len(rows),
                    "profiles": profile_counts,
                },
            )
        )
    if rows:
        categories = {_split_category(value) for value in split_counts if value}
        if "train" not in categories:
            issues.append(
                _issue(
                    "episode_plan_missing_train_split",
                    "Episode plan has no train split category",
                    plan_path,
                    severity="warning",
                    details={"splits": split_counts},
                )
            )
        if "test" not in categories:
            issues.append(
                _issue(
                    "episode_plan_missing_test_split",
                    "Episode plan has no test split category",
                    plan_path,
                    severity="warning",
                    details={"splits": split_counts},
                )
            )

    return GuardrailReport(
        name="episode_plan_guardrails",
        summary={
            "path": str(plan_path),
            "rows": len(rows),
            "datasets": dataset_ids,
            "capture_profiles": profile_counts,
            "splits": split_counts,
        },
        issues=tuple(issues),
    )


def check_dataset_trust(
    root: str | Path,
    *,
    require_splits: bool = True,
    require_activation_coverage: bool = True,
    require_artifacts: bool = True,
    require_outcome_balance: bool = True,
) -> GuardrailReport:
    """Validate a local dataset root for saved-trace research use.

    This opens an existing dataset and reads metadata/tables only. It does not
    run capture, replay, model loading, or simulator code.
    """

    dataset_root = Path(root)
    issues: list[GuardrailIssue] = []
    try:
        dataset = TraceDataset.open(dataset_root)
    except Exception as exc:
        return GuardrailReport(
            name="dataset_trust_gate",
            summary={"root": str(dataset_root)},
            issues=(
                _issue(
                    "dataset_open_failed",
                    "Could not open dataset root",
                    dataset_root,
                    details={"error": str(exc)},
                ),
            ),
        )

    issues.extend(_dataset_schema_issues(dataset_root, dataset))
    episode_index = dataset.episode_index
    model_sites = dataset.model_site_index
    trace_ids = set(_string_column(episode_index, "trace_id"))
    activation_trace_ids = set(_string_column(model_sites, "trace_id"))
    activation_coverage_ratio = (
        len(activation_trace_ids & trace_ids) / len(trace_ids) if trace_ids else 0.0
    )
    if require_activation_coverage and model_sites.empty:
        issues.append(
            _issue(
                "missing_activation_coverage",
                "Dataset has no model-site activation coverage in the VLA Lens overlay",
                dataset_root,
            )
        )
    elif require_activation_coverage and activation_coverage_ratio < 1.0:
        issues.append(
            _issue(
                "partial_activation_coverage",
                "Not every episode has model-site activation coverage",
                dataset_root,
                severity="warning",
                details={
                    "episodes": len(trace_ids),
                    "covered_episodes": len(activation_trace_ids & trace_ids),
                    "coverage_ratio": activation_coverage_ratio,
                },
            )
        )

    split_summary = _check_probe_splits(
        dataset,
        dataset_root,
        require_splits=require_splits,
        issues=issues,
    )
    outcome_counts = _counts(episode_index["outcome"]) if "outcome" in episode_index else {}
    if require_outcome_balance:
        known_outcomes = {
            outcome: count
            for outcome, count in outcome_counts.items()
            if outcome and outcome.lower() not in {"unknown", "nan", "none"}
        }
        if len(known_outcomes) < 2:
            issues.append(
                _issue(
                    "weak_outcome_balance",
                    "Research trust gate requires at least two known outcome classes",
                    dataset_root,
                    details={"outcomes": outcome_counts},
                )
            )

    artifact_summary = _check_artifact_freshness(
        dataset,
        require_artifacts=require_artifacts,
        issues=issues,
    )

    activation_summary = (
        model_sites.groupby(
            [column for column in ("tensor_type", "token_kind", "layer") if column in model_sites],
            dropna=False,
        )
        .size()
        .reset_index(name="rows")
        .head(20)
        .to_dict("records")
        if not model_sites.empty
        else []
    )
    return GuardrailReport(
        name="dataset_trust_gate",
        summary={
            "root": str(dataset_root),
            "episodes": int(len(episode_index)),
            "trace_ids": sorted(trace_ids),
            "activation_site_rows": int(len(model_sites)),
            "activation_coverage_ratio": activation_coverage_ratio,
            "activation_summary": _jsonable(activation_summary),
            "outcomes": outcome_counts,
            "splits": split_summary,
            "artifacts": artifact_summary,
        },
        issues=tuple(issues),
    )


def validate_probe_claim_artifact(
    artifact: LensArtifact | Mapping[str, Any],
    *,
    claimed_level: str | None = None,
) -> ProbeClaimGateResult:
    """Classify and validate the evidence level a probe artifact can support."""

    payload = artifact.to_dict() if isinstance(artifact, LensArtifact) else dict(artifact)
    declared = claimed_level or _declared_claim_level(payload)
    issues: list[GuardrailIssue] = []
    for level in CLAIM_LEVELS:
        missing = _missing_claim_paths(payload, level)
        if missing:
            continue
        classified = level
    if "classified" not in locals():
        classified = None

    if declared is not None and declared not in CLAIM_LEVELS:
        issues.append(
            GuardrailIssue(
                code="unknown_claim_level",
                message="Unknown probe claim level",
                details={"claimed_level": declared, "allowed": list(CLAIM_LEVELS)},
            )
        )
    elif declared is not None:
        missing = _missing_claim_paths(payload, declared)
        if missing:
            issues.append(
                GuardrailIssue(
                    code="claim_level_missing_evidence",
                    message="Probe artifact is missing evidence required for its claimed level",
                    details={"claimed_level": declared, "missing": missing},
                )
            )
    return ProbeClaimGateResult(
        classified_level=classified,
        claimed_level=declared,
        valid=not issues,
        issues=tuple(issues),
    )


def validate_audit_capture_contract_file(path: str | Path) -> GuardrailReport:
    """Load and validate an audit/circuit capture contract YAML."""

    contract_path = Path(path)
    if not contract_path.exists():
        return GuardrailReport(
            name="audit_capture_contract",
            summary={"path": str(contract_path)},
            issues=(
                _issue(
                    "missing_audit_contract",
                    "Audit contract file does not exist",
                    contract_path,
                ),
            ),
        )
    try:
        payload = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return GuardrailReport(
            name="audit_capture_contract",
            summary={"path": str(contract_path)},
            issues=(
                _issue(
                    "invalid_yaml",
                    "Could not parse audit capture contract YAML",
                    contract_path,
                    details={"error": str(exc)},
                ),
            ),
        )
    return validate_audit_capture_contract(payload, path=contract_path)


def validate_audit_capture_contract(
    payload: Mapping[str, Any],
    *,
    path: str | Path | None = None,
) -> GuardrailReport:
    """Validate an explicit audit/circuit capture contract, not a capture run."""

    contract_path = None if path is None else Path(path)
    issues: list[GuardrailIssue] = []
    if not isinstance(payload, Mapping):
        issues.append(
            _issue(
                "invalid_audit_contract",
                "Audit contract must be a mapping",
                contract_path,
            )
        )
        return GuardrailReport(name="audit_capture_contract", issues=tuple(issues))

    profile = str(payload.get("capture_profile") or "")
    template = bool(payload.get("template"))
    if profile not in AUDIT_CAPTURE_PROFILES:
        issues.append(
            _issue(
                "audit_contract_requires_audit_profile",
                "Audit capture contract must choose an audit capture profile",
                contract_path,
                details={"capture_profile": profile, "allowed": sorted(AUDIT_CAPTURE_PROFILES)},
            )
        )

    required_paths = (
        "kind",
        "question",
        "hypothesis",
        "source_dataset.root",
        "source_traces",
        "capture_profile",
        "sites",
        "episode_selection",
        "budget.max_episodes",
        "budget.max_estimated_gb",
        "evidence_plan.required_outputs",
        "stop_rules",
    )
    missing = [item for item in required_paths if _is_missing_path(payload, item)]
    if missing:
        issues.append(
            _issue(
                "audit_contract_missing_fields",
                "Audit capture contract is missing required planning fields",
                contract_path,
                details={"missing": missing},
            )
        )

    max_episodes = _optional_int(_path_get(payload, "budget.max_episodes"))
    if max_episodes is not None and max_episodes > 20:
        issues.append(
            _issue(
                "audit_contract_too_broad",
                "Audit contracts must be narrow circuit/debug plans, not broad capture plans",
                contract_path,
                details={"max_episodes": max_episodes},
            )
        )
    source_traces = _path_get(payload, "source_traces")
    if not template and isinstance(source_traces, str) and source_traces.lower() in {"all", "*"}:
        issues.append(
            _issue(
                "audit_contract_requires_explicit_traces",
                "Audit contracts must name explicit source traces or an explicit selection query",
                contract_path,
            )
        )
    if not template and isinstance(source_traces, Sequence) and not isinstance(source_traces, str):
        if not source_traces:
            issues.append(
                _issue(
                    "audit_contract_requires_explicit_traces",
                    "Audit contracts must include at least one source trace",
                    contract_path,
                )
            )
    broad_matrix_fields = {"benchmarks", "task_ids", "start_seed", "seeds_per_task"}
    present_matrix = sorted(field for field in broad_matrix_fields if field in payload)
    if present_matrix:
        issues.append(
            _issue(
                "audit_contract_contains_broad_matrix",
                "Audit contracts should not contain broad batch matrix fields",
                contract_path,
                details={"fields": present_matrix},
            )
        )

    return GuardrailReport(
        name="audit_capture_contract",
        summary={
            "path": None if contract_path is None else str(contract_path),
            "capture_profile": profile,
            "template": template,
            "max_episodes": max_episodes,
        },
        issues=tuple(issues),
    )


def _lint_yaml_config(path: Path, *, repo_root: Path) -> list[GuardrailIssue]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        return [
            _issue(
                "invalid_yaml",
                "Could not parse YAML config",
                path,
                details={"error": str(exc)},
            )
        ]
    if not isinstance(payload, Mapping):
        return [_issue("invalid_yaml", "YAML config should be a mapping", path)]
    if payload.get("kind") == "audit_capture_contract":
        return list(validate_audit_capture_contract(payload, path=path).issues)
    if "capture_profile" in payload or "output_root" in payload:
        return _lint_capture_config(path, payload)
    if "specs" in payload:
        return _lint_probe_campaign(path, payload, repo_root=repo_root)
    if "features" in payload or "target" in payload:
        return _lint_probe_spec(path, payload)
    return []


def _lint_capture_config(path: Path, payload: Mapping[str, Any]) -> list[GuardrailIssue]:
    issues: list[GuardrailIssue] = []
    name_parts = [
        path.name,
        str(payload.get("name") or ""),
        str(payload.get("dataset_id") or ""),
        str(payload.get("output_root") or ""),
    ]
    broad_1000 = any(_looks_broad_1000(value) for value in name_parts)
    if broad_1000 and payload.get("requires_episode_plan") is not True:
        issues.append(
            _issue(
                "broad_1000_requires_episode_plan",
                "Broad-1000 capture configs must require an explicit episode plan",
                path,
            )
        )

    profile = str(payload.get("capture_profile") or "")
    estimated_episodes = _estimate_config_episodes(payload)
    if profile not in CAPTURE_PROFILE_REQUIREMENTS:
        issues.append(
            _issue(
                "unknown_capture_profile",
                "Capture config uses an unknown capture profile",
                path,
                details={"capture_profile": profile},
            )
        )
    if profile in AUDIT_CAPTURE_PROFILES:
        too_many = estimated_episodes is not None and estimated_episodes > 20
        broadish = any(_looks_broad_capture(value) for value in name_parts)
        if too_many or broadish:
            issues.append(
                _issue(
                    "broad_config_audit_profile",
                    "Broad capture configs must not use audit capture profiles",
                    path,
                    details={
                        "capture_profile": profile,
                        "estimated_episodes": estimated_episodes,
                    },
                )
            )
    calibration = payload.get("full_calibration")
    if isinstance(calibration, Mapping) and calibration.get("enabled"):
        calibration_profile = str(calibration.get("capture_profile") or "")
        episodes = calibration.get("episodes")
        episode_count = len(episodes) if isinstance(episodes, Sequence) else None
        if calibration_profile in AUDIT_CAPTURE_PROFILES and episode_count and episode_count > 5:
            issues.append(
                _issue(
                    "full_calibration_too_broad",
                    "Audit calibration should stay tiny and explicitly selected",
                    path,
                    details={"episodes": episode_count, "capture_profile": calibration_profile},
                )
            )

    for field_name in RUNTIME_CONFIG_FIELDS:
        if field_name not in payload:
            continue
        value = str(payload.get(field_name) or "")
        severity = "warning"
        code = "runtime_field_ignored_by_wrapper"
        if field_name == "python_executable" and _is_unsafe_python_executable(value):
            severity = "error"
            code = "unsafe_python_executable"
        issues.append(
            _issue(
                code,
                "Runtime fields in capture configs are machine-local; wrappers should own them",
                path,
                severity=severity,
                details={"field": field_name, "value": value},
            )
        )
    return issues


def _lint_probe_campaign(
    path: Path,
    payload: Mapping[str, Any],
    *,
    repo_root: Path,
) -> list[GuardrailIssue]:
    issues: list[GuardrailIssue] = []
    specs = payload.get("specs")
    if not isinstance(specs, Sequence) or isinstance(specs, str) or not specs:
        return [
            _issue(
                "probe_campaign_missing_specs",
                "Probe campaign config should list spec paths",
                path,
            )
        ]
    for item in specs:
        spec_path = repo_root / str(item)
        if not spec_path.exists():
            issues.append(
                _issue(
                    "probe_campaign_missing_spec_file",
                    "Probe campaign references a missing spec file",
                    path,
                    details={"spec": str(item)},
                )
            )
    return issues


def _lint_probe_spec(path: Path, payload: Mapping[str, Any]) -> list[GuardrailIssue]:
    issues: list[GuardrailIssue] = []
    for field_name in ("capture_profile", *RUNTIME_CONFIG_FIELDS):
        if field_name in payload:
            issues.append(
                _issue(
                    "probe_spec_runtime_field",
                    "Probe specs should describe analysis only, not capture/runtime settings",
                    path,
                    details={"field": field_name},
                )
            )
    broad_1000 = _looks_broad_1000(path.name) or _looks_broad_1000(str(payload.get("name") or ""))
    split = payload.get("split")
    if broad_1000:
        if not isinstance(split, Mapping):
            issues.append(
                _issue(
                    "broad_probe_missing_split",
                    "Broad-1000 probe specs require explicit held-out split settings",
                    path,
                )
            )
        else:
            split_kind = str(split.get("kind") or "")
            if split_kind in {"random_row", "random_episode", ""}:
                issues.append(
                    _issue(
                        "broad_probe_weak_split",
                        "Broad-1000 probe specs should use task/layout-safe held-out splits",
                        path,
                        details={"split_kind": split_kind},
                    )
                )
        baselines = [str(item) for item in payload.get("baseline") or ()]
        baseline_text = {item.lower() for item in baselines}
        if not ({"benchmark", "task_id", "task"} & baseline_text):
            issues.append(
                _issue(
                    "broad_probe_missing_metadata_baseline",
                    "Broad-1000 probe specs need metadata baselines before decodability claims",
                    path,
                    details={"baseline": baselines},
                )
            )
    return issues


def _dataset_schema_issues(root: Path, dataset: TraceDataset) -> list[GuardrailIssue]:
    issues: list[GuardrailIssue] = []
    lerobot_roots = _lerobot_roots_for_dataset(root, dataset)
    if lerobot_roots:
        for dataset_root in lerobot_roots:
            result = validate_lerobot_v3_dataset(dataset_root)
            for error in result.errors:
                issues.append(
                    _issue(
                        f"schema_{error.code}",
                        error.message,
                        Path(error.path) if error.path else dataset_root,
                        details=error.details,
                    )
                )
            for warning in result.warnings:
                issues.append(
                    _issue(
                        f"schema_{warning.code}",
                        warning.message,
                        Path(warning.path) if warning.path else dataset_root,
                        severity="warning",
                        details=warning.details,
                    )
                )
        return issues

    result = validate_trace_dataset(dataset)
    for trace in result.traces:
        for error in trace.errors:
            issues.append(
                _issue(
                    f"schema_{error.get('code', 'trace_error')}",
                    str(error.get("message") or "Trace schema validation failed"),
                    root,
                    details={"trace_id": trace.trace_id, **dict(error)},
                )
            )
        for warning in trace.warnings:
            issues.append(
                _issue(
                    f"schema_{warning.get('code', 'trace_warning')}",
                    str(warning.get("message") or "Trace schema validation warning"),
                    root,
                    severity="warning",
                    details={"trace_id": trace.trace_id, **dict(warning)},
                )
            )
    return issues


def _check_probe_splits(
    dataset: TraceDataset,
    root: Path,
    *,
    require_splits: bool,
    issues: list[GuardrailIssue],
) -> dict[str, Any]:
    split_path = root / "probe_splits.csv"
    if not split_path.exists():
        if require_splits:
            issues.append(
                _issue(
                    "missing_probe_splits",
                    "Dataset trust gate requires probe_splits.csv at the checked root",
                    root,
                )
            )
        return {}

    try:
        table = pd.read_csv(split_path)
    except Exception as exc:
        issues.append(
            _issue(
                "unreadable_probe_splits",
                "Could not read probe_splits.csv",
                split_path,
                details={"error": str(exc)},
            )
        )
        return {"path": str(split_path)}
    missing = sorted({"trace_id", "split"} - set(table.columns))
    if missing:
        issues.append(
            _issue(
                "probe_splits_missing_columns",
                "probe_splits.csv is missing required columns",
                split_path,
                details={"missing": missing},
            )
        )
        return {"path": str(split_path), "rows": int(len(table))}
    trace_ids = set(_string_column(dataset.episode_index, "trace_id"))
    split_trace_ids = set(_string_column(table, "trace_id"))
    missing_traces = sorted(trace_ids - split_trace_ids)
    if missing_traces:
        issues.append(
            _issue(
                "probe_splits_missing_traces",
                "probe_splits.csv does not cover every opened episode",
                split_path,
                details={
                    "missing_trace_ids": missing_traces[:20],
                    "missing_count": len(missing_traces),
                },
            )
        )
    unknown_traces = sorted(split_trace_ids - trace_ids)
    if unknown_traces:
        issues.append(
            _issue(
                "probe_splits_unknown_traces",
                "probe_splits.csv references traces not present in the opened dataset",
                split_path,
                severity="warning",
                details={
                    "unknown_trace_ids": unknown_traces[:20],
                    "unknown_count": len(unknown_traces),
                },
            )
        )
    split_counts = _counts(table["split"])
    categories = {_split_category(value) for value in split_counts}
    for required in ("train", "test"):
        if required not in categories:
            issues.append(
                _issue(
                    f"probe_splits_missing_{required}",
                    f"probe_splits.csv has no {required} split category",
                    split_path,
                    details={"splits": split_counts},
                )
            )
    if "validation" not in categories:
        issues.append(
            _issue(
                "probe_splits_missing_validation",
                "probe_splits.csv has no validation split category",
                split_path,
                severity="warning",
                details={"splits": split_counts},
            )
        )
    return {"path": str(split_path), "rows": int(len(table)), "counts": split_counts}


def _check_artifact_freshness(
    dataset: TraceDataset,
    *,
    require_artifacts: bool,
    issues: list[GuardrailIssue],
) -> dict[str, Any]:
    table = dataset.artifact_index
    if table.empty:
        if require_artifacts:
            issues.append(
                _issue(
                    "missing_artifacts",
                    "Dataset trust gate requires at least one saved VLA Lens artifact",
                    dataset.root,
                )
            )
        return {"count": 0}
    trace_ids = set(_string_column(dataset.episode_index, "trace_id"))
    artifact_types = _counts(table["artifact_type"]) if "artifact_type" in table else {}
    checked = 0
    for row in table.to_dict("records"):
        checked += 1
        artifact_id = str(row.get("artifact_id") or "")
        artifact_path = _artifact_json_path(dataset, row)
        if artifact_path is None or not artifact_path.exists():
            issues.append(
                _issue(
                    "artifact_record_missing_json",
                    "Artifact index row points at missing artifact JSON",
                    dataset.root,
                    details={
                        "artifact_id": artifact_id,
                        "path": None if artifact_path is None else str(artifact_path),
                    },
                )
            )
            continue
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact = LensArtifact.from_dict(payload)
        except Exception as exc:
            issues.append(
                _issue(
                    "artifact_record_unreadable_json",
                    "Could not load artifact JSON",
                    artifact_path,
                    details={"artifact_id": artifact_id, "error": str(exc)},
                )
            )
            continue
        if not _parse_datetime(artifact.created_utc):
            issues.append(
                _issue(
                    "artifact_missing_created_utc",
                    "Artifact must record a parseable created_utc timestamp",
                    artifact_path,
                    details={"artifact_id": artifact.artifact_id},
                )
            )
        source_ids = set(str(value) for value in artifact.source_trace_ids)
        if not source_ids:
            issues.append(
                _issue(
                    "artifact_missing_source_traces",
                    "Artifact must record source_trace_ids for freshness checks",
                    artifact_path,
                    details={"artifact_id": artifact.artifact_id},
                )
            )
        elif not source_ids <= trace_ids:
            issues.append(
                _issue(
                    "artifact_unknown_source_traces",
                    "Artifact source_trace_ids are not all present in the opened dataset",
                    artifact_path,
                    details={
                        "artifact_id": artifact.artifact_id,
                        "unknown_trace_ids": sorted(source_ids - trace_ids),
                    },
                )
            )
        for array_name, relative_path in artifact.arrays.items():
            array_path = artifact_path.parent.parent.parent / str(relative_path)
            if not array_path.exists():
                issues.append(
                    _issue(
                        "artifact_missing_array",
                        "Artifact array path is missing",
                        artifact_path,
                        details={
                            "artifact_id": artifact.artifact_id,
                            "array": array_name,
                            "relative_path": relative_path,
                        },
                    )
                )
    return {"count": int(len(table)), "checked": checked, "types": artifact_types}


def _artifact_json_path(dataset: TraceDataset, row: Mapping[str, Any]) -> Path | None:
    relative = str(row.get("path") or "")
    if not relative:
        return None
    scope = str(row.get("artifact_scope") or row.get("scope") or "")
    if scope == "dataset":
        dataset_path = row.get("dataset_path")
        base = Path(str(dataset_path)) if dataset_path else dataset._dataset_artifact_root()
        return base / relative
    bundle_path = row.get("bundle_path")
    if bundle_path:
        return Path(str(bundle_path)) / relative
    return dataset.root / relative


def _lerobot_roots_for_dataset(root: Path, dataset: TraceDataset) -> tuple[Path, ...]:
    roots: set[Path] = set()
    if (root / LEROBOT_INFO_PATH).exists():
        roots.add(root)
    for bundle in dataset.bundles:
        bundle_root = getattr(bundle, "root", None)
        if bundle_root is not None and (Path(bundle_root) / LEROBOT_INFO_PATH).exists():
            roots.add(Path(bundle_root))
    return tuple(sorted(roots))


def _declared_claim_level(payload: Mapping[str, Any]) -> str | None:
    for path in (
        "method.claim_level",
        "method.claim_evidence.claim_level",
        "metrics.claim_level",
        "display.claim_level",
    ):
        value = _path_get(payload, path)
        if value:
            return str(value)
    return None


def _missing_claim_paths(payload: Mapping[str, Any], level: str) -> list[str]:
    required: list[str] = []
    for claim_level in CLAIM_LEVELS:
        required.extend(_claim_required_paths(claim_level))
        if claim_level == level:
            break
    return [path for path in required if _is_missing_path(payload, path)]


def _claim_required_paths(level: str) -> tuple[str, ...]:
    requirements = {
        "integration_smoke": (
            "artifact_id",
            "artifact_type",
            "source_trace_ids",
            "method.source.source_traces",
            "method.input.selector",
            "method.target.name",
            "method.split.kind",
            "method.evaluation.primary_metric",
            "metrics.sample_count",
            "display.data_quality",
        ),
        "decodable": (
            "method.split.group_key",
            "method.evaluation.eval_splits",
            "method.outputs.predictions",
            "method.outputs.per_split_metrics",
            "method.outputs.null_metrics",
            "method.metadata_baseline_columns",
            "metrics.best_delta",
            "metrics.best_baseline",
            "display.target_distribution",
        ),
        "candidate_mechanism": (
            "method.claim_evidence.mechanism_hypothesis",
            "method.claim_evidence.localization",
            "method.claim_evidence.intervention_plan",
            "method.claim_evidence.negative_controls",
        ),
        "causal_intervention": (
            "method.claim_evidence.intervention_artifacts",
            "method.claim_evidence.replay_reproduction",
            "method.claim_evidence.control_results",
            "method.claim_evidence.behavior_effect",
            "method.claim_evidence.rerun_verification",
        ),
    }
    return requirements[level]


def _estimate_config_episodes(payload: Mapping[str, Any]) -> int | None:
    try:
        seeds_per_task = int(payload["seeds_per_task"])
        benchmarks = _as_list(payload["benchmarks"])
        task_groups = payload["task_ids"]
        if not isinstance(task_groups, Mapping):
            return None
        task_count = sum(len(_as_list(values)) for values in task_groups.values())
        return len(benchmarks) * task_count * seeds_per_task
    except Exception:
        return None


def _is_unsafe_python_executable(value: str) -> bool:
    normalized = value.replace("\\", "/")
    return (
        "uv run" in normalized
        or normalized.endswith("/.venv/bin/python")
        or "/.venv/bin/python" in normalized
    )


def _looks_broad_1000(value: str) -> bool:
    text = _normalize_name(value)
    return "broad_1000" in text or "broad1000" in text


def _looks_broad_capture(value: str) -> bool:
    text = _normalize_name(value)
    return any(token in text for token in ("broad", "diverse_100", "diverse_500", "1000", "500"))


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _split_category(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("train"):
        return "train"
    if text.startswith(("val", "valid")):
        return "validation"
    if text.startswith("test"):
        return "test"
    return text or "missing"


def _string_column(frame: pd.DataFrame, column: str) -> list[str]:
    if frame.empty or column not in frame:
        return []
    return [str(value) for value in frame[column].dropna().tolist()]


def _counts(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError(f"Expected list-like value, got {value!r}")
    return list(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        return int(text) if text else None
    except (TypeError, ValueError):
        return None


def _path_get(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _is_missing_path(payload: Mapping[str, Any], path: str) -> bool:
    value = _path_get(payload, path)
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == "" or value.strip().startswith("<")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value) == 0
    if isinstance(value, Mapping):
        return len(value) == 0
    return False


def _parse_datetime(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _issue(
    code: str,
    message: str,
    path: str | Path | None = None,
    *,
    severity: str = "error",
    details: Mapping[str, Any] | None = None,
) -> GuardrailIssue:
    return GuardrailIssue(
        code=code,
        message=message,
        severity=severity,
        path=None if path is None else str(path),
        details=dict(details or {}),
    )


__all__ = [
    "AUDIT_CAPTURE_PROFILES",
    "CLAIM_LEVELS",
    "GuardrailIssue",
    "GuardrailReport",
    "ProbeClaimGateResult",
    "check_dataset_trust",
    "lint_episode_plan",
    "lint_research_configs",
    "validate_audit_capture_contract",
    "validate_audit_capture_contract_file",
    "validate_probe_claim_artifact",
]
