"""Typed analysis packages and deterministic decision-gate evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from vla_lens.research_child import child_plan_fingerprint
from vla_lens.research_plan import research_plan_fingerprint

ANALYSIS_SCHEMA_VERSION = 1
ANALYSIS_KIND = "vla_lens.research_analysis"
TOP_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "analysis_id",
        "program_id",
        "program_fingerprint",
        "study_id",
        "child_plan_id",
        "child_plan_fingerprint",
        "child_lock_fingerprint",
        "trial_manifest_sha256",
        "authorization_receipt_sha256",
        "attempt_ledger_sha256",
        "budget_record_sha256",
        "trial_accounting",
        "metric_results",
        "decision_values",
        "artifact_refs",
    }
)
METRIC_FIELDS = frozenset(
    {
        "metric_id",
        "estimate",
        "unit",
        "interval",
        "planned_independent_units",
        "independent_units",
        "source_artifact_id",
    }
)
INTERVAL_FIELDS = frozenset(
    {"low", "high", "method", "level", "grouping_unit", "replicates", "seed"}
)
UNIT_FIELDS = frozenset({"task_families", "scene_clusters", "noise_repeats", "rollouts"})
VALUE_FIELDS = frozenset({"id", "value", "unit", "evidence_artifact_id"})
ACCOUNTING_FIELDS = frozenset({"expected", "completed", "technical_failed", "excluded", "attempts"})
ARTIFACT_FIELDS = frozenset({"id", "type", "uri", "sha256"})


class ResearchAnalysisError(ValueError):
    """Raised when analysis bytes cannot drive a locked decision."""


@dataclass(frozen=True, slots=True)
class EvaluatedGate:
    """One gate evaluated from one typed analysis value."""

    id: str
    role: str
    value_id: str
    observed: float
    operator: str
    threshold: float
    unit: str
    passed: bool
    evidence_artifact_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "value_id": self.value_id,
            "observed": self.observed,
            "operator": self.operator,
            "threshold": self.threshold,
            "unit": self.unit,
            "passed": self.passed,
            "evidence_artifact_id": self.evidence_artifact_id,
        }


def validate_research_analysis(
    analysis: Mapping[str, Any],
    *,
    program: Mapping[str, Any],
    child_plan: Mapping[str, Any],
    child_lock_fingerprint: str,
) -> tuple[tuple[EvaluatedGate, ...], str]:
    """Validate exact analysis structure, evaluate gates, and derive outcome."""

    _exact(analysis, TOP_FIELDS, "analysis")
    expected = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "kind": ANALYSIS_KIND,
        "program_id": program.get("program_id"),
        "program_fingerprint": research_plan_fingerprint(program),
        "study_id": _get(child_plan, "study.id"),
        "child_plan_id": child_plan.get("child_plan_id"),
        "child_plan_fingerprint": child_plan_fingerprint(child_plan),
        "child_lock_fingerprint": child_lock_fingerprint,
        "trial_manifest_sha256": _get(child_plan, "trials.manifest.sha256"),
    }
    for field, value in expected.items():
        if analysis.get(field) != value:
            raise ResearchAnalysisError(f"Analysis does not match locked field {field!r}")
    for field in (
        "authorization_receipt_sha256",
        "attempt_ledger_sha256",
        "budget_record_sha256",
    ):
        if not _sha256(analysis.get(field)):
            raise ResearchAnalysisError(f"Analysis {field} must be a full sha256")
    _validate_accounting(analysis["trial_accounting"], child_plan)
    artifact_ids = _validate_artifacts(analysis["artifact_refs"])
    _validate_metrics(analysis["metric_results"], child_plan, artifact_ids)
    _validate_accounting_decision_values(analysis["decision_values"], analysis["trial_accounting"])
    gates = _evaluate_gates(
        analysis["decision_values"],
        _get(child_plan, "decision.gate_components"),
        artifact_ids,
    )
    return gates, derive_gate_outcome(gates)


def derive_gate_outcome(gates: Sequence[EvaluatedGate]) -> str:
    """Apply the only supported decision order to evaluated gate components."""

    by_role = {
        role: [gate.passed for gate in gates if gate.role == role]
        for role in ("integrity", "applicability", "positive", "negative")
    }
    if not by_role["integrity"] or not by_role["positive"] or not by_role["negative"]:
        raise ResearchAnalysisError(
            "Analysis lacks required integrity, positive, or negative gates"
        )
    if not all(by_role["integrity"]):
        return "invalid"
    if by_role["applicability"] and not all(by_role["applicability"]):
        return "not_applicable"
    positive = all(by_role["positive"])
    negative = all(by_role["negative"])
    if positive and negative:
        raise ResearchAnalysisError("Positive and negative gates cannot both pass")
    if positive:
        return "positive"
    if negative:
        return "negative"
    return "inconclusive"


def metric_by_id(analysis: Mapping[str, Any], metric_id: str) -> Mapping[str, Any]:
    """Return one unique typed metric result."""

    matches = [
        item
        for item in _records(analysis.get("metric_results"))
        if item.get("metric_id") == metric_id
    ]
    if len(matches) != 1:
        raise ResearchAnalysisError(f"Analysis needs exactly one metric {metric_id!r}")
    return matches[0]


def decision_value_by_id(analysis: Mapping[str, Any], value_id: str) -> Mapping[str, Any]:
    """Return one unique typed decision value."""

    matches = [
        item for item in _records(analysis.get("decision_values")) if item.get("id") == value_id
    ]
    if len(matches) != 1:
        raise ResearchAnalysisError(f"Analysis needs exactly one decision value {value_id!r}")
    return matches[0]


def _validate_accounting(value: Any, child: Mapping[str, Any]) -> None:
    _exact_mapping(value, ACCOUNTING_FIELDS, "trial_accounting")
    counts = {name: _nonnegative_int(value[name], name) for name in ACCOUNTING_FIELDS}
    if counts["expected"] != _get(child, "trials.expected_count"):
        raise ResearchAnalysisError("Analysis expected-trial count differs from the child")
    if counts["completed"] + counts["technical_failed"] + counts["excluded"] != counts["expected"]:
        raise ResearchAnalysisError("Analysis terminal trial counts must equal expected")
    if counts["attempts"] < counts["completed"] + counts["technical_failed"]:
        raise ResearchAnalysisError("Analysis attempt count hides completed or failed attempts")


def _validate_artifacts(value: Any) -> set[str]:
    if not _is_sequence(value) or not value:
        raise ResearchAnalysisError("Analysis needs typed evidence artifacts")
    ids: set[str] = set()
    for item in value:
        _exact_mapping(item, ARTIFACT_FIELDS, "artifact_refs[]")
        artifact_id = str(item["id"])
        if not artifact_id or artifact_id in ids:
            raise ResearchAnalysisError("Analysis artifact IDs must be unique and nonempty")
        if not _sha256(item["sha256"]):
            raise ResearchAnalysisError("Analysis artifact reference has an invalid sha256")
        ids.add(artifact_id)
    return ids


def _validate_metrics(value: Any, child: Mapping[str, Any], artifact_ids: set[str]) -> None:
    if not _is_sequence(value) or not value:
        raise ResearchAnalysisError("Analysis needs metric results")
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in value:
        _exact_mapping(item, METRIC_FIELDS, "metric_results[]")
        metric_id = str(item["metric_id"])
        if metric_id in by_id:
            raise ResearchAnalysisError("Analysis metric IDs must be unique")
        by_id[metric_id] = item
        estimate = _finite(item["estimate"], f"{metric_id} estimate")
        interval = item["interval"]
        _exact_mapping(interval, INTERVAL_FIELDS, f"{metric_id} interval")
        low = _finite(interval["low"], f"{metric_id} interval low")
        high = _finite(interval["high"], f"{metric_id} interval high")
        if not low <= estimate <= high:
            raise ResearchAnalysisError(f"Metric {metric_id!r} estimate lies outside its interval")
        inference = _get(child, "measurement.inference")
        for field in ("method", "level", "grouping_unit", "replicates", "seed"):
            if interval[field] != _get(inference, field):
                raise ResearchAnalysisError(
                    f"Metric {metric_id!r} interval changed locked inference field {field!r}"
                )
        planned_units = item["planned_independent_units"]
        units = item["independent_units"]
        _exact_mapping(planned_units, UNIT_FIELDS, f"{metric_id} planned_independent_units")
        _exact_mapping(units, UNIT_FIELDS, f"{metric_id} independent_units")
        expected_units = _get(child, "trials.expected_independent_units")
        for field in UNIT_FIELDS:
            planned = _nonnegative_int(planned_units[field], field)
            observed = _nonnegative_int(units[field], field)
            if planned != _get(expected_units, field):
                raise ResearchAnalysisError(
                    f"Metric {metric_id!r} changed planned evidence-unit count {field!r}"
                )
            if observed > planned:
                raise ResearchAnalysisError(
                    f"Metric {metric_id!r} has more contributing than planned units"
                )
        if item["source_artifact_id"] not in artifact_ids:
            raise ResearchAnalysisError(f"Metric {metric_id!r} cites unknown evidence")
    required = {
        str(_get(child, "measurement.primary.metric_id")),
        str(_get(child, "measurement.strongest_control_metric_id")),
    }
    if not required <= set(by_id):
        raise ResearchAnalysisError("Analysis lacks the locked primary or control metric")
    primary = by_id[str(_get(child, "measurement.primary.metric_id"))]
    control = by_id[str(_get(child, "measurement.strongest_control_metric_id"))]
    if primary["unit"] != _get(child, "measurement.primary.unit"):
        raise ResearchAnalysisError("Primary analysis unit differs from the child")
    if control["unit"] != primary["unit"]:
        raise ResearchAnalysisError("Control analysis must use the primary unit")


def _evaluate_gates(
    values: Any, components: Any, artifact_ids: set[str]
) -> tuple[EvaluatedGate, ...]:
    if not _is_sequence(values) or not _is_sequence(components):
        raise ResearchAnalysisError("Decision values and components must be lists")
    by_id: dict[str, Mapping[str, Any]] = {}
    for value in values:
        _exact_mapping(value, VALUE_FIELDS, "decision_values[]")
        value_id = str(value["id"])
        if value_id in by_id:
            raise ResearchAnalysisError("Decision value IDs must be unique")
        if value["evidence_artifact_id"] not in artifact_ids:
            raise ResearchAnalysisError("Decision value cites unknown evidence")
        _finite(value["value"], f"decision value {value_id}")
        by_id[value_id] = value
    expected_ids = {str(item["value_id"]) for item in _records(components)}
    if set(by_id) != expected_ids:
        raise ResearchAnalysisError("Analysis decision values do not exactly match locked gates")
    gates: list[EvaluatedGate] = []
    for component in _records(components):
        value = by_id[str(component["value_id"])]
        if value["unit"] != component["unit"]:
            raise ResearchAnalysisError("Decision value unit differs from its locked gate")
        observed = _finite(value["value"], "decision value")
        threshold = _finite(component["threshold"], "decision threshold")
        gates.append(
            EvaluatedGate(
                id=str(component["id"]),
                role=str(component["role"]),
                value_id=str(component["value_id"]),
                observed=observed,
                operator=str(component["operator"]),
                threshold=threshold,
                unit=str(component["unit"]),
                passed=_compare(observed, str(component["operator"]), threshold),
                evidence_artifact_id=str(value["evidence_artifact_id"]),
            )
        )
    return tuple(gates)


def _validate_accounting_decision_values(values: Any, accounting: Mapping[str, Any]) -> None:
    by_id = {str(item.get("id")): item for item in _records(values)}
    derived = {
        "completed_trial_count": accounting["completed"],
        "accounted_trial_count": (
            accounting["completed"] + accounting["technical_failed"] + accounting["excluded"]
        ),
        "attempt_count": accounting["attempts"],
        "expected_trial_count": accounting["expected"],
    }
    for value_id, expected in derived.items():
        if value_id in by_id and by_id[value_id].get("value") != expected:
            raise ResearchAnalysisError(
                f"Decision value {value_id!r} differs from typed trial accounting"
            )


def _compare(observed: float, operator: str, threshold: float) -> bool:
    operations = {
        "greater": observed > threshold,
        "greater_than_or_equal": observed >= threshold,
        "less": observed < threshold,
        "less_than_or_equal": observed <= threshold,
        "equal": observed == threshold,
        "not_equal": observed != threshold,
    }
    try:
        return operations[operator]
    except KeyError as exc:
        raise ResearchAnalysisError(f"Unknown decision operator {operator!r}") from exc


def _exact(value: Mapping[str, Any], fields: frozenset[str], label: str) -> None:
    if set(value) != fields:
        raise ResearchAnalysisError(
            f"{label} fields differ: missing={sorted(fields - set(value))}, "
            f"unknown={sorted(set(value) - fields)}"
        )


def _exact_mapping(value: Any, fields: frozenset[str], label: str) -> None:
    if not isinstance(value, Mapping):
        raise ResearchAnalysisError(f"{label} must be a mapping")
    _exact(value, fields, label)


def _get(payload: Any, path: str) -> Any:
    value = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ResearchAnalysisError(f"{label} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ResearchAnalysisError(f"{label} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ResearchAnalysisError(f"{label} must be finite")
    return parsed


def _nonnegative_int(value: Any, label: str) -> int:
    parsed = _finite(value, label)
    if parsed < 0 or not parsed.is_integer():
        raise ResearchAnalysisError(f"{label} must be a nonnegative integer")
    return int(parsed)


def _sha256(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _records(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in value if isinstance(item, Mapping)] if _is_sequence(value) else []


__all__ = [
    "ANALYSIS_KIND",
    "ANALYSIS_SCHEMA_VERSION",
    "EvaluatedGate",
    "ResearchAnalysisError",
    "decision_value_by_id",
    "derive_gate_outcome",
    "metric_by_id",
    "validate_research_analysis",
]
