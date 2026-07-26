from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from tests._support.research_summary_child import (
    child as _child,
)
from tests._support.research_summary_child import (
    locked_ref as _locked_ref,
)
from vla_lens.research_analysis import validate_research_analysis
from vla_lens.research_child import (
    check_research_child,
    child_plan_fingerprint,
    study_fingerprint,
)
from vla_lens.research_io import canonical_research_fingerprint
from vla_lens.research_plan import load_research_plan, research_plan_fingerprint
from vla_lens.research_state import CampaignState
from vla_lens.research_summary import (
    ResearchResultCardError,
    format_research_result_markdown,
    validate_research_result_card,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROGRAM = load_research_plan(
    REPO_ROOT / "configs/campaigns/rq024_controlled_scene_to_behavior.yaml"
)
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
SHA_D = "sha256:" + "d" * 64
SHA_E = "sha256:" + "e" * 64
SHA_F = "sha256:" + "f" * 64


def test_effect_result_renders_fixed_audited_human_summary():
    child = _child("GEOMETRY-DISCOVERY")
    lock = _lock(child)
    analysis = _effect_analysis(child, lock)
    card = _effect_card(child, lock, analysis)

    rendered = _format(card, child, lock, analysis)

    assert rendered.startswith("# Result: GEOMETRY-DISCOVERY")
    assert "Question: Does moving the target" in rendered
    assert "`geometry_response_gain` = 0.62 dimensionless_gain" in rendered
    assert "95.0% hierarchical_bootstrap interval" in rendered
    assert "6 task-object families and 24 scene clusters" in rendered
    assert "Next action: `reevaluate_program`" in rendered


def test_result_rejects_trial_drift_nonfinite_and_zero_independent_counts():
    child = _child("GEOMETRY-DISCOVERY")
    lock = _lock(child)
    analysis = _effect_analysis(child, lock)
    drift = _effect_card(child, lock, analysis)
    drift["trial_accounting"]["technical_failed"] = 2
    nonfinite = _effect_card(child, lock, analysis)
    nonfinite["primary_result"]["estimate"] = float("nan")
    zero = _effect_card(child, lock, analysis)
    zero["primary_result"]["independent_units"]["task_families"] = 0

    with pytest.raises(ResearchResultCardError, match="must equal expected"):
        _validate(drift, child, lock, analysis)
    with pytest.raises(ResearchResultCardError, match="finite"):
        _validate(nonfinite, child, lock, analysis)
    with pytest.raises(ResearchResultCardError, match="evidence-unit"):
        _validate(zero, child, lock, analysis)


def test_result_derives_verdict_and_next_action_instead_of_trusting_text():
    child = _child("GEOMETRY-DISCOVERY")
    lock = _lock(child)
    analysis = _effect_analysis(child, lock)
    wrong_verdict = _effect_card(child, lock, analysis)
    wrong_verdict["verdict"] = "confirmed_positive"
    wrong_action = _effect_card(child, lock, analysis)
    wrong_action["decision"]["next_action"] = "revise_child"

    with pytest.raises(ResearchResultCardError, match="Verdict must be derived"):
        _validate(wrong_verdict, child, lock, analysis)
    with pytest.raises(ResearchResultCardError, match="Next action"):
        _validate(wrong_action, child, lock, analysis)


def test_result_requires_clean_independent_audit_bound_to_actual_bytes():
    child = _child("GEOMETRY-DISCOVERY")
    lock = _lock(child)
    analysis = _effect_analysis(child, lock)
    failed = _effect_card(child, lock, analysis)
    failed["audit"]["status"] = "fail"
    same_agent = _effect_card(child, lock, analysis)
    same_agent["audit"]["auditor_id"] = child["prepared_by"]
    mismatched_bytes = _effect_card(child, lock, analysis)

    with pytest.raises(ResearchResultCardError, match="clean passing audit"):
        _validate(failed, child, lock, analysis)
    with pytest.raises(ResearchResultCardError, match="differ from"):
        _validate(same_agent, child, lock, analysis)
    with pytest.raises(ResearchResultCardError, match="audit-report bytes"):
        _validate(mismatched_bytes, child, lock, analysis, audit_report_sha256=SHA_D)


def test_result_rejects_program_child_metric_and_trial_manifest_drift():
    child = _child("GEOMETRY-DISCOVERY")
    lock = _lock(child)
    analysis = _effect_analysis(child, lock)
    program_drift = _effect_card(child, lock, analysis)
    program_drift["program_fingerprint"] = SHA_D
    metric_drift = _effect_card(child, lock, analysis)
    metric_drift["primary_result"]["metric_id"] = "after_the_fact_metric"
    trial_drift = _effect_card(child, lock, analysis)
    trial_drift["trial_manifest"]["sha256"] = SHA_D
    trial_drift["audit"]["subject_trial_manifest_sha256"] = SHA_D

    with pytest.raises(ResearchResultCardError, match="program_fingerprint"):
        _validate(program_drift, child, lock, analysis)
    with pytest.raises(ResearchResultCardError, match="Primary metric"):
        _validate(metric_drift, child, lock, analysis)
    with pytest.raises(ResearchResultCardError, match="trial manifest"):
        _validate(trial_drift, child, lock, analysis)


def test_preparation_gate_has_counts_not_a_fake_effect_interval():
    child = _child("FOUNDATION", result_kind="preparation_gate")
    lock = _lock(child)
    analysis = _preparation_analysis(child, lock)
    card = _preparation_card(child, lock, analysis)

    rendered = _format(card, child, lock, analysis)

    assert card["verdict"] == "gate_passed"
    assert "Discovery pool: 7 of 12 eligible" in rendered
    assert "Confirmation pool: 6 of 12 eligible" in rendered
    assert "Primary result" not in rendered
    assert "interval" not in rendered


def test_preparation_gate_rejects_flag_that_disagrees_with_counts():
    child = _child("FOUNDATION", result_kind="preparation_gate")
    lock = _lock(child)
    analysis = _preparation_analysis(child, lock)
    card = _preparation_card(child, lock, analysis)
    card["gate_result"]["gate_passed"] = False

    with pytest.raises(ResearchResultCardError, match="six-and-six"):
        _validate(card, child, lock, analysis)


def test_result_cannot_claim_positive_when_locked_numeric_gates_are_negative():
    child = _child("GEOMETRY-DISCOVERY")
    lock = _lock(child)
    analysis = deepcopy(_effect_analysis(child, lock))
    analysis["metric_results"][0]["estimate"] = 0.0
    analysis["metric_results"][0]["interval"].update({"low": -0.1, "high": 0.1})
    by_id = {item["id"]: item for item in analysis["decision_values"]}
    by_id["primary_interval_low"]["value"] = -0.1
    by_id["primary_interval_high"]["value"] = 0.1
    card = _effect_card(child, lock, analysis)
    card["primary_result"].update({"estimate": 0.0})
    card["primary_result"]["interval"].update({"low": -0.1, "high": 0.1})

    with pytest.raises(ResearchResultCardError, match="Declared outcome"):
        _validate(card, child, lock, analysis)


def test_result_rejects_nested_fields_that_would_otherwise_be_ignored():
    child = _child("GEOMETRY-DISCOVERY")
    lock = _lock(child)
    analysis = _effect_analysis(child, lock)
    card = _effect_card(child, lock, analysis)
    card["decision"]["posthoc_override"] = True

    with pytest.raises(ResearchResultCardError, match="unknown"):
        _validate(card, child, lock, analysis)


def test_confirmation_cannot_change_the_actual_discovery_protocol(tmp_path):
    source = _child("GEOMETRY-DISCOVERY")
    confirmation = _child("GEOMETRY-CONFIRMATION")
    confirmation["measurement"] = deepcopy(source["measurement"])
    confirmation["decision"] = deepcopy(source["decision"])
    confirmation["runtime"]["model"] = deepcopy(source["runtime"]["model"])
    confirmation["runtime"]["code"] = deepcopy(source["runtime"]["code"])
    confirmation["runtime"]["runner"]["entrypoint"] = source["runtime"]["runner"]["entrypoint"]
    confirmation["runtime"]["runner"]["config"] = deepcopy(source["runtime"]["runner"]["config"])
    confirmation["measurement"]["primary"]["minimum_useful_effect"] = 999
    source_path = tmp_path / "source-child.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    state = CampaignState(program_locked=True)
    state.terminal_results["GEOMETRY-DISCOVERY"] = {
        "result_event_id": "source-result",
        "result_event_sha256": SHA_D,
        "outcome": "positive",
    }
    state.events_by_id["source-result"] = {
        "payload": {"child_lock_event": {"event_id": "source-lock"}}
    }
    state.events_by_id["source-lock"] = {"payload": {"child_ref": {"path": source_path.name}}}

    check = check_research_child(
        confirmation,
        PROGRAM,
        repo_root=tmp_path,
        verify_files=True,
        campaign_state=state,
    )

    assert "confirmation_protocol_drift" in {issue.code for issue in check.issues}


def _effect_card(child: dict, lock: dict, analysis: dict) -> dict:
    study = next(item for item in PROGRAM["studies"] if item["id"] == child["study"]["id"])
    gates, _ = validate_research_analysis(
        analysis,
        program=PROGRAM,
        child_plan=child,
        child_lock_fingerprint=canonical_research_fingerprint(lock),
    )
    return {
        "schema_version": 2,
        "result_card_id": "rq024-geometry-discovery-result-r1",
        "result_kind": "effect_estimate",
        "program_id": PROGRAM["program_id"],
        "program_fingerprint": research_plan_fingerprint(PROGRAM),
        "study_id": study["id"],
        "study_fingerprint": study_fingerprint(study),
        "child_plan_id": child["child_plan_id"],
        "child_plan_fingerprint": child_plan_fingerprint(child),
        "child_lock_id": lock["lock_id"],
        "child_lock_fingerprint": canonical_research_fingerprint(lock),
        "reservation_id": lock["reservation_id"],
        "ledger_tip_before_result": SHA_F,
        "attempt_event_range": {"first_sequence": 1, "last_sequence": 10},
        "authorization_receipt": {"id": "authorization-r1", "sha256": SHA_C},
        "attempt_ledger": {"id": "attempts-r1", "sha256": SHA_D},
        "budget_record": {"id": "budget-r1", "sha256": SHA_E},
        "question": study["question"],
        "one_sentence_answer": "PI0.5 followed target displacement on discovery families.",
        "verdict": "exploratory_positive",
        "claim_type": "behavioral_response",
        "behavior_level": "closed_loop_rollout",
        "confirmation_status": "discovery",
        "predecessor_result_fingerprints": [SHA_D],
        "supersedes_result_fingerprints": [],
        "what_changed": ["target x/y position"],
        "held_fixed": ["instruction", "robot start", "camera", "matched noise"],
        "trial_manifest": {"id": "trials", "sha256": SHA_C},
        "analysis_package": {"id": "analysis", "sha256": SHA_B},
        "primary_result": {
            "metric_id": "geometry_response_gain",
            "estimate": 0.62,
            "unit": "dimensionless_gain",
            "null_value": 0.0,
            "minimum_useful_effect": 0.2,
            "interval": {
                "low": 0.31,
                "high": 0.81,
                "method": "hierarchical_bootstrap",
                "level": 0.95,
                "grouping_unit": "task_object_family",
                "replicates": 10000,
                "seed": 24001,
                "source_artifact_id": "per-family",
            },
            "independent_units": {
                "task_families": 6,
                "scene_clusters": 24,
                "noise_repeats": 4,
                "rollouts": 36,
            },
        },
        "strongest_control": {
            "metric_id": "irrelevant_object_gain",
            "name": "irrelevant-object displacement",
            "estimate": 0.08,
            "unit": "dimensionless_gain",
            "interval": {
                "low": -0.02,
                "high": 0.16,
                "method": "hierarchical_bootstrap",
                "level": 0.95,
                "grouping_unit": "task_object_family",
                "replicates": 10000,
                "seed": 24001,
            },
            "source_artifact_id": "control",
        },
        "trial_accounting": {
            "expected": 10,
            "completed": 9,
            "technical_failed": 1,
            "excluded": 0,
            "attempts": 10,
        },
        "supported_conclusion": study["allowed_conclusions"][0],
        "forbidden_conclusions": list(study["forbidden_conclusions"]),
        "strongest_surviving_alternative": "familiar layout response",
        "artifact_refs": [
            {
                "id": "per-family",
                "type": "per_family_metrics",
                "uri": "artifact://per-family",
                "sha256": SHA_A,
            },
            {
                "id": "control",
                "type": "control_metrics",
                "uri": "artifact://control",
                "sha256": SHA_B,
            },
            {
                "id": "audit-evidence",
                "type": "audit_evidence",
                "uri": "artifact://audit",
                "sha256": SHA_C,
            },
            {
                "id": "analysis",
                "type": "analysis_package",
                "uri": "artifact://analysis",
                "sha256": SHA_B,
            },
            {
                "id": "audit-r1",
                "type": "audit_report",
                "uri": "artifact://audit-r1",
                "sha256": SHA_A,
            },
            _result_artifact(lock["lock_id"], "child_lock_receipt", SHA_F),
            _result_artifact("authorization-r1", "child_authorization_receipt", SHA_C),
            _result_artifact("attempts-r1", "attempt_ledger", SHA_D),
            _result_artifact("budget-r1", "budget_record", SHA_E),
        ],
        "metric_ids": ["geometry_response_gain", "irrelevant_object_gain"],
        "decision": {
            "evaluated_gates": [gate.to_dict() for gate in gates],
            "derived_outcome": "positive",
            "next_action": "reevaluate_program",
        },
        "audit": {
            "status": "pass",
            "report_id": "audit-r1",
            "report_sha256": SHA_A,
            "auditor_id": "audit-agent",
            "subject_child_fingerprint": child_plan_fingerprint(child),
            "subject_child_lock_fingerprint": canonical_research_fingerprint(lock),
            "subject_trial_manifest_sha256": SHA_C,
            "subject_analysis_package_sha256": SHA_B,
            "checks": {"execution": "pass", "calculation": "pass", "claim": "pass"},
            "unresolved_errors": [],
        },
    }


def _preparation_card(child: dict, lock: dict, analysis: dict) -> dict:
    study = next(item for item in PROGRAM["studies"] if item["id"] == "FOUNDATION")
    gates, _ = validate_research_analysis(
        analysis,
        program=PROGRAM,
        child_plan=child,
        child_lock_fingerprint=canonical_research_fingerprint(lock),
    )
    return {
        "schema_version": 2,
        "result_card_id": "rq024-foundation-result-r1",
        "result_kind": "preparation_gate",
        "program_id": PROGRAM["program_id"],
        "program_fingerprint": research_plan_fingerprint(PROGRAM),
        "study_id": "FOUNDATION",
        "study_fingerprint": study_fingerprint(study),
        "child_plan_id": child["child_plan_id"],
        "child_plan_fingerprint": child_plan_fingerprint(child),
        "child_lock_id": lock["lock_id"],
        "child_lock_fingerprint": canonical_research_fingerprint(lock),
        "reservation_id": lock["reservation_id"],
        "ledger_tip_before_result": SHA_F,
        "attempt_event_range": {"first_sequence": 1, "last_sequence": 73},
        "authorization_receipt": {"id": "authorization-r1", "sha256": SHA_C},
        "attempt_ledger": {"id": "attempts-r1", "sha256": SHA_D},
        "budget_record": {"id": "budget-r1", "sha256": SHA_E},
        "question": study["question"],
        "one_sentence_answer": "Both fixed pools supplied six eligible families.",
        "verdict": "gate_passed",
        "claim_type": "preparation",
        "behavior_level": "closed_loop_rollout",
        "confirmation_status": "preparation",
        "predecessor_result_fingerprints": [],
        "supersedes_result_fingerprints": [],
        "what_changed": ["task family and reset seed"],
        "held_fixed": ["checkpoint and eligibility rule"],
        "trial_manifest": {"id": "trials", "sha256": SHA_C},
        "analysis_package": {"id": "analysis", "sha256": SHA_B},
        "gate_result": {
            "discovery_eligible": 7,
            "discovery_total": 12,
            "confirmation_eligible": 6,
            "confirmation_total": 12,
            "gate_passed": True,
            "selection_rule": "first six eligible in each pre-outcome pool rank",
            "diagnostics": ["contact telemetry is not yet available"],
        },
        "trial_accounting": {
            "expected": 72,
            "completed": 72,
            "technical_failed": 0,
            "excluded": 0,
            "attempts": 73,
        },
        "supported_conclusion": study["allowed_conclusions"][0],
        "forbidden_conclusions": list(study["forbidden_conclusions"]),
        "strongest_surviving_alternative": "competence may remain narrow to these families",
        "artifact_refs": [
            {
                "id": "foundation",
                "type": "gate_analysis",
                "uri": "artifact://foundation",
                "sha256": SHA_B,
            },
            {
                "id": "analysis",
                "type": "analysis_package",
                "uri": "artifact://analysis",
                "sha256": SHA_B,
            },
            {
                "id": "audit-foundation",
                "type": "audit_report",
                "uri": "artifact://audit",
                "sha256": SHA_A,
            },
            _result_artifact(lock["lock_id"], "child_lock_receipt", SHA_F),
            _result_artifact("authorization-r1", "child_authorization_receipt", SHA_C),
            _result_artifact("attempts-r1", "attempt_ledger", SHA_D),
            _result_artifact("budget-r1", "budget_record", SHA_E),
        ],
        "metric_ids": ["family_baseline_eligibility"],
        "decision": {
            "evaluated_gates": [gate.to_dict() for gate in gates],
            "derived_outcome": "positive",
            "next_action": "reevaluate_program",
        },
        "audit": {
            "status": "pass",
            "report_id": "audit-foundation",
            "report_sha256": SHA_A,
            "auditor_id": "audit-agent",
            "subject_child_fingerprint": child_plan_fingerprint(child),
            "subject_child_lock_fingerprint": canonical_research_fingerprint(lock),
            "subject_trial_manifest_sha256": SHA_C,
            "subject_analysis_package_sha256": SHA_B,
            "checks": {"execution": "pass", "calculation": "pass", "claim": "pass"},
            "unresolved_errors": [],
        },
    }


def _lock(child: dict) -> dict:
    study = next(item for item in PROGRAM["studies"] if item["id"] == child["study"]["id"])
    audit_types = ["schema", "design", "runner", "budget", *study["required_audits"]]
    return {
        "schema_version": 1,
        "kind": "vla_lens.research_child_lock",
        "lock_id": f"{child['child_plan_id']}-lock",
        "program_id": PROGRAM["program_id"],
        "program_fingerprint": research_plan_fingerprint(PROGRAM),
        "study_id": study["id"],
        "study_fingerprint": study_fingerprint(study),
        "child_plan_id": child["child_plan_id"],
        "child_plan_fingerprint": child_plan_fingerprint(child),
        "manifest_commit": "b" * 40,
        "locked_utc": "2026-07-26T12:00:00+00:00",
        "prepared_by": child["prepared_by"],
        "reservation_id": f"{child['child_plan_id']}-reservation",
        "prior_ledger_tip": SHA_F,
        "audits": [
            {
                "audit_type": audit_type,
                "auditor_id": f"audit-agent-{index}",
                "verdict": "pass",
                "subject_child_fingerprint": child_plan_fingerprint(child),
                "artifact": _locked_ref(f"lock-audit-{index}", SHA_A),
            }
            for index, audit_type in enumerate(audit_types)
        ],
    }


def _effect_analysis(child: dict, lock: dict) -> dict:
    inference = child["measurement"]["inference"]
    units = child["trials"]["expected_independent_units"]
    return _analysis(
        child,
        lock,
        trial_accounting={
            "expected": 10,
            "completed": 9,
            "technical_failed": 1,
            "excluded": 0,
            "attempts": 10,
        },
        metrics=[
            {
                "metric_id": "geometry_response_gain",
                "estimate": 0.62,
                "unit": "dimensionless_gain",
                "interval": {
                    "low": 0.31,
                    "high": 0.81,
                    **inference,
                },
                "planned_independent_units": dict(units),
                "independent_units": dict(units),
                "source_artifact_id": "per-family",
            },
            {
                "metric_id": "irrelevant_object_gain",
                "estimate": 0.08,
                "unit": "dimensionless_gain",
                "interval": {
                    "low": -0.02,
                    "high": 0.16,
                    **inference,
                },
                "planned_independent_units": dict(units),
                "independent_units": dict(units),
                "source_artifact_id": "control",
            },
        ],
        values=[
            _analysis_value("accounted_trial_count", 10, "trials", "per-family"),
            _analysis_value("primary_interval_low", 0.31, "dimensionless_gain", "per-family"),
            _analysis_value("primary_interval_high", 0.81, "dimensionless_gain", "per-family"),
        ],
        artifact_refs=[
            _result_artifact("per-family", "per_family_metrics", SHA_A),
            _result_artifact("control", "control_metrics", SHA_B),
        ],
    )


def _preparation_analysis(child: dict, lock: dict) -> dict:
    inference = child["measurement"]["inference"]
    units = child["trials"]["expected_independent_units"]
    return _analysis(
        child,
        lock,
        trial_accounting={
            "expected": 72,
            "completed": 72,
            "technical_failed": 0,
            "excluded": 0,
            "attempts": 73,
        },
        metrics=[
            {
                "metric_id": "family_baseline_eligibility",
                "estimate": 1.0,
                "unit": "binary_per_task_object_family",
                "interval": {"low": 1.0, "high": 1.0, **inference},
                "planned_independent_units": dict(units),
                "independent_units": dict(units),
                "source_artifact_id": "foundation",
            },
            {
                "metric_id": "irrelevant_object_gain",
                "estimate": 0.0,
                "unit": "binary_per_task_object_family",
                "interval": {"low": 0.0, "high": 0.0, **inference},
                "planned_independent_units": dict(units),
                "independent_units": dict(units),
                "source_artifact_id": "control",
            },
        ],
        values=[
            _analysis_value("discovery_eligible_count", 7, "task_object_families", "foundation"),
            _analysis_value("confirmation_eligible_count", 6, "task_object_families", "foundation"),
            _analysis_value("discovery_total_count", 12, "task_object_families", "foundation"),
            _analysis_value("confirmation_total_count", 12, "task_object_families", "foundation"),
            _analysis_value("completed_trial_count", 72, "rollout_trials", "foundation"),
            _analysis_value("minimum_pool_eligible_count", 6, "task_object_families", "foundation"),
        ],
        artifact_refs=[
            _result_artifact("foundation", "gate_analysis", SHA_B),
            _result_artifact("control", "control_metrics", SHA_A),
        ],
    )


def _analysis(
    child: dict,
    lock: dict,
    *,
    trial_accounting: dict,
    metrics: list[dict],
    values: list[dict],
    artifact_refs: list[dict],
) -> dict:
    return {
        "schema_version": 1,
        "kind": "vla_lens.research_analysis",
        "analysis_id": "analysis",
        "program_id": PROGRAM["program_id"],
        "program_fingerprint": research_plan_fingerprint(PROGRAM),
        "study_id": child["study"]["id"],
        "child_plan_id": child["child_plan_id"],
        "child_plan_fingerprint": child_plan_fingerprint(child),
        "child_lock_fingerprint": canonical_research_fingerprint(lock),
        "trial_manifest_sha256": SHA_C,
        "authorization_receipt_sha256": SHA_C,
        "attempt_ledger_sha256": SHA_D,
        "budget_record_sha256": SHA_E,
        "trial_accounting": trial_accounting,
        "metric_results": metrics,
        "decision_values": values,
        "artifact_refs": artifact_refs,
    }


def _analysis_value(value_id: str, value: float, unit: str, artifact_id: str) -> dict:
    return {
        "id": value_id,
        "value": value,
        "unit": unit,
        "evidence_artifact_id": artifact_id,
    }


def _result_artifact(artifact_id: str, artifact_type: str, sha256: str) -> dict:
    return {
        "id": artifact_id,
        "type": artifact_type,
        "uri": f"artifact://{artifact_id}",
        "sha256": sha256,
    }


def _validate(
    card: dict,
    child: dict,
    lock: dict,
    analysis: dict,
    *,
    audit_report_sha256: str = SHA_A,
) -> None:
    validate_research_result_card(
        card,
        program=PROGRAM,
        child_plan=child,
        child_lock=lock,
        analysis_package=analysis,
        lock_receipt_sha256=SHA_F,
        audit_report_sha256=audit_report_sha256,
        analysis_package_sha256=SHA_B,
        authorization_receipt_sha256=SHA_C,
        attempt_ledger_sha256=SHA_D,
        budget_record_sha256=SHA_E,
    )


def _format(card: dict, child: dict, lock: dict, analysis: dict) -> str:
    return format_research_result_markdown(
        card,
        program=PROGRAM,
        child_plan=child,
        child_lock=lock,
        analysis_package=analysis,
        lock_receipt_sha256=SHA_F,
        audit_report_sha256=SHA_A,
        analysis_package_sha256=SHA_B,
        authorization_receipt_sha256=SHA_C,
        attempt_ledger_sha256=SHA_D,
        budget_record_sha256=SHA_E,
    )
