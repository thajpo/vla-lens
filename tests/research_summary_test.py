from __future__ import annotations

from pathlib import Path

import pytest

from vla_lens.research_child import child_plan_fingerprint, study_fingerprint
from vla_lens.research_plan import load_research_plan, research_plan_fingerprint
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


def test_effect_result_renders_fixed_audited_human_summary():
    child = _child("GEOMETRY-DISCOVERY")
    card = _effect_card(child)

    rendered = format_research_result_markdown(card, program=PROGRAM, child_plan=child)

    assert rendered.startswith("# Result: GEOMETRY-DISCOVERY")
    assert "Question: Does moving the target" in rendered
    assert "`geometry_response_gain` = 0.62 dimensionless_gain" in rendered
    assert "95.0% hierarchical_bootstrap interval" in rendered
    assert "6 task-object families and 24 scene clusters" in rendered
    assert "Next action: `reevaluate_program`" in rendered


def test_result_rejects_trial_drift_nonfinite_and_zero_independent_counts():
    child = _child("GEOMETRY-DISCOVERY")
    drift = _effect_card(child)
    drift["trial_accounting"]["technical_failed"] = 2
    nonfinite = _effect_card(child)
    nonfinite["primary_result"]["estimate"] = float("nan")
    zero = _effect_card(child)
    zero["primary_result"]["independent_units"]["task_families"] = 0

    with pytest.raises(ResearchResultCardError, match="must equal expected"):
        validate_research_result_card(drift, program=PROGRAM, child_plan=child)
    with pytest.raises(ResearchResultCardError, match="finite"):
        validate_research_result_card(nonfinite, program=PROGRAM, child_plan=child)
    with pytest.raises(ResearchResultCardError, match="independent-unit"):
        validate_research_result_card(zero, program=PROGRAM, child_plan=child)


def test_result_derives_verdict_and_next_action_instead_of_trusting_text():
    child = _child("GEOMETRY-DISCOVERY")
    wrong_verdict = _effect_card(child)
    wrong_verdict["verdict"] = "confirmed_positive"
    wrong_action = _effect_card(child)
    wrong_action["decision"]["next_action"] = "revise_child"

    with pytest.raises(ResearchResultCardError, match="Verdict must be derived"):
        validate_research_result_card(wrong_verdict, program=PROGRAM, child_plan=child)
    with pytest.raises(ResearchResultCardError, match="Next action"):
        validate_research_result_card(wrong_action, program=PROGRAM, child_plan=child)


def test_result_requires_clean_independent_audit_bound_to_actual_bytes():
    child = _child("GEOMETRY-DISCOVERY")
    failed = _effect_card(child)
    failed["audit"]["status"] = "fail"
    same_agent = _effect_card(child)
    same_agent["audit"]["auditor_id"] = child["prepared_by"]
    mismatched_bytes = _effect_card(child)

    with pytest.raises(ResearchResultCardError, match="clean passing audit"):
        validate_research_result_card(failed, program=PROGRAM, child_plan=child)
    with pytest.raises(ResearchResultCardError, match="differ from"):
        validate_research_result_card(same_agent, program=PROGRAM, child_plan=child)
    with pytest.raises(ResearchResultCardError, match="audit-report bytes"):
        validate_research_result_card(
            mismatched_bytes,
            program=PROGRAM,
            child_plan=child,
            audit_report_sha256=SHA_D,
        )


def test_result_rejects_program_child_metric_and_trial_manifest_drift():
    child = _child("GEOMETRY-DISCOVERY")
    program_drift = _effect_card(child)
    program_drift["program_fingerprint"] = SHA_D
    metric_drift = _effect_card(child)
    metric_drift["primary_result"]["metric_id"] = "after_the_fact_metric"
    trial_drift = _effect_card(child)
    trial_drift["trial_manifest"]["sha256"] = SHA_D
    trial_drift["audit"]["subject_trial_manifest_sha256"] = SHA_D

    with pytest.raises(ResearchResultCardError, match="program_fingerprint"):
        validate_research_result_card(program_drift, program=PROGRAM, child_plan=child)
    with pytest.raises(ResearchResultCardError, match="Primary metric"):
        validate_research_result_card(metric_drift, program=PROGRAM, child_plan=child)
    with pytest.raises(ResearchResultCardError, match="trial manifest"):
        validate_research_result_card(trial_drift, program=PROGRAM, child_plan=child)


def test_preparation_gate_has_counts_not_a_fake_effect_interval():
    child = _child("FOUNDATION", result_kind="preparation_gate")
    card = _preparation_card(child)

    rendered = format_research_result_markdown(card, program=PROGRAM, child_plan=child)

    assert card["verdict"] == "gate_passed"
    assert "Discovery pool: 7 of 12 eligible" in rendered
    assert "Confirmation pool: 6 of 12 eligible" in rendered
    assert "Primary result" not in rendered
    assert "interval" not in rendered


def test_preparation_gate_rejects_flag_that_disagrees_with_counts():
    child = _child("FOUNDATION", result_kind="preparation_gate")
    card = _preparation_card(child)
    card["gate_result"]["gate_passed"] = False

    with pytest.raises(ResearchResultCardError, match="six-and-six"):
        validate_research_result_card(card, program=PROGRAM, child_plan=child)


def _child(study_id: str, *, result_kind: str = "effect_estimate") -> dict:
    study = next(item for item in PROGRAM["studies"] if item["id"] == study_id)
    predecessors = []
    for predecessor in study["entry_conditions"]["requires_all_completed"]:
        predecessors.append(
            {
                "study_id": predecessor,
                "result_card_id": f"{predecessor.lower()}-result",
                "fingerprint": SHA_D,
                "verdict": "gate_passed" if predecessor == "FOUNDATION" else "exploratory_positive",
            }
        )
    return {
        "schema_version": 1,
        "kind": "vla_lens.research_child",
        "child_plan_id": f"rq024-{study_id.lower()}-r1",
        "revision": 1,
        "prepared_by": "planner-agent",
        "program": {
            "path": "configs/campaigns/rq024_controlled_scene_to_behavior.yaml",
            "program_id": PROGRAM["program_id"],
            "fingerprint": research_plan_fingerprint(PROGRAM),
        },
        "study": {
            "id": study_id,
            "fingerprint": study_fingerprint(study),
            "phase": study["phase"],
        },
        "predecessor_results": predecessors,
        "claim": {
            "result_kind": result_kind,
            "question": study["question"],
            "allowed_conclusions": list(study["allowed_conclusions"]),
            "forbidden_conclusions": list(study["forbidden_conclusions"]),
        },
        "cohort": {
            "family_pool": study["data_scope"]["family_pool"],
            "pool_phase": study["data_scope"]["pool_phase"],
            "requires_gate": study["data_scope"]["requires_gate"],
            "read_namespaces": list(study["data_scope"]["read_namespaces"]),
            "write_namespace": study["data_scope"]["write_namespace"],
            "selection_allowed": study["data_scope"]["selection_allowed"],
            "manifest": _locked_ref("cohort", SHA_A),
            "exposure_log": _locked_ref("exposure", SHA_B),
        },
        "trials": {
            "manifest": _locked_ref("trials", SHA_C),
            "expected_count": 72 if study_id == "FOUNDATION" else 10,
            "stable_id_fields": ["child_plan_id", "trial_id"],
            "seed_domains": ["environment", "policy", "flow_noise"],
        },
        "measurement": {
            "primary": {
                "metric_id": study["primary_claim"]["metric_id"],
                "formula": study["primary_claim"]["definition"],
                "implementation_id": f"{study_id.lower()}-metric-v1",
                "unit": study["primary_claim"]["unit"],
                "direction": study["primary_claim"]["direction"],
                "minimum_useful_effect": (
                    "six eligible families in each fixed pool" if study_id == "FOUNDATION" else 0.2
                ),
            },
            "strongest_control_metric_id": "irrelevant_object_gain",
            "inference": {
                "method": "hierarchical_bootstrap",
                "level": 0.95,
                "grouping_unit": "task_object_family",
                "replicates": 10000,
                "seed": 24001,
            },
        },
        "decision": {
            "gate_components": [{"id": "locked_gate", "operator": "greater", "threshold": 0}],
            "positive_combiner": "all",
            "negative_rule": "locked equivalence or below-useful-effect gate",
            "inconclusive_rule": "neither positive nor negative gate passes",
            "invalid_conditions": ["trial matrix invalid"],
        },
        "runtime": {
            "model": {"repo_id": "pi05", "revision": "commit", "snapshot_manifest_sha256": SHA_A},
            "environment": {
                "backend": "rocm",
                "package_receipt": _locked_ref("environment", SHA_A),
                "camera_config_sha256": SHA_A,
                "controller_config_sha256": SHA_A,
                "preprocessor_config_sha256": SHA_A,
                "postprocessor_config_sha256": SHA_A,
            },
            "code": {"implementation_commit": "a" * 40, "source_tree_sha256": SHA_A},
            "runner": {
                "entrypoint": "scripts/pi05_batch_capture.sh",
                "argv": ["--backend", "rocm"],
                "config": _locked_ref("runner", SHA_A),
            },
        },
        "budget": {
            "max_model_calls": min(10, study["budget"]["max_model_calls"]),
            "max_action_generations": min(10, study["budget"]["max_action_generations"]),
            "max_full_rollouts": (
                72 if study_id == "FOUNDATION" else min(10, study["budget"]["max_full_rollouts"])
            ),
            "max_simulator_steps": min(100, study["budget"]["max_simulator_steps"]),
            "max_persistent_gb": min(1, study["budget"]["max_additional_persistent_gb"]),
            "max_ephemeral_gb": min(1, study["budget"]["max_ephemeral_gb"]),
            "min_free_space_gb": PROGRAM["program_budget"]["min_free_space_gb"],
        },
        "output": {
            "root": "/tmp/rq024-test",
            "namespace": study_id.lower(),
            "attempt_ledger": "events/trials",
            "required_artifact_types": ["analysis"],
        },
        "completion": {
            "valid_trial_statuses": ["completed"],
            "technical_retry_rule": "append only",
            "resume_identity_fields": ["child_fingerprint", "trial_id"],
        },
        "required_audits": ["schema", "design"],
    }


def _effect_card(child: dict) -> dict:
    study = next(item for item in PROGRAM["studies"] if item["id"] == child["study"]["id"])
    return {
        "schema_version": 2,
        "result_card_id": "rq024-geometry-discovery-result-r1",
        "result_kind": "effect_estimate",
        "program_id": PROGRAM["program_id"],
        "program_fingerprint": research_plan_fingerprint(PROGRAM),
        "study_id": study["id"],
        "child_plan_id": child["child_plan_id"],
        "child_plan_fingerprint": child_plan_fingerprint(child),
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
            "interval": {"low": -0.02, "high": 0.16},
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
        ],
        "metric_ids": ["geometry_response_gain", "irrelevant_object_gain"],
        "decision": {
            "integrity_checks": [_check("valid", True, "audit-evidence")],
            "applicability_checks": [],
            "positive_checks": [_check("positive-gate", True, "per-family")],
            "negative_checks": [_check("negative-gate", False, "per-family")],
            "derived_outcome": "positive",
            "next_action": "reevaluate_program",
        },
        "audit": {
            "status": "pass",
            "report_id": "audit-r1",
            "report_sha256": SHA_A,
            "auditor_id": "audit-agent",
            "subject_child_fingerprint": child_plan_fingerprint(child),
            "subject_trial_manifest_sha256": SHA_C,
            "subject_analysis_package_sha256": SHA_B,
            "checks": {"execution": "pass", "calculation": "pass", "claim": "pass"},
            "unresolved_errors": [],
        },
    }


def _preparation_card(child: dict) -> dict:
    study = next(item for item in PROGRAM["studies"] if item["id"] == "FOUNDATION")
    return {
        "schema_version": 2,
        "result_card_id": "rq024-foundation-result-r1",
        "result_kind": "preparation_gate",
        "program_id": PROGRAM["program_id"],
        "program_fingerprint": research_plan_fingerprint(PROGRAM),
        "study_id": "FOUNDATION",
        "child_plan_id": child["child_plan_id"],
        "child_plan_fingerprint": child_plan_fingerprint(child),
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
        ],
        "metric_ids": ["family_baseline_eligibility"],
        "decision": {
            "integrity_checks": [_check("valid", True, "foundation")],
            "applicability_checks": [],
            "positive_checks": [_check("six-and-six", True, "foundation")],
            "negative_checks": [_check("pool-shortfall", False, "foundation")],
            "derived_outcome": "positive",
            "next_action": "reevaluate_program",
        },
        "audit": {
            "status": "pass",
            "report_id": "audit-foundation",
            "report_sha256": SHA_A,
            "auditor_id": "audit-agent",
            "subject_child_fingerprint": child_plan_fingerprint(child),
            "subject_trial_manifest_sha256": SHA_C,
            "subject_analysis_package_sha256": SHA_B,
            "checks": {"execution": "pass", "calculation": "pass", "claim": "pass"},
            "unresolved_errors": [],
        },
    }


def _locked_ref(name: str, sha256: str) -> dict:
    return {"id": name, "type": f"{name}_manifest", "path": f"locked/{name}.json", "sha256": sha256}


def _check(check_id: str, passed: bool, artifact_id: str) -> dict:
    return {"id": check_id, "passed": passed, "evidence_artifact_id": artifact_id}
