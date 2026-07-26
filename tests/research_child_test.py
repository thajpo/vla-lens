from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from vla_lens.research_child import (
    check_research_child,
    check_research_child_lock,
    child_plan_fingerprint,
    load_research_child,
    study_fingerprint,
)
from vla_lens.research_plan import load_research_plan, research_plan_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
PROGRAM = load_research_plan(
    REPO_ROOT / "configs/campaigns/rq024_controlled_scene_to_behavior.yaml"
)
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64


def test_child_contract_binds_program_study_predecessor_metric_and_budget():
    child = _valid_foundation_child()

    check = check_research_child(child, PROGRAM)

    assert check.valid, check.to_dict()
    assert check.files_verified is False
    assert check.fingerprint == child_plan_fingerprint(child)


def test_child_contract_rejects_unknown_fields_and_weakened_parent_budget():
    child = _valid_foundation_child()
    child["surprise_override"] = True
    child["budget"]["max_model_calls"] = 2000

    check = check_research_child(child, PROGRAM)
    codes = {issue.code for issue in check.issues}

    assert not check.valid
    assert "unknown_contract_fields" in codes
    assert "child_budget_exceeds_study" in codes


def test_lock_receipt_is_separate_bound_and_independently_audited():
    child = _valid_foundation_child()
    receipt = _lock_receipt(child)

    check = check_research_child_lock(receipt, child, PROGRAM)

    assert check.valid, check.to_dict()
    assert check.audit_files_verified is False

    same_agent = deepcopy(receipt)
    same_agent["audits"][0]["auditor_id"] = child["prepared_by"]
    same_agent_check = check_research_child_lock(same_agent, child, PROGRAM)
    assert "audit_not_independent" in {issue.code for issue in same_agent_check.issues}


def test_foundation_template_is_honestly_blocked_until_exact_inputs_exist():
    template = load_research_child(
        REPO_ROOT / "configs/campaigns/rq024_foundation.child.template.yaml"
    )

    check = check_research_child(template, PROGRAM)

    assert not check.valid
    codes = {issue.code for issue in check.issues}
    assert "program_fingerprint_mismatch" in codes
    assert "invalid_child_hash" in codes


def _lock_receipt(child: dict) -> dict:
    study = next(item for item in PROGRAM["studies"] if item["id"] == child["study"]["id"])
    audits = []
    for audit_type in child["required_audits"]:
        audits.append(
            {
                "audit_type": audit_type,
                "auditor_id": f"{audit_type}-auditor",
                "verdict": "pass",
                "subject_child_fingerprint": child_plan_fingerprint(child),
                "artifact": {
                    "id": f"{audit_type}-audit",
                    "type": "research_child_audit",
                    "path": f"audits/{audit_type}.json",
                    "sha256": SHA_A if audit_type == "schema" else SHA_B,
                },
            }
        )
    return {
        "schema_version": 1,
        "kind": "vla_lens.research_child_lock",
        "lock_id": "geometry-discovery-lock-r1",
        "program_id": PROGRAM["program_id"],
        "program_fingerprint": child["program"]["fingerprint"],
        "study_id": study["id"],
        "study_fingerprint": study_fingerprint(study),
        "child_plan_id": child["child_plan_id"],
        "child_plan_fingerprint": child_plan_fingerprint(child),
        "manifest_commit": "b" * 40,
        "locked_utc": "2026-07-26T12:00:00+00:00",
        "prepared_by": child["prepared_by"],
        "first_output_absent": True,
        "audits": audits,
    }


def _valid_foundation_child() -> dict:
    child = deepcopy(
        load_research_child(REPO_ROOT / "configs/campaigns/rq024_foundation.child.template.yaml")
    )
    study = next(item for item in PROGRAM["studies"] if item["id"] == "FOUNDATION")
    child["prepared_by"] = "planner-agent"
    child["program"]["fingerprint"] = research_plan_fingerprint(PROGRAM)
    child["study"]["fingerprint"] = study_fingerprint(study)
    child["cohort"]["manifest"]["sha256"] = SHA_A
    child["cohort"]["exposure_log"]["sha256"] = SHA_B
    child["trials"]["manifest"]["sha256"] = SHA_C
    child["runtime"]["model"].update({"revision": "a" * 40, "snapshot_manifest_sha256": SHA_A})
    child["runtime"]["environment"].update(
        {
            "backend": "rocm",
            "camera_config_sha256": SHA_A,
            "controller_config_sha256": SHA_A,
            "preprocessor_config_sha256": SHA_A,
            "postprocessor_config_sha256": SHA_A,
        }
    )
    child["runtime"]["environment"]["package_receipt"]["sha256"] = SHA_A
    child["runtime"]["code"].update(
        {"implementation_commit": "a" * 40, "source_tree_sha256": SHA_A}
    )
    child["runtime"]["runner"]["argv"][1] = "rocm"
    child["runtime"]["runner"]["config"]["sha256"] = SHA_A
    return child
