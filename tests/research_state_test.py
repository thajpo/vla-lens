from __future__ import annotations

import json
from pathlib import Path

from vla_lens.research_child import child_plan_fingerprint, study_fingerprint
from vla_lens.research_io import (
    canonical_research_fingerprint,
    file_sha256,
    load_research_mapping,
)
from vla_lens.research_plan import research_plan_fingerprint
from vla_lens.research_state import reduce_campaign_events

REPO_ROOT = Path(__file__).resolve().parents[1]
PROGRAM_PATH = REPO_ROOT / "configs/campaigns/rq024_controlled_scene_to_behavior.yaml"
PROGRAM = load_research_mapping(PROGRAM_PATH)


def test_hash_matched_but_invalid_child_cannot_enter_campaign_state(tmp_path):
    program_path = tmp_path / "program.yaml"
    program_path.write_bytes(PROGRAM_PATH.read_bytes())
    schema_path = tmp_path / "schema.json"
    schema_path.write_text('{"valid": true}\n', encoding="utf-8")
    child_path = tmp_path / "invalid-child.json"
    child = {"schema_version": 1, "child_plan_id": "rq024-foundation-bogus"}
    child_path.write_text(json.dumps(child), encoding="utf-8")
    child_ref = _document_ref(tmp_path, child_path, "invalid-child", "research_child")
    child_fingerprint = child_plan_fingerprint(child)
    study = next(item for item in PROGRAM["studies"] if item["id"] == "FOUNDATION")
    documents: list[tuple[dict, str]] = []
    _append(
        documents,
        "program_locked",
        research_plan_fingerprint(PROGRAM),
        {
            "program_ref": _document_ref(
                tmp_path, program_path, "rq024-program", "research_program"
            ),
            "schema_check_ref": _artifact_ref(
                tmp_path, schema_path, "program-schema", "program_schema_check"
            ),
            "manifest_commit": "a" * 40,
        },
    )
    _append(
        documents,
        "child_prepared",
        child_fingerprint,
        {
            "child_ref": child_ref,
            "study_id": "FOUNDATION",
            "study_fingerprint": study_fingerprint(study),
            "study_instance_id": "foundation-instance-r1",
        },
    )
    audit_types = ["schema", "design", "runner", "budget", *study["required_audits"]]
    audit_refs = []
    lock_audits = []
    for index, audit_type in enumerate(audit_types):
        audit_id = f"audit-{index}"
        auditor_id = f"auditor-{index}"
        report_path = tmp_path / f"audit-{index}.json"
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "vla_lens.research_audit",
                    "audit_id": audit_id,
                    "audit_type": audit_type,
                    "subject_kind": "child_plan",
                    "subject_fingerprint": child_fingerprint,
                    "auditor_id": auditor_id,
                    "created_utc": "2026-07-26T12:00:00+00:00",
                    "verdict": "pass",
                    "checks": [
                        {
                            "id": "schema",
                            "status": "pass",
                            "evidence_refs": [
                                {
                                    "id": "invalid-child",
                                    "type": "research_child",
                                    "path": "invalid-child.json",
                                    "sha256": child_ref["sha256"],
                                }
                            ],
                        }
                    ],
                    "unresolved_errors": [],
                }
            ),
            encoding="utf-8",
        )
        report_ref = _artifact_ref(tmp_path, report_path, audit_id, "research_audit_report")
        event = _append(
            documents,
            "audit_completed",
            child_fingerprint,
            {
                "audit_id": audit_id,
                "audit_type": audit_type,
                "subject_kind": "child_plan",
                "subject_fingerprint": child_fingerprint,
                "auditor_id": auditor_id,
                "verdict": "pass",
                "report_ref": report_ref,
                "checks": [{"id": "schema", "status": "pass"}],
                "unresolved_errors": [],
            },
        )
        audit_refs.append(_event_ref(event, documents[-1][1]))
        lock_audits.append(
            {
                "audit_type": audit_type,
                "auditor_id": auditor_id,
                "verdict": "pass",
                "subject_child_fingerprint": child_fingerprint,
                "artifact": {
                    "id": report_ref["id"],
                    "type": report_ref["type"],
                    "path": report_ref["path"],
                    "sha256": report_ref["sha256"],
                },
            }
        )
    _append(
        documents,
        "budget_reserved",
        child_fingerprint,
        {
            "reservation_id": "foundation-reservation-r1",
            "child_ref": child_ref,
            "study_id": "FOUNDATION",
            "study_instance_id": "foundation-instance-r1",
            "hardware": True,
            "budget": _zero_budget(),
            "output_namespace": "rq024/foundation-r1",
        },
    )
    prior_tip = documents[-1][1]
    lock = {
        "schema_version": 1,
        "kind": "vla_lens.research_child_lock",
        "lock_id": "invalid-lock",
        "program_id": PROGRAM["program_id"],
        "program_fingerprint": research_plan_fingerprint(PROGRAM),
        "study_id": "FOUNDATION",
        "study_fingerprint": study_fingerprint(study),
        "child_plan_id": child["child_plan_id"],
        "child_plan_fingerprint": child_fingerprint,
        "manifest_commit": "a" * 40,
        "locked_utc": "2026-07-26T12:00:00+00:00",
        "prepared_by": "planner-agent",
        "reservation_id": "foundation-reservation-r1",
        "prior_ledger_tip": prior_tip,
        "audits": lock_audits,
    }
    lock_path = tmp_path / "invalid-lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    _append(
        documents,
        "child_locked",
        child_fingerprint,
        {
            "child_ref": child_ref,
            "study_id": "FOUNDATION",
            "study_fingerprint": study_fingerprint(study),
            "study_instance_id": "foundation-instance-r1",
            "lock_receipt_ref": _document_ref(
                tmp_path, lock_path, "invalid-lock", "research_child_lock"
            ),
            "reservation_id": "foundation-reservation-r1",
            "predecessor_result_events": [],
            "audit_events": audit_refs,
            "prior_ledger_tip": prior_tip,
        },
    )

    check = reduce_campaign_events(
        documents,
        PROGRAM,
        repo_root=tmp_path,
        verify_artifacts=True,
    )

    assert not check.valid
    assert "FOUNDATION" not in check.state.locked
    assert any(issue.code.startswith("locked_child_missing_fields") for issue in check.issues)


def _append(
    documents: list[tuple[dict, str]], event_type: str, subject: str, payload: dict
) -> dict:
    sequence = len(documents) + 1
    event = {
        "schema_version": 1,
        "program_id": PROGRAM["program_id"],
        "program_fingerprint": research_plan_fingerprint(PROGRAM),
        "sequence": sequence,
        "event_id": f"event-{sequence}",
        "event_type": event_type,
        "created_utc": f"2026-07-26T12:{sequence:02d}:00+00:00",
        "actor_id": "test-agent",
        "subject_id": f"subject-{sequence}",
        "subject_fingerprint": subject,
        "previous_event_sha256": documents[-1][1] if documents else None,
        "payload": payload,
    }
    event_hash = f"sha256:{sequence:064x}"
    documents.append((event, event_hash))
    return event


def _event_ref(event: dict, event_hash: str) -> dict:
    return {
        "sequence": event["sequence"],
        "event_id": event["event_id"],
        "event_sha256": event_hash,
    }


def _artifact_ref(root: Path, path: Path, artifact_id: str, artifact_type: str) -> dict:
    return {
        "id": artifact_id,
        "type": artifact_type,
        "root_id": "repo",
        "path": str(path.relative_to(root)),
        "sha256": file_sha256(path),
    }


def _document_ref(root: Path, path: Path, artifact_id: str, artifact_type: str) -> dict:
    document = load_research_mapping(path)
    return {
        **_artifact_ref(root, path, artifact_id, artifact_type),
        "content_fingerprint": canonical_research_fingerprint(document),
    }


def _zero_budget() -> dict:
    return {
        "model_calls": 0,
        "action_generations": 0,
        "full_rollouts": 0,
        "simulator_steps": 0,
        "probe_fits": 0,
        "persistent_gb": 0,
        "ephemeral_gb": 0,
    }
