from __future__ import annotations

import json
from pathlib import Path

import pytest

from vla_lens.research_events import (
    append_research_event,
    verify_research_event_ledger,
)
from vla_lens.research_io import load_research_mapping, write_bytes_create_only
from vla_lens.research_plan import load_research_plan, research_plan_fingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
PROGRAM = load_research_plan(
    REPO_ROOT / "configs/campaigns/rq024_controlled_scene_to_behavior.yaml"
)
SUBJECT = "sha256:" + "a" * 64


def test_campaign_events_are_append_only_and_hash_chained(tmp_path):
    root = tmp_path / "events"

    first_path, first_hash = append_research_event(
        root,
        PROGRAM,
        event_id="program-locked-r1",
        event_type="program_locked",
        actor_id="planner-agent",
        subject_id=PROGRAM["program_id"],
        subject_fingerprint=research_plan_fingerprint(PROGRAM),
        payload=_program_lock_payload(),
        created_utc="2026-07-26T12:00:00+00:00",
    )
    second_path, _ = append_research_event(
        root,
        PROGRAM,
        event_id="foundation-child-prepared-r1",
        event_type="child_prepared",
        actor_id="planner-agent",
        subject_id="rq024-foundation-r1",
        subject_fingerprint=SUBJECT,
        payload={
            "child_ref": _document_ref("foundation-child", SUBJECT),
            "study_id": "FOUNDATION",
            "study_fingerprint": _study_fingerprint("FOUNDATION"),
            "study_instance_id": "foundation-instance-r1",
        },
        created_utc="2026-07-26T12:01:00+00:00",
    )

    check = verify_research_event_ledger(root, PROGRAM)
    second = load_research_mapping(second_path)

    assert check.valid, check.to_dict()
    assert check.event_count == 2
    assert second["previous_event_sha256"] == first_hash
    assert first_path.name.startswith("000001-")
    assert second_path.name.startswith("000002-")


def test_campaign_event_tampering_is_detected(tmp_path):
    root = tmp_path / "events"
    first_path, _ = append_research_event(
        root,
        PROGRAM,
        event_id="program-locked-r1",
        event_type="program_locked",
        actor_id="planner-agent",
        subject_id=PROGRAM["program_id"],
        subject_fingerprint=research_plan_fingerprint(PROGRAM),
        payload=_program_lock_payload(),
    )
    append_research_event(
        root,
        PROGRAM,
        event_id="foundation-child-prepared-r1",
        event_type="child_prepared",
        actor_id="planner-agent",
        subject_id="rq024-foundation-r1",
        subject_fingerprint=SUBJECT,
        payload={
            "child_ref": _document_ref("foundation-child", SUBJECT),
            "study_id": "FOUNDATION",
            "study_fingerprint": _study_fingerprint("FOUNDATION"),
            "study_instance_id": "foundation-instance-r1",
        },
    )
    event = json.loads(first_path.read_text(encoding="utf-8"))
    event["payload"]["schema_valid"] = False
    first_path.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    check = verify_research_event_ledger(root, PROGRAM)

    assert not check.valid
    assert "event_chain_broken" in {issue.code for issue in check.issues}


def test_strict_json_and_create_only_evidence_reject_ambiguity(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"status": "pass", "status": "fail"}\n', encoding="utf-8")
    evidence = tmp_path / "evidence.json"

    with pytest.raises(ValueError, match="duplicate key"):
        load_research_mapping(duplicate)
    assert write_bytes_create_only(evidence, b"same\n") is True
    assert write_bytes_create_only(evidence, b"same\n") is False
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        write_bytes_create_only(evidence, b"different\n")


def test_state_reducer_rejects_pool_access_or_advancement_without_evidence(tmp_path):
    root = tmp_path / "events"
    append_research_event(
        root,
        PROGRAM,
        event_id="program-locked-r1",
        event_type="program_locked",
        actor_id="planner-agent",
        subject_id=PROGRAM["program_id"],
        subject_fingerprint=research_plan_fingerprint(PROGRAM),
        payload=_program_lock_payload(),
    )

    with pytest.raises(ValueError, match="illegal campaign transition"):
        append_research_event(
            root,
            PROGRAM,
            event_id="early-pool-access",
            event_type="pool_accessed",
            actor_id="runner-agent",
            subject_id="fake-child",
            subject_fingerprint=SUBJECT,
            payload={
                "child_lock_event": _event_ref(99, "fake-lock", SUBJECT),
                "family_pool": "confirmation",
                "namespace": "geometry_confirmation_scene_state",
                "access_mode": "measurement",
                "exposure_record_ref": _artifact_ref("exposure"),
                "data_refs": [],
            },
        )

    with pytest.raises(ValueError, match="illegal campaign transition"):
        append_research_event(
            root,
            PROGRAM,
            event_id="fake-advance",
            event_type="study_advanced",
            actor_id="planner-agent",
            subject_id="GEOMETRY-DISCOVERY",
            subject_fingerprint=SUBJECT,
            payload={
                "result_event": _event_ref(98, "fake-result", SUBJECT),
                "outcome": "positive",
                "program_action": "reevaluate_program",
                "newly_eligible_studies": ["GEOMETRY-CONFIRMATION"],
            },
        )


def test_empty_ledger_status_selects_program_lock_as_only_next_action(tmp_path):
    check = verify_research_event_ledger(tmp_path / "events", PROGRAM)
    state = check.state.to_dict(PROGRAM)

    assert check.valid
    assert state["status"]["next_action"]["action_id"] == "lock_program"
    assert state["status"]["hardware_authorized"] is False


def _program_lock_payload() -> dict:
    return {
        "program_ref": _document_ref("rq024-program", research_plan_fingerprint(PROGRAM)),
        "schema_check_ref": _artifact_ref("program-schema"),
        "manifest_commit": "a" * 40,
    }


def _study_fingerprint(study_id: str) -> str:
    study = next(item for item in PROGRAM["studies"] if item["id"] == study_id)
    from vla_lens.research_io import canonical_research_fingerprint

    return canonical_research_fingerprint(study)


def _artifact_ref(name: str) -> dict:
    return {
        "id": name,
        "type": "test_artifact",
        "root_id": "repo",
        "path": f"locked/{name}.json",
        "sha256": SUBJECT,
    }


def _document_ref(name: str, content_fingerprint: str) -> dict:
    return {**_artifact_ref(name), "content_fingerprint": content_fingerprint}


def _event_ref(sequence: int, event_id: str, event_hash: str) -> dict:
    return {"sequence": sequence, "event_id": event_id, "event_sha256": event_hash}
