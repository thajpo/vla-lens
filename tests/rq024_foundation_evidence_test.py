from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from vla_lens.research_io import canonical_research_fingerprint, file_sha256
from vla_lens.rq024_foundation_evidence import (
    FOUNDATION_TRIAL_FIELDS,
    FoundationEvidenceError,
    create_foundation_evidence_index,
    directory_sha256,
    seed_bundle_fingerprint,
    trial_row_fingerprint,
)

BUDGET = {
    "model_calls": 0,
    "action_generations": 0,
    "full_rollouts": 0,
    "simulator_steps": 0,
    "probe_fits": 0,
    "persistent_gb": 0,
    "ephemeral_gb": 0,
}
SEED_DOMAINS = ("layout", "reset", "environment", "policy", "flow_noise")


def test_indexes_all_attempts_and_creates_deterministic_manifest(tmp_path: Path) -> None:
    fixture = _write_complete_fixture(tmp_path)

    payload, created = create_foundation_evidence_index(**fixture["arguments"])

    assert created
    assert payload["trial_count"] == 72
    assert len(payload["trials"]) == 72
    completed = payload["trials"][0]
    excluded = payload["trials"][1]
    assert [attempt["disposition"] for attempt in completed["attempts"]] == [
        "failed",
        "completed",
    ]
    assert completed["accepted_terminal_attempt_id"] == "attempt-000-2"
    assert excluded["accepted_terminal_disposition"] == "excluded"
    measurements = completed["attempts"][-1]["trace"]["measurements"]
    assert set(measurements) == {
        "simulator_success",
        "mujoco_contact",
        "positive_gap_proximity",
        "object_motion",
        "evaluation",
    }
    capability = completed["attempts"][-1]["trace"]["contact_capability"]
    assert capability["sample_phase"] == "pre_action_control_step"
    assert capability["sampling_limitation"] == "synthetic control-step limitation"
    assert str(fixture["external_root"]) not in fixture["output"].read_text(encoding="utf-8")

    repeated, created_again = create_foundation_evidence_index(**fixture["arguments"])
    assert not created_again
    assert repeated == payload


def test_rejects_overwritten_external_trace(tmp_path: Path) -> None:
    fixture = _write_complete_fixture(tmp_path)
    scene_path = fixture["trace_root"] / (
        "vla_lens/episodes/episode_000000/tables/scene_state.parquet"
    )
    scene_path.write_bytes(scene_path.read_bytes() + b"tampered")

    with pytest.raises(FoundationEvidenceError, match="Artifact hash mismatch"):
        create_foundation_evidence_index(**fixture["arguments"])


def test_rejects_unapproved_external_root(tmp_path: Path) -> None:
    fixture = _write_complete_fixture(tmp_path)
    arguments = dict(fixture["arguments"])
    arguments["external_roots"] = {}

    with pytest.raises(FoundationEvidenceError, match="unapproved root_id"):
        create_foundation_evidence_index(**arguments)


def test_rejects_duplicate_trial_rows_before_reading_attempts(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    rows = _trial_rows()
    rows[1]["trial_id"] = rows[0]["trial_id"]
    trials = repo / "trials.csv"
    _write_trials(trials, rows)
    child = _child(file_sha256(trials))
    child_path = repo / "child.json"
    _write_json(child_path, child)

    with pytest.raises(FoundationEvidenceError, match="missing or duplicate trial_id"):
        create_foundation_evidence_index(
            repo_root=repo,
            child_path=child_path,
            trial_manifest_path=trials,
            event_root=repo / "missing-events",
            output_path=repo / "evidence.json",
        )


def _write_complete_fixture(tmp_path: Path) -> dict[str, object]:
    repo = tmp_path / "repo"
    external = tmp_path / "external"
    repo.mkdir()
    external.mkdir()
    rows = _trial_rows()
    trials_path = repo / "configs" / "trials.csv"
    _write_trials(trials_path, rows)
    child = _child(file_sha256(trials_path))
    child_path = repo / "configs" / "child.json"
    _write_json(child_path, child)
    child_fingerprint = canonical_research_fingerprint(child)

    trace_root = external / "traces" / rows[0]["trial_id"]
    _write_trace(trace_root, rows[0], child)
    trace_ref = {
        "id": "trace-000",
        "type": "lerobot_v3_trace",
        "root_id": "foundation",
        "path": f"traces/{rows[0]['trial_id']}",
        "sha256": directory_sha256(trace_root),
    }
    receipt_path = external / "receipts" / "attempt-000-2.json"
    receipt = {
        "schema_version": 1,
        "kind": "vla_lens.trial_runtime_receipt",
        "attempt_id": "attempt-000-2",
        "trial_id": rows[0]["trial_id"],
        "child_lock_fingerprint": _hash("child-lock"),
        "runtime_config_fingerprint": child["runtime"]["runner"]["config"]["sha256"],
        "seed_bundle_fingerprint": seed_bundle_fingerprint(rows[0]),
        "runtime_check_status": "pass",
        "terminal_status": "rollout_success",
        "output_refs": [trace_ref],
        "created_utc": "2026-01-01T00:00:00+00:00",
    }
    _write_json(receipt_path, receipt)
    receipt_ref = {
        "id": "receipt-000-2",
        "type": "trial_runtime_receipt",
        "root_id": "foundation",
        "path": "receipts/attempt-000-2.json",
        "sha256": file_sha256(receipt_path),
    }

    event_root = repo / "events"
    event_root.mkdir()
    ledger = _Ledger(event_root)
    lock = ledger.append(
        "child-locked",
        "child_locked",
        {
            "child_ref": {
                "id": child["child_plan_id"],
                "type": "research_child",
                "root_id": "repo",
                "path": "configs/child.json",
                "sha256": file_sha256(child_path),
                "content_fingerprint": child_fingerprint,
            },
            "study_id": "FOUNDATION",
            "study_fingerprint": _hash("study"),
            "study_instance_id": "foundation-r1",
            "lock_receipt_ref": _artifact_ref("lock", "lock.json"),
            "reservation_id": "foundation-reservation",
            "predecessor_result_events": [],
            "audit_events": [],
            "prior_ledger_tip": None,
        },
    )
    lock_ref = _event_ref(lock)

    first_failed = ledger.append(
        "attempt-000-1-start",
        "trial_attempt_started",
        _start_payload(rows[0], "attempt-000-1", 1, lock_ref, child),
    )
    first_terminal = ledger.append(
        "attempt-000-1-failed",
        "trial_attempt_failed",
        _failed_payload(_event_ref(first_failed)),
    )
    ledger.append(
        "attempt-000-1-retry",
        "deviation_recorded",
        _deviation_payload(_event_ref(first_terminal), "retry"),
    )
    completed_start = ledger.append(
        "attempt-000-2-start",
        "trial_attempt_started",
        _start_payload(rows[0], "attempt-000-2", 2, lock_ref, child),
    )
    ledger.append(
        "attempt-000-2-completed",
        "trial_attempt_completed",
        {
            "start_event": _event_ref(completed_start),
            "terminal_status": "rollout_success",
            "output_refs": [trace_ref],
            "actual_budget": dict(BUDGET),
            "runtime_receipt_ref": receipt_ref,
        },
    )

    for index, row in enumerate(rows[1:], start=1):
        attempt_id = f"attempt-{index:03d}-1"
        start = ledger.append(
            f"{attempt_id}-start",
            "trial_attempt_started",
            _start_payload(row, attempt_id, 1, lock_ref, child),
        )
        terminal = ledger.append(
            f"{attempt_id}-failed",
            "trial_attempt_failed",
            _failed_payload(_event_ref(start)),
        )
        ledger.append(
            f"{attempt_id}-excluded",
            "deviation_recorded",
            _deviation_payload(_event_ref(terminal), "exclude_trial"),
        )

    output = repo / "evidence" / "foundation-index.json"
    return {
        "arguments": {
            "repo_root": repo,
            "child_path": child_path,
            "trial_manifest_path": trials_path,
            "event_root": event_root,
            "output_path": output,
            "external_roots": {"foundation": external},
        },
        "external_root": external,
        "output": output,
        "trace_root": trace_root,
    }


def _trial_rows() -> list[dict[str, str]]:
    rows = []
    for index in range(72):
        row = dict.fromkeys(FOUNDATION_TRIAL_FIELDS, "")
        row.update(
            {
                "trial_id": f"trial-{index:03d}",
                "child_plan_id": "rq024-foundation-synthetic-r1",
                "dataset_id": "rq024-foundation-synthetic-r1",
                "benchmark": "synthetic",
                "task_id": str(index),
                "task_name": f"synthetic_task_{index:03d}",
                "split": "synthetic_baseline",
                "capture_profile": "rollout",
                "canonical_family_id": f"family-{index // 3:02d}",
                "candidate_position": str(index // 3),
                "pool": "discovery" if index % 2 == 0 else "confirmation",
                "pool_position": str(index // 6),
                "cell_id": "untouched_baseline",
                "replicate_id": str(index % 3),
                "layout_id": str(index),
            }
        )
        for domain_index, domain in enumerate(SEED_DOMAINS):
            value = index * 10 + domain_index
            row[f"{domain}_seed_identity"] = _hash(f"{domain}-{value}")
            row[f"{domain}_seed"] = str(value)
        row["seed"] = row["reset_seed"]
        rows.append(row)
    return rows


def _child(trials_sha256: str) -> dict[str, object]:
    hashes = {
        name: _hash(name)
        for name in (
            "snapshot",
            "environment",
            "camera",
            "controller",
            "preprocessor",
            "postprocessor",
            "source-tree",
            "effective-config",
        )
    }
    return {
        "schema_version": 1,
        "kind": "vla_lens.research_child",
        "child_plan_id": "rq024-foundation-synthetic-r1",
        "study": {"id": "FOUNDATION"},
        "trials": {
            "expected_count": 72,
            "manifest": {"id": "synthetic-trials", "sha256": trials_sha256},
        },
        "runtime": {
            "model": {
                "repo_id": "synthetic/pi05",
                "revision": "a" * 40,
                "snapshot_manifest_sha256": hashes["snapshot"],
            },
            "environment": {
                "backend": "synthetic",
                "package_receipt": {
                    "id": "synthetic-environment",
                    "path": "environment.json",
                    "sha256": hashes["environment"],
                },
                "camera_config_sha256": hashes["camera"],
                "controller_config_sha256": hashes["controller"],
                "preprocessor_config_sha256": hashes["preprocessor"],
                "postprocessor_config_sha256": hashes["postprocessor"],
            },
            "code": {
                "implementation_commit": "b" * 40,
                "source_tree_sha256": hashes["source-tree"],
            },
            "runner": {
                "entrypoint": "synthetic-runner",
                "argv": ["--synthetic"],
                "config": {"sha256": hashes["effective-config"]},
            },
        },
        "completion": {
            "valid_trial_statuses": ["rollout_success", "rollout_behavior_failure"]
        },
        "protocol_lock": {
            "locked_choices": {
                "simulator_contact_telemetry_capability_status": "synthetic-available"
            }
        },
    }


def _write_trace(root: Path, row: dict[str, str], child: dict[str, object]) -> None:
    bundle = root / "vla_lens" / "episodes" / "episode_000000"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "vla_lens" / "tables").mkdir(parents=True)
    (bundle / "tables").mkdir(parents=True)
    position_array = bundle / "arrays" / "scene_object_pos.zarr"
    position_array.mkdir(parents=True)
    (position_array / "0").write_bytes(b"synthetic object positions")
    features = {
        "episode_index": {"dtype": "int64"},
        "frame_index": {"dtype": "int64"},
        "timestamp": {"dtype": "float32"},
        "task_index": {"dtype": "int64"},
        "action": {"dtype": "float32", "shape": [1]},
        "observation.state": {"dtype": "float32", "shape": [2]},
    }
    _write_json(
        root / "meta" / "info.json",
        {
            "codebase_version": "v3.0",
            "data_path": "data/chunk-000/data.parquet",
            "features": features,
        },
    )
    _write_json(root / "meta" / "stats.json", {})
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "synthetic task"}) + "\n", encoding="utf-8"
    )
    pd.DataFrame({"episode_index": [0], "length": [1], "task_index": [0]}).to_parquet(
        root / "meta" / "episodes" / "chunk-000" / "episodes.parquet", index=False
    )
    pd.DataFrame(
        {
            "episode_index": [0],
            "frame_index": [0],
            "timestamp": [0.0],
            "task_index": [0],
            "action": [[0.0]],
            "observation.state": [[0.0, 0.0]],
        }
    ).to_parquet(root / "data" / "chunk-000" / "data.parquet", index=False)
    pd.DataFrame(
        {
            "episode_index": [0],
            "trace_id": [row["trial_id"]],
            "overlay_path": ["vla_lens/episodes/episode_000000"],
        }
    ).to_parquet(root / "vla_lens" / "tables" / "episode_refs.parquet", index=False)
    _write_json(
        root / "vla_lens" / "overlay.json",
        {
            "overlay_schema_version": "0.1.0",
            "robot_dataset_format": "lerobot_v3",
            "overlay_root": "vla_lens",
            "episodes": 1,
        },
    )
    runtime = child["runtime"]
    environment = runtime["environment"]
    capability = {
        "available": True,
        "verdict": "available_control_step_contact_manifold",
        "sample_phase": "pre_action_control_step",
        "sample_phase_semantics": "synthetic pre-action sample",
        "sampling_limitation": "synthetic control-step limitation",
        "distance_semantics": "signed distance",
    }
    _write_json(
        bundle / "manifest.json",
        {
            "trace_id": row["trial_id"],
            "episode_id": row["trial_id"],
            "task_id": row["task_id"],
            "prompt": "synthetic task",
            "model_id": runtime["model"]["repo_id"],
            "env_id": "synthetic",
            "robot_id": "synthetic",
            "outcome": "success",
            "length": 1,
            "schema_version": "0.1.0",
            "metadata": {
                "dataset_id": row["dataset_id"],
                "lerobot_episode_index": 0,
                "runtime_audit": {
                    "trial_id": row["trial_id"],
                    "child_plan_id": row["child_plan_id"],
                    "canonical_family_id": row["canonical_family_id"],
                    "pool": row["pool"],
                    "replicate_id": row["replicate_id"],
                    "seed_identities": {
                        domain: int(row[f"{domain}_seed"]) for domain in SEED_DOMAINS
                    },
                    "model_revision": runtime["model"]["revision"],
                    "snapshot_manifest_sha256": runtime["model"][
                        "snapshot_manifest_sha256"
                    ],
                    "camera_config_sha256": environment["camera_config_sha256"],
                    "controller_config_sha256": environment["controller_config_sha256"],
                    "preprocessor_config_sha256": environment["preprocessor_config_sha256"],
                    "postprocessor_config_sha256": environment["postprocessor_config_sha256"],
                },
                "capture_capabilities": {"simulator_contact_telemetry": capability},
            },
        },
    )
    pd.DataFrame(
        {
            "context_kind": ["object", "mujoco_contact"],
            "physical_contact": [False, False],
            "positive_gap_proximity": [False, True],
        }
    ).to_parquet(bundle / "tables" / "scene_state.parquet", index=False)
    pd.DataFrame(
        {"metric_name": ["success"], "metric_value": [1.0], "passed": [True]}
    ).to_parquet(bundle / "tables" / "evaluation.parquet", index=False)
    pd.DataFrame(
        {
            "name": ["scene_object_pos"],
            "relative_path": ["arrays/scene_object_pos.zarr"],
        }
    ).to_parquet(bundle / "tables" / "array_index.parquet", index=False)


def _start_payload(
    row: dict[str, str],
    attempt_id: str,
    ordinal: int,
    lock_ref: dict[str, object],
    child: dict[str, object],
) -> dict[str, object]:
    return {
        "child_lock_event": lock_ref,
        "reservation_id": "foundation-reservation",
        "trial_id": row["trial_id"],
        "attempt_id": attempt_id,
        "ordinal": ordinal,
        "trial_manifest_row_fingerprint": trial_row_fingerprint(row),
        "runtime_config_fingerprint": child["runtime"]["runner"]["config"]["sha256"],
        "seed_bundle_fingerprint": seed_bundle_fingerprint(row),
        "requested_budget": dict(BUDGET),
    }


def _failed_payload(start_ref: dict[str, object]) -> dict[str, object]:
    return {
        "start_event": start_ref,
        "failure_stage": "synthetic",
        "error_code": "synthetic_failure",
        "retryable": True,
        "log_refs": [],
        "actual_budget": dict(BUDGET),
    }


def _deviation_payload(
    terminal_ref: dict[str, object], disposition: str
) -> dict[str, object]:
    return {
        "target_event": terminal_ref,
        "category": "technical",
        "disposition": disposition,
        "reason": f"synthetic {disposition}",
        "evidence_refs": [],
    }


def _artifact_ref(identifier: str, path: str) -> dict[str, str]:
    return {
        "id": identifier,
        "type": "synthetic",
        "root_id": "repo",
        "path": path,
        "sha256": _hash(path),
        "content_fingerprint": _hash("child-lock"),
    }


class _Ledger:
    def __init__(self, root: Path):
        self.root = root
        self.previous: str | None = None
        self.sequence = 0

    def append(
        self, event_id: str, event_type: str, payload: dict[str, object]
    ) -> dict[str, object]:
        self.sequence += 1
        event = {
            "schema_version": 1,
            "program_id": "synthetic-program",
            "program_fingerprint": _hash("program"),
            "sequence": self.sequence,
            "event_id": event_id,
            "event_type": event_type,
            "created_utc": f"2026-01-01T00:00:{self.sequence:02d}+00:00",
            "actor_id": "synthetic-test",
            "subject_id": event_id,
            "subject_fingerprint": _hash(event_id),
            "previous_event_sha256": self.previous,
            "payload": payload,
        }
        path = self.root / f"{self.sequence:06d}-{event_id}.json"
        _write_json(path, event)
        self.previous = file_sha256(path)
        return {**event, "event_sha256": self.previous}


def _event_ref(event: dict[str, object]) -> dict[str, object]:
    return {
        "sequence": event["sequence"],
        "event_id": event["event_id"],
        "event_sha256": event["event_sha256"],
    }


def _write_trials(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode()).hexdigest()}"
