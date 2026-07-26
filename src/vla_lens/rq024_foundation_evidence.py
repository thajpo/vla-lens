"""Deterministic external-evidence index for the locked RQ-024 FOUNDATION child."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import pandas as pd

from vla_lens.capture.lerobot_v3 import (
    VLA_LENS_OVERLAY_MANIFEST,
    VLA_LENS_OVERLAY_REFERENCES,
    validate_lerobot_v3_dataset,
)
from vla_lens.dataset.common import OVERLAY_SCHEMA_VERSION
from vla_lens.research_io import (
    canonical_research_fingerprint,
    file_sha256,
    load_research_mapping,
    write_bytes_create_only,
)
from vla_lens.research_state import PAYLOAD_FIELDS
from vla_lens.traces.bundle import TraceBundle

EXPECTED_TRIAL_COUNT = 72
SEED_DOMAINS = ("layout", "reset", "environment", "policy", "flow_noise")
FOUNDATION_TRIAL_FIELDS = (
    "trial_id",
    "child_plan_id",
    "dataset_id",
    "benchmark",
    "task_id",
    "task_name",
    "split",
    "capture_profile",
    "canonical_family_id",
    "candidate_position",
    "pool",
    "pool_position",
    "cell_id",
    "replicate_id",
    "layout_seed_identity",
    "layout_seed",
    "reset_seed_identity",
    "reset_seed",
    "environment_seed_identity",
    "environment_seed",
    "policy_seed_identity",
    "policy_seed",
    "flow_noise_seed_identity",
    "flow_noise_seed",
    "layout_id",
    "seed",
)
EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "program_id",
        "program_fingerprint",
        "sequence",
        "event_id",
        "event_type",
        "created_utc",
        "actor_id",
        "subject_id",
        "subject_fingerprint",
        "previous_event_sha256",
        "payload",
    }
)
EVENT_REF_FIELDS = frozenset({"sequence", "event_id", "event_sha256"})
SHA256_ZERO = "sha256:" + ("0" * 64)


class FoundationEvidenceError(ValueError):
    """Raised when FOUNDATION evidence is incomplete, ambiguous, or mutable."""


def trial_row_fingerprint(row: Mapping[str, str]) -> str:
    """Return the canonical identity used by a trial-attempt start event."""

    return canonical_research_fingerprint(dict(row))


def seed_bundle(row: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """Return all locked seed values together with their independent identities."""

    bundle: dict[str, dict[str, Any]] = {}
    for domain in SEED_DOMAINS:
        identity = str(row.get(f"{domain}_seed_identity") or "")
        value = str(row.get(f"{domain}_seed") or "")
        if not _sha256(identity) or not value:
            raise FoundationEvidenceError(f"Trial row has an invalid {domain} seed identity")
        try:
            parsed = int(value)
        except ValueError as exc:
            raise FoundationEvidenceError(f"Trial row has a non-integer {domain} seed") from exc
        bundle[domain] = {"identity": identity, "value": parsed}
    return bundle


def seed_bundle_fingerprint(row: Mapping[str, str]) -> str:
    """Return the canonical fingerprint for :func:`seed_bundle`."""

    return canonical_research_fingerprint(seed_bundle(row))


def directory_sha256(path: str | Path) -> str:
    """Hash a directory as its sorted relative file paths, sizes, and byte hashes."""

    return _directory_record(Path(path))["sha256"]


def create_foundation_evidence_index(
    *,
    repo_root: str | Path,
    child_path: str | Path,
    trial_manifest_path: str | Path,
    event_root: str | Path,
    output_path: str | Path,
    external_roots: Mapping[str, str | Path] | None = None,
) -> tuple[Mapping[str, Any], bool]:
    """Verify locked attempts and create an immutable, deterministic evidence index.

    External artifact references use ``root_id`` plus a normalized relative path.
    Every non-``repo`` root must be explicitly supplied in ``external_roots``. The
    emitted index retains root IDs rather than machine-specific absolute paths.
    """

    repo = Path(repo_root).resolve()
    child_file = _repo_file(repo, child_path, "child")
    trials_file = _repo_file(repo, trial_manifest_path, "trial manifest")
    destination = _repo_destination(repo, output_path)
    child = load_research_mapping(child_file)
    rows = _load_trials(trials_file, child)
    roots = _approved_roots(repo, external_roots)
    events = _load_event_chain(Path(event_root), roots)

    child_fingerprint = canonical_research_fingerprint(child)
    attempts, deviations = _accepted_attempts(
        events,
        child,
        {
            "id": child["child_plan_id"],
            "path": child_file.relative_to(repo).as_posix(),
            "sha256": file_sha256(child_file),
            "content_fingerprint": child_fingerprint,
        },
        rows,
    )
    indexed_trials = []
    for row in rows:
        trial_id = row["trial_id"]
        indexed_trials.append(
            _index_trial(
                row,
                attempts[trial_id],
                deviations,
                child,
                roots,
            )
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "vla_lens.rq024_foundation_external_evidence_index",
        "child": {
            "child_plan_id": child["child_plan_id"],
            "fingerprint": child_fingerprint,
            "path": child_file.relative_to(repo).as_posix(),
            "sha256": file_sha256(child_file),
            "study_id": child["study"]["id"],
        },
        "trial_manifest": {
            "id": child["trials"]["manifest"]["id"],
            "path": trials_file.relative_to(repo).as_posix(),
            "sha256": file_sha256(trials_file),
            "row_count": len(rows),
        },
        "attempt_ledger": {
            **_approved_location(Path(event_root), roots),
            "event_count": len(events),
            "tip_sha256": events[-1]["event_sha256"],
        },
        "locked_runtime": _locked_runtime(child),
        "trial_count": len(indexed_trials),
        "trials": indexed_trials,
    }
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    created = write_bytes_create_only(destination, content)
    return payload, created


def _load_trials(path: Path, child: Mapping[str, Any]) -> list[dict[str, str]]:
    if child.get("study", {}).get("id") != "FOUNDATION":
        raise FoundationEvidenceError("Evidence index requires the FOUNDATION child")
    expected = child.get("trials", {}).get("expected_count")
    if expected != EXPECTED_TRIAL_COUNT:
        raise FoundationEvidenceError(
            f"FOUNDATION child must lock exactly {EXPECTED_TRIAL_COUNT} trials, observed {expected}"
        )
    declared = child.get("trials", {}).get("manifest", {})
    if declared.get("sha256") != file_sha256(path):
        raise FoundationEvidenceError("Trial manifest bytes differ from the locked child")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FOUNDATION_TRIAL_FIELDS:
            raise FoundationEvidenceError("Trial manifest header differs from the locked schema")
        rows = [dict(row) for row in reader]
    if len(rows) != EXPECTED_TRIAL_COUNT:
        raise FoundationEvidenceError(
            f"Trial manifest must contain exactly {EXPECTED_TRIAL_COUNT} rows"
        )
    trial_ids = [row.get("trial_id", "") for row in rows]
    if not all(trial_ids) or len(trial_ids) != len(set(trial_ids)):
        raise FoundationEvidenceError("Trial manifest has a missing or duplicate trial_id")
    child_id = str(child.get("child_plan_id") or "")
    if any(row.get("child_plan_id") != child_id for row in rows):
        raise FoundationEvidenceError("Trial manifest contains a cross-child row")
    for row in rows:
        seed_bundle(row)
    return sorted(rows, key=lambda row: row["trial_id"])


def _load_event_chain(root: Path, roots: Mapping[str, Path]) -> list[dict[str, Any]]:
    if root.is_symlink():
        raise FoundationEvidenceError("Attempt ledger root must not be a symlink")
    location = _approved_location(root, roots)
    resolved_root = roots[location["root_id"]] / location["path"]
    paths = sorted(path for path in resolved_root.glob("*.json") if path.is_file())
    if not paths:
        raise FoundationEvidenceError("Attempt ledger has no JSON events")
    events: list[dict[str, Any]] = []
    previous: str | None = None
    seen_ids: set[str] = set()
    program_identity: tuple[str, str] | None = None
    for expected_sequence, path in enumerate(paths, start=1):
        if path.is_symlink():
            raise FoundationEvidenceError(f"Attempt ledger event is symlinked: {path.name}")
        event = dict(load_research_mapping(path))
        if set(event) != EVENT_FIELDS:
            raise FoundationEvidenceError(f"Event {path.name} has the wrong envelope fields")
        if event.get("schema_version") != 1 or event.get("sequence") != expected_sequence:
            raise FoundationEvidenceError(f"Event {path.name} breaks the contiguous sequence")
        if not path.name.startswith(f"{expected_sequence:06d}-"):
            raise FoundationEvidenceError(f"Event {path.name} disagrees with its sequence")
        event_id = str(event.get("event_id") or "")
        if (
            not event_id
            or event_id in seen_ids
            or not str(event.get("actor_id") or "")
            or not str(event.get("subject_id") or "")
            or not _sha256(event.get("subject_fingerprint"))
        ):
            raise FoundationEvidenceError("Attempt ledger has a missing or duplicate event_id")
        seen_ids.add(event_id)
        observed_program = (
            str(event.get("program_id") or ""),
            str(event.get("program_fingerprint") or ""),
        )
        if (
            not observed_program[0]
            or not _sha256(observed_program[1])
            or (program_identity is not None and observed_program != program_identity)
        ):
            raise FoundationEvidenceError("Attempt ledger changes program identity")
        program_identity = observed_program
        if event.get("previous_event_sha256") != previous:
            raise FoundationEvidenceError(f"Event {path.name} breaks the SHA-256 chain")
        event_type = str(event.get("event_type") or "")
        expected_payload = PAYLOAD_FIELDS.get(event_type)
        payload = event.get("payload")
        if expected_payload is None or not isinstance(payload, Mapping):
            raise FoundationEvidenceError(f"Event {path.name} has an unsupported payload")
        if set(payload) != expected_payload:
            raise FoundationEvidenceError(f"Event {path.name} has the wrong payload fields")
        event_hash = file_sha256(path)
        event["event_sha256"] = event_hash
        events.append(event)
        previous = event_hash
    return events


def _accepted_attempts(
    events: Sequence[Mapping[str, Any]],
    child: Mapping[str, Any],
    expected_child_ref: Mapping[str, str],
    rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[Mapping[str, Any]]]]:
    by_id = {str(event["event_id"]): event for event in events}
    frozen_trials = {row["trial_id"]: row for row in rows}
    matching_locks: set[str] = set()
    for event in events:
        if event["event_type"] != "child_locked":
            continue
        child_ref = event["payload"]["child_ref"]
        if (
            all(child_ref.get(name) == value for name, value in expected_child_ref.items())
            and event["payload"].get("study_id") == child["study"]["id"]
        ):
            matching_locks.add(str(event["event_id"]))
    if len(matching_locks) != 1:
        raise FoundationEvidenceError("Expected exactly one ledger lock for the FOUNDATION child")

    starts: dict[str, Mapping[str, Any]] = {}
    start_lock_fingerprints: dict[str, str] = {}
    for event in events:
        if event["event_type"] != "trial_attempt_started":
            continue
        payload = event["payload"]
        trial_id = str(payload["trial_id"])
        lock = _resolve_event_ref(
            payload["child_lock_event"], by_id, before_sequence=int(event["sequence"])
        )
        if trial_id in frozen_trials and str(lock["event_id"]) not in matching_locks:
            raise FoundationEvidenceError(f"Trial {trial_id} is attached to another child lock")
        if str(lock["event_id"]) not in matching_locks:
            continue
        if payload["reservation_id"] != lock["payload"]["reservation_id"]:
            raise FoundationEvidenceError(f"Attempt {payload['attempt_id']} changes reservation")
        attempt_id = str(payload["attempt_id"])
        if not attempt_id or attempt_id in starts:
            raise FoundationEvidenceError("Attempt ledger has a missing or duplicate attempt_id")
        if trial_id not in frozen_trials:
            raise FoundationEvidenceError(f"Attempt {attempt_id} names an unknown trial")
        row = frozen_trials[trial_id]
        if payload["trial_manifest_row_fingerprint"] != trial_row_fingerprint(row):
            raise FoundationEvidenceError(f"Attempt {attempt_id} does not bind its trial row")
        if payload["seed_bundle_fingerprint"] != seed_bundle_fingerprint(row):
            raise FoundationEvidenceError(f"Attempt {attempt_id} does not bind all seed domains")
        locked_config = child["runtime"]["runner"]["config"]["sha256"]
        if payload["runtime_config_fingerprint"] != locked_config:
            raise FoundationEvidenceError(f"Attempt {attempt_id} uses another effective config")
        starts[attempt_id] = event
        start_lock_fingerprints[attempt_id] = str(
            lock["payload"]["lock_receipt_ref"]["content_fingerprint"]
        )
        if not _sha256(start_lock_fingerprints[attempt_id]):
            raise FoundationEvidenceError("Child lock has an invalid content fingerprint")

    closed: dict[str, dict[str, Any]] = {}
    deviations: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        event_type = event["event_type"]
        if event_type in {"trial_attempt_completed", "trial_attempt_failed"}:
            start = _resolve_event_ref(
                event["payload"]["start_event"],
                by_id,
                before_sequence=int(event["sequence"]),
            )
            attempt_id = str(start["payload"].get("attempt_id") or "")
            if attempt_id not in starts:
                continue
            if attempt_id in closed:
                raise FoundationEvidenceError(f"Attempt {attempt_id} has duplicate terminal events")
            closed[attempt_id] = {
                "start": starts[attempt_id],
                "terminal": event,
                "completed": event_type == "trial_attempt_completed",
                "child_lock_fingerprint": start_lock_fingerprints[attempt_id],
            }
        elif event_type == "deviation_recorded":
            payload = event["payload"]
            if payload["category"] not in {"technical", "protocol"} or payload[
                "disposition"
            ] not in {"retry", "exclude_trial", "invalidate_child", "continue"}:
                raise FoundationEvidenceError("Deviation has an invalid category or disposition")
            target = _resolve_event_ref(
                payload["target_event"], by_id, before_sequence=int(event["sequence"])
            )
            deviations[str(target["event_id"])].append(event)
    missing_closes = sorted(set(starts) - set(closed))
    if missing_closes:
        raise FoundationEvidenceError(
            f"Open attempts are not indexable: {', '.join(missing_closes)}"
        )

    by_trial: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for attempt in closed.values():
        by_trial[str(attempt["start"]["payload"]["trial_id"])].append(attempt)
    if set(by_trial) != set(frozen_trials):
        missing = sorted(set(frozen_trials) - set(by_trial))
        raise FoundationEvidenceError(f"Trials without terminal attempts: {', '.join(missing)}")
    for trial_id, trial_attempts in by_trial.items():
        try:
            trial_attempts.sort(key=lambda item: int(item["start"]["payload"]["ordinal"]))
            ordinals = [int(item["start"]["payload"]["ordinal"]) for item in trial_attempts]
        except (TypeError, ValueError) as exc:
            raise FoundationEvidenceError(f"Trial {trial_id} has a non-integer ordinal") from exc
        if ordinals != list(range(1, len(ordinals) + 1)):
            raise FoundationEvidenceError(
                f"Trial {trial_id} has a missing or duplicate retry ordinal"
            )
        completed = [item for item in trial_attempts if item["completed"]]
        if len(completed) > 1 or (completed and completed[0] is not trial_attempts[-1]):
            raise FoundationEvidenceError(f"Trial {trial_id} has ambiguous completed attempts")
        for attempt in trial_attempts[:-1]:
            if attempt["completed"] or not _has_disposition(attempt, deviations, "retry"):
                raise FoundationEvidenceError(f"Trial {trial_id} has an undispositioned retry")
        final = trial_attempts[-1]
        if not final["completed"] and not _has_disposition(final, deviations, "exclude_trial"):
            raise FoundationEvidenceError(
                f"Trial {trial_id} final failure lacks an explicit exclusion"
            )
    return dict(by_trial), dict(deviations)


def _index_trial(
    row: Mapping[str, str],
    attempts: Sequence[Mapping[str, Any]],
    deviations: Mapping[str, Sequence[Mapping[str, Any]]],
    child: Mapping[str, Any],
    roots: Mapping[str, Path],
) -> dict[str, Any]:
    records = [_index_attempt(item, deviations, row, child, roots) for item in attempts]
    final = records[-1]
    return {
        "trial": dict(row),
        "trial_manifest_row_fingerprint": trial_row_fingerprint(row),
        "seed_bundle": seed_bundle(row),
        "seed_bundle_fingerprint": seed_bundle_fingerprint(row),
        "attempts": records,
        "accepted_terminal_attempt_id": final["attempt_id"],
        "accepted_terminal_disposition": final["disposition"],
    }


def _index_attempt(
    attempt: Mapping[str, Any],
    deviations: Mapping[str, Sequence[Mapping[str, Any]]],
    row: Mapping[str, str],
    child: Mapping[str, Any],
    roots: Mapping[str, Path],
) -> dict[str, Any]:
    start = attempt["start"]
    terminal = attempt["terminal"]
    start_payload = start["payload"]
    terminal_payload = terminal["payload"]
    terminal_deviations = deviations.get(str(terminal["event_id"]), ())
    disposition = "completed" if attempt["completed"] else _final_disposition(terminal_deviations)
    refs_name = "output_refs" if attempt["completed"] else "log_refs"
    artifacts = [_index_artifact(ref, roots) for ref in terminal_payload[refs_name]]
    record: dict[str, Any] = {
        "attempt_id": start_payload["attempt_id"],
        "ordinal": start_payload["ordinal"],
        "disposition": disposition,
        "start_event": _event_identity(start),
        "terminal_event": _event_identity(terminal),
        "runtime_config_fingerprint": start_payload["runtime_config_fingerprint"],
        "seed_bundle_fingerprint": start_payload["seed_bundle_fingerprint"],
        "actual_budget": dict(terminal_payload["actual_budget"]),
        refs_name: artifacts,
        "deviations": [_deviation_record(item, roots) for item in terminal_deviations],
    }
    if attempt["completed"]:
        terminal_status = str(terminal_payload["terminal_status"])
        if terminal_status not in child["completion"]["valid_trial_statuses"]:
            raise FoundationEvidenceError(
                f"Attempt {start_payload['attempt_id']} has an invalid terminal status"
            )
        trace_artifacts = [item for item in artifacts if item["kind"] == "directory"]
        trace_artifacts = [
            item
            for item in trace_artifacts
            if (roots[item["root_id"]] / item["path"] / "meta" / "info.json").is_file()
        ]
        if len(trace_artifacts) != 1:
            raise FoundationEvidenceError(
                f"Attempt {start_payload['attempt_id']} must reference exactly one LeRobot trace"
            )
        trace = _verify_trace(trace_artifacts[0], row, child, terminal_status, roots)
        receipt = _index_artifact(terminal_payload["runtime_receipt_ref"], roots)
        _verify_runtime_receipt(
            receipt,
            start_payload,
            terminal_payload,
            str(attempt["child_lock_fingerprint"]),
            roots,
        )
        record.update(
            {
                "terminal_status": terminal_status,
                "runtime_receipt": receipt,
                "trace": trace,
            }
        )
    else:
        record.update(
            {
                "failure_stage": terminal_payload["failure_stage"],
                "error_code": terminal_payload["error_code"],
                "retryable": terminal_payload["retryable"],
            }
        )
    return record


def _verify_trace(
    artifact: Mapping[str, Any],
    row: Mapping[str, str],
    child: Mapping[str, Any],
    terminal_status: str,
    roots: Mapping[str, Path],
) -> dict[str, Any]:
    trace_root = roots[str(artifact["root_id"])] / str(artifact["path"])
    result = validate_lerobot_v3_dataset(trace_root)
    if not result.valid:
        codes = ", ".join(issue.code for issue in result.errors)
        raise FoundationEvidenceError(f"LeRobot trace {row['trial_id']} is invalid: {codes}")
    overlay_path = trace_root / VLA_LENS_OVERLAY_MANIFEST
    refs_path = trace_root / VLA_LENS_OVERLAY_REFERENCES
    if not overlay_path.is_file() or not refs_path.is_file():
        raise FoundationEvidenceError(f"Trace {row['trial_id']} lacks its overlay manifest")
    overlay = load_research_mapping(overlay_path)
    refs = pd.read_parquet(refs_path)
    if (
        set(overlay) != {
            "overlay_schema_version",
            "robot_dataset_format",
            "overlay_root",
            "episodes",
        }
        or overlay.get("overlay_schema_version") != OVERLAY_SCHEMA_VERSION
        or overlay.get("robot_dataset_format") != "lerobot_v3"
        or overlay.get("overlay_root") != "vla_lens"
        or overlay.get("episodes") != len(refs)
        or len(refs) != 1
        or not {"episode_index", "trace_id", "overlay_path"} <= set(refs.columns)
    ):
        raise FoundationEvidenceError(f"Trace {row['trial_id']} has a mismatched overlay manifest")
    bundle_relative = _safe_relative(str(refs.iloc[0]["overlay_path"]), "overlay_path")
    bundle_path = trace_root / bundle_relative
    manifest_path = bundle_path / TraceBundle.MANIFEST
    manifest = load_research_mapping(manifest_path)
    metadata = manifest.get("metadata")
    runtime = metadata.get("runtime_audit") if isinstance(metadata, Mapping) else None
    if not isinstance(runtime, Mapping):
        raise FoundationEvidenceError(f"Trace {row['trial_id']} lacks runtime audit metadata")
    expected_runtime = {
        "trial_id": row["trial_id"],
        "child_plan_id": row["child_plan_id"],
        "canonical_family_id": row["canonical_family_id"],
        "pool": row["pool"],
        "replicate_id": row["replicate_id"],
        "model_revision": child["runtime"]["model"]["revision"],
        "snapshot_manifest_sha256": child["runtime"]["model"]["snapshot_manifest_sha256"],
    }
    mismatched = [
        name
        for name, value in expected_runtime.items()
        if str(runtime.get(name)) != str(value)
    ]
    expected_seeds = {domain: seed_bundle(row)[domain]["value"] for domain in SEED_DOMAINS}
    if runtime.get("seed_identities") != expected_seeds:
        mismatched.append("seed_identities")
    environment = child["runtime"]["environment"]
    for name in (
        "camera_config_sha256",
        "controller_config_sha256",
        "preprocessor_config_sha256",
        "postprocessor_config_sha256",
    ):
        if runtime.get(name) != environment[name]:
            mismatched.append(name)
    if (
        str(manifest.get("task_id")) != str(row["task_id"])
        or manifest.get("model_id") != child["runtime"]["model"]["repo_id"]
        or not isinstance(metadata, Mapping)
        or metadata.get("dataset_id") != row["dataset_id"]
        or str(refs.iloc[0]["trace_id"]) != str(manifest.get("trace_id"))
        or str(metadata.get("lerobot_episode_index"))
        != str(refs.iloc[0]["episode_index"])
    ):
        mismatched.append("task_or_dataset")
    expected_outcome = {
        "rollout_success": "success",
        "rollout_behavior_failure": "failure",
    }[terminal_status]
    if manifest.get("outcome") != expected_outcome:
        mismatched.append("simulator_success")
    if mismatched:
        raise FoundationEvidenceError(
            f"Trace {row['trial_id']} identity mismatch: {', '.join(sorted(set(mismatched)))}"
        )
    capture_capabilities = metadata.get("capture_capabilities")
    capability = (
        capture_capabilities.get("simulator_contact_telemetry")
        if isinstance(capture_capabilities, Mapping)
        else None
    )
    required_capability = {
        "available",
        "verdict",
        "sample_phase",
        "sample_phase_semantics",
        "sampling_limitation",
        "distance_semantics",
    }
    if not isinstance(capability, Mapping) or not required_capability <= set(capability):
        raise FoundationEvidenceError(f"Trace {row['trial_id']} loses contact capability limits")
    scene_path = bundle_path / TraceBundle.SCENE_STATE
    evaluation_path = bundle_path / TraceBundle.EVALUATION
    array_index_path = bundle_path / TraceBundle.ARRAY_INDEX
    if (
        not scene_path.is_file()
        or not evaluation_path.is_file()
        or not array_index_path.is_file()
    ):
        raise FoundationEvidenceError(f"Trace {row['trial_id']} lacks physical evidence tables")
    array_index = pd.read_parquet(array_index_path)
    if not {"name", "relative_path"} <= set(array_index.columns):
        raise FoundationEvidenceError(f"Trace {row['trial_id']} has an invalid array index")
    position_rows = array_index.loc[array_index["name"].astype(str) == "scene_object_pos"]
    if len(position_rows) != 1:
        raise FoundationEvidenceError(
            f"Trace {row['trial_id']} must contain one object-position array"
        )
    position_relative = _safe_relative(
        str(position_rows.iloc[0]["relative_path"]), "object position array path"
    )
    position_path = bundle_path / position_relative
    _reject_symlinks(position_path, trace_root)
    if not position_path.exists():
        raise FoundationEvidenceError(f"Trace {row['trial_id']} lacks object-position bytes")
    return {
        "root": dict(artifact),
        "trace_sha256": artifact["sha256"],
        "overlay_manifest": _relative_file_record(overlay_path, trace_root),
        "overlay_references": _relative_file_record(refs_path, trace_root),
        "episode_manifest": _relative_file_record(manifest_path, trace_root),
        "checkpoint_revision": runtime["model_revision"],
        "effective_config_sha256": child["runtime"]["runner"]["config"]["sha256"],
        "measurements": {
            "simulator_success": {
                "value": manifest["outcome"] == "success",
                "source": _relative_file_record(manifest_path, trace_root),
                "field": "outcome",
            },
            "mujoco_contact": {
                "source": _relative_file_record(scene_path, trace_root),
                "predicate": "context_kind == 'mujoco_contact' and physical_contact == true",
            },
            "positive_gap_proximity": {
                "source": _relative_file_record(scene_path, trace_root),
                "predicate": (
                    "context_kind == 'mujoco_contact' and positive_gap_proximity == true"
                ),
            },
            "object_motion": {
                "object_metadata": _relative_file_record(scene_path, trace_root),
                "array_index": _relative_file_record(array_index_path, trace_root),
                "positions": _relative_path_record(position_path, trace_root),
                "array_name": "scene_object_pos",
            },
            "evaluation": {"source": _relative_file_record(evaluation_path, trace_root)},
        },
        "contact_capability": dict(capability),
    }


def _verify_runtime_receipt(
    artifact: Mapping[str, Any],
    start: Mapping[str, Any],
    terminal: Mapping[str, Any],
    child_lock_fingerprint: str,
    roots: Mapping[str, Path],
) -> None:
    if artifact["kind"] != "file":
        raise FoundationEvidenceError("Runtime receipt reference must name a file")
    receipt = load_research_mapping(roots[artifact["root_id"]] / artifact["path"])
    fields = {
        "schema_version",
        "kind",
        "attempt_id",
        "trial_id",
        "child_lock_fingerprint",
        "runtime_config_fingerprint",
        "seed_bundle_fingerprint",
        "runtime_check_status",
        "terminal_status",
        "output_refs",
        "created_utc",
    }
    expected = {
        "schema_version": 1,
        "kind": "vla_lens.trial_runtime_receipt",
        "attempt_id": start["attempt_id"],
        "trial_id": start["trial_id"],
        "child_lock_fingerprint": child_lock_fingerprint,
        "runtime_config_fingerprint": start["runtime_config_fingerprint"],
        "seed_bundle_fingerprint": start["seed_bundle_fingerprint"],
        "runtime_check_status": "pass",
        "terminal_status": terminal["terminal_status"],
        "output_refs": terminal["output_refs"],
    }
    if (
        set(receipt) != fields
        or any(receipt.get(name) != value for name, value in expected.items())
        or not str(receipt.get("created_utc") or "")
    ):
        raise FoundationEvidenceError(
            f"Attempt {start['attempt_id']} has a mismatched runtime receipt"
        )


def _index_artifact(reference: Mapping[str, Any], roots: Mapping[str, Path]) -> dict[str, Any]:
    required = {"id", "type", "root_id", "path", "sha256"}
    if not isinstance(reference, Mapping) or set(reference) != required:
        raise FoundationEvidenceError("Artifact reference has the wrong fields")
    root_id = str(reference["root_id"])
    if root_id not in roots:
        raise FoundationEvidenceError(f"Artifact uses unapproved root_id {root_id!r}")
    relative = _safe_relative(str(reference["path"]), "artifact path")
    target = roots[root_id] / relative
    _reject_symlinks(target, roots[root_id])
    if target.is_file():
        observed = {
            "kind": "file",
            "size_bytes": target.stat().st_size,
            "sha256": file_sha256(target),
            "sha256_kind": "file_bytes",
        }
    elif target.is_dir():
        observed = _directory_record(target)
    else:
        raise FoundationEvidenceError(f"Artifact is missing: {root_id}:{relative}")
    if observed["sha256"] != reference["sha256"]:
        raise FoundationEvidenceError(f"Artifact hash mismatch: {root_id}:{relative}")
    return {
        "id": reference["id"],
        "type": reference["type"],
        "root_id": root_id,
        "path": relative.as_posix(),
        **observed,
    }


def _directory_record(path: Path) -> dict[str, Any]:
    files = []
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise FoundationEvidenceError(f"Evidence directory contains a symlink: {child}")
        if child.is_file():
            files.append(
                {
                    "path": child.relative_to(path).as_posix(),
                    "size_bytes": child.stat().st_size,
                    "sha256": file_sha256(child),
                }
            )
    tree_bytes = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("ascii")
    return {
        "kind": "directory",
        "size_bytes": sum(item["size_bytes"] for item in files),
        "sha256": f"sha256:{hashlib.sha256(tree_bytes).hexdigest()}",
        "sha256_kind": "directory_tree_v1",
        "files": files,
    }


def _relative_file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "sha256_kind": "file_bytes",
    }


def _relative_path_record(path: Path, root: Path) -> dict[str, Any]:
    if path.is_file():
        observed = {
            "kind": "file",
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "sha256_kind": "file_bytes",
        }
    elif path.is_dir():
        observed = _directory_record(path)
    else:
        raise FoundationEvidenceError(f"Evidence path is missing: {path}")
    return {"path": path.relative_to(root).as_posix(), **observed}


def _resolve_event_ref(
    reference: Mapping[str, Any],
    by_id: Mapping[str, Mapping[str, Any]],
    *,
    before_sequence: int | None = None,
) -> Mapping[str, Any]:
    if not isinstance(reference, Mapping) or set(reference) != EVENT_REF_FIELDS:
        raise FoundationEvidenceError("Event reference has the wrong fields")
    event = by_id.get(str(reference["event_id"]))
    if (
        event is None
        or event["sequence"] != reference["sequence"]
        or event["event_sha256"] != reference["event_sha256"]
        or (before_sequence is not None and int(event["sequence"]) >= before_sequence)
    ):
        raise FoundationEvidenceError("Event reference does not resolve exactly")
    return event


def _has_disposition(
    attempt: Mapping[str, Any],
    deviations: Mapping[str, Sequence[Mapping[str, Any]]],
    disposition: str,
) -> bool:
    terminal_id = str(attempt["terminal"]["event_id"])
    return any(
        item["payload"].get("disposition") == disposition
        for item in deviations.get(terminal_id, ())
    )


def _final_disposition(deviations: Sequence[Mapping[str, Any]]) -> str:
    dispositions = [str(item["payload"]["disposition"]) for item in deviations]
    return "excluded" if "exclude_trial" in dispositions else "failed"


def _deviation_record(
    event: Mapping[str, Any], roots: Mapping[str, Path]
) -> dict[str, Any]:
    payload = event["payload"]
    return {
        "event": _event_identity(event),
        "category": payload["category"],
        "disposition": payload["disposition"],
        "reason": payload["reason"],
        "evidence_refs": [_index_artifact(item, roots) for item in payload["evidence_refs"]],
    }


def _event_identity(event: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sequence": event["sequence"],
        "event_id": event["event_id"],
        "event_sha256": event["event_sha256"],
    }


def _locked_runtime(child: Mapping[str, Any]) -> dict[str, Any]:
    runtime = child["runtime"]
    return {
        "checkpoint": dict(runtime["model"]),
        "environment": {
            "backend": runtime["environment"]["backend"],
            "package_receipt": dict(runtime["environment"]["package_receipt"]),
            **{
                name: runtime["environment"][name]
                for name in (
                    "camera_config_sha256",
                    "controller_config_sha256",
                    "preprocessor_config_sha256",
                    "postprocessor_config_sha256",
                )
            },
        },
        "code": dict(runtime["code"]),
        "runner": {
            "entrypoint": runtime["runner"]["entrypoint"],
            "argv": list(runtime["runner"]["argv"]),
            "effective_config_sha256": runtime["runner"]["config"]["sha256"],
        },
        "contact_capability_lock": child["protocol_lock"]["locked_choices"][
            "simulator_contact_telemetry_capability_status"
        ],
    }


def _approved_roots(
    repo: Path, external_roots: Mapping[str, str | Path] | None
) -> dict[str, Path]:
    roots = {"repo": repo}
    for root_id, raw_path in (external_roots or {}).items():
        if not root_id or root_id == "repo" or "/" in root_id or "\\" in root_id:
            raise FoundationEvidenceError(f"Invalid external root ID {root_id!r}")
        path = Path(raw_path)
        if _path_has_symlink(path) or not path.is_dir():
            raise FoundationEvidenceError(f"External root {root_id!r} is missing or symlinked")
        roots[root_id] = path.resolve()
    return roots


def _approved_location(path: Path, roots: Mapping[str, Path]) -> dict[str, str]:
    resolved = path.resolve()
    matches: list[tuple[int, str, Path]] = []
    for root_id, root in roots.items():
        try:
            relative = resolved.relative_to(root)
        except ValueError:
            continue
        matches.append((len(root.parts), root_id, relative))
    if not matches:
        raise FoundationEvidenceError(
            f"Path is outside the repo and approved external roots: {path}"
        )
    _, root_id, relative = max(matches)
    return {"root_id": root_id, "path": relative.as_posix() or "."}


def _repo_file(repo: Path, path: str | Path, label: str) -> Path:
    candidate = Path(path)
    candidate = candidate if candidate.is_absolute() else repo / candidate
    if _path_has_symlink(candidate):
        raise FoundationEvidenceError(f"{label} must not be symlinked")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(repo)
    except ValueError as exc:
        raise FoundationEvidenceError(f"{label} must be repo-relative") from exc
    _reject_symlinks(resolved, repo)
    if not resolved.is_file():
        raise FoundationEvidenceError(f"{label} is missing: {resolved}")
    return resolved


def _repo_destination(repo: Path, path: str | Path) -> Path:
    candidate = Path(path)
    candidate = candidate if candidate.is_absolute() else repo / candidate
    if _path_has_symlink(candidate):
        raise FoundationEvidenceError("Evidence index destination must not be symlinked")
    parent = candidate.parent.resolve()
    try:
        parent.relative_to(repo)
    except ValueError as exc:
        raise FoundationEvidenceError("Evidence index destination must be inside the repo") from exc
    _reject_symlinks(parent, repo)
    return parent / candidate.name


def _safe_relative(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or value == "." or ".." in path.parts or "\\" in value:
        raise FoundationEvidenceError(f"{label} must be a normalized relative path")
    if path.as_posix() != value:
        raise FoundationEvidenceError(f"{label} must be normalized")
    return path


def _reject_symlinks(path: Path, root: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise FoundationEvidenceError(f"Path escapes approved root: {path}") from exc
    current = root
    if root.is_symlink():
        raise FoundationEvidenceError(f"Approved root is symlinked: {root}")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise FoundationEvidenceError(f"Evidence path contains a symlink: {current}")


def _path_has_symlink(path: Path) -> bool:
    absolute = path.absolute()
    return any(candidate.is_symlink() for candidate in (absolute, *absolute.parents))


def _sha256(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == len(SHA256_ZERO)
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


__all__ = [
    "EXPECTED_TRIAL_COUNT",
    "FOUNDATION_TRIAL_FIELDS",
    "FoundationEvidenceError",
    "create_foundation_evidence_index",
    "directory_sha256",
    "seed_bundle",
    "seed_bundle_fingerprint",
    "trial_row_fingerprint",
]
