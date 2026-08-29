"""Immutable evidence indexing for the locked RQ-024 FOUNDATION trial matrix.

This module is normal-environment analysis code. It validates already accepted
campaign state and existing capture bytes; it never executes a model, simulator,
or campaign transition.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.capture.lerobot_v3 import (
    VLA_LENS_OVERLAY_MANIFEST,
    VLA_LENS_OVERLAY_REFERENCES,
    validate_lerobot_v3_dataset,
)
from vla_lens.research_events import verify_research_event_ledger
from vla_lens.research_io import (
    canonical_research_fingerprint,
    file_sha256,
    load_research_mapping,
    write_bytes_create_only,
)
from vla_lens.research_plan import load_research_plan, research_plan_fingerprint
from vla_lens.research_state import CampaignState
from vla_lens.traces import TraceBundle

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
ARTIFACT_REF_FIELDS = frozenset({"id", "type", "root_id", "path", "sha256"})
MEASUREMENT_RULE = (
    "Eligibility uses simulator success. Contact manifolds, object motion, and EEF "
    "distance remain separate measurements."
)


class FoundationEvidenceError(ValueError):
    """A stable, machine-readable evidence validation failure."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True, slots=True)
class FoundationContract:
    child: Mapping[str, Any]
    child_plan_id: str
    child_fingerprint: str
    trial_manifest_sha256: str
    trial_rows: tuple[Mapping[str, str], ...]
    trials_by_id: Mapping[str, Mapping[str, str]]
    checkpoint_revision: str
    runtime_config_sha256: str
    runtime_config_path: str
    runtime_config_fingerprint: str
    runtime_contract_sha256: str
    runtime_contract_path: str
    runtime_contract_fingerprint: str


@dataclass(frozen=True, slots=True)
class VerifiedFoundationLedger:
    program_fingerprint: str
    event_count: int
    ledger_tip: str | None
    state: CampaignState
    event_documents: tuple[tuple[Mapping[str, Any], str], ...]


def trial_row_fingerprint(row: Mapping[str, str]) -> str:
    return canonical_research_fingerprint(dict(row))


def seed_bundle(row: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    bundle: dict[str, dict[str, Any]] = {}
    for domain in SEED_DOMAINS:
        identity = str(row.get(f"{domain}_seed_identity") or "")
        value = str(row.get(f"{domain}_seed") or "")
        if not _is_sha256(identity):
            _fail("seed_binding_mismatch", f"invalid {domain} seed identity")
        try:
            seed = int(value)
        except ValueError:
            _fail("seed_binding_mismatch", f"invalid {domain} seed")
        bundle[domain] = {"identity": identity, "seed": seed}
    return bundle


def seed_bundle_fingerprint(row: Mapping[str, str]) -> str:
    return canonical_research_fingerprint(seed_bundle(row))


def _expected_trace_id(row: Mapping[str, str]) -> str:
    return (
        f"pi05_{row['capture_profile']}_{row['benchmark']}"
        f"_task{row['task_id']}_seed{row['seed']}"
    )


def directory_sha256(path: str | Path) -> str:
    """Return the v1 tree hash used for directory artifact references."""

    files = _directory_files(Path(path))
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("ascii")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def load_foundation_contract(
    *,
    child_path: str | Path,
    trial_manifest_path: str | Path,
    repo_root: str | Path,
) -> FoundationContract:
    """Load and verify the immutable child, runtime contract, and 72 trial rows."""

    root = Path(repo_root).resolve()
    child_file = _confined_file(child_path, root, "child")
    trials_file = Path(trial_manifest_path)
    if not trials_file.is_absolute():
        trials_file = root / trials_file
    if _path_has_symlink(trials_file) or not trials_file.is_file():
        _fail("invalid_trial_manifest", "trial CSV is missing or symlinked")
    trials_file = trials_file.resolve()
    child = load_research_mapping(child_file)
    rows = _load_trials(trials_file, child)
    config_ref = child.get("runtime", {}).get("runner", {}).get("config", {})
    config_path_value = config_ref.get("path") if isinstance(config_ref, Mapping) else None
    if config_path_value:
        config_file = _confined_file(config_path_value, root, "runtime config")
        config_payload = load_research_mapping(config_file)
        runtime_config_sha256 = file_sha256(config_file)
        runtime_config_path = config_file.relative_to(root).as_posix()
        runtime_config_fingerprint = canonical_research_fingerprint(config_payload)
    else:
        runtime_config_sha256 = str(config_ref.get("sha256") or "")
        runtime_config_path = ""
        runtime_config_fingerprint = runtime_config_sha256
    if not _is_sha256(runtime_config_sha256) or not _is_sha256(runtime_config_fingerprint):
        _fail("runtime_binding_mismatch", "locked runtime config has no usable identity")
    runtime_path = child_file.parent / "runtime_contract.json"
    runtime_fingerprint = _validate_runtime_contract(child_file.parent, child)
    return FoundationContract(
        child=child,
        child_plan_id=str(child["child_plan_id"]),
        child_fingerprint=canonical_research_fingerprint(child),
        trial_manifest_sha256=file_sha256(trials_file),
        trial_rows=tuple(rows),
        trials_by_id={row["trial_id"]: row for row in rows},
        checkpoint_revision=str(child["runtime"]["model"]["revision"]),
        runtime_config_sha256=runtime_config_sha256,
        runtime_config_path=runtime_config_path,
        runtime_config_fingerprint=runtime_config_fingerprint,
        runtime_contract_sha256=file_sha256(runtime_path),
        runtime_contract_path=runtime_path.relative_to(root).as_posix(),
        runtime_contract_fingerprint=runtime_fingerprint,
    )


def load_verified_foundation_ledger(
    *,
    program_path: str | Path,
    event_root: str | Path,
    repo_root: str | Path,
) -> VerifiedFoundationLedger:
    """Return only events and state accepted by the canonical campaign reducer."""

    root = Path(repo_root).resolve()
    program_file = _confined_file(program_path, root, "program")
    events_path = Path(event_root)
    if _path_has_symlink(events_path) or not _is_within(events_path.resolve(), root):
        _fail("invalid_event_ledger", "event root is outside the approved repo root")
    program = load_research_plan(program_file)
    check = verify_research_event_ledger(
        events_path,
        program,
        repo_root=root,
        verify_artifacts=False,
    )
    if not check.valid:
        _fail(
            "invalid_event_ledger",
            ", ".join(issue.code for issue in check.issues) or "campaign ledger is invalid",
        )
    documents = tuple(
        (load_research_mapping(path), file_sha256(path))
        for path in sorted(events_path.glob("*.json"))
        if path.is_file()
    )
    return VerifiedFoundationLedger(
        program_fingerprint=research_plan_fingerprint(program),
        event_count=check.event_count,
        ledger_tip=check.last_event_sha256,
        state=check.state,
        event_documents=documents,
    )


def build_foundation_evidence_index(
    *,
    child_path: str | Path,
    trial_manifest_path: str | Path,
    campaign_state: CampaignState,
    accepted_events: Sequence[tuple[Mapping[str, Any], str]],
    artifact_roots: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Build a deterministic index from reducer-accepted state and immutable bytes."""

    roots = _approved_roots(artifact_roots)
    repo_root = roots.get("repo")
    if repo_root is None:
        _fail("unapproved_artifact_root", "artifact_roots must include the repo root")
    contract = load_foundation_contract(
        child_path=child_path,
        trial_manifest_path=trial_manifest_path,
        repo_root=repo_root,
    )
    child_file = _confined_file(child_path, repo_root, "child")
    trials_file = _confined_file(trial_manifest_path, repo_root, "trial manifest")
    child = contract.child
    child_fingerprint = contract.child_fingerprint
    program = _validate_program_binding(child, repo_root)
    rows = contract.trial_rows
    runtime_config_fingerprint = contract.runtime_config_fingerprint
    runtime_contract_fingerprint = contract.runtime_contract_fingerprint
    lock = _validate_campaign_state(campaign_state, child, child_fingerprint)
    exclusions = _accepted_exclusions(
        accepted_events,
        campaign_state,
        program_id=str(program["program_id"]),
        program_fingerprint=research_plan_fingerprint(program),
    )
    artifacts = _ArtifactInventory(roots)
    dataset_cache: dict[tuple[str, str], dict[str, Any]] = {}

    attempts_by_trial: dict[str, list[Mapping[str, Any]]] = {}
    for attempt in campaign_state.closed_attempts.values():
        started = attempt.get("started")
        if not isinstance(started, Mapping):
            _fail("invalid_terminal_attempt", "closed attempt lacks its accepted start")
        attempts_by_trial.setdefault(str(started.get("trial_id") or ""), []).append(attempt)

    trial_records = []
    for row in rows:
        trial_id = row["trial_id"]
        trial_attempts = attempts_by_trial.pop(trial_id, [])
        if not trial_attempts:
            _fail("missing_terminal_attempt", f"{trial_id} has no terminal attempt")
        trial_records.append(
            _build_trial(
                row,
                trial_attempts,
                exclusions,
                lock,
                child,
                child_fingerprint,
                runtime_config_fingerprint,
                runtime_contract_fingerprint,
                artifacts,
                dataset_cache,
            )
        )
    if attempts_by_trial:
        _fail("cross_child_attempt", "campaign state contains attempts outside the trial matrix")

    return {
        "schema_version": 1,
        "kind": "rq024.foundation_external_evidence_index",
        "child": {
            "child_plan_id": child["child_plan_id"],
            "fingerprint": child_fingerprint,
            "path": child_file.relative_to(repo_root).as_posix(),
            "sha256": file_sha256(child_file),
        },
        "trial_manifest": {
            "id": child["trials"]["manifest"]["id"],
            "path": trials_file.relative_to(repo_root).as_posix(),
            "sha256": file_sha256(trials_file),
            "row_count": len(rows),
        },
        "program": {
            "program_id": program["program_id"],
            "fingerprint": research_plan_fingerprint(program),
        },
        "runtime": {
            "config": {
                "root_id": "repo",
                "path": contract.runtime_config_path,
                "sha256": contract.runtime_config_sha256,
                "fingerprint": contract.runtime_config_fingerprint,
            },
            "contract": {
                "root_id": "repo",
                "path": contract.runtime_contract_path,
                "size": (repo_root / contract.runtime_contract_path).stat().st_size,
                "sha256": contract.runtime_contract_sha256,
                "fingerprint": contract.runtime_contract_fingerprint,
            }
        },
        "runtime_config_fingerprint": runtime_config_fingerprint,
        "runtime_contract_fingerprint": contract.runtime_contract_fingerprint,
        "measurement_contract": {"foundation_rule": MEASUREMENT_RULE},
        "trials": trial_records,
        "artifacts": artifacts.records(),
    }


def write_foundation_evidence_index(path: str | Path, manifest: Mapping[str, Any]) -> bool:
    """Create the canonical JSON index, accepting only byte-identical repeats."""

    content = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return write_bytes_create_only(path, content)


def create_foundation_evidence_index(
    *,
    repo_root: str | Path,
    child_path: str | Path,
    trial_manifest_path: str | Path,
    event_root: str | Path,
    output_path: str | Path,
    external_roots: Mapping[str, str | Path] | None = None,
) -> tuple[Mapping[str, Any], bool]:
    """Create an index from an already isolated attempt-ledger export.

    The canonical campaign path uses :func:`load_verified_foundation_ledger` and
    :func:`build_foundation_evidence_index`. This compatibility entry point
    indexes standalone create-only attempt exports produced before that boundary
    was exposed.
    """

    root = Path(repo_root).resolve()
    child_file = _confined_file(child_path, root, "child")
    trials_file = _confined_file(trial_manifest_path, root, "trial manifest")
    child = load_research_mapping(child_file)
    rows = _load_trials(trials_file, child)
    roots = _approved_roots({"repo": root, **(external_roots or {})})
    events = _load_exported_attempt_events(event_root, roots)
    records = _build_exported_trials(events, rows, child, roots)
    payload = {
        "schema_version": 1,
        "kind": "vla_lens.rq024_foundation_external_evidence_index",
        "child": {
            "child_plan_id": child["child_plan_id"],
            "fingerprint": canonical_research_fingerprint(child),
        },
        "trial_manifest": {
            "id": child["trials"]["manifest"]["id"],
            "sha256": file_sha256(trials_file),
            "row_count": len(rows),
        },
        "trial_count": len(records),
        "trials": records,
    }
    destination = Path(output_path)
    if not destination.is_absolute():
        destination = root / destination
    if not _is_within(destination.parent.resolve(), root) or _path_has_symlink(destination):
        _fail("unsafe_artifact_path", "evidence index destination is unsafe")
    return payload, write_foundation_evidence_index(destination, payload)


def _load_exported_attempt_events(
    event_root: str | Path, roots: Mapping[str, Path]
) -> list[dict[str, Any]]:
    path = Path(event_root)
    if not path.is_absolute():
        path = roots["repo"] / path
    if _path_has_symlink(path) or not _is_within(path.resolve(), roots["repo"]):
        _fail("unsafe_artifact_path", "attempt event root is unsafe")
    documents = []
    previous = None
    for sequence, event_path in enumerate(sorted(path.glob("*.json")), start=1):
        event = dict(load_research_mapping(event_path))
        if event.get("sequence") != sequence or event.get("previous_event_sha256") != previous:
            _fail("invalid_event_ledger", "exported attempt event chain is invalid")
        event["event_sha256"] = file_sha256(event_path)
        documents.append(event)
        previous = event["event_sha256"]
    if not documents:
        _fail("invalid_event_ledger", "exported attempt ledger is empty")
    return documents


def _build_exported_trials(
    events: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, str]],
    child: Mapping[str, Any],
    roots: Mapping[str, Path],
) -> list[dict[str, Any]]:
    by_id = {str(event["event_id"]): event for event in events}
    lock_ids = {
        str(event["event_id"])
        for event in events
        if event.get("event_type") == "child_locked"
        and event["payload"]["child_ref"].get("id") == child["child_plan_id"]
        and event["payload"]["child_ref"].get("content_fingerprint")
        == canonical_research_fingerprint(child)
    }
    if len(lock_ids) != 1:
        _fail("cross_child_attempt", "export must contain one matching child lock")
    starts = {}
    terminals = {}
    deviations: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        payload = event["payload"]
        if event["event_type"] == "trial_attempt_started":
            lock_ref = payload["child_lock_event"]
            if lock_ref.get("event_id") in lock_ids:
                starts[str(payload["attempt_id"])] = event
        elif event["event_type"] in {"trial_attempt_completed", "trial_attempt_failed"}:
            start = by_id.get(str(payload["start_event"]["event_id"]))
            if start is not None and str(start["payload"].get("attempt_id")) in starts:
                terminals[str(start["payload"]["attempt_id"])] = event
        elif event["event_type"] == "deviation_recorded":
            deviations.setdefault(str(payload["target_event"]["event_id"]), []).append(event)
    by_trial: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = {}
    for attempt_id, start in starts.items():
        terminal = terminals.get(attempt_id)
        if terminal is None:
            _fail("missing_terminal_attempt", f"{attempt_id} is still open")
        by_trial.setdefault(str(start["payload"]["trial_id"]), []).append((start, terminal))
    records = []
    for row in rows:
        attempts = sorted(
            by_trial.get(row["trial_id"], ()), key=lambda item: int(item[0]["payload"]["ordinal"])
        )
        if not attempts:
            _fail("missing_terminal_attempt", f"{row['trial_id']} has no terminal")
        completed = [
            item for item in attempts if item[1]["event_type"] == "trial_attempt_completed"
        ]
        final = attempts[-1]
        final_deviations = deviations.get(str(final[1]["event_id"]), ())
        if completed:
            accepted = completed[-1]
            accepted_disposition = "completed"
        elif any(item["payload"]["disposition"] == "exclude_trial" for item in final_deviations):
            accepted = final
            accepted_disposition = "excluded"
        else:
            _fail("missing_terminal_attempt", f"{row['trial_id']} has no accepted terminal")
        attempt_records = []
        for start, terminal in attempts:
            start_payload = start["payload"]
            if start_payload["trial_manifest_row_fingerprint"] != trial_row_fingerprint(row):
                _fail("trial_binding_mismatch", "exported attempt uses another trial row")
            if start_payload["seed_bundle_fingerprint"] != seed_bundle_fingerprint(row):
                _fail("seed_binding_mismatch", "exported attempt uses another seed bundle")
            completed_attempt = terminal["event_type"] == "trial_attempt_completed"
            disposition = "completed" if completed_attempt else "failed"
            if terminal is final[1] and accepted_disposition == "excluded":
                disposition = "excluded"
            attempt_record: dict[str, Any] = {
                "attempt_id": start_payload["attempt_id"],
                "ordinal": start_payload["ordinal"],
                "disposition": disposition,
            }
            if completed_attempt:
                attempt_record["trace"] = _index_exported_trace(
                    terminal["payload"]["output_refs"], row, child, roots
                )
            attempt_records.append(attempt_record)
        records.append(
            {
                "trial": dict(row),
                "attempts": attempt_records,
                "accepted_terminal_attempt_id": accepted[0]["payload"]["attempt_id"],
                "accepted_terminal_disposition": accepted_disposition,
            }
        )
    return records


def _index_exported_trace(
    references: Sequence[Mapping[str, Any]],
    row: Mapping[str, str],
    child: Mapping[str, Any],
    roots: Mapping[str, Path],
) -> dict[str, Any]:
    datasets = []
    for reference in references:
        root_id = str(reference.get("root_id") or "")
        if root_id not in roots:
            _fail("unapproved_artifact_root", f"unapproved root_id {root_id!r}")
        relative = _safe_relative(str(reference.get("path") or ""), "artifact path")
        target = roots[root_id] / relative
        observed = directory_sha256(target) if target.is_dir() else file_sha256(target)
        if observed != reference.get("sha256"):
            _fail("artifact_hash_mismatch", "Artifact hash mismatch")
        if target.is_dir() and (target / "meta/info.json").is_file():
            datasets.append(target)
    if len(datasets) != 1:
        _fail("trace_binding_mismatch", "exported attempt must identify one trace root")
    trace_root = datasets[0]
    result = validate_lerobot_v3_dataset(trace_root)
    if not result.valid:
        _fail("invalid_lerobot_trace", "exported LeRobot trace is invalid")
    refs = pd.read_parquet(trace_root / VLA_LENS_OVERLAY_REFERENCES)
    if len(refs) != 1:
        _fail("trace_binding_mismatch", "exported trace root is not single-episode")
    manifest_path = trace_root / str(refs.iloc[0]["overlay_path"]) / "manifest.json"
    manifest = load_research_mapping(manifest_path)
    metadata = manifest.get("metadata")
    runtime = metadata.get("runtime_audit") if isinstance(metadata, Mapping) else None
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("trial_id") != row["trial_id"]
        or runtime.get("child_plan_id") != row["child_plan_id"]
        or str(manifest.get("task_id")) != row["task_id"]
    ):
        _fail("trace_binding_mismatch", "exported trace metadata differs from its trial")
    capability = metadata.get("capture_capabilities", {}).get("simulator_contact_telemetry")
    return {
        "measurements": {
            "simulator_success": manifest.get("outcome") == "success",
            "mujoco_contact": {"source": "tables/scene_state.parquet"},
            "positive_gap_proximity": {"source": "tables/scene_state.parquet"},
            "object_motion": {"source": "scene_object_pos"},
            "evaluation": {"source": "tables/evaluation.parquet"},
        },
        "contact_capability": dict(capability or {}),
        "trace_sha256": directory_sha256(trace_root),
    }


def _load_trials(path: Path, child: Mapping[str, Any]) -> list[dict[str, str]]:
    if child.get("study", {}).get("id") != "FOUNDATION":
        _fail("child_binding_mismatch", "child is not the FOUNDATION study")
    if child.get("trials", {}).get("expected_count") != EXPECTED_TRIAL_COUNT:
        _fail("invalid_trial_manifest", "child does not lock exactly 72 trials")
    if child.get("trials", {}).get("manifest", {}).get("sha256") != file_sha256(path):
        _fail("invalid_trial_manifest", "trial CSV differs from the child lock")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FOUNDATION_TRIAL_FIELDS:
            _fail("invalid_trial_manifest", "trial CSV header differs from the locked schema")
        rows = [dict(row) for row in reader]
    if len(rows) != EXPECTED_TRIAL_COUNT:
        _fail("invalid_trial_manifest", "trial CSV does not contain exactly 72 rows")
    ids = [row["trial_id"] for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        _fail("invalid_trial_manifest", "missing or duplicate trial_id")
    if any(row["child_plan_id"] != child["child_plan_id"] for row in rows):
        _fail("cross_child_attempt", "trial CSV contains a cross-child row")
    for row in rows:
        seed_bundle(row)
    return rows


def _validate_program_binding(
    child: Mapping[str, Any], repo_root: Path
) -> Mapping[str, Any]:
    declared = child.get("program")
    if not isinstance(declared, Mapping):
        _fail("program_binding_mismatch", "child lacks a parent program binding")
    relative = _safe_relative(str(declared.get("path") or ""), "program path")
    program_path = _confined_file(repo_root / relative, repo_root, "program")
    program = load_research_plan(program_path)
    if (
        program.get("program_id") != declared.get("program_id")
        or research_plan_fingerprint(program) != declared.get("fingerprint")
    ):
        _fail("program_binding_mismatch", "child and locked parent program disagree")
    return program


def _validate_runtime_contract(directory: Path, child: Mapping[str, Any]) -> str:
    path = directory / "runtime_contract.json"
    if not path.is_file() or path.is_symlink():
        _fail("runtime_binding_mismatch", "locked runtime contract is missing")
    contract = load_research_mapping(path)
    runtime = child["runtime"]
    environment = runtime["environment"]
    expected = {
        "model_id": runtime["model"]["repo_id"],
        "model_revision": runtime["model"]["revision"],
        "camera_config_sha256": environment["camera_config_sha256"],
        "controller_config_sha256": environment["controller_config_sha256"],
        "preprocessor_config_sha256": environment["preprocessor_config_sha256"],
        "postprocessor_config_sha256": environment["postprocessor_config_sha256"],
    }
    components = contract.get("components")
    observed = {
        "model_id": contract.get("model_id"),
        "model_revision": contract.get("model_revision"),
        **(
            {name: components.get(name) for name in expected if name.endswith("_sha256")}
            if isinstance(components, Mapping)
            else {}
        ),
    }
    if contract.get("kind") != "vla_lens.pi05_runtime_contract" or observed != expected:
        _fail("runtime_binding_mismatch", "runtime contract differs from the locked child")
    return canonical_research_fingerprint(contract)


def _validate_campaign_state(
    state: CampaignState, child: Mapping[str, Any], child_fingerprint: str
) -> Mapping[str, Any]:
    if state.events_by_id and not state.program_locked:
        _fail("unaccepted_campaign_event", "campaign state was not accepted after program lock")
    lock = state.locked.get("FOUNDATION")
    if not isinstance(lock, Mapping):
        _fail("cross_child_attempt", "campaign reducer has no FOUNDATION child lock")
    if (
        lock.get("child_plan_id") != child["child_plan_id"]
        or lock.get("child_plan_fingerprint") != child_fingerprint
    ):
        _fail("cross_child_attempt", "campaign reducer locked another child")
    reservation_id = str(lock.get("reservation_id") or "")
    reservation = state.reservations.get(reservation_id)
    if not isinstance(reservation, Mapping) or reservation.get("study_id") != "FOUNDATION":
        _fail("cross_child_attempt", "FOUNDATION reservation is absent or mismatched")
    if not _is_sha256(lock.get("event_sha256")):
        _fail("cross_child_attempt", "FOUNDATION lock event is not immutable")
    return lock


def _accepted_exclusions(
    accepted_events: Sequence[tuple[Mapping[str, Any], str]],
    state: CampaignState,
    *,
    program_id: str,
    program_fingerprint: str,
) -> dict[str, list[Mapping[str, Any]]]:
    exclusions: dict[str, list[Mapping[str, Any]]] = {}
    canonical_events_present = bool(state.events_by_id)
    provided_event_ids = {str(event.get("event_id") or "") for event, _ in accepted_events}
    if canonical_events_present and provided_event_ids != set(state.events_by_id):
        _fail("unaccepted_campaign_event", "accepted event set is incomplete or contains extras")
    for event, event_hash in accepted_events:
        event_id = str(event.get("event_id") or "")
        if not _is_sha256(event_hash):
            _fail("unaccepted_campaign_event", "accepted event lacks its reducer hash")
        if canonical_events_present:
            if (
                state.events_by_id.get(event_id) != event
                or state.event_hashes.get(event_id) != event_hash
                or event.get("program_id") != program_id
                or event.get("program_fingerprint") != program_fingerprint
            ):
                _fail("unaccepted_campaign_event", "event was not accepted for the locked program")
        if event.get("event_type") != "deviation_recorded":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping) or payload.get("disposition") != "exclude_trial":
            continue
        target = payload.get("target_event")
        if not isinstance(target, Mapping) or not _is_sha256(target.get("event_sha256")):
            _fail("unaccepted_campaign_event", "exclusion target is not immutable")
        exclusions.setdefault(str(target.get("event_id") or ""), []).append(
            {"event": event, "event_sha256": event_hash}
        )
    return exclusions


def _build_trial(
    row: Mapping[str, str],
    attempts: Sequence[Mapping[str, Any]],
    exclusions: Mapping[str, Sequence[Mapping[str, Any]]],
    lock: Mapping[str, Any],
    child: Mapping[str, Any],
    child_fingerprint: str,
    runtime_config_fingerprint: str,
    runtime_contract_fingerprint: str,
    artifacts: "_ArtifactInventory",
    dataset_cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    try:
        ordered = sorted(attempts, key=lambda item: int(item["started"]["ordinal"]))
        ordinals = [int(item["started"]["ordinal"]) for item in ordered]
    except (KeyError, TypeError, ValueError):
        _fail("duplicate_terminal_attempt", f"{row['trial_id']} has invalid attempt ordinals")
    if ordinals != list(range(1, len(ordered) + 1)):
        _fail("duplicate_terminal_attempt", f"{row['trial_id']} has non-contiguous attempts")
    completed = [attempt for attempt in ordered if attempt.get("completed") is True]
    if len(completed) > 1 or (completed and completed[0] is not ordered[-1]):
        _fail("duplicate_terminal_attempt", f"{row['trial_id']} has duplicate accepted terminals")
    final = ordered[-1]
    final_exclusions = exclusions.get(str(final.get("event_id") or ""), ())
    if completed:
        resolution = "completed"
        accepted = completed[0]
        if final_exclusions:
            _fail("duplicate_terminal_attempt", "completed attempt is also excluded")
    else:
        if len(final_exclusions) != 1:
            _fail("missing_terminal_attempt", f"{row['trial_id']} lacks one final exclusion")
        resolution = "excluded"
        accepted = final

    records = []
    for attempt in ordered:
        records.append(
            _build_attempt(
                row,
                attempt,
                accepted_terminal=attempt is accepted,
                exclusion=(
                    final_exclusions[0] if attempt is accepted and final_exclusions else None
                ),
                lock=lock,
                child=child,
                child_fingerprint=child_fingerprint,
                runtime_config_fingerprint=runtime_config_fingerprint,
                runtime_contract_fingerprint=runtime_contract_fingerprint,
                artifacts=artifacts,
                dataset_cache=dataset_cache,
            )
        )
    return {
        "trial_id": row["trial_id"],
        "trial_manifest_row": dict(row),
        "resolution": resolution,
        "accepted_terminal_attempt_id": accepted["started"]["attempt_id"],
        "attempts": records,
    }


def _build_attempt(
    row: Mapping[str, str],
    attempt: Mapping[str, Any],
    *,
    accepted_terminal: bool,
    exclusion: Mapping[str, Any] | None,
    lock: Mapping[str, Any],
    child: Mapping[str, Any],
    child_fingerprint: str,
    runtime_config_fingerprint: str,
    runtime_contract_fingerprint: str,
    artifacts: "_ArtifactInventory",
    dataset_cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    started = attempt["started"]
    attempt_id = str(started.get("attempt_id") or "")
    lock_ref = started.get("child_lock_event")
    if (
        started.get("reservation_id") != lock.get("reservation_id")
        or not isinstance(lock_ref, Mapping)
        or lock_ref.get("event_id") != lock.get("event_id")
        or lock_ref.get("event_sha256") != lock.get("event_sha256")
    ):
        _fail("cross_child_attempt", f"{attempt_id} is not attached to the locked child")
    row_fingerprint = trial_row_fingerprint(row)
    if started.get("trial_manifest_row_fingerprint") != row_fingerprint:
        _fail("trial_binding_mismatch", f"{attempt_id} does not bind its frozen trial row")
    seeds = seed_bundle(row)
    seed_fingerprint = canonical_research_fingerprint(seeds)
    if started.get("seed_bundle_fingerprint") != seed_fingerprint:
        _fail("seed_binding_mismatch", f"{attempt_id} does not bind all seed domains")
    if started.get("runtime_config_fingerprint") != runtime_config_fingerprint:
        _fail("runtime_binding_mismatch", f"{attempt_id} does not bind the runtime contract")

    completed = attempt.get("completed") is True
    ref_field = "output_refs" if completed else "log_refs"
    evidence = [artifacts.resolve(reference) for reference in attempt.get(ref_field, ())]
    exclusion_payload = None
    if exclusion is not None:
        exclusion_payload = exclusion["event"]["payload"]
        target = exclusion_payload["target_event"]
        if (
            target.get("event_id") != attempt.get("event_id")
            or target.get("event_sha256") != attempt.get("event_sha256")
            or target.get("sequence") != attempt.get("event_sequence")
        ):
            _fail("unaccepted_campaign_event", "exclusion targets another terminal attempt")
        evidence.extend(
            artifacts.resolve(reference) for reference in exclusion_payload.get("evidence_refs", ())
        )

    bindings = {
        "child_plan_id": child["child_plan_id"],
        "child_fingerprint": child_fingerprint,
        "trial_row_fingerprint": row_fingerprint,
        "checkpoint_revision": child["runtime"]["model"]["revision"],
        "runtime_contract_fingerprint": runtime_contract_fingerprint,
        "runtime_config_fingerprint": runtime_config_fingerprint,
        "effective_config_sha256": None,
        "seed_bundle_fingerprint": seed_fingerprint,
        "seed_identities": seeds,
    }
    measurements = None
    trace = None
    if completed:
        if attempt.get("terminal_status") not in child["completion"]["valid_trial_statuses"]:
            _fail("duplicate_terminal_attempt", f"{attempt_id} has an invalid terminal status")
        receipt_ref = attempt.get("runtime_receipt_ref")
        receipt_artifact = artifacts.resolve(receipt_ref)
        evidence.append(receipt_artifact)
        receipt = load_research_mapping(artifacts.path(receipt_artifact))
        _validate_runtime_receipt(
            receipt,
            started,
            attempt,
            lock,
            runtime_config_fingerprint,
        )
        trace = _select_trace_episode(
            row,
            started,
            attempt,
            receipt,
            child,
            child_fingerprint,
            runtime_config_fingerprint,
            artifacts,
            dataset_cache,
        )
        bindings["effective_config_sha256"] = trace["effective_config_sha256"]
        measurements = trace["measurements"]
    return {
        "attempt_id": attempt_id,
        "ordinal": int(started["ordinal"]),
        "terminal_kind": "completed" if completed else "failed",
        "accepted_terminal": accepted_terminal,
        "event_id": attempt.get("event_id"),
        "event_sha256": attempt.get("event_sha256"),
        "bindings": bindings,
        "evidence_refs": _deduplicate_artifacts(evidence),
        "exclusion": (
            None
            if exclusion_payload is None
            else {
                "category": exclusion_payload["category"],
                "disposition": exclusion_payload["disposition"],
                "reason": exclusion_payload["reason"],
                "event_sha256": exclusion["event_sha256"],
            }
        ),
        **(
            {
                "measurements": measurements,
                "trace_evidence": trace["identity"],
            }
            if completed
            else {}
        ),
        **(
            {}
            if completed
            else {
                "failure_stage": attempt.get("failure_stage"),
                "error_code": attempt.get("error_code"),
                "retryable": attempt.get("retryable"),
            }
        ),
    }


def _validate_runtime_receipt(
    receipt: Mapping[str, Any],
    started: Mapping[str, Any],
    terminal: Mapping[str, Any],
    lock: Mapping[str, Any],
    runtime_config_fingerprint: str,
) -> None:
    expected = {
        "schema_version": 1,
        "kind": "vla_lens.trial_runtime_receipt",
        "attempt_id": started["attempt_id"],
        "trial_id": started["trial_id"],
        "child_lock_fingerprint": lock["lock_receipt_ref"]["content_fingerprint"],
        "runtime_config_fingerprint": runtime_config_fingerprint,
        "seed_bundle_fingerprint": started["seed_bundle_fingerprint"],
        "runtime_check_status": "pass",
        "terminal_status": terminal["terminal_status"],
        "output_refs": terminal["output_refs"],
    }
    if any(receipt.get(name) != value for name, value in expected.items()):
        _fail("runtime_binding_mismatch", f"{started['attempt_id']} runtime receipt differs")
    if not str(receipt.get("created_utc") or ""):
        _fail("runtime_binding_mismatch", "runtime receipt has no creation identity")


def _select_trace_episode(
    row: Mapping[str, str],
    started: Mapping[str, Any],
    terminal: Mapping[str, Any],
    receipt: Mapping[str, Any],
    child: Mapping[str, Any],
    child_fingerprint: str,
    runtime_config_fingerprint: str,
    artifacts: "_ArtifactInventory",
    dataset_cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    datasets = []
    for reference in terminal["output_refs"]:
        artifact = artifacts.resolve(reference)
        target = artifacts.path(artifact)
        dataset = _dataset_for_artifact(artifact, target, artifacts, dataset_cache)
        if dataset is not None and dataset not in datasets:
            datasets.append(dataset)
        if dataset is None and target.is_file():
            output_manifest = load_research_mapping(target)
            if output_manifest.get("kind") != "vla_lens.trial_output_manifest":
                _fail(
                    "trace_binding_mismatch",
                    "completed output reference is not a trace manifest",
                )
            if (
                output_manifest.get("attempt_id") != started.get("attempt_id")
                or output_manifest.get("trial_id") != row["trial_id"]
                or output_manifest.get("trace_id") != _expected_trace_id(row)
                or output_manifest.get("runtime_config_fingerprint")
                != runtime_config_fingerprint
                or output_manifest.get("seed_bundle_fingerprint")
                != started.get("seed_bundle_fingerprint")
            ):
                _fail("trace_binding_mismatch", "output manifest identity differs from its trial")
            files = output_manifest.get("files")
            if not isinstance(files, list) or not files:
                _fail("trace_binding_mismatch", "output manifest has no immutable trace files")
            for file_reference in files:
                external = artifacts.resolve_external_file(file_reference)
                external_target = artifacts.path(external)
                dataset = _dataset_for_artifact(
                    external,
                    external_target,
                    artifacts,
                    dataset_cache,
                )
                if dataset is not None and dataset not in datasets:
                    datasets.append(dataset)
    if len(datasets) != 1:
        _fail("trace_binding_mismatch", "completed attempt must bind one capture root")
    dataset = datasets[0]
    environment = child["runtime"]["environment"]
    expected = {
        "trial_id": row["trial_id"],
        "child_plan_id": child["child_plan_id"],
        "canonical_family_id": row["canonical_family_id"],
        "pool": row["pool"],
        "replicate_id": row["replicate_id"],
        "layout_id": int(row["layout_id"]),
        "model_revision": child["runtime"]["model"]["revision"],
        "snapshot_manifest_sha256": child["runtime"]["model"]["snapshot_manifest_sha256"],
        "camera_config_sha256": environment["camera_config_sha256"],
        "controller_config_sha256": environment["controller_config_sha256"],
        "preprocessor_config_sha256": environment["preprocessor_config_sha256"],
        "postprocessor_config_sha256": environment["postprocessor_config_sha256"],
        "seed_identities": {
            domain: int(row[f"{domain}_seed"])
            for domain in SEED_DOMAINS
        },
    }
    matches = []
    for episode in dataset["episodes"]:
        manifest = episode["manifest"]
        metadata = manifest.get("metadata")
        audit = metadata.get("runtime_audit") if isinstance(metadata, Mapping) else None
        if (
            manifest.get("task_id") == str(row["task_id"])
            and manifest.get("env_id") == row["benchmark"]
            and manifest.get("model_id") == child["runtime"]["model"]["repo_id"]
            and isinstance(audit, Mapping)
            and all(audit.get(key) == value for key, value in expected.items())
            and manifest.get("trace_id") == _expected_trace_id(row)
        ):
            matches.append(episode)
    if len(matches) != 1:
        _fail(
            "trace_binding_mismatch",
            f"{row['trial_id']} resolves to {len(matches)} immutable matching episodes",
        )
    episode = matches[0]
    metadata = episode["manifest"]["metadata"]
    effective = _effective_config_fingerprint(metadata)
    if not _is_sha256(effective) or effective != dataset["effective_config_sha256"]:
        _fail("effective_config_binding_mismatch", "trace effective config is ambiguous")
    for source in (started, receipt):
        declared = source.get("effective_config_sha256")
        if declared is not None and declared != effective:
            _fail("effective_config_binding_mismatch", "attempt chain changes effective config")
    measurements = _validate_measurements(metadata, episode["path"])
    return {
        "effective_config_sha256": effective,
        "measurements": measurements,
        "identity": {
            "root_id": dataset["root_id"],
            "dataset_path": dataset["relative_path"],
            "episode_index": episode["episode_index"],
            "trace_id": episode["manifest"]["trace_id"],
            "episode_manifest": artifacts.file_record(
                dataset["root_id"], episode["path"] / "manifest.json"
            ),
        },
    }


def _dataset_for_artifact(
    artifact: Mapping[str, Any],
    target: Path,
    artifacts: "_ArtifactInventory",
    cache: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    approved_root = artifacts.roots[artifact["root_id"]]
    start = target if target.is_dir() else target.parent
    candidates = [start, *start.parents]
    dataset_root = next(
        (
            candidate
            for candidate in candidates
            if _is_within(candidate, approved_root)
            and (candidate / "meta/info.json").is_file()
            and (candidate / VLA_LENS_OVERLAY_MANIFEST).is_file()
        ),
        None,
    )
    if dataset_root is None:
        return None
    relative = dataset_root.relative_to(approved_root).as_posix()
    key = (str(artifact["root_id"]), relative)
    if key not in cache:
        cache[key] = _load_dataset(str(artifact["root_id"]), dataset_root, approved_root, artifacts)
    return cache[key]


def _load_dataset(
    root_id: str,
    root: Path,
    approved_root: Path,
    artifacts: "_ArtifactInventory",
) -> dict[str, Any]:
    result = validate_lerobot_v3_dataset(root)
    if not result.valid:
        _fail("invalid_lerobot_trace", ", ".join(issue.code for issue in result.errors))
    overlay = load_research_mapping(root / VLA_LENS_OVERLAY_MANIFEST)
    refs = pd.read_parquet(root / VLA_LENS_OVERLAY_REFERENCES)
    required_ref_fields = {"episode_index", "trace_id", "overlay_path"}
    if (
        overlay.get("robot_dataset_format") != "lerobot_v3"
        or overlay.get("overlay_root") != "vla_lens"
        or overlay.get("episodes") != len(refs)
        or not required_ref_fields <= set(refs.columns)
        or refs["episode_index"].duplicated().any()
    ):
        _fail("invalid_lerobot_trace", "overlay manifest and episode references disagree")
    episodes = []
    effective_configs: set[str] = set()
    for record in refs.sort_values("episode_index").to_dict("records"):
        relative = _safe_relative(str(record["overlay_path"]), "overlay episode path")
        manifest_path = root / relative / "manifest.json"
        if not manifest_path.is_file() or _path_has_symlink(manifest_path):
            _fail("invalid_lerobot_trace", "overlay episode manifest is missing or unsafe")
        manifest = load_research_mapping(manifest_path)
        if str(record["trace_id"]) != str(manifest.get("trace_id")):
            _fail("invalid_lerobot_trace", "overlay trace ID differs from episode manifest")
        metadata = manifest.get("metadata")
        effective = _effective_config_fingerprint(metadata)
        if effective is not None:
            effective_configs.add(effective)
        episodes.append(
            {
                "episode_index": int(record["episode_index"]),
                "manifest": manifest,
                "path": manifest_path.parent,
                "relative_manifest_path": manifest_path.relative_to(root).as_posix(),
            }
        )
    if len(effective_configs) != 1:
        _fail("effective_config_binding_mismatch", "capture root has multiple effective configs")
    artifacts.inventory_tree(root_id, root)
    return {
        "root_id": root_id,
        "relative_path": root.relative_to(approved_root).as_posix() or ".",
        "episodes": episodes,
        "effective_config_sha256": next(iter(effective_configs)),
    }


def _effective_config_fingerprint(metadata: Any) -> str | None:
    """Return the explicit or capture-plan identity for one trace."""

    if not isinstance(metadata, Mapping):
        return None
    explicit = metadata.get("effective_config_sha256")
    if _is_sha256(explicit):
        return str(explicit)
    capture_plan = metadata.get("capture_plan")
    if isinstance(capture_plan, Mapping):
        return canonical_research_fingerprint(capture_plan)
    return None


def _validate_measurements(metadata: Mapping[str, Any], episode_path: Path) -> dict[str, Any]:
    success = metadata.get("simulator_success")
    contact = metadata.get("mujoco_contact")
    proximity = metadata.get("proximity")
    motion = metadata.get("object_motion")
    capability = metadata.get("contact_capability")
    if not isinstance(success, bool):
        _fail("measurement_contract_mismatch", "simulator success must be boolean")
    if not isinstance(contact, Mapping) or not isinstance(contact.get("physical_contact"), bool):
        _fail("measurement_contract_mismatch", "contact evidence has no usable boolean")
    if not isinstance(proximity, Mapping) or not isinstance(
        proximity.get("positive_gap_within_contact_margin"), bool
    ):
        _fail("measurement_contract_mismatch", "proximity evidence has no usable boolean")
    if not isinstance(motion, Mapping) or not isinstance(motion.get("moved"), bool):
        _fail("measurement_contract_mismatch", "object-motion evidence has no usable boolean")
    required_capability = {
        "capability_status": str,
        "sample_phase": str,
        "force_optional": bool,
        "exhaustive_physics_substeps": bool,
        "terminal_post_action_contact_observed_after_autoreset": bool,
        "runtime_smoke_completed": bool,
    }
    if not isinstance(capability, Mapping) or any(
        not isinstance(capability.get(name), expected_type)
        for name, expected_type in required_capability.items()
    ):
        _fail("measurement_contract_mismatch", "contact capability schema is incomplete")
    if capability["sample_phase"] != "pre_action_control_step":
        _fail("measurement_contract_mismatch", "contact sample phase changed")
    _validate_physical_tables(episode_path, success, contact, proximity, capability)
    return {
        "simulator_success": success,
        "mujoco_contact": dict(contact),
        "proximity": dict(proximity),
        "object_motion": dict(motion),
        "contact_capability": dict(capability),
    }


def _validate_physical_tables(
    episode_path: Path,
    success: bool,
    contact: Mapping[str, Any],
    proximity: Mapping[str, Any],
    capability: Mapping[str, Any],
) -> None:
    evaluation_path = episode_path / "tables/evaluation.parquet"
    scene_path = episode_path / "tables/scene_state.parquet"
    if not evaluation_path.is_file() or not scene_path.is_file():
        _fail("measurement_contract_mismatch", "physical evidence tables are missing")
    evaluation = pd.read_parquet(evaluation_path)
    if not {"metric_name", "metric_value", "passed", "source"} <= set(evaluation.columns):
        _fail("measurement_contract_mismatch", "evaluation table schema is incomplete")
    success_rows = evaluation.loc[evaluation["metric_name"].astype(str) == "success"]
    values = {_strict_bool(value) for value in success_rows["passed"].tolist()}
    if values != {success}:
        _fail("measurement_contract_mismatch", "evaluation success is absent or inconsistent")

    scene = pd.read_parquet(scene_path)
    if "context_kind" not in scene:
        _fail("measurement_contract_mismatch", "scene-state schema lacks context_kind")
    contact_rows = scene.loc[scene["context_kind"].astype(str) == "mujoco_contact"]
    contact_fields = {
        "physical_contact",
        "positive_gap_proximity",
        "signed_distance_m",
        "distance_class",
        "sample_phase",
    }
    if contact_rows.empty or not contact_fields <= set(contact_rows.columns):
        _fail("measurement_contract_mismatch", "contact table schema is incomplete")
    physical_values = {_strict_bool(value) for value in contact_rows["physical_contact"]}
    proximity_values = {_strict_bool(value) for value in contact_rows["positive_gap_proximity"]}
    if any(physical_values) != contact["physical_contact"]:
        _fail("measurement_contract_mismatch", "contact rows disagree with physical contact")
    if any(proximity_values) != proximity["positive_gap_within_contact_margin"]:
        _fail("measurement_contract_mismatch", "contact rows disagree with proximity")
    if set(contact_rows["sample_phase"].astype(str)) != {capability["sample_phase"]}:
        _fail("measurement_contract_mismatch", "contact rows use another sample phase")

    object_rows = scene.loc[scene["context_kind"].astype(str) == "object"]
    if object_rows.empty or "pos_array_id" not in object_rows:
        _fail("measurement_contract_mismatch", "object rows lack a motion source")
    array_ids = {str(value) for value in object_rows["pos_array_id"] if str(value)}
    if len(array_ids) != 1:
        _fail("measurement_contract_mismatch", "object position source is missing or ambiguous")
    try:
        positions = np.asarray(TraceBundle.open(episode_path).array(next(iter(array_ids))))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise FoundationEvidenceError(
            "measurement_contract_mismatch", "object position array cannot be loaded"
        ) from exc
    if positions.ndim != 3 or positions.shape[0] < 2 or not np.isfinite(positions).all():
        _fail("measurement_contract_mismatch", "object position array is not usable")

    capability_rows = scene.loc[scene["context_kind"].astype(str) == "contact_capability"]
    if capability_rows.empty or not {"sample_phase", "exhaustive_physics_substeps"} <= set(
        capability_rows.columns
    ):
        _fail("measurement_contract_mismatch", "contact capability rows are incomplete")
    if set(capability_rows["sample_phase"].astype(str)) != {capability["sample_phase"]}:
        _fail("measurement_contract_mismatch", "contact capability sample phase changed")
    exhaustive = {_strict_bool(value) for value in capability_rows["exhaustive_physics_substeps"]}
    if exhaustive != {capability["exhaustive_physics_substeps"]}:
        _fail("measurement_contract_mismatch", "contact capability limitation changed")


class _ArtifactInventory:
    def __init__(self, roots: Mapping[str, Path]):
        self.roots = roots
        self._records: dict[tuple[str, str], dict[str, Any]] = {}

    def resolve(self, reference: Any) -> dict[str, Any]:
        if not isinstance(reference, Mapping) or set(reference) != ARTIFACT_REF_FIELDS:
            _fail("artifact_hash_mismatch", "artifact reference has the wrong fields")
        root_id = str(reference["root_id"])
        root = self.roots.get(root_id)
        if root is None:
            _fail("unapproved_artifact_root", f"root {root_id!r} was not approved")
        relative = _safe_relative(str(reference["path"]), "artifact path")
        target = root / relative
        if _path_has_symlink(target) or not _is_within(target.resolve(), root):
            _fail("unsafe_artifact_path", f"artifact path is unsafe: {relative}")
        if target.is_file():
            observed = file_sha256(target)
            size = target.stat().st_size
        elif target.is_dir():
            observed = directory_sha256(target)
            size = sum(item["size"] for item in _directory_files(target))
        else:
            _fail("artifact_hash_mismatch", f"artifact is missing: {root_id}:{relative}")
        if observed != reference["sha256"]:
            _fail("artifact_hash_mismatch", f"artifact bytes changed: {root_id}:{relative}")
        return self._add(root_id, relative.as_posix(), size, observed, reference)

    def inventory_tree(self, root_id: str, path: Path) -> None:
        root = self.roots[root_id]
        if _path_has_symlink(path) or not _is_within(path.resolve(), root):
            _fail("unsafe_artifact_path", "capture root is outside its approved root")
        for file_path in sorted(path.rglob("*")):
            if file_path.is_symlink():
                _fail("unsafe_artifact_path", f"capture contains symlink {file_path}")
            if file_path.is_file():
                relative = file_path.relative_to(root).as_posix()
                self._add(
                    root_id,
                    relative,
                    file_path.stat().st_size,
                    file_sha256(file_path),
                    None,
                )

    def resolve_external_file(self, reference: Any) -> dict[str, Any]:
        """Resolve a measured file path against one of the approved roots."""

        if not isinstance(reference, Mapping):
            _fail("artifact_hash_mismatch", "output file reference is not an object")
        allowed = {"path", "sha256", "size"}
        if set(reference) == {"root_id", "path", "sha256", "size"}:
            root_id = str(reference["root_id"])
            root = self.roots.get(root_id)
            if root is None:
                _fail("unapproved_artifact_root", f"root {root_id!r} was not approved")
            relative = _safe_relative(str(reference["path"]), "output file path")
            target = root / relative
        elif set(reference) == allowed and Path(str(reference["path"])).is_absolute():
            target = Path(str(reference["path"]))
            candidates = [
                (root_id, root)
                for root_id, root in self.roots.items()
                if _is_within(target.resolve(), root)
            ]
            if not candidates:
                _fail("unapproved_artifact_root", "output file is outside approved roots")
            root_id, root = max(candidates, key=lambda item: len(item[1].parts))
            relative = _safe_relative(
                target.resolve().relative_to(root).as_posix(), "output file path"
            )
        else:
            _fail("artifact_hash_mismatch", "output file reference has the wrong fields")
        if _path_has_symlink(target) or not target.is_file():
            _fail("unsafe_artifact_path", "output file is missing or symlinked")
        expected_size = reference.get("size")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            _fail("artifact_hash_mismatch", "output file size is invalid")
        observed_size = target.stat().st_size
        observed_hash = file_sha256(target)
        if observed_size != expected_size or observed_hash != reference.get("sha256"):
            _fail("artifact_hash_mismatch", "output file bytes changed")
        return self._add(root_id, relative.as_posix(), observed_size, observed_hash, None)

    def path(self, artifact: Mapping[str, Any]) -> Path:
        return self.roots[str(artifact["root_id"])] / str(artifact["path"])

    def file_record(self, root_id: str, path: Path) -> dict[str, Any]:
        root = self.roots[root_id]
        relative = path.relative_to(root).as_posix()
        record = self._records.get((root_id, relative))
        if record is None:
            _fail("artifact_hash_mismatch", f"file was not inventoried: {root_id}:{relative}")
        return dict(record)

    def records(self) -> list[dict[str, Any]]:
        return [self._records[key] for key in sorted(self._records)]

    def _add(
        self,
        root_id: str,
        path: str,
        size: int,
        sha256: str,
        reference: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        key = (root_id, path)
        record = {
            "root_id": root_id,
            "path": path,
            "size": int(size),
            "sha256": sha256,
        }
        if reference is not None:
            record.update({"id": reference["id"], "type": reference["type"]})
        prior = self._records.get(key)
        if prior is not None and (
            prior["size"] != record["size"] or prior["sha256"] != record["sha256"]
        ):
            _fail("artifact_hash_mismatch", f"artifact changed while indexing: {root_id}:{path}")
        if prior is None or reference is not None:
            self._records[key] = {**(prior or {}), **record}
        return dict(self._records[key])


def _approved_roots(values: Mapping[str, str | Path]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for root_id, value in values.items():
        if not root_id or "/" in root_id or "\\" in root_id or root_id in roots:
            _fail("unapproved_artifact_root", f"invalid root ID {root_id!r}")
        path = Path(value)
        if _path_has_symlink(path) or not path.is_dir():
            _fail("unapproved_artifact_root", f"root {root_id!r} is missing or symlinked")
        roots[root_id] = path.resolve()
    return roots


def _confined_file(path: str | Path, root: Path, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    if _path_has_symlink(candidate):
        _fail("unsafe_artifact_path", f"{label} is symlinked")
    resolved = candidate.resolve()
    if not _is_within(resolved, root) or not resolved.is_file():
        _fail("unsafe_artifact_path", f"{label} is outside the repo or missing")
    return resolved


def _directory_files(path: Path) -> list[dict[str, Any]]:
    records = []
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            _fail("unsafe_artifact_path", f"directory contains symlink {child}")
        if child.is_file():
            records.append(
                {
                    "path": child.relative_to(path).as_posix(),
                    "size": child.stat().st_size,
                    "sha256": file_sha256(child),
                }
            )
    return records


def _safe_relative(value: str, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or value == "."
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or path.as_posix() != value
    ):
        _fail("unsafe_artifact_path", f"{label} is not a normalized relative path")
    return path


def _path_has_symlink(path: Path) -> bool:
    absolute = path.absolute()
    return any(item.is_symlink() for item in (absolute, *absolute.parents))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if type(value).__module__ == "numpy" and type(value).__name__ == "bool_":
        return bool(value)
    _fail("measurement_contract_mismatch", "physical measurement is not boolean")


def _deduplicate_artifacts(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(str(item["root_id"]), str(item["path"])): dict(item) for item in values}
    return [by_key[key] for key in sorted(by_key)]


def _fail(code: str, message: str) -> None:
    raise FoundationEvidenceError(code, message)


__all__ = [
    "EXPECTED_TRIAL_COUNT",
    "FOUNDATION_TRIAL_FIELDS",
    "FoundationContract",
    "FoundationEvidenceError",
    "VerifiedFoundationLedger",
    "build_foundation_evidence_index",
    "create_foundation_evidence_index",
    "directory_sha256",
    "load_foundation_contract",
    "load_verified_foundation_ledger",
    "seed_bundle",
    "seed_bundle_fingerprint",
    "trial_row_fingerprint",
    "write_foundation_evidence_index",
]
