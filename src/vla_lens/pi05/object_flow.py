"""Derive object-role and interaction-flow labels for PI0.5 episodes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.analyzer import dataset_fingerprint
from vla_lens.artifacts import LensArtifact
from vla_lens.dataset import build_dataset_index
from vla_lens.pi05.interaction_metrics import (
    DEFAULT_THRESHOLDS,
    _alias_head,
    _eef_positions,
    _normalize_text,
    _normalized_phrase_index,
    _object_metric_rows,
    _object_positions,
    _object_prompt_aliases,
    _object_rows,
    _read_probe_splits,
    _sidecar_record,
    _target_match_sort_key,
    _trajectory_for_object,
)
from vla_lens.traces import TraceBundle, TraceDataset
from vla_lens.workbench import AnalysisRunSpec, save_analysis_run

ARTIFACT_TYPE = "pi05_object_flow"
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ObjectFlowArtifact:
    artifact: LensArtifact
    object_roles: pd.DataFrame
    interaction_events: pd.DataFrame
    flow_steps: pd.DataFrame
    timestep_labels: pd.DataFrame


def save_pi05_object_flow_artifact(
    dataset: TraceDataset,
    *,
    name: str = "PI0.5 object flow labels",
    thresholds: Mapping[str, float | int] | None = None,
    rebuild_index: bool = True,
) -> ObjectFlowArtifact:
    """Derive role-aware object-flow labels and save them as a dataset artifact."""
    threshold_values = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}
    split_sidecar = _read_probe_splits(dataset.root)
    role_records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    flow_records: list[dict[str, Any]] = []
    timestep_records: list[dict[str, Any]] = []

    for bundle in dataset.bundles:
        result = _bundle_object_flow(
            bundle,
            split_sidecar=split_sidecar,
            thresholds=threshold_values,
        )
        role_records.extend(result["roles"])
        event_records.extend(result["events"])
        flow_records.extend(result["flow_steps"])
        timestep_records.extend(result["timesteps"])

    object_roles = pd.DataFrame.from_records(role_records)
    interaction_events = pd.DataFrame.from_records(event_records)
    flow_steps = pd.DataFrame.from_records(flow_records)
    timestep_labels = pd.DataFrame.from_records(timestep_records)
    fingerprint = dataset_fingerprint(dataset)
    artifact = LensArtifact.create(
        artifact_type=ARTIFACT_TYPE,
        name=name,
        group_id=ARTIFACT_TYPE,
        scope="dataset",
        selector={"episodes": "all", "source": "pi05_trace_context"},
        method={
            "workflow": "save_pi05_object_flow_artifact",
            "schema_version": SCHEMA_VERSION,
            "dataset_fingerprint": fingerprint,
            "thresholds": threshold_values,
            "outputs": {
                "object_roles": "",
                "interaction_events": "",
                "flow_steps": "",
                "timestep_labels": "",
            },
        },
        metrics=_summary_metrics(object_roles, interaction_events, flow_steps, timestep_labels),
        display={
            "kind": ARTIFACT_TYPE,
            "summary": _summary_text(object_roles, flow_steps, timestep_labels),
            "primary_table": "timestep_labels",
            "role_table": "object_roles",
            "event_table": "interaction_events",
            "flow_table": "flow_steps",
            "interpretation_notes": [
                "Roles are deterministic labels inferred from prompt text and observed state.",
                "Observed events come from object poses and end-effector distance proxies.",
                "Role labels are automatic research labels, not manual ground truth.",
            ],
        },
        tags=("pi05", "object_flow", "labels", "postprocess"),
        source_trace_ids=tuple(
            sorted(str(value) for value in object_roles.get("trace_id", pd.Series()).unique())
        ),
    )
    saved = dataset.save_artifact(artifact)
    artifact_dir = (dataset._dataset_artifact_root() / str(saved.path or "")).parent
    paths = {
        "object_roles": artifact_dir / "object_roles.parquet",
        "interaction_events": artifact_dir / "interaction_events.parquet",
        "flow_steps": artifact_dir / "flow_steps.parquet",
        "timestep_labels": artifact_dir / "timestep_labels.parquet",
    }
    object_roles.to_parquet(paths["object_roles"], index=False)
    interaction_events.to_parquet(paths["interaction_events"], index=False)
    flow_steps.to_parquet(paths["flow_steps"], index=False)
    timestep_labels.to_parquet(paths["timestep_labels"], index=False)

    outputs = {
        key: str(path.relative_to(dataset.root))
        for key, path in paths.items()
    }
    updated = LensArtifact(
        artifact_id=saved.artifact_id,
        artifact_type=saved.artifact_type,
        name=saved.name,
        group_id=saved.group_id,
        scope=saved.scope,
        selector=saved.selector,
        method={**dict(saved.method), "outputs": outputs},
        metrics=saved.metrics,
        arrays=saved.arrays,
        display=saved.display,
        tags=saved.tags,
        created_utc=saved.created_utc,
        source_trace_ids=saved.source_trace_ids,
        path=saved.path,
    )
    (artifact_dir / "artifact.json").write_text(
        json.dumps(updated.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    artifact_index_path = dataset._dataset_artifact_root() / TraceBundle.ARTIFACT_INDEX
    existing = (
        pd.read_parquet(artifact_index_path)
        if artifact_index_path.exists()
        else pd.DataFrame()
    )
    updated_index = pd.concat(
        [existing, pd.DataFrame.from_records([updated.to_record()])],
        ignore_index=True,
    ).drop_duplicates(subset=["artifact_id"], keep="last")
    artifact_index_path.parent.mkdir(parents=True, exist_ok=True)
    updated_index.to_parquet(artifact_index_path, index=False)
    dataset.__dict__.pop("dataset_artifact_index", None)
    dataset.__dict__.pop("artifact_index", None)
    save_analysis_run(
        dataset,
        AnalysisRunSpec(
            run_id=updated.artifact_id,
            workflow=ARTIFACT_TYPE,
            inputs=updated.selector,
            outputs=(),
            provenance={"artifact_id": updated.artifact_id},
        ),
    )
    if rebuild_index:
        build_dataset_index(dataset.root, overwrite=True)
    return ObjectFlowArtifact(
        artifact=updated,
        object_roles=object_roles,
        interaction_events=interaction_events,
        flow_steps=flow_steps,
        timestep_labels=timestep_labels,
    )


def _bundle_object_flow(
    bundle: TraceBundle,
    *,
    split_sidecar: pd.DataFrame,
    thresholds: Mapping[str, float | int],
) -> dict[str, list[dict[str, Any]]]:
    episode = bundle.episode_record()
    sidecar = _sidecar_record(split_sidecar, bundle.manifest.trace_id)
    objects = _object_rows(bundle)
    positions, position_source = _object_positions(bundle)
    eef_positions = _eef_positions(bundle)
    prompt_text = _prompt_text(bundle, episode)
    prompt_matches = _prompt_object_matches(prompt_text, objects)
    metrics = _object_metric_rows(
        bundle,
        objects,
        positions,
        eef_positions=eef_positions,
        target_objects=(),
        thresholds=thresholds,
    )
    role_rows = [
        _role_record(
            bundle,
            episode,
            sidecar,
            metric,
            prompt_text=prompt_text,
            match=prompt_matches.get(str(metric["object_name"])),
            position_source=position_source,
        )
        for metric in metrics
    ]
    events = _event_records(bundle, episode, sidecar, role_rows)
    flow_steps = _flow_step_records(bundle, episode, sidecar, role_rows)
    timesteps = _timestep_records(
        bundle,
        episode,
        sidecar,
        role_rows,
        flow_steps,
        positions,
        eef_positions,
        thresholds,
    )
    return {
        "roles": role_rows,
        "events": events,
        "flow_steps": flow_steps,
        "timesteps": timesteps,
    }


def _prompt_text(bundle: TraceBundle, episode: Mapping[str, Any]) -> str:
    return " ".join(
        str(value)
        for value in [
            bundle.manifest.prompt,
            episode.get("task_name", ""),
            bundle.manifest.metadata.get("task_name", ""),
        ]
        if value
    )


def _prompt_object_matches(
    text: str,
    objects: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    normalized_text = _normalize_text(text)
    matches: list[dict[str, Any]] = []
    for obj in objects:
        base = str(obj["object_base_name"])
        best_match: dict[str, Any] | None = None
        for alias, strength in _object_prompt_aliases(base):
            index = _normalized_phrase_index(normalized_text, alias)
            if index < 0:
                continue
            candidate = {
                "object_name": str(obj["object_name"]),
                "object_base_name": base,
                "alias": alias,
                "strength": strength,
                "index": index,
            }
            if best_match is None or _target_match_sort_key(candidate) < _target_match_sort_key(
                best_match
            ):
                best_match = candidate
        if best_match is not None:
            matches.append(best_match)
    strong_heads = {
        _alias_head(str(item.get("alias") or ""))
        for item in matches
        if int(item.get("strength", 0)) >= 3
    }
    filtered = [
        item
        for item in matches
        if int(item.get("strength", 0)) >= 3
        or _alias_head(str(item.get("alias") or "")) not in strong_heads
    ]
    return {
        str(item["object_name"]): item
        for item in sorted(filtered, key=_target_match_sort_key)
    }


def _role_record(
    bundle: TraceBundle,
    episode: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    metric: Mapping[str, Any],
    *,
    prompt_text: str,
    match: Mapping[str, Any] | None,
    position_source: str,
) -> dict[str, Any]:
    mentioned = match is not None
    kind = str(metric.get("object_kind") or "")
    context = _match_context(prompt_text, match)
    observed_contacted = bool(metric.get("contacted"))
    observed_moved = bool(metric.get("moved"))
    observed_lifted = bool(metric.get("lifted"))
    fixture_action = _fixture_action(context, prompt_text)
    location_context = _location_context(context)
    pronoun_receptacle = _pronoun_reuses_fixture(prompt_text, match)
    role_fixture = mentioned and (
        kind == "fixture"
        or fixture_action != ""
        or _alias_head(str(match.get("alias") if match else "")) in _FIXTURE_HEADS
    )
    role_receptacle = mentioned and (
        location_context
        or pronoun_receptacle
        or (role_fixture and _has_put_or_place(prompt_text))
    )
    role_manipulated = mentioned and (
        (kind == "object" and not role_receptacle)
        or observed_lifted
        or (observed_moved and not role_fixture)
    )
    return {
        "trace_id": bundle.manifest.trace_id,
        "episode_id": bundle.manifest.episode_id,
        "dataset_id": episode.get("dataset_id") or sidecar.get("dataset_id") or "",
        "benchmark": sidecar.get("benchmark") or episode.get("benchmark") or bundle.manifest.env_id,
        "task_id": bundle.manifest.task_id,
        "split": sidecar.get("split") or episode.get("split") or "",
        "prompt": bundle.manifest.prompt,
        "object_index": int(metric["object_index"]),
        "object_name": str(metric["object_name"]),
        "object_base_name": str(metric["object_base_name"]),
        "object_kind": kind,
        "prompt_mentioned": mentioned,
        "prompt_alias": "" if match is None else str(match.get("alias") or ""),
        "prompt_match_index": np.nan if match is None else int(match.get("index", -1)),
        "prompt_match_strength": 0 if match is None else int(match.get("strength", 0)),
        "role_manipulated": bool(role_manipulated),
        "role_receptacle": bool(role_receptacle),
        "role_fixture": bool(role_fixture),
        "role_distractor": not mentioned and not observed_contacted and not observed_moved,
        "fixture_action": fixture_action,
        "observed_contacted": observed_contacted,
        "observed_moved": observed_moved,
        "observed_lifted": observed_lifted,
        "contact_onset_timestep": metric.get("contact_onset_timestep", np.nan),
        "movement_onset_timestep": metric.get("movement_onset_timestep", np.nan),
        "lift_onset_timestep": metric.get("lift_onset_timestep", np.nan),
        "max_displacement": metric.get("max_displacement", np.nan),
        "max_xy_displacement": metric.get("max_xy_displacement", np.nan),
        "max_z_delta": metric.get("max_z_delta", np.nan),
        "missing_position": bool(metric.get("missing_position", False)),
        "position_source": position_source,
    }


def _match_context(
    prompt_text: str,
    match: Mapping[str, Any] | None,
    *,
    window: int = 6,
) -> dict[str, Any]:
    normalized = _normalize_text(prompt_text)
    if match is None:
        return {"before": [], "after": [], "alias": "", "index": -1}
    alias = str(match.get("alias") or "")
    index = int(match.get("index", -1))
    before = normalized[: max(0, index)].strip("_").split("_") if index >= 0 else []
    after_start = max(0, index) + len(alias)
    after = normalized[after_start:].strip("_").split("_") if index >= 0 else []
    return {
        "before": [token for token in before if token][-window:],
        "after": [token for token in after if token][:window],
        "alias": alias,
        "index": index,
    }


def _fixture_action(context: Mapping[str, Any], prompt_text: str) -> str:
    tokens = list(context.get("before") or []) + [str(context.get("alias") or "")]
    normalized = _normalize_text(prompt_text)
    if "close" in tokens or re.search(r"(^|_)close(_|$)", normalized):
        return "close"
    if "open" in tokens or re.search(r"(^|_)open(_|$)", normalized):
        return "open"
    if "turn" in tokens and "on" in tokens:
        return "turn_on"
    if "turn" in tokens and "off" in tokens:
        return "turn_off"
    return ""


def _location_context(context: Mapping[str, Any]) -> bool:
    before = set(context.get("before") or [])
    return bool(before & _LOCATION_CUES)


def _pronoun_reuses_fixture(prompt_text: str, match: Mapping[str, Any] | None) -> bool:
    if match is None:
        return False
    normalized = _normalize_text(prompt_text)
    index = int(match.get("index", -1))
    if index < 0:
        return False
    tail = normalized[index:]
    return any(phrase in tail for phrase in ("on_it", "in_it", "into_it", "inside_it"))


def _has_put_or_place(prompt_text: str) -> bool:
    normalized = _normalize_text(prompt_text)
    return any(verb in normalized for verb in ("put", "place", "stack"))


def _event_records(
    bundle: TraceBundle,
    episode: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    role_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role in role_rows:
        for event_type, column in [
            ("contact", "contact_onset_timestep"),
            ("move", "movement_onset_timestep"),
            ("lift", "lift_onset_timestep"),
        ]:
            onset = role.get(column, np.nan)
            if pd.isna(onset):
                continue
            rows.append(
                {
                    "trace_id": bundle.manifest.trace_id,
                    "episode_id": bundle.manifest.episode_id,
                    "dataset_id": episode.get("dataset_id") or sidecar.get("dataset_id") or "",
                    "benchmark": sidecar.get("benchmark")
                    or episode.get("benchmark")
                    or bundle.manifest.env_id,
                    "task_id": bundle.manifest.task_id,
                    "split": sidecar.get("split") or episode.get("split") or "",
                    "object_name": role["object_name"],
                    "object_base_name": role["object_base_name"],
                    "object_kind": role["object_kind"],
                    "event_type": event_type,
                    "onset_timestep": int(onset),
                    "role_manipulated": bool(role.get("role_manipulated")),
                    "role_receptacle": bool(role.get("role_receptacle")),
                    "role_fixture": bool(role.get("role_fixture")),
                    "prompt_mentioned": bool(role.get("prompt_mentioned")),
                }
            )
    return sorted(rows, key=lambda row: (int(row["onset_timestep"]), str(row["object_name"])))


def _flow_step_records(
    bundle: TraceBundle,
    episode: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    role_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    roles = sorted(
        role_rows,
        key=lambda row: (
            _safe_prompt_index(row),
            _safe_onset(row, "movement_onset_timestep"),
            str(row["object_name"]),
        ),
    )
    receptacles = [row for row in roles if bool(row.get("role_receptacle"))]
    steps: list[dict[str, Any]] = []
    for role in roles:
        if bool(role.get("role_fixture")) and not bool(role.get("role_manipulated")):
            steps.append(
                _flow_step_record(
                    bundle,
                    episode,
                    sidecar,
                    role,
                    step_type=str(role.get("fixture_action") or "fixture_interaction"),
                    target_object=None,
                )
            )
        if bool(role.get("role_manipulated")):
            target = _receptacle_for_role(role, receptacles)
            steps.append(
                _flow_step_record(
                    bundle,
                    episode,
                    sidecar,
                    role,
                    step_type="manipulate_object",
                    target_object=target,
                )
            )
    steps = sorted(
        steps,
        key=lambda row: (
            int(row["prompt_order"]),
            _nan_last(row.get("observed_start_timestep")),
            str(row["object_name"]),
        ),
    )
    for index, row in enumerate(steps):
        row["flow_step_index"] = index
    return steps


def _flow_step_record(
    bundle: TraceBundle,
    episode: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    role: Mapping[str, Any],
    *,
    step_type: str,
    target_object: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "trace_id": bundle.manifest.trace_id,
        "episode_id": bundle.manifest.episode_id,
        "dataset_id": episode.get("dataset_id") or sidecar.get("dataset_id") or "",
        "benchmark": sidecar.get("benchmark") or episode.get("benchmark") or bundle.manifest.env_id,
        "task_id": bundle.manifest.task_id,
        "split": sidecar.get("split") or episode.get("split") or "",
        "flow_step_index": -1,
        "step_type": step_type,
        "object_name": role["object_name"],
        "object_base_name": role["object_base_name"],
        "target_object_name": "" if target_object is None else str(target_object["object_name"]),
        "target_object_base_name": ""
        if target_object is None
        else str(target_object["object_base_name"]),
        "prompt_order": _safe_prompt_index(role),
        "observed_start_timestep": _first_available_onset(role),
        "contact_onset_timestep": role.get("contact_onset_timestep", np.nan),
        "movement_onset_timestep": role.get("movement_onset_timestep", np.nan),
        "lift_onset_timestep": role.get("lift_onset_timestep", np.nan),
    }


def _receptacle_for_role(
    role: Mapping[str, Any],
    receptacles: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if not receptacles:
        return None
    role_index = _safe_prompt_index(role)
    later = [item for item in receptacles if _safe_prompt_index(item) > role_index]
    return later[0] if later else receptacles[0]


def _timestep_records(
    bundle: TraceBundle,
    episode: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    role_rows: Sequence[Mapping[str, Any]],
    flow_steps: Sequence[Mapping[str, Any]],
    positions: np.ndarray | None,
    eef_positions: np.ndarray | None,
    thresholds: Mapping[str, float | int],
) -> list[dict[str, Any]]:
    length = int(bundle.manifest.length)
    object_masks = _object_masks(role_rows, positions, eef_positions, thresholds, length)
    move_events = _sorted_step_events(flow_steps, "movement_onset_timestep")
    contact_events = _sorted_step_events(flow_steps, "contact_onset_timestep")
    lift_events = _sorted_step_events(flow_steps, "lift_onset_timestep")
    flow_events = sorted(
        flow_steps,
        key=lambda row: (
            _nan_last(row.get("observed_start_timestep")),
            int(row.get("flow_step_index", 0)),
        ),
    )
    records: list[dict[str, Any]] = []
    for timestep in range(length):
        current_contact = _current_mask_object(object_masks, timestep, "contact")
        current_moved = _current_mask_object(object_masks, timestep, "move")
        current_lifted = _current_mask_object(object_masks, timestep, "lift")
        active_step = _active_step(flow_steps, flow_events, timestep)
        next_flow = _next_event(flow_events, timestep, "observed_start_timestep")
        next_manipulated = _next_manipulated_object(flow_events, timestep)
        next_contact = _next_event(contact_events, timestep, "contact_onset_timestep")
        next_moved = _next_event(move_events, timestep, "movement_onset_timestep")
        next_lifted = _next_event(lift_events, timestep, "lift_onset_timestep")
        records.append(
            {
                "trace_id": bundle.manifest.trace_id,
                "episode_id": bundle.manifest.episode_id,
                "dataset_id": episode.get("dataset_id") or sidecar.get("dataset_id") or "",
                "benchmark": sidecar.get("benchmark")
                or episode.get("benchmark")
                or bundle.manifest.env_id,
                "task_id": bundle.manifest.task_id,
                "split": sidecar.get("split") or episode.get("split") or "",
                "timestep": timestep,
                "policy_call_index": _policy_call_for_timestep(bundle, timestep),
                "current_contact_object": _object_name(current_contact),
                "current_moved_object": _object_name(current_moved),
                "current_lifted_object": _object_name(current_lifted),
                "next_contact_object": _object_name(next_contact),
                "next_moved_object": _object_name(next_moved),
                "next_lifted_object": _object_name(next_lifted),
                "next_manipulated_object": _object_name(next_manipulated),
                "active_manipulated_object": _active_object_name(
                    current_contact,
                    current_moved,
                    current_lifted,
                    active_step,
                    next_manipulated,
                ),
                "active_receptacle_object": _target_object_name(active_step),
                "active_flow_step_index": _flow_step_index(active_step),
                "next_flow_step_index": _flow_step_index(next_flow),
                "task_phase": _task_phase(
                    current_contact,
                    current_moved,
                    current_lifted,
                    next_flow,
                ),
            }
        )
    return records


def _object_masks(
    role_rows: Sequence[Mapping[str, Any]],
    positions: np.ndarray | None,
    eef_positions: np.ndarray | None,
    thresholds: Mapping[str, float | int],
    length: int,
) -> dict[str, dict[str, Any]]:
    movement_threshold = float(thresholds["movement_distance_m"])
    lift_threshold = float(thresholds["lift_z_m"])
    contact_threshold = float(thresholds["contact_center_distance_m"])
    masks: dict[str, dict[str, Any]] = {}
    for role in role_rows:
        object_name = str(role["object_name"])
        index = int(role["object_index"])
        trajectory = _trajectory_for_object(positions, index)
        move_mask = np.zeros(length, dtype=bool)
        lift_mask = np.zeros(length, dtype=bool)
        contact_mask = np.zeros(length, dtype=bool)
        if trajectory is not None:
            count = min(length, len(trajectory))
            initial = trajectory[0]
            delta = trajectory[:count] - initial
            move_mask[:count] = np.linalg.norm(delta, axis=1) > movement_threshold
            lift_mask[:count] = delta[:, 2] > lift_threshold
            if eef_positions is not None:
                contact_count = min(count, len(eef_positions))
                distances = np.linalg.norm(
                    eef_positions[:contact_count] - trajectory[:contact_count],
                    axis=1,
                )
                contact_mask[:contact_count] = distances < contact_threshold
        masks[object_name] = {
            "role": role,
            "move": move_mask,
            "lift": lift_mask,
            "contact": contact_mask,
        }
    return masks


def _current_mask_object(
    object_masks: Mapping[str, Mapping[str, Any]],
    timestep: int,
    kind: str,
) -> Mapping[str, Any] | None:
    candidates = []
    for item in object_masks.values():
        mask = np.asarray(item[kind], dtype=bool)
        if timestep < len(mask) and bool(mask[timestep]):
            candidates.append(item["role"])
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda role: (
            not bool(role.get("role_manipulated")),
            _safe_prompt_index(role),
            str(role["object_name"]),
        ),
    )[0]


def _active_step(
    flow_steps: Sequence[Mapping[str, Any]],
    flow_events: Sequence[Mapping[str, Any]],
    timestep: int,
) -> Mapping[str, Any] | None:
    prior = [
        step
        for step in flow_steps
        if not pd.isna(step.get("observed_start_timestep", np.nan))
        and float(step["observed_start_timestep"]) <= timestep
    ]
    if prior:
        return sorted(prior, key=lambda row: int(row.get("flow_step_index", 0)))[-1]
    return _next_event(flow_events, timestep, "observed_start_timestep")


def _next_event(
    rows: Sequence[Mapping[str, Any]],
    timestep: int,
    column: str,
) -> Mapping[str, Any] | None:
    for row in rows:
        value = row.get(column, np.nan)
        if not pd.isna(value) and float(value) > timestep:
            return row
    return None


def _next_manipulated_object(
    flow_events: Sequence[Mapping[str, Any]],
    timestep: int,
) -> Mapping[str, Any] | None:
    for row in flow_events:
        if row.get("step_type") != "manipulate_object":
            continue
        value = row.get("observed_start_timestep", np.nan)
        if pd.isna(value) or float(value) > timestep:
            return row
    return None


def _task_phase(
    current_contact: Mapping[str, Any] | None,
    current_moved: Mapping[str, Any] | None,
    current_lifted: Mapping[str, Any] | None,
    next_flow: Mapping[str, Any] | None,
) -> str:
    if current_lifted is not None:
        return "lift_or_transport"
    if current_moved is not None:
        return "move_or_transport"
    if current_contact is not None:
        return "contact"
    if next_flow is not None:
        return "approach"
    return "idle_or_post"


def _sorted_step_events(
    flow_steps: Sequence[Mapping[str, Any]],
    column: str,
) -> list[Mapping[str, Any]]:
    return sorted(
        [row for row in flow_steps if not pd.isna(row.get(column, np.nan))],
        key=lambda row: (float(row[column]), int(row.get("flow_step_index", 0))),
    )


def _policy_call_for_timestep(bundle: TraceBundle, timestep: int) -> int | None:
    table = bundle.policy_calls
    if table.empty or "policy_call_index" not in table:
        return None
    if {"env_timestep_start", "env_timestep_end"}.issubset(table.columns):
        rows = table.loc[
            (table["env_timestep_start"].astype(int) <= timestep)
            & (table["env_timestep_end"].astype(int) >= timestep)
        ]
        if not rows.empty:
            return int(rows.iloc[-1]["policy_call_index"])
    if "observation_timestep" in table:
        prior = table.loc[table["observation_timestep"].astype(int) <= timestep]
        if not prior.empty:
            return int(prior.iloc[-1]["policy_call_index"])
    return None


def _object_name(row: Mapping[str, Any] | None) -> str:
    return "" if row is None else str(row.get("object_name") or "")


def _target_object_name(row: Mapping[str, Any] | None) -> str:
    return "" if row is None else str(row.get("target_object_name") or "")


def _flow_step_index(row: Mapping[str, Any] | None) -> float:
    return np.nan if row is None else float(row.get("flow_step_index", np.nan))


def _active_object_name(
    current_contact: Mapping[str, Any] | None,
    current_moved: Mapping[str, Any] | None,
    current_lifted: Mapping[str, Any] | None,
    active_step: Mapping[str, Any] | None,
    next_manipulated: Mapping[str, Any] | None,
) -> str:
    for row in [current_lifted, current_moved, current_contact]:
        if row is not None and bool(row.get("role_manipulated", False)):
            return str(row["object_name"])
    if active_step is not None and active_step.get("step_type") == "manipulate_object":
        return str(active_step.get("object_name") or "")
    if next_manipulated is not None:
        return str(next_manipulated.get("object_name") or "")
    return ""


def _safe_prompt_index(row: Mapping[str, Any]) -> int:
    value = row.get("prompt_match_index", row.get("prompt_order", np.nan))
    return int(value) if not pd.isna(value) else 10**9


def _safe_onset(row: Mapping[str, Any], column: str) -> float:
    value = row.get(column, np.nan)
    return float(value) if not pd.isna(value) else float("inf")


def _nan_last(value: Any) -> float:
    return float(value) if not pd.isna(value) else float("inf")


def _first_available_onset(role: Mapping[str, Any]) -> float:
    values = [
        role.get("contact_onset_timestep", np.nan),
        role.get("movement_onset_timestep", np.nan),
        role.get("lift_onset_timestep", np.nan),
    ]
    valid = [float(value) for value in values if not pd.isna(value)]
    return min(valid) if valid else np.nan


def _summary_metrics(
    object_roles: pd.DataFrame,
    interaction_events: pd.DataFrame,
    flow_steps: pd.DataFrame,
    timestep_labels: pd.DataFrame,
) -> dict[str, Any]:
    episode_count = (
        int(object_roles["trace_id"].nunique())
        if "trace_id" in object_roles
        else 0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_count": episode_count,
        "object_role_row_count": int(len(object_roles)),
        "interaction_event_count": int(len(interaction_events)),
        "flow_step_count": int(len(flow_steps)),
        "timestep_label_count": int(len(timestep_labels)),
        "manipulated_object_count": int(object_roles.get("role_manipulated", pd.Series()).sum()),
        "receptacle_object_count": int(object_roles.get("role_receptacle", pd.Series()).sum()),
        "fixture_object_count": int(object_roles.get("role_fixture", pd.Series()).sum()),
    }


def _summary_text(
    object_roles: pd.DataFrame,
    flow_steps: pd.DataFrame,
    timestep_labels: pd.DataFrame,
) -> str:
    episodes = int(object_roles["trace_id"].nunique()) if "trace_id" in object_roles else 0
    return (
        f"{episodes} episode(s); {len(object_roles)} object role row(s); "
        f"{len(flow_steps)} inferred flow step(s); {len(timestep_labels)} timestep label row(s)."
    )


_LOCATION_CUES = {
    "in",
    "into",
    "inside",
    "on",
    "onto",
    "to",
    "at",
    "top",
    "bottom",
    "left",
    "right",
    "front",
    "back",
    "drawer",
    "compartment",
}

_FIXTURE_HEADS = {
    "basket",
    "cabinet",
    "caddy",
    "drawer",
    "microwave",
    "plate",
    "rack",
    "stove",
}


__all__ = [
    "ARTIFACT_TYPE",
    "ObjectFlowArtifact",
    "save_pi05_object_flow_artifact",
]
