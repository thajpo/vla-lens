"""Post-process PI0.5 trace context into reusable interaction labels."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.analyzer import dataset_fingerprint
from vla_lens.artifacts import LensArtifact
from vla_lens.traces import TraceBundle, TraceDataset
from vla_lens.workbench import AnalysisRunSpec, save_analysis_run

ARTIFACT_TYPE = "pi05_interaction_metrics"
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class InteractionMetricsArtifact:
    artifact: LensArtifact
    episode_labels: pd.DataFrame
    object_metrics: pd.DataFrame


DEFAULT_THRESHOLDS: dict[str, float | int] = {
    "movement_distance_m": 0.025,
    "lift_z_m": 0.04,
    "contact_center_distance_m": 0.08,
    "consecutive_frames": 5,
    "first_tie_window": 3,
}


def save_pi05_interaction_metrics_artifact(
    dataset: TraceDataset,
    *,
    name: str = "PI0.5 interaction metrics",
    thresholds: Mapping[str, float | int] | None = None,
) -> InteractionMetricsArtifact:
    """Derive target/object interaction labels and save them as a dataset artifact."""
    threshold_values = {**DEFAULT_THRESHOLDS, **dict(thresholds or {})}
    split_sidecar = _read_probe_splits(dataset.root)
    episode_records: list[dict[str, Any]] = []
    object_records: list[dict[str, Any]] = []

    for bundle in dataset.bundles:
        result = _bundle_interaction_metrics(
            bundle,
            split_sidecar=split_sidecar,
            thresholds=threshold_values,
        )
        episode_records.append(result["episode"])
        object_records.extend(result["objects"])

    episode_labels = pd.DataFrame.from_records(episode_records)
    object_metrics = pd.DataFrame.from_records(object_records)
    fingerprint = dataset_fingerprint(dataset)
    artifact = LensArtifact.create(
        artifact_type=ARTIFACT_TYPE,
        name=name,
        group_id=ARTIFACT_TYPE,
        scope="dataset",
        selector={"episodes": "all", "source": "pi05_trace_context"},
        method={
            "workflow": "save_pi05_interaction_metrics_artifact",
            "schema_version": SCHEMA_VERSION,
            "dataset_fingerprint": fingerprint,
            "thresholds": threshold_values,
            "outputs": {
                "episode_labels": "",
                "object_metrics": "",
            },
        },
        metrics=_summary_metrics(episode_labels, object_metrics),
        display={
            "kind": ARTIFACT_TYPE,
            "summary": _summary_text(episode_labels),
            "primary_table": "interaction_episode_labels",
            "object_table": "interaction_object_metrics",
            "quality": _quality_summary(episode_labels),
            "interpretation_notes": [
                (
                    "These labels are derived from recorded scene/object state, "
                    "not manually annotated."
                ),
                "First-moved and first-lifted labels depend on fixed motion thresholds.",
                "Use metadata baselines before interpreting probe results as mechanistic evidence.",
            ],
        },
        tags=("pi05", "interaction", "labels", "postprocess"),
        source_trace_ids=tuple(sorted(str(value) for value in episode_labels["trace_id"])),
    )
    saved = dataset.save_artifact(artifact)
    artifact_dir = (dataset._dataset_artifact_root() / str(saved.path or "")).parent
    episode_path = artifact_dir / "interaction_episode_labels.parquet"
    object_path = artifact_dir / "interaction_object_metrics.parquet"
    episode_labels.to_parquet(episode_path, index=False)
    object_metrics.to_parquet(object_path, index=False)

    updated = LensArtifact(
        artifact_id=saved.artifact_id,
        artifact_type=saved.artifact_type,
        name=saved.name,
        group_id=saved.group_id,
        scope=saved.scope,
        selector=saved.selector,
        method={
            **dict(saved.method),
            "outputs": {
                "episode_labels": str(episode_path.relative_to(dataset.root)),
                "object_metrics": str(object_path.relative_to(dataset.root)),
            },
        },
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
    return InteractionMetricsArtifact(
        artifact=updated,
        episode_labels=episode_labels,
        object_metrics=object_metrics,
    )


def _bundle_interaction_metrics(
    bundle: TraceBundle,
    *,
    split_sidecar: pd.DataFrame,
    thresholds: Mapping[str, float | int],
) -> dict[str, Any]:
    episode = bundle.episode_record()
    sidecar = _sidecar_record(split_sidecar, bundle.manifest.trace_id)
    objects = _object_rows(bundle)
    positions, position_source = _object_positions(bundle)
    eef_positions = _eef_positions(bundle)
    target_objects, target_status = _parse_target_objects(bundle, objects)
    object_records = _object_metric_rows(
        bundle,
        objects,
        positions,
        eef_positions=eef_positions,
        target_objects=target_objects,
        thresholds=thresholds,
    )

    moved_candidates = [row for row in object_records if row["moved"]]
    lifted_candidates = [row for row in object_records if row["lifted"]]
    contacted_candidates = [row for row in object_records if row["contacted"]]
    first_moved = _first_event(moved_candidates, "movement_onset_timestep", thresholds)
    first_lifted = _first_event(lifted_candidates, "lift_onset_timestep", thresholds)
    first_contacted = _first_event(contacted_candidates, "contact_onset_timestep", thresholds)
    scene_family = _scene_family(str(episode.get("task_name") or bundle.manifest.prompt))
    task_verb = _task_verb(str(bundle.manifest.prompt or episode.get("task_name") or ""))

    record = {
        "trace_id": bundle.manifest.trace_id,
        "episode_id": bundle.manifest.episode_id,
        "dataset_id": episode.get("dataset_id") or sidecar.get("dataset_id") or "",
        "benchmark": sidecar.get("benchmark") or episode.get("benchmark") or bundle.manifest.env_id,
        "env_id": bundle.manifest.env_id,
        "task_id": bundle.manifest.task_id,
        "task_name": episode.get("task_name") or "",
        "prompt": bundle.manifest.prompt,
        "seed": sidecar.get("seed") if sidecar.get("seed") != "" else episode.get("seed"),
        "split": sidecar.get("split") or episode.get("split") or "",
        "outcome": bundle.manifest.outcome,
        "scene_family": scene_family,
        "layout_id": episode.get("layout_id") or episode.get("layout_episode_index") or "",
        "task_verb": task_verb,
        "target_objects": json.dumps(target_objects),
        "primary_target_object": target_objects[0] if target_objects else "",
        "target_parse_status": target_status,
        "first_moved_object": first_moved.get("object_name", ""),
        "first_moved_timestep": first_moved.get("movement_onset_timestep", np.nan),
        "first_moved_distance": first_moved.get("max_displacement", np.nan),
        "first_lifted_object": first_lifted.get("object_name", ""),
        "first_lifted_timestep": first_lifted.get("lift_onset_timestep", np.nan),
        "first_lifted_z_delta": first_lifted.get("max_z_delta", np.nan),
        "first_contacted_object": first_contacted.get("object_name", ""),
        "first_contact_timestep": first_contacted.get("contact_onset_timestep", np.nan),
        "contact_proxy_method": "eef_center_distance" if eef_positions is not None else "",
        "first_moved_is_target": _is_target(first_moved.get("object_name", ""), target_objects),
        "first_lifted_is_target": _is_target(first_lifted.get("object_name", ""), target_objects),
        "target_parse_failed": target_status == "failed",
        "multi_target_task": len(target_objects) > 1,
        "no_object_moved": not bool(moved_candidates),
        "ambiguous_first_moved": _ambiguous_first(
            moved_candidates,
            "movement_onset_timestep",
            thresholds,
        ),
        "no_object_lifted": not bool(lifted_candidates),
        "ambiguous_first_lifted": _ambiguous_first(
            lifted_candidates,
            "lift_onset_timestep",
            thresholds,
        ),
        "object_count": len(objects),
        "position_source": position_source,
    }
    return {"episode": record, "objects": object_records}


def _object_rows(bundle: TraceBundle) -> list[dict[str, Any]]:
    table = bundle.scene_state
    rows: list[dict[str, Any]] = []
    if table.empty:
        return rows
    for row in table.to_dict("records"):
        name = row.get("object_name") or row.get("name") or row.get("object_id")
        index = row.get("object_index")
        if name is None or index is None or pd.isna(index):
            continue
        kind = str(row.get("object_kind") or row.get("category") or "object")
        if kind == "site" or str(name).endswith("_region"):
            continue
        rows.append(
            {
                "object_index": int(index),
                "object_name": str(name),
                "object_base_name": _base_object_name(str(name)),
                "object_kind": kind,
            }
        )
    return sorted(rows, key=lambda item: int(item["object_index"]))


def _object_positions(bundle: TraceBundle) -> tuple[np.ndarray | None, str]:
    for name in ["scene_object_pos", "scene_object_geom_center"]:
        try:
            return np.asarray(bundle.array(name, mmap=True), dtype=np.float32), name
        except KeyError:
            continue
    try:
        poses = np.asarray(bundle.array("scene_object_poses", mmap=True), dtype=np.float32)
        if poses.ndim == 3 and poses.shape[-1] >= 3:
            return poses[..., :3], "scene_object_poses"
    except KeyError:
        pass
    return None, ""


def _eef_positions(bundle: TraceBundle) -> np.ndarray | None:
    for name in ["eef_pos", "robot_eef_pose"]:
        try:
            values = np.asarray(bundle.array(name, mmap=True), dtype=np.float32)
            if values.ndim >= 2 and values.shape[-1] >= 3:
                return values[..., :3]
        except KeyError:
            continue
    return None


def _object_metric_rows(
    bundle: TraceBundle,
    objects: Sequence[Mapping[str, Any]],
    positions: np.ndarray | None,
    *,
    eef_positions: np.ndarray | None,
    target_objects: Sequence[str],
    thresholds: Mapping[str, float | int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    consecutive = int(thresholds["consecutive_frames"])
    movement_threshold = float(thresholds["movement_distance_m"])
    lift_threshold = float(thresholds["lift_z_m"])
    contact_threshold = float(thresholds["contact_center_distance_m"])
    for obj in objects:
        index = int(obj["object_index"])
        trajectory = _trajectory_for_object(positions, index)
        if trajectory is None:
            displacement = np.array([], dtype=np.float32)
            xy_displacement = np.array([], dtype=np.float32)
            z_delta = np.array([], dtype=np.float32)
            movement_onset = None
            lift_onset = None
            contact_onset = None
            initial = final = np.full(3, np.nan, dtype=np.float32)
        else:
            initial = trajectory[0]
            final = trajectory[-1]
            delta = trajectory - initial
            displacement = np.linalg.norm(delta, axis=1)
            xy_displacement = np.linalg.norm(delta[:, :2], axis=1)
            z_delta = delta[:, 2]
            movement_onset = _first_consecutive(displacement > movement_threshold, consecutive)
            lift_onset = _first_consecutive(z_delta > lift_threshold, consecutive)
            if eef_positions is not None:
                count = min(len(eef_positions), len(trajectory))
                distances = np.linalg.norm(eef_positions[:count] - trajectory[:count], axis=1)
                contact_onset = _first_consecutive(distances < contact_threshold, consecutive=1)
            else:
                contact_onset = None
        records.append(
            {
                "trace_id": bundle.manifest.trace_id,
                "episode_id": bundle.manifest.episode_id,
                "object_index": index,
                "object_name": str(obj["object_name"]),
                "object_base_name": str(obj["object_base_name"]),
                "object_kind": str(obj["object_kind"]),
                "initial_x": _float(initial[0]),
                "initial_y": _float(initial[1]),
                "initial_z": _float(initial[2]),
                "final_x": _float(final[0]),
                "final_y": _float(final[1]),
                "final_z": _float(final[2]),
                "max_displacement": _float(np.nanmax(displacement))
                if displacement.size
                else np.nan,
                "max_xy_displacement": _float(np.nanmax(xy_displacement))
                if xy_displacement.size
                else np.nan,
                "max_z_delta": _float(np.nanmax(z_delta)) if z_delta.size else np.nan,
                "movement_onset_timestep": movement_onset if movement_onset is not None else np.nan,
                "lift_onset_timestep": lift_onset if lift_onset is not None else np.nan,
                "contact_onset_timestep": contact_onset if contact_onset is not None else np.nan,
                "moved": movement_onset is not None,
                "lifted": lift_onset is not None,
                "contacted": contact_onset is not None,
                "is_target_object": _is_target(str(obj["object_name"]), target_objects),
                "missing_position": trajectory is None,
            }
        )
    return records


def _trajectory_for_object(positions: np.ndarray | None, index: int) -> np.ndarray | None:
    if positions is None or positions.ndim != 3 or index >= positions.shape[1]:
        return None
    trajectory = np.asarray(positions[:, index, :3], dtype=np.float32)
    if trajectory.size == 0:
        return None
    return trajectory


def _first_event(
    rows: Sequence[Mapping[str, Any]],
    column: str,
    thresholds: Mapping[str, float | int],
) -> Mapping[str, Any]:
    valid = [row for row in rows if not pd.isna(row.get(column, np.nan))]
    if not valid:
        return {}
    return sorted(valid, key=lambda row: (float(row[column]), str(row["object_name"])))[0]


def _ambiguous_first(
    rows: Sequence[Mapping[str, Any]],
    column: str,
    thresholds: Mapping[str, float | int],
) -> bool:
    valid = sorted(
        [row for row in rows if not pd.isna(row.get(column, np.nan))],
        key=lambda row: float(row[column]),
    )
    if len(valid) < 2:
        return False
    return (
        float(valid[1][column]) - float(valid[0][column])
        <= float(thresholds["first_tie_window"])
    )


def _first_consecutive(mask: np.ndarray, consecutive: int) -> int | None:
    if mask.size == 0:
        return None
    run = 0
    for index, value in enumerate(mask.astype(bool)):
        run = run + 1 if value else 0
        if run >= consecutive:
            return index - consecutive + 1
    return None


def _parse_target_objects(
    bundle: TraceBundle,
    objects: Sequence[Mapping[str, Any]],
) -> tuple[list[str], str]:
    text = " ".join(
        str(value)
        for value in [
            bundle.manifest.prompt,
            bundle.manifest.task_id,
            bundle.manifest.metadata.get("task_name", ""),
        ]
        if value
    ).lower()
    targets: list[str] = []
    for obj in objects:
        base = str(obj["object_base_name"])
        words = base.replace("_", " ")
        if base and (base in _normalize_text(text) or words in text):
            targets.append(str(obj["object_name"]))
    targets = list(dict.fromkeys(targets))
    if not targets:
        return [], "failed"
    return targets, "multi" if len(targets) > 1 else "single"


def _scene_family(text: str) -> str:
    match = re.search(r"([A-Z_]+SCENE\d+)", text)
    if match:
        return match.group(1).lower()
    return ""


def _task_verb(text: str) -> str:
    lowered = text.lower()
    for verb in ["pick_up", "put", "stack", "open", "close", "turn_on", "turn_off"]:
        phrase = verb.replace("_", " ")
        if phrase in lowered or verb in lowered.replace(" ", "_"):
            return verb
    return lowered.split(" ", 1)[0] if lowered else ""


def _read_probe_splits(root: Path) -> pd.DataFrame:
    path = root / "probe_splits.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _sidecar_record(frame: pd.DataFrame, trace_id: str) -> dict[str, Any]:
    if frame.empty or "trace_id" not in frame:
        return {}
    rows = frame.loc[frame["trace_id"].astype(str) == trace_id]
    if rows.empty:
        return {}
    return rows.iloc[-1].fillna("").to_dict()


def _is_target(name: Any, targets: Sequence[str]) -> bool:
    if not name or not targets:
        return False
    base = _base_object_name(str(name))
    return any(base == _base_object_name(str(target)) for target in targets)


def _base_object_name(name: str) -> str:
    return re.sub(r"_[0-9]+$", "", str(name)).strip().lower()


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.lower())


def _summary_metrics(episode_labels: pd.DataFrame, object_metrics: pd.DataFrame) -> dict[str, Any]:
    total = max(1, len(episode_labels))
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_count": int(len(episode_labels)),
        "object_row_count": int(len(object_metrics)),
        "target_parse_success_rate": _rate(~episode_labels["target_parse_failed"], total),
        "no_object_moved_rate": _rate(episode_labels["no_object_moved"], total),
        "ambiguous_first_moved_rate": _rate(episode_labels["ambiguous_first_moved"], total),
        "no_object_lifted_rate": _rate(episode_labels["no_object_lifted"], total),
        "ambiguous_first_lifted_rate": _rate(episode_labels["ambiguous_first_lifted"], total),
    }


def _summary_text(episode_labels: pd.DataFrame) -> str:
    return (
        f"{len(episode_labels)} episode(s); "
        f"{int((~episode_labels['target_parse_failed']).sum())} target parse success; "
        f"{int((~episode_labels['no_object_moved']).sum())} with moved object."
    )


def _quality_summary(episode_labels: pd.DataFrame) -> dict[str, int]:
    keys = [
        "target_parse_failed",
        "multi_target_task",
        "no_object_moved",
        "ambiguous_first_moved",
        "no_object_lifted",
        "ambiguous_first_lifted",
    ]
    return {key: int(episode_labels[key].sum()) for key in keys if key in episode_labels}


def _rate(values: Sequence[Any], total: int) -> float:
    return float(np.asarray(values, dtype=bool).sum() / total)


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


__all__ = [
    "ARTIFACT_TYPE",
    "InteractionMetricsArtifact",
    "save_pi05_interaction_metrics_artifact",
]
