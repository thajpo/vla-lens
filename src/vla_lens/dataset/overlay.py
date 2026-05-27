"""VLA Lens overlay storage helpers for LeRobot dataset roots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vla_lens.capture.lerobot_v3 import (
    LEROBOT_ACTION,
    LEROBOT_EPISODE_INDEX,
    VLA_LENS_OVERLAY_DIR,
    VLA_LENS_OVERLAY_MANIFEST,
    VLA_LENS_OVERLAY_REFERENCES,
)
from vla_lens.capture.records import TraceRecord
from vla_lens.dataset.common import (
    OVERLAY_EPISODE_DIR,
    OVERLAY_ROOT_ARRAY_NAMES,
    OVERLAY_ROOT_ARRAY_PREFIXES,
    OVERLAY_SCHEMA_VERSION,
    TRACE_ACTION_ARRAY,
    TRACE_FRAME_PREFIX,
    _read_table,
    _write_json,
    _write_table,
)
from vla_lens.traces import ArraySpec, TraceBundle, TraceManifest


def _record_from_manifest(manifest: TraceManifest) -> TraceRecord:
    return TraceRecord(
        manifest=manifest,
        timesteps=pd.DataFrame({"timestep": np.arange(manifest.length, dtype=np.int32)}),
    )


def _overlay_bundle_for_episode(
    root: Path,
    refs: pd.DataFrame,
    episode_index: int,
) -> TraceBundle | None:
    overlay_path: Path | None = None
    if not refs.empty and LEROBOT_EPISODE_INDEX in refs:
        matches = refs.loc[refs[LEROBOT_EPISODE_INDEX].astype(int) == int(episode_index)]
        if not matches.empty and "overlay_path" in matches:
            overlay_path = root / str(matches.iloc[-1]["overlay_path"])
    if overlay_path is None:
        candidate = root / OVERLAY_EPISODE_DIR / f"episode_{episode_index:06d}"
        overlay_path = candidate if (candidate / TraceBundle.MANIFEST).exists() else None
    if overlay_path is None or not (overlay_path / TraceBundle.MANIFEST).exists():
        return None
    return TraceBundle.open(overlay_path)


def _write_overlay_bundle(
    root: Path,
    record: TraceRecord,
    *,
    episode_index: int,
    data_path: Path,
    overwrite: bool,
) -> TraceBundle:
    path = root / OVERLAY_EPISODE_DIR / f"episode_{episode_index:06d}"
    manifest = TraceManifest(
        trace_id=record.manifest.trace_id,
        episode_id=record.manifest.episode_id,
        task_id=record.manifest.task_id,
        prompt=record.manifest.prompt,
        model_id=record.manifest.model_id,
        env_id=record.manifest.env_id,
        robot_id=record.manifest.robot_id,
        outcome=record.manifest.outcome,
        length=record.manifest.length,
        schema_version=record.manifest.schema_version,
        metadata={
            **dict(record.manifest.metadata),
            "robot_dataset_format": "lerobot_v3",
            "lerobot_episode_index": int(episode_index),
            "lerobot_data_path": str(data_path),
        },
    )
    return TraceBundle.create(
        path,
        manifest=manifest,
        timesteps=record.timesteps,
        episode_arrays=_overlay_episode_arrays(record),
        model_arrays=record.model_arrays,
        policy_calls=record.policy_calls,
        generation_steps=record.generation_steps,
        streams=record.streams,
        token_spaces=record.token_spaces,
        tokens=record.tokens,
        robot_state=record.robot_state,
        scene_state=record.scene_state,
        camera_state=record.camera_state,
        evaluation=record.evaluation,
        image_preprocessing=record.image_preprocessing,
        prompt_metadata=record.prompt_metadata,
        action_normalization=_canonical_action_normalization(record.action_normalization),
        capture_request=record.capture_request,
        capture_plan=record.capture_plan,
        capture_report={
            **dict(record.capture_report),
            "dataset_format": "lerobot_v3_plus_vla_lens_overlay",
            "lerobot_episode_index": int(episode_index),
        },
        artifacts=record.artifacts,
        overwrite=overwrite,
    )


def _write_overlay_root(
    root: Path,
    record: TraceRecord,
    *,
    episode_index: int,
    overlay_bundle: TraceBundle,
) -> None:
    overlay_root = root / VLA_LENS_OVERLAY_DIR
    refs = _read_table(root / VLA_LENS_OVERLAY_REFERENCES)
    row = {
        LEROBOT_EPISODE_INDEX: int(episode_index),
        "trace_id": record.manifest.trace_id,
        "episode_id": record.manifest.episode_id,
        "length": int(record.manifest.length),
        "overlay_path": str(overlay_bundle.path.relative_to(root)),
    }
    if refs.empty:
        refs = pd.DataFrame.from_records([row])
    else:
        refs = refs.loc[refs[LEROBOT_EPISODE_INDEX].astype(int) != int(episode_index)]
        refs = pd.concat([refs, pd.DataFrame.from_records([row])], ignore_index=True)
    refs = refs.sort_values(LEROBOT_EPISODE_INDEX).reset_index(drop=True)
    _write_table(root / VLA_LENS_OVERLAY_REFERENCES, refs)
    _write_json(
        root / VLA_LENS_OVERLAY_MANIFEST,
        {
            "overlay_schema_version": OVERLAY_SCHEMA_VERSION,
            "robot_dataset_format": "lerobot_v3",
            "overlay_root": str(VLA_LENS_OVERLAY_DIR),
            "episodes": int(len(refs)),
        },
    )
    overlay_root.mkdir(parents=True, exist_ok=True)


def _overlay_episode_arrays(record: TraceRecord) -> dict[str, ArraySpec]:
    arrays: dict[str, ArraySpec] = {}
    for name, spec in record.episode_arrays.items():
        if name == TRACE_ACTION_ARRAY:
            continue
        if str(name).startswith(TRACE_FRAME_PREFIX):
            continue
        if name in OVERLAY_ROOT_ARRAY_NAMES:
            continue
        if str(name).startswith(OVERLAY_ROOT_ARRAY_PREFIXES):
            continue
        arrays[str(name)] = spec
    return arrays


def _canonical_action_normalization(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    if "unnormalized_action_array_ref" in out:
        out["unnormalized_action_array_ref"] = LEROBOT_ACTION
    return out


def _overlay_table(bundle: TraceBundle | None, name: str) -> pd.DataFrame:
    if bundle is None:
        return pd.DataFrame()
    return getattr(bundle, name)
