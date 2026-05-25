"""Object and predicate context extraction for PI0.5 captures."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.pi05.context_capture_common import (
    _env_candidates,
    _first_existing_attr,
    _flatten_mapping_keys,
    _lookup_mapping_path,
    _names_from_value,
    _numeric_array,
    _scene_snapshot_sequence,
    _squeeze_single_env,
    _Status,
)
from vla_lens.pi05.context_capture_scene import (
    _object_pose_from_scene_snapshots,
    _object_records_from_scene_snapshots,
)
from vla_lens.pi05.context_capture_types import OBJECT_POSE_SUFFIXES, PREDICATE_WORDS
from vla_lens.traces import ArraySpec


def extract_object_context(
    observations: Sequence[Mapping[str, Any]],
    env: Any | None = None,
    *,
    scene_snapshots: Sequence[Mapping[str, Any]] | None = None,
    status: "_Status | None" = None,
) -> tuple[pd.DataFrame, dict[str, ArraySpec]]:
    """Extract object labels, poses, joint-ish state, and predicates."""

    status = status or _Status()
    arrays: dict[str, ArraySpec] = {}
    snapshots = _scene_snapshot_sequence(scene_snapshots)
    object_records = _object_records_from_scene_snapshots(snapshots)
    names = [str(record["object_name"]) for record in object_records]
    if not names:
        names = _object_names(env, observations)
        object_records = [
            {"object_index": index, "object_name": name, "source": "env_or_observation"}
            for index, name in enumerate(names)
        ]

    snapshot_pose_sources = _object_pose_from_scene_snapshots(snapshots, names)
    pose_sources = _object_pose_sources(
        env,
        observations,
        names,
        scene_pose_sources=snapshot_pose_sources,
    )
    for object_field, expected_last_dim in {"pos": 3, "quat": 4}.items():
        values = pose_sources.get(object_field)
        if values is None:
            status.missing("object", object_field, f"no object {object_field} values were exposed")
            continue
        array_name = f"scene_object_{object_field}"
        axes = ["object", "xyzw" if object_field == "quat" else "xyz"]
        if values.ndim == 3:
            axes = ["timestep", *axes]
        source = (
            "libero.mujoco_snapshot"
            if object_field in snapshot_pose_sources
            else "env_or_observation"
        )
        arrays[array_name] = ArraySpec(
            values.astype(np.float32, copy=False),
            axes,
            metadata={"object_names": list(names), "source": source},
        )
        status.available("object", object_field, source, shape=values.shape)
        if values.shape[-1] != expected_last_dim:
            status.missing(
                "object",
                f"{object_field}_expected_dim",
                f"expected last dimension {expected_last_dim}, got {values.shape[-1]}",
            )

    geom_center = pose_sources.get("geom_center")
    if geom_center is not None:
        axes = ["object", "xyz"]
        if geom_center.ndim == 3:
            axes = ["timestep", *axes]
        arrays["scene_object_geom_center"] = ArraySpec(
            geom_center.astype(np.float32, copy=False),
            axes,
            metadata={"object_names": list(names), "source": "libero.mujoco_snapshot"},
        )
        status.available("object", "geom_center", "libero.mujoco_snapshot", shape=geom_center.shape)

    bbox_world = pose_sources.get("bbox_world")
    if bbox_world is not None:
        axes = ["object", "bound", "xyz"]
        if bbox_world.ndim == 4:
            axes = ["timestep", *axes]
        arrays["scene_object_bbox_world"] = ArraySpec(
            bbox_world.astype(np.float32, copy=False),
            axes,
            metadata={"object_names": list(names), "source": "libero.mujoco_snapshot"},
        )
        status.available("object", "bbox_world", "libero.mujoco_snapshot", shape=bbox_world.shape)

    geom_count = pose_sources.get("geom_count")
    if geom_count is not None:
        axes = ["object"]
        if geom_count.ndim == 2:
            axes = ["timestep", *axes]
        arrays["scene_object_geom_count"] = ArraySpec(
            geom_count.astype(np.float32, copy=False),
            axes,
            metadata={"object_names": list(names), "source": "libero.mujoco_snapshot"},
        )
        status.available("object", "geom_count", "libero.mujoco_snapshot", shape=geom_count.shape)

    joints = pose_sources.get("joints")
    if joints is None:
        status.missing("object", "joints", "no object joint or qpos values were exposed")
    else:
        axes = ["object", "joint"]
        if joints.ndim == 3:
            axes = ["timestep", *axes]
        arrays["scene_object_joints"] = ArraySpec(
            joints.astype(np.float32, copy=False),
            axes,
            metadata={"object_names": list(names)},
        )
        status.available("object", "joints", "env_or_observation", shape=joints.shape)

    for record in object_records:
        if "scene_object_pos" in arrays:
            record["pos_array_id"] = "scene_object_pos"
        if "scene_object_quat" in arrays:
            record["quat_array_id"] = "scene_object_quat"
        if "scene_object_joints" in arrays:
            record["joints_array_id"] = "scene_object_joints"
        if "scene_object_geom_center" in arrays:
            record["geom_center_array_id"] = "scene_object_geom_center"
        if "scene_object_bbox_world" in arrays:
            record["bbox_array_id"] = "scene_object_bbox_world"
        if "scene_object_geom_count" in arrays:
            record["geom_count_array_id"] = "scene_object_geom_count"

    predicate_names, predicate_values = _predicate_values(env, observations)
    if predicate_values is None:
        status.missing("object", "predicates", "no predicate-ish scalar values were exposed")
    else:
        arrays["scene_predicates"] = ArraySpec(
            predicate_values.astype(np.float32, copy=False),
            ["timestep", "predicate"] if predicate_values.ndim == 2 else ["predicate"],
            metadata={"predicate_names": list(predicate_names)},
        )
        status.available("object", "predicates", "env_or_observation", shape=predicate_values.shape)

    return pd.DataFrame.from_records(object_records), arrays


def _object_names(env: Any | None, observations: Sequence[Mapping[str, Any]]) -> list[str]:
    candidates = _env_candidates(env)
    for attr in (
        "object_names",
        "obj_names",
        "movable_object_names",
        "fixture_names",
        "objects_dict",
        "fixtures_dict",
        "object_sites_dict",
    ):
        values = _first_existing_attr(candidates, (attr,))
        names = _names_from_value(values)
        if names:
            return names
    objects = _first_existing_attr(candidates, ("objects", "object_cfgs", "fixtures"))
    names = _names_from_value(objects)
    if names:
        return names
    return _object_names_from_observations(observations)


def _object_names_from_observations(observations: Sequence[Mapping[str, Any]]) -> list[str]:
    if not observations:
        return []
    keys = set(_flatten_mapping_keys(observations[0]))
    names: set[str] = set()
    for key in keys:
        if key.startswith("robot") or key.endswith("_image"):
            continue
        for suffixes in OBJECT_POSE_SUFFIXES.values():
            for suffix in suffixes:
                if key.endswith(suffix):
                    names.add(key[: -len(suffix)])
    return sorted(name for name in names if name)


def _object_pose_sources(
    env: Any | None,
    observations: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    *,
    scene_pose_sources: Mapping[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    sources: dict[str, np.ndarray] = dict(scene_pose_sources or {})
    obs_sources = _object_pose_from_observations(observations, names)
    for key, value in obs_sources.items():
        sources.setdefault(key, value)
    if names:
        env_sources = _object_pose_from_env(env, names)
        for key, value in env_sources.items():
            sources.setdefault(key, value)
    return sources


def _object_pose_from_observations(
    observations: Sequence[Mapping[str, Any]],
    names: Sequence[str],
) -> dict[str, np.ndarray]:
    if not observations or not names:
        return {}
    out: dict[str, np.ndarray] = {}
    for object_field, suffixes in OBJECT_POSE_SUFFIXES.items():
        per_timestep: list[np.ndarray] = []
        for obs in observations:
            per_object: list[np.ndarray] = []
            for name in names:
                value = None
                for suffix in suffixes:
                    value = _lookup_mapping_path(obs, f"{name}{suffix}")
                    if value is not None:
                        break
                array = _numeric_array(value)
                if array is None:
                    per_object = []
                    break
                per_object.append(np.ravel(_squeeze_single_env(array)))
            if not per_object:
                per_timestep = []
                break
            per_timestep.append(np.stack(per_object, axis=0))
        if per_timestep:
            try:
                out[object_field] = np.stack(per_timestep, axis=0)
            except ValueError:
                pass
    return out


def _object_pose_from_env(env: Any | None, names: Sequence[str]) -> dict[str, np.ndarray]:
    candidates = _env_candidates(env)
    pose_mapping = _first_existing_attr(candidates, ("object_poses", "obj_poses"))
    out: dict[str, np.ndarray] = {}
    if isinstance(pose_mapping, Mapping):
        for field in ("pos", "quat", "joints"):
            values: list[np.ndarray] = []
            for name in names:
                item = pose_mapping.get(name)
                if isinstance(item, Mapping):
                    value = item.get(field)
                else:
                    value = getattr(item, field, None) if item is not None else None
                array = _numeric_array(value)
                if array is None:
                    values = []
                    break
                values.append(np.ravel(array))
            if values:
                out[field] = np.stack(values, axis=0)
    return out


def _predicate_values(
    env: Any | None,
    observations: Sequence[Mapping[str, Any]],
) -> tuple[list[str], np.ndarray | None]:
    from_obs = _predicates_from_observations(observations)
    if from_obs[1] is not None:
        return from_obs
    return _predicates_from_env(env)


def _predicates_from_observations(
    observations: Sequence[Mapping[str, Any]],
) -> tuple[list[str], np.ndarray | None]:
    if not observations:
        return [], None
    keys = [
        key
        for key in _flatten_mapping_keys(observations[0])
        if any(word in key for word in PREDICATE_WORDS)
    ]
    values: list[np.ndarray] = []
    names: list[str] = []
    for key in keys:
        series: list[float] = []
        for obs in observations:
            value = _lookup_mapping_path(obs, key)
            array = _numeric_array(value)
            if array is None or array.size != 1:
                series = []
                break
            series.append(float(np.ravel(array)[0]))
        if series:
            names.append(key)
            values.append(np.asarray(series, dtype=np.float32))
    if not values:
        return [], None
    return names, np.stack(values, axis=1)


def _predicates_from_env(env: Any | None) -> tuple[list[str], np.ndarray | None]:
    candidates = _env_candidates(env)
    value = _first_existing_attr(candidates, ("predicates", "predicate_values"))
    if value is None:
        for candidate in candidates:
            for name in ("get_predicates", "_eval_predicates", "check_success", "_check_success"):
                method = getattr(candidate, name, None)
                if callable(method):
                    try:
                        value = method()
                    except Exception:
                        continue
                    break
            if value is not None:
                break
    if isinstance(value, Mapping):
        names = list(map(str, value.keys()))
        scalars = [_numeric_array(item) for item in value.values()]
        if all(item is not None and item.size == 1 for item in scalars):
            values = [float(np.ravel(item)[0]) for item in scalars]
            return names, np.asarray(values, dtype=np.float32)
    array = _numeric_array(value)
    if array is not None and array.size:
        flat = np.ravel(array).astype(np.float32)
        return [f"predicate_{index}" for index in range(flat.shape[0])], flat
    return [], None
