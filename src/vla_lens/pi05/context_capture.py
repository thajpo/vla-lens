"""Cheap LIBERO / LeRobot context extraction helpers.

These helpers intentionally depend only on NumPy / pandas plus the local
``ArraySpec`` type.  Real LIBERO runs expose more state through robosuite and
MuJoCo, but tests and lightweight smoke runs often do not have those packages
or simulators available.  Missing optional fields are therefore represented as
status rows instead of exceptions.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.traces import ArraySpec


@dataclass(frozen=True, slots=True)
class ContextCaptureResult:
    """Normalized context payload ready to merge into a trace capture."""

    arrays: Mapping[str, ArraySpec] = dataclass_field(default_factory=dict)
    tables: Mapping[str, pd.DataFrame] = dataclass_field(default_factory=dict)

    @property
    def availability(self) -> pd.DataFrame:
        return self.tables.get("context_availability", pd.DataFrame())

    @property
    def unavailable(self) -> pd.DataFrame:
        availability = self.availability
        if availability.empty or "available" not in availability:
            return pd.DataFrame()
        return availability.loc[~availability["available"].astype(bool)].reset_index(drop=True)


ROBOT_FIELD_CANDIDATES: Mapping[str, tuple[str, ...]] = {
    "robot_joint_pos": (
        "robot_state.joints.pos",
        "robot0_joint_pos",
        "robot_joint_pos",
        "joint_pos",
        "joint_positions",
        "observation.robot_state.joints.pos",
        "observation.robot0_joint_pos",
        "observation.joint_positions",
    ),
    "robot_joint_vel": (
        "robot_state.joints.vel",
        "robot0_joint_vel",
        "robot_joint_vel",
        "joint_vel",
        "joint_velocities",
        "observation.robot_state.joints.vel",
        "observation.robot0_joint_vel",
        "observation.joint_velocities",
    ),
    "eef_pos": (
        "robot_state.eef.pos",
        "robot0_eef_pos",
        "eef_pos",
        "end_effector_pos",
        "observation.robot_state.eef.pos",
        "observation.robot0_eef_pos",
        "observation.eef_pos",
    ),
    "eef_quat": (
        "robot_state.eef.quat",
        "robot0_eef_quat",
        "eef_quat",
        "end_effector_quat",
        "observation.robot_state.eef.quat",
        "observation.robot0_eef_quat",
        "observation.eef_quat",
    ),
    "eef_mat": (
        "robot_state.eef.mat",
        "robot0_eef_mat",
        "robot0_eef_rot_mat",
        "eef_mat",
        "eef_rot_mat",
        "end_effector_mat",
        "observation.robot_state.eef.mat",
        "observation.robot0_eef_mat",
        "observation.eef_mat",
    ),
    "gripper_qpos": (
        "robot_state.gripper.qpos",
        "robot0_gripper_qpos",
        "gripper_qpos",
        "gripper_pos",
        "observation.robot_state.gripper.qpos",
        "observation.robot0_gripper_qpos",
        "observation.gripper_qpos",
    ),
    "gripper_qvel": (
        "robot_state.gripper.qvel",
        "robot0_gripper_qvel",
        "gripper_qvel",
        "gripper_vel",
        "observation.robot_state.gripper.qvel",
        "observation.robot0_gripper_qvel",
        "observation.gripper_qvel",
    ),
}

ENV_METADATA_FIELDS: Mapping[str, tuple[str, ...]] = {
    "task_id": ("task_id", "_task_id", "libero_task_id"),
    "task_name": ("task_name", "name", "_task_name"),
    "task_description": ("task_description", "language_instruction", "instruction", "language"),
    "layout_id": ("layout_id", "_layout_id", "problem_folder"),
    "bddl_file": ("bddl_file", "bddl_file_name", "_bddl_file_name"),
    "seed": ("seed", "_seed", "np_random_seed"),
    "reset_state": ("reset_state", "reset_states", "_reset_state", "_reset_states"),
    "init_state": ("init_state", "initial_state", "_init_state", "_initial_state"),
}

OBJECT_POSE_SUFFIXES = {
    "pos": ("_pos", "_position", "_xpos"),
    "quat": ("_quat", "_xquat"),
    "joints": ("_joint", "_joints", "_qpos", "_joint_qpos"),
}

PREDICATE_WORDS = (
    "predicate",
    "success",
    "contact",
    "grasp",
    "inside",
    "on_",
    "in_",
    "stack",
    "lift",
)


def capture_libero_context(
    observations: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    env: Any | None = None,
    scene_snapshots: Sequence[Mapping[str, Any]] | None = None,
    camera_snapshots: Sequence[Mapping[str, Any]] | None = None,
) -> ContextCaptureResult:
    """Extract cheap environment context from observations and an optional env.

    The returned ``arrays`` mapping can be passed to ``TraceBundle.create`` as
    episode arrays.  The returned tables are plain pandas frames intended for
    richer capture manifests or future .vlatrace table slots.
    """

    obs_sequence = _observation_sequence(observations)
    scene_snapshot_sequence = _scene_snapshot_sequence(scene_snapshots)
    arrays: dict[str, ArraySpec] = {}
    tables: dict[str, pd.DataFrame] = {}
    status = _Status()

    arrays.update(extract_robot_arrays(obs_sequence, status=status))
    env_metadata, env_arrays = extract_env_metadata(env, status=status)
    tables["episode_context"] = env_metadata
    arrays.update(env_arrays)

    object_table, object_arrays = extract_object_context(
        obs_sequence,
        env,
        scene_snapshots=scene_snapshot_sequence,
        status=status,
    )
    tables["objects"] = object_table
    arrays.update(object_arrays)

    camera_table, camera_arrays = extract_camera_context(
        obs_sequence,
        env,
        camera_snapshots=camera_snapshots,
        status=status,
    )
    tables["cameras"] = camera_table
    arrays.update(camera_arrays)

    tables["context_availability"] = status.frame()
    return ContextCaptureResult(arrays=arrays, tables=tables)


def capture_scene_snapshot(env: Any | None) -> dict[str, Any]:
    """Capture cheap scene object state directly from a nested LIBERO / MuJoCo env."""

    candidates = _env_candidates(env)
    sim = _first_existing_attr(candidates, ("sim", "_sim"))
    if sim is None:
        return {"objects": [], "source": "", "reason": "env.sim unavailable"}

    objects: list[dict[str, Any]] = []
    for descriptor in _scene_object_descriptors(candidates, sim):
        pose = _sample_scene_object_pose(sim, descriptor)
        if pose is None:
            continue
        objects.append({**descriptor, **pose})

    reason = "" if objects else "no MuJoCo object bodies or sites were resolved"
    return {"objects": objects, "source": "libero.mujoco", "reason": reason}


def capture_camera_snapshot(
    env: Any | None,
    observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture camera calibration at the current simulator timestep.

    Wrist / eye-in-hand cameras move with the robot, so camera extrinsics must be
    sampled at the same pre-action time as frames and object state.
    """

    observations = _observation_sequence(observation)
    names = _camera_names(env, observations)
    if not names:
        return {"cameras": [], "source": "", "reason": "no cameras were exposed"}

    candidates = _env_candidates(env)
    sim = _first_existing_attr(candidates, ("sim", "_sim"))
    if sim is None:
        return {"cameras": [], "source": "", "reason": "env.sim unavailable"}

    camera_utils = _robosuite_camera_utils()
    object_descriptors = _scene_object_descriptors(candidates, sim)
    cameras: list[dict[str, Any]] = []
    for name in names:
        height, width = _camera_resolution(name, observations, env)
        intrinsic, intrinsic_reason = _camera_intrinsic(
            camera_utils,
            sim,
            name,
            height=height,
            width=width,
        )
        extrinsic, extrinsic_reason = _camera_extrinsic(camera_utils, sim, name)
        object_bboxes = _camera_object_bboxes_from_segmentation(
            sim,
            name,
            height=height,
            width=width,
            object_descriptors=object_descriptors,
        )
        cameras.append(
            {
                "camera_name": name,
                "height": height,
                "width": width,
                "intrinsic": intrinsic,
                "intrinsic_reason": intrinsic_reason,
                "extrinsic": extrinsic,
                "extrinsic_reason": extrinsic_reason,
                "object_bboxes": object_bboxes,
            }
        )

    return {"cameras": cameras, "source": "robosuite.camera_utils", "reason": ""}


def extract_robot_arrays(
    observations: Sequence[Mapping[str, Any]],
    *,
    status: "_Status | None" = None,
) -> dict[str, ArraySpec]:
    """Extract robot proprioception arrays from formatted observations."""

    status = status or _Status()
    arrays: dict[str, ArraySpec] = {}
    if not observations:
        for field in ROBOT_FIELD_CANDIDATES:
            status.missing("robot", field, "no observations were provided")
        return arrays

    for field, candidates in ROBOT_FIELD_CANDIDATES.items():
        values, source = _stack_observation_field(observations, candidates)
        if values is None:
            if field == "eef_mat" and "eef_quat" in arrays:
                mat = _quat_array_to_matrix(arrays["eef_quat"].array)
                arrays["eef_mat"] = ArraySpec(
                    mat.astype(np.float32),
                    ["timestep", "row", "col"],
                    metadata={"source": "derived:eef_quat"},
                )
                status.available("robot", "eef_mat", "derived:eef_quat", shape=mat.shape)
                continue
            status.missing(
                "robot",
                field,
                f"none of these observation keys were present: {', '.join(candidates)}",
            )
            continue
        axes = _robot_axes(field, values)
        arrays[field] = ArraySpec(
            values.astype(np.float32, copy=False),
            axes,
            metadata={"source": str(source)},
        )
        status.available("robot", field, str(source), shape=values.shape)

    return arrays


def extract_env_metadata(
    env: Any | None,
    *,
    status: "_Status | None" = None,
) -> tuple[pd.DataFrame, dict[str, ArraySpec]]:
    """Extract reset/init state and task/layout metadata from wrapper objects."""

    status = status or _Status()
    arrays: dict[str, ArraySpec] = {}
    records: list[dict[str, Any]] = []
    if env is None:
        for field in ENV_METADATA_FIELDS:
            status.missing("env", field, "env is not available")
            records.append(_metadata_row(field, available=False, reason="env is not available"))
        return pd.DataFrame.from_records(records), arrays

    candidates = _env_candidates(env)
    for field, attr_names in ENV_METADATA_FIELDS.items():
        found = _first_attr(candidates, attr_names)
        if found is None:
            reason = f"no wrapper attribute found among: {', '.join(attr_names)}"
            status.missing("env", field, reason)
            records.append(
                _metadata_row(
                    field,
                    available=False,
                    reason=reason,
                )
            )
            continue

        value, source = found
        array = _numeric_array(value)
        if field in {"reset_state", "init_state"} and array is not None:
            name = f"scene_{field}"
            arrays[name] = ArraySpec(
                array.astype(np.float32, copy=False),
                _generic_axes(array, trailing_prefix="state"),
                metadata={"source": source},
            )
            records.append(
                _metadata_row(
                    field,
                    available=True,
                    source=source,
                    value=None,
                    array_name=name,
                    shape=array.shape,
                )
            )
        else:
            records.append(_metadata_row(field, available=True, source=source, value=value))
        status.available("env", field, source, shape=None if array is None else array.shape)

    return pd.DataFrame.from_records(records), arrays


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


def extract_camera_context(
    observations: Sequence[Mapping[str, Any]],
    env: Any | None = None,
    *,
    camera_snapshots: Sequence[Mapping[str, Any]] | None = None,
    status: "_Status | None" = None,
) -> tuple[pd.DataFrame, dict[str, ArraySpec]]:
    """Extract camera labels, image sizes, and robosuite calibration when present."""

    status = status or _Status()
    snapshot_sequence = _camera_snapshot_sequence(camera_snapshots)
    if snapshot_sequence:
        return _extract_camera_context_from_snapshots(snapshot_sequence, status=status)

    arrays: dict[str, ArraySpec] = {}
    names = _camera_names(env, observations)
    resolutions = [_camera_resolution(name, observations, env) for name in names]
    camera_utils = _robosuite_camera_utils()
    sim = _first_existing_attr(_env_candidates(env), ("sim", "_sim")) if env is not None else None

    intrinsics: list[np.ndarray] = []
    extrinsics: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        height, width = resolutions[index]
        intrinsic, intrinsic_reason = _camera_intrinsic(
            camera_utils,
            sim,
            name,
            height=height,
            width=width,
        )
        extrinsic, extrinsic_reason = _camera_extrinsic(camera_utils, sim, name)
        records.append(
            {
                "camera_index": index,
                "camera_name": name,
                "height": height,
                "width": width,
                "intrinsics_available": intrinsic is not None,
                "intrinsics_reason": intrinsic_reason,
                "extrinsics_available": extrinsic is not None,
                "extrinsics_reason": extrinsic_reason,
            }
        )
        if intrinsic is not None:
            intrinsics.append(intrinsic)
        if extrinsic is not None:
            extrinsics.append(extrinsic)

    if names:
        resolution_array = np.asarray(
            [[-1 if item is None else int(item) for item in pair] for pair in resolutions],
            dtype=np.int32,
        )
        arrays["camera_resolution"] = ArraySpec(
            resolution_array,
            ["camera", "height_width"],
            metadata={"camera_names": list(names)},
        )
        status.available("camera", "names", "env_or_observation", shape=(len(names),))
        status.available("camera", "resolution", "env_or_observation", shape=resolution_array.shape)
    else:
        status.missing("camera", "names", "no camera names or image observations were exposed")
        status.missing("camera", "resolution", "no cameras were exposed")

    if len(intrinsics) == len(names) and intrinsics:
        array = np.stack(intrinsics).astype(np.float32)
        arrays["camera_intrinsics"] = ArraySpec(
            array,
            ["camera", "row", "col"],
            metadata={"camera_names": list(names)},
        )
        status.available("camera", "intrinsics", "robosuite.camera_utils", shape=array.shape)
    else:
        reason = "robosuite camera_utils unavailable or env.sim missing"
        status.missing("camera", "intrinsics", reason)

    if len(extrinsics) == len(names) and extrinsics:
        array = np.stack(extrinsics).astype(np.float32)
        arrays["camera_extrinsics"] = ArraySpec(
            array,
            ["camera", "row", "col"],
            metadata={"camera_names": list(names)},
        )
        status.available("camera", "extrinsics", "robosuite.camera_utils", shape=array.shape)
    else:
        reason = "robosuite camera_utils unavailable or env.sim missing"
        status.missing("camera", "extrinsics", reason)

    return pd.DataFrame.from_records(records), arrays


def _extract_camera_context_from_snapshots(
    camera_snapshots: Sequence[Mapping[str, Any]],
    *,
    status: "_Status",
) -> tuple[pd.DataFrame, dict[str, ArraySpec]]:
    arrays: dict[str, ArraySpec] = {}
    names: list[str] = []
    object_names: list[str] = []
    for snapshot in camera_snapshots:
        for raw in _snapshot_cameras(snapshot):
            name = raw.get("camera_name")
            if name is None:
                continue
            key = str(name)
            if key not in names:
                names.append(key)
            for bbox in _snapshot_object_bboxes(raw):
                object_name = bbox.get("object_name")
                if object_name is None:
                    continue
                object_key = str(object_name)
                if object_key not in object_names:
                    object_names.append(object_key)

    if not names:
        status.missing("camera", "names", "camera snapshots had no camera rows")
        status.missing("camera", "resolution", "camera snapshots had no camera rows")
        status.missing("camera", "intrinsics", "camera snapshots had no camera rows")
        status.missing("camera", "extrinsics", "camera snapshots had no camera rows")
        return pd.DataFrame(), arrays

    name_to_index = {name: index for index, name in enumerate(names)}
    resolution = np.full((len(names), 2), -1, dtype=np.int32)
    intrinsics = np.full((len(names), 3, 3), np.nan, dtype=np.float32)
    extrinsics = np.full((len(camera_snapshots), len(names), 4, 4), np.nan, dtype=np.float32)
    intrinsic_reasons: dict[str, str] = {name: "" for name in names}
    extrinsic_reasons: dict[str, str] = {name: "" for name in names}
    has_intrinsic = {name: False for name in names}
    has_extrinsic = {name: False for name in names}
    object_bbox = np.full(
        (len(camera_snapshots), len(names), len(object_names), 4),
        np.nan,
        dtype=np.float32,
    )
    object_visible = np.zeros(
        (len(camera_snapshots), len(names), len(object_names)),
        dtype=np.uint8,
    )
    object_name_to_index = {name: index for index, name in enumerate(object_names)}

    for timestep, snapshot in enumerate(camera_snapshots):
        for raw in _snapshot_cameras(snapshot):
            name = raw.get("camera_name")
            if name is None:
                continue
            key = str(name)
            index = name_to_index.get(key)
            if index is None:
                continue
            height = _optional_int(raw.get("height"))
            width = _optional_int(raw.get("width"))
            if height is not None and width is not None:
                resolution[index] = [height, width]
            intrinsic = _numeric_matrix(raw.get("intrinsic"), 3, 3)
            if intrinsic is not None:
                intrinsics[index] = intrinsic
                has_intrinsic[key] = True
            elif not intrinsic_reasons[key]:
                intrinsic_reasons[key] = str(raw.get("intrinsic_reason") or "")
            extrinsic = _numeric_matrix(raw.get("extrinsic"), 4, 4)
            if extrinsic is not None:
                extrinsics[timestep, index] = extrinsic
                has_extrinsic[key] = True
            elif not extrinsic_reasons[key]:
                extrinsic_reasons[key] = str(raw.get("extrinsic_reason") or "")
            for bbox in _snapshot_object_bboxes(raw):
                object_name = bbox.get("object_name")
                object_index = object_name_to_index.get(str(object_name))
                if object_index is None:
                    continue
                pixel_bbox = _numeric_vector(bbox.get("bbox_pixel_xyxy"), 4)
                if pixel_bbox is None:
                    continue
                object_bbox[timestep, index, object_index] = pixel_bbox
                object_visible[timestep, index, object_index] = 1

    records = []
    for index, name in enumerate(names):
        records.append(
            {
                "camera_index": index,
                "camera_name": name,
                "height": int(resolution[index, 0]),
                "width": int(resolution[index, 1]),
                "intrinsics_available": bool(has_intrinsic[name]),
                "intrinsics_reason": intrinsic_reasons[name],
                "extrinsics_available": bool(has_extrinsic[name]),
                "extrinsics_reason": extrinsic_reasons[name],
                "extrinsics_time_varying": True,
            }
        )

    arrays["camera_resolution"] = ArraySpec(
        resolution,
        ["camera", "height_width"],
        metadata={"camera_names": list(names)},
    )
    status.available("camera", "names", "camera_snapshots", shape=(len(names),))
    status.available("camera", "resolution", "camera_snapshots", shape=resolution.shape)

    if all(has_intrinsic.values()):
        arrays["camera_intrinsics"] = ArraySpec(
            intrinsics,
            ["camera", "row", "col"],
            metadata={"camera_names": list(names)},
        )
        status.available("camera", "intrinsics", "camera_snapshots", shape=intrinsics.shape)
    else:
        missing = [name for name, available in has_intrinsic.items() if not available]
        status.missing("camera", "intrinsics", f"missing intrinsics for cameras: {missing}")

    if all(has_extrinsic.values()):
        arrays["camera_extrinsics"] = ArraySpec(
            extrinsics,
            ["timestep", "camera", "row", "col"],
            metadata={"camera_names": list(names), "time_aligned": True},
        )
        status.available("camera", "extrinsics", "camera_snapshots", shape=extrinsics.shape)
    else:
        missing = [name for name, available in has_extrinsic.items() if not available]
        status.missing("camera", "extrinsics", f"missing extrinsics for cameras: {missing}")

    if object_names and bool(np.any(object_visible)):
        arrays["camera_object_bbox"] = ArraySpec(
            object_bbox,
            ["timestep", "camera", "object", "bbox_xyxy"],
            metadata={
                "camera_names": list(names),
                "object_names": list(object_names),
                "bbox_format": "pixel_xyxy_exclusive",
                "source": "robosuite.segmentation",
            },
        )
        arrays["camera_object_visible"] = ArraySpec(
            object_visible,
            ["timestep", "camera", "object"],
            metadata={
                "camera_names": list(names),
                "object_names": list(object_names),
                "source": "robosuite.segmentation",
            },
        )
        status.available(
            "camera",
            "object_bbox",
            "robosuite.segmentation",
            shape=object_bbox.shape,
        )
    else:
        status.missing("camera", "object_bbox", "camera segmentation bboxes unavailable")

    return pd.DataFrame.from_records(records), arrays


class _Status:
    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def available(
        self,
        component: str,
        field: str,
        source: str,
        *,
        shape: Sequence[int] | None = None,
    ) -> None:
        self._records.append(
            {
                "component": component,
                "field": field,
                "available": True,
                "reason": "",
                "source": source,
                "shape": _shape_text(shape),
            }
        )

    def missing(self, component: str, field: str, reason: str) -> None:
        self._records.append(
            {
                "component": component,
                "field": field,
                "available": False,
                "reason": reason,
                "source": "",
                "shape": "",
            }
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self._records)


def _observation_sequence(
    observations: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    if observations is None:
        return []
    if isinstance(observations, Mapping):
        return [observations]
    return [item for item in observations if isinstance(item, Mapping)]


def _stack_observation_field(
    observations: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> tuple[np.ndarray | None, str | None]:
    source: str | None = None
    values: list[np.ndarray] = []
    for obs in observations:
        found = None
        for key in keys:
            value = _lookup_mapping_path(obs, key)
            if value is not None:
                found = value
                source = key
                break
        if found is None:
            return None, None
        array = _numeric_array(found)
        if array is None:
            return None, None
        values.append(_squeeze_single_env(array))
    try:
        return np.stack(values, axis=0), source
    except ValueError:
        return None, None


def _lookup_mapping_path(mapping: Mapping[str, Any], path: str) -> Any | None:
    if path in mapping:
        return mapping[path]
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _numeric_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        return None
    return array


def _squeeze_single_env(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim >= 2 and value.shape[0] == 1:
        return value[0]
    return value


def _robot_axes(field: str, values: np.ndarray) -> list[str]:
    if field == "eef_mat":
        return ["timestep", "row", "col"]
    if field in {"eef_pos"}:
        return ["timestep", "xyz"]
    if field == "eef_quat":
        return ["timestep", "xyzw"]
    if field.startswith("gripper_"):
        return ["timestep", "gripper_joint"]
    return ["timestep", "joint"]


def _quat_array_to_matrix(quats: np.ndarray) -> np.ndarray:
    quats = np.asarray(quats, dtype=np.float64)
    flat = quats.reshape(-1, quats.shape[-1])
    mats = np.stack([_quat_to_matrix(quat) for quat in flat], axis=0)
    return mats.reshape(*quats.shape[:-1], 3, 3)


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    if quat.shape[-1] != 4:
        return np.full((3, 3), np.nan, dtype=np.float32)
    x, y, z, w = [float(item) for item in quat]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return np.full((3, 3), np.nan, dtype=np.float32)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _env_candidates(env: Any | None) -> list[Any]:
    if env is None:
        return []
    candidates: list[Any] = []
    seen: set[int] = set()
    stack = [env]
    while stack:
        item = stack.pop(0)
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        candidates.append(item)
        for attr in ("unwrapped", "env", "_env", "gym_env", "base_env"):
            try:
                child = getattr(item, attr)
            except Exception:
                continue
            if child is not item:
                stack.append(child)
        for attr in ("envs",):
            try:
                children = getattr(item, attr)
            except Exception:
                continue
            if isinstance(children, Mapping):
                stack.extend(children.values())
            elif isinstance(children, Sequence) and not isinstance(children, (str, bytes)):
                stack.extend(children)
    return candidates


def _first_attr(candidates: Sequence[Any], names: Sequence[str]) -> tuple[Any, str] | None:
    for candidate in candidates:
        for name in names:
            try:
                value = getattr(candidate, name)
            except Exception:
                continue
            if value is not None and not callable(value):
                return value, f"{type(candidate).__name__}.{name}"
    return None


def _first_existing_attr(candidates: Sequence[Any], names: Sequence[str]) -> Any | None:
    found = _first_attr(candidates, names)
    return None if found is None else found[0]


def _metadata_row(
    field: str,
    *,
    available: bool,
    source: str = "",
    value: Any = None,
    reason: str = "",
    array_name: str = "",
    shape: Sequence[int] | None = None,
) -> dict[str, Any]:
    return {
        "field": field,
        "available": available,
        "source": source,
        "value": _scalar_text(value),
        "array_name": array_name,
        "shape": _shape_text(shape),
        "reason": reason,
    }


def _scene_snapshot_sequence(
    scene_snapshots: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    if scene_snapshots is None:
        return []
    return [snapshot for snapshot in scene_snapshots if isinstance(snapshot, Mapping)]


def _camera_snapshot_sequence(
    camera_snapshots: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    if camera_snapshots is None:
        return []
    return [snapshot for snapshot in camera_snapshots if isinstance(snapshot, Mapping)]


def _object_records_from_scene_snapshots(
    scene_snapshots: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for snapshot in scene_snapshots:
        for raw in _snapshot_objects(snapshot):
            name = raw.get("object_name")
            if name is None:
                continue
            key = str(name)
            if key in records:
                continue
            records[key] = {
                "object_index": len(records),
                "object_name": key,
                "object_kind": str(raw.get("object_kind") or ""),
                "source": str(raw.get("source") or "libero.mujoco_snapshot"),
                "body_id": _optional_int(raw.get("body_id")),
                "body_name": str(raw.get("body_name") or ""),
                "site_name": str(raw.get("site_name") or ""),
            }
    return list(records.values())


def _snapshot_objects(snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    objects = snapshot.get("objects")
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)):
        return []
    return [item for item in objects if isinstance(item, Mapping)]


def _snapshot_cameras(snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cameras = snapshot.get("cameras")
    if not isinstance(cameras, Sequence) or isinstance(cameras, (str, bytes)):
        return []
    return [item for item in cameras if isinstance(item, Mapping)]


def _snapshot_object_bboxes(camera_snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    bboxes = camera_snapshot.get("object_bboxes")
    if not isinstance(bboxes, Sequence) or isinstance(bboxes, (str, bytes)):
        return []
    return [item for item in bboxes if isinstance(item, Mapping)]


def _object_pose_from_scene_snapshots(
    scene_snapshots: Sequence[Mapping[str, Any]],
    names: Sequence[str],
) -> dict[str, np.ndarray]:
    if not scene_snapshots or not names:
        return {}
    positions = np.full((len(scene_snapshots), len(names), 3), np.nan, dtype=np.float32)
    quats = np.full((len(scene_snapshots), len(names), 4), np.nan, dtype=np.float32)
    geom_centers = np.full((len(scene_snapshots), len(names), 3), np.nan, dtype=np.float32)
    bboxes = np.full((len(scene_snapshots), len(names), 2, 3), np.nan, dtype=np.float32)
    geom_counts = np.full((len(scene_snapshots), len(names)), np.nan, dtype=np.float32)
    has_pos = False
    has_quat = False
    has_geom_center = False
    has_bbox = False
    has_geom_count = False
    name_to_index = {str(name): index for index, name in enumerate(names)}
    for timestep, snapshot in enumerate(scene_snapshots):
        for raw in _snapshot_objects(snapshot):
            index = name_to_index.get(str(raw.get("object_name")))
            if index is None:
                continue
            pos = _numeric_vector(raw.get("pos"), 3)
            if pos is not None:
                positions[timestep, index] = pos
                has_pos = True
            quat = _numeric_vector(raw.get("quat"), 4)
            if quat is not None:
                quats[timestep, index] = quat
                has_quat = True
            geom_center = _numeric_vector(raw.get("geom_center"), 3)
            if geom_center is not None:
                geom_centers[timestep, index] = geom_center
                has_geom_center = True
            bbox_min = _numeric_vector(raw.get("bbox_min"), 3)
            bbox_max = _numeric_vector(raw.get("bbox_max"), 3)
            if bbox_min is not None and bbox_max is not None:
                bboxes[timestep, index, 0] = bbox_min
                bboxes[timestep, index, 1] = bbox_max
                has_bbox = True
            geom_count = _optional_int(raw.get("geom_count"))
            if geom_count is not None:
                geom_counts[timestep, index] = float(geom_count)
                has_geom_count = True
    out: dict[str, np.ndarray] = {}
    if has_pos:
        out["pos"] = positions
    if has_quat:
        out["quat"] = quats
    if has_geom_center:
        out["geom_center"] = geom_centers
    if has_bbox:
        out["bbox_world"] = bboxes
    if has_geom_count:
        out["geom_count"] = geom_counts
    return out


def _scene_object_descriptors(
    candidates: Sequence[Any],
    sim: Any,
) -> list[dict[str, Any]]:
    descriptors: dict[str, dict[str, Any]] = {}

    body_ids = _first_existing_attr(candidates, ("obj_body_id", "object_body_ids"))
    if isinstance(body_ids, Mapping):
        for name, body_id in body_ids.items():
            key = str(name)
            descriptors[key] = {
                "object_name": key,
                "object_kind": _object_kind_for_name(candidates, key),
                "source": "libero.obj_body_id",
                "body_id": _resolve_body_id(sim, body_id),
                "body_name": _body_name_from_value(sim, body_id),
                "site_name": "",
            }

    for attr_name, object_kind in (
        ("objects_dict", "object"),
        ("fixtures_dict", "fixture"),
    ):
        mapping = _first_existing_attr(candidates, (attr_name,))
        if isinstance(mapping, Mapping):
            for name, item in mapping.items():
                key = str(name)
                body_name = str(
                    getattr(item, "root_body", None)
                    or getattr(item, "body_name", None)
                    or getattr(item, "name", None)
                    or key
                )
                current = descriptors.setdefault(
                    key,
                    {
                        "object_name": key,
                        "object_kind": object_kind,
                        "source": f"libero.{attr_name}",
                        "body_id": None,
                        "body_name": body_name,
                        "site_name": "",
                    },
                )
                current["object_kind"] = object_kind
                current["body_name"] = current.get("body_name") or body_name
                current["body_id"] = current.get("body_id") or _resolve_body_id(sim, body_name)

    for attr_name, object_kind in (("objects", "object"), ("fixtures", "fixture")):
        values = _first_existing_attr(candidates, (attr_name,))
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            continue
        for item in values:
            name = getattr(item, "name", None) or getattr(item, "object_name", None)
            if name is None:
                continue
            key = str(name)
            body_name = str(getattr(item, "root_body", None) or key)
            descriptors.setdefault(
                key,
                {
                    "object_name": key,
                    "object_kind": object_kind,
                    "source": f"libero.{attr_name}",
                    "body_id": _resolve_body_id(sim, body_name),
                    "body_name": body_name,
                    "site_name": "",
                },
            )

    sites = _first_existing_attr(candidates, ("object_sites_dict", "sites_dict"))
    if isinstance(sites, Mapping):
        for name, item in sites.items():
            key = str(name)
            site_name = str(getattr(item, "name", None) or key)
            descriptors.setdefault(
                key,
                {
                    "object_name": key,
                    "object_kind": "site",
                    "source": "libero.object_sites_dict",
                    "body_id": None,
                    "body_name": "",
                    "site_name": site_name,
                },
            )

    return list(descriptors.values())


def _object_kind_for_name(candidates: Sequence[Any], name: str) -> str:
    for attr_name, object_kind in (
        ("objects_dict", "object"),
        ("fixtures_dict", "fixture"),
        ("object_sites_dict", "site"),
    ):
        value = _first_existing_attr(candidates, (attr_name,))
        if isinstance(value, Mapping) and name in value:
            return object_kind
    return "object"


def _sample_scene_object_pose(
    sim: Any,
    descriptor: Mapping[str, Any],
) -> dict[str, Any] | None:
    if descriptor.get("site_name"):
        pose = _sample_site_pose(sim, str(descriptor["site_name"]))
    else:
        pose = _sample_body_pose(
            sim,
            body_id=_optional_int(descriptor.get("body_id")),
            body_name=str(descriptor.get("body_name") or ""),
        )
    if pose is None:
        return None
    geometry = _sample_object_geometry(sim, descriptor)
    if geometry:
        pose.update(geometry)
    return pose


def _sample_body_pose(
    sim: Any,
    *,
    body_id: int | None,
    body_name: str,
) -> dict[str, Any] | None:
    data = getattr(sim, "data", None)
    if data is None:
        return None
    pos = _named_data_vector(data, "get_body_xpos", body_name, 3)
    if pos is None and body_id is not None:
        pos = _indexed_data_vector(data, "body_xpos", body_id, 3)

    quat = _named_data_vector(data, "get_body_xquat", body_name, 4)
    if quat is None and body_id is not None:
        quat = _indexed_data_vector(data, "body_xquat", body_id, 4)
    if quat is not None:
        quat = _mujoco_quat_to_xyzw(quat)

    if quat is None:
        mat = _named_data_matrix(data, "get_body_xmat", body_name)
        if mat is None and body_id is not None:
            mat = _indexed_data_matrix(data, "body_xmat", body_id)
        if mat is not None:
            quat = _mat_to_quat_xyzw(mat)

    if pos is None and quat is None:
        return None
    return {
        "pos": pos if pos is not None else np.full(3, np.nan, dtype=np.float32),
        "quat": quat if quat is not None else np.full(4, np.nan, dtype=np.float32),
    }


def _sample_site_pose(sim: Any, site_name: str) -> dict[str, Any] | None:
    data = getattr(sim, "data", None)
    model = getattr(sim, "model", None)
    if data is None:
        return None
    site_id = _resolve_site_id(model, site_name)
    pos = _named_data_vector(data, "get_site_xpos", site_name, 3)
    if pos is None and site_id is not None:
        pos = _indexed_data_vector(data, "site_xpos", site_id, 3)

    mat = _named_data_matrix(data, "get_site_xmat", site_name)
    if mat is None and site_id is not None:
        mat = _indexed_data_matrix(data, "site_xmat", site_id)
    quat = _mat_to_quat_xyzw(mat) if mat is not None else None

    if pos is None and quat is None:
        return None
    return {
        "pos": pos if pos is not None else np.full(3, np.nan, dtype=np.float32),
        "quat": quat if quat is not None else np.full(4, np.nan, dtype=np.float32),
    }


def _sample_object_geometry(
    sim: Any,
    descriptor: Mapping[str, Any],
) -> dict[str, Any] | None:
    if descriptor.get("site_name"):
        return None
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        return None
    body_id = _optional_int(descriptor.get("body_id"))
    if body_id is None:
        body_id = _resolve_body_id(sim, descriptor.get("body_name"))
    if body_id is None:
        return None

    points: list[np.ndarray] = []
    geom_count = 0
    for geom_id in _geom_indices_for_body_tree(model, body_id):
        geom_points = _geom_world_points(model, data, geom_id)
        if geom_points is None:
            continue
        points.append(geom_points)
        geom_count += 1
    if not points:
        return None

    all_points = np.concatenate(points, axis=0)
    finite = np.all(np.isfinite(all_points), axis=1)
    if not np.any(finite):
        return None
    all_points = all_points[finite]
    bbox_min = np.min(all_points, axis=0).astype(np.float32)
    bbox_max = np.max(all_points, axis=0).astype(np.float32)
    return {
        "geom_center": ((bbox_min + bbox_max) * 0.5).astype(np.float32),
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "geom_count": int(geom_count),
    }


def _geom_indices_for_body_tree(model: Any, body_id: int) -> list[int]:
    geom_bodyid = getattr(model, "geom_bodyid", None)
    if geom_bodyid is None:
        return []
    body_ids = set(_body_tree_ids(model, body_id))
    geoms: list[int] = []
    try:
        geom_total = int(getattr(model, "ngeom", len(geom_bodyid)))
    except TypeError:
        geom_total = len(geom_bodyid)
    for geom_id in range(geom_total):
        try:
            if int(geom_bodyid[geom_id]) in body_ids:
                geoms.append(geom_id)
        except Exception:
            continue
    return geoms


def _body_tree_ids(model: Any, body_id: int) -> list[int]:
    parent_ids = getattr(model, "body_parentid", None)
    if parent_ids is None:
        return [body_id]
    try:
        body_total = int(getattr(model, "nbody", len(parent_ids)))
    except TypeError:
        body_total = len(parent_ids)
    descendants: list[int] = []
    for candidate in range(body_total):
        current = candidate
        seen: set[int] = set()
        while current not in seen and current >= 0:
            if current == body_id:
                descendants.append(candidate)
                break
            seen.add(current)
            try:
                current = int(parent_ids[current])
            except Exception:
                break
    return descendants or [body_id]


def _geom_world_points(model: Any, data: Any, geom_id: int) -> np.ndarray | None:
    center = _indexed_data_vector(data, "geom_xpos", geom_id, 3)
    if center is None:
        return None
    points = [center.astype(np.float32)]
    size = _indexed_data_vector(model, "geom_size", geom_id, 3)
    mat = _indexed_data_matrix(data, "geom_xmat", geom_id)
    if (
        size is not None
        and mat is not None
        and np.all(np.isfinite(size))
        and np.all(np.isfinite(mat))
    ):
        size = np.maximum(np.abs(size.astype(np.float32)), 0.0)
        offsets = np.asarray(
            [
                [sx, sy, sz]
                for sx in (-size[0], size[0])
                for sy in (-size[1], size[1])
                for sz in (-size[2], size[2])
            ],
            dtype=np.float32,
        )
        corners = center.astype(np.float32) + offsets @ mat.astype(np.float32).T
        points.extend(corners)
    return np.asarray(points, dtype=np.float32)


def _camera_object_bboxes_from_segmentation(
    sim: Any,
    camera_name: str,
    *,
    height: int | None,
    width: int | None,
    object_descriptors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if height is None or width is None or height <= 0 or width <= 0:
        return []
    segmentation = _render_camera_segmentation(
        sim,
        camera_name,
        height=int(height),
        width=int(width),
    )
    if segmentation is None:
        return []
    try:
        mujoco = importlib.import_module("mujoco")
        geom_obj_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    except Exception:
        geom_obj_type = 5

    model = getattr(sim, "model", None)
    if model is None:
        return []

    rows: list[dict[str, Any]] = []
    for descriptor in object_descriptors:
        object_name = descriptor.get("object_name")
        if object_name is None or descriptor.get("site_name"):
            continue
        body_id = _optional_int(descriptor.get("body_id"))
        if body_id is None:
            body_id = _resolve_body_id(sim, descriptor.get("body_name"))
        if body_id is None:
            continue
        geom_ids = _geom_indices_for_body_tree(model, body_id)
        if not geom_ids:
            continue
        mask = (segmentation[:, :, 0] == geom_obj_type) & np.isin(segmentation[:, :, 1], geom_ids)
        if not bool(np.any(mask)):
            continue
        ys, xs = np.where(mask)
        rows.append(
            {
                "object_name": str(object_name),
                "bbox_pixel_xyxy": np.asarray(
                    [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                    dtype=np.float32,
                ),
                "pixel_area": int(mask.sum()),
                "source": "robosuite.segmentation",
            }
        )
    return rows


def _render_camera_segmentation(
    sim: Any,
    camera_name: str,
    *,
    height: int,
    width: int,
) -> np.ndarray | None:
    """Render robust segmentation IDs, avoiding robosuite's uint8 overflow path."""

    try:
        mujoco = importlib.import_module("mujoco")
        binding_utils = importlib.import_module("robosuite.utils.binding_utils")
        context = getattr(sim, "_render_context_offscreen", None)
        model = getattr(sim, "model", None)
        if context is None or model is None:
            return None
        camera_id = model.camera_name2id(camera_name)
        lock = binding_utils._MjSim_render_lock
        with lock:
            context.render(
                width=width,
                height=height,
                camera_id=camera_id,
                segmentation=True,
            )
            viewport = mujoco.MjrRect(0, 0, width, height)
            rgb = np.empty((height, width, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(rgb=rgb, depth=None, viewport=viewport, con=context.con)
            seg_img = (
                rgb[:, :, 0].astype(np.int32)
                + rgb[:, :, 1].astype(np.int32) * (2**8)
                + rgb[:, :, 2].astype(np.int32) * (2**16)
            )
            seg_img[seg_img >= (context.scn.ngeom + 1)] = 0
            seg_ids = np.full((context.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32)
            for index in range(context.scn.ngeom):
                geom = context.scn.geoms[index]
                if geom.segid != -1:
                    seg_ids[geom.segid + 1, 0] = geom.objtype
                    seg_ids[geom.segid + 1, 1] = geom.objid
            return seg_ids[seg_img]
    except Exception:
        return None


def _named_data_vector(data: Any, method_name: str, name: str, size: int) -> np.ndarray | None:
    if not name:
        return None
    method = getattr(data, method_name, None)
    if not callable(method):
        return None
    try:
        return _numeric_vector(method(name), size)
    except Exception:
        return None


def _indexed_data_vector(data: Any, attr_name: str, index: int, size: int) -> np.ndarray | None:
    try:
        values = getattr(data, attr_name)
        return _numeric_vector(values[int(index)], size)
    except Exception:
        return None


def _named_data_matrix(data: Any, method_name: str, name: str) -> np.ndarray | None:
    if not name:
        return None
    method = getattr(data, method_name, None)
    if not callable(method):
        return None
    try:
        return _numeric_matrix(method(name), 3, 3)
    except Exception:
        return None


def _indexed_data_matrix(data: Any, attr_name: str, index: int) -> np.ndarray | None:
    try:
        values = getattr(data, attr_name)
        return _numeric_matrix(values[int(index)], 3, 3)
    except Exception:
        return None


def _numeric_vector(value: Any, size: int) -> np.ndarray | None:
    array = _numeric_array(value)
    if array is None:
        return None
    flat = np.ravel(array).astype(np.float32, copy=False)
    if flat.size < size:
        return None
    return flat[:size].copy()


def _numeric_matrix(value: Any, rows: int, cols: int) -> np.ndarray | None:
    array = _numeric_array(value)
    if array is None:
        return None
    try:
        return np.asarray(array, dtype=np.float32).reshape(rows, cols)
    except ValueError:
        return None


def _resolve_body_id(sim: Any, value: Any) -> int | None:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if value is None:
        return None
    model = getattr(sim, "model", None)
    method = getattr(model, "body_name2id", None)
    if callable(method):
        try:
            return int(method(str(value)))
        except Exception:
            return None
    return None


def _resolve_site_id(model: Any, value: Any) -> int | None:
    if model is None or value is None:
        return None
    method = getattr(model, "site_name2id", None)
    if callable(method):
        try:
            return int(method(str(value)))
        except Exception:
            return None
    return None


def _body_name_from_value(sim: Any, value: Any) -> str:
    if isinstance(value, str):
        return value
    body_id = _resolve_body_id(sim, value)
    if body_id is None:
        return ""
    model = getattr(sim, "model", None)
    method = getattr(model, "body_id2name", None)
    if callable(method):
        try:
            name = method(body_id)
            return "" if name is None else str(name)
        except Exception:
            pass
    names = getattr(model, "body_names", None)
    if isinstance(names, Sequence) and 0 <= body_id < len(names):
        return str(names[body_id])
    return ""


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mujoco_quat_to_xyzw(quat: np.ndarray) -> np.ndarray:
    value = _numeric_vector(quat, 4)
    if value is None:
        return np.full(4, np.nan, dtype=np.float32)
    return np.asarray([value[1], value[2], value[3], value[0]], dtype=np.float32)


def _mat_to_quat_xyzw(mat: np.ndarray) -> np.ndarray:
    matrix = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([x, y, z, w], dtype=np.float32)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.full(4, np.nan, dtype=np.float32)
    return quat / norm


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


def _names_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [str(key) for key in value]
    if isinstance(value, (str, bytes)):
        return [item.strip() for item in str(value).split(",") if item.strip()]
    if isinstance(value, Sequence):
        names: list[str] = []
        for item in value:
            if isinstance(item, (str, bytes)):
                names.append(str(item))
            else:
                name = getattr(item, "name", None) or getattr(item, "object_name", None)
                if name is not None:
                    names.append(str(name))
        return names
    return []


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


def _flatten_mapping_keys(mapping: Mapping[str, Any], prefix: str = "") -> list[str]:
    keys: list[str] = []
    for key, value in mapping.items():
        text = f"{prefix}.{key}" if prefix else str(key)
        keys.append(text)
        if isinstance(value, Mapping):
            keys.extend(_flatten_mapping_keys(value, text))
    return keys


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


def _camera_names(env: Any | None, observations: Sequence[Mapping[str, Any]]) -> list[str]:
    candidates = _env_candidates(env)
    values = _first_existing_attr(candidates, ("camera_names", "_camera_names", "cameras"))
    names = _names_from_value(values)
    if names:
        return [_normalize_camera_name(name) for name in names]
    if not observations:
        return []
    names = []
    for key in _flatten_mapping_keys(observations[0]):
        short = key.rsplit(".", 1)[-1]
        if short.endswith("_image"):
            names.append(short[: -len("_image")])
        elif short == "image":
            names.append("image")
    return sorted(set(names))


def _normalize_camera_name(name: str) -> str:
    text = str(name)
    return text[: -len("_image")] if text.endswith("_image") else text


def _camera_resolution(
    name: str,
    observations: Sequence[Mapping[str, Any]],
    env: Any | None,
) -> tuple[int | None, int | None]:
    image = _camera_image(name, observations)
    if image is not None and image.ndim >= 2:
        return int(image.shape[0]), int(image.shape[1])
    candidates = _env_candidates(env)
    height = _camera_dimension(candidates, name, ("camera_heights", "_camera_heights", "height"))
    width = _camera_dimension(candidates, name, ("camera_widths", "_camera_widths", "width"))
    return height, width


def _camera_image(name: str, observations: Sequence[Mapping[str, Any]]) -> np.ndarray | None:
    if not observations:
        return None
    obs = observations[0]
    for key in (f"{name}_image", f"observation.images.{name}", "image" if name == "image" else ""):
        if not key:
            continue
        value = _lookup_mapping_path(obs, key)
        array = _numeric_array(value)
        if array is not None:
            return _squeeze_single_env(array)
    return None


def _camera_dimension(
    candidates: Sequence[Any],
    name: str,
    attrs: Sequence[str],
) -> int | None:
    for attr in attrs:
        value = _first_existing_attr(candidates, (attr,))
        if isinstance(value, Mapping):
            item = value.get(name)
            if item is not None:
                return int(item)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            camera_name_attr = _first_existing_attr(candidates, ("camera_names", "_camera_names"))
            names = _names_from_value(camera_name_attr)
            if name in names:
                return int(value[names.index(name)])
        elif value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return None


def _robosuite_camera_utils() -> Any | None:
    try:
        return importlib.import_module("robosuite.utils.camera_utils")
    except Exception:
        return None


def _camera_intrinsic(
    camera_utils: Any | None,
    sim: Any | None,
    name: str,
    *,
    height: int | None,
    width: int | None,
) -> tuple[np.ndarray | None, str]:
    if camera_utils is None:
        return None, "robosuite camera_utils unavailable"
    if sim is None:
        return None, "env.sim unavailable"
    if height is None or width is None:
        return None, "camera resolution unavailable"
    func = getattr(camera_utils, "get_camera_intrinsic_matrix", None)
    if not callable(func):
        return None, "get_camera_intrinsic_matrix unavailable"
    try:
        return np.asarray(func(sim, name, int(height), int(width)), dtype=np.float32), ""
    except Exception as exc:
        return None, f"camera_utils intrinsic failed: {exc}"


def _camera_extrinsic(
    camera_utils: Any | None,
    sim: Any | None,
    name: str,
) -> tuple[np.ndarray | None, str]:
    if camera_utils is None:
        return None, "robosuite camera_utils unavailable"
    if sim is None:
        return None, "env.sim unavailable"
    func = getattr(camera_utils, "get_camera_extrinsic_matrix", None)
    if not callable(func):
        return None, "get_camera_extrinsic_matrix unavailable"
    try:
        return np.asarray(func(sim, name), dtype=np.float32), ""
    except Exception as exc:
        return None, f"camera_utils extrinsic failed: {exc}"


def _generic_axes(array: np.ndarray, *, trailing_prefix: str) -> list[str]:
    if array.ndim == 0:
        return []
    if array.ndim == 1:
        return [f"{trailing_prefix}_dim"]
    return [f"axis_{index}" for index in range(array.ndim)]


def _shape_text(shape: Sequence[int] | None) -> str:
    if shape is None:
        return ""
    return "x".join(str(int(item)) for item in shape)


def _scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    array = _numeric_array(value)
    if array is not None and array.size == 1:
        return str(np.ravel(array)[0].item())
    return repr(value)


__all__ = [
    "ContextCaptureResult",
    "capture_libero_context",
    "capture_scene_snapshot",
    "extract_camera_context",
    "extract_env_metadata",
    "extract_object_context",
    "extract_robot_arrays",
]
