"""Replayable, audited scene mutations for LIBERO counterfactual traces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class SceneMutationSpec:
    """A small simulator-state recipe that can be saved and replayed."""

    kind: str
    objects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"identity", "pose_exchange"}:
            raise ValueError("scene mutation kind must be identity or pose_exchange")
        if self.kind == "identity" and self.objects:
            raise ValueError("identity scene mutation must not name objects")
        if self.kind == "pose_exchange" and (
            len(self.objects) != 2 or not all(str(value).strip() for value in self.objects)
        ):
            raise ValueError("pose_exchange requires exactly two named objects")
        if len(self.objects) == 2 and self.objects[0] == self.objects[1]:
            raise ValueError("pose_exchange objects must be different")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "objects": list(self.objects)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SceneMutationSpec":
        objects = payload.get("objects")
        if objects is None:
            objects = ()
        if not isinstance(objects, (list, tuple)):
            raise ValueError("scene mutation objects must be a list")
        return cls(
            kind=str(payload.get("kind") or ""),
            objects=tuple(str(value) for value in objects),
        )


def scene_mutation_from_json(value: str | None) -> SceneMutationSpec | None:
    """Parse an inline JSON object or a path to one."""
    text = str(value or "").strip()
    if not text:
        return None
    payload = json.loads(text if text.startswith("{") else Path(text).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("scene mutation JSON must contain an object")
    return SceneMutationSpec.from_dict(payload)


def scene_mutation_from_metadata(value: Any) -> SceneMutationSpec | None:
    """Recover the recipe from either a saved spec or an execution report."""
    if not isinstance(value, Mapping) or not value:
        return None
    spec = value.get("spec") if isinstance(value.get("spec"), Mapping) else value
    return SceneMutationSpec.from_dict(spec)


def apply_scene_mutation(
    env: Any,
    spec: SceneMutationSpec,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Exchange two free-joint poses and return a freshly rendered observation."""
    vectorized = bool(getattr(env, "envs", None))
    base_env = env.envs[0] if vectorized else env
    raw_env = getattr(base_env, "_env", None)
    inner_env = getattr(raw_env, "env", None)
    sim = getattr(raw_env, "sim", None)
    objects = getattr(inner_env, "objects_dict", None)
    if raw_env is None or inner_env is None or sim is None or not isinstance(objects, Mapping):
        raise TypeError("pose exchange requires a LeRobot LiberoEnv backed by MuJoCo")

    before_qpos = np.asarray(sim.data.qpos, dtype=np.float64).copy()
    before_qvel = np.asarray(sim.data.qvel, dtype=np.float64).copy()
    if spec.kind == "identity":
        raw_observation = _refresh_raw_observation(raw_env, inner_env, sim)
        observation = base_env._format_raw_obs(raw_observation)
        after_qpos = np.asarray(sim.data.qpos, dtype=np.float64).copy()
        after_qvel = np.asarray(sim.data.qvel, dtype=np.float64).copy()
        qpos_max = _max_abs(after_qpos - before_qpos)
        qvel_max = _max_abs(after_qvel - before_qvel)
        if qpos_max != 0.0 or qvel_max != 0.0:
            raise RuntimeError("identity scene refresh changed simulator state")
        report = {
            "spec": spec.to_dict(),
            "objects": {},
            "before_qpos_sha256": _array_sha256(before_qpos),
            "after_qpos_sha256": _array_sha256(after_qpos),
            "changed_qpos_indices": [],
            "outside_object_qpos_max_abs": qpos_max,
            "qvel_max_abs": qvel_max,
            "observation_refreshed_without_step": True,
        }
        return (_batch_observation(observation) if vectorized else observation), report

    first_name, second_name = spec.objects
    missing = [name for name in spec.objects if name not in objects]
    if missing:
        raise KeyError(f"pose exchange objects are not movable scene objects: {missing}")
    first_joint = _free_joint(objects[first_name], first_name)
    second_joint = _free_joint(objects[second_name], second_name)
    first_indices = _joint_qpos_indices(sim.model, first_joint)
    second_indices = _joint_qpos_indices(sim.model, second_joint)
    if len(first_indices) != 7 or len(second_indices) != 7:
        raise ValueError("pose exchange currently requires two 7-value free joints")

    first_pose = np.asarray(sim.data.get_joint_qpos(first_joint), dtype=np.float64).copy()
    second_pose = np.asarray(sim.data.get_joint_qpos(second_joint), dtype=np.float64).copy()
    sim.data.set_joint_qpos(first_joint, second_pose)
    sim.data.set_joint_qpos(second_joint, first_pose)
    raw_observation = _refresh_raw_observation(raw_env, inner_env, sim)
    observation = base_env._format_raw_obs(raw_observation)

    after_qpos = np.asarray(sim.data.qpos, dtype=np.float64).copy()
    after_qvel = np.asarray(sim.data.qvel, dtype=np.float64).copy()
    allowed = set((*first_indices, *second_indices))
    outside = [index for index in range(len(before_qpos)) if index not in allowed]
    outside_max = _max_abs(after_qpos[outside] - before_qpos[outside])
    qvel_max = _max_abs(after_qvel - before_qvel)
    if outside_max != 0.0:
        raise RuntimeError("pose exchange changed simulator qpos outside the two objects")
    if qvel_max != 0.0:
        raise RuntimeError("pose exchange changed simulator velocities")
    if not np.array_equal(after_qpos[list(first_indices)], second_pose):
        raise RuntimeError(f"pose exchange did not assign {second_name}'s pose to {first_name}")
    if not np.array_equal(after_qpos[list(second_indices)], first_pose):
        raise RuntimeError(f"pose exchange did not assign {first_name}'s pose to {second_name}")

    report = {
        "spec": spec.to_dict(),
        "objects": {
            first_name: {
                "joint": first_joint,
                "qpos_indices": list(first_indices),
                "before_qpos_wxyz": first_pose.tolist(),
                "after_qpos_wxyz": second_pose.tolist(),
            },
            second_name: {
                "joint": second_joint,
                "qpos_indices": list(second_indices),
                "before_qpos_wxyz": second_pose.tolist(),
                "after_qpos_wxyz": first_pose.tolist(),
            },
        },
        "before_qpos_sha256": _array_sha256(before_qpos),
        "after_qpos_sha256": _array_sha256(after_qpos),
        "changed_qpos_indices": np.flatnonzero(after_qpos != before_qpos).astype(int).tolist(),
        "outside_object_qpos_max_abs": outside_max,
        "qvel_max_abs": qvel_max,
        "observation_refreshed_without_step": True,
    }
    return (_batch_observation(observation) if vectorized else observation), report


def _free_joint(value: Any, object_name: str) -> str:
    joints = getattr(value, "joints", None)
    if not isinstance(joints, (list, tuple)) or not joints:
        raise ValueError(f"scene object {object_name!r} has no MuJoCo joint")
    return str(joints[-1])


def _refresh_raw_observation(raw_env: Any, inner_env: Any, sim: Any) -> Mapping[str, Any]:
    sim.forward()
    raw_env.check_success()
    raw_env._post_process()
    raw_env._update_observables(force=True)
    return inner_env._get_observations()


def _joint_qpos_indices(model: Any, joint: str) -> tuple[int, ...]:
    address = model.get_joint_qpos_addr(joint)
    if isinstance(address, tuple):
        return tuple(range(int(address[0]), int(address[1])))
    return (int(address),)


def _batch_observation(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _batch_observation(item) for key, item in value.items()}
    array = np.asarray(value)
    return np.expand_dims(array, axis=0)


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _max_abs(array: np.ndarray) -> float:
    return float(np.max(np.abs(array))) if array.size else 0.0


__all__ = [
    "SceneMutationSpec",
    "apply_scene_mutation",
    "scene_mutation_from_json",
    "scene_mutation_from_metadata",
]
