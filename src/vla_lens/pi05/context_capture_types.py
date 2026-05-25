"""PI0.5 context capture schemas and field conventions."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Mapping

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
