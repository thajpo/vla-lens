"""Scripted policy backend used to validate the rollout interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class ScriptedPickPolicyConfig:
    target_object: str = "cubeA"
    approach_height: float = 0.10
    grasp_offset: float = 0.01
    lift_height: float = 0.18
    position_gain: float = 8.0
    open_gripper: float = -1.0
    close_gripper: float = 1.0
    approach_steps: int = 40
    descend_steps: int = 35
    close_steps: int = 25
    lift_steps: int = 40

    @property
    def total_steps(self) -> int:
        return self.approach_steps + self.descend_steps + self.close_steps + self.lift_steps


class ScriptedPickPolicy:
    """Simple scripted controller that picks one named cube."""

    def __init__(self, config: ScriptedPickPolicyConfig):
        self.config = config
        self.policy_id = "scripted_pick"
        self.target_object = config.target_object
        self._scene_metadata: dict[str, Any] | None = None

    def reset(self, initial_obs: dict[str, np.ndarray], scene_metadata: dict[str, Any]) -> None:
        self._scene_metadata = scene_metadata

    def act(self, obs: dict[str, np.ndarray], step: int) -> tuple[np.ndarray, dict[str, Any]]:
        if self.target_object not in {"cubeA", "cubeB"}:
            raise ValueError(f"Unsupported scripted target object: {self.target_object}")

        target_pos = np.array(obs[f"{self.target_object}_pos"])
        eef_pos = np.array(obs["robot0_eef_pos"])
        action = np.zeros(7, dtype=np.float64)

        approach_end = self.config.approach_steps
        descend_end = approach_end + self.config.descend_steps
        close_end = descend_end + self.config.close_steps

        if step < approach_end:
            desired = target_pos + np.array([0.0, 0.0, self.config.approach_height])
            grip = self.config.open_gripper
            phase = "approach"
        elif step < descend_end:
            desired = target_pos + np.array([0.0, 0.0, self.config.grasp_offset])
            grip = self.config.open_gripper
            phase = "descend"
        elif step < close_end:
            desired = target_pos + np.array([0.0, 0.0, self.config.grasp_offset])
            grip = self.config.close_gripper
            phase = "close"
        else:
            desired = target_pos + np.array([0.0, 0.0, self.config.lift_height])
            grip = self.config.close_gripper
            phase = "lift"

        delta = (desired - eef_pos) * self.config.position_gain
        action[:3] = np.clip(delta, -1.0, 1.0)
        action[-1] = grip
        return action, {"phase": phase}
