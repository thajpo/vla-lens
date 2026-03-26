"""Experiment-facing wrapper for the robosuite Stack environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import imageio.v2 as imageio
import numpy as np

from openvla_steering.env.robosuite_env import RobosuiteEnvConfig, make_env
from openvla_steering.utils.io import ensure_parent
from openvla_steering.utils.seeding import set_global_seed


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


@dataclass(slots=True)
class ObjectMetadata:
    object_id: str
    color_name: str
    rgba: list[float]


@dataclass(slots=True)
class StackTaskMetadata:
    cubeA: ObjectMetadata
    cubeB: ObjectMetadata

    def object_ids(self) -> tuple[str, str]:
        return self.cubeA.object_id, self.cubeB.object_id

    def color_map(self) -> dict[str, str]:
        return {
            self.cubeA.object_id: self.cubeA.color_name,
            self.cubeB.object_id: self.cubeB.color_name,
        }


@dataclass(slots=True)
class RolloutSummary:
    rollout_id: str
    seed: int
    target_object: str
    target_color_name: str
    selected_object: str | None
    selected_color_name: str | None
    success: bool
    initial_cubeA_pos: list[float]
    initial_cubeB_pos: list[float]
    final_cubeA_pos: list[float]
    final_cubeB_pos: list[float]
    final_eef_pos: list[float]
    cubeA_height_gain: float
    cubeB_height_gain: float
    max_reward: float
    final_reward: float
    steps_executed: int
    done: bool
    video_path: str | None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


class StackTaskEnv:
    """Small experiment wrapper around robosuite's built-in Stack task."""

    def __init__(self, env_config: RobosuiteEnvConfig, task_metadata: StackTaskMetadata):
        self.env_config = env_config
        self.task_metadata = task_metadata
        self._env: Any | None = None
        self._current_seed: int | None = None

    @property
    def env(self) -> Any:
        if self._env is None:
            raise RuntimeError("Environment has not been reset yet.")
        return self._env

    def reset(self, seed: int) -> dict[str, np.ndarray]:
        set_global_seed(seed)
        self._current_seed = seed
        if self._env is not None:
            self._env.close()
        self._env = make_env(self.env_config, seed=seed)
        return self.env.reset()

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def current_observation(self) -> dict[str, np.ndarray]:
        return self.env._get_observations()

    def scene_metadata(self) -> dict[str, Any]:
        obs = self.current_observation()
        return {
            "env_name": self.env.__class__.__name__,
            "seed": self._current_seed,
            "cubeA_object_id": self.task_metadata.cubeA.object_id,
            "cubeA_color_name": self.task_metadata.cubeA.color_name,
            "cubeA_rgba": self.task_metadata.cubeA.rgba,
            "cubeB_object_id": self.task_metadata.cubeB.object_id,
            "cubeB_color_name": self.task_metadata.cubeB.color_name,
            "cubeB_rgba": self.task_metadata.cubeB.rgba,
            "cubeA_pos": obs["cubeA_pos"].round(6).tolist(),
            "cubeB_pos": obs["cubeB_pos"].round(6).tolist(),
            "robot0_eef_pos": obs["robot0_eef_pos"].round(6).tolist(),
        }

    def render_frame(self, camera_name: str, width: int = 640, height: int = 480) -> np.ndarray:
        frame = self.env.sim.render(
            camera_name=camera_name,
            width=width,
            height=height,
            depth=False,
        )
        return np.flipud(frame)

    def _scripted_action(
        self, obs: dict[str, np.ndarray], policy: ScriptedPickPolicyConfig, step: int
    ) -> tuple[np.ndarray, str]:
        if policy.target_object not in {"cubeA", "cubeB"}:
            raise ValueError(f"Unsupported scripted target object: {policy.target_object}")

        target_pos = np.array(obs[f"{policy.target_object}_pos"])
        eef_pos = np.array(obs["robot0_eef_pos"])
        action = np.zeros(self.env.action_dim, dtype=np.float64)

        approach_end = policy.approach_steps
        descend_end = approach_end + policy.descend_steps
        close_end = descend_end + policy.close_steps

        if step < approach_end:
            desired = target_pos + np.array([0.0, 0.0, policy.approach_height])
            grip = policy.open_gripper
            phase = "approach"
        elif step < descend_end:
            desired = target_pos + np.array([0.0, 0.0, policy.grasp_offset])
            grip = policy.open_gripper
            phase = "descend"
        elif step < close_end:
            desired = target_pos + np.array([0.0, 0.0, policy.grasp_offset])
            grip = policy.close_gripper
            phase = "close"
        else:
            desired = target_pos + np.array([0.0, 0.0, policy.lift_height])
            grip = policy.close_gripper
            phase = "lift"

        delta = (desired - eef_pos) * policy.position_gain
        action[:3] = np.clip(delta, -1.0, 1.0)
        action[-1] = grip
        return action, phase

    def infer_selected_object(
        self,
        initial_obs: dict[str, np.ndarray],
        final_obs: dict[str, np.ndarray],
        height_threshold: float = 0.08,
    ) -> tuple[str | None, float, float]:
        cubeA_gain = float(final_obs["cubeA_pos"][2] - initial_obs["cubeA_pos"][2])
        cubeB_gain = float(final_obs["cubeB_pos"][2] - initial_obs["cubeB_pos"][2])
        best_name = "cubeA" if cubeA_gain >= cubeB_gain else "cubeB"
        best_gain = max(cubeA_gain, cubeB_gain)
        if best_gain < height_threshold:
            return None, cubeA_gain, cubeB_gain
        return best_name, cubeA_gain, cubeB_gain

    def run_scripted_pick(
        self,
        seed: int,
        policy: ScriptedPickPolicyConfig,
        save_video: bool = False,
        video_path: str | None = None,
        camera_name: str = "frontview",
        log_every: int = 25,
    ) -> tuple[RolloutSummary, list[str]]:
        initial_obs = self.reset(seed=seed)
        initial_snapshot = {
            "cubeA_pos": np.array(initial_obs["cubeA_pos"]),
            "cubeB_pos": np.array(initial_obs["cubeB_pos"]),
        }

        frames: list[np.ndarray] = []
        debug_lines: list[str] = [
            f"Reset complete. Observation keys: {sorted(initial_obs.keys())}",
            f"Action dimension: {self.env.action_dim}",
            f"Scene metadata: {self.scene_metadata()}",
        ]

        obs = initial_obs
        max_reward = float("-inf")
        final_reward = 0.0
        done = False
        executed_steps = 0

        for step in range(policy.total_steps):
            action, phase = self._scripted_action(obs, policy, step)
            obs, reward, done, _ = self.env.step(action)
            final_reward = float(reward)
            max_reward = max(max_reward, final_reward)
            executed_steps = step + 1

            if save_video:
                frames.append(self.render_frame(camera_name=camera_name))

            if step % log_every == 0:
                target_dist = float(
                    np.linalg.norm(np.array(obs["robot0_eef_pos"]) - np.array(obs[f"{policy.target_object}_pos"]))
                )
                debug_lines.append(
                    f"step={step} phase={phase} reward={reward:.4f} done={done} target_dist={target_dist:.6f}"
                )
            if done:
                debug_lines.append(f"Episode ended early at step {step}")
                break

        selected_object, cubeA_gain, cubeB_gain = self.infer_selected_object(initial_obs, obs)
        success = selected_object == policy.target_object
        color_map = self.task_metadata.color_map()

        saved_video_path: str | None = None
        if save_video and frames and video_path is not None:
            output_path = ensure_parent(video_path)
            imageio.mimsave(output_path, frames, fps=20)
            saved_video_path = str(output_path)
            debug_lines.append(f"Saved video to {saved_video_path}")

        summary = RolloutSummary(
            rollout_id=uuid4().hex[:12],
            seed=seed,
            target_object=policy.target_object,
            target_color_name=color_map[policy.target_object],
            selected_object=selected_object,
            selected_color_name=color_map[selected_object] if selected_object is not None else None,
            success=success,
            initial_cubeA_pos=initial_snapshot["cubeA_pos"].round(6).tolist(),
            initial_cubeB_pos=initial_snapshot["cubeB_pos"].round(6).tolist(),
            final_cubeA_pos=np.array(obs["cubeA_pos"]).round(6).tolist(),
            final_cubeB_pos=np.array(obs["cubeB_pos"]).round(6).tolist(),
            final_eef_pos=np.array(obs["robot0_eef_pos"]).round(6).tolist(),
            cubeA_height_gain=round(cubeA_gain, 6),
            cubeB_height_gain=round(cubeB_gain, 6),
            max_reward=round(max_reward, 6),
            final_reward=round(final_reward, 6),
            steps_executed=executed_steps,
            done=done,
            video_path=saved_video_path,
        )
        return summary, debug_lines


def default_video_path(root: str | Path, seed: int, target_object: str) -> str:
    return str(Path(root) / f"seed_{seed:04d}_{target_object}.mp4")
