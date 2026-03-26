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
class ObjectMetadata:
    object_id: str
    color_name: str
    rgba: list[float]


@dataclass(slots=True)
class RolloutSummary:
    rollout_id: str
    seed: int
    policy_id: str
    target_object: str | None
    target_color_name: str | None
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

    def __init__(self, env_config: RobosuiteEnvConfig):
        self.env_config = env_config
        self._env: Any | None = None
        self._current_seed: int | None = None
        self._object_metadata = {
            "cubeA": ObjectMetadata(object_id="cubeA", color_name="red", rgba=[1.0, 0.0, 0.0, 1.0]),
            "cubeB": ObjectMetadata(object_id="cubeB", color_name="green", rgba=[0.0, 1.0, 0.0, 1.0]),
        }

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
            "cubeA_object_id": self._object_metadata["cubeA"].object_id,
            "cubeA_color_name": self._object_metadata["cubeA"].color_name,
            "cubeA_rgba": self._object_metadata["cubeA"].rgba,
            "cubeB_object_id": self._object_metadata["cubeB"].object_id,
            "cubeB_color_name": self._object_metadata["cubeB"].color_name,
            "cubeB_rgba": self._object_metadata["cubeB"].rgba,
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

    def color_name_for(self, object_id: str | None) -> str | None:
        if object_id is None:
            return None
        metadata = self._object_metadata.get(object_id)
        return metadata.color_name if metadata is not None else None

    def run_policy_rollout(
        self,
        seed: int,
        policy: Any,
        num_steps: int,
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

        if hasattr(policy, "reset"):
            policy.reset(initial_obs=initial_obs, scene_metadata=self.scene_metadata())

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

        for step in range(num_steps):
            action, info = policy.act(obs=obs, step=step)
            phase = info.get("phase", "policy")
            obs, reward, done, _ = self.env.step(action)
            final_reward = float(reward)
            max_reward = max(max_reward, final_reward)
            executed_steps = step + 1

            if save_video:
                frames.append(self.render_frame(camera_name=camera_name))

            if step % log_every == 0:
                target_object = getattr(policy, "target_object", None)
                target_dist = None
                if target_object in {"cubeA", "cubeB"}:
                    target_dist = float(
                        np.linalg.norm(np.array(obs["robot0_eef_pos"]) - np.array(obs[f"{target_object}_pos"]))
                    )
                debug_lines.append(
                    f"step={step} phase={phase} reward={reward:.4f} done={done} "
                    f"target_dist={target_dist if target_dist is not None else 'n/a'}"
                )
            if done:
                debug_lines.append(f"Episode ended early at step {step}")
                break

        selected_object, cubeA_gain, cubeB_gain = self.infer_selected_object(initial_obs, obs)
        target_object = getattr(policy, "target_object", None)
        success = selected_object == target_object if target_object is not None else selected_object is not None

        saved_video_path: str | None = None
        if save_video and frames and video_path is not None:
            output_path = ensure_parent(video_path)
            imageio.mimsave(output_path, frames, fps=20)
            saved_video_path = str(output_path)
            debug_lines.append(f"Saved video to {saved_video_path}")

        summary = RolloutSummary(
            rollout_id=uuid4().hex[:12],
            seed=seed,
            policy_id=getattr(policy, "policy_id", policy.__class__.__name__),
            target_object=target_object,
            target_color_name=self.color_name_for(target_object),
            selected_object=selected_object,
            selected_color_name=self.color_name_for(selected_object),
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


def default_video_path(root: str | Path, seed: int, target_object: str | None, policy_id: str) -> str:
    target_suffix = target_object if target_object is not None else policy_id
    return str(Path(root) / f"seed_{seed:04d}_{target_suffix}.mp4")
