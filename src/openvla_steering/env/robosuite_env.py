"""Small robosuite wrapper for the first visualization loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import robosuite as suite


@dataclass(slots=True)
class RobosuiteEnvConfig:
    name: str = "Stack"
    robot: str = "Panda"
    horizon: int = 200
    control_freq: int = 20
    use_renderer: bool = False
    use_camera_obs: bool = False
    has_offscreen_renderer: bool = True
    reward_shaping: bool = True
    hard_reset: bool = True
    ignore_done: bool = False
    camera_name: str = "frontview"
    camera_height: int = 224
    camera_width: int = 224
    camera_depth: bool = False
    seed: int | None = None


def make_env(config: RobosuiteEnvConfig, seed: int | None = None) -> Any:
    return suite.make(
        env_name=config.name,
        robots=config.robot,
        horizon=config.horizon,
        control_freq=config.control_freq,
        has_renderer=config.use_renderer,
        has_offscreen_renderer=config.has_offscreen_renderer,
        use_camera_obs=config.use_camera_obs,
        reward_shaping=config.reward_shaping,
        hard_reset=config.hard_reset,
        ignore_done=config.ignore_done,
        camera_names=config.camera_name,
        camera_heights=config.camera_height,
        camera_widths=config.camera_width,
        camera_depths=config.camera_depth,
        seed=config.seed if seed is None else seed,
    )


def get_scene_metadata(env: Any) -> dict[str, list[float] | str]:
    obs = env._get_observations()
    metadata: dict[str, list[float] | str] = {"env_name": env.__class__.__name__}

    if "cubeA_pos" in obs:
        metadata["cubeA_pos"] = obs["cubeA_pos"].round(6).tolist()
    if "cubeB_pos" in obs:
        metadata["cubeB_pos"] = obs["cubeB_pos"].round(6).tolist()
    if "robot0_eef_pos" in obs:
        metadata["robot0_eef_pos"] = obs["robot0_eef_pos"].round(6).tolist()

    return metadata
