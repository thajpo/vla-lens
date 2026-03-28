from __future__ import annotations

import hydra
from omegaconf import DictConfig

from openvla_steering.env import RobosuiteEnvConfig, StackTaskEnv


@hydra.main(version_base=None, config_path="../configs", config_name="scene")
def main(cfg: DictConfig) -> None:
    env = StackTaskEnv(
        RobosuiteEnvConfig(
            name=str(cfg.env.name).capitalize(),
            robot=str(cfg.env.robot),
            horizon=int(cfg.env.horizon),
            control_freq=int(cfg.env.control_freq),
            use_renderer=bool(cfg.env.use_renderer),
            use_camera_obs=bool(cfg.env.use_camera_obs),
            has_offscreen_renderer=bool(cfg.env.has_offscreen_renderer),
            reward_shaping=bool(cfg.env.reward_shaping),
            hard_reset=bool(cfg.env.hard_reset),
            ignore_done=bool(cfg.env.ignore_done),
            camera_name=str(cfg.env.camera_name),
            camera_height=int(cfg.env.camera_height),
            camera_width=int(cfg.env.camera_width),
            camera_depth=bool(cfg.env.camera_depth),
        )
    )

    seed = int(cfg.run.seed)
    env.reset(seed=seed)
    first = env.scene_metadata()
    env.reset(seed=seed)
    second = env.scene_metadata()
    env.close()

    print(f"First reset metadata: {first}")
    print(f"Second reset metadata: {second}")
    print(f"Matched scene: {first == second}")


if __name__ == "__main__":
    main()
