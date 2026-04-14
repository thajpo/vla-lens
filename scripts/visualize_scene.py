from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from openvla_steering.env import RobosuiteEnvConfig, StackTaskEnv
from openvla_steering.model import build_policy_from_config


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
    policy = build_policy_from_config(cfg)
    summary, debug_lines = env.run_policy_rollout(
        seed=int(cfg.run.seed),
        policy=policy,
        num_steps=getattr(getattr(policy, "config", None), "total_steps", int(cfg.env.horizon)),
        save_video=bool(cfg.run.save_video),
        video_path=str(Path(str(cfg.run.video_path))),
        camera_name=str(cfg.env.camera_name),
    )
    for line in debug_lines:
        print(line)
    print(
        f"Rollout summary: target={summary.target_object}/{summary.target_color_name} "
        f"selected={summary.selected_object}/{summary.selected_color_name} "
        f"success={summary.success} video_path={summary.video_path}"
    )
    env.close()


if __name__ == "__main__":
    main()
