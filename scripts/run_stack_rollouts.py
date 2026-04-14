from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from openvla_steering.env import RobosuiteEnvConfig, StackTaskEnv, default_video_path
from openvla_steering.model import build_policy_from_config
from openvla_steering.utils.io import write_records_parquet


@hydra.main(version_base=None, config_path="../configs", config_name="scene")
def main(cfg: DictConfig) -> None:
    backend = str(cfg.model.backend)
    if backend in {"openvla", "minivla"}:
        if not bool(cfg.env.use_camera_obs):
            raise RuntimeError(f"{backend} rollouts require `env.use_camera_obs=true`.")
        if not bool(cfg.env.has_offscreen_renderer):
            raise RuntimeError(f"{backend} rollouts require `env.has_offscreen_renderer=true`.")

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
    print(f"Running rollouts with backend={backend}")

    records: list[dict[str, object]] = []
    seed_start = int(cfg.run.seed)
    num_rollouts = int(cfg.run.num_rollouts)
    video_root = Path(str(cfg.run.video_dir))
    for offset in range(num_rollouts):
        seed = seed_start + offset
        video_path = None
        if bool(cfg.run.save_video):
            video_path = default_video_path(
                video_root, seed=seed, target_object=policy.target_object, policy_id=policy.policy_id
            )
        summary, debug_lines = env.run_policy_rollout(
            seed=seed,
            policy=policy,
            num_steps=getattr(getattr(policy, "config", None), "total_steps", int(cfg.env.horizon)),
            save_video=bool(cfg.run.save_video),
            video_path=video_path,
            camera_name=str(cfg.env.camera_name),
        )
        for line in debug_lines:
            print(f"[seed={seed}] {line}")
        records.append(summary.to_record())
        print(
            f"[seed={seed}] target={summary.target_object}/{summary.target_color_name} "
            f"selected={summary.selected_object}/{summary.selected_color_name} "
            f"success={summary.success} cubeA_gain={summary.cubeA_height_gain:.4f} "
            f"cubeB_gain={summary.cubeB_height_gain:.4f}"
        )

    env.close()
    output_path = write_records_parquet(records, str(cfg.run.rollout_log_path))
    print(f"Wrote {len(records)} rollout records to {output_path}")


if __name__ == "__main__":
    main()
