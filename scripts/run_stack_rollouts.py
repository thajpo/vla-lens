from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from openvla_steering.env import (
    ObjectMetadata,
    RobosuiteEnvConfig,
    ScriptedPickPolicyConfig,
    StackTaskMetadata,
    StackTaskEnv,
    default_video_path,
)
from openvla_steering.utils.io import write_records_parquet


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
        ),
        StackTaskMetadata(
            cubeA=ObjectMetadata(
                object_id="cubeA",
                color_name=str(cfg.objects.cubeA.color_name),
                rgba=[float(value) for value in cfg.objects.cubeA.rgba],
            ),
            cubeB=ObjectMetadata(
                object_id="cubeB",
                color_name=str(cfg.objects.cubeB.color_name),
                rgba=[float(value) for value in cfg.objects.cubeB.rgba],
            ),
        ),
    )
    policy = ScriptedPickPolicyConfig(
        target_object=str(cfg.policy.target_object),
        approach_height=float(cfg.policy.approach_height),
        grasp_offset=float(cfg.policy.grasp_offset),
        lift_height=float(cfg.policy.lift_height),
        position_gain=float(cfg.policy.position_gain),
        open_gripper=float(cfg.policy.open_gripper),
        close_gripper=float(cfg.policy.close_gripper),
        approach_steps=int(cfg.policy.phase_steps.approach),
        descend_steps=int(cfg.policy.phase_steps.descend),
        close_steps=int(cfg.policy.phase_steps.close),
        lift_steps=int(cfg.policy.phase_steps.lift),
    )

    records: list[dict[str, object]] = []
    seed_start = int(cfg.run.seed)
    num_rollouts = int(cfg.run.num_rollouts)
    video_root = Path(str(cfg.run.video_dir))
    for offset in range(num_rollouts):
        seed = seed_start + offset
        video_path = None
        if bool(cfg.run.save_video):
            video_path = default_video_path(video_root, seed=seed, target_object=policy.target_object)
        summary, debug_lines = env.run_scripted_pick(
            seed=seed,
            policy=policy,
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
