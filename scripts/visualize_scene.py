from __future__ import annotations

from pathlib import Path

import hydra
import imageio.v2 as imageio
import numpy as np
from omegaconf import DictConfig

from openvla_steering.env.robosuite_env import RobosuiteEnvConfig, get_scene_metadata, make_env
from openvla_steering.utils.seeding import set_global_seed


def _random_action(env) -> np.ndarray:
    low, high = env.action_spec
    return np.random.uniform(low=low, high=high)


def _scripted_pick_action(env, obs: dict, cfg: DictConfig, step: int) -> np.ndarray:
    target_name = str(cfg.policy.target_object)
    if target_name not in {"cubeA", "cubeB"}:
        raise ValueError(f"Unsupported scripted target object: {target_name}")

    target_pos = np.array(obs[f"{target_name}_pos"])
    eef_pos = np.array(obs["robot0_eef_pos"])
    action = np.zeros(env.action_dim, dtype=np.float64)

    phase_steps = cfg.policy.phase_steps
    approach_end = int(phase_steps.approach)
    descend_end = approach_end + int(phase_steps.descend)
    close_end = descend_end + int(phase_steps.close)

    if step < approach_end:
        desired = target_pos + np.array([0.0, 0.0, float(cfg.policy.approach_height)])
        grip = float(cfg.policy.open_gripper)
        phase = "approach"
    elif step < descend_end:
        desired = target_pos + np.array([0.0, 0.0, float(cfg.policy.grasp_offset)])
        grip = float(cfg.policy.open_gripper)
        phase = "descend"
    elif step < close_end:
        desired = target_pos + np.array([0.0, 0.0, float(cfg.policy.grasp_offset)])
        grip = float(cfg.policy.close_gripper)
        phase = "close"
    else:
        desired = target_pos + np.array([0.0, 0.0, float(cfg.policy.lift_height)])
        grip = float(cfg.policy.close_gripper)
        phase = "lift"

    delta = (desired - eef_pos) * float(cfg.policy.position_gain)
    action[:3] = np.clip(delta, -1.0, 1.0)
    action[-1] = grip
    return action, phase


def _maybe_make_parent(path_str: str) -> None:
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


@hydra.main(version_base=None, config_path="../configs", config_name="scene")
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.run.seed))

    env = make_env(
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
        )
    )

    obs = env.reset()
    print(f"Reset complete. Observation keys: {sorted(obs.keys())}")
    print(f"Action dimension: {env.action_dim}")
    print(f"Scene metadata: {get_scene_metadata(env)}")

    frames: list[np.ndarray] = []
    for step in range(int(cfg.run.steps)):
        if str(cfg.policy.name) == "scripted_pick":
            action, phase = _scripted_pick_action(env, obs, cfg, step)
        else:
            action = _random_action(env)
            phase = "random"
        obs, reward, done, info = env.step(action)
        if cfg.run.save_video:
            frame = env.sim.render(
                camera_name=str(cfg.env.camera_name),
                width=640,
                height=480,
                depth=False,
            )
            frames.append(np.flipud(frame))
        if step % 25 == 0:
            target_name = str(cfg.policy.target_object)
            target_pos = obs.get(f"{target_name}_pos")
            eef_pos = obs.get("robot0_eef_pos")
            dist = None
            if target_pos is not None and eef_pos is not None:
                dist = float(np.linalg.norm(np.array(eef_pos) - np.array(target_pos)))
            print(
                f"step={step} phase={phase} reward={reward:.4f} done={done} "
                f"target_dist={dist if dist is not None else 'n/a'}"
            )
        if done:
            print(f"Episode ended early at step {step}")
            break

    if cfg.run.save_video and frames:
        _maybe_make_parent(str(cfg.run.video_path))
        imageio.mimsave(str(cfg.run.video_path), frames, fps=20)
        print(f"Saved video to {cfg.run.video_path}")

    env.close()


if __name__ == "__main__":
    main()
