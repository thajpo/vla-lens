"""Synthetic trace dataset generation for developing the VLA-lens substrate."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from vla_lens.artifacts import LensArtifact
from vla_lens.traces import ActivationSpec, ArraySpec, TraceBundle, TraceDataset, TraceManifest


def create_synthetic_trace_dataset(
    root: str | Path,
    *,
    num_episodes: int = 3,
    timesteps: int = 24,
    layers: int = 6,
    seed: int = 0,
    overwrite: bool = False,
) -> TraceDataset:
    """Create a small trace dataset with camera frames, flow actions, and artifacts.

    The synthetic data exists to make the trace/index/dashboard contract concrete before
    model-specific capture adapters are ready.
    """
    root = Path(root)
    if overwrite and root.exists():
        for bundle_path in root.glob("synthetic_*.vlatrace"):
            shutil.rmtree(bundle_path)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for episode_idx in range(num_episodes):
        trace_id = f"synthetic_{episode_idx:03d}"
        outcome = "success" if episode_idx % 2 == 0 else "failure"
        task_id = "pick_red_cube" if episode_idx != 2 else "pick_blue_cube"
        prompt = "pick up the red cube" if task_id == "pick_red_cube" else "pick up the blue cube"
        bundle_path = root / f"{trace_id}.vlatrace"

        episode = _make_episode_arrays(
            timesteps=timesteps,
            episode_idx=episode_idx,
            outcome=outcome,
            rng=rng,
        )
        streams, token_spaces, tokens = _make_token_layout()
        activation_arrays = _make_activation_arrays(
            timesteps=timesteps,
            layers=layers,
            outcome=outcome,
            target_patch_by_timestep=episode["target_patch_by_timestep"],
            rng=rng,
        )
        context_arrays, context_tables = _make_context_payload(
            timesteps=timesteps,
            episode_idx=episode_idx,
            outcome=outcome,
        )

        manifest = TraceManifest(
            trace_id=trace_id,
            episode_id=trace_id,
            task_id=task_id,
            prompt=prompt,
            model_id="synthetic-flow-policy",
            env_id="synthetic-tabletop",
            robot_id="synthetic-7dof",
            outcome=outcome,
            length=timesteps,
            metadata={
                "split": "train" if episode_idx < max(1, num_episodes - 1) else "test",
                "target_object": "red_cube" if task_id == "pick_red_cube" else "blue_cube",
                "seed": seed + episode_idx,
            },
        )
        timestep_index = pd.DataFrame(
            {
                "timestep": np.arange(timesteps, dtype=np.int32),
                "phase": _phase_labels(timesteps),
                "reward": np.linspace(0.0, 1.0 if outcome == "success" else 0.25, timesteps),
            }
        )
        timestep_index["policy_call_index"] = timestep_index["timestep"] // 4
        timestep_index["horizon_index"] = timestep_index["timestep"] % 4
        policy_call_timesteps = np.arange(0, timesteps, 4, dtype=np.int32)
        policy_calls = pd.DataFrame(
            {
                "policy_call_index": np.arange(len(policy_call_timesteps), dtype=np.int32),
                "episode_id": trace_id,
                "observation_timestep": policy_call_timesteps,
                "env_timestep_start": policy_call_timesteps,
                "env_timestep_end": np.minimum(policy_call_timesteps + 3, timesteps - 1),
                "model_id": "synthetic-flow-policy",
                "model_family": "synthetic",
                "model_call_kind": "policy_action_chunk",
                "action_generator_kind": "flow_matching",
                "action_horizon": 8,
                "action_dim": 7,
            }
        )
        generation_steps = _make_generation_steps(policy_calls, generation_step_count=5)
        bundle = TraceBundle.create(
            bundle_path,
            manifest=manifest,
            timesteps=timestep_index,
            policy_calls=policy_calls,
            generation_steps=generation_steps,
            streams=streams,
            token_spaces=token_spaces,
            tokens=tokens,
            episode_arrays={
                "frames.main": ArraySpec(
                    episode["frames_main"],
                    ["timestep", "height", "width", "rgb"],
                ),
                "frames.wrist": ArraySpec(
                    episode["frames_wrist"],
                    ["timestep", "height", "width", "rgb"],
                ),
                "executed_actions": ArraySpec(
                    episode["executed_actions"],
                    ["timestep", "action_dim"],
                ),
                "action_chunks": ArraySpec(
                    episode["action_chunks"][policy_call_timesteps],
                    ["policy_call", "horizon", "action_dim"],
                ),
                "generation_actions": ArraySpec(
                    episode["generation_actions"][policy_call_timesteps],
                    ["policy_call", "generation_step", "horizon", "action_dim"],
                ),
                **context_arrays,
            },
            model_arrays=activation_arrays,
            robot_state=context_tables["robot_state"],
            scene_state=context_tables["scene_state"],
            camera_state=context_tables["camera_state"],
            evaluation=context_tables["evaluation"],
            image_preprocessing=context_tables["image_preprocessing"],
            prompt_metadata=context_tables["prompt_metadata"],
            action_normalization=context_tables["action_normalization"],
            capture_request={"requested_profile": "mechanistic_all"},
            capture_plan={"actual_profile": "mechanistic_all", "complete": True},
            capture_report={
                "requested_profile": "mechanistic_all",
                "actual_profile": "mechanistic_all",
                "complete": True,
                "captured_cheap_fields": sorted(context_tables),
                "unavailable_cheap_fields": [],
                "missing_model_sites": [],
            },
            overwrite=overwrite,
        )
        _save_synthetic_artifacts(
            bundle,
            layers=layers,
            timesteps=timesteps,
            target_patch_by_timestep=episode["target_patch_by_timestep"],
            outcome=outcome,
        )

    return TraceDataset.open(root)


def _make_episode_arrays(
    *,
    timesteps: int,
    episode_idx: int,
    outcome: str,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    height, width = 96, 128
    horizon, action_dim, generation_steps = 8, 7, 5
    frames_main = np.zeros((timesteps, height, width, 3), dtype=np.uint8)
    frames_wrist = np.zeros_like(frames_main)
    target_patch_by_timestep = np.zeros((timesteps, 2), dtype=np.int32)

    target_start = np.array([24 + 7 * episode_idx, 30], dtype=np.float32)
    target_end = np.array([88, 62], dtype=np.float32)
    distractor = np.array([92, 34 + 5 * episode_idx], dtype=np.float32)

    for t in range(timesteps):
        alpha = t / max(1, timesteps - 1)
        target = (1.0 - alpha) * target_start + alpha * target_end
        target_patch_by_timestep[t] = _patch_index(target, height=height, width=width)
        frames_main[t] = _base_frame(height, width, tint=(235, 237, 232))
        frames_wrist[t] = _base_frame(height, width, tint=(228, 233, 239))
        _draw_square(frames_main[t], target, size=12, color=(214, 54, 48))
        _draw_square(frames_main[t], distractor, size=12, color=(54, 103, 202))
        _draw_square(
            frames_wrist[t],
            target * np.array([0.9, 1.05]),
            size=14,
            color=(214, 54, 48),
        )
        _draw_square(
            frames_wrist[t],
            distractor * np.array([1.05, 0.92]),
            size=10,
            color=(54, 103, 202),
        )

    final_direction = np.array([0.16, -0.02, 0.04, 0.0, 0.0, 0.03, 0.1], dtype=np.float32)
    if outcome == "failure":
        final_direction = np.array([-0.04, 0.15, -0.02, 0.0, 0.0, -0.02, -0.05], dtype=np.float32)

    action_chunks = np.zeros((timesteps, horizon, action_dim), dtype=np.float32)
    generation_actions = np.zeros(
        (timesteps, generation_steps, horizon, action_dim),
        dtype=np.float32,
    )
    for t in range(timesteps):
        commitment = 0.25 + 0.75 * (t / max(1, timesteps - 1))
        for h in range(horizon):
            action_chunks[t, h] = commitment * final_direction * (1.0 + 0.08 * h)
            action_chunks[t, h, 2] += 0.02 * np.sin((t + h) / 3.0)
        for s in range(generation_steps):
            alpha = (s + 1) / generation_steps
            noise = rng.normal(0.0, 0.06 * (1.0 - alpha), size=(horizon, action_dim))
            generation_actions[t, s] = alpha * action_chunks[t] + noise

    executed_actions = action_chunks[:, 0] + rng.normal(0.0, 0.005, size=(timesteps, action_dim))
    return {
        "frames_main": frames_main,
        "frames_wrist": frames_wrist,
        "executed_actions": executed_actions.astype(np.float32),
        "action_chunks": action_chunks,
        "generation_actions": generation_actions.astype(np.float32),
        "target_patch_by_timestep": target_patch_by_timestep,
    }


def _make_activation_arrays(
    *,
    timesteps: int,
    layers: int,
    outcome: str,
    target_patch_by_timestep: np.ndarray,
    rng: np.random.Generator,
) -> list[ActivationSpec]:
    patch_tokens = 32
    horizon_tokens = 8
    hidden = 12
    specs: list[ActivationSpec] = []

    for layer in range(layers):
        image_acts = rng.normal(
            0.0,
            0.05,
            size=(timesteps, patch_tokens, hidden),
        ).astype(np.float32)
        action_acts = rng.normal(
            0.0,
            0.05,
            size=(timesteps, horizon_tokens, hidden),
        ).astype(np.float32)
        layer_gain = (layer + 1) / layers
        for t in range(timesteps):
            row, col = target_patch_by_timestep[t]
            target_token = int(row * 4 + col)
            wrist_token = 16 + target_token
            time_gain = t / max(1, timesteps - 1)
            sign = 1.0 if outcome == "success" else -0.6
            image_acts[t, target_token, 0] += sign * layer_gain * (0.4 + time_gain)
            image_acts[t, wrist_token, 1] += sign * layer_gain * (0.2 + time_gain)
            action_acts[t, :, 0] += sign * layer_gain * time_gain
            action_acts[t, :, 2] += layer_gain * np.linspace(-0.2, 0.2, horizon_tokens)
        specs.append(
            ActivationSpec(
                name=f"backbone.layers.{layer}.resid",
                array=image_acts,
                axes=["timestep", "token", "channel"],
                module=f"backbone.layers.{layer}.resid",
                layer=layer,
                tensor_type="resid",
                token_kind="image_patch",
                token_space_id="synthetic.image_prefix",
            )
        )
        specs.append(
            ActivationSpec(
                name=f"action_head.layers.{layer}.resid",
                array=action_acts,
                axes=["timestep", "token", "channel"],
                module=f"action_head.layers.{layer}.resid",
                layer=layer,
                tensor_type="resid",
                token_kind="action",
                token_space_id="synthetic.action_suffix",
            )
        )
    return specs


def _make_generation_steps(
    policy_calls: pd.DataFrame,
    *,
    generation_step_count: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for call_index in policy_calls["policy_call_index"].astype(int):
        for generation_step in range(generation_step_count):
            records.append(
                {
                    "policy_call_index": call_index,
                    "generation_step": generation_step,
                    "process_kind": "flow_matching",
                    "t": float(generation_step / max(1, generation_step_count - 1)),
                    "dt": float(1.0 / generation_step_count),
                    "sigma": np.nan,
                    "scheduler_index": generation_step,
                    "decode_index": np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def _make_context_payload(
    *,
    timesteps: int,
    episode_idx: int,
    outcome: str,
) -> tuple[dict[str, ArraySpec], dict[str, pd.DataFrame]]:
    t = np.linspace(0.0, 1.0, timesteps, dtype=np.float32)
    joint_positions = np.stack(
        [0.2 * np.sin(t * np.pi + dim * 0.2) for dim in range(7)],
        axis=1,
    ).astype(np.float32)
    joint_velocities = np.gradient(joint_positions, axis=0).astype(np.float32)
    eef_pose = np.stack(
        [
            0.35 + 0.12 * t,
            -0.08 + 0.02 * episode_idx + 0.03 * t,
            0.22 + 0.05 * t,
            np.ones_like(t),
            np.zeros_like(t),
            np.zeros_like(t),
            np.zeros_like(t),
        ],
        axis=1,
    ).astype(np.float32)
    gripper_state = np.stack([1.0 - t, t], axis=1).astype(np.float32)
    object_poses = np.zeros((timesteps, 2, 7), dtype=np.float32)
    object_poses[:, 0, :3] = np.stack([0.45 + 0.08 * t, -0.12 + 0.02 * t, 0.04 + 0 * t], axis=1)
    object_poses[:, 1, :3] = np.array([0.32, 0.18 + 0.01 * episode_idx, 0.04], dtype=np.float32)
    object_poses[:, :, 3] = 1.0
    camera_extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, None, :, :], timesteps, axis=0)
    camera_extrinsics = np.repeat(camera_extrinsics, 2, axis=1)
    camera_extrinsics[:, 1, 0, 3] = eef_pose[:, 0]
    camera_extrinsics[:, 1, 1, 3] = eef_pose[:, 1]
    camera_extrinsics[:, 1, 2, 3] = eef_pose[:, 2]

    arrays = {
        "robot_joint_positions": ArraySpec(joint_positions, ["timestep", "joint"]),
        "robot_joint_velocities": ArraySpec(joint_velocities, ["timestep", "joint"]),
        "robot_eef_pose": ArraySpec(eef_pose, ["timestep", "pose_component"]),
        "robot_gripper_state": ArraySpec(gripper_state, ["timestep", "gripper_component"]),
        "scene_object_poses": ArraySpec(object_poses, ["timestep", "object", "pose_component"]),
        "camera_extrinsics": ArraySpec(
            camera_extrinsics,
            ["timestep", "camera", "matrix_row", "matrix_col"],
        ),
    }
    reward = np.linspace(0.0, 1.0 if outcome == "success" else 0.25, timesteps)
    tables = {
        "robot_state": pd.DataFrame.from_records(
            [
                {
                    "field_name": "joint_positions",
                    "array_id": "robot_joint_positions",
                    "entity_id": "synthetic_arm",
                    "unit": "radian",
                    "source": "synthetic",
                },
                {
                    "field_name": "joint_velocities",
                    "array_id": "robot_joint_velocities",
                    "entity_id": "synthetic_arm",
                    "unit": "radian_per_second",
                    "source": "synthetic",
                },
                {
                    "field_name": "eef_pose",
                    "array_id": "robot_eef_pose",
                    "entity_id": "synthetic_eef",
                    "unit": "meter_quaternion",
                    "source": "synthetic",
                },
                {
                    "field_name": "gripper_state",
                    "array_id": "robot_gripper_state",
                    "entity_id": "synthetic_gripper",
                    "unit": "normalized",
                    "source": "synthetic",
                },
            ]
        ),
        "scene_state": pd.DataFrame.from_records(
            [
                {
                    "object_id": "target_cube",
                    "name": "target_cube",
                    "category": "cube",
                    "pose_array_id": "scene_object_poses",
                    "object_index": 0,
                    "source": "synthetic",
                },
                {
                    "object_id": "distractor_cube",
                    "name": "distractor_cube",
                    "category": "cube",
                    "pose_array_id": "scene_object_poses",
                    "object_index": 1,
                    "source": "synthetic",
                },
            ]
        ),
        "camera_state": pd.DataFrame.from_records(
            [
                {
                    "camera_id": "main",
                    "name": "main",
                    "width": 128,
                    "height": 96,
                    "intrinsics": "[[100.0, 0.0, 64.0], [0.0, 100.0, 48.0], [0.0, 0.0, 1.0]]",
                    "extrinsics_array_id": "camera_extrinsics",
                    "camera_index": 0,
                    "extrinsics_static": True,
                },
                {
                    "camera_id": "wrist",
                    "name": "wrist",
                    "width": 128,
                    "height": 96,
                    "intrinsics": "[[96.0, 0.0, 64.0], [0.0, 96.0, 48.0], [0.0, 0.0, 1.0]]",
                    "extrinsics_array_id": "camera_extrinsics",
                    "camera_index": 1,
                    "extrinsics_static": False,
                },
            ]
        ),
        "evaluation": pd.DataFrame.from_records(
            [
                {
                    "timestep": idx,
                    "metric_name": "reward",
                    "metric_value": float(value),
                    "threshold": np.nan,
                    "passed": bool(value > 0.5),
                    "source": "synthetic",
                }
                for idx, value in enumerate(reward)
            ]
            + [
                {
                    "timestep": timesteps - 1,
                    "metric_name": "success",
                    "metric_value": float(outcome == "success"),
                    "threshold": 1.0,
                    "passed": outcome == "success",
                    "source": "synthetic",
                }
            ]
        ),
        "image_preprocessing": pd.DataFrame.from_records(
            [
                {
                    "policy_call_index": -1,
                    "camera_id": "main",
                    "raw_shape": "[96, 128, 3]",
                    "processed_shape": "[96, 128, 3]",
                    "resize_mode": "none",
                    "value_range": "uint8_0_255",
                },
                {
                    "policy_call_index": -1,
                    "camera_id": "wrist",
                    "raw_shape": "[96, 128, 3]",
                    "processed_shape": "[96, 128, 3]",
                    "resize_mode": "none",
                    "value_range": "uint8_0_255",
                },
            ]
        ),
        "prompt_metadata": pd.DataFrame.from_records(
            [
                {
                    "prompt_id": "synthetic_prompt",
                    "raw_task": "pick up the target cube",
                    "cleaned_task": "pick up the target cube",
                    "formatted_prompt": "Task: pick up the target cube; Action:",
                    "state_bin_count": np.nan,
                    "source": "synthetic",
                }
            ]
        ),
        "action_normalization": pd.DataFrame.from_records(
            [
                {
                    "normalization_id": "synthetic_action_identity",
                    "mode": "identity",
                    "stats_ref": "",
                    "action_dim_names": ('["x", "y", "z", "roll", "pitch", "yaw", "gripper"]'),
                    "normalized_action_array_ref": "action_chunks",
                    "unnormalized_action_array_ref": "executed_actions",
                }
            ]
        ),
    }
    return arrays, tables


def _save_synthetic_artifacts(
    bundle: TraceBundle,
    *,
    layers: int,
    timesteps: int,
    target_patch_by_timestep: np.ndarray,
    outcome: str,
) -> None:
    probe_margin = np.zeros((layers, timesteps), dtype=np.float32)
    attention = np.zeros((layers, timesteps, 4, 4), dtype=np.float32)
    sign = 1.0 if outcome == "success" else -0.65
    for layer in range(layers):
        layer_gain = (layer + 1) / layers
        for t in range(timesteps):
            time_gain = t / max(1, timesteps - 1)
            probe_margin[layer, t] = sign * layer_gain * (0.15 + time_gain)
            row, col = target_patch_by_timestep[t]
            attention[layer, t] += 0.03
            attention[layer, t, row, col] += max(0.0, probe_margin[layer, t]) + 0.25
            attention[layer, t] /= attention[layer, t].sum()

    probe = LensArtifact.create(
        artifact_type="probe_suite",
        name="target-object probe by layer/time",
        group_id="target_object_probes",
        selector={
            "module": "backbone.layers.*.resid",
            "token_kind": "image_patch",
            "reduce_tokens": "mean",
        },
        method={"probe": "synthetic_logistic_probe", "sweep": ["layer", "timestep"]},
        metrics={
            "best_margin": float(probe_margin.max()),
            "worst_margin": float(probe_margin.min()),
        },
        display={"primary_array": "margin", "axes": ["layer", "timestep"]},
        tags=("synthetic", "probe"),
    )
    bundle.save_artifact(probe, arrays={"margin": probe_margin})

    attention_artifact = LensArtifact.create(
        artifact_type="attention_map",
        name="action-token attention to main camera",
        group_id="attention_overlays",
        selector={
            "query": "action_token:horizon_0",
            "key": "image_patch:main",
            "module": "backbone.layers.*.attn",
        },
        method={"method": "synthetic_attention_mass"},
        display={
            "primary_array": "attention",
            "axes": ["layer", "timestep", "patch_row", "patch_col"],
            "camera": "main",
        },
        tags=("synthetic", "attention"),
    )
    bundle.save_artifact(attention_artifact, arrays={"attention": attention})


def _make_token_layout() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    streams = pd.DataFrame.from_records(
        [
            {
                "stream_id": "synthetic.image.main",
                "name": "main camera image patches",
                "modality": "image",
                "camera_id": "main",
                "description": "Synthetic main-camera 4x4 patch grid.",
            },
            {
                "stream_id": "synthetic.image.wrist",
                "name": "wrist camera image patches",
                "modality": "image",
                "camera_id": "wrist",
                "description": "Synthetic wrist-camera 4x4 patch grid.",
            },
            {
                "stream_id": "synthetic.action",
                "name": "action suffix",
                "modality": "action",
                "camera_id": "",
                "description": "Continuous action-horizon suffix tokens.",
            },
        ]
    )
    token_spaces = pd.DataFrame.from_records(
        [
            {
                "token_space_id": "synthetic.image_prefix",
                "policy_call_index": -1,
                "segment": "image_prefix",
                "stream_id": "synthetic.image.main",
                "token_count": 32,
            },
            {
                "token_space_id": "synthetic.action_suffix",
                "policy_call_index": -1,
                "segment": "action_suffix",
                "stream_id": "synthetic.action",
                "token_count": 8,
            },
        ]
    )
    records: list[dict[str, object]] = []
    global_token = 0
    for camera in ["main", "wrist"]:
        for row in range(4):
            for col in range(4):
                records.append(
                    {
                        "token_space_id": "synthetic.image_prefix",
                        "token_index": global_token,
                        "global_prefix_index": global_token,
                        "modality": "image",
                        "segment": "image_prefix",
                        "token_kind": "image_patch",
                        "token_type": "image_patch",
                        "stream_id": f"synthetic.image.{camera}",
                        "camera_id": camera,
                        "patch_row": row,
                        "patch_col": col,
                        "patch_x0": col * 32,
                        "patch_x1": (col + 1) * 32,
                        "patch_y0": row * 24,
                        "patch_y1": (row + 1) * 24,
                        "pixel_y0": row * 24,
                        "pixel_y1": (row + 1) * 24,
                        "pixel_x0": col * 32,
                        "pixel_x1": (col + 1) * 32,
                        "is_padding": False,
                        "attention_mask": True,
                    }
                )
                global_token += 1
    for horizon_index in range(8):
        records.append(
            {
                "token_space_id": "synthetic.action_suffix",
                "token_index": horizon_index,
                "global_prefix_index": np.nan,
                "modality": "action",
                "segment": "action_suffix",
                "token_kind": "action",
                "token_type": "continuous_action",
                "stream_id": "synthetic.action",
                "action_horizon_index": horizon_index,
                "is_padding": False,
                "attention_mask": True,
            }
        )
    return streams, token_spaces, pd.DataFrame.from_records(records)


def _base_frame(height: int, width: int, tint: tuple[int, int, int]) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    y_grad = np.linspace(0, 18, height, dtype=np.uint8)[:, None]
    x_grad = np.linspace(0, 12, width, dtype=np.uint8)[None, :]
    for channel, base in enumerate(tint):
        frame[:, :, channel] = np.clip(base - y_grad + x_grad, 0, 255)
    frame[70:75, 12:116] = (176, 180, 170)
    return frame


def _draw_square(
    frame: np.ndarray,
    center_xy: np.ndarray,
    *,
    size: int,
    color: tuple[int, int, int],
) -> None:
    x, y = center_xy.astype(int)
    half = size // 2
    y0, y1 = max(0, y - half), min(frame.shape[0], y + half)
    x0, x1 = max(0, x - half), min(frame.shape[1], x + half)
    frame[y0:y1, x0:x1] = color


def _patch_index(center_xy: np.ndarray, *, height: int, width: int) -> np.ndarray:
    x, y = center_xy
    col = int(np.clip(x / width * 4, 0, 3))
    row = int(np.clip(y / height * 4, 0, 3))
    return np.array([row, col], dtype=np.int32)


def _phase_labels(timesteps: int) -> list[str]:
    labels = []
    for t in range(timesteps):
        alpha = t / max(1, timesteps - 1)
        if alpha < 0.33:
            labels.append("approach")
        elif alpha < 0.66:
            labels.append("commit")
        else:
            labels.append("execute")
    return labels


__all__ = ["create_synthetic_trace_dataset"]
