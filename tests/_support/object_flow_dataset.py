from __future__ import annotations

import numpy as np
import pandas as pd

from vla_lens import ArraySpec, TraceBundle, TraceDataset, TraceManifest


def object_flow_dataset(root) -> TraceDataset:
    bundle_path = root / "flow_trace"
    timesteps = 10
    positions = np.zeros((timesteps, 2, 3), dtype=np.float32)
    positions[:, 0] = np.array([0.0, 0.0, 0.02], dtype=np.float32)
    positions[:, 1] = np.array([0.2, 0.0, 0.02], dtype=np.float32)
    positions[3:, 0, 0] += 0.06
    positions[4:, 0, 2] += 0.06
    eef = np.repeat(np.array([[0.4, 0.4, 0.1]], dtype=np.float32), timesteps, axis=0)
    eef[1:3] = np.array([0.0, 0.0, 0.04], dtype=np.float32)
    manifest = TraceManifest(
        trace_id="flow_trace",
        episode_id="flow_trace",
        task_id="0",
        prompt="put the red cube in the blue bowl",
        model_id="pi05",
        env_id="libero_object",
        robot_id="panda",
        outcome="success",
        length=timesteps,
        metadata={"task_name": "LIBERO_OBJECT_put_the_red_cube_in_the_blue_bowl"},
    )
    timestep_index = pd.DataFrame(
        {
            "timestep": np.arange(timesteps),
            "policy_call_index": np.arange(timesteps) // 3,
            "horizon_index": np.arange(timesteps) % 3,
        }
    )
    bundle = TraceBundle.create(
        bundle_path,
        manifest=manifest,
        timesteps=timestep_index,
        policy_calls=pd.DataFrame(
            {
                "policy_call_index": [0, 1, 2, 3],
                "episode_id": ["flow_trace"] * 4,
                "observation_timestep": [0, 3, 6, 9],
                "env_timestep_start": [0, 3, 6, 9],
                "env_timestep_end": [2, 5, 8, 9],
            }
        ),
        generation_steps=pd.DataFrame(
            {
                "policy_call_index": [0, 1, 2, 3],
                "generation_step": [0, 0, 0, 0],
            }
        ),
        streams=pd.DataFrame({"stream_id": ["action"], "name": ["action"], "modality": ["action"]}),
        token_spaces=pd.DataFrame(
            {"token_space_id": ["action"], "stream_id": ["action"], "token_count": [1]}
        ),
        tokens=pd.DataFrame({"token_space_id": ["action"], "token_index": [0]}),
        scene_state=pd.DataFrame(
            {
                "object_index": [0, 1],
                "object_name": ["red_cube_1", "blue_bowl_1"],
                "object_kind": ["object", "object"],
            }
        ),
        episode_arrays={
            "scene_object_pos": ArraySpec(positions, ["timestep", "object", "xyz"]),
            "eef_pos": ArraySpec(eef, ["timestep", "xyz"]),
            "executed_actions": ArraySpec(
                np.zeros((timesteps, 1), dtype=np.float32),
                ["timestep", "action_dim"],
            ),
            "action_chunks": ArraySpec(
                np.zeros((4, 3, 1), dtype=np.float32),
                ["policy_call", "horizon", "action_dim"],
            ),
            "generation_actions": ArraySpec(
                np.zeros((4, 1, 3, 1), dtype=np.float32),
                ["policy_call", "generation_step", "horizon", "action_dim"],
            ),
        },
    )
    return TraceDataset(root, [bundle])
