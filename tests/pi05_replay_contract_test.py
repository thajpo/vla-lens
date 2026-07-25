from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vla_lens.pi05.replay import policy_call_replay_inputs, replay_config_from_bundle
from vla_lens.traces import ArraySpec, TraceBundle, TraceManifest


def _replay_bundle(
    tmp_path,
    *,
    include_exact_noise: bool,
    generation_dtype: str = "float16",
    task_id: str = "7",
    metadata: dict[str, object] | None = None,
) -> TraceBundle:
    episode_arrays = {
        "executed_actions": ArraySpec(
            np.zeros((6, 2), dtype=np.float32),
            ["timestep", "action_dim"],
        ),
        "action_chunks": ArraySpec(
            np.arange(8, dtype=np.float32).reshape(1, 4, 2),
            ["policy_call", "horizon", "action_dim"],
        ),
        "generation_actions": ArraySpec(
            np.arange(24, dtype=generation_dtype).reshape(1, 3, 4, 2),
            ["policy_call", "generation_step", "horizon", "action_dim"],
        ),
    }
    if include_exact_noise:
        episode_arrays["flow_initial_noise"] = ArraySpec(
            np.linspace(-1.0, 1.0, 8, dtype=np.float32).reshape(1, 4, 2),
            ["policy_call", "horizon", "action_dim"],
        )

    return TraceBundle.create(
        tmp_path / "trace-a",
        manifest=TraceManifest(
            trace_id="trace-a",
            episode_id="trace-a",
            task_id=task_id,
            prompt="pick up the mug",
            model_id="lerobot/pi05_libero_finetuned",
            env_id="libero_object",
            robot_id="libero_panda",
            outcome="failure",
            length=6,
            metadata=metadata
            if metadata is not None
            else {
                "seed": 1234,
                "environment": {
                    "benchmark": "libero_object",
                    "task_id": 7,
                    "layout_id": 3,
                    "obs_size": 224,
                },
            },
        ),
        timesteps=pd.DataFrame(
            {
                "timestep": range(6),
                "policy_call_index": [0] * 6,
                "horizon_index": range(6),
            }
        ),
        policy_calls=pd.DataFrame(
            {
                "policy_call_index": [0],
                "observation_timestep": [2],
                "env_timestep_start": [2],
                "env_timestep_end": [5],
                "preprocess_id": ["lerobot.default"],
                "postprocess_id": ["lerobot.default"],
            }
        ),
        generation_steps=pd.DataFrame(
            {"policy_call_index": [0, 0, 0], "generation_step": [0, 1, 2]}
        ),
        episode_arrays=episode_arrays,
    )


def test_replay_config_resolves_current_capture_manifest_and_environment_metadata(tmp_path):
    bundle = _replay_bundle(tmp_path, include_exact_noise=True)

    config = replay_config_from_bundle(bundle)

    assert config.benchmark == "libero_object"
    assert config.task_id == 7
    assert config.layout_id == 3
    assert config.seed == 1234
    assert config.obs_size == 224
    assert config.horizon == 6


def test_replay_config_keeps_scene_mutation_recipe(tmp_path):
    mutation = {
        "spec": {
            "kind": "pose_exchange",
            "objects": ["black_book_1", "white_yellow_mug_1"],
        },
        "outside_object_qpos_max_abs": 0.0,
    }
    bundle = _replay_bundle(
        tmp_path,
        include_exact_noise=True,
        metadata={
            "seed": 1234,
            "environment": {
                "benchmark": "libero_object",
                "task_id": 7,
                "layout_id": 3,
                "obs_size": 224,
                "scene_mutation": mutation,
            },
        },
    )

    config = replay_config_from_bundle(bundle)

    assert config.scene_mutation == mutation


def test_replay_config_preserves_legacy_flat_metadata_aliases(tmp_path):
    bundle = _replay_bundle(
        tmp_path,
        include_exact_noise=False,
        task_id="",
        metadata={
            "benchmark": "libero_goal",
            "task_id": 5,
            "layout_episode_index": 2,
            "env_seed": 99,
            "obs_size": 128,
        },
    )

    config = replay_config_from_bundle(bundle)

    assert config.benchmark == "libero_goal"
    assert config.task_id == 5
    assert config.layout_id == 2
    assert config.seed == 99
    assert config.obs_size == 128


def test_replay_config_does_not_invent_layout_zero_when_capture_omits_layout(tmp_path):
    bundle = _replay_bundle(
        tmp_path,
        include_exact_noise=True,
        metadata={
            "environment": {
                "benchmark": "libero_object",
                "task_id": 7,
                "seed": 1234,
                "obs_size": 224,
            }
        },
    )

    config = replay_config_from_bundle(bundle)

    assert config.layout_id is None


def test_policy_call_replay_summary_excludes_tensor_values(tmp_path):
    bundle = _replay_bundle(tmp_path, include_exact_noise=True)

    summary = policy_call_replay_inputs(bundle, 0).summary()

    assert summary["initial_noise"] == {
        "ref": "flow_initial_noise[0]",
        "exactness": "exact",
        "shape": [4, 2],
        "dtype": "float32",
    }
    assert summary["stored_action_chunk"] == {"shape": [4, 2], "dtype": "float32"}


def test_policy_call_replay_inputs_prefer_exact_flow_initial_noise(tmp_path):
    bundle = _replay_bundle(tmp_path, include_exact_noise=True)

    inputs = policy_call_replay_inputs(bundle, 0)

    assert inputs.trace_id == "trace-a"
    assert inputs.policy_call_index == 0
    assert inputs.observation_timestep == 2
    assert inputs.initial_noise_ref == "flow_initial_noise[0]"
    assert inputs.initial_noise_exactness == "exact"
    assert inputs.initial_noise.dtype == np.float32
    np.testing.assert_array_equal(inputs.stored_action_chunk, bundle.action_chunks()[0])
    np.testing.assert_array_equal(inputs.initial_noise, bundle.array("flow_initial_noise")[0])
    assert inputs.policy_call["preprocess_id"] == "lerobot.default"


def test_policy_call_replay_inputs_fall_back_to_quantized_generation_step_zero(tmp_path):
    bundle = _replay_bundle(tmp_path, include_exact_noise=False)

    inputs = policy_call_replay_inputs(bundle, 0)

    assert inputs.initial_noise_ref == "generation_actions[0,0]"
    assert inputs.initial_noise_exactness == "quantized"
    assert inputs.initial_noise.dtype == np.float16
    np.testing.assert_array_equal(inputs.initial_noise, bundle.generation_actions()[0, 0])


def test_policy_call_replay_inputs_reject_unknown_policy_call(tmp_path):
    bundle = _replay_bundle(tmp_path, include_exact_noise=True)

    with pytest.raises(KeyError, match="policy call 5"):
        policy_call_replay_inputs(bundle, 5)
