from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from vla_lens import ArraySpec, TraceDataset, TraceManifest, validate_lerobot_v3_dataset
from vla_lens.capture.records import TraceRecord
from vla_lens.lerobot_dataset import write_lerobot_trace_record
from vla_lens.traces import ModelSiteSpec


def test_write_lerobot_root_and_open_with_trace_dataset_api(tmp_path):
    record = _minimal_record("trace-a")

    bundle = write_lerobot_trace_record(record, tmp_path)
    dataset = TraceDataset.open(tmp_path)
    validation = validate_lerobot_v3_dataset(tmp_path)

    assert validation.valid, validation.to_dict()
    assert bundle.manifest.trace_id == "trace-a"
    assert len(dataset.bundles) == 1
    opened = dataset.bundle("trace-a")
    assert opened.actions().shape == (2, 2)
    assert opened.array("action").shape == (2, 2)
    assert "executed_actions" not in set(opened.array_index["name"].astype(str))
    assert opened.array("observation.state").shape == (2, 3)
    assert opened.cameras() == ["main"]
    assert opened.frames("main").shape[:3] == (2, 16, 16)
    assert opened.action_chunks().shape == (1, 2, 2)
    assert set(opened.model_sites["name"].astype(str)) == {"model.layer0.hidden"}


def test_lerobot_root_without_overlay_still_visualizes_robot_episode(tmp_path):
    write_lerobot_trace_record(_minimal_record("trace-a"), tmp_path)
    overlay = tmp_path / "vla_lens"
    if overlay.exists():
        shutil.rmtree(overlay)

    dataset = TraceDataset.open(tmp_path)
    bundle = dataset.bundles[0]

    assert bundle.manifest.trace_id == "episode_000000"
    assert bundle.actions().shape == (2, 2)
    assert bundle.cameras() == ["main"]
    assert bundle.model_sites.empty


def test_multiple_records_append_to_one_lerobot_root(tmp_path):
    write_lerobot_trace_record(_minimal_record("trace-a"), tmp_path)
    write_lerobot_trace_record(_minimal_record("trace-b"), tmp_path)

    dataset = TraceDataset.open(tmp_path)
    refs = pd.read_parquet(tmp_path / "vla_lens" / "tables" / "episode_refs.parquet")

    assert [bundle.manifest.trace_id for bundle in dataset.bundles] == ["trace-a", "trace-b"]
    assert refs["episode_index"].tolist() == [0, 1]
    assert refs["trace_id"].tolist() == ["trace-a", "trace-b"]


def test_trace_dataset_open_discovers_nested_lerobot_roots(tmp_path):
    first_root = (
        tmp_path / "traces" / "dataset-a" / "mechanistic_sampled" / "libero_object" / "task_00"
    )
    second_root = (
        tmp_path / "traces" / "dataset-a" / "mechanistic_sampled" / "libero_goal" / "task_01"
    )
    write_lerobot_trace_record(_minimal_record("trace-a"), first_root)
    write_lerobot_trace_record(_minimal_record("trace-b"), second_root)

    dataset = TraceDataset.open(tmp_path)

    assert dataset.root == tmp_path
    assert sorted(bundle.manifest.trace_id for bundle in dataset.bundles) == [
        "trace-a",
        "trace-b",
    ]
    assert dataset.bundle("trace-a").actions().shape == (2, 2)
    assert dataset.bundle("trace-b").cameras() == ["main"]


def test_nested_lerobot_roots_without_overlay_get_namespaced_trace_ids(tmp_path):
    first_root = tmp_path / "batch" / "libero_object" / "task_00"
    second_root = tmp_path / "batch" / "libero_goal" / "task_01"
    write_lerobot_trace_record(_minimal_record("trace-a"), first_root)
    write_lerobot_trace_record(_minimal_record("trace-b"), second_root)
    shutil.rmtree(first_root / "vla_lens")
    shutil.rmtree(second_root / "vla_lens")

    dataset = TraceDataset.open(tmp_path)
    trace_ids = sorted(bundle.manifest.trace_id for bundle in dataset.bundles)

    assert trace_ids == [
        "batch-libero_goal-task_01__episode_000000",
        "batch-libero_object-task_00__episode_000000",
    ]
    assert dataset.bundle(trace_ids[0]).actions().shape == (2, 2)
    assert len(dataset._bundle_by_trace_id) == 2


def test_duplicate_overlay_trace_ids_fail_loudly_for_nested_roots(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    write_lerobot_trace_record(_minimal_record("trace-a"), first_root)
    write_lerobot_trace_record(_minimal_record("trace-a"), second_root)

    with pytest.raises(ValueError, match="Duplicate trace_id"):
        TraceDataset.open(tmp_path)


def _minimal_record(trace_id: str) -> TraceRecord:
    length = 2
    manifest = TraceManifest(
        trace_id=trace_id,
        episode_id=trace_id,
        task_id="0",
        prompt="pick the cube",
        model_id="test-model",
        env_id="test-env",
        robot_id="test-robot",
        outcome="success",
        length=length,
        metadata={
            "task_name": "pick the cube",
            "capture_profile": "mechanistic_sampled",
            "action_space": {"action_names": ["x", "y"]},
        },
    )
    frames = np.zeros((length, 16, 16, 3), dtype=np.uint8)
    frames[1, :, :, 0] = 255
    return TraceRecord(
        manifest=manifest,
        timesteps=pd.DataFrame(
            {
                "timestep": [0, 1],
                "reward": [0.0, 1.0],
                "done": [False, True],
                "policy_call_index": [0, 0],
                "horizon_index": [0, 1],
            }
        ),
        episode_arrays={
            "executed_actions": ArraySpec(
                np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
                ["timestep", "action_dim"],
            ),
            "frames.main": ArraySpec(frames, ["timestep", "height", "width", "channel"]),
            "robot_joint_pos": ArraySpec(
                np.asarray([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1]], dtype=np.float32),
                ["timestep", "joint"],
            ),
            "action_chunks": ArraySpec(
                np.zeros((1, 2, 2), dtype=np.float32),
                ["policy_call", "horizon", "action_dim"],
            ),
        },
        model_arrays=[
            ModelSiteSpec(
                name="model.layer0.hidden",
                array=np.zeros((1, 2, 4), dtype=np.float32),
                axes=["policy_call", "token", "channel"],
                module="model.layer0",
                layer=0,
            )
        ],
        policy_calls=pd.DataFrame(
            {
                "policy_call_index": [0],
                "observation_timestep": [0],
                "env_timestep_start": [0],
                "env_timestep_end": [1],
            }
        ),
        generation_steps=pd.DataFrame({"policy_call_index": [0], "generation_step": [0]}),
        streams=pd.DataFrame({"stream_id": ["action"], "name": ["action"], "modality": ["action"]}),
        token_spaces=pd.DataFrame(
            {"token_space_id": ["action"], "stream_id": ["action"], "token_count": [2]}
        ),
        tokens=pd.DataFrame({"token_space_id": ["action"], "token_index": [0]}),
        action_normalization=pd.DataFrame(
            {
                "normalization_id": ["identity"],
                "normalized_action_array_ref": ["action_chunks"],
                "unnormalized_action_array_ref": ["executed_actions"],
            }
        ),
        capture_report={"missing_model_sites": []},
    )
