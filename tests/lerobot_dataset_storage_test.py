from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from vla_lens import (
    ActivationQuery,
    ArraySpec,
    LensArtifact,
    TraceDataset,
    TraceManifest,
    query_table,
    table_catalog,
    validate_lerobot_v3_dataset,
    validate_workbench_contracts,
)
from vla_lens.capture.records import TraceRecord
from vla_lens.lerobot_dataset import write_lerobot_trace_record
from vla_lens.pi05.batch_capture import CaptureCommand, _expected_trace_exists
from vla_lens.pi05.plan_capture import _command_expected_traces_exist, _validate_task_root
from vla_lens.server import _dataset_signature
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


def test_lerobot_overlay_model_sites_materialize_with_selectors(tmp_path):
    write_lerobot_trace_record(_minimal_record("trace-a"), tmp_path)
    dataset = TraceDataset.open(tmp_path)

    X, rows = dataset.select_model_sites(
        ActivationQuery(name="model.layer0.hidden", reduce_tokens="mean")
    ).to_matrix(cache=False)

    assert X.shape == (1, 4)
    assert rows["trace_id"].tolist() == ["trace-a"]
    assert rows["activation"].tolist() == ["model.layer0.hidden"]


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


def test_lerobot_stats_are_recomputed_across_appended_records(tmp_path):
    first = _minimal_record("trace-a")
    second = _minimal_record("trace-b")
    second.episode_arrays["executed_actions"].array[:] = np.asarray(
        [[10.0, 20.0], [30.0, 40.0]],
        dtype=np.float32,
    )

    write_lerobot_trace_record(first, tmp_path)
    write_lerobot_trace_record(second, tmp_path)

    stats = json.loads((tmp_path / "meta" / "stats.json").read_text(encoding="utf-8"))

    assert stats["action"]["min"] == pytest.approx([0.1, 0.2])
    assert stats["action"]["max"] == pytest.approx([30.0, 40.0])
    assert stats["action"]["mean"] == pytest.approx([10.1, 15.15])


def test_lerobot_task_index_is_dense_for_nonzero_source_task_id(tmp_path):
    write_lerobot_trace_record(_minimal_record("trace-a", task_id="5"), tmp_path)

    task_rows = pd.read_parquet(tmp_path / "meta" / "tasks.parquet").reset_index()
    data_rows = pd.read_parquet(tmp_path / "data" / "chunk-000" / "file-000.parquet")

    assert task_rows["task_index"].tolist() == [0]
    assert data_rows["task_index"].tolist() == [0, 0]
    assert TraceDataset.open(tmp_path).bundle("trace-a").manifest.task_id == "5"


def test_length_changing_overwrite_is_rejected(tmp_path):
    write_lerobot_trace_record(_minimal_record("trace-a", length=2), tmp_path)
    write_lerobot_trace_record(_minimal_record("trace-b", length=2), tmp_path)

    with pytest.raises(ValueError, match="different length"):
        write_lerobot_trace_record(_minimal_record("trace-a", length=3), tmp_path, overwrite=True)


def test_lerobot_overwrite_does_not_leave_stale_camera_features(tmp_path):
    write_lerobot_trace_record(_minimal_record("trace-a", include_frames=True), tmp_path)
    write_lerobot_trace_record(
        _minimal_record("trace-a", include_frames=False),
        tmp_path,
        overwrite=True,
    )

    info = json.loads((tmp_path / "meta" / "info.json").read_text(encoding="utf-8"))
    dataset = TraceDataset.open(tmp_path)

    assert dataset.bundle("trace-a").cameras() == []
    assert not any(str(name).startswith("observation.images.") for name in info["features"])
    assert not list((tmp_path / "videos").rglob("*.mp4"))


def test_lerobot_append_rejects_robot_feature_schema_drift(tmp_path):
    write_lerobot_trace_record(_minimal_record("trace-a", include_frames=True), tmp_path)

    with pytest.raises(ValueError, match="feature schemas"):
        write_lerobot_trace_record(_minimal_record("trace-b", include_frames=False), tmp_path)


def test_lerobot_validator_checks_required_fields_in_data_shards(tmp_path):
    write_lerobot_trace_record(_minimal_record("trace-a"), tmp_path)
    data_path = tmp_path / "data" / "chunk-000" / "file-000.parquet"
    data = pd.read_parquet(data_path).drop(columns=["action"])
    data.to_parquet(data_path, index=False)

    validation = validate_lerobot_v3_dataset(tmp_path)

    assert not validation.valid
    assert "missing_data_field" in {issue.code for issue in validation.errors}


def test_plan_capture_completion_checks_lerobot_trace_ids(tmp_path):
    task_root = (
        tmp_path / "traces" / "dataset-a" / "mechanistic_sampled" / "libero_goal" / "task_01"
    )
    write_lerobot_trace_record(_minimal_record("trace-a"), task_root)
    command = CaptureCommand(
        dataset_id="dataset-a",
        benchmark="libero_goal",
        task_id=1,
        start_seed=1000,
        episodes=2,
        capture_profile="mechanistic_sampled",
        output_root=task_root,
        expected_paths=(task_root, task_root),
        expected_trace_ids=("trace-a", "trace-b"),
        command=("python", "-m", "vla_lens.pi05.capture"),
    )

    assert not _command_expected_traces_exist(command)

    write_lerobot_trace_record(_minimal_record("trace-b"), task_root)

    assert _command_expected_traces_exist(command)


def test_plan_capture_validates_lerobot_task_root(tmp_path):
    write_lerobot_trace_record(_minimal_record("trace-a"), tmp_path)

    _validate_task_root(tmp_path)


def test_batch_completion_check_rejects_stale_overlay_refs(tmp_path):
    write_lerobot_trace_record(_minimal_record("trace-a"), tmp_path)
    assert _expected_trace_exists(tmp_path, "trace-a")

    shutil.rmtree(tmp_path / "vla_lens" / "episodes" / "episode_000000")

    assert not _expected_trace_exists(tmp_path, "trace-a")


def test_lerobot_overlay_bundle_artifacts_load_through_trace_dataset(tmp_path):
    write_lerobot_trace_record(_minimal_record("trace-a"), tmp_path)
    dataset = TraceDataset.open(tmp_path)
    artifact = LensArtifact.create(artifact_type="probe", name="Bundle probe")
    saved = dataset.bundle("trace-a").save_artifact(
        artifact,
        arrays={"weights": np.asarray([[1.0, 2.0]], dtype=np.float32)},
    )

    reopened = TraceDataset.open(tmp_path)
    loaded = reopened.load_artifact(saved.artifact_id)
    weights = reopened.load_artifact_array(loaded, "weights")

    assert loaded.artifact_id == saved.artifact_id
    assert weights.shape == (1, 2)


def test_lerobot_dataset_artifacts_are_stored_under_overlay(tmp_path):
    write_lerobot_trace_record(_minimal_record("trace-a"), tmp_path)
    write_lerobot_trace_record(_minimal_record("trace-b"), tmp_path)
    dataset = TraceDataset.open(tmp_path)

    artifact = LensArtifact.create(artifact_type="probe", name="Dataset probe")
    saved = dataset.save_artifact(
        artifact,
        arrays={"weights": np.asarray([[1.0]], dtype=np.float32)},
    )

    assert (tmp_path / "vla_lens" / "artifacts" / saved.artifact_id / "artifact.json").exists()
    assert (tmp_path / "vla_lens" / "tables" / "artifact_index.parquet").exists()
    assert not (tmp_path / "tables" / "artifact_index.parquet").exists()
    assert TraceDataset.open(tmp_path).load_artifact_array(saved, "weights").shape == (1, 1)


def test_lerobot_roots_pass_workbench_contract_and_table_queries(tmp_path):
    write_lerobot_trace_record(_minimal_record("trace-a"), tmp_path)
    dataset = TraceDataset.open(tmp_path)

    validation = validate_workbench_contracts(dataset)
    model_sites = query_table(dataset, table="model_sites", limit=5)
    tables = {table.table_id: table.to_dict() for table in table_catalog(dataset)}

    assert validation["valid"], validation
    assert model_sites["total"] == 1
    assert model_sites["rows"][0]["trace_id"] == "trace-a"
    assert "model_sites" in tables
    assert "vla_lens/episodes" in tables["model_sites"]["storage"]["uri"]


def test_lerobot_dataset_signature_tracks_overlay_appends(tmp_path):
    before = _dataset_signature(tmp_path)
    write_lerobot_trace_record(_minimal_record("trace-a"), tmp_path)
    after_first = _dataset_signature(tmp_path)
    write_lerobot_trace_record(_minimal_record("trace-b"), tmp_path)
    after_second = _dataset_signature(tmp_path)

    assert before[0] == 0
    assert after_first[0] == 1
    assert after_second[0] == 2
    assert after_first != after_second


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


def _minimal_record(
    trace_id: str,
    *,
    length: int = 2,
    task_id: str = "0",
    include_frames: bool = True,
    include_observation_state: bool = True,
) -> TraceRecord:
    manifest = TraceManifest(
        trace_id=trace_id,
        episode_id=trace_id,
        task_id=task_id,
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
    frames[-1, :, :, 0] = 255
    actions = np.stack(
        [
            np.linspace(0.1, 0.3, length, dtype=np.float32),
            np.linspace(0.2, 0.4, length, dtype=np.float32),
        ],
        axis=1,
    )
    robot_joint_pos = np.stack(
        [
            np.linspace(1.0, 1.1, length, dtype=np.float32),
            np.linspace(2.0, 2.1, length, dtype=np.float32),
            np.linspace(3.0, 3.1, length, dtype=np.float32),
        ],
        axis=1,
    )
    episode_arrays = {
        "executed_actions": ArraySpec(
            actions,
            ["timestep", "action_dim"],
        ),
        "action_chunks": ArraySpec(
            np.zeros((1, 2, 2), dtype=np.float32),
            ["policy_call", "horizon", "action_dim"],
        ),
    }
    if include_frames:
        episode_arrays["frames.main"] = ArraySpec(
            frames,
            ["timestep", "height", "width", "channel"],
        )
    if include_observation_state:
        episode_arrays["robot_joint_pos"] = ArraySpec(
            robot_joint_pos,
            ["timestep", "joint"],
        )
    return TraceRecord(
        manifest=manifest,
        timesteps=pd.DataFrame(
            {
                "timestep": np.arange(length, dtype=np.int64),
                "reward": np.linspace(0.0, 1.0, length, dtype=np.float32),
                "done": [False] * max(0, length - 1) + [True],
                "policy_call_index": np.zeros(length, dtype=np.int64),
                "horizon_index": np.arange(length, dtype=np.int64),
            }
        ),
        episode_arrays=episode_arrays,
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
