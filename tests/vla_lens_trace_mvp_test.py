from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import vla_lens.workbench as workbench_module
from vla_lens import (
    FULL_REQUIRED_MODEL_SITE_ROLES,
    ActivationQuery,
    ActivationSpec,
    AnalysisRunSpec,
    ArraySpec,
    InterventionRunSpec,
    SavedWorkspace,
    SelectionState,
    TraceBundle,
    TraceDataset,
    TraceManifest,
    UnitRef,
    cohort_from_selection,
    compare_cohorts,
    create_synthetic_trace_dataset,
    graph_from_selection,
    list_analysis_runs,
    list_cohorts,
    list_workspaces,
    normalize_axis_values,
    projection_points,
    query_table,
    resolve_selection,
    resolve_workspace,
    save_analysis_run,
    save_cohort,
    save_intervention_run,
    save_pi05_interaction_metrics_artifact,
    save_workspace,
    spatial_overlay_contracts,
    table_catalog,
    unit_profile,
    validate_trace_bundle,
    validate_workbench_contracts,
    workbench_manifest,
)
from vla_lens.action_generation import save_action_generation_artifact
from vla_lens.analyzer import diagnostics_status, run_dataset_diagnostics
from vla_lens.pi05.capture import AUDIT_WINDOWED_LAYERS
from vla_lens.probes import dump_probe_spec, train_probe_artifact_from_spec
from vla_lens.server import (
    _activation_sites_payload,
    _artifact_detail_payload,
    _artifacts_payload,
    _attention_map_payload,
    _create_action_generation_payload,
    _create_outcome_probe_payload,
    _create_target_object_probe_payload,
    _dataset_diagnostics_payload,
    _dataset_payload,
    _episode_interactions_payload,
    _episode_metrics_payload,
    _episode_video_path,
    _expert_token_details_payload,
    _lens_array_meta_payload,
    _lens_array_slice_payload,
    _lens_arrays_payload,
    _object_camera_overlay_payload,
    _prompt_attention_payload,
    _resolve_selection_payload,
    _run_dataset_diagnostics_payload,
    _save_analysis_run_payload,
    _save_cohort_from_selection_payload,
    _save_intervention_run_payload,
    _save_workspace_payload,
    _table_query_payload,
    _unit_profile_payload,
    _workbench_payload,
)
from vla_lens.target_object import save_target_object_encoding_artifact
from vla_lens.traces import ModelSiteSpec as TraceModelSiteSpec
from vla_lens.workbench import ImageFrameSpec, LensArraySpec, StorageRef, TableSpec


def _make_minimal_trace(
    path,
    *,
    profile: str = "rollout",
    model_sites: list[TraceModelSiteSpec] | None = None,
    streams: dict[str, list[object]] | None = None,
    token_spaces: dict[str, list[object]] | None = None,
    tokens: dict[str, list[object]] | None = None,
    include_frames: bool = False,
    action_normalization: dict[str, list[object]] | None = None,
    extra_episode_arrays: dict[str, ArraySpec] | None = None,
    scene_state: pd.DataFrame | None = None,
    camera_state: pd.DataFrame | None = None,
    metadata: dict[str, object] | None = None,
) -> TraceBundle:
    length = 2
    manifest = TraceManifest(
        trace_id=path.stem,
        episode_id=path.stem,
        task_id="minimal",
        prompt="minimal",
        model_id="minimal-model",
        env_id="minimal-env",
        robot_id="minimal-robot",
        outcome="unknown",
        length=length,
        metadata={"capture_profile": profile, **(metadata or {})},
    )
    timesteps = {
        "timestep": [0, 1],
        "reward": [0.0, 0.0],
        "policy_call_index": [0, 0],
        "horizon_index": [0, 1],
    }
    policy_calls = {
        "policy_call_index": [0],
        "episode_id": [path.stem],
        "observation_timestep": [0],
        "env_timestep_start": [0],
        "env_timestep_end": [1],
    }
    episode_arrays = {
        "executed_actions": ArraySpec(
            np.zeros((length, 1), dtype=np.float32),
            ["timestep", "action_dim"],
        ),
        "action_chunks": ArraySpec(
            np.zeros((1, length, 1), dtype=np.float32),
            ["policy_call", "horizon", "action_dim"],
        ),
        "generation_actions": ArraySpec(
            np.zeros((1, 1, length, 1), dtype=np.float32),
            ["policy_call", "generation_step", "horizon", "action_dim"],
        ),
    }
    if include_frames:
        episode_arrays["frames.main"] = ArraySpec(
            np.zeros((length, 16, 16, 3), dtype=np.uint8),
            ["timestep", "height", "width", "channel"],
        )
    if extra_episode_arrays:
        episode_arrays.update(extra_episode_arrays)
    return TraceBundle.create(
        path,
        manifest=manifest,
        timesteps=pd.DataFrame(timesteps),
        policy_calls=pd.DataFrame(policy_calls),
        generation_steps=pd.DataFrame({"policy_call_index": [0], "generation_step": [0]}),
        streams=pd.DataFrame(
            streams or {"stream_id": ["action"], "name": ["action"], "modality": ["action"]}
        ),
        token_spaces=pd.DataFrame(
            token_spaces
            or {"token_space_id": ["action"], "stream_id": ["action"], "token_count": [2]}
        ),
        tokens=pd.DataFrame(
            tokens or {"token_space_id": ["action"], "token_index": [0], "token_kind": ["action"]}
        ),
        action_normalization=pd.DataFrame(
            action_normalization
            or {
                "normalization_id": ["identity"],
                "mode": ["identity"],
                "stats_ref": [""],
            }
        ),
        scene_state=scene_state,
        camera_state=camera_state,
        episode_arrays=episode_arrays,
        model_arrays=model_sites or (),
        capture_report={"missing_model_sites": []},
    )


def test_synthetic_dataset_indexes_and_stats(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=3, timesteps=12)

    assert len(dataset.episode_index) == 3
    assert {"trace_id", "task_id", "outcome", "path"}.issubset(dataset.episode_index.columns)
    assert not dataset.timestep_index.empty
    assert not dataset.stats.by_task().empty
    coverage = dataset.stats.activation_coverage()
    assert set(coverage["token_kind"]) >= {"image_patch", "action"}


def test_trace_bundle_writes_probe_provenance_fingerprints(tmp_path):
    bundle = _make_minimal_trace(tmp_path / "fingerprinted.vlatrace")

    fingerprints = bundle.fingerprints
    assert fingerprints["algorithm"] == "sha256"
    assert fingerprints["fingerprint_schema_version"] == 1
    assert fingerprints["trajectory_fingerprint"].startswith("sha256:")
    assert fingerprints["context_fingerprint"].startswith("sha256:")
    assert fingerprints["trace_schema_fingerprint"].startswith("sha256:")
    assert fingerprints["trace_fingerprint"].startswith("sha256:")
    assert bundle.manifest.metadata["fingerprints"] == fingerprints
    assert bundle.capture_report["fingerprints"] == fingerprints
    assert validate_trace_bundle(bundle).valid

    changed_actions = ArraySpec(
        np.ones((2, 1), dtype=np.float32),
        ["timestep", "action_dim"],
    )
    changed = _make_minimal_trace(
        tmp_path / "fingerprinted_changed.vlatrace",
        extra_episode_arrays={"executed_actions": changed_actions},
    )
    assert changed.fingerprints["trajectory_fingerprint"] != fingerprints["trajectory_fingerprint"]


def test_dataset_payload_uses_workbench_contract_not_legacy_schema(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=3, timesteps=12)

    manifest = workbench_manifest(dataset)
    payload = _dataset_payload(dataset)

    assert "schema" not in payload
    assert payload["workbench"]["dataset_id"] == manifest["dataset_id"]
    assert payload["workbench"]["axes"]["layer"]["values"]
    assert any(
        workflow["workflow_id"] == "target_object_encoding"
        for workflow in payload["workbench"]["workflow_presets"]
    )


def test_dataset_payload_groups_counterfactual_pairs(tmp_path):
    clean = _make_minimal_trace(
        tmp_path / "pi05_mechanistic_sampled_libero_goal_task1_seed42_clean.vlatrace",
        metadata={
            "capture_design": "paired_counterfactual",
            "trace_variant": "clean",
            "counterfactual_group_id": "group-1",
            "counterfactual_role": "clean",
            "counterfactual_type": "prompt_target_swap",
            "pair_index": 0,
            "paired_trace_id": "pi05_mechanistic_sampled_libero_goal_task1_seed42_corrupt",
            "changed_fields": ["prompt.target_object"],
            "matched_fields": ["benchmark", "task_id", "seed"],
            "target_object_id": "mug",
        },
    )
    corrupt = _make_minimal_trace(
        tmp_path / "pi05_mechanistic_sampled_libero_goal_task1_seed42_corrupt.vlatrace",
        metadata={
            "capture_design": "paired_counterfactual",
            "trace_variant": "corrupt",
            "counterfactual": {
                "group_id": "group-1",
                "role": "corrupt",
                "type": "prompt_target_swap",
                "pair_index": 1,
                "paired_trace_id": clean.manifest.trace_id,
                "changed_fields": ["prompt.target_object"],
                "matched_fields": ["benchmark", "task_id", "seed"],
                "target_object_id": "bowl",
            },
        },
    )
    dataset = TraceDataset(tmp_path, [clean, corrupt])

    payload = _dataset_payload(dataset)

    assert payload["counterfactual_pairs"] == [
        {
            "group_id": "group-1",
            "type": "prompt_target_swap",
            "changed_fields": ["prompt.target_object"],
            "matched_fields": ["benchmark", "task_id", "seed"],
            "members": [
                {
                    "trace_id": clean.manifest.trace_id,
                    "episode_id": clean.manifest.episode_id,
                    "role": "clean",
                    "pair_index": 0,
                    "paired_trace_id": corrupt.manifest.trace_id,
                    "target_object_id": "mug",
                    "counterfactual_target_object_id": "",
                    "outcome": "unknown",
                    "prompt": "minimal",
                },
                {
                    "trace_id": corrupt.manifest.trace_id,
                    "episode_id": corrupt.manifest.episode_id,
                    "role": "corrupt",
                    "pair_index": 1,
                    "paired_trace_id": clean.manifest.trace_id,
                    "target_object_id": "bowl",
                    "counterfactual_target_object_id": "",
                    "outcome": "unknown",
                    "prompt": "minimal",
                },
            ],
        }
    ]


def test_workbench_manifest_exposes_axis_native_handles(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=3, timesteps=12)

    manifest = workbench_manifest(dataset)
    payload = _workbench_payload(dataset)

    assert "layer" in manifest["axes"]
    assert any(array["kind"] == "image_sequence" for array in manifest["lens_arrays"])
    assert any(array["kind"] == "tensor" for array in manifest["lens_arrays"])
    assert manifest["image_frames"]
    assert manifest["media"]
    assert {frame["storage"]["format"] for frame in manifest["image_frames"]} == {"jpeg"}
    assert any(item["kind"] == "jpeg_sequence" for item in manifest["media"])
    assert manifest["model_sites"]
    assert any(panel["panel_type"] == "heatmap" for panel in manifest["panel_recipes"])
    assert any(
        workflow["workflow_id"] == "target_object_encoding"
        for workflow in manifest["workflow_presets"]
    )
    assert any(table["table_id"] == "timesteps" for table in manifest["tables"])
    assert {table["storage"]["format"] for table in manifest["tables"]} == {"parquet"}
    assert payload["tables"] == manifest["tables"]
    assert any(
        score["score_type"] == "intervention_delta" for score in manifest["overlay_score_types"]
    )
    assert any(edge["edge_type"] == "attention_weight" for edge in manifest["graph_edge_types"])
    assert payload["dataset_id"] == manifest["dataset_id"]


def test_empty_dataset_has_structural_workbench_contract(tmp_path):
    dataset = TraceDataset(tmp_path / "empty", [])

    manifest = workbench_manifest(dataset)
    validation = validate_workbench_contracts(dataset)

    assert manifest["dataset_id"] == "empty"
    assert manifest["lens_arrays"] == []
    assert manifest["tables"] == []
    assert manifest["image_frames"] == []
    assert manifest["media"] == []
    assert manifest["model_sites"] == []
    assert validation["valid"] is True


def test_unit_profile_links_unit_to_top_episode_examples(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=3, timesteps=8)
    site = workbench_manifest(dataset)["model_sites"][0]
    unit = UnitRef(kind="neuron", site_id=site["site_id"], index=0)

    profile = unit_profile(dataset, unit, top_k=5)
    api_profile = _unit_profile_payload(
        dataset,
        {
            "kind": ["neuron"],
            "site_id": [site["site_id"]],
            "unit": ["0"],
            "top_k": ["5"],
        },
    )

    assert profile["unit_ref"]["index"] == 0
    assert profile["top_examples"]
    assert profile["lens_arrays"]
    assert profile["suggested_panels"]
    assert api_profile["top_examples"]


def test_lens_array_data_plane_exposes_bounded_slices(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=8)
    arrays = _lens_arrays_payload(dataset)
    tensor = next(
        array
        for array in arrays["lens_arrays"]
        if array["kind"] == "tensor" and "timestep" in array["dims"]
    )

    meta = _lens_array_meta_payload(dataset, tensor["array_id"])
    sliced = _lens_array_slice_payload(
        dataset,
        tensor["array_id"],
        {
            "selection": {"timestep": {"start": 0, "end": 1}},
            "max_values": 8,
        },
    )

    assert arrays["total"] >= 1
    assert meta["array_id"] == tensor["array_id"]
    assert sliced["array"]["array_id"] == tensor["array_id"]
    assert sliced["shape"][0] == 2
    assert "summary" in sliced
    assert "values" in sliced or "preview" in sliced


def test_trace_creation_writes_zarr_arrays_and_parquet_indexes(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=8)
    bundle = dataset.bundles[0]
    frame_mask = bundle.array_index["name"].astype(str).str.startswith("frames.")
    frame_rows = bundle.array_index.loc[frame_mask]
    dense_rows = bundle.array_index.loc[~frame_mask]

    assert (bundle.path / "tables" / "array_index.parquet").exists()
    assert (bundle.path / "tables" / "model_sites.parquet").exists()
    assert (bundle.path / "tables" / "policy_calls.parquet").exists()
    assert set(frame_rows["storage_format"]) == {"jpeg"}
    assert set(dense_rows["storage_format"]) == {"zarr"}
    assert set(bundle.model_sites["storage_format"]) == {"zarr"}
    assert all(str(path).endswith(".zarr") for path in dense_rows["relative_path"])
    assert all(str(path).startswith("media/frames/") for path in frame_rows["relative_path"])
    assert not list(bundle.path.glob("arrays/**/*.npy"))
    assert list((bundle.path / str(frame_rows.iloc[0]["relative_path"])).glob("*.jpg"))
    assert bundle.actions(mmap=True).shape[0] == 8
    assert bundle.frames("main", mmap=True).shape == (8, 96, 128, 3)
    assert (bundle.path / "tables" / "generation_steps.parquet").exists()
    assert (bundle.path / "tables" / "streams.parquet").exists()
    assert (bundle.path / "tables" / "token_spaces.parquet").exists()
    assert (bundle.path / "tables" / "robot_state.parquet").exists()
    assert (bundle.path / "tables" / "scene_state.parquet").exists()
    assert (bundle.path / "tables" / "camera_state.parquet").exists()
    assert (bundle.path / "tables" / "evaluation.parquet").exists()
    assert not bundle.robot_state.empty
    assert not bundle.scene_state.empty
    assert not bundle.camera_state.empty
    assert not bundle.evaluation.empty


def test_object_camera_overlay_projects_scene_objects_to_frame(tmp_path):
    intrinsics = np.array([[[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]]], dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)[None, :, :]
    object_pos = np.array(
        [
            [[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [0.2, 0.0, 1.0]],
        ],
        dtype=np.float32,
    )
    object_quat = np.zeros((2, 2, 4), dtype=np.float32)
    object_quat[..., 3] = 1.0
    bundle = _make_minimal_trace(
        tmp_path / "object_overlay.vlatrace",
        include_frames=True,
        extra_episode_arrays={
            "scene_object_pos": ArraySpec(
                object_pos,
                ["timestep", "object", "xyz"],
                metadata={"object_names": ["cube", "mug"]},
            ),
            "scene_object_quat": ArraySpec(
                object_quat,
                ["timestep", "object", "xyzw"],
                metadata={"object_names": ["cube", "mug"]},
            ),
            "camera_intrinsics": ArraySpec(
                intrinsics,
                ["camera", "row", "col"],
                metadata={"camera_names": ["agentview"]},
            ),
            "camera_extrinsics": ArraySpec(
                extrinsics,
                ["camera", "row", "col"],
                metadata={"camera_names": ["agentview"]},
            ),
            "camera_resolution": ArraySpec(
                np.array([[16, 16]], dtype=np.int32),
                ["camera", "height_width"],
                metadata={"camera_names": ["agentview"]},
            ),
        },
        scene_state=pd.DataFrame.from_records(
            [
                {
                    "context_kind": "object",
                    "object_index": 0,
                    "object_name": "cube",
                    "object_kind": "object",
                    "source": "test",
                    "pos_array_id": "scene_object_pos",
                    "quat_array_id": "scene_object_quat",
                },
                {
                    "context_kind": "object",
                    "object_index": 1,
                    "object_name": "mug",
                    "object_kind": "object",
                    "source": "test",
                    "pos_array_id": "scene_object_pos",
                    "quat_array_id": "scene_object_quat",
                },
                {
                    "context_kind": "object",
                    "object_index": 99,
                    "object_name": "debug_site",
                    "object_kind": "site",
                    "source": "test",
                },
            ]
        ),
        camera_state=pd.DataFrame.from_records(
            [{"camera_name": "agentview", "height": 16, "width": 16}]
        ),
    )

    payload = _object_camera_overlay_payload(
        bundle,
        {"camera": ["main"], "timestep": ["0"]},
    )

    assert payload["available"] is True
    assert payload["calibration_camera"] == "agentview"
    assert payload["visible_count"] == 2
    assert [item["object_name"] for item in payload["objects"]] == ["cube", "mug"]
    assert payload["objects"][0]["in_frame"] is True
    assert payload["objects"][0]["pixel_x"] == 8.0
    assert payload["objects"][0]["pixel_y"] == 8.0
    assert payload["objects"][0]["x"] == 0.5
    assert payload["objects"][0]["y"] == 0.5
    assert payload["objects"][0]["projection_kind"] == "object_pose_center"


def test_object_camera_overlay_prefers_geometry_bbox_anchor(tmp_path):
    intrinsics = np.array([[[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]]], dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)[None, :, :]
    object_pos = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
    object_bbox = np.array([[[[0.1, 0.1, 1.0], [0.3, 0.3, 1.0]]]], dtype=np.float32)
    bundle = _make_minimal_trace(
        tmp_path / "object_overlay_bbox.vlatrace",
        include_frames=True,
        extra_episode_arrays={
            "scene_object_pos": ArraySpec(
                object_pos,
                ["timestep", "object", "xyz"],
                metadata={"object_names": ["cube"]},
            ),
            "scene_object_bbox_world": ArraySpec(
                object_bbox,
                ["timestep", "object", "bound", "xyz"],
                metadata={"object_names": ["cube"]},
            ),
            "camera_intrinsics": ArraySpec(
                intrinsics,
                ["camera", "row", "col"],
                metadata={"camera_names": ["agentview"]},
            ),
            "camera_extrinsics": ArraySpec(
                extrinsics,
                ["camera", "row", "col"],
                metadata={"camera_names": ["agentview"]},
            ),
            "camera_resolution": ArraySpec(
                np.array([[16, 16]], dtype=np.int32),
                ["camera", "height_width"],
                metadata={"camera_names": ["agentview"]},
            ),
        },
        scene_state=pd.DataFrame.from_records(
            [
                {
                    "context_kind": "object",
                    "object_index": 0,
                    "object_name": "cube",
                    "object_kind": "object",
                    "source": "test",
                    "pos_array_id": "scene_object_pos",
                    "bbox_array_id": "scene_object_bbox_world",
                }
            ]
        ),
        camera_state=pd.DataFrame.from_records(
            [{"camera_name": "agentview", "height": 16, "width": 16}]
        ),
    )

    payload = _object_camera_overlay_payload(
        bundle,
        {"camera": ["main"], "timestep": ["0"]},
    )

    assert payload["visible_count"] == 1
    assert payload["objects"][0]["projection_kind"] == "object_geometry_bbox"
    assert payload["objects"][0]["pixel_x"] == 10.0
    assert payload["objects"][0]["pixel_y"] == 10.0
    assert payload["objects"][0]["bbox"]["x0"] == 9.0 / 16.0
    assert payload["objects"][0]["bbox"]["y0"] == 9.0 / 16.0
    assert payload["objects"][0]["bbox"]["x1"] == 11.0 / 16.0
    assert payload["objects"][0]["bbox"]["y1"] == 11.0 / 16.0


def test_object_camera_overlay_prefers_camera_segmentation_bbox(tmp_path):
    intrinsics = np.array([[[10.0, 0.0, 8.0], [0.0, 10.0, 8.0], [0.0, 0.0, 1.0]]], dtype=np.float32)
    extrinsics = np.eye(4, dtype=np.float32)[None, :, :]
    object_pos = np.array([[[0.0, 0.0, 1.0]]], dtype=np.float32)
    camera_bbox = np.array([[[[2.0, 4.0, 10.0, 12.0]]]], dtype=np.float32)
    bundle = _make_minimal_trace(
        tmp_path / "object_overlay_camera_bbox.vlatrace",
        include_frames=True,
        extra_episode_arrays={
            "scene_object_pos": ArraySpec(
                object_pos,
                ["timestep", "object", "xyz"],
                metadata={"object_names": ["cube"]},
            ),
            "camera_object_bbox": ArraySpec(
                camera_bbox,
                ["timestep", "camera", "object", "bbox_xyxy"],
                metadata={
                    "camera_names": ["agentview"],
                    "object_names": ["cube"],
                    "bbox_format": "pixel_xyxy_exclusive",
                },
            ),
            "camera_object_visible": ArraySpec(
                np.ones((1, 1, 1), dtype=np.uint8),
                ["timestep", "camera", "object"],
                metadata={"camera_names": ["agentview"], "object_names": ["cube"]},
            ),
            "camera_intrinsics": ArraySpec(
                intrinsics,
                ["camera", "row", "col"],
                metadata={"camera_names": ["agentview"]},
            ),
            "camera_extrinsics": ArraySpec(
                extrinsics,
                ["camera", "row", "col"],
                metadata={"camera_names": ["agentview"]},
            ),
            "camera_resolution": ArraySpec(
                np.array([[16, 16]], dtype=np.int32),
                ["camera", "height_width"],
                metadata={"camera_names": ["agentview"]},
            ),
        },
        scene_state=pd.DataFrame.from_records(
            [
                {
                    "context_kind": "object",
                    "object_index": 0,
                    "object_name": "cube",
                    "object_kind": "object",
                    "source": "test",
                    "pos_array_id": "scene_object_pos",
                }
            ]
        ),
        camera_state=pd.DataFrame.from_records(
            [{"camera_name": "agentview", "height": 16, "width": 16}]
        ),
    )

    payload = _object_camera_overlay_payload(
        bundle,
        {"camera": ["main"], "timestep": ["0"]},
    )

    assert payload["objects"][0]["projection_kind"] == "camera_segmentation_bbox"
    assert payload["objects"][0]["pixel_x"] == 6.0
    assert payload["objects"][0]["pixel_y"] == 8.0
    assert payload["objects"][0]["bbox"]["source"] == "camera_segmentation"


def test_workbench_storage_refs_use_zarr_for_dense_and_jpeg_for_frames(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=8)
    manifest = workbench_manifest(dataset)
    dense_arrays = [
        array for array in manifest["lens_arrays"] if array["kind"] in {"tensor", "artifact_array"}
    ]
    image_arrays = [array for array in manifest["lens_arrays"] if array["kind"] == "image_sequence"]

    assert dense_arrays
    assert image_arrays
    assert {array["storage"]["format"] for array in dense_arrays} == {"zarr"}
    assert all(array["storage"]["chunks"] for array in dense_arrays if array["shape"])
    assert all(array["storage"]["compression"] == "zstd" for array in dense_arrays)
    assert {array["storage"]["format"] for array in image_arrays} == {"jpeg"}
    assert all(array["storage"]["compression"] == "jpeg" for array in image_arrays)


def test_artifact_arrays_are_saved_as_zarr_storage_refs(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=3, timesteps=8)
    saved = save_action_generation_artifact(dataset, name="Action generation zarr")
    reopened = TraceDataset.open(dataset.root)
    artifact = reopened.load_artifact(saved.artifact.artifact_id)
    manifest = workbench_manifest(reopened)
    artifact_arrays = [
        array
        for array in manifest["lens_arrays"]
        if array["array_id"].startswith(f"artifact.{saved.artifact.artifact_id}.")
    ]

    assert all(path.endswith(".zarr") for path in artifact.arrays.values())
    assert artifact_arrays
    assert {array["storage"]["format"] for array in artifact_arrays} == {"zarr"}
    assert (
        reopened.load_artifact_array(artifact, "delta_to_final").shape == saved.delta_to_final.shape
    )


def test_duckdb_table_query_filters_parquet_indexes(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=8)
    trace_id = dataset.bundles[0].manifest.trace_id

    queried = query_table(
        dataset,
        table="arrays",
        filters={"trace_id": [trace_id], "storage_format": ["zarr"]},
        columns=["trace_id", "name", "storage_format", "relative_path"],
        limit=20,
    )

    assert queried["total"] >= 1
    assert queried["columns"] == ["trace_id", "name", "storage_format", "relative_path"]
    assert all(row["trace_id"] == trace_id for row in queried["rows"])
    assert all(row["storage_format"] == "zarr" for row in queried["rows"])
    assert all(row["relative_path"].endswith(".zarr") for row in queried["rows"])


def test_table_catalog_exposes_parquet_storage_refs(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=8)

    tables = {table.table_id: table.to_dict() for table in table_catalog(dataset)}

    expected = {
        "timesteps",
        "policy_calls",
        "tokens",
        "array_index",
        "model_sites",
        "artifact_index",
    }
    assert expected.issubset(tables)
    assert all(table["storage"]["format"] == "parquet" for table in tables.values())
    assert tables["timesteps"]["row_count"] == 16
    assert "trace_id" in tables["timesteps"]["columns"]
    assert list((dataset.root).glob(tables["timesteps"]["storage"]["uri"]))


def test_trace_dataset_open_discovers_nested_trace_bundles(tmp_path):
    root = tmp_path / "external-dataset"
    _make_minimal_trace(
        root / "traces" / "mechanistic_light" / "libero_object" / "task_00" / "a.vlatrace"
    )
    _make_minimal_trace(
        root / "traces" / "mechanistic_light" / "libero_goal" / "task_01" / "b.vlatrace"
    )

    dataset = TraceDataset.open(root)
    tables = {table.table_id: table.to_dict() for table in table_catalog(dataset)}
    payload = _dataset_payload(dataset)

    assert len(dataset.bundles) == 2
    assert {bundle.manifest.trace_id for bundle in dataset.bundles} == {"a", "b"}
    assert payload["root"] == str(root)
    assert len(payload["episodes"]) == 2
    assert list(root.glob(tables["timesteps"]["storage"]["uri"]))


def test_richer_trace_tables_are_cataloged_and_queryable(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=8)

    tables = {table.table_id: table.to_dict() for table in table_catalog(dataset)}
    generation_steps = query_table(
        dataset,
        table="generation_steps",
        filters={"generation_step": [1]},
        columns=["trace_id", "policy_call_index", "generation_step", "t"],
    )
    streams = query_table(dataset, table="streams", columns=["trace_id", "stream_id", "modality"])
    token_spaces = query_table(
        dataset,
        table="token_spaces",
        columns=["trace_id", "token_space_id", "token_count"],
    )
    context = query_table(
        dataset,
        table="context",
        filters={"context_table": ["robot_state"]},
        columns=["trace_id", "context_table", "field_name", "array_id"],
    )

    assert {"generation_steps", "streams", "token_spaces", "context"}.issubset(tables)
    assert tables["context"]["provenance"]["context_tables"]
    assert generation_steps["rows"][0]["generation_step"] == 1
    assert any(row["stream_id"] == "synthetic.action" for row in streams["rows"])
    assert any(row["token_space_id"] == "synthetic.action_suffix" for row in token_spaces["rows"])
    assert context["rows"][0]["context_table"] == "robot_state"


def test_model_site_catalog_exposes_richer_schema_fields(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=8)
    bundle = dataset.bundles[0]
    table_path = bundle.path / TraceBundle.MODEL_SITES
    sites = pd.read_parquet(table_path)
    sites.loc[0, "family"] = "expert"
    sites.loc[0, "role"] = "action_generation"
    sites.loc[0, "segment"] = "action_head"
    sites.loc[0, "materialization"] = "summary"
    sites.loc[0, "exactness"] = "approximate"
    sites.loc[0, "token_space_id"] = "action_tokens"
    sites.loc[0, "query_token_space_id"] = "action_queries"
    sites.loc[0, "key_token_space_id"] = "vlm_keys"
    sites.loc[0, "parent_site_id"] = "expert.parent"
    sites.loc[0, "summary_type"] = "mean"
    sites.to_parquet(table_path, index=False)
    manifest = workbench_manifest(TraceDataset.open(dataset.root))
    site = next(item for item in manifest["model_sites"] if item["family"] == "expert")

    assert site["site_type"] == "action_generation"
    assert site["segment"] == "action_head"
    assert site["materialization"] == "summary"
    assert site["exactness"] == "approximate"
    assert site["token_space_id"] == "action_tokens"
    assert site["refs"]["query_token_space_id"] == "action_queries"
    assert site["refs"]["key_token_space_id"] == "vlm_keys"
    assert site["refs"]["parent_site_id"] == "expert.parent"
    assert site["summary"]["row_count"] >= 1


def test_validation_rejects_token_space_reference_errors(tmp_path):
    bundle = _make_minimal_trace(
        tmp_path / "bad_tokens.vlatrace",
        tokens={
            "token_space_id": ["missing_space"],
            "token_index": [0],
            "token_kind": ["action"],
        },
        token_spaces={
            "token_space_id": ["known_space"],
            "stream_id": ["action"],
            "token_count": [1],
        },
        streams={"stream_id": ["action"], "name": ["action"], "modality": ["action"]},
    )

    result = validate_trace_bundle(bundle)

    assert not result.valid
    assert any(error["code"] == "invalid_reference" for error in result.errors)


def test_validation_requires_exact_raw_full_sites(tmp_path):
    model_sites = [
        TraceModelSiteSpec(
            name=f"pi05.full.{role}",
            array=np.zeros((1,), dtype=np.float32),
            axes=["scalar"],
            module=f"pi05/{role}",
            tensor_type=role,
            role=role,
            materialization="raw",
            exactness="exact",
        )
        for role in FULL_REQUIRED_MODEL_SITE_ROLES
    ]
    role = FULL_REQUIRED_MODEL_SITE_ROLES[0]
    model_sites[0] = TraceModelSiteSpec(
        name=f"pi05.full.{role}",
        array=np.zeros((1,), dtype=np.float32),
        axes=["scalar"],
        module=f"pi05/{role}",
        tensor_type=role,
        role=role,
        materialization="summary",
        exactness="lossy_summary",
    )
    bundle = _make_minimal_trace(
        tmp_path / "partial_full.vlatrace",
        profile="full",
        model_sites=model_sites,
    )

    result = validate_trace_bundle(bundle)

    assert not result.valid
    assert any(error["code"] == "profile_full_missing_raw_sites" for error in result.errors)


def test_validation_accepts_complete_exact_raw_full_sites(tmp_path):
    model_sites = [
        TraceModelSiteSpec(
            name=f"pi05.full.{role}",
            array=np.zeros((1,), dtype=np.float32),
            axes=["scalar"],
            module=f"pi05/{role}",
            tensor_type=role,
            role=role,
            materialization="raw",
            exactness="exact",
        )
        for role in FULL_REQUIRED_MODEL_SITE_ROLES
    ]
    bundle = _make_minimal_trace(
        tmp_path / "complete_full.vlatrace",
        profile="full",
        model_sites=model_sites,
    )

    result = validate_trace_bundle(bundle)

    assert result.valid


def test_validation_accepts_audit_windowed_profile(tmp_path):
    bundle = _make_minimal_trace(
        tmp_path / "audit_windowed_validation.vlatrace",
        profile="audit_windowed",
        model_sites=[
            TraceModelSiteSpec(
                name="pi05.vlm.layers.0.prefix.hidden_tokens",
                array=np.zeros((1, 2, 3), dtype=np.float32),
                axes=["policy_call", "token", "channel"],
                module="pi05.vlm.layers.0",
                layer=0,
                tensor_type="hidden_tokens",
            ),
            TraceModelSiteSpec(
                name="pi05.vlm.layers.0.prefix.attention",
                array=np.zeros((1, 1, 2, 2), dtype=np.float32),
                axes=["policy_call", "head", "query_token", "key_token"],
                module="pi05.vlm.layers.0.attention",
                layer=0,
                tensor_type="attention",
            ),
        ],
    )

    result = validate_trace_bundle(bundle)

    assert result.valid
    assert not any(warning["code"] == "unknown_capture_profile" for warning in result.warnings)


def test_activation_sites_payload_includes_runtime_kv_collection(tmp_path):
    bundle = _make_minimal_trace(
        tmp_path / "kv_collection.vlatrace",
        model_sites=[
            TraceModelSiteSpec(
                name="pi05.vlm.layers.0.kv_cache.key",
                array=np.zeros((1, 1, 2, 3), dtype=np.float32),
                axes=["policy_call", "kv_head", "cached_token", "head_channel"],
                module="pi05.vlm.layers.0.attention",
                layer=0,
                tensor_type="kv_cache",
                token_kind="prefix",
                family="cache",
                role="kv_cache_key",
                token_space_id="pi05.prefix",
            ),
            TraceModelSiteSpec(
                name="pi05.vlm.layers.0.kv_cache.value",
                array=np.zeros((1, 1, 2, 3), dtype=np.float32),
                axes=["policy_call", "kv_head", "cached_token", "head_channel"],
                module="pi05.vlm.layers.0.attention",
                layer=0,
                tensor_type="kv_cache",
                token_kind="prefix",
                family="cache",
                role="kv_cache_value",
                token_space_id="pi05.prefix",
            ),
        ],
    )

    payload = _activation_sites_payload(bundle)

    assert payload["runtime_collections"][0]["id"] == "pi05.vlm.past_key_values"
    assert payload["runtime_collections"][0]["materialized"] is False
    assert payload["runtime_collections"][0]["aggregation"] == "none"
    assert {member["site_name"] for member in payload["runtime_collections"][0]["members"]} == {
        "pi05.vlm.layers.0.kv_cache.key",
        "pi05.vlm.layers.0.kv_cache.value",
    }


def test_activation_sites_payload_includes_per_layer_kv_architecture_edges(tmp_path):
    model_sites = []
    for layer in (0, 4):
        model_sites.extend(
            [
                TraceModelSiteSpec(
                    name=f"pi05.vlm.layers.{layer}.kv_cache.key",
                    array=np.zeros((1, 1, 2, 3), dtype=np.float32),
                    axes=["policy_call", "kv_head", "cached_token", "head_channel"],
                    module=f"pi05.vlm.layers.{layer}.attention",
                    layer=layer,
                    tensor_type="kv_cache",
                    token_kind="prefix",
                    family="cache",
                    role="kv_cache_key",
                    token_space_id="pi05.prefix",
                ),
                TraceModelSiteSpec(
                    name=f"pi05.vlm.layers.{layer}.kv_cache.value",
                    array=np.zeros((1, 1, 2, 3), dtype=np.float32),
                    axes=["policy_call", "kv_head", "cached_token", "head_channel"],
                    module=f"pi05.vlm.layers.{layer}.attention",
                    layer=layer,
                    tensor_type="kv_cache",
                    token_kind="prefix",
                    family="cache",
                    role="kv_cache_value",
                    token_space_id="pi05.prefix",
                ),
                TraceModelSiteSpec(
                    name=f"pi05.expert.layers.{layer}.by_step.attention",
                    array=np.zeros((1, 1, 1, 2, 4), dtype=np.float32),
                    axes=["policy_call", "generation_step", "head", "query_token", "key_token"],
                    module=f"pi05.expert.layers.{layer}.attention",
                    layer=layer,
                    tensor_type="attention",
                    token_kind="action",
                    family="attention",
                    role="attention_probs",
                    segment="action_expert",
                    query_token_space_id="pi05.action_suffix",
                    key_token_space_id="pi05.expert_context",
                ),
            ]
        )
    bundle = _make_minimal_trace(tmp_path / "kv_architecture.vlatrace", model_sites=model_sites)

    payload = _activation_sites_payload(bundle)

    edges = payload["architecture"]["edges"]
    assert [edge["id"] for edge in edges] == [
        "pi05.vlm.layers.0.kv_to_expert.layers.0",
        "pi05.vlm.layers.4.kv_to_expert.layers.4",
    ]
    assert [edge["layer"] for edge in edges] == [0, 4]
    assert all(edge["kind"] == "per_layer_kv_conditioning" for edge in edges)
    assert all(edge["source_token_space"] == "pi05.prefix" for edge in edges)
    assert all(edge["query_token_space"] == "pi05.action_suffix" for edge in edges)
    assert all(edge["key_token_space"] == "pi05.expert_context" for edge in edges)
    assert all(edge["materialized"] is False for edge in edges)
    assert edges[0]["source_sites"] == [
        "pi05.vlm.layers.0.kv_cache.key",
        "pi05.vlm.layers.0.kv_cache.value",
    ]
    assert any(node["id"] == "pi05.vlm.layers.4" for node in payload["architecture"]["nodes"])
    assert any(node["id"] == "pi05.expert.layers.4" for node in payload["architecture"]["nodes"])


def test_activation_sites_payload_includes_audit_windowed_kv_edges(tmp_path):
    model_sites = []
    for layer in AUDIT_WINDOWED_LAYERS:
        model_sites.extend(
            [
                TraceModelSiteSpec(
                    name=f"pi05.vlm.layers.{layer}.kv_cache.key",
                    array=np.zeros((1, 1, 2, 3), dtype=np.float32),
                    axes=["policy_call", "kv_head", "cached_token", "head_channel"],
                    module=f"pi05.vlm.layers.{layer}.attention",
                    layer=layer,
                    tensor_type="kv_cache",
                    token_kind="prefix",
                    family="cache",
                    role="kv_cache_key",
                    token_space_id="pi05.prefix",
                ),
                TraceModelSiteSpec(
                    name=f"pi05.vlm.layers.{layer}.kv_cache.value",
                    array=np.zeros((1, 1, 2, 3), dtype=np.float32),
                    axes=["policy_call", "kv_head", "cached_token", "head_channel"],
                    module=f"pi05.vlm.layers.{layer}.attention",
                    layer=layer,
                    tensor_type="kv_cache",
                    token_kind="prefix",
                    family="cache",
                    role="kv_cache_value",
                    token_space_id="pi05.prefix",
                ),
                TraceModelSiteSpec(
                    name=f"pi05.expert.layers.{layer}.by_step.attention",
                    array=np.zeros((1, 1, 1, 2, 4), dtype=np.float32),
                    axes=["policy_call", "generation_step", "head", "query_token", "key_token"],
                    module=f"pi05.expert.layers.{layer}.attention",
                    layer=layer,
                    tensor_type="attention",
                    token_kind="action",
                    family="attention",
                    role="attention_probs",
                    segment="action_expert",
                    query_token_space_id="pi05.action_suffix",
                    key_token_space_id="pi05.expert_context",
                ),
            ]
        )
    bundle = _make_minimal_trace(
        tmp_path / "audit_windowed_kv_architecture.vlatrace",
        model_sites=model_sites,
    )

    payload = _activation_sites_payload(bundle)

    members = payload["runtime_collections"][0]["members"]
    edges = payload["architecture"]["edges"]
    assert len(members) == 20
    assert {int(member["layer"]) for member in members} == set(AUDIT_WINDOWED_LAYERS)
    assert [edge["layer"] for edge in edges] == list(AUDIT_WINDOWED_LAYERS)
    assert len(edges) == 10
    assert all(edge["kind"] == "per_layer_kv_conditioning" for edge in edges)
    assert all(edge["source"].endswith(str(edge["layer"])) for edge in edges)
    assert all(edge["target"].endswith(str(edge["layer"])) for edge in edges)


def test_activation_sites_payload_keeps_empty_architecture_for_non_pi05_sites(tmp_path):
    bundle = _make_minimal_trace(
        tmp_path / "generic_activation.vlatrace",
        model_sites=[
            TraceModelSiteSpec(
                name="toy.layers.0.hidden",
                array=np.zeros((1, 2), dtype=np.float32),
                axes=["token", "channel"],
                module="toy.layers.0",
                layer=0,
                tensor_type="hidden_tokens",
            )
        ],
    )

    payload = _activation_sites_payload(bundle)

    assert payload["architecture"] == {}


def test_expert_token_details_project_attention_to_image_and_prompt_tokens(tmp_path):
    hidden = np.array([[[[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]]]], dtype=np.float32)
    attention = np.array(
        [
            [
                [
                    [
                        [0.05, 0.10, 0.20, 0.25, 0.30, 0.05, 0.05],
                        [0.10, 0.20, 0.40, 0.10, 0.15, 0.03, 0.02],
                    ]
                ]
            ]
        ],
        dtype=np.float32,
    )
    bundle = _make_minimal_trace(
        tmp_path / "attention_details.vlatrace",
        include_frames=True,
        camera_state=pd.DataFrame.from_records(
            [{"camera_id": "main", "name": "main", "width": 16, "height": 16}]
        ),
        streams={
            "stream_id": ["prefix", "image_main", "language", "action_suffix"],
            "name": ["prefix", "main", "language", "action_suffix"],
            "modality": ["multimodal", "image", "language", "action"],
        },
        token_spaces={
            "token_space_id": ["pi05.prefix", "pi05.action_suffix"],
            "stream_id": ["prefix", "action_suffix"],
            "token_count": [5, 2],
        },
        tokens={
            "token_space_id": ["pi05.prefix"] * 5 + ["pi05.action_suffix"] * 2,
            "token_index": [0, 1, 2, 3, 4, 0, 1],
            "token_kind": ["image", "image", "image", "image", "language", "action", "action"],
            "token_type": ["image_patch"] * 4 + ["text", "continuous_action", "continuous_action"],
            "camera_id": ["main", "main", "main", "main", None, None, None],
            "patch_row": [0, 0, 1, 1, None, None, None],
            "patch_col": [0, 1, 0, 1, None, None, None],
            "token_id": [None, None, None, None, 42, None, None],
            "token_piece": [None, None, None, None, "cube", None, None],
            "attention_mask": [None, None, None, None, True, None, None],
            "policy_call_index": [0] * 7,
        },
        model_sites=[
            TraceModelSiteSpec(
                name="pi05.vlm.prefix.image_hidden_tokens",
                array=np.zeros((1, 4, 3), dtype=np.float32),
                axes=["policy_call", "token", "channel"],
                module="pi05.vlm.prefix",
                tensor_type="hidden_tokens",
                token_kind="image",
                family="representation",
                role="hidden_state",
                token_space_id="pi05.prefix",
                metadata={"patches_per_image": 4, "image_slots": 1, "grid_size": 2},
            ),
            TraceModelSiteSpec(
                name="pi05.expert.layers.0.by_step.hidden_tokens",
                array=hidden,
                axes=["policy_call", "generation_step", "token", "channel"],
                module="pi05.expert.layers.0",
                layer=0,
                tensor_type="hidden_tokens",
                token_kind="action",
                family="representation",
                role="hidden_state",
                token_space_id="pi05.action_suffix",
            ),
            TraceModelSiteSpec(
                name="pi05.expert.layers.0.attention.attention_probs",
                array=attention,
                axes=["policy_call", "generation_step", "head", "query_token", "key_token"],
                module="pi05.expert.layers.0",
                layer=0,
                tensor_type="attention_probs",
                family="attention",
                role="attention_probs",
                query_token_space_id="pi05.action_suffix",
                key_token_space_id="pi05.prefix",
            ),
        ],
    )

    details = _expert_token_details_payload(
        bundle,
        {
            "name": ["pi05.expert.layers.0.by_step.hidden_tokens"],
            "call_index": ["0"],
            "generation_step": ["0"],
            "token_index": ["1"],
            "feature": ["2"],
        },
    )
    prompt = _prompt_attention_payload(bundle, {"call_index": ["0"], "generation_step": ["0"]})

    assert details["available"] is True
    assert details["attention_site"] == "pi05.expert.layers.0.attention.attention_probs"
    assert np.isclose(details["attention_coarse"]["image"], 0.8)
    assert np.isclose(details["attention_coarse"]["prompt"], 0.15)
    assert np.isclose(details["attention_coarse"]["action_suffix"], 0.05)
    assert np.allclose(details["maps"]["main"]["values"], [[0.1, 0.2], [0.4, 0.1]])
    assert details["top_image_patches"][0]["camera"] == "main"
    assert details["top_image_patches"][0]["row"] == 1
    assert details["top_image_patches"][0]["col"] == 0
    assert details["top_image_patches"][0]["token_index"] == 2
    assert np.isclose(details["top_image_patches"][0]["attention"], 0.4)
    assert details["top_prompt_tokens"][0]["token_piece"] == "cube"
    assert np.isclose(details["top_prompt_tokens"][0]["attention"], 0.15)
    assert prompt["available"] is True
    assert np.isclose(prompt["expert_coarse"]["prompt"], 0.225)
    selected_map = _attention_map_payload(
        bundle,
        {
            "kind": ["expert"],
            "call_index": ["0"],
            "generation_step": ["0"],
            "head": ["0"],
            "query_token": ["1"],
        },
    )
    averaged_map = _attention_map_payload(
        bundle,
        {"kind": ["expert"], "call_index": ["0"], "generation_step": ["0"]},
    )
    assert selected_map["available"] is True
    assert selected_map["head_mode"] == "selected"
    assert selected_map["query_mode"] == "selected"
    assert selected_map["head"] == 0
    assert selected_map["query_token"] == 1
    assert np.allclose(selected_map["maps"]["main"]["values"], [[0.1, 0.2], [0.4, 0.1]])
    assert averaged_map["query_mode"] == "average"
    assert np.allclose(averaged_map["maps"]["main"]["values"], [[0.075, 0.15], [0.3, 0.175]])


def test_lens_array_dims_exist_in_axis_registry(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=8)
    manifest = workbench_manifest(dataset)
    axes = set(manifest["axes"])

    for array in manifest["lens_arrays"]:
        assert set(array["dims"]).issubset(axes)


def test_lens_array_slice_respects_max_payload(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=8)
    tensor = next(
        array
        for array in _lens_arrays_payload(dataset)["lens_arrays"]
        if array["kind"] == "tensor" and np.prod(array["shape"]) > 16
    )

    sliced = _lens_array_slice_payload(
        dataset,
        tensor["array_id"],
        {"selection": {}, "max_values": 4},
    )

    assert sliced["truncated"] is True
    assert "preview" in sliced
    assert np.asarray(sliced["preview"]).size <= 4


def test_selection_state_aliases_normalize_to_canonical_axes(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=8)

    normalized = normalize_axis_values(
        {"step": [2], "time": [2], "frame_idx": [2], "patch": [{"row": 1, "col": 2}]}
    )
    resolved = resolve_selection(
        dataset,
        {
            "selection_id": "alias_cell",
            "axis_values": {"step": [2], "token_kind": ["action"], "layer": [1]},
        },
    )

    assert "timestep" in normalized
    assert "step" not in normalized
    assert "image_patch" in normalized
    assert resolved["selection"]["axis_values"]["timestep"] == [2]


def test_metadata_table_query_projects_and_filters(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=3, timesteps=8)
    expected_trace = dataset.bundles[0].manifest.trace_id

    queried = query_table(
        dataset,
        table="episodes",
        filters={"trace_id": [expected_trace]},
        columns=["trace_id", "target_object"],
        limit=10,
    )
    api_queried = _table_query_payload(
        dataset,
        {
            "table": "timesteps",
            "filters": {"trace_id": [expected_trace], "timestep": {"start": 0, "end": 2}},
            "columns": ["trace_id", "timestep"],
            "limit": 5,
        },
    )
    token_queried = query_table(
        dataset,
        table="tokens",
        filters={"token_kind": ["action"]},
        columns=["trace_id", "token_kind"],
        limit=10,
    )

    assert queried["total"] == 1
    assert queried["columns"] == ["trace_id", "target_object"]
    assert queried["rows"][0]["trace_id"] == expected_trace
    assert api_queried["total"] == 3
    assert api_queried["returned"] == 3
    assert token_queried["total"] >= 1
    assert all(row["token_kind"] == "action" for row in token_queried["rows"])


def test_selection_resolution_links_selection_to_views(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=3, timesteps=12)
    selection = SelectionState(
        selection_id="sel_test",
        axis_values={"layer": [1], "token_kind": ["action"], "timestep": {"start": 2, "end": 2}},
        source_panel_id="probe_heatmap",
    )

    resolved = resolve_selection(dataset, selection, request=["representative_examples"])
    api_resolved = _resolve_selection_payload(
        dataset,
        {
            "selection": selection.to_dict(),
            "request": ["representative_examples"],
        },
    )

    assert resolved["selection"]["axis_values"]["layer"] == [1]
    assert resolved["episodes"]
    assert resolved["lens_arrays"]
    assert resolved["model_sites"]
    assert resolved["suggested_panels"]
    assert resolved["examples"]["matching"]
    assert api_resolved["examples"]["matching"]


def test_workbench_persists_cohorts_and_workspaces(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=3, timesteps=12)
    selection = SelectionState(
        selection_id="sel_action_layer",
        axis_values={"layer": [1], "token_kind": ["action"], "timestep": {"start": 2, "end": 2}},
        source_panel_id="probe_heatmap",
    )

    cohort = save_cohort(
        dataset,
        cohort_from_selection(dataset, selection, label="Action layer cohort"),
    )
    run = save_analysis_run(
        dataset,
        AnalysisRunSpec(
            run_id="target_object_probe_test",
            workflow="target_object_encoding",
            inputs={"selection": selection.to_dict()},
            outputs=("probe.metric_cube",),
        ),
    )
    workspace = save_workspace(
        dataset,
        SavedWorkspace(
            workspace_id="ws_action_layer",
            dataset_id="demo",
            panels=({"panel_type": "heatmap", "array_id": "probe.metric_cube"},),
            selection=selection,
            cohorts=(cohort.cohort_id,),
            analysis_runs=(run.run_id,),
        ),
    )
    manifest = workbench_manifest(dataset)

    assert list_cohorts(dataset)[0].cohort_id == cohort.cohort_id
    assert list_analysis_runs(dataset)[0].run_id == run.run_id
    assert list_workspaces(dataset)[0].workspace_id == workspace.workspace_id
    assert manifest["cohorts"][0]["members"]["trace_id"]
    assert manifest["analysis_runs"][0]["workflow"] == "target_object_encoding"
    assert manifest["saved_workspaces"][0]["cohorts"] == [cohort.cohort_id]


def test_workbench_persistence_api_payloads(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=8)
    selection = SelectionState(
        selection_id="sel_api",
        axis_values={"episode": [dataset.bundles[0].manifest.trace_id]},
        source_panel_id="examples.table",
    )

    cohort_payload = _save_cohort_from_selection_payload(
        dataset,
        {
            "selection": selection.to_dict(),
            "label": "API cohort",
        },
    )
    workspace_payload = _save_workspace_payload(
        dataset,
        {
            "workspace": {
                "workspace_id": "api_workspace",
                "dataset_id": "demo",
                "panels": [{"panel_type": "episode.viewer"}],
                "selection": selection.to_dict(),
                "cohorts": [cohort_payload["cohort"]["cohort_id"]],
            }
        },
    )
    run_payload = _save_analysis_run_payload(
        dataset,
        {
            "analysis_run": {
                "run_id": "api_run",
                "workflow": "unit_explorer",
                "inputs": {"unit": 0},
                "outputs": ["top_examples"],
            }
        },
    )

    assert cohort_payload["total"] == 1
    assert cohort_payload["cohort"]["members"]["trace_id"] == [dataset.bundles[0].manifest.trace_id]
    assert workspace_payload["total"] == 1
    assert workspace_payload["workspace"]["cohorts"] == [cohort_payload["cohort"]["cohort_id"]]
    assert run_payload["total"] == 1
    assert run_payload["analysis_run"]["workflow"] == "unit_explorer"


def test_activation_query_materializes_and_caches(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=8)
    query = ActivationQuery(
        module="backbone.layers.*.resid",
        layers=[0, 1],
        token_kind="image_patch",
        timesteps=[0, 1, 2],
        reduce_tokens="mean",
    )

    X, rows = dataset.select_model_sites(query).to_matrix(cache=True)
    X_cached, rows_cached = dataset.select_model_sites(query).to_matrix(cache=True)

    assert X.shape == (2 * 2 * 3, 12)
    assert rows.shape[0] == X.shape[0]
    assert np.allclose(X, X_cached)
    assert rows_cached.equals(rows)
    assert (dataset.cache_dir() / "features").exists()


def test_activation_selector_slices_generation_step_axis(tmp_path):
    root = tmp_path / "generation_selector"
    array = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    TraceBundle.create(
        root / "trace.vlatrace",
        manifest=TraceManifest(
            trace_id="trace",
            episode_id="trace",
            task_id="0",
            prompt="test",
            model_id="synthetic",
            env_id="unit",
            robot_id="none",
            outcome="success",
            length=2,
        ),
        timesteps=pd.DataFrame(
            {"timestep": [0, 1], "policy_call_index": [0, 1], "horizon_index": [0, 0]}
        ),
        policy_calls=pd.DataFrame(
            {
                "policy_call_index": [0, 1],
                "episode_id": ["trace", "trace"],
                "observation_timestep": [0, 1],
            }
        ),
        generation_steps=pd.DataFrame(
            {
                "policy_call_index": [0, 0, 0, 1, 1, 1],
                "generation_step": [0, 1, 2, 0, 1, 2],
            }
        ),
        streams=pd.DataFrame(
            {"stream_id": ["action"], "name": ["action"], "modality": ["action"]}
        ),
        token_spaces=pd.DataFrame(
            {"token_space_id": ["action"], "stream_id": ["action"], "token_count": [4]}
        ),
        tokens=pd.DataFrame(
            {
                "token_space_id": ["action"] * 4,
                "token_index": [0, 1, 2, 3],
                "token_kind": ["action"] * 4,
            }
        ),
        model_arrays=[
            ActivationSpec(
                name="pi05.expert.layers.0.hidden",
                array=array,
                axes=["policy_call", "generation_step", "token", "channel"],
                module="pi05.expert.layers.0",
                layer=0,
                tensor_type="hidden_tokens",
                token_kind="action",
                token_space_id="action",
            )
        ],
    )
    dataset = TraceDataset.open(root)

    query = ActivationQuery(
        module="pi05.expert.layers.*",
        tensor_type="hidden_tokens",
        token_kind="action",
        policy_calls=[0, 1],
        generation_step="final",
        reduce_tokens="mean",
    )
    X, rows = dataset.select_model_sites(query).to_matrix(cache=False)

    assert X.shape == (2, 5)
    assert rows["generation_step"].tolist() == ["final", "final"]
    assert np.allclose(X, array[:, -1, :, :].mean(axis=1))


def test_trace_dataset_can_open_single_bundle(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=6)
    bundle_path = dataset.bundles[0].path

    single = TraceDataset.open(bundle_path)

    assert len(single.bundles) == 1
    assert single.bundles[0].manifest.trace_id == dataset.bundles[0].manifest.trace_id


def test_probe_workflow_saves_dataset_artifact_from_yaml_spec(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=5, timesteps=8)
    spec = dump_probe_spec(
        {
            "name": "Outcome smoke probe",
            "target": {"kind": "outcome"},
            "features": {
                "module": "action_head.layers.*.resid",
                "tensor_type": "resid",
                "token_kind": "action",
                "reduction": "mean",
            },
            "split": {"kind": "random_episode"},
            "baseline": ["majority_class", "benchmark"],
            "sweep": "layer",
        }
    )
    import yaml

    saved = train_probe_artifact_from_spec(dataset, yaml.safe_load(spec))

    assert saved.artifact.artifact_type == "probe_suite"
    assert not saved.results.empty
    assert saved.artifact.display["split_summary"]
    assert saved.artifact.display["interpretation_notes"]
    assert saved.artifact.display["best_result_details"]
    assert saved.artifact.display["data_quality"]
    method = saved.artifact.method
    assert method["probe_artifact_schema_version"] == 3
    for key in [
        "lineage",
        "source",
        "input",
        "target",
        "examples",
        "split",
        "normalization",
        "probe",
        "evaluation",
        "prediction_retention",
        "outputs",
    ]:
        assert key in method
    assert method["split"]["group_key"] == "trace_id"
    assert method["prediction_retention"]["mode"] == "row_level_eval"
    assert method["source"]["source_episodes"]
    assert method["source"]["source_episodes"][0]["trace_fingerprint"].startswith("sha256:")
    assert method["source"]["source_collection_fingerprint"].startswith("sha256:")
    assert method["input"]["feature_matrix_fingerprint"].startswith("sha256:")
    assert method["target"]["target_fingerprint"].startswith("sha256:")
    assert method["examples"]["row_index_fingerprint"].startswith("sha256:")
    artifact_dir = dataset.root / "artifacts" / saved.artifact.artifact_id
    assert (artifact_dir / "metrics.json").exists()
    predictions_path = artifact_dir / "predictions.parquet"
    assert predictions_path.exists()
    predictions = pd.read_parquet(predictions_path)
    assert {
        "example_id",
        "split",
        "trace_id",
        "policy_call_index",
        "model_site_id",
        "target_name",
        "prediction_value",
        "model",
        "eval_split",
        "primary_metric",
    }.issubset(predictions.columns)
    assert len(predictions) == saved.artifact.metrics["prediction_row_count"]
    assert "weights" in saved.artifact.arrays
    assert "feature_mean" in saved.artifact.arrays
    for output_path in method["outputs"].values():
        assert (dataset.root / output_path).exists()

    reopened = TraceDataset.open(dataset.root)
    artifact = reopened.load_artifact(saved.artifact.artifact_id)
    assert artifact.name == "Outcome smoke probe"
    assert artifact.scope == "dataset"
    assert saved.artifact.artifact_id in set(reopened.artifact_index["artifact_id"])


def test_probe_workflow_uses_probe_split_sidecar(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=6, timesteps=8)
    trace_ids = [bundle.manifest.trace_id for bundle in dataset.bundles]
    pd.DataFrame(
        {
            "dataset_id": ["demo"] * len(trace_ids),
            "trace_id": trace_ids,
            "benchmark": [
                "libero_object",
                "libero_object",
                "libero_goal",
                "libero_goal",
                "libero_10",
                "libero_10",
            ],
            "task_id": [0, 0, 1, 1, 2, 2],
            "seed": [1000, 1001, 1002, 1003, 1004, 1005],
            "split": [
                "train",
                "train",
                "train",
                "train",
                "val_heldout_task",
                "test_heldout_task",
            ],
            "capture_profile": ["mechanistic_sampled"] * len(trace_ids),
        }
    ).to_csv(dataset.root / "probe_splits.csv", index=False)
    save_pi05_interaction_metrics_artifact(dataset)
    spec = dump_probe_spec(
        {
            "name": "Sidecar split probe",
            "target": {"kind": "task_id"},
            "features": {
                "module": "action_head.layers.*.resid",
                "tensor_type": "resid",
                "token_kind": "action",
                "reduction": "mean",
            },
            "split": {
                "kind": "random_episode",
                "column": "split_sidecar_split",
                "selection_value": "val_heldout_task",
                "test_value": "test_heldout_task",
                "eval_values": ["val_heldout_task", "test_heldout_task"],
            },
            "baseline": ["benchmark"],
            "probe": {"models": ["linear", "mlp"]},
            "sweep": "layer",
        }
    )
    import yaml

    saved = train_probe_artifact_from_spec(dataset, yaml.safe_load(spec))

    assert set(saved.rows["split_sidecar_split"]) == {
        "train",
        "val_heldout_task",
        "test_heldout_task",
    }
    assert saved.artifact.display["split_summary"]["episodes"]["test_heldout_task"] > 0
    assert saved.artifact.method["split"]["test_heldout_task_traces"]
    assert saved.artifact.method["split"]["selection_value"] == "val_heldout_task"
    assert saved.artifact.method["probe"]["models"] == ["linear", "mlp"]
    assert set(saved.results["split_value"]) == {"val_heldout_task", "test_heldout_task"}
    assert set(saved.results["model"]) == {"linear", "mlp"}
    assert "benchmark" in saved.artifact.method["metadata_baseline_columns"]
    assert "benchmark" in saved.artifact.display["source_columns"]
    assert "first_moved_object" in saved.artifact.display["source_columns"]


def test_pi05_interaction_metrics_derives_object_labels(tmp_path):
    root = tmp_path / "interaction"
    bundle_path = root / "red_cube_trace.vlatrace"
    timesteps = 8
    positions = np.zeros((timesteps, 2, 3), dtype=np.float32)
    positions[:, 0] = np.array([0.0, 0.0, 0.02], dtype=np.float32)
    positions[:, 1] = np.array([0.2, 0.0, 0.02], dtype=np.float32)
    positions[2:, 0, 0] += 0.05
    positions[3:, 0, 2] += 0.06
    eef = np.repeat(np.array([[0.05, 0.0, 0.08]], dtype=np.float32), timesteps, axis=0)
    manifest = TraceManifest(
        trace_id="red_cube_trace",
        episode_id="red_cube_trace",
        task_id="0",
        prompt="pick up the red cube",
        model_id="pi05",
        env_id="libero_object",
        robot_id="panda",
        outcome="success",
        length=timesteps,
        metadata={"task_name": "LIVING_ROOM_SCENE1_pick_up_the_red_cube", "seed": 1},
    )
    TraceBundle.create(
        bundle_path,
        manifest=manifest,
        timesteps=pd.DataFrame({"timestep": np.arange(timesteps)}),
        policy_calls=pd.DataFrame(
            {
                "policy_call_index": [0],
                "episode_id": ["red_cube_trace"],
                "observation_timestep": [0],
            }
        ),
        generation_steps=pd.DataFrame({"policy_call_index": [0], "generation_step": [0]}),
        streams=pd.DataFrame({"stream_id": ["action"], "name": ["action"], "modality": ["action"]}),
        token_spaces=pd.DataFrame(
            {"token_space_id": ["action"], "stream_id": ["action"], "token_count": [1]}
        ),
        tokens=pd.DataFrame({"token_space_id": ["action"], "token_index": [0]}),
        scene_state=pd.DataFrame(
            {
                "object_index": [0, 1],
                "object_name": ["red_cube_1", "blue_cube_1"],
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
                np.zeros((1, timesteps, 1), dtype=np.float32),
                ["policy_call", "horizon", "action_dim"],
            ),
            "generation_actions": ArraySpec(
                np.zeros((1, 1, timesteps, 1), dtype=np.float32),
                ["policy_call", "generation_step", "horizon", "action_dim"],
            ),
        },
    )
    dataset = TraceDataset.open(root)

    saved = save_pi05_interaction_metrics_artifact(dataset)
    row = saved.episode_labels.iloc[0]

    assert saved.artifact.artifact_type == "pi05_interaction_metrics"
    assert row["primary_target_object"] == "red_cube_1"
    assert row["first_moved_object"] == "red_cube_1"
    assert row["first_lifted_object"] == "red_cube_1"
    assert bool(row["first_moved_is_target"])
    assert (dataset.root / saved.artifact.method["outputs"]["episode_labels"]).exists()
    assert list_analysis_runs(dataset)[0].workflow == "pi05_interaction_metrics"
    red_object = saved.object_metrics.loc[
        saved.object_metrics["object_name"] == "red_cube_1"
    ].iloc[0]
    assert np.isclose(red_object["max_xy_displacement"], 0.05)
    assert red_object["max_displacement"] > red_object["max_xy_displacement"]

    payload = _episode_interactions_payload(dataset, {"trace_id": ["red_cube_trace"]})
    assert payload["available"]
    assert payload["artifact_id"] == saved.artifact.artifact_id
    assert payload["episode"]["primary_target_object"] == "red_cube_1"
    assert payload["episode"]["first_moved_timestep"] == 2
    assert payload["episode"]["first_lifted_timestep"] == 3
    assert payload["episode"]["target_objects"] == ["red_cube_1"]
    assert payload["quality"]["target_parse_failed"] is False
    assert payload["objects"][0]["object_name"] == "red_cube_1"
    assert payload["objects"][0]["is_target_object"] is True

    stale_episode_path = dataset.root / saved.artifact.method["outputs"]["episode_labels"]
    stale_episode_path.write_text("not a parquet file", encoding="utf-8")
    stale_payload = _episode_interactions_payload(dataset, {"trace_id": ["red_cube_trace"]})
    assert stale_payload["available"] is False
    assert stale_payload["reason"] == "Interaction metrics artifact has no episode label table."


def test_probe_workflow_resolves_trace_context_targets(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=6, timesteps=8)
    selector = ActivationQuery(
        module="action_head.layers.*.resid",
        tensor_type="resid",
        token_kind="action",
        timesteps="all",
        reduce_tokens="mean",
    )

    reward_probe = train_probe_artifact_from_spec(
        dataset,
        {
            "name": "Reward context probe",
            "target": {
                "name": "reward",
                "kind": "regression",
                "source": "evaluation",
                "metric_name": "reward",
            },
            "features": selector.to_dict(),
            "split": {"kind": "random_episode"},
            "sweep": "layer",
        },
    )
    assert "reward" in reward_probe.rows
    assert reward_probe.artifact.method["target"]["source"] == "evaluation"
    assert reward_probe.artifact.method["target"]["kind"] == "regression"

    object_probe = train_probe_artifact_from_spec(
        dataset,
        {
            "name": "Target cube x probe",
            "target": {
                "name": "target_cube_x",
                "kind": "regression",
                "source": "scene_state",
                "field": "pose",
                "selector": {"object": "target_cube", "component": "x"},
            },
            "features": selector.to_dict(),
            "split": {"kind": "random_episode"},
            "sweep": "layer",
        },
    )
    assert "target_cube_x" in object_probe.rows
    assert object_probe.rows["target_cube_x"].notna().all()
    assert object_probe.artifact.method["target"]["selector"]["object"] == "target_cube"

    threshold_probe = train_probe_artifact_from_spec(
        dataset,
        {
            "name": "Reward threshold probe",
            "target": {
                "name": "reward_high",
                "kind": "classification",
                "source": "evaluation",
                "metric_name": "reward",
                "transform": {"kind": "threshold", "operator": ">", "value": 0.5},
            },
            "features": selector.to_dict(),
            "split": {"kind": "random_episode"},
            "sweep": "layer",
        },
    )
    assert set(threshold_probe.rows["reward_high"].unique()).issubset({False, True})
    assert threshold_probe.artifact.method["target"]["transform"]["kind"] == "threshold"

    action_probe = train_probe_artifact_from_spec(
        dataset,
        {
            "name": "Action chunk probe",
            "target": {
                "name": "chunk_action_0",
                "kind": "regression",
                "source": "array",
                "array_id": "action_chunks",
                "selector": {"horizon": 0, "action_dim": 0},
            },
            "features": selector.to_dict(),
            "split": {"kind": "random_episode"},
            "sweep": "layer",
        },
    )
    first_row = action_probe.rows.iloc[0]
    first_bundle = dataset.bundle(str(first_row["trace_id"]))
    expected = first_bundle.action_chunks()[
        int(first_row["policy_call_index"]),
        0,
        0,
    ]
    assert np.isclose(first_row["chunk_action_0"], expected)

    final_generation_probe = train_probe_artifact_from_spec(
        dataset,
        {
            "name": "Final generation action probe",
            "target": {
                "name": "final_generation_action",
                "kind": "regression",
                "source": "array",
                "array_id": "generation_actions",
                "alignment": {"generation_step": "final"},
                "selector": {"horizon": -1, "action_dim": -1},
            },
            "features": selector.to_dict(),
            "split": {"kind": "random_episode"},
            "sweep": "layer",
        },
    )
    first_row = final_generation_probe.rows.iloc[0]
    first_bundle = dataset.bundle(str(first_row["trace_id"]))
    expected = first_bundle.generation_actions()[
        int(first_row["policy_call_index"]),
        -1,
        -1,
        -1,
    ]
    assert np.isclose(first_row["final_generation_action"], expected)

    negative_generation_probe = train_probe_artifact_from_spec(
        dataset,
        {
            "name": "Negative generation action probe",
            "target": {
                "name": "negative_generation_action",
                "kind": "regression",
                "source": "array",
                "array_id": "generation_actions",
                "generation_step": -1,
                "selector": {"horizon": 0, "action_dim": 0},
            },
            "features": selector.to_dict(),
            "split": {"kind": "random_episode"},
            "sweep": "layer",
        },
    )
    first_row = negative_generation_probe.rows.iloc[0]
    first_bundle = dataset.bundle(str(first_row["trace_id"]))
    expected = first_bundle.generation_actions()[
        int(first_row["policy_call_index"]),
        -1,
        0,
        0,
    ]
    assert np.isclose(first_row["negative_generation_action"], expected)

    for bundle in dataset.bundles:
        final_success = pd.DataFrame.from_records(
            [
                {
                    "timestep": bundle.manifest.length - 1,
                    "metric_name": "final_only_success",
                    "metric_value": float(bundle.manifest.outcome == "success"),
                    "threshold": 1.0,
                    "passed": bundle.manifest.outcome == "success",
                    "source": "test",
                }
            ]
        )
        final_success.to_parquet(bundle.path / "tables" / "evaluation.parquet", index=False)
        bundle.__dict__.pop("evaluation", None)
    with pytest.raises(ValueError, match="could not be resolved"):
        train_probe_artifact_from_spec(
            dataset,
            {
                "name": "Sparse success probe",
                "target": {
                    "name": "final_only_success",
                    "kind": "classification",
                    "source": "evaluation",
                    "metric_name": "final_only_success",
                },
                "features": selector.to_dict(),
                "split": {"kind": "random_episode"},
                "sweep": "layer",
            },
        )


def test_artifact_api_payloads_include_probe_and_attention(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=3, timesteps=8)
    payload = _artifacts_payload(dataset)

    assert payload["total"] >= 2
    assert payload["counts"]["probe_suite"] >= 1
    assert payload["counts"]["attention_map"] >= 1

    artifact_id = payload["artifacts"][0]["artifact_id"]
    detail = _artifact_detail_payload(dataset, artifact_id)
    assert detail["artifact"]["artifact_id"] == artifact_id
    assert "selector" in detail["artifact"]


def test_episode_metrics_use_action_metadata_and_axis_labels(tmp_path):
    bundle = _make_minimal_trace(
        tmp_path / "metrics.vlatrace",
        action_normalization={
            "normalization_id": ["libero_action"],
            "mode": ["checkpoint"],
            "stats_ref": ["policy_preprocessor"],
            "action_dim_names": ['["eef_delta_x"]'],
            "metadata": [
                '{"action_labels":["EEF delta x"],"action_units":["normalized OSC command"]}'
            ],
        },
    )

    payload = _episode_metrics_payload(bundle)
    metrics = {metric["key"]: metric for metric in payload["metrics"]}

    assert metrics["action_dim_0"]["label"] == "EEF delta x"
    assert metrics["action_dim_0"]["y_label"] == "EEF delta x"
    assert metrics["action_dim_0"]["y_unit"] == "normalized OSC command"
    assert metrics["action_dim_0"]["x_label"] == "Environment timestep"
    assert metrics["generation_start"]["domain"] == "call"
    assert metrics["generation_start"]["x_values"] == [0.0]
    assert metrics["generation_start"]["x_label"] == "Policy call timestep"


def test_episode_video_is_saved_as_model_call_artifact(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=6)
    bundle = dataset.bundles[0]

    video_path = _episode_video_path(bundle, camera="main", fps=2, max_width=96)

    assert video_path.exists()
    assert video_path.relative_to(bundle.path).parts[:2] == ("artifacts", "videos")

    reopened = TraceDataset.open(bundle.path)
    videos = reopened.artifact_index.loc[
        reopened.artifact_index["artifact_type"].astype(str) == "episode_video"
    ]
    assert len(videos) == 1
    artifact = reopened.load_artifact(str(videos.iloc[0]["artifact_id"]))
    assert artifact.metrics["frame_count"] == 2
    assert artifact.display["relative_path"].endswith(".mp4")


def test_action_generation_artifact_saves_arrays(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=3, timesteps=8)

    saved = save_action_generation_artifact(dataset, name="Action generation smoke")

    assert saved.artifact.artifact_type == "action_generation"
    assert saved.commitment.ndim == 3
    assert saved.executed_error.ndim == 2
    assert saved.delta_to_final is not None and saved.delta_to_final.ndim == 4
    assert saved.step_delta is not None and saved.step_delta.ndim == 4
    assert saved.final_vs_executed is not None and saved.final_vs_executed.ndim == 4
    assert saved.artifact.metrics["episode_count"] == 3

    reopened = TraceDataset.open(dataset.root)
    artifact = reopened.load_artifact(saved.artifact.artifact_id)
    assert artifact.display["episodes"]
    assert artifact.metrics["outcome_summary"]
    assert artifact.display["episodes"][0]["unstable_calls"]
    assert reopened.load_artifact_array(artifact, "commitment").shape == saved.commitment.shape
    assert (
        reopened.load_artifact_array(artifact, "delta_to_final").shape == saved.delta_to_final.shape
    )
    assert reopened.load_artifact_array(artifact, "step_delta").shape == saved.step_delta.shape
    assert (
        reopened.load_artifact_array(artifact, "final_vs_executed").shape
        == saved.final_vs_executed.shape
    )


def test_action_stabilization_arrays_use_canonical_dims_coords_and_outputs(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=3, timesteps=8)
    saved = save_action_generation_artifact(dataset, name="Action stabilization contract")
    reopened = TraceDataset.open(dataset.root)
    manifest = workbench_manifest(reopened)
    arrays = {
        array["array_id"]: array
        for array in manifest["lens_arrays"]
        if array["array_id"].startswith(f"artifact.{saved.artifact.artifact_id}.")
    }
    run = next(
        run for run in manifest["analysis_runs"] if run["run_id"] == saved.artifact.artifact_id
    )

    assert set(run["outputs"]) >= {"delta_to_final", "step_delta", "final_vs_executed"}
    assert arrays[f"artifact.{saved.artifact.artifact_id}.delta_to_final"]["dims"] == [
        "episode",
        "policy_call",
        "generation_step",
        "action_horizon",
    ]
    assert arrays[f"artifact.{saved.artifact.artifact_id}.step_delta"]["dims"] == [
        "episode",
        "policy_call",
        "generation_step",
        "action_horizon",
    ]
    assert arrays[f"artifact.{saved.artifact.artifact_id}.final_vs_executed"]["dims"] == [
        "episode",
        "policy_call",
        "action_horizon",
        "action_dim",
    ]
    for name in ["delta_to_final", "step_delta", "final_vs_executed"]:
        spec = arrays[f"artifact.{saved.artifact.artifact_id}.{name}"]
        assert spec["storage"]["format"] == "zarr"
        assert spec["storage"]["chunks"]
        assert spec["provenance"]["analysis_run_id"] == saved.artifact.artifact_id
        assert all(dim in spec["coords"] for dim in spec["dims"])


def test_dataset_diagnostics_artifact_tracks_fingerprint(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=4, timesteps=8)

    assert diagnostics_status(dataset)["stale"] is True
    artifact = run_dataset_diagnostics(dataset)

    reopened = TraceDataset.open(dataset.root)
    status = diagnostics_status(reopened)
    assert status["stale"] is False
    assert artifact.artifact_type == "dataset_diagnostics"
    assert artifact.metrics["dataset_fingerprint"] == status["fingerprint"]
    assert artifact.display["split_feasibility"]
    assert artifact.display["recommended_artifacts"]


def test_dataset_diagnostics_api_can_run(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=4, timesteps=8)

    before = _dataset_diagnostics_payload(dataset)
    after = _run_dataset_diagnostics_payload(dataset)

    assert before["stale"] is True
    assert after["stale"] is False
    assert after["latest"]["artifact_type"] == "dataset_diagnostics"


def test_dashboard_can_create_default_artifacts(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=5, timesteps=8)

    target_object = _create_target_object_probe_payload(dataset)
    probe = _create_outcome_probe_payload(dataset)
    generation = _create_action_generation_payload(dataset)

    assert target_object["artifact"]["artifact_type"] == "target_object_encoding"
    assert target_object["artifact"]["method"]["workflow"] == "target_object_encoding"
    assert target_object["artifact"]["method"]["split"]["unit"] == "episode"
    assert target_object["arrays"]["metric_cube"]
    assert probe["artifact"]["artifact_type"] == "probe_suite"
    assert probe["artifact"]["display"]["best_result_details"]
    assert generation["artifact"]["artifact_type"] == "action_generation"
    assert generation["artifact"]["metrics"]["outcome_summary"]


def test_target_object_encoding_saves_metric_cubes_and_analysis_run(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=6, timesteps=8)

    saved = save_target_object_encoding_artifact(dataset, max_timesteps=3)

    assert saved.artifact.artifact_type == "target_object_encoding"
    assert saved.metric_cube.shape == (
        len(saved.layers),
        len(saved.timesteps),
        len(saved.token_kinds),
    )
    assert np.isfinite(saved.metric_cube).any()
    assert saved.artifact.display["confusion_matrix"]
    assert saved.artifact.display["linked_examples"]
    reopened = TraceDataset.open(dataset.root)
    artifact = reopened.load_artifact(saved.artifact.artifact_id)
    assert reopened.load_artifact_array(artifact, "metric_cube").shape == saved.metric_cube.shape
    manifest = workbench_manifest(reopened)
    run = next(
        run for run in manifest["analysis_runs"] if run["run_id"] == saved.artifact.artifact_id
    )
    assert run["workflow"] == "target_object_encoding"
    assert set(run["outputs"]) == {"metric_cube", "baseline_cube", "delta_cube"}
    assert run["provenance"]["artifact_id"] == saved.artifact.artifact_id
    assert any(
        array["array_id"].endswith(".metric_cube")
        and array["dims"] == ["layer", "timestep", "token_kind"]
        for array in manifest["lens_arrays"]
    )
    assert not list(dataset.root.glob(".vla_cache/**/*.npy"))
    assert list(dataset.root.glob(".vla_cache/**/X.zarr"))


def test_target_object_metric_cube_has_dims_layer_timestep_token_kind(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=6, timesteps=8)
    saved = save_target_object_encoding_artifact(dataset, max_timesteps=3)
    reopened = TraceDataset.open(dataset.root)

    manifest = workbench_manifest(reopened)
    arrays = {
        item["array_id"]: item
        for item in manifest["lens_arrays"]
        if item["array_id"].startswith(f"artifact.{saved.artifact.artifact_id}.")
    }
    array = next(
        item
        for item in manifest["lens_arrays"]
        if item["array_id"] == f"artifact.{saved.artifact.artifact_id}.metric_cube"
    )

    assert array["dims"] == ["layer", "timestep", "token_kind"]
    assert array["shape"] == list(saved.metric_cube.shape)
    for name in ["metric_cube", "baseline_cube", "delta_cube"]:
        spec = arrays[f"artifact.{saved.artifact.artifact_id}.{name}"]
        assert spec["dims"] == ["layer", "timestep", "token_kind"]
        assert spec["shape"] == list(saved.metric_cube.shape)
        assert spec["storage"]["format"] == "zarr"
        assert spec["coords"]["layer"] == list(saved.layers)
        assert spec["coords"]["timestep"] == list(saved.timesteps)
        assert spec["coords"]["token_kind"] == list(saved.token_kinds)


def test_lens_array_slice_accepts_semantic_axis_coordinates(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=6, timesteps=8)
    saved = save_target_object_encoding_artifact(dataset, max_timesteps=3)
    reopened = TraceDataset.open(dataset.root)
    array_id = f"artifact.{saved.artifact.artifact_id}.metric_cube"
    layer = saved.layers[-1]
    timestep = saved.timesteps[-1]
    token_kind = saved.token_kinds[-1]

    sliced = _lens_array_slice_payload(
        reopened,
        array_id,
        {
            "selection": {
                "layer": [layer],
                "timestep": [timestep],
                "token_kind": [token_kind],
            },
            "max_values": 8,
        },
    )

    expected = saved.metric_cube[
        [len(saved.layers) - 1],
        [len(saved.timesteps) - 1],
        [len(saved.token_kinds) - 1],
    ]
    assert sliced["shape"] == [1]
    assert np.asarray(sliced["values"]).shape == (1,)
    assert np.allclose(np.asarray(sliced["values"]), expected, equal_nan=True)


def test_selection_analysis_run_limits_artifact_arrays_to_selected_run(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=6, timesteps=8)
    first = save_target_object_encoding_artifact(dataset, name="first", max_timesteps=2)
    second = save_target_object_encoding_artifact(dataset, name="second", max_timesteps=2)
    reopened = TraceDataset.open(dataset.root)
    best = first.artifact.display["best_cell"]

    resolved = resolve_selection(
        reopened,
        SelectionState(
            selection_id="first_only",
            axis_values={
                "layer": [best["layer"]],
                "timestep": [best["timestep"]],
                "token_kind": [best["token_kind"]],
                "analysis_run": [first.artifact.artifact_id],
            },
        ),
    )
    array_ids = {array["array_id"] for array in resolved["lens_arrays"]}

    assert any(first.artifact.artifact_id in array_id for array_id in array_ids)
    assert not any(second.artifact.artifact_id in array_id for array_id in array_ids)


def test_heatmap_cell_selection_resolves_to_examples(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=6, timesteps=8)
    saved = save_target_object_encoding_artifact(dataset, max_timesteps=3)
    cell = saved.artifact.display["cell_details"][0]
    selection = SelectionState(
        selection_id="cell",
        axis_values={
            "layer": [cell["layer"]],
            "timestep": [cell["timestep"]],
            "token_kind": [cell["token_kind"]],
            "metric": ["score"],
            "analysis_run": [saved.artifact.artifact_id],
        },
        source_panel_id="workbench.heatmap",
    )

    resolved = resolve_selection(TraceDataset.open(dataset.root), selection)

    assert resolved["examples"]["matching"]
    assert resolved["target_object_cell"]["confusion_matrix"]


def test_target_object_selection_resolves_to_examples_arrays_panels_and_provenance(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=6, timesteps=8)
    saved = save_target_object_encoding_artifact(dataset, max_timesteps=3)
    best = saved.artifact.display["best_cell"]
    selection = {
        "selection_id": "target_object_cell",
        "source_panel_id": "workbench.heatmap",
        "axis_values": {
            "layer": [best["layer"]],
            "timestep": [best["timestep"]],
            "token_kind": [best["token_kind"]],
            "metric": ["score"],
            "analysis_run": [saved.artifact.artifact_id],
        },
    }

    resolved = _resolve_selection_payload(TraceDataset.open(dataset.root), {"selection": selection})

    assert resolved["examples"]["matching"]
    assert any(array["array_id"].endswith(".metric_cube") for array in resolved["lens_arrays"])
    assert any(panel["panel_type"] == "heatmap" for panel in resolved["suggested_panels"])
    assert resolved["provenance"]["analysis_run"] == saved.artifact.artifact_id
    assert resolved["valid_references"]["episodes"]
    assert resolved["valid_references"]["timestep"] == best["timestep"]


def test_selection_resolver_returns_compatible_panels(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=6, timesteps=8)
    saved = save_target_object_encoding_artifact(dataset, max_timesteps=2)
    cell = saved.artifact.display["cell_details"][0]

    resolved = resolve_selection(
        TraceDataset.open(dataset.root),
        {
            "selection_id": "panel_cell",
            "axis_values": {
                "layer": [cell["layer"]],
                "timestep": [cell["timestep"]],
                "token_kind": [cell["token_kind"]],
                "analysis_run": [saved.artifact.artifact_id],
            },
        },
    )
    panel_types = {panel["panel_type"] for panel in resolved["suggested_panels"]}

    assert {"heatmap", "examples.table"}.issubset(panel_types)


def test_cohort_from_selection_has_definition_and_members(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=6, timesteps=8)
    saved = save_target_object_encoding_artifact(dataset, max_timesteps=2)
    cell = saved.artifact.display["cell_details"][0]
    selection = SelectionState(
        selection_id="save_cell",
        axis_values={
            "layer": [cell["layer"]],
            "timestep": [cell["timestep"]],
            "token_kind": [cell["token_kind"]],
            "analysis_run": [saved.artifact.artifact_id],
        },
        source_panel_id="workbench.heatmap",
    )

    cohort = save_cohort(
        TraceDataset.open(dataset.root),
        cohort_from_selection(TraceDataset.open(dataset.root), selection, label="Saved cell"),
    )

    assert cohort.definition["source"] == "selection"
    assert cohort.definition["analysis_run"] == saved.artifact.artifact_id
    assert cohort.members["trace_id"]
    assert cohort.members["example_id"]


def test_saved_workspace_round_trips_panels_selection_cohorts_and_analysis_run(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=6, timesteps=8)
    saved = save_target_object_encoding_artifact(dataset, max_timesteps=2)
    selection = SelectionState(
        selection_id="workspace_cell",
        axis_values={
            "layer": [saved.artifact.display["best_cell"]["layer"]],
            "timestep": [saved.artifact.display["best_cell"]["timestep"]],
            "token_kind": [saved.artifact.display["best_cell"]["token_kind"]],
            "analysis_run": [saved.artifact.artifact_id],
        },
    )
    cohort = save_cohort(
        dataset,
        cohort_from_selection(dataset, selection, label="Workspace cohort"),
    )

    workspace_payload = _save_workspace_payload(
        dataset,
        {
            "workspace_id": "ws_target_object_cell",
            "panels": [
                {
                    "panel_type": "heatmap",
                    "array_id": f"artifact.{saved.artifact.artifact_id}.metric_cube",
                },
                {"panel_type": "examples.table"},
                {"panel_type": "episode.viewer"},
            ],
            "selection": selection.to_dict(),
            "cohorts": [cohort.cohort_id],
            "analysis_runs": [saved.artifact.artifact_id],
        },
    )
    reopened = TraceDataset.open(dataset.root)
    workspace = workbench_manifest(reopened)["saved_workspaces"][0]

    assert workspace_payload["workspace"]["selection"]["axis_values"]["layer"]
    assert workspace["panels"][0]["panel_type"] == "heatmap"
    assert workspace["cohorts"] == [cohort.cohort_id]
    assert workspace["analysis_runs"] == [saved.artifact.artifact_id]


def test_probe_artifact_is_registered_as_analysis_run_not_primary_ui_object(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=6, timesteps=8)
    saved = save_target_object_encoding_artifact(dataset, max_timesteps=2)
    manifest = workbench_manifest(TraceDataset.open(dataset.root))

    assert any(
        run["run_id"] == saved.artifact.artifact_id and run["workflow"] == "target_object_encoding"
        for run in manifest["analysis_runs"]
    )
    assert any(
        workflow["workflow_id"] == "target_object_encoding"
        for workflow in manifest["workflow_presets"]
    )


def test_non_intervention_edges_are_not_marked_causal(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=8)
    manifest = workbench_manifest(dataset)
    edge_types = {edge["edge_type"]: edge["causal"] for edge in manifest["graph_edge_types"]}
    overlay_types = {
        score["score_type"]: score["causal"] for score in manifest["overlay_score_types"]
    }

    assert edge_types["attention_weight"] is False
    assert edge_types["gradient_attribution"] is False
    assert edge_types["correlation"] is False
    assert edge_types["intervention_delta"] is True
    assert overlay_types["probe_contribution"] is False
    assert overlay_types["patch_ablation_delta"] is True


def test_panel_registry_and_contract_validation_are_canonical(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=8)

    manifest = workbench_manifest(dataset)
    validation = validate_workbench_contracts(dataset)

    assert validation["valid"] is True
    assert "inspector" in manifest["panel_registry"]
    assert "action.horizon_heatmap" in manifest["panel_registry"]
    assert "unit.profile" in manifest["panel_registry"]
    assert not validation["invalid_array_dims"]
    assert not validation["invalid_panel_axes"]
    assert not validation["invalid_workflow_panels"]
    assert not validation["invalid_storage"]
    assert not validation["invalid_tables"]
    assert not validation["invalid_media"]
    assert not validation["invalid_analysis_outputs"]


def test_contract_validation_reports_invalid_dims_and_storage(tmp_path, monkeypatch):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=8)
    valid_arrays = workbench_module.lens_array_catalog(dataset)
    invalid = LensArraySpec(
        array_id="invalid.array",
        kind="tensor",
        label="invalid",
        storage=StorageRef(format="zarr", uri="missing.zarr", relative_to="dataset"),
        dims=("not_an_axis",),
        shape=(1,),
        dtype="float32",
    )

    monkeypatch.setattr(
        workbench_module,
        "lens_array_catalog",
        lambda _dataset: (*valid_arrays, invalid),
    )
    validation = validate_workbench_contracts(dataset)

    assert validation["valid"] is False
    assert validation["invalid_array_dims"]
    assert validation["invalid_storage"]


def test_contract_validation_reports_invalid_table_storage(tmp_path, monkeypatch):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=8)
    valid_tables = table_catalog(dataset)
    invalid = TableSpec(
        table_id="invalid_table",
        label="Invalid Table",
        storage=StorageRef(format="json", uri="missing.json", relative_to="dataset"),
        columns=("trace_id",),
        row_count=1,
    )

    monkeypatch.setattr(
        workbench_module,
        "table_catalog",
        lambda _dataset: (*valid_tables, invalid),
    )
    validation = validate_workbench_contracts(dataset)

    assert validation["valid"] is False
    assert validation["invalid_tables"]
    assert validation["invalid_tables"][0]["reason"] == "table_not_parquet"


def test_contract_validation_reports_invalid_frame_media_storage(tmp_path, monkeypatch):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=8)
    valid_frames = workbench_module.image_frame_catalog(dataset)
    invalid = ImageFrameSpec(
        frame_id="invalid.frames",
        trace_id=dataset.bundles[0].manifest.trace_id,
        episode_id=dataset.bundles[0].manifest.episode_id,
        camera="main",
        storage=StorageRef(format="png", uri="missing", relative_to="bundle"),
        dims=("timestep", "height", "width", "rgb"),
        shape=(1, 1, 1, 3),
        frame_count=1,
    )

    monkeypatch.setattr(
        workbench_module,
        "image_frame_catalog",
        lambda _dataset: (*valid_frames, invalid),
    )
    validation = validate_workbench_contracts(dataset)

    assert validation["valid"] is False
    assert validation["invalid_media"]
    assert validation["invalid_media"][0]["reason"] == "image_frame_not_jpeg"


def test_saved_workspace_reload_resolves_selection(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=3, timesteps=8)
    selection = SelectionState(
        selection_id="reload",
        axis_values={"episode": [dataset.bundles[0].manifest.trace_id], "timestep": [0]},
    )
    saved = save_workspace(
        dataset,
        SavedWorkspace(
            workspace_id="reload_workspace",
            dataset_id="demo",
            panels=({"panel_type": "episode.viewer"}, {"panel_type": "examples.table"}),
            selection=selection,
        ),
    )

    resolved = resolve_workspace(dataset, saved.workspace_id)

    assert resolved["workspace"]["workspace_id"] == saved.workspace_id
    assert resolved["resolved_selection"]["examples"]["matching"]
    assert resolved["panel_registry"]["episode.viewer"]["renderer"] == "media"


def test_action_stabilization_selection_resolves_to_cell_arrays_and_panels(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=3, timesteps=8)
    saved = save_action_generation_artifact(dataset, name="Action stabilization smoke")
    reopened = TraceDataset.open(dataset.root)
    selection = SelectionState(
        selection_id="action_cell",
        axis_values={
            "episode": [0],
            "policy_call": [0],
            "generation_step": [1],
            "action_horizon": [0],
            "analysis_run": [saved.artifact.artifact_id],
        },
        source_panel_id="action.horizon_heatmap",
    )

    resolved = resolve_selection(reopened, selection)

    assert resolved["action_stabilization_cell"]["delta_to_final"] is not None
    assert resolved["examples"]["matching"]
    assert any(array["array_id"].endswith(".delta_to_final") for array in resolved["lens_arrays"])
    assert any(
        panel["panel_type"] == "action.horizon_heatmap" for panel in resolved["suggested_panels"]
    )
    assert any(run.workflow == "action_stabilization" for run in list_analysis_runs(reopened))


def test_unit_profile_returns_histograms_and_noncausal_associations(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=4, timesteps=8)
    save_target_object_encoding_artifact(dataset, max_timesteps=2)
    reopened = TraceDataset.open(dataset.root)
    site = workbench_manifest(reopened)["model_sites"][0]

    profile = unit_profile(
        reopened,
        UnitRef(kind="neuron", site_id=site["site_id"], index=0),
        selection={
            "selection_id": "unit",
            "axis_values": {"layer": [site["layer"]], "token_kind": [site["token_kind"]]},
        },
    )

    assert profile["axis_histograms"]
    assert profile["causal"] is False
    assert profile["association_kind"] == "observational_activation"
    assert all(item["causal"] is False for item in profile["probe_associations"])


def test_spatial_overlay_contracts_preserve_score_type_labels(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=8)

    contracts = spatial_overlay_contracts(dataset)

    assert contracts
    assert all("score_type" in contract for contract in contracts)
    assert all(contract["score_type"] != "importance" for contract in contracts)
    assert any(contract["score_type"] == "attention_weight" for contract in contracts)


def test_cohort_comparison_produces_delta_tables(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=4, timesteps=8)
    left = save_cohort(
        dataset,
        cohort_from_selection(
            dataset,
            SelectionState(selection_id="left", axis_values={"outcome": ["success"]}),
            label="successes",
        ),
    )
    right = save_cohort(
        dataset,
        cohort_from_selection(
            dataset,
            SelectionState(selection_id="right", axis_values={"outcome": ["failure"]}),
            label="failures",
        ),
    )

    comparison = compare_cohorts(dataset, left.cohort_id, right.cohort_id)

    assert comparison["summary"]["left_count"] >= 0
    assert "outcome" in comparison["tables"]
    assert comparison["tables"]["object"]


def test_projection_and_graph_contracts_are_selection_linked(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=3, timesteps=8)

    projection = graph_selection = {
        "selection_id": "discovery",
        "axis_values": {"token_kind": ["action"], "layer": [0]},
    }
    points = projection_points(dataset, selection=projection, limit=12)
    graph = graph_from_selection(dataset, graph_selection)

    assert points["semantic_role"] == "cohort_discovery_not_explanation"
    assert points["points"]
    assert graph["nodes"]
    assert graph["edges"]
    assert all(edge["causal"] is False for edge in graph["edges"])


def test_intervention_records_are_saved_readouts_not_live_execution(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=8)
    run = save_intervention_run(
        dataset,
        InterventionRunSpec(
            run_id="saved_intervention",
            intervention_type="intervention_delta",
            target={"site_id": "policy.layer12", "unit": 3},
            baseline={"cohort_id": "baseline"},
            intervention={"cohort_id": "patched"},
            readouts={"success_delta": 0.25},
            outputs=("intervention_delta",),
            provenance={"source": "saved_readout"},
        ),
    )
    payload = _save_intervention_run_payload(
        dataset,
        {
            "run_id": "saved_ablation",
            "intervention_type": "ablation_effect",
            "target": {"site_id": "policy.layer8"},
            "outputs": ["ablation_effect"],
        },
    )

    assert run.intervention_type == "intervention_delta"
    assert payload["intervention_run"]["intervention_type"] == "ablation_effect"
    assert any(item.run_id == "saved_intervention" for item in list_analysis_runs(dataset))
