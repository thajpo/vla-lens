# ruff: noqa: F403,F405
from tests._support.vla_lens_trace_mvp import *


def test_synthetic_dataset_indexes_and_stats(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=3, timesteps=12)

    assert len(dataset.episode_index) == 3
    assert {"trace_id", "task_id", "outcome", "path"}.issubset(dataset.episode_index.columns)
    assert not dataset.timestep_index.empty
    assert not dataset.stats.by_task().empty
    coverage = dataset.stats.activation_coverage()
    assert set(coverage["token_kind"]) >= {"image_patch", "action"}


def test_trace_bundle_writes_probe_provenance_fingerprints(tmp_path):
    bundle = _make_minimal_trace(tmp_path / "fingerprinted")

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
        tmp_path / "fingerprinted_changed",
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
        tmp_path / "pi05_mechanistic_sampled_libero_goal_task1_seed42_clean",
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
        tmp_path / "pi05_mechanistic_sampled_libero_goal_task1_seed42_corrupt",
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
    assert {frame["storage"]["format"] for frame in manifest["image_frames"]} == {"mp4"}
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
        score["score_type"] == "intervention_record" for score in manifest["overlay_score_types"]
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
    overlay = bundle.overlay_bundle
    assert overlay is not None
    robot_image_rows = bundle.array_index.loc[
        bundle.array_index["name"].astype(str).str.startswith("observation.images.")
    ]
    overlay_rows = overlay.array_index

    assert (bundle.path / "meta" / "info.json").exists()
    assert (bundle.path / "data").exists()
    assert (overlay.path / "tables" / "array_index.parquet").exists()
    assert (overlay.path / "tables" / "model_sites.parquet").exists()
    assert (overlay.path / "tables" / "policy_calls.parquet").exists()
    assert set(robot_image_rows["storage_format"]) == {"mp4"}
    assert "executed_actions" not in set(overlay_rows["name"].astype(str))
    assert not any(str(name).startswith("frames.") for name in overlay_rows["name"])
    assert set(overlay_rows["storage_format"]) == {"zarr"}
    assert set(bundle.model_sites["storage_format"]) == {"zarr"}
    assert all(str(path).endswith(".zarr") for path in overlay_rows["relative_path"])
    assert not list(bundle.path.glob("arrays/**/*.npy"))
    assert (bundle.path / str(robot_image_rows.iloc[0]["relative_path"])).exists()
    assert bundle.actions(mmap=True).shape[0] == 8
    assert bundle.frames("main", mmap=True).shape == (8, 96, 128, 3)
    assert (overlay.path / "tables" / "generation_steps.parquet").exists()
    assert (overlay.path / "tables" / "streams.parquet").exists()
    assert (overlay.path / "tables" / "token_spaces.parquet").exists()
    assert (overlay.path / "tables" / "robot_state.parquet").exists()
    assert (overlay.path / "tables" / "scene_state.parquet").exists()
    assert (overlay.path / "tables" / "camera_state.parquet").exists()
    assert (overlay.path / "tables" / "evaluation.parquet").exists()
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
        tmp_path / "object_overlay",
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
        tmp_path / "object_overlay_bbox",
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
        tmp_path / "object_overlay_camera_bbox",
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
    zarr_arrays = [array for array in dense_arrays if array["storage"]["format"] == "zarr"]
    parquet_arrays = [
        array for array in dense_arrays if array["storage"]["format"] == "parquet_column"
    ]
    assert zarr_arrays
    assert parquet_arrays
    assert all(array["storage"]["chunks"] for array in zarr_arrays if array["shape"])
    assert all(array["storage"]["compression"] == "zstd" for array in zarr_arrays)
    assert all(array["storage"]["compression"] == "snappy" for array in parquet_arrays)
    assert {array["storage"]["format"] for array in image_arrays} == {"mp4"}
    assert all(array["storage"]["compression"] == "h264" for array in image_arrays)


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


def test_trace_dataset_open_rejects_nested_overlay_bundles(tmp_path):
    root = tmp_path / "external-dataset"
    _make_minimal_trace(
        root / "traces" / "mechanistic_light" / "libero_object" / "task_00" / "a"
    )
    _make_minimal_trace(
        root / "traces" / "mechanistic_light" / "libero_goal" / "task_01" / "b"
    )

    with pytest.raises(FileNotFoundError, match="No LeRobot v3 dataset root"):
        TraceDataset.open(root)
