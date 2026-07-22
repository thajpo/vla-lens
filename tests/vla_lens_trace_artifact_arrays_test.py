# ruff: noqa: F403,F405

from tests._support.object_flow_dataset import object_flow_dataset
from tests._support.vla_lens_trace_mvp import *
from vla_lens.pi05.object_flow import save_pi05_object_flow_artifact
from vla_lens.probes import run_probe_suite
from vla_lens.probes.suite import trained_label_shuffle_metrics
from vla_lens.probes.workflow_artifacts import _grouped_bootstrap_intervals
from vla_lens.probes.workflow_prepare import _apply_row_expansion, _ensure_selection_split


def test_object_role_row_expansion_and_regression_metadata_baselines(tmp_path):
    dataset = object_flow_dataset(tmp_path / "object-row-expansion")
    save_pi05_object_flow_artifact(dataset, rebuild_index=False)
    rows = pd.DataFrame({"trace_id": ["flow_trace", "flow_trace"], "timestep": [1, 4]})
    X = np.arange(6, dtype=np.float32).reshape(2, 3)

    expanded_X, expanded_rows, summary = _apply_row_expansion(
        X,
        rows,
        dataset,
        {"kind": "object_roles", "prefix": "probe_object"},
    )

    assert summary["input_rows"] == 2
    assert summary["output_rows"] == 4
    assert expanded_X.shape == (4, 3)
    assert set(expanded_rows["probe_object_name"]) == {"red_cube_1", "blue_bowl_1"}
    assert "probe_object_role_manipulated" in expanded_rows

    probe_rows = pd.DataFrame(
        {
            "split": ["train", "train", "test", "test"],
            "object": ["a", "b", "a", "b"],
            "target": [0.0, 10.0, 0.0, 10.0],
        }
    )
    features = {"uninformative": np.zeros((4, 2), dtype=np.float32)}
    results = run_probe_suite(
        probe_rows,
        features,
        ["target"],
        metadata_baseline_columns=["object"],
        target_kinds={"target": "regression"},
    )

    row = results.iloc[0]
    assert row["metadata_baseline"] == "object"
    assert row["baseline_score"] > row["score"]
    assert any(
        baseline["baseline"] == "object"
        for baseline in row["details"]["metadata_baselines"]
    )


def test_default_probe_battery_includes_linear_and_mlp_with_trained_nulls():
    rows = pd.DataFrame(
        {
            "split": ["train"] * 16 + ["val"] * 8,
            "trace_id": [f"episode-{index // 2}" for index in range(24)],
            "episode_id": [f"episode-{index // 2}" for index in range(24)],
            "timestep": list(range(24)),
            "target": [False, True] * 12,
        }
    )
    features = np.arange(48, dtype=np.float32).reshape(24, 2)

    results = run_probe_suite(
        rows,
        {"features": features},
        ["target"],
        eval_values=["val"],
    )
    nulls = trained_label_shuffle_metrics(
        rows,
        features,
        "target",
        split_column="split",
        train_value="train",
        eval_value="val",
        probe_type="classification",
        model_name="mlp",
        runs=3,
    )

    assert set(results["model"]) == {"linear", "mlp"}
    assert set(nulls["null_kind"]) == {"trained_label_shuffle"}
    assert len(nulls) == 3


def test_trained_null_permutes_constant_labels_by_episode():
    rows = pd.DataFrame(
        {
            "split": ["train"] * 8 + ["validation"] * 4,
            "trace_id": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
            "target": [False] * 4 + [True] * 4 + [False, True, False, True],
        }
    )
    nulls = trained_label_shuffle_metrics(
        rows,
        np.arange(24, dtype=np.float32).reshape(12, 2),
        "target",
        split_column="split",
        train_value="train",
        eval_value="validation",
        probe_type="classification",
        model_name="linear",
        group_column="trace_id",
        runs=3,
    )

    assert set(nulls["shuffle_strategy"]) == {"permute_group_labels"}
    assert set(nulls["shuffle_group_column"]) == {"trace_id"}


def test_automatic_validation_split_is_grouped_and_keeps_test_locked():
    rows = pd.DataFrame(
        {
            "trace_id": [f"trace-{index}" for index in range(8)],
            "task_id": [0, 0, 1, 1, 2, 2, 3, 3],
            "split": ["train"] * 6 + ["test"] * 2,
        }
    )

    split_rows, summary = _ensure_selection_split(
        rows,
        "split",
        train_value="train",
        selection_value="validation",
        test_value="test",
        split_kind="heldout_task",
    )

    assert summary["created"] is True
    assert summary["group_column"] == "task_id"
    assert set(split_rows.loc[split_rows["split"] == "test", "task_id"]) == {3}
    train_tasks = set(split_rows.loc[split_rows["split"] == "train", "task_id"])
    validation_tasks = set(split_rows.loc[split_rows["split"] == "validation", "task_id"])
    assert train_tasks.isdisjoint(validation_tasks)


def test_probe_selection_cannot_use_final_test_split():
    rows = pd.DataFrame({"trace_id": ["a", "b"], "split": ["train", "test"]})

    with pytest.raises(ValueError, match="must differ from test_value"):
        _ensure_selection_split(
            rows,
            "split",
            train_value="train",
            selection_value="test",
            test_value="test",
            split_kind="random_episode",
        )


def test_probe_intervals_resample_whole_episodes():
    predictions = pd.DataFrame(
        {
            "split": ["test"] * 8,
            "trace_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "target_kind": ["classification"] * 8,
            "actual": [False, True] * 4,
            "predicted": [False, True, False, False, True, True, False, True],
            "correct": [True, True, True, False, False, True, True, True],
            "baseline_prediction": [False] * 8,
        }
    )

    intervals = _grouped_bootstrap_intervals(predictions, samples=100)

    assert intervals.iloc[0]["group_column"] == "trace_id"
    assert intervals.iloc[0]["group_count"] == 4
    assert intervals.iloc[0]["low"] <= intervals.iloc[0]["estimate"]
    assert intervals.iloc[0]["high"] >= intervals.iloc[0]["estimate"]
    assert intervals.iloc[0]["delta_low"] <= intervals.iloc[0]["delta_estimate"]
    assert intervals.iloc[0]["delta_high"] >= intervals.iloc[0]["delta_estimate"]


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
        tmp_path / "metrics",
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
