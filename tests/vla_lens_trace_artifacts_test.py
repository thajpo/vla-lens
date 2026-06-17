# ruff: noqa: F403,F405
import json

from tests._support.vla_lens_trace_mvp import *
from vla_lens.pi05.object_flow import save_pi05_object_flow_artifact
from vla_lens.pi05.policy_call_labels import save_pi05_policy_call_labels_artifact
from vla_lens.probes.workflow_prepare import _attach_episode_metadata


def test_pi05_interaction_metrics_derives_object_labels(tmp_path):
    root = tmp_path / "interaction"
    bundle_path = root / "red_cube_trace"
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
    bundle = TraceBundle.create(
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
    dataset = TraceDataset(root, [bundle])

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


def test_pi05_interaction_metrics_matches_backend_object_aliases(tmp_path):
    root = tmp_path / "interaction-aliases"
    bundle_path = root / "alias_trace"
    timesteps = 4
    positions = np.zeros((timesteps, 2, 3), dtype=np.float32)
    manifest = TraceManifest(
        trace_id="alias_trace",
        episode_id="alias_trace",
        task_id="0",
        prompt="pick up the book and place it in the back compartment of the caddy",
        model_id="pi05",
        env_id="libero_10",
        robot_id="panda",
        outcome="success",
        length=timesteps,
    )
    bundle = TraceBundle.create(
        bundle_path,
        manifest=manifest,
        timesteps=pd.DataFrame({"timestep": np.arange(timesteps)}),
        policy_calls=pd.DataFrame(
            {
                "policy_call_index": [0],
                "episode_id": ["alias_trace"],
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
                "object_name": ["black_book_1", "desk_caddy_1"],
                "object_kind": ["object", "fixture"],
            }
        ),
        episode_arrays={
            "scene_object_pos": ArraySpec(positions, ["timestep", "object", "xyz"]),
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
    dataset = TraceDataset(root, [bundle])

    saved = save_pi05_interaction_metrics_artifact(dataset)
    row = saved.episode_labels.iloc[0]

    assert row["target_parse_status"] == "multi"
    assert row["primary_target_object"] == "black_book_1"
    assert json.loads(row["target_objects"]) == ["black_book_1", "desk_caddy_1"]


def test_pi05_object_flow_derives_role_and_timestep_labels(tmp_path):
    dataset = _object_flow_dataset(tmp_path / "object-flow")

    saved = save_pi05_object_flow_artifact(dataset, rebuild_index=False)

    roles = saved.object_roles.set_index("object_name")
    assert saved.artifact.artifact_type == "pi05_object_flow"
    assert bool(roles.loc["red_cube_1", "role_manipulated"])
    assert not bool(roles.loc["red_cube_1", "role_receptacle"])
    assert bool(roles.loc["blue_bowl_1", "role_receptacle"])
    assert not bool(roles.loc["blue_bowl_1", "role_manipulated"])

    step = saved.flow_steps.iloc[0]
    assert step["object_name"] == "red_cube_1"
    assert step["target_object_name"] == "blue_bowl_1"
    assert step["step_type"] == "manipulate_object"

    timestep_labels = saved.timestep_labels.set_index("timestep")
    assert timestep_labels.loc[0, "next_manipulated_object"] == "red_cube_1"
    assert timestep_labels.loc[1, "active_manipulated_object"] == "red_cube_1"
    assert timestep_labels.loc[1, "active_receptacle_object"] == "blue_bowl_1"
    assert timestep_labels.loc[0, "task_phase"] == "approach"
    assert timestep_labels.loc[1, "task_phase"] == "contact"
    assert (dataset.root / saved.artifact.method["outputs"]["timestep_labels"]).exists()


def test_probe_metadata_attaches_object_flow_timestep_labels(tmp_path):
    dataset = _object_flow_dataset(tmp_path / "object-flow-probe-rows")
    save_pi05_object_flow_artifact(dataset, rebuild_index=False)
    rows = pd.DataFrame(
        {
            "trace_id": ["flow_trace", "flow_trace"],
            "timestep": [0, 4],
            "policy_call_index": [0, 1],
        }
    )

    merged = _attach_episode_metadata(rows, dataset)

    assert "next_manipulated_object" in merged
    assert "active_manipulated_object" in merged
    assert merged.loc[0, "next_manipulated_object"] == "red_cube_1"
    assert merged.loc[1, "active_manipulated_object"] == "red_cube_1"
    assert merged.loc[1, "active_receptacle_object"] == "blue_bowl_1"


def test_pi05_policy_call_labels_align_object_flow_to_policy_calls(tmp_path):
    dataset = _object_flow_dataset(tmp_path / "policy-call-labels")
    save_pi05_object_flow_artifact(dataset, rebuild_index=False)

    saved = save_pi05_policy_call_labels_artifact(dataset, rebuild_index=False)

    labels = saved.policy_call_labels
    assert saved.artifact.artifact_type == "pi05_policy_call_labels"
    assert labels["policy_call_index"].tolist() == [0, 1, 2, 3]
    first = labels.iloc[0]
    second = labels.iloc[1]
    assert first["next_manipulated_object"] == "red_cube_1"
    assert bool(first["is_pre_contact"])
    assert bool(first["is_pre_motion"])
    assert not bool(second["is_pre_contact"])
    assert not bool(second["is_pre_motion"])
    assert json.loads(first["candidate_objects"]) == ["blue_bowl_1", "red_cube_1"]
    assert (dataset.root / saved.artifact.method["outputs"]["policy_call_labels"]).exists()


def test_probe_metadata_attaches_policy_call_labels(tmp_path):
    dataset = _object_flow_dataset(tmp_path / "policy-call-probe-rows")
    save_pi05_interaction_metrics_artifact(dataset)
    save_pi05_object_flow_artifact(dataset, rebuild_index=False)
    save_pi05_policy_call_labels_artifact(dataset, rebuild_index=False)
    rows = pd.DataFrame(
        {
            "trace_id": ["flow_trace", "flow_trace"],
            "timestep": [0, 3],
            "policy_call_index": [0, 1],
        }
    )

    merged = _attach_episode_metadata(rows, dataset)

    assert "is_pre_contact" in merged
    assert "is_pre_motion" in merged
    assert "next_manipulated_is_target" in merged
    assert "next_manipulated_is_primary_target" in merged
    assert bool(merged.loc[0, "is_pre_contact"])
    assert bool(merged.loc[0, "is_pre_motion"])
    assert bool(merged.loc[0, "next_manipulated_is_target"])
    assert bool(merged.loc[0, "next_manipulated_is_primary_target"])
    assert not bool(merged.loc[1, "is_pre_contact"])
    assert not bool(merged.loc[1, "is_pre_motion"])
    assert not bool(merged.loc[1, "next_manipulated_is_target"])
    assert not bool(merged.loc[1, "next_manipulated_is_primary_target"])
    assert "target_contact_within_1_policy_calls" in merged
    assert "target_motion_within_1_policy_calls" in merged
    assert bool(merged.loc[0, "target_contact_in_future"])
    assert bool(merged.loc[0, "target_contact_within_1_policy_calls"])
    assert bool(merged.loc[0, "target_motion_within_1_policy_calls"])
    assert not bool(merged.loc[1, "target_contact_in_future"])
    assert not bool(merged.loc[1, "target_contact_within_1_policy_calls"])
    assert not bool(merged.loc[1, "target_motion_within_1_policy_calls"])


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
        assert bundle.overlay_bundle is not None
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
        final_success.to_parquet(
            bundle.overlay_bundle.path / "tables" / "evaluation.parquet",
            index=False,
        )
        bundle.__dict__.pop("evaluation", None)
        bundle.overlay_bundle.__dict__.pop("evaluation", None)
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


def _object_flow_dataset(root) -> TraceDataset:
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
