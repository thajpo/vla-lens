# ruff: noqa: F403,F405
from tests._support.vla_lens_trace_mvp import *


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
    bundle = TraceBundle.create(
        root / "trace",
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
    dataset = TraceDataset(root, [bundle])

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


def test_trace_dataset_open_rejects_overlay_bundle_root(tmp_path):
    bundle = _make_minimal_trace(tmp_path / "single_overlay")

    with pytest.raises(FileNotFoundError, match="No LeRobot v3 dataset root"):
        TraceDataset.open(bundle.path)


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
    artifact_dir = dataset.root / "vla_lens" / "artifacts" / saved.artifact.artifact_id
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


def test_probe_episode_payloads_link_artifacts_to_dataset_traces(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=6, timesteps=8)
    trace_ids = [bundle.manifest.trace_id for bundle in dataset.bundles]
    pd.DataFrame(
        {
            "trace_id": trace_ids,
            "split": ["train", "train", "validation", "validation", "test", "test"],
        }
    ).to_csv(dataset.root / "probe_splits.csv", index=False)
    spec = dump_probe_spec(
        {
            "name": "Episode UI outcome probe",
            "target": {"kind": "outcome"},
            "features": {
                "module": "action_head.layers.*.resid",
                "tensor_type": "resid",
                "token_kind": "action",
                "reduction": "mean",
            },
            "split": {"kind": "random_episode"},
            "baseline": ["majority_class"],
            "probe": {"models": ["linear"]},
            "sweep": "layer",
        }
    )
    import yaml

    train_probe_artifact_from_spec(dataset, yaml.safe_load(spec))
    reopened = TraceDataset.open(dataset.root)

    index_payload = _probe_index_payload(reopened)
    assert index_payload["total"] >= 1
    assert index_payload["trace_count"] == len(trace_ids)
    assert index_payload["split_source"] == "probe_splits.csv"
    probe_index = next(
        probe for probe in index_payload["probes"] if probe["name"] == "Episode UI outcome probe"
    )
    assert probe_index["by_trace"][trace_ids[0]]["split_category"] == "train"
    assert probe_index["by_trace"][trace_ids[2]]["split_category"] == "validation"
    assert probe_index["by_trace"][trace_ids[-1]]["split_category"] == "test"

    episode_payload = _episode_probes_payload(reopened, {"trace_id": [trace_ids[-1]]})
    assert episode_payload["trace_id"] == trace_ids[-1]
    assert episode_payload["total"] >= 1
    assert episode_payload["available_count"] >= 1
    episode_probe = next(
        probe for probe in episode_payload["probes"] if probe["name"] == "Episode UI outcome probe"
    )
    assert episode_probe["available"] is True
    assert episode_probe["row_count"] > 0
    assert episode_probe["episode_summary"]["best_row"]
    assert {"actual", "predicted", "confidence", "correct"}.issubset(
        episode_probe["episode_summary"]
    )


def test_observational_comparisons_rank_real_probe_candidates(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=6, timesteps=8)
    trace_ids = [bundle.manifest.trace_id for bundle in dataset.bundles]
    pd.DataFrame(
        {
            "trace_id": trace_ids,
            "split": ["train", "train", "validation", "validation", "test", "test"],
        }
    ).to_csv(dataset.root / "probe_splits.csv", index=False)
    spec = dump_probe_spec(
        {
            "name": "Comparison outcome probe",
            "target": {"kind": "outcome"},
            "features": {
                "module": "action_head.layers.*.resid",
                "tensor_type": "resid",
                "token_kind": "action",
                "reduction": "mean",
            },
            "split": {"kind": "random_episode"},
            "baseline": ["majority_class"],
            "probe": {"models": ["linear"]},
            "sweep": "layer",
        }
    )
    import yaml

    saved = train_probe_artifact_from_spec(dataset, yaml.safe_load(spec))
    reopened = TraceDataset.open(dataset.root)

    payload = _observational_comparisons_payload(
        reopened,
        {
            "trace_id": [trace_ids[0]],
            "probe_id": [saved.artifact.artifact_id],
            "limit": ["4"],
        },
    )

    assert payload["artifact_type"] == "observational_counterfactual_comparison"
    assert payload["comparison_kind"] == "nearest_neighbor_existing_trace"
    assert payload["causal"] is False
    assert payload["source"]["episode"]["trace_id"] == trace_ids[0]
    assert payload["source"]["probe"]["split_category"] == "train"
    assert payload["candidates"]
    assert all(candidate["trace_id"] != trace_ids[0] for candidate in payload["candidates"])
    assert all(candidate["contract"]["causal"] is False for candidate in payload["candidates"])
    assert payload["candidates"][0]["probe"]["split_category"] in {"test", "validation"}
    assert "training-set probe record" not in payload["candidates"][0]["reasons"]


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
    reopened = TraceDataset.open(dataset.root)
    index_payload = _probe_index_payload(reopened)
    probe_index = next(
        probe for probe in index_payload["probes"] if probe["name"] == "Sidecar split probe"
    )
    assert probe_index["by_trace"][trace_ids[4]]["split_category"] == "validation"
    assert probe_index["by_trace"][trace_ids[5]]["split_category"] == "test"
    assert "benchmark" in saved.artifact.method["metadata_baseline_columns"]
    assert "benchmark" in saved.artifact.display["source_columns"]
    assert "first_moved_object" in saved.artifact.display["source_columns"]
