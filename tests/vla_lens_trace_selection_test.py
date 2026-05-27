# ruff: noqa: F403,F405
from tests._support.vla_lens_trace_mvp import *


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
