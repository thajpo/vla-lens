# ruff: noqa: F403,F405
import json

from tests._support.object_flow_dataset import object_flow_dataset
from tests._support.vla_lens_trace_mvp import *
from vla_lens.pi05.object_flow import save_pi05_object_flow_artifact
from vla_lens.pi05.policy_call_labels import save_pi05_policy_call_labels_artifact
from vla_lens.probes.workflow_prepare import _attach_episode_metadata
from vla_lens.probes.workflow_targets import _normalize_target_spec, _resolve_probe_target


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
    dataset = object_flow_dataset(tmp_path / "object-flow")

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
    dataset = object_flow_dataset(tmp_path / "object-flow-probe-rows")
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
    dataset = object_flow_dataset(tmp_path / "policy-call-labels")
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
    dataset = object_flow_dataset(tmp_path / "policy-call-probe-rows")
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
    assert "next_manipulated_present" in merged
    assert "next_manipulated_is_target" in merged
    assert "next_manipulated_is_primary_target" in merged
    assert bool(merged.loc[0, "is_pre_contact"])
    assert bool(merged.loc[0, "is_pre_motion"])
    assert bool(merged.loc[0, "next_manipulated_present"])
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

    rows = pd.DataFrame(
        {
            "trace_id": [dataset.bundles[0].manifest.trace_id],
            "timestep": [0],
            "active_manipulated_object": ["target_cube"],
        }
    )
    dynamic_rows = _resolve_probe_target(
        dataset,
        rows,
        _normalize_target_spec(
            {
                "name": "active_cube_x",
                "kind": "regression",
                "source": "scene_state",
                "field": "pose",
                "object_column": "active_manipulated_object",
                "selector": {"component": "x"},
            }
        ),
    )
    expected_active_x = dataset.bundles[0].array("scene_object_poses", mmap=True)[0, 0, 0]
    assert np.isclose(dynamic_rows.loc[0, "active_cube_x"], expected_active_x)

    relative_rows = _resolve_probe_target(
        dataset,
        rows,
        _normalize_target_spec(
            {
                "name": "active_cube_relative_x",
                "kind": "regression",
                "source": "scene_state",
                "field": "pose",
                "object_column": "active_manipulated_object",
                "selector": {"component": "x"},
                "relative_to": {"array_id": "robot_eef_pose"},
            }
        ),
    )
    expected_eef_x = dataset.bundles[0].array("robot_eef_pose", mmap=True)[0, 0]
    assert np.isclose(
        relative_rows.loc[0, "active_cube_relative_x"],
        expected_active_x - expected_eef_x,
    )

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
