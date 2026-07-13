# ruff: noqa: F403,F405
from dataclasses import replace

from tests._support.vla_lens_trace_mvp import *
from vla_lens.artifacts import LensArtifact
from vla_lens.dataset import build_dataset_index


def test_probe_episode_lens_view_resolves_selection_and_contributors(tmp_path):
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
            "name": "Episode LensView outcome probe",
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

    payload = _discovery_artifact_episode_lens_view_payload(
        reopened,
        saved.artifact.artifact_id,
        {"trace_id": [trace_ids[-1]], "top_k": ["5"]},
    )
    view = payload["view"]
    site_readout = view["view"]["site_readout"]

    assert view["schema_version"] == "episode_lens_view.v1"
    assert view["family"] == "probe_suite"
    assert view["current_selection"]["trace_id"] == trace_ids[-1]
    assert view["inspector"]["default_ranking_id"] == "probe_contributors"
    assert view["inspector"]["pipeline_marks"]
    assert view["inspector"]["timeline_marks"]
    assert len(view["inspector"]["timeline_marks"]) == len(view["view"]["temporal_readout"]["rows"])
    assert any(mark.get("selected") for mark in view["inspector"]["timeline_marks"])
    assert {ranking["id"] for ranking in view["inspector"]["rankings"]} == {
        "probe_contributors",
        "raw_activations",
    }
    assert view["view"]["probe"]["training_spec"]["policy_calls"] == "all"
    assert view["recommended_selection"]["model_site_id"]
    assert view["resolved_selection"]["model_site_id"] == site_readout["model_site_id"]
    assert view["readout"]["verdict"] in {
        "correct",
        "wrong",
        "high_conf_wrong",
        "ambiguous",
        "unscored",
        "unknown",
    }
    assert site_readout["available"] is True
    assert site_readout["probe_contribution_ranking_available"] is True
    assert site_readout["feature_contributors_available"] is True
    assert site_readout["top_k"] == 5
    assert len(site_readout["feature_contributors"]) <= 5
    first = site_readout["feature_contributors"][0]
    assert first["contribution"] == pytest.approx(first["normalized_activation"] * first["weight"])
    assert first["abs_contribution"] == pytest.approx(abs(first["contribution"]))
    assert first["feature_ref"]["model_site_id"] == view["resolved_selection"]["model_site_id"]
    assert site_readout["logit_reconstruction"]["reconstructed_logit"] == pytest.approx(
        site_readout["logit_reconstruction"]["bias"]
        + site_readout["logit_reconstruction"]["total_contribution_sum"]
    )
    intervention = next(
        action for action in view["actions"] if action["kind"] == "send_to_intervention"
    )
    assert intervention["enabled"] is True
    assert intervention["seed"]["trace_id"] == trace_ids[-1]
    assert intervention["seed"]["model_site_id"] == view["resolved_selection"]["model_site_id"]
    assert intervention["seed"]["token_space"] == site_readout["token_space_id"]

    artifact = reopened.load_artifact(saved.artifact.artifact_id)
    method = dict(artifact.method)
    method_input = dict(method["input"])
    selector = dict(method_input["selector"])
    selector["policy_calls"] = [0, 1]
    method_input["selector"] = selector
    method["input"] = method_input
    reopened.save_artifact(replace(artifact, method=method))
    build_dataset_index(reopened.root)
    reopened = TraceDataset.open(reopened.root)
    payload = _discovery_artifact_episode_lens_view_payload(
        reopened,
        saved.artifact.artifact_id,
        {"trace_id": [trace_ids[-1]], "top_k": ["3"]},
    )
    assert payload["view"]["view"]["probe"]["trained_policy_call_scope"] == "selected"
    assert payload["view"]["view"]["probe"]["training_spec"]["policy_calls"] == [0, 1]


def test_probe_episode_lens_view_marks_nonlinear_contributors_unavailable(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=6, timesteps=8)
    trace_ids = [bundle.manifest.trace_id for bundle in dataset.bundles]
    spec = dump_probe_spec(
        {
            "name": "Nonlinear guard outcome probe",
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
    artifact = reopened.load_artifact(saved.artifact.artifact_id)
    method = dict(artifact.method)
    probe = dict(method["probe"])
    best_state = dict(probe["best_model_state"])
    best_state["model"] = "mlp"
    probe["best_model_state"] = best_state
    method["probe"] = probe
    reopened.save_artifact(replace(artifact, method=method))
    reopened = TraceDataset.open(dataset.root)

    payload = _discovery_artifact_episode_lens_view_payload(
        reopened,
        saved.artifact.artifact_id,
        {"trace_id": [trace_ids[-1]]},
    )
    site_readout = payload["view"]["view"]["site_readout"]

    assert site_readout["raw_activation_ranking_available"] is True
    assert site_readout["probe_contribution_ranking_available"] is False
    assert "nonlinear" in site_readout["feature_contributors_unavailable_reason"]


def test_unsupported_episode_lens_view_family_returns_unavailable_payload(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=4)
    trace_id = dataset.bundles[0].manifest.trace_id
    artifact = dataset.save_artifact(
        LensArtifact.create(
            artifact_type="sae_feature",
            name="Future SAE lens",
            scope="dataset",
        )
    )
    build_dataset_index(dataset.root)
    reopened = TraceDataset.open(dataset.root)

    payload = _discovery_artifact_episode_lens_view_payload(
        reopened,
        artifact.artifact_id,
        {"trace_id": [trace_id]},
    )
    view = payload["view"]

    assert view["available"] is False
    assert view["family"] == "sae_feature"
    assert "does not support episode LensViews yet" in view["unavailable_reason"]
    assert view["current_selection"]["trace_id"] == trace_id
    assert view["actions"][0]["kind"] == "open_artifact_debug"
