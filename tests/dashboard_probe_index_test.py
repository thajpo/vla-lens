from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from vla_lens import create_synthetic_trace_dataset
from vla_lens.artifacts import LensArtifact
from vla_lens.dataset.index import (
    ARTIFACT_COLUMNS,
    ARTIFACT_INDEX,
    PROBE_EPISODE_COLUMNS,
    PROBE_EPISODE_INDEX,
    PROBE_PREDICTION_COLUMNS,
    PROBE_PREDICTIONS,
    index_manifest_path,
)
from vla_lens.server.fastapi_app import create_dashboard_app


def test_fastapi_indexed_probe_episode_page_includes_split_record(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    trace_id = dataset.bundles[0].manifest.trace_id
    _write_index_table(
        dataset.root,
        "probe_predictions",
        PROBE_PREDICTIONS,
        pd.DataFrame(
            {
                "probe_id": ["probe-a"],
                "trace_id": [trace_id],
                "split": ["heldout-validation"],
                "split_category": ["validation"],
                "actual": ["success"],
                "predicted": ["failure"],
                "confidence": [0.91],
                "correct": [False],
                "model": ["linear"],
                "feature": ["action_head.layers.0"],
            }
        ),
        PROBE_PREDICTION_COLUMNS,
    )
    _write_index_table(
        dataset.root,
        "probe_episode_index",
        PROBE_EPISODE_INDEX,
        pd.DataFrame(
            {
                "probe_id": ["probe-a"],
                "trace_id": [trace_id],
                "split": ["heldout-validation"],
                "split_category": ["validation"],
                "actual": ["success"],
                "predicted": ["failure"],
                "confidence": [0.91],
                "correct": [False],
                "correct_rate": [0.0],
                "model": ["linear"],
                "feature": ["action_head.layers.0"],
                "policy_call_index": [3],
                "row_count": [4],
            }
        ),
        PROBE_EPISODE_COLUMNS,
    )
    client = TestClient(create_dashboard_app(dataset.root))

    payload = client.get("/api/episodes", params={"limit": "1", "probe_id": "probe-a"}).json()
    probe_record = payload["episodes"][0]["probe_record"]

    assert probe_record["available"] is True
    assert probe_record["split"] == "heldout-validation"
    assert probe_record["split_category"] == "validation"
    assert probe_record["correct"] is False
    assert probe_record["correct_rate"] == 0.0
    assert probe_record["policy_call_index"] == 3
    assert probe_record["row_count"] == 4


def test_fastapi_probe_evidence_defaults_to_scored_split_records(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=2)
    trace_ids = [bundle.manifest.trace_id for bundle in dataset.bundles]
    _write_index_table(
        dataset.root,
        "artifact_index",
        ARTIFACT_INDEX,
        pd.DataFrame(
            {
                "artifact_id": ["probe-a"],
                "artifact_type": ["probe_suite"],
                "name": ["Probe A"],
                "metrics": ['{"target":"outcome","best_model":"linear"}'],
            }
        ),
        ARTIFACT_COLUMNS,
    )
    _write_index_table(
        dataset.root,
        "probe_predictions",
        PROBE_PREDICTIONS,
        pd.DataFrame(
            {
                "probe_id": ["probe-a"],
                "trace_id": [trace_ids[1]],
                "split": ["val_heldout_task"],
                "split_category": ["validation"],
                "actual": ["success"],
                "predicted": ["failure"],
                "confidence": [0.91],
                "correct": [False],
                "model": ["linear"],
                "feature": ["action_head.layers.0"],
            }
        ),
        PROBE_PREDICTION_COLUMNS,
    )
    _write_index_table(
        dataset.root,
        "probe_episode_index",
        PROBE_EPISODE_INDEX,
        pd.DataFrame(
            {
                "probe_id": ["probe-a"],
                "trace_id": [trace_ids[1]],
                "split": ["val_heldout_task"],
                "split_category": ["validation"],
                "actual": ["success"],
                "predicted": ["failure"],
                "confidence": [0.91],
                "correct": [False],
                "correct_rate": [0.0],
                "model": ["linear"],
                "feature": ["action_head.layers.0"],
                "policy_call_index": [2],
                "row_count": [2],
            }
        ),
        PROBE_EPISODE_COLUMNS,
    )
    client = TestClient(create_dashboard_app(dataset.root))

    payload = client.get("/api/probes/probe-a/evidence", params={"limit": "1"}).json()
    episode = payload["episodes"][0]

    assert payload["total"] == 1
    assert episode["trace_id"] == trace_ids[1]
    assert episode["probe_record"]["split_category"] == "validation"
    assert episode["probe_record"]["correct"] is False


def test_fastapi_discovery_artifact_families_exposes_contracts(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    client = TestClient(create_dashboard_app(dataset.root))

    response = client.get("/api/discovery-artifact-families")
    payload = response.json()

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    family_by_type = {family["artifact_type"]: family for family in payload["families"]}
    assert family_by_type["probe_suite"]["target_kind"] == "probe_direction"
    assert family_by_type["probe_suite"]["available"] is True
    assert family_by_type["probe_suite"]["reason"] == ""
    assert "add_direction" in family_by_type["probe_suite"]["operators"]
    assert "sae_feature" in family_by_type


def test_fastapi_discovery_artifact_episode_ranking_delegates_to_probe_interest(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=3, timesteps=2)
    trace_ids = [bundle.manifest.trace_id for bundle in dataset.bundles]
    _write_index_table(
        dataset.root,
        "artifact_index",
        ARTIFACT_INDEX,
        pd.DataFrame(
            {
                "artifact_id": ["probe-a"],
                "artifact_type": ["probe_suite"],
                "name": ["Probe A"],
                "metrics": ['{"target":"outcome","best_model":"linear"}'],
            }
        ),
        ARTIFACT_COLUMNS,
    )
    _write_index_table(
        dataset.root,
        "probe_episode_index",
        PROBE_EPISODE_INDEX,
        pd.DataFrame(
            {
                "probe_id": ["probe-a", "probe-a"],
                "trace_id": [trace_ids[0], trace_ids[2]],
                "split": ["train", "test"],
                "split_category": ["train", "test"],
                "actual": ["success", "success"],
                "predicted": ["success", "failure"],
                "confidence": [0.61, 0.96],
                "correct": [True, False],
                "correct_rate": [1.0, 0.0],
                "model": ["linear", "linear"],
                "feature": ["action_head.layers.0", "action_head.layers.0"],
                "policy_call_index": [1, 5],
                "row_count": [1, 4],
            }
        ),
        PROBE_EPISODE_COLUMNS,
    )
    client = TestClient(create_dashboard_app(dataset.root))

    generic = client.get(
        "/api/discovery-artifacts/probe-a/episodes",
        params={"limit": "3", "rank_by": "interest"},
    ).json()
    legacy = client.get(
        "/api/episodes",
        params={"limit": "3", "probe_id": "probe-a", "sort": "probe_interest"},
    ).json()

    assert generic["available"] is True
    assert generic["artifact"]["artifact_type"] == "probe_suite"
    assert generic["family"]["target_kind"] == "probe_direction"
    assert [episode["trace_id"] for episode in generic["episodes"]] == [
        episode["trace_id"] for episode in legacy["episodes"]
    ]
    assert generic["episodes"][0]["trace_id"] == trace_ids[2]
    assert generic["episodes"][0]["probe_record"]["correct"] is False

    probe_index = client.get("/api/probe-index").json()
    probe = probe_index["probes"][0]
    assert probe["review_stats"]["highConfidence"] == 1
    assert probe["review_stats_by_split"]["train"]["correct"] == 1
    assert probe["review_stats_by_split"]["train"]["wrong"] == 0
    assert probe["review_stats_by_split"]["test"]["wrong"] == 1
    assert probe["review_stats_by_split"]["test"]["highConfidence"] == 1
    assert probe["review_stats_by_split"]["test"]["highConfWrong"] == 1

    explicit_sort = client.get(
        "/api/discovery-artifacts/probe-a/episodes",
        params={"limit": "3", "sort": "episode_index"},
    ).json()
    assert [episode["trace_id"] for episode in explicit_sort["episodes"]] == trace_ids


def test_fastapi_discovery_artifact_readout_returns_probe_summary(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    trace_id = dataset.bundles[0].manifest.trace_id
    _write_index_table(
        dataset.root,
        "artifact_index",
        ARTIFACT_INDEX,
        pd.DataFrame(
            {
                "artifact_id": ["probe-a"],
                "artifact_type": ["probe_suite"],
                "name": ["Probe A"],
                "metrics": [
                    (
                        '{"target":"outcome","best_model":"linear",'
                        '"best_feature":"action_head.layers.0"}'
                    )
                ],
            }
        ),
        ARTIFACT_COLUMNS,
    )
    _write_index_table(
        dataset.root,
        "probe_episode_index",
        PROBE_EPISODE_INDEX,
        pd.DataFrame(
            {
                "probe_id": ["probe-a"],
                "trace_id": [trace_id],
                "split": ["test"],
                "split_category": ["test"],
                "actual": ["success"],
                "predicted": ["failure"],
                "confidence": [0.93],
                "correct": [False],
                "correct_rate": [0.0],
                "model": ["linear"],
                "feature": ["action_head.layers.0"],
                "model_site_id": ["pi05.action_head.input"],
                "token_space_id": ["pi05.action_horizon"],
                "policy_call_index": [4],
                "row_count": [3],
            }
        ),
        PROBE_EPISODE_COLUMNS,
    )
    client = TestClient(create_dashboard_app(dataset.root))

    payload = client.get(
        "/api/discovery-artifacts/probe-a/readout",
        params={"trace_id": trace_id},
    ).json()

    assert payload["available"] is True
    assert payload["readout_type"] == "probe_prediction"
    assert payload["summary"]["actual"] == "success"
    assert payload["summary"]["predicted"] == "failure"
    assert payload["summary"]["confidence"] == 0.93
    assert payload["summary"]["correct"] is False
    assert payload["summary"]["policy_call_index"] == 4
    assert payload["target_hint"]["model_site"] == "pi05.action_head.input"
    assert payload["target_hint"]["token_space"] == "pi05.action_horizon"
    assert payload["row_count"] == 3
    assert payload["rows"][0]["policy_call_index"] == 4


def test_fastapi_discovery_artifact_target_uses_family_contract_and_overrides(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    trace_id = dataset.bundles[0].manifest.trace_id
    _write_index_table(
        dataset.root,
        "artifact_index",
        ARTIFACT_INDEX,
        pd.DataFrame(
            {
                "artifact_id": ["probe-a"],
                "artifact_type": ["probe_suite"],
                "name": ["Probe A"],
                "metrics": ['{"target":"outcome","best_model":"linear"}'],
                "arrays": ['{"direction":"artifacts/probe-a/direction.npy"}'],
            }
        ),
        ARTIFACT_COLUMNS,
    )
    client = TestClient(create_dashboard_app(dataset.root))

    payload = client.get(
        "/api/discovery-artifacts/probe-a/target",
        params={
            "trace_id": trace_id,
            "policy_call": "4",
            "model_site": "pi05.action_head.input",
            "token_space": "pi05.action_horizon",
        },
    ).json()

    assert payload["available"] is True
    assert payload["target"]["kind"] == "probe_direction"
    assert payload["target"]["source_artifact_id"] == "probe-a"
    assert payload["target"]["source_artifact_type"] == "probe_suite"
    assert payload["target"]["model_site"] == "pi05.action_head.input"
    assert payload["target"]["token_space"] == "pi05.action_horizon"
    assert payload["target"]["metadata"]["trace_id"] == trace_id
    assert payload["target"]["metadata"]["policy_call_index"] == "4"
    assert payload["target"]["metadata"]["artifact_family"] == "probe_suite"


def test_fastapi_episode_probes_reads_index_without_scoring_side_effects(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    trace_id = dataset.bundles[0].manifest.trace_id
    _write_index_table(
        dataset.root,
        "artifact_index",
        ARTIFACT_INDEX,
        pd.DataFrame(
            {
                "artifact_id": ["probe-a"],
                "artifact_type": ["probe_suite"],
                "name": ["Probe A"],
                "metrics": ['{"target":"outcome","best_model":"linear"}'],
            }
        ),
        ARTIFACT_COLUMNS,
    )
    _write_index_table(
        dataset.root,
        "probe_episode_index",
        PROBE_EPISODE_INDEX,
        pd.DataFrame(
            {
                "probe_id": ["probe-a"],
                "trace_id": [trace_id],
                "split": ["test"],
                "split_category": ["test"],
                "actual": ["success"],
                "predicted": ["failure"],
                "confidence": [0.93],
                "correct": [False],
                "correct_rate": [0.0],
                "model": ["linear"],
                "feature": ["action_head.layers.0"],
                "policy_call_index": [4],
                "row_count": [3],
            }
        ),
        PROBE_EPISODE_COLUMNS,
    )
    output_root = dataset.root / "workbench" / "episode_probe_predictions"
    client = TestClient(create_dashboard_app(dataset.root))

    payload = client.get("/api/episode-probes", params={"trace_id": trace_id}).json()
    probe = payload["probes"][0]

    assert payload["available_count"] == 1
    assert probe["available"] is True
    assert probe["row_count"] == 3
    assert probe["episode_summary"]["correct"] is False
    assert probe["episode_summary"]["best_row"]["policy_call_index"] == 4
    assert not output_root.exists()


def test_fastapi_probe_studies_promotes_diagnostics_readouts(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=4)
    trace_id = dataset.bundles[0].manifest.trace_id
    saved = dataset.save_artifact(
        LensArtifact.create(
            artifact_type="probe_suite",
            name="Object-of-action probe",
            selector={
                "site": "pi05.expert.layers.by_step.hidden_tokens",
                "layers": [0, 4],
                "token_space": "pi05.action_suffix",
            },
            method={
                "normalization": {"method": "standardize"},
                "probe": {
                    "type": "classification",
                    "library": "sklearn",
                    "hyperparams": {
                        "linear": {
                            "model": "LogisticRegression",
                            "max_iter": 1000,
                            "class_weight": "balanced",
                        }
                    },
                    "primary_model": "linear",
                    "trained_on_split": "train",
                },
                "target": {"kind": "classification"},
                "evaluation": {"primary_metric": "balanced_accuracy"},
            },
            metrics={"target": "next_manipulated_object"},
        )
    )
    diagnostics = dataset._dataset_artifact_root() / "artifacts" / saved.artifact_id / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    (diagnostics / "summary.json").write_text(
        json.dumps(
            {
                "target": "next_manipulated_object",
                "selected_layer": "4",
                "selection_split": "val_heldout_task",
                "test_split": "test_heldout_task",
                "selected_layer_selection_balanced_accuracy": 0.62,
                "selected_layer_test_balanced_accuracy": 0.58,
                "selection_aware_null": {
                    "runs": 2,
                    "selection_score_mean": 0.21,
                    "selection_score_std": 0.03,
                    "selection_p_value": 0.333,
                    "test_score_mean": 0.18,
                    "test_score_std": 0.02,
                    "test_p_value": 0.333,
                },
                "feature_rows": 30,
                "policy_call_count": 15,
                "episode_count": 2,
                "layer_count": 2,
                "class_count": 3,
                "split_policy_call_counts": {
                    "train": 8,
                    "val_heldout_task": 4,
                    "test_heldout_task": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "target": [
                "next_manipulated_object",
                "next_manipulated_object",
                "task_phase",
                "next_action_type",
            ],
            "status": ["ok", "ok", "ok", "skipped"],
            "layer": [0, 4, 4, None],
            "split": ["val_heldout_task", "test_heldout_task", "test_heldout_task", None],
            "row_count": [4, 3, 3, None],
            "policy_call_count": [4, 3, 3, None],
            "class_count": [2, 2, 3, None],
            "balanced_accuracy": [0.55, 0.58, 0.74, None],
            "accuracy": [0.5, 0.67, 0.75, None],
            "macro_f1": [0.52, 0.6, 0.7, None],
            "top1_accuracy": [0.5, 0.67, 0.75, None],
            "top2_accuracy": [0.75, 1.0, 1.0, None],
            "top3_accuracy": [1.0, 1.0, 1.0, None],
            "train_balanced_accuracy": [0.9, 0.9, 0.88, None],
            "train_gap_balanced_accuracy": [0.35, 0.32, 0.14, None],
            "reason": [None, None, None, "missing reliable label"],
        }
    ).to_parquet(diagnostics / "readout_battery_metrics.parquet", index=False)
    pd.DataFrame(
        {
            "run": [0, 0, 1, 1],
            "layer": [4, 4, 4, 4],
            "split": [
                "val_heldout_task",
                "test_heldout_task",
                "val_heldout_task",
                "test_heldout_task",
            ],
            "score": [0.2, 0.18, 0.22, 0.17],
            "row_count": [4, 3, 4, 3],
            "policy_call_count": [4, 3, 4, 3],
            "selected_layer": [4, 4, 4, 4],
        }
    ).to_parquet(diagnostics / "selection_aware_null.parquet", index=False)
    pd.DataFrame(
        {
            "layer": [4],
            "split": ["test_heldout_task"],
            "actual": ["red_block_1"],
            "predicted": ["blue_block_1"],
            "row_count": [1],
            "policy_call_count": [1],
        }
    ).to_parquet(diagnostics / "confusion_matrix.parquet", index=False)
    pd.DataFrame(
        {
            "layer": [4],
            "split": ["test_heldout_task"],
            "class": ["red_block_1"],
            "row_support": [2],
            "policy_call_support": [2],
            "precision": [0.5],
            "recall": [0.5],
            "f1": [0.5],
            "one_vs_rest_balanced_accuracy": [0.75],
        }
    ).to_parquet(diagnostics / "per_class_metrics.parquet", index=False)
    pd.DataFrame(
        {
            "layer": [4],
            "split": ["test_heldout_task"],
            "lead_kind": ["contact_lead"],
            "lead_bucket": ["1_policy_call"],
            "balanced_accuracy": [0.58],
            "policy_call_count": [3],
        }
    ).to_parquet(diagnostics / "lead_time_metrics.parquet", index=False)
    pd.DataFrame(
        {
            "split": ["test_heldout_task"],
            "class": ["red_block_1"],
            "policy_call_count": [3],
        }
    ).to_parquet(diagnostics / "policy_call_support_by_class_split.parquet", index=False)
    pd.DataFrame(
        {
            "split": ["test_heldout_task"],
            "trace_id": [trace_id],
            "episode_id": [trace_id],
            "task_id": ["pick"],
            "prompt": ["pick up the red block"],
            "timestep": [4],
            "policy_call_index": [1],
            "layer": [4],
            "model_site_id": ["pi05.expert.layers.4.by_step.hidden_tokens"],
            "token_space_id": ["pi05.action_suffix"],
            "actual": ["red_block_1"],
            "predicted": ["blue_block_1"],
            "correct": [False],
            "confidence": [0.91],
            "task_phase": ["approach"],
            "contact_lead_bucket": ["1_policy_call"],
            "events_after": ["contact:red_block_1@5"],
        }
    ).to_parquet(diagnostics / "probe_error_browser.parquet", index=False)

    client = TestClient(create_dashboard_app(dataset.root))

    response = client.get("/api/probe-studies")
    payload = response.json()
    study_by_target = {
        item["target"]: item
        for item in payload["studies"]
        if item["artifact_id"] == saved.artifact_id
    }
    study = study_by_target["next_manipulated_object"]
    phase_study = study_by_target["task_phase"]

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert set(study_by_target) == {"next_manipulated_object", "task_phase"}
    assert study["study_id"] == f"{saved.artifact_id}::target=next_manipulated_object"
    assert phase_study["study_id"] == f"{saved.artifact_id}::target=task_phase"
    assert phase_study["source_artifact_id"] == saved.artifact_id
    assert phase_study["name"] == "Task phase"
    assert study["diagnostics_available"] is True
    assert study["counts"]["readout_count"] == 2
    assert study["counts"]["skipped_readout_count"] == 0
    assert study["counts"]["target_count"] == 1
    assert study["counts"]["null_run_count"] == 2
    assert study["objective"] == "Multiclass logistic regression"
    assert study["training_summary"]["objective"] == "Multiclass logistic regression"
    assert study["training_summary"]["preprocessing"] == "standardized X"
    assert "class-balanced" in study["training_summary"]["hyperparameters"]
    assert "max iter 1000" in study["training_summary"]["hyperparameters"]
    assert study["readouts"][0]["target"] == "next_manipulated_object"
    assert study["readouts"][0]["trained_probe_id"] == "NMO-L0-VAL-HELDOUT-TASK"
    assert study["readouts"][1]["split_category"] == "test"
    assert study["skipped_readouts"] == []
    assert {row["target"] for row in study["readouts"]} == {"next_manipulated_object"}
    assert phase_study["counts"]["readout_count"] == 1
    assert phase_study["counts"]["target_count"] == 1
    assert phase_study["readouts"][0]["target"] == "task_phase"
    assert phase_study["readouts"][0]["trained_probe_id"] == "TPH-L4-TEST-HELDOUT-TASK"
    assert phase_study["readouts"][0]["balanced_accuracy"] == 0.74
    assert phase_study["controls"] == []
    assert phase_study["error_examples"] == []
    assert study["controls"][0]["real_score"] == 0.62
    assert study["controls"][0]["p_value"] == 0.333
    assert study["error_examples"][0]["trace_id"] == trace_id

    episodes_response = client.get(
        f"/api/probe-studies/{saved.artifact_id}/episodes",
        params={
            "target": "next_manipulated_object",
            "layer": "4",
            "split": "test_heldout_task",
            "prediction": "incorrect",
            "limit": "5",
        },
    )
    episodes_payload = episodes_response.json()

    assert episodes_response.status_code == 200
    assert episodes_response.headers["Cache-Control"] == "no-store"
    assert episodes_payload["available"] is True
    assert episodes_payload["total"] == 1
    assert episodes_payload["episodes"][0]["trace_id"] == trace_id
    assert episodes_payload["episodes"][0]["probe_record"]["actual"] == "red_block_1"
    assert episodes_payload["episodes"][0]["probe_record"]["predicted"] == "blue_block_1"
    assert episodes_payload["episodes"][0]["probe_record"]["policy_call_index"] == 1
    assert episodes_payload["summary"]["episode_count"] == 1
    assert episodes_payload["summary"]["policy_call_count"] == 1
    assert episodes_payload["summary"]["wrong"] == 1
    assert episodes_payload["summary"]["high_conf_wrong"] == 1
    assert episodes_payload["summary"]["split_counts"]["test_heldout_task"]["wrong"] == 1
    assert episodes_payload["summary"]["split_counts"]["test_heldout_task"]["high_conf_wrong"] == 1

    unavailable = client.get(
        f"/api/probe-studies/{saved.artifact_id}/episodes",
        params={"target": "task_phase", "layer": "4", "split": "test_heldout_task"},
    ).json()

    assert unavailable["available"] is False
    assert unavailable["reason"] == (
        "Episode table rows were exported only for next_manipulated_object."
    )


def _write_index_table(
    root: Path,
    table_name: str,
    relative_path: Path,
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> None:
    out = frame.copy()
    for column in columns:
        if column not in out:
            out[column] = pd.Series(dtype=object)
    out = out.loc[:, list(columns)]
    (root / relative_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(root / relative_path, index=False)
    manifest_path = index_manifest_path(root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tables"][table_name] = {
        "path": str(relative_path),
        "rows": int(len(out)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
