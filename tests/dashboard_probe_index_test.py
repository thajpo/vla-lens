from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from vla_lens import create_synthetic_trace_dataset
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
