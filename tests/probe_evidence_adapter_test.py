from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from vla_lens import create_synthetic_trace_dataset
from vla_lens.dataset.index import (
    ARTIFACT_COLUMNS,
    ARTIFACT_INDEX,
    EPISODE_INDEX,
    MODEL_SITE_COLUMNS,
    MODEL_SITE_INDEX,
    PROBE_EPISODE_COLUMNS,
    PROBE_EPISODE_INDEX,
    PROBE_PREDICTION_COLUMNS,
    PROBE_PREDICTIONS,
    index_manifest_path,
)
from vla_lens.probe_evidence import (
    FailureCaseEvidence,
    RankedMomentsEvidence,
    default_probe_panel_specs,
    primitives_by_kind,
    select_available_panels,
)
from vla_lens.probe_evidence_adapter import probe_evidence_bundle_from_index
from vla_lens.server.fastapi_app import create_dashboard_app


def test_indexed_probe_evidence_adapter_builds_valid_bundle(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=2)
    trace_ids = [bundle.manifest.trace_id for bundle in dataset.bundles]
    _write_probe_fixture(dataset.root, trace_ids, labeled=True)

    bundle = probe_evidence_bundle_from_index(dataset.root, "probe-a")

    assert bundle.family == "probe"
    assert bundle.artifact.lens_id == "probe-a"
    assert bundle.run.lens_run_id == "indexed:probe-a:demo"
    assert bundle.geometry.temporal_scope == "policy_call"
    assert bundle.geometry.output_kind == "scalar"
    assert bundle.geometry.input_basis == "pooled_layer_activation"
    assert bundle.geometry.locus_kind == "model_locus"
    assert set(bundle.capabilities) == {
        "score_series",
        "thresholding",
        "ranked_moments",
            "uncertainty",
            "prediction",
            "model_locus_view",
            "failure_cases",
        }

    assert len(primitives_by_kind(bundle, "score_series")) == 2
    assert len(primitives_by_kind(bundle, "prediction")) == 2
    assert len(primitives_by_kind(bundle, "model_locus")) == 2
    rankings = {
        primitive.ranking: primitive
        for primitive in primitives_by_kind(bundle, "ranked_moments")
        if isinstance(primitive, RankedMomentsEvidence)
    }
    assert rankings["top"].moments[0].episode_id == trace_ids[1]
    assert rankings["top"].moments[0].score == 0.91
    assert rankings["bottom"].moments[0].episode_id == trace_ids[0]
    assert rankings["uncertain"].moments[0].episode_id == trace_ids[0]
    assert "false_positive" not in rankings
    failure = primitives_by_kind(bundle, "failure_case")[0]
    assert isinstance(failure, FailureCaseEvidence)
    assert failure.ranking == "high_confidence_wrong"
    assert failure.moments[0].episode_id == trace_ids[1]

    panels = {
        item.panel_id: item
        for item in select_available_panels(bundle, default_probe_panel_specs())
    }
    assert panels["score_series"].available is True
    assert panels["ranked_moments"].available is True
    assert panels["prediction"].available is True
    assert panels["model_locus"].available is True
    assert panels["failure_cases"].available is True
    assert panels["contribution"].available is False
    assert panels["contribution"].reason == "pooled_representation"


def test_indexed_probe_evidence_adapter_reports_missing_labels(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    trace_ids = [bundle.manifest.trace_id for bundle in dataset.bundles]
    _write_probe_fixture(dataset.root, trace_ids, labeled=False)

    bundle = probe_evidence_bundle_from_index(dataset.root, "probe-a")

    reasons = {reason.capability: reason for reason in bundle.unavailable}
    assert reasons["failure_cases"].reason == "missing_labels"
    assert "no labels or proxy targets" in reasons["failure_cases"].message
    assert "failure_cases" not in bundle.capabilities
    panels = {
        item.panel_id: item
        for item in select_available_panels(bundle, default_probe_panel_specs())
    }
    assert panels["failure_cases"].available is False
    assert panels["failure_cases"].reason == "missing_labels"


def test_indexed_probe_evidence_adapter_reports_missing_model_locus(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    trace_ids = [bundle.manifest.trace_id for bundle in dataset.bundles]
    _write_probe_fixture(
        dataset.root,
        trace_ids,
        labeled=True,
        include_row_locus=False,
        include_artifact_source=False,
    )

    bundle = probe_evidence_bundle_from_index(dataset.root, "probe-a")

    assert bundle.geometry.locus_kind == "none"
    assert "model_locus_view" not in bundle.capabilities
    assert not primitives_by_kind(bundle, "model_locus")
    reasons = {reason.capability: reason for reason in bundle.unavailable}
    assert reasons["model_locus_view"].reason == "missing_model_locus"
    panels = {
        item.panel_id: item
        for item in select_available_panels(bundle, default_probe_panel_specs())
    }
    assert panels["model_locus"].available is False
    assert panels["model_locus"].reason == "missing_model_locus"


def test_indexed_probe_evidence_adapter_uses_artifact_source_model_locus(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    trace_ids = [bundle.manifest.trace_id for bundle in dataset.bundles]
    _write_probe_fixture(
        dataset.root,
        trace_ids,
        labeled=True,
        include_row_locus=False,
        include_artifact_source=True,
    )

    bundle = probe_evidence_bundle_from_index(dataset.root, "probe-a")

    assert bundle.geometry.locus_kind == "model_locus"
    assert "model_locus_view" in bundle.capabilities
    model_locus = primitives_by_kind(bundle, "model_locus")[0]
    assert model_locus.locus.model_site_id == "action_head.layers.0"


def test_indexed_probe_evidence_adapter_derives_correctness_when_possible(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=2)
    trace_ids = [bundle.manifest.trace_id for bundle in dataset.bundles]
    _write_probe_fixture(dataset.root, trace_ids, labeled=True, include_correct=False)

    bundle = probe_evidence_bundle_from_index(dataset.root, "probe-a")

    assert "failure_cases" in bundle.capabilities
    failure = primitives_by_kind(bundle, "failure_case")[0]
    assert failure.moments[0].episode_id == trace_ids[1]


def test_indexed_probe_evidence_adapter_ignores_partial_label_unknowns(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=2)
    trace_ids = [bundle.manifest.trace_id for bundle in dataset.bundles]
    _write_probe_fixture(dataset.root, trace_ids, labeled=True, include_correct=False)
    predictions = pd.read_parquet(dataset.root / PROBE_PREDICTIONS)
    predictions.loc[predictions["trace_id"].astype(str) == trace_ids[0], "prediction_value"] = None
    predictions.loc[predictions["trace_id"].astype(str) == trace_ids[0], "predicted"] = None
    predictions.to_parquet(dataset.root / PROBE_PREDICTIONS, index=False)

    bundle = probe_evidence_bundle_from_index(dataset.root, "probe-a")

    failure = primitives_by_kind(bundle, "failure_case")[0]
    assert [moment.episode_id for moment in failure.moments] == [trace_ids[1]]


def test_indexed_probe_evidence_adapter_reports_failure_not_computed(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    trace_ids = [bundle.manifest.trace_id for bundle in dataset.bundles]
    _write_probe_fixture(
        dataset.root,
        trace_ids,
        labeled=True,
        include_correct=False,
        include_prediction_labels=False,
    )

    bundle = probe_evidence_bundle_from_index(dataset.root, "probe-a")

    assert "failure_cases" not in bundle.capabilities
    assert not primitives_by_kind(bundle, "failure_case")
    reasons = {reason.capability: reason for reason in bundle.unavailable}
    assert reasons["failure_cases"].reason == "not_computed"


def test_indexed_probe_evidence_adapter_supports_layer_only_source_locus(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    trace_ids = [bundle.manifest.trace_id for bundle in dataset.bundles]
    _write_probe_fixture(
        dataset.root,
        trace_ids,
        labeled=True,
        include_row_locus=False,
        include_artifact_source=True,
        include_metric_feature=False,
        include_source_module=False,
    )

    bundle = probe_evidence_bundle_from_index(dataset.root, "probe-a")

    assert bundle.geometry.locus_kind == "model_locus"
    assert "model_locus_view" in bundle.capabilities
    model_locus = primitives_by_kind(bundle, "model_locus")[0]
    assert model_locus.locus.layer == 0
    assert model_locus.locus.model_site_id is None
    assert model_locus.source_label == "Layer 0"


def test_fastapi_probe_evidence_bundle_route_returns_canonical_payload(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=2)
    trace_ids = [bundle.manifest.trace_id for bundle in dataset.bundles]
    _write_probe_fixture(dataset.root, trace_ids, labeled=True)
    client = TestClient(create_dashboard_app(dataset.root))

    response = client.get("/api/probes/probe-a/evidence-bundle", params={"limit": "1"})
    payload = response.json()

    assert response.status_code == 200
    assert payload["family"] == "probe"
    assert payload["artifact"]["lens_id"] == "probe-a"
    assert payload["run"]["lens_run_id"] == "indexed:probe-a:demo"
    assert payload["geometry"]["temporal_scope"] == "policy_call"
    ranked = [item for item in payload["primitives"] if item["kind"] == "ranked_moments"]
    assert {item["ranking"] for item in ranked} >= {"top", "bottom", "uncertain"}
    assert len(next(item for item in ranked if item["ranking"] == "top")["moments"]) == 1
    assert any(
        reason["capability"] == "contribution_breakdown"
        for reason in payload["unavailable"]
    )


def test_indexed_probe_evidence_adapter_filters_bundle_to_selected_dataset(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=2)
    trace_ids = [bundle.manifest.trace_id for bundle in dataset.bundles]
    _write_probe_fixture(dataset.root, trace_ids, labeled=True)
    _set_episode_dataset_ids(dataset.root, {trace_ids[0]: "dataset-a", trace_ids[1]: "dataset-b"})

    bundle = probe_evidence_bundle_from_index(dataset.root, "probe-a", dataset_id="dataset-a")

    assert bundle.run.dataset_id == "dataset-a"
    assert bundle.run.lens_run_id == "indexed:probe-a:dataset-a"
    assert bundle.run.episode_ids == (trace_ids[0],)
    prediction = primitives_by_kind(bundle, "prediction")[0]
    assert prediction.episode_id == trace_ids[0]
    ranked = [
        moment.episode_id
        for primitive in primitives_by_kind(bundle, "ranked_moments")
        if isinstance(primitive, RankedMomentsEvidence)
        for moment in primitive.moments
    ]
    assert set(ranked) == {trace_ids[0]}


def test_fastapi_probe_evidence_bundle_route_honors_dataset_id_query(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=2)
    trace_ids = [bundle.manifest.trace_id for bundle in dataset.bundles]
    _write_probe_fixture(dataset.root, trace_ids, labeled=True)
    _set_episode_dataset_ids(dataset.root, {trace_ids[0]: "dataset-a", trace_ids[1]: "dataset-b"})
    client = TestClient(create_dashboard_app(dataset.root))

    response = client.get(
        "/api/probes/probe-a/evidence-bundle",
        params={"dataset_id": "dataset-b", "limit": "5"},
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["run"]["dataset_id"] == "dataset-b"
    assert payload["run"]["lens_run_id"] == "indexed:probe-a:dataset-b"
    assert payload["run"]["episode_ids"] == [trace_ids[1]]
    predictions = [item for item in payload["primitives"] if item["kind"] == "prediction"]
    assert [item["episode_id"] for item in predictions] == [trace_ids[1]]


def _write_probe_fixture(
    root: Path,
    trace_ids: list[str],
    *,
    labeled: bool,
    include_row_locus: bool = True,
    include_artifact_source: bool = True,
    include_metric_feature: bool = True,
    include_source_module: bool = True,
    include_correct: bool = True,
    include_prediction_labels: bool = True,
) -> None:
    metrics = {
        "target": "outcome",
        "best_model": "linear",
        "best_score": 0.8,
        "best_delta": 0.2,
    }
    if include_artifact_source and include_metric_feature:
        metrics["best_feature"] = "action_head.layers.0"
    method = {
        "split": {"selection_value": "validation"},
    }
    if include_artifact_source:
        selector = {
            "layers": [0],
            "reduce_tokens": "mean",
            "tensor_type": "residual",
        }
        if include_source_module:
            selector["module"] = "action_head"
        method["input"] = {
            "selector": selector
        }
    _write_index_table(
        root,
        "artifact_index",
        ARTIFACT_INDEX,
        pd.DataFrame(
            {
                "artifact_id": ["probe-a"],
                "artifact_type": ["probe_suite"],
                "name": ["Probe A"],
                "created_utc": ["2026-06-09T00:00:00+00:00"],
                "metrics": [json.dumps(metrics)],
                "method": [json.dumps(method)],
                "display": [json.dumps({"target": "outcome"})],
                "arrays": [json.dumps({})],
            }
        ),
        ARTIFACT_COLUMNS,
    )
    _write_index_table(
        root,
        "model_site_index",
        MODEL_SITE_INDEX,
        pd.DataFrame(
            {
                "trace_id": [trace_ids[0]],
                "episode_id": ["episode-0"],
                "site_id": ["action_head.layers.0"],
                "name": ["action_head.layers.0"],
                "module": ["action_head"],
                "layer": [0],
                "tensor_type": ["residual"],
            }
        ),
        MODEL_SITE_COLUMNS,
    )
    frame = pd.DataFrame(
        {
            "probe_id": ["probe-a" for _ in trace_ids],
            "probe_name": ["Probe A" for _ in trace_ids],
            "target": ["outcome" for _ in trace_ids],
            "trace_id": trace_ids,
            "episode_id": [f"episode-{idx}" for idx, _ in enumerate(trace_ids)],
            "split": ["validation" for _ in trace_ids],
            "split_category": ["validation" for _ in trace_ids],
            "actual": (
                ["success", "success"][: len(trace_ids)]
                if labeled
                else [None] * len(trace_ids)
            ),
            "predicted": (
                ["success", "failure"][: len(trace_ids)]
                if include_prediction_labels
                else [None] * len(trace_ids)
            ),
            "target_value": (
                ["success", "success"][: len(trace_ids)]
                if labeled
                else [None] * len(trace_ids)
            ),
            "prediction_value": (
                ["success", "failure"][: len(trace_ids)]
                if include_prediction_labels
                else [None] * len(trace_ids)
            ),
            "confidence": [0.54, 0.91][: len(trace_ids)],
            "correct": (
                [True, False][: len(trace_ids)]
                if labeled and include_correct
                else [None] * len(trace_ids)
            ),
            "model": ["linear" for _ in trace_ids],
            "layer": [0 for _ in trace_ids],
            "policy_call_index": [1, 4][: len(trace_ids)],
            "timestep": [2, 8][: len(trace_ids)],
            "primary_metric": ["balanced_accuracy" for _ in trace_ids],
        }
    )
    if include_row_locus:
        frame["feature"] = ["action_head.layers.0" for _ in trace_ids]
        frame["model_site_id"] = ["action_head.layers.0" for _ in trace_ids]
    _write_index_table(
        root,
        "probe_predictions",
        PROBE_PREDICTIONS,
        frame,
        PROBE_PREDICTION_COLUMNS,
    )
    episode_frame = frame.copy()
    episode_frame["row_count"] = 1
    episode_frame["aggregate_policy"] = "representative"
    episode_frame["representative_rank"] = list(range(len(episode_frame)))
    _write_index_table(
        root,
        "probe_episode_index",
        PROBE_EPISODE_INDEX,
        episode_frame,
        PROBE_EPISODE_COLUMNS,
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


def _set_episode_dataset_ids(root: Path, mapping: dict[str, str]) -> None:
    episodes = pd.read_parquet(root / EPISODE_INDEX)
    for trace_id, dataset_id in mapping.items():
        episodes.loc[episodes["trace_id"].astype(str) == trace_id, "dataset_id"] = dataset_id
    episodes.to_parquet(root / EPISODE_INDEX, index=False)
