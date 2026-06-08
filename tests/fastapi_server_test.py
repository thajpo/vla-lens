from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import vla_lens.server.indexed as indexed_api
import vla_lens.server.state as state_api
from vla_lens import create_synthetic_trace_dataset
from vla_lens.dataset import DatasetIndexError
from vla_lens.dataset.index import (
    index_manifest_path,
)
from vla_lens.server import _dataset_signature
from vla_lens.server.fastapi_app import create_dashboard_app

EXPECTED_DASHBOARD_ROUTE_METHODS = {
    ("/", "get"),
    ("/api/health", "get"),
    ("/api/dataset", "get"),
    ("/api/counterfactual-pairs", "get"),
    ("/api/observational-comparisons", "get"),
    ("/api/workbench", "get"),
    ("/api/workbench/validate", "get"),
    ("/api/spatial-overlays", "get"),
    ("/api/lens-arrays", "get"),
    ("/api/lens-arrays/{array_id}", "get"),
    ("/api/cohorts", "get"),
    ("/api/analysis-runs", "get"),
    ("/api/workspaces", "get"),
    ("/api/workspaces/{workspace_id}/resolve", "get"),
    ("/api/intervention-runs", "get"),
    ("/api/intervention-runs/{run_id}", "get"),
    ("/api/unit-profile", "get"),
    ("/api/dataset-diagnostics", "get"),
    ("/api/episode-annotations", "get"),
    ("/api/dataset-diagnostics/run", "get"),
    ("/api/artifacts", "get"),
    ("/api/artifacts/{artifact_id}", "get"),
    ("/api/discovery-artifact-families", "get"),
    ("/api/discovery-artifacts/{artifact_id}/episodes", "get"),
    ("/api/discovery-artifacts/{artifact_id}/readout", "get"),
    ("/api/discovery-artifacts/{artifact_id}/episode-lens-view", "get"),
    ("/api/discovery-artifacts/{artifact_id}/target", "get"),
    ("/api/episodes", "get"),
    ("/api/episodes/{trace_id}", "get"),
    ("/api/episodes/{trace_id}/neighbors", "get"),
    ("/api/frame", "get"),
    ("/api/episode-video", "get"),
    ("/api/policy-calls", "get"),
    ("/api/action-norm", "get"),
    ("/api/generation-commitment", "get"),
    ("/api/episode-metrics", "get"),
    ("/api/episode-interactions", "get"),
    ("/api/episode-probes", "get"),
    ("/api/probe-index", "get"),
    ("/api/probes/{probe_id}/evidence", "get"),
    ("/api/activation-sites", "get"),
    ("/api/activation-slice", "get"),
    ("/api/image-token-map", "get"),
    ("/api/object-camera-overlay", "get"),
    ("/api/attention-map", "get"),
    ("/api/patch-features", "get"),
    ("/api/prompt-attention", "get"),
    ("/api/prompt-feature-map", "get"),
    ("/api/expert-token-activations", "get"),
    ("/api/expert-token-details", "get"),
    ("/api/dataset-diagnostics/run", "post"),
    ("/api/episode-annotations", "post"),
    ("/api/selections/resolve", "post"),
    ("/api/cohorts", "post"),
    ("/api/cohorts/from-selection", "post"),
    ("/api/cohorts/compare", "post"),
    ("/api/analysis-runs", "post"),
    ("/api/intervention-runs", "post"),
    ("/api/interventions/preflight", "post"),
    ("/api/workspaces", "post"),
    ("/api/projection", "post"),
    ("/api/graph", "post"),
    ("/api/tables/query", "post"),
    ("/api/lens-arrays/{array_id}/slice", "post"),
    ("/api/artifacts/create/outcome-probe", "post"),
    ("/api/artifacts/create/target-object-probe", "post"),
    ("/api/artifacts/create/action-generation", "post"),
}


def test_fastapi_dataset_payload_reports_model_sites_without_workbench_payload(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    client = TestClient(create_dashboard_app(dataset.root))

    response = client.get("/api/dataset")
    payload = response.json()

    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("application/json")
    assert response.headers["Cache-Control"] == "private, max-age=2"
    assert "episodes" not in payload
    assert payload["episode_count"] == 1
    assert payload["activation_sites"] == len(dataset.model_site_index)
    assert payload["capabilities"]["flags"]["robot_episodes"] is True
    assert payload["capabilities"]["flags"]["model_sites"] is True
    assert payload["capabilities"]["flags"]["action_generation"] is True
    assert payload["capabilities"]["model_site_prefixes"] == ["action_head", "backbone"]
    assert "workbench" not in payload


def test_fastapi_episode_page_payload_keeps_metadata_bounded(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    client = TestClient(create_dashboard_app(dataset.root))

    payload = client.get("/api/episodes?limit=1").json()
    metadata = payload["episodes"][0]["metadata"]

    assert set(metadata) <= {"benchmark", "capture_profile", "dataset_id", "profile", "seed"}


def test_fastapi_indexed_pages_do_not_revalidate_index_on_request(tmp_path, monkeypatch):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    app = create_dashboard_app(dataset.root)
    app.state.dashboard.dataset_signature_checked_at = 0.0
    app.state.dashboard.dataset_signature_check_interval_s = -1.0

    def fail_request_validation(_root):
        raise AssertionError("indexed endpoints should trust startup index validation")

    monkeypatch.setattr(state_api, "validate_dataset_index", fail_request_validation)
    monkeypatch.setattr(
        indexed_api,
        "validate_dataset_index",
        fail_request_validation,
        raising=False,
    )

    response = TestClient(app).get("/api/episodes?limit=1")

    assert response.status_code == 200
    assert response.json()["total"] == 1


def test_dataset_signature_uses_manifest_count_without_recursive_episode_ref_scan(
    tmp_path,
    monkeypatch,
):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=2)
    original_glob = Path.glob

    def guarded_glob(self, pattern):
        if str(pattern).startswith("**/"):
            raise AssertionError("dataset signatures must not recursively scan episode refs")
        return original_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", guarded_glob)

    assert _dataset_signature(dataset.root)[0] == 2


def test_fastapi_startup_requires_dataset_index(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    index_manifest_path(dataset.root).unlink()

    with pytest.raises(DatasetIndexError) as exc_info:
        create_dashboard_app(dataset.root)

    message = str(exc_info.value)
    assert "Dataset index is missing" in message
    assert "scripts/build_vla_lens_index.py" in message


def test_fastapi_health_endpoint_is_gateway_readiness_probe(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    client = TestClient(create_dashboard_app(dataset.root))

    response = client.get("/api/health")
    payload = response.json()

    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("application/json")
    assert response.headers["Cache-Control"] == "no-store"
    assert payload["status"] == "ok"
    assert payload["api"] == "/api/dataset"
    assert payload["dataset"]["episodes"] == 1
    assert payload["dataset"]["activation_sites"] == len(dataset.model_site_index)


def test_fastapi_client_errors_match_json_contract(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    client = TestClient(create_dashboard_app(dataset.root))

    response = client.get("/api/frame")
    payload = response.json()

    assert response.status_code == 400
    assert response.headers["Content-Type"].startswith("application/json")
    assert response.headers["Cache-Control"] == "no-store"
    assert payload["message"] == "Missing query parameter: trace_id"


def test_fastapi_blank_query_values_match_legacy_missing_param(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    client = TestClient(create_dashboard_app(dataset.root))

    response = client.get("/api/frame?trace_id=&camera=main&timestep=0&source=trace")
    payload = response.json()

    assert response.status_code == 400
    assert response.headers["Content-Type"].startswith("application/json")
    assert response.headers["Cache-Control"] == "no-store"
    assert payload["message"] == "Missing query parameter: trace_id"


def test_fastapi_unknown_api_root_uses_dashboard_error_json(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    client = TestClient(create_dashboard_app(dataset.root))

    response = client.get("/api")
    payload = response.json()

    assert response.status_code == 404
    assert response.headers["Content-Type"].startswith("application/json")
    assert response.headers["Cache-Control"] == "no-store"
    assert payload == {"error": "Not Found", "message": "Unknown route: /api"}


def test_fastapi_trailing_slash_routes_keep_error_json_contract(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    client = TestClient(create_dashboard_app(dataset.root), follow_redirects=False)

    response = client.get("/api/dataset/")
    payload = response.json()

    assert response.status_code == 404
    assert "location" not in response.headers
    assert response.headers["Content-Type"].startswith("application/json")
    assert response.headers["Cache-Control"] == "no-store"
    assert payload == {"error": "Not Found", "message": "Unknown route: /api/dataset/"}


def test_fastapi_artifact_ids_cannot_escape_dataset_root(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    client = TestClient(create_dashboard_app(dataset.root))

    response = client.get("/api/artifacts/..%2F..%2Fescape")
    payload = response.json()

    assert response.status_code == 400
    assert response.headers["Content-Type"].startswith("application/json")
    assert "Invalid artifact_id" in payload["message"]


def test_fastapi_frame_endpoint_returns_jpeg(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    trace_id = dataset.bundles[0].manifest.trace_id
    client = TestClient(create_dashboard_app(dataset.root))

    response = client.get(
        "/api/frame",
        params={
            "trace_id": trace_id,
            "camera": "main",
            "timestep": "0",
            "source": "trace",
            "v": "test-fingerprint",
        },
    )

    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("image/jpeg")
    assert response.headers["Cache-Control"] == "public, max-age=31536000, immutable"
    assert response.content.startswith(b"\xff\xd8")


def test_fastapi_post_clears_cached_workbench_payload(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    client = TestClient(create_dashboard_app(dataset.root))

    before = client.get("/api/workbench").json()
    assert before["saved_workspaces"] == []

    response = client.post(
        "/api/workspaces",
        json={
            "workspace_id": "regression_workspace",
            "dataset_id": "demo",
            "panels": [],
        },
    )
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"

    after = client.get("/api/workbench").json()
    assert [workspace["workspace_id"] for workspace in after["saved_workspaces"]] == [
        "regression_workspace"
    ]


def test_fastapi_openapi_matches_dashboard_get_post_route_surface(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    client = TestClient(create_dashboard_app(dataset.root))

    response = client.get("/api/openapi.json")
    payload = response.json()
    route_methods = {
        (path, method)
        for path, methods in payload["paths"].items()
        for method in methods
        if method in {"get", "post"}
    }

    assert response.status_code == 200
    assert route_methods == EXPECTED_DASHBOARD_ROUTE_METHODS


def test_fastapi_representative_get_routes_return_legacy_payload_shapes(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=3)
    client = TestClient(create_dashboard_app(dataset.root))
    trace_id = dataset.bundles[0].manifest.trace_id
    site_name = str(dataset.model_site_index.iloc[0]["name"])
    array_id = client.get("/api/lens-arrays").json()["lens_arrays"][0]["array_id"]
    artifact_id = client.get("/api/artifacts").json()["artifacts"][0]["artifact_id"]

    client.post(
        "/api/workspaces",
        json={"workspace_id": "shape_workspace", "dataset_id": "demo", "panels": []},
    )

    cases = [
        ("/", {}, {"service", "api"}),
        ("/api/counterfactual-pairs", {}, {"pairs", "count"}),
        (
            "/api/observational-comparisons",
            {"trace_id": trace_id, "limit": "1"},
            {"source_trace_id", "candidates", "total_candidates"},
        ),
        ("/api/workbench/validate", {}, {"valid", "trace_validation"}),
        ("/api/spatial-overlays", {}, {"overlays"}),
        (f"/api/lens-arrays/{array_id}", {}, {"array_id", "dims", "storage"}),
        ("/api/unit-profile", {"unit": "0"}, {"unit_ref", "top_examples", "lens_arrays"}),
        ("/api/dataset-diagnostics", {}, {"fingerprint", "stale", "latest"}),
        ("/api/episode-annotations", {"trace_id": trace_id}, {"annotation"}),
        ("/api/dataset-diagnostics/run", {}, {"fingerprint", "artifact"}),
        (f"/api/artifacts/{artifact_id}", {}, {"artifact", "arrays"}),
        ("/api/episodes", {"limit": "1"}, {"episodes", "total", "facets", "next_offset"}),
        (f"/api/episodes/{trace_id}", {}, {"trace_id", "cameras", "arrays"}),
        (
            f"/api/episodes/{trace_id}/neighbors",
            {},
            {"trace_id", "previous_trace_id", "next_trace_id"},
        ),
        ("/api/policy-calls", {"trace_id": trace_id}, {"calls", "count"}),
        ("/api/action-norm", {"trace_id": trace_id}, {"values"}),
        ("/api/generation-commitment", {"trace_id": trace_id}, {"values"}),
        ("/api/episode-metrics", {"trace_id": trace_id}, {"domains", "metrics"}),
        ("/api/episode-interactions", {"trace_id": trace_id}, {"trace_id", "objects"}),
        ("/api/episode-probes", {"trace_id": trace_id}, {"trace_id", "probes"}),
        ("/api/probe-index", {}, {"probes", "total", "trace_count"}),
        ("/api/activation-sites", {"trace_id": trace_id}, {"sites", "architecture"}),
        (
            "/api/activation-slice",
            {"trace_id": trace_id, "name": site_name, "call_index": "0"},
            {"name", "shape", "feature_count"},
        ),
        (
            "/api/image-token-map",
            {"trace_id": trace_id, "name": site_name, "call_index": "0"},
            {"available", "name", "token_kind"},
        ),
        (
            "/api/object-camera-overlay",
            {"trace_id": trace_id, "camera": "main", "timestep": "0"},
            {"available", "camera", "objects"},
        ),
        (
            "/api/attention-map",
            {"trace_id": trace_id, "call_index": "0", "camera": "main"},
            {"available", "kind", "query_token"},
        ),
        (
            "/api/patch-features",
            {
                "trace_id": trace_id,
                "name": site_name,
                "call_index": "0",
                "camera": "main",
                "row": "0",
                "col": "0",
            },
            {"available", "name", "patch_row", "patch_col"},
        ),
        (
            "/api/prompt-attention",
            {"trace_id": trace_id, "call_index": "0"},
            {"available", "kind", "prompt"},
        ),
        (
            "/api/prompt-feature-map",
            {"trace_id": trace_id, "name": site_name, "call_index": "0"},
            {"available", "name", "token_space_id"},
        ),
        (
            "/api/expert-token-activations",
            {"trace_id": trace_id, "name": site_name, "call_index": "0"},
            {"available", "name", "values"},
        ),
        (
            "/api/expert-token-details",
            {"trace_id": trace_id, "name": site_name, "call_index": "0"},
            {"available", "name", "token_index"},
        ),
        ("/api/cohorts", {}, {"cohorts", "total"}),
        ("/api/analysis-runs", {}, {"analysis_runs", "total"}),
        ("/api/workspaces", {}, {"workspaces", "total"}),
        (
            "/api/workspaces/shape_workspace/resolve",
            {},
            {"workspace", "resolved_selection", "panel_registry"},
        ),
        ("/api/intervention-runs", {}, {"intervention_runs", "total"}),
    ]

    for path, params, keys in cases:
        response = client.get(path, params=params)
        payload = response.json()

        assert response.status_code == 200, path
        assert response.headers["Content-Type"].startswith("application/json"), path
        assert keys <= set(payload), path


def test_fastapi_representative_post_routes_return_legacy_payload_shapes(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=2, timesteps=3)
    client = TestClient(create_dashboard_app(dataset.root))
    trace_id = dataset.bundles[0].manifest.trace_id
    lens_arrays = client.get("/api/lens-arrays").json()["lens_arrays"]
    array_id = next(array["array_id"] for array in lens_arrays if array["kind"] == "tensor")
    artifacts = client.get("/api/artifacts").json()["artifacts"]
    source_artifact_id = next(
        artifact["artifact_id"]
        for artifact in artifacts
        if artifact["artifact_type"] == "probe_suite"
    )
    selection = {"selection_id": "episode_selection", "axis_values": {"episode": [trace_id]}}

    cases: list[tuple[str, dict[str, Any], set[str]]] = [
        (
            "/api/episode-annotations",
            {"trace_id": trace_id, "starred": True, "notes": "review"},
            {"annotation"},
        ),
        ("/api/selections/resolve", selection, {"selection", "episodes", "lens_arrays"}),
        (
            "/api/cohorts",
            {"cohort_id": "left", "label": "Left", "members": {"trace_id": [trace_id]}},
            {"cohort", "cohorts", "total"},
        ),
        (
            "/api/cohorts/from-selection",
            {"selection": selection, "cohort_id": "right", "label": "Right"},
            {"cohort", "cohorts", "total"},
        ),
        ("/api/cohorts/compare", {"left": "left", "right": "right"}, {"left", "right", "summary"}),
        (
            "/api/analysis-runs",
            {"run_id": "analysis_1", "workflow": "pytest", "inputs": {}, "outputs": []},
            {"analysis_run", "analysis_runs", "total"},
        ),
        (
            "/api/intervention-runs",
            {
                "run_id": "intervention_1",
                "intervention_type": "intervention_record",
                "target": {},
            },
            {"intervention_run", "intervention_runs", "total"},
        ),
        (
            "/api/interventions/preflight",
            {
                "runtime_adapter": "synthetic",
                "target": {
                    "kind": "probe_direction",
                    "source_artifact_id": source_artifact_id,
                    "source_artifact_type": "probe_suite",
                    "model_site": "action_head.layers.0.resid",
                    "token_space": "synthetic.action_suffix",
                    "model_family": "synthetic",
                },
                "baseline": {
                    "context": {
                        "trace_id": trace_id,
                        "policy_call_index": 0,
                    },
                },
                "intervention": {
                    "request": {
                        "schedule": {"policy_calls": [0], "tokens": "action"},
                        "outcome": {"kind": "action", "basis": ["raw", "gripper"]},
                    },
                },
            },
            {"preflight"},
        ),
        (
            "/api/workspaces",
            {"workspace_id": "workspace_1", "dataset_id": "demo", "panels": []},
            {"workspace", "workspaces", "total"},
        ),
        ("/api/projection", {"selection": selection, "limit": 3}, {"selection", "points"}),
        ("/api/graph", selection, {"selection", "nodes", "edges"}),
        ("/api/tables/query", {"table": "episodes", "limit": 1}, {"table", "rows", "total"}),
        (
            f"/api/lens-arrays/{array_id}/slice",
            {"selection": {"timestep": 0}, "max_values": 10},
            {"array", "selection", "shape", "values"},
        ),
        ("/api/dataset-diagnostics/run", {}, {"fingerprint", "artifact"}),
        ("/api/artifacts/create/action-generation", {}, {"artifact", "artifacts"}),
    ]

    for path, body, keys in cases:
        response = client.post(path, json=body)
        payload = response.json()

        assert response.status_code == 200, path
        assert response.headers["Content-Type"].startswith("application/json"), path
        assert response.headers["Cache-Control"] == "no-store", path
        assert keys <= set(payload), path


def test_fastapi_intervention_run_detail_route_saves_lists_and_reloads(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    body = {
        "run_id": "detail_intervention",
        "intervention_type": "intervention_record",
        "target": {
            "kind": "manual",
            "model_site": "pi05.expert.layers.12.hidden_tokens",
        },
        "baseline": {
            "context": {
                "dataset_id": "demo",
                "dataset_fingerprint": "fingerprint-1",
                "trace_id": "trace-1",
                "policy_call_index": 0,
            }
        },
        "intervention": {
            "request": {
                "operator": {"operator": "add_direction", "strength": 1.0},
                "schedule": {"policy_calls": [0], "tokens": "action"},
                "outcome": {"kind": "action", "basis": ["raw"]},
            }
        },
        "readouts": {
            "status": "inspected_only",
            "title": "Saved inspected record",
            "preflight": {"status": "inspected_only"},
            "trials": [],
            "outcomes": [],
            "claim": {"claim_strength": ["observation"]},
        },
        "outputs": ["array://stored-original"],
        "provenance": {
            "schema_kind": "vla_lens.intervention_run",
            "schema_version": "0.1.0",
            "dataset_id": "demo",
            "dataset_fingerprint": "fingerprint-1",
            "trace_id": "trace-1",
            "policy_call_index": 0,
            "created_utc": "2026-06-06T00:00:00+00:00",
        },
    }
    client = TestClient(create_dashboard_app(dataset.root))

    saved = client.post("/api/intervention-runs", json=body)
    listed = client.get("/api/intervention-runs")
    opened = client.get("/api/intervention-runs/detail_intervention")
    restarted = TestClient(create_dashboard_app(dataset.root))
    reopened = restarted.get("/api/intervention-runs/detail_intervention")
    missing = restarted.get("/api/intervention-runs/missing")

    assert saved.status_code == 200
    assert listed.status_code == 200
    assert opened.status_code == 200
    assert reopened.status_code == 200
    assert missing.status_code == 404
    assert listed.json()["total"] == 1
    detail = opened.json()["intervention_run"]
    assert detail["run_id"] == "detail_intervention"
    assert detail["intervention_type"] == "intervention_record"
    assert detail["baseline"]["context"]["trace_id"] == "trace-1"
    assert detail["target"]["kind"] == "manual"
    assert detail["intervention"]["request"]["operator"]["operator"] == "add_direction"
    assert detail["readouts"]["status"] == "inspected_only"
    assert detail["readouts"]["claim"]["claim_strength"] == ["observation"]
    assert detail["provenance"]["schema_kind"] == "vla_lens.intervention_run"
    assert reopened.json()["intervention_run"] == detail


@pytest.mark.parametrize(
    "path",
    [
        "/api/workbench",
        "/api/workbench/validate",
        "/api/cohorts",
        "/api/analysis-runs",
        "/api/workspaces",
        "/api/intervention-runs",
        "/api/dataset-diagnostics",
        "/api/episode-annotations",
        "/api/artifacts",
    ],
)
def test_fastapi_mutable_read_routes_are_no_store(tmp_path, path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    client = TestClient(create_dashboard_app(dataset.root))

    response = client.get(path)

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"


def test_fastapi_mutable_read_detail_routes_are_no_store(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    client = TestClient(create_dashboard_app(dataset.root))
    artifact_id = client.get("/api/artifacts").json()["artifacts"][0]["artifact_id"]
    client.post(
        "/api/workspaces",
        json={"workspace_id": "cache_workspace", "dataset_id": "demo", "panels": []},
    )
    client.post(
        "/api/intervention-runs",
        json={
            "run_id": "cache_intervention",
            "intervention_type": "intervention_record",
            "target": {"kind": "manual"},
            "readouts": {"status": "inspected_only"},
        },
    )

    artifact_response = client.get(f"/api/artifacts/{artifact_id}")
    workspace_response = client.get("/api/workspaces/cache_workspace/resolve")
    intervention_response = client.get("/api/intervention-runs/cache_intervention")

    assert artifact_response.status_code == 200
    assert artifact_response.headers["Cache-Control"] == "no-store"
    assert workspace_response.status_code == 200
    assert workspace_response.headers["Cache-Control"] == "no-store"
    assert intervention_response.status_code == 200
    assert intervention_response.headers["Cache-Control"] == "no-store"


def test_fastapi_openapi_includes_dashboard_routes(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    client = TestClient(create_dashboard_app(dataset.root))

    response = client.get("/api/openapi.json")
    payload = response.json()

    assert response.status_code == 200
    assert "/api/health" in payload["paths"]
    assert "/api/dataset" in payload["paths"]
    assert "/api/lens-arrays/{array_id}/slice" in payload["paths"]
