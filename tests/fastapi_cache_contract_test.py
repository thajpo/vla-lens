from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import vla_lens.server.state as state_api
from vla_lens import create_synthetic_trace_dataset
from vla_lens.server.fastapi_app import create_dashboard_app


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
        "/api/evidence-pins",
        "/api/artifacts",
        "/api/probe-studies",
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


def test_dashboard_state_cached_payload_reads_signature_inside_lock(tmp_path):
    state = state_api.DashboardState.__new__(state_api.DashboardState)
    state.root = tmp_path
    state.dataset = object()
    state.dataset_signature = (1, 1)
    state.payload_cache = {"dataset": ((1, 1), {"stale": True})}
    state.dataset_lock = _MutatingLock(state, signature=(2, 2))

    payload = state.cached_payload(
        "dataset",
        lambda dataset: {"fresh": dataset is state.dataset},
    )

    assert payload == {"fresh": True}
    assert state.payload_cache["dataset"] == ((2, 2), {"fresh": True})


class _MutatingLock:
    def __init__(self, state: state_api.DashboardState, *, signature: tuple[int, int]):
        self.state = state
        self.signature = signature

    def __enter__(self):
        self.state.dataset_signature = self.signature
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False
