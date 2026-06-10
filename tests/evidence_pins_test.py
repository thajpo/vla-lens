from __future__ import annotations

from fastapi.testclient import TestClient

from vla_lens import create_synthetic_trace_dataset
from vla_lens.server.fastapi_app import create_dashboard_app


def test_evidence_pins_persist_research_selection(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    client = TestClient(create_dashboard_app(dataset.root))
    trace_id = dataset.bundles[0].manifest.trace_id

    response = client.post(
        "/api/evidence-pins",
        json={
            "label": "Target contacted evidence",
            "note": "review this moment",
            "selection": {
                "dataset_id": "demo",
                "episode_id": trace_id,
                "lens_id": "probe-target-contacted",
                "lens_run_id": "run-probe-target-contacted",
                "ranking": "top",
                "timestep": 7,
            },
            "evidence": {
                "primitive_kind": "prediction",
                "score": 0.91,
                "prediction": True,
                "model_site_id": "action_head.layers.8.resid",
                "selected_contributor": "dim_42",
            },
        },
    )

    assert response.status_code == 200
    pin = response.json()["pin"]
    assert pin["selection"]["episode_id"] == trace_id
    assert pin["selection"]["lens_run_id"] == "run-probe-target-contacted"
    assert pin["evidence"]["primitive_kind"] == "prediction"
    assert pin["evidence"]["selected_contributor"] == "dim_42"
    assert pin["note"] == "review this moment"
    assert client.get("/api/evidence-pins").json()["pins"] == [pin]


def test_evidence_pin_requires_episode_research_state(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    client = TestClient(create_dashboard_app(dataset.root))

    response = client.post(
        "/api/evidence-pins",
        json={
            "label": "bad",
            "selection": {
                "dataset_id": "demo",
                "lens_id": "probe-a",
                "lens_run_id": "run-a",
                "policy_call": 0,
            },
            "evidence": {
                "model_site_id": "action_head.layers.8.resid",
                "primitive_kind": "prediction",
            },
        },
    )

    assert response.status_code == 400
    assert "selection.episode_id" in response.json()["message"]


def test_evidence_pin_rejects_episode_bookmark_without_evidence_ref(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    client = TestClient(create_dashboard_app(dataset.root))
    trace_id = dataset.bundles[0].manifest.trace_id

    response = client.post(
        "/api/evidence-pins",
        json={
            "label": "bad",
            "selection": {
                "dataset_id": "demo",
                "episode_id": trace_id,
                "lens_id": "probe-a",
                "lens_run_id": "run-a",
            },
            "evidence": {"primitive_kind": "prediction"},
        },
    )

    assert response.status_code == 400
    assert "selection.timestep or selection.policy_call" in response.json()["message"]

    response = client.post(
        "/api/evidence-pins",
        json={
            "label": "bad",
            "selection": {
                "dataset_id": "demo",
                "episode_id": trace_id,
                "lens_id": "probe-a",
                "lens_run_id": "run-a",
                "policy_call": 0,
            },
            "evidence": {"model_site_id": "action_head.layers.8.resid"},
        },
    )

    assert response.status_code == 400
    assert "evidence.primitive_kind" in response.json()["message"]

    response = client.post(
        "/api/evidence-pins",
        json={
            "label": "bad",
            "selection": {
                "dataset_id": "demo",
                "episode_id": trace_id,
                "lens_id": "probe-a",
                "lens_run_id": "run-a",
                "policy_call": 0,
            },
            "evidence": {"primitive_kind": "prediction"},
        },
    )

    assert response.status_code == 400
    assert "evidence.model_site_id" in response.json()["message"]


def test_evidence_pins_preserve_review_categories(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    client = TestClient(create_dashboard_app(dataset.root))
    trace_id = dataset.bundles[0].manifest.trace_id
    cases = [
        ("top", "ranked_moments"),
        ("bottom", "ranked_moments"),
        ("uncertain", "ranked_moments"),
        ("false_positive", "failure_case"),
        ("false_negative", "failure_case"),
        (None, "manual"),
    ]

    for index, (ranking, primitive_kind) in enumerate(cases):
        response = client.post(
            "/api/evidence-pins",
            json={
                "label": primitive_kind,
                "selection": {
                    "dataset_id": "demo",
                    "episode_id": trace_id,
                    "lens_id": "probe-a",
                    "lens_run_id": "run-a",
                    "policy_call": index,
                    "ranking": ranking,
                },
                "evidence": {
                    "model_site_id": "action_head.layers.8.resid",
                    "primitive_kind": primitive_kind,
                },
            },
        )
        assert response.status_code == 200

    pins = client.get("/api/evidence-pins").json()["pins"]
    assert [pin["selection"].get("ranking") for pin in pins] == [case[0] for case in cases]
    assert [pin["evidence"]["primitive_kind"] for pin in pins] == [case[1] for case in cases]


def test_evidence_pin_accepts_model_site_from_research_selection(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    client = TestClient(create_dashboard_app(dataset.root))
    trace_id = dataset.bundles[0].manifest.trace_id

    response = client.post(
        "/api/evidence-pins",
        json={
            "label": "Selection model site",
            "selection": {
                "dataset_id": "demo",
                "episode_id": trace_id,
                "lens_id": "probe-a",
                "lens_run_id": "run-a",
                "model_locus": {"model_site_id": "action_head.layers.8.resid"},
                "policy_call": 0,
            },
            "evidence": {"primitive_kind": "prediction"},
        },
    )

    assert response.status_code == 200
    pin = response.json()["pin"]
    assert pin["selection"]["model_locus"]["model_site_id"] == "action_head.layers.8.resid"
