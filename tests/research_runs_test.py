from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from vla_lens import create_synthetic_trace_dataset
from vla_lens.server.fastapi_app import create_dashboard_app
from vla_lens.workbench import (
    ResearchProgress,
    ResearchResultSummary,
    ResearchRunSpec,
    get_research_run,
    list_research_runs,
    save_research_run,
    workbench_manifest,
)


def _run(**overrides) -> ResearchRunSpec:
    payload = {
        "run_id": "rq-015-object-regions",
        "parent_run_id": "semantic-object-wave",
        "kind": "probe",
        "name": "Object identity in known regions",
        "question": "Does an object's image region contain its identity?",
        "status": "completed",
        "stage": "completed",
        "progress": ResearchProgress(completed=5, total=5, unit="layers"),
        "artifact_ids": ("probe-object-regions",),
        "result": ResearchResultSummary(
            metric="balanced accuracy",
            score=0.76,
            baseline=0.51,
            delta=0.25,
            verdict="beats the strongest control",
        ),
        "created_utc": "2026-07-22T10:00:00+00:00",
        "updated_utc": "2026-07-22T11:00:00+00:00",
        "started_utc": "2026-07-22T10:01:00+00:00",
        "completed_utc": "2026-07-22T11:00:00+00:00",
        "provenance": {"cache_fingerprint": "abc123", "worker": "probe-1"},
    }
    payload.update(overrides)
    return ResearchRunSpec(**payload)


def test_research_run_roundtrip_preserves_reconstructable_progress_and_result(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    run = _run()

    save_research_run(dataset, run)
    loaded = get_research_run(dataset, run.run_id)

    assert loaded == run
    assert loaded.progress.fraction == 1.0
    assert loaded.result.delta == pytest.approx(0.25)
    path = dataset.root / "vla_lens" / "workbench" / "research_runs" / f"{run.run_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["progress"] == {
        "completed": 5,
        "total": 5,
        "unit": "layers",
        "fraction": 1.0,
    }
    assert not list(path.parent.glob("*.tmp"))


def test_research_run_save_atomically_replaces_one_run_file(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    save_research_run(dataset, _run(status="running", stage="training"))
    save_research_run(dataset, _run(status="completed", stage="completed"))

    runs = list_research_runs(dataset)

    assert len(runs) == 1
    assert runs[0].status == "completed"
    run_root = dataset.root / "vla_lens" / "workbench" / "research_runs"
    assert len(list(run_root.glob("*.json"))) == 1


def test_research_progress_accepts_legacy_fraction_and_rejects_invalid_counts():
    progress = ResearchProgress.from_value(0.42)

    assert progress.to_dict() == {
        "completed": 42,
        "total": 100,
        "unit": "percent",
        "fraction": 0.42,
    }
    with pytest.raises(ValueError, match="cannot exceed"):
        ResearchProgress(completed=2, total=1)


def test_research_run_api_and_workbench_manifest_expose_lifecycle_records(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    save_research_run(dataset, _run())
    client = TestClient(create_dashboard_app(dataset.root))

    listing = client.get("/api/research-runs")
    detail = client.get("/api/research-runs/rq-015-object-regions")
    missing = client.get("/api/research-runs/missing")

    assert listing.status_code == 200
    assert listing.headers["Cache-Control"] == "no-store"
    assert listing.json()["research_runs"][0]["question"].startswith("Does an object")
    assert detail.status_code == 200
    assert detail.json()["research_run"]["result"]["baseline"] == pytest.approx(0.51)
    assert missing.status_code == 404
    assert workbench_manifest(dataset)["research_runs"][0]["run_id"] == run_id(detail.json())


def run_id(payload: dict[str, object]) -> str:
    run = payload["research_run"]
    assert isinstance(run, dict)
    return str(run["run_id"])
