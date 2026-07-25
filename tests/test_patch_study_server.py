from __future__ import annotations

import json

from fastapi.testclient import TestClient

from vla_lens.server.fastapi_app import create_dashboard_app
from vla_lens.synthetic import create_synthetic_trace_dataset


def test_patch_study_endpoint_returns_saved_cohort_analysis(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "dataset", num_episodes=1)
    path = dataset.root / "vla_lens" / "patch_studies" / "study-a" / "analysis.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_kind": "vla_lens.patch_study_analysis",
                "study_id": "study-a",
                "pair_count": 5,
                "summary": [{"layer": 0, "token_region": "full_prefix"}],
            }
        ),
        encoding="utf-8",
    )

    response = TestClient(create_dashboard_app(dataset.root)).get("/api/patch-studies")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.json()["total"] == 1
    assert response.json()["patch_studies"][0]["study_id"] == "study-a"
