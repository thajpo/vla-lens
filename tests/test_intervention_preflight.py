from __future__ import annotations

import subprocess
import sys
from typing import Any

from fastapi.testclient import TestClient

from vla_lens.interventions.preflight import PREFLIGHT_CHECK_NAMES, intervention_preflight
from vla_lens.server.fastapi_app import create_dashboard_app
from vla_lens.synthetic import create_synthetic_trace_dataset
from vla_lens.traces import TraceDataset


def _probe_artifact_id(dataset: TraceDataset) -> str:
    rows = dataset.artifact_index
    matches = rows.loc[rows["artifact_type"].astype(str) == "probe_suite"]
    assert not matches.empty
    return str(matches.iloc[0]["artifact_id"])


def _request(
    dataset: TraceDataset,
    *,
    source_artifact_id: str | None = None,
    basis: list[str] | None = None,
) -> dict[str, Any]:
    trace_id = dataset.bundles[0].manifest.trace_id
    return {
        "runtime_adapter": "synthetic",
        "target": {
            "kind": "probe_direction",
            "source_artifact_id": source_artifact_id or _probe_artifact_id(dataset),
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
                "outcome": {"kind": "action", "basis": basis or ["raw", "gripper"]},
            },
        },
    }


def _checks(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {check["name"]: check for check in result["checks"]}


def test_intervention_preflight_complete_metadata_is_inspected_only(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=8)

    result = intervention_preflight(dataset, _request(dataset))
    payload = result.to_dict()
    checks = _checks(payload)

    assert payload["status"] == "inspected_only"
    assert set(checks) == set(PREFLIGHT_CHECK_NAMES)
    assert checks["policy_call_exists"]["status"] == "ok"
    assert checks["stored_action_exists"]["status"] == "ok"
    assert checks["stored_action_chunk_exists"]["status"] == "ok"
    assert checks["source_artifact_exists"]["status"] == "ok"
    assert checks["target_site_declared_in_model_site_index"]["status"] == "ok"
    assert checks["token_space_declared"]["status"] == "ok"
    assert checks["action_decoder_metadata_available"]["status"] == "ok"
    assert checks["action_basis_metadata_available"]["status"] == "ok"
    assert checks["runtime_adapter_declared"]["status"] == "ok"
    assert checks["model_runtime_available"]["status"] == "unavailable"
    assert checks["runtime_environment_safe"]["status"] == "ok"
    assert payload["target_resolution"]["token_space"] == "synthetic.action_suffix"
    assert payload["action_basis_status"]["gripper"]["available"] is True
    assert "model_runtime_available" in payload["missing_capabilities"]


def test_intervention_preflight_missing_source_artifact_fails(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=8)

    result = intervention_preflight(
        dataset,
        _request(dataset, source_artifact_id="missing-source-artifact"),
    ).to_dict()
    checks = _checks(result)

    assert result["status"] == "failed"
    assert checks["source_artifact_exists"]["status"] == "failed"
    assert "missing-source-artifact" in checks["source_artifact_exists"]["message"]
    assert "source_artifact_exists" in result["missing_capabilities"]


def test_intervention_preflight_missing_named_action_basis_is_partial(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=8)

    result = intervention_preflight(dataset, _request(dataset, basis=["raw", "jaw"])).to_dict()
    checks = _checks(result)

    assert result["status"] == "partial"
    assert checks["action_basis_metadata_available"]["status"] == "partial"
    assert result["action_basis_status"]["raw"]["available"] is True
    assert result["action_basis_status"]["jaw"]["available"] is False
    assert "action_basis_metadata_available" in result["missing_capabilities"]


def test_intervention_preflight_route_returns_runtime_free_result(tmp_path):
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=8)
    client = TestClient(create_dashboard_app(dataset.root))

    response = client.post("/api/interventions/preflight", json=_request(dataset))
    payload = response.json()
    checks = _checks(payload["preflight"])

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert payload["preflight"]["status"] == "inspected_only"
    assert set(checks) == set(PREFLIGHT_CHECK_NAMES)
    assert checks["model_runtime_available"]["status"] == "unavailable"
    assert payload["preflight"]["runtime_environment"]["mode"] == "metadata_preflight"


def test_intervention_preflight_import_does_not_load_heavy_runtime_dependencies():
    code = """
import sys
import vla_lens.interventions.preflight
banned = {"torch", "lerobot", "libero", "robosuite"}
loaded = sorted(name for name in banned if name in sys.modules)
if loaded:
    raise SystemExit("loaded heavy modules: " + ", ".join(loaded))
"""

    subprocess.run([sys.executable, "-c", code], check=True)
