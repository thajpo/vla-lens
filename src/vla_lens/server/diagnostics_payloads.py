"""Dataset diagnostics and one-click artifact payload helpers."""

from __future__ import annotations

from typing import Any

from vla_lens.action_generation import save_action_generation_artifact
from vla_lens.analyzer import diagnostics_status, run_dataset_diagnostics
from vla_lens.probes.workflow import train_probe_artifact_from_spec
from vla_lens.server.artifacts import _artifacts_payload
from vla_lens.server.common import _jsonable
from vla_lens.target_object import save_target_object_encoding_artifact
from vla_lens.traces import TraceDataset


def _dataset_diagnostics_payload(dataset: TraceDataset) -> dict[str, Any]:
    status = diagnostics_status(dataset)
    return _diagnostics_payload(status)


def _run_dataset_diagnostics_payload(dataset: TraceDataset) -> dict[str, Any]:
    artifact = run_dataset_diagnostics(dataset)
    status = diagnostics_status(dataset)
    payload = _diagnostics_payload(status)
    payload["artifact"] = _jsonable(artifact.to_dict())
    return payload


def _create_outcome_probe_payload(dataset: TraceDataset) -> dict[str, Any]:
    saved = train_probe_artifact_from_spec(dataset, _default_outcome_probe_spec(dataset))
    return {
        "artifact": _jsonable(saved.artifact.to_dict()),
        "results": _jsonable(saved.results.to_dict("records")),
        "artifacts": _artifacts_payload(dataset),
    }


def _create_target_object_probe_payload(dataset: TraceDataset) -> dict[str, Any]:
    saved = save_target_object_encoding_artifact(dataset, name="Dashboard target-object encoding")
    return {
        "artifact": _jsonable(saved.artifact.to_dict()),
        "arrays": {
            "metric_cube": list(saved.metric_cube.shape),
            "baseline_cube": list(saved.baseline_cube.shape),
            "delta_cube": list(saved.delta_cube.shape),
        },
        "artifacts": _artifacts_payload(dataset),
    }


def _create_action_generation_payload(dataset: TraceDataset) -> dict[str, Any]:
    saved = save_action_generation_artifact(dataset, name="Dashboard action generation")
    return {
        "artifact": _jsonable(saved.artifact.to_dict()),
        "artifacts": _artifacts_payload(dataset),
    }


def _default_target_object_probe_spec(dataset: TraceDataset) -> dict[str, Any]:
    spec = _default_outcome_probe_spec(dataset)
    spec["name"] = "Dashboard target-object encoding probe"
    spec["target"] = {"kind": "target_object"}
    spec["split"] = {"kind": "random_episode"}
    spec["baseline"] = ["majority_class", "benchmark", "task"]
    spec["sweep"] = "layer"
    return spec


def _default_outcome_probe_spec(dataset: TraceDataset) -> dict[str, Any]:
    model_sites = dataset.model_site_index
    module = "pi05.expert.layers.*"
    tensor_type = "hidden_mean"
    token_kind = "action"
    if not model_sites.empty and "module" in model_sites:
        modules = model_sites["module"].astype(str)
        if modules.str.contains("pi05.expert", regex=False).any():
            module = "pi05.expert.layers.*"
            tensor_type = "hidden_mean"
            token_kind = "action"
        elif modules.str.contains("action_head", regex=False).any():
            module = "action_head.layers.*.resid"
            tensor_type = "resid"
            token_kind = "action"
        else:
            module = str(modules.iloc[0])
            if "tensor_type" in model_sites:
                tensor_type = str(model_sites["tensor_type"].astype(str).iloc[0])
            token_kind = None
    return {
        "name": "Dashboard outcome probe",
        "target": {"kind": "outcome"},
        "features": {
            "module": module,
            "tensor_type": tensor_type,
            "token_kind": token_kind,
            "layers": None,
            "timesteps": "all",
            "generation_step": None,
            "reduction": "mean",
        },
        "split": {"kind": "heldout_benchmark"},
        "baseline": ["majority_class", "benchmark", "target_object", "task"],
        "sweep": "layer",
    }


def _diagnostics_payload(status: dict[str, Any]) -> dict[str, Any]:
    latest = status.get("latest")
    return {
        "fingerprint": status.get("fingerprint"),
        "stale": bool(status.get("stale", True)),
        "latest": _jsonable(latest) if latest else None,
    }
