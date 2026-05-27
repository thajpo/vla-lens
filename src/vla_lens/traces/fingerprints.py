"""Trace fingerprint computation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import zarr

from vla_lens.traces.io import _read_table
from vla_lens.traces.types import TraceManifest

_ARRAY_INDEX = "tables/array_index.parquet"
_TIMESTEPS = "tables/timesteps.parquet"
_POLICY_CALLS = "tables/policy_calls.parquet"
_ROBOT_STATE = "tables/robot_state.parquet"
_SCENE_STATE = "tables/scene_state.parquet"
_CAMERA_STATE = "tables/camera_state.parquet"
_EVALUATION = "tables/evaluation.parquet"
_IMAGE_PREPROCESSING = "tables/image_preprocessing.parquet"
_PROMPT_METADATA = "tables/prompt_metadata.parquet"
_ACTION_NORMALIZATION = "tables/action_normalization.parquet"
_CAPTURE_REQUEST = "tables/capture_request.json"
_CAPTURE_PLAN = "tables/capture_plan.json"
_CAPTURE_REPORT = "tables/capture_report.json"
_MODEL_SITES = "tables/model_sites.parquet"
_STREAMS = "tables/streams.parquet"
_TOKEN_SPACES = "tables/token_spaces.parquet"
_TOKENS = "tables/tokens.parquet"
_GENERATION_STEPS = "tables/generation_steps.parquet"


def _compute_trace_fingerprints(path: Path, *, manifest: TraceManifest) -> dict[str, Any]:
    array_index = _read_table(path / _ARRAY_INDEX)
    trajectory_payload = {
        "tables": {
            "timesteps": _table_fingerprint_payload(_read_table(path / _TIMESTEPS)),
            "policy_calls": _table_fingerprint_payload(
                _read_table(path / _POLICY_CALLS)
            ),
        },
        "arrays": _fingerprint_arrays(
            path,
            array_index,
            names={
                "executed_actions",
                "action_chunks",
                "generation_actions",
                "generation_velocities",
            },
        ),
    }
    context_payload = {
        "manifest_context": {
            "trace_id": manifest.trace_id,
            "episode_id": manifest.episode_id,
            "task_id": manifest.task_id,
            "prompt": manifest.prompt,
            "model_id": manifest.model_id,
            "env_id": manifest.env_id,
            "robot_id": manifest.robot_id,
            "outcome": manifest.outcome,
            "length": manifest.length,
        },
        "tables": {
            "robot_state": _table_fingerprint_payload(_read_table(path / _ROBOT_STATE)),
            "scene_state": _table_fingerprint_payload(_read_table(path / _SCENE_STATE)),
            "camera_state": _table_fingerprint_payload(
                _read_table(path / _CAMERA_STATE)
            ),
            "evaluation": _table_fingerprint_payload(_read_table(path / _EVALUATION)),
            "image_preprocessing": _table_fingerprint_payload(
                _read_table(path / _IMAGE_PREPROCESSING)
            ),
            "prompt_metadata": _table_fingerprint_payload(
                _read_table(path / _PROMPT_METADATA)
            ),
            "action_normalization": _table_fingerprint_payload(
                _read_table(path / _ACTION_NORMALIZATION)
            ),
        },
        "arrays": _fingerprint_arrays(
            path,
            array_index,
            prefixes=("robot_", "scene_", "camera_", "evaluation_"),
        ),
    }
    trace_schema_payload = {
        "manifest": _without_fingerprint_fields(manifest.to_dict()),
        "capture_request": _without_fingerprint_fields(
            _read_json(path / _CAPTURE_REQUEST)
        ),
        "capture_plan": _without_fingerprint_fields(_read_json(path / _CAPTURE_PLAN)),
        "capture_report": _without_fingerprint_fields(
            _read_json(path / _CAPTURE_REPORT)
        ),
        "tables": {
            "array_index": _table_fingerprint_payload(array_index),
            "model_sites": _table_fingerprint_payload(_read_table(path / _MODEL_SITES)),
            "streams": _table_fingerprint_payload(_read_table(path / _STREAMS)),
            "token_spaces": _table_fingerprint_payload(
                _read_table(path / _TOKEN_SPACES)
            ),
            "tokens": _table_fingerprint_payload(_read_table(path / _TOKENS)),
            "policy_calls": _table_fingerprint_payload(
                _read_table(path / _POLICY_CALLS)
            ),
            "generation_steps": _table_fingerprint_payload(
                _read_table(path / _GENERATION_STEPS)
            ),
        },
    }

    trajectory_fingerprint = _hash_json_payload(trajectory_payload)
    context_fingerprint = _hash_json_payload(context_payload)
    trace_schema_fingerprint = _hash_json_payload(trace_schema_payload)
    component_payload = {
        "trajectory_fingerprint": trajectory_fingerprint,
        "context_fingerprint": context_fingerprint,
        "trace_schema_fingerprint": trace_schema_fingerprint,
    }
    return {
        "fingerprint_schema_version": 1,
        "algorithm": "sha256",
        **component_payload,
        "trace_fingerprint": _hash_json_payload(component_payload),
        "components": {
            "trajectory": _fingerprint_component_summary(trajectory_payload),
            "context": _fingerprint_component_summary(context_payload),
            "trace_schema": _fingerprint_component_summary(trace_schema_payload),
        },
    }


def _fingerprint_component_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    tables = payload.get("tables")
    if isinstance(tables, Mapping):
        summary["tables"] = {
            str(name): {
                "rows": value.get("rows"),
                "columns": value.get("columns"),
                "fingerprint": value.get("fingerprint"),
            }
            for name, value in tables.items()
            if isinstance(value, Mapping)
        }
    arrays = payload.get("arrays")
    if isinstance(arrays, Mapping):
        summary["arrays"] = {
            str(name): {
                "shape": value.get("shape"),
                "dtype": value.get("dtype"),
                "fingerprint": value.get("fingerprint"),
            }
            for name, value in arrays.items()
            if isinstance(value, Mapping)
        }
    return summary


def _table_fingerprint_payload(frame: pd.DataFrame) -> dict[str, Any]:
    columns = sorted(str(column) for column in frame.columns)
    if frame.empty:
        records: list[dict[str, Any]] = []
    else:
        records = [
            {str(column): _jsonable_cell(row[column]) for column in columns}
            for _, row in frame.loc[:, columns].iterrows()
        ]
    payload = {"columns": columns, "rows": int(len(frame)), "records": records}
    return {
        "columns": columns,
        "rows": int(len(frame)),
        "fingerprint": _hash_json_payload(payload),
    }


def _fingerprint_arrays(
    bundle_path: Path,
    array_index: pd.DataFrame,
    *,
    names: set[str] | None = None,
    prefixes: tuple[str, ...] = (),
) -> dict[str, dict[str, Any]]:
    if array_index.empty or "name" not in array_index:
        return {}
    records: dict[str, dict[str, Any]] = {}
    for row in array_index.to_dict("records"):
        name = str(row.get("name") or "")
        if names is not None and name not in names:
            continue
        if prefixes and not name.startswith(prefixes):
            continue
        relative_path = Path(str(row.get("relative_path") or ""))
        if relative_path.suffix != ".zarr":
            continue
        records[name] = _array_fingerprint_payload(bundle_path / relative_path)
    return records


def _array_fingerprint_payload(path: Path) -> dict[str, Any]:
    array = zarr.open_array(str(path), mode="r")
    value = np.asarray(array[:])
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(value)
    digest.update(json.dumps([int(item) for item in contiguous.shape]).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.view(np.uint8))
    return {
        "shape": [int(item) for item in contiguous.shape],
        "dtype": str(contiguous.dtype),
        "fingerprint": f"sha256:{digest.hexdigest()}",
    }


def _validate_artifact_id(artifact_id: str) -> None:
    value = str(artifact_id)
    if not value or Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"Invalid artifact_id: {artifact_id!r}")
    if value in {".", ".."}:
        raise ValueError(f"Invalid artifact_id: {artifact_id!r}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _without_fingerprint_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_fingerprint_fields(item)
            for key, item in value.items()
            if str(key) != "fingerprints"
        }
    if isinstance(value, list):
        return [_without_fingerprint_fields(item) for item in value]
    return value


def _hash_json_payload(payload: Any) -> str:
    encoded = json.dumps(_jsonable_cell(payload), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _jsonable_cell(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable_cell(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_cell(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable_cell(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable_cell(value.item())
    if pd.isna(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value
