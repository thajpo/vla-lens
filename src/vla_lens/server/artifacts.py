"""Dataset artifact response helpers."""

from __future__ import annotations

from typing import Any

from vla_lens.server.common import (
    _array_summary,
    _json_parse,
    _json_scalar,
    _jsonable,
)
from vla_lens.server.metrics import _manifest_payload
from vla_lens.traces import TraceBundle, TraceDataset


def _episode_payload(bundle: TraceBundle) -> dict[str, Any]:
    return {
        **_manifest_payload(bundle),
        "cameras": bundle.cameras(),
        "artifacts": (
            bundle.artifact_index.to_dict("records") if not bundle.artifact_index.empty else []
        ),
        "arrays": bundle.array_index.to_dict("records") if not bundle.array_index.empty else [],
    }


def _artifacts_payload(dataset: TraceDataset) -> dict[str, Any]:
    artifacts = [_artifact_record_payload(record) for record in _artifact_records(dataset)]
    counts: dict[str, int] = {}
    for artifact in artifacts:
        key = str(artifact.get("artifact_type") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return {
        "artifacts": artifacts,
        "counts": counts,
        "total": len(artifacts),
    }


def _artifact_detail_payload(dataset: TraceDataset, artifact_id: str) -> dict[str, Any]:
    artifact = dataset.load_artifact(artifact_id)
    arrays: list[dict[str, Any]] = []
    for name, path in artifact.arrays.items():
        array = dataset.load_artifact_array(artifact, name, mmap=True)
        arrays.append(
            {
                "name": name,
                "path": path,
                "shape": [int(item) for item in array.shape],
                "dtype": str(array.dtype),
                "summary": _array_summary(array),
            }
        )
    return {
        "artifact": _jsonable(artifact.to_dict()),
        "arrays": arrays,
    }


def _artifact_summary(dataset: TraceDataset) -> dict[str, Any]:
    records = _artifact_records(dataset)
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get("artifact_type") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return {"total": len(records), "counts": counts}


def _artifact_records(dataset: TraceDataset) -> list[dict[str, Any]]:
    table = dataset.artifact_index
    if table.empty:
        return []
    return table.sort_values("created_utc", ascending=False, na_position="last").to_dict("records")


def _artifact_record_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = {str(key): _json_scalar(value) for key, value in record.items()}
    for key in ["selector", "method", "metrics", "arrays", "display", "tags", "source_trace_ids"]:
        payload[key] = _jsonable(_json_parse(payload.get(key)))
    return payload
