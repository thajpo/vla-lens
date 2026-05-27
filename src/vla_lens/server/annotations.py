"""Episode annotation persistence helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


def _episode_annotations_payload(root: Path, *, trace_id: str | None = None) -> dict[str, Any]:
    annotations = _read_episode_annotations(root)
    if trace_id:
        return {
            "annotation": annotations.get(
                trace_id,
                _empty_episode_annotation(trace_id),
            )
        }
    return {"annotations": annotations}


def _save_episode_annotation_payload(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    trace_id = str(payload.get("trace_id") or "").strip()
    if not trace_id:
        raise ValueError("Episode annotation requires trace_id")
    annotations = _read_episode_annotations(root)
    current = annotations.get(trace_id, _empty_episode_annotation(trace_id))
    annotation = {
        **current,
        "trace_id": trace_id,
        "starred": bool(payload.get("starred", current.get("starred", False))),
        "notes": str(payload.get("notes", current.get("notes", ""))),
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    annotations[trace_id] = annotation
    path = _episode_annotations_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(annotations, indent=2, sort_keys=True), encoding="utf-8")
    return {"annotation": annotation}


def _read_episode_annotations(root: Path) -> dict[str, dict[str, Any]]:
    path = _episode_annotations_path(root)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    annotations: dict[str, dict[str, Any]] = {}
    for trace_id, value in payload.items():
        if isinstance(value, Mapping):
            annotations[str(trace_id)] = {
                **_empty_episode_annotation(str(trace_id)),
                **dict(value),
                "trace_id": str(trace_id),
            }
    return annotations


def _episode_annotations_path(root: Path) -> Path:
    return root / "annotations" / "episode_annotations.json"


def _empty_episode_annotation(trace_id: str) -> dict[str, Any]:
    return {"trace_id": trace_id, "starred": False, "notes": "", "updated_utc": None}
