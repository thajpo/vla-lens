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


def _evidence_pins_payload(root: Path) -> dict[str, Any]:
    pins = _read_evidence_pins(root)
    return {"pins": pins, "total": len(pins)}


def _save_evidence_pin_payload(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    selection = payload.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Evidence pin requires selection")
    for field in ("dataset_id", "episode_id", "lens_id", "lens_run_id"):
        if not str(selection.get(field) or "").strip():
            raise ValueError(f"Evidence pin requires selection.{field}")
    if selection.get("timestep") is None and selection.get("policy_call") is None:
        raise ValueError("Evidence pin requires selection.timestep or selection.policy_call")
    evidence = payload.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("Evidence pin requires evidence")
    if not str(evidence.get("primitive_kind") or "").strip():
        raise ValueError("Evidence pin requires evidence.primitive_kind")
    selection_locus = selection.get("model_locus")
    selection_model_site = (
        selection_locus.get("model_site_id")
        if isinstance(selection_locus, Mapping)
        else None
    )
    if not str(evidence.get("model_site_id") or selection_model_site or "").strip():
        raise ValueError(
            "Evidence pin requires evidence.model_site_id or "
            "selection.model_locus.model_site_id"
        )
    pins = _read_evidence_pins(root)
    now = datetime.now(UTC).isoformat()
    pin_id = payload.get("pin_id") or (
        f"pin_{len(pins) + 1}_{now.replace(':', '').replace('.', '')}"
    )
    pin = {
        "pin_id": str(pin_id),
        "created_utc": now,
        "label": str(payload.get("label") or "Pinned evidence"),
        "note": str(payload.get("note") or ""),
        "selection": dict(selection),
        "evidence": dict(evidence),
    }
    pins.append(pin)
    path = _evidence_pins_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pins, indent=2, sort_keys=True), encoding="utf-8")
    return {"pin": pin, "pins": pins, "total": len(pins)}


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


def _read_evidence_pins(root: Path) -> list[dict[str, Any]]:
    path = _evidence_pins_path(root)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, Mapping)]


def _episode_annotations_path(root: Path) -> Path:
    return root / "annotations" / "episode_annotations.json"


def _evidence_pins_path(root: Path) -> Path:
    return root / "annotations" / "evidence_pins.json"


def _empty_episode_annotation(trace_id: str) -> dict[str, Any]:
    return {"trace_id": trace_id, "starred": False, "notes": "", "updated_utc": None}
