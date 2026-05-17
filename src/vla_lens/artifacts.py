"""Saved analysis artifacts for trace-aligned VLA interpretability.

A ``LensArtifact`` is the durable record of a probe, attribution map,
intervention result, or visualization summary.  The object is intentionally
small: it stores provenance and points at arrays on disk rather than trying to
own every analysis-specific payload shape.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str, fallback: str = "artifact") -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-._").lower()
    return slug or fallback


def make_artifact_id(name: str, artifact_type: str) -> str:
    prefix = slugify(f"{artifact_type}-{name}", fallback=artifact_type)
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


@dataclass(frozen=True, slots=True)
class LensArtifact:
    """Provenance and display metadata for a generated VLA-lens result."""

    artifact_id: str
    artifact_type: str
    name: str
    group_id: str | None = None
    scope: str = "bundle"
    selector: Mapping[str, Any] = field(default_factory=dict)
    method: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    arrays: Mapping[str, str] = field(default_factory=dict)
    display: Mapping[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    created_utc: str = field(default_factory=utc_now_iso)
    source_trace_ids: tuple[str, ...] = ()
    path: str | None = None

    @classmethod
    def create(
        cls,
        *,
        artifact_type: str,
        name: str,
        group_id: str | None = None,
        scope: str = "bundle",
        selector: Mapping[str, Any] | None = None,
        method: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        arrays: Mapping[str, str] | None = None,
        display: Mapping[str, Any] | None = None,
        tags: tuple[str, ...] = (),
        source_trace_ids: tuple[str, ...] = (),
    ) -> "LensArtifact":
        return cls(
            artifact_id=make_artifact_id(name, artifact_type),
            artifact_type=artifact_type,
            name=name,
            group_id=group_id,
            scope=scope,
            selector=selector or {},
            method=method or {},
            metrics=metrics or {},
            arrays=arrays or {},
            display=display or {},
            tags=tags,
            source_trace_ids=source_trace_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["tags"] = list(self.tags)
        payload["source_trace_ids"] = list(self.source_trace_ids)
        return payload

    def to_record(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "name": self.name,
            "group_id": self.group_id,
            "scope": self.scope,
            "selector": json.dumps(self.selector, sort_keys=True),
            "method": json.dumps(self.method, sort_keys=True),
            "metrics": json.dumps(self.metrics, sort_keys=True),
            "arrays": json.dumps(self.arrays, sort_keys=True),
            "display": json.dumps(self.display, sort_keys=True),
            "tags": json.dumps(list(self.tags)),
            "source_trace_ids": json.dumps(list(self.source_trace_ids)),
            "created_utc": self.created_utc,
            "path": self.path,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LensArtifact":
        return cls(
            artifact_id=str(payload["artifact_id"]),
            artifact_type=str(payload["artifact_type"]),
            name=str(payload["name"]),
            group_id=_optional_str(payload.get("group_id")),
            scope=str(payload.get("scope", "bundle")),
            selector=dict(payload.get("selector") or {}),
            method=dict(payload.get("method") or {}),
            metrics=dict(payload.get("metrics") or {}),
            arrays=dict(payload.get("arrays") or {}),
            display=dict(payload.get("display") or {}),
            tags=tuple(payload.get("tags") or ()),
            created_utc=str(payload.get("created_utc") or utc_now_iso()),
            source_trace_ids=tuple(payload.get("source_trace_ids") or ()),
            path=_optional_str(payload.get("path")),
        )


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    return str(value)
