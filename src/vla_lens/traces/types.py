"""Trace schema dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from vla_lens.traces.layout import SCHEMA_VERSION


class TraceManifest:
    """Rollout-level metadata for one trace bundle."""

    trace_id: str
    episode_id: str
    task_id: str
    prompt: str
    model_id: str
    env_id: str
    robot_id: str
    outcome: str
    length: int
    schema_version: str = SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TraceManifest":
        return cls(
            trace_id=str(payload["trace_id"]),
            episode_id=str(payload.get("episode_id", payload["trace_id"])),
            task_id=str(payload.get("task_id", "")),
            prompt=str(payload.get("prompt", "")),
            model_id=str(payload.get("model_id", "")),
            env_id=str(payload.get("env_id", "")),
            robot_id=str(payload.get("robot_id", "")),
            outcome=str(payload.get("outcome", "unknown")),
            length=int(payload.get("length", 0)),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class ArraySpec:
    """Array payload plus axis metadata used when writing a bundle."""

    array: np.ndarray
    axes: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelSiteSpec:
    """Model tensor payload plus the semantic site it came from."""

    name: str
    array: np.ndarray
    axes: Sequence[str]
    module: str
    layer: int | None = None
    tensor_type: str = "activation"
    token_kind: str | None = None
    generation_step: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    family: str | None = None
    role: str | None = None
    segment: str | None = None
    materialization: str = "raw"
    exactness: str = "exact"
    token_space_id: str | None = None
    query_token_space_id: str | None = None
    key_token_space_id: str | None = None
    parent_site_id: str | None = None
    summary_type: str | None = None
    capture_family: str | None = None
    view_kind: str | None = None
    capture_role: str | None = None
    default_view: bool | None = None
    derived_from: Sequence[str] | None = None
    derivation: str | None = None


ActivationSpec = ModelSiteSpec
