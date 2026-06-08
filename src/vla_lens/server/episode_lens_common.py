"""Shared EpisodeLensView payload helpers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from vla_lens.server.common import _jsonable
from vla_lens.server.indexed import _query_value
from vla_lens.traces import TraceDataset


def _episode_lens_selection_from_query(
    trace_id: str,
    query: Mapping[str, list[str]],
) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "timestep": _query_optional_int(query, "timestep"),
        "policy_call_index": _query_optional_int(query, "policy_call_index", "policy_call"),
        "model_site_id": _query_value(query, "model_site_id")
        or _query_value(query, "model_site")
        or _query_value(query, "site"),
        "layer": _query_optional_int(query, "layer"),
        "feature": _query_optional_int(query, "feature"),
        "mode": _query_value(query, "mode") or "features",
    }

def _episode_lens_episode_payload(dataset: TraceDataset, trace_id: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"trace_id": trace_id, "dataset_id": None, "episode_index": None}
    episodes = dataset.episodes(trace_id=trace_id)
    if episodes.empty:
        return payload
    row = episodes.iloc[0].to_dict()
    payload["dataset_id"] = _first_present(
        row.get("dataset_id"),
        (row.get("metadata") or {}).get("dataset_id")
        if isinstance(row.get("metadata"), Mapping)
        else None,
    )
    payload["episode_index"] = _optional_int(row.get("episode_index"))
    return _jsonable(payload)

def _lens_payload(
    artifact: Mapping[str, Any],
    family: Mapping[str, Any],
    *,
    spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_type = str(artifact.get("artifact_type") or family.get("artifact_type") or "unknown")
    return {
        "artifact_id": str(artifact.get("artifact_id") or ""),
        "artifact_type": artifact_type,
        "family": artifact_type,
        "display_name": str(artifact.get("name") or artifact.get("artifact_id") or artifact_type),
        "spec": dict(spec or {}),
    }

def _query_optional_int(query: Mapping[str, list[str]], *names: str) -> int | None:
    for name in names:
        value = _query_value(query, name)
        parsed = _optional_int(value)
        if parsed is not None:
            return parsed
    return None

def _ranking_mode(query: Mapping[str, list[str]]) -> str:
    value = _query_value(query, "ranking_mode") or "probe_contribution"
    if value not in {"probe_contribution", "raw_activation"}:
        return "probe_contribution"
    return value

def _top_k(query: Mapping[str, list[str]]) -> int:
    value = _query_optional_int(query, "top_k")
    if value is None:
        return 25
    return max(1, min(100, value))

def _record_correct(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "correct"}:
            return True
        if lowered in {"false", "0", "no", "wrong", "incorrect"}:
            return False
    return None

def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(number)

def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return float(number) if np.isfinite(number) else None

def _site_name_for_id(records: list[Mapping[str, Any]], value: str) -> str | None:
    for record in records:
        name = str(record.get("name") or "")
        site_id = str(record.get("site_id") or "")
        if value in {name, site_id}:
            return name or site_id
    return None

def _layer_for_site(records: list[Mapping[str, Any]], site: Any) -> int | None:
    if not site:
        return None
    for record in records:
        if str(record.get("name") or record.get("site_id") or "") == str(site):
            return _optional_int(record.get("layer"))
    return None

def _human_label(value: Any) -> str:
    text = str(value or "").replace("_", " ").replace("-", " ").strip()
    return " ".join(
        part.upper() if part in {"eef", "vlm", "mlp"} else part.capitalize()
        for part in text.split()
    )

def _site_label(site_name: str, layer: int | None, record: Mapping[str, Any]) -> str:
    parts = []
    if layer is not None:
        parts.append(f"Layer {layer}")
    tensor = record.get("tensor_type")
    token = record.get("token_kind")
    if token or tensor:
        parts.append(_human_label(" ".join(str(item) for item in [token, tensor] if item)))
    return " · ".join(parts) if parts else site_name

def _short_site_label(site_name: str) -> str:
    return site_name.rsplit(".", 1)[-1] if site_name else "site"

def _site_from_feature_name(feature: str) -> str | None:
    if not feature:
        return None
    if "model_site_id=" in feature:
        return _feature_part(feature, "model_site_id")
    if "activation=" in feature:
        return _feature_part(feature, "activation")
    return None

def _layer_from_feature_name(feature: str) -> int | None:
    layer = _feature_part(feature, "layer")
    if layer is None:
        text = feature.strip().lower()
        if text.startswith("layer "):
            layer = text.split(" ", 1)[1]
    return _optional_int(layer)

def _site_for_best_state(
    records: list[Mapping[str, Any]],
    best_state: Mapping[str, Any],
) -> str | None:
    feature = str(best_state.get("feature") or "")
    explicit = _site_from_feature_name(feature)
    if explicit:
        return _site_name_for_id(records, explicit) or explicit
    layer = _layer_from_feature_name(feature)
    if layer is None:
        return None
    for record in records:
        if _optional_int(record.get("layer")) == layer:
            return str(record.get("name") or record.get("model_site_id") or "")
    return None

def _feature_part(feature: str, key: str) -> str | None:
    for part in feature.split(","):
        if "=" not in part:
            continue
        name, value = part.split("=", 1)
        if name.strip() == key:
            return value.strip()
    return None

def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in {None, ""}:
            return value
    return None
