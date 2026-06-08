"""Discovery-artifact dashboard payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from vla_lens.dataset.index import ARTIFACT_INDEX
from vla_lens.interventions.families import (
    artifact_family_for_type,
    artifact_family_registry,
    target_from_discovery_artifact,
)
from vla_lens.server.common import _json_parse, _jsonable
from vla_lens.server.episode_lens_common import (
    _episode_lens_selection_from_query,
    _first_present,
)
from vla_lens.server.episode_lens_probe import (
    _probe_suite_episode_lens_view,
    _unavailable_episode_lens_view,
)
from vla_lens.server.indexed import _query_value, _read_table, indexed_episodes_payload
from vla_lens.server.indexed_probes import indexed_episode_probes_payload
from vla_lens.traces import TraceDataset


def discovery_artifact_families_payload() -> dict[str, Any]:
    """Return discovery-artifact family contracts exposed to the dashboard."""
    families = [
        {"available": True, **contract.to_dict(), "reason": ""}
        for contract in artifact_family_registry()
    ]
    return {"families": families, "total": len(families)}


def discovery_artifact_episodes_payload(
    root: Path,
    artifact_id: str,
    query: Mapping[str, list[str]],
) -> dict[str, Any]:
    """Return episodes ranked by one discovery artifact."""
    artifact = _discovery_artifact_payload(root, artifact_id)
    family = _family_payload(artifact)
    if not family["available"]:
        return {
            **_empty_episode_page(query),
            "artifact": artifact,
            "family": family,
            "available": False,
            "reason": family["reason"],
            "rank_by": _query_value(query, "rank_by") or "interest",
        }
    if artifact["artifact_type"] != "probe_suite":
        return {
            **_empty_episode_page(query),
            "artifact": artifact,
            "family": family,
            "available": False,
            "reason": (
                f"Discovery artifact type '{artifact['artifact_type']}' does not support "
                "episode ranking yet."
            ),
            "rank_by": _query_value(query, "rank_by") or "interest",
        }

    episode_query = _probe_episode_query(artifact_id, query)
    payload = indexed_episodes_payload(root, episode_query)
    return {
        **payload,
        "artifact": artifact,
        "family": family,
        "available": True,
        "reason": "",
        "rank_by": _query_value(query, "rank_by") or "interest",
    }


def discovery_artifact_readout_payload(
    root: Path,
    artifact_id: str,
    query: Mapping[str, list[str]],
) -> dict[str, Any]:
    """Return what one discovery artifact says about one selected episode."""
    trace_id = _query_value(query, "trace_id") or ""
    if not trace_id:
        raise KeyError("Missing query parameter: trace_id")
    artifact = _discovery_artifact_payload(root, artifact_id)
    family = _family_payload(artifact)
    if not family["available"]:
        return _unavailable_readout(artifact, family, trace_id, family["reason"])
    if artifact["artifact_type"] != "probe_suite":
        return _unavailable_readout(
            artifact,
            family,
            trace_id,
            (
                f"Discovery artifact type '{artifact['artifact_type']}' does not support "
                "episode readouts yet."
            ),
        )

    probes = indexed_episode_probes_payload(root, {"trace_id": [trace_id]})
    probe = next(
        (
            item
            for item in probes.get("probes", [])
            if str(item.get("artifact_id") or "") == artifact_id
        ),
        None,
    )
    if probe is None:
        raise KeyError(artifact_id)
    best_row = (
        probe.get("episode_summary", {}).get("best_row")
        if isinstance(probe.get("episode_summary"), Mapping)
        else {}
    )
    best_row = best_row if isinstance(best_row, Mapping) else {}
    summary = (
        probe.get("episode_summary") if isinstance(probe.get("episode_summary"), Mapping) else {}
    )
    policy_call = _first_present(
        _query_value(query, "policy_call") or _query_value(query, "policy_call_index"),
        best_row.get("policy_call_index"),
    )
    model_site = _first_present(
        _query_value(query, "model_site") or _query_value(query, "site"),
        best_row.get("model_site_id"),
        best_row.get("model"),
        artifact.get("best_model"),
    )
    feature = _first_present(best_row.get("feature"), artifact.get("best_feature"))
    return {
        "artifact": artifact,
        "family": family,
        "available": bool(probe.get("available")),
        "reason": (
            "" if probe.get("available") else "This artifact has no readout for the selected trace."
        ),
        "readout_type": "probe_prediction",
        "trace_id": trace_id,
        "summary": {
            "title": str(artifact.get("name") or artifact_id),
            "target": artifact.get("target"),
            "actual": summary.get("actual"),
            "predicted": summary.get("predicted"),
            "confidence": summary.get("confidence"),
            "correct": summary.get("correct"),
            "correct_rate": summary.get("correct_rate"),
            "split": best_row.get("split") or best_row.get("eval_split"),
            "policy_call_index": policy_call,
            "model_site": model_site,
            "feature": feature,
        },
        "target_hint": {
            "source_artifact_id": artifact_id,
            "source_artifact_type": artifact.get("artifact_type"),
            "policy_call_index": policy_call,
            "model_site": model_site,
            "feature": feature,
            "token_space": _query_value(query, "token_space") or best_row.get("token_space_id"),
        },
        "rows": probe.get("rows") if isinstance(probe.get("rows"), list) else [],
        "row_count": int(probe.get("row_count") or 0),
    }


def discovery_artifact_target_payload(
    dataset: TraceDataset,
    artifact_id: str,
    query: Mapping[str, list[str]],
) -> dict[str, Any]:
    """Convert a discovery artifact into an intervention TargetSpec candidate."""
    artifact = _discovery_artifact_payload(dataset.root, artifact_id)
    family = _family_payload(artifact)
    if not family["available"]:
        return {
            "artifact": artifact,
            "family": family,
            "available": False,
            "reason": family["reason"],
            "target": None,
        }
    metadata = {
        key: value
        for key, value in {
            "trace_id": _query_value(query, "trace_id"),
            "policy_call_index": _query_value(query, "policy_call")
            or _query_value(query, "policy_call_index"),
        }.items()
        if value not in {None, ""}
    }
    target = target_from_discovery_artifact(
        artifact,
        model_site=_query_value(query, "model_site") or _query_value(query, "site"),
        token_space=_query_value(query, "token_space"),
        metadata=metadata,
    )
    return {
        "artifact": artifact,
        "family": family,
        "available": True,
        "reason": "",
        "target": target.to_dict(),
    }


def discovery_artifact_episode_lens_view_payload(
    dataset: TraceDataset,
    artifact_id: str,
    query: Mapping[str, list[str]],
) -> dict[str, Any]:
    """Return an episode-conditioned view model for one discovery artifact."""
    trace_id = _query_value(query, "trace_id") or ""
    if not trace_id:
        raise KeyError("Missing query parameter: trace_id")
    artifact = _discovery_artifact_payload(dataset.root, artifact_id)
    family = _family_payload(artifact)
    current_selection = _episode_lens_selection_from_query(trace_id, query)
    if not family["available"]:
        return {
            "view": _unavailable_episode_lens_view(
                dataset,
                artifact,
                family,
                trace_id,
                current_selection,
                family["reason"],
            )
        }
    if artifact.get("artifact_type") != "probe_suite":
        return {
            "view": _unavailable_episode_lens_view(
                dataset,
                artifact,
                family,
                trace_id,
                current_selection,
                (
                    f"Discovery artifact type '{artifact.get('artifact_type')}' does not "
                    "support episode LensViews yet."
                ),
            )
        }

    artifact_object = dataset.load_artifact(artifact_id)
    return {
        "view": _probe_suite_episode_lens_view(
            dataset,
            artifact,
            artifact_object,
            family,
            trace_id,
            current_selection,
            query,
        )
    }


def _discovery_artifact_payload(root: Path, artifact_id: str) -> dict[str, Any]:
    artifacts = _read_table(root / ARTIFACT_INDEX)
    if artifacts.empty or "artifact_id" not in artifacts:
        raise KeyError(artifact_id)
    rows = artifacts.loc[artifacts["artifact_id"].astype(str) == artifact_id]
    if rows.empty:
        raise KeyError(artifact_id)
    row = rows.iloc[0].to_dict()
    return _artifact_row_payload(row)


def _artifact_row_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(key): value for key, value in row.items() if not _is_missing(value)}
    for key in ("selector", "method", "metrics", "arrays", "display", "tags", "source_trace_ids"):
        if key in payload:
            payload[key] = _jsonable(_json_parse(payload.get(key)))
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
    if isinstance(metrics, Mapping):
        payload.setdefault("target", metrics.get("target"))
        payload.setdefault("best_model", metrics.get("best_model"))
        payload.setdefault("best_feature", metrics.get("best_feature"))
    return _jsonable(payload)


def _family_payload(artifact: Mapping[str, Any]) -> dict[str, Any]:
    artifact_type = str(artifact.get("artifact_type") or "")
    try:
        contract = artifact_family_for_type(artifact_type)
    except KeyError:
        return {
            "available": False,
            "artifact_type": artifact_type,
            "reason": f"Unknown discovery artifact family '{artifact_type}'.",
        }
    return {"available": True, **contract.to_dict(), "reason": ""}


def _probe_episode_query(
    artifact_id: str,
    query: Mapping[str, list[str]],
) -> dict[str, list[str]]:
    mapped: dict[str, list[str]] = {key: list(value) for key, value in query.items()}
    mapped["probe_id"] = [artifact_id]
    mapped.setdefault("sort", ["probe_interest"])
    rank_by = _query_value(query, "rank_by")
    if rank_by == "interest" or (rank_by in {"", None} and not _query_value(query, "sort")):
        mapped["sort"] = ["probe_interest"]
    for generic, probe_specific in (
        ("split", "probe_split"),
        ("prediction", "probe_prediction"),
        ("cohort_preset", "probe_cohort_preset"),
    ):
        value = _query_value(query, generic)
        if value and value != "all" and not _query_value(query, probe_specific):
            mapped[probe_specific] = [value]
    return mapped


def _empty_episode_page(query: Mapping[str, list[str]]) -> dict[str, Any]:
    limit = int(_query_value(query, "limit") or 100)
    offset = int(_query_value(query, "offset") or 0)
    return {
        "episodes": [],
        "total": 0,
        "limit": limit,
        "offset": offset,
        "next_offset": None,
        "facets": {},
        "sort": "unsupported",
    }


def _unavailable_readout(
    artifact: Mapping[str, Any],
    family: Mapping[str, Any],
    trace_id: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "family": family,
        "available": False,
        "reason": reason,
        "readout_type": "unavailable",
        "trace_id": trace_id,
        "summary": {},
        "target_hint": {},
        "rows": [],
        "row_count": 0,
    }


































































































def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    return False


__all__ = [
    "discovery_artifact_episodes_payload",
    "discovery_artifact_episode_lens_view_payload",
    "discovery_artifact_families_payload",
    "discovery_artifact_readout_payload",
    "discovery_artifact_target_payload",
]
