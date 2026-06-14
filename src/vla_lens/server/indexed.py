"""Indexed dashboard payloads backed by local Parquet tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import duckdb
import pandas as pd

from vla_lens.dataset.index import (
    ARTIFACT_INDEX,
    EPISODE_INDEX,
    INDEX_MANIFEST,
    INDEX_SCHEMA_VERSION,
    MODEL_SITE_INDEX,
    PROBE_EPISODE_INDEX,
    PROBE_PREDICTIONS,
)
from vla_lens.server.common import _json_parse, _json_scalar, _jsonable
from vla_lens.table_io import read_optional_parquet

MAX_EPISODE_LIMIT = 500
DEFAULT_EPISODE_LIMIT = 100
SEARCH_COLUMNS = (
    "trace_id",
    "episode_id",
    "task_id",
    "prompt",
    "outcome",
    "dataset_id",
    "benchmark",
    "profile",
    "seed",
)
EPISODE_SORT_COLUMNS = {
    "episode_index": "episode_index",
    "trace_id": "trace_id",
    "task_id": "task_id",
    "outcome": "outcome",
    "length": "length",
}


def indexed_dataset_payload(
    root: Path,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = manifest or _read_index_manifest(root)
    episodes = _read_table(root / EPISODE_INDEX)
    model_sites = _read_table(root / MODEL_SITE_INDEX)
    artifacts = _read_table(root / ARTIFACT_INDEX)
    probes = _read_table(root / PROBE_PREDICTIONS)
    probe_episodes = _read_table(root / PROBE_EPISODE_INDEX)
    artifact_counts = _value_counts(artifacts, "artifact_type")
    camera_names = _json_list_union(episodes.get("camera_names", pd.Series(dtype=object)))
    model_site_prefixes = sorted(
        {
            str(name).split(".", 1)[0]
            for name in model_sites.get("name", pd.Series(dtype=object)).dropna().tolist()
            if str(name).strip()
        }
    )
    array_names = _json_list_union(episodes.get("array_names", pd.Series(dtype=object)))
    flags = {
        "robot_episodes": len(episodes) > 0,
        "cameras": bool(camera_names),
        "policy_calls": _numeric_sum(episodes, "policy_call_count") > 0,
        "model_sites": len(model_sites) > 0,
        "token_spaces": _numeric_sum(episodes, "token_space_count") > 0,
        "image_token_maps": _has_indexed_image_token_maps(
            model_sites,
            _numeric_sum(episodes, "token_space_count"),
        ),
        "attention_maps": _column_has_value(model_sites, "tensor_type", "attention"),
        "action_chunks": "action_chunks" in array_names,
        "action_generation": "generation_actions" in array_names,
        "architecture_graph": len(model_sites) > 0,
        "probe_artifacts": artifact_counts.get("probe_suite", 0) > 0,
        "intervention_artifacts": artifact_counts.get("intervention_run", 0) > 0,
    }
    return {
        "root": str(root),
        "episode_count": int(len(episodes)),
        "activation_sites": int(len(model_sites)),
        "capabilities": {
            "available": sorted(name for name, available in flags.items() if available),
            "flags": flags,
            "camera_names": camera_names,
            "model_families": _model_families_from_index(episodes),
            "model_site_prefixes": model_site_prefixes,
        },
        "artifacts": {"total": int(len(artifacts)), "counts": artifact_counts},
        "probes": {
            "total_predictions": int(len(probes)),
            "total_episode_records": int(len(probe_episodes)),
            "probe_count": int(probes["probe_id"].nunique()) if "probe_id" in probes else 0,
        },
        "index": {
            "schema_version": manifest.get("schema_version", INDEX_SCHEMA_VERSION),
            "dataset_fingerprint": manifest.get("dataset_fingerprint"),
            "indexed_episode_count": manifest.get("indexed_episode_count"),
            "updated_utc": manifest.get("updated_utc"),
            "tables": manifest.get("tables", {}),
        },
    }


def indexed_episodes_payload(root: Path, query: Mapping[str, list[str]]) -> dict[str, Any]:
    limit = _clamped_int(_query_value(query, "limit"), DEFAULT_EPISODE_LIMIT, 1, MAX_EPISODE_LIMIT)
    offset = _clamped_int(_query_value(query, "offset"), 0, 0, 10**12)
    sort = _query_value(query, "sort") or "episode_index"
    params, where_sql, source_sql = _episode_query_parts(root, query)
    sort_sql = _episode_sort_sql(sort, probe_query=bool(_query_value(query, "probe_id")))
    con = duckdb.connect(database=":memory:")
    try:
        total = con.execute(
            f"SELECT count(*) FROM ({source_sql}) AS source {where_sql}",
            params,
        ).fetchone()[0]
        frame = con.execute(
            f"""
            SELECT *
            FROM ({source_sql}) AS source
            {where_sql}
            ORDER BY {sort_sql}, trace_id
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).df()
        rows = [_episode_payload_from_row(row) for row in frame.to_dict("records")]
        return {
            "episodes": rows,
            "total": int(total),
            "limit": limit,
            "offset": offset,
            "next_offset": offset + limit if offset + limit < int(total) else None,
            "facets": _episode_facets(con, root),
            "sort": sort_sql,
        }
    finally:
        con.close()


def indexed_episode_neighbors_payload(root: Path, trace_id: str) -> dict[str, Any]:
    path = _quote_literal(str(root / EPISODE_INDEX))
    con = duckdb.connect(database=":memory:")
    try:
        rows = con.execute(
            f"""
            WITH ordered AS (
              SELECT trace_id,
                     lag(trace_id) OVER (ORDER BY episode_index, trace_id) AS previous_trace_id,
                     lead(trace_id) OVER (ORDER BY episode_index, trace_id) AS next_trace_id
              FROM read_parquet({path})
            )
            SELECT *
            FROM ordered
            WHERE trace_id = ?
            """,
            [trace_id],
        ).df()
    finally:
        con.close()
    if rows.empty:
        raise KeyError(trace_id)
    row = rows.iloc[0].to_dict()
    return {
        "trace_id": trace_id,
        "previous_trace_id": _none_if_missing(row.get("previous_trace_id")),
        "next_trace_id": _none_if_missing(row.get("next_trace_id")),
    }


def indexed_artifacts_payload(root: Path) -> dict[str, Any]:
    artifacts = _read_table(root / ARTIFACT_INDEX)
    if artifacts.empty:
        return {"artifacts": [], "counts": {}, "total": 0}
    if "created_utc" in artifacts:
        artifacts = artifacts.sort_values("created_utc", ascending=False, na_position="last")
    records = [_artifact_payload(row) for row in artifacts.to_dict("records")]
    return {
        "artifacts": records,
        "counts": _value_counts(artifacts, "artifact_type"),
        "total": len(records),
    }


def indexed_artifact_summary(root: Path) -> dict[str, Any]:
    artifacts = _read_table(root / ARTIFACT_INDEX)
    return {"total": int(len(artifacts)), "counts": _value_counts(artifacts, "artifact_type")}


def counterfactual_pairs_from_index(root: Path) -> dict[str, Any]:
    episodes = _read_table(root / EPISODE_INDEX)
    groups: dict[str, list[dict[str, Any]]] = {}
    group_meta: dict[str, dict[str, Any]] = {}
    for row in episodes.to_dict("records"):
        metadata = _json_parse(row.get("metadata")) or {}
        if not isinstance(metadata, Mapping):
            continue
        counterfactual = metadata.get("counterfactual")
        counterfactual = dict(counterfactual) if isinstance(counterfactual, Mapping) else {}
        group_id = str(
            counterfactual.get("group_id") or metadata.get("counterfactual_group_id") or ""
        )
        if not group_id:
            continue
        groups.setdefault(group_id, []).append(
            {
                "trace_id": row.get("trace_id"),
                "episode_id": row.get("episode_id"),
                "role": counterfactual.get("role") or metadata.get("counterfactual_role") or "",
                "pair_index": _json_scalar(
                    counterfactual.get("pair_index") or metadata.get("pair_index")
                ),
                "paired_trace_id": counterfactual.get("paired_trace_id")
                or metadata.get("paired_trace_id")
                or "",
                "target_object_id": counterfactual.get("target_object_id")
                or metadata.get("target_object_id")
                or "",
                "counterfactual_target_object_id": counterfactual.get(
                    "counterfactual_target_object_id"
                )
                or metadata.get("counterfactual_target_object_id")
                or "",
                "outcome": row.get("outcome"),
                "prompt": row.get("prompt"),
            }
        )
        group_meta.setdefault(
            group_id,
            {
                "group_id": group_id,
                "type": counterfactual.get("type") or metadata.get("counterfactual_type") or "",
                "changed_fields": counterfactual.get("changed_fields") or [],
                "matched_fields": counterfactual.get("matched_fields") or [],
            },
        )
    pairs = [{**group_meta[group_id], "members": members} for group_id, members in groups.items()]
    pairs.sort(key=lambda item: str(item.get("group_id") or ""))
    return {"pairs": pairs, "count": len(pairs)}


def _episode_query_parts(
    root: Path,
    query: Mapping[str, list[str]],
) -> tuple[list[Any], str, str]:
    episodes = f"read_parquet({_quote_literal(str(root / EPISODE_INDEX))})"
    source_sql = f"SELECT * FROM {episodes}"
    params: list[Any] = []
    probe_id = _query_value(query, "probe_id")
    if probe_id:
        probe_episodes = f"read_parquet({_quote_literal(str(root / PROBE_EPISODE_INDEX))})"
        source_sql = f"""
        SELECT e.*, p.probe_available, p.probe_row_count, p.probe_split, p.probe_split_category,
               p.probe_confidence, p.probe_correct, p.probe_actual, p.probe_predicted,
               p.probe_correct_rate, p.probe_model, p.probe_feature, p.probe_policy_call_index
        FROM {episodes} AS e
        LEFT JOIN (
          SELECT trace_id,
                 true AS probe_available,
                 row_count AS probe_row_count,
                 split AS probe_split,
                 split_category AS probe_split_category,
                 try_cast(confidence AS DOUBLE) AS probe_confidence,
                 try_cast(correct_rate AS DOUBLE) AS probe_correct_rate,
                 correct AS probe_correct,
                 actual AS probe_actual,
                 predicted AS probe_predicted,
                 model AS probe_model,
                 feature AS probe_feature,
                 try_cast(policy_call_index AS INTEGER) AS probe_policy_call_index
          FROM {probe_episodes}
          WHERE probe_id = ?
        ) AS p
        ON e.trace_id = p.trace_id
        """
        params.append(probe_id)
    clauses: list[str] = []
    for key in ("dataset_id", "benchmark", "task_id", "outcome", "profile"):
        value = _query_value(query, key)
        if value and value != "all":
            clauses.append(f"cast({key} AS VARCHAR) = ?")
            params.append(value)
    q = _query_value(query, "q")
    if q:
        haystack = " || ' ' || ".join(
            f"lower(coalesce(cast({column} AS VARCHAR), ''))" for column in SEARCH_COLUMNS
        )
        clauses.append(f"contains({haystack}, ?)")
        params.append(q.strip().lower())
    if probe_id:
        _append_probe_filters(clauses, params, query)
    where_sql = "WHERE " + " AND ".join(clauses) if clauses else ""
    return params, where_sql, source_sql


def _append_probe_filters(
    clauses: list[str],
    params: list[Any],
    query: Mapping[str, list[str]],
) -> None:
    split = _query_value(query, "probe_split")
    if split and split != "all":
        clauses.append("probe_split_category = ?")
        params.append(split)
    prediction = _query_value(query, "probe_prediction")
    if prediction == "scored":
        clauses.append("probe_available = true")
    elif prediction == "unscored":
        clauses.append("probe_available IS NULL")
    elif prediction == "correct":
        clauses.append("probe_correct = true")
    elif prediction == "incorrect":
        clauses.append("probe_correct = false")
    elif prediction == "high_confidence":
        clauses.append("probe_confidence >= 0.8")
    elif prediction == "low_confidence":
        clauses.append("probe_available = true AND probe_confidence < 0.8")
    preset = _query_value(query, "probe_cohort_preset")
    if preset == "needs_review":
        clauses.append(
            "probe_available = true AND "
            "(probe_split_category IN ('validation', 'test') OR probe_correct = false "
            "OR probe_confidence IS NULL OR probe_confidence < 0.65)"
        )
    elif preset == "heldout_wrong":
        clauses.append("probe_split_category IN ('validation', 'test') AND probe_correct = false")
    elif preset == "confident_wrong":
        clauses.append("probe_correct = false AND probe_confidence >= 0.8")
    elif preset == "heldout_scored":
        clauses.append("probe_split_category IN ('validation', 'test') AND probe_available = true")
    elif preset == "train_sanity":
        clauses.append("probe_split_category = 'train'")


def _episode_sort_sql(sort: str, *, probe_query: bool) -> str:
    if sort == "probe_interest" and probe_query:
        return """
            (
              CASE
                WHEN probe_split_category = 'test' THEN 110
                WHEN probe_split_category = 'validation' THEN 80
                WHEN probe_split_category = 'train' THEN -90
                ELSE -10
              END
              + CASE WHEN probe_available = true THEN 0 ELSE -40 END
              + CASE WHEN probe_correct = false THEN 260 ELSE 0 END
              + CASE
                  WHEN probe_correct = false AND probe_confidence >= 0.8 THEN 140
                  ELSE 0
                END
              + CASE
                  WHEN probe_correct IS NULL AND probe_available = true THEN 45
                  ELSE 0
                END
              + least(40, coalesce(probe_row_count, 0) * 4)
            ) DESC,
            episode_index
        """
    return EPISODE_SORT_COLUMNS.get(sort, "episode_index")


def _episode_payload_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {}
    for key in ("dataset_id", "benchmark", "seed"):
        value = row.get(key)
        if value not in {None, ""}:
            metadata[key] = value
    profile = row.get("profile")
    if profile not in {None, ""}:
        metadata["profile"] = profile
        metadata["capture_profile"] = profile
    payload = {
        "trace_id": str(row.get("trace_id") or ""),
        "episode_id": str(row.get("episode_id") or ""),
        "episode_index": _int_or_none(row.get("episode_index")),
        "task_id": _none_if_missing(row.get("task_id")),
        "prompt": _none_if_missing(row.get("prompt")),
        "model_id": _none_if_missing(row.get("model_id")),
        "env_id": _none_if_missing(row.get("env_id")),
        "robot_id": _none_if_missing(row.get("robot_id")),
        "outcome": _none_if_missing(row.get("outcome")),
        "length": _int_or_none(row.get("length")),
        "schema_version": _none_if_missing(row.get("schema_version")),
        "metadata": metadata,
    }
    probe_record = _probe_record_from_row(row)
    if probe_record:
        payload["probe_record"] = probe_record
    return payload


def _probe_record_from_row(row: Mapping[str, Any]) -> dict[str, Any] | None:
    available = _bool_or_none(row.get("probe_available"))
    correct = _bool_or_none(row.get("probe_correct"))
    if available is not True and _none_if_missing(row.get("probe_row_count")) is None:
        return None
    correct_rate = _float_or_none(row.get("probe_correct_rate"))
    if correct_rate is None:
        correct_rate = 1.0 if correct is True else 0.0 if correct is False else None
    return {
        "trace_id": str(row.get("trace_id") or ""),
        "split": _none_if_missing(row.get("probe_split")),
        "split_category": _none_if_missing(row.get("probe_split_category")),
        "available": available is True,
        "row_count": _int_or_none(row.get("probe_row_count")) or 0,
        "best_row_count": _int_or_none(row.get("probe_row_count")) or 0,
        "actual": _json_scalar(row.get("probe_actual")),
        "predicted": _json_scalar(row.get("probe_predicted")),
        "confidence": _float_or_none(row.get("probe_confidence")),
        "correct": correct,
        "correct_rate": correct_rate,
        "model": _none_if_missing(row.get("probe_model")),
        "feature": _none_if_missing(row.get("probe_feature")),
        "policy_call_index": _int_or_none(row.get("probe_policy_call_index")),
    }


def _episode_facets(con: duckdb.DuckDBPyConnection, root: Path) -> dict[str, list[dict[str, Any]]]:
    facets: dict[str, list[dict[str, Any]]] = {}
    path = _quote_literal(str(root / EPISODE_INDEX))
    for column in ("dataset_id", "benchmark", "task_id", "outcome", "profile"):
        frame = con.execute(
            f"""
            SELECT cast({column} AS VARCHAR) AS value, count(*) AS count
            FROM read_parquet({path})
            WHERE {column} IS NOT NULL AND cast({column} AS VARCHAR) != ''
            GROUP BY value
            ORDER BY value
            """
        ).df()
        facets[column] = [
            {"value": str(row["value"]), "count": int(row["count"])}
            for row in frame.to_dict("records")
        ]
    return facets


def _artifact_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(key): _json_scalar(value) for key, value in row.items()}
    for key in ["selector", "method", "metrics", "arrays", "display", "tags", "source_trace_ids"]:
        payload[key] = _jsonable(_json_parse(payload.get(key)))
    return payload


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame:
        return {}
    counts = frame[column].dropna().astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items() if str(key)}


def _read_table(path: Path) -> pd.DataFrame:
    return read_optional_parquet(path, context="indexed dashboard")


def _read_index_manifest(root: Path) -> dict[str, Any]:
    try:
        return json.loads((root / INDEX_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _json_list_union(values: pd.Series) -> list[str]:
    out: set[str] = set()
    for value in values.dropna().tolist():
        parsed = _json_parse(value)
        if isinstance(parsed, list):
            out.update(str(item) for item in parsed if str(item).strip())
    return sorted(out)


def _model_families_from_index(episodes: pd.DataFrame) -> list[str]:
    families: set[str] = set()
    for value in episodes.get("metadata", pd.Series(dtype=object)).dropna().tolist():
        metadata = _json_parse(value)
        if not isinstance(metadata, Mapping):
            continue
        _add_model_family(families, metadata.get("model_family"))
        model = metadata.get("model")
        if isinstance(model, Mapping):
            _add_model_family(families, model.get("model_family"))
        descriptor = metadata.get("model_descriptor")
        if isinstance(descriptor, Mapping):
            _add_model_family(families, descriptor.get("model_family"))
    return sorted(families)


def _add_model_family(families: set[str], value: Any) -> None:
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _add_model_family(families, item)
        return
    text = str(value or "").strip()
    if text:
        families.add(text)


def _numeric_sum(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _column_has_value(frame: pd.DataFrame, column: str, value: str) -> bool:
    return column in frame and bool((frame[column].astype(str) == value).any())


def _has_indexed_image_token_maps(model_sites: pd.DataFrame, token_space_count: float) -> bool:
    if model_sites.empty or "axes" not in model_sites or "token_kind" not in model_sites:
        return False
    token_kinds = model_sites["token_kind"].astype(str)
    axes_text = model_sites["axes"].astype(str)
    has_token_channel_site = axes_text.str.contains("token", na=False) & axes_text.str.contains(
        "channel", na=False
    )
    has_direct_image_site = token_kinds.isin({"image", "image_patch", "image_patches"})
    has_prefix_site_with_layout = token_kinds.isin({"prefix", "image"}) & (token_space_count > 0)
    return bool(
        (has_token_channel_site & (has_direct_image_site | has_prefix_site_with_layout)).any()
    )


def _query_value(query: Mapping[str, list[str]], name: str) -> str:
    raw = (query.get(name) or [""])[0]
    return str(raw).strip()


def _clamped_int(raw: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        if value.lower() in {"true", "1"}:
            return True
        if value.lower() in {"false", "0"}:
            return False
    return None


def _none_if_missing(value: Any) -> Any | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value)
    return None if text.lower() in {"", "nan", "none", "null"} else value


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


__all__ = [
    "counterfactual_pairs_from_index",
    "indexed_artifact_summary",
    "indexed_artifacts_payload",
    "indexed_dataset_payload",
    "indexed_episode_neighbors_payload",
    "indexed_episodes_payload",
]
