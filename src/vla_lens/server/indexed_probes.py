"""Indexed probe payloads backed by local Parquet tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from vla_lens.dataset.index import ARTIFACT_INDEX, PROBE_EPISODE_INDEX
from vla_lens.server.common import _json_parse, _json_scalar, _jsonable
from vla_lens.server.indexed import (
    _bool_or_none,
    _clamped_int,
    _float_or_none,
    _int_or_none,
    _none_if_missing,
    _query_value,
    _read_table,
    _value_counts,
    indexed_episodes_payload,
)


def indexed_probe_index_payload(root: Path) -> dict[str, Any]:
    artifacts = _read_table(root / ARTIFACT_INDEX)
    probe_episodes = _read_table(root / PROBE_EPISODE_INDEX)
    probes: list[dict[str, Any]] = []
    for artifact in artifacts.loc[
        artifacts.get("artifact_type", pd.Series(dtype=object)).astype(str) == "probe_suite"
    ].to_dict("records"):
        probe_id = str(artifact.get("artifact_id") or "")
        rows = (
            probe_episodes.loc[probe_episodes["probe_id"].astype(str) == probe_id]
            if not probe_episodes.empty and "probe_id" in probe_episodes
            else pd.DataFrame()
        )
        probes.append(_probe_summary_payload(artifact, rows))
    probes.sort(key=lambda item: (-_probe_review_score(item), str(item.get("name") or "")))
    return {
        "probes": probes,
        "total": len(probes),
        "trace_count": _episode_count(root),
        "split_source": None,
    }


def indexed_probe_evidence_payload(
    root: Path,
    probe_id: str,
    query: Mapping[str, list[str]],
) -> dict[str, Any]:
    limit = _clamped_int(_query_value(query, "limit"), 12, 1, 100)
    evidence_query = {key: list(value) for key, value in query.items()}
    evidence_query["probe_id"] = [probe_id]
    evidence_query["limit"] = [str(limit)]
    evidence_query.setdefault("probe_prediction", ["scored"])
    evidence_query.setdefault("sort", ["probe_interest"])
    episodes_payload = indexed_episodes_payload(root, evidence_query)
    probe = next(
        (
            item
            for item in indexed_probe_index_payload(root)["probes"]
            if str(item.get("artifact_id")) == probe_id
        ),
        None,
    )
    if probe is None:
        raise KeyError(probe_id)
    return {
        "probe": probe,
        "episodes": episodes_payload["episodes"],
        "total": episodes_payload["total"],
        "limit": limit,
    }


def indexed_episode_probes_payload(root: Path, query: Mapping[str, list[str]]) -> dict[str, Any]:
    trace_id = _query_value(query, "trace_id") or ""
    artifacts = _read_table(root / ARTIFACT_INDEX)
    probe_episodes = _read_table(root / PROBE_EPISODE_INDEX)
    probes: list[dict[str, Any]] = []
    for artifact in artifacts.loc[
        artifacts.get("artifact_type", pd.Series(dtype=object)).astype(str) == "probe_suite"
    ].to_dict("records"):
        probe_id = str(artifact.get("artifact_id") or "")
        rows = (
            probe_episodes.loc[
                (probe_episodes["probe_id"].astype(str) == probe_id)
                & (probe_episodes["trace_id"].astype(str) == trace_id)
            ]
            if not probe_episodes.empty and {"probe_id", "trace_id"} <= set(probe_episodes.columns)
            else pd.DataFrame()
        )
        probes.append(_episode_probe_payload(artifact, rows))
    return {
        "trace_id": trace_id,
        "probes": probes,
        "available_count": sum(1 for probe in probes if probe["available"]),
        "total": len(probes),
    }


def _probe_summary_payload(artifact: Mapping[str, Any], rows: pd.DataFrame) -> dict[str, Any]:
    metrics = _json_parse(artifact.get("metrics")) or {}
    prediction_summary = _probe_prediction_summary(rows)
    split_summary = _value_counts(rows, "split_category")
    stats = {
        "scored": prediction_summary.get("scored", 0),
        "unscored": max(0, _episode_count_from_rows(rows) - prediction_summary.get("scored", 0)),
        "correct": prediction_summary.get("correct", 0),
        "wrong": prediction_summary.get("incorrect", 0),
        "heldoutScored": _count_probe_rows(rows, heldout=True, scored=True),
        "heldoutWrong": _count_probe_rows(rows, heldout=True, correct=False),
        "confidentWrong": _count_probe_rows(rows, correct=False, min_confidence=0.8),
        "train": int(split_summary.get("train", 0)),
        "validation": int(split_summary.get("validation", 0)),
        "test": int(split_summary.get("test", 0)),
    }
    return {
        "artifact_id": str(artifact.get("artifact_id") or ""),
        "name": str(artifact.get("name") or artifact.get("artifact_id") or ""),
        "target": metrics.get("target") if isinstance(metrics, Mapping) else None,
        "best_model": metrics.get("best_model") if isinstance(metrics, Mapping) else None,
        "best_feature": metrics.get("best_feature") if isinstance(metrics, Mapping) else None,
        "best_score": _float_or_none(metrics.get("best_score"))
        if isinstance(metrics, Mapping)
        else None,
        "best_delta": _float_or_none(metrics.get("best_delta"))
        if isinstance(metrics, Mapping)
        else None,
        "split_summary": split_summary,
        "prediction_summary": prediction_summary,
        "review_stats": stats,
    }


def _episode_probe_payload(artifact: Mapping[str, Any], rows: pd.DataFrame) -> dict[str, Any]:
    metrics = _json_parse(artifact.get("metrics")) or {}
    display = _json_parse(artifact.get("display")) or {}
    summary = _indexed_probe_episode_summary(rows, metrics)
    row_payloads = _indexed_probe_prediction_rows(rows)
    return {
        "artifact_id": str(artifact.get("artifact_id") or ""),
        "name": str(artifact.get("name") or artifact.get("artifact_id") or ""),
        "target": metrics.get("target") if isinstance(metrics, Mapping) else None,
        "metrics": _jsonable(metrics if isinstance(metrics, Mapping) else {}),
        "best_result": _jsonable(
            display.get("best_result_details")
            if isinstance(display, Mapping)
            else {}
        )
        or {},
        "target_distribution": _jsonable(
            display.get("target_distribution")
            if isinstance(display, Mapping)
            else {}
        )
        or {},
        "episode_summary": summary,
        "rows": row_payloads,
        "row_count": _int_or_none(rows.iloc[0].get("row_count")) if not rows.empty else 0,
        "available": bool(len(rows)),
    }


def _indexed_probe_episode_summary(
    rows: pd.DataFrame,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if rows.empty:
        return {}
    row = rows.iloc[0].to_dict()
    correct = _bool_or_none(row.get("correct"))
    correct_rate = _float_or_none(row.get("correct_rate"))
    if correct_rate is None:
        correct_rate = 1.0 if correct is True else 0.0 if correct is False else None
    confidence = _float_or_none(row.get("confidence"))
    best_row = _indexed_probe_prediction_rows(rows.head(1))
    return {
        "actual": _json_scalar(row.get("actual")),
        "predicted": _json_scalar(row.get("predicted")),
        "confidence": confidence,
        "correct": correct,
        "correct_rate": correct_rate,
        "all_cell_correct_rate": correct_rate,
        "all_cell_mean_confidence": confidence,
        "best_feature": _none_if_missing(row.get("feature"))
        or str(metrics.get("best_feature") or ""),
        "best_model": _none_if_missing(row.get("model"))
        or str(metrics.get("best_model") or ""),
        "best_row": best_row[0] if best_row else {},
    }


def _indexed_probe_prediction_rows(rows: pd.DataFrame) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    out: list[dict[str, Any]] = []
    for row in rows.to_dict("records"):
        target = _none_if_missing(row.get("target_name")) or _none_if_missing(row.get("target"))
        actual = _json_scalar(row.get("target_value", row.get("actual")))
        predicted = _json_scalar(row.get("prediction_value", row.get("predicted")))
        out.append(
            {
                "trace_id": _none_if_missing(row.get("trace_id")),
                "episode_id": _none_if_missing(row.get("episode_id")),
                "task_id": _none_if_missing(row.get("task_id")),
                "split": _none_if_missing(row.get("split")),
                "target_name": target,
                "target_value": actual,
                "actual": actual,
                "predicted": predicted,
                "prediction_value": predicted,
                "confidence": _float_or_none(row.get("confidence")),
                "correct": _bool_or_none(row.get("correct")),
                "model": _none_if_missing(row.get("model")),
                "feature": _none_if_missing(row.get("feature")),
                "layer": _int_or_none(row.get("layer")),
                "policy_call_index": _int_or_none(row.get("policy_call_index")),
                "timestep": _int_or_none(row.get("timestep")),
                "target_timestep": _int_or_none(row.get("target_timestep")),
                "generation_step": _json_scalar(row.get("generation_step")),
                "model_site_id": _none_if_missing(row.get("model_site_id")),
                "token_space_id": _none_if_missing(row.get("token_space_id")),
                "eval_split": _none_if_missing(row.get("eval_split"))
                or _none_if_missing(row.get("split")),
                "primary_metric": _none_if_missing(row.get("primary_metric")),
            }
        )
    return out


def _probe_prediction_summary(rows: pd.DataFrame) -> dict[str, int]:
    if rows.empty:
        return {"scored": 0, "unscored": 0, "correct": 0, "incorrect": 0, "unknown": 0}
    correct = rows.get("correct", pd.Series(dtype=object))
    return {
        "scored": int(len(rows)),
        "unscored": 0,
        "correct": int((correct == True).sum()),  # noqa: E712
        "incorrect": int((correct == False).sum()),  # noqa: E712
        "unknown": int((correct.isna()).sum()) if hasattr(correct, "isna") else 0,
    }


def _count_probe_rows(
    rows: pd.DataFrame,
    *,
    heldout: bool | None = None,
    scored: bool | None = None,
    correct: bool | None = None,
    min_confidence: float | None = None,
) -> int:
    if rows.empty:
        return 0
    mask = pd.Series(True, index=rows.index)
    if heldout is True and "split_category" in rows:
        mask = mask & rows["split_category"].astype(str).isin({"validation", "test"})
    if scored is True:
        mask = mask & rows["trace_id"].notna()
    if correct is not None and "correct" in rows:
        mask = mask & (rows["correct"] == correct)
    if min_confidence is not None and "confidence" in rows:
        mask = mask & (pd.to_numeric(rows["confidence"], errors="coerce") >= min_confidence)
    return int(mask.sum())


def _probe_review_score(probe: Mapping[str, Any]) -> float:
    stats = probe.get("review_stats") if isinstance(probe.get("review_stats"), Mapping) else {}
    return (
        float(stats.get("heldoutWrong") or 0) * 80
        + float(stats.get("confidentWrong") or 0) * 70
        + float(stats.get("heldoutScored") or 0) * 8
        + float(stats.get("wrong") or 0) * 12
        + float(probe.get("best_delta") or 0) * 120
        + float(probe.get("best_score") or 0) * 12
    )


def _episode_count(root: Path) -> int:
    from vla_lens.dataset.index import EPISODE_INDEX

    return int(len(_read_table(root / EPISODE_INDEX)))


def _episode_count_from_rows(rows: pd.DataFrame) -> int:
    if rows.empty or "trace_id" not in rows:
        return 0
    return int(rows["trace_id"].nunique())


__all__ = [
    "indexed_episode_probes_payload",
    "indexed_probe_evidence_payload",
    "indexed_probe_index_payload",
]
