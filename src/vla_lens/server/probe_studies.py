"""Probe-study payloads for research-question oriented probe browsing."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from vla_lens.dataset.index import EPISODE_INDEX
from vla_lens.server.common import _jsonable
from vla_lens.server.indexed import (
    DEFAULT_EPISODE_LIMIT,
    MAX_EPISODE_LIMIT,
    _clamped_int,
    _episode_payload_from_row,
    _query_value,
    _read_table,
)
from vla_lens.server.probe_study_diagnostics import (
    _artifact_for_id,
    _control_payloads,
    _counts,
    _diagnostics,
    _error_examples,
    _first_present,
    _primary_target,
    _readouts_from_artifact,
    _readouts_from_diagnostics,
    _records_with_target,
    _study_id,
    _study_name,
    _study_targets,
)
from vla_lens.server.probe_study_formatting import (
    _clean_bool,
    _clean_float,
    _clean_int,
    _clean_scalar,
    _input_label,
    _load_artifact_or_record,
    _mapping,
    _objective_label,
    _optional_text,
    _output_label,
    _prediction_label,
    _question_label,
    _split_category,
    _training_summary,
)
from vla_lens.traces import TraceDataset

EPISODE_SEARCH_COLUMNS = (
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

EPISODE_FILTER_COLUMNS = ("dataset_id", "benchmark", "task_id", "outcome", "profile")
PROBE_READOUT_SORT_COLUMNS = {
    "episode_index": "episode_index",
    "trace_id": "trace_id",
    "task_id": "task_id",
    "outcome": "outcome",
    "length": "length",
}


def probe_studies_payload(dataset: TraceDataset) -> dict[str, Any]:
    """Return probe-suite artifacts as studies with first-class readouts."""

    studies: list[dict[str, Any]] = []
    artifacts = dataset.artifact_index
    if artifacts.empty or "artifact_type" not in artifacts:
        return {"studies": [], "total": 0}

    probe_rows = artifacts.loc[artifacts["artifact_type"].astype(str) == "probe_suite"]
    if "created_utc" in probe_rows:
        probe_rows = probe_rows.sort_values("created_utc", ascending=False, na_position="last")

    for record in probe_rows.to_dict("records"):
        artifact = _load_artifact_or_record(dataset, record)
        artifact_id = str(artifact.get("artifact_id") or record.get("artifact_id") or "")
        summary, tables = _diagnostics(dataset, artifact_id, artifact)
        metrics = _mapping(artifact.get("metrics"))
        display = _mapping(artifact.get("display"))
        primary_target = _primary_target(summary, metrics, display)
        targets = _study_targets(tables, primary_target)
        for target in targets:
            studies.append(
                _study_payload(
                    dataset,
                    record,
                    artifact=artifact,
                    summary=summary,
                    tables=tables,
                    primary_target=primary_target,
                    target=target,
                    family_count=len(targets),
                )
            )
    return {"studies": studies, "total": len(studies)}


def probe_study_episodes_payload(
    dataset: TraceDataset,
    artifact_id: str,
    query: Mapping[str, list[str]],
) -> dict[str, Any]:
    """Return episode rows scoped to one diagnostic probe readout."""

    artifact = _artifact_for_id(dataset, artifact_id)
    summary, tables = _diagnostics(dataset, artifact_id, artifact)
    frame = tables["errors"]
    target = (
        _query_value(query, "target")
        or _optional_text(summary.get("target"))
        or _optional_text(_mapping(artifact.get("metrics")).get("target"))
    )
    layer = _query_value(query, "layer")
    split = _query_value(query, "split")
    limit = _clamped_int(_query_value(query, "limit"), DEFAULT_EPISODE_LIMIT, 1, MAX_EPISODE_LIMIT)
    offset = _clamped_int(_query_value(query, "offset"), 0, 0, 10**12)
    sort = _query_value(query, "sort") or "probe_interest"
    episodes = _read_table(dataset.root / EPISODE_INDEX)

    if frame.empty:
        return _empty_readout_episode_page(
            artifact_id,
            target,
            layer,
            split,
            limit,
            offset,
            sort,
            _episode_facets_from_frame(episodes),
            "This probe artifact does not include diagnostic episode rows.",
        )

    primary_target = _optional_text(summary.get("target")) or _optional_text(
        _mapping(artifact.get("metrics")).get("target")
    )
    if "target" not in frame.columns:
        if target and primary_target and target != primary_target:
            return _empty_readout_episode_page(
                artifact_id,
                target,
                layer,
                split,
                limit,
                offset,
                sort,
                _episode_facets_from_frame(episodes),
                f"Episode table rows were exported only for {primary_target}.",
            )
        frame = frame.copy()
        frame["target"] = primary_target or target
    elif target:
        frame = frame.loc[frame["target"].astype(str) == target]

    split_summary_rows = _filter_readout_layer_rows(frame, layer)
    scoped = _filter_readout_rows(frame, query)
    scoped = _attach_episode_index(scoped, episodes)
    scoped = _filter_episode_metadata(scoped, query)
    representatives = _representative_episode_rows(scoped)
    representatives = _sort_readout_episode_rows(representatives, sort)
    summary = _readout_episode_summary(
        scoped,
        representatives,
        split_summary_rows=split_summary_rows,
    )

    total = int(len(representatives))
    page = representatives.iloc[offset : offset + limit]
    rows = [_readout_episode_payload(row) for row in page.to_dict("records")]
    return {
        "artifact_id": artifact_id,
        "available": True,
        "reason": "",
        "target": target or primary_target or None,
        "layer": _clean_scalar(layer) if layer else None,
        "split": split or None,
        "summary": summary,
        "episodes": rows,
        "total": total,
        "limit": limit,
        "offset": offset,
        "next_offset": offset + limit if offset + limit < total else None,
        "facets": _episode_facets_from_frame(episodes),
        "sort": sort,
    }


def _study_payload(
    dataset: TraceDataset,
    record: Mapping[str, Any],
    *,
    artifact: Mapping[str, Any] | None = None,
    summary: Mapping[str, Any] | None = None,
    tables: Mapping[str, pd.DataFrame] | None = None,
    primary_target: str | None = None,
    target: str | None = None,
    family_count: int = 1,
) -> dict[str, Any]:
    artifact = artifact or _load_artifact_or_record(dataset, record)
    artifact_id = str(artifact.get("artifact_id") or record.get("artifact_id") or "")
    artifact_name = str(artifact.get("name") or artifact_id)
    metrics = _mapping(artifact.get("metrics"))
    method = _mapping(artifact.get("method"))
    selector = _mapping(artifact.get("selector"))
    display = _mapping(artifact.get("display"))
    summary = summary or {}
    tables = tables or _diagnostics(dataset, artifact_id, artifact)[1]
    primary_target = (
        primary_target if primary_target is not None else _primary_target(summary, metrics, display)
    )
    target = target if target is not None else primary_target
    study_id = _study_id(artifact_id, target)
    name = _study_name(artifact_name, target, family_count)

    readouts, skipped = _readouts_from_diagnostics(tables, target, primary_target, summary)
    source = "diagnostics"
    if not readouts and not skipped:
        readouts = _readouts_from_artifact(display, metrics, target, summary)
        source = "artifact"

    lead_time = _records_with_target(tables["lead_time"], target, primary_target=primary_target)
    per_class = _records_with_target(tables["per_class"], target, primary_target=primary_target)
    confusion = _records_with_target(
        tables["confusion"],
        target,
        primary_target=primary_target,
        limit=600,
    )
    class_support = _records_with_target(
        tables["class_support"],
        target,
        primary_target=primary_target,
    )
    error_examples = _error_examples(tables["errors"], target=target, primary_target=primary_target)
    controls = _control_payloads(
        summary,
        tables["selection_null"],
        readouts,
        target=target,
        primary_target=primary_target,
    )

    return {
        "study_id": study_id,
        "artifact_id": artifact_id,
        "artifact_type": "probe_suite",
        "source_artifact_id": artifact_id,
        "source_artifact_name": artifact_name,
        "name": name,
        "created_utc": _clean_scalar(artifact.get("created_utc")),
        "target": target or None,
        "question_label": _question_label(target),
        "prediction": _prediction_label(target),
        "input": _input_label(selector, method),
        "output": _output_label(summary, metrics),
        "objective": _objective_label(method, summary, metrics),
        "training_summary": _training_summary(method, summary, metrics),
        "diagnostics_available": any(not table.empty for table in tables.values()) or bool(summary),
        "source": source,
        "counts": _counts(
            summary,
            readouts,
            skipped,
            tables,
            target=target,
            primary_target=primary_target,
        ),
        "summary": _jsonable(summary),
        "readouts": _jsonable(readouts),
        "skipped_readouts": _jsonable(skipped),
        "controls": _jsonable(controls),
        "lead_time": _jsonable(lead_time),
        "per_class": _jsonable(per_class),
        "confusion": _jsonable(confusion),
        "class_support": _jsonable(class_support),
        "error_examples": _jsonable(error_examples),
    }


def _empty_readout_episode_page(
    artifact_id: str,
    target: str,
    layer: str,
    split: str,
    limit: int,
    offset: int,
    sort: str,
    facets: dict[str, list[dict[str, Any]]],
    reason: str,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id,
        "available": False,
        "reason": reason,
        "target": target or None,
        "layer": _clean_scalar(layer) if layer else None,
        "split": split or None,
        "summary": _empty_readout_episode_summary(),
        "episodes": [],
        "total": 0,
        "limit": limit,
        "offset": offset,
        "next_offset": None,
        "facets": facets,
        "sort": sort,
    }


def _filter_readout_rows(frame: pd.DataFrame, query: Mapping[str, list[str]]) -> pd.DataFrame:
    out = frame.copy()
    layer = _query_value(query, "layer")
    if layer and layer != "all" and "layer" in out:
        out = out.loc[out["layer"].map(_optional_text) == layer]
    split = _query_value(query, "split")
    if split and split != "all" and "split" in out:
        out = out.loc[out["split"].astype(str) == split]
    out["split_category"] = out.get("split", pd.Series(dtype=object)).map(_split_category)

    split_category = _query_value(query, "split_category")
    if split_category and split_category != "all":
        out = out.loc[out["split_category"] == split_category]

    correct = out.get("correct", pd.Series(dtype=object)).map(_clean_bool)
    confidence = pd.to_numeric(out.get("confidence", pd.Series(dtype=object)), errors="coerce")
    available = out.get("predicted", pd.Series(dtype=object)).map(_clean_scalar).notna()
    prediction = _query_value(query, "prediction")
    if prediction == "scored":
        out = out.loc[available]
        correct = correct.loc[out.index]
        confidence = confidence.loc[out.index]
    elif prediction == "unscored":
        out = out.loc[~available]
        correct = correct.loc[out.index]
        confidence = confidence.loc[out.index]
    elif prediction == "correct":
        out = out.loc[correct.eq(True)]
        correct = correct.loc[out.index]
        confidence = confidence.loc[out.index]
    elif prediction == "incorrect":
        out = out.loc[correct.eq(False)]
        correct = correct.loc[out.index]
        confidence = confidence.loc[out.index]
    elif prediction == "high_confidence":
        out = out.loc[confidence >= 0.8]
        correct = correct.loc[out.index]
        confidence = confidence.loc[out.index]
    elif prediction == "low_confidence":
        out = out.loc[available & (confidence < 0.8)]
        correct = correct.loc[out.index]
        confidence = confidence.loc[out.index]

    preset = _query_value(query, "cohort_preset") or _query_value(query, "probe_cohort_preset")
    split_categories = out["split_category"].astype(str)
    available = out.get("predicted", pd.Series(dtype=object)).map(_clean_scalar).notna()
    correct = out.get("correct", pd.Series(dtype=object)).map(_clean_bool)
    confidence = pd.to_numeric(out.get("confidence", pd.Series(dtype=object)), errors="coerce")
    if preset == "needs_review":
        out = out.loc[
            split_categories.isin(["validation", "test"])
            | correct.eq(False)
            | confidence.isna()
            | (confidence < 0.65)
        ]
    elif preset == "heldout_wrong":
        out = out.loc[split_categories.isin(["validation", "test"]) & correct.eq(False)]
    elif preset == "confident_wrong":
        out = out.loc[correct.eq(False) & (confidence >= 0.8)]
    elif preset == "heldout_scored":
        out = out.loc[split_categories.isin(["validation", "test"]) & available]
    elif preset == "train_sanity":
        out = out.loc[split_categories == "train"]
    return out


def _filter_readout_layer_rows(frame: pd.DataFrame, layer: str) -> pd.DataFrame:
    out = frame.copy()
    if layer and layer != "all" and "layer" in out:
        out = out.loc[out["layer"].map(_optional_text) == layer]
    return out


def _attach_episode_index(frame: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if episodes.empty or "trace_id" not in episodes:
        return out
    episode_cols = [column for column in episodes.columns if column != "trace_id"]
    lookup = episodes.loc[:, ["trace_id", *episode_cols]].add_prefix("episode_")
    lookup = lookup.rename(columns={"episode_trace_id": "trace_id"})
    return out.merge(lookup, on="trace_id", how="left")


def _filter_episode_metadata(frame: pd.DataFrame, query: Mapping[str, list[str]]) -> pd.DataFrame:
    out = frame.copy()
    for column in EPISODE_FILTER_COLUMNS:
        value = _query_value(query, column)
        if not value or value == "all":
            continue
        series = _readout_episode_field(out, column).astype(str)
        out = out.loc[series == value]
    q = _query_value(query, "q")
    if q:
        haystack = pd.Series("", index=out.index, dtype=object)
        for column in EPISODE_SEARCH_COLUMNS:
            haystack = haystack + " " + _readout_episode_field(out, column).fillna("").astype(str)
        out = out.loc[haystack.str.lower().str.contains(q.strip().lower(), regex=False, na=False)]
    return out


def _representative_episode_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["_probe_row_count"] = out.groupby("trace_id")["trace_id"].transform("count")
    out["_probe_interest"] = out.apply(_readout_interest_score, axis=1)
    if "episode_episode_index" not in out:
        out["episode_episode_index"] = pd.NA
    out = out.sort_values(
        ["_probe_interest", "episode_episode_index", "trace_id"],
        ascending=[False, True, True],
        na_position="last",
    )
    return out.drop_duplicates("trace_id", keep="first")


def _sort_readout_episode_rows(frame: pd.DataFrame, sort: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if sort in {"lens_interest", "probe_interest"}:
        return out.sort_values(
            ["_probe_interest", "episode_episode_index", "trace_id"],
            ascending=[False, True, True],
            na_position="last",
        )
    column = PROBE_READOUT_SORT_COLUMNS.get(sort, "episode_index")
    field = f"episode_{column}" if f"episode_{column}" in out else column
    return out.sort_values([field, "trace_id"], ascending=[True, True], na_position="last")


def _readout_episode_summary(
    scoped_rows: pd.DataFrame,
    representatives: pd.DataFrame,
    *,
    split_summary_rows: pd.DataFrame | None = None,
) -> dict[str, Any]:
    split_counts = _split_summary_counts(
        split_summary_rows if split_summary_rows is not None else scoped_rows
    )
    if representatives.empty:
        summary = _empty_readout_episode_summary()
        summary["policy_call_count"] = int(len(scoped_rows))
        summary["split_counts"] = split_counts
        return summary
    correct = representatives.get("correct", pd.Series(dtype=object)).map(_clean_bool)
    confidence = pd.to_numeric(
        representatives.get("confidence", pd.Series(dtype=object)),
        errors="coerce",
    )
    available = representatives.get("predicted", pd.Series(dtype=object)).map(_clean_scalar).notna()
    wrong = correct.eq(False)
    return {
        "policy_call_count": int(len(scoped_rows)),
        "episode_count": int(len(representatives)),
        "scored": int(available.sum()),
        "unscored": int((~available).sum()),
        "correct": int(correct.eq(True).sum()),
        "wrong": int(wrong.sum()),
        "high_confidence": int((confidence >= 0.8).sum()),
        "high_conf_wrong": int((wrong & (confidence >= 0.8)).sum()),
        "split_counts": split_counts,
    }


def _empty_readout_episode_summary() -> dict[str, Any]:
    return {
        "policy_call_count": 0,
        "episode_count": 0,
        "scored": 0,
        "unscored": 0,
        "correct": 0,
        "wrong": 0,
        "high_confidence": 0,
        "high_conf_wrong": 0,
        "split_counts": {},
    }


def _split_summary_counts(rows: pd.DataFrame) -> dict[str, dict[str, int]]:
    if rows.empty or "split" not in rows:
        return {}
    out: dict[str, dict[str, int]] = {}
    for split, group in rows.groupby("split", dropna=False):
        split_name = _optional_text(split)
        if not split_name:
            continue
        correct = group.get("correct", pd.Series(dtype=object)).map(_clean_bool)
        confidence = pd.to_numeric(
            group.get("confidence", pd.Series(dtype=object)),
            errors="coerce",
        )
        available = group.get("predicted", pd.Series(dtype=object)).map(_clean_scalar).notna()
        wrong = correct.eq(False)
        out[split_name] = {
            "policy_call_count": int(len(group)),
            "scored": int(available.sum()),
            "correct": int(correct.eq(True).sum()),
            "wrong": int(wrong.sum()),
            "high_confidence": int((confidence >= 0.8).sum()),
            "high_conf_wrong": int((wrong & (confidence >= 0.8)).sum()),
        }
    return out


def _readout_interest_score(row: Mapping[str, Any]) -> float:
    split = _split_category(_optional_text(row.get("split")))
    correct = _clean_bool(row.get("correct"))
    confidence = _clean_float(row.get("confidence"))
    score = {"test": 110, "validation": 80, "train": -90}.get(split or "", -10)
    if correct is False:
        score += 260
    if correct is False and confidence is not None and confidence >= 0.8:
        score += 140
    if correct is None:
        score += 45
    score += min(40, (_clean_int(row.get("_probe_row_count")) or 1) * 4)
    if confidence is not None:
        score += min(20, confidence * 20)
    return float(score)


def _readout_episode_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    payload_row = {
        "trace_id": _first_present(row.get("trace_id"), row.get("episode_trace_id")),
        "episode_id": _first_present(row.get("episode_id"), row.get("episode_episode_id")),
        "episode_index": _first_present(row.get("episode_index"), row.get("episode_episode_index")),
        "task_id": _first_present(row.get("task_id"), row.get("episode_task_id")),
        "prompt": _first_present(row.get("prompt"), row.get("episode_prompt")),
        "model_id": row.get("episode_model_id"),
        "env_id": row.get("episode_env_id"),
        "robot_id": row.get("episode_robot_id"),
        "outcome": row.get("episode_outcome"),
        "length": row.get("episode_length"),
        "schema_version": row.get("episode_schema_version"),
        "dataset_id": row.get("episode_dataset_id"),
        "benchmark": row.get("episode_benchmark"),
        "profile": row.get("episode_profile"),
        "seed": row.get("episode_seed"),
        "probe_available": True,
        "probe_row_count": row.get("_probe_row_count"),
        "probe_split": row.get("split"),
        "probe_split_category": row.get("split_category"),
        "probe_confidence": row.get("confidence"),
        "probe_correct": row.get("correct"),
        "probe_correct_rate": 1.0
        if _clean_bool(row.get("correct")) is True
        else 0.0
        if _clean_bool(row.get("correct")) is False
        else None,
        "probe_actual": row.get("actual"),
        "probe_predicted": row.get("predicted"),
        "probe_model": row.get("model_site_id"),
        "probe_feature": _readout_feature_label(row),
        "probe_policy_call_index": row.get("policy_call_index"),
    }
    return _episode_payload_from_row(payload_row)


def _readout_feature_label(row: Mapping[str, Any]) -> str:
    model_site = _optional_text(row.get("model_site_id"))
    token_space = _optional_text(row.get("token_space_id"))
    layer = _optional_text(row.get("layer"))
    pieces = [
        piece for piece in [model_site, token_space, f"layer {layer}" if layer else ""] if piece
    ]
    return " / ".join(pieces)


def _readout_episode_field(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame:
        values = frame[column]
    else:
        values = pd.Series([None] * len(frame), index=frame.index, dtype=object)
    episode_column = f"episode_{column}"
    if episode_column in frame:
        values = values.where(values.map(_clean_scalar).notna(), frame[episode_column])
    return values


def _episode_facets_from_frame(episodes: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    facets: dict[str, list[dict[str, Any]]] = {}
    for column in EPISODE_FILTER_COLUMNS:
        if episodes.empty or column not in episodes:
            facets[column] = []
            continue
        counts = episodes[column].dropna().astype(str).value_counts().sort_index()
        facets[column] = [
            {"value": str(value), "count": int(count)}
            for value, count in counts.items()
            if str(value).strip()
        ]
    return facets
