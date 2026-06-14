"""Probe-study payloads for research-question oriented probe browsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from vla_lens.dataset.index import EPISODE_INDEX
from vla_lens.server.common import _json_parse, _jsonable
from vla_lens.server.indexed import (
    DEFAULT_EPISODE_LIMIT,
    MAX_EPISODE_LIMIT,
    _clamped_int,
    _episode_payload_from_row,
    _query_value,
    _read_table,
)
from vla_lens.traces import TraceDataset

READOUT_COLUMNS = [
    "readout_id",
    "target",
    "status",
    "source",
    "layer",
    "split",
    "split_category",
    "row_count",
    "policy_call_count",
    "class_count",
    "balanced_accuracy",
    "accuracy",
    "macro_f1",
    "top1_accuracy",
    "top2_accuracy",
    "top3_accuracy",
    "train_balanced_accuracy",
    "train_gap_balanced_accuracy",
    "reason",
    "is_primary_target",
    "is_selected_layer",
    "is_selection_split",
    "is_test_split",
]

ERROR_BROWSER_COLUMNS = [
    "split",
    "trace_id",
    "episode_id",
    "task_id",
    "prompt",
    "timestep",
    "policy_call_index",
    "layer",
    "model_site_id",
    "token_space_id",
    "actual",
    "predicted",
    "correct",
    "confidence",
    "top1_label",
    "top1_confidence",
    "top2_label",
    "top2_confidence",
    "top3_label",
    "top3_confidence",
    "task_phase",
    "next_manipulated_object",
    "active_manipulated_object",
    "active_receptacle_object",
    "contact_lead_bucket",
    "motion_lead_bucket",
    "events_before",
    "events_after",
]

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
        studies.append(_study_payload(dataset, record))
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

    primary_target = (
        _optional_text(summary.get("target"))
        or _optional_text(_mapping(artifact.get("metrics")).get("target"))
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


def _study_payload(dataset: TraceDataset, record: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _load_artifact_or_record(dataset, record)
    artifact_id = str(artifact.get("artifact_id") or record.get("artifact_id") or "")
    name = str(artifact.get("name") or artifact_id)
    metrics = _mapping(artifact.get("metrics"))
    method = _mapping(artifact.get("method"))
    selector = _mapping(artifact.get("selector"))
    display = _mapping(artifact.get("display"))
    summary, tables = _diagnostics(dataset, artifact_id, artifact)
    target = str(
        summary.get("target")
        or metrics.get("target")
        or display.get("target")
        or ""
    )

    readouts, skipped = _readouts_from_diagnostics(tables, target, summary)
    source = "diagnostics"
    if not readouts and not skipped:
        readouts = _readouts_from_artifact(display, metrics, target, summary)
        source = "artifact"

    lead_time = _records_with_target(tables["lead_time"], target)
    per_class = _records_with_target(tables["per_class"], target)
    confusion = _records_with_target(tables["confusion"], target, limit=600)
    class_support = _records_with_target(tables["class_support"], target)
    error_examples = _error_examples(tables["errors"])
    controls = _control_payloads(summary, tables["selection_null"])

    return {
        "artifact_id": artifact_id,
        "artifact_type": "probe_suite",
        "name": name,
        "created_utc": _clean_scalar(artifact.get("created_utc")),
        "target": target or None,
        "question_label": _question_label(target),
        "prediction": _prediction_label(target),
        "input": _input_label(selector, method),
        "output": _output_label(summary, metrics),
        "objective": _objective_label(method),
        "diagnostics_available": any(not table.empty for table in tables.values())
        or bool(summary),
        "source": source,
        "counts": _counts(summary, readouts, skipped, tables),
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
        out = out.loc[correct == True]  # noqa: E712
        correct = correct.loc[out.index]
        confidence = confidence.loc[out.index]
    elif prediction == "incorrect":
        out = out.loc[correct == False]  # noqa: E712
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
            | (correct == False)  # noqa: E712
            | confidence.isna()
            | (confidence < 0.65)
        ]
    elif preset == "heldout_wrong":
        out = out.loc[split_categories.isin(["validation", "test"]) & (correct == False)]  # noqa: E712
    elif preset == "confident_wrong":
        out = out.loc[(correct == False) & (confidence >= 0.8)]  # noqa: E712
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
    wrong = correct == False  # noqa: E712
    return {
        "policy_call_count": int(len(scoped_rows)),
        "episode_count": int(len(representatives)),
        "scored": int(available.sum()),
        "unscored": int((~available).sum()),
        "correct": int((correct == True).sum()),  # noqa: E712
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
        wrong = correct == False  # noqa: E712
        out[split_name] = {
            "policy_call_count": int(len(group)),
            "scored": int(available.sum()),
            "correct": int((correct == True).sum()),  # noqa: E712
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
        "probe_correct_rate": 1.0 if _clean_bool(row.get("correct")) is True else 0.0
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
        piece
        for piece in [model_site, token_space, f"layer {layer}" if layer else ""]
        if piece
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


def _artifact_for_id(dataset: TraceDataset, artifact_id: str) -> dict[str, Any]:
    artifacts = dataset.artifact_index
    if not artifacts.empty and "artifact_id" in artifacts:
        rows = artifacts.loc[artifacts["artifact_id"].astype(str) == artifact_id]
        if not rows.empty:
            return _load_artifact_or_record(dataset, rows.iloc[0].to_dict())
    try:
        return dataset.load_artifact(artifact_id).to_dict()
    except Exception:
        return {"artifact_id": artifact_id}


def _first_present(*values: Any) -> Any:
    for value in values:
        if _clean_scalar(value) is not None:
            return value
    return None


def _diagnostics(
    dataset: TraceDataset,
    artifact_id: str,
    artifact: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    diagnostics_dir = _artifact_dir(dataset, artifact_id, artifact) / "diagnostics"
    summary: dict[str, Any] = {}
    try:
        summary_path = diagnostics_dir / "summary.json"
        if summary_path.exists():
            parsed = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                summary = parsed
    except (OSError, json.JSONDecodeError):
        summary = {}

    tables = {
        "layer_split": _read_parquet(diagnostics_dir / "layer_split_metrics.parquet"),
        "battery": _read_parquet(diagnostics_dir / "readout_battery_metrics.parquet"),
        "selection_null": _read_parquet(diagnostics_dir / "selection_aware_null.parquet"),
        "lead_time": _read_parquet(diagnostics_dir / "lead_time_metrics.parquet"),
        "per_class": _read_parquet(diagnostics_dir / "per_class_metrics.parquet"),
        "confusion": _read_parquet(diagnostics_dir / "confusion_matrix.parquet"),
        "errors": _read_parquet(diagnostics_dir / "probe_error_browser.parquet"),
        "class_support": _read_parquet(
            diagnostics_dir / "policy_call_support_by_class_split.parquet"
        ),
    }
    return summary, tables


def _artifact_dir(dataset: TraceDataset, artifact_id: str, artifact: Mapping[str, Any]) -> Path:
    root = dataset._dataset_artifact_root()
    raw_path = artifact.get("path")
    if raw_path:
        path = Path(str(raw_path))
        if not path.is_absolute() and ".." not in path.parts:
            return (root / path).parent
    return root / "artifacts" / artifact_id


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _readouts_from_diagnostics(
    tables: Mapping[str, pd.DataFrame],
    primary_target: str,
    summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    battery = tables["battery"]
    if battery.empty:
        layer_split = tables["layer_split"]
        if layer_split.empty:
            return [], []
        battery = layer_split.copy()
        battery["target"] = primary_target
        battery["status"] = "ok"
        battery["reason"] = None

    selected_layer = _optional_text(summary.get("selected_layer"))
    selection_split = _optional_text(summary.get("selection_split"))
    test_split = _optional_text(summary.get("test_split"))
    readouts: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in battery.to_dict("records"):
        target = _optional_text(row.get("target")) or primary_target
        if target != primary_target:
            continue
        status = _optional_text(row.get("status")) or "ok"
        layer = _clean_scalar(row.get("layer"))
        split = _optional_text(row.get("split"))
        item = {
            **_readout_metric_fields(row),
            "readout_id": _readout_id(target, layer, split, status),
            "target": target,
            "status": status,
            "source": "diagnostic",
            "layer": layer,
            "split": split or None,
            "split_category": _split_category(split),
            "reason": _clean_scalar(row.get("reason")),
            "is_primary_target": target == primary_target,
            "is_selected_layer": bool(selected_layer and str(layer) == selected_layer),
            "is_selection_split": bool(selection_split and split == selection_split),
            "is_test_split": bool(test_split and split == test_split),
        }
        item = {key: item.get(key) for key in READOUT_COLUMNS}
        if status == "ok":
            readouts.append(item)
        else:
            skipped.append(item)
    return readouts, skipped


def _readouts_from_artifact(
    display: Mapping[str, Any],
    metrics: Mapping[str, Any],
    primary_target: str,
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    results = display.get("results")
    if not isinstance(results, list):
        return []
    readouts: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        if not isinstance(result, Mapping):
            continue
        split = _optional_text(result.get("split_value") or result.get("eval_split"))
        layer = _clean_scalar(result.get("sweep_layer", result.get("layer")))
        row = {
            "readout_id": f"artifact:{index}",
            "target": primary_target,
            "status": "ok",
            "source": "artifact",
            "layer": layer,
            "split": split or None,
            "split_category": _split_category(split),
            "row_count": _clean_int(result.get("n_test") or result.get("row_count")),
            "policy_call_count": _clean_int(result.get("policy_call_count")),
            "class_count": None,
            "balanced_accuracy": _clean_float(result.get("score")),
            "accuracy": None,
            "macro_f1": None,
            "top1_accuracy": None,
            "top2_accuracy": None,
            "top3_accuracy": None,
            "train_balanced_accuracy": None,
            "train_gap_balanced_accuracy": None,
            "reason": None,
            "is_primary_target": True,
            "is_selected_layer": str(layer) == _optional_text(summary.get("selected_layer")),
            "is_selection_split": split == _optional_text(metrics.get("best_eval_split")),
            "is_test_split": False,
        }
        readouts.append(row)
    return readouts


def _readout_metric_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "row_count": _clean_int(row.get("row_count")),
        "policy_call_count": _clean_int(row.get("policy_call_count")),
        "class_count": _clean_int(row.get("class_count")),
        "balanced_accuracy": _clean_float(row.get("balanced_accuracy")),
        "accuracy": _clean_float(row.get("accuracy")),
        "macro_f1": _clean_float(row.get("macro_f1")),
        "top1_accuracy": _clean_float(row.get("top1_accuracy")),
        "top2_accuracy": _clean_float(row.get("top2_accuracy")),
        "top3_accuracy": _clean_float(row.get("top3_accuracy")),
        "train_balanced_accuracy": _clean_float(row.get("train_balanced_accuracy")),
        "train_gap_balanced_accuracy": _clean_float(row.get("train_gap_balanced_accuracy")),
    }


def _records_with_target(
    frame: pd.DataFrame,
    target: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    out = frame.copy()
    if target and "target" not in out:
        out["target"] = target
    if limit is not None:
        out = out.head(limit)
    return [_clean_record(row) for row in out.to_dict("records")]


def _error_examples(
    frame: pd.DataFrame,
    limit: int = 600,
    per_layer_split: int = 40,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    out = frame.copy()
    if "correct" in out:
        correct = out["correct"].map(_clean_bool)
        out["_wrong"] = correct == False  # noqa: E712
    else:
        out["_wrong"] = False
    out["_heldout"] = out.get("split", pd.Series(dtype=object)).astype(str) != "train"
    out["_confidence"] = pd.to_numeric(
        out.get("confidence", pd.Series(dtype=object)),
        errors="coerce",
    )
    out = out.sort_values(
        ["_wrong", "_heldout", "_confidence"],
        ascending=[False, False, False],
        na_position="last",
    )
    if {"layer", "split"} <= set(out.columns):
        groups = [
            group.head(per_layer_split)
            for _, group in out.groupby(["layer", "split"], dropna=False, sort=True)
        ]
        out = pd.concat(groups, ignore_index=True) if groups else out.head(0)
        out = out.sort_values(
            ["_wrong", "_heldout", "_confidence"],
            ascending=[False, False, False],
            na_position="last",
        )
    columns = [column for column in ERROR_BROWSER_COLUMNS if column in out.columns]
    out = out.loc[:, columns].head(limit)
    return [_clean_record(row) for row in out.to_dict("records")]


def _control_payloads(
    summary: Mapping[str, Any],
    null_rows: pd.DataFrame,
) -> list[dict[str, Any]]:
    null_summary = summary.get("selection_aware_null")
    if not isinstance(null_summary, Mapping) and null_rows.empty:
        return []
    if not isinstance(null_summary, Mapping):
        null_summary = {}
    selected_layer_counts: dict[str, int] = {}
    if not null_rows.empty and "selected_layer" in null_rows:
        counts = null_rows["selected_layer"].dropna().astype(str).value_counts().sort_index()
        selected_layer_counts = {str(key): int(value) for key, value in counts.items()}
    runs = _clean_int(null_summary.get("runs"))
    if runs is None and not null_rows.empty and "run" in null_rows:
        runs = int(null_rows["run"].nunique())
    selected_layer = _clean_scalar(summary.get("selected_layer"))
    return [
        {
            "kind": "selection_aware_null",
            "label": "Validation selection",
            "split": _clean_scalar(summary.get("selection_split")),
            "runs": runs,
            "selected_layer": selected_layer,
            "real_score": _clean_float(summary.get("selected_layer_selection_balanced_accuracy")),
            "null_score_mean": _clean_float(null_summary.get("selection_score_mean")),
            "null_score_std": _clean_float(null_summary.get("selection_score_std")),
            "p_value": _clean_float(null_summary.get("selection_p_value")),
            "selected_layer_counts": selected_layer_counts,
        },
        {
            "kind": "selection_aware_null",
            "label": "Heldout test",
            "split": _clean_scalar(summary.get("test_split")),
            "runs": runs,
            "selected_layer": selected_layer,
            "real_score": _clean_float(summary.get("selected_layer_test_balanced_accuracy")),
            "null_score_mean": _clean_float(null_summary.get("test_score_mean")),
            "null_score_std": _clean_float(null_summary.get("test_score_std")),
            "p_value": _clean_float(null_summary.get("test_p_value")),
            "selected_layer_counts": selected_layer_counts,
        },
    ]


def _counts(
    summary: Mapping[str, Any],
    readouts: list[Mapping[str, Any]],
    skipped: list[Mapping[str, Any]],
    tables: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    null_rows = tables["selection_null"]
    null_runs = (
        int(null_rows["run"].nunique())
        if not null_rows.empty and "run" in null_rows
        else _clean_int(_mapping(summary.get("selection_aware_null")).get("runs"))
    )
    return {
        "readout_count": len(readouts),
        "skipped_readout_count": len(skipped),
        "target_count": len({str(row.get("target")) for row in readouts if row.get("target")}),
        "layer_count": _clean_int(summary.get("layer_count")),
        "feature_rows": _clean_int(summary.get("feature_rows")),
        "policy_call_count": _clean_int(summary.get("policy_call_count")),
        "episode_count": _clean_int(summary.get("episode_count")),
        "class_count": _clean_int(summary.get("class_count")),
        "null_run_count": null_runs,
        "null_eval_row_count": int(len(null_rows)) if not null_rows.empty else 0,
        "split_policy_call_counts": _jsonable(summary.get("split_policy_call_counts") or {}),
    }


def _load_artifact_or_record(
    dataset: TraceDataset,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_id = str(record.get("artifact_id") or "")
    try:
        artifact = dataset.load_artifact(artifact_id)
        return artifact.to_dict()
    except Exception:
        return _artifact_record_payload(record)


def _artifact_record_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = {str(key): _clean_scalar(value) for key, value in record.items()}
    for key in ["selector", "method", "metrics", "arrays", "display", "tags", "source_trace_ids"]:
        payload[key] = _json_parse(payload.get(key))
    return payload


def _mapping(value: Any) -> dict[str, Any]:
    parsed = _json_parse(value)
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _readout_id(target: str, layer: Any, split: str, status: str) -> str:
    layer_value = layer if layer is not None else "all"
    split_value = split or "all"
    return f"{target or 'target'}|layer:{layer_value}|split:{split_value}|{status}"


def _split_category(split: str) -> str | None:
    text = str(split or "").lower()
    if not text:
        return None
    if text == "train" or text.startswith("train"):
        return "train"
    if "val" in text or "validation" in text:
        return "validation"
    if "test" in text:
        return "test"
    return text


def _question_label(target: str) -> str:
    if target == "next_manipulated_object":
        return "Which object will the robot manipulate next before contact?"
    if target == "active_manipulated_object":
        return "Which object is the robot currently manipulating?"
    if target == "active_receptacle_object":
        return "Which receptacle is active in the current interaction?"
    if target == "task_phase":
        return "Which object-centric phase is the robot in?"
    return target.replace("_", " ") if target else "Probe study"


def _prediction_label(target: str) -> str:
    labels = {
        "next_manipulated_object": "Next manipulated object",
        "active_manipulated_object": "Active manipulated object",
        "active_receptacle_object": "Active receptacle",
        "task_phase": "Task phase",
    }
    return labels.get(target, target.replace("_", " ").strip().capitalize() if target else "Target")


def _input_label(selector: Mapping[str, Any], method: Mapping[str, Any]) -> str:
    site = selector.get("site") or selector.get("model_site")
    token_space = selector.get("token_space")
    layers = selector.get("layers")
    if not site and isinstance(method.get("feature_cache"), Mapping):
        feature_cache = method["feature_cache"]
        site = feature_cache.get("site") or feature_cache.get("model_site")
        token_space = feature_cache.get("token_space")
        layers = feature_cache.get("layers") or layers
    pieces = ["Expert hidden states"]
    if site:
        pieces.append(str(site))
    if layers:
        if isinstance(layers, list):
            pieces.append(f"layers {', '.join(str(item) for item in layers)}")
        else:
            pieces.append(f"layers {layers}")
    if token_space:
        pieces.append(str(token_space))
    return " / ".join(pieces)


def _output_label(summary: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    class_count = _clean_int(summary.get("class_count") or metrics.get("class_count"))
    if class_count:
        return f"{class_count} object classes"
    return "Class label"


def _objective_label(method: Mapping[str, Any]) -> str:
    probe = method.get("probe")
    if isinstance(probe, Mapping):
        model = probe.get("model") or probe.get("classifier")
        if model:
            return str(model).replace("_", " ")
    model = method.get("model") or method.get("classifier")
    return str(model).replace("_", " ") if model else "Linear readout"


def _clean_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _clean_scalar(value) for key, value in row.items()}


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return _clean_scalar(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        return value if value == value and value not in {float("inf"), float("-inf")} else None
    if isinstance(value, Path):
        return str(value)
    return value


def _clean_float(value: Any) -> float | None:
    value = _clean_scalar(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in {float("inf"), float("-inf")} else None


def _clean_int(value: Any) -> int | None:
    number = _clean_float(value)
    return int(number) if number is not None else None


def _clean_bool(value: Any) -> bool | None:
    value = _clean_scalar(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    return None


def _optional_text(value: Any) -> str:
    value = _clean_scalar(value)
    return "" if value is None else str(value)


__all__ = ["probe_studies_payload", "probe_study_episodes_payload"]
