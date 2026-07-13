"""Probe-study artifact diagnostics and readout aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from vla_lens.server.common import _jsonable
from vla_lens.server.probe_study_formatting import (
    _clean_bool,
    _clean_float,
    _clean_int,
    _clean_record,
    _clean_scalar,
    _load_artifact_or_record,
    _mapping,
    _optional_text,
    _prediction_label,
    _readout_id,
    _split_category,
    _trained_probe_id,
)
from vla_lens.table_io import read_optional_parquet
from vla_lens.traces import TraceDataset

READOUT_COLUMNS = [
    "readout_id",
    "trained_probe_id",
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
    return read_optional_parquet(path, context="probe diagnostics")


def _primary_target(
    summary: Mapping[str, Any],
    metrics: Mapping[str, Any],
    display: Mapping[str, Any],
) -> str:
    return str(summary.get("target") or metrics.get("target") or display.get("target") or "")


def _study_targets(tables: Mapping[str, pd.DataFrame], primary_target: str) -> list[str]:
    """Return logical probe-family targets with trained readouts.

    Older diagnostics artifacts can contain a battery of targets inside one
    physical artifact. The UI should expose those as separate question families,
    while skipped-only targets are kept out of the selector.
    """

    battery = tables["battery"]
    if battery.empty or "target" not in battery:
        return [primary_target] if primary_target else []
    targets: list[str] = []
    for row in battery.to_dict("records"):
        target = _optional_text(row.get("target")) or primary_target
        status = (_optional_text(row.get("status")) or "ok").lower()
        if not target or status != "ok" or target in targets:
            continue
        targets.append(target)
    if not targets and primary_target:
        targets.append(primary_target)
    if primary_target in targets:
        targets = [primary_target, *[target for target in targets if target != primary_target]]
    return targets


def _study_id(artifact_id: str, target: str) -> str:
    if not target:
        return artifact_id
    return f"{artifact_id}::target={target}"


def _study_name(artifact_name: str, target: str, family_count: int) -> str:
    if target:
        label = _prediction_label(target)
        return label if family_count > 1 else artifact_name or label
    return artifact_name or "Probe family"


def _readouts_from_diagnostics(
    tables: Mapping[str, pd.DataFrame],
    target_filter: str,
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

    selection_split = _optional_text(summary.get("selection_split"))
    test_split = _optional_text(summary.get("test_split"))
    selected_layer = _selected_layer_for_target(
        battery,
        target_filter=target_filter,
        primary_target=primary_target,
        selection_split=selection_split,
        summary=summary,
    )
    readouts: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in battery.to_dict("records"):
        target = _optional_text(row.get("target")) or primary_target
        if target_filter and target != target_filter:
            continue
        status = _optional_text(row.get("status")) or "ok"
        layer = _clean_scalar(row.get("layer"))
        split = _optional_text(row.get("split"))
        item = {
            **_readout_metric_fields(row),
            "readout_id": _readout_id(target, layer, split, status),
            "trained_probe_id": _optional_text(row.get("trained_probe_id"))
            or _trained_probe_id(target, layer, split),
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


def _selected_layer_for_target(
    battery: pd.DataFrame,
    *,
    target_filter: str,
    primary_target: str,
    selection_split: str,
    summary: Mapping[str, Any],
) -> str:
    fallback = _optional_text(summary.get("selected_layer"))
    if not target_filter or target_filter == primary_target:
        return fallback
    if battery.empty or not selection_split:
        return fallback
    rows = battery.copy()
    if "target" in rows:
        target = target_filter or primary_target
        if target:
            rows = rows.loc[rows["target"].astype(str) == target]
    elif target_filter and primary_target and target_filter != primary_target:
        return ""
    if rows.empty or "split" not in rows or "layer" not in rows or "balanced_accuracy" not in rows:
        return fallback
    rows = rows.loc[rows["split"].astype(str) == selection_split].copy()
    if "status" in rows:
        rows = rows.loc[rows["status"].fillna("ok").astype(str) == "ok"]
    rows["__score"] = pd.to_numeric(rows["balanced_accuracy"], errors="coerce")
    rows = rows.dropna(subset=["__score"])
    if rows.empty:
        return fallback
    best = rows.sort_values(["__score", "layer"], ascending=[False, True]).iloc[0]
    return _optional_text(best.get("layer")) or fallback


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
            "trained_probe_id": _trained_probe_id(primary_target, layer, split),
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
    primary_target: str = "",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    out = frame.copy()
    if target and "target" in out:
        out = out.loc[out["target"].astype(str) == target]
    elif target and primary_target and target != primary_target:
        return []
    elif target:
        out["target"] = target
    if limit is not None:
        out = out.head(limit)
    return [_clean_record(row) for row in out.to_dict("records")]


def _error_examples(
    frame: pd.DataFrame,
    *,
    target: str = "",
    primary_target: str = "",
    limit: int = 600,
    per_layer_split: int = 40,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    out = frame.copy()
    if target and "target" in out:
        out = out.loc[out["target"].astype(str) == target]
    elif target and primary_target and target != primary_target:
        return []
    elif target:
        out["target"] = target
    if "correct" in out:
        correct = out["correct"].map(_clean_bool)
        out["_wrong"] = correct.eq(False)
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
    readouts: Sequence[Mapping[str, Any]],
    *,
    target: str = "",
    primary_target: str = "",
) -> list[dict[str, Any]]:
    if target and not null_rows.empty and "target" in null_rows:
        null_rows = null_rows.loc[null_rows["target"].astype(str) == target]
    elif target and primary_target and target != primary_target:
        return []
    null_summary = summary.get("selection_aware_null")
    if not isinstance(null_summary, Mapping) and null_rows.empty:
        return []
    if not isinstance(null_summary, Mapping):
        null_summary = {}
    selected_layer_counts: dict[str, int] = {}
    if not null_rows.empty and "selected_layer" in null_rows:
        counts = null_rows["selected_layer"].dropna().astype(str).value_counts().sort_index()
        selected_layer_counts = {str(key): int(value) for key, value in counts.items()}
    runs = None
    if not null_rows.empty and "run" in null_rows:
        runs = int(null_rows["run"].nunique())
    if runs is None:
        runs = _clean_int(null_summary.get("runs"))
    selection_split = _optional_text(summary.get("selection_split"))
    test_split = _optional_text(summary.get("test_split"))
    selected_layer = _selected_readout_layer(readouts, selection_split) or _optional_text(
        summary.get("selected_layer")
    )
    selection_readout = _readout_for_layer_split(readouts, selected_layer, selection_split)
    test_readout = _readout_for_layer_split(readouts, selected_layer, test_split)
    selection_stats = _null_score_stats(
        null_rows,
        selection_split,
        _clean_float(selection_readout.get("balanced_accuracy")),
    )
    test_stats = _null_score_stats(
        null_rows,
        test_split,
        _clean_float(test_readout.get("balanced_accuracy")),
    )
    if runs is None and not null_rows.empty and "run" in null_rows:
        runs = int(null_rows["run"].nunique())
    use_summary_scores = not target or target == primary_target
    selection_real = (
        _clean_float(summary.get("selected_layer_selection_balanced_accuracy"))
        if use_summary_scores
        else None
    )
    test_real = (
        _clean_float(summary.get("selected_layer_test_balanced_accuracy"))
        if use_summary_scores
        else None
    )
    selection_null_mean = (
        _clean_float(null_summary.get("selection_score_mean")) if use_summary_scores else None
    )
    selection_null_std = (
        _clean_float(null_summary.get("selection_score_std")) if use_summary_scores else None
    )
    selection_p_value = (
        _clean_float(null_summary.get("selection_p_value")) if use_summary_scores else None
    )
    test_null_mean = (
        _clean_float(null_summary.get("test_score_mean")) if use_summary_scores else None
    )
    test_null_std = _clean_float(null_summary.get("test_score_std")) if use_summary_scores else None
    test_p_value = _clean_float(null_summary.get("test_p_value")) if use_summary_scores else None
    controls = [
        {
            "kind": "selection_aware_null",
            "label": "Validation selection",
            "split": _clean_scalar(selection_split),
            "runs": runs,
            "selected_layer": selected_layer,
            "real_score": selection_real
            if selection_real is not None
            else _clean_float(selection_readout.get("balanced_accuracy")),
            "null_score_mean": selection_null_mean
            if selection_null_mean is not None
            else selection_stats["mean"],
            "null_score_std": selection_null_std
            if selection_null_std is not None
            else selection_stats["std"],
            "p_value": selection_p_value
            if selection_p_value is not None
            else selection_stats["p_value"],
            "selected_layer_counts": selected_layer_counts,
        },
        {
            "kind": "selection_aware_null",
            "label": "Heldout test",
            "split": _clean_scalar(test_split),
            "runs": runs,
            "selected_layer": selected_layer,
            "real_score": test_real
            if test_real is not None
            else _clean_float(test_readout.get("balanced_accuracy")),
            "null_score_mean": test_null_mean
            if test_null_mean is not None
            else test_stats["mean"],
            "null_score_std": test_null_std
            if test_null_std is not None
            else test_stats["std"],
            "p_value": test_p_value
            if test_p_value is not None
            else test_stats["p_value"],
            "selected_layer_counts": selected_layer_counts,
        },
    ]
    return [
        control
        for control in controls
        if control["real_score"] is not None or control["null_score_mean"] is not None
    ]


def _selected_readout_layer(readouts: Sequence[Mapping[str, Any]], selection_split: str) -> str:
    selected = [
        readout
        for readout in readouts
        if readout.get("is_selected_layer")
        and (not selection_split or readout.get("split") == selection_split)
    ]
    if selected:
        return _optional_text(selected[0].get("layer"))
    selection_readouts = [
        readout
        for readout in readouts
        if not selection_split or readout.get("split") == selection_split
    ]
    scored = [
        readout
        for readout in selection_readouts
        if _clean_float(readout.get("balanced_accuracy")) is not None
    ]
    if not scored:
        return ""
    best = sorted(
        scored,
        key=lambda readout: (
            _clean_float(readout.get("balanced_accuracy")) or float("-inf"),
            str(readout.get("layer") or ""),
        ),
        reverse=True,
    )[0]
    return _optional_text(best.get("layer"))


def _readout_for_layer_split(
    readouts: Sequence[Mapping[str, Any]],
    selected_layer: str,
    split: str,
) -> Mapping[str, Any]:
    for readout in readouts:
        if _optional_text(readout.get("layer")) == selected_layer and (
            not split or readout.get("split") == split
        ):
            return readout
    return {}


def _null_score_stats(
    null_rows: pd.DataFrame,
    split: str,
    real_score: float | None,
) -> dict[str, float | None]:
    if null_rows.empty or not split or "split" not in null_rows or "score" not in null_rows:
        return {"mean": None, "p_value": None, "std": None}
    scores = pd.to_numeric(
        null_rows.loc[null_rows["split"].astype(str) == split, "score"],
        errors="coerce",
    ).dropna()
    if scores.empty:
        return {"mean": None, "p_value": None, "std": None}
    p_value = None
    if real_score is not None:
        p_value = float((1 + int((scores >= real_score).sum())) / (len(scores) + 1))
    return {
        "mean": float(scores.mean()),
        "p_value": p_value,
        "std": float(scores.std(ddof=0)),
    }


def _counts(
    summary: Mapping[str, Any],
    readouts: list[Mapping[str, Any]],
    skipped: list[Mapping[str, Any]],
    tables: Mapping[str, pd.DataFrame],
    *,
    target: str = "",
    primary_target: str = "",
) -> dict[str, Any]:
    null_rows = tables["selection_null"]
    null_rows_for_target = _target_scoped_frame(null_rows, target, primary_target)
    null_runs = (
        int(null_rows_for_target["run"].nunique())
        if not null_rows_for_target.empty and "run" in null_rows_for_target
        else _clean_int(_mapping(summary.get("selection_aware_null")).get("runs"))
        if not target or target == primary_target
        else None
    )
    return {
        "readout_count": len(readouts),
        "skipped_readout_count": len(skipped),
        "target_count": len({str(row.get("target")) for row in readouts if row.get("target")}),
        "layer_count": _clean_int(summary.get("layer_count")),
        "feature_rows": _clean_int(summary.get("feature_rows")),
        "policy_call_count": _clean_int(summary.get("policy_call_count")),
        "episode_count": _clean_int(summary.get("episode_count")),
        "class_count": _max_count(readouts, "class_count")
        or _clean_int(summary.get("class_count")),
        "null_run_count": null_runs,
        "null_eval_row_count": (
            int(len(null_rows_for_target)) if not null_rows_for_target.empty else 0
        ),
        "split_policy_call_counts": _jsonable(summary.get("split_policy_call_counts") or {}),
    }


def _target_scoped_frame(frame: pd.DataFrame, target: str, primary_target: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if target and "target" in frame:
        return frame.loc[frame["target"].astype(str) == target]
    if target and primary_target and target != primary_target:
        return frame.head(0)
    return frame


def _max_count(rows: list[Mapping[str, Any]], key: str) -> int | None:
    values = [_clean_int(row.get(key)) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else None
