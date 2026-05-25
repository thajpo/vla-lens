"""Probes dashboard server helpers."""


from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.artifacts import LensArtifact
from vla_lens.probes.workflow import (
    _apply_missing_policy,
    _attach_episode_metadata,
    _normalize_target_spec,
    _resolve_probe_target,
    _target_name,
)
from vla_lens.selectors import ActivationQuery
from vla_lens.server_common import (
    _dominant_value,
    _is_missing_scalar,
    _json_parse,
    _json_scalar,
    _jsonable,
    _mean_numeric,
    _optional_bool,
    _optional_float,
    _optional_int,
    _optional_text,
    _query_one,
    _safe_filename,
)
from vla_lens.traces import TraceDataset


def _episode_interactions_payload(
    dataset: TraceDataset,
    query: dict[str, list[str]],
) -> dict[str, Any]:
    trace_id = query.get("trace_id", [""])[0]
    if not trace_id:
        return {
            "available": False,
            "reason": "Missing trace_id.",
            "trace_id": "",
            "objects": [],
        }
    artifact = _latest_interaction_metrics_artifact(dataset)
    if artifact is None:
        return {
            "available": False,
            "reason": "No pi05_interaction_metrics artifact found.",
            "trace_id": trace_id,
            "objects": [],
        }

    episode_table = _interaction_metrics_table(dataset, artifact, "episode_labels")
    object_table = _interaction_metrics_table(dataset, artifact, "object_metrics")
    if episode_table.empty or "trace_id" not in episode_table:
        return {
            "available": False,
            "reason": "Interaction metrics artifact has no episode label table.",
            "trace_id": trace_id,
            "artifact_id": artifact.artifact_id,
            "objects": [],
        }
    episode_rows = episode_table[episode_table["trace_id"].astype(str) == trace_id]
    if episode_rows.empty:
        return {
            "available": False,
            "reason": "Trace is not present in the interaction metrics artifact.",
            "trace_id": trace_id,
            "artifact_id": artifact.artifact_id,
            "objects": [],
        }

    episode_row = episode_rows.iloc[0].to_dict()
    object_rows = (
        object_table[object_table["trace_id"].astype(str) == trace_id]
        if not object_table.empty and "trace_id" in object_table
        else pd.DataFrame()
    )
    objects = [_interaction_object_payload(row) for row in object_rows.to_dict("records")]
    objects = sorted(
        objects,
        key=lambda item: (
            not bool(item["is_target_object"]),
            not (bool(item["moved"]) or bool(item["lifted"]) or bool(item["contacted"])),
            str(item["object_name"]),
        ),
    )
    return {
        "available": True,
        "trace_id": trace_id,
        "artifact_id": artifact.artifact_id,
        "episode": _interaction_episode_payload(episode_row),
        "quality": _interaction_quality_payload(episode_row),
        "objects": objects,
    }

def _latest_interaction_metrics_artifact(dataset: TraceDataset) -> LensArtifact | None:
    table = dataset.artifact_index
    if table.empty or "artifact_type" not in table:
        return None
    matches = table[table["artifact_type"].astype(str) == "pi05_interaction_metrics"]
    if matches.empty:
        return None
    if "created_utc" in matches:
        matches = matches.sort_values("created_utc", ascending=False, na_position="last")
    try:
        return dataset.load_artifact(str(matches.iloc[0]["artifact_id"]))
    except (FileNotFoundError, KeyError, ValueError):
        return None

def _interaction_metrics_table(
    dataset: TraceDataset,
    artifact: LensArtifact,
    key: str,
) -> pd.DataFrame:
    outputs = artifact.method.get("outputs") if isinstance(artifact.method, Mapping) else None
    relative_path = outputs.get(key) if isinstance(outputs, Mapping) else None
    if not relative_path:
        return pd.DataFrame()
    path = dataset.root / str(relative_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

def _episode_probes_payload(
    dataset: TraceDataset,
    query: Mapping[str, list[str]],
) -> dict[str, Any]:
    from vla_lens.server_dataset import _artifact_records

    trace_id = _query_one(query, "trace_id")
    probes: list[dict[str, Any]] = []
    for record in _artifact_records(dataset):
        if str(record.get("artifact_type")) != "probe_suite":
            continue
        artifact_id = str(record.get("artifact_id"))
        try:
            artifact = dataset.load_artifact(artifact_id)
        except (FileNotFoundError, KeyError, ValueError):
            continue
        predictions = _probe_prediction_table(dataset, artifact)
        if predictions.empty or "trace_id" not in predictions:
            episode_predictions = pd.DataFrame()
        else:
            episode_predictions = predictions.loc[predictions["trace_id"].astype(str) == trace_id]
        if episode_predictions.empty:
            episode_predictions = _saved_episode_probe_predictions(dataset, artifact, trace_id)
        if episode_predictions.empty:
            episode_predictions = _score_and_save_episode_probe(dataset, artifact, trace_id)
        probes.append(
            {
                "artifact_id": artifact.artifact_id,
                "name": artifact.name,
                "target": artifact.metrics.get("target") or artifact.display.get("target"),
                "metrics": _jsonable(artifact.metrics),
                "best_result": _jsonable(artifact.display.get("best_result_details") or {}),
                "target_distribution": _jsonable(artifact.display.get("target_distribution") or {}),
                "episode_summary": _probe_episode_summary(episode_predictions, artifact),
                "rows": _probe_prediction_rows(episode_predictions),
                "row_count": int(len(episode_predictions)),
                "available": bool(len(episode_predictions)),
            }
        )
    return {
        "trace_id": trace_id,
        "probes": probes,
        "available_count": sum(1 for probe in probes if probe["available"]),
        "total": len(probes),
    }

def _probe_index_payload(dataset: TraceDataset) -> dict[str, Any]:
    from vla_lens.server_dataset import _artifact_records

    split_sidecar = _probe_split_sidecar(dataset.root)
    trace_ids = [bundle.manifest.trace_id for bundle in dataset.bundles]
    probes: list[dict[str, Any]] = []
    for record in _artifact_records(dataset):
        if str(record.get("artifact_type")) != "probe_suite":
            continue
        artifact_id = str(record.get("artifact_id"))
        probe = _probe_index_artifact_payload(
            dataset,
            artifact_id,
            split_sidecar=split_sidecar,
            trace_ids=trace_ids,
        )
        if probe:
            probes.append(probe)
    return {
        "probes": probes,
        "total": len(probes),
        "trace_count": len(trace_ids),
        "split_source": "probe_splits.csv" if split_sidecar else None,
    }

def _probe_index_artifact_payload(
    dataset: TraceDataset,
    artifact_id: str,
    *,
    split_sidecar: dict[str, dict[str, Any]] | None = None,
    trace_ids: Sequence[str] | None = None,
) -> dict[str, Any] | None:
    try:
        artifact = dataset.load_artifact(artifact_id)
    except (FileNotFoundError, KeyError, ValueError):
        return None
    if artifact.artifact_type != "probe_suite":
        return None
    split_sidecar = (
        split_sidecar if split_sidecar is not None else _probe_split_sidecar(dataset.root)
    )
    trace_ids = list(trace_ids) if trace_ids is not None else [
        bundle.manifest.trace_id for bundle in dataset.bundles
    ]
    predictions = _probe_prediction_table(dataset, artifact)
    saved_predictions = _saved_probe_prediction_tables(dataset, artifact)
    if not saved_predictions.empty:
        predictions = (
            saved_predictions
            if predictions.empty
            else pd.concat([predictions, saved_predictions], ignore_index=True, sort=False)
        )
    by_trace = _probe_index_by_trace(
        trace_ids,
        split_sidecar,
        predictions,
        artifact,
    )
    return {
        "artifact_id": artifact.artifact_id,
        "name": artifact.name,
        "target": artifact.metrics.get("target") or artifact.display.get("target"),
        "best_model": artifact.metrics.get("best_model"),
        "best_feature": artifact.metrics.get("best_feature"),
        "best_score": artifact.metrics.get("best_score"),
        "best_delta": artifact.metrics.get("best_delta"),
        "split_summary": _probe_index_split_summary(by_trace),
        "prediction_summary": _probe_index_prediction_summary(by_trace),
        "by_trace": by_trace,
    }

def _probe_split_sidecar(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "probe_splits.csv"
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except Exception:
        return {}
    if frame.empty or "trace_id" not in frame:
        return {}
    frame = frame.drop_duplicates(subset=["trace_id"], keep="last")
    return {
        str(row.get("trace_id")): {
            str(key): _json_scalar(value)
            for key, value in row.items()
            if key != "trace_id" and not _is_missing_scalar(value)
        }
        for row in frame.to_dict("records")
    }

def _saved_probe_prediction_tables(dataset: TraceDataset, artifact: LensArtifact) -> pd.DataFrame:
    root = dataset.root / "workbench" / "episode_probe_predictions" / artifact.artifact_id
    if not root.exists():
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("*.parquet")):
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)

def _probe_index_by_trace(
    trace_ids: Sequence[str],
    split_sidecar: dict[str, dict[str, Any]],
    predictions: pd.DataFrame,
    artifact: LensArtifact,
) -> dict[str, dict[str, Any]]:
    prediction_groups: dict[str, pd.DataFrame] = {}
    if not predictions.empty and "trace_id" in predictions:
        for trace_id, group in predictions.groupby(predictions["trace_id"].astype(str), sort=False):
            prediction_groups[str(trace_id)] = group.copy()
    by_trace: dict[str, dict[str, Any]] = {}
    for trace_id in trace_ids:
        sidecar = split_sidecar.get(trace_id, {})
        rows = prediction_groups.get(trace_id, pd.DataFrame())
        best_rows = _best_probe_rows(rows, artifact)
        split = _probe_index_split(sidecar, rows)
        row_count = int(len(rows))
        best_row_count = int(len(best_rows))
        correct_rate = _mean_numeric(best_rows.get("correct", pd.Series(dtype=float)))
        confidence = _mean_numeric(best_rows.get("confidence", pd.Series(dtype=float)))
        by_trace[trace_id] = {
            "trace_id": trace_id,
            "split": split,
            "split_category": _probe_split_category(split),
            "sidecar": _jsonable(sidecar),
            "available": bool(row_count),
            "row_count": row_count,
            "best_row_count": best_row_count,
            "actual": _json_scalar(
                _dominant_value(best_rows.get("actual", pd.Series(dtype=object)))
            ),
            "predicted": _json_scalar(
                _dominant_value(best_rows.get("predicted", pd.Series(dtype=object)))
            ),
            "confidence": confidence,
            "correct": None if correct_rate is None else bool(correct_rate >= 0.5),
            "correct_rate": correct_rate,
            "eval_split": _dominant_value(best_rows.get("eval_split", pd.Series(dtype=object))),
            "model": _dominant_value(best_rows.get("model", pd.Series(dtype=object))),
            "feature": _dominant_value(best_rows.get("feature", pd.Series(dtype=object))),
        }
    return by_trace

def _best_probe_rows(predictions: pd.DataFrame, artifact: LensArtifact) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    best_model = str(artifact.metrics.get("best_model") or "")
    best_feature = str(artifact.metrics.get("best_feature") or "")
    rows = predictions
    if best_model and "model" in rows:
        model_rows = rows.loc[rows["model"].astype(str) == best_model]
        if not model_rows.empty:
            rows = model_rows
    if best_feature and "feature" in rows:
        feature_rows = rows.loc[rows["feature"].astype(str) == best_feature]
        if not feature_rows.empty:
            rows = feature_rows
    return rows

def _probe_index_split(sidecar: Mapping[str, Any], rows: pd.DataFrame) -> str:
    sidecar_split = sidecar.get("split")
    if not _is_missing_scalar(sidecar_split):
        return str(sidecar_split)
    for column in ("split", "eval_split"):
        if column in rows:
            value = _dominant_value(rows[column])
            if not _is_missing_scalar(value):
                return str(value)
    return ""

def _probe_split_category(split: str) -> str:
    text = str(split or "").strip().lower().replace("-", "_")
    if not text:
        return "unknown"
    if text in {"train", "training"}:
        return "train"
    if text.startswith("test"):
        return "test"
    if text.startswith("val") or text in {"valid", "validation"}:
        return "validation"
    if "heldout" in text or "held_out" in text:
        return "validation"
    return "unknown"

def _probe_index_split_summary(by_trace: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in by_trace.values():
        category = str(item.get("split_category") or "unknown")
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))

def _probe_index_prediction_summary(by_trace: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    summary = {"scored": 0, "unscored": 0, "correct": 0, "incorrect": 0, "unknown": 0}
    for item in by_trace.values():
        if item.get("available"):
            summary["scored"] += 1
        else:
            summary["unscored"] += 1
        correct = item.get("correct")
        if correct is True:
            summary["correct"] += 1
        elif correct is False:
            summary["incorrect"] += 1
        else:
            summary["unknown"] += 1
    return summary

def _probe_prediction_table(dataset: TraceDataset, artifact: LensArtifact) -> pd.DataFrame:
    outputs = artifact.method.get("outputs") if isinstance(artifact.method, Mapping) else None
    relative_path = outputs.get("predictions") if isinstance(outputs, Mapping) else None
    if not relative_path:
        return pd.DataFrame()
    path = dataset.root / str(relative_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

def _saved_episode_probe_predictions(
    dataset: TraceDataset,
    artifact: LensArtifact,
    trace_id: str,
) -> pd.DataFrame:
    path = _episode_probe_prediction_path(dataset, artifact, trace_id)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()

def _score_and_save_episode_probe(
    dataset: TraceDataset,
    artifact: LensArtifact,
    trace_id: str,
) -> pd.DataFrame:
    probe_method = artifact.method.get("probe") if isinstance(artifact.method, Mapping) else None
    state = probe_method.get("best_model_state") if isinstance(probe_method, Mapping) else None
    if not isinstance(state, Mapping) or str(state.get("model") or "linear") != "linear":
        return pd.DataFrame()
    required_arrays = ["weights", "bias", "feature_mean", "feature_scale"]
    if any(name not in artifact.arrays for name in required_arrays):
        return pd.DataFrame()
    selector_payload = dict(artifact.selector or {})
    selector_payload["episodes"] = {"trace_id": trace_id}
    selector = ActivationQuery(**selector_payload)
    try:
        feature_matrix = dataset.select_model_sites(selector).materialize(cache=True)
    except Exception:
        return pd.DataFrame()
    X, rows = feature_matrix.X, feature_matrix.rows
    if rows.empty or X.size == 0:
        return pd.DataFrame()
    rows = _filter_rows_for_probe_feature(rows, str(state.get("feature") or ""))
    if rows.empty:
        return pd.DataFrame()
    X = X[rows.index.to_numpy()]
    rows = rows.reset_index(drop=True)
    target_spec = _normalize_target_spec(str(artifact.metrics.get("target") or "target"))
    target_name = _target_name(target_spec)
    try:
        rows = _attach_episode_metadata(rows, dataset)
        rows = _resolve_probe_target(dataset, rows, target_spec)
        X, rows, _missing = _apply_missing_policy(
            X,
            rows,
            target_name,
            policy=str(target_spec.get("missing_policy") or "drop"),
        )
    except Exception:
        rows[target_name] = None
    if rows.empty or X.size == 0:
        return pd.DataFrame()
    weights = np.asarray(
        dataset.load_artifact_array(artifact, "weights", mmap=True),
        dtype=np.float32,
    )
    bias = np.asarray(dataset.load_artifact_array(artifact, "bias", mmap=True), dtype=np.float32)
    mean = np.asarray(
        dataset.load_artifact_array(artifact, "feature_mean", mmap=True),
        dtype=np.float32,
    )
    scale = np.asarray(
        dataset.load_artifact_array(artifact, "feature_scale", mmap=True),
        dtype=np.float32,
    )
    normalized = (X.astype(np.float32, copy=False) - mean) / np.where(scale == 0, 1.0, scale)
    logits = normalized @ weights.T + bias.reshape(1, -1)
    classes = [str(item) for item in state.get("classes") or []]
    predicted, confidence = _linear_probe_predictions(logits, classes)
    actual = rows[target_name].astype(str) if target_name in rows else pd.Series([None] * len(rows))
    out = rows.copy()
    out["artifact_id"] = artifact.artifact_id
    out["target_name"] = target_name
    out["target_value"] = actual
    out["actual"] = actual
    out["predicted"] = predicted
    out["prediction_value"] = predicted
    out["confidence"] = confidence
    out["correct"] = actual.astype(str).to_numpy() == np.asarray(predicted, dtype=str)
    out["model"] = str(state.get("model") or "linear")
    out["feature"] = str(state.get("feature") or artifact.metrics.get("best_feature") or "")
    out["eval_split"] = "on_demand_episode"
    out["primary_metric"] = str(
        state.get("primary_metric") or artifact.metrics.get("best_primary_metric") or ""
    )
    out["generation_step"] = out.get("generation_step", selector.generation_step)
    path = _episode_probe_prediction_path(dataset, artifact, trace_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out

def _episode_probe_prediction_path(
    dataset: TraceDataset,
    artifact: LensArtifact,
    trace_id: str,
) -> Path:
    return (
        dataset.root
        / "workbench"
        / "episode_probe_predictions"
        / artifact.artifact_id
        / f"{_safe_filename(trace_id)}.parquet"
    )

def _filter_rows_for_probe_feature(rows: pd.DataFrame, feature: str) -> pd.DataFrame:
    filtered = rows.copy()
    for column, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([A-Za-z0-9_.-]+)", feature):
        if column not in filtered:
            continue
        numeric = pd.to_numeric(filtered[column], errors="coerce")
        try:
            target = float(value)
        except ValueError:
            filtered = filtered.loc[filtered[column].astype(str) == value]
        else:
            filtered = filtered.loc[np.isclose(numeric, target, equal_nan=False)]
    simple = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*) ([A-Za-z0-9_.-]+)", feature)
    if simple:
        column, value = simple.groups()
        if column in filtered:
            numeric = pd.to_numeric(filtered[column], errors="coerce")
            try:
                target = float(value)
            except ValueError:
                filtered = filtered.loc[filtered[column].astype(str) == value]
            else:
                filtered = filtered.loc[np.isclose(numeric, target, equal_nan=False)]
    return filtered

def _linear_probe_predictions(
    logits: np.ndarray,
    classes: Sequence[str],
) -> tuple[list[str], list[float]]:
    if logits.ndim != 2 or logits.shape[0] == 0:
        return [], []
    if logits.shape[1] == 1:
        positive = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        negative = 1.0 - positive
        if len(classes) >= 2:
            labels = np.where(positive >= 0.5, classes[1], classes[0])
        else:
            labels = np.where(positive >= 0.5, "True", "False")
        confidence = np.maximum(positive, negative)
        return [str(item) for item in labels], [float(item) for item in confidence]
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    indices = probs.argmax(axis=1)
    labels = [classes[index] if index < len(classes) else str(index) for index in indices]
    confidence = probs.max(axis=1)
    return labels, [float(item) for item in confidence]

def _probe_prediction_rows(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    if predictions.empty:
        return []
    columns = [
        "trace_id",
        "episode_id",
        "task_id",
        "split",
        "target_name",
        "target_value",
        "actual",
        "predicted",
        "prediction_value",
        "confidence",
        "correct",
        "model",
        "feature",
        "layer",
        "policy_call_index",
        "timestep",
        "target_timestep",
        "generation_step",
        "model_site_id",
        "token_space_id",
        "eval_split",
        "primary_metric",
    ]
    available = [column for column in columns if column in predictions.columns]
    rows = predictions.loc[:, available].copy()
    if "confidence" in rows:
        rows = rows.sort_values(
            ["correct", "confidence"],
            ascending=[True, False],
            na_position="last",
        )
    return [_jsonable(row) for row in rows.head(500).to_dict("records")]

def _probe_episode_summary(
    predictions: pd.DataFrame,
    artifact: LensArtifact,
) -> dict[str, Any]:
    if predictions.empty:
        return {}
    best_model = str(artifact.metrics.get("best_model") or "")
    best_feature = str(artifact.metrics.get("best_feature") or "")
    rows = predictions
    if best_model and "model" in rows:
        rows = rows.loc[rows["model"].astype(str) == best_model]
    best_rows = rows
    if best_feature and "feature" in rows:
        matching = rows.loc[rows["feature"].astype(str) == best_feature]
        if not matching.empty:
            best_rows = matching
    if best_rows.empty:
        best_rows = rows if not rows.empty else predictions
    actual = _dominant_value(best_rows.get("actual", pd.Series(dtype=object)))
    predicted = _dominant_value(best_rows.get("predicted", pd.Series(dtype=object)))
    confidence = _mean_numeric(best_rows.get("confidence", pd.Series(dtype=float)))
    correct = _mean_numeric(best_rows.get("correct", pd.Series(dtype=float)))
    best_row = _probe_prediction_rows(best_rows.head(1))
    all_correct = _mean_numeric(rows.get("correct", pd.Series(dtype=float)))
    all_confidence = _mean_numeric(rows.get("confidence", pd.Series(dtype=float)))
    return {
        "actual": _json_scalar(actual),
        "predicted": _json_scalar(predicted),
        "confidence": confidence,
        "correct": None if correct is None else bool(correct >= 0.5),
        "correct_rate": correct,
        "all_cell_correct_rate": all_correct,
        "all_cell_mean_confidence": all_confidence,
        "best_feature": best_feature,
        "best_model": best_model,
        "best_row": best_row[0] if best_row else {},
    }

def _interaction_episode_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    target_objects = _json_parse(row.get("target_objects"))
    if not isinstance(target_objects, list):
        target_objects = []
    parsed_targets = [_optional_text(value) for value in target_objects]
    return {
        "primary_target_object": _optional_text(row.get("primary_target_object")),
        "target_objects": [value for value in parsed_targets if value],
        "target_parse_status": _optional_text(row.get("target_parse_status")),
        "first_moved_object": _optional_text(row.get("first_moved_object")),
        "first_moved_timestep": _optional_int(row.get("first_moved_timestep")),
        "first_moved_is_target": _optional_bool(row.get("first_moved_is_target")),
        "first_lifted_object": _optional_text(row.get("first_lifted_object")),
        "first_lifted_timestep": _optional_int(row.get("first_lifted_timestep")),
        "first_lifted_is_target": _optional_bool(row.get("first_lifted_is_target")),
        "first_contacted_object": _optional_text(row.get("first_contacted_object")),
        "first_contact_timestep": _optional_int(row.get("first_contact_timestep")),
        "scene_family": _optional_text(row.get("scene_family")),
        "task_verb": _optional_text(row.get("task_verb")),
    }

def _interaction_quality_payload(row: Mapping[str, Any]) -> dict[str, bool]:
    keys = [
        "target_parse_failed",
        "multi_target_task",
        "no_object_moved",
        "ambiguous_first_moved",
        "no_object_lifted",
        "ambiguous_first_lifted",
    ]
    return {key: _optional_bool(row.get(key)) for key in keys}

def _interaction_object_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "object_name": _optional_text(row.get("object_name")),
        "object_base_name": _optional_text(row.get("object_base_name")),
        "object_kind": _optional_text(row.get("object_kind")),
        "is_target_object": _optional_bool(row.get("is_target_object")),
        "moved": _optional_bool(row.get("moved")),
        "lifted": _optional_bool(row.get("lifted")),
        "contacted": _optional_bool(row.get("contacted")),
        "movement_onset_timestep": _optional_int(row.get("movement_onset_timestep")),
        "lift_onset_timestep": _optional_int(row.get("lift_onset_timestep")),
        "contact_onset_timestep": _optional_int(row.get("contact_onset_timestep")),
        "max_displacement": _optional_float(row.get("max_displacement")),
        "max_z_delta": _optional_float(row.get("max_z_delta")),
    }
