from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from vla_lens.artifacts import LensArtifact
from vla_lens.server.common import (
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
)
from vla_lens.table_io import read_optional_parquet
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
    path = _artifact_output_path(dataset, str(relative_path))
    return read_optional_parquet(path, context=f"interaction metrics {key}")


def _artifact_output_path(dataset: TraceDataset, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    dataset_path = dataset.root / path
    if dataset_path.exists():
        return dataset_path
    return dataset._dataset_artifact_root() / path


def _episode_probes_payload(
    dataset: TraceDataset,
    query: Mapping[str, list[str]],
) -> dict[str, Any]:
    from vla_lens.server.indexed_probes import indexed_episode_probes_payload

    return indexed_episode_probes_payload(dataset.root, query)


def _probe_index_payload(dataset: TraceDataset) -> dict[str, Any]:
    from vla_lens.server.dataset import _artifact_records

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
    trace_ids = (
        list(trace_ids)
        if trace_ids is not None
        else [bundle.manifest.trace_id for bundle in dataset.bundles]
    )
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
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
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
        frame = read_optional_parquet(path, context="saved probe prediction")
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
    from vla_lens.probes.score_cache import read_probe_score_cache

    score_cache = read_probe_score_cache(dataset, artifact.artifact_id)
    if not score_cache.empty:
        return score_cache
    outputs = artifact.method.get("outputs") if isinstance(artifact.method, Mapping) else None
    relative_path = (
        outputs.get("scored_predictions") or outputs.get("predictions")
        if isinstance(outputs, Mapping)
        else None
    )
    if not relative_path:
        return pd.DataFrame()
    path = dataset.root / str(relative_path)
    if not path.exists():
        path = dataset._dataset_artifact_root() / str(relative_path)
    return read_optional_parquet(path, context="probe prediction")


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
