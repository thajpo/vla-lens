"""Motion-aware follow-up for object geometry probes.

The first geometry study averaged stationary and moving policy calls. This study
keeps those cases separate, saves every comparison prediction, and asks whether
activations add information beyond task context and the robot's own movement.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from vla_lens.artifacts import LensArtifact, make_artifact_id
from vla_lens.probes.geometry_study import (
    GEOMETRY_STUDY_SCHEMA_VERSION,
    _activation_query,
    _align_labeled_geometry_rows,
    _apply_split_contract,
    _canonical_quaternion_rows,
    _geometry_metadata_rows,
    _group_label,
    _limit_rows_by_episode,
    _limited_episode_ids,
    _normalize_spec,
    _required_split_values,
    _source_required_split_values,
    _validate_episode_limit,
    _vectors,
    geometry_target_table,
)
from vla_lens.traces import TraceDataset

MOTION_STUDY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class MotionStudyResult:
    """Saved motion study and all reconstructable result tables."""

    artifact: LensArtifact | None
    candidates: pd.DataFrame
    models: pd.DataFrame
    predictions: pd.DataFrame
    comparisons: pd.DataFrame
    examples: pd.DataFrame
    object_motion: pd.DataFrame
    matched_scenes: pd.DataFrame
    timings: Mapping[str, float]


@dataclass(slots=True)
class _SelectedModel:
    record: dict[str, Any]
    rows: pd.DataFrame
    truth: np.ndarray
    prediction: np.ndarray
    masks: dict[str, np.ndarray]
    baseline_predictions: dict[str, np.ndarray]
    baseline_records: list[dict[str, Any]]


def run_motion_probe_study(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    *,
    save: bool = True,
) -> MotionStudyResult:
    """Run motion detection and moving-only geometry prediction."""

    normalized = _normalize_motion_spec(spec)
    started = time.perf_counter()
    timings: dict[str, float] = {}
    candidates: list[dict[str, Any]] = []
    selected: dict[tuple[str, str, str, float], _SelectedModel] = {}
    base_rows: pd.DataFrame | None = None
    source_trace_ids: set[str] = set()
    required_split_values = _required_split_values(normalized["split"])
    _validate_episode_limit(normalized.get("limit_episodes"), required_split_values)
    limited_episode_ids = _limited_episode_ids(
        dataset,
        normalized.get("limit_episodes"),
        required_split_values=_source_required_split_values(normalized["split"]),
    )

    for feature_spec in normalized["features"]:
        feature_started = time.perf_counter()
        query = _activation_query(feature_spec)
        if limited_episode_ids is not None:
            query = replace(
                query,
                episodes={**dict(query.episodes), "trace_id": limited_episode_ids},
            )
        matrix = dataset.select_model_sites(query).materialize(cache=True)
        timings[f"feature:{feature_spec['id']}:materialize_seconds"] = (
            time.perf_counter() - feature_started
        )

        prepare_started = time.perf_counter()
        rows, X = _prepare_feature_rows(
            dataset,
            matrix.rows,
            matrix.X,
            normalized,
        )
        if base_rows is None:
            base_rows = rows.copy()
        source_trace_ids.update(rows["trace_id"].astype(str).unique())
        timings[f"feature:{feature_spec['id']}:prepare_seconds"] = (
            time.perf_counter() - prepare_started
        )

        fit_started = time.perf_counter()
        feature_candidates, feature_selected = _fit_feature_motion_models(
            X,
            rows,
            feature_id=str(feature_spec["id"]),
            sweep_columns=[str(value) for value in feature_spec.get("sweep", ["layer"])],
            split=normalized["split"],
            pca_dims=[int(value) for value in normalized["probe"]["pca_dims"]],
            ridge_alphas=[float(value) for value in normalized["probe"]["ridge_alphas"]],
            baseline_columns=[str(value) for value in normalized["baseline_columns"]],
            thresholds=normalized["movement"],
        )
        candidates.extend(feature_candidates)
        for key, value in feature_selected.items():
            current = selected.get(key)
            if current is None or _is_better(value.record, current.record):
                selected[key] = value
        timings[f"feature:{feature_spec['id']}:fit_seconds"] = (
            time.perf_counter() - fit_started
        )

    model_rows, prediction_rows = _selected_output_rows(selected, normalized["split"])
    candidate_frame = pd.DataFrame.from_records(candidates)
    model_frame = pd.DataFrame.from_records(model_rows)
    prediction_frame = pd.DataFrame.from_records(prediction_rows)
    comparison_frame = _comparison_table(prediction_frame)
    example_frame = _example_table(prediction_frame, model_frame)
    object_motion, matched_scenes = _object_motion_tables(
        dataset,
        base_rows if base_rows is not None else pd.DataFrame(),
        normalized,
    )
    timings["total_seconds"] = time.perf_counter() - started

    artifact = (
        _save_motion_study(
            dataset,
            normalized,
            candidate_frame,
            model_frame,
            prediction_frame,
            comparison_frame,
            example_frame,
            object_motion,
            matched_scenes,
            timings,
            source_trace_ids,
        )
        if save
        else None
    )
    return MotionStudyResult(
        artifact=artifact,
        candidates=candidate_frame,
        models=model_frame,
        predictions=prediction_frame,
        comparisons=comparison_frame,
        examples=example_frame,
        object_motion=object_motion,
        matched_scenes=matched_scenes,
        timings=timings,
    )


def _prepare_feature_rows(
    dataset: TraceDataset,
    feature_rows: pd.DataFrame,
    features: np.ndarray,
    spec: Mapping[str, Any],
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = _geometry_metadata_rows(dataset, feature_rows, cache=True)
    rows = _apply_split_contract(rows, spec["split"])
    rows, X = _limit_rows_by_episode(
        rows,
        features,
        spec.get("limit_episodes"),
        split_column=str(spec["split"]["column"]),
        required_split_values=_required_split_values(spec["split"]),
    )
    object_column = str(spec["object_column"])
    rows, X = _align_labeled_geometry_rows(
        dataset,
        rows,
        X,
        object_column=object_column,
    )
    rows["object_name"] = rows[object_column].astype(str)
    rows["task_key"] = _task_keys(rows)
    rows["period"] = np.where(
        pd.to_numeric(rows.get("policy_call_index"), errors="coerce").fillna(0).astype(int)
        <= 1,
        "initial_or_first_interval",
        "later_interval",
    )
    finite = np.isfinite(X).any(axis=1)
    finite &= np.isfinite(_vectors(rows["position_previous_delta"], 3)).all(axis=1)
    rows = rows.loc[finite].reset_index(drop=True)
    X = X[finite]
    # The first policy call has no previous interval and is not a movement example.
    usable = ~rows["is_first_policy_call"].astype(bool).to_numpy()
    return rows.loc[usable].reset_index(drop=True), X[usable]


def _fit_feature_motion_models(
    X: np.ndarray,
    rows: pd.DataFrame,
    *,
    feature_id: str,
    sweep_columns: Sequence[str],
    split: Mapping[str, Any],
    pca_dims: Sequence[int],
    ridge_alphas: Sequence[float],
    baseline_columns: Sequence[str],
    thresholds: Mapping[str, Sequence[float]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, float], _SelectedModel]]:
    candidates: list[dict[str, Any]] = []
    selected: dict[tuple[str, str, str, float], _SelectedModel] = {}
    split_column = str(split["column"])
    split_values = {
        "train": str(split["train_value"]),
        "selection": str(split["selection_value"]),
        "test": str(split["test_value"]),
    }
    target_values = _motion_targets(rows)
    group_columns = [column for column in sweep_columns if column in rows]
    grouped: list[tuple[Any, np.ndarray]] = [((), rows.index.to_numpy())]
    if group_columns:
        grouped = list(rows.groupby(group_columns, dropna=False, sort=True).indices.items())

    for group_value, raw_indices in grouped:
        indices = np.asarray(raw_indices, dtype=np.int64)
        group_rows = rows.iloc[indices].reset_index(drop=True)
        group_X = np.asarray(X[indices], dtype=np.float64)
        split_array = group_rows[split_column].astype(str).to_numpy()
        base_masks = {
            name: split_array == value for name, value in split_values.items()
        }
        if not all(mask.any() for mask in base_masks.values()):
            continue
        projected = _project_features(group_X, base_masks["train"], pca_dims)
        designs = _comparison_designs(
            group_rows,
            train_mask=base_masks["train"],
            categorical_columns=baseline_columns,
        )
        group_label = _group_label(group_columns, group_value)

        for target_name, target in target_values.items():
            truth = target["values"][indices]
            amounts = target["amounts"][indices]
            for threshold in thresholds[target_name]:
                threshold = float(threshold)
                detection_candidates, detection_best = _fit_detection_models(
                    projected,
                    designs,
                    group_rows,
                    amounts,
                    base_masks,
                    threshold,
                    ridge_alphas,
                    feature_id,
                    group_label,
                    target_name,
                )
                candidates.extend(detection_candidates)
                key = (feature_id, "movement_detection", target_name, threshold)
                current = selected.get(key)
                if current is None or _is_better(detection_best.record, current.record):
                    selected[key] = detection_best

                regression_candidates, regression_best = _fit_regression_models(
                    projected,
                    designs,
                    group_rows,
                    truth,
                    amounts,
                    base_masks,
                    threshold,
                    ridge_alphas,
                    feature_id,
                    group_label,
                    target_name,
                )
                candidates.extend(regression_candidates)
                key = (feature_id, "moving_geometry", target_name, threshold)
                current = selected.get(key)
                if current is None or _is_better(regression_best.record, current.record):
                    selected[key] = regression_best
    return candidates, selected


def _fit_detection_models(
    projected: Mapping[int, np.ndarray],
    designs: Mapping[str, np.ndarray],
    rows: pd.DataFrame,
    amounts: np.ndarray,
    base_masks: Mapping[str, np.ndarray],
    threshold: float,
    alphas: Sequence[float],
    feature_id: str,
    feature_group: str,
    target: str,
) -> tuple[list[dict[str, Any]], _SelectedModel]:
    labels = amounts > threshold
    candidates: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    best_prediction: np.ndarray | None = None
    for pca_dim, values in projected.items():
        for alpha in alphas:
            prediction = _fit_binary_ridge(values, labels, base_masks["train"], alpha)
            record = _candidate_record(
                "movement_detection",
                feature_id,
                feature_group,
                target,
                threshold,
                pca_dim,
                alpha,
                _classification_metrics(labels, prediction, rows, base_masks),
            )
            candidates.append(record)
            if best_record is None or _is_better(record, best_record):
                best_record = record
                best_prediction = prediction
    assert best_record is not None and best_prediction is not None
    baseline_predictions, baseline_records = _detection_baselines(
        designs,
        labels,
        rows,
        base_masks,
        alphas,
        target,
        threshold,
    )
    return candidates, _SelectedModel(
        record=best_record,
        rows=rows,
        truth=labels.astype(np.float64).reshape(-1, 1),
        prediction=best_prediction.reshape(-1, 1),
        masks=dict(base_masks),
        baseline_predictions={
            name: value.reshape(-1, 1) for name, value in baseline_predictions.items()
        },
        baseline_records=baseline_records,
    )


def _fit_regression_models(
    projected: Mapping[int, np.ndarray],
    designs: Mapping[str, np.ndarray],
    rows: pd.DataFrame,
    truth: np.ndarray,
    amounts: np.ndarray,
    base_masks: Mapping[str, np.ndarray],
    threshold: float,
    alphas: Sequence[float],
    feature_id: str,
    feature_group: str,
    target: str,
) -> tuple[list[dict[str, Any]], _SelectedModel]:
    masks = {name: mask & (amounts > threshold) for name, mask in base_masks.items()}
    if not all(mask.any() for mask in masks.values()):
        raise ValueError(f"No moving rows for {target} above {threshold}")
    candidates: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    best_prediction: np.ndarray | None = None
    for pca_dim, values in projected.items():
        for alpha in alphas:
            prediction = _fit_ridge(values, truth, masks["train"], alpha)
            record = _candidate_record(
                "moving_geometry",
                feature_id,
                feature_group,
                target,
                threshold,
                pca_dim,
                alpha,
                _geometry_metrics(target, truth, prediction, rows, masks),
            )
            candidates.append(record)
            if best_record is None or _is_better(record, best_record):
                best_record = record
                best_prediction = prediction
    assert best_record is not None and best_prediction is not None
    baseline_predictions, baseline_records = _regression_baselines(
        designs,
        truth,
        rows,
        masks,
        alphas,
        target,
        threshold,
    )
    return candidates, _SelectedModel(
        record=best_record,
        rows=rows,
        truth=truth,
        prediction=best_prediction,
        masks=masks,
        baseline_predictions=baseline_predictions,
        baseline_records=baseline_records,
    )


def _project_features(
    X: np.ndarray,
    train_mask: np.ndarray,
    pca_dims: Sequence[int],
) -> dict[int, np.ndarray]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X[train_mask])
    all_scaled = scaler.transform(X)
    max_dim = min(max(pca_dims), train_scaled.shape[0] - 1, train_scaled.shape[1])
    projector = PCA(
        n_components=max_dim,
        svd_solver="randomized",
        iterated_power=2,
        random_state=0,
    )
    projector.fit(train_scaled)
    projected = projector.transform(all_scaled)
    return {
        min(int(dim), max_dim): projected[:, : min(int(dim), max_dim)]
        for dim in sorted(set(pca_dims))
    }


def _comparison_designs(
    rows: pd.DataFrame,
    *,
    train_mask: np.ndarray,
    categorical_columns: Sequence[str],
) -> dict[str, np.ndarray]:
    available = [column for column in categorical_columns if column in rows]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    context = encoder.fit_transform(rows.loc[train_mask, available].astype(str))
    context_all = encoder.transform(rows[available].astype(str))
    robot = _robot_features(rows)
    scaler = StandardScaler()
    scaler.fit(robot[train_mask])
    robot_all = scaler.transform(robot)
    del context
    return {
        "context": context_all,
        "robot_movement": robot_all,
        "context_and_robot": np.concatenate([context_all, robot_all], axis=1),
    }


def _robot_features(rows: pd.DataFrame) -> np.ndarray:
    eef_position = _vectors(rows["eef_position_previous_delta"], 3)
    eef_quat = _canonical_quaternion_rows(
        _vectors(rows["eef_orientation_previous_relative_quat"], 4)
    )
    eef_rotation = Rotation.from_quat(eef_quat).as_rotvec()
    action_mean = np.stack(rows["executed_action_mean"].map(np.asarray))
    action_sum = np.stack(rows["executed_action_sum"].map(np.asarray))
    action_std = np.stack(rows["executed_action_std"].map(np.asarray))
    return np.concatenate(
        [eef_position, eef_rotation, action_mean, action_sum, action_std], axis=1
    ).astype(np.float64)


def _fit_binary_ridge(
    X: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    train_labels = labels[train_mask].astype(int)
    counts = np.bincount(train_labels, minlength=2)
    weights = np.array(
        [len(train_labels) / (2 * max(1, counts[value])) for value in train_labels]
    )
    model = Ridge(alpha=float(alpha))
    model.fit(X[train_mask], train_labels.astype(float), sample_weight=weights)
    return np.asarray(model.predict(X), dtype=np.float64)


def _fit_ridge(
    X: np.ndarray,
    truth: np.ndarray,
    train_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    model = Ridge(alpha=float(alpha))
    model.fit(X[train_mask], truth[train_mask])
    return np.asarray(model.predict(X), dtype=np.float64)


def _detection_baselines(
    designs: Mapping[str, np.ndarray],
    labels: np.ndarray,
    rows: pd.DataFrame,
    masks: Mapping[str, np.ndarray],
    alphas: Sequence[float],
    target: str,
    threshold: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    predictions = {"always_not_moving": np.zeros(len(rows), dtype=np.float64)}
    records = [
        _baseline_record(
            "movement_detection",
            "always_not_moving",
            target,
            threshold,
            _classification_metrics(labels, predictions["always_not_moving"], rows, masks),
        )
    ]
    for name, design in designs.items():
        best_record: dict[str, Any] | None = None
        best_prediction: np.ndarray | None = None
        for alpha in alphas:
            prediction = _fit_binary_ridge(design, labels, masks["train"], alpha)
            record = _baseline_record(
                "movement_detection",
                name,
                target,
                threshold,
                _classification_metrics(labels, prediction, rows, masks),
                alpha=alpha,
            )
            if best_record is None or _is_better(record, best_record):
                best_record, best_prediction = record, prediction
        assert best_record is not None and best_prediction is not None
        records.append(best_record)
        predictions[name] = best_prediction
    return predictions, records


def _regression_baselines(
    designs: Mapping[str, np.ndarray],
    truth: np.ndarray,
    rows: pd.DataFrame,
    masks: Mapping[str, np.ndarray],
    alphas: Sequence[float],
    target: str,
    threshold: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    zero = np.zeros_like(truth)
    average = np.repeat(truth[masks["train"]].mean(axis=0, keepdims=True), len(rows), axis=0)
    predictions = {"no_movement": zero, "average_movement": average}
    records = [
        _baseline_record(
            "moving_geometry",
            name,
            target,
            threshold,
            _geometry_metrics(target, truth, prediction, rows, masks),
        )
        for name, prediction in predictions.items()
    ]
    for name, design in designs.items():
        best_record: dict[str, Any] | None = None
        best_prediction: np.ndarray | None = None
        for alpha in alphas:
            prediction = _fit_ridge(design, truth, masks["train"], alpha)
            record = _baseline_record(
                "moving_geometry",
                name,
                target,
                threshold,
                _geometry_metrics(target, truth, prediction, rows, masks),
                alpha=alpha,
            )
            if best_record is None or _is_better(record, best_record):
                best_record, best_prediction = record, prediction
        assert best_record is not None and best_prediction is not None
        records.append(best_record)
        predictions[name] = best_prediction
    return predictions, records


def _classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    rows: pd.DataFrame,
    masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split_name in ["selection", "test"]:
        mask = masks[split_name]
        y = labels[mask].astype(int)
        score = np.asarray(scores)[mask]
        predicted = score >= 0.5
        result[f"{split_name}_balanced_accuracy"] = float(
            balanced_accuracy_score(y, predicted)
        )
        result[f"{split_name}_accuracy"] = float(np.mean(predicted == y))
        result[f"{split_name}_positive_rate"] = float(np.mean(y))
        result[f"{split_name}_rows"] = int(mask.sum())
        result[f"{split_name}_tasks"] = int(rows.loc[mask, "task_key"].nunique())
        result[f"{split_name}_roc_auc"] = _safe_binary_metric(roc_auc_score, y, score)
        result[f"{split_name}_average_precision"] = _safe_binary_metric(
            average_precision_score, y, score
        )
    return result


def _geometry_metrics(
    target: str,
    truth: np.ndarray,
    predicted: np.ndarray,
    rows: pd.DataFrame,
    masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split_name in ["selection", "test"]:
        mask = masks[split_name]
        diagnostics = _geometry_row_diagnostics(target, truth[mask], predicted[mask])
        episode_error = (
            pd.DataFrame(
                {
                    "trace_id": rows.loc[mask, "trace_id"].astype(str).to_numpy(),
                    "error": diagnostics["vector_error"],
                }
            )
            .groupby("trace_id")["error"]
            .mean()
        )
        result[f"{split_name}_error"] = float(episode_error.mean())
        result[f"{split_name}_row_error"] = float(
            np.mean(diagnostics["vector_error"])
        )
        result[f"{split_name}_magnitude_error"] = float(
            np.mean(diagnostics["magnitude_error"])
        )
        finite_direction = diagnostics["direction_error"][
            np.isfinite(diagnostics["direction_error"])
        ]
        result[f"{split_name}_direction_error"] = (
            float(np.mean(finite_direction)) if finite_direction.size else None
        )
        result[f"{split_name}_rows"] = int(mask.sum())
        result[f"{split_name}_tasks"] = int(rows.loc[mask, "task_key"].nunique())
    return result


def _geometry_row_diagnostics(
    target: str,
    truth: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, np.ndarray]:
    truth = np.asarray(truth, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    truth_norm = np.linalg.norm(truth, axis=1)
    predicted_norm = np.linalg.norm(predicted, axis=1)
    if target == "position":
        vector_error = np.linalg.norm(predicted - truth, axis=1)
        truth_amount = truth_norm
        predicted_amount = predicted_norm
    else:
        relative = Rotation.from_rotvec(predicted).inv() * Rotation.from_rotvec(truth)
        vector_error = np.degrees(relative.magnitude())
        truth_amount = np.degrees(truth_norm)
        predicted_amount = np.degrees(predicted_norm)
    direction_error = np.full(len(truth), np.nan, dtype=np.float64)
    valid = (truth_norm > 1e-9) & (predicted_norm > 1e-9)
    if valid.any():
        cosine = np.sum(truth[valid] * predicted[valid], axis=1) / (
            truth_norm[valid] * predicted_norm[valid]
        )
        direction_error[valid] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return {
        "vector_error": vector_error,
        "truth_amount": truth_amount,
        "predicted_amount": predicted_amount,
        "magnitude_error": np.abs(predicted_amount - truth_amount),
        "direction_error": direction_error,
    }


def _motion_targets(rows: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    position = _vectors(rows["position_previous_delta"], 3)
    quaternion = _canonical_quaternion_rows(
        _vectors(rows["orientation_previous_relative_quat"], 4)
    )
    rotation = Rotation.from_quat(quaternion).as_rotvec()
    return {
        "position": {"values": position, "amounts": np.linalg.norm(position, axis=1)},
        "rotation": {
            "values": rotation,
            "amounts": np.degrees(np.linalg.norm(rotation, axis=1)),
        },
    }


def _candidate_record(
    analysis: str,
    feature_id: str,
    feature_group: str,
    target: str,
    threshold: float,
    pca_dim: int,
    alpha: float,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "analysis": analysis,
        "model": "activation",
        "feature_id": feature_id,
        "feature_group": feature_group,
        "target": target,
        "threshold": threshold,
        "pca_dim": int(pca_dim),
        "ridge_alpha": float(alpha),
        **dict(metrics),
    }


def _baseline_record(
    analysis: str,
    name: str,
    target: str,
    threshold: float,
    metrics: Mapping[str, Any],
    *,
    alpha: float | None = None,
) -> dict[str, Any]:
    return {
        "analysis": analysis,
        "model": name,
        "feature_id": "comparison",
        "feature_group": "all",
        "target": target,
        "threshold": threshold,
        "pca_dim": None,
        "ridge_alpha": alpha,
        **dict(metrics),
    }


def _is_better(candidate: Mapping[str, Any], current: Mapping[str, Any]) -> bool:
    if candidate["analysis"] == "movement_detection":
        left = (
            float(candidate["selection_balanced_accuracy"]),
            float(candidate.get("selection_average_precision") or 0.0),
        )
        right = (
            float(current["selection_balanced_accuracy"]),
            float(current.get("selection_average_precision") or 0.0),
        )
        return left > right
    return float(candidate["selection_error"]) < float(current["selection_error"])


def _selected_output_rows(
    selected: Mapping[tuple[str, str, str, float], _SelectedModel],
    split: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    split_names = {
        "selection": str(split["selection_value"]),
        "test": str(split["test_value"]),
    }
    for (feature_id, analysis, target, threshold), chosen in sorted(selected.items()):
        model_rows.append({**dict(chosen.record), "compared_feature_id": feature_id})
        model_rows.extend(
            {**record, "compared_feature_id": feature_id}
            for record in chosen.baseline_records
        )
        all_predictions = {"activation": chosen.prediction, **chosen.baseline_predictions}
        for split_role, split_name in split_names.items():
            mask = chosen.masks[split_role]
            scoped_rows = chosen.rows.loc[mask].reset_index(drop=True)
            truth = chosen.truth[mask]
            for model_name, raw_prediction in all_predictions.items():
                predicted = np.asarray(raw_prediction)[mask]
                if analysis == "movement_detection":
                    for index, row in scoped_rows.iterrows():
                        score = float(predicted[index, 0])
                        label = bool(truth[index, 0])
                        predictions.append(
                            _source_prediction_row(
                                row,
                                analysis,
                                target,
                                threshold,
                                split_name,
                                model_name,
                                chosen.record,
                                {
                                    "label": label,
                                    "prediction_score": score,
                                    "prediction_label": bool(score >= 0.5),
                                    "correct": bool((score >= 0.5) == label),
                                },
                            )
                        )
                else:
                    diagnostics = _geometry_row_diagnostics(target, truth, predicted)
                    for index, row in scoped_rows.iterrows():
                        predictions.append(
                            _source_prediction_row(
                                row,
                                analysis,
                                target,
                                threshold,
                                split_name,
                                model_name,
                                chosen.record,
                                {
                                    "target_value": truth[index].tolist(),
                                    "prediction_value": predicted[index].tolist(),
                                    "movement_amount": float(
                                        diagnostics["truth_amount"][index]
                                    ),
                                    "predicted_amount": float(
                                        diagnostics["predicted_amount"][index]
                                    ),
                                    "vector_error": float(
                                        diagnostics["vector_error"][index]
                                    ),
                                    "magnitude_error": float(
                                        diagnostics["magnitude_error"][index]
                                    ),
                                    "direction_error": _finite_or_none(
                                        diagnostics["direction_error"][index]
                                    ),
                                },
                            )
                        )
    return model_rows, predictions


def _source_prediction_row(
    row: pd.Series,
    analysis: str,
    target: str,
    threshold: float,
    split_name: str,
    model_name: str,
    selected: Mapping[str, Any],
    values: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "analysis": analysis,
        "target": target,
        "threshold": threshold,
        "split": split_name,
        "model": model_name,
        "selected_feature_id": selected["feature_id"],
        "selected_feature_group": selected["feature_group"],
        "trace_id": str(row["trace_id"]),
        "episode_id": row.get("episode_id"),
        "task_key": str(row["task_key"]),
        "benchmark": row.get("benchmark"),
        "task_id": row.get("task_id"),
        "task_name": row.get("task_name"),
        "task_phase": row.get("task_phase"),
        "period": row.get("period"),
        "timestep": int(row["timestep"]),
        "policy_call_index": int(row.get("policy_call_index", 0)),
        "object_name": str(row["object_name"]),
        "frame_cameras": ["main", "wrist"],
        **dict(values),
    }


def _comparison_table(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    keys = ["selected_feature_id", "analysis", "target", "threshold", "split"]
    for group_key, group in predictions.groupby(keys, dropna=False, sort=True):
        feature_id, analysis, target, threshold, split_name = group_key
        activation = group.loc[group["model"] == "activation"]
        if activation.empty:
            continue
        for model_name in sorted(set(group["model"]) - {"activation"}):
            comparison = group.loc[group["model"] == model_name]
            row_keys = ["trace_id", "timestep", "task_key"]
            if analysis == "movement_detection":
                left = activation[
                    row_keys + ["label", "prediction_label"]
                ].rename(columns={"prediction_label": "activation_prediction"})
                right = comparison[row_keys + ["prediction_label"]].rename(
                    columns={"prediction_label": "comparison_prediction"}
                )
                higher_is_better = True
            else:
                left = activation[row_keys + ["vector_error"]].rename(
                    columns={"vector_error": "activation_value"}
                )
                right = comparison[row_keys + ["vector_error"]].rename(
                    columns={"vector_error": "comparison_value"}
                )
                higher_is_better = False
            paired = left.merge(right, on=row_keys, validate="one_to_one")
            if higher_is_better:
                task_values = _task_balanced_accuracy(paired)
            else:
                task_values = paired.groupby("task_key")[
                    ["activation_value", "comparison_value"]
                ].mean()
            difference = (
                task_values["activation_value"] - task_values["comparison_value"]
            ).to_numpy(dtype=np.float64)
            if not higher_is_better:
                difference *= -1.0
            ci_low, ci_high, p_value = _paired_uncertainty(difference)
            records.append(
                {
                    "analysis": analysis,
                    "feature_id": feature_id,
                    "target": target,
                    "threshold": float(threshold),
                    "split": split_name,
                    "comparison_model": model_name,
                    "tasks": int(len(task_values)),
                    "activation_mean": float(task_values["activation_value"].mean()),
                    "comparison_mean": float(task_values["comparison_value"].mean()),
                    "activation_advantage": float(difference.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "two_sided_p_value": p_value,
                    "advantage_definition": (
                        "activation accuracy minus comparison accuracy"
                        if higher_is_better
                        else "comparison error minus activation error"
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _task_balanced_accuracy(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for task_key, group in rows.groupby("task_key", sort=False):
        labels = group["label"].astype(bool).to_numpy()
        if len(np.unique(labels)) < 2:
            continue
        records.append(
            {
                "task_key": task_key,
                "activation_value": balanced_accuracy_score(
                    labels, group["activation_prediction"].astype(bool)
                ),
                "comparison_value": balanced_accuracy_score(
                    labels, group["comparison_prediction"].astype(bool)
                ),
            }
        )
    if not records:
        return pd.DataFrame(columns=["activation_value", "comparison_value"])
    return pd.DataFrame.from_records(records).set_index("task_key")


def _paired_uncertainty(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return np.nan, np.nan, np.nan
    seed = int(np.round(np.sum(np.abs(values)) * 1e9)) % (2**32)
    rng = np.random.default_rng(seed)
    count = 20_000
    samples = values[rng.integers(0, len(values), size=(count, len(values)))].mean(axis=1)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(count, len(values)))
    permuted = (signs * values).mean(axis=1)
    observed = float(values.mean())
    p_value = float((np.count_nonzero(np.abs(permuted) >= abs(observed)) + 1) / (count + 1))
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high), p_value


def _example_table(predictions: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or models.empty:
        return pd.DataFrame()
    records: list[pd.DataFrame] = []
    selected_baselines = models.loc[
        (models["analysis"] == "moving_geometry") & (models["model"] != "activation")
    ].copy()
    for (feature_id, target, threshold), baseline_rows in selected_baselines.groupby(
        ["compared_feature_id", "target", "threshold"], sort=True
    ):
        baseline = baseline_rows.sort_values("selection_error").iloc[0]["model"]
        scoped = predictions.loc[
            (predictions["analysis"] == "moving_geometry")
            & (predictions["selected_feature_id"] == feature_id)
            & (predictions["target"] == target)
            & (predictions["threshold"] == threshold)
            & predictions["split"].astype(str).str.startswith("test")
        ]
        keys = ["trace_id", "timestep"]
        activation = scoped.loc[scoped["model"] == "activation"].copy()
        comparison = scoped.loc[scoped["model"] == baseline, keys + ["vector_error"]].rename(
            columns={"vector_error": "comparison_error"}
        )
        merged = activation.merge(comparison, on=keys, validate="one_to_one")
        merged["comparison_model"] = baseline
        merged["activation_advantage"] = merged["comparison_error"] - merged["vector_error"]
        records.append(merged.nlargest(12, "activation_advantage"))
        records.append(merged.nsmallest(12, "activation_advantage"))
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def _object_motion_tables(
    dataset: TraceDataset,
    base_rows: pd.DataFrame,
    spec: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if base_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    roles = _object_role_table(dataset)
    if roles.empty:
        return pd.DataFrame(), pd.DataFrame()
    scene_keys = base_rows[
        [
            "trace_id",
            "episode_id",
            "timestep",
            "policy_call_index",
            str(spec["split"]["column"]),
            "primary_target_object",
            "task_key",
            "task_phase",
        ]
    ].drop_duplicates()
    expanded = scene_keys.merge(
        roles[
            [
                "trace_id",
                "object_index",
                "object_name",
                "object_base_name",
                "role_manipulated",
                "role_receptacle",
                "role_distractor",
                "prompt_mentioned",
            ]
        ],
        on="trace_id",
        how="inner",
        validate="many_to_many",
    )
    geometry = geometry_target_table(dataset, expanded, object_column="object_name", cache=True)
    expanded = expanded.merge(
        geometry,
        on=["trace_id", "timestep", "object_name"],
        how="inner",
        validate="many_to_one",
    )
    position = _vectors(expanded["position_previous_delta"], 3)
    quaternion = _canonical_quaternion_rows(
        _vectors(expanded["orientation_previous_relative_quat"], 4)
    )
    expanded["position_change_m"] = np.linalg.norm(position, axis=1)
    expanded["rotation_change_deg"] = np.degrees(
        Rotation.from_quat(quaternion).magnitude()
    )
    expanded["is_primary_target"] = (
        expanded["object_name"].astype(str)
        == expanded["primary_target_object"].astype(str)
    )
    expanded["initial_settling_interval"] = (
        pd.to_numeric(expanded["policy_call_index"], errors="coerce").fillna(0).astype(int)
        <= 1
    )
    expanded["position_group"] = pd.cut(
        expanded["position_change_m"],
        bins=[-np.inf, 0.01, 0.10, np.inf],
        labels=["still_or_small", "moderate", "large"],
    ).astype(str)
    expanded["rotation_group"] = pd.cut(
        expanded["rotation_change_deg"],
        bins=[-np.inf, 1.0, 15.0, np.inf],
        labels=["still_or_small", "moderate", "large"],
    ).astype(str)
    matched = _matched_scene_rows(expanded)
    keep = [
        "trace_id",
        "episode_id",
        "task_key",
        "timestep",
        "policy_call_index",
        str(spec["split"]["column"]),
        "task_phase",
        "primary_target_object",
        "object_index",
        "object_name",
        "object_base_name",
        "role_manipulated",
        "role_receptacle",
        "role_distractor",
        "prompt_mentioned",
        "is_primary_target",
        "initial_settling_interval",
        "position_previous_delta",
        "orientation_previous_relative_quat",
        "position_change_m",
        "rotation_change_deg",
        "position_group",
        "rotation_group",
    ]
    return expanded[keep].reset_index(drop=True), matched


def _matched_scene_rows(objects: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (trace_id, timestep), group in objects.groupby(["trace_id", "timestep"], sort=False):
        target = group.loc[group["is_primary_target"]]
        others = group.loc[~group["is_primary_target"]]
        if target.empty:
            continue
        target_row = target.iloc[0]
        records.append(
            {
                "trace_id": str(trace_id),
                "timestep": int(timestep),
                "policy_call_index": int(target_row["policy_call_index"]),
                "task_key": target_row["task_key"],
                "split": target_row.get("split"),
                "task_phase": target_row.get("task_phase"),
                "target_object": target_row["object_name"],
                "target_position_change_m": float(target_row["position_change_m"]),
                "target_rotation_change_deg": float(target_row["rotation_change_deg"]),
                "other_object_count": int(len(others)),
                "stationary_other_count": int((others["position_change_m"] <= 0.01).sum()),
                "moving_other_count": int((others["position_change_m"] > 0.01).sum()),
                "max_other_position_change_m": (
                    float(others["position_change_m"].max()) if not others.empty else None
                ),
                "initial_settling_interval": bool(target_row["initial_settling_interval"]),
            }
        )
    return pd.DataFrame.from_records(records)


def _object_role_table(dataset: TraceDataset) -> pd.DataFrame:
    index = dataset.artifact_index
    matches = index.loc[index.get("artifact_type", "").astype(str) == "pi05_object_flow"]
    if matches.empty:
        return pd.DataFrame()
    artifact_id = str(matches.sort_values("created_utc").iloc[-1]["artifact_id"])
    artifact = dataset.load_artifact(artifact_id)
    path = artifact.method.get("outputs", {}).get("object_roles")
    if not path:
        return pd.DataFrame()
    return pd.read_parquet(dataset.root / str(path))


def _save_motion_study(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    candidates: pd.DataFrame,
    models: pd.DataFrame,
    predictions: pd.DataFrame,
    comparisons: pd.DataFrame,
    examples: pd.DataFrame,
    object_motion: pd.DataFrame,
    matched_scenes: pd.DataFrame,
    timings: Mapping[str, float],
    source_trace_ids: Sequence[str],
) -> LensArtifact:
    artifact_id = make_artifact_id("geometry_motion_study", str(spec["name"]))
    relative_dir = Path("artifacts") / artifact_id
    outputs = {
        name: str(relative_dir / f"{name}.parquet")
        for name in [
            "candidates",
            "models",
            "predictions",
            "comparisons",
            "examples",
            "object_motion",
            "matched_scenes",
        ]
    }
    artifact = LensArtifact(
        artifact_id=artifact_id,
        artifact_type="geometry_motion_study",
        name=str(spec["name"]),
        group_id="geometry_motion_studies",
        scope="dataset",
        selector={"features": spec["features"], "object_column": spec["object_column"]},
        method={
            "workflow": "run_motion_probe_study",
            "schema_version": MOTION_STUDY_SCHEMA_VERSION,
            "geometry_schema_version": GEOMETRY_STUDY_SCHEMA_VERSION,
            "research": {
                key: spec[key]
                for key in ["question", "hypothesis_family", "intended_claim"]
                if spec.get(key)
            },
            "source_artifact_ids": list(spec.get("source_artifact_ids") or []),
            "split": spec["split"],
            "movement": spec["movement"],
            "probe": spec["probe"],
            "baseline_columns": spec["baseline_columns"],
            "comparison_contract": {
                "context": "categorical task, scene, object, phase, and policy-call data",
                "robot_movement": "end-effector change and executed-action summaries",
                "context_and_robot": "combined context and robot movement",
                "uncertainty_unit": "task",
                "confirmation_status": (
                    "exploratory; final held-out tasks were viewed before this follow-up"
                ),
            },
            "outputs": outputs,
            "timings_seconds": dict(timings),
        },
        metrics={
            "candidate_count": int(len(candidates)),
            "model_count": int(len(models)),
            "prediction_count": int(len(predictions)),
            "object_motion_rows": int(len(object_motion)),
            "total_seconds": float(timings.get("total_seconds", 0.0)),
        },
        display={
            "kind": "geometry_motion_study",
            "status": "exploratory",
            "summary": _display_summary(models, comparisons),
        },
        tags=("probe", "geometry", "motion", "exploratory"),
        source_trace_ids=tuple(sorted(str(value) for value in source_trace_ids)),
    )
    saved = dataset.save_artifact(artifact)
    artifact_dir = dataset._dataset_artifact_root() / relative_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [
        ("candidates", candidates),
        ("models", models),
        ("predictions", predictions),
        ("comparisons", comparisons),
        ("examples", examples),
        ("object_motion", object_motion),
        ("matched_scenes", matched_scenes),
    ]:
        frame.to_parquet(artifact_dir / f"{name}.parquet", index=False)
    return saved


def _display_summary(models: pd.DataFrame, comparisons: pd.DataFrame) -> list[dict[str, Any]]:
    if models.empty:
        return []
    selected = models.loc[models["model"] == "activation"].copy()
    columns = [
        "analysis",
        "target",
        "threshold",
        "feature_id",
        "feature_group",
        "selection_error",
        "test_error",
        "selection_balanced_accuracy",
        "test_balanced_accuracy",
    ]
    records = selected[[column for column in columns if column in selected]].to_dict("records")
    if not comparisons.empty:
        test = comparisons.loc[comparisons["split"].astype(str).str.startswith("test")]
        records.extend(test.nlargest(min(8, len(test)), "activation_advantage").to_dict("records"))
    return json.loads(json.dumps(records, default=_json_default))


def _normalize_motion_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize_spec(spec)
    normalized.setdefault(
        "movement",
        {
            "position": [0.01, 0.10],
            "rotation": [1.0, 15.0],
        },
    )
    movement = normalized["movement"]
    normalized["movement"] = {
        "position": [float(value) for value in movement.get("position", [0.01, 0.10])],
        "rotation": [float(value) for value in movement.get("rotation", [1.0, 15.0])],
    }
    return normalized


def _task_keys(rows: pd.DataFrame) -> pd.Series:
    benchmark = rows.get("benchmark", pd.Series("", index=rows.index)).astype(str)
    task_name = rows.get("task_name", rows.get("task_id", pd.Series("", index=rows.index))).astype(
        str
    )
    return benchmark + ":" + task_name


def _safe_binary_metric(function: Any, labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(function(labels, scores))


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


__all__ = ["MOTION_STUDY_SCHEMA_VERSION", "MotionStudyResult", "run_motion_probe_study"]
