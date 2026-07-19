"""Joint scene-object identity and location probe studies.

One model activation is decoded into one scene-sized output.  The output has a
fixed slot for every exact object identity observed in the dataset, so repeated
instances such as ``akita_black_bowl_1`` and ``akita_black_bowl_2`` remain
distinct.  Missing XYZ labels are handled by independent masked output heads;
all heads share the same input and are evaluated together as one scene map.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from vla_lens.artifacts import LensArtifact, make_artifact_id
from vla_lens.probes.geometry_study import (
    _activation_query,
    _apply_split_contract,
    _geometry_metadata_rows,
    _limit_rows_by_episode,
    _limited_episode_ids,
    _required_split_values,
    _source_required_split_values,
    _validate_episode_limit,
)
from vla_lens.traces import TraceDataset

SCENE_MAP_STUDY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SceneMapTargets:
    """Scene-sized labels aligned to one feature row per policy call."""

    vocabulary: tuple[str, ...]
    presence: np.ndarray
    visibility: np.ndarray
    position: np.ndarray
    initial_position: np.ndarray
    previous_position: np.ndarray
    role_manipulated: np.ndarray
    role_distractor: np.ndarray


@dataclass(frozen=True, slots=True)
class SceneMapStudyResult:
    """Saved study and its human- and machine-readable result tables."""

    artifact: LensArtifact | None
    candidates: pd.DataFrame
    selections: pd.DataFrame
    comparisons: pd.DataFrame
    predictions: pd.DataFrame
    vocabulary: pd.DataFrame
    examples: pd.DataFrame
    timings: Mapping[str, float]


@dataclass(slots=True)
class _Selected:
    record: dict[str, Any]
    rows: pd.DataFrame
    truth: np.ndarray
    prediction: np.ndarray
    masks: dict[str, np.ndarray]
    supported: np.ndarray
    threshold: float | None = None
    design: np.ndarray | None = None
    scene_targets: SceneMapTargets | None = None


def run_scene_map_probe_study(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    *,
    save: bool = True,
) -> SceneMapStudyResult:
    """Decode all object identities and locations from each saved activation."""

    normalized = _normalize_spec(spec)
    started = time.perf_counter()
    timings: dict[str, float] = {}
    candidate_records: list[dict[str, Any]] = []
    selected_models: list[_Selected] = []
    prediction_frames: list[pd.DataFrame] = []
    comparison_records: list[dict[str, Any]] = []
    vocabulary_frame: pd.DataFrame | None = None
    source_trace_ids: set[str] = set()
    required_split_values = _required_split_values(normalized["split"])
    _validate_episode_limit(normalized.get("limit_episodes"), required_split_values)
    limited_ids = _limited_episode_ids(
        dataset,
        normalized.get("limit_episodes"),
        required_split_values=_source_required_split_values(normalized["split"]),
    )

    for feature_spec in normalized["features"]:
        feature_id = str(feature_spec["id"])
        step_started = time.perf_counter()
        query = _activation_query(feature_spec)
        if limited_ids is not None:
            query = replace(
                query,
                episodes={**dict(query.episodes), "trace_id": limited_ids},
            )
        matrix = dataset.select_model_sites(query).materialize(cache=True)
        timings[f"feature:{feature_id}:materialize_seconds"] = time.perf_counter() - step_started

        step_started = time.perf_counter()
        rows = _geometry_metadata_rows(dataset, matrix.rows, cache=True)
        rows = _apply_split_contract(rows, normalized["split"])
        rows, X = _limit_rows_by_episode(
            rows,
            matrix.X,
            normalized.get("limit_episodes"),
            split_column=str(normalized["split"]["column"]),
            required_split_values=required_split_values,
        )
        rows = rows.reset_index(drop=True)
        X = np.asarray(X, dtype=np.float32)
        targets, target_vocabulary = scene_map_target_table(dataset, rows, cache=True)
        source_trace_ids.update(rows["trace_id"].astype(str).unique())
        if vocabulary_frame is None:
            vocabulary_frame = target_vocabulary
        elif tuple(vocabulary_frame["object_name"]) != targets.vocabulary:
            raise ValueError("Feature families produced different scene-map vocabularies")
        timings[f"feature:{feature_id}:targets_seconds"] = time.perf_counter() - step_started

        step_started = time.perf_counter()
        candidates, selections, comparisons, predictions = _fit_feature(
            X,
            rows,
            targets,
            feature_id=feature_id,
            sweep_columns=[str(value) for value in feature_spec.get("sweep", ["layer"])],
            split=normalized["split"],
            pca_dims=[int(value) for value in normalized["probe"]["pca_dims"]],
            ridge_alphas=[float(value) for value in normalized["probe"]["ridge_alphas"]],
            min_train_episodes=int(normalized["probe"]["min_train_episodes"]),
            context_columns=[str(value) for value in normalized["context_columns"]],
            projection_cache_root=(
                dataset.cache_dir() / "scene_map_projections" / matrix.cache_key
            ),
        )
        candidate_records.extend(candidates)
        selected_models.extend(selections)
        comparison_records.extend(comparisons)
        prediction_frames.extend(predictions)
        timings[f"feature:{feature_id}:fit_seconds"] = time.perf_counter() - step_started

    candidate_frame = pd.DataFrame.from_records(candidate_records)
    selection_frame = pd.DataFrame.from_records([item.record for item in selected_models])
    comparison_frame = pd.DataFrame.from_records(comparison_records)
    prediction_frame = (
        pd.concat(
            [frame.dropna(axis=1, how="all") for frame in prediction_frames],
            ignore_index=True,
        )
        if prediction_frames
        else pd.DataFrame()
    )
    vocabulary_frame = vocabulary_frame if vocabulary_frame is not None else pd.DataFrame()
    examples = _example_scenes(prediction_frame, vocabulary_frame)
    timings["total_seconds"] = time.perf_counter() - started
    artifact = (
        _save_scene_map_study(
            dataset,
            normalized,
            candidate_frame,
            selection_frame,
            comparison_frame,
            prediction_frame,
            vocabulary_frame,
            examples,
            timings,
            source_trace_ids,
        )
        if save
        else None
    )
    return SceneMapStudyResult(
        artifact=artifact,
        candidates=candidate_frame,
        selections=selection_frame,
        comparisons=comparison_frame,
        predictions=prediction_frame,
        vocabulary=vocabulary_frame,
        examples=examples,
        timings=timings,
    )


def scene_map_target_table(
    dataset: TraceDataset,
    rows: pd.DataFrame,
    *,
    cache: bool = True,
) -> tuple[SceneMapTargets, pd.DataFrame]:
    """Build scene-sized identity, visibility, and XYZ labels for feature rows."""

    roles = _object_role_table(dataset)
    if roles.empty:
        raise ValueError("Scene-map study requires a pi05_object_flow object_roles table")
    roles = roles.loc[roles["trace_id"].astype(str).isin(rows["trace_id"].astype(str))].copy()
    vocabulary = tuple(sorted(str(value) for value in roles["object_name"].dropna().unique()))
    if not vocabulary:
        raise ValueError("No exact object identities were found")
    keys = rows[["trace_id", "timestep"]].copy()
    keys["trace_id"] = keys["trace_id"].astype(str)
    keys["timestep"] = pd.to_numeric(keys["timestep"], errors="coerce").fillna(0).astype(int)
    key = _target_cache_key(dataset, keys, roles, vocabulary)
    cache_path = dataset.cache_dir() / "scene_map_targets" / key / "objects.parquet"
    if cache and cache_path.exists():
        long = pd.read_parquet(cache_path)
    else:
        long = _build_scene_object_rows(dataset, keys, roles)
        if cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            long.to_parquet(cache_path, index=False)

    row_key = pd.MultiIndex.from_frame(keys[["trace_id", "timestep"]])
    row_lookup: dict[tuple[str, int], list[int]] = {}
    for index, value in enumerate(row_key):
        row_lookup.setdefault((str(value[0]), int(value[1])), []).append(index)
    object_lookup = {name: index for index, name in enumerate(vocabulary)}
    n_rows, n_objects = len(rows), len(vocabulary)
    presence = np.zeros((n_rows, n_objects), dtype=np.float64)
    visibility = np.zeros((n_rows, n_objects), dtype=np.float64)
    position = np.full((n_rows, n_objects, 3), np.nan, dtype=np.float64)
    initial = np.full_like(position, np.nan)
    previous = np.full_like(position, np.nan)
    manipulated = np.zeros((n_rows, n_objects), dtype=bool)
    distractor = np.zeros((n_rows, n_objects), dtype=bool)
    for record in long.to_dict("records"):
        row_indices = row_lookup.get((str(record["trace_id"]), int(record["timestep"])), [])
        object_index = object_lookup.get(str(record["object_name"]))
        if not row_indices or object_index is None:
            continue
        for row_index in row_indices:
            presence[row_index, object_index] = 1.0
            visibility[row_index, object_index] = float(bool(record["visible_any"]))
            position[row_index, object_index] = np.asarray(record["position"], dtype=np.float64)
            initial[row_index, object_index] = np.asarray(
                record["initial_position"], dtype=np.float64
            )
            previous[row_index, object_index] = np.asarray(
                record["previous_position"], dtype=np.float64
            )
            manipulated[row_index, object_index] = bool(record["role_manipulated"])
            distractor[row_index, object_index] = bool(record["role_distractor"])

    split_column = "split" if "split" in rows else None
    vocabulary_records: list[dict[str, Any]] = []
    for object_index, object_name in enumerate(vocabulary):
        record: dict[str, Any] = {"object_index": object_index, "object_name": object_name}
        object_roles = roles.loc[roles["object_name"].astype(str) == object_name]
        if not object_roles.empty:
            record["object_base_name"] = object_roles.iloc[0].get("object_base_name")
        if split_column:
            episode_split = rows[["trace_id", split_column]].drop_duplicates()
            for split_value, group in episode_split.groupby(split_column, dropna=False):
                trace_ids = set(group["trace_id"].astype(str))
                record[f"episodes_{split_value}"] = int(
                    object_roles["trace_id"].astype(str).isin(trace_ids).sum()
                )
        vocabulary_records.append(record)
    return (
        SceneMapTargets(
            vocabulary=vocabulary,
            presence=presence,
            visibility=visibility,
            position=position,
            initial_position=initial,
            previous_position=previous,
            role_manipulated=manipulated,
            role_distractor=distractor,
        ),
        pd.DataFrame.from_records(vocabulary_records),
    )


def _build_scene_object_rows(
    dataset: TraceDataset,
    keys: pd.DataFrame,
    roles: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    roles_by_trace = {str(key): value for key, value in roles.groupby("trace_id", sort=False)}
    for trace_id, trace_rows in keys.groupby("trace_id", sort=False):
        trace_roles = roles_by_trace.get(str(trace_id))
        if trace_roles is None:
            continue
        bundle = dataset.bundle(str(trace_id))
        positions = np.asarray(bundle.array("scene_object_pos", mmap=True))
        timesteps = sorted(int(value) for value in trace_rows["timestep"].unique())
        previous = {
            timestep: timesteps[max(0, index - 1)] for index, timestep in enumerate(timesteps)
        }
        visible, visible_names = _visibility_array(bundle)
        visible_lookup = {name: index for index, name in enumerate(visible_names)}
        for timestep in timesteps:
            step = min(max(0, timestep), positions.shape[0] - 1)
            previous_step = min(max(0, previous[timestep]), positions.shape[0] - 1)
            for role in trace_roles.to_dict("records"):
                object_index = int(role["object_index"])
                object_name = str(role["object_name"])
                visible_index = visible_lookup.get(object_name)
                visible_any = bool(
                    visible is not None
                    and visible_index is not None
                    and step < visible.shape[0]
                    and visible_index < visible.shape[-1]
                    and np.asarray(visible[step, ..., visible_index]).astype(bool).any()
                )
                records.append(
                    {
                        "trace_id": str(trace_id),
                        "timestep": int(timestep),
                        "object_name": object_name,
                        "object_base_name": role.get("object_base_name"),
                        "position": np.asarray(
                            positions[step, object_index], dtype=np.float64
                        ).tolist(),
                        "initial_position": np.asarray(
                            positions[0, object_index], dtype=np.float64
                        ).tolist(),
                        "previous_position": np.asarray(
                            positions[previous_step, object_index], dtype=np.float64
                        ).tolist(),
                        "visible_any": visible_any,
                        "role_manipulated": bool(role.get("role_manipulated", False)),
                        "role_distractor": bool(role.get("role_distractor", False)),
                    }
                )
    return pd.DataFrame.from_records(records)


def _visibility_array(bundle: Any) -> tuple[np.ndarray | None, list[str]]:
    try:
        value = np.asarray(bundle.array("camera_object_visible", mmap=True))
    except KeyError:
        return None, []
    match = bundle.array_index.loc[
        bundle.array_index["name"].astype(str) == "camera_object_visible"
    ]
    if match.empty:
        return value, []
    metadata = match.iloc[0].get("metadata")
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    names = list((metadata or {}).get("object_names") or [])
    return value, [str(name) for name in names]


def _fit_feature(
    X: np.ndarray,
    rows: pd.DataFrame,
    targets: SceneMapTargets,
    *,
    feature_id: str,
    sweep_columns: Sequence[str],
    split: Mapping[str, Any],
    pca_dims: Sequence[int],
    ridge_alphas: Sequence[float],
    min_train_episodes: int,
    context_columns: Sequence[str],
    projection_cache_root: Path | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[_Selected],
    list[dict[str, Any]],
    list[pd.DataFrame],
]:
    split_column = str(split["column"])
    split_values = {
        "train": str(split["train_value"]),
        "selection": str(split["selection_value"]),
        "test": str(split["test_value"]),
    }
    group_columns = [column for column in sweep_columns if column in rows]
    grouped: list[tuple[Any, np.ndarray]] = [((), np.arange(len(rows)))]
    if group_columns:
        grouped = [
            (key, np.asarray(value, dtype=np.int64))
            for key, value in rows.groupby(group_columns, dropna=False, sort=True).indices.items()
        ]
    candidates: list[dict[str, Any]] = []
    best: dict[str, _Selected] = {}
    for group_value, indices in grouped:
        group_rows = rows.iloc[indices].reset_index(drop=True)
        group_targets = _slice_scene_targets(targets, indices)
        group_X = np.asarray(X[indices], dtype=np.float64)
        masks = {
            name: group_rows[split_column].astype(str).to_numpy() == value
            for name, value in split_values.items()
        }
        if not all(mask.any() for mask in masks.values()):
            continue
        supported = _supported_objects(
            group_targets.presence,
            group_rows,
            masks["train"],
            min_train_episodes,
        )
        group_label = _group_label(group_columns, group_value)
        projected = _project(
            group_X,
            masks["train"],
            pca_dims,
            cache_path=(
                projection_cache_root
                / f"{_projection_group_key(group_rows, split_column, group_label, pca_dims)}.npz"
                if projection_cache_root is not None
                else None
            ),
        )
        for pca_dim, design in projected.items():
            for alpha in ridge_alphas:
                presence_prediction = _fit_joint_ridge(
                    design, group_targets.presence, masks["train"], alpha
                )
                presence_metrics = _identity_metrics(
                    group_targets.presence,
                    presence_prediction,
                    group_rows,
                    masks,
                    supported,
                )
                presence_record = _candidate_record(
                    feature_id,
                    group_label,
                    "scene_identity",
                    pca_dim,
                    alpha,
                    presence_metrics,
                )
                candidates.append(presence_record)
                _consider_best(
                    best,
                    "scene_identity",
                    _Selected(
                        record=presence_record,
                        rows=group_rows,
                        truth=group_targets.presence,
                        prediction=presence_prediction,
                        masks=masks,
                        supported=supported,
                        threshold=float(presence_metrics["selection_threshold"]),
                        design=design,
                        scene_targets=group_targets,
                    ),
                )

                visibility_supported = supported & (
                    group_targets.visibility[masks["train"]].sum(axis=0) > 0
                )
                visibility_prediction = _fit_joint_ridge(
                    design, group_targets.visibility, masks["train"], alpha
                )
                visibility_metrics = _identity_metrics(
                    group_targets.visibility,
                    visibility_prediction,
                    group_rows,
                    masks,
                    visibility_supported,
                )
                visibility_record = _candidate_record(
                    feature_id,
                    group_label,
                    "visible_identity",
                    pca_dim,
                    alpha,
                    visibility_metrics,
                )
                candidates.append(visibility_record)
                _consider_best(
                    best,
                    "visible_identity",
                    _Selected(
                        record=visibility_record,
                        rows=group_rows,
                        truth=group_targets.visibility,
                        prediction=visibility_prediction,
                        masks=masks,
                        supported=visibility_supported,
                        threshold=float(visibility_metrics["selection_threshold"]),
                        design=design,
                        scene_targets=group_targets,
                    ),
                )

                position_prediction, position_supported = _fit_masked_positions(
                    design,
                    group_targets.position,
                    group_rows,
                    masks["train"],
                    alpha,
                    min_train_episodes,
                )
                position_metrics = _position_metrics(
                    group_targets,
                    position_prediction,
                    group_rows,
                    masks,
                    position_supported,
                )
                position_record = _candidate_record(
                    feature_id,
                    group_label,
                    "object_position",
                    pca_dim,
                    alpha,
                    position_metrics,
                )
                candidates.append(position_record)
                _consider_best(
                    best,
                    "object_position",
                    _Selected(
                        record=position_record,
                        rows=group_rows,
                        truth=group_targets.position,
                        prediction=position_prediction,
                        masks=masks,
                        supported=position_supported,
                        design=design,
                        scene_targets=group_targets,
                    ),
                )

    selected = list(best.values())
    comparisons: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    for neural in selected:
        target = str(neural.record["target"])
        if neural.scene_targets is None:
            raise ValueError("Selected scene-map model is missing aligned targets")
        aligned_targets = neural.scene_targets
        neural.record["model_name"] = "activation"
        comparisons.append(_comparison_record(neural.record, "activation"))
        predictions.append(_prediction_frame(neural, feature_id, "activation"))

        context = _context_design(neural.rows, neural.masks["train"], context_columns)
        context_selected = _select_comparison_model(
            context,
            neural,
            targets=aligned_targets,
            target=target,
            alphas=ridge_alphas,
            model_name="prompt_and_scene_context",
        )
        comparisons.append(_comparison_record(context_selected.record, "prompt_and_scene_context"))
        predictions.append(
            _prediction_frame(context_selected, feature_id, "prompt_and_scene_context")
        )

        if neural.design is None:
            raise ValueError("Selected activation model is missing its feature design")
        combined = np.concatenate([neural.design, context], axis=1)
        combined_selected = _select_comparison_model(
            combined,
            neural,
            targets=aligned_targets,
            target=target,
            alphas=ridge_alphas,
            model_name="activation_and_context",
        )
        comparisons.append(_comparison_record(combined_selected.record, "activation_and_context"))
        predictions.append(
            _prediction_frame(combined_selected, feature_id, "activation_and_context")
        )

        if target == "scene_identity":
            frequency = np.repeat(
                neural.truth[neural.masks["train"]].mean(axis=0, keepdims=True),
                len(neural.rows),
                axis=0,
            )
            baseline = _selected_from_prediction(
                neural,
                frequency,
                target,
                "training_frequency",
                aligned_targets,
            )
            comparisons.append(_comparison_record(baseline.record, "training_frequency"))
            predictions.append(_prediction_frame(baseline, feature_id, "training_frequency"))
        elif target == "visible_identity":
            frequency = np.repeat(
                neural.truth[neural.masks["train"]].mean(axis=0, keepdims=True),
                len(neural.rows),
                axis=0,
            )
            baseline = _selected_from_prediction(
                neural,
                frequency,
                target,
                "training_visibility_frequency",
                aligned_targets,
            )
            comparisons.append(_comparison_record(baseline.record, "training_visibility_frequency"))
            predictions.append(
                _prediction_frame(baseline, feature_id, "training_visibility_frequency")
            )
        else:
            for name, value in [
                ("initial_position", aligned_targets.initial_position),
                ("previous_position", aligned_targets.previous_position),
            ]:
                baseline = _selected_from_prediction(neural, value, target, name, aligned_targets)
                comparisons.append(_comparison_record(baseline.record, name))
                predictions.append(_prediction_frame(baseline, feature_id, name))
    return candidates, selected, comparisons, predictions


def _project(
    X: np.ndarray,
    train_mask: np.ndarray,
    pca_dims: Sequence[int],
    *,
    cache_path: Path | None = None,
) -> dict[int, np.ndarray]:
    if cache_path is not None and cache_path.exists():
        with np.load(cache_path) as cached:
            return {
                int(name.removeprefix("dim_")): np.asarray(cached[name])
                for name in cached.files
                if name.startswith("dim_")
            }
    scaler = StandardScaler()
    train = scaler.fit_transform(X[train_mask])
    all_scaled = scaler.transform(X)
    max_dim = min(max(pca_dims), train.shape[0] - 1, train.shape[1])
    projector = PCA(
        n_components=max_dim,
        svd_solver="randomized",
        iterated_power=2,
        random_state=0,
    )
    projector.fit(train)
    values = projector.transform(all_scaled)
    result = {
        min(int(dim), max_dim): values[:, : min(int(dim), max_dim)] for dim in sorted(set(pca_dims))
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            **{f"dim_{dim}": value for dim, value in result.items()},
        )
        os.replace(temporary, cache_path)
    return result


def _fit_joint_ridge(
    X: np.ndarray,
    truth: np.ndarray,
    train_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    model = Ridge(alpha=float(alpha), solver="lsqr")
    model.fit(X[train_mask], truth[train_mask])
    return np.asarray(model.predict(X), dtype=np.float64)


def _fit_masked_positions(
    X: np.ndarray,
    truth: np.ndarray,
    rows: pd.DataFrame,
    train_mask: np.ndarray,
    alpha: float,
    min_train_episodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.full_like(truth, np.nan, dtype=np.float64)
    supported = np.zeros(truth.shape[1], dtype=bool)
    for object_index in range(truth.shape[1]):
        available = train_mask & np.isfinite(truth[:, object_index]).all(axis=1)
        episode_count = rows.loc[available, "trace_id"].astype(str).nunique()
        if episode_count < min_train_episodes:
            continue
        model = Ridge(alpha=float(alpha), solver="lsqr")
        model.fit(X[available], truth[available, object_index])
        prediction[:, object_index] = model.predict(X)
        supported[object_index] = True
    return prediction, supported


def _identity_metrics(
    truth: np.ndarray,
    scores: np.ndarray,
    rows: pd.DataFrame,
    masks: Mapping[str, np.ndarray],
    supported: np.ndarray,
) -> dict[str, Any]:
    threshold = _best_threshold(truth[masks["selection"]], scores[masks["selection"]], supported)
    result: dict[str, Any] = {"selection_threshold": threshold}
    for split_name in ["selection", "test"]:
        mask = masks[split_name]
        split_truth = truth[mask]
        split_scores = scores[mask]
        predicted = split_scores >= threshold
        if supported.any():
            y = split_truth[:, supported].astype(bool)
            p = predicted[:, supported]
            score = split_scores[:, supported]
            tp = int(np.logical_and(y, p).sum())
            fp = int(np.logical_and(~y, p).sum())
            fn = int(np.logical_and(y, ~p).sum())
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            intersections = np.logical_and(y, p).sum(axis=1)
            unions = np.logical_or(y, p).sum(axis=1)
            jaccard = intersections / np.maximum(1, unions)
            exact = np.all(y == p, axis=1)
            macro_ap = _macro_average_precision(y, score)
            predicted_count_error = np.abs(p.sum(axis=1) - y.sum(axis=1))
        else:
            precision = recall = f1 = macro_ap = float("nan")
            jaccard = exact = predicted_count_error = np.full(mask.sum(), np.nan)
        full_truth = split_truth.astype(bool)
        full_predicted = np.zeros_like(full_truth, dtype=bool)
        full_predicted[:, supported] = predicted[:, supported]
        full_intersections = np.logical_and(full_truth, full_predicted).sum(axis=1)
        full_unions = np.logical_or(full_truth, full_predicted).sum(axis=1)
        full_jaccard = full_intersections / np.maximum(1, full_unions)
        full_exact = np.all(full_truth == full_predicted, axis=1)
        episode = rows.loc[mask, "trace_id"].astype(str).to_numpy()
        result.update(
            {
                f"{split_name}_precision": float(precision),
                f"{split_name}_recall": float(recall),
                f"{split_name}_f1": float(f1),
                f"{split_name}_macro_average_precision": float(macro_ap),
                f"{split_name}_scene_jaccard": _episode_weighted_mean(jaccard, episode),
                f"{split_name}_exact_scene_rate": _episode_weighted_mean(
                    exact.astype(float), episode
                ),
                f"{split_name}_full_scene_jaccard": _episode_weighted_mean(
                    full_jaccard.astype(float), episode
                ),
                f"{split_name}_full_exact_scene_rate": _episode_weighted_mean(
                    full_exact.astype(float), episode
                ),
                f"{split_name}_count_error": _episode_weighted_mean(
                    predicted_count_error.astype(float), episode
                ),
                f"{split_name}_rows": int(mask.sum()),
                f"{split_name}_supported_objects": int(supported.sum()),
                f"{split_name}_unseen_positive_count": int(split_truth[:, ~supported].sum()),
            }
        )
    return result


def _best_threshold(truth: np.ndarray, scores: np.ndarray, supported: np.ndarray) -> float:
    if not supported.any():
        return 0.5
    y = truth[:, supported].astype(bool)
    values = scores[:, supported]
    candidates = np.unique(
        np.concatenate(
            [
                np.linspace(-0.1, 1.1, 49),
                np.quantile(values, np.linspace(0.05, 0.95, 19)),
            ]
        )
    )
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        predicted = values >= float(threshold)
        tp = np.logical_and(y, predicted).sum()
        fp = np.logical_and(~y, predicted).sum()
        fn = np.logical_and(y, ~predicted).sum()
        score = 2 * tp / max(1, 2 * tp + fp + fn)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold


def _macro_average_precision(truth: np.ndarray, scores: np.ndarray) -> float:
    values: list[float] = []
    for index in range(truth.shape[1]):
        column = truth[:, index].astype(int)
        if np.unique(column).size < 2:
            continue
        values.append(float(average_precision_score(column, scores[:, index])))
    return float(np.mean(values)) if values else float("nan")


def _position_metrics(
    targets: SceneMapTargets,
    predicted: np.ndarray,
    rows: pd.DataFrame,
    masks: Mapping[str, np.ndarray],
    supported: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    moved = np.linalg.norm(targets.position - targets.initial_position, axis=2)
    for split_name in ["selection", "test"]:
        row_mask = masks[split_name]
        finite = np.isfinite(targets.position).all(axis=2)
        available = row_mask[:, None] & finite & supported[None, :]
        errors = np.linalg.norm(predicted - targets.position, axis=2)
        coordinate_errors = np.abs(predicted - targets.position)
        result[f"{split_name}_error_m"] = _scene_position_mean(errors, available, rows)
        for axis, name in enumerate("xyz"):
            result[f"{split_name}_{name}_mae_m"] = _scene_position_mean(
                coordinate_errors[:, :, axis], available, rows
            )
        for slice_name, slice_mask in [
            ("visible", targets.visibility.astype(bool)),
            ("not_visible", ~targets.visibility.astype(bool)),
            ("moved_1cm", moved > 0.01),
            ("moved_10cm", moved > 0.10),
            ("manipulated", targets.role_manipulated),
            ("distractor", targets.role_distractor),
        ]:
            selected = available & slice_mask
            result[f"{split_name}_{slice_name}_error_m"] = _scene_position_mean(
                errors, selected, rows
            )
            result[f"{split_name}_{slice_name}_count"] = int(selected.sum())
        result[f"{split_name}_position_count"] = int(available.sum())
        result[f"{split_name}_supported_objects"] = int(supported.sum())
        result[f"{split_name}_unseen_position_count"] = int(
            (row_mask[:, None] & finite & ~supported[None, :]).sum()
        )
    return result


def _scene_position_mean(
    values: np.ndarray,
    available: np.ndarray,
    rows: pd.DataFrame,
) -> float:
    selected_rows = np.flatnonzero(available.any(axis=1))
    if not len(selected_rows):
        return float("nan")
    scene_values = np.array(
        [float(np.mean(values[index, available[index]])) for index in selected_rows]
    )
    episodes = rows.iloc[selected_rows]["trace_id"].astype(str).to_numpy()
    return _episode_weighted_mean(scene_values, episodes)


def _episode_weighted_mean(values: np.ndarray, episodes: np.ndarray) -> float:
    frame = pd.DataFrame({"episode": episodes, "value": np.asarray(values, dtype=float)})
    frame = frame.loc[np.isfinite(frame["value"])]
    if frame.empty:
        return float("nan")
    return float(frame.groupby("episode", sort=False)["value"].mean().mean())


def _candidate_record(
    feature_id: str,
    feature_group: str,
    target: str,
    pca_dim: int,
    alpha: float,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "feature_id": feature_id,
        "feature_group": feature_group,
        "target": target,
        "pca_dim": int(pca_dim),
        "ridge_alpha": float(alpha),
        **dict(metrics),
    }


def _consider_best(best: dict[str, _Selected], target: str, candidate: _Selected) -> None:
    current = best.get(target)
    if current is None or _selection_score(candidate.record, target) > _selection_score(
        current.record, target
    ):
        best[target] = candidate


def _selection_score(record: Mapping[str, Any], target: str) -> float:
    if target == "object_position":
        value = float(record.get("selection_error_m", float("inf")))
        return -value if np.isfinite(value) else -float("inf")
    value = float(record.get("selection_scene_jaccard", -float("inf")))
    return value if np.isfinite(value) else -float("inf")


def _context_design(
    rows: pd.DataFrame,
    train_mask: np.ndarray,
    categorical_columns: Sequence[str],
) -> np.ndarray:
    prompt = rows.get("prompt", pd.Series("", index=rows.index)).fillna("").astype(str)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=256)
    train_text = vectorizer.fit_transform(prompt[train_mask])
    all_text = vectorizer.transform(prompt)
    if train_text.shape[1] > 1:
        dim = min(32, train_text.shape[0] - 1, train_text.shape[1] - 1)
        if dim > 0:
            reducer = TruncatedSVD(n_components=dim, random_state=0)
            reducer.fit(train_text)
            text_design = reducer.transform(all_text)
        else:
            text_design = all_text.toarray()
    else:
        text_design = all_text.toarray()
    available = [column for column in categorical_columns if column in rows]
    parts = [np.asarray(text_design, dtype=np.float64)]
    if available:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
        encoder.fit(rows.loc[train_mask, available].fillna("").astype(str))
        parts.append(encoder.transform(rows[available].fillna("").astype(str)))
    timing = (
        pd.to_numeric(
            rows.get("policy_call_index", pd.Series(0, index=rows.index)), errors="coerce"
        )
        .fillna(0)
        .to_numpy(dtype=np.float64)[:, None]
    )
    scaler = StandardScaler()
    scaler.fit(timing[train_mask])
    parts.append(scaler.transform(timing))
    return np.concatenate(parts, axis=1)


def _select_comparison_model(
    design: np.ndarray,
    reference: _Selected,
    *,
    targets: SceneMapTargets,
    target: str,
    alphas: Sequence[float],
    model_name: str,
) -> _Selected:
    best: _Selected | None = None
    for alpha in alphas:
        if target == "object_position":
            prediction, supported = _fit_masked_positions(
                design,
                targets.position,
                reference.rows,
                reference.masks["train"],
                float(alpha),
                min_train_episodes=1,
            )
            supported &= reference.supported
            metrics = _position_metrics(
                targets, prediction, reference.rows, reference.masks, supported
            )
        else:
            truth = targets.presence if target == "scene_identity" else targets.visibility
            prediction = _fit_joint_ridge(design, truth, reference.masks["train"], float(alpha))
            supported = reference.supported
            metrics = _identity_metrics(
                truth, prediction, reference.rows, reference.masks, supported
            )
        record = {
            "feature_id": reference.record["feature_id"],
            "feature_group": reference.record["feature_group"],
            "target": target,
            "pca_dim": reference.record["pca_dim"],
            "ridge_alpha": float(alpha),
            "model_name": model_name,
            **metrics,
        }
        selected = _Selected(
            record=record,
            rows=reference.rows,
            truth=targets.position
            if target == "object_position"
            else (targets.presence if target == "scene_identity" else targets.visibility),
            prediction=prediction,
            masks=reference.masks,
            supported=supported,
            threshold=(
                float(metrics["selection_threshold"]) if target != "object_position" else None
            ),
            design=design,
            scene_targets=targets,
        )
        if best is None or _selection_score(record, target) > _selection_score(best.record, target):
            best = selected
    if best is None:
        raise ValueError(f"No comparison model could be fit for {target}")
    return best


def _selected_from_prediction(
    reference: _Selected,
    prediction: np.ndarray,
    target: str,
    model_name: str,
    targets: SceneMapTargets,
) -> _Selected:
    if target == "object_position":
        metrics = _position_metrics(
            targets, prediction, reference.rows, reference.masks, reference.supported
        )
        threshold = None
        truth = targets.position
    else:
        truth = targets.presence if target == "scene_identity" else targets.visibility
        metrics = _identity_metrics(
            truth, prediction, reference.rows, reference.masks, reference.supported
        )
        threshold = float(metrics["selection_threshold"])
    record = {
        "feature_id": reference.record["feature_id"],
        "feature_group": reference.record["feature_group"],
        "target": target,
        "pca_dim": None,
        "ridge_alpha": None,
        "model_name": model_name,
        **metrics,
    }
    return _Selected(
        record=record,
        rows=reference.rows,
        truth=truth,
        prediction=np.asarray(prediction, dtype=np.float64),
        masks=reference.masks,
        supported=reference.supported,
        threshold=threshold,
        scene_targets=targets,
    )


def _comparison_record(record: Mapping[str, Any], model_name: str) -> dict[str, Any]:
    return {**dict(record), "model_name": model_name}


def _prediction_frame(
    selected: _Selected,
    feature_id: str,
    model_name: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for split_name in ["selection", "test"]:
        indices = np.flatnonzero(selected.masks[split_name])
        for index in indices:
            source = selected.rows.iloc[index]
            truth = selected.truth[index]
            prediction = selected.prediction[index]
            records.append(
                {
                    "feature_id": feature_id,
                    "feature_group": selected.record["feature_group"],
                    "target": selected.record["target"],
                    "model_name": model_name,
                    "split": split_name,
                    "trace_id": str(source["trace_id"]),
                    "episode_id": source.get("episode_id"),
                    "benchmark": source.get("benchmark"),
                    "task_id": source.get("task_id"),
                    "prompt": source.get("prompt"),
                    "timestep": int(source["timestep"]),
                    "policy_call_index": _optional_int(source.get("policy_call_index")),
                    "threshold": selected.threshold,
                    "supported": selected.supported.astype(np.uint8).tolist(),
                    "truth": np.asarray(truth, dtype=np.float32).reshape(-1).tolist(),
                    "prediction": np.asarray(prediction, dtype=np.float32).reshape(-1).tolist(),
                }
            )
    return pd.DataFrame.from_records(records)


def _example_scenes(
    predictions: pd.DataFrame,
    vocabulary: pd.DataFrame,
) -> pd.DataFrame:
    if predictions.empty or vocabulary.empty:
        return pd.DataFrame()
    names = vocabulary.sort_values("object_index")["object_name"].astype(str).tolist()
    test = predictions.loc[predictions["split"].astype(str) == "test"].copy()
    records: list[dict[str, Any]] = []
    for (feature_id, target, model_name), group in test.groupby(
        ["feature_id", "target", "model_name"], sort=False
    ):
        scored: list[tuple[float, pd.Series]] = []
        for _, row in group.iterrows():
            truth = np.asarray(row["truth"], dtype=float)
            prediction = np.asarray(row["prediction"], dtype=float)
            supported = np.asarray(row["supported"], dtype=bool)
            if target == "object_position":
                truth = truth.reshape(-1, 3)
                prediction = prediction.reshape(-1, 3)
                available = supported & np.isfinite(truth).all(axis=1)
                score = (
                    float(np.linalg.norm(prediction[available] - truth[available], axis=1).mean())
                    if available.any()
                    else float("nan")
                )
            else:
                y = truth[supported] >= 0.5
                p = prediction[supported] >= float(row.get("threshold") or 0.5)
                score = float(np.logical_xor(y, p).sum())
            if np.isfinite(score):
                scored.append((score, row))
        if not scored:
            continue
        scored.sort(key=lambda value: value[0])
        choices = [("best", scored[0]), ("worst", scored[-1])]
        for label, (score, row) in choices:
            truth = np.asarray(row["truth"], dtype=float)
            prediction = np.asarray(row["prediction"], dtype=float)
            if target == "object_position":
                truth = truth.reshape(-1, 3)
                prediction = prediction.reshape(-1, 3)
                details = [
                    {
                        "object_name": names[index],
                        "truth_xyz": truth[index].tolist(),
                        "predicted_xyz": prediction[index].tolist(),
                        "error_m": float(np.linalg.norm(prediction[index] - truth[index])),
                    }
                    for index in range(len(names))
                    if np.isfinite(truth[index]).all()
                ]
            else:
                threshold = float(row.get("threshold") or 0.5)
                details = [
                    {
                        "object_name": names[index],
                        "truth": bool(truth[index] >= 0.5),
                        "score": float(prediction[index]),
                        "predicted": bool(prediction[index] >= threshold),
                    }
                    for index in range(len(names))
                ]
            records.append(
                {
                    "feature_id": feature_id,
                    "target": target,
                    "model_name": model_name,
                    "example_kind": label,
                    "score": score,
                    "trace_id": row["trace_id"],
                    "timestep": int(row["timestep"]),
                    "prompt": row.get("prompt"),
                    "objects_json": json.dumps(details, sort_keys=True),
                }
            )
    return pd.DataFrame.from_records(records)


def _slice_scene_targets(
    targets: SceneMapTargets,
    indices: np.ndarray,
) -> SceneMapTargets:
    return SceneMapTargets(
        vocabulary=targets.vocabulary,
        presence=targets.presence[indices],
        visibility=targets.visibility[indices],
        position=targets.position[indices],
        initial_position=targets.initial_position[indices],
        previous_position=targets.previous_position[indices],
        role_manipulated=targets.role_manipulated[indices],
        role_distractor=targets.role_distractor[indices],
    )


def _supported_objects(
    presence: np.ndarray,
    rows: pd.DataFrame,
    train_mask: np.ndarray,
    min_train_episodes: int,
) -> np.ndarray:
    supported = np.zeros(presence.shape[1], dtype=bool)
    for object_index in range(presence.shape[1]):
        positive = train_mask & (presence[:, object_index] > 0.5)
        supported[object_index] = (
            rows.loc[positive, "trace_id"].astype(str).nunique() >= min_train_episodes
        )
    return supported


def _group_label(columns: Sequence[str], value: Any) -> str:
    if not columns:
        return "all"
    values = value if isinstance(value, tuple) else (value,)
    return ",".join(f"{column}={item}" for column, item in zip(columns, values, strict=False))


def _projection_group_key(
    rows: pd.DataFrame,
    split_column: str,
    group_label: str,
    pca_dims: Sequence[int],
) -> str:
    columns = [
        column
        for column in [
            "trace_id",
            "timestep",
            "policy_call_index",
            "layer",
            split_column,
        ]
        if column in rows
    ]
    row_hash = pd.util.hash_pandas_object(rows[columns], index=False).to_numpy(dtype=np.uint64)
    payload = {
        "schema": SCENE_MAP_STUDY_SCHEMA_VERSION,
        "group": group_label,
        "pca_dims": sorted(set(int(value) for value in pca_dims)),
        "rows": hashlib.sha256(row_hash.tobytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def _object_role_table(dataset: TraceDataset) -> pd.DataFrame:
    index = dataset.artifact_index
    if index.empty or "artifact_type" not in index:
        return pd.DataFrame()
    matches = index.loc[index["artifact_type"].astype(str) == "pi05_object_flow"]
    if matches.empty:
        return pd.DataFrame()
    sort_column = "created_utc" if "created_utc" in matches else "artifact_id"
    artifact_id = str(matches.sort_values(sort_column).iloc[-1]["artifact_id"])
    artifact = dataset.load_artifact(artifact_id)
    relative = artifact.method.get("outputs", {}).get("object_roles")
    if not relative:
        return pd.DataFrame()
    return pd.read_parquet(dataset.root / str(relative))


def _target_cache_key(
    dataset: TraceDataset,
    keys: pd.DataFrame,
    roles: pd.DataFrame,
    vocabulary: Sequence[str],
) -> str:
    row_hash = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
    role_columns = [
        column
        for column in ["trace_id", "object_index", "object_name", "role_manipulated"]
        if column in roles
    ]
    role_hash = pd.util.hash_pandas_object(
        roles[role_columns].sort_values(role_columns), index=False
    ).to_numpy(dtype=np.uint64)
    payload = {
        "schema": SCENE_MAP_STUDY_SCHEMA_VERSION,
        "dataset_root": str(dataset.root.resolve()),
        "rows": hashlib.sha256(row_hash.tobytes()).hexdigest(),
        "roles": hashlib.sha256(role_hash.tobytes()).hexdigest(),
        "vocabulary": list(vocabulary),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def _normalize_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(spec)
    normalized.setdefault("name", "PI0.5 joint scene map study")
    normalized.setdefault(
        "split",
        {
            "kind": "heldout_task",
            "column": "split",
            "train_value": "train",
            "selection_value": "val_heldout_task",
            "test_value": "test_heldout_task",
        },
    )
    normalized.setdefault(
        "probe",
        {
            "pca_dims": [64, 128],
            "ridge_alphas": [1.0, 10.0],
            "min_train_episodes": 5,
        },
    )
    normalized["probe"].setdefault("min_train_episodes", 5)
    normalized.setdefault("context_columns", ["benchmark", "scene_family", "task_phase"])
    features = normalized.get("features")
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)) or not features:
        raise ValueError("Scene-map study requires a non-empty features list")
    normalized["features"] = [dict(value) for value in features]
    for index, feature in enumerate(normalized["features"]):
        feature.setdefault("id", f"feature_{index}")
        feature.setdefault("sweep", ["layer"])
    return normalized


def _save_scene_map_study(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    candidates: pd.DataFrame,
    selections: pd.DataFrame,
    comparisons: pd.DataFrame,
    predictions: pd.DataFrame,
    vocabulary: pd.DataFrame,
    examples: pd.DataFrame,
    timings: Mapping[str, float],
    source_trace_ids: Sequence[str],
) -> LensArtifact:
    artifact_id = make_artifact_id(str(spec["name"]), "scene_map_probe_study")
    relative_dir = Path("artifacts") / artifact_id
    tables = {
        "candidates": candidates,
        "selections": selections,
        "comparisons": comparisons,
        "scene_predictions": predictions,
        "vocabulary": vocabulary,
        "examples": examples,
    }
    outputs = {name: str(relative_dir / f"{name}.parquet") for name in tables}
    artifact = LensArtifact(
        artifact_id=artifact_id,
        artifact_type="scene_map_probe_study",
        name=str(spec["name"]),
        group_id="scene_map_probe_studies",
        scope="dataset",
        selector={"features": spec["features"]},
        method={
            "workflow": "run_scene_map_probe_study",
            "schema_version": SCENE_MAP_STUDY_SCHEMA_VERSION,
            "research": {
                key: spec[key]
                for key in ["question", "hypothesis_family", "intended_claim"]
                if spec.get(key)
            },
            "split": spec["split"],
            "probe": spec["probe"],
            "context_columns": spec["context_columns"],
            "target_contract": {
                "scene_identity": "multi-label exact object instance presence",
                "visible_identity": "multi-label visibility in either captured camera",
                "object_position": "masked XYZ head per exact object instance",
                "position_metric": (
                    "mean Euclidean error per scene, then equal mean across episodes"
                ),
                "identity_metric": (
                    "scene Jaccard and micro F1 at a validation-selected threshold"
                ),
                "unknown_identity_policy": (
                    "retain held-out identities missing from training and count them separately"
                ),
            },
            "storage_contract": {
                "activation_arrays": "referenced from the source capture, never copied",
                "scene_predictions": ("compact list columns ordered by vocabulary.object_index"),
            },
            "timings_seconds": dict(timings),
            "outputs": outputs,
        },
        metrics={
            "candidate_count": int(len(candidates)),
            "selection_count": int(len(selections)),
            "comparison_count": int(len(comparisons)),
            "prediction_count": int(len(predictions)),
            "object_vocabulary_size": int(len(vocabulary)),
            "total_seconds": float(timings.get("total_seconds", 0.0)),
        },
        display={
            "kind": "scene_map_probe_study",
            "status": "exploratory",
            "comparisons": _json_records(comparisons),
        },
        tags=("probe", "scene-map", "object-identity", "object-location", "exploratory"),
        source_trace_ids=tuple(sorted(str(value) for value in source_trace_ids)),
    )
    saved = dataset.save_artifact(artifact)
    artifact_dir = dataset._dataset_artifact_root() / relative_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_parquet(artifact_dir / f"{name}.parquet", index=False)
    return saved


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.to_json(orient="records"))


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)
