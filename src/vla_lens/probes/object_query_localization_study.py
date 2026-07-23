"""Locate a named object with a shared query-conditioned visual-token head."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from vla_lens.artifacts import LensArtifact, make_artifact_id
from vla_lens.probes.image_location_study import _box_iou
from vla_lens.probes.object_study_common import (
    ContextEncoder,
    FittedClassifier,
    VisualObjectData,
    classifier_arrays,
    fit_classifier,
    fit_context_encoder,
    grouped_paired_interval,
    normalized_patch_centers,
    prepare_visual_object_data,
    row_identity,
    split_masks,
)
from vla_lens.traces import TraceDataset

OBJECT_QUERY_LOCALIZATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ObjectQueryLocalizationStudyResult:
    """Saved explicit object-query localization experiment."""

    artifact: LensArtifact | None
    candidates: pd.DataFrame
    selections: pd.DataFrame
    patch_metrics: pd.DataFrame
    scene_metrics: pd.DataFrame
    comparisons: pd.DataFrame
    matched_displacement: pd.DataFrame
    matched_displacement_summary: pd.DataFrame
    reconstruction_check: pd.DataFrame
    timings: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class _QueryFit:
    method: str
    layer: int | None
    model: str
    alpha: float
    fitted: FittedClassifier
    record: Mapping[str, Any]


def run_object_query_localization_study(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    *,
    save: bool = True,
) -> ObjectQueryLocalizationStudyResult:
    """Fit and evaluate a shared ``(object query, patch) -> overlap`` decoder."""

    normalized = _normalize_spec(spec)
    started = time.perf_counter()
    timings: dict[str, float] = {}

    step = time.perf_counter()
    data = prepare_visual_object_data(dataset, normalized)
    split = dict(data.source.method.get("split") or {})
    masks = split_masks(data.readouts.rows, split)
    supported = _supported_objects(
        data,
        masks["train"],
        min_train_episodes=int(normalized["probe"]["min_train_episodes"]),
    )
    train_examples = _sample_patch_examples(
        data,
        masks["train"],
        supported,
        max_examples=int(normalized["sampling"]["max_train_examples"]),
        seed=int(normalized["sampling"]["random_state"]),
    )
    selection_examples = _sample_patch_examples(
        data,
        masks["selection"],
        supported,
        max_examples=int(normalized["sampling"]["max_selection_examples"]),
        seed=int(normalized["sampling"]["random_state"]) + 1,
    )
    centers = normalized_patch_centers(data.readouts.token_metadata)
    context_encoder = fit_context_encoder(
        data.readouts.rows,
        masks["train"],
        [str(value) for value in normalized["context_columns"]],
    )
    context = context_encoder.transform(data.readouts.rows)
    timings["prepare_seconds"] = time.perf_counter() - step

    step = time.perf_counter()
    candidates, selected = _fit_candidates(
        data,
        train_examples,
        selection_examples,
        supported,
        centers,
        context,
        normalized["probe"],
    )
    shuffled_fit, shuffled_row_remap = _fit_within_task_shuffle(
        data,
        selected["activation_query"],
        train_examples,
        supported,
        centers,
        normalized["probe"],
    )
    selected["within_task_shuffled_activation"] = shuffled_fit
    timings["fit_seconds"] = time.perf_counter() - step

    step = time.perf_counter()
    selection_indices = np.flatnonzero(masks["selection"])
    test_indices = np.flatnonzero(masks["test"])
    fixed_map = _fixed_spatial_map(data, masks["train"], supported)
    spatial_permutation = np.random.default_rng(
        int(normalized["sampling"]["random_state"]) + 2
    ).permutation(len(centers))
    selection_scores, method_order = _all_method_scores(
        data,
        selection_indices,
        supported,
        centers,
        context,
        selected,
        fixed_map,
        spatial_permutation=spatial_permutation,
    )
    presence_threshold = _select_presence_threshold(
        selection_scores[method_order.index("activation_query")],
        data.visible[selection_indices],
        supported,
    )
    candidate_validation_ap = _mean_visible_patch_ap(
        data,
        selection_indices,
        selection_scores[method_order.index("activation_query")],
        supported,
    )
    fixed_map_validation_ap = _mean_visible_patch_ap(
        data,
        selection_indices,
        selection_scores[method_order.index("fixed_object_spatial_map")],
        supported,
    )
    validation_gate_passed = candidate_validation_ap > fixed_map_validation_ap
    if validation_gate_passed:
        test_scores, _ = _all_method_scores(
            data,
            test_indices,
            supported,
            centers,
            context,
            selected,
            fixed_map,
            spatial_permutation=spatial_permutation,
        )
        patch_metrics, scene_metrics = _evaluation_tables(
            data,
            test_indices,
            test_scores,
            method_order,
            supported,
            centers,
            presence_threshold=presence_threshold,
            patch_threshold=float(normalized["analysis"]["patch_threshold"]),
        )
        comparisons = _comparison_table(
            patch_metrics,
            scene_metrics,
            bootstrap_samples=int(normalized["analysis"]["bootstrap_samples"]),
        )
        matched_displacement = _matched_displacement_table(
            dataset,
            data,
            selected["activation_query"],
            supported,
            centers,
            str(normalized.get("matched_scene_artifact_id") or ""),
        )
        matched_displacement_summary = _matched_displacement_summary(
            matched_displacement,
            bootstrap_samples=int(normalized["analysis"]["bootstrap_samples"]),
        )
        reconstruction_check = _reconstruction_check(
            selected,
            test_scores,
            method_order,
            data,
            test_indices,
            supported,
            centers,
            context,
        )
    else:
        test_indices = np.array([], dtype=int)
        test_scores = np.zeros(
            (
                len(method_order),
                0,
                len(supported),
                data.patch_targets.shape[-1],
            ),
            dtype=np.float32,
        )
        patch_metrics = pd.DataFrame()
        scene_metrics = pd.DataFrame()
        comparisons = pd.DataFrame()
        matched_displacement = pd.DataFrame()
        matched_displacement_summary = pd.DataFrame()
        reconstruction_check = pd.DataFrame()
    selections = pd.DataFrame.from_records([dict(value.record) for value in selected.values()])
    timings["evaluate_seconds"] = time.perf_counter() - step
    timings["total_seconds"] = time.perf_counter() - started

    artifact = (
        _save_study(
            dataset,
            normalized,
            data,
            supported,
            train_examples,
            selection_examples,
            candidates,
            selections,
            patch_metrics,
            scene_metrics,
            comparisons,
            matched_displacement,
            matched_displacement_summary,
            reconstruction_check,
            selected,
            context_encoder,
            fixed_map,
            shuffled_row_remap,
            spatial_permutation,
            test_indices,
            test_scores,
            method_order,
            presence_threshold,
            validation_gate_passed,
            candidate_validation_ap,
            fixed_map_validation_ap,
            timings,
        )
        if save
        else None
    )
    return ObjectQueryLocalizationStudyResult(
        artifact,
        candidates,
        selections,
        patch_metrics,
        scene_metrics,
        comparisons,
        matched_displacement,
        matched_displacement_summary,
        reconstruction_check,
        timings,
    )


def _supported_objects(
    data: VisualObjectData,
    train_mask: np.ndarray,
    *,
    min_train_episodes: int,
) -> np.ndarray:
    supported = np.zeros(len(data.targets.vocabulary), dtype=bool)
    for object_index in range(len(supported)):
        supported[object_index] = data.readouts.rows.loc[
            train_mask & data.visible[:, object_index], "trace_id"
        ].astype(str).nunique() >= int(min_train_episodes)
    return supported


def _sample_patch_examples(
    data: VisualObjectData,
    row_mask: np.ndarray,
    supported: np.ndarray,
    *,
    max_examples: int,
    seed: int,
) -> pd.DataFrame:
    """Keep positives and a deterministic 1:3 near/wrong/background mixture."""

    rng = np.random.default_rng(seed)
    tokens = data.readouts.token_metadata.reset_index(drop=True)
    token_rows = tokens.get("patch_row", pd.Series(np.zeros(len(tokens), dtype=int))).to_numpy(int)
    token_cols = tokens.get("patch_col", pd.Series(np.arange(len(tokens)))).to_numpy(int)
    positive_records: list[tuple[int, int, int]] = []
    for row_index in np.flatnonzero(row_mask):
        for object_index in np.flatnonzero(data.visible[row_index] & supported):
            positive_records.extend(
                (int(row_index), int(object_index), int(patch_index))
                for patch_index in np.flatnonzero(data.patch_targets[row_index, object_index])
            )
    max_positive = max(1, int(max_examples) // 4)
    if len(positive_records) > max_positive:
        chosen = np.sort(rng.choice(len(positive_records), size=max_positive, replace=False))
        positive_records = [positive_records[index] for index in chosen]
    records: list[dict[str, Any]] = []
    for row_index, object_index, positive_patch in positive_records:
        target = data.patch_targets[row_index, object_index]
        other = data.patch_targets[row_index].copy()
        other[object_index] = False
        wrong_pool = np.flatnonzero(other.any(axis=0) & ~target)
        background_pool = np.flatnonzero(~data.patch_targets[row_index].any(axis=0) & ~target)
        target_indices = np.flatnonzero(target)
        row_distance = np.min(
            np.abs(token_rows[:, None] - token_rows[target_indices][None, :])
            + np.abs(token_cols[:, None] - token_cols[target_indices][None, :]),
            axis=1,
        )
        near_pool = np.flatnonzero((row_distance <= 1) & ~target)
        all_negative = np.flatnonzero(~target)
        pools = {
            "near_box": near_pool,
            "wrong_object": wrong_pool,
            "background": background_pool,
        }
        records.append(
            _example_record(data, row_index, object_index, positive_patch, 1, "positive")
        )
        for kind, pool in pools.items():
            candidates = pool if len(pool) else all_negative
            selected_patch = int(rng.choice(candidates))
            records.append(_example_record(data, row_index, object_index, selected_patch, 0, kind))
    frame = pd.DataFrame.from_records(records)
    if frame.empty or frame["label"].nunique() < 2:
        raise ValueError("Object-query sampling did not produce both labels")
    return frame.reset_index(drop=True)


def _example_record(
    data: VisualObjectData,
    row_index: int,
    object_index: int,
    patch_index: int,
    label: int,
    kind: str,
) -> dict[str, Any]:
    source = data.readouts.rows.iloc[row_index]
    return {
        **row_identity(source),
        "source_row_index": int(row_index),
        "object_index": int(object_index),
        "object_name": str(data.targets.vocabulary[object_index]),
        "patch_index": int(patch_index),
        "label": int(label),
        "sample_kind": kind,
    }


def _query_design(
    data: VisualObjectData,
    examples: pd.DataFrame,
    supported: np.ndarray,
    centers: np.ndarray,
    *,
    layer_index: int | None,
    context: np.ndarray | None = None,
    row_remap: np.ndarray | None = None,
) -> np.ndarray:
    object_indices = np.flatnonzero(supported)
    object_lookup = {int(value): index for index, value in enumerate(object_indices)}
    rows = examples["source_row_index"].to_numpy(dtype=int)
    patches = examples["patch_index"].to_numpy(dtype=int)
    queries = examples["object_index"].to_numpy(dtype=int)
    one_hot = np.zeros((len(examples), len(object_indices)), dtype=np.float32)
    one_hot[np.arange(len(examples)), [object_lookup[int(value)] for value in queries]] = 1.0
    pieces = [centers[patches], one_hot]
    if context is not None:
        pieces.insert(0, context[rows])
    if layer_index is not None:
        source_rows = row_remap[rows] if row_remap is not None else rows
        pieces.insert(0, data.compact[source_rows, layer_index, patches])
    return np.column_stack(pieces).astype(np.float32)


def _fit_candidates(
    data: VisualObjectData,
    train_examples: pd.DataFrame,
    selection_examples: pd.DataFrame,
    supported: np.ndarray,
    centers: np.ndarray,
    context: np.ndarray,
    settings: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, _QueryFit]]:
    records: list[dict[str, Any]] = []
    selected: dict[str, _QueryFit] = {}
    y_train = train_examples["label"].to_numpy(dtype=int)
    y_selection = selection_examples["label"].to_numpy(dtype=int)
    variants: list[tuple[str, int | None, np.ndarray, np.ndarray]] = []
    requested_layers = {int(value) for value in settings["layers"]}
    missing_layers = requested_layers - {int(value) for value in data.readouts.layers}
    if missing_layers:
        raise ValueError(
            f"Object-query source is missing requested layers {sorted(missing_layers)}"
        )
    for layer_index, layer in enumerate(data.readouts.layers):
        if int(layer) not in requested_layers:
            continue
        variants.append(
            (
                "activation_query",
                int(layer),
                _query_design(data, train_examples, supported, centers, layer_index=layer_index),
                _query_design(
                    data, selection_examples, supported, centers, layer_index=layer_index
                ),
            )
        )
    variants.extend(
        [
            (
                "query_xy",
                None,
                _query_design(data, train_examples, supported, centers, layer_index=None),
                _query_design(data, selection_examples, supported, centers, layer_index=None),
            ),
            (
                "prompt_scene_query_xy",
                None,
                _query_design(
                    data, train_examples, supported, centers, layer_index=None, context=context
                ),
                _query_design(
                    data, selection_examples, supported, centers, layer_index=None, context=context
                ),
            ),
        ]
    )
    for method, layer, train_values, selection_values in variants:
        for model in settings["models"]:
            for alpha in settings["alphas"]:
                fitted = fit_classifier(
                    train_values,
                    y_train,
                    model=str(model),
                    alpha=float(alpha),
                    hidden_units=int(settings["mlp_hidden_units"]),
                    max_iter=int(settings["max_iter"]),
                    random_state=int(settings["random_state"]),
                )
                probabilities = fitted.predict_proba(selection_values)
                positive_column = int(np.flatnonzero(fitted.classes == 1)[0])
                ap = float(average_precision_score(y_selection, probabilities[:, positive_column]))
                record = {
                    "method": method,
                    "layer": layer,
                    "model": str(model),
                    "alpha": float(alpha),
                    "feature_dim": int(train_values.shape[1]),
                    "training_example_count": int(len(train_values)),
                    "selection_example_count": int(len(selection_values)),
                    "selection_patch_average_precision": ap,
                    "iterations": fitted.n_iter,
                    "converged": fitted.converged,
                }
                records.append(record)
                candidate = _QueryFit(method, layer, str(model), float(alpha), fitted, record)
                current = selected.get(method)
                if current is None or ap > float(
                    current.record["selection_patch_average_precision"]
                ):
                    selected[method] = candidate
    return pd.DataFrame.from_records(records), selected


def _fit_within_task_shuffle(
    data: VisualObjectData,
    selected: _QueryFit,
    train_examples: pd.DataFrame,
    supported: np.ndarray,
    centers: np.ndarray,
    settings: Mapping[str, Any],
) -> tuple[_QueryFit, np.ndarray]:
    rows = data.readouts.rows
    task_keys = (
        rows["benchmark"].fillna("").astype(str)
        + ":"
        + rows.get("task_name", rows["task_id"]).fillna("").astype(str)
    )
    remap = np.arange(len(rows), dtype=int)
    rng = np.random.default_rng(int(settings["random_state"]) + 30)
    for _, group in rows.groupby(task_keys, sort=True):
        indices = group.index.to_numpy(dtype=int)
        remap[indices] = rng.permutation(indices)
    layer_index = list(data.readouts.layers).index(int(selected.layer))
    values = _query_design(
        data,
        train_examples,
        supported,
        centers,
        layer_index=layer_index,
        row_remap=remap,
    )
    fitted = fit_classifier(
        values,
        train_examples["label"].to_numpy(dtype=int),
        model=selected.model,
        alpha=selected.alpha,
        hidden_units=int(settings["mlp_hidden_units"]),
        max_iter=int(settings["max_iter"]),
        random_state=int(settings["random_state"]) + 31,
    )
    record = {
        "method": "within_task_shuffled_activation",
        "layer": selected.layer,
        "model": selected.model,
        "alpha": selected.alpha,
        "feature_dim": int(values.shape[1]),
        "training_example_count": int(len(values)),
        "selection_patch_average_precision": float("nan"),
        "iterations": fitted.n_iter,
        "converged": fitted.converged,
    }
    return (
        _QueryFit(
            "within_task_shuffled_activation",
            selected.layer,
            selected.model,
            selected.alpha,
            fitted,
            record,
        ),
        remap,
    )


def _score_fit(
    data: VisualObjectData,
    row_indices: np.ndarray,
    supported: np.ndarray,
    centers: np.ndarray,
    fitted: _QueryFit,
    context: np.ndarray,
    *,
    wrong_query: bool = False,
    patch_permutation: np.ndarray | None = None,
) -> np.ndarray:
    object_indices = np.flatnonzero(supported)
    scores = np.zeros((len(row_indices), len(supported), len(centers)), dtype=np.float32)
    class_column = int(np.flatnonzero(fitted.fitted.classes == 1)[0])
    layer_index = (
        list(data.readouts.layers).index(int(fitted.layer)) if fitted.layer is not None else None
    )
    query_order = np.roll(object_indices, 1) if wrong_query else object_indices
    for output_row, row_index in enumerate(row_indices):
        for query_position, object_index in enumerate(object_indices):
            query_index = int(query_order[query_position])
            one_hot = np.zeros((len(centers), len(object_indices)), dtype=np.float32)
            one_hot[:, np.flatnonzero(object_indices == query_index)[0]] = 1.0
            pieces = [centers, one_hot]
            if fitted.method == "prompt_scene_query_xy":
                pieces.insert(0, np.repeat(context[row_index][None, :], len(centers), axis=0))
            if layer_index is not None:
                token_values = data.compact[row_index, layer_index]
                if patch_permutation is not None:
                    token_values = token_values[patch_permutation]
                pieces.insert(0, token_values)
            values = np.column_stack(pieces)
            scores[output_row, object_index] = fitted.fitted.predict_proba(values)[:, class_column]
    return scores


def _fixed_spatial_map(
    data: VisualObjectData, train_mask: np.ndarray, supported: np.ndarray
) -> np.ndarray:
    result = np.zeros((len(supported), data.patch_targets.shape[-1]), dtype=np.float32)
    for object_index in np.flatnonzero(supported):
        available = train_mask & data.visible[:, object_index]
        if available.any():
            result[object_index] = data.patch_targets[available, object_index].mean(axis=0)
    return result


def _all_method_scores(
    data: VisualObjectData,
    row_indices: np.ndarray,
    supported: np.ndarray,
    centers: np.ndarray,
    context: np.ndarray,
    selected: Mapping[str, _QueryFit],
    fixed_map: np.ndarray,
    *,
    spatial_permutation: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    candidate = selected["activation_query"]
    methods = [
        "activation_query",
        "fixed_object_spatial_map",
        "query_xy",
        "prompt_scene_query_xy",
        "wrong_object_query",
        "within_task_shuffled_activation",
        "fixed_patch_position_permutation",
    ]
    values = [
        _score_fit(data, row_indices, supported, centers, candidate, context),
        np.broadcast_to(fixed_map[None, :, :], (len(row_indices), *fixed_map.shape)).copy(),
        _score_fit(data, row_indices, supported, centers, selected["query_xy"], context),
        _score_fit(
            data, row_indices, supported, centers, selected["prompt_scene_query_xy"], context
        ),
        _score_fit(data, row_indices, supported, centers, candidate, context, wrong_query=True),
        _score_fit(
            data,
            row_indices,
            supported,
            centers,
            selected["within_task_shuffled_activation"],
            context,
        ),
        _score_fit(
            data,
            row_indices,
            supported,
            centers,
            candidate,
            context,
            patch_permutation=spatial_permutation,
        ),
    ]
    return np.stack(values).astype(np.float32), methods


def _select_presence_threshold(
    scores: np.ndarray,
    visible: np.ndarray,
    supported: np.ndarray,
) -> float:
    object_indices = np.flatnonzero(supported)
    maxima = scores[:, object_indices].max(axis=2)
    truth = visible[:, object_indices]
    candidates = np.unique(np.quantile(maxima, np.linspace(0.05, 0.95, 19)))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        predicted = maxima >= threshold
        intersection = np.logical_and(predicted, truth).sum(axis=1)
        union = np.logical_or(predicted, truth).sum(axis=1)
        score = float(np.mean(intersection / np.maximum(union, 1)))
        if score > best_score:
            best_threshold, best_score = float(threshold), score
    return best_threshold


def _mean_visible_patch_ap(
    data: VisualObjectData,
    row_indices: np.ndarray,
    scores: np.ndarray,
    supported: np.ndarray,
) -> float:
    values = []
    for local_row, row_index in enumerate(row_indices):
        for object_index in np.flatnonzero(data.visible[row_index] & supported):
            truth = data.patch_targets[row_index, object_index]
            values.append(
                float(average_precision_score(truth.astype(int), scores[local_row, object_index]))
            )
    return float(np.mean(values)) if values else float("nan")


def _evaluation_tables(
    data: VisualObjectData,
    row_indices: np.ndarray,
    score_tensor: np.ndarray,
    method_order: Sequence[str],
    supported: np.ndarray,
    centers: np.ndarray,
    *,
    presence_threshold: float,
    patch_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    patch_records: list[dict[str, Any]] = []
    scene_records: list[dict[str, Any]] = []
    object_indices = np.flatnonzero(supported)
    tokens = data.readouts.token_metadata.reset_index(drop=True)
    image_width = float(tokens["pixel_x1"].max())
    image_height = float(tokens["pixel_y1"].max())
    for method_index, method in enumerate(method_order):
        values = score_tensor[method_index]
        for local_row, row_index in enumerate(row_indices):
            source = data.readouts.rows.iloc[row_index]
            predicted_presence = values[local_row, object_indices].max(axis=1) >= presence_threshold
            truth_presence = data.visible[row_index, object_indices]
            intersection = int(np.logical_and(predicted_presence, truth_presence).sum())
            union = int(np.logical_or(predicted_presence, truth_presence).sum())
            scene_records.append(
                {
                    "method": method,
                    **row_identity(source),
                    "scene_jaccard": float(intersection / max(1, union)),
                    "true_object_count": int(truth_presence.sum()),
                    "predicted_object_count": int(predicted_presence.sum()),
                    "presence_threshold": float(presence_threshold),
                }
            )
            for object_index in np.flatnonzero(data.visible[row_index] & supported):
                scores = values[local_row, object_index]
                truth = data.patch_targets[row_index, object_index]
                peak = int(np.argmax(scores))
                truth_box = data.boxes_px[row_index, object_index]
                true_center = np.array(
                    [
                        0.5 * (truth_box[0] + truth_box[2]) / image_width,
                        0.5 * (truth_box[1] + truth_box[3]) / image_height,
                    ]
                )
                center_error = float(
                    np.linalg.norm(
                        (centers[peak] - true_center) * np.array([image_width, image_height])
                    )
                )
                selected_patches = np.flatnonzero(scores >= patch_threshold)
                if not len(selected_patches):
                    selected_patches = np.array([peak])
                predicted_box = np.array(
                    [
                        tokens.iloc[selected_patches]["pixel_x0"].min() / image_width,
                        tokens.iloc[selected_patches]["pixel_y0"].min() / image_height,
                        tokens.iloc[selected_patches]["pixel_x1"].max() / image_width,
                        tokens.iloc[selected_patches]["pixel_y1"].max() / image_height,
                    ]
                )
                normalized_truth_box = truth_box / np.array(
                    [image_width, image_height, image_width, image_height]
                )
                patch_records.append(
                    {
                        "method": method,
                        **row_identity(source),
                        "source_row_index": int(row_index),
                        "object_index": int(object_index),
                        "object_name": str(data.targets.vocabulary[object_index]),
                        "average_precision": float(
                            average_precision_score(truth.astype(int), scores)
                        ),
                        "peak_center_error_px": center_error,
                        "iou": float(_box_iou(predicted_box, normalized_truth_box)),
                        "truth_bbox_xyxy": truth_box.tolist(),
                        "predicted_bbox_normalized": predicted_box.tolist(),
                        "peak_patch_index": peak,
                    }
                )
    return pd.DataFrame.from_records(patch_records), pd.DataFrame.from_records(scene_records)


def _comparison_table(
    patch_metrics: pd.DataFrame,
    scene_metrics: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    baselines = sorted(set(patch_metrics["method"]) - {"activation_query"})
    for baseline in baselines:
        patch_candidate = patch_metrics.loc[patch_metrics["method"] == "activation_query"]
        patch_baseline = patch_metrics.loc[patch_metrics["method"] == baseline]
        paired = patch_candidate.merge(
            patch_baseline[
                ["trace_id", "object_index", "average_precision", "peak_center_error_px", "iou"]
            ],
            on=["trace_id", "object_index"],
            suffixes=("_candidate", "_baseline"),
        )
        for metric, direction in [
            ("average_precision", 1.0),
            ("peak_center_error_px", -1.0),
            ("iou", 1.0),
        ]:
            candidate = direction * paired[f"{metric}_candidate"].to_numpy()
            reference = direction * paired[f"{metric}_baseline"].to_numpy()
            for unit, groups in [
                ("benchmark_task", paired["task_key"].to_numpy()),
                ("instruction", paired["instruction_key"].to_numpy()),
                ("object", paired["object_index"].to_numpy()),
            ]:
                records.append(
                    {
                        "candidate": "activation_query",
                        "baseline": baseline,
                        "metric": metric,
                        "unit": unit,
                        **grouped_paired_interval(
                            candidate,
                            reference,
                            groups,
                            bootstrap_samples=bootstrap_samples,
                            seed=20260830 + len(records),
                        ),
                    }
                )
        scene_candidate = scene_metrics.loc[scene_metrics["method"] == "activation_query"]
        scene_baseline = scene_metrics.loc[scene_metrics["method"] == baseline]
        scene_paired = scene_candidate.merge(
            scene_baseline[["trace_id", "scene_jaccard"]],
            on="trace_id",
            suffixes=("_candidate", "_baseline"),
        )
        for unit, groups in [
            ("benchmark_task", scene_paired["task_key"].to_numpy()),
            ("instruction", scene_paired["instruction_key"].to_numpy()),
        ]:
            records.append(
                {
                    "candidate": "activation_query",
                    "baseline": baseline,
                    "metric": "scene_jaccard",
                    "unit": unit,
                    **grouped_paired_interval(
                        scene_paired["scene_jaccard_candidate"].to_numpy(),
                        scene_paired["scene_jaccard_baseline"].to_numpy(),
                        groups,
                        bootstrap_samples=bootstrap_samples,
                        seed=20260850 + len(records),
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _matched_displacement_table(
    dataset: TraceDataset,
    data: VisualObjectData,
    fitted: _QueryFit,
    supported: np.ndarray,
    centers: np.ndarray,
    artifact_id: str,
) -> pd.DataFrame:
    if not artifact_id:
        return pd.DataFrame()
    artifact = dataset.load_artifact(artifact_id)
    outputs = dict(artifact.method.get("outputs") or {})
    relative = outputs.get("matched_pairs")
    if relative is None:
        raise KeyError(f"Matched-scene artifact {artifact_id!r} has no matched_pairs output")
    pairs = pd.read_parquet(dataset._dataset_artifact_root() / str(relative))
    row_lookup = {str(row.trace_id): int(index) for index, row in data.readouts.rows.iterrows()}
    object_lookup = {str(name): index for index, name in enumerate(data.targets.vocabulary)}
    requested_rows = sorted(
        {
            row_lookup[trace]
            for trace in set(pairs["left_trace_id"].astype(str))
            | set(pairs["right_trace_id"].astype(str))
            if trace in row_lookup
        }
    )
    if not requested_rows:
        return pd.DataFrame()
    row_indices = np.asarray(requested_rows, dtype=int)
    scores = _score_fit(
        data,
        row_indices,
        supported,
        centers,
        fitted,
        np.zeros((len(data.readouts.rows), 0), dtype=np.float32),
    )
    score_lookup = {row_index: position for position, row_index in enumerate(requested_rows)}
    tokens = data.readouts.token_metadata
    width = float(tokens["pixel_x1"].max())
    height = float(tokens["pixel_y1"].max())
    pixel_scale = np.array([width, height])
    records: list[dict[str, Any]] = []
    for pair in pairs.itertuples():
        left_trace, right_trace = str(pair.left_trace_id), str(pair.right_trace_id)
        object_name = str(pair.moved_object_name)
        if (
            left_trace not in row_lookup
            or right_trace not in row_lookup
            or object_name not in object_lookup
        ):
            continue
        object_index = object_lookup[object_name]
        if not supported[object_index]:
            continue
        left_index, right_index = row_lookup[left_trace], row_lookup[right_trace]
        if (
            not data.visible[left_index, object_index]
            or not data.visible[right_index, object_index]
        ):
            continue
        left_peak = int(np.argmax(scores[score_lookup[left_index], object_index]))
        right_peak = int(np.argmax(scores[score_lookup[right_index], object_index]))
        predicted_delta = (centers[right_peak] - centers[left_peak]) * pixel_scale
        left_box, right_box = (
            data.boxes_px[left_index, object_index],
            data.boxes_px[right_index, object_index],
        )
        true_delta = np.array(
            [
                0.5 * (right_box[0] + right_box[2] - left_box[0] - left_box[2]),
                0.5 * (right_box[1] + right_box[3] - left_box[1] - left_box[3]),
            ]
        )
        error = float(np.linalg.norm(predicted_delta - true_delta))
        zero_error = float(np.linalg.norm(true_delta))
        denominator = float(np.linalg.norm(predicted_delta) * np.linalg.norm(true_delta))
        records.append(
            {
                "pair_id": str(pair.pair_id),
                "scene_key": str(pair.scene_key),
                "split": str(pair.split),
                "left_trace_id": left_trace,
                "right_trace_id": right_trace,
                "object_index": int(object_index),
                "object_name": object_name,
                "predicted_delta_px": predicted_delta.tolist(),
                "true_delta_px": true_delta.tolist(),
                "displacement_error_px": error,
                "zero_displacement_error_px": zero_error,
                "error_improvement_over_zero_px": zero_error - error,
                "direction_cosine": float(np.dot(predicted_delta, true_delta) / denominator)
                if denominator > 0
                else float("nan"),
            }
        )
    return pd.DataFrame.from_records(records)


def _matched_displacement_summary(
    pairs: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    records = []
    groups = pairs["scene_key"].astype(str).to_numpy()
    for metric in ["error_improvement_over_zero_px", "direction_cosine"]:
        values = pairs[metric].to_numpy(dtype=float)
        records.append(
            {
                "metric": metric,
                "unit": "matched_scene",
                **grouped_paired_interval(
                    values,
                    np.zeros(len(values), dtype=float),
                    groups,
                    bootstrap_samples=bootstrap_samples,
                    seed=20260870 + len(records),
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _reconstruction_check(
    selected: Mapping[str, _QueryFit],
    test_scores: np.ndarray,
    method_order: Sequence[str],
    data: VisualObjectData,
    test_indices: np.ndarray,
    supported: np.ndarray,
    centers: np.ndarray,
    context: np.ndarray,
) -> pd.DataFrame:
    records = []
    for method in [
        "activation_query",
        "query_xy",
        "prompt_scene_query_xy",
        "within_task_shuffled_activation",
    ]:
        rebuilt = _score_fit(data, test_indices[:1], supported, centers, selected[method], context)
        expected = test_scores[method_order.index(method), :1]
        records.append(
            {
                "method": method,
                "value_count": int(rebuilt.size),
                "max_absolute_error": float(np.max(np.abs(rebuilt - expected))),
            }
        )
    return pd.DataFrame.from_records(records)


def _projection_arrays(data: VisualObjectData) -> dict[str, np.ndarray]:
    projection = data.readouts.channel_projection
    return {
        "channel_input_center": projection.input_center,
        "channel_input_scale": projection.input_scale,
        "channel_pca_center": projection.pca_center,
        "channel_components": projection.components,
        "channel_explained_variance_ratio": projection.explained_variance_ratio,
    }


def _save_study(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    data: VisualObjectData,
    supported: np.ndarray,
    train_examples: pd.DataFrame,
    selection_examples: pd.DataFrame,
    candidates: pd.DataFrame,
    selections: pd.DataFrame,
    patch_metrics: pd.DataFrame,
    scene_metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    matched_displacement: pd.DataFrame,
    matched_displacement_summary: pd.DataFrame,
    reconstruction_check: pd.DataFrame,
    selected: Mapping[str, _QueryFit],
    context_encoder: ContextEncoder,
    fixed_map: np.ndarray,
    shuffled_row_remap: np.ndarray,
    spatial_permutation: np.ndarray,
    test_indices: np.ndarray,
    test_scores: np.ndarray,
    method_order: Sequence[str],
    presence_threshold: float,
    validation_gate_passed: bool,
    candidate_validation_ap: float,
    fixed_map_validation_ap: float,
    timings: Mapping[str, float],
) -> LensArtifact:
    artifact_id = make_artifact_id(str(spec["name"]), "object_query_localization_study")
    relative_dir = Path("artifacts") / artifact_id
    tables = {
        "train_examples": train_examples,
        "selection_examples": selection_examples,
        "candidates": candidates,
        "selections": selections,
        "patch_metrics": patch_metrics,
        "scene_metrics": scene_metrics,
        "comparisons": comparisons,
        "matched_displacement": matched_displacement,
        "matched_displacement_summary": matched_displacement_summary,
        "reconstruction_check": reconstruction_check,
        "evaluation_rows": data.readouts.rows.iloc[test_indices].reset_index(drop=True),
        "source_rows": data.readouts.rows,
        "source_sites": data.readouts.source_sites,
        "token_metadata": data.readouts.token_metadata,
        "vocabulary": data.vocabulary,
        "context_categories": pd.DataFrame.from_records(
            [
                {"column": column, "categories": categories.astype(str).tolist()}
                for column, categories in zip(
                    context_encoder.columns, context_encoder.encoder.categories_, strict=True
                )
            ]
        ),
        "within_task_shuffle_rows": pd.DataFrame(
            {
                "source_row_index": np.arange(len(shuffled_row_remap), dtype=np.int64),
                "shuffled_source_row_index": shuffled_row_remap.astype(np.int64),
            }
        ),
    }
    outputs = {name: str(relative_dir / f"{name}.parquet") for name in tables}
    arrays = {
        **_projection_arrays(data),
        "supported_objects": supported.astype(np.uint8),
        "fixed_object_spatial_map": fixed_map,
        "fixed_patch_position_permutation": spatial_permutation.astype(np.int64),
        "test_patch_scores": test_scores,
        "test_patch_truth": data.patch_targets[test_indices].astype(np.uint8),
        "test_visible": data.visible[test_indices].astype(np.uint8),
    }
    model_contract: dict[str, Any] = {}
    for index, (method, fitted) in enumerate(sorted(selected.items())):
        prefix = f"selected_{index}"
        arrays.update(classifier_arrays(prefix, fitted.fitted))
        model_contract[method] = {
            "prefix": prefix,
            "layer": fitted.layer,
            "model": fitted.model,
            "alpha": fitted.alpha,
            "feature_dim": int(len(fitted.fitted.feature_mean)),
        }
    artifact = LensArtifact(
        artifact_id=artifact_id,
        artifact_type="object_query_localization_study",
        name=str(spec["name"]),
        group_id="scene_map_probe_studies",
        scope="dataset",
        selector={
            "source_probe_artifact_id": data.source.artifact_id,
            "matched_scene_artifact_id": spec.get("matched_scene_artifact_id"),
            "feature": data.source.selector.get("feature"),
        },
        method={
            "workflow": "run_object_query_localization_study",
            "schema_version": OBJECT_QUERY_LOCALIZATION_SCHEMA_VERSION,
            "research": {
                key: spec[key]
                for key in ["question", "hypothesis_family", "intended_claim"]
                if spec.get(key)
            },
            "split": data.source.method.get("split"),
            "probe": spec["probe"],
            "sampling": spec["sampling"],
            "analysis": {
                **spec["analysis"],
                "presence_threshold_selected_on_validation": presence_threshold,
                "test_gate": {
                    "passed": validation_gate_passed,
                    "candidate_validation_patch_ap": candidate_validation_ap,
                    "fixed_map_validation_patch_ap": fixed_map_validation_ap,
                    "rule": "candidate validation patch AP must exceed fixed spatial map",
                },
            },
            "models": model_contract,
            "controls": [
                "fixed_object_spatial_map",
                "query_xy",
                "prompt_scene_query_xy",
                "wrong_object_query",
                "within_task_shuffled_activation",
                "fixed_patch_position_permutation",
            ],
            "outputs": outputs,
            "array_axes": {
                "test_patch_scores": ["method", "evaluation_row", "object", "patch"],
                "test_patch_truth": ["evaluation_row", "object", "patch"],
                "test_visible": ["evaluation_row", "object"],
            },
            "array_method_order": {"test_patch_scores": list(method_order)},
            "storage_contract": {
                "raw_activations": "referenced from capture and never copied",
                "compact_token_cache": {
                    "key": data.compact_cache_key,
                    "cache_hit": data.compact_cache_hit,
                    "rebuildable": True,
                },
                "image_box_cache": {
                    "key": data.box_cache_key,
                    "cache_hit": data.box_cache_hit,
                    "rebuildable": True,
                },
                "saved_evidence": (
                    "exact sampled examples, rows, token coordinates, fitted query heads, "
                    "test heatmaps, boxes, scene predictions, controls, grouped intervals, "
                    "and matched-scene displacement"
                ),
            },
            "timings_seconds": dict(timings),
        },
        metrics={
            "training_example_count": int(len(train_examples)),
            "selection_example_count": int(len(selection_examples)),
            "supported_object_count": int(supported.sum()),
            "test_row_count": int(len(test_indices)),
            "patch_metric_count": int(len(patch_metrics)),
            "matched_pair_count": int(len(matched_displacement)),
            "total_seconds": float(timings["total_seconds"]),
        },
        display={
            "kind": "object_query_localization_study",
            "status": "exploratory",
            "selections": json.loads(selections.to_json(orient="records")),
            "comparisons": json.loads(comparisons.to_json(orient="records")),
        },
        tags=("probe", "object-query", "object-location", "visual-tokens", "exploratory"),
        source_trace_ids=tuple(sorted(data.readouts.rows["trace_id"].astype(str).unique())),
    )
    artifact_dir = dataset._dataset_artifact_root() / relative_dir
    artifact_dir.mkdir(parents=True, exist_ok=False)
    try:
        for name, frame in tables.items():
            frame.to_parquet(artifact_dir / f"{name}.parquet", index=False)
        return dataset.save_artifact(artifact, arrays=arrays)
    except BaseException:
        shutil.rmtree(artifact_dir)
        raise


def _normalize_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(spec)
    if not normalized.get("source_probe_artifact_id"):
        raise ValueError("Object-query study requires source_probe_artifact_id")
    normalized.setdefault("name", "PI0.5 explicit visual object-query localization study")
    normalized.setdefault("camera_name", "agentview")
    normalized.setdefault("context_columns", ["benchmark", "scene_family", "task_phase", "prompt"])
    probe = dict(normalized.get("probe") or {})
    probe.setdefault("layers", [0, 4, 8, 12, 17])
    probe.setdefault("models", ["linear", "mlp"])
    probe.setdefault("alphas", [0.0001])
    probe.setdefault("mlp_hidden_units", 64)
    probe.setdefault("max_iter", 300)
    probe.setdefault("min_train_episodes", 5)
    probe.setdefault("random_state", 20260830)
    normalized["probe"] = probe
    sampling = dict(normalized.get("sampling") or {})
    sampling.setdefault("max_train_examples", 400_000)
    sampling.setdefault("max_selection_examples", 100_000)
    sampling.setdefault("negative_ratio", 3)
    sampling.setdefault("random_state", 20260830)
    if int(sampling["negative_ratio"]) != 3:
        raise ValueError(
            "Object-query sampling currently requires negative_ratio=3 "
            "(near-box, wrong-object, background)"
        )
    normalized["sampling"] = sampling
    analysis = dict(normalized.get("analysis") or {})
    analysis.setdefault("io_workers", 8)
    analysis.setdefault("bootstrap_samples", 2000)
    analysis.setdefault("patch_threshold", 0.5)
    normalized["analysis"] = analysis
    return normalized
