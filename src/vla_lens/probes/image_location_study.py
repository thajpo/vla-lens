"""Direct image-plane object localization probes for captured visual tokens."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPRegressor

from vla_lens.artifacts import LensArtifact, make_artifact_id
from vla_lens.probes.identity_localization_study import (
    _artifact_table,
    _grouped_bootstrap,
    _object_boxes,
    _patch_metrics,
)
from vla_lens.probes.scene_map_study import (
    _context_design,
    _supported_objects,
    scene_map_target_table,
)
from vla_lens.probes.token_representations import (
    LayerTokenReadouts,
    ProjectionState,
    build_layer_token_readouts,
    read_compressed_token_layers,
)
from vla_lens.traces import TraceDataset

IMAGE_LOCATION_STUDY_SCHEMA_VERSION = 2
IMAGE_BOX_CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ImageLocationStudyResult:
    """Saved direct-localization study and its review tables."""

    artifact: LensArtifact | None
    candidates: pd.DataFrame
    selections: pd.DataFrame
    capacity_checks: pd.DataFrame
    patch_metrics: pd.DataFrame
    box_metrics: pd.DataFrame
    comparisons: pd.DataFrame
    capacity_comparisons: pd.DataFrame
    shuffled_controls: pd.DataFrame
    object_summary: pd.DataFrame
    examples: pd.DataFrame
    reconstruction_check: pd.DataFrame
    timings: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class _PatchFit:
    record: Mapping[str, Any]
    coefficients: np.ndarray
    intercepts: np.ndarray
    supported: np.ndarray
    prediction: np.ndarray


@dataclass(frozen=True, slots=True)
class _BoxFit:
    record: Mapping[str, Any]
    coefficients: np.ndarray
    intercepts: np.ndarray
    supported: np.ndarray
    prediction: np.ndarray


@dataclass(frozen=True, slots=True)
class _PatchMLPFit:
    record: Mapping[str, Any]
    input_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray
    prediction: np.ndarray


@dataclass(frozen=True, slots=True)
class _BoxMLPFit:
    record: Mapping[str, Any]
    input_weights: np.ndarray
    hidden_bias: np.ndarray
    output_weights: np.ndarray
    output_bias: np.ndarray
    supported: np.ndarray
    prediction: np.ndarray


def run_image_location_probe_study(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    *,
    save: bool = True,
) -> ImageLocationStudyResult:
    """Fit local patch and whole-scene image-box probes on identical rows."""

    normalized = _normalize_spec(spec)
    started = time.perf_counter()
    timings: dict[str, float] = {}

    step = time.perf_counter()
    source = dataset.load_artifact(str(normalized["source_probe_artifact_id"]))
    source_rows = _artifact_table(dataset, source, "source_rows")
    token_metadata = _artifact_table(dataset, source, "token_metadata")
    vocabulary = _artifact_table(dataset, source, "vocabulary")
    source_probe = dict(source.method.get("probe") or {})
    split = dict(source.method.get("split") or {})
    readouts = build_layer_token_readouts(
        dataset,
        dict(source.selector.get("feature") or {}),
        split,
        readout_dim=max(int(value) for value in source_probe["readout_dims"]),
        token_channel_dim=int(source_probe["token_channel_dim"]),
        channel_sample_count=int(source_probe["channel_sample_count"]),
        projection_fit_rows=int(source_probe["projection_fit_rows"]),
        io_workers=int(normalized["analysis"]["io_workers"]),
        cache=True,
    )
    _require_same_rows(source_rows, readouts.rows)
    compact = read_compressed_token_layers(
        dataset,
        readouts.rows,
        readouts.source_sites,
        readouts.token_metadata,
        layers=readouts.layers,
        channel_projection=readouts.channel_projection,
        generation_step=source.selector.get("feature", {}).get("generation_step"),
        io_workers=int(normalized["analysis"]["io_workers"]),
        cache=True,
    )
    timings["prepare_representations_seconds"] = time.perf_counter() - step

    step = time.perf_counter()
    targets, target_vocabulary = scene_map_target_table(dataset, readouts.rows, cache=True)
    expected_names = vocabulary.sort_values("object_index")["object_name"].astype(str).tolist()
    if list(targets.vocabulary) != expected_names:
        raise ValueError("Source probe vocabulary does not match current scene targets")
    boxes_px, visible, box_cache_key, box_cache_hit = _cached_object_boxes(
        dataset,
        readouts.rows,
        token_metadata,
        targets.presence,
        targets.vocabulary,
        camera_name=str(normalized["camera_name"]),
    )
    image_width = float(token_metadata["pixel_x1"].max())
    image_height = float(token_metadata["pixel_y1"].max())
    boxes = _normalize_boxes(boxes_px, image_width, image_height)
    box_targets = _center_size(boxes)
    patch_targets = _box_patch_targets(boxes_px, visible, token_metadata)
    object_names = list(targets.vocabulary)
    masks = _split_masks(readouts.rows, split)
    supported = _supported_objects(
        visible.astype(np.float32),
        readouts.rows,
        masks["train"],
        int(normalized["probe"]["min_train_episodes"]),
    )
    timings["prepare_targets_seconds"] = time.perf_counter() - step

    step = time.perf_counter()
    patch_candidates, patch_fit = _fit_patch_candidates(
        compact.values,
        readouts.layers,
        patch_targets,
        visible,
        supported,
        masks,
        token_metadata,
        alphas=[float(value) for value in normalized["probe"]["ridge_alphas"]],
    )
    timings["fit_patch_candidates_seconds"] = time.perf_counter() - step

    step = time.perf_counter()
    box_candidates, box_fit = _fit_box_candidates(
        readouts,
        box_targets,
        visible,
        supported,
        masks,
        image_width=image_width,
        image_height=image_height,
        readout_dims=[int(value) for value in normalized["probe"]["readout_dims"]],
        alphas=[float(value) for value in normalized["probe"]["ridge_alphas"]],
    )
    context_fit = _fit_context_box_baseline(
        readouts.rows,
        box_targets,
        visible,
        supported,
        masks,
        context_columns=[str(value) for value in normalized["context_columns"]],
        alphas=[float(value) for value in normalized["probe"]["baseline_ridge_alphas"]],
        image_width=image_width,
        image_height=image_height,
    )
    mean_prediction = _mean_box_baseline(box_targets, visible, masks["train"])
    fixed_patch_map = _fixed_patch_baseline(patch_targets, visible, masks["train"])
    fixed_patch_prediction = np.broadcast_to(
        fixed_patch_map[None, :, :], patch_fit.prediction.shape
    ).copy()
    timings["fit_box_candidates_seconds"] = time.perf_counter() - step

    step = time.perf_counter()
    patch_mlp_fit, box_mlp_fit = _fit_nonlinear_capacity_checks(
        compact.values,
        readouts,
        patch_targets,
        box_targets,
        visible,
        masks,
        patch_fit,
        box_fit,
        normalized["nonlinear"],
        image_width=image_width,
        image_height=image_height,
    )
    timings["fit_nonlinear_capacity_seconds"] = time.perf_counter() - step

    step = time.perf_counter()
    test_mask = masks["test"]
    patch_metrics = pd.concat(
        [
            _patch_metric_table(
                "activation_patch_head",
                patch_fit.prediction,
                patch_targets,
                boxes_px,
                visible,
                patch_fit.supported,
                readouts.rows,
                test_mask,
                token_metadata,
                object_names,
            ),
            _patch_metric_table(
                "fixed_object_spatial_map",
                fixed_patch_prediction,
                patch_targets,
                boxes_px,
                visible,
                supported,
                readouts.rows,
                test_mask,
                token_metadata,
                object_names,
            ),
        ],
        ignore_index=True,
    )
    box_metrics = pd.concat(
        [
            _box_metric_table(
                "activation_box_decoder",
                box_fit.prediction,
                box_targets,
                visible,
                box_fit.supported,
                readouts.rows,
                test_mask,
                image_width,
                image_height,
                object_names,
            ),
            _box_metric_table(
                "per_object_training_mean",
                mean_prediction,
                box_targets,
                visible,
                supported,
                readouts.rows,
                test_mask,
                image_width,
                image_height,
                object_names,
            ),
            _box_metric_table(
                "prompt_and_scene_context",
                context_fit.prediction,
                box_targets,
                visible,
                context_fit.supported,
                readouts.rows,
                test_mask,
                image_width,
                image_height,
                object_names,
            ),
        ],
        ignore_index=True,
    )
    if patch_mlp_fit is not None:
        patch_metrics = pd.concat(
            [
                patch_metrics,
                _patch_metric_table(
                    "activation_patch_head_mlp",
                    patch_mlp_fit.prediction,
                    patch_targets,
                    boxes_px,
                    visible,
                    supported,
                    readouts.rows,
                    test_mask,
                    token_metadata,
                    object_names,
                ),
            ],
            ignore_index=True,
        )
    if box_mlp_fit is not None:
        box_metrics = pd.concat(
            [
                box_metrics,
                _box_metric_table(
                    "activation_box_decoder_mlp",
                    box_mlp_fit.prediction,
                    box_targets,
                    visible,
                    box_mlp_fit.supported,
                    readouts.rows,
                    test_mask,
                    image_width,
                    image_height,
                    object_names,
                ),
            ],
            ignore_index=True,
        )
    comparisons = _comparison_table(
        patch_metrics,
        box_metrics,
        bootstrap_samples=int(normalized["analysis"]["bootstrap_samples"]),
    )
    capacity_comparisons = _capacity_comparison_table(
        patch_metrics,
        box_metrics,
        bootstrap_samples=int(normalized["analysis"]["bootstrap_samples"]),
    )
    shuffled_controls = _shuffled_controls(
        compact.values,
        readouts,
        patch_fit,
        box_fit,
        patch_targets,
        box_targets,
        visible,
        masks,
        token_metadata,
        image_width,
        image_height,
        repeats=int(normalized["analysis"]["shuffle_repeats"]),
    )
    reconstruction_check = _reconstruction_check(
        compact.values,
        readouts,
        patch_fit,
        box_fit,
        patch_mlp_fit,
        box_mlp_fit,
    )
    object_summary = _object_summary(patch_metrics, box_metrics, object_names)
    examples = _example_table(
        patch_metrics,
        box_metrics,
        count=int(normalized["analysis"]["example_count"]),
    )
    timings["evaluate_seconds"] = time.perf_counter() - step
    timings["total_seconds"] = time.perf_counter() - started

    candidates = pd.concat([patch_candidates, box_candidates], ignore_index=True, sort=False)
    selections = pd.DataFrame.from_records(
        [dict(patch_fit.record), dict(box_fit.record), dict(context_fit.record)]
    )
    capacity_checks = pd.DataFrame.from_records(
        [dict(fitted.record) for fitted in [patch_mlp_fit, box_mlp_fit] if fitted is not None]
    )
    artifact = (
        _save_study(
            dataset,
            normalized,
            source,
            readouts,
            compact.cache_key,
            compact.cache_hit,
            box_cache_key,
            box_cache_hit,
            target_vocabulary,
            boxes,
            visible,
            patch_targets,
            candidates,
            selections,
            capacity_checks,
            patch_metrics,
            box_metrics,
            comparisons,
            capacity_comparisons,
            shuffled_controls,
            object_summary,
            examples,
            reconstruction_check,
            patch_fit,
            box_fit,
            patch_mlp_fit,
            box_mlp_fit,
            context_fit,
            fixed_patch_map,
            mean_prediction[0],
            masks["test"],
            timings,
        )
        if save
        else None
    )
    return ImageLocationStudyResult(
        artifact=artifact,
        candidates=candidates,
        selections=selections,
        capacity_checks=capacity_checks,
        patch_metrics=patch_metrics,
        box_metrics=box_metrics,
        comparisons=comparisons,
        capacity_comparisons=capacity_comparisons,
        shuffled_controls=shuffled_controls,
        object_summary=object_summary,
        examples=examples,
        reconstruction_check=reconstruction_check,
        timings=timings,
    )


def _fit_patch_candidates(
    compact: np.ndarray,
    layers: Sequence[int],
    targets: np.ndarray,
    visible: np.ndarray,
    supported: np.ndarray,
    masks: Mapping[str, np.ndarray],
    tokens: pd.DataFrame,
    *,
    alphas: Sequence[float],
) -> tuple[pd.DataFrame, _PatchFit]:
    records: list[dict[str, Any]] = []
    best: _PatchFit | None = None
    for layer_index, layer in enumerate(layers):
        X = compact[:, layer_index]
        for alpha in alphas:
            coefficients, intercepts, prediction = _fit_patch_decoder(
                X, targets, masks["train"], float(alpha)
            )
            metrics = _patch_candidate_metrics(
                prediction,
                targets,
                visible,
                supported,
                masks,
                tokens,
            )
            record = {
                "study_part": "patch_head",
                "representation": "local_token",
                "layer": int(layer),
                "ridge_alpha": float(alpha),
                **metrics,
            }
            records.append(record)
            fitted = _PatchFit(record, coefficients, intercepts, supported.copy(), prediction)
            if best is None or _patch_selection_score(record) > _patch_selection_score(best.record):
                best = fitted
    if best is None:
        raise ValueError("Patch probe battery produced no candidates")
    return pd.DataFrame.from_records(records), best


def _fit_patch_decoder(
    values: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(values, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32).transpose(0, 2, 1)
    model = Ridge(alpha=float(alpha))
    model.fit(X[train_mask].reshape(-1, X.shape[-1]), y[train_mask].reshape(-1, y.shape[-1]))
    prediction = model.predict(X.reshape(-1, X.shape[-1])).reshape(len(X), X.shape[1], y.shape[-1])
    return (
        np.asarray(model.coef_, dtype=np.float32),
        np.asarray(model.intercept_, dtype=np.float32),
        np.asarray(prediction.transpose(0, 2, 1), dtype=np.float32),
    )


def _fit_box_candidates(
    readouts: LayerTokenReadouts,
    targets: np.ndarray,
    visible: np.ndarray,
    supported: np.ndarray,
    masks: Mapping[str, np.ndarray],
    *,
    image_width: float,
    image_height: float,
    readout_dims: Sequence[int],
    alphas: Sequence[float],
) -> tuple[pd.DataFrame, _BoxFit]:
    records: list[dict[str, Any]] = []
    best: _BoxFit | None = None
    representations = {"pooled": readouts.pooled, "tokenwise": readouts.tokenwise}
    for representation, all_values in representations.items():
        for layer_index, layer in enumerate(readouts.layers):
            for requested_dim in sorted(set(int(value) for value in readout_dims)):
                dim = min(requested_dim, all_values.shape[-1])
                X = all_values[:, layer_index, :dim]
                for alpha in alphas:
                    coefficients, intercepts, fitted_supported, prediction = _fit_box_decoder(
                        X,
                        targets,
                        visible,
                        masks["train"],
                        supported,
                        float(alpha),
                    )
                    metrics = _box_candidate_metrics(
                        prediction,
                        targets,
                        visible,
                        fitted_supported,
                        masks,
                        image_width,
                        image_height,
                    )
                    record = {
                        "study_part": "box_decoder",
                        "representation": representation,
                        "layer": int(layer),
                        "readout_dim": int(dim),
                        "ridge_alpha": float(alpha),
                        **metrics,
                    }
                    records.append(record)
                    fitted = _BoxFit(
                        record,
                        coefficients,
                        intercepts,
                        fitted_supported,
                        prediction,
                    )
                    if best is None or _box_selection_score(record) > _box_selection_score(
                        best.record
                    ):
                        best = fitted
    if best is None:
        raise ValueError("Box probe battery produced no candidates")
    return pd.DataFrame.from_records(records), best


def _fit_box_decoder(
    X: np.ndarray,
    targets: np.ndarray,
    visible: np.ndarray,
    train_mask: np.ndarray,
    supported: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(X, dtype=np.float64)
    object_count = targets.shape[1]
    coefficients = np.zeros((object_count, 4, values.shape[1]), dtype=np.float32)
    intercepts = np.zeros((object_count, 4), dtype=np.float32)
    prediction = np.full(targets.shape, np.nan, dtype=np.float32)
    fitted_supported = np.asarray(supported, dtype=bool).copy()
    for object_index in np.flatnonzero(fitted_supported):
        available = train_mask & visible[:, object_index]
        if not available.any():
            fitted_supported[object_index] = False
            continue
        model = Ridge(alpha=float(alpha))
        model.fit(values[available], targets[available, object_index])
        coefficients[object_index] = np.asarray(model.coef_, dtype=np.float32)
        intercepts[object_index] = np.asarray(model.intercept_, dtype=np.float32)
        prediction[:, object_index] = model.predict(values)
    return coefficients, intercepts, fitted_supported, prediction


def _fit_context_box_baseline(
    rows: pd.DataFrame,
    targets: np.ndarray,
    visible: np.ndarray,
    supported: np.ndarray,
    masks: Mapping[str, np.ndarray],
    *,
    context_columns: Sequence[str],
    alphas: Sequence[float],
    image_width: float,
    image_height: float,
) -> _BoxFit:
    design = _context_design(rows, masks["train"], context_columns)
    best: _BoxFit | None = None
    for alpha in alphas:
        coefficients, intercepts, fitted_supported, prediction = _fit_box_decoder(
            design,
            targets,
            visible,
            masks["train"],
            supported,
            float(alpha),
        )
        metrics = _box_candidate_metrics(
            prediction,
            targets,
            visible,
            fitted_supported,
            masks,
            image_width,
            image_height,
        )
        record = {
            "study_part": "box_baseline",
            "representation": "prompt_and_scene_context",
            "layer": None,
            "readout_dim": int(design.shape[1]),
            "ridge_alpha": float(alpha),
            **metrics,
        }
        fitted = _BoxFit(record, coefficients, intercepts, fitted_supported, prediction)
        if best is None or _box_selection_score(record) > _box_selection_score(best.record):
            best = fitted
    if best is None:
        raise ValueError("Context box baseline produced no candidates")
    return best


def _fit_nonlinear_capacity_checks(
    compact: np.ndarray,
    readouts: LayerTokenReadouts,
    patch_targets: np.ndarray,
    box_targets: np.ndarray,
    visible: np.ndarray,
    masks: Mapping[str, np.ndarray],
    patch_fit: _PatchFit,
    box_fit: _BoxFit,
    settings: Mapping[str, Any],
    *,
    image_width: float,
    image_height: float,
) -> tuple[_PatchMLPFit | None, _BoxMLPFit | None]:
    if not bool(settings.get("enabled", True)):
        return None, None
    hidden = int(settings["hidden_units"])
    patch_layer = list(readouts.layers).index(int(patch_fit.record["layer"]))
    patch_values = compact[:, patch_layer]
    flat_values = patch_values[masks["train"]].reshape(-1, patch_values.shape[-1])
    flat_targets = (
        patch_targets[masks["train"]]
        .transpose(0, 2, 1)
        .reshape(-1, patch_targets.shape[1])
        .astype(np.float32)
    )
    max_train_tokens = int(settings["patch_max_train_tokens"])
    if len(flat_values) > max_train_tokens:
        rng = np.random.default_rng(int(settings["random_state"]))
        chosen = np.sort(rng.choice(len(flat_values), size=max_train_tokens, replace=False))
        flat_values = flat_values[chosen]
        flat_targets = flat_targets[chosen]
    patch_model = MLPRegressor(
        hidden_layer_sizes=(hidden,),
        activation="relu",
        solver="adam",
        batch_size=int(settings["patch_batch_size"]),
        learning_rate_init=float(settings["learning_rate_init"]),
        max_iter=int(settings["patch_max_iter"]),
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=int(settings["n_iter_no_change"]),
        random_state=int(settings["random_state"]),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        patch_model.fit(flat_values, flat_targets)
    patch_warning = any(issubclass(item.category, ConvergenceWarning) for item in caught)
    patch_prediction = patch_model.predict(
        patch_values.reshape(-1, patch_values.shape[-1])
    ).reshape(len(patch_values), patch_values.shape[1], patch_targets.shape[1])
    patch_prediction = patch_prediction.transpose(0, 2, 1).astype(np.float32)
    patch_metrics = _patch_candidate_metrics(
        patch_prediction,
        patch_targets,
        visible,
        patch_fit.supported,
        masks,
        readouts.token_metadata,
    )
    patch_record = {
        "study_part": "patch_capacity_check",
        "representation": "local_token",
        "model": "mlp",
        "activation": "relu",
        "solver": "adam",
        "early_stopping": True,
        "layer": int(patch_fit.record["layer"]),
        "readout_dim": int(patch_values.shape[-1]),
        "hidden_units": hidden,
        "max_iter": int(settings["patch_max_iter"]),
        "iterations": int(patch_model.n_iter_),
        "converged": not patch_warning,
        "training_token_count": int(len(flat_values)),
        **patch_metrics,
    }
    patch_result = _PatchMLPFit(
        patch_record,
        np.asarray(patch_model.coefs_[0], dtype=np.float32),
        np.asarray(patch_model.intercepts_[0], dtype=np.float32),
        np.asarray(patch_model.coefs_[1], dtype=np.float32),
        np.asarray(patch_model.intercepts_[1], dtype=np.float32),
        patch_prediction,
    )

    box_layer = list(readouts.layers).index(int(box_fit.record["layer"]))
    box_values = (
        readouts.tokenwise if box_fit.record["representation"] == "tokenwise" else readouts.pooled
    )[:, box_layer, : int(box_fit.record["readout_dim"])]
    box_result = _fit_box_mlp(
        box_values,
        box_targets,
        visible,
        masks,
        box_fit.supported,
        hidden_units=hidden,
        max_iter=int(settings["box_max_iter"]),
        random_state=int(settings["random_state"]),
        record_base={
            "study_part": "box_capacity_check",
            "representation": str(box_fit.record["representation"]),
            "model": "mlp",
            "activation": "relu",
            "solver": "lbfgs",
            "early_stopping": False,
            "layer": int(box_fit.record["layer"]),
            "readout_dim": int(box_fit.record["readout_dim"]),
        },
        image_width=image_width,
        image_height=image_height,
    )
    return patch_result, box_result


def _fit_box_mlp(
    values: np.ndarray,
    targets: np.ndarray,
    visible: np.ndarray,
    masks: Mapping[str, np.ndarray],
    supported: np.ndarray,
    *,
    hidden_units: int,
    max_iter: int,
    random_state: int,
    record_base: Mapping[str, Any],
    image_width: float,
    image_height: float,
) -> _BoxMLPFit:
    object_count = targets.shape[1]
    input_weights = np.zeros((object_count, values.shape[1], hidden_units), dtype=np.float32)
    hidden_bias = np.zeros((object_count, hidden_units), dtype=np.float32)
    output_weights = np.zeros((object_count, hidden_units, 4), dtype=np.float32)
    output_bias = np.zeros((object_count, 4), dtype=np.float32)
    prediction = np.full(targets.shape, np.nan, dtype=np.float32)
    fitted_supported = np.asarray(supported, dtype=bool).copy()
    converged = 0
    iterations: list[int] = []
    for object_index in np.flatnonzero(fitted_supported):
        available = masks["train"] & visible[:, object_index]
        if available.sum() < 2:
            fitted_supported[object_index] = False
            continue
        model = MLPRegressor(
            hidden_layer_sizes=(hidden_units,),
            activation="relu",
            solver="lbfgs",
            max_iter=max_iter,
            random_state=random_state + int(object_index),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(values[available], targets[available, object_index])
        if not any(issubclass(item.category, ConvergenceWarning) for item in caught):
            converged += 1
        iterations.append(int(model.n_iter_))
        input_weights[object_index] = np.asarray(model.coefs_[0], dtype=np.float32)
        hidden_bias[object_index] = np.asarray(model.intercepts_[0], dtype=np.float32)
        output_weights[object_index] = np.asarray(model.coefs_[1], dtype=np.float32)
        output_bias[object_index] = np.asarray(model.intercepts_[1], dtype=np.float32)
        prediction[:, object_index] = model.predict(values)
    metrics = _box_candidate_metrics(
        prediction,
        targets,
        visible,
        fitted_supported,
        masks,
        image_width=image_width,
        image_height=image_height,
    )
    record = {
        **dict(record_base),
        "hidden_units": hidden_units,
        "max_iter": max_iter,
        "supported_head_count": int(fitted_supported.sum()),
        "converged_head_count": int(converged),
        "mean_iterations": float(np.mean(iterations)) if iterations else float("nan"),
        **metrics,
    }
    return _BoxMLPFit(
        record,
        input_weights,
        hidden_bias,
        output_weights,
        output_bias,
        fitted_supported,
        prediction,
    )


def _patch_candidate_metrics(
    prediction: np.ndarray,
    targets: np.ndarray,
    visible: np.ndarray,
    supported: np.ndarray,
    masks: Mapping[str, np.ndarray],
    tokens: pd.DataFrame,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split_name in ["selection", "test"]:
        values = _patch_instance_scores(
            prediction,
            targets,
            visible,
            supported,
            masks[split_name],
            tokens,
        )
        out[f"{split_name}_mean_average_precision"] = float(values["ap"].mean())
        out[f"{split_name}_peak_center_error_px"] = float(values["peak_center_error_px"].mean())
        out[f"{split_name}_visible_object_count"] = int(len(values))
    return out


def _box_candidate_metrics(
    prediction: np.ndarray,
    targets: np.ndarray,
    visible: np.ndarray,
    supported: np.ndarray,
    masks: Mapping[str, np.ndarray],
    image_width: float,
    image_height: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split_name in ["selection", "test"]:
        values = _box_instance_scores(
            prediction,
            targets,
            visible,
            supported,
            masks[split_name],
            image_width,
            image_height,
        )
        out[f"{split_name}_center_error_px"] = float(values["center_error_px"].mean())
        out[f"{split_name}_mean_iou"] = float(values["iou"].mean())
        out[f"{split_name}_size_mae"] = float(values["size_mae"].mean())
        out[f"{split_name}_visible_object_count"] = int(len(values))
    return out


def _patch_instance_scores(
    prediction: np.ndarray,
    targets: np.ndarray,
    visible: np.ndarray,
    supported: np.ndarray,
    row_mask: np.ndarray,
    tokens: pd.DataFrame,
) -> pd.DataFrame:
    patch_x = 0.5 * (
        tokens["pixel_x0"].to_numpy(dtype=float) + tokens["pixel_x1"].to_numpy(dtype=float)
    )
    patch_y = 0.5 * (
        tokens["pixel_y0"].to_numpy(dtype=float) + tokens["pixel_y1"].to_numpy(dtype=float)
    )
    records: list[dict[str, Any]] = []
    for row_index in np.flatnonzero(row_mask):
        for object_index in np.flatnonzero(visible[row_index] & supported):
            truth = targets[row_index, object_index]
            scores = prediction[row_index, object_index]
            peak = int(np.nanargmax(scores))
            target_indices = np.flatnonzero(truth)
            target_x = float(patch_x[target_indices].mean())
            target_y = float(patch_y[target_indices].mean())
            records.append(
                {
                    "row_index": int(row_index),
                    "object_index": int(object_index),
                    "ap": float(average_precision_score(truth, scores)),
                    "peak_center_error_px": float(
                        np.hypot(patch_x[peak] - target_x, patch_y[peak] - target_y)
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _box_instance_scores(
    prediction: np.ndarray,
    targets: np.ndarray,
    visible: np.ndarray,
    supported: np.ndarray,
    row_mask: np.ndarray,
    image_width: float,
    image_height: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row_index in np.flatnonzero(row_mask):
        for object_index in np.flatnonzero(visible[row_index] & supported):
            truth = targets[row_index, object_index]
            predicted = prediction[row_index, object_index]
            center_error = np.hypot(
                (predicted[0] - truth[0]) * image_width,
                (predicted[1] - truth[1]) * image_height,
            )
            records.append(
                {
                    "row_index": int(row_index),
                    "object_index": int(object_index),
                    "center_error_px": float(center_error),
                    "size_mae": float(np.abs(predicted[2:] - truth[2:]).mean()),
                    "iou": _box_iou(_center_size_to_xyxy(predicted), _center_size_to_xyxy(truth)),
                }
            )
    return pd.DataFrame.from_records(records)


def _patch_metric_table(
    method: str,
    prediction: np.ndarray,
    targets: np.ndarray,
    boxes_px: np.ndarray,
    visible: np.ndarray,
    supported: np.ndarray,
    rows: pd.DataFrame,
    row_mask: np.ndarray,
    tokens: pd.DataFrame,
    object_names: Sequence[str],
) -> pd.DataFrame:
    instance = _patch_instance_scores(prediction, targets, visible, supported, row_mask, tokens)
    records: list[dict[str, Any]] = []
    for item in instance.itertuples():
        row = rows.iloc[item.row_index]
        object_index = int(item.object_index)
        target_mask = targets[item.row_index, object_index]
        wrong_index, wrong_mask = _wrong_object_patch_mask(
            targets[item.row_index], visible[item.row_index], object_index, target_mask
        )
        metrics = _patch_metrics(prediction[item.row_index, object_index], target_mask, wrong_mask)
        records.append(
            {
                **_row_identity(row),
                "method": method,
                "object_index": object_index,
                "object_name": str(object_names[object_index]),
                "wrong_object_index": wrong_index,
                "bbox_xyxy": boxes_px[item.row_index, object_index].tolist(),
                "peak_center_error_px": float(item.peak_center_error_px),
                **metrics,
            }
        )
    return pd.DataFrame.from_records(records)


def _box_metric_table(
    method: str,
    prediction: np.ndarray,
    targets: np.ndarray,
    visible: np.ndarray,
    supported: np.ndarray,
    rows: pd.DataFrame,
    row_mask: np.ndarray,
    image_width: float,
    image_height: float,
    object_names: Sequence[str],
) -> pd.DataFrame:
    instance = _box_instance_scores(
        prediction,
        targets,
        visible,
        supported,
        row_mask,
        image_width,
        image_height,
    )
    records: list[dict[str, Any]] = []
    for item in instance.itertuples():
        row = rows.iloc[item.row_index]
        records.append(
            {
                **_row_identity(row),
                "method": method,
                "object_index": int(item.object_index),
                "object_name": str(object_names[int(item.object_index)]),
                "center_error_px": float(item.center_error_px),
                "size_mae": float(item.size_mae),
                "iou": float(item.iou),
                "truth_center_size": targets[item.row_index, item.object_index].tolist(),
                "prediction_center_size": prediction[item.row_index, item.object_index].tolist(),
            }
        )
    return pd.DataFrame.from_records(records)


def _comparison_table(
    patch_metrics: pd.DataFrame,
    box_metrics: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    specs = [
        (
            patch_metrics,
            "activation_patch_head",
            "fixed_object_spatial_map",
            "average_precision",
            1.0,
        ),
        (
            patch_metrics,
            "activation_patch_head",
            "fixed_object_spatial_map",
            "peak_center_error_px",
            -1.0,
        ),
        (
            box_metrics,
            "activation_box_decoder",
            "per_object_training_mean",
            "center_error_px",
            -1.0,
        ),
        (
            box_metrics,
            "activation_box_decoder",
            "prompt_and_scene_context",
            "center_error_px",
            -1.0,
        ),
        (
            box_metrics,
            "activation_box_decoder",
            "per_object_training_mean",
            "iou",
            1.0,
        ),
        (
            box_metrics,
            "activation_box_decoder",
            "prompt_and_scene_context",
            "iou",
            1.0,
        ),
    ]
    keys = ["trace_id", "object_index"]
    for frame, candidate, baseline, metric, direction in specs:
        candidate_rows = frame.loc[frame["method"] == candidate]
        baseline_rows = frame.loc[frame["method"] == baseline]
        paired = candidate_rows.merge(
            baseline_rows[keys + [metric]],
            on=keys,
            suffixes=("_candidate", "_baseline"),
        )
        improvement = direction * (paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"])
        for unit, groups in [
            ("episode", paired["trace_id"].astype(str).to_numpy()),
            ("benchmark_task", paired["task_key"].astype(str).to_numpy()),
            ("instruction", paired["instruction_key"].astype(str).to_numpy()),
            ("object", paired["object_index"].astype(str).to_numpy()),
        ]:
            records.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric,
                    "unit": unit,
                    **_grouped_bootstrap(
                        improvement.to_numpy(),
                        groups,
                        bootstrap_samples=bootstrap_samples,
                        seed=20260730 + len(records),
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _capacity_comparison_table(
    patch_metrics: pd.DataFrame,
    box_metrics: pd.DataFrame,
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    specs = [
        (
            patch_metrics,
            "activation_patch_head_mlp",
            "activation_patch_head",
            "average_precision",
            1.0,
        ),
        (
            patch_metrics,
            "activation_patch_head_mlp",
            "fixed_object_spatial_map",
            "average_precision",
            1.0,
        ),
        (
            box_metrics,
            "activation_box_decoder_mlp",
            "activation_box_decoder",
            "center_error_px",
            -1.0,
        ),
        (
            box_metrics,
            "activation_box_decoder_mlp",
            "prompt_and_scene_context",
            "center_error_px",
            -1.0,
        ),
        (
            box_metrics,
            "activation_box_decoder_mlp",
            "activation_box_decoder",
            "iou",
            1.0,
        ),
        (
            box_metrics,
            "activation_box_decoder_mlp",
            "prompt_and_scene_context",
            "iou",
            1.0,
        ),
    ]
    keys = ["trace_id", "object_index"]
    for frame, candidate, baseline, metric, direction in specs:
        candidate_rows = frame.loc[frame["method"] == candidate]
        baseline_rows = frame.loc[frame["method"] == baseline]
        if candidate_rows.empty or baseline_rows.empty:
            continue
        paired = candidate_rows.merge(
            baseline_rows[keys + [metric]],
            on=keys,
            suffixes=("_candidate", "_baseline"),
        )
        improvement = direction * (paired[f"{metric}_candidate"] - paired[f"{metric}_baseline"])
        for unit, groups in [
            ("episode", paired["trace_id"].astype(str).to_numpy()),
            ("benchmark_task", paired["task_key"].astype(str).to_numpy()),
            ("instruction", paired["instruction_key"].astype(str).to_numpy()),
            ("object", paired["object_index"].astype(str).to_numpy()),
        ]:
            records.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "metric": metric,
                    "unit": unit,
                    **_grouped_bootstrap(
                        improvement.to_numpy(),
                        groups,
                        bootstrap_samples=bootstrap_samples,
                        seed=20260801 + len(records),
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _shuffled_controls(
    compact: np.ndarray,
    readouts: LayerTokenReadouts,
    patch_fit: _PatchFit,
    box_fit: _BoxFit,
    patch_targets: np.ndarray,
    box_targets: np.ndarray,
    visible: np.ndarray,
    masks: Mapping[str, np.ndarray],
    tokens: pd.DataFrame,
    image_width: float,
    image_height: float,
    *,
    repeats: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    train_indices = np.flatnonzero(masks["train"])
    patch_layer = list(readouts.layers).index(int(patch_fit.record["layer"]))
    box_layer = list(readouts.layers).index(int(box_fit.record["layer"]))
    box_values = (
        readouts.tokenwise if box_fit.record["representation"] == "tokenwise" else readouts.pooled
    )[:, box_layer, : int(box_fit.record["readout_dim"])]
    for repeat in range(max(0, int(repeats))):
        rng = np.random.default_rng(20260730 + repeat)
        permutation = rng.permutation(train_indices)
        shuffled_patch = patch_targets.copy()
        shuffled_patch[train_indices] = patch_targets[permutation]
        _, _, patch_prediction = _fit_patch_decoder(
            compact[:, patch_layer],
            shuffled_patch,
            masks["train"],
            float(patch_fit.record["ridge_alpha"]),
        )
        patch_values = _patch_instance_scores(
            patch_prediction,
            patch_targets,
            visible,
            patch_fit.supported,
            masks["test"],
            tokens,
        )
        shuffled_boxes = box_targets.copy()
        shuffled_visible = visible.copy()
        shuffled_boxes[train_indices] = box_targets[permutation]
        shuffled_visible[train_indices] = visible[permutation]
        _, _, box_supported, box_prediction = _fit_box_decoder(
            box_values,
            shuffled_boxes,
            shuffled_visible,
            masks["train"],
            box_fit.supported,
            float(box_fit.record["ridge_alpha"]),
        )
        box_values_scored = _box_instance_scores(
            box_prediction,
            box_targets,
            visible,
            box_supported,
            masks["test"],
            image_width,
            image_height,
        )
        records.extend(
            [
                {
                    "study_part": "patch_head",
                    "repeat": repeat,
                    "mean_average_precision": float(patch_values["ap"].mean()),
                    "peak_center_error_px": float(patch_values["peak_center_error_px"].mean()),
                },
                {
                    "study_part": "box_decoder",
                    "repeat": repeat,
                    "center_error_px": float(box_values_scored["center_error_px"].mean()),
                    "mean_iou": float(box_values_scored["iou"].mean()),
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def _reconstruction_check(
    compact: np.ndarray,
    readouts: LayerTokenReadouts,
    patch_fit: _PatchFit,
    box_fit: _BoxFit,
    patch_mlp_fit: _PatchMLPFit | None = None,
    box_mlp_fit: _BoxMLPFit | None = None,
) -> pd.DataFrame:
    patch_layer = list(readouts.layers).index(int(patch_fit.record["layer"]))
    patch_rebuilt = (
        np.einsum(
            "rtc,oc->rot",
            compact[:, patch_layer],
            patch_fit.coefficients,
            optimize=True,
        )
        + patch_fit.intercepts[None, :, None]
    )
    box_layer = list(readouts.layers).index(int(box_fit.record["layer"]))
    box_values = (
        readouts.tokenwise if box_fit.record["representation"] == "tokenwise" else readouts.pooled
    )[:, box_layer, : int(box_fit.record["readout_dim"])]
    box_rebuilt = (
        np.einsum("rd,ocd->roc", box_values, box_fit.coefficients, optimize=True)
        + box_fit.intercepts[None, :, :]
    )
    box_mask = np.broadcast_to(box_fit.supported[None, :, None], box_rebuilt.shape)
    records = [
        {
            "study_part": "patch_head",
            "value_count": int(patch_rebuilt.size),
            "max_absolute_error": float(np.max(np.abs(patch_rebuilt - patch_fit.prediction))),
        },
        {
            "study_part": "box_decoder",
            "value_count": int(box_mask.sum()),
            "max_absolute_error": float(
                np.max(np.abs(box_rebuilt[box_mask] - box_fit.prediction[box_mask]))
            ),
        },
    ]
    if patch_mlp_fit is not None:
        patch_hidden = np.maximum(
            0.0,
            compact[:, patch_layer] @ patch_mlp_fit.input_weights + patch_mlp_fit.hidden_bias,
        )
        patch_mlp_rebuilt = (
            patch_hidden @ patch_mlp_fit.output_weights + patch_mlp_fit.output_bias
        ).transpose(0, 2, 1)
        records.append(
            {
                "study_part": "patch_head_mlp",
                "value_count": int(patch_mlp_rebuilt.size),
                "max_absolute_error": float(
                    np.max(np.abs(patch_mlp_rebuilt - patch_mlp_fit.prediction))
                ),
            }
        )
    if box_mlp_fit is not None:
        box_hidden = np.maximum(
            0.0,
            np.einsum(
                "rd,odh->roh",
                box_values,
                box_mlp_fit.input_weights,
                optimize=True,
            )
            + box_mlp_fit.hidden_bias[None, :, :],
        )
        box_mlp_rebuilt = (
            np.einsum(
                "roh,ohc->roc",
                box_hidden,
                box_mlp_fit.output_weights,
                optimize=True,
            )
            + box_mlp_fit.output_bias[None, :, :]
        )
        box_mlp_mask = np.broadcast_to(box_mlp_fit.supported[None, :, None], box_mlp_rebuilt.shape)
        records.append(
            {
                "study_part": "box_decoder_mlp",
                "value_count": int(box_mlp_mask.sum()),
                "max_absolute_error": float(
                    np.max(
                        np.abs(box_mlp_rebuilt[box_mlp_mask] - box_mlp_fit.prediction[box_mlp_mask])
                    )
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _object_summary(
    patch_metrics: pd.DataFrame,
    box_metrics: pd.DataFrame,
    object_names: Sequence[str],
) -> pd.DataFrame:
    patch = patch_metrics.pivot_table(
        index="object_index", columns="method", values="average_precision", aggfunc="mean"
    )
    peak = patch_metrics.pivot_table(
        index="object_index", columns="method", values="peak_center_error_px", aggfunc="mean"
    )
    center = box_metrics.pivot_table(
        index="object_index", columns="method", values="center_error_px", aggfunc="mean"
    )
    iou = box_metrics.pivot_table(
        index="object_index", columns="method", values="iou", aggfunc="mean"
    )
    indices = sorted(set(patch.index) | set(center.index))
    records: list[dict[str, Any]] = []
    for object_index in indices:
        records.append(
            {
                "object_index": int(object_index),
                "object_name": str(object_names[int(object_index)]),
                "patch_average_precision": _table_value(
                    patch, object_index, "activation_patch_head"
                ),
                "fixed_map_average_precision": _table_value(
                    patch, object_index, "fixed_object_spatial_map"
                ),
                "patch_ap_lift": _table_value(patch, object_index, "activation_patch_head")
                - _table_value(patch, object_index, "fixed_object_spatial_map"),
                "patch_mlp_average_precision": _table_value(
                    patch, object_index, "activation_patch_head_mlp"
                ),
                "patch_mlp_ap_lift": _table_value(patch, object_index, "activation_patch_head_mlp")
                - _table_value(patch, object_index, "fixed_object_spatial_map"),
                "patch_peak_error_px": _table_value(peak, object_index, "activation_patch_head"),
                "box_center_error_px": _table_value(center, object_index, "activation_box_decoder"),
                "mean_baseline_center_error_px": _table_value(
                    center, object_index, "per_object_training_mean"
                ),
                "context_baseline_center_error_px": _table_value(
                    center, object_index, "prompt_and_scene_context"
                ),
                "box_iou": _table_value(iou, object_index, "activation_box_decoder"),
                "box_mlp_center_error_px": _table_value(
                    center, object_index, "activation_box_decoder_mlp"
                ),
                "box_mlp_iou": _table_value(iou, object_index, "activation_box_decoder_mlp"),
            }
        )
    return pd.DataFrame.from_records(records)


def _example_table(
    patch_metrics: pd.DataFrame,
    box_metrics: pd.DataFrame,
    *,
    count: int,
) -> pd.DataFrame:
    patch = patch_metrics.loc[patch_metrics["method"] == "activation_patch_head"].copy()
    box = box_metrics.loc[box_metrics["method"] == "activation_box_decoder"].copy()
    count = max(1, int(count))
    frames = [
        patch.nlargest(count, "average_precision").assign(
            study_part="patch_head", example_kind="best"
        ),
        patch.nsmallest(count, "average_precision").assign(
            study_part="patch_head", example_kind="worst"
        ),
        box.nsmallest(count, "center_error_px").assign(
            study_part="box_decoder", example_kind="best"
        ),
        box.nlargest(count, "center_error_px").assign(
            study_part="box_decoder", example_kind="worst"
        ),
    ]
    return pd.concat(frames, ignore_index=True, sort=False)


def _mean_box_baseline(
    targets: np.ndarray, visible: np.ndarray, train_mask: np.ndarray
) -> np.ndarray:
    means = np.full((targets.shape[1], 4), np.nan, dtype=np.float32)
    for object_index in range(targets.shape[1]):
        available = train_mask & visible[:, object_index]
        if available.any():
            means[object_index] = targets[available, object_index].mean(axis=0)
    return np.broadcast_to(means[None, :, :], targets.shape).copy()


def _fixed_patch_baseline(
    targets: np.ndarray, visible: np.ndarray, train_mask: np.ndarray
) -> np.ndarray:
    fixed = np.zeros((targets.shape[1], targets.shape[2]), dtype=np.float32)
    for object_index in range(targets.shape[1]):
        available = train_mask & visible[:, object_index]
        if available.any():
            fixed[object_index] = targets[available, object_index].mean(axis=0)
    return fixed


def _normalize_boxes(boxes: np.ndarray, image_width: float, image_height: float) -> np.ndarray:
    scale = np.asarray([image_width, image_height, image_width, image_height])
    return np.asarray(boxes, dtype=np.float32) / scale


def _cached_object_boxes(
    dataset: TraceDataset,
    rows: pd.DataFrame,
    tokens: pd.DataFrame,
    presence: np.ndarray,
    vocabulary: Sequence[str],
    *,
    camera_name: str,
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    row_columns = [
        column for column in ["trace_id", "timestep", "policy_call_index"] if column in rows
    ]
    row_hash = pd.util.hash_pandas_object(rows[row_columns], index=False).to_numpy(dtype=np.uint64)
    token_columns = ["token_index", "pixel_x0", "pixel_x1", "pixel_y0", "pixel_y1"]
    available_token_columns = [column for column in token_columns if column in tokens]
    token_hash = pd.util.hash_pandas_object(tokens[available_token_columns], index=False).to_numpy(
        dtype=np.uint64
    )
    payload = {
        "schema": IMAGE_BOX_CACHE_SCHEMA_VERSION,
        "rows": hashlib.sha256(row_hash.tobytes()).hexdigest(),
        "tokens": hashlib.sha256(token_hash.tobytes()).hexdigest(),
        "presence": hashlib.sha256(np.asarray(presence, dtype=np.uint8).tobytes()).hexdigest(),
        "vocabulary": [str(value) for value in vocabulary],
        "camera_name": str(camera_name),
    }
    cache_key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    cache_path = dataset.cache_dir() / "image_object_boxes" / cache_key
    metadata_path = cache_path / "metadata.json"
    boxes_path = cache_path / "boxes.npy"
    visible_path = cache_path / "visible.npy"
    if metadata_path.exists() and boxes_path.exists() and visible_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("schema_version") == IMAGE_BOX_CACHE_SCHEMA_VERSION
            and metadata.get("cache_key") == cache_key
        ):
            return (
                np.asarray(np.load(boxes_path, allow_pickle=False)),
                np.asarray(np.load(visible_path, allow_pickle=False), dtype=bool),
                cache_key,
                True,
            )
    boxes, visible = _object_boxes(
        dataset,
        rows,
        tokens,
        presence,
        vocabulary,
        camera_name=camera_name,
    )
    cache_path.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        metadata_path.unlink()
    temporary_boxes = cache_path / "boxes.tmp.npy"
    temporary_visible = cache_path / "visible.tmp.npy"
    np.save(temporary_boxes, boxes)
    np.save(temporary_visible, visible.astype(np.uint8))
    os.replace(temporary_boxes, boxes_path)
    os.replace(temporary_visible, visible_path)
    temporary_metadata = cache_path / "metadata.tmp.json"
    temporary_metadata.write_text(
        json.dumps(
            {
                "schema_version": IMAGE_BOX_CACHE_SCHEMA_VERSION,
                "cache_key": cache_key,
                "shape": [int(value) for value in boxes.shape],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    os.replace(temporary_metadata, metadata_path)
    return boxes, visible, cache_key, False


def _center_size(boxes: np.ndarray) -> np.ndarray:
    values = np.asarray(boxes, dtype=np.float32)
    return np.stack(
        [
            0.5 * (values[..., 0] + values[..., 2]),
            0.5 * (values[..., 1] + values[..., 3]),
            values[..., 2] - values[..., 0],
            values[..., 3] - values[..., 1],
        ],
        axis=-1,
    )


def _center_size_to_xyxy(value: np.ndarray) -> np.ndarray:
    cx, cy, width, height = np.asarray(value, dtype=float)
    width = max(0.0, float(width))
    height = max(0.0, float(height))
    return np.clip(
        [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
        0.0,
        1.0,
    )


def _box_iou(left: np.ndarray, right: np.ndarray) -> float:
    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    right_area = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    return intersection / max(1e-12, left_area + right_area - intersection)


def _box_patch_targets(
    boxes_px: np.ndarray, visible: np.ndarray, tokens: pd.DataFrame
) -> np.ndarray:
    x0 = tokens["pixel_x0"].to_numpy(dtype=float)[None, None, :]
    x1 = tokens["pixel_x1"].to_numpy(dtype=float)[None, None, :]
    y0 = tokens["pixel_y0"].to_numpy(dtype=float)[None, None, :]
    y1 = tokens["pixel_y1"].to_numpy(dtype=float)[None, None, :]
    boxes = np.asarray(boxes_px, dtype=float)
    mask = (
        (x1 > boxes[..., 0, None])
        & (x0 < boxes[..., 2, None])
        & (y1 > boxes[..., 1, None])
        & (y0 < boxes[..., 3, None])
    )
    return mask & visible[..., None]


def _wrong_object_patch_mask(
    masks: np.ndarray,
    visible: np.ndarray,
    target_index: int,
    target_mask: np.ndarray,
) -> tuple[int | None, np.ndarray]:
    best_index: int | None = None
    best_mask = np.zeros_like(target_mask, dtype=bool)
    best_overlap = float("inf")
    for object_index in np.flatnonzero(visible):
        if int(object_index) == target_index:
            continue
        mask = np.asarray(masks[object_index], dtype=bool)
        overlap = float(np.logical_and(mask, target_mask).sum() / max(1, mask.sum()))
        if mask.any() and overlap < best_overlap:
            best_index, best_mask, best_overlap = int(object_index), mask, overlap
    return best_index, best_mask


def _split_masks(rows: pd.DataFrame, split: Mapping[str, Any]) -> dict[str, np.ndarray]:
    column = str(split["column"])
    return {
        name: rows[column].astype(str).to_numpy() == str(split[f"{name}_value"])
        for name in ["train", "selection", "test"]
    }


def _row_identity(row: pd.Series) -> dict[str, Any]:
    return {
        "trace_id": str(row["trace_id"]),
        "episode_id": row.get("episode_id"),
        "benchmark": row.get("benchmark"),
        "task_id": row.get("task_id"),
        "task_key": f"{row.get('benchmark')}:{row.get('task_name', row.get('task_id'))}",
        "instruction_key": str(row.get("prompt")),
        "prompt": row.get("prompt"),
    }


def _patch_selection_score(record: Mapping[str, Any]) -> float:
    return float(record["selection_mean_average_precision"])


def _box_selection_score(record: Mapping[str, Any]) -> float:
    return -float(record["selection_center_error_px"])


def _table_value(table: pd.DataFrame, index: int, column: str) -> float:
    if index not in table.index or column not in table:
        return float("nan")
    return float(table.loc[index, column])


def _require_same_rows(left: pd.DataFrame, right: pd.DataFrame) -> None:
    columns = ["trace_id", "timestep", "policy_call_index"]
    available = [column for column in columns if column in left and column in right]
    if len(left) != len(right) or not left[available].reset_index(drop=True).equals(
        right[available].reset_index(drop=True)
    ):
        raise ValueError("Source artifact rows do not match reconstructed token readouts")


def _projection_arrays(prefix: str, projection: ProjectionState) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_input_center": projection.input_center,
        f"{prefix}_input_scale": projection.input_scale,
        f"{prefix}_pca_center": projection.pca_center,
        f"{prefix}_components": projection.components,
        f"{prefix}_explained_variance_ratio": projection.explained_variance_ratio,
    }


def _save_study(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    source: LensArtifact,
    readouts: LayerTokenReadouts,
    compact_cache_key: str,
    compact_cache_hit: bool,
    box_cache_key: str,
    box_cache_hit: bool,
    vocabulary: pd.DataFrame,
    boxes: np.ndarray,
    visible: np.ndarray,
    patch_targets: np.ndarray,
    candidates: pd.DataFrame,
    selections: pd.DataFrame,
    capacity_checks: pd.DataFrame,
    patch_metrics: pd.DataFrame,
    box_metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    capacity_comparisons: pd.DataFrame,
    shuffled_controls: pd.DataFrame,
    object_summary: pd.DataFrame,
    examples: pd.DataFrame,
    reconstruction_check: pd.DataFrame,
    patch_fit: _PatchFit,
    box_fit: _BoxFit,
    patch_mlp_fit: _PatchMLPFit | None,
    box_mlp_fit: _BoxMLPFit | None,
    context_fit: _BoxFit,
    fixed_patch_map: np.ndarray,
    mean_box: np.ndarray,
    test_mask: np.ndarray,
    timings: Mapping[str, float],
) -> LensArtifact:
    artifact_id = make_artifact_id(str(spec["name"]), "image_location_probe_study")
    relative_dir = Path("artifacts") / artifact_id
    tables = {
        "candidates": candidates,
        "selections": selections,
        "capacity_checks": capacity_checks,
        "patch_metrics": patch_metrics,
        "box_metrics": box_metrics,
        "comparisons": comparisons,
        "capacity_comparisons": capacity_comparisons,
        "shuffled_controls": shuffled_controls,
        "object_summary": object_summary,
        "examples": examples,
        "reconstruction_check": reconstruction_check,
        "source_rows": readouts.rows,
        "source_sites": readouts.source_sites,
        "token_metadata": readouts.token_metadata,
        "vocabulary": vocabulary,
        "evaluation_rows": readouts.rows.loc[test_mask].reset_index(drop=True),
    }
    outputs = {name: str(relative_dir / f"{name}.parquet") for name in tables}
    test_indices = np.flatnonzero(test_mask)
    patch_score_methods = ["activation_patch_head"]
    patch_score_values = [patch_fit.prediction[test_indices]]
    if patch_mlp_fit is not None:
        patch_score_methods.append("activation_patch_head_mlp")
        patch_score_values.append(patch_mlp_fit.prediction[test_indices])
    patch_score_methods.append("fixed_object_spatial_map")
    patch_score_values.append(
        np.broadcast_to(fixed_patch_map[None, :, :], patch_fit.prediction[test_indices].shape)
    )
    box_prediction_methods = ["activation_box_decoder"]
    box_prediction_values = [box_fit.prediction[test_indices]]
    if box_mlp_fit is not None:
        box_prediction_methods.append("activation_box_decoder_mlp")
        box_prediction_values.append(box_mlp_fit.prediction[test_indices])
    box_prediction_methods.extend(["per_object_training_mean", "prompt_and_scene_context"])
    box_prediction_values.extend(
        [
            np.broadcast_to(mean_box[None, :, :], box_fit.prediction[test_indices].shape),
            context_fit.prediction[test_indices],
        ]
    )
    arrays = {
        **_projection_arrays("channel", readouts.channel_projection),
        **_projection_arrays("pooled", readouts.pooled_projection),
        **_projection_arrays("tokenwise", readouts.tokenwise_projection),
        "patch_coefficients": patch_fit.coefficients,
        "patch_intercepts": patch_fit.intercepts,
        "patch_supported": patch_fit.supported.astype(np.uint8),
        "box_coefficients": box_fit.coefficients,
        "box_intercepts": box_fit.intercepts,
        "box_supported": box_fit.supported.astype(np.uint8),
        "fixed_patch_map": fixed_patch_map,
        "mean_box_center_size": mean_box,
        "test_patch_scores": np.stack(patch_score_values).astype(np.float32),
        "test_box_predictions": np.stack(box_prediction_values).astype(np.float32),
        "test_box_truth": boxes[test_indices].astype(np.float32),
        "test_patch_truth": patch_targets[test_indices].astype(np.uint8),
        "test_visible": visible[test_indices].astype(np.uint8),
    }
    if patch_mlp_fit is not None:
        arrays.update(
            {
                "patch_mlp_input_weights": patch_mlp_fit.input_weights,
                "patch_mlp_hidden_bias": patch_mlp_fit.hidden_bias,
                "patch_mlp_output_weights": patch_mlp_fit.output_weights,
                "patch_mlp_output_bias": patch_mlp_fit.output_bias,
            }
        )
    if box_mlp_fit is not None:
        arrays.update(
            {
                "box_mlp_input_weights": box_mlp_fit.input_weights,
                "box_mlp_hidden_bias": box_mlp_fit.hidden_bias,
                "box_mlp_output_weights": box_mlp_fit.output_weights,
                "box_mlp_output_bias": box_mlp_fit.output_bias,
                "box_mlp_supported": box_mlp_fit.supported.astype(np.uint8),
            }
        )
    artifact = LensArtifact(
        artifact_id=artifact_id,
        artifact_type="image_location_probe_study",
        name=str(spec["name"]),
        group_id="scene_map_probe_studies",
        scope="dataset",
        selector={
            "source_probe_artifact_id": source.artifact_id,
            "feature": source.selector.get("feature"),
        },
        method={
            "workflow": "run_image_location_probe_study",
            "schema_version": IMAGE_LOCATION_STUDY_SCHEMA_VERSION,
            "research": {
                key: spec[key]
                for key in ["question", "hypothesis_family", "intended_claim"]
                if spec.get(key)
            },
            "split": source.method.get("split"),
            "probe": spec["probe"],
            "nonlinear": spec["nonlinear"],
            "analysis": spec["analysis"],
            "representations": {
                "local_patch_head": (
                    "one shared named-object ridge head applied independently to each "
                    "compressed image token"
                ),
                "box_decoder": (
                    "one masked named-object center/size ridge head from pooled or "
                    "token-preserving scene readouts"
                ),
                "nonlinear_capacity_check": (
                    "one fixed small MLP at each validation-selected linear input; "
                    "capacity check only, not a second layer or architecture search"
                ),
            },
            "controls": {
                "patch": "per-object training mean spatial map and shuffled scenes",
                "box": ("per-object training mean, prompt/scene context, and shuffled scenes"),
                "wrong_region": "another visible object in the same episode",
            },
            "storage_contract": {
                "raw_activations": "referenced from capture and never copied",
                "compact_patch_cache": {
                    "key": compact_cache_key,
                    "cache_hit": compact_cache_hit,
                    "rebuildable": True,
                },
                "image_box_cache": {
                    "key": box_cache_key,
                    "cache_hit": box_cache_hit,
                    "rebuildable": True,
                },
                "saved_evidence": (
                    "selected activation-probe projections and parameters, test scores "
                    "and boxes, exact row and token mappings, baseline predictions, "
                    "controls, uncertainty, and examples"
                ),
            },
            "array_axes": {
                "patch_coefficients": ["object", "compressed_channel"],
                "patch_intercepts": ["object"],
                "box_coefficients": ["object", "center_size", "readout"],
                "box_intercepts": ["object", "center_size"],
                "patch_mlp_input_weights": ["compressed_channel", "hidden"],
                "patch_mlp_output_weights": ["hidden", "object"],
                "box_mlp_input_weights": ["object", "readout", "hidden"],
                "box_mlp_output_weights": ["object", "hidden", "center_size"],
                "test_patch_scores": ["method", "evaluation_row", "object", "patch"],
                "test_box_predictions": [
                    "method",
                    "evaluation_row",
                    "object",
                    "center_size",
                ],
                "test_box_truth": ["evaluation_row", "object", "xyxy"],
                "test_patch_truth": ["evaluation_row", "object", "patch"],
                "test_visible": ["evaluation_row", "object"],
            },
            "array_method_order": {
                "test_patch_scores": patch_score_methods,
                "test_box_predictions": box_prediction_methods,
            },
            "outputs": outputs,
            "timings_seconds": dict(timings),
        },
        metrics={
            "candidate_count": int(len(candidates)),
            "visible_test_object_count": int(visible[test_indices].sum()),
            "patch_metric_count": int(len(patch_metrics)),
            "box_metric_count": int(len(box_metrics)),
            "comparison_count": int(len(comparisons)),
            "total_seconds": float(timings.get("total_seconds", 0.0)),
        },
        display={
            "kind": "image_location_probe_study",
            "status": "exploratory",
            "selections": selections.to_dict("records"),
            "capacity_checks": capacity_checks.to_dict("records"),
            "comparisons": comparisons.to_dict("records"),
            "capacity_comparisons": capacity_comparisons.to_dict("records"),
        },
        tags=(
            "probe",
            "object-location",
            "image-plane",
            "patch-head",
            "box-decoder",
            "exploratory",
        ),
        source_trace_ids=tuple(sorted(readouts.rows["trace_id"].astype(str).unique())),
    )
    artifact_dir = dataset._dataset_artifact_root() / relative_dir
    artifact_dir.mkdir(parents=True, exist_ok=False)
    try:
        for name, frame in tables.items():
            frame.to_parquet(artifact_dir / f"{name}.parquet", index=False)
        saved = dataset.save_artifact(artifact, arrays=arrays)
    except BaseException:
        shutil.rmtree(artifact_dir)
        raise
    return saved


def _normalize_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(spec)
    if not normalized.get("source_probe_artifact_id"):
        raise ValueError("Image-location study requires source_probe_artifact_id")
    normalized.setdefault("name", "PI0.5 broad 1000 explicit image-plane object location study")
    normalized.setdefault("camera_name", "agentview")
    normalized.setdefault("context_columns", ["benchmark", "scene_family", "task_phase"])
    probe = dict(normalized.get("probe") or {})
    probe.setdefault("readout_dims", [32, 64])
    probe.setdefault("ridge_alphas", [0.01, 1.0, 10.0])
    probe.setdefault("baseline_ridge_alphas", [0.01, 1.0])
    probe.setdefault("min_train_episodes", 5)
    normalized["probe"] = probe
    nonlinear = dict(normalized.get("nonlinear") or {})
    nonlinear.setdefault("enabled", True)
    nonlinear.setdefault("hidden_units", 32)
    nonlinear.setdefault("patch_max_train_tokens", 100_000)
    nonlinear.setdefault("patch_batch_size", 1024)
    nonlinear.setdefault("patch_max_iter", 80)
    nonlinear.setdefault("box_max_iter", 200)
    nonlinear.setdefault("learning_rate_init", 0.001)
    nonlinear.setdefault("n_iter_no_change", 5)
    nonlinear.setdefault("random_state", 20260801)
    normalized["nonlinear"] = nonlinear
    analysis = dict(normalized.get("analysis") or {})
    analysis.setdefault("io_workers", 8)
    analysis.setdefault("bootstrap_samples", 2000)
    analysis.setdefault("shuffle_repeats", 10)
    analysis.setdefault("example_count", 12)
    normalized["analysis"] = analysis
    return normalized
