"""Classify visible object identity from its known image region."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, balanced_accuracy_score, recall_score

from vla_lens.artifacts import LensArtifact, make_artifact_id
from vla_lens.probes.object_study_common import (
    FittedClassifier,
    VisualObjectData,
    classifier_arrays,
    fit_classifier,
    fit_context_encoder,
    grouped_paired_interval,
    prepare_visual_object_data,
    row_identity,
)
from vla_lens.traces import TraceDataset

OBJECT_ROI_IDENTITY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ObjectROIIdentityStudyResult:
    """Saved known-region object-identity experiment."""

    artifact: LensArtifact | None
    candidates: pd.DataFrame
    selections: pd.DataFrame
    predictions: pd.DataFrame
    per_object: pd.DataFrame
    comparisons: pd.DataFrame
    shuffled_controls: pd.DataFrame
    reconstruction_check: pd.DataFrame
    timings: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class _IdentityFit:
    method: str
    layer: int | None
    model: str
    alpha: float
    fitted: FittedClassifier
    probabilities: np.ndarray
    record: Mapping[str, Any]


def run_object_roi_identity_study(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    *,
    save: bool = True,
) -> ObjectROIIdentityStudyResult:
    """Test whether a known object region contains identity beyond scene priors."""

    normalized = _normalize_spec(spec)
    started = time.perf_counter()
    timings: dict[str, float] = {}

    step = time.perf_counter()
    data = prepare_visual_object_data(dataset, normalized)
    split = dict(data.source.method.get("split") or {})
    instances, supported = _object_instances(
        data,
        split,
        min_train_episodes=int(normalized["probe"]["min_train_episodes"]),
    )
    features = _instance_features(data, instances)
    train_mask, selection_mask, test_mask = _instance_split_masks(instances, split)
    context_encoder = fit_context_encoder(
        instances,
        train_mask,
        [str(value) for value in normalized["context_columns"]],
    )
    context = np.column_stack(
        [
            context_encoder.transform(instances),
            np.stack(instances["bbox_normalized"].to_numpy()),
        ]
    ).astype(np.float32)
    timings["prepare_seconds"] = time.perf_counter() - step

    step = time.perf_counter()
    candidates, selected = _fit_battery(
        instances,
        features,
        context,
        data.readouts.layers,
        train_mask,
        selection_mask,
        test_mask,
        normalized["probe"],
    )
    timings["fit_seconds"] = time.perf_counter() - step

    step = time.perf_counter()
    predictions = _prediction_table(selected, instances, test_mask)
    per_object = _per_object_table(
        selected,
        instances,
        test_mask,
        data.targets.vocabulary,
        bootstrap_samples=int(normalized["analysis"]["bootstrap_samples"]),
    )
    comparisons = _comparison_table(
        selected,
        instances,
        test_mask,
        bootstrap_samples=int(normalized["analysis"]["bootstrap_samples"]),
    )
    shuffled_controls = _shuffled_controls(
        selected["object_roi"],
        features,
        instances,
        train_mask,
        selection_mask,
        test_mask,
        normalized["probe"],
        repeats=int(normalized["analysis"]["shuffle_repeats"]),
    )
    reconstruction_check = _reconstruction_check(selected)
    selections = pd.DataFrame.from_records([dict(value.record) for value in selected.values()])
    timings["evaluate_seconds"] = time.perf_counter() - step
    timings["total_seconds"] = time.perf_counter() - started

    artifact = (
        _save_study(
            dataset,
            normalized,
            data,
            instances,
            supported,
            candidates,
            selections,
            predictions,
            per_object,
            comparisons,
            shuffled_controls,
            reconstruction_check,
            selected,
            context_encoder,
            timings,
        )
        if save
        else None
    )
    return ObjectROIIdentityStudyResult(
        artifact,
        candidates,
        selections,
        predictions,
        per_object,
        comparisons,
        shuffled_controls,
        reconstruction_check,
        timings,
    )


def _object_instances(
    data: VisualObjectData,
    split: Mapping[str, Any],
    *,
    min_train_episodes: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rows = data.readouts.rows
    split_column = str(split["column"])
    train_rows = rows[split_column].astype(str).to_numpy() == str(split["train_value"])
    supported = np.zeros(len(data.targets.vocabulary), dtype=bool)
    for object_index in range(len(supported)):
        trace_count = (
            rows.loc[train_rows & data.visible[:, object_index], "trace_id"].astype(str).nunique()
        )
        supported[object_index] = trace_count >= int(min_train_episodes)

    image_width = float(data.readouts.token_metadata["pixel_x1"].max())
    image_height = float(data.readouts.token_metadata["pixel_y1"].max())
    scale = np.array([image_width, image_height, image_width, image_height])
    records: list[dict[str, Any]] = []
    for row_index, source in data.readouts.rows.iterrows():
        visible_indices = np.flatnonzero(data.visible[row_index] & supported)
        union = data.patch_targets[row_index].any(axis=0)
        background = np.flatnonzero(~union)
        for object_index in visible_indices:
            target_patches = np.flatnonzero(data.patch_targets[row_index, object_index])
            wrong_candidates = [
                int(value) for value in visible_indices if int(value) != int(object_index)
            ]
            wrong_index = wrong_candidates[0] if wrong_candidates else None
            wrong_patches = (
                np.flatnonzero(data.patch_targets[row_index, wrong_index])
                if wrong_index is not None
                else background
            )
            if not len(background):
                background = np.flatnonzero(~data.patch_targets[row_index, object_index])
            records.append(
                {
                    **row_identity(source),
                    "source_row_index": int(row_index),
                    "split": str(source[split_column]),
                    "object_index": int(object_index),
                    "object_name": str(data.targets.vocabulary[object_index]),
                    "wrong_object_index": wrong_index,
                    "wrong_object_name": (
                        str(data.targets.vocabulary[wrong_index])
                        if wrong_index is not None
                        else None
                    ),
                    "bbox_xyxy": data.boxes_px[row_index, object_index].tolist(),
                    "bbox_normalized": (data.boxes_px[row_index, object_index] / scale)
                    .astype(np.float32)
                    .tolist(),
                    "roi_patch_indices": target_patches.astype(int).tolist(),
                    "wrong_roi_patch_indices": wrong_patches.astype(int).tolist(),
                    "background_patch_indices": background.astype(int).tolist(),
                }
            )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("No visible supported object instances were found")
    return frame, supported


def _instance_features(
    data: VisualObjectData,
    instances: pd.DataFrame,
) -> dict[str, np.ndarray]:
    shape = (len(instances), len(data.readouts.layers), data.compact.shape[-1])
    result = {
        method: np.zeros(shape, dtype=np.float32)
        for method in ("object_roi", "whole_image", "wrong_object_roi", "background_roi")
    }
    for instance_index, item in enumerate(instances.itertuples()):
        values = data.compact[int(item.source_row_index)]
        masks = {
            "object_roi": np.asarray(item.roi_patch_indices, dtype=int),
            "wrong_object_roi": np.asarray(item.wrong_roi_patch_indices, dtype=int),
            "background_roi": np.asarray(item.background_patch_indices, dtype=int),
        }
        result["whole_image"][instance_index] = values.mean(axis=1)
        for method, indices in masks.items():
            if not len(indices):
                result[method][instance_index] = values.mean(axis=1)
            else:
                result[method][instance_index] = values[:, indices].mean(axis=1)
    return result


def _instance_split_masks(
    instances: pd.DataFrame, split: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = instances["split"].astype(str).to_numpy()
    return tuple(values == str(split[f"{name}_value"]) for name in ("train", "selection", "test"))  # type: ignore[return-value]


def _fit_battery(
    instances: pd.DataFrame,
    features: Mapping[str, np.ndarray],
    context: np.ndarray,
    layers: Sequence[int],
    train_mask: np.ndarray,
    selection_mask: np.ndarray,
    test_mask: np.ndarray,
    settings: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, _IdentityFit]]:
    labels = instances["object_index"].to_numpy(dtype=int)
    records: list[dict[str, Any]] = []
    selected: dict[str, _IdentityFit] = {}
    methods = list(features) + ["task_scene_box"]
    requested_layers = {int(value) for value in settings["layers"]}
    missing_layers = requested_layers - {int(value) for value in layers}
    if missing_layers:
        raise ValueError(
            f"ROI identity source is missing requested layers {sorted(missing_layers)}"
        )
    for method in methods:
        layer_values = (
            [(None, context)]
            if method == "task_scene_box"
            else [
                (int(layer), features[method][:, index])
                for index, layer in enumerate(layers)
                if int(layer) in requested_layers
            ]
        )
        for layer, values in layer_values:
            for model in settings["models"]:
                for alpha in settings["alphas"]:
                    fitted = fit_classifier(
                        values[train_mask],
                        labels[train_mask],
                        model=str(model),
                        alpha=float(alpha),
                        hidden_units=int(settings["mlp_hidden_units"]),
                        max_iter=int(settings["max_iter"]),
                        random_state=int(settings["random_state"]),
                    )
                    probabilities = fitted.predict_proba(values)
                    record = {
                        "method": method,
                        "layer": layer,
                        "model": str(model),
                        "alpha": float(alpha),
                        "feature_dim": int(values.shape[1]),
                        "training_instance_count": int(train_mask.sum()),
                        "class_count": int(len(fitted.classes)),
                        "iterations": fitted.n_iter,
                        "converged": fitted.converged,
                        **_classification_metrics(
                            labels[selection_mask],
                            probabilities[selection_mask],
                            fitted.classes,
                            "selection",
                        ),
                    }
                    records.append(record)
                    candidate = _IdentityFit(
                        method, layer, str(model), float(alpha), fitted, probabilities, record
                    )
                    current = selected.get(method)
                    if current is None or _selection_score(record) > _selection_score(
                        current.record
                    ):
                        selected[method] = candidate
    final_selected: dict[str, _IdentityFit] = {}
    for method, fitted in selected.items():
        record = {
            **dict(fitted.record),
            **_classification_metrics(
                labels[test_mask],
                fitted.probabilities[test_mask],
                fitted.fitted.classes,
                "test",
            ),
        }
        final_selected[method] = _IdentityFit(
            fitted.method,
            fitted.layer,
            fitted.model,
            fitted.alpha,
            fitted.fitted,
            fitted.probabilities,
            record,
        )
    return pd.DataFrame.from_records(records), final_selected


def _classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    prediction = classes[np.argmax(probabilities, axis=1)]
    recalls = recall_score(labels, prediction, labels=classes, average=None, zero_division=0)
    average_precisions = []
    for class_index, class_value in enumerate(classes):
        truth = labels == class_value
        if truth.any() and (~truth).any():
            average_precisions.append(
                float(average_precision_score(truth.astype(int), probabilities[:, class_index]))
            )
    return {
        f"{prefix}_macro_balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        f"{prefix}_macro_average_precision": (
            float(np.mean(average_precisions)) if average_precisions else float("nan")
        ),
        f"{prefix}_macro_recall": float(np.mean(recalls)),
    }


def _selection_score(record: Mapping[str, Any]) -> tuple[float, float]:
    return (
        float(record["selection_macro_balanced_accuracy"]),
        float(record["selection_macro_average_precision"]),
    )


def _prediction_table(
    selected: Mapping[str, _IdentityFit],
    instances: pd.DataFrame,
    test_mask: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    indices = np.flatnonzero(test_mask)
    for method, fitted in selected.items():
        for instance_index in indices:
            probabilities = fitted.probabilities[instance_index]
            prediction_index = int(np.argmax(probabilities))
            source = instances.iloc[instance_index]
            records.append(
                {
                    "method": method,
                    "layer": fitted.layer,
                    "model": fitted.model,
                    "instance_index": int(instance_index),
                    **{
                        key: source.get(key)
                        for key in [
                            "trace_id",
                            "task_key",
                            "instruction_key",
                            "object_index",
                            "object_name",
                            "bbox_xyxy",
                            "roi_patch_indices",
                            "wrong_roi_patch_indices",
                            "background_patch_indices",
                        ]
                    },
                    "predicted_object_index": int(fitted.fitted.classes[prediction_index]),
                    "confidence": float(probabilities[prediction_index]),
                    "correct": bool(
                        fitted.fitted.classes[prediction_index] == int(source["object_index"])
                    ),
                    "class_order": fitted.fitted.classes.astype(int).tolist(),
                    "probabilities": probabilities.astype(np.float32).tolist(),
                }
            )
    return pd.DataFrame.from_records(records)


def _per_object_table(
    selected: Mapping[str, _IdentityFit],
    instances: pd.DataFrame,
    test_mask: np.ndarray,
    vocabulary: Sequence[str],
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    labels = instances["object_index"].to_numpy(dtype=int)
    records: list[dict[str, Any]] = []
    for method, fitted in selected.items():
        prediction = fitted.fitted.classes[np.argmax(fitted.probabilities, axis=1)]
        for object_index in fitted.fitted.classes.astype(int):
            mask = test_mask & (labels == object_index)
            correct = (prediction[mask] == object_index).astype(float)
            uncertainty = grouped_paired_interval(
                correct,
                np.zeros(len(correct), dtype=float),
                instances.loc[mask, "trace_id"].to_numpy(),
                bootstrap_samples=bootstrap_samples,
                seed=20260815 + len(records),
            )
            records.append(
                {
                    "method": method,
                    "object_index": int(object_index),
                    "object_name": str(vocabulary[object_index]),
                    "test_count": int(mask.sum()),
                    "recall": uncertainty["mean"],
                    "recall_ci95_low": uncertainty["ci95_low"],
                    "recall_ci95_high": uncertainty["ci95_high"],
                    "episode_count": uncertainty["group_count"],
                }
            )
    return pd.DataFrame.from_records(records)


def _comparison_table(
    selected: Mapping[str, _IdentityFit],
    instances: pd.DataFrame,
    test_mask: np.ndarray,
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    labels = instances.loc[test_mask, "object_index"].to_numpy(dtype=int)
    candidate = selected["object_roi"]
    candidate_correct = (
        candidate.fitted.classes[np.argmax(candidate.probabilities[test_mask], axis=1)] == labels
    ).astype(float)
    records: list[dict[str, Any]] = []
    for baseline_name, baseline in selected.items():
        if baseline_name == "object_roi":
            continue
        baseline_correct = (
            baseline.fitted.classes[np.argmax(baseline.probabilities[test_mask], axis=1)] == labels
        ).astype(float)
        for unit, groups in [
            ("benchmark_task", instances.loc[test_mask, "task_key"].to_numpy()),
            ("instruction", instances.loc[test_mask, "instruction_key"].to_numpy()),
            ("object", instances.loc[test_mask, "object_index"].to_numpy()),
        ]:
            records.append(
                {
                    "candidate": "object_roi",
                    "baseline": baseline_name,
                    "metric": "correct_rate_improvement",
                    "unit": unit,
                    **grouped_paired_interval(
                        candidate_correct,
                        baseline_correct,
                        groups,
                        bootstrap_samples=bootstrap_samples,
                        seed=20260810 + len(records),
                    ),
                }
            )
    return pd.DataFrame.from_records(records)


def _shuffled_controls(
    selected: _IdentityFit,
    features: Mapping[str, np.ndarray],
    instances: pd.DataFrame,
    train_mask: np.ndarray,
    selection_mask: np.ndarray,
    test_mask: np.ndarray,
    settings: Mapping[str, Any],
    *,
    repeats: int,
) -> pd.DataFrame:
    labels = instances["object_index"].to_numpy(dtype=int)
    layer_index = int(np.flatnonzero(np.asarray(settings["layers"]) == selected.layer)[0])
    values = features["object_roi"][:, layer_index]
    train_indices = np.flatnonzero(train_mask)
    records = []
    for repeat in range(max(0, int(repeats))):
        rng = np.random.default_rng(20260820 + repeat)
        shuffled = labels.copy()
        shuffled[train_indices] = shuffled[rng.permutation(train_indices)]
        fitted = fit_classifier(
            values[train_mask],
            shuffled[train_mask],
            model=selected.model,
            alpha=selected.alpha,
            hidden_units=int(settings["mlp_hidden_units"]),
            max_iter=int(settings["max_iter"]),
            random_state=int(settings["random_state"]) + repeat + 1,
        )
        probabilities = fitted.predict_proba(values)
        records.append(
            {
                "repeat": repeat,
                "layer": selected.layer,
                "model": selected.model,
                **_classification_metrics(
                    labels[selection_mask],
                    probabilities[selection_mask],
                    fitted.classes,
                    "selection",
                ),
                **_classification_metrics(
                    labels[test_mask], probabilities[test_mask], fitted.classes, "test"
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _reconstruction_check(selected: Mapping[str, _IdentityFit]) -> pd.DataFrame:
    records = []
    for method, value in selected.items():
        rebuilt = value.fitted.predict_proba(_fit_input_placeholder(value))
        records.append(
            {
                "method": method,
                "note": "checked during fit; saved parameters are NumPy replayable",
                "class_count": int(rebuilt.shape[1]),
            }
        )
    return pd.DataFrame.from_records(records)


def _fit_input_placeholder(value: _IdentityFit) -> np.ndarray:
    # A zero raw input checks every stored array has a compatible replay shape.
    return np.zeros((1, len(value.fitted.feature_mean)), dtype=np.float64)


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
    instances: pd.DataFrame,
    supported: np.ndarray,
    candidates: pd.DataFrame,
    selections: pd.DataFrame,
    predictions: pd.DataFrame,
    per_object: pd.DataFrame,
    comparisons: pd.DataFrame,
    shuffled_controls: pd.DataFrame,
    reconstruction_check: pd.DataFrame,
    selected: Mapping[str, _IdentityFit],
    context_encoder: Any,
    timings: Mapping[str, float],
) -> LensArtifact:
    artifact_id = make_artifact_id(str(spec["name"]), "object_roi_identity_study")
    relative_dir = Path("artifacts") / artifact_id
    context_categories = pd.DataFrame.from_records(
        [
            {"column": column, "categories": categories.astype(str).tolist()}
            for column, categories in zip(
                context_encoder.columns, context_encoder.encoder.categories_, strict=True
            )
        ]
    )
    tables = {
        "instances": instances,
        "candidates": candidates,
        "selections": selections,
        "predictions": predictions,
        "per_object": per_object,
        "comparisons": comparisons,
        "shuffled_controls": shuffled_controls,
        "reconstruction_check": reconstruction_check,
        "source_rows": data.readouts.rows,
        "source_sites": data.readouts.source_sites,
        "token_metadata": data.readouts.token_metadata,
        "vocabulary": data.vocabulary,
        "context_categories": context_categories,
    }
    outputs = {name: str(relative_dir / f"{name}.parquet") for name in tables}
    arrays = {**_projection_arrays(data), "supported_objects": supported.astype(np.uint8)}
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
        artifact_type="object_roi_identity_study",
        name=str(spec["name"]),
        group_id="scene_map_probe_studies",
        scope="dataset",
        selector={
            "source_probe_artifact_id": data.source.artifact_id,
            "feature": data.source.selector.get("feature"),
            "cohort": "initial visible supported object instances",
        },
        method={
            "workflow": "run_object_roi_identity_study",
            "schema_version": OBJECT_ROI_IDENTITY_SCHEMA_VERSION,
            "research": {
                key: spec[key]
                for key in ["question", "hypothesis_family", "intended_claim"]
                if spec.get(key)
            },
            "split": data.source.method.get("split"),
            "probe": spec["probe"],
            "analysis": spec["analysis"],
            "models": model_contract,
            "controls": [
                "whole_image",
                "task_scene_box",
                "wrong_object_roi",
                "background_roi",
                "shuffled_training_labels",
            ],
            "outputs": outputs,
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
                    "exact object rows and patch membership, train-fitted projection, "
                    "selected model parameters, test predictions, controls, and grouped "
                    "intervals"
                ),
            },
            "timings_seconds": dict(timings),
        },
        metrics={
            "instance_count": int(len(instances)),
            "supported_object_count": int(supported.sum()),
            "candidate_count": int(len(candidates)),
            "test_prediction_count": int(len(predictions)),
            "total_seconds": float(timings["total_seconds"]),
        },
        display={
            "kind": "object_roi_identity_study",
            "status": "exploratory",
            "selections": json.loads(selections.to_json(orient="records")),
            "comparisons": json.loads(comparisons.to_json(orient="records")),
        },
        tags=("probe", "object-identity", "object-conditioned", "visual-tokens", "exploratory"),
        source_trace_ids=tuple(sorted(instances["trace_id"].astype(str).unique())),
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
        raise ValueError("ROI identity study requires source_probe_artifact_id")
    normalized.setdefault("name", "PI0.5 known-region visible-object identity study")
    normalized.setdefault("camera_name", "agentview")
    normalized.setdefault("context_columns", ["benchmark", "scene_family", "task_phase", "prompt"])
    probe = dict(normalized.get("probe") or {})
    probe.setdefault("layers", [0, 4, 8, 12, 17])
    probe.setdefault("models", ["linear", "mlp"])
    probe.setdefault("alphas", [0.0001])
    probe.setdefault("mlp_hidden_units", 64)
    probe.setdefault("max_iter", 300)
    probe.setdefault("min_train_episodes", 5)
    probe.setdefault("random_state", 20260810)
    normalized["probe"] = probe
    analysis = dict(normalized.get("analysis") or {})
    analysis.setdefault("io_workers", 8)
    analysis.setdefault("bootstrap_samples", 2000)
    analysis.setdefault("shuffle_repeats", 10)
    normalized["analysis"] = analysis
    return normalized
