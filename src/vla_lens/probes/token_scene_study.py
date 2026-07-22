"""Matched pooled, tokenwise, layer, and model-capacity scene-object study."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from vla_lens.artifacts import LensArtifact, make_artifact_id
from vla_lens.probes.scene_map_study import SceneMapTargets, scene_map_target_table
from vla_lens.probes.structured_scene_models import (
    FittedMLP,
    FittedSceneRepresentation,
    SceneLinearDecoder,
    SceneMLPDecoder,
    fit_structured_scene_representations,
)
from vla_lens.probes.token_representations import (
    LayerTokenReadouts,
    ProjectionState,
    build_layer_token_readouts,
)
from vla_lens.traces import TraceDataset

TOKEN_SCENE_STUDY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TokenSceneStudyResult:
    """Saved token-preserving scene study and its review tables."""

    artifact: LensArtifact | None
    candidates: pd.DataFrame
    selections: pd.DataFrame
    predictions: pd.DataFrame
    object_results: pd.DataFrame
    paired_comparisons: pd.DataFrame
    layer_weights: pd.DataFrame
    token_importance: pd.DataFrame
    examples: pd.DataFrame
    vocabulary: pd.DataFrame
    timings: Mapping[str, float]


def run_token_scene_probe_study(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    *,
    save: bool = True,
) -> TokenSceneStudyResult:
    """Compare pooled/token-preserving linear and MLP decoders on identical rows."""

    normalized = _normalize_spec(spec)
    started = time.perf_counter()
    timings: dict[str, float] = {}
    step_started = time.perf_counter()
    readouts = build_layer_token_readouts(
        dataset,
        normalized["feature"],
        normalized["split"],
        readout_dim=max(int(value) for value in normalized["probe"]["readout_dims"]),
        token_channel_dim=int(normalized["probe"]["token_channel_dim"]),
        channel_sample_count=int(normalized["probe"]["channel_sample_count"]),
        projection_fit_rows=int(normalized["probe"]["projection_fit_rows"]),
        io_workers=int(normalized["probe"]["io_workers"]),
        limit_episodes=normalized.get("limit_episodes"),
        cache=bool(normalized["probe"].get("cache", True)),
    )
    timings["prepare_readouts_seconds"] = time.perf_counter() - step_started

    step_started = time.perf_counter()
    targets, vocabulary = scene_map_target_table(dataset, readouts.rows, cache=True)
    candidates, selected = fit_structured_scene_representations(
        {"pooled": readouts.pooled, "tokenwise": readouts.tokenwise},
        readouts.rows,
        targets,
        readouts.layers,
        normalized["split"],
        readout_dims=[int(value) for value in normalized["probe"]["readout_dims"]],
        ridge_alphas=[float(value) for value in normalized["probe"]["ridge_alphas"]],
        min_train_episodes=int(normalized["probe"]["min_train_episodes"]),
        mixture_iterations=int(normalized["probe"]["mixture_iterations"]),
        mixture_regularization=float(normalized["probe"]["mixture_regularization"]),
        models=[str(value) for value in normalized["probe"]["models"]],
        mlp_hidden_layer_sizes=[
            int(value) for value in normalized["probe"]["mlp_hidden_layer_sizes"]
        ],
        mlp_max_iter=int(normalized["probe"]["mlp_max_iter"]),
        mlp_workers=int(normalized["probe"]["mlp_workers"]),
    )
    timings["fit_seconds"] = time.perf_counter() - step_started

    selections = pd.DataFrame.from_records([dict(value.record) for value in selected])
    predictions = _prediction_table(
        selected, readouts.rows, targets, normalized["split"]
    )
    object_results = _object_result_table(
        selected, readouts.rows, targets, normalized["split"]
    )
    paired_comparisons = _paired_comparison_table(
        selected,
        readouts.rows,
        targets,
        normalized["split"],
        bootstrap_samples=int(normalized["probe"]["bootstrap_samples"]),
    )
    layer_weights = _layer_weight_table(selected, readouts.layers)
    token_importance = _token_importance_table(
        selected, readouts, targets.vocabulary
    )
    examples = _example_table(selected, readouts.rows, targets, normalized["split"])
    decoder_parameters = _decoder_parameter_table(selected)
    timings["total_seconds"] = time.perf_counter() - started
    artifact = (
        _save_study(
            dataset,
            normalized,
            readouts,
            candidates,
            selections,
            predictions,
            object_results,
            paired_comparisons,
            layer_weights,
            token_importance,
            examples,
            vocabulary,
            decoder_parameters,
            timings,
        )
        if save
        else None
    )
    return TokenSceneStudyResult(
        artifact=artifact,
        candidates=candidates,
        selections=selections,
        predictions=predictions,
        object_results=object_results,
        paired_comparisons=paired_comparisons,
        layer_weights=layer_weights,
        token_importance=token_importance,
        examples=examples,
        vocabulary=vocabulary,
        timings=timings,
    )


def _prediction_table(
    selected: Sequence[FittedSceneRepresentation],
    rows: pd.DataFrame,
    targets: SceneMapTargets,
    split: Mapping[str, Any],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    split_column = str(split["column"])
    split_values = {
        "selection": str(split["selection_value"]),
        "test": str(split["test_value"]),
    }
    for fitted in selected:
        target = str(fitted.record["target"])
        truth = targets.presence if target == "scene_identity" else targets.position
        for split_name, split_value in split_values.items():
            indices = np.flatnonzero(
                rows[split_column].astype(str).to_numpy() == split_value
            )
            for index in indices:
                source = rows.iloc[index]
                records.append(
                    {
                        "variant": _variant(fitted),
                        "representation": fitted.record["representation"],
                        "structure": fitted.record["structure"],
                        "target": target,
                        "split": split_name,
                        "trace_id": str(source["trace_id"]),
                        "episode_id": source.get("episode_id"),
                        "benchmark": source.get("benchmark"),
                        "task_id": source.get("task_id"),
                        "prompt": source.get("prompt"),
                        "timestep": int(source["timestep"]),
                        "policy_call_index": _optional_int(
                            source.get("policy_call_index")
                        ),
                        "readout_dim": int(fitted.record["readout_dim"]),
                        "ridge_alpha": float(fitted.record["ridge_alpha"]),
                        "threshold": fitted.record.get("selection_threshold"),
                        "supported": fitted.decoder.supported.astype(np.uint8).tolist(),
                        "truth": np.asarray(truth[index], dtype=np.float32)
                        .reshape(-1)
                        .tolist(),
                        "prediction": np.asarray(
                            fitted.prediction[index], dtype=np.float32
                        )
                        .reshape(-1)
                        .tolist(),
                    }
                )
    return pd.DataFrame.from_records(records)


def _object_result_table(
    selected: Sequence[FittedSceneRepresentation],
    rows: pd.DataFrame,
    targets: SceneMapTargets,
    split: Mapping[str, Any],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    split_column = str(split["column"])
    split_values = {
        "selection": str(split["selection_value"]),
        "test": str(split["test_value"]),
    }
    moved = np.linalg.norm(targets.position - targets.initial_position, axis=2)
    for fitted in selected:
        target = str(fitted.record["target"])
        for split_name, split_value in split_values.items():
            mask = rows[split_column].astype(str).to_numpy() == split_value
            episodes = rows.loc[mask, "trace_id"].astype(str).to_numpy()
            for object_index, object_name in enumerate(targets.vocabulary):
                supported = bool(fitted.decoder.supported[object_index])
                record: dict[str, Any] = {
                    "variant": _variant(fitted),
                    "representation": fitted.record["representation"],
                    "structure": fitted.record["structure"],
                    "target": target,
                    "split": split_name,
                    "object_index": object_index,
                    "object_name": object_name,
                    "supported": supported,
                }
                if target == "scene_identity":
                    truth = targets.presence[mask, object_index].astype(bool)
                    scores = fitted.prediction[mask, object_index]
                    threshold = float(
                        fitted.record.get("selection_threshold", 0.5)
                    )
                    predicted = scores >= threshold
                    tp = int(np.logical_and(truth, predicted).sum())
                    fp = int(np.logical_and(~truth, predicted).sum())
                    fn = int(np.logical_and(truth, ~predicted).sum())
                    record.update(
                        {
                            "positive_count": int(truth.sum()),
                            "precision": tp / max(1, tp + fp),
                            "recall": tp / max(1, tp + fn),
                            "f1": 2 * tp / max(1, 2 * tp + fp + fn),
                            "average_precision": (
                                float(average_precision_score(truth.astype(int), scores))
                                if np.unique(truth).size > 1 and supported
                                else float("nan")
                            ),
                        }
                    )
                else:
                    truth = targets.position[mask, object_index]
                    predicted = fitted.prediction[mask, object_index]
                    finite = np.isfinite(truth).all(axis=1) & supported
                    errors = np.linalg.norm(predicted - truth, axis=1)
                    coordinate = np.abs(predicted - truth)
                    record.update(
                        {
                            "position_count": int(finite.sum()),
                            "error_m": _episode_mean(errors, episodes, finite),
                            "x_mae_m": _episode_mean(coordinate[:, 0], episodes, finite),
                            "y_mae_m": _episode_mean(coordinate[:, 1], episodes, finite),
                            "z_mae_m": _episode_mean(coordinate[:, 2], episodes, finite),
                            "moved_10cm_error_m": _episode_mean(
                                errors,
                                episodes,
                                finite & (moved[mask, object_index] > 0.10),
                            ),
                        }
                    )
                records.append(record)
    return pd.DataFrame.from_records(records)


def _layer_weight_table(
    selected: Sequence[FittedSceneRepresentation], layers: Sequence[int]
) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "variant": _variant(fitted),
                "representation": fitted.record["representation"],
                "structure": fitted.record["structure"],
                "target": fitted.record["target"],
                "layer": int(layer),
                "weight": float(fitted.layer_weights[index]),
            }
            for fitted in selected
            for index, layer in enumerate(layers)
        ]
    )


def _paired_comparison_table(
    selected: Sequence[FittedSceneRepresentation],
    rows: pd.DataFrame,
    targets: SceneMapTargets,
    split: Mapping[str, Any],
    *,
    bootstrap_samples: int,
) -> pd.DataFrame:
    """Compare matched variants with equal-weight episode and task resampling."""

    by_key = {
        (_variant(fitted), str(fitted.record["target"])): fitted
        for fitted in selected
    }
    comparisons = [
        ("tokenwise__single_layer", "pooled__single_layer"),
        ("pooled__learned_layer_mix", "pooled__single_layer"),
        ("tokenwise__learned_layer_mix", "tokenwise__single_layer"),
        ("tokenwise__learned_layer_mix", "pooled__single_layer"),
    ]
    models = sorted({str(fitted.record["model"]) for fitted in selected})
    test_mask = rows[str(split["column"])].astype(str).to_numpy() == str(
        split["test_value"]
    )
    records: list[dict[str, Any]] = []
    for comparison_index, (candidate_name, reference_name) in enumerate(comparisons):
        for model in models:
            candidate_variant = f"{candidate_name}__{model}"
            reference_variant = f"{reference_name}__{model}"
            for target in ["scene_identity", "object_position"]:
                candidate = by_key.get((candidate_variant, target))
                reference = by_key.get((reference_variant, target))
                if candidate is None or reference is None:
                    continue
                candidate_score = _row_score(candidate, targets)
                reference_score = _row_score(reference, targets)
                improvement = (
                    candidate_score - reference_score
                    if target == "scene_identity"
                    else reference_score - candidate_score
                )
                for unit_index, (unit, groups) in enumerate(
                    [
                        ("episode", rows["trace_id"].astype(str).to_numpy()),
                        (
                            "task",
                            rows.get("task_id", rows["trace_id"])
                            .fillna("")
                            .astype(str)
                            .to_numpy(),
                        ),
                    ]
                ):
                    summary = _paired_bootstrap_summary(
                        improvement[test_mask],
                        groups[test_mask],
                        bootstrap_samples=bootstrap_samples,
                        seed=20260719 + comparison_index * 100 + unit_index,
                    )
                    records.append(
                        {
                            "candidate": candidate_variant,
                            "reference": reference_variant,
                            "model": model,
                            "target": target,
                            "unit": unit,
                            "metric": (
                                "scene_jaccard_improvement"
                                if target == "scene_identity"
                                else "error_reduction_m"
                            ),
                            **summary,
                        }
                    )
    return pd.DataFrame.from_records(records)


def _row_score(
    fitted: FittedSceneRepresentation, targets: SceneMapTargets
) -> np.ndarray:
    if fitted.record["target"] == "scene_identity":
        supported = fitted.decoder.supported
        truth = targets.presence[:, supported].astype(bool)
        predicted = fitted.prediction[:, supported] >= float(
            fitted.record.get("selection_threshold", 0.5)
        )
        intersection = np.logical_and(truth, predicted).sum(axis=1)
        union = np.logical_or(truth, predicted).sum(axis=1)
        return intersection / np.maximum(1, union)
    truth = targets.position
    available = np.isfinite(truth).all(axis=2) & fitted.decoder.supported[None, :]
    errors = np.linalg.norm(fitted.prediction - truth, axis=2)
    return np.asarray(
        [
            float(np.mean(errors[index, available[index]]))
            if available[index].any()
            else float("nan")
            for index in range(len(truth))
        ]
    )


def _paired_bootstrap_summary(
    improvement: np.ndarray,
    groups: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {"group": np.asarray(groups).astype(str), "improvement": improvement}
    )
    frame = frame.loc[np.isfinite(frame["improvement"])]
    grouped = frame.groupby("group", sort=True)["improvement"].mean().to_numpy()
    if not len(grouped):
        return {
            "unit_count": 0,
            "mean_improvement": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "probability_improvement": float("nan"),
        }
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        grouped,
        size=(max(1, int(bootstrap_samples)), len(grouped)),
        replace=True,
    ).mean(axis=1)
    probability = float(np.mean(draws > 0.0))
    return {
        "unit_count": int(len(grouped)),
        "mean_improvement": float(grouped.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "probability_improvement": probability,
    }


def _token_importance_table(
    selected: Sequence[FittedSceneRepresentation],
    readouts: LayerTokenReadouts,
    vocabulary: Sequence[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    projection = readouts.tokenwise_projection
    token_metadata = readouts.token_metadata.reset_index(drop=True)
    for fitted in selected:
        if fitted.record["representation"] != "tokenwise":
            continue
        dim = int(fitted.record["readout_dim"])
        components = projection.components[:dim]
        inverse_scale = 1.0 / projection.input_scale
        coefficients = fitted.decoder.coefficients
        if fitted.record["target"] == "scene_identity":
            flat = coefficients @ components
            flat *= inverse_scale[None, :]
            importance = np.linalg.norm(
                flat.reshape(len(vocabulary), readouts.token_count, readouts.channel_dim),
                axis=2,
            )
        else:
            flat = np.einsum("ocd,df->ocf", coefficients, components, optimize=True)
            flat *= inverse_scale[None, None, :]
            importance = np.linalg.norm(
                flat.reshape(
                    len(vocabulary), 3, readouts.token_count, readouts.channel_dim
                ),
                axis=(1, 3),
            )
        for object_index, object_name in enumerate(vocabulary):
            values = np.nan_to_num(importance[object_index], nan=0.0)
            weighted, fractions = _weighted_token_importance(
                values,
                fitted.layer_weights,
            )
            for layer_index, layer in enumerate(readouts.layers):
                layer_weight = float(fitted.layer_weights[layer_index])
                for token_position, value in enumerate(weighted[layer_index]):
                    metadata = (
                        dict(token_metadata.iloc[token_position])
                        if token_position < len(token_metadata)
                        else {}
                    )
                    records.append(
                        {
                            "variant": _variant(fitted),
                            "target": fitted.record["target"],
                            "object_index": object_index,
                            "object_name": object_name,
                            "layer": int(layer),
                            "layer_weight": layer_weight,
                            "token_position": token_position,
                            "token_index": _optional_int(
                                metadata.get("token_index", token_position)
                            ),
                            "token_kind": metadata.get("token_kind"),
                            "stream_id": metadata.get("stream_id"),
                            "action_horizon_index": _optional_int(
                                metadata.get("action_horizon_index")
                            ),
                            "patch_row": _optional_int(metadata.get("patch_row")),
                            "patch_col": _optional_int(metadata.get("patch_col")),
                            "importance": float(value),
                            "within_object_fraction": float(
                                fractions[layer_index, token_position]
                            ),
                        }
                    )
    return pd.DataFrame.from_records(records)


def _weighted_token_importance(
    values: np.ndarray,
    layer_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weighted = np.abs(np.asarray(layer_weights, dtype=np.float64))[:, None] * np.asarray(
        values,
        dtype=np.float64,
    )[None, :]
    total = float(weighted.sum())
    fractions = weighted / total if total > 0 else np.zeros_like(weighted)
    return weighted, fractions


def _example_table(
    selected: Sequence[FittedSceneRepresentation],
    rows: pd.DataFrame,
    targets: SceneMapTargets,
    split: Mapping[str, Any],
) -> pd.DataFrame:
    test = rows[str(split["column"])].astype(str).to_numpy() == str(split["test_value"])
    indices = np.flatnonzero(test)
    records: list[dict[str, Any]] = []
    for fitted in selected:
        target = str(fitted.record["target"])
        scores: list[tuple[float, int]] = []
        for index in indices:
            if target == "scene_identity":
                truth = targets.presence[index].astype(bool)
                predicted = fitted.prediction[index] >= float(
                    fitted.record.get("selection_threshold", 0.5)
                )
                score = float(np.logical_xor(truth, predicted).sum())
            else:
                truth = targets.position[index]
                available = (
                    np.isfinite(truth).all(axis=1) & fitted.decoder.supported
                )
                score = (
                    float(
                        np.linalg.norm(
                            fitted.prediction[index, available] - truth[available],
                            axis=1,
                        ).mean()
                    )
                    if available.any()
                    else float("nan")
                )
            if np.isfinite(score):
                scores.append((score, int(index)))
        if not scores:
            continue
        scores.sort()
        for kind, (score, index) in [("best", scores[0]), ("worst", scores[-1])]:
            source = rows.iloc[index]
            records.append(
                {
                    "variant": _variant(fitted),
                    "target": target,
                    "example_kind": kind,
                    "score": score,
                    "trace_id": str(source["trace_id"]),
                    "timestep": int(source["timestep"]),
                    "policy_call_index": _optional_int(
                        source.get("policy_call_index")
                    ),
                    "prompt": source.get("prompt"),
                    "truth": np.asarray(
                        targets.presence[index]
                        if target == "scene_identity"
                        else targets.position[index],
                        dtype=np.float32,
                    )
                    .reshape(-1)
                    .tolist(),
                    "prediction": np.asarray(
                        fitted.prediction[index], dtype=np.float32
                    )
                    .reshape(-1)
                    .tolist(),
                }
            )
    return pd.DataFrame.from_records(records)


def _decoder_parameter_table(
    selected: Sequence[FittedSceneRepresentation],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for fitted in selected:
        decoder = fitted.decoder
        common = {
            "variant": _variant(fitted),
            "target": fitted.record["target"],
            "model": fitted.record["model"],
            "supported": decoder.supported.astype(np.uint8).tolist(),
        }
        if isinstance(decoder, SceneLinearDecoder):
            records.append(
                {
                    **common,
                    "object_index": None,
                    "parameter_kind": "linear",
                    "coefficient_shape": json.dumps(list(decoder.coefficients.shape)),
                    "coefficients": np.asarray(decoder.coefficients)
                    .reshape(-1)
                    .tolist(),
                    "intercept_shape": json.dumps(list(decoder.intercepts.shape)),
                    "intercepts": np.asarray(decoder.intercepts)
                    .reshape(-1)
                    .tolist(),
                }
            )
            continue
        if not isinstance(decoder, SceneMLPDecoder):
            raise TypeError(f"Unknown scene decoder {type(decoder)!r}")
        for object_index, network in enumerate(decoder.networks):
            if network is None:
                continue
            records.extend(
                _mlp_parameter_records(
                    common,
                    network,
                    object_index=(
                        object_index if decoder.target == "object_position" else None
                    ),
                )
            )
    return pd.DataFrame.from_records(records)


def _mlp_parameter_records(
    common: Mapping[str, Any],
    network: FittedMLP,
    *,
    object_index: int | None,
) -> list[dict[str, Any]]:
    records = [
        {
            **dict(common),
            "object_index": object_index,
            "parameter_kind": "standardizer",
            "feature_mean": network.feature_mean.tolist(),
            "feature_scale": network.feature_scale.tolist(),
            "out_activation": network.out_activation,
        }
    ]
    for layer, (weights, biases) in enumerate(
        zip(network.weights, network.biases, strict=True)
    ):
        records.append(
            {
                **dict(common),
                "object_index": object_index,
                "parameter_kind": "mlp_layer",
                "mlp_layer": layer,
                "coefficient_shape": json.dumps(list(weights.shape)),
                "coefficients": weights.reshape(-1).tolist(),
                "intercept_shape": json.dumps(list(biases.shape)),
                "intercepts": biases.reshape(-1).tolist(),
                "out_activation": network.out_activation,
            }
        )
    return records


def _save_study(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    readouts: LayerTokenReadouts,
    candidates: pd.DataFrame,
    selections: pd.DataFrame,
    predictions: pd.DataFrame,
    object_results: pd.DataFrame,
    paired_comparisons: pd.DataFrame,
    layer_weights: pd.DataFrame,
    token_importance: pd.DataFrame,
    examples: pd.DataFrame,
    vocabulary: pd.DataFrame,
    decoder_parameters: pd.DataFrame,
    timings: Mapping[str, float],
) -> LensArtifact:
    artifact_id = make_artifact_id(str(spec["name"]), "token_scene_probe_study")
    relative_dir = Path("artifacts") / artifact_id
    tables = {
        "candidates": candidates,
        "selections": selections,
        "scene_predictions": predictions,
        "object_results": object_results,
        "paired_comparisons": paired_comparisons,
        "layer_weights": layer_weights,
        "token_importance": token_importance,
        "examples": examples,
        "vocabulary": vocabulary,
        "source_rows": readouts.rows,
        "source_sites": readouts.source_sites,
        "token_metadata": readouts.token_metadata,
        "decoder_parameters": decoder_parameters,
    }
    outputs = {name: str(relative_dir / f"{name}.parquet") for name in tables}
    arrays = {
        **_projection_array_map("channel", readouts.channel_projection),
        **_projection_array_map("pooled", readouts.pooled_projection),
        **_projection_array_map("tokenwise", readouts.tokenwise_projection),
    }
    artifact = LensArtifact(
        artifact_id=artifact_id,
        artifact_type="token_scene_probe_study",
        name=str(spec["name"]),
        group_id="scene_map_probe_studies",
        scope="dataset",
        selector={"feature": spec["feature"]},
        method={
            "workflow": "run_token_scene_probe_study",
            "schema_version": TOKEN_SCENE_STUDY_SCHEMA_VERSION,
            "research": {
                key: spec[key]
                for key in ["question", "hypothesis_family", "intended_claim"]
                if spec.get(key)
            },
            "split": spec["split"],
            "probe": spec["probe"],
            "representations": {
                "pooled": "mean tokens, then a shared training-only PCA",
                "tokenwise": (
                    "training-only channel PCA applied inside each token, then flatten "
                    "distinct token positions and apply a shared training-only PCA"
                ),
                "single_layer": "validation selects one captured layer",
                "learned_layer_mix": (
                    "non-negative layer weights summing to one; weights and other "
                    "hyperparameters selected on validation, decoder fit on train"
                ),
                "object_query": (
                    "identity presence is decoded as a whole roster; XYZ uses one "
                    "separately fitted head per named object, so each location can be "
                    "queried and inspected rather than scored only as one flattened map"
                ),
                "model_capacity": (
                    "linear and small one-hidden-layer MLP readouts are selected only "
                    "on validation and reported separately"
                ),
            },
            "storage_contract": {
                "raw_activations": "referenced from capture and never copied",
                "temporary_token_tensor": "memory only",
                "evictable_cache": readouts.cache_key,
                "saved_reconstruction": (
                    "source rows/sites, token metadata, all PCA transforms, selected "
                    "decoder parameters, layer weights, and predictions"
                ),
            },
            "outputs": outputs,
            "timings_seconds": dict(timings),
        },
        metrics={
            "candidate_count": int(len(candidates)),
            "selection_count": int(len(selections)),
            "prediction_count": int(len(predictions)),
            "object_result_count": int(len(object_results)),
            "paired_comparison_count": int(len(paired_comparisons)),
            "token_importance_count": int(len(token_importance)),
            "source_row_count": int(len(readouts.rows)),
            "layer_count": int(len(readouts.layers)),
            "token_count": int(readouts.token_count),
            "total_seconds": float(timings.get("total_seconds", 0.0)),
        },
        display={
            "kind": "token_scene_probe_study",
            "status": "exploratory",
            "comparisons": selections.to_dict("records"),
        },
        tags=(
            "probe",
            "scene-map",
            "tokenwise",
            "layer-mixture",
            "object-identity",
            "object-location",
            "exploratory",
        ),
        source_trace_ids=tuple(
            sorted(str(value) for value in readouts.rows["trace_id"].unique())
        ),
    )
    saved = dataset.save_artifact(artifact, arrays=arrays)
    artifact_dir = dataset._dataset_artifact_root() / relative_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_parquet(artifact_dir / f"{name}.parquet", index=False)
    return saved


def _projection_array_map(
    prefix: str, projection: ProjectionState
) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_input_center": projection.input_center,
        f"{prefix}_input_scale": projection.input_scale,
        f"{prefix}_pca_center": projection.pca_center,
        f"{prefix}_components": projection.components,
        f"{prefix}_explained_variance_ratio": projection.explained_variance_ratio,
    }


def _normalize_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(spec)
    normalized.setdefault(
        "name", "PI0.5 token-preserving joint object identity and location study"
    )
    feature = normalized.get("feature")
    if not isinstance(feature, Mapping):
        raise ValueError("Token scene study requires one feature mapping")
    normalized["feature"] = dict(feature)
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
    probe = dict(normalized.get("probe") or {})
    probe.setdefault("readout_dims", [64, 128])
    probe.setdefault("ridge_alphas", [1.0, 10.0])
    probe.setdefault("models", ["linear", "mlp"])
    probe.setdefault("mlp_hidden_layer_sizes", [64])
    probe.setdefault("mlp_max_iter", 300)
    probe.setdefault("mlp_workers", 4)
    probe.setdefault("token_channel_dim", 16)
    probe.setdefault("channel_sample_count", 50_000)
    probe.setdefault("projection_fit_rows", 10_000)
    probe.setdefault("io_workers", 8)
    probe.setdefault("min_train_episodes", 5)
    probe.setdefault("mixture_iterations", 5)
    probe.setdefault("mixture_regularization", 1e-3)
    probe.setdefault("bootstrap_samples", 2_000)
    probe.setdefault("cache", True)
    normalized["probe"] = probe
    return normalized


def _variant(fitted: FittedSceneRepresentation) -> str:
    return (
        f"{fitted.record['representation']}__{fitted.record['structure']}__"
        f"{fitted.record['model']}"
    )


def _episode_mean(values: np.ndarray, episodes: np.ndarray, mask: np.ndarray) -> float:
    frame = pd.DataFrame(
        {
            "episode": np.asarray(episodes)[mask],
            "value": np.asarray(values, dtype=float)[mask],
        }
    )
    frame = frame.loc[np.isfinite(frame["value"])]
    if frame.empty:
        return float("nan")
    return float(frame.groupby("episode", sort=False)["value"].mean().mean())


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return int(value)
