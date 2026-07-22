"""Test whether a saved identity probe's evidence falls on the named object."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from vla_lens.artifacts import LensArtifact, make_artifact_id
from vla_lens.probes.matched_scene_study import (
    _bbox_patch_mask,
    _object_bbox,
    _pixel_patch_difference,
    _random_ranking_expected_ap,
)
from vla_lens.probes.scene_map_study import scene_map_target_table
from vla_lens.probes.token_representations import (
    ProjectionState,
    read_compressed_token_layer,
)
from vla_lens.traces import TraceBundle, TraceDataset

IDENTITY_LOCALIZATION_STUDY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class IdentityLocalizationStudyResult:
    """Saved localization evidence and compact review tables."""

    artifact: LensArtifact | None
    episode_object_metrics: pd.DataFrame
    summary: pd.DataFrame
    object_summary: pd.DataFrame
    matched_pair_metrics: pd.DataFrame
    matched_pair_summary: pd.DataFrame
    examples: pd.DataFrame
    reconstruction_check: pd.DataFrame
    timings: Mapping[str, float]


def linear_token_contributions(
    compressed_tokens: np.ndarray,
    projection: ProjectionState,
    coefficients: np.ndarray,
    intercepts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exactly split linear probe scores into one signed value per token."""

    values = np.asarray(compressed_tokens, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("Compressed tokens must have row/token/channel axes")
    flat = values.reshape(len(values), -1)
    weights = np.asarray(coefficients, dtype=np.float64)
    biases = np.asarray(intercepts, dtype=np.float64).reshape(-1)
    if weights.ndim != 2 or len(weights) != len(biases):
        raise ValueError("Linear identity parameters have incompatible shapes")
    if flat.shape[1] != projection.input_center.size:
        raise ValueError(
            "Token tensor does not match the saved tokenwise projection: "
            f"{flat.shape[1]} != {projection.input_center.size}"
        )
    if weights.shape[1] > projection.components.shape[0]:
        raise ValueError("Decoder uses more PCA dimensions than the saved projection")

    components = np.asarray(
        projection.components[: weights.shape[1]], dtype=np.float64
    )
    centered = (
        (flat - projection.input_center) / projection.input_scale
        - projection.pca_center
    )
    feature_weights = weights @ components
    feature_contributions = centered[:, None, :] * feature_weights[None, :, :]
    token_contributions = feature_contributions.reshape(
        len(values), len(weights), values.shape[1], values.shape[2]
    ).sum(axis=3)
    scores = token_contributions.sum(axis=2) + biases[None, :]
    static_token_strength = np.linalg.norm(
        feature_weights.reshape(len(weights), values.shape[1], values.shape[2]), axis=2
    )
    return (
        token_contributions.astype(np.float32),
        scores.astype(np.float32),
        static_token_strength.astype(np.float32),
    )


def run_identity_localization_study(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    *,
    save: bool = True,
) -> IdentityLocalizationStudyResult:
    """Replay one saved identity probe and compare patch evidence with object boxes."""

    normalized = _normalize_spec(spec)
    started = time.perf_counter()
    timings: dict[str, float] = {}

    step = time.perf_counter()
    source = dataset.load_artifact(str(normalized["source_probe_artifact_id"]))
    source_rows = _artifact_table(dataset, source, "source_rows")
    source_sites = _artifact_table(dataset, source, "source_sites")
    token_metadata = _artifact_table(dataset, source, "token_metadata")
    vocabulary = _artifact_table(dataset, source, "vocabulary")
    selections = _artifact_table(dataset, source, "selections")
    decoder_parameters = _artifact_table(dataset, source, "decoder_parameters")
    selection = _selected_probe(selections, normalized)
    variant = str(normalized["variant"])
    coefficients, intercepts = _linear_parameters(
        decoder_parameters, variant=variant, target="scene_identity"
    )
    selected_layer = int(selection["selected_layer"])
    readout_dim = int(selection["readout_dim"])
    coefficients = coefficients[:, :readout_dim]
    split = dict(source.method.get("split") or {})
    split_column = str(split.get("column", "split"))
    evaluation_value = str(
        normalized.get("evaluation_value") or split.get("test_value", "test_heldout_task")
    )
    evaluation_rows = source_rows.loc[
        source_rows[split_column].astype(str) == evaluation_value
    ].copy()
    evaluation_rows = evaluation_rows.reset_index(drop=True)
    evaluation_rows["evaluation_array_index"] = np.arange(
        len(evaluation_rows), dtype=np.int64
    )
    if evaluation_rows.empty:
        raise ValueError(f"Source probe has no rows for split {evaluation_value!r}")
    timings["load_contract_seconds"] = time.perf_counter() - step

    step = time.perf_counter()
    channel_projection = _artifact_projection(dataset, source, "channel")
    tokenwise_projection = _artifact_projection(dataset, source, "tokenwise")
    generation_step = source.selector.get("feature", {}).get("generation_step")
    compressed = read_compressed_token_layer(
        dataset,
        evaluation_rows,
        source_sites,
        token_metadata,
        layer=selected_layer,
        channel_projection=channel_projection,
        generation_step=generation_step,
        io_workers=int(normalized["analysis"]["io_workers"]),
    )
    contributions, probe_scores, static_strength = linear_token_contributions(
        compressed,
        tokenwise_projection,
        coefficients,
        intercepts,
    )
    timings["replay_contributions_seconds"] = time.perf_counter() - step

    step = time.perf_counter()
    reconstruction_check = _reconstruction_check(
        dataset,
        source,
        evaluation_rows,
        variant,
        probe_scores,
    )
    max_error = float(reconstruction_check["max_absolute_error"].max())
    if max_error > float(normalized["analysis"]["replay_tolerance"]):
        raise ValueError(
            "Replayed probe scores do not match the saved predictions; "
            f"maximum absolute error is {max_error:.6g}"
        )
    targets, target_vocabulary = scene_map_target_table(dataset, source_rows, cache=True)
    expected_names = vocabulary.sort_values("object_index")["object_name"].astype(str).tolist()
    if list(targets.vocabulary) != expected_names:
        raise ValueError("Source probe vocabulary does not match current scene targets")
    source_indices = evaluation_rows["representation_row_index"].to_numpy(dtype=np.int64)
    evaluation_presence = targets.presence[source_indices]
    evaluation_manipulated = targets.role_manipulated[source_indices]
    evaluation_distractor = targets.role_distractor[source_indices]
    boxes, visible = _object_boxes(
        dataset,
        evaluation_rows,
        token_metadata,
        evaluation_presence,
        targets.vocabulary,
        camera_name=str(normalized["camera_name"]),
    )
    episode_object_metrics = _episode_object_metrics(
        evaluation_rows,
        token_metadata,
        evaluation_presence,
        targets.vocabulary,
        boxes,
        visible,
        contributions,
        probe_scores,
        static_strength,
        evaluation_manipulated,
        evaluation_distractor,
        threshold=float(selection["selection_threshold"]),
    )
    summary = _summary_table(
        episode_object_metrics,
        bootstrap_samples=int(normalized["analysis"]["bootstrap_samples"]),
    )
    object_summary = _object_summary(
        episode_object_metrics,
        bootstrap_samples=int(normalized["analysis"]["bootstrap_samples"]),
    )
    examples = _example_table(
        episode_object_metrics,
        count=int(normalized["analysis"]["example_count"]),
    )
    timings["score_episodes_seconds"] = time.perf_counter() - step

    step = time.perf_counter()
    matched_pair_metrics = pd.DataFrame()
    matched_pair_summary = pd.DataFrame()
    matched_id = normalized.get("matched_scene_artifact_id")
    if matched_id:
        matched_source = dataset.load_artifact(str(matched_id))
        matched_pairs = _artifact_table(dataset, matched_source, "matched_pairs")
        matched_pair_metrics = _matched_pair_metrics(
            dataset,
            matched_pairs,
            evaluation_rows,
            token_metadata,
            targets.vocabulary,
            contributions,
            camera_id=str(normalized["camera_id"]),
            camera_name=str(normalized["camera_name"]),
        )
        matched_pair_summary = _matched_summary(
            matched_pair_metrics,
            bootstrap_samples=int(normalized["analysis"]["bootstrap_samples"]),
        )
    timings["score_matched_pairs_seconds"] = time.perf_counter() - step
    timings["total_seconds"] = time.perf_counter() - started

    artifact = (
        _save_study(
            dataset,
            normalized,
            source,
            evaluation_rows,
            source_sites,
            token_metadata,
            target_vocabulary,
            episode_object_metrics,
            summary,
            object_summary,
            matched_pair_metrics,
            matched_pair_summary,
            examples,
            reconstruction_check,
            contributions,
            probe_scores,
            static_strength,
            boxes,
            visible,
            selected_layer=selected_layer,
            readout_dim=readout_dim,
            timings=timings,
        )
        if save
        else None
    )
    return IdentityLocalizationStudyResult(
        artifact=artifact,
        episode_object_metrics=episode_object_metrics,
        summary=summary,
        object_summary=object_summary,
        matched_pair_metrics=matched_pair_metrics,
        matched_pair_summary=matched_pair_summary,
        examples=examples,
        reconstruction_check=reconstruction_check,
        timings=timings,
    )


def _selected_probe(selections: pd.DataFrame, spec: Mapping[str, Any]) -> pd.Series:
    matches = selections.loc[
        (selections["target"].astype(str) == "scene_identity")
        & (selections["model"].astype(str) == "linear")
        & (
            selections.apply(
                lambda row: (
                    f"{row['representation']}__{row['structure']}__{row['model']}"
                ),
                axis=1,
            )
            == str(spec["variant"])
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one selected identity probe {spec['variant']!r}; got {len(matches)}"
        )
    row = matches.iloc[0]
    if pd.isna(row.get("selected_layer")):
        raise ValueError("Identity localization currently requires one selected layer")
    return row


def _linear_parameters(
    parameters: pd.DataFrame, *, variant: str, target: str
) -> tuple[np.ndarray, np.ndarray]:
    matches = parameters.loc[
        (parameters["variant"].astype(str) == variant)
        & (parameters["target"].astype(str) == target)
        & (parameters["parameter_kind"].astype(str) == "linear")
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one saved linear decoder; got {len(matches)}")
    row = matches.iloc[0]
    coefficient_shape = tuple(json.loads(str(row["coefficient_shape"])))
    intercept_shape = tuple(json.loads(str(row["intercept_shape"])))
    return (
        np.asarray(row["coefficients"], dtype=np.float64).reshape(coefficient_shape),
        np.asarray(row["intercepts"], dtype=np.float64).reshape(intercept_shape),
    )


def _artifact_projection(
    dataset: TraceDataset, artifact: LensArtifact, prefix: str
) -> ProjectionState:
    def load(suffix: str) -> np.ndarray:
        return np.asarray(dataset.load_artifact_array(artifact, f"{prefix}_{suffix}"))

    return ProjectionState(
        input_center=load("input_center"),
        input_scale=load("input_scale"),
        pca_center=load("pca_center"),
        components=load("components"),
        explained_variance_ratio=load("explained_variance_ratio"),
    )


def _artifact_table(
    dataset: TraceDataset, artifact: LensArtifact, name: str
) -> pd.DataFrame:
    outputs = dict(artifact.method.get("outputs") or {})
    relative = outputs.get(name)
    if relative is None:
        raise KeyError(f"Artifact {artifact.artifact_id!r} has no output {name!r}")
    return pd.read_parquet(dataset._dataset_artifact_root() / str(relative))


def _reconstruction_check(
    dataset: TraceDataset,
    source: LensArtifact,
    rows: pd.DataFrame,
    variant: str,
    scores: np.ndarray,
) -> pd.DataFrame:
    predictions = _artifact_table(dataset, source, "scene_predictions")
    predictions = predictions.loc[
        (predictions["variant"].astype(str) == variant)
        & (predictions["target"].astype(str) == "scene_identity")
    ]
    by_trace = {
        str(row.trace_id): np.asarray(row.prediction, dtype=np.float64)
        for row in predictions.itertuples()
    }
    errors = []
    for index, trace_id in enumerate(rows["trace_id"].astype(str)):
        if trace_id not in by_trace:
            raise KeyError(f"Saved identity predictions are missing trace {trace_id!r}")
        errors.append(float(np.max(np.abs(scores[index] - by_trace[trace_id]))))
    return pd.DataFrame.from_records(
        [
            {
                "row_count": len(errors),
                "mean_absolute_error": float(np.mean(errors)),
                "max_absolute_error": float(np.max(errors)),
            }
        ]
    )


def _object_boxes(
    dataset: TraceDataset,
    rows: pd.DataFrame,
    tokens: pd.DataFrame,
    presence: np.ndarray,
    vocabulary: Sequence[str],
    *,
    camera_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.full((len(rows), len(vocabulary), 4), np.nan, dtype=np.float32)
    visible = np.zeros((len(rows), len(vocabulary)), dtype=bool)
    for row_index, trace_id in enumerate(rows["trace_id"].astype(str)):
        bundle = dataset.bundle(trace_id)
        for object_index in np.flatnonzero(presence[row_index]):
            bbox = _object_bbox(bundle, str(vocabulary[object_index]), camera_name)
            if bbox is None:
                continue
            mask = _bbox_patch_mask(tokens, bbox)
            if mask.any() and not mask.all():
                boxes[row_index, object_index] = bbox
                visible[row_index, object_index] = True
    return boxes, visible


def _episode_object_metrics(
    rows: pd.DataFrame,
    tokens: pd.DataFrame,
    presence: np.ndarray,
    vocabulary: Sequence[str],
    boxes: np.ndarray,
    visible: np.ndarray,
    contributions: np.ndarray,
    probe_scores: np.ndarray,
    static_strength: np.ndarray,
    role_manipulated: np.ndarray,
    role_distractor: np.ndarray,
    *,
    threshold: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    methods = {
        "positive_contribution": lambda values: np.maximum(values, 0.0),
        "signed_contribution": lambda values: values,
        "absolute_contribution": lambda values: np.abs(values),
    }
    for row_index, row in rows.iterrows():
        present_indices = np.flatnonzero(
            np.asarray(presence[row_index], dtype=bool) & visible[row_index]
        )
        for object_index in present_indices:
            target_mask = _bbox_patch_mask(tokens, boxes[row_index, object_index])
            wrong_name, wrong_mask = _wrong_object_mask(
                tokens,
                boxes[row_index],
                visible[row_index],
                vocabulary,
                int(object_index),
                target_mask,
            )
            static_metrics = _patch_metrics(
                static_strength[object_index], target_mask, wrong_mask
            )
            common = {
                "evaluation_array_index": int(row["evaluation_array_index"]),
                "trace_id": str(row["trace_id"]),
                "episode_id": row.get("episode_id"),
                "benchmark": row.get("benchmark"),
                "task_id": row.get("task_id"),
                "task_key": f"{row.get('benchmark')}:{row.get('task_id')}",
                "instruction_key": str(row.get("prompt")),
                "prompt": row.get("prompt"),
                "object_index": int(object_index),
                "object_name": str(vocabulary[object_index]),
                "wrong_object_name": wrong_name,
                "probe_score": float(probe_scores[row_index, object_index]),
                "probe_threshold": threshold,
                "probe_predicted_present": bool(
                    probe_scores[row_index, object_index] >= threshold
                ),
                "role_manipulated": bool(role_manipulated[row_index, object_index]),
                "role_distractor": bool(role_distractor[row_index, object_index]),
                "bbox_xyxy": boxes[row_index, object_index].tolist(),
                "static_average_precision": static_metrics["average_precision"],
            }
            for method, transform in methods.items():
                metrics = _patch_metrics(
                    transform(contributions[row_index, object_index]),
                    target_mask,
                    wrong_mask,
                )
                records.append({**common, "method": method, **metrics})
            records.append(
                {
                    **common,
                    "method": "static_coefficient_strength",
                    **static_metrics,
                }
            )
    return pd.DataFrame.from_records(records)


def _wrong_object_mask(
    tokens: pd.DataFrame,
    boxes: np.ndarray,
    visible: np.ndarray,
    vocabulary: Sequence[str],
    target_index: int,
    target_mask: np.ndarray,
) -> tuple[str | None, np.ndarray]:
    best_name: str | None = None
    best_mask = np.zeros(len(tokens), dtype=bool)
    best_overlap = float("inf")
    for object_index in np.flatnonzero(visible):
        if int(object_index) == target_index:
            continue
        mask = _bbox_patch_mask(tokens, boxes[object_index])
        overlap = float(np.logical_and(mask, target_mask).sum() / max(1, mask.sum()))
        if mask.any() and not mask.all() and overlap < best_overlap:
            best_name = str(vocabulary[object_index])
            best_mask = mask
            best_overlap = overlap
    return best_name, best_mask


def _patch_metrics(
    scores: np.ndarray, target_mask: np.ndarray, wrong_mask: np.ndarray
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    target = np.asarray(target_mask, dtype=bool)
    wrong = np.asarray(wrong_mask, dtype=bool)
    positive_count = int(target.sum())
    top = np.argsort(values)[-positive_count:]
    random_ap = _random_ranking_expected_ap(len(values), positive_count)
    magnitude = np.abs(values)
    total_magnitude = float(magnitude.sum())
    return {
        "patch_count": int(len(values)),
        "target_patch_count": positive_count,
        "target_patch_prevalence": float(target.mean()),
        "random_expected_average_precision": random_ap,
        "average_precision": float(average_precision_score(target, values)),
        "average_precision_minus_random": float(
            average_precision_score(target, values) - random_ap
        ),
        "roc_auc": float(roc_auc_score(target, values)),
        "top_k_recall": float(target[top].sum() / max(1, positive_count)),
        "target_mean_score": float(values[target].mean()),
        "outside_mean_score": float(values[~target].mean()),
        "target_minus_outside": float(values[target].mean() - values[~target].mean()),
        "target_absolute_mass_fraction": (
            float(magnitude[target].sum() / total_magnitude)
            if total_magnitude > 0
            else 0.0
        ),
        "wrong_object_patch_count": int(wrong.sum()),
        "wrong_object_average_precision": (
            float(average_precision_score(wrong, values))
            if wrong.any() and not wrong.all()
            else float("nan")
        ),
        "wrong_object_mean_score": (
            float(values[wrong].mean()) if wrong.any() else float("nan")
        ),
        "target_minus_wrong_object": (
            float(values[target].mean() - values[wrong].mean())
            if wrong.any()
            else float("nan")
        ),
    }


def _summary_table(
    metrics: pd.DataFrame, *, bootstrap_samples: int
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for method, frame in metrics.groupby("method", sort=True):
        comparisons = {
            "average_precision_minus_random": frame["average_precision_minus_random"].to_numpy(),
            "average_precision_minus_static": (
                frame["average_precision"] - frame["static_average_precision"]
            ).to_numpy(),
            "target_minus_wrong_object": frame[
                "target_minus_wrong_object"
            ].to_numpy(),
        }
        for metric_name, values in comparisons.items():
            for unit_index, (unit, groups) in enumerate(
                [
                    ("episode", frame["trace_id"].astype(str).to_numpy()),
                    ("benchmark_task", frame["task_key"].astype(str).to_numpy()),
                    (
                        "instruction",
                        frame["instruction_key"].astype(str).to_numpy(),
                    ),
                ]
            ):
                records.append(
                    {
                        "method": method,
                        "metric": metric_name,
                        "unit": unit,
                        **_grouped_bootstrap(
                            values,
                            groups,
                            bootstrap_samples=bootstrap_samples,
                            seed=20260722 + len(records) * 10 + unit_index,
                        ),
                    }
                )
    return pd.DataFrame.from_records(records)


def _grouped_bootstrap(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    frame = pd.DataFrame(
        {"group": np.asarray(groups).astype(str), "value": np.asarray(values, dtype=float)}
    ).dropna()
    grouped = frame.groupby("group", sort=True)["value"].mean().to_numpy()
    if not len(grouped):
        return {
            "group_count": 0,
            "mean": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "probability_positive": float("nan"),
        }
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        grouped,
        size=(max(1, int(bootstrap_samples)), len(grouped)),
        replace=True,
    ).mean(axis=1)
    return {
        "group_count": int(len(grouped)),
        "mean": float(grouped.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "probability_positive": float(np.mean(draws > 0.0)),
    }


def _object_summary(
    metrics: pd.DataFrame, *, bootstrap_samples: int
) -> pd.DataFrame:
    focus = metrics.loc[metrics["method"] == "positive_contribution"]
    records: list[dict[str, Any]] = []
    for object_index, frame in focus.groupby("object_index", sort=True):
        random_lift = frame["average_precision_minus_random"].to_numpy()
        static_lift = (
            frame["average_precision"] - frame["static_average_precision"]
        ).to_numpy()
        groups = frame["trace_id"].astype(str).to_numpy()
        random_summary = _grouped_bootstrap(
            random_lift,
            groups,
            bootstrap_samples=bootstrap_samples,
            seed=20260724 + int(object_index) * 2,
        )
        static_summary = _grouped_bootstrap(
            static_lift,
            groups,
            bootstrap_samples=bootstrap_samples,
            seed=20260725 + int(object_index) * 2,
        )
        records.append(
            {
                "object_index": int(object_index),
                "object_name": str(frame.iloc[0]["object_name"]),
                "episode_count": int(frame["trace_id"].nunique()),
                "benchmark_task_count": int(frame["task_key"].nunique()),
                "mean_target_patch_count": float(frame["target_patch_count"].mean()),
                "mean_average_precision": float(frame["average_precision"].mean()),
                "mean_random_average_precision": float(
                    frame["random_expected_average_precision"].mean()
                ),
                "mean_static_average_precision": float(
                    frame["static_average_precision"].mean()
                ),
                "probe_recall": float(frame["probe_predicted_present"].mean()),
                "manipulated_fraction": float(frame["role_manipulated"].mean()),
                "distractor_fraction": float(frame["role_distractor"].mean()),
                "random_lift": random_summary["mean"],
                "random_lift_ci95_low": random_summary["ci95_low"],
                "random_lift_ci95_high": random_summary["ci95_high"],
                "static_lift": static_summary["mean"],
                "static_lift_ci95_low": static_summary["ci95_low"],
                "static_lift_ci95_high": static_summary["ci95_high"],
            }
        )
    return pd.DataFrame.from_records(records)


def _example_table(metrics: pd.DataFrame, *, count: int) -> pd.DataFrame:
    focus = metrics.loc[metrics["method"] == "positive_contribution"].copy()
    if focus.empty:
        return focus
    count = max(1, int(count))
    best = focus.nlargest(count, "average_precision").assign(example_kind="best")
    worst = focus.nsmallest(count, "average_precision").assign(example_kind="worst")
    return pd.concat([best, worst], ignore_index=True)


def _matched_pair_metrics(
    dataset: TraceDataset,
    pairs: pd.DataFrame,
    rows: pd.DataFrame,
    tokens: pd.DataFrame,
    vocabulary: Sequence[str],
    contributions: np.ndarray,
    *,
    camera_id: str,
    camera_name: str,
) -> pd.DataFrame:
    row_lookup = {
        str(row.trace_id): int(row.evaluation_array_index) for row in rows.itertuples()
    }
    object_lookup = {str(name): index for index, name in enumerate(vocabulary)}
    records: list[dict[str, Any]] = []
    for pair in pairs.itertuples():
        left_trace = str(pair.left_trace_id)
        right_trace = str(pair.right_trace_id)
        object_name = str(pair.moved_object_name)
        if (
            left_trace not in row_lookup
            or right_trace not in row_lookup
            or object_name not in object_lookup
        ):
            continue
        left = dataset.bundle(left_trace)
        right = dataset.bundle(right_trace)
        left_box = _object_bbox(left, object_name, camera_name)
        right_box = _object_bbox(right, object_name, camera_name)
        if left_box is None or right_box is None:
            continue
        target = _bbox_patch_mask(tokens, left_box) | _bbox_patch_mask(tokens, right_box)
        if not target.any() or target.all():
            continue
        wrong_name, wrong = _pair_wrong_mask(
            left,
            right,
            tokens,
            list(pair.stationary_object_names),
            camera_name,
            target,
        )
        object_index = object_lookup[object_name]
        left_index = row_lookup[left_trace]
        right_index = row_lookup[right_trace]
        contribution_change = np.abs(
            contributions[right_index, object_index]
            - contributions[left_index, object_index]
        )
        score_sets = [("probe_contribution_change", contribution_change)]
        pixel_change = _pixel_patch_difference(left, right, tokens, camera_id)
        if pixel_change is not None:
            score_sets.append(("raw_pixel_change", pixel_change))
        common = {
            "pair_id": str(pair.pair_id),
            "scene_key": str(pair.scene_key),
            "split": str(pair.split),
            "left_trace_id": left_trace,
            "right_trace_id": right_trace,
            "moved_object_name": object_name,
            "moved_distance_m": float(pair.moved_distance_m),
            "wrong_object_name": wrong_name,
        }
        for method, values in score_sets:
            records.append({**common, "method": method, **_patch_metrics(values, target, wrong)})
    return pd.DataFrame.from_records(records)


def _pair_wrong_mask(
    left: TraceBundle,
    right: TraceBundle,
    tokens: pd.DataFrame,
    object_names: Sequence[str],
    camera_name: str,
    target: np.ndarray,
) -> tuple[str | None, np.ndarray]:
    best_name: str | None = None
    best_mask = np.zeros(len(tokens), dtype=bool)
    best_overlap = float("inf")
    for object_name in object_names:
        left_box = _object_bbox(left, str(object_name), camera_name)
        right_box = _object_bbox(right, str(object_name), camera_name)
        if left_box is None or right_box is None:
            continue
        mask = _bbox_patch_mask(tokens, left_box) | _bbox_patch_mask(tokens, right_box)
        overlap = float(np.logical_and(mask, target).sum() / max(1, mask.sum()))
        if mask.any() and not mask.all() and overlap < best_overlap:
            best_name, best_mask, best_overlap = str(object_name), mask, overlap
    return best_name, best_mask


def _matched_summary(
    metrics: pd.DataFrame, *, bootstrap_samples: int
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame()
    for method, frame in metrics.groupby("method", sort=True):
        records.append(
            {
                "method": method,
                "metric": "average_precision_minus_random",
                **_grouped_bootstrap(
                    frame["average_precision_minus_random"].to_numpy(),
                    frame["scene_key"].astype(str).to_numpy(),
                    bootstrap_samples=bootstrap_samples,
                    seed=20260723 + len(records),
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _save_study(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    source: LensArtifact,
    rows: pd.DataFrame,
    source_sites: pd.DataFrame,
    token_metadata: pd.DataFrame,
    vocabulary: pd.DataFrame,
    episode_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    object_summary: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    pair_summary: pd.DataFrame,
    examples: pd.DataFrame,
    reconstruction: pd.DataFrame,
    contributions: np.ndarray,
    probe_scores: np.ndarray,
    static_strength: np.ndarray,
    boxes: np.ndarray,
    visible: np.ndarray,
    *,
    selected_layer: int,
    readout_dim: int,
    timings: Mapping[str, float],
) -> LensArtifact:
    artifact_id = make_artifact_id(str(spec["name"]), "identity_localization_study")
    relative_dir = Path("artifacts") / artifact_id
    tables = {
        "evaluation_rows": rows,
        "source_sites": source_sites.loc[
            pd.to_numeric(source_sites["layer"], errors="coerce") == selected_layer
        ].copy(),
        "token_metadata": token_metadata,
        "vocabulary": vocabulary,
        "episode_object_metrics": episode_metrics,
        "summary": summary,
        "object_summary": object_summary,
        "matched_pair_metrics": pair_metrics,
        "matched_pair_summary": pair_summary,
        "examples": examples,
        "reconstruction_check": reconstruction,
    }
    outputs = {name: str(relative_dir / f"{name}.parquet") for name in tables}
    arrays = {
        "signed_patch_contributions": contributions,
        "probe_scores": probe_scores,
        "static_patch_strength": static_strength,
        "object_bbox_xyxy": boxes,
        "object_visible": visible.astype(np.uint8),
    }
    artifact = LensArtifact(
        artifact_id=artifact_id,
        artifact_type="identity_localization_study",
        name=str(spec["name"]),
        group_id="scene_map_probe_diagnostics",
        scope="dataset",
        selector={
            "source_probe_artifact_id": source.artifact_id,
            "variant": spec["variant"],
            "evaluation_value": spec["evaluation_value"],
        },
        method={
            "workflow": "run_identity_localization_study",
            "schema_version": IDENTITY_LOCALIZATION_STUDY_SCHEMA_VERSION,
            "research": {
                key: spec[key]
                for key in ["question", "hypothesis_family", "intended_claim"]
                if spec.get(key)
            },
            "source_probe_artifact_id": source.artifact_id,
            "matched_scene_artifact_id": spec.get("matched_scene_artifact_id"),
            "selected_layer": selected_layer,
            "readout_dim": readout_dim,
            "analysis": spec["analysis"],
            "controls": {
                "random": "exact expected average precision for a random patch ranking",
                "fixed_probe_preference": "object decoder coefficient magnitude by patch",
                "wrong_region": "another visible object in the same frame",
                "matched_scene_positive": "raw pixel change for the same moved-object pair",
            },
            "interpretation": (
                "signed patch values exactly sum with the decoder intercept to the saved "
                "linear probe score; they explain the probe, not a causal VLA mechanism"
            ),
            "array_axes": {
                "signed_patch_contributions": ["evaluation_row", "object", "patch"],
                "probe_scores": ["evaluation_row", "object"],
                "static_patch_strength": ["object", "patch"],
                "object_bbox_xyxy": ["evaluation_row", "object", "xyxy"],
                "object_visible": ["evaluation_row", "object"],
            },
            "storage_contract": {
                "raw_activations": "referenced from capture and never copied",
                "temporary_compressed_tokens": "memory only for the selected layer",
                "saved_evidence": (
                    "compact patch contributions, scores, boxes, exact row/object/patch "
                    "mappings, controls, grouped uncertainty, and replay error"
                ),
            },
            "outputs": outputs,
            "timings_seconds": dict(timings),
        },
        metrics={
            "evaluation_row_count": int(len(rows)),
            "visible_object_count": int(visible.sum()),
            "episode_object_metric_count": int(len(episode_metrics)),
            "matched_pair_metric_count": int(len(pair_metrics)),
            "replay_max_absolute_error": float(reconstruction["max_absolute_error"].max()),
            "total_seconds": float(timings.get("total_seconds", 0.0)),
        },
        display={
            "kind": "identity_localization_study",
            "status": "exploratory",
            "summary": json.loads(summary.to_json(orient="records")),
            "matched_pair_summary": json.loads(pair_summary.to_json(orient="records")),
        },
        tags=(
            "diagnostic",
            "probe-attribution",
            "object-identity",
            "object-localization",
            "episode-view",
            "exploratory",
        ),
        source_trace_ids=tuple(sorted(rows["trace_id"].astype(str).unique())),
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
        raise ValueError("Identity localization requires source_probe_artifact_id")
    normalized.setdefault(
        "name", "PI0.5 held-out object identity patch localization study"
    )
    normalized.setdefault("variant", "tokenwise__single_layer__linear")
    normalized.setdefault("evaluation_value", "test_heldout_task")
    normalized.setdefault("camera_id", "main")
    normalized.setdefault("camera_name", "agentview")
    analysis = dict(normalized.get("analysis") or {})
    analysis.setdefault("io_workers", 8)
    analysis.setdefault("bootstrap_samples", 2000)
    analysis.setdefault("example_count", 12)
    analysis.setdefault("replay_tolerance", 1e-4)
    normalized["analysis"] = analysis
    return normalized
