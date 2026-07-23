"""Shared data and replay contracts for object-conditioned visual probes."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from vla_lens.artifacts import LensArtifact
from vla_lens.probes.identity_localization_study import _artifact_table, _grouped_bootstrap
from vla_lens.probes.image_location_study import (
    _box_patch_targets,
    _cached_object_boxes,
    _require_same_rows,
)
from vla_lens.probes.scene_map_study import SceneMapTargets, scene_map_target_table
from vla_lens.probes.token_representations import (
    LayerTokenReadouts,
    build_layer_token_readouts,
    read_compressed_token_layers,
)
from vla_lens.traces import TraceDataset


@dataclass(frozen=True, slots=True)
class VisualObjectData:
    """Compact visual tokens plus aligned scene-object targets."""

    source: LensArtifact
    readouts: LayerTokenReadouts
    compact: np.ndarray
    compact_cache_key: str
    compact_cache_hit: bool
    targets: SceneMapTargets
    vocabulary: pd.DataFrame
    boxes_px: np.ndarray
    visible: np.ndarray
    patch_targets: np.ndarray
    box_cache_key: str
    box_cache_hit: bool


@dataclass(frozen=True, slots=True)
class FittedClassifier:
    """A small classifier whose prediction can be replayed with NumPy."""

    model: str
    classes: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    n_iter: int
    converged: bool

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        hidden = (np.asarray(values, dtype=np.float64) - self.feature_mean) / self.feature_scale
        for index, (weights, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            hidden = hidden @ weights + bias
            if index < len(self.weights) - 1:
                hidden = np.maximum(hidden, 0.0)
        if hidden.ndim == 1:
            hidden = hidden[:, None]
        if hidden.shape[1] == 1 and len(self.classes) == 2:
            positive = 1.0 / (1.0 + np.exp(-np.clip(hidden[:, 0], -40.0, 40.0)))
            return np.column_stack([1.0 - positive, positive])
        shifted = hidden - np.max(hidden, axis=1, keepdims=True)
        exponent = np.exp(np.clip(shifted, -40.0, 40.0))
        return exponent / exponent.sum(axis=1, keepdims=True)

    def predict(self, values: np.ndarray) -> np.ndarray:
        return self.classes[np.argmax(self.predict_proba(values), axis=1)]


@dataclass(frozen=True, slots=True)
class ContextEncoder:
    """Train-fitted categorical encoder used by non-activation baselines."""

    columns: tuple[str, ...]
    encoder: OneHotEncoder

    def transform(self, rows: pd.DataFrame) -> np.ndarray:
        frame = _categorical_frame(rows, self.columns)
        return np.asarray(self.encoder.transform(frame), dtype=np.float32)


def prepare_visual_object_data(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
) -> VisualObjectData:
    """Rebuild compact token values and boxes from an accepted source artifact."""

    source_id = str(spec.get("source_probe_artifact_id") or "")
    if not source_id:
        raise ValueError("Object-conditioned studies require source_probe_artifact_id")
    source = dataset.load_artifact(source_id)
    source_rows = _artifact_table(dataset, source, "source_rows")
    source_vocabulary = _artifact_table(dataset, source, "vocabulary")
    source_probe = dict(source.method.get("probe") or {})
    split = dict(source.method.get("split") or {})
    feature = dict(source.selector.get("feature") or {})
    analysis = dict(spec.get("analysis") or {})
    readouts = build_layer_token_readouts(
        dataset,
        feature,
        split,
        readout_dim=max(int(value) for value in source_probe["readout_dims"]),
        token_channel_dim=int(source_probe["token_channel_dim"]),
        channel_sample_count=int(source_probe["channel_sample_count"]),
        projection_fit_rows=int(source_probe["projection_fit_rows"]),
        io_workers=int(analysis.get("io_workers", 8)),
        cache=True,
    )
    _require_same_rows(source_rows, readouts.rows)
    compact_result = read_compressed_token_layers(
        dataset,
        readouts.rows,
        readouts.source_sites,
        readouts.token_metadata,
        layers=readouts.layers,
        channel_projection=readouts.channel_projection,
        generation_step=feature.get("generation_step"),
        io_workers=int(analysis.get("io_workers", 8)),
        cache=True,
    )
    targets, vocabulary = scene_map_target_table(dataset, readouts.rows, cache=True)
    expected_names = (
        source_vocabulary.sort_values("object_index")["object_name"].astype(str).tolist()
    )
    if list(targets.vocabulary) != expected_names:
        raise ValueError("Source probe vocabulary does not match current scene targets")
    boxes, visible, box_key, box_hit = _cached_object_boxes(
        dataset,
        readouts.rows,
        readouts.token_metadata,
        targets.presence,
        targets.vocabulary,
        camera_name=str(spec.get("camera_name", "agentview")),
    )
    patch_targets = _box_patch_targets(boxes, visible, readouts.token_metadata)
    return VisualObjectData(
        source=source,
        readouts=readouts,
        compact=compact_result.values,
        compact_cache_key=compact_result.cache_key,
        compact_cache_hit=compact_result.cache_hit,
        targets=targets,
        vocabulary=vocabulary,
        boxes_px=boxes,
        visible=visible,
        patch_targets=patch_targets,
        box_cache_key=box_key,
        box_cache_hit=box_hit,
    )


def split_masks(rows: pd.DataFrame, split: Mapping[str, Any]) -> dict[str, np.ndarray]:
    column = str(split["column"])
    return {
        name: rows[column].astype(str).to_numpy() == str(split[f"{name}_value"])
        for name in ("train", "selection", "test")
    }


def fit_classifier(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    model: str,
    alpha: float,
    hidden_units: int,
    max_iter: int,
    random_state: int,
) -> FittedClassifier:
    """Fit the standard linear or one-hidden-layer probe and retain replay state."""

    X = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels)
    scaler = StandardScaler().fit(X)
    scaled = scaler.transform(X)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        if model == "linear":
            estimator = LogisticRegression(
                C=1.0 / max(float(alpha), 1e-12),
                max_iter=int(max_iter),
                class_weight="balanced",
                random_state=int(random_state),
            )
        elif model == "mlp":
            estimator = MLPClassifier(
                hidden_layer_sizes=(int(hidden_units),),
                activation="relu",
                solver="adam",
                alpha=float(alpha),
                max_iter=int(max_iter),
                random_state=int(random_state),
            )
        else:
            raise ValueError(f"Unknown classifier model {model!r}")
        estimator.fit(scaled, y)
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    if model == "linear":
        coefficient = np.asarray(estimator.coef_, dtype=np.float64)
        weights = (coefficient.T,)
        biases = (np.asarray(estimator.intercept_, dtype=np.float64),)
    else:
        weights = tuple(np.asarray(value, dtype=np.float64) for value in estimator.coefs_)
        biases = tuple(np.asarray(value, dtype=np.float64) for value in estimator.intercepts_)
    result = FittedClassifier(
        model=model,
        classes=np.asarray(estimator.classes_),
        feature_mean=np.asarray(scaler.mean_, dtype=np.float64),
        feature_scale=np.asarray(scaler.scale_, dtype=np.float64),
        weights=weights,
        biases=biases,
        n_iter=int(np.max(np.atleast_1d(estimator.n_iter_))),
        converged=converged,
    )
    sample_count = min(len(X), 1024)
    sklearn_probabilities = np.asarray(
        estimator.predict_proba(scaled[:sample_count]), dtype=np.float64
    )
    replay_probabilities = result.predict_proba(X[:sample_count])
    if not np.allclose(sklearn_probabilities, replay_probabilities, atol=1e-9, rtol=1e-7):
        error = float(np.max(np.abs(sklearn_probabilities - replay_probabilities)))
        raise RuntimeError(
            f"Saved {model} classifier parameters do not replay fitted scores; "
            f"max absolute error={error:.3g}"
        )
    return result


def classifier_arrays(prefix: str, fitted: FittedClassifier) -> dict[str, np.ndarray]:
    arrays = {
        f"{prefix}_classes": fitted.classes,
        f"{prefix}_feature_mean": fitted.feature_mean,
        f"{prefix}_feature_scale": fitted.feature_scale,
    }
    for index, value in enumerate(fitted.weights):
        arrays[f"{prefix}_weights_{index}"] = value
    for index, value in enumerate(fitted.biases):
        arrays[f"{prefix}_biases_{index}"] = value
    return arrays


def fit_context_encoder(
    rows: pd.DataFrame,
    train_mask: np.ndarray,
    columns: Sequence[str],
) -> ContextEncoder:
    available = tuple(str(column) for column in columns if column in rows)
    if not available:
        raise ValueError("No requested context columns are available")
    frame = _categorical_frame(rows, available)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    encoder.fit(frame.loc[train_mask])
    return ContextEncoder(available, encoder)


def grouped_paired_interval(
    candidate: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Equal-weight grouped interval for a per-example score improvement."""

    return _grouped_bootstrap(
        np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float),
        np.asarray(groups).astype(str),
        bootstrap_samples=int(bootstrap_samples),
        seed=int(seed),
    )


def row_identity(row: pd.Series) -> dict[str, Any]:
    return {
        "trace_id": str(row["trace_id"]),
        "episode_id": row.get("episode_id"),
        "benchmark": row.get("benchmark"),
        "task_id": row.get("task_id"),
        "task_key": f"{row.get('benchmark')}:{row.get('task_name', row.get('task_id'))}",
        "instruction_key": str(row.get("prompt")),
        "prompt": row.get("prompt"),
    }


def normalized_patch_centers(tokens: pd.DataFrame) -> np.ndarray:
    width = float(tokens["pixel_x1"].max())
    height = float(tokens["pixel_y1"].max())
    return np.column_stack(
        [
            0.5 * (tokens["pixel_x0"].to_numpy(float) + tokens["pixel_x1"].to_numpy(float)) / width,
            0.5
            * (tokens["pixel_y0"].to_numpy(float) + tokens["pixel_y1"].to_numpy(float))
            / height,
        ]
    ).astype(np.float32)


def _categorical_frame(rows: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {column: rows[column].fillna("<missing>").astype(str) for column in columns},
        index=rows.index,
    )
