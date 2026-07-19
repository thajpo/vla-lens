"""Linear decoders for matched pooled, tokenwise, and layer-mixture studies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from vla_lens.probes.scene_map_study import (
    SceneMapTargets,
    _identity_metrics,
    _position_metrics,
    _supported_objects,
)


@dataclass(frozen=True, slots=True)
class SceneLinearDecoder:
    """A linear whole-scene decoder with masked per-object XYZ heads."""

    target: str
    coefficients: np.ndarray
    intercepts: np.ndarray
    supported: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float64)
        if self.target == "scene_identity":
            prediction = np.zeros((len(values), len(self.supported)), dtype=np.float64)
            if self.supported.any():
                prediction[:, self.supported] = (
                    values @ self.coefficients[self.supported].T
                    + self.intercepts[self.supported]
                )
            return prediction
        prediction = np.full(
            (len(values), len(self.supported), 3), np.nan, dtype=np.float64
        )
        for object_index in np.flatnonzero(self.supported):
            prediction[:, object_index] = (
                values @ self.coefficients[object_index].T
                + self.intercepts[object_index]
            )
        return prediction


@dataclass(frozen=True, slots=True)
class FittedSceneRepresentation:
    """One validation-selected representation and its final predictions."""

    record: Mapping[str, Any]
    decoder: SceneLinearDecoder
    prediction: np.ndarray
    layer_weights: np.ndarray


def fit_structured_scene_representations(
    readouts: Mapping[str, np.ndarray],
    rows: pd.DataFrame,
    targets: SceneMapTargets,
    layers: Sequence[int],
    split: Mapping[str, Any],
    *,
    readout_dims: Sequence[int],
    ridge_alphas: Sequence[float],
    min_train_episodes: int,
    mixture_iterations: int = 5,
    mixture_regularization: float = 1e-3,
) -> tuple[pd.DataFrame, list[FittedSceneRepresentation]]:
    """Fit matched single-layer and learned-mixture models for two scene targets."""

    split_column = str(split["column"])
    masks = {
        "train": rows[split_column].astype(str).to_numpy() == str(split["train_value"]),
        "selection": rows[split_column].astype(str).to_numpy()
        == str(split["selection_value"]),
        "test": rows[split_column].astype(str).to_numpy() == str(split["test_value"]),
    }
    if not all(mask.any() for mask in masks.values()):
        counts = {name: int(mask.sum()) for name, mask in masks.items()}
        raise ValueError(f"Structured scene study requires all three splits; got {counts}")
    supported = _supported_objects(
        targets.presence,
        rows,
        masks["train"],
        int(min_train_episodes),
    )
    candidate_records: list[dict[str, Any]] = []
    best: dict[tuple[str, str], FittedSceneRepresentation] = {}
    for representation, full_values in readouts.items():
        values = np.asarray(full_values, dtype=np.float32)
        if values.ndim != 3 or values.shape[1] != len(layers):
            raise ValueError(
                f"{representation} readouts must have shape row/layer/channel; got "
                f"{values.shape}"
            )
        for requested_dim in sorted(set(int(value) for value in readout_dims)):
            dim = min(requested_dim, values.shape[-1])
            X = values[:, :, :dim]
            for target_name in ["scene_identity", "object_position"]:
                truth = (
                    targets.presence
                    if target_name == "scene_identity"
                    else targets.position
                )
                for alpha in ridge_alphas:
                    for layer_index, layer in enumerate(layers):
                        decoder = fit_scene_decoder(
                            X[:, layer_index],
                            truth,
                            rows,
                            masks["train"],
                            supported,
                            target=target_name,
                            alpha=float(alpha),
                            min_train_episodes=min_train_episodes,
                        )
                        prediction = decoder.predict(X[:, layer_index])
                        metrics = scene_metrics(
                            targets,
                            target_name,
                            prediction,
                            rows,
                            masks,
                            decoder.supported,
                        )
                        weights = np.zeros(len(layers), dtype=np.float64)
                        weights[layer_index] = 1.0
                        record = _candidate_record(
                            representation,
                            "single_layer",
                            target_name,
                            dim,
                            alpha,
                            layers,
                            weights,
                            metrics,
                            selected_layer=int(layer),
                            mixture_iterations=0,
                        )
                        candidate_records.append(record)
                        _consider_best(
                            best,
                            FittedSceneRepresentation(record, decoder, prediction, weights),
                        )

                    decoder, prediction, weights, iterations = fit_layer_mixture(
                        X,
                        truth,
                        rows,
                        masks,
                        supported,
                        target=target_name,
                        alpha=float(alpha),
                        min_train_episodes=min_train_episodes,
                        max_iterations=mixture_iterations,
                        weight_regularization=mixture_regularization,
                    )
                    metrics = scene_metrics(
                        targets,
                        target_name,
                        prediction,
                        rows,
                        masks,
                        decoder.supported,
                    )
                    record = _candidate_record(
                        representation,
                        "learned_layer_mix",
                        target_name,
                        dim,
                        alpha,
                        layers,
                        weights,
                        metrics,
                        selected_layer=None,
                        mixture_iterations=iterations,
                    )
                    candidate_records.append(record)
                    _consider_best(
                        best,
                        FittedSceneRepresentation(record, decoder, prediction, weights),
                    )
    selected = sorted(
        best.values(),
        key=lambda item: (
            str(item.record["representation"]),
            str(item.record["structure"]),
            str(item.record["target"]),
        ),
    )
    return pd.DataFrame.from_records(candidate_records), selected


def fit_scene_decoder(
    X: np.ndarray,
    truth: np.ndarray,
    rows: pd.DataFrame,
    train_mask: np.ndarray,
    supported: np.ndarray,
    *,
    target: str,
    alpha: float,
    min_train_episodes: int,
) -> SceneLinearDecoder:
    """Fit one linear scene decoder while preserving missing-position masks."""

    values = np.asarray(X, dtype=np.float64)
    feature_dim = values.shape[1]
    if target == "scene_identity":
        coefficients = np.zeros((truth.shape[1], feature_dim), dtype=np.float64)
        intercepts = np.zeros(truth.shape[1], dtype=np.float64)
        if supported.any():
            model = Ridge(alpha=float(alpha), solver="lsqr")
            model.fit(values[train_mask], truth[train_mask][:, supported])
            coefficients[supported] = np.atleast_2d(model.coef_)
            intercepts[supported] = np.atleast_1d(model.intercept_)
        return SceneLinearDecoder(target, coefficients, intercepts, supported.copy())
    if target != "object_position":
        raise ValueError(f"Unknown structured scene target {target!r}")
    coefficients = np.full((truth.shape[1], 3, feature_dim), np.nan, dtype=np.float64)
    intercepts = np.full((truth.shape[1], 3), np.nan, dtype=np.float64)
    position_supported = supported.copy()
    for object_index in np.flatnonzero(supported):
        available = train_mask & np.isfinite(truth[:, object_index]).all(axis=1)
        episode_count = rows.loc[available, "trace_id"].astype(str).nunique()
        if episode_count < int(min_train_episodes):
            position_supported[object_index] = False
            continue
        model = Ridge(alpha=float(alpha), solver="lsqr")
        model.fit(values[available], truth[available, object_index])
        coefficients[object_index] = np.atleast_2d(model.coef_)
        intercepts[object_index] = np.atleast_1d(model.intercept_)
    return SceneLinearDecoder(target, coefficients, intercepts, position_supported)


def fit_layer_mixture(
    X: np.ndarray,
    truth: np.ndarray,
    rows: pd.DataFrame,
    masks: Mapping[str, np.ndarray],
    supported: np.ndarray,
    *,
    target: str,
    alpha: float,
    min_train_episodes: int,
    max_iterations: int,
    weight_regularization: float,
) -> tuple[SceneLinearDecoder, np.ndarray, np.ndarray, int]:
    """Alternate between a shared decoder and simplex-constrained layer weights."""

    layer_count = X.shape[1]
    weights = np.full(layer_count, 1.0 / layer_count, dtype=np.float64)
    completed = 0
    for iteration in range(max(1, int(max_iterations))):
        mixed = np.einsum("nlk,l->nk", X, weights, optimize=True)
        decoder = fit_scene_decoder(
            mixed,
            truth,
            rows,
            masks["train"],
            supported,
            target=target,
            alpha=alpha,
            min_train_episodes=min_train_episodes,
        )
        per_layer = np.stack(
            [decoder.predict(X[:, layer_index]) for layer_index in range(layer_count)],
            axis=1,
        )
        updated = _optimize_layer_weights(
            per_layer,
            truth,
            masks["selection"],
            decoder.supported,
            initial=weights,
            regularization=weight_regularization,
        )
        completed = iteration + 1
        if np.max(np.abs(updated - weights)) < 1e-4:
            weights = updated
            break
        weights = updated
    mixed = np.einsum("nlk,l->nk", X, weights, optimize=True)
    decoder = fit_scene_decoder(
        mixed,
        truth,
        rows,
        masks["train"],
        supported,
        target=target,
        alpha=alpha,
        min_train_episodes=min_train_episodes,
    )
    return decoder, decoder.predict(mixed), weights, completed


def _optimize_layer_weights(
    per_layer_prediction: np.ndarray,
    truth: np.ndarray,
    selection_mask: np.ndarray,
    supported: np.ndarray,
    *,
    initial: np.ndarray,
    regularization: float,
) -> np.ndarray:
    selected_prediction = per_layer_prediction[selection_mask]
    selected_truth = truth[selection_mask]
    if selected_truth.ndim == 2:
        valid = np.broadcast_to(supported[None, :], selected_truth.shape)
    else:
        valid = np.broadcast_to(
            supported[None, :, None], selected_truth.shape
        ).copy()
        valid &= np.isfinite(selected_truth)
    if not valid.any():
        return initial
    columns = [selected_prediction[:, layer][valid] for layer in range(len(initial))]
    design = np.stack(columns, axis=1).astype(np.float64, copy=False)
    target = selected_truth[valid].astype(np.float64, copy=False)
    uniform = np.full(len(initial), 1.0 / len(initial), dtype=np.float64)

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        residual = design @ weights - target
        value = float(np.mean(residual**2))
        gradient = 2.0 * (design.T @ residual) / max(1, len(residual))
        if regularization:
            value += float(regularization * np.sum((weights - uniform) ** 2))
            gradient += 2.0 * regularization * (weights - uniform)
        return value, gradient

    result = minimize(
        objective,
        np.asarray(initial, dtype=np.float64),
        method="SLSQP",
        jac=True,
        bounds=[(0.0, 1.0)] * len(initial),
        constraints={"type": "eq", "fun": lambda value: float(np.sum(value) - 1.0)},
        options={"maxiter": 100, "ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.x).all():
        return initial
    weights = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, 1.0)
    return weights / max(1e-12, float(weights.sum()))


def scene_metrics(
    targets: SceneMapTargets,
    target: str,
    prediction: np.ndarray,
    rows: pd.DataFrame,
    masks: Mapping[str, np.ndarray],
    supported: np.ndarray,
) -> dict[str, Any]:
    if target == "scene_identity":
        return _identity_metrics(
            targets.presence, prediction, rows, masks, supported
        )
    return _position_metrics(targets, prediction, rows, masks, supported)


def _candidate_record(
    representation: str,
    structure: str,
    target: str,
    readout_dim: int,
    alpha: float,
    layers: Sequence[int],
    weights: np.ndarray,
    metrics: Mapping[str, Any],
    *,
    selected_layer: int | None,
    mixture_iterations: int,
) -> dict[str, Any]:
    return {
        "representation": representation,
        "structure": structure,
        "target": target,
        "readout_dim": int(readout_dim),
        "ridge_alpha": float(alpha),
        "selected_layer": selected_layer,
        "layers": json.dumps([int(value) for value in layers]),
        "layer_weights": json.dumps([float(value) for value in weights]),
        "mixture_iterations": int(mixture_iterations),
        **dict(metrics),
    }


def _consider_best(
    best: dict[tuple[str, str], FittedSceneRepresentation],
    candidate: FittedSceneRepresentation,
) -> None:
    key = (str(candidate.record["target"]), _variant_name(candidate.record))
    current = best.get(key)
    if current is None or _selection_score(candidate.record) > _selection_score(
        current.record
    ):
        best[key] = candidate


def _variant_name(record: Mapping[str, Any]) -> str:
    return f"{record['representation']}__{record['structure']}"


def _selection_score(record: Mapping[str, Any]) -> float:
    if record["target"] == "object_position":
        value = float(record.get("selection_error_m", float("inf")))
        return -value if np.isfinite(value) else -float("inf")
    value = float(record.get("selection_scene_jaccard", -float("inf")))
    return value if np.isfinite(value) else -float("inf")
