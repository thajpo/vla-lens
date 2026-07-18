"""Vector-aware object geometry probe studies.

This module intentionally sits beside the scalar probe workflow. Geometry targets
are joint objects: XYZ should be evaluated with Euclidean error and rotations with
geodesic error. Treating every coordinate as an independent probe both obscures
that geometry and repeats expensive feature preparation and linear algebra.
"""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from vla_lens.artifacts import LensArtifact, make_artifact_id
from vla_lens.probes.workflow_prepare import _attach_episode_metadata
from vla_lens.selectors import ActivationQuery
from vla_lens.traces import TraceDataset

GEOMETRY_STUDY_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class GeometryTarget:
    """One jointly evaluated geometry target."""

    name: str
    kind: str
    basis: str
    values: np.ndarray
    truth: np.ndarray
    baseline_values: Mapping[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class GeometryStudyResult:
    """Saved study plus its review tables."""

    artifact: LensArtifact | None
    candidates: pd.DataFrame
    selections: pd.DataFrame
    predictions: pd.DataFrame
    timings: Mapping[str, float]


def run_geometry_probe_study(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    *,
    save: bool = True,
) -> GeometryStudyResult:
    """Run and save one staged, vector-aware geometry study."""

    normalized = _normalize_spec(spec)
    timings: dict[str, float] = {}
    started = time.perf_counter()
    all_candidates: list[dict[str, Any]] = []
    all_selections: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    limited_episode_ids = _limited_episode_ids(dataset, normalized.get("limit_episodes"))

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

        metadata_started = time.perf_counter()
        rows = _geometry_metadata_rows(dataset, matrix.rows, cache=True)
        rows = _apply_split_contract(rows, normalized["split"])
        rows, X = _limit_rows_by_episode(rows, matrix.X, normalized.get("limit_episodes"))
        rows = rows.reset_index(drop=True)
        object_column = str(normalized["object_column"])
        present = rows[object_column].notna() & (rows[object_column].astype(str) != "")
        rows = rows.loc[present].reset_index(drop=True)
        X = np.asarray(X[present.to_numpy()], dtype=np.float32)
        geometry = geometry_target_table(dataset, rows, object_column=object_column, cache=True)
        rows = rows.merge(
            geometry,
            on=["trace_id", "timestep", object_column],
            how="inner",
            validate="many_to_one",
        )
        rows["__geometry_object_name"] = rows[object_column]
        source_indices = rows.pop("__feature_row_index").to_numpy(dtype=np.int64)
        X = X[source_indices]
        targets = _geometry_targets(rows)
        finite = _finite_target_mask(targets)
        rows = rows.loc[finite].reset_index(drop=True)
        X = X[finite]
        targets = _slice_targets(targets, finite)
        timings[f"feature:{feature_spec['id']}:targets_seconds"] = (
            time.perf_counter() - metadata_started
        )

        fit_started = time.perf_counter()
        candidates, selections, predictions = _fit_feature_study(
            X,
            rows,
            targets,
            feature_id=str(feature_spec["id"]),
            split_column=str(normalized["split"]["column"]),
            train_value=str(normalized["split"]["train_value"]),
            selection_value=str(normalized["split"]["selection_value"]),
            test_value=str(normalized["split"]["test_value"]),
            sweep_columns=[str(value) for value in feature_spec.get("sweep", ["layer"])],
            pca_dims=[int(value) for value in normalized["probe"]["pca_dims"]],
            ridge_alphas=[float(value) for value in normalized["probe"]["ridge_alphas"]],
            baseline_columns=[str(value) for value in normalized["baseline_columns"]],
        )
        all_candidates.extend(candidates)
        all_selections.extend(selections)
        all_predictions.extend(predictions)
        timings[f"feature:{feature_spec['id']}:fit_seconds"] = time.perf_counter() - fit_started

    timings["total_seconds"] = time.perf_counter() - started
    candidate_frame = pd.DataFrame.from_records(all_candidates)
    selection_frame = pd.DataFrame.from_records(all_selections)
    prediction_frame = pd.DataFrame.from_records(all_predictions)
    artifact = (
        _save_geometry_study(
            dataset,
            normalized,
            candidate_frame,
            selection_frame,
            prediction_frame,
            timings,
        )
        if save
        else None
    )
    return GeometryStudyResult(
        artifact=artifact,
        candidates=candidate_frame,
        selections=selection_frame,
        predictions=prediction_frame,
        timings=timings,
    )


def geometry_target_table(
    dataset: TraceDataset,
    rows: pd.DataFrame,
    *,
    object_column: str,
    cache: bool = True,
) -> pd.DataFrame:
    """Resolve current, initial, previous, and relative poses once per source row."""

    keys = rows[["trace_id", "timestep", object_column]].copy()
    keys["timestep"] = pd.to_numeric(keys["timestep"], errors="coerce").fillna(0).astype(int)
    keys[object_column] = keys[object_column].astype(str)
    keys = keys.drop_duplicates().sort_values(["trace_id", "timestep", object_column])
    key = _target_cache_key(keys, object_column)
    cache_path = dataset.cache_dir() / "geometry_targets" / key / "targets.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    records: list[dict[str, Any]] = []
    for trace_id, trace_rows in keys.groupby("trace_id", sort=False):
        bundle = dataset.bundle(str(trace_id))
        scene = bundle.scene_state
        object_rows = scene.loc[
            scene.get("object_kind", pd.Series("", index=scene.index)).astype(str) == "object"
        ]
        object_indices = {
            str(row["object_name"]): int(row["object_index"])
            for row in object_rows.to_dict("records")
            if row.get("object_name") not in {None, ""}
        }
        positions = bundle.array("scene_object_pos", mmap=True)
        quaternions = bundle.array("scene_object_quat", mmap=True)
        eef_positions = bundle.array("eef_pos", mmap=True)
        eef_quaternions = bundle.array("eef_quat", mmap=True)
        available_timesteps = sorted(int(value) for value in trace_rows["timestep"].unique())
        previous_by_timestep = {
            timestep: available_timesteps[max(0, index - 1)]
            for index, timestep in enumerate(available_timesteps)
        }
        for row in trace_rows.to_dict("records"):
            object_name = str(row[object_column])
            if object_name not in object_indices:
                continue
            object_index = object_indices[object_name]
            timestep = _bounded_index(int(row["timestep"]), int(positions.shape[0]))
            previous_timestep = _bounded_index(
                previous_by_timestep[int(row["timestep"])], int(positions.shape[0])
            )
            position = np.asarray(positions[timestep, object_index], dtype=np.float64)
            initial_position = np.asarray(positions[0, object_index], dtype=np.float64)
            previous_position = np.asarray(
                positions[previous_timestep, object_index], dtype=np.float64
            )
            eef_position = np.asarray(eef_positions[timestep], dtype=np.float64)
            quat = _canonical_quaternion(quaternions[timestep, object_index])
            initial_quat = _canonical_quaternion(quaternions[0, object_index])
            previous_quat = _canonical_quaternion(quaternions[previous_timestep, object_index])
            eef_quat = _canonical_quaternion(eef_quaternions[timestep])
            records.append(
                {
                    "trace_id": str(trace_id),
                    "timestep": int(row["timestep"]),
                    object_column: object_name,
                    "position_world": position.tolist(),
                    "position_initial": initial_position.tolist(),
                    "position_previous": previous_position.tolist(),
                    "position_initial_delta": (position - initial_position).tolist(),
                    "position_previous_delta": (position - previous_position).tolist(),
                    "position_eef_relative": (position - eef_position).tolist(),
                    "position_eef_initial_baseline": (initial_position - eef_position).tolist(),
                    "position_eef_previous_baseline": (previous_position - eef_position).tolist(),
                    "orientation_world_quat": quat.tolist(),
                    "orientation_initial_quat": initial_quat.tolist(),
                    "orientation_previous_quat": previous_quat.tolist(),
                    "orientation_initial_relative_quat": _relative_quaternion(
                        initial_quat, quat
                    ).tolist(),
                    "orientation_previous_relative_quat": _relative_quaternion(
                        previous_quat, quat
                    ).tolist(),
                    "orientation_eef_relative_quat": _relative_quaternion(
                        eef_quat, quat
                    ).tolist(),
                    "orientation_eef_initial_baseline_quat": _relative_quaternion(
                        eef_quat, initial_quat
                    ).tolist(),
                    "orientation_eef_previous_baseline_quat": _relative_quaternion(
                        eef_quat, previous_quat
                    ).tolist(),
                }
            )
    frame = pd.DataFrame.from_records(records)
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path, index=False)
    return frame


def _geometry_metadata_rows(
    dataset: TraceDataset,
    rows: pd.DataFrame,
    *,
    cache: bool,
) -> pd.DataFrame:
    key_columns = [
        column
        for column in ["trace_id", "timestep", "policy_call", "policy_call_index"]
        if column in rows
    ]
    source_keys = rows[key_columns].drop_duplicates().reset_index(drop=True)
    key = _metadata_cache_key(dataset, source_keys)
    cache_path = dataset.cache_dir() / "geometry_base_metadata" / key / "rows.parquet"
    if cache and cache_path.exists():
        attached = pd.read_parquet(cache_path)
    else:
        attached = _attach_episode_metadata(source_keys, dataset)
        if cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            attached.to_parquet(cache_path, index=False)
    overlap = [
        column for column in attached if column in rows and column not in set(key_columns)
    ]
    if overlap:
        attached = attached.drop(columns=overlap)
    return rows.merge(attached, on=key_columns, how="left", validate="many_to_one")


def _apply_split_contract(rows: pd.DataFrame, split: Mapping[str, Any]) -> pd.DataFrame:
    kind = str(split.get("kind") or "existing")
    if kind in {"existing", "heldout_task", "sidecar"}:
        return rows
    if kind != "within_task_episode":
        raise ValueError(f"Unknown geometry split kind: {kind!r}")
    group_column = str(split.get("group_column") or "task_id")
    split_column = str(split.get("column") or "geometry_split")
    if group_column not in rows:
        raise KeyError(f"Within-task geometry split requires {group_column!r}")
    seed = int(split.get("seed", 20260718))
    trace_groups = rows[["trace_id", group_column]].drop_duplicates()
    assignments: dict[str, str] = {}
    for _, group in trace_groups.groupby(group_column, dropna=False, sort=True):
        traces = sorted(
            (str(value) for value in group["trace_id"].unique()),
            key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
        )
        count = len(traces)
        if count < 3:
            continue
        train_stop = max(1, int(np.floor(count * 0.6)))
        selection_stop = max(train_stop + 1, int(np.floor(count * 0.8)))
        selection_stop = min(selection_stop, count - 1)
        for index, trace_id in enumerate(traces):
            if index < train_stop:
                value = str(split["train_value"])
            elif index < selection_stop:
                value = str(split["selection_value"])
            else:
                value = str(split["test_value"])
            assignments[trace_id] = value
    out = rows.copy()
    out[split_column] = out["trace_id"].astype(str).map(assignments)
    return out


def _fit_feature_study(
    X: np.ndarray,
    rows: pd.DataFrame,
    targets: Sequence[GeometryTarget],
    *,
    feature_id: str,
    split_column: str,
    train_value: str,
    selection_value: str,
    test_value: str,
    sweep_columns: Sequence[str],
    pca_dims: Sequence[int],
    ridge_alphas: Sequence[float],
    baseline_columns: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    best: dict[str, dict[str, Any]] = {}
    baseline_cache: dict[str, dict[str, dict[str, Any]]] = {}
    blocks, combined_targets = _target_blocks(targets)
    group_columns = [column for column in sweep_columns if column in rows]
    grouped = [((), rows.index.to_numpy())]
    if group_columns:
        grouped = list(rows.groupby(group_columns, dropna=False, sort=True).indices.items())

    for group_value, indices in grouped:
        group_indices = np.asarray(indices, dtype=np.int64)
        group_rows = rows.iloc[group_indices].reset_index(drop=True)
        group_X = np.asarray(X[group_indices], dtype=np.float64)
        group_Y = np.asarray(combined_targets[group_indices], dtype=np.float64)
        split = group_rows[split_column].astype(str).to_numpy()
        train_mask = split == train_value
        selection_mask = split == selection_value
        test_mask = split == test_value
        if not train_mask.any() or not selection_mask.any() or not test_mask.any():
            continue

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(group_X[train_mask])
        all_scaled = scaler.transform(group_X)
        max_dim = min(max(pca_dims), train_scaled.shape[0] - 1, train_scaled.shape[1])
        if max_dim < 1:
            continue
        projector = PCA(
            n_components=max_dim,
            svd_solver="randomized",
            iterated_power=2,
            random_state=0,
        )
        train_projected = projector.fit_transform(train_scaled)
        all_projected = projector.transform(all_scaled)
        target_mean = group_Y[train_mask].mean(axis=0)
        centered_target = group_Y[train_mask] - target_mean
        group_label = _group_label(group_columns, group_value)
        baseline_key = _row_alignment_key(group_rows, split_column)
        baseline = baseline_cache.get(baseline_key)
        if baseline is None:
            baseline = _baseline_selection(
                group_rows,
                targets=_slice_targets(targets, group_indices),
                blocks=blocks,
                combined_targets=group_Y,
                train_mask=train_mask,
                selection_mask=selection_mask,
                test_mask=test_mask,
                baseline_columns=baseline_columns,
                ridge_alphas=ridge_alphas,
            )
            baseline_cache[baseline_key] = baseline

        for pca_dim in sorted(set(min(int(value), max_dim) for value in pca_dims)):
            Z_train = train_projected[:, :pca_dim]
            gram = Z_train.T @ Z_train
            cross = Z_train.T @ centered_target
            for alpha in ridge_alphas:
                coefficients = np.linalg.solve(
                    gram + float(alpha) * np.eye(pca_dim, dtype=np.float64), cross
                )
                predicted = all_projected[:, :pca_dim] @ coefficients + target_mean
                for target in targets:
                    start, stop = blocks[target.name]
                    target_truth = target.truth[group_indices]
                    target_predicted = predicted[:, start:stop]
                    selection_metrics = _target_metrics(
                        target,
                        target_truth[selection_mask],
                        target_predicted[selection_mask],
                        group_rows.loc[selection_mask, "trace_id"],
                    )
                    test_metrics = _target_metrics(
                        target,
                        target_truth[test_mask],
                        target_predicted[test_mask],
                        group_rows.loc[test_mask, "trace_id"],
                    )
                    record = {
                        "feature_id": feature_id,
                        "feature_group": group_label,
                        "target": target.name,
                        "target_kind": target.kind,
                        "target_basis": target.basis,
                        "pca_dim": int(pca_dim),
                        "ridge_alpha": float(alpha),
                        "train_rows": int(train_mask.sum()),
                        "selection_rows": int(selection_mask.sum()),
                        "test_rows": int(test_mask.sum()),
                        "selection_error": selection_metrics["episode_mean_error"],
                        "test_error": test_metrics["episode_mean_error"],
                        "selection_row_error": selection_metrics["row_mean_error"],
                        "test_row_error": test_metrics["row_mean_error"],
                        "error_unit": selection_metrics["error_unit"],
                        "selection_baseline": baseline[target.name]["name"],
                        "selection_baseline_error": baseline[target.name]["selection_error"],
                        "test_baseline_error": baseline[target.name]["test_error"],
                        "selection_delta": (
                            baseline[target.name]["selection_error"]
                            - selection_metrics["episode_mean_error"]
                        ),
                        "test_delta": (
                            baseline[target.name]["test_error"]
                            - test_metrics["episode_mean_error"]
                        ),
                        "selection_metrics": selection_metrics,
                        "test_metrics": test_metrics,
                    }
                    candidates.append(record)
                    best_key = target.name
                    current = best.get(best_key)
                    if current is None or record["selection_error"] < current["record"][
                        "selection_error"
                    ]:
                        best[best_key] = {
                            "record": record,
                            "rows": group_rows,
                            "truth": target_truth,
                            "predicted": target_predicted,
                            "selection_mask": selection_mask,
                            "test_mask": test_mask,
                            "baseline": baseline[target.name],
                        }

    for target_name, selected in best.items():
        record = dict(selected["record"])
        record["selection_split"] = selection_value
        record["test_split"] = test_value
        selections.append(record)
        target = next(item for item in targets if item.name == target_name)
        for split_name, mask in [
            (selection_value, selected["selection_mask"]),
            (test_value, selected["test_mask"]),
        ]:
            split_rows = selected["rows"].loc[mask].reset_index(drop=True)
            truth = selected["truth"][mask]
            predicted = selected["predicted"][mask]
            errors = _target_row_errors(target, truth, predicted)
            physical_prediction = (
                predicted
                if target.kind == "position"
                else _decode_orientation(predicted, target.basis)
            )
            for index, source in split_rows.iterrows():
                predictions.append(
                    {
                        "feature_id": feature_id,
                        "feature_group": record["feature_group"],
                        "target": target_name,
                        "target_kind": target.kind,
                        "target_basis": target.basis,
                        "split": split_name,
                        "trace_id": str(source["trace_id"]),
                        "episode_id": source.get("episode_id"),
                        "timestep": int(source["timestep"]),
                        "policy_call_index": _optional_int(source.get("policy_call_index")),
                        "object_name": source.get("__geometry_object_name"),
                        "target_value": np.asarray(truth[index]).tolist(),
                        "prediction_value": np.asarray(physical_prediction[index]).tolist(),
                        "prediction_representation": np.asarray(predicted[index]).tolist(),
                        "error": float(errors[index]),
                        "error_unit": record["error_unit"],
                    }
                )
    return candidates, selections, predictions


def _baseline_selection(
    rows: pd.DataFrame,
    *,
    targets: Sequence[GeometryTarget],
    blocks: Mapping[str, tuple[int, int]],
    combined_targets: np.ndarray,
    train_mask: np.ndarray,
    selection_mask: np.ndarray,
    test_mask: np.ndarray,
    baseline_columns: Sequence[str],
    ridge_alphas: Sequence[float],
) -> dict[str, dict[str, Any]]:
    predictions: dict[str, np.ndarray] = {
        "train_mean": np.repeat(
            combined_targets[train_mask].mean(axis=0, keepdims=True), len(rows), axis=0
        )
    }
    available = [column for column in baseline_columns if column in rows]
    if available:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
        encoded_train = encoder.fit_transform(rows.loc[train_mask, available].astype(str))
        encoded_all = encoder.transform(rows[available].astype(str))
        mean = combined_targets[train_mask].mean(axis=0)
        centered = combined_targets[train_mask] - mean
        gram = encoded_train.T @ encoded_train
        cross = encoded_train.T @ centered
        for alpha in ridge_alphas:
            coefficients = np.linalg.solve(
                gram + float(alpha) * np.eye(gram.shape[0], dtype=np.float64), cross
            )
            predictions[f"metadata_ridge_alpha={alpha:g}"] = encoded_all @ coefficients + mean

    selected: dict[str, dict[str, Any]] = {}
    for target in targets:
        start, stop = blocks[target.name]
        target_truth = target.truth
        candidates = {
            name: values[:, start:stop] for name, values in predictions.items()
        }
        candidates.update(target.baseline_values)
        best: dict[str, Any] | None = None
        for name, values in candidates.items():
            selection_metrics = _target_metrics(
                target,
                target_truth[selection_mask],
                np.asarray(values)[selection_mask],
                rows.loc[selection_mask, "trace_id"],
            )
            test_metrics = _target_metrics(
                target,
                target_truth[test_mask],
                np.asarray(values)[test_mask],
                rows.loc[test_mask, "trace_id"],
            )
            record = {
                "name": name,
                "selection_error": selection_metrics["episode_mean_error"],
                "test_error": test_metrics["episode_mean_error"],
            }
            if best is None or record["selection_error"] < best["selection_error"]:
                best = record
        assert best is not None
        selected[target.name] = best
    return selected


def _geometry_targets(rows: pd.DataFrame) -> list[GeometryTarget]:
    position_world = _vectors(rows["position_world"], 3)
    position_initial = _vectors(rows["position_initial"], 3)
    position_previous = _vectors(rows["position_previous"], 3)
    position_delta = _vectors(rows["position_initial_delta"], 3)
    position_previous_delta = _vectors(rows["position_previous_delta"], 3)
    position_eef = _vectors(rows["position_eef_relative"], 3)
    eef_initial = _vectors(rows["position_eef_initial_baseline"], 3)
    eef_previous = _vectors(rows["position_eef_previous_baseline"], 3)
    world_quat = _quaternion_rows(rows["orientation_world_quat"])
    initial_quat = _quaternion_rows(rows["orientation_initial_quat"])
    previous_quat = _quaternion_rows(rows["orientation_previous_quat"])
    initial_relative = _quaternion_rows(rows["orientation_initial_relative_quat"])
    previous_relative = _quaternion_rows(rows["orientation_previous_relative_quat"])
    eef_relative = _quaternion_rows(rows["orientation_eef_relative_quat"])
    eef_initial_quat = _quaternion_rows(rows["orientation_eef_initial_baseline_quat"])
    eef_previous_quat = _quaternion_rows(rows["orientation_eef_previous_baseline_quat"])

    targets = [
        GeometryTarget(
            "position_world",
            "position",
            "xyz",
            position_world,
            position_world,
            {"initial_pose": position_initial, "previous_call_pose": position_previous},
        ),
        GeometryTarget(
            "position_initial_delta",
            "position",
            "xyz",
            position_delta,
            position_delta,
            {
                "zero_displacement": np.zeros_like(position_delta),
                "previous_call_pose": position_previous - position_initial,
            },
        ),
        GeometryTarget(
            "position_previous_delta",
            "position",
            "xyz",
            position_previous_delta,
            position_previous_delta,
            {"zero_change": np.zeros_like(position_previous_delta)},
        ),
        GeometryTarget(
            "position_eef_relative",
            "position",
            "xyz",
            position_eef,
            position_eef,
            {"initial_pose": eef_initial, "previous_call_pose": eef_previous},
        ),
    ]
    for basis in ["quaternion", "rotation_6d", "rotation_vector", "euler_sincos"]:
        targets.append(
            GeometryTarget(
                f"orientation_world_{basis}",
                "orientation",
                basis,
                _encode_orientation(world_quat, basis),
                world_quat,
                {
                    "initial_pose": _encode_orientation(initial_quat, basis),
                    "previous_call_pose": _encode_orientation(previous_quat, basis),
                },
            )
        )
    identity = np.repeat(np.array([[0.0, 0.0, 0.0, 1.0]]), len(rows), axis=0)
    targets.extend(
        [
            GeometryTarget(
                "orientation_initial_relative_rotation_6d",
                "orientation",
                "rotation_6d",
                _encode_orientation(initial_relative, "rotation_6d"),
                initial_relative,
                {
                    "identity_rotation": _encode_orientation(identity, "rotation_6d"),
                    "previous_call_pose": _encode_orientation(
                        _relative_quaternion_rows(initial_quat, previous_quat), "rotation_6d"
                    ),
                },
            ),
            GeometryTarget(
                "orientation_previous_relative_rotation_6d",
                "orientation",
                "rotation_6d",
                _encode_orientation(previous_relative, "rotation_6d"),
                previous_relative,
                {"identity_rotation": _encode_orientation(identity, "rotation_6d")},
            ),
            GeometryTarget(
                "orientation_eef_relative_rotation_6d",
                "orientation",
                "rotation_6d",
                _encode_orientation(eef_relative, "rotation_6d"),
                eef_relative,
                {
                    "initial_pose": _encode_orientation(eef_initial_quat, "rotation_6d"),
                    "previous_call_pose": _encode_orientation(
                        eef_previous_quat, "rotation_6d"
                    ),
                },
            ),
        ]
    )
    return targets


def _target_metrics(
    target: GeometryTarget,
    truth: np.ndarray,
    predicted: np.ndarray,
    trace_ids: pd.Series,
) -> dict[str, Any]:
    errors = _target_row_errors(target, truth, predicted)
    frame = pd.DataFrame({"trace_id": np.asarray(trace_ids).astype(str), "error": errors})
    episode_errors = frame.groupby("trace_id", sort=False)["error"].mean()
    unit = "meters" if target.kind == "position" else "degrees"
    metrics: dict[str, Any] = {
        "row_mean_error": float(np.mean(errors)),
        "row_median_error": float(np.median(errors)),
        "episode_mean_error": float(episode_errors.mean()),
        "episode_median_error": float(episode_errors.median()),
        "episodes": int(len(episode_errors)),
        "rows": int(len(errors)),
        "error_unit": unit,
    }
    if target.kind == "position":
        metrics["axis_mae"] = np.mean(np.abs(predicted - truth), axis=0).tolist()
    return metrics


def _target_row_errors(
    target: GeometryTarget,
    truth: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    if target.kind == "position":
        return np.linalg.norm(np.asarray(predicted) - np.asarray(truth), axis=1)
    predicted_quat = _decode_orientation(np.asarray(predicted), target.basis)
    relative = Rotation.from_quat(predicted_quat).inv() * Rotation.from_quat(truth)
    return np.degrees(relative.magnitude())


def _target_blocks(
    targets: Sequence[GeometryTarget],
) -> tuple[dict[str, tuple[int, int]], np.ndarray]:
    blocks: dict[str, tuple[int, int]] = {}
    values: list[np.ndarray] = []
    offset = 0
    for target in targets:
        width = int(target.values.shape[1])
        blocks[target.name] = (offset, offset + width)
        values.append(target.values)
        offset += width
    return blocks, np.concatenate(values, axis=1)


def _encode_orientation(quaternions: np.ndarray, basis: str) -> np.ndarray:
    rotations = Rotation.from_quat(_canonical_quaternion_rows(quaternions))
    if basis == "quaternion":
        return _canonical_quaternion_rows(rotations.as_quat())
    if basis == "rotation_6d":
        matrices = rotations.as_matrix()
        return np.concatenate([matrices[:, :, 0], matrices[:, :, 1]], axis=1)
    if basis == "rotation_vector":
        return rotations.as_rotvec()
    if basis == "euler_sincos":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Gimbal lock detected.*")
            euler = rotations.as_euler("xyz")
        return np.concatenate([np.sin(euler), np.cos(euler)], axis=1)
    raise ValueError(f"Unknown orientation basis: {basis!r}")


def _decode_orientation(values: np.ndarray, basis: str) -> np.ndarray:
    if basis == "quaternion":
        return _canonical_quaternion_rows(values)
    if basis == "rotation_vector":
        return _canonical_quaternion_rows(Rotation.from_rotvec(values).as_quat())
    if basis == "euler_sincos":
        euler = np.arctan2(values[:, :3], values[:, 3:6])
        return _canonical_quaternion_rows(Rotation.from_euler("xyz", euler).as_quat())
    if basis == "rotation_6d":
        first = _safe_normalize(values[:, :3], fallback=np.array([1.0, 0.0, 0.0]))
        second_raw = values[:, 3:6] - np.sum(values[:, 3:6] * first, axis=1)[:, None] * first
        second = _safe_normalize(second_raw, fallback=np.array([0.0, 1.0, 0.0]))
        third = np.cross(first, second)
        matrices = np.stack([first, second, third], axis=2)
        return _canonical_quaternion_rows(Rotation.from_matrix(matrices).as_quat())
    raise ValueError(f"Unknown orientation basis: {basis!r}")


def _save_geometry_study(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    candidates: pd.DataFrame,
    selections: pd.DataFrame,
    predictions: pd.DataFrame,
    timings: Mapping[str, float],
) -> LensArtifact:
    artifact_id = make_artifact_id(str(spec["name"]), "geometry_probe_study")
    relative_dir = Path("artifacts") / artifact_id
    outputs = {
        "candidates": str(relative_dir / "candidates.parquet"),
        "selections": str(relative_dir / "selections.parquet"),
        "predictions": str(relative_dir / "predictions.parquet"),
    }
    display_records = [] if selections.empty else _json_records(selections)
    artifact = LensArtifact(
        artifact_id=artifact_id,
        artifact_type="geometry_probe_study",
        name=str(spec["name"]),
        group_id="geometry_probe_studies",
        scope="dataset",
        selector={"features": spec["features"], "object_column": spec["object_column"]},
        method={
            "workflow": "run_geometry_probe_study",
            "schema_version": GEOMETRY_STUDY_SCHEMA_VERSION,
            "research": {
                key: spec[key]
                for key in ["question", "hypothesis_family", "intended_claim"]
                if spec.get(key)
            },
            "split": spec["split"],
            "probe": spec["probe"],
            "baseline_columns": spec["baseline_columns"],
            "target_contract": {
                "position_metric": "episode-weighted mean Euclidean distance",
                "orientation_metric": "episode-weighted mean SO(3) geodesic angle",
                "orientation_bases": [
                    "canonical quaternion xyzw",
                    "6D rotation",
                    "rotation vector",
                    "Euler sine/cosine",
                ],
            },
            "timings_seconds": dict(timings),
            "outputs": outputs,
        },
        metrics={
            "candidate_count": int(len(candidates)),
            "selection_count": int(len(selections)),
            "prediction_count": int(len(predictions)),
            "total_seconds": float(timings.get("total_seconds", 0.0)),
        },
        display={
            "kind": "geometry_probe_study",
            "status": "exploratory",
            "selections": display_records,
        },
        tags=("probe", "geometry", "exploratory"),
        source_trace_ids=tuple(
            sorted(str(value) for value in predictions.get("trace_id", pd.Series()).unique())
        ),
    )
    saved = dataset.save_artifact(artifact)
    artifact_dir = dataset._dataset_artifact_root() / relative_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(artifact_dir / "candidates.parquet", index=False)
    selections.to_parquet(artifact_dir / "selections.parquet", index=False)
    predictions.to_parquet(artifact_dir / "predictions.parquet", index=False)
    return saved


def _normalize_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(spec)
    normalized.setdefault("name", "PI0.5 object geometry study")
    normalized.setdefault("object_column", "primary_target_object")
    normalized.setdefault(
        "split",
        {
            "column": "split",
            "train_value": "train",
            "selection_value": "val_heldout_task",
            "test_value": "test_heldout_task",
        },
    )
    normalized.setdefault(
        "probe", {"pca_dims": [64, 128], "ridge_alphas": [0.1, 1.0, 10.0, 100.0]}
    )
    if "baseline_columns" not in normalized:
        normalized["baseline_columns"] = normalized.get(
            "baseline",
            [
                "benchmark",
                "task_id",
                "scene_family",
                "primary_target_object",
                "task_phase",
                "policy_call_index",
            ],
        )
    features = normalized.get("features")
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)) or not features:
        raise ValueError("Geometry study requires a non-empty features list")
    normalized["features"] = [dict(value) for value in features]
    for index, feature in enumerate(normalized["features"]):
        feature.setdefault("id", f"feature_{index}")
        feature.setdefault("sweep", ["layer"])
    return normalized


def _activation_query(spec: Mapping[str, Any]) -> ActivationQuery:
    return ActivationQuery(
        episodes=dict(spec.get("episodes") or {}),
        name=spec.get("name"),
        module=spec.get("module"),
        layers=spec.get("layers"),
        tensor_type=spec.get("tensor_type"),
        token_kind=spec.get("token_kind"),
        timesteps=spec.get("timesteps", "all"),
        policy_calls=spec.get("policy_calls", "all"),
        generation_step=spec.get("generation_step"),
        reduce_tokens=spec.get("reduction", "mean"),
        dtype=str(spec.get("dtype", "float32")),
    )


def _limit_rows_by_episode(
    rows: pd.DataFrame,
    X: np.ndarray,
    limit: Any,
) -> tuple[pd.DataFrame, np.ndarray]:
    if limit in {None, 0, ""}:
        out = rows.reset_index(drop=True).reset_index(names="__feature_row_index")
        return out, X
    requested = int(limit)
    episode_rows = rows[["trace_id", "split"]].drop_duplicates()
    split_values = sorted(str(value) for value in episode_rows["split"].dropna().unique())
    per_split = max(1, requested // max(1, len(split_values)))
    episode_ids: list[str] = []
    for split_value in split_values:
        matches = episode_rows.loc[
            episode_rows["split"].astype(str) == split_value, "trace_id"
        ]
        episode_ids.extend(sorted(str(value) for value in matches.unique())[:per_split])
    if len(episode_ids) < requested:
        remaining = sorted(
            set(str(value) for value in rows["trace_id"].unique()) - set(episode_ids)
        )
        episode_ids.extend(remaining[: requested - len(episode_ids)])
    mask = rows["trace_id"].astype(str).isin(episode_ids).to_numpy()
    out = rows.loc[mask].reset_index(drop=True).reset_index(names="__feature_row_index")
    return out, X[mask]


def _limited_episode_ids(dataset: TraceDataset, limit: Any) -> list[str] | None:
    if limit in {None, 0, ""}:
        return None
    requested = int(limit)
    episodes = dataset.episode_index[["trace_id"]].copy()
    split_path = dataset.root / "probe_splits.csv"
    if split_path.exists():
        splits = pd.read_csv(split_path, usecols=["trace_id", "split"])
        episodes = episodes.merge(splits, on="trace_id", how="left")
    else:
        episodes["split"] = "all"
    split_values = sorted(str(value) for value in episodes["split"].dropna().unique())
    per_split = max(1, requested // max(1, len(split_values)))
    selected: list[str] = []
    for split_value in split_values:
        matches = episodes.loc[
            episodes["split"].astype(str) == split_value, "trace_id"
        ]
        selected.extend(sorted(str(value) for value in matches.unique())[:per_split])
    if len(selected) < requested:
        remaining = sorted(
            set(str(value) for value in episodes["trace_id"].unique()) - set(selected)
        )
        selected.extend(remaining[: requested - len(selected)])
    return selected[:requested]


def _finite_target_mask(targets: Sequence[GeometryTarget]) -> np.ndarray:
    mask = np.ones(len(targets[0].values), dtype=bool)
    for target in targets:
        mask &= np.isfinite(target.values).all(axis=1)
        mask &= np.isfinite(target.truth).all(axis=1)
    return mask


def _slice_targets(
    targets: Sequence[GeometryTarget], indices: np.ndarray
) -> list[GeometryTarget]:
    return [
        GeometryTarget(
            target.name,
            target.kind,
            target.basis,
            target.values[indices],
            target.truth[indices],
            {name: values[indices] for name, values in target.baseline_values.items()},
        )
        for target in targets
    ]


def _vectors(values: pd.Series, width: int) -> np.ndarray:
    return np.stack([np.asarray(value, dtype=np.float64).reshape(width) for value in values])


def _quaternion_rows(values: pd.Series) -> np.ndarray:
    return _canonical_quaternion_rows(_vectors(values, 4))


def _canonical_quaternion(value: Any) -> np.ndarray:
    return _canonical_quaternion_rows(np.asarray(value, dtype=np.float64).reshape(1, 4))[0]


def _canonical_quaternion_rows(values: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(values, dtype=np.float64).reshape(-1, 4).copy()
    norms = np.linalg.norm(quaternions, axis=1)
    invalid = (~np.isfinite(norms)) | (norms < 1e-12)
    quaternions[invalid] = np.array([0.0, 0.0, 0.0, 1.0])
    norms[invalid] = 1.0
    quaternions /= norms[:, None]
    quaternions[quaternions[:, 3] < 0] *= -1.0
    return quaternions


def _relative_quaternion(reference: np.ndarray, value: np.ndarray) -> np.ndarray:
    relative = Rotation.from_quat(reference).inv() * Rotation.from_quat(value)
    return _canonical_quaternion(relative.as_quat())


def _relative_quaternion_rows(reference: np.ndarray, value: np.ndarray) -> np.ndarray:
    relative = Rotation.from_quat(reference).inv() * Rotation.from_quat(value)
    return _canonical_quaternion_rows(relative.as_quat())


def _safe_normalize(values: np.ndarray, *, fallback: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    norms = np.linalg.norm(out, axis=1)
    invalid = (~np.isfinite(norms)) | (norms < 1e-12)
    out[invalid] = fallback
    norms[invalid] = np.linalg.norm(fallback)
    return out / norms[:, None]


def _target_cache_key(rows: pd.DataFrame, object_column: str) -> str:
    payload = {
        "schema": GEOMETRY_STUDY_SCHEMA_VERSION,
        "object_column": object_column,
        "rows": rows.to_dict("records"),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _metadata_cache_key(dataset: TraceDataset, rows: pd.DataFrame) -> str:
    row_columns = [
        column
        for column in [
            "trace_id",
            "timestep",
            "policy_call_index",
        ]
        if column in rows
    ]
    row_hash = pd.util.hash_pandas_object(rows[row_columns], index=False).to_numpy(
        dtype=np.uint64
    )
    relevant_types = {
        "pi05_interaction_metrics",
        "pi05_object_flow",
        "pi05_policy_call_labels",
    }
    artifact_index = dataset.artifact_index
    relevant_artifacts: list[str] = []
    if not artifact_index.empty and "artifact_type" in artifact_index:
        relevant_artifacts = sorted(
            str(value)
            for value in artifact_index.loc[
                artifact_index["artifact_type"].astype(str).isin(relevant_types), "artifact_id"
            ].dropna()
        )
    split_path = dataset.root / "probe_splits.csv"
    payload = {
        "schema": GEOMETRY_STUDY_SCHEMA_VERSION,
        "rows": hashlib.sha256(row_hash.tobytes()).hexdigest(),
        "artifacts": relevant_artifacts,
        "split_mtime_ns": split_path.stat().st_mtime_ns if split_path.exists() else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def _row_alignment_key(rows: pd.DataFrame, split_column: str) -> str:
    columns = [
        column
        for column in ["trace_id", "timestep", "__geometry_object_name", split_column]
        if column in rows
    ]
    hashed = pd.util.hash_pandas_object(rows[columns], index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()[:20]


def _bounded_index(index: int, count: int) -> int:
    return max(0, min(int(index), max(0, count - 1)))


def _group_label(columns: Sequence[str], value: Any) -> str:
    if not columns:
        return "all"
    values = value if isinstance(value, tuple) else (value,)
    return ",".join(
        f"{column}={item}" for column, item in zip(columns, values, strict=False)
    )


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


__all__ = [
    "GEOMETRY_STUDY_SCHEMA_VERSION",
    "GeometryStudyResult",
    "GeometryTarget",
    "geometry_target_table",
    "run_geometry_probe_study",
]
