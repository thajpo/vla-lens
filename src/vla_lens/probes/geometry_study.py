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
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from vla_lens.artifacts import LensArtifact, make_artifact_id
from vla_lens.cache import CacheBuildMetadata, CacheManager, fingerprint_payload
from vla_lens.probes.workflow_prepare import _attach_episode_metadata
from vla_lens.selectors import ActivationQuery
from vla_lens.traces import TraceDataset

GEOMETRY_STUDY_SCHEMA_VERSION = 4
LONG_CACHE_BUILD_TIMEOUT_S = 2 * 60 * 60
GEOMETRY_TARGET_NAMES = (
    "position_world",
    "position_initial_delta",
    "position_previous_delta",
    "position_eef_relative",
    "orientation_world_quaternion",
    "orientation_world_rotation_6d",
    "orientation_world_rotation_vector",
    "orientation_world_euler_sincos",
    "orientation_initial_relative_rotation_6d",
    "orientation_previous_relative_rotation_6d",
    "orientation_eef_relative_rotation_6d",
)


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


@dataclass(frozen=True, slots=True)
class GeometryReadoutState:
    """Replayable train-fitted transform and readout arrays."""

    readout_id: str
    contract: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]


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
    all_fitted_readouts: dict[str, GeometryReadoutState] = {}
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

        metadata_started = time.perf_counter()
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
        object_column = str(normalized["object_column"])
        rows, X = _align_labeled_geometry_rows(
            dataset,
            rows,
            X,
            object_column=object_column,
        )
        rows["__geometry_object_name"] = rows[object_column]
        targets = _select_geometry_targets(
            _geometry_targets(rows), normalized.get("targets")
        )
        finite = _finite_target_mask(targets)
        rows = rows.loc[finite].reset_index(drop=True)
        X = X[finite]
        targets = _slice_targets(targets, finite)
        source_trace_ids.update(rows["trace_id"].astype(str).unique())
        timings[f"feature:{feature_spec['id']}:targets_seconds"] = (
            time.perf_counter() - metadata_started
        )

        fit_started = time.perf_counter()
        candidates, selections, predictions, fitted_readouts = _fit_feature_study(
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
            models=[str(value) for value in normalized["probe"]["models"]],
            mlp_seed=int(normalized["probe"]["mlp_random_state"]),
            mlp_max_iter=int(normalized["probe"]["mlp_max_iter"]),
            bootstrap_samples=int(normalized["probe"]["bootstrap_samples"]),
            bootstrap_group_column=str(normalized["probe"]["bootstrap_group_column"]),
            baseline_columns=[str(value) for value in normalized["baseline_columns"]],
        )
        all_candidates.extend(candidates)
        all_selections.extend(selections)
        all_predictions.extend(predictions)
        all_fitted_readouts.update(
            {readout.readout_id: readout for readout in fitted_readouts}
        )
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
            tuple(all_fitted_readouts.values()),
            timings,
            source_trace_ids,
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


def _align_labeled_geometry_rows(
    dataset: TraceDataset,
    rows: pd.DataFrame,
    features: np.ndarray,
    *,
    object_column: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    present = rows[object_column].notna() & rows[object_column].astype(str).ne("")
    labeled_rows = rows.loc[present].reset_index(drop=True)
    geometry = geometry_target_table(
        dataset,
        labeled_rows,
        object_column=object_column,
        cache=True,
    )
    labeled_rows = labeled_rows.merge(
        geometry,
        on=["trace_id", "timestep", object_column],
        how="inner",
        validate="many_to_one",
    )
    source_indices = labeled_rows.pop("__feature_row_index").to_numpy(dtype=np.int64)
    return labeled_rows, np.asarray(features[source_indices], dtype=np.float32)


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
    if cache:
        return _managed_parquet_cache(
            dataset,
            namespace="geometry_targets",
            key=key,
            filename="targets.parquet",
            builder=lambda: geometry_target_table(
                dataset,
                rows,
                object_column=object_column,
                cache=False,
            ),
        )

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
        actions = bundle.actions(mmap=True)
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
            previous_eef_position = np.asarray(
                eef_positions[previous_timestep], dtype=np.float64
            )
            quat = _canonical_quaternion(quaternions[timestep, object_index])
            initial_quat = _canonical_quaternion(quaternions[0, object_index])
            previous_quat = _canonical_quaternion(quaternions[previous_timestep, object_index])
            eef_quat = _canonical_quaternion(eef_quaternions[timestep])
            previous_eef_quat = _canonical_quaternion(eef_quaternions[previous_timestep])
            action_start = min(previous_timestep, timestep)
            action_stop = max(previous_timestep, timestep)
            action_segment = np.asarray(actions[action_start:action_stop], dtype=np.float64)
            if action_segment.size:
                action_mean = action_segment.mean(axis=0)
                action_sum = action_segment.sum(axis=0)
                action_std = action_segment.std(axis=0)
            else:
                action_width = int(actions.shape[-1])
                action_mean = np.zeros(action_width, dtype=np.float64)
                action_sum = np.zeros(action_width, dtype=np.float64)
                action_std = np.zeros(action_width, dtype=np.float64)
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
                    "eef_position_previous_delta": (
                        eef_position - previous_eef_position
                    ).tolist(),
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
                    "eef_orientation_previous_relative_quat": _relative_quaternion(
                        previous_eef_quat, eef_quat
                    ).tolist(),
                    "executed_action_mean": action_mean.tolist(),
                    "executed_action_sum": action_sum.tolist(),
                    "executed_action_std": action_std.tolist(),
                    "is_first_policy_call": bool(timestep == previous_timestep),
                }
            )
    frame = pd.DataFrame.from_records(records)
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
    if cache:
        attached = _managed_parquet_cache(
            dataset,
            namespace="geometry_base_metadata",
            key=key,
            filename="rows.parquet",
            builder=lambda: _attach_episode_metadata(source_keys, dataset),
        )
    else:
        attached = _attach_episode_metadata(source_keys, dataset)
    overlap = [
        column for column in attached if column in rows and column not in set(key_columns)
    ]
    if overlap:
        attached = attached.drop(columns=overlap)
    return rows.merge(attached, on=key_columns, how="left", validate="many_to_one")


def _managed_parquet_cache(
    dataset: TraceDataset,
    *,
    namespace: str,
    key: str,
    filename: str,
    builder: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """Build one small derived table once and adopt valid pre-manager entries."""

    manager = CacheManager(dataset.cache_dir())
    recipe = {
        "schema_version": GEOMETRY_STUDY_SCHEMA_VERSION,
        "kind": namespace,
        "cache_key": key,
        "filename": filename,
    }

    def valid(path: Path) -> bool:
        try:
            pd.read_parquet(path / filename)
        except (OSError, ValueError, KeyError):
            return False
        return True

    def metadata(path: Path) -> CacheBuildMetadata:
        frame = pd.read_parquet(path / filename)
        return CacheBuildMetadata(
            shape=tuple(int(value) for value in frame.shape),
            axes=("row", "column"),
            row_count=int(len(frame)),
            rebuild={"kind": namespace, "cache_key": key},
        )

    def build(path: Path) -> CacheBuildMetadata:
        frame = builder()
        frame.to_parquet(path / filename, index=False)
        return CacheBuildMetadata(
            shape=tuple(int(value) for value in frame.shape),
            axes=("row", "column"),
            row_count=int(len(frame)),
            rebuild={"kind": namespace, "cache_key": key},
        )

    cache_path, _, _ = manager.get_or_build(
        namespace=namespace,
        key=key,
        recipe=recipe,
        source_fingerprint=fingerprint_payload({"cache_key": key}),
        builder=build,
        validator=valid,
        legacy_metadata=metadata,
        timeout_s=LONG_CACHE_BUILD_TIMEOUT_S,
    )
    with manager.lock_for(namespace, key):
        return pd.read_parquet(cache_path / filename)


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
    models: Sequence[str],
    mlp_seed: int,
    mlp_max_iter: int,
    bootstrap_samples: int,
    bootstrap_group_column: str,
    baseline_columns: Sequence[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[GeometryReadoutState],
]:
    candidates: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    best: dict[tuple[str, str], dict[str, Any]] = {}
    fitted_states: dict[str, GeometryReadoutState] = {}
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
            Z_all = all_projected[:, :pca_dim]
            model_fits: list[dict[str, Any]] = []
            if "ridge" in models:
                gram = Z_train.T @ Z_train
                cross = Z_train.T @ centered_target
                for alpha in ridge_alphas:
                    coefficients = np.linalg.solve(
                        gram + float(alpha) * np.eye(pca_dim, dtype=np.float64), cross
                    )
                    predicted = Z_all @ coefficients + target_mean
                    readout_id = _geometry_readout_id(
                        feature_id, group_label, "ridge", pca_dim, float(alpha)
                    )
                    state = _ridge_readout_state(
                        readout_id=readout_id,
                        feature_id=feature_id,
                        feature_group=group_label,
                        pca_dim=pca_dim,
                        alpha=float(alpha),
                        scaler=scaler,
                        projector=projector,
                        coefficients=coefficients,
                        target_mean=target_mean,
                        blocks=blocks,
                        train_rows=int(train_mask.sum()),
                    )
                    model_fits.append(
                        {
                            "model": "ridge",
                            "ridge_alpha": float(alpha),
                            "predicted": predicted,
                            "readout_id": readout_id,
                            "state": state,
                            "converged": True,
                            "n_iter": None,
                            "final_loss": None,
                            "report_all_test_candidates": len(models) == 1,
                        }
                    )
            if "mlp" in models:
                network = MLPRegressor(
                    hidden_layer_sizes=(64,),
                    alpha=1e-4,
                    max_iter=mlp_max_iter,
                    random_state=mlp_seed,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    network.fit(Z_train, group_Y[train_mask])
                predicted = np.asarray(network.predict(Z_all), dtype=np.float64)
                readout_id = _geometry_readout_id(
                    feature_id, group_label, "mlp", pca_dim, None
                )
                state = _mlp_readout_state(
                    readout_id=readout_id,
                    feature_id=feature_id,
                    feature_group=group_label,
                    pca_dim=pca_dim,
                    scaler=scaler,
                    projector=projector,
                    network=network,
                    blocks=blocks,
                    train_rows=int(train_mask.sum()),
                    random_state=mlp_seed,
                    max_iter=mlp_max_iter,
                )
                model_fits.append(
                    {
                        "model": "mlp",
                        "ridge_alpha": None,
                        "predicted": predicted,
                        "readout_id": readout_id,
                        "state": state,
                        "converged": int(network.n_iter_) < mlp_max_iter,
                        "n_iter": int(network.n_iter_),
                        "final_loss": float(network.loss_),
                        "report_all_test_candidates": False,
                    }
                )

            for fitted in model_fits:
                fitted_states[str(fitted["readout_id"])] = fitted["state"]
                predicted = np.asarray(fitted["predicted"], dtype=np.float64)
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
                    report_test = bool(fitted["report_all_test_candidates"])
                    test_metrics = (
                        _target_metrics(
                            target,
                            target_truth[test_mask],
                            target_predicted[test_mask],
                            group_rows.loc[test_mask, "trace_id"],
                        )
                        if report_test
                        else None
                    )
                    record = {
                        "feature_id": feature_id,
                        "feature_group": group_label,
                        "model": fitted["model"],
                        "readout_id": fitted["readout_id"],
                        "target": target.name,
                        "target_kind": target.kind,
                        "target_basis": target.basis,
                        "pca_dim": int(pca_dim),
                        "ridge_alpha": fitted["ridge_alpha"],
                        "mlp_hidden_units": 64 if fitted["model"] == "mlp" else None,
                        "mlp_alpha": 1e-4 if fitted["model"] == "mlp" else None,
                        "max_iter": mlp_max_iter if fitted["model"] == "mlp" else None,
                        "random_state": mlp_seed if fitted["model"] == "mlp" else None,
                        "converged": fitted["converged"],
                        "n_iter": fitted["n_iter"],
                        "final_loss": fitted["final_loss"],
                        "train_rows": int(train_mask.sum()),
                        "selection_rows": int(selection_mask.sum()),
                        "test_rows": int(test_mask.sum()),
                        "selection_error": selection_metrics["episode_mean_error"],
                        "test_error": (
                            test_metrics["episode_mean_error"] if test_metrics else None
                        ),
                        "selection_row_error": selection_metrics["row_mean_error"],
                        "test_row_error": test_metrics["row_mean_error"] if test_metrics else None,
                        "error_unit": selection_metrics["error_unit"],
                        "selection_baseline": baseline[target.name]["name"],
                        "selection_baseline_error": baseline[target.name]["selection_error"],
                        "test_baseline_error": (
                            baseline[target.name]["test_error"] if report_test else None
                        ),
                        "selection_delta": (
                            baseline[target.name]["selection_error"]
                            - selection_metrics["episode_mean_error"]
                        ),
                        "test_delta": (
                            baseline[target.name]["test_error"]
                            - test_metrics["episode_mean_error"]
                            if test_metrics
                            else None
                        ),
                        "selection_metrics": selection_metrics,
                        "test_metrics": test_metrics,
                    }
                    candidates.append(record)
                    best_key = (target.name, str(fitted["model"]))
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
                            "state": fitted["state"],
                        }

    for (target_name, model_name), selected in best.items():
        record = dict(selected["record"])
        record["selection_split"] = selection_value
        record["test_split"] = test_value
        target = next(item for item in targets if item.name == target_name)
        validation_gate_passed = float(record["selection_delta"]) > 0.0
        test_reported = model_name == "ridge" or validation_gate_passed
        record["validation_gate_passed"] = validation_gate_passed
        record["test_reported"] = test_reported
        record["promotion_status"] = (
            "eligible_from_validation"
            if validation_gate_passed
            else "validation_did_not_beat_baseline"
        )
        if test_reported and record.get("test_metrics") is None:
            test_metrics = _target_metrics(
                target,
                selected["truth"][selected["test_mask"]],
                selected["predicted"][selected["test_mask"]],
                selected["rows"].loc[selected["test_mask"], "trace_id"],
            )
            record["test_metrics"] = test_metrics
            record["test_error"] = test_metrics["episode_mean_error"]
            record["test_row_error"] = test_metrics["row_mean_error"]
            record["test_baseline_error"] = selected["baseline"]["test_error"]
            record["test_delta"] = selected["baseline"]["test_error"] - record["test_error"]
        if not test_reported:
            for column in (
                "test_error",
                "test_row_error",
                "test_baseline_error",
                "test_delta",
                "test_metrics",
            ):
                record[column] = None

        split_specs = [(selection_value, selected["selection_mask"])]
        if test_reported:
            split_specs.append((test_value, selected["test_mask"]))
        for split_name, mask in split_specs:
            split_rows = selected["rows"].loc[mask].reset_index(drop=True)
            truth = selected["truth"][mask]
            predicted = selected["predicted"][mask]
            baseline_predicted = np.asarray(selected["baseline"]["predicted"])[mask]
            errors = _target_row_errors(target, truth, predicted)
            baseline_errors = _target_row_errors(target, truth, baseline_predicted)
            interval = _paired_grouped_interval(
                target,
                truth,
                predicted,
                baseline_predicted,
                split_rows,
                group_column=bootstrap_group_column,
                samples=bootstrap_samples,
                seed=_stable_seed(mlp_seed, record["readout_id"], split_name, target_name),
            )
            prefix = "selection" if split_name == selection_value else "test"
            for name, value in interval.items():
                record[f"{prefix}_{name}"] = value
            physical_prediction = (
                predicted
                if target.kind == "position"
                else _decode_orientation(predicted, target.basis)
            )
            physical_baseline = (
                baseline_predicted
                if target.kind == "position"
                else _decode_orientation(baseline_predicted, target.basis)
            )
            for index, source in split_rows.iterrows():
                predictions.append(
                    {
                        "feature_id": feature_id,
                        "feature_group": record["feature_group"],
                        "model": model_name,
                        "readout_id": record["readout_id"],
                        "pca_dim": int(record["pca_dim"]),
                        "ridge_alpha": record["ridge_alpha"],
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
                        "baseline": record["selection_baseline"],
                        "baseline_value": np.asarray(physical_baseline[index]).tolist(),
                        "baseline_representation": np.asarray(
                            baseline_predicted[index]
                        ).tolist(),
                        "baseline_error": float(baseline_errors[index]),
                        "probe_minus_baseline": float(
                            baseline_errors[index] - errors[index]
                        ),
                        "error_unit": record["error_unit"],
                    }
                )
        selections.append(record)

    for target_name in sorted({str(record["target"]) for record in selections}):
        eligible = [
            record
            for record in selections
            if record["target"] == target_name and record["validation_gate_passed"]
        ]
        promoted_id = (
            min(eligible, key=lambda record: float(record["selection_error"]))["readout_id"]
            if eligible
            else None
        )
        for record in selections:
            if record["target"] != target_name:
                continue
            record["promoted"] = record["readout_id"] == promoted_id
            if record["promoted"]:
                record["promotion_status"] = "validation_selected"

    selected_ids = {str(record["readout_id"]) for record in selections}
    return (
        candidates,
        selections,
        predictions,
        [fitted_states[readout_id] for readout_id in sorted(selected_ids)],
    )


def _geometry_readout_id(
    feature_id: str,
    feature_group: str,
    model: str,
    pca_dim: int,
    ridge_alpha: float | None,
) -> str:
    payload = {
        "feature_id": feature_id,
        "feature_group": feature_group,
        "model": model,
        "pca_dim": pca_dim,
        "ridge_alpha": ridge_alpha,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return f"geometry-{model}-{digest}"


def _transform_state(
    *,
    feature_id: str,
    feature_group: str,
    pca_dim: int,
    scaler: StandardScaler,
    projector: PCA,
    blocks: Mapping[str, tuple[int, int]],
    train_rows: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays = {
        "feature_mean": np.asarray(scaler.mean_, dtype=np.float64),
        "feature_scale": np.asarray(scaler.scale_, dtype=np.float64),
        "pca_mean": np.asarray(projector.mean_, dtype=np.float64),
        "pca_components": np.asarray(projector.components_[:pca_dim], dtype=np.float64),
    }
    contract: dict[str, Any] = {
        "feature_id": feature_id,
        "feature_group": feature_group,
        "pca_dim": int(pca_dim),
        "train_rows": int(train_rows),
        "transform_fit_split": "train",
        "target_blocks": {name: [int(start), int(stop)] for name, (start, stop) in blocks.items()},
        "array_names": {name: name for name in arrays},
    }
    return contract, arrays


def _ridge_readout_state(
    *,
    readout_id: str,
    feature_id: str,
    feature_group: str,
    pca_dim: int,
    alpha: float,
    scaler: StandardScaler,
    projector: PCA,
    coefficients: np.ndarray,
    target_mean: np.ndarray,
    blocks: Mapping[str, tuple[int, int]],
    train_rows: int,
) -> GeometryReadoutState:
    contract, arrays = _transform_state(
        feature_id=feature_id,
        feature_group=feature_group,
        pca_dim=pca_dim,
        scaler=scaler,
        projector=projector,
        blocks=blocks,
        train_rows=train_rows,
    )
    arrays.update(
        {
            "readout_weights": np.asarray(coefficients, dtype=np.float64),
            "readout_bias": np.asarray(target_mean, dtype=np.float64),
        }
    )
    contract.update(
        {
            "readout_id": readout_id,
            "model": "ridge",
            "ridge_alpha": float(alpha),
            "array_names": {name: name for name in arrays},
        }
    )
    return GeometryReadoutState(readout_id, contract, arrays)


def _mlp_readout_state(
    *,
    readout_id: str,
    feature_id: str,
    feature_group: str,
    pca_dim: int,
    scaler: StandardScaler,
    projector: PCA,
    network: MLPRegressor,
    blocks: Mapping[str, tuple[int, int]],
    train_rows: int,
    random_state: int,
    max_iter: int,
) -> GeometryReadoutState:
    contract, arrays = _transform_state(
        feature_id=feature_id,
        feature_group=feature_group,
        pca_dim=pca_dim,
        scaler=scaler,
        projector=projector,
        blocks=blocks,
        train_rows=train_rows,
    )
    weight_names: list[str] = []
    bias_names: list[str] = []
    for index, (weights, biases) in enumerate(
        zip(network.coefs_, network.intercepts_, strict=True)
    ):
        weight_name = f"layer_weights_{index}"
        bias_name = f"layer_biases_{index}"
        arrays[weight_name] = np.asarray(weights, dtype=np.float64)
        arrays[bias_name] = np.asarray(biases, dtype=np.float64)
        weight_names.append(weight_name)
        bias_names.append(bias_name)
    contract.update(
        {
            "readout_id": readout_id,
            "model": "mlp",
            "hidden_layer_sizes": [64],
            "alpha": 1e-4,
            "max_iter": int(max_iter),
            "random_state": int(random_state),
            "n_iter": int(network.n_iter_),
            "final_loss": float(network.loss_),
            "converged": int(network.n_iter_) < int(max_iter),
            "out_activation": str(network.out_activation_),
            "array_names": {
                **{name: name for name in arrays if not name.startswith("layer_")},
                "layer_weights": weight_names,
                "layer_biases": bias_names,
            },
        }
    )
    return GeometryReadoutState(readout_id, contract, arrays)


def predict_geometry_readout(
    contract: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    features: np.ndarray,
) -> np.ndarray:
    """Replay a saved geometry readout from its explicit numeric state."""
    names = dict(contract["array_names"])

    def array(name: str) -> np.ndarray:
        return np.asarray(arrays[str(names[name])], dtype=np.float64)

    scale = array("feature_scale")
    scaled = (np.asarray(features, dtype=np.float64) - array("feature_mean")) / scale
    projected = (scaled - array("pca_mean")) @ array("pca_components").T
    model = str(contract["model"])
    if model == "ridge":
        return projected @ array("readout_weights") + array("readout_bias")
    if model != "mlp":
        raise ValueError(f"Unknown saved geometry model: {model!r}")
    values = projected
    weight_names = [str(value) for value in names["layer_weights"]]
    bias_names = [str(value) for value in names["layer_biases"]]
    for index, (weight_name, bias_name) in enumerate(
        zip(weight_names, bias_names, strict=True)
    ):
        values = values @ np.asarray(arrays[weight_name]) + np.asarray(arrays[bias_name])
        if index < len(weight_names) - 1:
            values = np.maximum(values, 0.0)
    if str(contract.get("out_activation") or "identity") != "identity":
        raise ValueError("Only identity-output MLP geometry readouts are supported")
    return np.asarray(values, dtype=np.float64)


def _paired_grouped_interval(
    target: GeometryTarget,
    truth: np.ndarray,
    predicted: np.ndarray,
    baseline_predicted: np.ndarray,
    rows: pd.DataFrame,
    *,
    group_column: str,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    improvement = _target_row_errors(target, truth, baseline_predicted) - _target_row_errors(
        target, truth, predicted
    )
    resolved_group_column = group_column if group_column in rows else "trace_id"
    groups = rows[resolved_group_column].astype(str)
    if resolved_group_column == "task_id" and "benchmark" in rows:
        groups = rows["benchmark"].astype(str) + ":" + groups
    frame = pd.DataFrame(
        {
            "group": groups.to_numpy(),
            "trace_id": rows["trace_id"].astype(str).to_numpy(),
            "improvement": improvement,
        }
    )
    episode_means = (
        frame.groupby(["group", "trace_id"], sort=True)["improvement"].mean().reset_index()
    )
    grouped = episode_means.groupby("group", sort=True)["improvement"].mean().to_numpy()
    if not len(grouped):
        return {
            "confidence_group_column": resolved_group_column,
            "confidence_group_count": 0,
            "mean_probe_minus_baseline": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "probability_improvement": float("nan"),
            "bootstrap_samples": int(samples),
        }
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        grouped,
        size=(max(1, int(samples)), len(grouped)),
        replace=True,
    ).mean(axis=1)
    return {
        "confidence_group_column": resolved_group_column,
        "confidence_group_count": int(len(grouped)),
        "mean_probe_minus_baseline": float(grouped.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "probability_improvement": float(np.mean(draws > 0.0)),
        "bootstrap_samples": int(samples),
    }


def _stable_seed(seed: int, *values: Any) -> int:
    encoded = json.dumps([int(seed), *values], sort_keys=True, default=str).encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:4], "big")


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
                "predicted": np.asarray(values),
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


def _select_geometry_targets(
    targets: Sequence[GeometryTarget],
    allowlist: Sequence[str] | None,
) -> list[GeometryTarget]:
    if allowlist is None:
        return list(targets)
    requested = [str(value) for value in allowlist]
    available = {target.name: target for target in targets}
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError("Unknown geometry targets: " + ", ".join(unknown))
    if not requested:
        raise ValueError("Geometry target allowlist must not be empty")
    return [available[name] for name in dict.fromkeys(requested)]


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
    fitted_readouts: Sequence[GeometryReadoutState],
    timings: Mapping[str, float],
    source_trace_ids: Sequence[str],
) -> LensArtifact:
    artifact_id = make_artifact_id(str(spec["name"]), "geometry_probe_study")
    relative_dir = Path("artifacts") / artifact_id
    outputs = {
        "candidates": str(relative_dir / "candidates.parquet"),
        "selections": str(relative_dir / "selections.parquet"),
        "predictions": str(relative_dir / "predictions.parquet"),
        "confidence_intervals": str(relative_dir / "confidence_intervals.parquet"),
        "fitted_readouts": str(relative_dir / "fitted_readouts.json"),
        "fitted_arrays": str(relative_dir / "fitted_arrays.npz"),
    }
    display_records = [] if selections.empty else _json_records(selections)
    artifact = LensArtifact(
        artifact_id=artifact_id,
        artifact_type="geometry_probe_study",
        name=str(spec["name"]),
        group_id="geometry_probe_studies",
        scope="dataset",
        selector={
            "features": spec["features"],
            "object_column": spec["object_column"],
            "targets": spec.get("targets", list(GEOMETRY_TARGET_NAMES)),
        },
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
            "model_selection": {
                "selection_split_only": True,
                "mlp_test_gate": "best validation MLP must beat its matching baseline",
                "promotion_uses_test": False,
                "confidence_interval": (
                    "paired bootstrap of probe-minus-baseline over whole groups"
                ),
            },
            "timings_seconds": dict(timings),
            "outputs": outputs,
        },
        metrics={
            "candidate_count": int(len(candidates)),
            "selection_count": int(len(selections)),
            "prediction_count": int(len(predictions)),
            "fitted_readout_count": int(len(fitted_readouts)),
            "total_seconds": float(timings.get("total_seconds", 0.0)),
        },
        display={
            "kind": "geometry_probe_study",
            "status": "exploratory",
            "selections": display_records,
        },
        tags=("probe", "geometry", "exploratory"),
        source_trace_ids=tuple(sorted(str(value) for value in source_trace_ids)),
    )
    saved = dataset.save_artifact(artifact)
    artifact_dir = dataset._dataset_artifact_root() / relative_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(artifact_dir / "candidates.parquet", index=False)
    selections.to_parquet(artifact_dir / "selections.parquet", index=False)
    predictions.to_parquet(artifact_dir / "predictions.parquet", index=False)
    _geometry_confidence_table(selections).to_parquet(
        artifact_dir / "confidence_intervals.parquet", index=False
    )
    readout_manifest, fitted_arrays = _serialized_geometry_readouts(fitted_readouts)
    (artifact_dir / "fitted_readouts.json").write_text(
        json.dumps(readout_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    np.savez_compressed(artifact_dir / "fitted_arrays.npz", **fitted_arrays)
    return saved


def _geometry_confidence_table(selections: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in selections.to_dict("records"):
        for prefix in ("selection", "test"):
            interval_low = row.get(f"{prefix}_ci95_low")
            if interval_low is None or pd.isna(interval_low):
                continue
            records.append(
                {
                    "feature_id": row["feature_id"],
                    "feature_group": row["feature_group"],
                    "model": row["model"],
                    "readout_id": row["readout_id"],
                    "target": row["target"],
                    "split": row[f"{prefix}_split"],
                    "group_column": row[f"{prefix}_confidence_group_column"],
                    "group_count": row[f"{prefix}_confidence_group_count"],
                    "mean_probe_minus_baseline": row[
                        f"{prefix}_mean_probe_minus_baseline"
                    ],
                    "ci95_low": row[f"{prefix}_ci95_low"],
                    "ci95_high": row[f"{prefix}_ci95_high"],
                    "probability_improvement": row[
                        f"{prefix}_probability_improvement"
                    ],
                    "bootstrap_samples": row[f"{prefix}_bootstrap_samples"],
                }
            )
    return pd.DataFrame.from_records(records)


def _serialized_geometry_readouts(
    fitted_readouts: Sequence[GeometryReadoutState],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    manifest: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for readout in fitted_readouts:
        prefix = f"{readout.readout_id}__"
        renamed = {f"{prefix}{name}": np.asarray(value) for name, value in readout.arrays.items()}
        arrays.update(renamed)
        contract = dict(readout.contract)
        array_names: dict[str, Any] = {}
        for name, value in dict(contract["array_names"]).items():
            if isinstance(value, str):
                array_names[name] = f"{prefix}{value}"
            else:
                array_names[name] = [f"{prefix}{item}" for item in value]
        contract["array_names"] = array_names
        contract["array_fingerprints"] = {
            name: _geometry_array_fingerprint(value) for name, value in renamed.items()
        }
        manifest.append(contract)
    return manifest, arrays


def _geometry_array_fingerprint(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape)).encode())
    digest.update(array.tobytes())
    return f"sha256:{digest.hexdigest()}"


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
    probe = dict(normalized.get("probe") or {})
    probe.setdefault("models", ["ridge"])
    probe.setdefault("pca_dims", [64, 128])
    probe.setdefault("ridge_alphas", [0.1, 1.0, 10.0, 100.0])
    probe.setdefault("mlp_hidden_units", 64)
    probe.setdefault("mlp_alpha", 1e-4)
    probe.setdefault("mlp_max_iter", 300)
    probe.setdefault("mlp_random_state", 0)
    probe.setdefault("bootstrap_samples", 2_000)
    probe.setdefault(
        "bootstrap_group_column",
        "trace_id"
        if str(normalized["split"].get("kind") or "existing") == "within_task_episode"
        else "task_id",
    )
    models = [str(value) for value in probe["models"]]
    unknown_models = sorted(set(models) - {"ridge", "mlp"})
    if unknown_models:
        raise ValueError("Unknown geometry probe models: " + ", ".join(unknown_models))
    if not models:
        raise ValueError("Geometry probe models must not be empty")
    probe["models"] = list(dict.fromkeys(models))
    if int(probe["mlp_hidden_units"]) != 64:
        raise ValueError("The standard geometry MLP uses exactly 64 hidden units")
    if float(probe["mlp_alpha"]) != 1e-4:
        raise ValueError("The standard geometry MLP uses alpha=1e-4")
    if int(probe["mlp_max_iter"]) != 300:
        raise ValueError("The standard geometry MLP uses max_iter=300")
    if not probe["pca_dims"]:
        raise ValueError("Geometry probe pca_dims must not be empty")
    normalized["probe"] = probe
    if "targets" in normalized:
        targets = normalized["targets"]
        if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
            raise ValueError("Geometry targets must be a list")
        requested_targets = [str(value) for value in targets]
        unknown_targets = sorted(set(requested_targets) - set(GEOMETRY_TARGET_NAMES))
        if unknown_targets:
            raise ValueError("Unknown geometry targets: " + ", ".join(unknown_targets))
        if not requested_targets:
            raise ValueError("Geometry targets must not be empty")
        normalized["targets"] = list(dict.fromkeys(requested_targets))
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
    *,
    split_column: str = "split",
    required_split_values: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if limit in {None, 0, ""}:
        out = rows.reset_index(drop=True).reset_index(names="__feature_row_index")
        return out, X
    requested = int(limit)
    episode_rows = rows[["trace_id", split_column]].drop_duplicates()
    episode_ids = _balanced_episode_ids(
        episode_rows,
        requested,
        split_column=split_column,
        required_split_values=required_split_values,
    )
    mask = rows["trace_id"].astype(str).isin(episode_ids).to_numpy()
    out = rows.loc[mask].reset_index(drop=True).reset_index(names="__feature_row_index")
    return out, X[mask]


def _limited_episode_ids(
    dataset: TraceDataset,
    limit: Any,
    *,
    required_split_values: Sequence[str] | None = None,
) -> list[str] | None:
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
    return _balanced_episode_ids(
        episodes,
        requested,
        split_column="split",
        required_split_values=required_split_values,
    )


def _balanced_episode_ids(
    episodes: pd.DataFrame,
    requested: int,
    *,
    split_column: str,
    required_split_values: Sequence[str] | None,
) -> list[str]:
    if required_split_values is not None and not required_split_values:
        return sorted(str(value) for value in episodes["trace_id"].unique())[:requested]
    available_values = sorted(
        str(value) for value in episodes[split_column].dropna().unique()
    )
    split_values = list(dict.fromkeys(required_split_values or available_values))
    _validate_episode_limit(requested, split_values)
    missing = sorted(set(split_values) - set(available_values))
    if missing:
        raise ValueError(
            "Episode limit cannot preserve missing required splits: " + ", ".join(missing)
        )
    queues = {
        split_value: sorted(
            str(value)
            for value in episodes.loc[
                episodes[split_column].astype(str) == split_value, "trace_id"
            ].unique()
        )
        for split_value in split_values
    }
    selected: list[str] = []
    for split_value in split_values:
        if queues[split_value]:
            selected.append(queues[split_value].pop(0))
    while len(selected) < requested and any(queues.values()):
        for split_value in split_values:
            if len(selected) >= requested:
                break
            if queues[split_value]:
                selected.append(queues[split_value].pop(0))
    return selected


def _required_split_values(split: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(split[key]) for key in ("train_value", "selection_value", "test_value")
        )
    )


def _source_required_split_values(split: Mapping[str, Any]) -> tuple[str, ...]:
    if str(split.get("kind") or "existing") == "within_task_episode":
        return ()
    return _required_split_values(split)


def _validate_episode_limit(limit: Any, required_split_values: Sequence[str]) -> None:
    if limit in {None, 0, ""}:
        return
    requested = int(limit)
    required_count = len(set(str(value) for value in required_split_values))
    if requested < required_count:
        raise ValueError(
            f"limit_episodes={requested} cannot cover all {required_count} required splits"
        )


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
    "GeometryReadoutState",
    "GeometryStudyResult",
    "GeometryTarget",
    "geometry_target_table",
    "predict_geometry_readout",
    "run_geometry_probe_study",
]
