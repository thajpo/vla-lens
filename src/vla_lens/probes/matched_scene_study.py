"""Matched-scene localization study for object-induced representation changes."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from vla_lens.artifacts import LensArtifact, make_artifact_id
from vla_lens.probes.geometry_study import (
    _apply_split_contract,
    _geometry_metadata_rows,
)
from vla_lens.probes.scene_map_study import SceneMapTargets, scene_map_target_table
from vla_lens.probes.workflow_artifacts import _artifact_dir
from vla_lens.traces import TraceBundle, TraceDataset

MATCHED_SCENE_STUDY_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class MatchedSceneStudyResult:
    artifact: LensArtifact | None
    pairs: pd.DataFrame
    patch_scores: pd.DataFrame
    pair_metrics: pd.DataFrame
    summary: pd.DataFrame
    source_sites: pd.DataFrame
    token_metadata: pd.DataFrame
    timings: Mapping[str, float]


def run_matched_scene_localization_study(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    *,
    save: bool = True,
) -> MatchedSceneStudyResult:
    """Localize the one changed object in otherwise matched initial scenes."""

    normalized = _normalize_spec(spec)
    started = time.perf_counter()
    timings: dict[str, float] = {}

    step = time.perf_counter()
    rows = _episode_rows(dataset, normalized["split"])
    targets, vocabulary = scene_map_target_table(dataset, rows, cache=True)
    pairs = _matched_pairs(
        rows,
        targets,
        movement_threshold_m=float(normalized["matching"]["movement_threshold_m"]),
        stationary_threshold_m=float(normalized["matching"]["stationary_threshold_m"]),
        robot_threshold_m=float(normalized["matching"]["robot_threshold_m"]),
        dataset=dataset,
    )
    limit_pairs = normalized.get("limit_pairs_per_split")
    if limit_pairs is not None:
        pairs = (
            pairs.sort_values(["split", "pair_id"])
            .groupby("split", group_keys=False)
            .head(int(limit_pairs))
            .reset_index(drop=True)
        )
    if pairs.empty:
        raise ValueError("No matched scene pairs satisfied the declared controls")
    timings["match_pairs_seconds"] = time.perf_counter() - step

    step = time.perf_counter()
    feature_sites = _feature_sites(normalized["feature"])
    pair_records = [dict(value) for value in pairs.to_dict("records")]
    workers = max(1, int(normalized["analysis"]["io_workers"]))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        outputs = list(
            executor.map(
                lambda pair: _score_pair(
                    dataset,
                    pair,
                    feature_sites,
                    camera_id=str(normalized["feature"]["camera_id"]),
                    camera_name=str(normalized["feature"]["camera_name"]),
                ),
                pair_records,
            )
        )
    patch_frames = [value[0] for value in outputs if not value[0].empty]
    metric_frames = [value[1] for value in outputs if not value[1].empty]
    eligible_ids = {value for frame in metric_frames for value in frame["pair_id"].unique()}
    pairs = pairs.loc[pairs["pair_id"].isin(eligible_ids)].reset_index(drop=True)
    patch_scores = (
        pd.concat(patch_frames, ignore_index=True) if patch_frames else pd.DataFrame()
    )
    pair_metrics = (
        pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    )
    if pair_metrics.empty:
        raise ValueError("Matched pairs did not have usable image tokens and object boxes")
    timings["score_patches_seconds"] = time.perf_counter() - step

    step = time.perf_counter()
    summary = _summary_table(
        pair_metrics,
        split=normalized["split"],
        bootstrap_samples=int(normalized["analysis"]["bootstrap_samples"]),
    )
    source_trace_ids = sorted(
        set(pairs["left_trace_id"].astype(str)) | set(pairs["right_trace_id"].astype(str))
    )
    source_sites = dataset.model_site_index.loc[
        dataset.model_site_index["trace_id"].astype(str).isin(source_trace_ids)
        & dataset.model_site_index["name"].astype(str).isin(
            [str(value["name"]) for value in feature_sites]
        )
    ].copy()
    token_metadata = _source_token_metadata(
        dataset,
        source_trace_ids,
        camera_id=str(normalized["feature"]["camera_id"]),
    )
    timings["summarize_seconds"] = time.perf_counter() - step
    timings["total_seconds"] = time.perf_counter() - started

    artifact = (
        _save_study(
            dataset,
            normalized,
            pairs,
            patch_scores,
            pair_metrics,
            summary,
            source_sites,
            token_metadata,
            vocabulary,
            timings,
        )
        if save
        else None
    )
    return MatchedSceneStudyResult(
        artifact=artifact,
        pairs=pairs,
        patch_scores=patch_scores,
        pair_metrics=pair_metrics,
        summary=summary,
        source_sites=source_sites,
        token_metadata=token_metadata,
        timings=timings,
    )


def _episode_rows(dataset: TraceDataset, split: Mapping[str, Any]) -> pd.DataFrame:
    columns = ["trace_id", "episode_id", "env_id", "task_id", "prompt", "seed"]
    rows = dataset.episode_index[
        [column for column in columns if column in dataset.episode_index]
    ].drop_duplicates("trace_id")
    rows["timestep"] = 0
    rows["policy_call_index"] = 0
    rows = _geometry_metadata_rows(dataset, rows, cache=True)
    rows = _apply_split_contract(rows, split)
    return rows.sort_values("trace_id").reset_index(drop=True)


def _matched_pairs(
    rows: pd.DataFrame,
    targets: SceneMapTargets,
    *,
    movement_threshold_m: float,
    stationary_threshold_m: float,
    robot_threshold_m: float,
    dataset: TraceDataset | None = None,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    positions = np.asarray(targets.position, dtype=np.float64)
    presence = np.asarray(targets.presence, dtype=bool)
    group_columns = ["env_id", "task_id", "prompt", "split"]
    missing = [column for column in group_columns if column not in rows]
    if missing:
        raise KeyError(f"Matched-scene rows are missing group columns: {missing}")
    robot_positions = _initial_robot_positions(dataset, rows) if dataset is not None else {}
    for key, group in rows.groupby(group_columns, dropna=False, sort=True):
        env_id, task_id, prompt, split_value = key
        indices = sorted(int(value) for value in group.index)
        for left_index, right_index in combinations(indices, 2):
            if not np.array_equal(presence[left_index], presence[right_index]):
                continue
            common = presence[left_index] & np.isfinite(positions[left_index]).all(axis=1)
            common &= np.isfinite(positions[right_index]).all(axis=1)
            if common.sum() < 2:
                continue
            distance = np.linalg.norm(positions[right_index] - positions[left_index], axis=1)
            moved = np.flatnonzero(common & (distance > movement_threshold_m))
            if len(moved) != 1:
                continue
            moved_index = int(moved[0])
            stationary = common.copy()
            stationary[moved_index] = False
            max_stationary = float(distance[stationary].max()) if stationary.any() else 0.0
            if max_stationary > stationary_threshold_m:
                continue
            left_trace = str(rows.loc[left_index, "trace_id"])
            right_trace = str(rows.loc[right_index, "trace_id"])
            robot_delta = _robot_delta(robot_positions, left_trace, right_trace)
            if robot_delta is not None and robot_delta > robot_threshold_m:
                continue
            pair_id = hashlib.sha256(f"{left_trace}\n{right_trace}".encode()).hexdigest()[:20]
            delta = positions[right_index, moved_index] - positions[left_index, moved_index]
            records.append(
                {
                    "pair_id": pair_id,
                    "env_id": env_id,
                    "task_id": task_id,
                    "prompt": prompt,
                    "split": split_value,
                    "scene_key": f"{env_id}:{task_id}:{prompt}",
                    "left_trace_id": left_trace,
                    "right_trace_id": right_trace,
                    "left_seed": rows.loc[left_index].get("seed"),
                    "right_seed": rows.loc[right_index].get("seed"),
                    "moved_object_index": moved_index,
                    "moved_object_name": targets.vocabulary[moved_index],
                    "moved_object_delta": delta.tolist(),
                    "moved_distance_m": float(np.linalg.norm(delta)),
                    "max_stationary_distance_m": max_stationary,
                    "robot_initial_delta_m": robot_delta,
                    "role_manipulated": bool(
                        targets.role_manipulated[left_index, moved_index]
                        or targets.role_manipulated[right_index, moved_index]
                    ),
                    "role_distractor": bool(
                        targets.role_distractor[left_index, moved_index]
                        or targets.role_distractor[right_index, moved_index]
                    ),
                    "common_object_names": [
                        targets.vocabulary[index] for index in np.flatnonzero(common)
                    ],
                    "stationary_object_names": [
                        targets.vocabulary[index] for index in np.flatnonzero(stationary)
                    ],
                }
            )
    return pd.DataFrame.from_records(records)


def _initial_robot_positions(
    dataset: TraceDataset | None, rows: pd.DataFrame
) -> dict[str, np.ndarray]:
    if dataset is None:
        return {}
    out: dict[str, np.ndarray] = {}
    for trace_id in rows["trace_id"].astype(str):
        try:
            out[trace_id] = np.asarray(
                dataset.bundle(trace_id).array("eef_pos", mmap=True)[0], dtype=np.float64
            )
        except KeyError:
            continue
    return out


def _robot_delta(
    values: Mapping[str, np.ndarray], left_trace: str, right_trace: str
) -> float | None:
    if left_trace not in values or right_trace not in values:
        return None
    return float(np.linalg.norm(values[right_trace] - values[left_trace]))


def _feature_sites(feature: Mapping[str, Any]) -> list[dict[str, Any]]:
    sites: list[dict[str, Any]] = []
    if bool(feature.get("include_input_embeddings", True)):
        sites.append(
            {
                "feature_id": "image_input_embeddings",
                "name": "pi05.vlm.prefix.image_hidden_tokens",
                "layer": "input",
            }
        )
    for layer in feature.get("layers", [0, 4, 8, 12, 17]):
        sites.append(
            {
                "feature_id": f"vlm_layer_{int(layer)}",
                "name": f"pi05.vlm.layers.{int(layer)}.prefix.hidden_tokens",
                # Keep this column one type so every result table can be saved
                # without losing the human-readable input/pixel labels.
                "layer": str(int(layer)),
            }
        )
    return sites


def _score_pair(
    dataset: TraceDataset,
    pair: Mapping[str, Any],
    feature_sites: Sequence[Mapping[str, Any]],
    *,
    camera_id: str,
    camera_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    left = dataset.bundle(str(pair["left_trace_id"]))
    right = dataset.bundle(str(pair["right_trace_id"]))
    left_tokens = _camera_tokens(left, camera_id)
    right_tokens = _camera_tokens(right, camera_id)
    if not _same_patch_topology(left_tokens, right_tokens):
        return pd.DataFrame(), pd.DataFrame()
    moved_name = str(pair["moved_object_name"])
    left_bbox = _object_bbox(left, moved_name, camera_name)
    right_bbox = _object_bbox(right, moved_name, camera_name)
    if left_bbox is None or right_bbox is None:
        return pd.DataFrame(), pd.DataFrame()
    target_mask = _bbox_patch_mask(left_tokens, left_bbox) | _bbox_patch_mask(
        left_tokens, right_bbox
    )
    if not target_mask.any() or target_mask.all():
        return pd.DataFrame(), pd.DataFrame()
    wrong_name, wrong_mask = _stationary_control_mask(
        left,
        right,
        left_tokens,
        pair.get("stationary_object_names") or [],
        camera_name,
        target_mask,
    )
    patch_records: list[dict[str, Any]] = []
    metric_records: list[dict[str, Any]] = []
    score_sets: list[tuple[str, Any, np.ndarray]] = []
    pixel_scores = _pixel_patch_difference(left, right, left_tokens, camera_id)
    if pixel_scores is not None:
        score_sets.append(("raw_pixels", "pixels", pixel_scores))
    token_indices = left_tokens["token_index"].to_numpy(dtype=np.int64)
    for site in feature_sites:
        try:
            left_values = _site_tokens(left, str(site["name"]), token_indices)
            right_values = _site_tokens(right, str(site["name"]), token_indices)
        except (KeyError, ValueError):
            continue
        delta = right_values - left_values
        raw = np.linalg.norm(delta, axis=1)
        scale = 0.5 * (
            np.linalg.norm(left_values, axis=1) + np.linalg.norm(right_values, axis=1)
        )
        relative = raw / np.maximum(scale, 1e-8)
        score_sets.append((str(site["feature_id"]), site["layer"], relative))
    for feature_id, layer, scores in score_sets:
        metrics = _localization_metrics(scores, target_mask, wrong_mask)
        metric_records.append(
            {
                **_pair_identity(pair),
                "feature_id": feature_id,
                "layer": layer,
                "wrong_object_name": wrong_name,
                **metrics,
            }
        )
        for index, token in left_tokens.reset_index(drop=True).iterrows():
            patch_records.append(
                {
                    **_pair_identity(pair),
                    "feature_id": feature_id,
                    "layer": layer,
                    "token_index": int(token["token_index"]),
                    "patch_row": int(token["patch_row"]),
                    "patch_col": int(token["patch_col"]),
                    "pixel_x0": int(token["pixel_x0"]),
                    "pixel_x1": int(token["pixel_x1"]),
                    "pixel_y0": int(token["pixel_y0"]),
                    "pixel_y1": int(token["pixel_y1"]),
                    "change_score": float(scores[index]),
                    "target_region": bool(target_mask[index]),
                    "stationary_control_region": bool(wrong_mask[index]),
                }
            )
    return pd.DataFrame.from_records(patch_records), pd.DataFrame.from_records(metric_records)


def _pair_identity(pair: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": pair["pair_id"],
        "scene_key": pair["scene_key"],
        "split": pair["split"],
        "moved_object_name": pair["moved_object_name"],
        "moved_distance_m": pair["moved_distance_m"],
        "role_manipulated": bool(pair["role_manipulated"]),
        "role_distractor": bool(pair["role_distractor"]),
    }


def _camera_tokens(bundle: TraceBundle, camera_id: str) -> pd.DataFrame:
    tokens = bundle.tokens.copy()
    mask = tokens["token_kind"].astype(str) == "image"
    if "camera_id" in tokens:
        mask &= tokens["camera_id"].astype(str) == camera_id
    if "policy_call_index" in tokens:
        mask &= pd.to_numeric(tokens["policy_call_index"], errors="coerce").fillna(0) == 0
    columns = [
        "token_index",
        "patch_row",
        "patch_col",
        "pixel_x0",
        "pixel_x1",
        "pixel_y0",
        "pixel_y1",
    ]
    return (
        tokens.loc[mask, columns]
        .drop_duplicates("token_index")
        .sort_values("token_index")
        .reset_index(drop=True)
    )


def _same_patch_topology(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if left.empty or len(left) != len(right):
        return False
    return left.equals(right)


def _site_tokens(
    bundle: TraceBundle, site_name: str, token_indices: np.ndarray
) -> np.ndarray:
    site = bundle.model_sites.loc[bundle.model_sites["name"].astype(str) == site_name]
    if site.empty:
        raise KeyError(site_name)
    axes = _parse_axes(site.iloc[0]["axes"])
    array = bundle.model_site(site_name, mmap=True)
    selection: list[Any] = [slice(None)] * len(axes)
    if "policy_call" not in axes or "token" not in axes or "channel" not in axes:
        raise ValueError(f"Site {site_name!r} is not a policy-call token tensor")
    selection[axes.index("policy_call")] = 0
    selection[axes.index("token")] = token_indices
    values = np.asarray(array.oindex[tuple(selection)], dtype=np.float32)
    remaining = [
        axis
        for axis, value in zip(axes, selection, strict=True)
        if not isinstance(value, int)
    ]
    token_axis = remaining.index("token")
    channel_axis = remaining.index("channel")
    values = np.moveaxis(values, [token_axis, channel_axis], [0, 1])
    return values.reshape(len(token_indices), -1)


def _object_bbox(
    bundle: TraceBundle, object_name: str, camera_name: str
) -> np.ndarray | None:
    match = bundle.array_index.loc[
        bundle.array_index["name"].astype(str) == "camera_object_bbox"
    ]
    if match.empty:
        return None
    metadata = match.iloc[0].get("metadata")
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    object_names = [str(value) for value in (metadata or {}).get("object_names", [])]
    camera_names = [str(value) for value in (metadata or {}).get("camera_names", [])]
    if object_name not in object_names or camera_name not in camera_names:
        return None
    object_index = object_names.index(object_name)
    camera_index = camera_names.index(camera_name)
    visible = np.asarray(bundle.array("camera_object_visible", mmap=True))
    if not bool(visible[0, camera_index, object_index]):
        return None
    bbox = np.asarray(bundle.array("camera_object_bbox", mmap=True))[
        0, camera_index, object_index
    ].astype(np.float64)
    return bbox if np.isfinite(bbox).all() else None


def _bbox_patch_mask(tokens: pd.DataFrame, bbox: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return (
        (tokens["pixel_x1"].to_numpy(dtype=float) > x0)
        & (tokens["pixel_x0"].to_numpy(dtype=float) < x1)
        & (tokens["pixel_y1"].to_numpy(dtype=float) > y0)
        & (tokens["pixel_y0"].to_numpy(dtype=float) < y1)
    )


def _stationary_control_mask(
    left: TraceBundle,
    right: TraceBundle,
    tokens: pd.DataFrame,
    object_names: Sequence[str],
    camera_name: str,
    target_mask: np.ndarray,
) -> tuple[str | None, np.ndarray]:
    best_name: str | None = None
    best_mask = np.zeros(len(tokens), dtype=bool)
    best_overlap = 1.0
    for object_name in sorted(str(value) for value in object_names):
        left_bbox = _object_bbox(left, object_name, camera_name)
        right_bbox = _object_bbox(right, object_name, camera_name)
        if left_bbox is None or right_bbox is None:
            continue
        mask = _bbox_patch_mask(tokens, left_bbox) | _bbox_patch_mask(tokens, right_bbox)
        if not mask.any() or mask.all():
            continue
        overlap = float(np.logical_and(mask, target_mask).sum() / max(1, mask.sum()))
        if overlap < best_overlap:
            best_name, best_mask, best_overlap = object_name, mask, overlap
    return best_name, best_mask


def _pixel_patch_difference(
    left: TraceBundle,
    right: TraceBundle,
    tokens: pd.DataFrame,
    camera_id: str,
) -> np.ndarray | None:
    try:
        left_frame = np.asarray(left.frames(camera_id, mmap=True)[0], dtype=np.float32)
        right_frame = np.asarray(right.frames(camera_id, mmap=True)[0], dtype=np.float32)
    except KeyError:
        return None
    difference = np.abs(right_frame - left_frame).mean(axis=-1)
    scores = []
    for token in tokens.itertuples():
        patch = difference[
            int(token.pixel_y0) : int(token.pixel_y1),
            int(token.pixel_x0) : int(token.pixel_x1),
        ]
        scores.append(float(patch.mean()) if patch.size else 0.0)
    return np.asarray(scores, dtype=np.float64)


def _localization_metrics(
    scores: np.ndarray, target_mask: np.ndarray, wrong_mask: np.ndarray
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    truth = np.asarray(target_mask, dtype=bool)
    control = np.asarray(wrong_mask, dtype=bool)
    positive_count = int(truth.sum())
    top = np.argsort(values)[-positive_count:]
    outside = values[~truth]
    return {
        "patch_count": int(len(values)),
        "target_patch_count": positive_count,
        "random_average_precision": positive_count / max(1, len(values)),
        "target_average_precision": float(average_precision_score(truth, values)),
        "target_roc_auc": float(roc_auc_score(truth, values)),
        "target_top_k_recall": float(truth[top].sum() / max(1, positive_count)),
        "target_mean_change": float(values[truth].mean()),
        "outside_mean_change": float(outside.mean()),
        "target_minus_outside": float(values[truth].mean() - outside.mean()),
        "stationary_control_patch_count": int(control.sum()),
        "stationary_control_average_precision": (
            float(average_precision_score(control, values))
            if control.any() and not control.all()
            else float("nan")
        ),
    }


def _summary_table(
    pair_metrics: pd.DataFrame,
    *,
    split: Mapping[str, Any],
    bootstrap_samples: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for (split_value, feature_id, layer), group in pair_metrics.groupby(
        ["split", "feature_id", "layer"], dropna=False, sort=True
    ):
        lift = group["target_average_precision"] - group["random_average_precision"]
        control_lift = (
            group["target_average_precision"]
            - group["stationary_control_average_precision"]
        )
        low, high = _group_bootstrap_interval(
            lift.to_numpy(dtype=float),
            group["scene_key"].astype(str).to_numpy(),
            samples=bootstrap_samples,
            seed=0,
        )
        scene_keys = group["scene_key"].astype(str).to_numpy()
        records.append(
            {
                "split": split_value,
                "feature_id": feature_id,
                "layer": layer,
                "pair_count": int(len(group)),
                "scene_count": int(group["scene_key"].nunique()),
                "mean_average_precision": _equal_weight_group_mean(
                    group["target_average_precision"].to_numpy(dtype=float), scene_keys
                ),
                "mean_random_average_precision": _equal_weight_group_mean(
                    group["random_average_precision"].to_numpy(dtype=float), scene_keys
                ),
                "mean_average_precision_lift": _equal_weight_group_mean(
                    lift.to_numpy(dtype=float), scene_keys
                ),
                "scene_bootstrap_lift_low": low,
                "scene_bootstrap_lift_high": high,
                "mean_target_vs_stationary_control": _equal_weight_group_mean(
                    control_lift.to_numpy(dtype=float), scene_keys
                ),
                "mean_roc_auc": _equal_weight_group_mean(
                    group["target_roc_auc"].to_numpy(dtype=float), scene_keys
                ),
                "mean_top_k_recall": _equal_weight_group_mean(
                    group["target_top_k_recall"].to_numpy(dtype=float), scene_keys
                ),
            }
        )
    out = pd.DataFrame.from_records(records)
    selection_value = str(split["selection_value"])
    candidates = out.loc[
        (out["split"].astype(str) == selection_value)
        & (out["feature_id"].astype(str) != "raw_pixels")
    ]
    selected = (
        str(
            candidates.sort_values("mean_average_precision_lift", ascending=False)
            .iloc[0]["feature_id"]
        )
        if not candidates.empty
        else None
    )
    out["selected_on_validation"] = out["feature_id"].astype(str) == str(selected)
    return out


def _group_bootstrap_interval(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    finite = np.isfinite(values)
    values = values[finite]
    groups = groups[finite]
    unique = np.unique(groups)
    if len(unique) < 2 or samples < 1:
        mean = float(np.mean(values)) if len(values) else float("nan")
        return mean, mean
    group_means = np.asarray([values[groups == value].mean() for value in unique])
    rng = np.random.default_rng(seed)
    draws = group_means[rng.integers(0, len(group_means), size=(samples, len(group_means)))]
    estimates = draws.mean(axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _equal_weight_group_mean(values: np.ndarray, groups: np.ndarray) -> float:
    finite = np.isfinite(values)
    values = values[finite]
    groups = groups[finite]
    unique = np.unique(groups)
    if not len(unique):
        return float("nan")
    return float(np.mean([values[groups == value].mean() for value in unique]))


def _source_token_metadata(
    dataset: TraceDataset, trace_ids: Sequence[str], *, camera_id: str
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for trace_id in trace_ids:
        frame = _camera_tokens(dataset.bundle(trace_id), camera_id).copy()
        frame.insert(0, "trace_id", trace_id)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _save_study(
    dataset: TraceDataset,
    spec: Mapping[str, Any],
    pairs: pd.DataFrame,
    patch_scores: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    source_sites: pd.DataFrame,
    token_metadata: pd.DataFrame,
    vocabulary: pd.DataFrame,
    timings: Mapping[str, float],
) -> LensArtifact:
    artifact_id = make_artifact_id(str(spec["name"]), "matched_scene_localization_study")
    relative_dir = Path("artifacts") / artifact_id
    tables = {
        "matched_pairs": pairs,
        "patch_scores": patch_scores,
        "pair_metrics": pair_metrics,
        "summary": summary,
        "source_sites": source_sites,
        "token_metadata": token_metadata,
        "vocabulary": vocabulary,
    }
    outputs = {name: str(relative_dir / f"{name}.parquet") for name in tables}
    trace_ids = sorted(
        set(pairs["left_trace_id"].astype(str)) | set(pairs["right_trace_id"].astype(str))
    )
    trace_fingerprints = {
        trace_id: str(
            dataset.bundle(trace_id).fingerprints.get("trace_fingerprint") or ""
        )
        for trace_id in trace_ids
    }
    artifact = LensArtifact(
        artifact_id=artifact_id,
        artifact_type="matched_scene_localization_study",
        name=str(spec["name"]),
        group_id="scene_map_probe_studies",
        scope="dataset",
        selector={"feature": spec["feature"], "matching": spec["matching"]},
        method={
            "workflow": "run_matched_scene_localization_study",
            "schema_version": MATCHED_SCENE_STUDY_SCHEMA_VERSION,
            "research": {
                key: spec[key]
                for key in ["question", "hypothesis_family", "intended_claim"]
                if spec.get(key)
            },
            "split": spec["split"],
            "matching": spec["matching"],
            "analysis": spec["analysis"],
            "controls": {
                "positive": "raw main-camera pixel patch difference",
                "negative": "stationary object region in the same matched pair",
                "random": "positive patch prevalence",
            },
            "evaluation": {
                "unit": "unordered matched episode pair",
                "grouped_uncertainty": "equal-weight scene key bootstrap",
                "selection": "feature source selected only on the validation split",
            },
            "source_trace_fingerprints": trace_fingerprints,
            "outputs": outputs,
            "timings_seconds": dict(timings),
            "storage_contract": {
                "raw_activations": "referenced from capture and never copied",
                "saved_evidence": (
                    "matched trace pairs, exact source sites and token boxes, per-patch "
                    "scores, pair metrics, controls, and grouped uncertainty"
                ),
            },
        },
        metrics={
            "pair_count": int(len(pairs)),
            "scene_count": int(pairs["scene_key"].nunique()),
            "patch_score_count": int(len(patch_scores)),
            "source_trace_count": int(len(trace_ids)),
            "total_seconds": float(timings.get("total_seconds", 0.0)),
        },
        display={
            "kind": "matched_scene_localization_study",
            "status": "exploratory",
            "summary": json.loads(summary.to_json(orient="records")),
        },
        tags=(
            "probe",
            "scene-map",
            "matched-scenes",
            "visual-tokens",
            "object-localization",
            "exploratory",
        ),
        source_trace_ids=tuple(trace_ids),
    )
    artifact_dir = _artifact_dir(dataset, artifact)
    artifact_dir.mkdir(parents=True, exist_ok=False)
    try:
        for name, frame in tables.items():
            frame.to_parquet(artifact_dir / f"{name}.parquet", index=False)
        saved = dataset.save_artifact(artifact)
    except BaseException:
        shutil.rmtree(artifact_dir)
        raise
    return saved


def _normalize_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(spec)
    normalized.setdefault(
        "name", "PI0.5 matched initial-scene visual object localization study"
    )
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
    feature = dict(normalized.get("feature") or {})
    feature.setdefault("camera_id", "main")
    feature.setdefault("camera_name", "agentview")
    feature.setdefault("layers", [0, 4, 8, 12, 17])
    feature.setdefault("include_input_embeddings", True)
    normalized["feature"] = feature
    matching = dict(normalized.get("matching") or {})
    matching.setdefault("movement_threshold_m", 0.01)
    matching.setdefault("stationary_threshold_m", 0.01)
    matching.setdefault("robot_threshold_m", 0.005)
    normalized["matching"] = matching
    analysis = dict(normalized.get("analysis") or {})
    analysis.setdefault("io_workers", 8)
    analysis.setdefault("bootstrap_samples", 2_000)
    normalized["analysis"] = analysis
    return normalized


def _parse_axes(value: Any) -> list[str]:
    if isinstance(value, str):
        value = json.loads(value)
    return [str(item) for item in value]
