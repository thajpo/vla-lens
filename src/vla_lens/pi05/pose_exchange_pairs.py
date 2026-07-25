"""Validate captured pose exchanges and derive object-conditioned token mappings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.interventions import (
    CounterfactualPairManifest,
    CounterfactualRecipe,
    DonorSpec,
    PolicyCallRef,
    RecipientSpec,
    TraceRef,
)
from vla_lens.traces import TraceBundle, TraceDataset

ROBOT_ARRAYS = (
    "robot_joint_pos",
    "robot_joint_vel",
    "eef_pos",
    "eef_quat",
    "eef_mat",
    "gripper_qpos",
    "gripper_qvel",
)
CAMERA_ARRAYS = ("camera_intrinsics", "camera_extrinsics", "camera_resolution")
OBJECT_TOKEN_COUNT = 12
BACKGROUND_TOKEN_COUNT = 24


def build_pose_exchange_pair_manifests(
    dataset: TraceDataset,
    rows: Sequence[Any],
) -> tuple[CounterfactualPairManifest, ...]:
    """Validate every captured pair and preserve the exact token mapping."""
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        grouped.setdefault(str(row.pair_id), {})[str(row.role)] = row
    manifests = []
    for pair_id, roles in grouped.items():
        if set(roles) != {"recipient", "donor"}:
            raise ValueError(f"pose-exchange pair {pair_id} needs recipient and donor traces")
        recipient_row = roles["recipient"]
        donor_row = roles["donor"]
        recipient = dataset.bundle(str(recipient_row.trace_id))
        donor = dataset.bundle(str(donor_row.trace_id))
        manifests.append(
            _pair_manifest(
                dataset,
                pair_id=pair_id,
                recipient=recipient,
                donor=donor,
                target_object=str(recipient_row.target_object),
                distractor_object=str(recipient_row.distractor_object),
                layout_id=int(recipient_row.layout_id),
                seed=int(recipient_row.seed),
            )
        )
    return tuple(manifests)


def save_pose_exchange_pair_manifests(
    root: Path,
    manifests: Sequence[CounterfactualPairManifest],
) -> Path:
    path = root / "vla_lens" / "counterfactual_pairs" / "rq019_pose_exchange_pairs.json"
    _write_json_atomic(
        path,
        {
            "schema_kind": "vla_lens.counterfactual_pair_collection",
            "schema_version": 1,
            "pair_count": len(manifests),
            "valid_pair_count": sum(
                bool(pair.validation.get("pair_valid")) for pair in manifests
            ),
            "pairs": [pair.to_dict() for pair in manifests],
        },
    )
    return path


def _pair_manifest(
    dataset: TraceDataset,
    *,
    pair_id: str,
    recipient: TraceBundle,
    donor: TraceBundle,
    target_object: str,
    distractor_object: str,
    layout_id: int,
    seed: int,
) -> CounterfactualPairManifest:
    recipient_objects = _object_indices(recipient)
    donor_objects = _object_indices(donor)
    if recipient_objects != donor_objects:
        raise ValueError(f"pair {pair_id} object indexes differ between traces")
    for name in (target_object, distractor_object):
        if name not in recipient_objects:
            raise KeyError(f"pair {pair_id} trace does not contain object {name!r}")

    target_index = recipient_objects[target_object]
    distractor_index = recipient_objects[distractor_object]
    recipient_pos = _array(recipient, "scene_object_pos")[0]
    donor_pos = _array(donor, "scene_object_pos")[0]
    recipient_quat = _array(recipient, "scene_object_quat")[0]
    donor_quat = _array(donor, "scene_object_quat")[0]
    swapped_target = _same_pose(
        donor_pos[target_index],
        donor_quat[target_index],
        recipient_pos[distractor_index],
        recipient_quat[distractor_index],
    )
    swapped_distractor = _same_pose(
        donor_pos[distractor_index],
        donor_quat[distractor_index],
        recipient_pos[target_index],
        recipient_quat[target_index],
    )
    other_indices = [
        index
        for index in range(len(recipient_pos))
        if index not in {target_index, distractor_index}
    ]
    other_scene_fixed = bool(
        np.array_equal(recipient_pos[other_indices], donor_pos[other_indices])
        and np.array_equal(recipient_quat[other_indices], donor_quat[other_indices])
    )
    robot_differences = {
        name: _array_max_abs_difference(recipient, donor, name)
        for name in ROBOT_ARRAYS
    }
    camera_differences = {
        name: _array_max_abs_difference(recipient, donor, name)
        for name in CAMERA_ARRAYS
    }
    identity = {
        "model_id": recipient.manifest.model_id == donor.manifest.model_id,
        "prompt": recipient.manifest.prompt == donor.manifest.prompt,
        "task_id": recipient.manifest.task_id == donor.manifest.task_id,
        "action_shape": recipient.action_chunks().shape == donor.action_chunks().shape,
        "layout_id": _environment_value(recipient, "layout_id")
        == _environment_value(donor, "layout_id")
        == layout_id,
        "seed": _environment_value(recipient, "seed")
        == _environment_value(donor, "seed")
        == seed,
        "robot_state": all(value == 0.0 for value in robot_differences.values()),
        "camera_state": all(value == 0.0 for value in camera_differences.values()),
        "target_pose_exchanged": swapped_target,
        "distractor_pose_exchanged": swapped_distractor,
        "other_scene_state": other_scene_fixed,
    }
    natural_delta = np.asarray(donor.action_chunks()[0], dtype=np.float32) - np.asarray(
        recipient.action_chunks()[0], dtype=np.float32
    )
    natural_l2 = float(np.linalg.norm(natural_delta))
    token_regions = _pair_token_regions(
        recipient,
        donor,
        target_index=target_index,
        distractor_index=distractor_index,
    )
    # The two captures intentionally preserve their own saved diffusion noise.
    # Their action difference is useful context, but it is not a valid causal
    # gate because noise can also change the action.  The patch-study runner
    # later recomputes the natural effect with the recipient's noise shared by
    # both scenes.
    pair_valid = all(identity.values())
    dataset_id = str(recipient.manifest.metadata.get("dataset_id") or dataset.root.name)
    root_id = str(dataset.root)
    return CounterfactualPairManifest(
        pair_id=pair_id,
        recipe=CounterfactualRecipe(
            kind="pose_exchange",
            target_object=target_object,
            distractor_object=distractor_object,
            changed_variables=(f"{target_object}.pose", f"{distractor_object}.pose"),
            held_fixed={
                "model_id": recipient.manifest.model_id,
                "instruction": recipient.manifest.prompt,
                "task_id": recipient.manifest.task_id,
                "benchmark": recipient.manifest.env_id,
                "layout_id": layout_id,
                "seed": seed,
                "camera": "agentview_image,robot0_eye_in_hand_image",
                "robot_state": "exact after matched observation refresh",
                "policy_call_index": 0,
            },
            parameters={
                "target_object_index": target_index,
                "distractor_object_index": distractor_index,
            },
        ),
        recipient=RecipientSpec(
            trace=TraceRef(
                trace_id=recipient.manifest.trace_id,
                dataset_id=dataset_id,
                dataset_root_id=root_id,
                episode_id=recipient.manifest.episode_id,
            ),
            policy_call=PolicyCallRef(
                trace_id=recipient.manifest.trace_id,
                policy_call_index=0,
                timestep=0,
                frame_index=0,
            ),
        ),
        donor=DonorSpec(
            trace=TraceRef(
                trace_id=donor.manifest.trace_id,
                dataset_id=dataset_id,
                dataset_root_id=root_id,
                episode_id=donor.manifest.episode_id,
            ),
            policy_call=PolicyCallRef(
                trace_id=donor.manifest.trace_id,
                policy_call_index=0,
                timestep=0,
                frame_index=0,
            ),
            metadata={"shared_noise_source": "recipient"},
        ),
        compatibility=identity,
        validation={
            "pair_valid": pair_valid,
            "natural_action_delta_l2_separate_saved_noise": natural_l2,
            "natural_action_delta_max_abs_separate_saved_noise": float(
                np.max(np.abs(natural_delta))
            ),
            "robot_max_abs_differences": robot_differences,
            "camera_max_abs_differences": camera_differences,
            "token_regions": token_regions,
            "token_selection": {
                "object_tokens_per_region": OBJECT_TOKEN_COUNT,
                "background_tokens": BACKGROUND_TOKEN_COUNT,
                "camera_id": "main",
                "method": "bbox-overlap then nearest patch centers",
            },
        },
        media={
            "recipient": {"trace_id": recipient.manifest.trace_id, "timestep": 0},
            "donor": {"trace_id": donor.manifest.trace_id, "timestep": 0},
        },
        provenance={
            "source": "pi05_pose_exchange_capture",
            "recipient_scene_mutation": _environment_value(recipient, "scene_mutation"),
            "donor_scene_mutation": _environment_value(donor, "scene_mutation"),
        },
    )


def _pair_token_regions(
    recipient: TraceBundle,
    donor: TraceBundle,
    *,
    target_index: int,
    distractor_index: int,
) -> dict[str, Any]:
    recipient_tokens = _main_image_tokens(recipient)
    donor_tokens = _main_image_tokens(donor)
    recipient_boxes = _array(recipient, "camera_object_bbox")[0, 0]
    donor_boxes = _array(donor, "camera_object_bbox")[0, 0]
    recipient_target, recipient_distractor = _disjoint_tokens_for_bboxes(
        recipient_tokens,
        recipient_boxes[target_index],
        recipient_boxes[distractor_index],
        count=OBJECT_TOKEN_COUNT,
    )
    donor_target, donor_distractor = _disjoint_tokens_for_bboxes(
        donor_tokens,
        donor_boxes[target_index],
        donor_boxes[distractor_index],
        count=OBJECT_TOKEN_COUNT,
    )
    recipient_both = _unique_exact((*recipient_target, *recipient_distractor))
    donor_both = _unique_exact((*donor_target, *donor_distractor))
    if len(recipient_both) != 2 * OBJECT_TOKEN_COUNT or len(donor_both) != 2 * OBJECT_TOKEN_COUNT:
        raise AssertionError("joint object-token assignment did not produce disjoint regions")
    recipient_background = _background_tokens(
        recipient_tokens,
        (recipient_boxes[target_index], recipient_boxes[distractor_index]),
        count=BACKGROUND_TOKEN_COUNT,
    )
    donor_background = _background_tokens(
        donor_tokens,
        (donor_boxes[target_index], donor_boxes[distractor_index]),
        count=BACKGROUND_TOKEN_COUNT,
    )
    broad_regions = _broad_prefix_regions(recipient.tokens, donor.tokens)
    return {
        "target": {"recipient": recipient_target, "donor": donor_target},
        "distractor": {
            "recipient": recipient_distractor,
            "donor": donor_distractor,
        },
        "both": {"recipient": recipient_both, "donor": donor_both},
        "complement": {
            "recipient": recipient_background,
            "donor": donor_background,
        },
        **broad_regions,
    }


def _broad_prefix_regions(
    recipient_tokens: pd.DataFrame,
    donor_tokens: pd.DataFrame,
) -> dict[str, dict[str, list[int]]]:
    """Resolve broad patch scopes that distinguish localization from hook failure."""
    selectors = {
        "main_camera": {"token_kind": "image", "camera_id": "main"},
        "wrist_camera": {"token_kind": "image", "camera_id": "wrist"},
        "active_images": {"token_kind": "image", "prefix_mask": True},
        "language_active": {"token_kind": "language", "attention_mask": True},
        "full_prefix": {"token_space_id": "pi05.prefix"},
    }
    regions: dict[str, dict[str, list[int]]] = {}
    for name, selector in selectors.items():
        recipient = _selected_token_indices(recipient_tokens, selector)
        donor = _selected_token_indices(donor_tokens, selector)
        if not recipient or len(recipient) != len(donor):
            raise ValueError(f"broad token region {name!r} differs between pair traces")
        regions[name] = {"recipient": recipient, "donor": donor}
    return regions


def _selected_token_indices(
    tokens: pd.DataFrame,
    selector: Mapping[str, Any],
) -> list[int]:
    selected = tokens.loc[tokens["policy_call_index"].astype(int) == 0]
    for column, value in selector.items():
        if column not in selected:
            raise ValueError(f"token table lacks {column!r} for broad region selection")
        if isinstance(value, bool):
            mask = selected[column].eq(value).fillna(False)
        else:
            mask = selected[column].astype(str) == str(value)
        selected = selected.loc[mask]
    return selected["token_index"].astype(int).tolist()


def _tokens_for_bbox(tokens: pd.DataFrame, bbox: np.ndarray, *, count: int) -> list[int]:
    scored = _token_bbox_scores(tokens, bbox)
    ordered = scored.sort_values(
        ["overlap", "distance", "token_index"],
        ascending=[False, True, True],
    )
    return ordered.head(count)["token_index"].astype(int).tolist()


def _disjoint_tokens_for_bboxes(
    tokens: pd.DataFrame,
    first_bbox: np.ndarray,
    second_bbox: np.ndarray,
    *,
    count: int,
) -> tuple[list[int], list[int]]:
    """Choose equally sized, non-overlapping patch sets for two nearby objects.

    We alternate between each object's ranked list.  When both objects want the
    same patch, the object whose turn comes first keeps it and the other moves
    to its next-best unused patch.  Alternating which object picks first on each
    round avoids systematically giving every ambiguous patch to one object.
    """
    if len(tokens) < 2 * count:
        raise ValueError("not enough image patches for two disjoint object regions")
    rankings = [
        _tokens_for_bbox(tokens, first_bbox, count=len(tokens)),
        _tokens_for_bbox(tokens, second_bbox, count=len(tokens)),
    ]
    selected: list[list[int]] = [[], []]
    used: set[int] = set()
    cursors = [0, 0]
    for round_index in range(count):
        for object_index in ((0, 1) if round_index % 2 == 0 else (1, 0)):
            ranking = rankings[object_index]
            while cursors[object_index] < len(ranking):
                token_index = int(ranking[cursors[object_index]])
                cursors[object_index] += 1
                if token_index not in used:
                    selected[object_index].append(token_index)
                    used.add(token_index)
                    break
            else:
                raise ValueError("could not assign disjoint image patches to both objects")
    return selected[0], selected[1]


def _background_tokens(
    tokens: pd.DataFrame,
    boxes: tuple[np.ndarray, np.ndarray],
    *,
    count: int,
) -> list[int]:
    first = _token_bbox_scores(tokens, boxes[0])
    second = _token_bbox_scores(tokens, boxes[1])
    scored = first[["token_index", "center_x", "center_y"]].copy()
    scored["overlap"] = first["overlap"].to_numpy() + second["overlap"].to_numpy()
    centers = np.asarray([_bbox_center(box) for box in boxes], dtype=np.float64)
    token_centers = scored[["center_x", "center_y"]].to_numpy(dtype=np.float64)
    distances = np.linalg.norm(token_centers[:, None, :] - centers[None, :, :], axis=-1)
    scored["distance"] = distances.min(axis=1)
    candidates = scored.loc[scored["overlap"] == 0.0]
    ordered = candidates.sort_values(
        ["distance", "token_index"], ascending=[False, True]
    )
    if len(ordered) < count:
        raise ValueError("not enough background image patches outside the two object boxes")
    return ordered.head(count)["token_index"].astype(int).tolist()


def _token_bbox_scores(tokens: pd.DataFrame, bbox: np.ndarray) -> pd.DataFrame:
    x0, y0, x1, y1 = np.asarray(bbox, dtype=np.float64)
    scored = tokens[["token_index", "pixel_x0", "pixel_x1", "pixel_y0", "pixel_y1"]].copy()
    overlap_x = np.maximum(
        0.0,
        np.minimum(scored["pixel_x1"].to_numpy(), x1)
        - np.maximum(scored["pixel_x0"].to_numpy(), x0),
    )
    overlap_y = np.maximum(
        0.0,
        np.minimum(scored["pixel_y1"].to_numpy(), y1)
        - np.maximum(scored["pixel_y0"].to_numpy(), y0),
    )
    scored["overlap"] = overlap_x * overlap_y
    scored["center_x"] = (scored["pixel_x0"] + scored["pixel_x1"]) / 2.0
    scored["center_y"] = (scored["pixel_y0"] + scored["pixel_y1"]) / 2.0
    center_x, center_y = _bbox_center(bbox)
    scored["distance"] = np.hypot(
        scored["center_x"] - center_x,
        scored["center_y"] - center_y,
    )
    return scored


def _bbox_center(bbox: np.ndarray) -> tuple[float, float]:
    x0, y0, x1, y1 = np.asarray(bbox, dtype=np.float64)
    return float((x0 + x1) / 2.0), float((y0 + y1) / 2.0)


def _main_image_tokens(bundle: TraceBundle) -> pd.DataFrame:
    tokens = bundle.tokens
    selected = tokens.loc[
        (tokens["policy_call_index"].astype(int) == 0)
        & (tokens["token_kind"].astype(str) == "image")
        & (tokens["camera_id"].astype(str) == "main")
    ].copy()
    if len(selected) < BACKGROUND_TOKEN_COUNT:
        raise ValueError(f"trace {bundle.manifest.trace_id} lacks main-camera patch tokens")
    return selected.reset_index(drop=True)


def _object_indices(bundle: TraceBundle) -> dict[str, int]:
    rows = bundle.scene_state.dropna(subset=["object_index", "object_name"])
    return {
        str(row.object_name): int(row.object_index)
        for row in rows.itertuples(index=False)
    }


def _environment_value(bundle: TraceBundle, key: str) -> Any:
    environment = bundle.manifest.metadata.get("environment")
    return environment.get(key) if isinstance(environment, Mapping) else None


def _array(bundle: TraceBundle, name: str) -> np.ndarray:
    return np.asarray(bundle.array(name, mmap=True))


def _array_max_abs_difference(left: TraceBundle, right: TraceBundle, name: str) -> float:
    left_array = _array(left, name)
    right_array = _array(right, name)
    if left_array.shape != right_array.shape:
        return float("inf")
    delta = np.asarray(left_array, dtype=np.float64) - np.asarray(right_array, dtype=np.float64)
    return float(np.max(np.abs(delta))) if delta.size else 0.0


def _same_pose(
    left_pos: np.ndarray,
    left_quat: np.ndarray,
    right_pos: np.ndarray,
    right_quat: np.ndarray,
) -> bool:
    return bool(np.array_equal(left_pos, right_pos) and np.array_equal(left_quat, right_quat))


def _unique_exact(values: Sequence[int]) -> list[int]:
    return list(dict.fromkeys(int(value) for value in values))


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


__all__ = [
    "build_pose_exchange_pair_manifests",
    "save_pose_exchange_pair_manifests",
]
