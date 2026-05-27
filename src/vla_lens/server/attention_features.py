"""Activation-token and image-patch feature helpers."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from vla_lens.server.attention_tokens import _camera_patch_layout, _image_token_index_for_patch
from vla_lens.server.common import (
    _json_scalar,
    _policy_call_axis_selection,
    _query_call_index,
    _query_one,
    _take_axis_value,
    _take_axis_values,
    _take_policy_call_value,
)
from vla_lens.server.metrics import _policy_calls
from vla_lens.traces import TraceBundle


def _patch_features_payload(bundle: TraceBundle, query: dict[str, list[str]]) -> dict[str, Any]:
    name = _query_one(query, "name")
    calls = _policy_calls(bundle)
    if not calls:
        return {"available": False, "reason": "No policy calls are available."}
    call = calls[_query_call_index(query)]
    camera = _query_one(query, "camera")
    row = int(_query_one(query, "row"))
    col = int(_query_one(query, "col"))
    feature = int(query.get("feature", ["0"])[0])
    matches = bundle.model_sites.loc[bundle.model_sites["name"].astype(str) == name]
    if matches.empty:
        raise KeyError(name)
    record = matches.iloc[0]
    token_matrix = _activation_token_matrix(bundle, name, call, query)
    cameras = bundle.cameras()
    if camera not in cameras:
        raise KeyError(f"Unknown camera {camera!r}; available={cameras}")
    token_index, row, col = _image_token_index_for_patch(
        bundle,
        record,
        call,
        token_matrix.shape[0],
        camera,
        row,
        col,
    )
    if token_index is None:
        layout = _camera_patch_layout(bundle, token_matrix.shape[0], text_tokens=0)
        camera_index = cameras.index(camera)
        grid_size = int(layout["grid_size"])
        row = max(0, min(row, grid_size - 1))
        col = max(0, min(col, grid_size - 1))
        token_index = camera_index * int(layout["patches_per_image"]) + row * grid_size + col
    token_index = max(0, min(int(token_index), token_matrix.shape[0] - 1))
    vector = token_matrix[token_index]
    safe = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(np.abs(safe))[::-1]
    positive = np.argsort(safe)[::-1]
    negative = np.argsort(safe)
    feature = max(0, min(feature, vector.shape[0] - 1))
    feature_rank = int(np.where(order == feature)[0][0]) + 1 if vector.size else None
    return {
        "available": True,
        "name": name,
        "call": call,
        "camera": camera,
        "patch_row": row,
        "patch_col": col,
        "token_index": int(token_index),
        "feature": feature,
        "feature_value": _json_scalar(float(vector[feature])),
        "feature_rank_by_abs": feature_rank,
        "feature_count": int(vector.shape[0]),
        "top_abs": [
            {"index": int(index), "value": _json_scalar(float(vector[index]))}
            for index in order[:32]
        ],
        "top_positive": [
            {"index": int(index), "value": _json_scalar(float(vector[index]))}
            for index in positive[:16]
        ],
        "top_negative": [
            {"index": int(index), "value": _json_scalar(float(vector[index]))}
            for index in negative[:16]
        ],
    }

def _activation_token_matrix(
    bundle: TraceBundle,
    name: str,
    call: dict[str, Any],
    query: dict[str, list[str]],
) -> np.ndarray:
    matches = bundle.model_sites.loc[bundle.model_sites["name"].astype(str) == name]
    if matches.empty:
        raise KeyError(name)
    record = matches.iloc[0]
    axes = json.loads(str(record.get("axes") or "[]"))
    array = bundle.model_site(name, mmap=True)
    value, remaining_axes = _take_policy_call_value(array, axes, call)
    if "generation_step" in remaining_axes:
        generation_step = query.get("generation_step", [""])[0]
        step = int(generation_step) if generation_step not in {"", None} else 0
        value = _take_axis_value(value, remaining_axes, "generation_step", step)
        remaining_axes = [axis for axis in remaining_axes if axis != "generation_step"]
    matrix = np.asarray(value, dtype=np.float32)
    if "token" in remaining_axes:
        token_axis = remaining_axes.index("token")
        matrix = np.moveaxis(matrix, token_axis, 0)
        if matrix.ndim == 1:
            matrix = matrix.reshape(matrix.shape[0], 1)
        else:
            matrix = matrix.reshape(matrix.shape[0], -1)
    elif matrix.ndim != 2:
        matrix = matrix.reshape(-1, matrix.shape[-1])
    if matrix.ndim != 2:
        raise ValueError(f"Expected token x channel activation for {name!r}, got {matrix.shape}")
    return matrix

def _activation_token_feature_vector(
    bundle: TraceBundle,
    name: str,
    call: dict[str, Any],
    query: dict[str, list[str]],
    feature: int,
) -> tuple[np.ndarray, int, int]:
    matches = bundle.model_sites.loc[bundle.model_sites["name"].astype(str) == name]
    if matches.empty:
        raise KeyError(name)
    record = matches.iloc[0]
    axes = json.loads(str(record.get("axes") or "[]"))
    array = bundle.model_site(name, mmap=True)
    selections = _policy_call_axis_selection(axes, call)
    if "generation_step" in axes:
        generation_step = query.get("generation_step", [""])[0]
        step = int(generation_step) if generation_step not in {"", None} else 0
        selections["generation_step"] = step
    if "channel" not in axes:
        raise ValueError(f"Expected channel axis for {name!r}, got axes={axes!r}")
    channel_axis = axes.index("channel")
    feature_count = int(array.shape[channel_axis])
    feature = max(0, min(int(feature), max(0, feature_count - 1)))
    selections["channel"] = feature
    value, remaining_axes = _take_axis_values(array, axes, selections)
    vector = np.asarray(value, dtype=np.float32)
    if "token" in remaining_axes:
        token_axis = remaining_axes.index("token")
        vector = np.moveaxis(vector, token_axis, 0).reshape(vector.shape[token_axis], -1)
        vector = vector[:, 0]
    else:
        vector = vector.reshape(-1)
    return vector, feature, feature_count
