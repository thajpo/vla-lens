"""Attention dashboard server helpers."""


from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Mapping

import numpy as np

from vla_lens.server_common import (
    _json_scalar,
    _optional_int,
    _patches_per_image,
    _policy_call_axis_selection,
    _query_call_index,
    _query_one,
    _round,
    _take_axis_value,
    _take_axis_values,
    _take_policy_call_value,
)
from vla_lens.server_metrics import (
    _policy_calls,
)
from vla_lens.traces import TraceBundle

_NUMERIC_TOKEN_RE = re.compile(r"^-?\d+(?:\.0)?$")


def _attention_map_payload(bundle: TraceBundle, query: dict[str, list[str]]) -> dict[str, Any]:
    calls = _policy_calls(bundle)
    if not calls:
        return {"available": False, "reason": "No policy calls are available."}
    kind = query.get("kind", ["expert"])[0]
    call = calls[_query_call_index(query)]
    generation_step = int(query.get("generation_step", ["0"])[0] or 0)
    name = query.get("name", [""])[0] or None
    head = query.get("head", [""])[0]
    head_index = int(head) if head not in {"", None} else None
    query_token = query.get("query_token", [""])[0]
    query_token_index = int(query_token) if query_token not in {"", None} else None
    try:
        key_mass, selected_site, axis_selection = _attention_key_mass_from_trace(
            bundle,
            kind,
            call,
            generation_step,
            name=name,
            head=head_index,
            query_token=query_token_index,
        )
    except KeyError:
        return _not_captured_in_profile(
            f"{kind} attention maps require attention arrays stored in the .vlatrace bundle.",
            kind=kind,
            generation_step=generation_step,
            name=name,
            head=head_index,
            query_token=query_token_index,
        )
    except ValueError as error:
        return {
            "available": False,
            "reason": "selected_axis_unavailable",
            "detail": str(error),
            "kind": kind,
            "generation_step": generation_step,
            "name": name,
            "head": head_index,
            "query_token": query_token_index,
        }
    layout = _attention_camera_layout(bundle, key_mass.shape[0])
    return {
        "available": True,
        "kind": kind,
        "call": call,
        "generation_step": generation_step,
        **axis_selection,
        "site": selected_site,
        "source": "vlatrace",
        **layout,
        "coarse": {
            "image": _json_scalar(float(np.nansum(key_mass[: int(layout["image_tokens"])]))),
            "prompt": None,
            "action_suffix": None,
        },
        "maps": _camera_maps_from_trace_key_mass(bundle, key_mass, layout),
    }

def _expert_token_attention_payload(
    bundle: TraceBundle,
    source_name: str,
    call: dict[str, Any],
    generation_step: int,
    token_index: int,
) -> dict[str, Any] | None:
    """Return action-query attention over prefix image/text tokens for one expert token."""
    attention = _expert_attention_for_token(
        bundle,
        source_name,
        call,
        generation_step,
        token_index,
    )
    if attention is None:
        return None
    key_mass, site_name = attention
    prefix_rows = _token_rows_for_space(bundle, call, "pi05.prefix")
    if prefix_rows.empty:
        return None

    prefix_count = _token_count(prefix_rows)
    prefix_mass = np.asarray(key_mass[:prefix_count], dtype=np.float32)
    action_mass = np.asarray(key_mass[prefix_count:], dtype=np.float32)
    maps, top_image_patches, image_mass = _image_attention_from_prefix_rows(
        bundle,
        prefix_rows,
        prefix_mass,
    )
    top_prompt_tokens, prompt_mass, prompt, prompt_tokens = _prompt_attention_from_prefix_rows(
        prefix_rows,
        prefix_mass,
    )
    return {
        "attention_site": site_name,
        "attention_coarse": {
            "image": _json_scalar(float(image_mass)),
            "prompt": _json_scalar(float(prompt_mass)),
            "action_suffix": (
                _json_scalar(float(np.nansum(action_mass))) if action_mass.size else 0.0
            ),
        },
        "top_prompt_tokens": top_prompt_tokens,
        "prompt_tokens": prompt_tokens,
        "top_image_patches": top_image_patches,
        "maps": maps,
        "prompt": bundle.manifest.prompt or prompt,
    }

def _expert_attention_for_token(
    bundle: TraceBundle,
    source_name: str,
    call: dict[str, Any],
    generation_step: int,
    token_index: int,
) -> tuple[np.ndarray, str] | None:
    matches = bundle.model_sites.loc[bundle.model_sites["name"].astype(str) == source_name]
    if matches.empty:
        return None
    source = matches.iloc[0]
    layer = source.get("layer")
    candidates = _expert_attention_site_candidates(bundle, layer)
    if candidates.empty:
        return None
    record = candidates.iloc[0]
    name = str(record["name"])
    axes = json.loads(str(record.get("axes") or "[]"))
    array = bundle.model_site(name, mmap=True)
    selections = _policy_call_axis_selection(axes, call)
    if "generation_step" in axes:
        selections["generation_step"] = generation_step
    if "query_token" in axes:
        selections["query_token"] = token_index
    value, remaining_axes = _take_axis_values(array, axes, selections)
    value_array = np.asarray(value, dtype=np.float32)
    if "head" in remaining_axes:
        value_array = np.nanmean(value_array, axis=remaining_axes.index("head"))
        remaining_axes = [axis for axis in remaining_axes if axis != "head"]
    if "key_token" in remaining_axes:
        key_axis = remaining_axes.index("key_token")
        value_array = np.moveaxis(value_array, key_axis, -1)
    return np.asarray(value_array, dtype=np.float32).reshape(-1), name

def _expert_attention_site_candidates(bundle: TraceBundle, layer: Any) -> Any:
    if bundle.model_sites.empty:
        return bundle.model_sites
    table = bundle.model_sites.copy()
    names = table["name"].astype(str)
    table = table.loc[names.str.contains(".expert.layers.", regex=False)].copy()
    if table.empty:
        return table
    if layer is not None and str(layer) != "nan":
        numeric_layer = float(layer)
        table = table.loc[table.get("layer").astype(float) == numeric_layer].copy()
    if table.empty:
        return table
    axes = table.get("axes", "").astype(str)
    roles = table.get("role", "").astype(str)
    tensor_types = table.get("tensor_type", "").astype(str)
    names = table["name"].astype(str)
    table = table.loc[
        axes.str.contains("query_token")
        & axes.str.contains("key_token")
        & (
            (roles == "attention_probs")
            | (tensor_types == "attention_probs")
            | names.str.endswith(".by_step.attention")
        )
    ].copy()
    if table.empty:
        return table
    table["_priority"] = np.select(
        [
            table["name"].astype(str).str.endswith(".attention.attention_probs"),
            table["name"].astype(str).str.endswith(".by_step.attention"),
        ],
        [0, 1],
        default=2,
    )
    return table.sort_values(["_priority", "name"])

def _attention_key_mass_from_trace(
    bundle: TraceBundle,
    kind: str,
    call: dict[str, Any],
    generation_step: int,
    *,
    name: str | None = None,
    head: int | None = None,
    query_token: int | None = None,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    matches = _attention_site_matches(bundle, kind, name=name)
    if matches.empty:
        raise KeyError(kind)
    record = matches.iloc[-1]
    name = str(record["name"])
    axes = json.loads(str(record.get("axes") or "[]"))
    array = bundle.model_site(name, mmap=True)
    selections = _policy_call_axis_selection(axes, call)
    if "generation_step" in axes:
        selections["generation_step"] = generation_step
    if head is not None:
        if "head" not in axes:
            raise ValueError("Selected head is not available for this attention capture.")
        selections["head"] = head
    if query_token is not None:
        if "query_token" not in axes:
            raise ValueError("Selected looking slot is not available for this attention capture.")
        selections["query_token"] = query_token
    value, remaining_axes = _take_axis_values(array, axes, selections)
    value_array = np.asarray(value, dtype=np.float32)
    axis_selection: dict[str, Any] = {
        "head": None,
        "head_mode": "average",
        "query_token": None,
        "query_mode": "average",
    }
    if head is not None:
        axis_selection["head"] = int(head)
        axis_selection["head_mode"] = "selected"
    if query_token is not None:
        axis_selection["query_token"] = int(query_token)
        axis_selection["query_mode"] = "selected"
    if "head" in remaining_axes:
        head_axis = remaining_axes.index("head")
        value_array = np.nanmean(value_array, axis=head_axis)
        axis_selection["head_mode"] = "average"
        remaining_axes = [axis for axis in remaining_axes if axis != "head"]
    if "query_token" in remaining_axes:
        query_axis = remaining_axes.index("query_token")
        value_array = np.nanmean(value_array, axis=query_axis)
        axis_selection["query_mode"] = "average"
        remaining_axes = [axis for axis in remaining_axes if axis != "query_token"]
    return value_array.reshape(-1), name, axis_selection

def _attention_site_matches(bundle: TraceBundle, kind: str, *, name: str | None = None) -> Any:
    if bundle.model_sites.empty:
        return bundle.model_sites
    table = bundle.model_sites.copy()
    names = table["name"].astype(str)
    if name:
        table = table.loc[names == name].copy()
        names = table["name"].astype(str)
    roles = table.get("role", "").astype(str)
    tensor_types = table.get("tensor_type", "").astype(str)
    table = table.loc[
        (tensor_types.isin({"attention", "attention_probs"}) | (roles == "attention_probs"))
        & names.str.contains(f"pi05.{kind}.", regex=False)
    ].copy()
    if table.empty:
        return table
    table["_layer_sort"] = table.get("layer", 0).fillna(0).astype(float)
    table["_key_mass_sort"] = names.loc[table.index].str.contains("attention_key_mass").astype(int)
    return table.sort_values(["_key_mass_sort", "_layer_sort", "name"])

def _camera_maps_from_trace_key_mass(
    bundle: TraceBundle,
    key_mass: np.ndarray,
    layout: dict[str, int],
) -> dict[str, Any]:
    maps: dict[str, Any] = {}
    grid_size = int(layout["grid_size"])
    patches_per_image = int(layout["patches_per_image"])
    image_slots = int(layout["image_slots"])
    for camera_index, camera in enumerate(bundle.cameras()):
        if camera_index >= image_slots:
            continue
        start = camera_index * patches_per_image
        end = start + patches_per_image
        if end > key_mass.shape[0]:
            continue
        values = key_mass[start:end].reshape(grid_size, grid_size)
        maps[camera] = {
            "values": _round(values),
            "token_start": start,
            "token_end": end - 1,
            "min": _json_scalar(float(np.nanmin(values))),
            "max": _json_scalar(float(np.nanmax(values))),
        }
    return maps

def _attention_camera_layout(bundle: TraceBundle, key_count: int) -> dict[str, int]:
    if not bundle.model_sites.empty:
        matches = bundle.model_sites.loc[
            bundle.model_sites["name"].astype(str) == "pi05.vlm.prefix.image_hidden_tokens"
        ]
        if not matches.empty:
            metadata = json.loads(str(matches.iloc[0].get("metadata") or "{}"))
            image_tokens = int(metadata.get("patches_per_image") or 0) * int(
                metadata.get("image_slots") or 0
            )
            if image_tokens > 0 and key_count >= image_tokens:
                return _camera_patch_layout_from_record(
                    bundle,
                    matches.iloc[0],
                    key_count,
                    text_tokens=key_count - image_tokens,
                )
    return _camera_patch_layout(bundle, key_count, text_tokens=0)

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

def _camera_patch_layout(
    bundle: TraceBundle,
    token_count: int,
    *,
    text_tokens: int,
) -> dict[str, int]:
    image_tokens = max(0, token_count - text_tokens)
    camera_count = max(1, len(bundle.cameras()))
    candidate = image_tokens // camera_count if image_tokens % camera_count == 0 else image_tokens
    root = int(round(float(np.sqrt(max(1, candidate)))))
    patches_per_image = candidate if root * root == candidate else _patches_per_image(image_tokens)
    grid_size = int(round(float(np.sqrt(patches_per_image)))) if patches_per_image else 0
    image_slots = image_tokens // patches_per_image if patches_per_image else 0
    return {
        "grid_size": grid_size,
        "grid_height": grid_size,
        "grid_width": grid_size,
        "patches_per_image": patches_per_image,
        "image_tokens": image_tokens,
        "text_tokens": text_tokens,
        "image_slots": min(image_slots, len(bundle.cameras())),
    }

def _camera_patch_layout_from_record(
    bundle: TraceBundle,
    record: Any,
    token_count: int,
    *,
    text_tokens: int,
) -> dict[str, int]:
    metadata = json.loads(str(record.get("metadata") or "{}"))
    patches_per_image = int(metadata.get("patches_per_image") or 0)
    grid_height = int(metadata.get("grid_height") or metadata.get("grid_size") or 0)
    grid_width = int(metadata.get("grid_width") or metadata.get("grid_size") or 0)
    if patches_per_image > 0 and grid_height > 0 and grid_width > 0:
        image_tokens = max(0, token_count - text_tokens)
        return {
            "grid_size": grid_height if grid_height == grid_width else 0,
            "grid_height": grid_height,
            "grid_width": grid_width,
            "patches_per_image": patches_per_image,
            "image_tokens": image_tokens,
            "text_tokens": text_tokens,
            "image_slots": min(image_tokens // patches_per_image, len(bundle.cameras())),
        }
    return _camera_patch_layout(bundle, token_count, text_tokens=text_tokens)

def _token_rows_for_space(
    bundle: TraceBundle,
    call: dict[str, Any],
    token_space_id: str,
) -> Any:
    rows = bundle.tokens
    if rows.empty or "token_space_id" not in rows:
        return rows.iloc[0:0].copy()
    rows = rows.loc[rows["token_space_id"].astype(str) == token_space_id].copy()
    if rows.empty:
        return rows
    if "policy_call_index" in rows:
        call_index = int(call.get("model_call_index", call.get("index", 0)))
        call_rows = rows.loc[rows["policy_call_index"].astype(int) == call_index].copy()
        if not call_rows.empty:
            rows = call_rows
    return rows.sort_values("token_index").reset_index(drop=True)

def _token_count(rows: Any) -> int:
    if rows.empty or "token_index" not in rows:
        return 0
    return int(rows["token_index"].max()) + 1

def _image_token_rows_for_site(
    bundle: TraceBundle,
    record: Any,
    call: dict[str, Any],
    token_count: int,
) -> Any:
    token_space_id = str(record.get("token_space_id") or "")
    if not token_space_id or token_space_id.lower() == "nan":
        return bundle.tokens.iloc[0:0].copy()
    rows = _token_rows_for_space(bundle, call, token_space_id)
    if rows.empty:
        return rows
    token_kind = rows.get("token_kind", "").astype(str)
    image_rows = rows.loc[token_kind == "image"].copy()
    if image_rows.empty or "token_index" not in image_rows:
        return image_rows
    image_rows = image_rows.loc[image_rows["token_index"].astype(int) < token_count].copy()
    return image_rows.sort_values("token_index").reset_index(drop=True)

def _camera_patch_maps_from_token_rows(
    bundle: TraceBundle,
    image_rows: Any,
    feature_values: np.ndarray,
) -> tuple[dict[str, Any], dict[str, int | None]]:
    maps: dict[str, Any] = {}
    grid_heights: list[int] = []
    grid_widths: list[int] = []
    patch_counts: list[int] = []
    for camera in bundle.cameras():
        camera_rows = image_rows.loc[image_rows.get("camera_id", "").astype(str) == camera].copy()
        if camera_rows.empty:
            continue
        grid_height = int(camera_rows.get("patch_row", 0).max()) + 1
        grid_width = int(camera_rows.get("patch_col", 0).max()) + 1
        values = np.full((grid_height, grid_width), np.nan, dtype=np.float32)
        for row in camera_rows.to_dict("records"):
            token_index = int(row.get("token_index", 0))
            if token_index >= feature_values.shape[0]:
                continue
            patch_row = int(row.get("patch_row", 0))
            patch_col = int(row.get("patch_col", 0))
            if patch_row < grid_height and patch_col < grid_width:
                values[patch_row, patch_col] = float(feature_values[token_index])
        grid_heights.append(grid_height)
        grid_widths.append(grid_width)
        patch_counts.append(int(len(camera_rows)))
        finite_values = np.nan_to_num(values, nan=0.0)
        maps[camera] = {
            "values": _round(finite_values),
            "token_start": int(camera_rows["token_index"].min()),
            "token_end": int(camera_rows["token_index"].max()),
            "active_tokens": int(len(camera_rows)),
            "min": _json_scalar(float(np.nanmin(values))),
            "max": _json_scalar(float(np.nanmax(values))),
        }
    grid_height = grid_heights[0] if grid_heights and len(set(grid_heights)) == 1 else None
    grid_width = grid_widths[0] if grid_widths and len(set(grid_widths)) == 1 else None
    patches_per_image = patch_counts[0] if patch_counts and len(set(patch_counts)) == 1 else 0
    return maps, {
        "grid_size": grid_height if grid_height is not None and grid_height == grid_width else None,
        "grid_height": grid_height or 0,
        "grid_width": grid_width or 0,
        "patches_per_image": patches_per_image,
    }

def _image_token_index_for_patch(
    bundle: TraceBundle,
    record: Any,
    call: dict[str, Any],
    token_count: int,
    camera: str,
    row: int,
    col: int,
) -> tuple[int | None, int, int]:
    image_rows = _image_token_rows_for_site(bundle, record, call, token_count)
    if image_rows.empty:
        return None, row, col
    camera_rows = image_rows.loc[image_rows.get("camera_id", "").astype(str) == camera].copy()
    if camera_rows.empty:
        return None, row, col
    max_row = int(camera_rows.get("patch_row", 0).max())
    max_col = int(camera_rows.get("patch_col", 0).max())
    row = max(0, min(row, max_row))
    col = max(0, min(col, max_col))
    matches = camera_rows.loc[
        (camera_rows.get("patch_row", 0).astype(int) == row)
        & (camera_rows.get("patch_col", 0).astype(int) == col)
    ]
    if matches.empty:
        return None, row, col
    return int(matches.iloc[0].get("token_index", 0)), row, col

def _image_attention_from_prefix_rows(
    bundle: TraceBundle,
    prefix_rows: Any,
    prefix_mass: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    image_rows = prefix_rows.loc[prefix_rows.get("token_kind", "").astype(str) == "image"].copy()
    if image_rows.empty:
        return {}, [], 0.0
    cameras = bundle.cameras()
    maps: dict[str, Any] = {}
    patch_rows: list[dict[str, Any]] = []
    image_mass = 0.0
    for camera in cameras:
        camera_rows = image_rows.loc[image_rows.get("camera_id", "").astype(str) == camera].copy()
        if camera_rows.empty:
            continue
        max_row = int(camera_rows.get("patch_row", 0).max())
        max_col = int(camera_rows.get("patch_col", 0).max())
        values = np.full((max_row + 1, max_col + 1), np.nan, dtype=np.float32)
        for row in camera_rows.to_dict("records"):
            token_index = int(row.get("token_index", 0))
            if token_index >= prefix_mass.shape[0]:
                continue
            patch_row = int(row.get("patch_row", 0))
            patch_col = int(row.get("patch_col", 0))
            attention = float(prefix_mass[token_index])
            values[patch_row, patch_col] = attention
            image_mass += attention
            patch_rows.append(
                {
                    "camera": camera,
                    "row": patch_row,
                    "col": patch_col,
                    "token_index": token_index,
                    "attention": _json_scalar(attention),
                }
            )
        if values.size:
            maps[camera] = {
                "values": _round(np.nan_to_num(values, nan=0.0)),
                "token_start": int(camera_rows["token_index"].min()),
                "token_end": int(camera_rows["token_index"].max()),
                "min": _json_scalar(float(np.nanmin(values))),
                "max": _json_scalar(float(np.nanmax(values))),
            }
    patch_rows.sort(key=lambda item: float(item.get("attention") or 0.0), reverse=True)
    return maps, patch_rows[:24], image_mass

def _prompt_attention_from_prefix_rows(
    prefix_rows: Any,
    prefix_mass: np.ndarray,
) -> tuple[list[dict[str, Any]], float, str, list[dict[str, Any]]]:
    text_rows = prefix_rows.loc[prefix_rows.get("token_kind", "").astype(str) == "language"].copy()
    if text_rows.empty:
        return [], 0.0, "", []
    if "attention_mask" in text_rows:
        active = text_rows["attention_mask"].astype(bool)
        active_rows = text_rows.loc[active].copy()
    else:
        active_rows = text_rows
    if active_rows.empty:
        return [], 0.0, "", []
    start = int(text_rows["token_index"].min())
    token_records: list[dict[str, Any]] = []
    prompt_pieces: list[str] = []
    prompt_mass = 0.0
    for row in active_rows.to_dict("records"):
        token_index = int(row.get("token_index", 0))
        if token_index >= prefix_mass.shape[0]:
            continue
        attention = float(prefix_mass[token_index])
        prompt_mass += attention
        token_piece = _display_token_piece(row)
        prompt_pieces.append(token_piece)
        token_records.append(
            {
                "local_index": token_index - start,
                "prefix_index": token_index,
                "token_id": _json_scalar(row.get("token_id")),
                "token_piece": token_piece,
                "attention": _json_scalar(attention),
            }
        )
    top_records = sorted(
        token_records,
        key=lambda item: float(item.get("attention") or 0.0),
        reverse=True,
    )
    return (
        top_records[:24],
        prompt_mass,
        _join_token_pieces(prompt_pieces),
        token_records,
    )

def _display_token_piece(row: Mapping[str, Any]) -> str:
    """Return a human-readable token piece for numeric tokenizer rows."""

    raw_piece = row.get("token_piece")
    token_id = _optional_int(row.get("token_id"))
    piece = "" if raw_piece is None else str(raw_piece)
    if token_id is not None and (not piece or _NUMERIC_TOKEN_RE.match(piece)):
        decoded = _decode_paligemma_token(token_id)
        if decoded:
            piece = decoded
    return _clean_token_piece(piece)

@lru_cache(maxsize=4096)
def _decode_paligemma_token(token_id: int) -> str:
    tokenizer = _paligemma_tokenizer()
    if tokenizer is None:
        return ""
    try:
        piece = tokenizer.convert_ids_to_tokens([int(token_id)])
    except Exception:
        return ""
    if isinstance(piece, str):
        return piece
    if piece:
        return str(piece[0])
    return ""

@lru_cache(maxsize=1)
def _paligemma_tokenizer() -> Any | None:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            "google/paligemma-3b-pt-224",
            local_files_only=True,
        )
    except Exception:
        return None

def _clean_token_piece(piece: str) -> str:
    text = str(piece)
    text = text.replace("<0x0A>", "\n")
    text = text.replace("Ċ", "\n")
    return text

def _join_token_pieces(pieces: list[str]) -> str:
    text = "".join(piece.replace("▁", " ") for piece in pieces)
    text = text.replace("  ", " ")
    return text.strip()

def _not_captured_in_profile(reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "available": False,
        "reason": "not_captured_in_profile",
        "detail": reason,
        **extra,
    }

def _prompt_attention_payload(bundle: TraceBundle, query: dict[str, list[str]]) -> dict[str, Any]:
    calls = _policy_calls(bundle)
    if not calls:
        return {"available": False, "reason": "No policy calls are available."}
    kind = query.get("kind", ["expert"])[0]
    call = calls[_query_call_index(query)]
    generation_step = int(query.get("generation_step", ["0"])[0] or 0)
    name = query.get("name", [""])[0] or None
    head = query.get("head", [""])[0]
    head_index = int(head) if head not in {"", None} else None
    query_token = query.get("query_token", [""])[0]
    query_token_index = int(query_token) if query_token not in {"", None} else None
    try:
        key_mass, selected_site, axis_selection = _attention_key_mass_from_trace(
            bundle,
            kind,
            call,
            generation_step,
            name=name,
            head=head_index,
            query_token=query_token_index,
        )
    except KeyError:
        return _not_captured_in_profile(
            f"Prompt attention requires {kind} attention arrays stored in the .vlatrace bundle.",
            kind=kind,
            prompt=bundle.manifest.prompt,
            generation_step=generation_step,
            name=name,
            head=head_index,
            query_token=query_token_index,
        )
    except ValueError as error:
        return {
            "available": False,
            "reason": "selected_axis_unavailable",
            "detail": str(error),
            "kind": kind,
            "prompt": bundle.manifest.prompt,
            "generation_step": generation_step,
            "name": name,
            "head": head_index,
            "query_token": query_token_index,
        }
    prefix_rows = _token_rows_for_space(bundle, call, "pi05.prefix")
    if prefix_rows.empty:
        return _not_captured_in_profile(
            "Prompt attention requires token layout rows for pi05.prefix.",
            prompt=bundle.manifest.prompt,
            kind=kind,
            generation_step=generation_step,
            attention_site=selected_site,
        )
    prefix_count = _token_count(prefix_rows)
    prefix_mass = np.asarray(key_mass[:prefix_count], dtype=np.float32)
    _maps, top_image_patches, image_mass = _image_attention_from_prefix_rows(
        bundle,
        prefix_rows,
        prefix_mass,
    )
    top_text_tokens, prompt_mass, prompt, prompt_tokens = _prompt_attention_from_prefix_rows(
        prefix_rows,
        prefix_mass,
    )
    action_mass = np.asarray(key_mass[prefix_count:], dtype=np.float32)
    active_text_tokens = len(top_text_tokens)
    if "attention_mask" in prefix_rows:
        text_rows = prefix_rows.loc[prefix_rows.get("token_kind", "").astype(str) == "language"]
        active_text_tokens = int(text_rows.get("attention_mask", []).astype(bool).sum())
    return {
        "available": True,
        "kind": kind,
        "call": call,
        "generation_step": generation_step,
        **axis_selection,
        "attention_site": selected_site,
        "prompt": bundle.manifest.prompt or prompt,
        "active_text_tokens": active_text_tokens,
        "allocated_text_slots": int(
            (prefix_rows.get("token_kind", "").astype(str) == "language").sum()
        ),
        "expert_coarse": {
            "image": _json_scalar(float(image_mass)),
            "prompt": _json_scalar(float(prompt_mass)),
            "action_suffix": (
                _json_scalar(float(np.nansum(action_mass))) if action_mass.size else 0.0
            ),
        },
        "top_text_tokens": top_text_tokens,
        "prompt_tokens": prompt_tokens,
        "top_image_patches": top_image_patches,
    }

def _expert_token_model_sites_payload(
    bundle: TraceBundle,
    query: dict[str, list[str]],
) -> dict[str, Any]:
    name = _query_one(query, "name")
    calls = _policy_calls(bundle)
    if not calls:
        return {"available": False, "reason": "No policy calls are available."}
    call = calls[_query_call_index(query)]
    feature = int(query.get("feature", ["0"])[0])
    generation_step = int(query.get("generation_step", ["0"])[0] or 0)
    token_matrix = _activation_token_matrix(bundle, name, call, query)
    feature = max(0, min(feature, token_matrix.shape[1] - 1))
    values = token_matrix[:, feature]
    return {
        "available": True,
        "name": name,
        "call": call,
        "generation_step": generation_step,
        "feature": feature,
        "feature_count": int(token_matrix.shape[1]),
        "values": _round(values),
        "min": _json_scalar(float(np.nanmin(values))),
        "max": _json_scalar(float(np.nanmax(values))),
        "note": "Expert model_sites live on action/noise tokens, not image patch tokens.",
    }

def _expert_token_details_payload(
    bundle: TraceBundle,
    query: dict[str, list[str]],
) -> dict[str, Any]:
    name = _query_one(query, "name")
    calls = _policy_calls(bundle)
    if not calls:
        return {"available": False, "reason": "No policy calls are available."}
    call = calls[_query_call_index(query)]
    feature = int(query.get("feature", ["0"])[0])
    token_index = int(query.get("token_index", ["0"])[0])
    generation_step = int(query.get("generation_step", ["0"])[0] or 0)

    token_matrix = _activation_token_matrix(bundle, name, call, query)
    if token_matrix.ndim != 2:
        raise ValueError(f"Expected action-token x channel tensor, got {token_matrix.shape}")
    token_index = max(0, min(token_index, token_matrix.shape[0] - 1))
    feature = max(0, min(feature, token_matrix.shape[1] - 1))
    vector = token_matrix[token_index]
    safe = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(np.abs(safe))[::-1]
    feature_rank = int(np.where(order == feature)[0][0]) + 1 if vector.size else None

    action = _action_vector_for_token(bundle, call, token_index)
    attention = _expert_token_attention_payload(
        bundle,
        name,
        call,
        generation_step,
        token_index,
    )
    return {
        "available": True,
        "name": name,
        "call": call,
        "generation_step": generation_step,
        "token_index": token_index,
        "token_count": int(token_matrix.shape[0]),
        "feature": feature,
        "feature_value": _json_scalar(float(vector[feature])),
        "feature_rank_by_abs": feature_rank,
        "feature_count": int(vector.shape[0]),
        "top_abs": [
            {"index": int(index), "value": _json_scalar(float(vector[index]))}
            for index in order[:24]
        ],
        "attention_site": attention.get("attention_site") if attention else None,
        "attention_coarse": attention.get("attention_coarse") if attention else None,
        "top_prompt_tokens": attention.get("top_prompt_tokens", []) if attention else [],
        "prompt_tokens": attention.get("prompt_tokens", []) if attention else [],
        "top_image_patches": attention.get("top_image_patches", []) if attention else [],
        "maps": attention.get("maps", {}) if attention else {},
        "action": action,
        "note": (
            "This is one expert query/action token from the .vlatrace activation store. "
            "Image and prompt rows are attention mass from the matching expert layer/query token."
            if attention
            else "Attention details are unavailable unless captured into .vlatrace."
        ),
    }

def _action_vector_for_token(
    bundle: TraceBundle,
    call: dict[str, Any],
    token_index: int,
) -> dict[str, Any] | None:
    try:
        array = np.asarray(bundle.action_chunks(mmap=True)[int(call["index"])], dtype=np.float32)
    except KeyError:
        return None
    if array.ndim < 2 or array.shape[0] <= 0:
        return None
    token_index = max(0, min(int(token_index), array.shape[0] - 1))
    vector = np.asarray(array[token_index], dtype=np.float32).reshape(-1)
    safe = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(np.abs(safe))[::-1][:10]
    return {
        "source": "vlatrace.action_chunks",
        "dim": int(vector.shape[0]),
        "norm": _json_scalar(float(np.linalg.norm(safe))),
        "top_abs": [
            {"index": int(index), "value": _json_scalar(float(vector[index]))} for index in order
        ],
    }
