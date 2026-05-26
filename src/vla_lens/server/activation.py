"""Activation dashboard server helpers."""


from __future__ import annotations

import json
from typing import Any

import numpy as np

from vla_lens.server.attention import (
    _activation_token_feature_vector,
    _activation_token_matrix,
    _camera_patch_layout_from_record,
    _camera_patch_maps_from_token_rows,
    _image_token_rows_for_site,
    _not_captured_in_profile,
    _prompt_attention_from_prefix_rows,
    _token_count,
    _token_rows_for_space,
)
from vla_lens.server.common import (
    _json_list,
    _json_scalar,
    _query_call_index,
    _query_float,
    _query_one,
    _round,
    _take_axis_value,
    _take_policy_call_value,
)
from vla_lens.server.metrics import (
    _policy_calls,
)
from vla_lens.traces import TraceBundle


def _activation_sites_payload(bundle: TraceBundle) -> dict[str, Any]:
    if bundle.model_sites.empty:
        return {"sites": [], "runtime_collections": [], "architecture": {}}
    rows = []
    for record in bundle.model_sites.to_dict("records"):
        axes = json.loads(str(record.get("axes") or "[]"))
        metadata = json.loads(str(record.get("metadata") or "{}"))
        rows.append(
            {
                "name": str(record["name"]),
                "site_id": str(record.get("site_id") or record["name"]),
                "module": str(record.get("module") or ""),
                "layer": _json_scalar(record.get("layer")),
                "tensor_type": str(record.get("tensor_type") or ""),
                "token_kind": _json_scalar(record.get("token_kind")),
                "family": _json_scalar(record.get("family")),
                "role": _json_scalar(record.get("role")),
                "segment": _json_scalar(record.get("segment")),
                "materialization": _json_scalar(record.get("materialization")),
                "exactness": _json_scalar(record.get("exactness")),
                "token_space_id": _json_scalar(record.get("token_space_id")),
                "query_token_space_id": _json_scalar(record.get("query_token_space_id")),
                "key_token_space_id": _json_scalar(record.get("key_token_space_id")),
                "parent_site_id": _json_scalar(record.get("parent_site_id")),
                "summary_type": _json_scalar(record.get("summary_type")),
                "capture_family": _json_scalar(record.get("capture_family")),
                "view_kind": _json_scalar(record.get("view_kind")),
                "capture_role": _json_scalar(record.get("capture_role")),
                "default_view": _json_scalar(record.get("default_view")),
                "derived_from": _json_list(record.get("derived_from")),
                "derivation": _json_scalar(record.get("derivation")),
                "axes": axes,
                "shape": json.loads(str(record.get("shape") or "[]")),
                "dtype": str(record.get("dtype") or ""),
                "metadata": metadata,
            }
        )
    return {
        "sites": rows,
        "runtime_collections": _activation_runtime_collections(rows),
        "architecture": _activation_architecture(rows),
    }

def _activation_runtime_collections(sites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kv_members = []
    for site in sites:
        name = str(site.get("name") or "")
        role = str(site.get("role") or "")
        if ".vlm.layers." not in name or ".kv_cache." not in name:
            continue
        component = "key" if role.endswith("key") or name.endswith(".key") else "value"
        kv_members.append(
            {
                "layer": site.get("layer"),
                "component": component,
                "site_name": name,
            }
        )
    if not kv_members:
        return []
    kv_members.sort(key=lambda item: (int(item["layer"] or 0), str(item["component"])))
    return [
        {
            "id": "pi05.vlm.past_key_values",
            "label": "Layer-wise prefix K/V",
            "kind": "runtime_collection",
            "materialized": False,
            "aggregation": "none",
            "members": kv_members,
        }
    ]

def _activation_architecture(sites: list[dict[str, Any]]) -> dict[str, Any]:
    if not any(str(site.get("name") or "").startswith("pi05.") for site in sites):
        return {}

    vlm_layers = sorted(_captured_layers(sites, stack="vlm"))
    expert_layers = sorted(_captured_layers(sites, stack="expert"))
    nodes = [
        {
            "id": "pi05.vlm.prefix",
            "label": "Inputs",
            "kind": "inputs",
            "stage": "prefix",
            "captured": any(
                str(site.get("name") or "").startswith("pi05.vlm.prefix") for site in sites
            ),
        }
    ]
    nodes.extend(
        {
            "id": f"pi05.vlm.layers.{layer}",
            "label": f"VLM L{layer}",
            "kind": "vlm_layer",
            "stage": "prefix",
            "layer": layer,
            "captured": True,
        }
        for layer in vlm_layers
    )
    nodes.append(
        {
            "id": "pi05.expert.by_step.input_embeddings",
            "label": "x_t",
            "kind": "denoise_state",
            "stage": "action_denoiser",
            "captured": any(
                str(site.get("name") or "") == "pi05.expert.by_step.input_embeddings"
                for site in sites
            ),
        }
    )
    nodes.extend(
        {
            "id": f"pi05.expert.layers.{layer}",
            "label": f"Expert L{layer}",
            "kind": "expert_layer",
            "stage": "action_denoiser",
            "layer": layer,
            "captured": True,
        }
        for layer in expert_layers
    )
    nodes.extend(
        [
            {
                "id": "pi05.action_head",
                "label": "Head",
                "kind": "action_head",
                "stage": "output",
                "captured": any(
                    str(site.get("name") or "").startswith("pi05.action_head")
                    and str(site.get("role") or "") != "action_head_output"
                    for site in sites
                ),
            },
            {
                "id": "pi05.action_output",
                "label": "Action",
                "kind": "action_output",
                "stage": "output",
                "captured": any(
                    str(site.get("role") or "") == "action_head_output" for site in sites
                ),
            },
        ]
    )
    edges = _activation_architecture_edges(sites)
    return {"nodes": nodes, "edges": edges} if nodes or edges else {}

def _activation_architecture_edges(sites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    vlm_kv = _vlm_kv_sites_by_layer(sites)
    expert_layers = _captured_layers(sites, stack="expert")
    attention_by_layer = {
        int(site["layer"]): site
        for site in sites
        if _site_layer(site) is not None
        and ".expert.layers." in str(site.get("name") or "")
        and (
            str(site.get("role") or "") == "attention_probs"
            or str(site.get("tensor_type") or "") == "attention"
        )
    }
    edges = []
    for layer in sorted(set(vlm_kv) & expert_layers):
        source_sites = vlm_kv[layer]
        if not {"key", "value"}.issubset(source_sites):
            continue
        attention_site = attention_by_layer.get(layer)
        query_token_space = (
            attention_site.get("query_token_space_id") if attention_site else "pi05.action_suffix"
        )
        key_token_space = (
            attention_site.get("key_token_space_id") if attention_site else "pi05.expert_context"
        )
        edges.append(
            {
                "id": f"pi05.vlm.layers.{layer}.kv_to_expert.layers.{layer}",
                "kind": "per_layer_kv_conditioning",
                "source": f"pi05.vlm.layers.{layer}",
                "target": f"pi05.expert.layers.{layer}",
                "layer": layer,
                "source_sites": [source_sites["key"], source_sites["value"]],
                "target_site_family": (
                    attention_site.get("name")
                    if attention_site
                    else f"pi05.expert.layers.{layer}.by_step.attention"
                ),
                "source_token_space": "pi05.prefix",
                "query_token_space": query_token_space,
                "key_token_space": key_token_space,
                "runtime_collection": "pi05.vlm.past_key_values",
                "materialized": False,
            }
        )
    return edges

def _vlm_kv_sites_by_layer(sites: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for site in sites:
        layer = _site_layer(site)
        name = str(site.get("name") or "")
        if layer is None or ".vlm.layers." not in name or ".kv_cache." not in name:
            continue
        role = str(site.get("role") or "")
        if role.endswith("key") or name.endswith(".key"):
            component = "key"
        elif role.endswith("value") or name.endswith(".value"):
            component = "value"
        else:
            continue
        out.setdefault(layer, {})[component] = name
    return out

def _captured_layers(sites: list[dict[str, Any]], *, stack: str) -> set[int]:
    marker = f".{stack}.layers."
    layers = set()
    for site in sites:
        name = str(site.get("name") or "")
        layer = _site_layer(site)
        if layer is not None and marker in name:
            layers.add(layer)
    return layers

def _site_layer(site: dict[str, Any]) -> int | None:
    layer = site.get("layer")
    try:
        if layer is None:
            return None
        if isinstance(layer, float) and not np.isfinite(layer):
            return None
        return int(layer)
    except (TypeError, ValueError, OverflowError):
        return None

def _activation_slice_payload(bundle: TraceBundle, query: dict[str, list[str]]) -> dict[str, Any]:
    name = _query_one(query, "name")
    calls = _policy_calls(bundle)
    if not calls:
        return {"name": name, "values": [], "top_abs": [], "selected": None}
    call = calls[_query_call_index(query)]
    generation_step = query.get("generation_step", [""])[0]
    matches = bundle.model_sites.loc[bundle.model_sites["name"].astype(str) == name]
    if matches.empty:
        raise KeyError(name)
    record = matches.iloc[0]
    axes = json.loads(str(record.get("axes") or "[]"))
    array = bundle.model_site(name, mmap=True)
    value, remaining_axes = _take_policy_call_value(array, axes, call)
    if "generation_step" in axes:
        step = int(generation_step) if generation_step not in {"", None} else 0
        value = _take_axis_value(value, remaining_axes, "generation_step", step)
        remaining_axes = [axis for axis in remaining_axes if axis != "generation_step"]
    value_array = np.asarray(value, dtype=np.float32)
    if "channel" not in remaining_axes:
        return {
            "name": name,
            "selected": call,
            "axes": remaining_axes,
            "shape": [int(item) for item in array.shape],
            "feature_count": 0,
            "feature": 0,
            "feature_value": None,
            "top_abs": [],
            "reason": "Selected site has no channel feature axis.",
        }
    channel_axis = remaining_axes.index("channel")
    channel_count = int(value_array.shape[channel_axis])
    channel_matrix = np.moveaxis(value_array, channel_axis, -1).reshape(-1, channel_count)
    vector = np.nanmean(channel_matrix, axis=0)
    remaining_axes = [axis for axis in remaining_axes if axis != "channel"]
    clip_percent = _query_float(query, "clip_percent", 0.0)
    clip_percent = min(20.0, max(0.0, clip_percent))
    try:
        top_k = int(query.get("top_k", ["12"])[0])
    except (TypeError, ValueError):
        top_k = 12
    top_k = max(1, min(top_k, 256))
    order, clip_info = _rank_feature_vector(vector, clip_percent=clip_percent, limit=top_k)
    feature = int(query.get("feature", ["0"])[0])
    feature = max(0, min(feature, max(0, vector.shape[0] - 1)))
    return {
        "name": name,
        "selected": call,
        "axes": remaining_axes,
        "shape": [int(item) for item in array.shape],
        "feature_count": int(vector.shape[0]),
        "feature": feature,
        "feature_value": _json_scalar(float(vector[feature])) if vector.size else None,
        "clip_percent": clip_percent,
        "clip": clip_info,
        "top_abs": [
            {"index": int(index), "value": _json_scalar(float(vector[index]))} for index in order
        ],
    }

def _rank_feature_vector(
    vector: np.ndarray,
    *,
    clip_percent: float = 0.0,
    limit: int = 24,
) -> tuple[np.ndarray, dict[str, Any]]:
    finite_mask = np.isfinite(vector)
    finite_values = vector[finite_mask]
    if finite_values.size == 0:
        return np.asarray([], dtype=np.int64), {
            "enabled": clip_percent > 0,
            "kept": 0,
            "total": int(vector.size),
        }

    lower: float | None = None
    upper: float | None = None
    keep_mask = finite_mask.copy()
    if clip_percent > 0:
        lower = float(np.percentile(finite_values, clip_percent))
        upper = float(np.percentile(finite_values, 100.0 - clip_percent))
        keep_mask &= vector >= lower
        keep_mask &= vector <= upper

    candidates = np.flatnonzero(keep_mask)
    if candidates.size == 0:
        return candidates, {
            "enabled": clip_percent > 0,
            "lower": _json_scalar(lower),
            "upper": _json_scalar(upper),
            "kept": 0,
            "total": int(vector.size),
        }
    candidate_values = np.nan_to_num(vector[candidates], nan=0.0, posinf=0.0, neginf=0.0)
    ranked = candidates[np.argsort(np.abs(candidate_values))[::-1][:limit]]
    return ranked, {
        "enabled": clip_percent > 0,
        "lower": _json_scalar(lower),
        "upper": _json_scalar(upper),
        "kept": int(candidates.size),
        "total": int(vector.size),
    }

def _image_token_map_payload(bundle: TraceBundle, query: dict[str, list[str]]) -> dict[str, Any]:
    name = _query_one(query, "name")
    calls = _policy_calls(bundle)
    if not calls:
        return {"available": False, "reason": "No policy calls are available."}
    call = calls[_query_call_index(query)]
    feature = int(query.get("feature", ["0"])[0])
    matches = bundle.model_sites.loc[bundle.model_sites["name"].astype(str) == name]
    if matches.empty:
        raise KeyError(name)
    record = matches.iloc[0]
    axes = json.loads(str(record.get("axes") or "[]"))
    if "token" not in axes:
        return _not_captured_in_profile(
            f"Image heatmaps require model_sites with a token axis. {name!r} has axes={axes!r}.",
            name=name,
            token_kind=_json_scalar(record.get("token_kind")),
            axes=axes,
        )
    feature_values, feature, feature_count = _activation_token_feature_vector(
        bundle,
        name,
        call,
        query,
        feature,
    )
    image_rows = _image_token_rows_for_site(bundle, record, call, feature_values.shape[0])
    if not image_rows.empty:
        maps, layout = _camera_patch_maps_from_token_rows(
            bundle,
            image_rows,
            feature_values,
        )
        prefix_rows = _token_rows_for_space(bundle, call, str(record.get("token_space_id") or ""))
        text_tokens = (
            int((prefix_rows.get("token_kind", "").astype(str) == "language").sum())
            if not prefix_rows.empty
            else 0
        )
        return {
            "available": True,
            "name": name,
            "feature": feature,
            "feature_count": feature_count,
            "call": call,
            "source": "vlatrace",
            "grid_size": layout["grid_size"],
            "grid_height": layout["grid_height"],
            "grid_width": layout["grid_width"],
            "patches_per_image": layout["patches_per_image"],
            "image_tokens": int(len(image_rows)),
            "text_tokens": text_tokens,
            "image_slots": len(maps),
            "maps": maps,
            "note": (
                "Mapped image-token rows from token layout. This site is a mixed prefix sequence, "
                "so language tokens are excluded from the camera heatmap."
            ),
        }
    if str(record.get("token_kind") or "") != "image":
        return _not_captured_in_profile(
            "Image heatmaps require either an image-token site or token layout rows that mark "
            f"the image-token subset. {name!r} is token_kind={record.get('token_kind')!r}.",
            name=name,
            token_kind=_json_scalar(record.get("token_kind")),
            token_space_id=_json_scalar(record.get("token_space_id")),
        )
    text_tokens = 0
    layout = _camera_patch_layout_from_record(
        bundle,
        record,
        feature_values.shape[0],
        text_tokens=text_tokens,
    )
    image_tokens = int(layout["image_tokens"])
    patches_per_image = int(layout["patches_per_image"])
    grid_height = int(layout["grid_height"])
    grid_width = int(layout["grid_width"])
    maps: dict[str, Any] = {}
    cameras = bundle.cameras()
    image_slots = image_tokens // patches_per_image if patches_per_image else 0
    for camera_index, camera in enumerate(cameras):
        if camera_index >= image_slots:
            continue
        start = camera_index * patches_per_image
        end = start + patches_per_image
        values = feature_values[start:end].reshape(grid_height, grid_width)
        maps[camera] = {
            "values": _round(values),
            "token_start": start,
            "token_end": end - 1,
            "active_tokens": None,
            "min": _json_scalar(float(np.nanmin(values))),
            "max": _json_scalar(float(np.nanmax(values))),
        }
    return {
        "available": True,
        "name": name,
        "feature": feature,
        "feature_count": feature_count,
        "call": call,
        "source": "vlatrace",
        "grid_size": grid_height if grid_height == grid_width else None,
        "grid_height": grid_height,
        "grid_width": grid_width,
        "patches_per_image": patches_per_image,
        "image_tokens": image_tokens,
        "text_tokens": text_tokens,
        "image_slots": image_slots,
        "maps": maps,
        "note": (
            f"Inferred PI0.5/PaliGemma prefix layout: {image_slots} image slots x "
            f"{grid_height}x{grid_width} patches, followed by {text_tokens} text token slots."
        ),
    }

def _prompt_feature_map_payload(bundle: TraceBundle, query: dict[str, list[str]]) -> dict[str, Any]:
    name = _query_one(query, "name")
    calls = _policy_calls(bundle)
    if not calls:
        return {"available": False, "reason": "No policy calls are available."}
    call = calls[_query_call_index(query)]
    feature = int(query.get("feature", ["0"])[0])
    matches = bundle.model_sites.loc[bundle.model_sites["name"].astype(str) == name]
    if matches.empty:
        raise KeyError(name)
    record = matches.iloc[0]
    axes = json.loads(str(record.get("axes") or "[]"))
    if "token" not in axes or "channel" not in axes:
        return _not_captured_in_profile(
            "Prompt feature maps require a model site with token and channel axes.",
            name=name,
            axes=axes,
        )
    if str(record.get("role") or "") == "image_prefix_hidden_tokens":
        return _not_captured_in_profile(
            "This site stores image-prefix tokens only, "
            "so it has no aligned prompt-token features.",
            name=name,
            role=_json_scalar(record.get("role")),
        )
    token_space_id = str(record.get("token_space_id") or "")
    prefix_rows = _token_rows_for_space(bundle, call, token_space_id)
    if prefix_rows.empty:
        return _not_captured_in_profile(
            "Prompt feature maps require token layout rows for the selected token space.",
            name=name,
            token_space_id=token_space_id,
        )
    token_matrix = _activation_token_matrix(bundle, name, call, query)
    feature = max(0, min(feature, token_matrix.shape[1] - 1))
    text_rows = prefix_rows.loc[prefix_rows.get("token_kind", "").astype(str) == "language"].copy()
    if text_rows.empty:
        return _not_captured_in_profile(
            "The selected token space has no language-token rows.",
            name=name,
            token_space_id=token_space_id,
        )
    values = np.full((_token_count(prefix_rows),), np.nan, dtype=np.float32)
    limit = min(values.shape[0], token_matrix.shape[0])
    values[:limit] = token_matrix[:limit, feature]
    _top, prompt_mass, prompt, prompt_tokens = _prompt_attention_from_prefix_rows(
        prefix_rows, values
    )
    active_text_tokens = len(prompt_tokens)
    if "attention_mask" in text_rows:
        active_text_tokens = int(text_rows.get("attention_mask", []).astype(bool).sum())
    return {
        "available": True,
        "kind": "feature",
        "name": name,
        "call": call,
        "feature": feature,
        "feature_count": int(token_matrix.shape[1]),
        "prompt": bundle.manifest.prompt or prompt,
        "active_text_tokens": active_text_tokens,
        "allocated_text_slots": int(len(text_rows)),
        "expert_coarse": {"prompt": _json_scalar(float(prompt_mass))},
        "top_text_tokens": sorted(
            prompt_tokens,
            key=lambda item: abs(float(item.get("attention") or 0.0)),
            reverse=True,
        )[:24],
        "prompt_tokens": prompt_tokens,
        "top_image_patches": [],
        "note": (
            "Prompt tokens are colored by the selected hidden feature value, not attention mass."
        ),
    }
