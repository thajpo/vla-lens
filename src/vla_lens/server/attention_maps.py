"""Attention-map response helpers."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from vla_lens.server.attention_tokens import (
    _camera_patch_layout,
    _camera_patch_layout_from_record,
    _not_captured_in_profile,
)
from vla_lens.server.common import (
    _json_scalar,
    _policy_call_axis_selection,
    _query_call_index,
    _round,
    _take_axis_values,
)
from vla_lens.server.metrics import _policy_calls
from vla_lens.traces import TraceBundle


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
            f"{kind} attention maps require attention arrays stored in the VLA Lens overlay.",
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
        "source": "vla_lens_overlay",
        **layout,
        "coarse": {
            "image": _json_scalar(float(np.nansum(key_mass[: int(layout["image_tokens"])]))),
            "prompt": None,
            "action_suffix": None,
        },
        "maps": _camera_maps_from_trace_key_mass(bundle, key_mass, layout),
    }

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
