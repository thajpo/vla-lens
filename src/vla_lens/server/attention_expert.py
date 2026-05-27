"""Expert-token and prompt-attention payload helpers."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from vla_lens.server.attention_features import _activation_token_matrix
from vla_lens.server.attention_maps import _attention_key_mass_from_trace
from vla_lens.server.attention_tokens import (
    _image_attention_from_prefix_rows,
    _not_captured_in_profile,
    _prompt_attention_from_prefix_rows,
    _token_count,
    _token_rows_for_space,
)
from vla_lens.server.common import (
    _json_scalar,
    _policy_call_axis_selection,
    _query_call_index,
    _query_one,
    _round,
    _take_axis_values,
)
from vla_lens.server.metrics import _policy_calls
from vla_lens.traces import TraceBundle


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
            f"Prompt attention requires {kind} attention arrays stored in the VLA Lens overlay.",
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
            "This is one expert query/action token from the VLA Lens overlay. "
            "Image and prompt rows are attention mass from the matching expert layer/query token."
            if attention
            else "Attention details are unavailable unless captured into the VLA Lens overlay."
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
        "source": "vla_lens_overlay.action_chunks",
        "dim": int(vector.shape[0]),
        "norm": _json_scalar(float(np.linalg.norm(safe))),
        "top_abs": [
            {"index": int(index), "value": _json_scalar(float(vector[index]))} for index in order
        ],
    }
