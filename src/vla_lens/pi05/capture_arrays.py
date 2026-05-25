"""PI0.5 capture arrays helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from vla_lens.pi05.capture_hooks import (
    _empty_step_array,
)
from vla_lens.pi05.capture_schema import (
    CaptureCall,
    CapturePlan,
    EpisodeBuffer,
    canonical_profile,
)
from vla_lens.pi05.capture_utils import (
    _pad_time,
    _patch_grid_shape,
    _stack_call_arrays,
    _stack_layer_calls,
    _stack_optional_calls,
    _trace_cameras,
)
from vla_lens.pi05.full_capture import (
    pi05_full_site_declarations,
)
from vla_lens.pi05.token_metadata import (
    EXPERT_CONTEXT_TOKEN_SPACE_ID,
)
from vla_lens.traces import ArraySpec, ModelSiteSpec


def _episode_arrays(buffer: EpisodeBuffer, length: int) -> dict[str, ArraySpec]:
    arrays: dict[str, ArraySpec] = {
        "executed_actions": ArraySpec(
            _pad_time(buffer.executed_actions, length),
            ["timestep", "action_dim"],
        )
    }
    if buffer.frames:
        arrays["frames.main"] = ArraySpec(
            np.stack(buffer.frames),
            ["timestep", "height", "width", "rgb"],
        )
    if buffer.wrist_frames:
        arrays["frames.wrist"] = ArraySpec(
            np.stack(buffer.wrist_frames),
            ["timestep", "height", "width", "rgb"],
        )
    if buffer.calls:
        arrays["action_chunks"] = ArraySpec(
            _stack_call_arrays(buffer.calls, "final_action_chunk"),
            ["policy_call", "horizon", "action_dim"],
        )
        arrays["generation_actions"] = ArraySpec(
            _stack_call_arrays(buffer.calls, "denoising_actions"),
            ["policy_call", "generation_step", "horizon", "action_dim"],
        )
        velocities = _stack_optional_calls(buffer.calls, "denoise_velocities")
        if velocities is not None:
            arrays["generation_velocities"] = ArraySpec(
                velocities.astype(np.float32),
                ["policy_call", "generation_step", "horizon", "action_dim"],
            )
    return arrays

def _model_arrays(
    buffer: EpisodeBuffer,
    plan: CapturePlan,
) -> list[ModelSiteSpec]:
    if canonical_profile(plan.profile) == "rollout" or not buffer.calls:
        return []
    specs: list[ModelSiteSpec] = []
    image_hidden = _stack_optional_calls(buffer.calls, "prefix_image_hidden")
    if image_hidden is not None:
        patches_per_image = next(
            (
                int(call.prefix_patches_per_image)
                for call in buffer.calls
                if call.prefix_patches_per_image
            ),
            image_hidden.shape[1] // max(1, len(_trace_cameras(buffer))),
        )
        image_slots = next(
            (int(call.prefix_image_slots) for call in buffer.calls if call.prefix_image_slots),
            image_hidden.shape[1] // max(1, patches_per_image),
        )
        grid_height, grid_width = _patch_grid_shape(patches_per_image)
        specs.append(
            ModelSiteSpec(
                name="pi05.vlm.prefix.image_hidden_tokens",
                array=image_hidden,
                axes=["policy_call", "token", "channel"],
                module="pi05.vlm.prefix",
                tensor_type="hidden_tokens",
                token_kind="image",
                family="representation",
                role="image_prefix_hidden_tokens",
                segment="vlm_prefix",
                token_space_id="pi05.prefix",
                metadata={
                    "camera_order": _trace_cameras(buffer),
                    "patches_per_image": patches_per_image,
                    "grid_height": grid_height,
                    "grid_width": grid_width,
                    "image_slots": image_slots,
                },
                capture_family="representation",
                view_kind="features",
                capture_role="primary",
                default_view=True,
            )
        )

    for layer in sorted(set(plan.vlm_layers) | set(plan.expert_layers)):
        vlm_hidden = _stack_layer_calls(buffer.calls, "vlm_hidden_by_layer", layer)
        if vlm_hidden is not None:
            vlm_axes = ["policy_call", "channel"]
            if plan.vlm_hidden == "tokens":
                vlm_axes = ["policy_call", "token", "channel"]
            specs.append(
                ModelSiteSpec(
                    name=f"pi05.vlm.layers.{layer}.prefix.hidden_{plan.vlm_hidden}",
                    array=vlm_hidden,
                    axes=vlm_axes,
                    module=f"pi05.vlm.layers.{layer}",
                    layer=layer,
                    tensor_type=f"hidden_{plan.vlm_hidden}",
                    token_kind="prefix",
                    family="representation",
                    role="hidden_state",
                    segment="vlm_prefix",
                    token_space_id="pi05.prefix",
                    capture_family="representation",
                    view_kind="features",
                    capture_role="primary",
                    default_view=True,
                )
            )

        vlm_attention = _stack_layer_calls(buffer.calls, "vlm_attention_by_layer", layer)
        if vlm_attention is not None:
            specs.append(
                _attention_spec(
                    family="vlm",
                    layer=layer,
                    array=vlm_attention,
                    resolution=plan.vlm_attention,
                    by_step=False,
                    token_kind="prefix",
                    segment="vlm_prefix",
                )
            )
        vlm_kv_key = _stack_layer_calls(buffer.calls, "vlm_kv_key_by_layer", layer)
        if vlm_kv_key is not None:
            specs.append(
                ModelSiteSpec(
                    name=f"pi05.vlm.layers.{layer}.kv_cache.key",
                    array=vlm_kv_key,
                    axes=["policy_call", "kv_head", "cached_token", "head_channel"],
                    module=f"pi05.vlm.layers.{layer}.attention",
                    layer=layer,
                    tensor_type="kv_cache",
                    token_kind="prefix",
                    family="cache",
                    role="kv_cache_key",
                    segment="vlm_prefix",
                    materialization="raw",
                    exactness="exact",
                    token_space_id="pi05.prefix",
                    metadata={
                        "capture_scope": "mechanistic_bridge",
                        "included_in_profile": plan.profile,
                    },
                    capture_family="cache",
                    view_kind="cache",
                    capture_role="primary",
                    default_view=False,
                )
            )
        vlm_kv_value = _stack_layer_calls(buffer.calls, "vlm_kv_value_by_layer", layer)
        if vlm_kv_value is not None:
            specs.append(
                ModelSiteSpec(
                    name=f"pi05.vlm.layers.{layer}.kv_cache.value",
                    array=vlm_kv_value,
                    axes=["policy_call", "kv_head", "cached_token", "head_channel"],
                    module=f"pi05.vlm.layers.{layer}.attention",
                    layer=layer,
                    tensor_type="kv_cache",
                    token_kind="prefix",
                    family="cache",
                    role="kv_cache_value",
                    segment="vlm_prefix",
                    materialization="raw",
                    exactness="exact",
                    token_space_id="pi05.prefix",
                    metadata={
                        "capture_scope": "mechanistic_bridge",
                        "included_in_profile": plan.profile,
                    },
                    capture_family="cache",
                    view_kind="cache",
                    capture_role="primary",
                    default_view=False,
                )
            )

        expert_hidden = _stack_layer_calls(buffer.calls, "expert_hidden_by_layer", layer)
        if expert_hidden is not None:
            expert_axes = ["policy_call", "generation_step", "channel"]
            if plan.expert_hidden == "tokens":
                expert_axes = ["policy_call", "generation_step", "token", "channel"]
            specs.append(
                ModelSiteSpec(
                    name=f"pi05.expert.layers.{layer}.by_step.hidden_{plan.expert_hidden}",
                    array=expert_hidden,
                    axes=expert_axes,
                    module=f"pi05.expert.layers.{layer}",
                    layer=layer,
                    tensor_type=f"hidden_{plan.expert_hidden}",
                    token_kind="action",
                    family="representation",
                    role="hidden_state",
                    segment="action_expert",
                    token_space_id="pi05.action_suffix",
                    capture_family="representation",
                    view_kind="features",
                    capture_role="primary",
                    default_view=True,
                )
            )

        expert_attention = _stack_layer_calls(buffer.calls, "expert_attention_by_layer", layer)
        if expert_attention is not None:
            specs.append(
                _attention_spec(
                    family="expert",
                    layer=layer,
                    array=expert_attention,
                    resolution=plan.expert_attention,
                    by_step=True,
                    token_kind="action",
                    segment="action_expert",
                )
            )

    attention = _stack_optional_calls(buffer.calls, "attention_mass")
    if attention is not None and plan.expert_attention != "full":
        specs.append(
            ModelSiteSpec(
                name="pi05.expert.by_step.attention_key_mass",
                array=attention,
                axes=["policy_call", "generation_step", "key_token"],
                module="pi05.expert",
                tensor_type="attention",
                token_kind="action",
                family="derived",
                role="attention_key_mass_summary",
                segment="action_expert",
                materialization="summary",
                exactness="lossy_summary",
                metadata={
                    "attention_resolution": "key_mass",
                    "source": "final captured expert layer averaged over heads and queries",
                },
                capture_family="attention",
                view_kind="attention",
                capture_role="derived_summary",
                default_view=False,
                derived_from=tuple(
                    f"pi05.expert.layers.{layer}.by_step.attention" for layer in plan.expert_layers
                ),
                derivation="mean_over_heads_and_queries",
            )
        )
    generation_input_embeddings = _stack_optional_calls(buffer.calls, "generation_input_embeddings")
    if generation_input_embeddings is not None:
        specs.append(
            ModelSiteSpec(
                name="pi05.expert.by_step.input_embeddings",
                array=generation_input_embeddings,
                axes=["policy_call", "generation_step", "token", "channel"],
                module="pi05.expert",
                tensor_type="embedding",
                token_kind="action",
                family="embedding",
                role="input_embeddings",
                segment="action_expert",
                materialization="raw",
                exactness="exact",
                token_space_id="pi05.action_suffix",
                metadata={
                    "capture_scope": "mechanistic_bridge",
                    "included_in_profile": plan.profile,
                },
                capture_family="representation",
                view_kind="features",
                capture_role="primary",
                default_view=True,
            )
        )
    action_head_input = _stack_optional_calls(buffer.calls, "action_head_input")
    if action_head_input is not None:
        specs.append(
            ModelSiteSpec(
                name="pi05.action_head.input",
                array=action_head_input,
                axes=["policy_call", "generation_step", "token", "channel"],
                module="pi05.action_head",
                tensor_type="action_head",
                token_kind="action",
                family="action_head",
                role="action_head_input",
                segment="action_head",
                materialization="raw",
                exactness="exact",
                token_space_id="pi05.action_suffix",
                metadata={
                    "capture_scope": "mechanistic_bridge",
                    "included_in_profile": plan.profile,
                },
                capture_family="action_head",
                view_kind="action",
                capture_role="primary",
                default_view=True,
            )
        )
    action_head_output = _stack_optional_calls(buffer.calls, "action_head_output")
    if action_head_output is not None:
        specs.append(
            ModelSiteSpec(
                name="pi05.action_head.output",
                array=action_head_output,
                axes=["policy_call", "generation_step", "horizon", "action_dim"],
                module="pi05.action_head",
                tensor_type="action_head",
                token_kind="action",
                family="action_head",
                role="action_head_output",
                segment="action_head",
                materialization="raw",
                exactness="exact",
                token_space_id="pi05.action_suffix",
                metadata={
                    "capture_scope": "mechanistic_bridge",
                    "included_in_profile": plan.profile,
                },
                capture_family="action_head",
                view_kind="action",
                capture_role="primary",
                default_view=True,
            )
        )
    if plan.capture_internals_sites:
        existing_names = {spec.name for spec in specs}
        specs.extend(
            spec for spec in _full_model_site_specs(buffer, plan) if spec.name not in existing_names
        )
    return specs

def _full_model_site_specs(
    buffer: EpisodeBuffer,
    plan: CapturePlan,
) -> list[ModelSiteSpec]:
    profile = canonical_profile(plan.profile)
    declarations = {
        item.name: item
        for item in pi05_full_site_declarations(
            vlm_layers=plan.vlm_layers,
            expert_layers=plan.expert_layers,
        )
    }
    specs: list[ModelSiteSpec] = []
    for name, declaration in declarations.items():
        if profile == "internals_sampled" and not _is_selected_internal_site(declaration):
            continue
        if profile in {"audit_sampled", "audit_windowed"} and not _is_audit_sampled_site(
            declaration
        ):
            continue
        stacked = _stack_full_site_calls(buffer.calls, name)
        if stacked is None:
            continue
        specs.append(
            declaration.spec(
                stacked,
                metadata={
                    "capture_profile": profile,
                    "included_in_profile": profile,
                    "required_for_audit_sampled": profile == "audit_sampled",
                    "required_for_audit_windowed": profile == "audit_windowed",
                    "numeric_lossy": str(stacked.dtype) != "float32"
                    and np.issubdtype(stacked.dtype, np.floating),
                    "semantic_lossy": False,
                    "summary_lossy": False,
                },
            )
        )
    return specs

def _is_selected_internal_site(declaration: Any) -> bool:
    selected_roles = {
        "q",
        "k",
        "v",
        "attention_probs",
        "o_proj",
        "mlp_gate",
        "mlp_up",
        "mlp_intermediate",
        "mlp_down",
        "adarms_gate",
        "kv_cache_key",
        "kv_cache_value",
    }
    return str(declaration.role) in selected_roles

def _is_audit_sampled_site(declaration: Any) -> bool:
    role = str(declaration.role)
    name = str(declaration.name)
    segment = str(declaration.segment)
    if role == "input_embeddings":
        return name == "pi05.expert.by_step.input_embeddings"
    if role in {"action_head_input", "action_head_output"}:
        return True
    if role in {"kv_cache_key", "kv_cache_value"}:
        return segment == "vlm_prefix"
    return role in _AUDIT_SAMPLED_RAW_ROLES

_AUDIT_SAMPLED_RAW_ROLES = {
    "residual_pre_attention",
    "attention_norm_output",
    "q",
    "k",
    "v",
    "pre_mask_scores",
    "post_mask_logits",
    "attention_probs",
    "attn_output_pre_o_proj",
    "o_proj",
    "residual_post_attention",
    "residual_pre_mlp",
    "mlp_norm_output",
    "mlp_gate",
    "mlp_up",
    "mlp_intermediate",
    "mlp_down",
    "mlp_output",
    "residual_post_mlp",
    "adarms_gate",
    "adarms_scale",
    "adarms_shift",
}

def _stack_full_site_calls(calls: list[CaptureCall], site_name: str) -> np.ndarray | None:
    sample = next(
        (
            call.full_site_arrays.get(site_name)
            for call in calls
            if call.full_site_arrays.get(site_name) is not None
        ),
        None,
    )
    if sample is None:
        return None
    sample_array = np.asarray(sample)
    out = _empty_step_array(len(calls), sample_array)
    for index, call in enumerate(calls):
        value = call.full_site_arrays.get(site_name)
        if value is not None:
            out[index] = np.asarray(value, dtype=sample_array.dtype)
    return out

def _attention_spec(
    *,
    family: str,
    layer: int,
    array: np.ndarray,
    resolution: str,
    by_step: bool,
    token_kind: str,
    segment: str,
) -> ModelSiteSpec:
    suffix = "attention" if resolution == "full" else "attention_key_mass"
    axes = ["policy_call"]
    if by_step:
        axes.append("generation_step")
    axes.append("head")
    if resolution == "full":
        axes.extend(["query_token", "key_token"])
    else:
        axes.append("key_token")
    return ModelSiteSpec(
        name=f"pi05.{family}.layers.{layer}.{'by_step.' if by_step else 'prefix.'}{suffix}",
        array=array,
        axes=axes,
        module=f"pi05.{family}.layers.{layer}",
        layer=layer,
        tensor_type="attention",
        token_kind=token_kind,
        family="attention",
        role="attention_probs" if resolution == "full" else "attention_key_mass_summary",
        segment=segment,
        materialization="raw" if resolution == "full" else "summary",
        exactness="exact" if resolution == "full" else "lossy_summary",
        metadata={"attention_resolution": resolution},
        token_space_id="pi05.prefix" if family == "vlm" else EXPERT_CONTEXT_TOKEN_SPACE_ID,
        query_token_space_id="pi05.prefix" if family == "vlm" else "pi05.action_suffix",
        key_token_space_id="pi05.prefix" if family == "vlm" else EXPERT_CONTEXT_TOKEN_SPACE_ID,
        capture_family="attention",
        view_kind="attention",
        capture_role="primary" if resolution == "full" else "derived_summary",
        default_view=True,
    )
