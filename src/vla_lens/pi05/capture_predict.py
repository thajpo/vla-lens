"""PI0.5 capture predict helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from vla_lens.pi05.capture_hooks import (
    _capture_expert_step_inputs,
    _capture_full_attention_sites,
    _install_full_layer_hooks,
    _PI05FullSiteRecorder,
    _rope_metadata_array,
)
from vla_lens.pi05.capture_schema import (
    CaptureCall,
    CapturePlan,
)
from vla_lens.pi05.capture_utils import (
    _to_numpy,
)
from vla_lens.pi05.full_capture import (
    IncompletePI05FullCaptureError,
)


def _predict_action_chunk(
    policy: Any,
    obs: dict[str, Any],
    call_index: int,
    step: int,
    plan: CapturePlan,
) -> CaptureCall:
    import torch
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    from transformers.models.gemma import modeling_gemma

    model = policy.model
    full_recorder = _PI05FullSiteRecorder(plan) if plan.capture_internals_sites else None
    capture: dict[str, list[np.ndarray]] = {
        "x_t": [],
        "denoise_velocities": [],
        "prefix_image_hidden": [],
        "generation_input_embeddings": [],
        "action_head_input": [],
        "action_head_output": [],
    }
    initial_noise: np.ndarray | None = None
    current_denoise_step: dict[str, int | None] = {"index": None}
    vlm_hidden_by_layer: dict[int, np.ndarray] = {}
    vlm_attention_by_layer: dict[int, np.ndarray] = {}
    expert_hidden_by_layer: dict[int, list[np.ndarray]] = defaultdict(list)
    expert_attention_by_layer: dict[int, list[np.ndarray]] = defaultdict(list)
    vlm_kv_key_by_layer: dict[int, np.ndarray] = {}
    vlm_kv_value_by_layer: dict[int, np.ndarray] = {}

    original_denoise = model.denoise_step
    original_embed_prefix = model.embed_prefix
    original_embed_suffix = model.embed_suffix
    original_action_out_forward = model.action_out_proj.forward
    original_embed_image = model.paligemma_with_expert.embed_image
    original_attention = modeling_gemma.eager_attention_forward
    vlm_model = model.paligemma_with_expert.paligemma.model.language_model
    expert_model = model.paligemma_with_expert.gemma_expert.model
    original_vlm_forward = vlm_model.forward
    original_expert_forward = expert_model.forward
    original_vlm_rotary_forward = vlm_model.rotary_emb.forward
    original_expert_rotary_forward = expert_model.rotary_emb.forward
    full_hook_handles: list[Any] = []
    patched_mlps: list[tuple[Any, Any]] = []
    if full_recorder is not None:
        full_hook_handles, patched_mlps = _install_full_layer_hooks(
            full_recorder,
            plan,
            current_denoise_step,
            vlm_model=vlm_model,
            expert_model=expert_model,
        )
    vlm_attention_modules = {
        id(layer.self_attn): layer_idx for layer_idx, layer in enumerate(vlm_model.layers)
    }
    expert_attention_modules = {
        id(layer.self_attn): layer_idx for layer_idx, layer in enumerate(expert_model.layers)
    }

    def embed_image_wrapper(image: Any) -> Any:
        out = original_embed_image(image)
        capture["prefix_image_hidden"].append(_to_numpy(out, dtype=plan.np_dtype).squeeze(0))
        return out

    def embed_prefix_wrapper(*args: Any, **kwargs: Any) -> Any:
        out = original_embed_prefix(*args, **kwargs)
        if full_recorder is not None:
            prefix_embs, prefix_pad_masks, prefix_att_masks = out
            full_recorder.capture(
                "pi05.vlm.prefix.input_embeddings",
                prefix_embs,
                dtype=plan.np_dtype,
            )
            full_recorder.capture(
                "pi05.inputs.attention_mask",
                prefix_pad_masks,
                dtype=np.bool_,
            )
            full_recorder.capture(
                "pi05.inputs.causal_mask",
                make_att_2d_masks(prefix_pad_masks, prefix_att_masks),
                dtype=np.bool_,
            )
            full_recorder.capture(
                "pi05.inputs.position_ids",
                torch.cumsum(prefix_pad_masks, dim=1) - 1,
                dtype=np.int64,
            )
            full_recorder.capture(
                "pi05.inputs.rope.metadata",
                _rope_metadata_array(vlm_model),
                dtype=np.float32,
                squeeze_batch=False,
            )
        return out

    def embed_suffix_wrapper(*args: Any, **kwargs: Any) -> Any:
        out = original_embed_suffix(*args, **kwargs)
        suffix_embs = out[0]
        if plan.capture_bridge_sites and current_denoise_step["index"] is not None:
            capture["generation_input_embeddings"].append(
                _to_numpy(suffix_embs, dtype=plan.np_dtype).squeeze(0)
            )
        if full_recorder is not None and current_denoise_step["index"] is not None:
            full_recorder.capture(
                "pi05.expert.by_step.input_embeddings",
                suffix_embs,
                dtype=plan.np_dtype,
                generation_step=current_denoise_step["index"],
            )
        return out

    def vlm_rotary_wrapper(*args: Any, **kwargs: Any) -> Any:
        out = original_vlm_rotary_forward(*args, **kwargs)
        if full_recorder is not None:
            cos, sin = out
            full_recorder.capture("pi05.inputs.rope.cos", cos, dtype=plan.np_dtype)
            full_recorder.capture("pi05.inputs.rope.sin", sin, dtype=plan.np_dtype)
        return out

    def expert_rotary_wrapper(*args: Any, **kwargs: Any) -> Any:
        out = original_expert_rotary_forward(*args, **kwargs)
        if full_recorder is not None and current_denoise_step["index"] is not None:
            cos, sin = out
            full_recorder.capture(
                "pi05.expert.by_step.rope.cos",
                cos,
                dtype=plan.np_dtype,
                generation_step=current_denoise_step["index"],
            )
            full_recorder.capture(
                "pi05.expert.by_step.rope.sin",
                sin,
                dtype=plan.np_dtype,
                generation_step=current_denoise_step["index"],
            )
        return out

    def action_out_forward_wrapper(*args: Any, **kwargs: Any) -> Any:
        out = original_action_out_forward(*args, **kwargs)
        if plan.capture_bridge_sites and current_denoise_step["index"] is not None and args:
            capture["action_head_input"].append(_to_numpy(args[0], dtype=plan.np_dtype).squeeze(0))
            capture["action_head_output"].append(_to_numpy(out, dtype=plan.np_dtype).squeeze(0))
        if full_recorder is not None and current_denoise_step["index"] is not None and args:
            step_index = current_denoise_step["index"]
            full_recorder.capture(
                "pi05.action_head.input",
                args[0],
                dtype=plan.np_dtype,
                generation_step=step_index,
            )
            full_recorder.capture(
                "pi05.action_head.output",
                out,
                dtype=plan.np_dtype,
                generation_step=step_index,
            )
        return out

    def vlm_forward_wrapper(*args: Any, **kwargs: Any) -> Any:
        if plan.vlm_hidden != "none":
            kwargs["output_hidden_states"] = True
        out = original_vlm_forward(*args, **kwargs)
        if plan.vlm_hidden != "none":
            _capture_hidden_layers(
                getattr(out, "hidden_states", None),
                layers=plan.vlm_layers,
                resolution=plan.vlm_hidden,
                dtype=plan.np_dtype,
                target=vlm_hidden_by_layer,
                append=False,
            )
        return out

    def expert_forward_wrapper(*args: Any, **kwargs: Any) -> Any:
        if plan.expert_hidden != "none":
            kwargs["output_hidden_states"] = True
        out = original_expert_forward(*args, **kwargs)
        if plan.expert_hidden != "none" and current_denoise_step["index"] is not None:
            _capture_hidden_layers(
                getattr(out, "hidden_states", None),
                layers=plan.expert_layers,
                resolution=plan.expert_hidden,
                dtype=plan.np_dtype,
                target=expert_hidden_by_layer,
                append=True,
            )
        return out

    def attention_wrapper(*args: Any, **kwargs: Any) -> Any:
        if len(args) >= 1:
            module = args[0]
        else:
            module = kwargs.get("module")
        can_capture_attention = (
            full_recorder is not None and len(args) >= 5 and (len(args) >= 6 or "scaling" in kwargs)
        )
        if can_capture_attention:
            query, key, value, attention_mask = args[1:5]
            scaling = args[5] if len(args) >= 6 else kwargs["scaling"]
            dropout = kwargs.get("dropout", args[6] if len(args) > 6 else 0.0)
            key_states = modeling_gemma.repeat_kv(key, module.num_key_value_groups)
            value_states = modeling_gemma.repeat_kv(value, module.num_key_value_groups)
            pre_mask_scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            post_mask_logits = (
                pre_mask_scores if attention_mask is None else pre_mask_scores + attention_mask
            )
            attention_probs = torch.nn.functional.softmax(
                post_mask_logits,
                dim=-1,
                dtype=torch.float32,
            ).to(query.dtype)
            attn_weights = torch.nn.functional.dropout(
                attention_probs,
                p=float(dropout),
                training=module.training,
            )
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            out = (attn_output, attn_weights)
        else:
            out = original_attention(*args, **kwargs)
        if not isinstance(out, tuple) or len(out) < 2:
            return out
        attn_weights = out[1]
        module_id = id(module)
        if module_id in vlm_attention_modules and plan.vlm_attention != "none":
            layer = vlm_attention_modules[module_id]
            if layer in plan.vlm_layers:
                if plan.capture_bridge_sites and len(args) >= 4:
                    query, key, value = args[1:4]
                    vlm_kv_key_by_layer[layer] = _to_numpy(key, dtype=plan.np_dtype).squeeze(0)
                    vlm_kv_value_by_layer[layer] = _to_numpy(value, dtype=plan.np_dtype).squeeze(0)
                if can_capture_attention:
                    query, key, value = args[1:4]
                    _capture_full_attention_sites(
                        full_recorder,
                        plan,
                        stack="vlm",
                        layer=layer,
                        generation_step=None,
                        query=query,
                        key=key,
                        value=value,
                        pre_mask_scores=pre_mask_scores,
                        post_mask_logits=post_mask_logits,
                        attention_probs=attention_probs,
                        attn_output=out[0],
                    )
                vlm_attention_by_layer[layer] = _attention_to_numpy(
                    attn_weights,
                    resolution=plan.vlm_attention,
                    dtype=plan.np_dtype,
                )
        elif (
            module_id in expert_attention_modules
            and plan.expert_attention != "none"
            and current_denoise_step["index"] is not None
        ):
            layer = expert_attention_modules[module_id]
            if layer in plan.expert_layers:
                if can_capture_attention:
                    query, key, value = args[1:4]
                    _capture_full_attention_sites(
                        full_recorder,
                        plan,
                        stack="expert",
                        layer=layer,
                        generation_step=current_denoise_step["index"],
                        query=query,
                        key=key,
                        value=value,
                        pre_mask_scores=pre_mask_scores,
                        post_mask_logits=post_mask_logits,
                        attention_probs=attention_probs,
                        attn_output=out[0],
                    )
                expert_attention_by_layer[layer].append(
                    _attention_to_numpy(
                        attn_weights,
                        resolution=plan.expert_attention,
                        dtype=plan.np_dtype,
                    )
                )
        return out

    def denoise_wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal initial_noise
        x_t = kwargs.get("x_t") if "x_t" in kwargs else args[2]
        prefix_pad_masks = (
            kwargs.get("prefix_pad_masks") if "prefix_pad_masks" in kwargs else args[0]
        )
        denoise_index = len(capture["x_t"])
        if denoise_index == 0:
            initial_noise = _to_numpy(x_t, dtype=np.float32).squeeze(0)
        current_denoise_step["index"] = denoise_index
        capture["x_t"].append(_to_numpy(x_t, dtype=plan.np_dtype).squeeze(0))
        if full_recorder is not None:
            _capture_expert_step_inputs(
                full_recorder,
                make_att_2d_masks,
                prefix_pad_masks=prefix_pad_masks,
                x_t=x_t,
                generation_step=denoise_index,
            )
        out = original_denoise(*args, **kwargs)
        capture["denoise_velocities"].append(_to_numpy(out, dtype=plan.np_dtype).squeeze(0))
        current_denoise_step["index"] = None
        return out

    model.paligemma_with_expert.embed_image = embed_image_wrapper
    model.embed_prefix = embed_prefix_wrapper
    model.embed_suffix = embed_suffix_wrapper
    model.denoise_step = denoise_wrapper
    model.action_out_proj.forward = action_out_forward_wrapper
    vlm_model.forward = vlm_forward_wrapper
    expert_model.forward = expert_forward_wrapper
    vlm_model.rotary_emb.forward = vlm_rotary_wrapper
    expert_model.rotary_emb.forward = expert_rotary_wrapper
    modeling_gemma.eager_attention_forward = attention_wrapper
    try:
        with torch.no_grad():
            chunk = policy.predict_action_chunk(obs)
    finally:
        model.denoise_step = original_denoise
        model.embed_prefix = original_embed_prefix
        model.embed_suffix = original_embed_suffix
        model.action_out_proj.forward = original_action_out_forward
        model.paligemma_with_expert.embed_image = original_embed_image
        vlm_model.forward = original_vlm_forward
        expert_model.forward = original_expert_forward
        vlm_model.rotary_emb.forward = original_vlm_rotary_forward
        expert_model.rotary_emb.forward = original_expert_rotary_forward
        modeling_gemma.eager_attention_forward = original_attention
        for handle in full_hook_handles:
            handle.remove()
        for mlp, original_forward in patched_mlps:
            mlp.forward = original_forward

    final_chunk = _to_numpy(chunk, dtype=plan.np_dtype).squeeze(0)
    denoising = np.stack(capture["x_t"], axis=0).astype(plan.np_dtype)
    velocities = np.stack(capture["denoise_velocities"], axis=0).astype(plan.np_dtype)
    generation_input_embeddings = (
        np.stack(capture["generation_input_embeddings"], axis=0).astype(plan.np_dtype)
        if capture["generation_input_embeddings"]
        else None
    )
    action_head_input = (
        np.stack(capture["action_head_input"], axis=0).astype(plan.np_dtype)
        if capture["action_head_input"]
        else None
    )
    action_head_output = (
        np.stack(capture["action_head_output"], axis=0).astype(plan.np_dtype)
        if capture["action_head_output"]
        else None
    )
    prefix_image_hidden = (
        np.concatenate(capture["prefix_image_hidden"], axis=0).astype(plan.np_dtype)
        if capture["prefix_image_hidden"]
        else None
    )
    prefix_patches_per_image = (
        int(capture["prefix_image_hidden"][0].shape[0]) if capture["prefix_image_hidden"] else None
    )
    prefix_image_slots = (
        len(capture["prefix_image_hidden"]) if capture["prefix_image_hidden"] else None
    )
    attention = _expert_attention_key_mass(expert_attention_by_layer, dtype=plan.np_dtype)
    full_site_arrays: dict[str, np.ndarray] = {}
    if full_recorder is not None:
        full_site_arrays = full_recorder.finalized_arrays(generation_steps=denoising.shape[0])
        missing = full_recorder.missing_names(full_site_arrays)
        if missing and plan.capture_audit_full_sites:
            preview = ", ".join(missing[:12])
            suffix = "..." if len(missing) > 12 else ""
            raise IncompletePI05FullCaptureError(
                f"PI0.5 full capture missed {len(missing)} required raw sites: {preview}{suffix}"
            )
    return CaptureCall(
        call_index=call_index,
        env_timestep=step,
        final_action_chunk=final_chunk.astype(plan.np_dtype),
        denoising_actions=denoising,
        suffix_hidden=velocities,
        initial_noise=initial_noise,
        prefix_image_hidden=prefix_image_hidden,
        prefix_patches_per_image=prefix_patches_per_image,
        prefix_image_slots=prefix_image_slots,
        attention_mass=attention,
        denoise_velocities=velocities,
        vlm_hidden_by_layer=vlm_hidden_by_layer,
        vlm_attention_by_layer=vlm_attention_by_layer,
        expert_hidden_by_layer={
            layer: np.stack(values, axis=0).astype(plan.np_dtype)
            for layer, values in expert_hidden_by_layer.items()
            if values
        },
        expert_attention_by_layer={
            layer: np.stack(values, axis=0).astype(plan.np_dtype)
            for layer, values in expert_attention_by_layer.items()
            if values
        },
        vlm_kv_key_by_layer=vlm_kv_key_by_layer,
        vlm_kv_value_by_layer=vlm_kv_value_by_layer,
        generation_input_embeddings=generation_input_embeddings,
        action_head_input=action_head_input,
        action_head_output=action_head_output,
        full_site_arrays=full_site_arrays,
    )

def _capture_hidden_layers(
    hidden_states: Any,
    *,
    layers: tuple[int, ...],
    resolution: str,
    dtype: np.dtype,
    target: dict[int, Any],
    append: bool,
) -> None:
    if resolution == "none" or hidden_states is None:
        return
    hidden_tuple = tuple(hidden_states)
    if not hidden_tuple:
        return
    for layer in layers:
        hidden_index = min(layer + 1, len(hidden_tuple) - 1)
        array = _hidden_to_numpy(hidden_tuple[hidden_index], resolution=resolution, dtype=dtype)
        if append:
            target[layer].append(array)
        else:
            target[layer] = array

def _hidden_to_numpy(value: Any, *, resolution: str, dtype: np.dtype) -> np.ndarray:
    array = _to_numpy(value, dtype=dtype).squeeze(0)
    if resolution == "mean":
        return np.nanmean(array, axis=0).astype(dtype)
    return array.astype(dtype)

def _attention_to_numpy(value: Any, *, resolution: str, dtype: np.dtype) -> np.ndarray:
    array = _to_numpy(value, dtype=np.float32).squeeze(0)
    if resolution == "key_mass":
        array = np.nanmean(array, axis=-2)
    return array.astype(dtype)

def _expert_attention_key_mass(
    attention_by_layer: dict[int, list[np.ndarray]],
    *,
    dtype: np.dtype,
) -> np.ndarray | None:
    if not attention_by_layer:
        return None
    final_layer = max(attention_by_layer)
    values = attention_by_layer.get(final_layer) or []
    if not values:
        return None
    array = np.stack(values, axis=0).astype(np.float32)
    if array.ndim == 4:
        # denoise_step x head x query_token x key_token
        return np.nanmean(array, axis=(1, 2)).astype(dtype)
    if array.ndim == 3:
        # denoise_step x head x key_token
        return np.nanmean(array, axis=1).astype(dtype)
    return None
