"""PI0.5 capture hooks helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np

from vla_lens.pi05.capture_schema import (
    CapturePlan,
)
from vla_lens.pi05.full_capture import (
    pi05_full_site_declarations,
)


class _PI05FullSiteRecorder:
    """Per-policy-call recorder for exact raw PI0.5 full-capture sites."""

    def __init__(self, plan: CapturePlan):
        self.declarations = pi05_full_site_declarations(
            vlm_layers=plan.vlm_layers,
            expert_layers=plan.expert_layers,
        )
        self._declarations_by_name = {item.name: item for item in self.declarations}
        self._arrays: dict[str, np.ndarray] = {}
        self._step_arrays: dict[str, dict[int, np.ndarray]] = defaultdict(dict)

    def capture(
        self,
        name: str,
        value: Any,
        *,
        dtype: np.dtype | str | None = None,
        generation_step: int | None = None,
        squeeze_batch: bool = True,
    ) -> None:
        """Store one declared tensor, optionally under a denoising step."""
        if name not in self._declarations_by_name:
            raise KeyError(f"Unknown PI0.5 full capture site: {name}")
        array = _capture_numpy(value, dtype=dtype)
        if squeeze_batch and array.ndim > 0 and array.shape[0] == 1:
            array = np.squeeze(array, axis=0)
        if generation_step is None:
            self._arrays[name] = array
        else:
            self._step_arrays[name][int(generation_step)] = array

    def finalized_arrays(self, *, generation_steps: int) -> dict[str, np.ndarray]:
        """Return captured tensors with per-step captures stacked and padded."""
        arrays = dict(self._arrays)
        for name, by_step in self._step_arrays.items():
            if not by_step:
                continue
            sample = np.asarray(next(iter(by_step.values())))
            out = _empty_step_array(generation_steps, sample)
            for step, value in by_step.items():
                if 0 <= step < generation_steps:
                    out[step] = np.asarray(value, dtype=sample.dtype)
            arrays[name] = out
        return arrays

    def missing_names(self, arrays: Mapping[str, np.ndarray]) -> tuple[str, ...]:
        """Return declared full-capture site names absent from a finalized call."""
        captured = set(arrays)
        return tuple(item.name for item in self.declarations if item.name not in captured)

def _empty_step_array(generation_steps: int, sample: np.ndarray) -> np.ndarray:
    """Create a padded step-major array for sparse denoising-step captures."""
    if np.issubdtype(sample.dtype, np.floating):
        fill_value: float | int | bool = np.nan
    elif np.issubdtype(sample.dtype, np.bool_):
        fill_value = False
    else:
        fill_value = -1
    return np.full((generation_steps, *sample.shape), fill_value, dtype=sample.dtype)

def _capture_numpy(value: Any, *, dtype: np.dtype | str | None = None) -> np.ndarray:
    """Detach a Torch-like value and return a contiguous numpy array."""
    if hasattr(value, "detach"):
        value = value.detach()
    if dtype is not None and np.dtype(dtype).kind == "f" and hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "to"):
        value = value.to("cpu")
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if dtype is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            array = array.astype(np.dtype(dtype), copy=False)
    return np.ascontiguousarray(array)

def _capture_step(
    current_denoise_step: Mapping[str, int | None],
    stack: str,
) -> int | None:
    if stack == "expert":
        return current_denoise_step.get("index")
    return None

def _full_site_prefix(stack: str, layer: int) -> str:
    return f"pi05.{stack}.layers.{layer}"

def _capture_full_tensor(
    recorder: _PI05FullSiteRecorder | None,
    plan: CapturePlan,
    current_denoise_step: Mapping[str, int | None],
    *,
    stack: str,
    name: str,
    value: Any,
    dtype: np.dtype | str | None = None,
    squeeze_batch: bool = True,
) -> None:
    if recorder is None:
        return
    generation_step = _capture_step(current_denoise_step, stack)
    if stack == "expert" and generation_step is None:
        return
    recorder.capture(
        name,
        value,
        dtype=plan.np_dtype if dtype is None else dtype,
        generation_step=generation_step,
        squeeze_batch=squeeze_batch,
    )

def _register_forward_pre_hook(module: Any, hook: Any) -> Any:
    try:
        return module.register_forward_pre_hook(hook, with_kwargs=True)
    except TypeError:
        return module.register_forward_pre_hook(lambda mod, args: hook(mod, args, {}))

def _register_forward_hook(module: Any, hook: Any) -> Any:
    try:
        return module.register_forward_hook(hook, with_kwargs=True)
    except TypeError:
        return module.register_forward_hook(lambda mod, args, output: hook(mod, args, {}, output))

def _install_full_layer_hooks(
    recorder: _PI05FullSiteRecorder,
    plan: CapturePlan,
    current_denoise_step: Mapping[str, int | None],
    *,
    vlm_model: Any,
    expert_model: Any,
) -> tuple[list[Any], list[tuple[Any, Any]]]:
    """Install true-full capture hooks on selected VLM and expert layers."""
    handles: list[Any] = []
    patched_mlps: list[tuple[Any, Any]] = []

    for stack, model in (("vlm", vlm_model), ("expert", expert_model)):
        stack_layers = plan.vlm_layers if stack == "vlm" else plan.expert_layers
        for layer_idx, layer in enumerate(getattr(model, "layers", ())):
            if layer_idx not in stack_layers:
                continue
            prefix = _full_site_prefix(stack, int(layer_idx))
            handles.extend(
                _install_full_single_layer_hooks(
                    recorder,
                    plan,
                    current_denoise_step,
                    stack=stack,
                    layer=layer,
                    prefix=prefix,
                )
            )
            patched_mlps.append((layer.mlp, layer.mlp.forward))
            layer.mlp.forward = _make_full_mlp_forward(
                layer.mlp,
                recorder,
                plan,
                current_denoise_step,
                stack=stack,
                prefix=prefix,
            )

    return handles, patched_mlps

def _install_full_single_layer_hooks(
    recorder: _PI05FullSiteRecorder,
    plan: CapturePlan,
    current_denoise_step: Mapping[str, int | None],
    *,
    stack: str,
    layer: Any,
    prefix: str,
) -> list[Any]:
    """Install raw residual, norm, attention, and MLP hooks for one layer."""
    handles: list[Any] = []

    def capture(name: str, value: Any, *, dtype: np.dtype | str | None = None) -> None:
        _capture_full_tensor(
            recorder,
            plan,
            current_denoise_step,
            stack=stack,
            name=name,
            value=value,
            dtype=dtype,
        )

    def self_attn_pre_hook(_module: Any, args: tuple[Any, ...], _kwargs: Mapping[str, Any]) -> None:
        if args:
            capture(f"{prefix}.residual_pre_attention", args[0])

    def capture_adarms(
        module: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        normed: Any,
        *,
        norm_site: str,
    ) -> None:
        if stack != "expert":
            return
        cond = kwargs.get("cond") if isinstance(kwargs, Mapping) else None
        if cond is None and len(args) > 1:
            cond = args[1]
        dense = getattr(module, "dense", None)
        if cond is None or dense is None:
            return
        modulation = dense(cond)
        x = args[0] if args else normed
        if len(getattr(x, "shape", ())) == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        capture(f"{prefix}.{norm_site}.scale", scale)
        capture(f"{prefix}.{norm_site}.shift", shift)
        capture(f"{prefix}.{norm_site}.gate", gate)

    def input_norm_hook(
        module: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> None:
        normed = output[0] if isinstance(output, tuple) else output
        capture(f"{prefix}.attention_norm_output", normed)
        capture_adarms(module, args, kwargs, normed, norm_site="attention_adarms")

    def post_norm_pre_hook(_module: Any, args: tuple[Any, ...], _kwargs: Mapping[str, Any]) -> None:
        if not args:
            return
        residual = args[0]
        capture(f"{prefix}.residual_post_attention", residual)
        capture(f"{prefix}.residual_pre_mlp", residual)

    def post_norm_hook(
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: Mapping[str, Any],
        output: Any,
    ) -> None:
        normed = output[0] if isinstance(output, tuple) else output
        capture(f"{prefix}.mlp_norm_output", normed)
        capture_adarms(_module, _args, _kwargs, normed, norm_site="mlp_adarms")

    def o_proj_hook(
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: Mapping[str, Any],
        output: Any,
    ) -> None:
        capture(f"{prefix}.attention.o_proj", output)

    def layer_hook(
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: Mapping[str, Any],
        output: Any,
    ) -> None:
        value = output[0] if isinstance(output, tuple) else output
        capture(f"{prefix}.residual_post_mlp", value)

    handles.append(_register_forward_pre_hook(layer.self_attn, self_attn_pre_hook))
    handles.append(_register_forward_hook(layer.input_layernorm, input_norm_hook))
    handles.append(_register_forward_pre_hook(layer.post_attention_layernorm, post_norm_pre_hook))
    handles.append(_register_forward_hook(layer.post_attention_layernorm, post_norm_hook))
    handles.append(_register_forward_hook(layer.self_attn.o_proj, o_proj_hook))
    handles.append(_register_forward_hook(layer, layer_hook))
    return handles

def _make_full_mlp_forward(
    mlp: Any,
    recorder: _PI05FullSiteRecorder,
    plan: CapturePlan,
    current_denoise_step: Mapping[str, int | None],
    *,
    stack: str,
    prefix: str,
) -> Any:
    """Wrap a PI0.5 MLP forward pass to expose intermediate gate/up/down sites."""
    def capture(name: str, value: Any) -> None:
        _capture_full_tensor(
            recorder,
            plan,
            current_denoise_step,
            stack=stack,
            name=name,
            value=value,
        )

    def mlp_forward(x: Any) -> Any:
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        intermediate = mlp.act_fn(gate) * up
        down = mlp.down_proj(intermediate)
        capture(f"{prefix}.mlp.gate", gate)
        capture(f"{prefix}.mlp.up", up)
        capture(f"{prefix}.mlp.intermediate", intermediate)
        capture(f"{prefix}.mlp.down", down)
        capture(f"{prefix}.mlp.output", down)
        return down

    return mlp_forward

def _capture_full_attention_sites(
    recorder: _PI05FullSiteRecorder,
    plan: CapturePlan,
    *,
    stack: str,
    layer: int,
    generation_step: int | None,
    query: Any,
    key: Any,
    value: Any,
    pre_mask_scores: Any,
    post_mask_logits: Any,
    attention_probs: Any,
    attn_output: Any,
) -> None:
    prefix = _full_site_prefix(stack, int(layer))
    step_kwargs = {} if generation_step is None else {"generation_step": generation_step}
    recorder.capture(
        f"{prefix}.attention.q",
        query,
        dtype=plan.np_dtype,
        **step_kwargs,
    )
    recorder.capture(
        f"{prefix}.attention.k",
        key,
        dtype=plan.np_dtype,
        **step_kwargs,
    )
    recorder.capture(
        f"{prefix}.attention.v",
        value,
        dtype=plan.np_dtype,
        **step_kwargs,
    )
    recorder.capture(
        f"{prefix}.attention.pre_mask_scores",
        pre_mask_scores,
        dtype=plan.np_dtype,
        **step_kwargs,
    )
    recorder.capture(
        f"{prefix}.attention.post_mask_logits",
        post_mask_logits,
        dtype=plan.np_dtype,
        **step_kwargs,
    )
    recorder.capture(
        f"{prefix}.attention.attention_probs",
        attention_probs,
        dtype=plan.np_dtype,
        **step_kwargs,
    )
    recorder.capture(
        f"{prefix}.attention.attn_output_pre_o_proj",
        _flatten_attention_output(attn_output),
        dtype=plan.np_dtype,
        **step_kwargs,
    )
    recorder.capture(
        f"{prefix}.kv_cache.key",
        key,
        dtype=plan.np_dtype,
        **step_kwargs,
    )
    recorder.capture(
        f"{prefix}.kv_cache.value",
        value,
        dtype=plan.np_dtype,
        **step_kwargs,
    )

def _capture_expert_step_inputs(
    recorder: _PI05FullSiteRecorder,
    make_att_2d_masks_fn: Any,
    *,
    prefix_pad_masks: Any,
    x_t: Any,
    generation_step: int,
) -> None:
    import torch

    suffix_len = int(x_t.shape[1])
    batch_size = int(prefix_pad_masks.shape[0])
    prefix_len = int(prefix_pad_masks.shape[1])
    suffix_pad_masks = torch.ones(
        batch_size,
        suffix_len,
        dtype=torch.bool,
        device=x_t.device,
    )
    suffix_att_masks = torch.tensor(
        [1, *([0] * max(0, suffix_len - 1))],
        dtype=torch.bool,
        device=x_t.device,
    )[None, :].expand(batch_size, suffix_len)
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
        batch_size,
        suffix_len,
        prefix_len,
    )
    suffix_att_2d_masks = make_att_2d_masks_fn(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    recorder.capture(
        "pi05.expert.by_step.attention_mask",
        suffix_pad_masks,
        dtype=np.bool_,
        generation_step=generation_step,
    )
    recorder.capture(
        "pi05.expert.by_step.causal_mask",
        full_att_2d_masks,
        dtype=np.bool_,
        generation_step=generation_step,
    )
    recorder.capture(
        "pi05.expert.by_step.position_ids",
        position_ids,
        dtype=np.int64,
        generation_step=generation_step,
    )

def _flatten_attention_output(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) != 4:
        return value
    return value.reshape(shape[0], shape[1], shape[2] * shape[3]).contiguous()

def _rope_metadata_array(model: Any) -> np.ndarray:
    config = getattr(model, "config", None)
    rotary = getattr(model, "rotary_emb", None)
    values = [
        getattr(config, "head_dim", np.nan),
        getattr(config, "max_position_embeddings", np.nan),
        getattr(config, "rope_theta", np.nan),
        getattr(rotary, "base", np.nan),
    ]
    return np.asarray(
        [float(value) if value is not None else np.nan for value in values],
        dtype=np.float32,
    )
