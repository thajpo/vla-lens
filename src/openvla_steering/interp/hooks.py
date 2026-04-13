"""
hooks.py

Forward hook manager for capturing activations from named modules in MiniVLA.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Module name constants (verify with discover_modules() on a loaded model)
# ---------------------------------------------------------------------------

def llm_layer_names(n_layers: int = 24) -> list[str]:
    """Return the canonical module names for each LLM transformer layer."""
    return [f"llm_backbone.llm.model.layers.{i}" for i in range(n_layers)]


def discover_modules(model: nn.Module, substring: str = "") -> list[str]:
    """
    Print and return all named module paths in the model, optionally filtered
    by a substring. Use this to verify LLM layer names and find VQ module paths.

    Example:
        discover_modules(vla, "vq")
        discover_modules(vla, "layers")
    """
    matches = [
        name for name, _ in model.named_modules()
        if substring.lower() in name.lower() and name
    ]
    for name in matches:
        print(name)
    return matches


# ---------------------------------------------------------------------------
# HookManager
# ---------------------------------------------------------------------------

class HookManager:
    """
    Context manager that registers forward hooks on named modules in a model,
    captures their outputs, and clears state on exit.

    Usage:
        with HookManager(model, ["llm_backbone.llm.model.layers.14"]) as hm:
            output = model(...)
            act = hm.get("llm_backbone.llm.model.layers.14")  # (seq_len, hidden_dim)

    The manager captures the *output* of each named module (the full residual
    stream tensor, not just the attention component).

    Module names must exist in model.named_modules() — a KeyError is raised
    during construction if any name is not found, before any forward pass.
    """

    def __init__(
        self,
        model: nn.Module,
        module_names: list[str],
        first_pass_only: bool = True,
    ) -> None:
        """
        Args:
            model: The VLA model.
            module_names: Module paths to hook (must exist in model.named_modules()).
            first_pass_only: If True (default), only capture the FIRST forward
                pass through each module per context-manager invocation.

                This is critical for VLA generation: predict_action calls
                model.generate, which does N forward passes (one per VQ action
                token) with KV caching. After the prefill pass, subsequent
                passes only process a single new token — the full instruction
                and image context are NOT re-processed. Capturing only the
                first pass gives the full-sequence activation that includes the
                color-word token position. Subsequent passes produce shape
                (1, hidden_dim), not (seq_len, hidden_dim).
        """
        self.model = model
        self.module_names = list(module_names)
        self.first_pass_only = first_pass_only
        self._cache: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []
        self._call_count: dict[str, int] = {}

        # Validate all names up front — fail loudly rather than silently skip
        named = {name for name, _ in model.named_modules()}
        missing = [n for n in module_names if n not in named]
        if missing:
            raise KeyError(
                f"Module(s) not found in model:\n"
                + "\n".join(f"  {m}" for m in missing)
                + "\nUse discover_modules(model) to list available names."
            )

    def __enter__(self) -> "HookManager":
        self._cache.clear()
        self._handles.clear()
        self._call_count = {n: 0 for n in self.module_names}

        for name, module in self.model.named_modules():
            if name in self.module_names:
                def make_hook(n: str):
                    def hook(module, input, output):
                        if self.first_pass_only and self._call_count[n] > 0:
                            return  # skip all passes after the first
                        self._call_count[n] += 1
                        if isinstance(output, tuple):
                            tensor = output[0]
                        else:
                            tensor = output
                        # Detach and move to CPU to avoid holding GPU memory
                        self._cache[n] = tensor.detach().cpu()
                    return hook
                handle = module.register_forward_hook(make_hook(name))
                self._handles.append(handle)

        return self

    def __exit__(self, *_) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        # Do not clear _cache here — caller reads it after exit

    def get(self, module_name: str) -> torch.Tensor:
        """
        Return the captured activation for a module.

        Returns:
            Tensor of shape (seq_len, hidden_dim) — the first batch dimension
            is squeezed if batch_size == 1, as is standard for single-image
            VLA inference.
        """
        if module_name not in self._cache:
            raise KeyError(
                f"No activation cached for '{module_name}'. "
                f"Available: {list(self._cache.keys())}"
            )
        tensor = self._cache[module_name]
        # Squeeze batch dimension for single-sample inference
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        return tensor

    def clear(self) -> None:
        """Clear the activation cache. Call between episodes."""
        self._cache.clear()

    def dry_run(self, model_inputs: dict) -> dict[str, tuple[int, ...]]:
        """
        Run a single forward pass and return the shape of each captured module's
        output without storing full tensors. Useful for verifying module names
        and tensor shapes before a full data collection run.

        Args:
            model_inputs: keyword arguments to pass to model.forward()

        Returns:
            Dict mapping module_name -> output tensor shape
        """
        shapes = {}
        with self:
            with torch.no_grad():
                self.model(**model_inputs)
            for name in self.module_names:
                if name in self._cache:
                    shapes[name] = tuple(self._cache[name].shape)
        return shapes


# ---------------------------------------------------------------------------
# Token position resolution
# ---------------------------------------------------------------------------
#
# ARCHITECTURE NOTE — MiniVLA full-sequence layout:
#
# The LLM sees a concatenated sequence of:
#   [BOS_token] [n_patches vision tokens] [rest of input_ids (tokens 1..N)]
#
# Concretely for MiniVLA with 256 projected patches and 44 input_ids:
#   Position 0        : BOS (first input_id token)
#   Positions 1..256  : 256 vision patch embeddings (inserted after BOS)
#   Positions 257..299: remaining input_ids (tokens 1..43)
#
# So the total sequence length = 1 + n_patches + (len(input_ids) - 1)
#                              = len(input_ids) + n_patches
#
# Captured hidden-state tensors have this full shape: (total_seq_len, hidden_dim).
# Positions computed from instruction text alone must be offset by n_patches
# (except position 0 which stays at 0).
#
# N_VISION_PATCHES is a constant for MiniVLA (224px, 14px patch, register tokens included).
# Verify with: discover_modules(model, "embed_tokens") + one prefill pass shape check.

N_VISION_PATCHES = 256  # confirmed: projected_patches shape=(1, 256, 896)


def find_color_word_in_full_seq(
    tokenizer,
    full_input_ids: list[int],
    instruction: str,
    n_vision_patches: int = N_VISION_PATCHES,
) -> int:
    """
    Find the last subword token of the color word in the FULL LLM input_ids
    (before vision patch insertion), then shift by n_vision_patches to get the
    position in the full concatenated sequence (with patches inserted after BOS).

    Args:
        full_input_ids: The actual input_ids list captured during the prefill
                        (e.g., from a hook on embed_tokens). Shape: [seq_len].
        instruction: Task instruction string (used to find offset_mapping).
        n_vision_patches: Number of vision patch tokens inserted after position 0.

    Returns:
        Integer index into the full (vision+text) sequence.
    """
    lower = instruction.lower()
    for color in ["red", "white"]:
        if color not in lower:
            continue
        char_start = lower.index(color)
        char_end = char_start + len(color)

        try:
            encoding = tokenizer(instruction, return_offsets_mapping=True, add_special_tokens=False)
            offsets = encoding["offset_mapping"]
            inst_ids = encoding["input_ids"]
        except Exception:
            # No offset mapping support — fall back to last text token
            return -1

        color_token_indices = [
            i for i, (s, e) in enumerate(offsets)
            if s < char_end and e > char_start and s != e
        ]
        if not color_token_indices:
            return -1  # fallback

        # Index in instruction-only tokens (no special tokens)
        inst_idx = color_token_indices[-1]

        # Find this instruction token's position in full_input_ids by matching token IDs.
        # full_input_ids = [BOS?, ...system_prompt..., inst_token_0, ..., inst_token_N, EOS?]
        # We scan for the first occurrence of inst_ids[inst_idx] that also has
        # inst_ids[inst_idx-1] appearing one position before (context match).
        target_id = inst_ids[inst_idx]
        for k in range(1, len(full_input_ids)):
            if full_input_ids[k] == target_id:
                # Verify surrounding context
                if inst_idx > 0:
                    prev_matches = (k > 0 and full_input_ids[k - 1] == inst_ids[inst_idx - 1])
                else:
                    prev_matches = True
                if prev_matches:
                    # k is the position in full_input_ids (text space, before patch insertion)
                    # Map to full LLM sequence: position 0 stays, all others shift by n_patches
                    if k == 0:
                        return 0
                    return k + n_vision_patches

        # Fallback: color word token not matched in full_input_ids
        return -1

    raise ValueError(f"No color word ('red' or 'white') found in instruction: {instruction!r}")


def resolve_token_position_in_full_seq(
    spec: str,
    full_input_ids: list[int],
    instruction: str,
    tokenizer,
    n_vision_patches: int = N_VISION_PATCHES,
) -> int:
    """
    Resolve a position spec to an index in the FULL LLM sequence
    (vision patches + text tokens concatenated).

    Args:
        spec: One of "color_word", "final", "eos"
        full_input_ids: The text-only input_ids from the prefill pass
                        (before vision patch insertion), captured via hook.
        instruction: Task instruction string.
        tokenizer: HuggingFace tokenizer.
        n_vision_patches: Number of vision tokens inserted after position 0.

    Returns:
        Integer index into the full (seq_len,) activation tensor.
        Returns -1 for "final" and "eos" to use Python last-element indexing.
    """
    if spec == "final":
        # Last position in the full sequence — the final text token or EOS.
        # Using -1 (Python last-element) is unambiguous regardless of n_vision_patches.
        return -1

    if spec == "eos":
        eos_id = tokenizer.eos_token_id
        if eos_id in full_input_ids:
            k = len(full_input_ids) - 1 - full_input_ids[::-1].index(eos_id)
            if k == 0:
                return 0
            return k + n_vision_patches
        return -1  # fallback: last position

    if spec == "color_word":
        return find_color_word_in_full_seq(
            tokenizer, full_input_ids, instruction, n_vision_patches
        )

    raise ValueError(f"Unknown position spec {spec!r}. Expected: 'color_word', 'final', 'eos'")
