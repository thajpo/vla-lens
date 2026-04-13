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

    def __init__(self, model: nn.Module, module_names: list[str]) -> None:
        self.model = model
        self.module_names = list(module_names)
        self._cache: dict[str, torch.Tensor] = {}
        self._handles: list[Any] = []

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

        for name, module in self.model.named_modules():
            if name in self.module_names:
                # Close over `name` with a default arg to avoid late-binding
                def make_hook(n: str):
                    def hook(module, input, output):
                        # output may be a tensor or a tuple (e.g., from attention layers)
                        # For transformer decoder layers, output is typically a tuple
                        # where the first element is the hidden state tensor.
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

def resolve_token_position(tokenizer, instruction: str, spec: str) -> int:
    """
    Resolve a position spec to a token index within the tokenized instruction.

    Args:
        tokenizer: HuggingFace tokenizer (must support return_offsets_mapping)
        instruction: The task instruction string
        spec: One of "color_word", "final", "eos"

    Returns:
        Integer index into the tokenized sequence.

    Raises:
        ValueError if spec is "color_word" and no color word is found.
    """
    if spec == "final":
        tokens = tokenizer.encode(instruction)
        return len(tokens) - 1

    if spec == "eos":
        tokens = tokenizer.encode(instruction)
        eos_id = tokenizer.eos_token_id
        if eos_id in tokens:
            return tokens.index(eos_id)
        return len(tokens) - 1

    if spec == "color_word":
        lower = instruction.lower()
        for color in ["red", "white"]:
            if color not in lower:
                continue
            char_start = lower.index(color)
            char_end = char_start + len(color)

            try:
                encoding = tokenizer(
                    instruction,
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                )
                offsets = encoding["offset_mapping"]
            except Exception:
                # Fallback: tokenize without offset mapping if tokenizer doesn't support it
                tokens = tokenizer.encode(instruction, add_special_tokens=False)
                # Approximate: return midpoint of sequence
                import warnings
                warnings.warn(
                    "Tokenizer does not support offset_mapping; "
                    "falling back to sequence midpoint for color_word position."
                )
                return len(tokens) // 2

            color_token_indices = [
                i for i, (s, e) in enumerate(offsets)
                if s < char_end and e > char_start and s != e
            ]
            if not color_token_indices:
                raise ValueError(
                    f"Color word '{color}' found in instruction text but not "
                    f"in token offset mapping. Tokenized sequence:\n"
                    f"  {tokenizer.convert_ids_to_tokens(encoding['input_ids'])}"
                )
            # Use last subword token — accumulates full word representation via causal attention
            return color_token_indices[-1]

        raise ValueError(
            f"No color word ('red' or 'white') found in instruction: {instruction!r}"
        )

    raise ValueError(f"Unknown position spec {spec!r}. Expected: 'color_word', 'final', 'eos'")
