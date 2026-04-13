"""
patching.py

Activation patching (causal tracing) protocol for Experiment 5.

Protocol:
  1. Clean forward pass: correct instruction -> record activations + action
  2. Corrupt forward pass: wrong instruction -> record activations + action
  3. Patched forward pass: wrong instruction, but replace h[layer][position]
     with the clean activation -> record action

Recovery score = 1 - ||a_patched - a_clean|| / ||a_corrupt - a_clean||

All evaluation is OFFLINE — observations are loaded from disk, not collected
from a live environment. See scripts/patch_and_rollout.py for the driver.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .hooks import HookManager


# ---------------------------------------------------------------------------
# Forward pass helpers
# ---------------------------------------------------------------------------

def _run_forward(model, obs_image, instruction: str, unnorm_key: str) -> dict[str, Any]:
    """
    Run a single inference step and return action + VQ debug info.

    Args:
        model: Loaded VLA model
        obs_image: PIL.Image or numpy array (uint8, HxWx3)
        instruction: Task instruction string
        unnorm_key: Dataset normalization key (e.g. "libero_90")

    Returns:
        dict with 'action' (np.ndarray shape (7,)) and any debug fields
    """
    from PIL import Image as PILImage
    if not isinstance(obs_image, PILImage.Image):
        obs_image = PILImage.fromarray(obs_image).convert("RGB")
    action = model.predict_action(obs_image, instruction, unnorm_key=unnorm_key)
    return {"action": action}


# ---------------------------------------------------------------------------
# Core patching functions
# ---------------------------------------------------------------------------

def run_clean_and_corrupt(
    model: nn.Module,
    obs_image,
    clean_instruction: str,
    corrupt_instruction: str,
    layer_names: list[str],
    unnorm_key: str,
    token_positions: list[int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Run clean and corrupt forward passes, returning activations and actions.

    Args:
        model: Loaded VLA model
        obs_image: Single observation image (PIL or numpy uint8)
        clean_instruction: The correct task instruction
        corrupt_instruction: The wrong task instruction (color word swapped)
        layer_names: Module names to capture activations from
        unnorm_key: Dataset normalization key
        token_positions: If provided, slice the captured activation to these
                         token indices before storing. None = store full sequence.

    Returns:
        (clean_result, corrupt_result) — each is a dict with keys:
            'action':      (7,) numpy array
            'activations': {module_name: tensor} — (seq_len, hidden_dim) or
                           (len(token_positions), hidden_dim) if positions given
    """
    def _capture(instruction):
        with HookManager(model, layer_names) as hm:
            result = _run_forward(model, obs_image, instruction, unnorm_key)
            acts = {}
            for name in layer_names:
                tensor = hm.get(name)  # (seq_len, hidden_dim)
                if token_positions is not None:
                    tensor = tensor[token_positions]
                acts[name] = tensor
        result["activations"] = acts
        return result

    clean = _capture(clean_instruction)
    corrupt = _capture(corrupt_instruction)
    return clean, corrupt


def run_patched(
    model: nn.Module,
    obs_image,
    corrupt_instruction: str,
    clean_cache: dict[str, torch.Tensor],
    patch_layer: str,
    patch_position: int,
    unnorm_key: str,
) -> np.ndarray:
    """
    Run a forward pass with the corrupt instruction but with one layer's
    activation replaced by the clean activation at a specific token position.

    Args:
        model: Loaded VLA model
        obs_image: Single observation image
        corrupt_instruction: The wrong task instruction
        clean_cache: {module_name: tensor} from run_clean_and_corrupt
        patch_layer: Module name to patch
        patch_position: Token index to replace (e.g., index of color word)
        unnorm_key: Dataset normalization key

    Returns:
        Decoded action as (7,) numpy array.
    """
    if patch_layer not in clean_cache:
        raise KeyError(
            f"patch_layer '{patch_layer}' not found in clean_cache. "
            f"Available: {list(clean_cache.keys())}"
        )

    clean_activation = clean_cache[patch_layer]  # (seq_len, hidden_dim) on CPU

    patch_handles = []

    def make_patch_hook(clean_act: torch.Tensor, position: int):
        """Hook that replaces one token position in the module's output."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
                # Clone to avoid in-place modification of the computation graph
                hidden = hidden.clone()
                # clean_act may be (seq_len, hidden_dim); move to same device
                src = clean_act[position].to(hidden.device)
                hidden[:, position, :] = src.unsqueeze(0)
                return (hidden,) + output[1:]
            else:
                output = output.clone()
                src = clean_act[position].to(output.device)
                output[:, position, :] = src.unsqueeze(0)
                return output
        return hook

    # Register the patch hook
    for name, module in model.named_modules():
        if name == patch_layer:
            handle = module.register_forward_hook(
                make_patch_hook(clean_activation, patch_position)
            )
            patch_handles.append(handle)
            break

    if not patch_handles:
        raise KeyError(f"Module '{patch_layer}' not found in model.")

    try:
        result = _run_forward(model, obs_image, corrupt_instruction, unnorm_key)
    finally:
        for handle in patch_handles:
            handle.remove()

    return result["action"]


def recovery_score(
    clean_action: np.ndarray,
    corrupt_action: np.ndarray,
    patched_action: np.ndarray,
) -> float:
    """
    Compute the activation patching recovery score.

    Score = 1 - ||a_patched - a_clean|| / ||a_corrupt - a_clean||

    1.0 = full recovery (patched == clean)
    0.0 = no recovery (patched == corrupt)
    Negative values are possible if the patch moved the action further from clean.

    Clipped to [0, 1] for reporting.
    """
    denom = np.linalg.norm(corrupt_action - clean_action)
    if denom < 1e-8:
        # Clean and corrupt actions are the same — patching is uninformative
        return float("nan")
    raw = 1.0 - np.linalg.norm(patched_action - clean_action) / denom
    return float(np.clip(raw, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Batch patching sweep
# ---------------------------------------------------------------------------

def patch_sweep(
    model: nn.Module,
    obs_image,
    clean_instruction: str,
    corrupt_instruction: str,
    layer_names: list[str],
    patch_positions: list[int],
    unnorm_key: str,
) -> list[dict[str, Any]]:
    """
    Run a full patch sweep over all (layer, position) combinations for a
    single (observation, instruction pair).

    Returns:
        List of dicts, one per patch site, with keys:
            patch_layer, patch_position,
            clean_action, corrupt_action, patched_action,
            recovery_score, vq_code_match_clean, vq_code_match_corrupt
    """
    clean, corrupt = run_clean_and_corrupt(
        model, obs_image,
        clean_instruction, corrupt_instruction,
        layer_names, unnorm_key,
    )

    clean_action = clean["action"]
    corrupt_action = corrupt["action"]

    records = []
    for layer in layer_names:
        for pos in patch_positions:
            patched_action = run_patched(
                model, obs_image,
                corrupt_instruction,
                clean["activations"],
                patch_layer=layer,
                patch_position=pos,
                unnorm_key=unnorm_key,
            )
            score = recovery_score(clean_action, corrupt_action, patched_action)
            records.append({
                "patch_layer": layer,
                "patch_position": pos,
                "clean_action": clean_action.tolist(),
                "corrupt_action": corrupt_action.tolist(),
                "patched_action": patched_action.tolist(),
                "recovery_score": score,
                # Strict match: are the output actions close enough to be "the same decision"?
                # Using 0.1 L2 threshold on normalized action space as a heuristic
                "close_to_clean": bool(
                    np.linalg.norm(patched_action - clean_action) < 0.1
                ),
                "close_to_corrupt": bool(
                    np.linalg.norm(patched_action - corrupt_action) < 0.1
                ),
            })

    return records
