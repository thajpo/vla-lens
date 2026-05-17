"""Named PI0.5 activation selectors.

Selectors are deliberately string-friendly so scripts, configs, and agents can
refer to activation sites without repeating schema-specific dictionary access.

Examples:
    vlm.layer.8.mean
    vlm.final.mean
    vlm.handoff.kv.layer.8.flat
    expert.layer.17.final_step.mean
    expert.layer.8.flow_step.3.mean
    flow.step.4.flat
    action.final.flat
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

Reduction = Literal["mean", "flat", "tokens"]


@dataclass(frozen=True, slots=True)
class ActivationSelector:
    family: str
    source: str
    layer: int | None = None
    flow_step: int | None = None
    reduction: Reduction = "mean"

    def name(self) -> str:
        parts = [self.family, self.source]
        if self.layer is not None:
            parts.extend(["layer", str(self.layer)])
        if self.flow_step is not None:
            parts.extend(["flow_step", str(self.flow_step)])
        parts.append(self.reduction)
        return ".".join(parts)

    def extract(self, vlm: dict[str, Any], expert: dict[str, Any]) -> np.ndarray:
        tensor = self._resolve_tensor(vlm, expert)
        return reduce_tensor(tensor, self.reduction)

    def _resolve_tensor(self, vlm: dict[str, Any], expert: dict[str, Any]) -> torch.Tensor:
        if self.family == "vlm" and self.source == "layer":
            if self.layer is None:
                raise ValueError("vlm.layer selector requires a layer")
            if self.layer in vlm.get("layer_hidden_states", {}):
                return vlm["layer_hidden_states"][self.layer]
            if str(self.layer) in vlm.get("layer_hidden_states", {}):
                return vlm["layer_hidden_states"][str(self.layer)]
            if self.layer == _metadata_int(vlm, "vlm_final_layer"):
                return vlm["final_layer_hidden_state"]
            raise KeyError(f"VLM layer {self.layer} not present in payload")

        if self.family == "vlm" and self.source == "final":
            return vlm["final_layer_hidden_state"]

        if self.family == "vlm" and self.source == "prefix":
            return vlm["prefix_final_hidden_state"]

        if self.family == "vlm" and self.source == "handoff_kv":
            if self.layer is None:
                raise ValueError("vlm.handoff.kv selector requires a layer")
            key_value = vlm["prefix_past_key_values"][self.layer]
            tensors = [item for item in key_value if isinstance(item, torch.Tensor)]
            if not tensors:
                raise KeyError(f"No tensor KV entries found for layer {self.layer}")
            return torch.cat([item.reshape(-1).to(torch.float32) for item in tensors])

        if self.family == "expert" and self.source == "layer_final":
            if self.layer is None:
                raise ValueError("expert.layer final-step selector requires a layer")
            selected = expert["expert_selected_hidden_final_step"]
            return selected.get(self.layer, selected.get(str(self.layer)))

        if self.family == "expert" and self.source == "layer_flow":
            if self.layer is None or self.flow_step is None:
                raise ValueError("expert.layer flow-step selector requires layer and flow_step")
            by_step = expert["expert_selected_hidden_by_step"]
            selected = by_step[self.flow_step]
            return selected.get(self.layer, selected.get(str(self.layer)))

        if self.family == "flow" and self.source == "x_t":
            if self.flow_step is None:
                raise ValueError("flow.step selector requires a flow step")
            return expert["flow_x_t"][self.flow_step]

        if self.family == "action" and self.source == "final":
            return expert["final_action_chunk"]

        raise ValueError(f"Unsupported selector: {self}")


def parse_selector(value: str) -> ActivationSelector:
    parts = value.split(".")
    if parts[:2] == ["vlm", "layer"]:
        return ActivationSelector("vlm", "layer", layer=int(parts[2]), reduction=_last(parts))
    if parts[:2] == ["vlm", "final"]:
        return ActivationSelector("vlm", "final", reduction=_last(parts))
    if parts[:2] == ["vlm", "prefix"]:
        return ActivationSelector("vlm", "prefix", reduction=_last(parts))
    if parts[:4] == ["vlm", "handoff", "kv", "layer"]:
        return ActivationSelector("vlm", "handoff_kv", layer=int(parts[4]), reduction=_last(parts))
    if parts[:2] == ["expert", "layer"] and "final_step" in parts:
        return ActivationSelector(
            "expert",
            "layer_final",
            layer=int(parts[2]),
            reduction=_last(parts),
        )
    if parts[:2] == ["expert", "layer"] and "flow_step" in parts:
        flow_idx = parts.index("flow_step")
        return ActivationSelector(
            "expert",
            "layer_flow",
            layer=int(parts[2]),
            flow_step=int(parts[flow_idx + 1]),
            reduction=_last(parts),
        )
    if parts[:2] == ["flow", "step"]:
        return ActivationSelector("flow", "x_t", flow_step=int(parts[2]), reduction=_last(parts))
    if parts[:2] == ["action", "final"]:
        return ActivationSelector("action", "final", reduction=_last(parts))
    raise ValueError(f"Cannot parse activation selector: {value}")


def reduce_tensor(tensor: torch.Tensor | None, reduction: Reduction) -> np.ndarray:
    if tensor is None:
        raise KeyError("Selected activation is missing")
    value = tensor.detach().to(dtype=torch.float32, device="cpu")
    if reduction == "flat":
        return value.reshape(-1).numpy()
    if reduction == "tokens":
        if value.dim() >= 3 and value.shape[0] == 1:
            value = value.squeeze(0)
        return value.numpy()
    if reduction == "mean":
        if value.dim() == 1:
            return value.numpy()
        if value.dim() >= 3 and value.shape[0] == 1:
            value = value.squeeze(0)
        return value.reshape(-1, value.shape[-1]).mean(axis=0).numpy()
    raise ValueError(f"Unknown reduction: {reduction}")


def _last(parts: list[str]) -> Reduction:
    if parts[-1] in {"mean", "flat", "tokens"}:
        return parts[-1]  # type: ignore[return-value]
    return "mean"


def _metadata_int(payload: dict[str, Any], key: str) -> int | None:
    metadata = payload.get("capture_metadata") or {}
    value = metadata.get(key)
    return int(value) if value is not None else None
