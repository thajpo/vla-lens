"""Named PI0.5 runtime sites that support reconstructable source patching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from vla_lens.pi05.capture_schema import ALL_PI05_LAYERS

PatchStack = Literal["vlm_prefix", "expert_action"]

_VLM_SITE = re.compile(r"pi05\.vlm\.layers\.(\d+)\.prefix\.hidden_tokens")
_EXPERT_SITE = re.compile(r"pi05\.expert\.layers\.(\d+)\.by_step\.hidden_tokens")


@dataclass(frozen=True, slots=True)
class PI05PatchSite:
    """A live tensor address plus the axes needed to patch it correctly."""

    model_site: str
    stack: PatchStack
    layer: int
    token_space: str
    axes: tuple[str, ...]
    repeated_by_generation_step: bool

    def to_runtime_record(self) -> dict[str, Any]:
        return {
            "name": self.model_site,
            "model_site": self.model_site,
            "stack": self.stack,
            "layer": self.layer,
            "tensor_type": "hidden_tokens",
            "token_space_id": self.token_space,
            "axes": list(self.axes),
            "repeated_by_generation_step": self.repeated_by_generation_step,
            "materialization": "runtime_only",
            "saved_activation": False,
        }


def parse_pi05_patch_site(
    model_site: str,
    *,
    declared_layer: int | None = None,
) -> PI05PatchSite:
    """Resolve one supported live site without importing Torch or LeRobot."""

    value = str(model_site).strip()
    match = _VLM_SITE.fullmatch(value)
    if match is not None:
        site = PI05PatchSite(
            model_site=value,
            stack="vlm_prefix",
            layer=int(match.group(1)),
            token_space="pi05.prefix",
            axes=("token", "channel"),
            repeated_by_generation_step=False,
        )
    else:
        match = _EXPERT_SITE.fullmatch(value)
        if match is None:
            raise ValueError(f"Unsupported PI0.5 source-patch site {value!r}")
        site = PI05PatchSite(
            model_site=value,
            stack="expert_action",
            layer=int(match.group(1)),
            token_space="pi05.action_suffix",
            axes=("generation_step", "token", "channel"),
            repeated_by_generation_step=True,
        )
    if site.layer not in ALL_PI05_LAYERS:
        raise ValueError(f"PI0.5 layer {site.layer} is outside {ALL_PI05_LAYERS}")
    if declared_layer is not None and int(declared_layer) != site.layer:
        raise ValueError(
            f"Declared layer {declared_layer} disagrees with {site.model_site!r}"
        )
    return site


def pi05_runtime_patch_sites() -> tuple[str, ...]:
    """Return every lightweight capture site that can be rebuilt at replay time."""

    return tuple(
        [
            f"pi05.vlm.layers.{layer}.prefix.hidden_tokens"
            for layer in ALL_PI05_LAYERS
        ]
        + [
            f"pi05.expert.layers.{layer}.by_step.hidden_tokens"
            for layer in ALL_PI05_LAYERS
        ]
    )


def pi05_patch_module(policy: Any, site: PI05PatchSite) -> Any:
    """Resolve the concrete decoder module for a parsed site."""

    root = getattr(getattr(policy, "model", None), "paligemma_with_expert", None)
    if site.stack == "vlm_prefix":
        paligemma = getattr(root, "paligemma", None)
        candidates = (
            getattr(paligemma, "language_model", None),
            getattr(getattr(paligemma, "model", None), "language_model", None),
        )
    else:
        expert = getattr(root, "gemma_expert", None)
        candidates = (
            getattr(expert, "model", None),
            getattr(getattr(expert, "model", None), "language_model", None),
        )
    for model in candidates:
        layers = getattr(model, "layers", None)
        if layers is not None and 0 <= site.layer < len(layers):
            return layers[site.layer]
    raise ValueError(f"Loaded PI0.5 runtime does not expose {site.model_site!r}")


__all__ = [
    "PI05PatchSite",
    "parse_pi05_patch_site",
    "pi05_patch_module",
    "pi05_runtime_patch_sites",
]
