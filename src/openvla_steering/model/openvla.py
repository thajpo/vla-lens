"""OpenVLA backend shell.

This intentionally starts narrow:
- load processor + model if dependencies are available
- expose the same policy interface as scripted backends
- fail fast when the current env observations are insufficient
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from openvla_steering.model.adapters import build_openvla_prompt, extract_camera_image

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OpenVLAPolicyConfig:
    model_id: str = "openvla/openvla-7b"
    instruction: str = "pick up the target block"
    trust_remote_code: bool = True
    device: str = "auto"
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    unnorm_key: str | None = None
    action_dim: int = 7
    camera_name: str = "frontview"


class OpenVLAPolicy:
    """Thin backend shell for OpenVLA inference."""

    def __init__(self, config: OpenVLAPolicyConfig):
        self.config = config
        self.policy_id = "openvla"
        self.target_object: str | None = None
        self._processor: Any | None = None
        self._model: Any | None = None
        self._load_error: Exception | None = None
        self._scene_metadata: dict[str, Any] | None = None
        self._logged_first_action = False
        self._try_load()

    def _try_load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except Exception as exc:  # pragma: no cover - dependency-gated path
            self._load_error = exc
            return

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": dtype_map.get(self.config.torch_dtype, torch.bfloat16),
            "low_cpu_mem_usage": True,
        }
        if self.config.attn_implementation is not None:
            kwargs["attn_implementation"] = self.config.attn_implementation

        self._processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.config.model_id,
            **kwargs,
        )

        if self.config.device != "auto":
            self._model = self._model.to(self.config.device)

    def reset(self, initial_obs: dict[str, np.ndarray], scene_metadata: dict[str, Any]) -> None:
        self.target_object = None
        self._scene_metadata = scene_metadata

    def act(self, obs: dict[str, np.ndarray], step: int) -> tuple[np.ndarray, dict[str, Any]]:
        if self._load_error is not None:
            raise RuntimeError(
                "OpenVLA backend is not available because required dependencies are missing. "
                "Install a ROCm-compatible PyTorch build first, then install transformers and "
                "the minimal OpenVLA inference dependencies."
            ) from self._load_error

        image = extract_camera_image(obs, camera_name=self.config.camera_name)
        prompt = build_openvla_prompt(self.config.instruction, scene_metadata=self._scene_metadata)
        image_pil = Image.fromarray(image)

        inputs = self._processor(text=prompt, images=image_pil, return_tensors="pt")

        if hasattr(self._model, "device"):
            model_device = self._model.device
            inputs = {
                key: value.to(model_device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        model_dtype = getattr(self._model, "dtype", None)
        if model_dtype is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

        unnorm_key = self.config.unnorm_key
        norm_stats = getattr(self._model, "norm_stats", None)
        if isinstance(norm_stats, dict):
            if unnorm_key is None and len(norm_stats) == 1:
                unnorm_key = next(iter(norm_stats))
            elif unnorm_key is None and len(norm_stats) > 1:
                raise RuntimeError(
                    "OpenVLA action unnormalization is ambiguous. "
                    f"Available unnorm keys: {sorted(norm_stats)}"
                )

        raw_action = self._model.predict_action(unnorm_key=unnorm_key, **inputs)
        action = np.asarray(raw_action, dtype=np.float32)

        if not self._logged_first_action:
            LOGGER.info(
                "OpenVLA first action: shape=%s min=%.6f max=%.6f unnorm_key=%s",
                action.shape,
                float(np.min(action)),
                float(np.max(action)),
                unnorm_key,
            )
            self._logged_first_action = True

        if action.ndim != 1:
            action = action.reshape(-1)
        if action.shape[0] != self.config.action_dim:
            available_dim = None
            if hasattr(self._model, "get_action_dim"):
                try:
                    available_dim = int(self._model.get_action_dim(unnorm_key))
                except Exception:
                    available_dim = None
            raise RuntimeError(
                "OpenVLA returned an unexpected action dimension. "
                f"expected={self.config.action_dim} got={action.shape[0]} "
                f"model_action_dim={available_dim} unnorm_key={unnorm_key}"
            )

        action = np.clip(action, -1.0, 1.0)

        return action, {
            "phase": "openvla",
            "prompt": prompt,
            "image_shape": tuple(image.shape),
            "unnorm_key": unnorm_key,
        }
