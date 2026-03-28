"""Adapters between robosuite observations and model inputs."""

from __future__ import annotations

from typing import Any

import numpy as np


def extract_camera_image(
    obs: dict[str, np.ndarray],
    camera_name: str,
) -> np.ndarray:
    image_key = f"{camera_name}_image"
    if image_key not in obs:
        available = sorted(key for key in obs if key.endswith("_image"))
        raise RuntimeError(
            f"Expected image observation key '{image_key}', but it was not present. "
            f"Available image keys: {available}"
        )
    image = obs[image_key]
    if image.ndim != 3:
        raise RuntimeError(f"Expected image with 3 dims, got shape {image.shape}")
    return image


def build_openvla_prompt(
    instruction: str,
    scene_metadata: dict[str, Any] | None = None,
) -> str:
    if not scene_metadata:
        return instruction
    return instruction

