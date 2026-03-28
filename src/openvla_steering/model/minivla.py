"""MiniVLA backend via the upstream Prismatic loader."""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from openvla_steering.model.adapters import build_openvla_prompt, extract_camera_image
from openvla_steering.model.openvla import OpenVLAPolicy, OpenVLAPolicyConfig

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class MiniVLAPolicyConfig(OpenVLAPolicyConfig):
    model_id: str = "Stanford-ILIAD/minivla-vq-bridge-prismatic"
    unnorm_key: str | None = "bridge_dataset"


class MiniVLAPolicy(OpenVLAPolicy):
    """Thin backend shell for MiniVLA inference."""

    def __init__(self, config: MiniVLAPolicyConfig):
        super().__init__(config)
        self.policy_id = "minivla"

    def _try_load(self) -> None:
        try:
            import torch
            from huggingface_hub import snapshot_download
        except Exception as exc:  # pragma: no cover - dependency-gated path
            self._load_error = exc
            return

        repo_root = Path(__file__).resolve().parents[3]
        prismatic_root = repo_root / "third_party" / "openvla-mini"
        vq_root = repo_root / "third_party" / "vq_bet_official"
        if not prismatic_root.exists():
            self._load_error = RuntimeError(
                f"Missing vendored MiniVLA code at {prismatic_root}. "
                "Clone Stanford-ILIAD/openvla-mini into third_party/openvla-mini."
            )
            return
        if not vq_root.exists():
            self._load_error = RuntimeError(
                f"Missing vendored VQ-Bet code at {vq_root}. "
                "Clone jayLEE0301/vq_bet_official into third_party/vq_bet_official."
            )
            return

        os.environ.setdefault("PRISMATIC_DATA_ROOT", str(repo_root / "artifacts" / "prismatic_data"))
        if str(prismatic_root) not in sys.path:
            sys.path.insert(0, str(prismatic_root))
        if str(vq_root) not in sys.path:
            sys.path.insert(0, str(vq_root))

        try:
            from prismatic.models.load import load_vla
        except Exception as exc:  # pragma: no cover - dependency-gated path
            self._load_error = exc
            return

        try:
            snapshot_dir = Path(
                snapshot_download(
                    repo_id=self.config.model_id,
                    allow_patterns=[
                        "config.json",
                        "dataset_statistics.json",
                        "checkpoints/*.pt",
                    ],
                )
            )
            checkpoint_paths = sorted((snapshot_dir / "checkpoints").glob("step-*.pt"))
            if not checkpoint_paths:
                raise RuntimeError(f"No MiniVLA checkpoint found under {snapshot_dir / 'checkpoints'}")
            checkpoint_path = checkpoint_paths[-1]
            self._model = load_vla(str(checkpoint_path), image_sequence_len=1)
        except Exception as exc:  # pragma: no cover - model-load path
            self._load_error = exc
            return

        if self.config.device == "auto":
            target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            target_device = torch.device(self.config.device)
        half_dtype = getattr(self._model.llm_backbone, "half_precision_dtype", torch.bfloat16)
        self._model.vision_backbone.to(dtype=half_dtype)
        self._model.llm_backbone.to(dtype=half_dtype)
        self._model.to(dtype=half_dtype)
        self._model.to(target_device)

    def act(self, obs: dict[str, np.ndarray], step: int) -> tuple[np.ndarray, dict[str, Any]]:
        if self._load_error is not None:
            raise RuntimeError(
                "MiniVLA backend is not available because the upstream Prismatic loader or "
                "checkpoint dependencies could not be initialized."
            ) from self._load_error

        image = extract_camera_image(obs, camera_name=self.config.camera_name)
        prompt = build_openvla_prompt(self.config.instruction, scene_metadata=self._scene_metadata)
        image_pil = Image.fromarray(image).convert("RGB")

        raw_action = self._model.predict_action(
            image_pil,
            prompt,
            unnorm_key=self.config.unnorm_key,
            do_sample=False,
        )
        action = np.asarray(raw_action, dtype=np.float32)

        if not self._logged_first_action:
            LOGGER.info(
                "MiniVLA first action: shape=%s min=%.6f max=%.6f unnorm_key=%s",
                action.shape,
                float(np.min(action)),
                float(np.max(action)),
                self.config.unnorm_key,
            )
            self._logged_first_action = True

        if action.ndim != 1:
            action = action.reshape(-1)
        if action.shape[0] != self.config.action_dim:
            available_dim = None
            if hasattr(self._model, "get_action_dim"):
                try:
                    available_dim = int(self._model.get_action_dim(self.config.unnorm_key))
                except Exception:
                    available_dim = None
            raise RuntimeError(
                "MiniVLA returned an unexpected action dimension. "
                f"expected={self.config.action_dim} got={action.shape[0]} "
                f"model_action_dim={available_dim} unnorm_key={self.config.unnorm_key}"
            )

        action = np.clip(action, -1.0, 1.0)
        return action, {
            "phase": "minivla",
            "prompt": prompt,
            "image_shape": tuple(image.shape),
            "unnorm_key": self.config.unnorm_key,
        }
