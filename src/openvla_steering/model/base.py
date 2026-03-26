"""Base interfaces for policy backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(slots=True)
class PolicyStep:
    action: np.ndarray
    info: dict[str, Any]


class PolicyBackend(Protocol):
    policy_id: str
    target_object: str | None

    def reset(self, initial_obs: dict[str, np.ndarray], scene_metadata: dict[str, Any]) -> None:
        """Reset policy state for a new rollout."""

    def act(self, obs: dict[str, np.ndarray], step: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Produce one action and optional step metadata."""

