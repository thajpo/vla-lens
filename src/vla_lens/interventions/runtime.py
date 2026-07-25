"""Generic intervention runtime protocols.

This module is deliberately runtime-light. Concrete model adapters, including
PI0.5, can implement these protocols inside their dedicated environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from vla_lens.interventions.serialization import jsonable


@dataclass(frozen=True, slots=True)
class RuntimeTrialOutput:
    """One action-producing runtime trial returned by an adapter."""

    trial_id: str
    trial_kind: str
    action_chunk: Any
    array_outputs: Mapping[str, Any] = field(default_factory=dict)
    control_kind: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    status: str = "ok"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "trial_kind": self.trial_kind,
            "control_kind": self.control_kind,
            "array_outputs": {
                str(name): {
                    "shape": list(getattr(value, "shape", ())),
                    "dtype": str(getattr(value, "dtype", "unknown")),
                }
                for name, value in self.array_outputs.items()
            },
            "metrics": jsonable(self.metrics),
            "runtime": jsonable(self.runtime),
            "status": self.status,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


class ActionInterventionExecutor(Protocol):
    """Protocol for adapters that can produce action chunks for intervention trials."""

    def run_noop(self, request: Mapping[str, Any]) -> RuntimeTrialOutput:
        """Run the selected policy call without intervention."""

    def run_intervention(self, request: Mapping[str, Any]) -> RuntimeTrialOutput:
        """Run the selected policy call with the requested intervention."""

    def run_control(
        self,
        request: Mapping[str, Any],
        *,
        control_kind: str,
    ) -> RuntimeTrialOutput:
        """Run one optional control trial."""


__all__ = ["ActionInterventionExecutor", "RuntimeTrialOutput"]
