"""Declarative PI0.5 intervention specifications.

These specs are intentionally lightweight.  Existing scripts can interpret them
to run concrete swaps/patches while reports can serialize the exact intended
site, donor, recipient, and control condition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class InterventionType(StrEnum):
    KV_LAYER_REPLACE = "kv_layer_replace"
    KV_WRONG_LAYER_CONTROL = "kv_wrong_layer_control"
    EXPERT_HIDDEN_REPLACE = "expert_hidden_replace"
    FLOW_STATE_REPLACE = "flow_state_replace"
    DIRECTION_ADD = "direction_add"
    DIRECTION_PROJECT_OUT = "direction_project_out"


@dataclass(frozen=True, slots=True)
class InterventionSpec:
    kind: InterventionType
    recipient_rollout_id: str
    donor_rollout_id: str | None = None
    call_index: int = 0
    layer: int | None = None
    donor_layer: int | None = None
    flow_step: int | None = None
    token_index: int | None = None
    scale: float = 1.0
    label: str | None = None

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["kind"] = str(self.kind)
        return record

    @classmethod
    def kv_rescue(
        cls,
        *,
        recipient_rollout_id: str,
        donor_rollout_id: str,
        layer: int,
        call_index: int = 0,
        label: str | None = None,
    ) -> "InterventionSpec":
        return cls(
            kind=InterventionType.KV_LAYER_REPLACE,
            recipient_rollout_id=recipient_rollout_id,
            donor_rollout_id=donor_rollout_id,
            call_index=call_index,
            layer=layer,
            label=label,
        )

    @classmethod
    def kv_wrong_layer_control(
        cls,
        *,
        recipient_rollout_id: str,
        donor_rollout_id: str,
        layer: int,
        donor_layer: int,
        call_index: int = 0,
        label: str | None = None,
    ) -> "InterventionSpec":
        return cls(
            kind=InterventionType.KV_WRONG_LAYER_CONTROL,
            recipient_rollout_id=recipient_rollout_id,
            donor_rollout_id=donor_rollout_id,
            call_index=call_index,
            layer=layer,
            donor_layer=donor_layer,
            label=label,
        )
