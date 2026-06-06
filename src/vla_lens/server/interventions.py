"""Server payload helpers for intervention APIs."""

from __future__ import annotations

from typing import Any, Mapping

from vla_lens.interventions.preflight import intervention_preflight
from vla_lens.traces import TraceDataset


def _intervention_preflight_payload(
    dataset: TraceDataset,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    result = intervention_preflight(dataset, payload)
    return {"preflight": result.to_dict()}


__all__ = ["_intervention_preflight_payload"]
