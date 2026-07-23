"""Small orchestration helpers for sharing expensive campaign precomputes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from vla_lens.selectors import ActivationQuery
from vla_lens.traces import TraceDataset


@dataclass(frozen=True, slots=True)
class PreparedFeature:
    """Result of preparing one unique feature matrix."""

    cache_key: str
    names: tuple[str, ...]
    shape: tuple[int, ...]
    row_count: int
    built: bool


@dataclass(frozen=True, slots=True)
class CampaignPreparation:
    """Summary of a deduplicated precompute pass."""

    campaign_id: str
    requested_count: int
    unique_count: int
    features: tuple[PreparedFeature, ...]


def prepare_feature_campaign(
    dataset: TraceDataset,
    campaign: Mapping[str, Any],
) -> CampaignPreparation:
    """Materialize each unique feature selector once.

    ``precomputes`` is a list of ``{"name": ..., "selector": {...}}``
    records.  Identical selectors collapse to one cache build even when several
    experiments use different names for that input.
    """
    campaign_id = str(campaign.get("campaign_id") or campaign.get("id") or "campaign")
    raw_requests = campaign.get("precomputes", campaign.get("features", ()))
    if not isinstance(raw_requests, Sequence) or isinstance(raw_requests, (str, bytes)):
        raise ValueError("Campaign precomputes must be a list")

    requests: dict[str, tuple[ActivationQuery, list[str]]] = {}
    for index, item in enumerate(raw_requests):
        if not isinstance(item, Mapping):
            raise ValueError(f"Campaign precompute {index} must be a mapping")
        selector_payload = item.get("selector")
        if not isinstance(selector_payload, Mapping):
            raise ValueError(f"Campaign precompute {index} requires a selector mapping")
        selector = ActivationQuery.from_dict(selector_payload)
        view = dataset.select_model_sites(selector)
        key = view.cache_key()
        name = str(item.get("name") or f"feature-{index + 1}")
        if key not in requests:
            requests[key] = (selector, [])
        requests[key][1].append(name)

    prepared: list[PreparedFeature] = []
    for key, (selector, names) in requests.items():
        matrix = dataset.select_model_sites(selector).materialize(cache=True)
        prepared.append(
            PreparedFeature(
                cache_key=key,
                names=tuple(names),
                shape=tuple(int(item) for item in matrix.X.shape),
                row_count=int(len(matrix.rows)),
                built=matrix.cache_built,
            )
        )
    return CampaignPreparation(
        campaign_id=campaign_id,
        requested_count=len(raw_requests),
        unique_count=len(prepared),
        features=tuple(prepared),
    )


__all__ = ["CampaignPreparation", "PreparedFeature", "prepare_feature_campaign"]
