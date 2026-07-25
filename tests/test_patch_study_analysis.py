from __future__ import annotations

import pytest

from vla_lens.interventions.patch_study_analysis import summarize_patch_records


def test_patch_study_summary_groups_pairs_and_preserves_uncertainty():
    records = [
        {
            "pair_id": f"pair-{index}",
            "layer": 4,
            "token_region": "active_images",
            "verdict": "localized_transfer",
            "natural_delta_norm": 8.0,
            "patch_delta_norm": 7.0,
            "direction_agreement": 0.98,
            "transfer_fraction": transfer,
            "donor_gap_remaining": 0.1,
            "donor_recovery": 0.8,
            "off_direction_norm": 0.2,
            "off_direction_fraction": 0.03,
        }
        for index, transfer in enumerate((0.8, 0.9, 1.0, 0.85, 0.95))
    ]

    summaries = summarize_patch_records(
        records,
        bootstrap_samples=2_000,
        seed=7,
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["pair_count"] == 5
    assert summary["localized_pair_count"] == 5
    assert summary["transfer_mean"] == pytest.approx(0.9)
    assert summary["transfer_ci95_low"] < summary["transfer_mean"]
    assert summary["transfer_ci95_high"] > summary["transfer_mean"]
