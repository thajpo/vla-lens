from __future__ import annotations

from dataclasses import replace

import pytest

from vla_lens.probe_evidence import (
    ContributionItem,
    ProbeEvidenceContractError,
    ResearchSelectionState,
    attention_head_grouped_probe_bundle,
    default_probe_panel_specs,
    pooled_no_contribution_probe_bundle,
    probe_episode_lens_adapter,
    raw_layer_contribution_probe_bundle,
    sae_feature_contribution_probe_bundle,
    scalar_timestep_probe_bundle,
    select_available_panels,
    select_contribution_claim_level,
    select_contribution_rows,
    select_current_moment_evidence,
    select_top_moments,
    select_unavailable_reasons,
)


def test_select_available_panels_covers_all_golden_probe_fixtures():
    for bundle in (
        scalar_timestep_probe_bundle(),
        pooled_no_contribution_probe_bundle(),
        raw_layer_contribution_probe_bundle(),
        sae_feature_contribution_probe_bundle(),
        attention_head_grouped_probe_bundle(),
    ):
        panels = {
            panel.panel_id: panel
            for panel in select_available_panels(bundle, default_probe_panel_specs())
        }

        assert set(panels) == {
            "probe_provenance",
            "score_series",
            "ranked_moments",
            "prediction",
            "contribution",
            "model_locus",
            "failure_cases",
            "unavailable_reasons",
        }
        assert panels["probe_provenance"].available is True
        assert panels["unavailable_reasons"].available is True


def test_select_top_moments_preserves_ranking_and_limit_without_fabricating_data():
    bundle = scalar_timestep_probe_bundle()

    assert select_top_moments(bundle)[0].score == 0.91
    assert select_top_moments(bundle, "bottom")[0].episode_id == "episode-2"
    assert select_top_moments(bundle, "uncertain") == ()
    assert select_top_moments(bundle, limit=0) == ()

    with pytest.raises(ProbeEvidenceContractError, match="limit"):
        select_top_moments(bundle, limit=-1)


def test_select_current_moment_evidence_narrows_to_active_scalar_timestep():
    bundle = scalar_timestep_probe_bundle()
    selection = ResearchSelectionState(
        dataset_id=bundle.run.dataset_id,
        lens_id=bundle.artifact.lens_id,
        lens_run_id=bundle.run.lens_run_id,
        episode_id="episode-1",
        timestep=7,
        ranking="top",
    )

    current = select_current_moment_evidence(bundle, selection)

    assert [moment.score for moment in current.ranked_moments] == [0.91]
    assert [prediction.confidence for prediction in current.predictions] == [0.91]
    assert [series.episode_id for series in current.score_series] == ["episode-1"]
    assert current.contributions == ()
    assert current.model_loci == ()
    assert current.unavailable


def test_select_current_moment_evidence_keeps_global_model_locus_when_selection_matches_episode():
    bundle = raw_layer_contribution_probe_bundle()
    selection = ResearchSelectionState(
        dataset_id=bundle.run.dataset_id,
        lens_id=bundle.artifact.lens_id,
        lens_run_id=bundle.run.lens_run_id,
        episode_id="episode-1",
        policy_call=3,
    )

    current = select_current_moment_evidence(bundle, selection)

    assert len(current.predictions) == 1
    assert len(current.contributions) == 1
    assert len(current.model_loci) == 1
    assert current.model_loci[0].locus.layer == 8


def test_select_contribution_rows_and_claim_level_are_selection_aware():
    bundle = raw_layer_contribution_probe_bundle()
    selection = ResearchSelectionState(episode_id="episode-1", policy_call=3)

    rows = select_contribution_rows(bundle, selection)

    assert [row.key for row in rows] == ["dim_42"]
    assert select_contribution_rows(bundle, ResearchSelectionState(episode_id="episode-2")) == ()
    assert select_contribution_claim_level(bundle, selection) == "numeric_only"
    assert (
        select_contribution_claim_level(sae_feature_contribution_probe_bundle())
        == "human_labeled_feature"
    )
    assert select_contribution_claim_level(scalar_timestep_probe_bundle()) is None


def test_select_unavailable_reasons_filters_by_panel_and_capability():
    bundle = scalar_timestep_probe_bundle()

    contribution = select_unavailable_reasons(bundle, panel_id="contribution")
    model_locus = select_unavailable_reasons(bundle, capability="model_locus_view")

    assert [reason.reason for reason in contribution] == ["missing_contribution_basis"]
    assert [reason.panel_id for reason in model_locus] == ["model_locus"]


def test_probe_episode_lens_adapter_defaults_and_exposes_honest_episode_seams():
    bundle = raw_layer_contribution_probe_bundle()
    selection = probe_episode_lens_adapter.default_selection(bundle)

    assert selection.episode_id == "episode-1"
    assert selection.policy_call == 3
    assert selection.ranking == "top"

    annotations = probe_episode_lens_adapter.pipeline_annotations(bundle, selection)
    channel_rows = probe_episode_lens_adapter.channel_ranking(bundle, selection)
    timeline_rows = probe_episode_lens_adapter.timeline_rows(bundle)

    assert {annotation.source for annotation in annotations} == {"model_locus", "contribution"}
    assert [row.key for row in channel_rows] == ["dim_42"]
    assert channel_rows[0].claim_level == "numeric_only"
    assert {row.source for row in timeline_rows} == {"ranked_moment", "prediction"}
    assert probe_episode_lens_adapter.intervention_seed(bundle, selection) is None


def test_probe_episode_lens_adapter_does_not_create_fake_channels_for_scalar_probe():
    bundle = scalar_timestep_probe_bundle()
    selection = probe_episode_lens_adapter.default_selection(bundle)

    assert selection.episode_id == "episode-1"
    assert selection.timestep == 7
    assert probe_episode_lens_adapter.channel_ranking(bundle, selection) == ()
    assert probe_episode_lens_adapter.pipeline_annotations(bundle, selection) == ()


def test_probe_episode_lens_adapter_preserves_origin_claims_for_multiple_contributions():
    bundle = raw_layer_contribution_probe_bundle()
    original = next(item for item in bundle.primitives if item.kind == "contribution")
    second = replace(
        original,
        basis="sae_feature",
        claim_level="human_labeled_feature",
        items=(
            ContributionItem(
                key="sae_99",
                value=0.99,
                rank=2,
                sign="positive",
                label="synthetic feature",
            ),
        ),
    )
    bundle = replace(bundle, primitives=(*bundle.primitives, second))

    rows = probe_episode_lens_adapter.channel_ranking(
        bundle,
        ResearchSelectionState(episode_id="episode-1", policy_call=3),
    )

    assert [(row.key, row.basis, row.claim_level) for row in rows] == [
        ("dim_42", "raw_activation_dimension", "numeric_only"),
        ("sae_99", "sae_feature", "human_labeled_feature"),
    ]


def test_probe_episode_lens_adapter_default_selection_preserves_fallback_ranking():
    bundle = scalar_timestep_probe_bundle()
    bundle = replace(
        bundle,
        primitives=tuple(
            primitive
            for primitive in bundle.primitives
            if not (primitive.kind == "ranked_moments" and primitive.ranking == "top")
        ),
    )

    selection = probe_episode_lens_adapter.default_selection(bundle)
    current = select_current_moment_evidence(bundle, selection)

    assert selection.ranking == "bottom"
    assert selection.episode_id == "episode-2"
    assert selection.timestep == 1
    assert [moment.episode_id for moment in current.ranked_moments] == ["episode-2"]
