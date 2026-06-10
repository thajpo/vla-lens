from __future__ import annotations

from dataclasses import replace

import pytest

from vla_lens.probe_evidence import (
    ContributionItem,
    FailureCaseEvidence,
    PanelSpec,
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

PANEL_EXPECTATIONS = {
    "scalar timestep": (
        scalar_timestep_probe_bundle,
        {
            "contribution": False,
            "failure_cases": False,
            "model_locus": False,
            "prediction": True,
            "probe_provenance": True,
            "ranked_moments": True,
            "score_series": True,
            "unavailable_reasons": True,
        },
    ),
    "pooled layer": (
        pooled_no_contribution_probe_bundle,
        {
            "contribution": False,
            "failure_cases": False,
            "model_locus": False,
            "prediction": True,
            "probe_provenance": True,
            "ranked_moments": True,
            "score_series": True,
            "unavailable_reasons": True,
        },
    ),
    "raw layer vector": (
        raw_layer_contribution_probe_bundle,
        {
            "contribution": True,
            "failure_cases": False,
            "model_locus": True,
            "prediction": True,
            "probe_provenance": True,
            "ranked_moments": True,
            "score_series": True,
            "unavailable_reasons": True,
        },
    ),
    "SAE feature": (
        sae_feature_contribution_probe_bundle,
        {
            "contribution": True,
            "failure_cases": False,
            "model_locus": True,
            "prediction": False,
            "probe_provenance": True,
            "ranked_moments": False,
            "score_series": False,
            "unavailable_reasons": True,
        },
    ),
    "attention head grouped": (
        attention_head_grouped_probe_bundle,
        {
            "contribution": True,
            "failure_cases": False,
            "model_locus": True,
            "prediction": False,
            "probe_provenance": True,
            "ranked_moments": False,
            "score_series": False,
            "unavailable_reasons": True,
        },
    ),
}


def test_select_available_panels_covers_all_golden_probe_fixtures():
    for name, (bundle_factory, expected) in PANEL_EXPECTATIONS.items():
        bundle = bundle_factory()
        panels = {
            panel.panel_id: panel
            for panel in select_available_panels(bundle, default_probe_panel_specs())
        }

        assert set(panels) == set(expected), name
        assert {panel_id: panels[panel_id].available for panel_id in panels} == expected, name


def test_select_available_panels_isolates_capability_gating_from_primitive_presence():
    bundle = scalar_timestep_probe_bundle()
    comparison_provenance = PanelSpec(
        panel_id="comparison_provenance",
        consumes=("provenance",),
        requires_capabilities=("comparison",),
        unavailable_copy="Comparison provenance is unavailable.",
    )
    score_backed_contribution = PanelSpec(
        panel_id="score_backed_contribution",
        consumes=("contribution",),
        requires_capabilities=("score_series",),
        unavailable_copy="Contribution is unavailable.",
    )

    missing_capability, missing_primitive = select_available_panels(
        bundle,
        (comparison_provenance, score_backed_contribution),
    )

    assert missing_capability.available is False
    assert missing_capability.reason == "missing capability: comparison"
    assert missing_primitive.available is False
    assert missing_primitive.reason == "missing evidence primitive: contribution"


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


def test_select_current_moment_evidence_narrows_failure_case_primitives():
    bundle = scalar_timestep_probe_bundle()
    bundle = replace(
        bundle,
        geometry=replace(
            bundle.geometry,
            capabilities=(*bundle.geometry.capabilities, "failure_cases"),
        ),
        capabilities=(*bundle.capabilities, "failure_cases"),
        primitives=(
            *bundle.primitives,
            FailureCaseEvidence(
                kind="failure_case",
                lens_id=bundle.artifact.lens_id,
                lens_run_id=bundle.run.lens_run_id,
                ranking="false_positive",
                moments=(
                    replace(select_top_moments(bundle)[0], episode_id="episode-1"),
                    replace(select_top_moments(bundle)[0], episode_id="episode-2"),
                ),
            ),
        ),
    )

    current = select_current_moment_evidence(
        bundle,
        ResearchSelectionState(
            episode_id="episode-1",
            ranking="false_positive",
            timestep=7,
        ),
    )

    assert [(moment.episode_id, moment.score) for moment in current.failure_moments] == [
        ("episode-1", 0.91)
    ]


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
    failure_cases = select_unavailable_reasons(bundle, panel_id="failure_cases")

    assert [reason.reason for reason in contribution] == ["missing_contribution_basis"]
    assert [reason.panel_id for reason in model_locus] == ["model_locus"]
    assert [reason.reason for reason in failure_cases] == ["missing_labels"]
    assert "no labels or proxy targets" in str(failure_cases[0].message)


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


def test_probe_evidence_interaction_contract_checklist_is_executable():
    bundle = raw_layer_contribution_probe_bundle()

    select_probe_panels = {
        panel.panel_id: panel
        for panel in select_available_panels(bundle, default_probe_panel_specs())
    }
    assert select_probe_panels["score_series"].available is True
    assert select_probe_panels["ranked_moments"].available is True
    assert select_probe_panels["model_locus"].available is True

    moment_selection = probe_episode_lens_adapter.default_selection(bundle)
    moment_evidence = select_current_moment_evidence(bundle, moment_selection)
    assert (
        moment_selection.dataset_id,
        moment_selection.lens_id,
        moment_selection.lens_run_id,
        moment_selection.episode_id,
        moment_selection.policy_call,
        moment_selection.ranking,
    ) == ("demo", "probe-grasp-intent", "run-probe-grasp-intent", "episode-1", 3, "top")
    assert moment_evidence.predictions[0].prediction is True
    assert moment_evidence.ranked_moments[0].score == 0.88

    source_annotations = probe_episode_lens_adapter.pipeline_annotations(
        bundle, moment_selection
    )
    assert source_annotations[0].model_locus.model_site_id == "action_head.layers.8.resid"

    contributor_selection = replace(moment_selection, feature_id="dim_42")
    rows = select_contribution_rows(bundle, contributor_selection)
    assert [(row.key, row.value, row.sign) for row in rows] == [
        ("dim_42", 0.42, "positive")
    ]
    assert select_contribution_claim_level(bundle, contributor_selection) == "numeric_only"
