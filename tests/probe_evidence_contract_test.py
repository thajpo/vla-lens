from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from vla_lens.probe_evidence import (
    ContributionEvidence,
    ContributionItem,
    EvidenceCohortRef,
    FailureCaseEvidence,
    LensGeometry,
    PanelSpec,
    PredictionEvidence,
    ProbeEvidenceBundle,
    ProbeEvidenceContractError,
    ProbeLensArtifact,
    ResearchSelectionState,
    attention_head_grouped_probe_bundle,
    default_probe_panel_specs,
    pooled_no_contribution_probe_bundle,
    primitive_kinds,
    ranked_moments,
    raw_layer_contribution_probe_bundle,
    sae_feature_contribution_probe_bundle,
    scalar_timestep_probe_bundle,
    select_available_panels,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "probe_evidence"


def availability_by_panel(bundle: ProbeEvidenceBundle) -> dict[str, object]:
    panels = select_available_panels(bundle, default_probe_panel_specs())
    return {item.panel_id: item for item in panels}


def test_scalar_timestep_probe_exposes_core_evidence_without_fake_panels():
    bundle = scalar_timestep_probe_bundle()

    assert bundle.family == "probe"
    assert bundle.geometry.temporal_scope == "timestep"
    assert bundle.geometry.input_basis == "pooled_layer_activation"
    assert primitive_kinds(bundle) == {"provenance", "score_series", "ranked_moments", "prediction"}
    assert ranked_moments(bundle, "top")[0].timestep == 7

    panels = availability_by_panel(bundle)
    assert panels["probe_provenance"].available is True
    assert panels["score_series"].available is True
    assert panels["ranked_moments"].available is True
    assert panels["prediction"].available is True
    assert panels["contribution"].available is False
    assert panels["contribution"].reason == "missing_contribution_basis"
    assert "scores but not decomposed inputs" in str(panels["contribution"].message)
    assert panels["model_locus"].available is False
    assert panels["model_locus"].reason == "pooled_representation"


def test_pooled_no_contribution_fixture_is_explicit_alias_for_scalar_timestep():
    pooled = pooled_no_contribution_probe_bundle()
    scalar = scalar_timestep_probe_bundle()

    assert pooled.to_dict() == scalar.to_dict()
    assert pooled.geometry.input_basis == "pooled_layer_activation"


def test_raw_layer_contribution_is_numeric_only_not_semantic_feature_claim():
    bundle = raw_layer_contribution_probe_bundle()

    panels = availability_by_panel(bundle)
    assert panels["contribution"].available is True
    assert panels["model_locus"].available is True

    contribution = next(item for item in bundle.primitives if item.kind == "contribution")
    assert contribution.basis == "raw_activation_dimension"
    assert contribution.claim_level == "numeric_only"
    assert contribution.items[0].model_locus.channel_index == 42


def test_feature_and_head_fixtures_encode_stronger_but_distinct_claim_levels_and_panels():
    sae = sae_feature_contribution_probe_bundle()
    head = attention_head_grouped_probe_bundle()

    sae_contribution = next(item for item in sae.primitives if item.kind == "contribution")
    head_contribution = next(item for item in head.primitives if item.kind == "contribution")

    assert sae.geometry.input_basis == "sae_feature"
    assert sae_contribution.basis == "sae_feature"
    assert sae_contribution.claim_level == "human_labeled_feature"
    assert sae_contribution.items[0].label == "contact-like feature"

    assert head.geometry.input_basis == "attention_head_output"
    assert head_contribution.basis == "attention_head_output"
    assert head_contribution.claim_level == "grouped_model_locus"
    assert head_contribution.items[0].label == "Layer 6 head 3"

    for bundle in (sae, head):
        panels = availability_by_panel(bundle)
        assert panels["contribution"].available is True
        assert panels["model_locus"].available is True


def test_bundle_rejects_capabilities_not_declared_by_geometry():
    base = raw_layer_contribution_probe_bundle()
    artifact = base.artifact
    run = base.run
    geometry = LensGeometry(
        temporal_scope="episode",
        output_kind="scalar",
        input_basis="pooled_layer_activation",
        locus_kind="none",
        capabilities=("score_series",),
    )

    with pytest.raises(ProbeEvidenceContractError, match="bundle capabilities"):
        ProbeEvidenceBundle(
            bundle_id="bundle-x",
            artifact=artifact,
            run=run,
            geometry=geometry,
            capabilities=("score_series", "model_locus_view"),
        )


def test_bundle_rejects_primitives_from_another_lens_or_run():
    base = raw_layer_contribution_probe_bundle()
    primitive = next(item for item in base.primitives if item.kind == "score_series")
    mismatched_lens = replace(primitive, lens_id="other-probe")

    with pytest.raises(ProbeEvidenceContractError, match="primitive lens_id"):
        ProbeEvidenceBundle(
            bundle_id=base.bundle_id,
            artifact=base.artifact,
            run=base.run,
            geometry=base.geometry,
            capabilities=base.capabilities,
            primitives=(mismatched_lens,),
        )

    mismatched_run = replace(primitive, lens_run_id="other-run")

    with pytest.raises(ProbeEvidenceContractError, match="primitive lens_run_id"):
        ProbeEvidenceBundle(
            bundle_id=base.bundle_id,
            artifact=base.artifact,
            run=base.run,
            geometry=base.geometry,
            capabilities=base.capabilities,
            primitives=(mismatched_run,),
        )


def test_bundle_rejects_primitives_without_required_capability():
    base = raw_layer_contribution_probe_bundle()
    contribution = next(item for item in base.primitives if item.kind == "contribution")
    geometry = LensGeometry(
        temporal_scope="policy_call",
        output_kind="scalar",
        input_basis="layer_activation",
        locus_kind="model_locus",
        capabilities=("model_locus_view",),
    )

    with pytest.raises(ProbeEvidenceContractError, match="requires capability"):
        ProbeEvidenceBundle(
            bundle_id=base.bundle_id,
            artifact=base.artifact,
            run=base.run,
            geometry=geometry,
            capabilities=geometry.capabilities,
            primitives=(contribution,),
        )


def test_literal_validation_rejects_invalid_contract_values():
    base = raw_layer_contribution_probe_bundle()

    with pytest.raises(ProbeEvidenceContractError, match="basis"):
        ContributionEvidence(
            kind="contribution",
            lens_id=base.artifact.lens_id,
            lens_run_id=base.run.lens_run_id,
            episode_id="episode-1",
            basis="made_up_basis",
            claim_level="numeric_only",
        )

    with pytest.raises(ProbeEvidenceContractError, match="sign"):
        ContributionItem(key="dim_1", value=0.1, rank=1, sign="up")

    with pytest.raises(ProbeEvidenceContractError, match="split"):
        PredictionEvidence(
            kind="prediction",
            lens_id=base.artifact.lens_id,
            lens_run_id=base.run.lens_run_id,
            episode_id="episode-1",
            prediction=True,
            split="holdout",
        )

    with pytest.raises(ProbeEvidenceContractError, match="source"):
        EvidenceCohortRef(
            cohort_id="cohort-1",
            source="query",
            selection=ResearchSelectionState(dataset_id="demo"),
            count=1,
        )

    with pytest.raises(ProbeEvidenceContractError, match="ranking"):
        FailureCaseEvidence(
            kind="failure_case",
            lens_id=base.artifact.lens_id,
            lens_run_id=base.run.lens_run_id,
            ranking="maybe_wrong",
        )


def test_missing_labels_disable_failure_panel_with_precise_reason():
    bundle = scalar_timestep_probe_bundle()
    failure_panel = PanelSpec(
        panel_id="failure_cases",
        consumes=("failure_case",),
        requires_capabilities=("failure_cases",),
        unavailable_copy="Failure cases are unavailable.",
    )

    availability = select_available_panels(bundle, (failure_panel,))[0]

    assert availability.available is False
    assert availability.reason == "missing_labels"
    assert "no labels or proxy targets" in str(availability.message)


def test_geometry_mismatch_blocks_panel_even_when_capability_and_primitive_exist():
    bundle = raw_layer_contribution_probe_bundle()
    sae_only_model_panel = PanelSpec(
        panel_id="sae_only_model_locus",
        consumes=("model_locus",),
        requires_capabilities=("model_locus_view",),
        requires_geometry={"input_basis": "sae_feature"},
        unavailable_copy="SAE-only model locus panel unavailable.",
    )

    availability = select_available_panels(bundle, (sae_only_model_panel,))[0]

    assert availability.available is False
    assert "geometry input_basis" in str(availability.reason)


def test_shared_json_fixtures_match_python_fixture_builders():
    expected = {
        "scalar_timestep.json": scalar_timestep_probe_bundle().to_dict(),
        "pooled_no_contribution.json": pooled_no_contribution_probe_bundle().to_dict(),
        "raw_layer_contribution.json": raw_layer_contribution_probe_bundle().to_dict(),
        "sae_feature_contribution.json": sae_feature_contribution_probe_bundle().to_dict(),
        "attention_head_grouped.json": attention_head_grouped_probe_bundle().to_dict(),
    }

    for filename, payload in expected.items():
        with (FIXTURE_DIR / filename).open() as handle:
            assert json.load(handle) == payload


def test_bundle_rejects_non_probe_artifacts():
    with pytest.raises(ProbeEvidenceContractError, match="lens_type='probe'"):
        ProbeLensArtifact(
            lens_id="sae-x",
            lens_version="v1",
            name="SAE X",
            lens_type="sae",
        )
