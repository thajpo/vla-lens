from __future__ import annotations

import json

import pytest

from vla_lens.artifacts import LensArtifact
from vla_lens.interventions import (
    ArtifactFamilyContract,
    ControlSpec,
    InterventionOperatorSpec,
    OutcomeSpec,
    artifact_family_for_type,
    artifact_family_registry,
    legal_operators_for_artifact,
    legal_outcomes_for_artifact,
    required_controls_for_artifact_claim,
    target_from_discovery_artifact,
)


def test_sae_feature_artifact_maps_to_feature_target_contract():
    artifact = LensArtifact.create(
        artifact_type="sae_feature",
        name="gripper close SAE feature",
        selector={
            "target": {
                "model_site": "pi05.expert.layers.8.mlp",
                "layer": 8,
                "tensor_type": "mlp_activation",
                "token_space": "pi05.expert_context",
                "feature_index": 42,
            }
        },
        arrays={"decoder_vector": "arrays/sae_feature_42_decoder.npy"},
    )

    target = target_from_discovery_artifact(artifact)

    assert target.kind == "feature"
    assert target.source_artifact_id == artifact.artifact_id
    assert target.source_artifact_type == "sae_feature"
    assert target.model_site == "pi05.expert.layers.8.mlp"
    assert target.layer == 8
    assert target.representation["kind"] == "feature_index"
    assert target.representation["feature_index"] == 42
    assert target.representation["array_ref"] == "arrays/sae_feature_42_decoder.npy"
    assert "feature_clamp" in target.metadata["legal_operators"]
    assert "action" in target.metadata["legal_outcomes"]


def test_attention_alias_artifact_maps_to_edge_target_contract():
    artifact = LensArtifact.create(
        artifact_type="attribution_map",
        name="action token attention to object",
        selector={
            "target": {
                "model_site": "pi05.expert.layers.4.attention",
                "tensor_type": "attention_probs",
                "token_selector": {"query": "action_suffix", "key": "image_patch"},
            },
            "edge": {"query_token": 3, "key_token": 12, "head": 1},
        },
    )

    target = target_from_discovery_artifact(artifact, token_space="pi05.expert_context")

    assert target.kind == "edge"
    assert target.source_artifact_type == "attribution_map"
    assert target.metadata["artifact_family"] == "attention_map"
    assert target.token_space == "pi05.expert_context"
    assert target.token_selector == {"query": "action_suffix", "key": "image_patch"}
    assert target.representation["kind"] == "attention_edge"
    assert target.representation["edge"] == {"query_token": 3, "key_token": 12, "head": 1}
    assert legal_operators_for_artifact("attribution_map") == ("attention_patch", "head_ablate")
    assert "attention" in legal_outcomes_for_artifact("attribution_map")


def test_future_family_contracts_reuse_core_validators():
    contracts = artifact_family_registry()

    assert {contract.artifact_type for contract in contracts} >= {
        "sae_feature",
        "transcoder_feature",
        "crosscoder_feature",
        "attention_map",
    }
    for contract in contracts:
        for operator in contract.operators:
            InterventionOperatorSpec(operator=operator)
        for outcome in contract.outcomes:
            OutcomeSpec(kind=outcome)
        for claim_label in contract.required_controls:
            controls = required_controls_for_artifact_claim(contract.artifact_type, claim_label)
            for control in controls:
                ControlSpec(kind=control)


def test_artifact_family_contract_roundtrip_and_unknown_family():
    contract = artifact_family_for_type("transcoder_feature")

    loaded = ArtifactFamilyContract.from_dict(json.loads(json.dumps(contract.to_dict())))

    assert loaded == contract
    assert loaded.target_kind == "path"
    assert "path_patch" in loaded.operators
    with pytest.raises(KeyError, match="Unknown intervention artifact family"):
        artifact_family_for_type("unregistered_future_artifact")
