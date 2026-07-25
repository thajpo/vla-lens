from __future__ import annotations

import json

import numpy as np

from vla_lens.interventions import (
    ActionArrayRef,
    CounterfactualPairManifest,
    CounterfactualRecipe,
    DonorSpec,
    PatchDecisionThresholds,
    PatchStudyArtifact,
    PatchStudySpec,
    PatchTrialManifest,
    PolicyCallRef,
    RecipientSpec,
    TraceRef,
    counterfactual_action_metrics,
    evaluate_patch_trial,
)


def _roundtrip(value, factory):
    return factory(json.loads(json.dumps(value.to_dict())))


def _pair() -> CounterfactualPairManifest:
    return CounterfactualPairManifest(
        pair_id="pair-caddy-mug-001",
        recipe=CounterfactualRecipe(
            kind="pose_exchange",
            target_object="desk_caddy_1",
            distractor_object="red_coffee_mug_1",
            changed_variables=(
                "desk_caddy_1.pose",
                "red_coffee_mug_1.pose",
            ),
            held_fixed={
                "instruction": "put the caddy in the basket",
                "camera": "main",
                "robot_state": "identical",
                "checkpoint": "openpi/pi05_libero",
            },
        ),
        recipient=RecipientSpec(
            trace=TraceRef(trace_id="recipient-trace", dataset_id="counterfactuals"),
            policy_call=PolicyCallRef(trace_id="recipient-trace", policy_call_index=0),
        ),
        donor=DonorSpec(
            trace=TraceRef(trace_id="donor-trace", dataset_id="counterfactuals"),
            policy_call=PolicyCallRef(trace_id="donor-trace", policy_call_index=0),
        ),
        compatibility={"model": True, "prompt": True, "action_shape": True},
        validation={"pair_valid": True},
    )


def _action_ref(role: str, sha: str) -> ActionArrayRef:
    return ActionArrayRef(
        array_ref=f"artifact://actions/{role}",
        role=role,
        shape=(50, 7),
        sha256=sha,
        coordinates={"action_dim": ["x", "y", "z", "rx", "ry", "rz", "gripper"]},
    )


def test_counterfactual_pair_and_study_artifact_roundtrip():
    pair = _pair()
    trial = PatchTrialManifest(
        trial_id="trial-layer-8-roi",
        pair_id=pair.pair_id,
        trial_kind="patched",
        action=_action_ref("patched", "abc123"),
        noise_ref="noise://shared/7",
        target={"model_site": "pi05.vlm.layers.8.prefix.hidden_tokens"},
        operation={"operator": "source_patch", "alpha": 1.0},
        token_indices=(11, 12, 13),
        token_mapping_sha256="tokens123",
        hook_calls=1,
    )
    spec = PatchStudySpec(
        study_id="study-pose-exchange-pilot",
        question="Does source patching transfer the donor action change?",
        hypothesis="An object-local layer should move the action toward the donor.",
        pair_ids=(pair.pair_id,),
        sites=({"layer": 8, "source": "residual"},),
        controls=("recipient_self_patch", "shuffled_tokens", "random_matched_norm"),
        shared_noise_refs=("noise://shared/7",),
    )
    artifact = PatchStudyArtifact(
        study=spec,
        pairs=(pair,),
        trials=(trial,),
        action_arrays=(trial.action,),
        decisions=(
            evaluate_patch_trial(
                counterfactual_action_metrics(
                    np.zeros((50, 7)),
                    np.ones((50, 7)),
                    np.full((50, 7), 0.5),
                ),
                controls=(),
            ),
        ),
        permanent_outputs=("pairs.parquet", "trials.parquet", "actions.zarr"),
        disposable_cache_refs=("cache://donor/layer8",),
    )

    assert _roundtrip(pair, CounterfactualPairManifest.from_dict) == pair
    assert _roundtrip(trial, PatchTrialManifest.from_dict) == trial
    assert _roundtrip(spec, PatchStudySpec.from_dict) == spec
    assert _roundtrip(artifact, PatchStudyArtifact.from_dict) == artifact
    assert artifact.action_arrays[0].dims == ("action_horizon", "action_dim")


def test_action_array_ref_requires_explicit_two_dimensional_shape():
    try:
        ActionArrayRef(array_ref="array://bad", role="patched", shape=(350,))
    except ValueError as error:
        assert "action_horizon" in str(error)
    else:
        raise AssertionError("flat action arrays must be rejected")


def test_counterfactual_metrics_report_direction_transfer_and_donor_gap():
    recipient = np.array([[0.0, 0.0], [0.0, 0.0]])
    donor = np.array([[2.0, 0.0], [0.0, 0.0]])
    patched = np.array([[1.0, 0.0], [0.0, 0.5]])

    metrics = counterfactual_action_metrics(recipient, donor, patched)

    assert metrics.natural_delta_norm == 2.0
    assert np.isclose(metrics.patch_delta_norm, np.sqrt(1.25))
    assert np.isclose(metrics.direction_agreement, 1 / np.sqrt(1.25))
    assert metrics.transfer_fraction == 0.5
    assert np.isclose(metrics.donor_gap_remaining, np.sqrt(1.25) / 2)
    assert np.isclose(metrics.donor_recovery, 1 - np.sqrt(1.25) / 2)
    assert metrics.off_direction_norm == 0.5


def test_patch_trial_verdicts_follow_predeclared_gates():
    thresholds = PatchDecisionThresholds(
        minimum_natural_delta_norm=0.1,
        minimum_direction_agreement=0.5,
        minimum_transfer_fraction=0.2,
        maximum_donor_gap_remaining=0.9,
        minimum_control_margin=0.1,
    )
    main = counterfactual_action_metrics(
        np.zeros((2, 2)), np.ones((2, 2)), np.full((2, 2), 0.6)
    )
    weak_control = counterfactual_action_metrics(
        np.zeros((2, 2)), np.ones((2, 2)), np.full((2, 2), 0.1)
    )
    strong_control = counterfactual_action_metrics(
        np.zeros((2, 2)), np.ones((2, 2)), np.full((2, 2), 0.55)
    )

    localized = evaluate_patch_trial(main, thresholds=thresholds)
    specific = evaluate_patch_trial(main, controls=(weak_control,), thresholds=thresholds)
    nonspecific = evaluate_patch_trial(main, controls=(strong_control,), thresholds=thresholds)

    assert localized.verdict == "localized_transfer"
    assert specific.verdict == "specific_action_transfer"
    assert specific.supports_specificity is True
    assert nonspecific.verdict == "nonspecific"
    assert evaluate_patch_trial(main, pair_valid=False).verdict == "pair_invalid"
    assert evaluate_patch_trial(main, replay_valid=False).verdict == "replay_invalid"
    assert evaluate_patch_trial(main, hook_valid=False).verdict == "hook_invalid"


def test_patch_trial_rejects_absent_natural_effect_before_transfer_claim():
    metrics = counterfactual_action_metrics(
        np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2))
    )

    decision = evaluate_patch_trial(metrics)

    assert decision.verdict == "natural_effect_absent"
    assert decision.supports_specificity is False
