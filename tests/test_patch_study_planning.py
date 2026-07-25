from __future__ import annotations

from vla_lens.interventions import (
    CounterfactualPairManifest,
    CounterfactualRecipe,
    DonorSpec,
    PatchStudySpec,
    PolicyCallRef,
    RecipientSpec,
    TraceRef,
    build_patch_trial_request,
    expand_patch_study,
)


def _pair(pair_id: str = "pair-1") -> CounterfactualPairManifest:
    return CounterfactualPairManifest(
        pair_id=pair_id,
        recipe=CounterfactualRecipe(
            kind="pose_exchange",
            target_object="caddy",
            distractor_object="mug",
            changed_variables=("caddy.pose", "mug.pose"),
            held_fixed={"prompt": True, "robot": True},
        ),
        recipient=RecipientSpec(
            trace=TraceRef(trace_id=f"{pair_id}-recipient"),
            policy_call=PolicyCallRef(
                trace_id=f"{pair_id}-recipient", policy_call_index=2
            ),
        ),
        donor=DonorSpec(
            trace=TraceRef(trace_id=f"{pair_id}-donor"),
            policy_call=PolicyCallRef(trace_id=f"{pair_id}-donor", policy_call_index=2),
        ),
        validation={
            "token_regions": {
                "target": {"recipient": [1, 2], "donor": [3, 4]},
                "distractor": {"recipient": [5, 6], "donor": [7, 8]},
            }
        },
    )


def _study() -> PatchStudySpec:
    return PatchStudySpec(
        study_id="rq020-pilot",
        question="Where does patching transfer the donor action?",
        hypothesis="An object-local layer moves the action toward the donor.",
        pair_ids=("pair-1",),
        sites=(
            {"layer": 0},
            {
                "layer": 4,
                "model_site": "pi05.vlm.layers.{layer}.prefix.hidden_tokens",
            },
        ),
        controls=("recipient_self_patch", "wrong_region", "random_matched_norm"),
        shared_noise_refs=("recipient.flow_initial_noise[2]",),
        axes={
            "token_regions": ["target", "distractor"],
            "wrong_region_by_region": {
                "target": "distractor",
                "distractor": "target",
            },
        },
    )


def _template() -> dict:
    return {
        "runtime_adapter": "pi05",
        "target": {
            "kind": "activation_slice",
            "model_family": "pi05",
            "model_site": "placeholder",
            "token_space": "pi05.prefix",
        },
        "intervention": {
            "request": {
                "operator": {
                    "operator": "source_patch",
                    "strength": 1.0,
                    "parameters": {"mode": "donor_source_patch"},
                },
                "schedule": {
                    "policy_calls": [2],
                    "generation_steps": "all",
                    "tokens": "target_tokens",
                },
                "controls": [
                    {"kind": "random_matched_norm", "parameters": {"seed": 7}}
                ],
                "outcome": {"kind": "action", "basis": ["raw"]},
            }
        },
    }


def test_patch_study_expansion_is_deterministic_and_resolves_pair_tokens():
    first = expand_patch_study(_study(), (_pair(),))
    second = expand_patch_study(_study(), (_pair(),))

    assert first == second
    assert len(first) == 4
    assert [(trial.layer, trial.token_region) for trial in first] == [
        (0, "target"),
        (0, "distractor"),
        (4, "target"),
        (4, "distractor"),
    ]
    assert first[0].recipient_token_indices == (1, 2)
    assert first[0].donor_token_indices == (3, 4)
    assert first[0].wrong_recipient_token_indices == (5, 6)


def test_patch_trial_request_contains_every_runtime_and_reconstruction_choice():
    study = _study()
    pair = _pair()
    trial = expand_patch_study(study, (pair,))[2]

    request = build_patch_trial_request(
        _template(), study=study, pair=pair, trial=trial
    )

    assert request["run_id"] == trial.trial_id
    assert request["baseline"]["context"] == {
        "trace_id": "pair-1-recipient",
        "policy_call_index": 2,
    }
    assert request["donor"]["trace"]["trace_id"] == "pair-1-donor"
    assert request["target"]["layer"] == 4
    assert request["target"]["token_selector"]["indices"] == [1, 2]
    nested = request["intervention"]["request"]
    assert nested["operator"]["parameters"]["donor_token_indices"] == [3, 4]
    assert [control["kind"] for control in nested["controls"]] == [
        "recipient_self_patch",
        "wrong_region",
        "random_matched_norm",
    ]
    wrong = nested["controls"][1]["parameters"]
    assert wrong == {"recipient_indices": [5, 6], "donor_indices": [7, 8]}
    assert nested["controls"][2]["parameters"]["seed"] == 7
    assert (
        nested["outcome"]["parameters"]["patch_decision_thresholds"]
        == study.thresholds.to_dict()
    )
