from __future__ import annotations

from vla_lens.pi05.pose_exchange_study import build_pose_exchange_study_job


def _pair(pair_id: str, *, valid: bool = True) -> dict:
    trace_id = f"{pair_id}-recipient"
    return {
        "pair_id": pair_id,
        "recipient": {
            "trace": {"trace_id": trace_id},
            "policy_call": {"trace_id": trace_id, "policy_call_index": 0},
        },
        "validation": {"pair_valid": valid},
    }


def test_localization_job_uses_all_valid_pairs_without_repeated_controls():
    job = build_pose_exchange_study_job(
        {
            "schema_kind": "pairs",
            "schema_version": 1,
            "pairs": [_pair("a"), _pair("bad", valid=False)],
        },
        study_id="localize",
        phase="localization",
        layers=(0, 8),
        token_regions=("target", "both"),
    )

    assert job["study"]["pair_ids"] == ["a"]
    assert job["study"]["controls"] == []
    assert job["study"]["shared_noise_refs"] == ["a-recipient.flow_initial_noise[0]"]
    assert job["request_template"]["intervention"]["request"]["controls"] == []


def test_confirmation_job_adds_specificity_controls_and_wrong_regions():
    job = build_pose_exchange_study_job(
        {"pairs": [_pair("a"), _pair("b")]},
        study_id="confirm",
        phase="confirmation",
        pair_ids=("b",),
        layers=(12,),
        token_regions=("target",),
        control_seed=7,
    )

    assert job["study"]["pair_ids"] == ["b"]
    assert job["study"]["axes"]["wrong_region_by_region"] == {
        "target": "distractor"
    }
    assert job["study"]["axes"]["generation_steps"] == "all"
    controls = job["request_template"]["intervention"]["request"]["controls"]
    assert [control["kind"] for control in controls] == list(job["study"]["controls"])
    assert controls[2]["parameters"]["seed"] == 7


def test_expert_job_defaults_to_full_action_suffix_and_step_aligned_sites():
    job = build_pose_exchange_study_job(
        {"pairs": [_pair("a")]},
        study_id="expert-handoff",
        phase="confirmation",
        stream="expert_action",
        control_seed=11,
    )

    study = job["study"]
    assert study["axes"] == {
        "phase": "confirmation",
        "stream": "expert_action",
        "token_regions": ["action_all"],
        "generation_steps": "all",
    }
    assert [site["layer"] for site in study["sites"]] == [0, 4, 8, 12, 16, 17]
    assert study["sites"][-1]["model_site"] == (
        "pi05.expert.layers.17.by_step.hidden_tokens"
    )
    template = job["request_template"]
    assert template["target"]["token_space"] == "pi05.action_suffix"
    controls = template["intervention"]["request"]["controls"]
    assert [control["kind"] for control in controls] == [
        "recipient_self_patch",
        "donor_self_patch",
        "alpha_zero",
        "shuffled_donor",
        "random_matched_norm",
    ]


def test_expert_job_can_target_a_denoising_step_range_without_manual_json():
    job = build_pose_exchange_study_job(
        {"pairs": [_pair("a")]},
        study_id="expert-early-steps",
        phase="localization",
        stream="expert_action",
        layers=(17,),
        generation_steps={"start": 0, "end": 3},
    )

    selector = {"start": 0, "end": 3}
    assert job["study"]["axes"]["generation_steps"] == selector
    assert (
        job["request_template"]["intervention"]["request"]["schedule"][
            "generation_steps"
        ]
        == selector
    )
