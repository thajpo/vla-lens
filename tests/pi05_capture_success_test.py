from __future__ import annotations

import numpy as np

from vla_lens.pi05.capture import (
    PROFILE_EXPERT_LAYERS,
    PROFILE_VLM_LAYERS,
    CaptureCall,
    CapturePlan,
    EpisodeBuffer,
    _capture_design_metadata,
    _declared_pi05_sites,
    _episode_success,
    _model_arrays,
    _trace_id_for_seed,
    canonical_profile,
    parse_args,
    profile_dimensions,
)


def test_episode_success_uses_any_late_success_info() -> None:
    buffer = EpisodeBuffer(
        trace_id="trace",
        task_id=0,
        task_name="task",
        prompt="task",
        seed=0,
    )
    buffer.infos = [{"is_success": False}, {"is_success": False}, {"is_success": True}]
    buffer.rewards = [0.0, 0.0, 1.0]

    assert _episode_success(buffer) is True


def test_counterfactual_capture_args_make_variant_trace_id_and_metadata() -> None:
    args = parse_args(
        [
            "--capture-profile",
            "mechanistic_sampled",
            "--benchmark",
            "libero_goal",
            "--task-id",
            "1",
            "--capture-design",
            "paired_counterfactual",
            "--counterfactual-group-id",
            "group-1",
            "--counterfactual-role",
            "Clean Trace",
            "--counterfactual-type",
            "prompt_target_swap",
            "--paired-trace-id",
            "pi05_mechanistic_sampled_libero_goal_task1_seed42_corrupt",
            "--changed-fields",
            '["prompt.target_object"]',
            "--matched-fields",
            "benchmark,task_id,seed",
            "--target-object-id",
            "mug",
        ]
    )

    assert _trace_id_for_seed(args, 42) == (
        "pi05_mechanistic_sampled_libero_goal_task1_seed42_clean_trace"
    )
    metadata = _capture_design_metadata(args)
    assert metadata["capture_design"] == "paired_counterfactual"
    assert metadata["trace_variant"] == "clean_trace"
    assert metadata["counterfactual_group_id"] == "group-1"
    assert metadata["counterfactual_role"] == "clean_trace"
    assert metadata["changed_fields"] == ["prompt.target_object"]
    assert metadata["matched_fields"] == ["benchmark", "task_id", "seed"]
    assert metadata["counterfactual"]["target_object_id"] == "mug"


def test_episode_success_false_when_success_signal_never_passes() -> None:
    buffer = EpisodeBuffer(
        trace_id="trace",
        task_id=0,
        task_name="task",
        prompt="task",
        seed=0,
    )
    buffer.infos = [{"is_success": False}, {"is_success": False}]
    buffer.rewards = [0.0, 1.0]

    assert _episode_success(buffer) is False


def test_episode_success_falls_back_to_reward_without_success_signal() -> None:
    buffer = EpisodeBuffer(
        trace_id="trace",
        task_id=0,
        task_name="task",
        prompt="task",
        seed=0,
    )
    buffer.infos = [{}, {}]
    buffer.rewards = [0.0, 1.0]

    assert _episode_success(buffer) is True


def test_profile_aliases_resolve_to_canonical_capture_profiles() -> None:
    assert canonical_profile("representation") == "features"
    assert canonical_profile("mechanistic_light") == "mechanistic_sampled"
    assert canonical_profile("mechanistic_heavy") == "mechanistic_all"
    assert canonical_profile("full") == "audit_full"


def test_sampled_profiles_use_same_vlm_and_expert_layer_pairs() -> None:
    expected_layers = (0, 4, 8, 12, 17)

    for profile in ("mechanistic_sampled", "internals_sampled", "audit_sampled"):
        assert PROFILE_VLM_LAYERS[profile] == expected_layers
        assert PROFILE_EXPERT_LAYERS[profile] == expected_layers
        assert profile_dimensions(profile)["layer_coverage"] == {
            "vlm": "sampled_5",
            "expert": "sampled_5",
        }


def test_audit_sampled_profile_dimensions_are_distinct_from_internals_sampled() -> None:
    audit = profile_dimensions("audit_sampled")
    internals = profile_dimensions("internals_sampled")

    assert audit["families"]["internals"] == "sampled_audit"
    assert internals["families"]["internals"] == "selected_ops"
    assert audit["families"]["state_setup"] == "none"


def test_custom_plan_metadata_reflects_resolved_families() -> None:
    plan = CapturePlan(
        profile="custom",
        vlm_layers=(0, 4, 8, 12, 17),
        expert_layers=(0, 4, 8, 12, 17),
        vlm_hidden="tokens",
        vlm_attention="none",
        expert_hidden="tokens",
        expert_attention="none",
        storage_dtype="float16",
    )

    metadata = plan.to_metadata()

    assert metadata["profile_dimensions"] == {
        "layer_coverage": {"vlm": "sampled_5", "expert": "sampled_5"},
        "families": {
            "representations": "tokens",
            "attention": "none",
            "cache": "none",
            "action_head": "none",
            "internals": "none",
            "state_setup": "none",
        },
    }
    assert profile_dimensions("custom") == metadata["profile_dimensions"]


def test_mechanistic_sampled_declares_bridge_and_action_head_sites() -> None:
    plan = CapturePlan(
        profile="mechanistic_sampled",
        vlm_layers=(0, 4),
        expert_layers=(0,),
        vlm_hidden="tokens",
        vlm_attention="full",
        expert_hidden="tokens",
        expert_attention="full",
        storage_dtype="float16",
    )

    sites = set(_declared_pi05_sites(plan))

    assert "pi05.vlm.layers.0.kv_cache.key" in sites
    assert "pi05.vlm.layers.0.kv_cache.value" in sites
    assert "pi05.vlm.layers.4.kv_cache.key" in sites
    assert "pi05.expert.by_step.input_embeddings" in sites
    assert "pi05.action_head.input" in sites
    assert "pi05.action_head.output" in sites
    assert plan.to_metadata()["captures_bridge_sites"] is True


def test_audit_sampled_declares_circuit_boundaries_without_state_setup() -> None:
    plan = CapturePlan(
        profile="audit_sampled",
        vlm_layers=(0,),
        expert_layers=(0,),
        vlm_hidden="tokens",
        vlm_attention="full",
        expert_hidden="tokens",
        expert_attention="full",
        storage_dtype="float16",
    )

    sites = set(_declared_pi05_sites(plan))

    assert "pi05.vlm.layers.0.residual_pre_attention" in sites
    assert "pi05.vlm.layers.0.attention.pre_mask_scores" in sites
    assert "pi05.vlm.layers.0.attention.post_mask_logits" in sites
    assert "pi05.vlm.layers.0.attention.attn_output_pre_o_proj" in sites
    assert "pi05.vlm.layers.0.mlp.output" in sites
    assert "pi05.expert.layers.0.attention_adarms.scale" in sites
    assert "pi05.expert.layers.0.mlp_adarms.shift" in sites
    assert "pi05.expert.layers.0.residual_post_mlp" in sites
    assert "pi05.vlm.layers.0.kv_cache.key" in sites
    assert "pi05.vlm.layers.0.kv_cache.value" in sites
    assert "pi05.expert.by_step.input_embeddings" in sites
    assert "pi05.action_head.input" in sites
    assert "pi05.action_head.output" in sites

    assert "pi05.inputs.attention_mask" not in sites
    assert "pi05.inputs.rope.cos" not in sites
    assert "pi05.expert.by_step.position_ids" not in sites
    assert "pi05.expert.layers.0.kv_cache.key" not in sites


def test_mechanistic_sampled_materializes_bridge_and_action_head_model_sites() -> None:
    plan = CapturePlan(
        profile="mechanistic_sampled",
        vlm_layers=(0,),
        expert_layers=(0,),
        vlm_hidden="tokens",
        vlm_attention="full",
        expert_hidden="tokens",
        expert_attention="full",
        storage_dtype="float16",
    )
    buffer = EpisodeBuffer(
        trace_id="trace",
        task_id=0,
        task_name="task",
        prompt="task",
        seed=0,
    )
    buffer.calls = [
        CaptureCall(
            call_index=0,
            env_timestep=0,
            final_action_chunk=np.zeros((2, 7), dtype=np.float16),
            denoising_actions=np.zeros((3, 2, 7), dtype=np.float16),
            suffix_hidden=np.zeros((3, 2, 7), dtype=np.float16),
            vlm_kv_key_by_layer={0: np.zeros((2, 5, 4), dtype=np.float16)},
            vlm_kv_value_by_layer={0: np.ones((2, 5, 4), dtype=np.float16)},
            generation_input_embeddings=np.zeros((3, 2, 8), dtype=np.float16),
            action_head_input=np.ones((3, 2, 8), dtype=np.float16),
            action_head_output=np.ones((3, 2, 7), dtype=np.float16),
        )
    ]

    specs = {spec.name: spec for spec in _model_arrays(buffer, plan)}

    assert specs["pi05.vlm.layers.0.kv_cache.key"].axes == [
        "policy_call",
        "kv_head",
        "cached_token",
        "head_channel",
    ]
    assert specs["pi05.vlm.layers.0.kv_cache.key"].array.shape == (1, 2, 5, 4)
    assert specs["pi05.vlm.layers.0.kv_cache.key"].role == "kv_cache_key"
    assert specs["pi05.vlm.layers.0.kv_cache.value"].role == "kv_cache_value"
    assert specs["pi05.expert.by_step.input_embeddings"].axes == [
        "policy_call",
        "generation_step",
        "token",
        "channel",
    ]
    assert specs["pi05.action_head.input"].axes == [
        "policy_call",
        "generation_step",
        "token",
        "channel",
    ]
    assert specs["pi05.action_head.output"].axes == [
        "policy_call",
        "generation_step",
        "horizon",
        "action_dim",
    ]
    assert specs["pi05.action_head.output"].materialization == "raw"
    assert specs["pi05.action_head.output"].exactness == "exact"


def test_audit_sampled_materializes_boundary_sites_with_coordinate_metadata() -> None:
    plan = CapturePlan(
        profile="audit_sampled",
        vlm_layers=(0,),
        expert_layers=(0,),
        vlm_hidden="tokens",
        vlm_attention="full",
        expert_hidden="tokens",
        expert_attention="full",
        storage_dtype="float16",
    )
    buffer = EpisodeBuffer(
        trace_id="trace",
        task_id=0,
        task_name="task",
        prompt="task",
        seed=0,
    )
    buffer.calls = [
        CaptureCall(
            call_index=0,
            env_timestep=0,
            final_action_chunk=np.zeros((2, 7), dtype=np.float16),
            denoising_actions=np.zeros((3, 2, 7), dtype=np.float16),
            suffix_hidden=np.zeros((3, 2, 7), dtype=np.float16),
            full_site_arrays={
                "pi05.vlm.layers.0.residual_pre_attention": np.zeros((5, 8), dtype=np.float16),
                "pi05.vlm.layers.0.attention.q": np.zeros((1, 5, 4), dtype=np.float16),
                "pi05.vlm.layers.0.attention.post_mask_logits": np.zeros(
                    (1, 5, 5), dtype=np.float16
                ),
                "pi05.vlm.layers.0.attention.attn_output_pre_o_proj": np.zeros(
                    (5, 8), dtype=np.float16
                ),
                "pi05.vlm.layers.0.mlp.output": np.zeros((5, 8), dtype=np.float16),
                "pi05.inputs.rope.cos": np.zeros((5, 4), dtype=np.float16),
                "pi05.expert.layers.0.attention_adarms.scale": np.zeros(
                    (3, 2, 8), dtype=np.float16
                ),
                "pi05.expert.layers.0.residual_post_mlp": np.zeros((3, 2, 8), dtype=np.float16),
                "pi05.expert.layers.0.kv_cache.key": np.zeros((3, 1, 2, 4), dtype=np.float16),
            },
        )
    ]

    specs = {spec.name: spec for spec in _model_arrays(buffer, plan)}

    assert "pi05.vlm.layers.0.residual_pre_attention" in specs
    assert "pi05.vlm.layers.0.attention.post_mask_logits" in specs
    assert "pi05.vlm.layers.0.attention.attn_output_pre_o_proj" in specs
    assert "pi05.vlm.layers.0.mlp.output" in specs
    assert "pi05.expert.layers.0.attention_adarms.scale" in specs
    assert "pi05.inputs.rope.cos" not in specs
    assert "pi05.expert.layers.0.kv_cache.key" not in specs

    logits = specs["pi05.vlm.layers.0.attention.post_mask_logits"]
    assert logits.metadata["capture_profile"] == "audit_sampled"
    assert logits.metadata["required_for_audit_sampled"] is True
    assert logits.metadata["coordinate_system_version"] == "pi05_attention_v1"
    assert logits.metadata["q_state"] == "post_rope"
    assert logits.metadata["k_state"] == "post_rope_pre_repeat_kv"
    assert logits.metadata["formula"] == "pre_mask_scores + additive_attention_mask"


def test_token_profiles_do_not_store_redundant_hidden_mean_or_attention_key_mass() -> None:
    plan = CapturePlan(
        profile="mechanistic_sampled",
        vlm_layers=(0,),
        expert_layers=(0,),
        vlm_hidden="tokens",
        vlm_attention="full",
        expert_hidden="tokens",
        expert_attention="full",
        storage_dtype="float16",
    )
    buffer = EpisodeBuffer(
        trace_id="trace",
        task_id=0,
        task_name="task",
        prompt="task",
        seed=0,
    )
    buffer.calls = [
        CaptureCall(
            call_index=0,
            env_timestep=0,
            final_action_chunk=np.zeros((2, 7), dtype=np.float16),
            denoising_actions=np.zeros((3, 2, 7), dtype=np.float16),
            suffix_hidden=np.zeros((3, 2, 7), dtype=np.float16),
            expert_hidden_by_layer={0: np.zeros((3, 2, 8), dtype=np.float16)},
            expert_attention_by_layer={0: np.zeros((3, 1, 2, 6), dtype=np.float16)},
        )
    ]

    names = {spec.name for spec in _model_arrays(buffer, plan)}

    assert "pi05.expert.layers.0.by_step.hidden_tokens" in names
    assert "pi05.expert.layers.0.by_step.hidden_mean" not in names
    assert "pi05.expert.layers.0.by_step.attention" in names
    assert "pi05.expert.by_step.attention_key_mass" not in names
    attention = {spec.name: spec for spec in _model_arrays(buffer, plan)}[
        "pi05.expert.layers.0.by_step.attention"
    ]
    assert attention.query_token_space_id == "pi05.action_suffix"
    assert attention.key_token_space_id == "pi05.expert_context"
