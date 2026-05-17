from __future__ import annotations

import numpy as np
import pytest

from vla_lens.pi05.full_capture import (
    IncompletePI05FullCaptureError,
    PerSiteCaptureBuffer,
    make_raw_model_site_spec,
    materialize_raw_model_site_specs,
    missing_pi05_full_sites,
    pi05_full_capture_status,
    pi05_full_site_declarations,
    required_pi05_full_site_names,
)


def test_pi05_full_declarations_include_required_raw_site_roles():
    declarations = pi05_full_site_declarations(vlm_layers=(0,), expert_layers=(0,))
    by_role = {declaration.role: declaration for declaration in declarations}
    names = {declaration.name for declaration in declarations}

    assert {"q", "k", "v"}.issubset(by_role)
    assert {"pre_mask_scores", "post_mask_logits", "attention_probs"}.issubset(by_role)
    assert "pi05.vlm.layers.0.attention.pre_mask_scores" in names
    assert "pi05.vlm.layers.0.attention.post_mask_logits" in names
    assert "pi05.expert.layers.0.attention.pre_mask_scores" in names
    assert "pi05.expert.layers.0.attention.post_mask_logits" in names
    assert {
        "attn_output_pre_o_proj",
        "o_proj",
        "residual_pre_attention",
        "residual_post_attention",
        "residual_pre_mlp",
        "residual_post_mlp",
        "input_embeddings",
        "attention_norm_output",
        "mlp_norm_output",
        "adarms_scale",
        "adarms_shift",
        "adarms_gate",
        "mlp_gate",
        "mlp_up",
        "mlp_intermediate",
        "mlp_down",
        "mlp_output",
        "action_head_input",
        "action_head_output",
        "attention_mask",
        "position_ids",
        "rope_cos",
        "rope_sin",
        "rope_metadata",
        "kv_cache_key",
        "kv_cache_value",
    }.issubset(by_role)


def test_pi05_full_declarations_have_segment_and_axis_metadata():
    declaration = next(
        item
        for item in pi05_full_site_declarations(vlm_layers=(0,), expert_layers=())
        if item.name == "pi05.vlm.layers.0.attention.pre_mask_scores"
    )
    spec = make_raw_model_site_spec(declaration, np.zeros((1, 2, 3, 3), dtype=np.float32))

    assert spec.materialization == "raw"
    assert spec.exactness == "exact"
    assert spec.family == "attention"
    assert spec.role == "pre_mask_scores"
    assert spec.segment == "vlm_prefix"
    assert tuple(spec.axes) == ("policy_call", "head", "query_token", "key_token")
    assert spec.metadata["site_family"] == "attention"
    assert spec.metadata["site_role"] == "pre_mask_scores"
    assert spec.metadata["site_segment"] == "vlm_prefix"
    assert spec.metadata["site_axes"] == ("policy_call", "head", "query_token", "key_token")


def test_expert_sites_are_generation_step_aligned():
    declaration = next(
        item
        for item in pi05_full_site_declarations(vlm_layers=(), expert_layers=(3,))
        if item.name == "pi05.expert.layers.3.attention.post_mask_logits"
    )

    assert declaration.layer == 3
    assert declaration.segment == "action_expert"
    assert declaration.token_space_id is None
    assert declaration.query_token_space_id == "pi05.action_suffix"
    assert declaration.key_token_space_id == "pi05.expert_context"
    assert declaration.axes == (
        "policy_call",
        "generation_step",
        "head",
        "query_token",
        "key_token",
    )


def test_per_site_capture_buffer_rejects_unknown_sites_and_requires_completeness():
    declarations = pi05_full_site_declarations(vlm_layers=(0,), expert_layers=())
    buffer = PerSiteCaptureBuffer(declarations)

    with pytest.raises(KeyError):
        buffer.capture("pi05.vlm.layers.99.attention.q", np.zeros((1,)))

    first = declarations[0]
    buffer.capture(first.name, np.zeros((1,), dtype=np.float32))

    assert first.name in buffer.captured_names
    assert not buffer.is_complete()
    with pytest.raises(IncompletePI05FullCaptureError):
        buffer.specs()

    partial_specs = buffer.specs(require_complete=False)
    assert len(partial_specs) == 1
    assert partial_specs[0].name == first.name
    assert partial_specs[0].materialization == "raw"
    assert partial_specs[0].exactness == "exact"


def test_materialize_raw_model_site_specs_can_require_or_allow_partial_capture():
    declarations = pi05_full_site_declarations(vlm_layers=(), expert_layers=())
    first = declarations[0]
    captures = {first.name: np.zeros((1,), dtype=np.float32)}

    with pytest.raises(IncompletePI05FullCaptureError):
        materialize_raw_model_site_specs(captures, declarations=declarations)

    specs = materialize_raw_model_site_specs(
        captures,
        declarations=declarations,
        require_complete=False,
    )

    assert [spec.name for spec in specs] == [first.name]
    assert specs[0].metadata["required_for_true_full"] is True


def test_full_status_enables_when_required_site_names_are_present():
    names = required_pi05_full_site_names(vlm_layers=(), expert_layers=())
    status = pi05_full_capture_status(
        names,
        declarations=pi05_full_site_declarations(vlm_layers=(), expert_layers=()),
    )

    assert status.complete is True
    assert status.implemented is True
    assert status.enabled is True
    assert missing_pi05_full_sites(names, vlm_layers=(), expert_layers=()) == ()
