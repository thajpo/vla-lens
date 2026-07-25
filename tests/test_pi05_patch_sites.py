from __future__ import annotations

from types import SimpleNamespace

import pytest

from vla_lens.pi05.patch_sites import (
    parse_pi05_patch_site,
    pi05_patch_module,
    pi05_runtime_patch_sites,
)


def test_patch_sites_name_vlm_and_step_aligned_expert_axes():
    vlm = parse_pi05_patch_site("pi05.vlm.layers.12.prefix.hidden_tokens")
    expert = parse_pi05_patch_site(
        "pi05.expert.layers.16.by_step.hidden_tokens",
        declared_layer=16,
    )

    assert vlm.stack == "vlm_prefix"
    assert vlm.token_space == "pi05.prefix"
    assert vlm.axes == ("token", "channel")
    assert vlm.repeated_by_generation_step is False
    assert expert.stack == "expert_action"
    assert expert.token_space == "pi05.action_suffix"
    assert expert.axes == ("generation_step", "token", "channel")
    assert expert.repeated_by_generation_step is True
    assert expert.to_runtime_record()["materialization"] == "runtime_only"


def test_patch_site_registry_and_module_resolution_cover_all_layers():
    sites = pi05_runtime_patch_sites()
    assert len(sites) == 36
    assert "pi05.vlm.layers.17.prefix.hidden_tokens" in sites
    assert "pi05.expert.layers.17.by_step.hidden_tokens" in sites

    vlm_layers = [object()]
    expert_layers = [object()]
    policy = SimpleNamespace(
        model=SimpleNamespace(
            paligemma_with_expert=SimpleNamespace(
                paligemma=SimpleNamespace(language_model=SimpleNamespace(layers=vlm_layers)),
                gemma_expert=SimpleNamespace(model=SimpleNamespace(layers=expert_layers)),
            )
        )
    )
    assert pi05_patch_module(
        policy, parse_pi05_patch_site("pi05.vlm.layers.0.prefix.hidden_tokens")
    ) is vlm_layers[0]
    assert pi05_patch_module(
        policy,
        parse_pi05_patch_site("pi05.expert.layers.0.by_step.hidden_tokens"),
    ) is expert_layers[0]


def test_patch_site_parser_rejects_ambiguous_or_mismatched_sites():
    with pytest.raises(ValueError, match="Unsupported"):
        parse_pi05_patch_site("pi05.expert.layers.0.hidden_tokens")
    with pytest.raises(ValueError, match="disagrees"):
        parse_pi05_patch_site(
            "pi05.expert.layers.4.by_step.hidden_tokens", declared_layer=8
        )
