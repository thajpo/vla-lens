"""Compatibility facade for attention server helpers."""

from __future__ import annotations

from vla_lens.server.attention_expert import (
    _action_vector_for_token,
    _expert_attention_for_token,
    _expert_attention_site_candidates,
    _expert_token_attention_payload,
    _expert_token_details_payload,
    _expert_token_model_sites_payload,
    _prompt_attention_payload,
)
from vla_lens.server.attention_features import (
    _activation_token_feature_vector,
    _activation_token_matrix,
    _patch_features_payload,
)
from vla_lens.server.attention_maps import (
    _attention_camera_layout,
    _attention_key_mass_from_trace,
    _attention_map_payload,
    _attention_site_matches,
    _camera_maps_from_trace_key_mass,
)
from vla_lens.server.attention_tokens import (
    _camera_patch_layout,
    _camera_patch_layout_from_record,
    _camera_patch_maps_from_token_rows,
    _clean_token_piece,
    _decode_paligemma_token,
    _display_token_piece,
    _image_attention_from_prefix_rows,
    _image_token_index_for_patch,
    _image_token_rows_for_site,
    _join_token_pieces,
    _not_captured_in_profile,
    _paligemma_tokenizer,
    _prompt_attention_from_prefix_rows,
    _token_count,
    _token_rows_for_space,
)

__all__ = [
    "_action_vector_for_token",
    "_activation_token_feature_vector",
    "_activation_token_matrix",
    "_attention_camera_layout",
    "_attention_key_mass_from_trace",
    "_attention_map_payload",
    "_attention_site_matches",
    "_camera_maps_from_trace_key_mass",
    "_camera_patch_layout",
    "_camera_patch_layout_from_record",
    "_camera_patch_maps_from_token_rows",
    "_clean_token_piece",
    "_decode_paligemma_token",
    "_display_token_piece",
    "_expert_attention_for_token",
    "_expert_attention_site_candidates",
    "_expert_token_attention_payload",
    "_expert_token_details_payload",
    "_expert_token_model_sites_payload",
    "_image_attention_from_prefix_rows",
    "_image_token_index_for_patch",
    "_image_token_rows_for_site",
    "_join_token_pieces",
    "_not_captured_in_profile",
    "_paligemma_tokenizer",
    "_patch_features_payload",
    "_prompt_attention_from_prefix_rows",
    "_prompt_attention_payload",
    "_token_count",
    "_token_rows_for_space",
]
