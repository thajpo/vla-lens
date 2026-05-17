from __future__ import annotations

import json

import numpy as np

from vla_lens.pi05.token_metadata import (
    ACTION_SUFFIX_TOKEN_SPACE_ID,
    EXPERT_CONTEXT_TOKEN_SPACE_ID,
    PREFIX_TOKEN_SPACE_ID,
    action_normalization_metadata,
    build_pi05_token_metadata,
    image_prefix_token_layout,
    normalize_language_tokens,
)


class FakeTokenizer:
    def __call__(self, prompt: str):
        words = prompt.split()
        return {
            "input_ids": [[101, *range(200, 200 + len(words)), 102, 0]],
            "attention_mask": [[1, *([1] * len(words)), 1, 0]],
            "special_tokens_mask": [[1, *([0] * len(words)), 1, 1]],
        }

    def convert_ids_to_tokens(self, ids):
        return [f"tok_{item}" for item in ids]


def test_normalize_language_tokens_accepts_fake_tokenizer_and_masks():
    tokens = normalize_language_tokens("pick up cube", tokenizer=FakeTokenizer())

    assert tokens.input_ids == (101, 200, 201, 202, 102, 0)
    assert tokens.token_pieces == ("tok_101", "tok_200", "tok_201", "tok_202", "tok_102", "tok_0")
    assert tokens.attention_mask == (True, True, True, True, True, False)
    assert tokens.special_tokens_mask == (True, False, False, False, True, True)
    assert tokens.active_token_count == 5

    rows = tokens.records(start_index=10)
    assert rows[0]["token_index"] == 10
    assert rows[0]["token_id"] == 101
    assert rows[-1]["prefix_mask"] is False


def test_image_prefix_layout_keeps_masked_empty_camera_slot():
    layout = image_prefix_token_layout(
        [{"camera_id": "main"}, {"camera_id": "wrist"}],
        image_slots=3,
        patches_per_image=4,
        grid_shape=(2, 2),
        image_size=(224, 224),
    )

    assert layout.camera_order == ("main", "wrist", "empty_camera_0")
    assert layout.camera_present == (True, True, False)
    assert layout.camera_prefix_mask == (True, True, False)
    assert layout.token_count == 12

    records = layout.records()
    empty_records = [row for row in records if row["camera_id"] == "empty_camera_0"]
    assert len(empty_records) == 4
    assert {row["prefix_mask"] for row in empty_records} == {False}
    assert {row["is_empty_camera_slot"] for row in empty_records} == {True}
    assert empty_records[0]["token_index"] == 8
    assert empty_records[-1]["pixel_x1"] == 224


def test_build_pi05_token_metadata_tables_are_trace_ready():
    metadata = build_pi05_token_metadata(
        prompt="pick red mug",
        language={
            "input_ids": [11, 12, 0],
            "pieces": ["pick", " red", "<pad>"],
            "attention_mask": [1, 1, 0],
        },
        cameras=[{"camera_id": "main"}, {"camera_id": "wrist", "prefix_mask": False}],
        image_slots=3,
        patches_per_image=4,
        grid_shape=(2, 2),
        image_size=(224, 224),
        image_preprocessing={"resize_size": [224, 224], "mean": [0.5, 0.5, 0.5]},
        action=np.zeros((8, 7), dtype=np.float32),
        action_normalization={
            "q01": np.arange(7),
            "q99": np.arange(7) + 10,
            "mask": [1, 1, 1, 1, 1, 1, 0],
        },
        policy_call_index=3,
        observation_timestep=12,
        env_timestep_start=12,
        env_timestep_end=19,
    )

    assert set(metadata.streams["stream_id"]) == {
        "prefix",
        "language",
        "action_suffix",
        "expert_context",
        "image_main",
        "image_wrist",
        "image_empty_camera_0",
    }
    token_spaces = metadata.token_spaces.set_index("token_space_id")
    assert token_spaces.loc[PREFIX_TOKEN_SPACE_ID, "token_count"] == 15
    assert token_spaces.loc[ACTION_SUFFIX_TOKEN_SPACE_ID, "token_count"] == 8
    assert token_spaces.loc[EXPERT_CONTEXT_TOKEN_SPACE_ID, "token_count"] == 23
    expert_context = json.loads(token_spaces.loc[EXPERT_CONTEXT_TOKEN_SPACE_ID, "metadata"])
    assert expert_context["segments"] == [PREFIX_TOKEN_SPACE_ID, ACTION_SUFFIX_TOKEN_SPACE_ID]

    tokens = metadata.tokens
    language_rows = tokens.loc[tokens["token_kind"] == "language"]
    assert language_rows["token_index"].tolist() == [12, 13, 14]
    assert language_rows["token_id"].tolist() == [11, 12, 0]
    assert language_rows["prefix_mask"].tolist() == [True, True, False]

    action_rows = tokens.loc[tokens["token_kind"] == "action"]
    assert action_rows["token_type"].unique().tolist() == ["continuous_action"]
    assert action_rows["token_value_type"].unique().tolist() == ["continuous"]
    assert action_rows["token_id"].isna().all()
    assert action_rows["action_horizon_index"].tolist() == list(range(8))

    empty_rows = tokens.loc[tokens["camera_id"] == "empty_camera_0"]
    assert empty_rows["prefix_mask"].tolist() == [False, False, False, False]
    assert empty_rows["is_empty_camera_slot"].tolist() == [True, True, True, True]

    policy_call = metadata.policy_calls.iloc[0]
    assert policy_call["policy_call_index"] == 3
    assert policy_call["observation_timestep"] == 12
    policy_metadata = json.loads(policy_call["metadata"])
    assert policy_metadata["image_prefix"]["image_slots"] == 3
    assert policy_metadata["action_suffix"]["action_tokens_are_token_ids"] is False
    assert policy_metadata["action_normalization_metadata"]["mask"][-1] is False


def test_action_normalization_metadata_is_json_safe():
    metadata = action_normalization_metadata(
        {"q01": np.array([0.0, 1.0]), "q99": np.array([2.0, 3.0]), "mask": np.array([1, 0])},
        normalization_type="percentile",
        unnormalize_key="libero",
    )

    dumped = json.dumps(metadata)
    assert "percentile" in dumped
    assert metadata["mask"] == [True, False]
