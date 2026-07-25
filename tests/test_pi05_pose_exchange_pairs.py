from __future__ import annotations

import numpy as np
import pandas as pd

from vla_lens.pi05.pose_exchange_pairs import (
    _background_tokens,
    _broad_prefix_regions,
    _disjoint_tokens_for_bboxes,
    _tokens_for_bbox,
)


def _tokens() -> pd.DataFrame:
    rows = []
    index = 0
    for row in range(4):
        for col in range(4):
            rows.append(
                {
                    "token_index": index,
                    "pixel_x0": col * 16,
                    "pixel_x1": (col + 1) * 16,
                    "pixel_y0": row * 16,
                    "pixel_y1": (row + 1) * 16,
                }
            )
            index += 1
    return pd.DataFrame(rows)


def test_object_tokens_prioritize_bbox_overlap_then_nearest_patches():
    selected = _tokens_for_bbox(
        _tokens(), np.array([16.0, 16.0, 32.0, 32.0]), count=4
    )

    assert selected[0] == 5
    assert len(selected) == 4
    assert len(set(selected)) == 4


def test_background_tokens_are_outside_and_far_from_both_objects():
    selected = _background_tokens(
        _tokens(),
        (
            np.array([0.0, 0.0, 16.0, 16.0]),
            np.array([48.0, 48.0, 64.0, 64.0]),
        ),
        count=4,
    )

    assert len(selected) == 4
    assert 0 not in selected
    assert 15 not in selected


def test_nearby_objects_receive_disjoint_balanced_token_regions():
    first, second = _disjoint_tokens_for_bboxes(
        _tokens(),
        np.array([16.0, 16.0, 32.0, 32.0]),
        np.array([20.0, 16.0, 36.0, 32.0]),
        count=6,
    )

    assert len(first) == len(second) == 6
    assert set(first).isdisjoint(second)
    assert first[0] == 5


def test_broad_prefix_regions_separate_cameras_language_and_full_prefix():
    tokens = pd.DataFrame(
        [
            {
                "policy_call_index": 0,
                "token_index": 0,
                "token_space_id": "pi05.prefix",
                "token_kind": "image",
                "camera_id": "main",
                "prefix_mask": True,
                "attention_mask": True,
            },
            {
                "policy_call_index": 0,
                "token_index": 1,
                "token_space_id": "pi05.prefix",
                "token_kind": "image",
                "camera_id": "wrist",
                "prefix_mask": True,
                "attention_mask": True,
            },
            {
                "policy_call_index": 0,
                "token_index": 2,
                "token_space_id": "pi05.prefix",
                "token_kind": "image",
                "camera_id": "empty_camera_0",
                "prefix_mask": False,
                "attention_mask": False,
            },
            {
                "policy_call_index": 0,
                "token_index": 3,
                "token_space_id": "pi05.prefix",
                "token_kind": "language",
                "camera_id": None,
                "prefix_mask": True,
                "attention_mask": True,
            },
            {
                "policy_call_index": 0,
                "token_index": 4,
                "token_space_id": "pi05.prefix",
                "token_kind": "language",
                "camera_id": None,
                "prefix_mask": False,
                "attention_mask": False,
            },
        ]
    )

    regions = _broad_prefix_regions(tokens, tokens)

    assert regions["main_camera"]["recipient"] == [0]
    assert regions["wrist_camera"]["recipient"] == [1]
    assert regions["active_images"]["recipient"] == [0, 1]
    assert regions["language_active"]["recipient"] == [3]
    assert regions["full_prefix"]["recipient"] == [0, 1, 2, 3, 4]
