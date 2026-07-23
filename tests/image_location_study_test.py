from __future__ import annotations

import numpy as np
import pandas as pd

import vla_lens.probes.image_location_study as image_location_study
from vla_lens.probes.image_location_study import (
    _box_iou,
    _box_patch_targets,
    _cached_object_boxes,
    _fit_box_decoder,
    _fit_patch_decoder,
    _patch_selection_score,
)


class _CacheDataset:
    def __init__(self, root):
        self.root = root

    def cache_dir(self):
        return self.root


def test_box_patch_targets_marks_only_intersecting_image_tokens():
    tokens = pd.DataFrame(
        {
            "pixel_x0": [0, 8, 0, 8],
            "pixel_x1": [8, 16, 8, 16],
            "pixel_y0": [0, 0, 8, 8],
            "pixel_y1": [8, 8, 16, 16],
        }
    )
    boxes = np.array([[[2, 2, 7, 7], [9, 1, 15, 7]]], dtype=np.float32)

    targets = _box_patch_targets(boxes, np.array([[True, True]]), tokens)

    np.testing.assert_array_equal(
        targets,
        np.array([[[True, False, False, False], [False, True, False, False]]]),
    )


def test_local_patch_decoder_keeps_tokens_separate():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(12, 4, 3))
    token_targets = np.stack(
        [
            2.0 * values[..., 0] - values[..., 1],
            -0.5 * values[..., 0] + 3.0 * values[..., 2],
        ],
        axis=1,
    )

    coefficients, intercepts, prediction = _fit_patch_decoder(
        values, token_targets, np.ones(12, dtype=bool), alpha=1e-9
    )

    assert coefficients.shape == (2, 3)
    assert intercepts.shape == (2,)
    np.testing.assert_allclose(prediction, token_targets, atol=2e-6)


def test_named_object_box_decoder_masks_visibility_and_unsupported_heads():
    rng = np.random.default_rng(7)
    values = rng.normal(size=(20, 3))
    targets = np.zeros((20, 2, 4), dtype=np.float64)
    weights = rng.normal(size=(4, 3))
    targets[:, 0] = values @ weights.T + np.array([0.1, 0.2, 0.3, 0.4])
    visible = np.ones((20, 2), dtype=bool)

    coefficients, intercepts, supported, prediction = _fit_box_decoder(
        values,
        targets,
        visible,
        np.ones(20, dtype=bool),
        np.array([True, False]),
        alpha=1e-9,
    )

    assert coefficients.shape == (2, 4, 3)
    assert intercepts.shape == (2, 4)
    np.testing.assert_array_equal(supported, [True, False])
    np.testing.assert_allclose(prediction[:, 0], targets[:, 0], atol=1e-7)
    assert np.isnan(prediction[:, 1]).all()


def test_box_iou_uses_area_overlap():
    assert _box_iou(np.array([0.0, 0.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0])) == 1.0
    assert np.isclose(
        _box_iou(
            np.array([0.0, 0.0, 0.5, 0.5]),
            np.array([0.25, 0.25, 0.75, 0.75]),
        ),
        1.0 / 7.0,
    )


def test_patch_candidate_selection_reads_validation_not_test_score():
    better_validation = {
        "selection_mean_average_precision": 0.6,
        "test_mean_average_precision": 0.1,
    }
    tempting_test = {
        "selection_mean_average_precision": 0.5,
        "test_mean_average_precision": 0.9,
    }

    assert _patch_selection_score(better_validation) > _patch_selection_score(tempting_test)


def test_object_box_cache_avoids_reopening_episode_arrays(tmp_path, monkeypatch):
    rows = pd.DataFrame({"trace_id": ["one"], "timestep": [0], "policy_call_index": [0]})
    tokens = pd.DataFrame(
        {
            "token_index": [0],
            "pixel_x0": [0],
            "pixel_x1": [16],
            "pixel_y0": [0],
            "pixel_y1": [16],
        }
    )
    expected_boxes = np.array([[[1, 2, 3, 4]]], dtype=np.float32)
    expected_visible = np.array([[True]])
    calls = 0

    def fake_boxes(*args, **kwargs):
        nonlocal calls
        calls += 1
        return expected_boxes, expected_visible

    monkeypatch.setattr(image_location_study, "_object_boxes", fake_boxes)
    arguments = (
        _CacheDataset(tmp_path),
        rows,
        tokens,
        np.array([[True]]),
        ["mug"],
    )

    first_boxes, first_visible, first_key, first_hit = _cached_object_boxes(
        *arguments, camera_name="agentview"
    )
    second_boxes, second_visible, second_key, second_hit = _cached_object_boxes(
        *arguments, camera_name="agentview"
    )

    assert calls == 1
    assert not first_hit
    assert second_hit
    assert first_key == second_key
    np.testing.assert_array_equal(first_boxes, second_boxes)
    np.testing.assert_array_equal(first_visible, second_visible)
