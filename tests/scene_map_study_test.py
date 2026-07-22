from __future__ import annotations

import numpy as np
import pandas as pd

from vla_lens.probes.scene_map_study import (
    SceneMapTargets,
    _fit_masked_positions,
    _identity_metrics,
    _position_metrics,
    _project,
)


def test_identity_metrics_score_complete_scenes_and_keep_unseen_failures_visible():
    rows = pd.DataFrame(
        {
            "trace_id": ["train", "val", "test"],
        }
    )
    truth = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
    )
    scores = truth.copy()
    supported = np.array([True, True, False])
    masks = {
        "train": np.array([True, False, False]),
        "selection": np.array([False, True, False]),
        "test": np.array([False, False, True]),
    }

    metrics = _identity_metrics(truth, scores, rows, masks, supported)

    assert metrics["test_scene_jaccard"] == 1.0
    assert metrics["test_full_scene_jaccard"] == 0.5
    assert metrics["test_full_exact_scene_rate"] == 0.0
    assert metrics["test_unseen_positive_count"] == 1
    assert metrics["test_supported_objects"] == 2


def test_masked_location_heads_fit_each_object_only_where_it_exists():
    rows = pd.DataFrame({"trace_id": ["a", "b", "c", "d"]})
    design = np.arange(4, dtype=np.float64)[:, None]
    truth = np.full((4, 2, 3), np.nan, dtype=np.float64)
    truth[:, 0, 0] = np.arange(4)
    truth[:, 0, 1:] = 0.0
    truth[:3, 1, 1] = np.arange(3)
    truth[:3, 1, [0, 2]] = 0.0

    predicted, supported = _fit_masked_positions(
        design,
        truth,
        rows,
        np.array([True, True, True, False]),
        alpha=1e-8,
        min_train_episodes=3,
    )

    assert supported.tolist() == [True, True]
    np.testing.assert_allclose(predicted[:, 0, 0], np.arange(4), atol=1e-6)
    np.testing.assert_allclose(predicted[:, 1, 1], np.arange(4), atol=1e-6)


def test_position_metric_averages_objects_within_scene_then_episodes():
    rows = pd.DataFrame({"trace_id": ["long", "long", "short"]})
    truth = np.zeros((3, 2, 3), dtype=np.float64)
    prediction = truth.copy()
    prediction[0, :, 0] = 1.0
    prediction[1, :, 0] = 1.0
    prediction[2, :, 0] = 3.0
    presence = np.ones((3, 2), dtype=np.float64)
    targets = SceneMapTargets(
        vocabulary=("one", "two"),
        presence=presence,
        visibility=presence,
        position=truth,
        initial_position=truth,
        previous_position=truth,
        role_manipulated=np.zeros_like(presence, dtype=bool),
        role_distractor=np.zeros_like(presence, dtype=bool),
    )
    masks = {
        "train": np.array([False, False, False]),
        "selection": np.array([True, True, True]),
        "test": np.array([True, True, True]),
    }

    metrics = _position_metrics(
        targets,
        prediction,
        rows,
        masks,
        supported=np.array([True, True]),
    )

    assert metrics["test_error_m"] == 2.0
    assert metrics["test_x_mae_m"] == 2.0
    assert metrics["test_y_mae_m"] == 0.0


def test_project_reuses_evictable_reduced_feature_cache(tmp_path):
    values = np.arange(120, dtype=np.float64).reshape(20, 6)
    train = np.array([True] * 12 + [False] * 8)
    cache_path = tmp_path / "projection.npz"

    first = _project(values, train, [2, 4], cache_path=cache_path)
    second = _project(values + 1000.0, train, [2, 4], cache_path=cache_path)

    assert cache_path.exists()
    assert first.keys() == second.keys()
    for dim in first:
        np.testing.assert_array_equal(first[dim], second[dim])
