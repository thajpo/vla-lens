from __future__ import annotations

import numpy as np
import pandas as pd

from vla_lens.probes.matched_scene_study import (
    _bbox_patch_mask,
    _feature_sites,
    _localization_metrics,
    _matched_pairs,
    _random_ranking_expected_ap,
    _summary_table,
)
from vla_lens.probes.scene_map_study import SceneMapTargets


def test_matched_pairs_keeps_one_changed_object_inside_one_scene():
    rows = pd.DataFrame(
        {
            "trace_id": ["a", "b", "c", "d"],
            "env_id": ["suite", "suite", "suite", "other"],
            "task_id": [1, 1, 1, 1],
            "prompt": ["move it"] * 4,
            "split": ["train"] * 4,
            "seed": [1, 2, 3, 4],
        }
    )
    positions = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.02, 0.0, 0.0], [1.001, 0.0, 0.0]],
            [[0.05, 0.0, 0.0], [1.03, 0.0, 0.0]],
            [[0.02, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    )
    targets = SceneMapTargets(
        vocabulary=("moved", "still"),
        presence=np.ones((4, 2)),
        visibility=np.ones((4, 2)),
        position=positions,
        initial_position=positions,
        previous_position=positions,
        role_manipulated=np.array([[True, False]] * 4),
        role_distractor=np.array([[False, True]] * 4),
    )

    pairs = _matched_pairs(
        rows,
        targets,
        movement_threshold_m=0.01,
        stationary_threshold_m=0.01,
        robot_threshold_m=0.005,
    )

    assert len(pairs) == 1
    assert pairs.iloc[0]["left_trace_id"] == "a"
    assert pairs.iloc[0]["right_trace_id"] == "b"
    assert pairs.iloc[0]["moved_object_name"] == "moved"
    assert np.isclose(pairs.iloc[0]["moved_distance_m"], 0.02)


def test_bbox_patch_mask_uses_pixel_overlap():
    tokens = pd.DataFrame(
        {
            "pixel_x0": [0, 16, 0, 16],
            "pixel_x1": [16, 32, 16, 32],
            "pixel_y0": [0, 0, 16, 16],
            "pixel_y1": [16, 16, 32, 32],
        }
    )

    mask = _bbox_patch_mask(tokens, np.array([15, 17, 30, 30]))

    np.testing.assert_array_equal(mask, [False, False, True, True])


def test_feature_site_layers_have_one_parquet_safe_type():
    sites = _feature_sites({"include_input_embeddings": True, "layers": [0, 17]})

    assert [site["layer"] for site in sites] == ["input", "0", "17"]


def test_localization_metrics_separate_target_from_stationary_control():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    target = np.array([True, True, False, False])
    control = np.array([False, False, True, False])

    metrics = _localization_metrics(scores, target, control)

    assert metrics["target_average_precision"] == 1.0
    assert metrics["target_top_k_recall"] == 1.0
    assert metrics["target_average_precision"] > metrics[
        "stationary_control_average_precision"
    ]


def test_random_ranking_average_precision_is_not_patch_prevalence():
    expected = _random_ranking_expected_ap(256, 22)

    assert np.isclose(expected, 0.104306, atol=1e-6)
    assert expected > 22 / 256


def test_summary_selects_feature_on_validation_only():
    rows = []
    for split, good_a, good_b in [
        ("val", 0.8, 0.3),
        ("test", 0.2, 0.9),
    ]:
        for feature, score in [("a", good_a), ("b", good_b)]:
            rows.append(
                {
                    "split": split,
                    "feature_id": feature,
                    "layer": feature,
                    "pair_id": f"{split}-{feature}",
                    "scene_key": f"scene-{split}",
                    "target_average_precision": score,
                    "random_ranking_expected_average_precision": 0.1,
                    "stationary_control_average_precision": 0.2,
                    "target_roc_auc": score,
                    "target_top_k_recall": score,
                }
            )

    summary = _summary_table(
        pd.DataFrame.from_records(rows),
        split={"selection_value": "val"},
        bootstrap_samples=10,
    )

    selected = summary.loc[summary["selected_on_validation"]]
    assert set(selected["feature_id"]) == {"a"}
    assert set(selected["split"]) == {"val", "test"}


def test_summary_weights_scene_groups_instead_of_pair_count():
    rows = [
        {
            "split": "val",
            "feature_id": "feature",
            "layer": "17",
            "pair_id": f"repeated-{index}",
            "scene_key": "repeated-scene",
            "target_average_precision": 0.9,
            "random_ranking_expected_average_precision": 0.1,
            "stationary_control_average_precision": 0.1,
            "target_roc_auc": 0.9,
            "target_top_k_recall": 0.9,
        }
        for index in range(9)
    ]
    rows.append(
        {
            "split": "val",
            "feature_id": "feature",
            "layer": "17",
            "pair_id": "single",
            "scene_key": "single-scene",
            "target_average_precision": 0.1,
            "random_ranking_expected_average_precision": 0.1,
            "stationary_control_average_precision": 0.1,
            "target_roc_auc": 0.1,
            "target_top_k_recall": 0.1,
        }
    )

    summary = _summary_table(
        pd.DataFrame.from_records(rows),
        split={"selection_value": "val"},
        bootstrap_samples=10,
    )

    assert np.isclose(summary.iloc[0]["mean_average_precision"], 0.5)
    assert np.isclose(summary.iloc[0]["mean_average_precision_lift"], 0.4)
