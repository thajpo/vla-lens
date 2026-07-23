from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from vla_lens.probes.object_query_localization_study import (
    _fit_candidates,
    _mean_visible_patch_ap,
    _query_design,
    _sample_patch_examples,
    _score_fit,
    _select_presence_threshold,
)
from vla_lens.probes.object_roi_identity_study import _instance_features
from vla_lens.probes.object_study_common import fit_classifier


def test_saved_linear_and_mlp_classifier_parameters_replay_training_signal():
    rng = np.random.default_rng(4)
    values = rng.normal(size=(120, 4))
    labels = (values[:, 0] - 0.5 * values[:, 1] > 0).astype(int)

    for model in ["linear", "mlp"]:
        fitted = fit_classifier(
            values,
            labels,
            model=model,
            alpha=1e-4,
            hidden_units=64,
            max_iter=300,
            random_state=7,
        )

        probabilities = fitted.predict_proba(values)
        assert probabilities.shape == (120, 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-7)
        assert np.mean(fitted.predict(values) == labels) > 0.8


def test_roi_features_preserve_16_channels_and_average_only_selected_patches():
    compact = np.arange(2 * 2 * 4 * 3, dtype=np.float32).reshape(2, 2, 4, 3)
    data = SimpleNamespace(
        compact=compact,
        readouts=SimpleNamespace(layers=(0, 4)),
    )
    instances = pd.DataFrame(
        {
            "source_row_index": [0],
            "roi_patch_indices": [[0, 1]],
            "wrong_roi_patch_indices": [[2]],
            "background_patch_indices": [[3]],
        }
    )

    features = _instance_features(data, instances)

    assert features["object_roi"].shape == (1, 2, 3)
    np.testing.assert_allclose(features["object_roi"][0], compact[0, :, [0, 1]].mean(axis=0))
    np.testing.assert_allclose(features["whole_image"][0], compact[0].mean(axis=1))
    np.testing.assert_allclose(features["wrong_object_roi"][0], compact[0, :, 2])


def test_query_sampling_keeps_one_positive_and_three_named_negative_kinds():
    rows = pd.DataFrame(
        {
            "trace_id": ["episode"],
            "episode_id": ["episode"],
            "benchmark": ["suite"],
            "task_id": ["task"],
            "task_name": ["task"],
            "prompt": ["move object"],
        }
    )
    tokens = pd.DataFrame(
        {
            "patch_row": [0, 0, 1, 1, 2, 2],
            "patch_col": [0, 1, 0, 1, 0, 1],
        }
    )
    patch_targets = np.zeros((1, 2, 6), dtype=bool)
    patch_targets[0, 0, :2] = True
    patch_targets[0, 1, 2:4] = True
    data = SimpleNamespace(
        readouts=SimpleNamespace(rows=rows, token_metadata=tokens),
        patch_targets=patch_targets,
        visible=np.array([[True, True]]),
        targets=SimpleNamespace(vocabulary=("red_cube", "blue_bowl")),
    )

    examples = _sample_patch_examples(
        data,
        np.array([True]),
        np.array([True, True]),
        max_examples=100,
        seed=9,
    )

    counts = examples["sample_kind"].value_counts().to_dict()
    assert counts == {
        "positive": 4,
        "near_box": 4,
        "wrong_object": 4,
        "background": 4,
    }
    assert len(examples) == 4 * int(examples["label"].sum())


def test_query_design_combines_local_token_xy_and_query_identity_without_pooling():
    compact = np.arange(2 * 1 * 3 * 2, dtype=np.float32).reshape(2, 1, 3, 2)
    data = SimpleNamespace(compact=compact)
    examples = pd.DataFrame(
        {
            "source_row_index": [0, 1],
            "patch_index": [2, 0],
            "object_index": [0, 2],
        }
    )
    centers = np.array([[0.1, 0.1], [0.5, 0.1], [0.9, 0.1]], dtype=np.float32)

    design = _query_design(
        data,
        examples,
        supported=np.array([True, False, True]),
        centers=centers,
        layer_index=0,
    )

    assert design.shape == (2, 6)
    np.testing.assert_array_equal(design[:, :2], [compact[0, 0, 2], compact[1, 0, 0]])
    np.testing.assert_array_equal(design[:, 2:4], [centers[2], centers[0]])
    np.testing.assert_array_equal(design[:, 4:], [[1, 0], [0, 1]])


def test_presence_threshold_is_selected_from_validation_scene_jaccard():
    scores = np.zeros((2, 3, 2), dtype=np.float32)
    scores[0, 0] = 0.9
    scores[0, 1] = 0.1
    scores[0, 2] = 0.2
    scores[1, 0] = 0.1
    scores[1, 1] = 0.8
    scores[1, 2] = 0.2
    visible = np.array([[True, False, False], [False, True, False]])

    threshold = _select_presence_threshold(scores, visible, np.ones(3, dtype=bool))

    assert 0.2 < threshold < 0.8


def test_query_candidate_is_selected_on_sampled_validation_and_scores_full_grid():
    rows = pd.DataFrame(
        {
            "trace_id": [f"episode-{index}" for index in range(8)],
            "benchmark": ["suite"] * 8,
            "task_id": [str(index) for index in range(8)],
            "prompt": ["find it"] * 8,
        }
    )
    compact = np.zeros((8, 1, 4, 2), dtype=np.float32)
    compact[:, 0, :, 0] = np.array([1.0, 1.0, -1.0, -1.0])
    compact[:, 0, :, 1] = np.array([-1.0, -1.0, 1.0, 1.0])
    patch_targets = np.zeros((8, 2, 4), dtype=bool)
    patch_targets[:, 0, :2] = True
    patch_targets[:, 1, 2:] = True
    data = SimpleNamespace(
        compact=compact,
        readouts=SimpleNamespace(layers=(4,), rows=rows),
        patch_targets=patch_targets,
        visible=np.ones((8, 2), dtype=bool),
    )
    records = []
    for row_index in range(8):
        for object_index in range(2):
            for patch_index in range(4):
                records.append(
                    {
                        "source_row_index": row_index,
                        "object_index": object_index,
                        "patch_index": patch_index,
                        "label": int(patch_targets[row_index, object_index, patch_index]),
                    }
                )
    examples = pd.DataFrame.from_records(records)
    train = examples.loc[examples["source_row_index"] < 5].reset_index(drop=True)
    selection = examples.loc[examples["source_row_index"] >= 5].reset_index(drop=True)
    centers = np.array(
        [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]],
        dtype=np.float32,
    )

    candidates, selected = _fit_candidates(
        data,
        train,
        selection,
        np.array([True, True]),
        centers,
        np.zeros((8, 0), dtype=np.float32),
        {
            "layers": [4],
            "models": ["mlp"],
            "alphas": [1e-4],
            "mlp_hidden_units": 64,
            "max_iter": 300,
            "random_state": 4,
        },
    )
    scores = _score_fit(
        data,
        np.array([7]),
        np.array([True, True]),
        centers,
        selected["activation_query"],
        np.zeros((8, 0), dtype=np.float32),
    )

    assert len(candidates) == 3
    assert selected["activation_query"].layer == 4
    assert scores.shape == (1, 2, 4)
    assert _mean_visible_patch_ap(data, np.array([7]), scores, np.array([True, True])) > 0.9
