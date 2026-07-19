from __future__ import annotations

import numpy as np
import pandas as pd

from vla_lens import create_synthetic_trace_dataset
from vla_lens.probes.scene_map_study import SceneMapTargets
from vla_lens.probes.structured_scene_models import (
    fit_layer_mixture,
    fit_structured_scene_representations,
)
from vla_lens.probes.token_representations import build_layer_token_readouts


def test_token_readouts_keep_layer_and_token_structure_in_a_small_cache(tmp_path):
    dataset = create_synthetic_trace_dataset(
        tmp_path / "dataset", num_episodes=4, timesteps=8, layers=3
    )
    feature = {
        "module": "action_head.layers.*.resid",
        "tensor_type": "resid",
        "token_kind": "action",
        "layers": [0, 1, 2],
        "timesteps": "all",
        "dtype": "float32",
    }
    split = {
        "kind": "existing",
        "column": "split",
        "train_value": "train",
        "selection_value": "test",
        "test_value": "test",
    }
    settings = {
        "readout_dim": 4,
        "token_channel_dim": 3,
        "channel_sample_count": 128,
        "projection_fit_rows": 64,
        "io_workers": 2,
        "cache": True,
    }

    first = build_layer_token_readouts(dataset, feature, split, **settings)
    second = build_layer_token_readouts(dataset, feature, split, **settings)

    assert first.layers == (0, 1, 2)
    assert first.token_count == 8
    assert first.channel_dim == 3
    assert first.pooled.shape == (32, 3, 4)
    assert first.tokenwise.shape == (32, 3, 4)
    assert first.tokenwise_projection.components.shape[1] == 8 * 3
    assert first.rows["timestep"].tolist()[:8] == list(range(8))
    np.testing.assert_array_equal(first.pooled, second.pooled)
    np.testing.assert_array_equal(first.tokenwise, second.tokenwise)


def test_learned_layer_mixture_favors_the_layer_with_position_signal():
    rng = np.random.default_rng(4)
    row_count, layer_count, feature_dim = 90, 3, 4
    values = rng.normal(size=(row_count, layer_count, feature_dim))
    signal = values[:, 2, 0]
    position = np.zeros((row_count, 1, 3), dtype=np.float64)
    position[:, 0, 0] = signal
    position[:, 0, 1] = signal * 0.5
    rows = pd.DataFrame(
        {
            "trace_id": [f"episode_{index}" for index in range(row_count)],
            "split": ["train"] * 50 + ["selection"] * 20 + ["test"] * 20,
        }
    )
    masks = {
        value: rows["split"].astype(str).to_numpy() == value
        for value in ["train", "selection", "test"]
    }

    _, prediction, weights, _ = fit_layer_mixture(
        values,
        position,
        rows,
        masks,
        supported=np.array([True]),
        target="object_position",
        alpha=0.1,
        min_train_episodes=1,
        max_iterations=5,
        weight_regularization=1e-3,
    )

    assert weights[2] > 0.9
    error = np.linalg.norm(prediction[masks["test"]] - position[masks["test"]], axis=2)
    assert float(np.nanmean(error)) < 0.02


def test_matched_study_selects_all_four_representation_variants():
    rng = np.random.default_rng(7)
    row_count, layer_count, feature_dim = 75, 3, 6
    pooled = rng.normal(size=(row_count, layer_count, feature_dim))
    tokenwise = pooled + rng.normal(scale=0.05, size=pooled.shape)
    signal = pooled[:, 1, 0]
    presence = np.stack([signal > 0, signal <= 0], axis=1).astype(float)
    position = np.zeros((row_count, 2, 3), dtype=np.float64)
    position[:, 0, 0] = signal
    position[:, 1, 1] = signal
    rows = pd.DataFrame(
        {
            "trace_id": [f"episode_{index}" for index in range(row_count)],
            "split": ["train"] * 45 + ["selection"] * 15 + ["test"] * 15,
        }
    )
    targets = SceneMapTargets(
        vocabulary=("one", "two"),
        presence=presence,
        visibility=presence,
        position=position,
        initial_position=np.zeros_like(position),
        previous_position=np.zeros_like(position),
        role_manipulated=np.ones_like(presence, dtype=bool),
        role_distractor=np.zeros_like(presence, dtype=bool),
    )

    _, selected = fit_structured_scene_representations(
        {"pooled": pooled, "tokenwise": tokenwise},
        rows,
        targets,
        layers=[0, 1, 2],
        split={
            "column": "split",
            "train_value": "train",
            "selection_value": "selection",
            "test_value": "test",
        },
        readout_dims=[4, 6],
        ridge_alphas=[0.1],
        min_train_episodes=1,
        mixture_iterations=3,
    )

    variants = {
        (item.record["representation"], item.record["structure"])
        for item in selected
    }
    assert variants == {
        ("pooled", "single_layer"),
        ("pooled", "learned_layer_mix"),
        ("tokenwise", "single_layer"),
        ("tokenwise", "learned_layer_mix"),
    }
    assert len(selected) == 8
