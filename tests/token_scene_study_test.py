from __future__ import annotations

import numpy as np
import pandas as pd

from vla_lens import create_synthetic_trace_dataset
from vla_lens.probes.scene_map_study import SceneMapTargets
from vla_lens.probes.structured_scene_models import (
    FittedSceneRepresentation,
    SceneMLPDecoder,
    fit_layer_mixture,
    fit_scene_decoder,
    fit_structured_scene_representations,
)
from vla_lens.probes.token_representations import (
    _selected_token_metadata,
    build_layer_token_readouts,
    read_compressed_token_layers,
)
from vla_lens.probes.token_scene_study import (
    _activation_baseline_comparison_table,
    _decoder_parameter_table,
    _fit_no_activation_baselines,
    _paired_bootstrap_summary,
    _paired_comparison_table,
    _task_keys,
    _weighted_token_importance,
)


def test_task_keys_do_not_merge_same_numeric_id_across_benchmarks():
    rows = pd.DataFrame(
        {
            "benchmark": ["libero_goal", "libero_object"],
            "task_id": [8, 8],
        }
    )

    assert _task_keys(rows).tolist() == ["libero_goal:8", "libero_object:8"]


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

    compact_first = read_compressed_token_layers(
        dataset,
        first.rows,
        first.source_sites,
        first.token_metadata,
        layers=first.layers,
        channel_projection=first.channel_projection,
        io_workers=2,
    )
    compact_second = read_compressed_token_layers(
        dataset,
        first.rows,
        first.source_sites,
        first.token_metadata,
        layers=first.layers,
        channel_projection=first.channel_projection,
        io_workers=2,
    )
    assert compact_first.values.shape == (32, 3, 8, 3)
    assert not compact_first.cache_hit
    assert compact_second.cache_hit
    np.testing.assert_array_equal(compact_first.values, compact_second.values)


def test_dynamic_token_metadata_does_not_duplicate_model_token_positions(tmp_path):
    dataset = create_synthetic_trace_dataset(
        tmp_path / "dataset", num_episodes=2, timesteps=4, layers=2
    )
    bundle = dataset.bundles[0]
    original = bundle.tokens
    repeated = pd.concat(
        [original.assign(policy_call_index=index) for index in range(3)],
        ignore_index=True,
    )
    bundle.__dict__["tokens"] = repeated

    metadata = _selected_token_metadata(bundle, "action")
    readouts = build_layer_token_readouts(
        dataset,
        {
            "module": "action_head.layers.*.resid",
            "tensor_type": "resid",
            "token_kind": "action",
            "layers": [0, 1],
            "timesteps": "all",
            "dtype": "float32",
        },
        {
            "kind": "existing",
            "column": "split",
            "train_value": "train",
            "selection_value": "test",
            "test_value": "test",
        },
        readout_dim=2,
        token_channel_dim=2,
        channel_sample_count=32,
        projection_fit_rows=16,
        io_workers=1,
        cache=False,
    )

    assert metadata["token_index"].tolist() == list(range(8))
    assert readouts.token_count == 8


def test_token_readouts_can_select_one_metadata_stream(tmp_path):
    dataset = create_synthetic_trace_dataset(
        tmp_path / "dataset", num_episodes=2, timesteps=4, layers=2
    )
    for bundle in dataset.bundles:
        tokens = bundle.tokens.copy()
        tokens["camera_id"] = "wrist"
        action_indices = tokens.index[tokens["token_kind"].astype(str) == "action"]
        tokens.loc[action_indices[:4], "camera_id"] = "main"
        bundle.__dict__["tokens"] = tokens

    readouts = build_layer_token_readouts(
        dataset,
        {
            "module": "action_head.layers.*.resid",
            "tensor_type": "resid",
            "token_kind": "action",
            "token_filters": {"camera_id": "main"},
            "layers": [0, 1],
            "timesteps": "all",
            "dtype": "float32",
        },
        {
            "kind": "existing",
            "column": "split",
            "train_value": "train",
            "selection_value": "test",
            "test_value": "test",
        },
        readout_dim=2,
        token_channel_dim=2,
        channel_sample_count=32,
        projection_fit_rows=16,
        io_workers=1,
        cache=False,
    )

    assert readouts.token_count == 4
    assert set(readouts.token_metadata["camera_id"]) == {"main"}


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


def test_paired_bootstrap_reports_effect_size_and_uncertainty():
    summary = _paired_bootstrap_summary(
        improvement=np.array([0.1, 0.2, 0.3, 0.4]),
        groups=np.array(["a", "a", "b", "b"]),
        bootstrap_samples=500,
        seed=7,
    )

    assert summary["unit_count"] == 2
    assert summary["mean_improvement"] == 0.25
    assert summary["ci95_low"] > 0.0
    assert summary["probability_improvement"] == 1.0


def test_token_importance_fractions_include_layer_weights():
    weighted, fractions = _weighted_token_importance(
        np.array([1.0, 3.0]),
        np.array([0.0, 1.0]),
    )

    np.testing.assert_array_equal(weighted, [[0.0, 0.0], [1.0, 3.0]])
    np.testing.assert_array_equal(fractions, [[0.0, 0.0], [0.25, 0.75]])
    assert fractions.sum() == 1.0


def test_token_readouts_reject_different_trace_token_topologies(tmp_path):
    dataset = create_synthetic_trace_dataset(
        tmp_path / "dataset", num_episodes=4, timesteps=4, layers=2
    )
    bundle = dataset.bundles[1]
    bundle.__dict__["tokens"] = bundle.tokens.loc[
        ~(
            (bundle.tokens["token_kind"].astype(str) == "action")
            & (bundle.tokens["token_index"].astype(int) == 7)
        )
    ].copy()

    with np.testing.assert_raises_regex(ValueError, "identical token counts, indices"):
        build_layer_token_readouts(
            dataset,
            {
                "module": "action_head.layers.*.resid",
                "tensor_type": "resid",
                "token_kind": "action",
                "layers": [0, 1],
                "timesteps": "all",
                "dtype": "float32",
            },
            {
                "kind": "existing",
                "column": "split",
                "train_value": "train",
                "selection_value": "test",
                "test_value": "test",
            },
            readout_dim=2,
            token_channel_dim=2,
            channel_sample_count=32,
            projection_fit_rows=16,
            io_workers=1,
            cache=False,
        )


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
            "task_id": np.arange(row_count) % 5,
            "prompt": [
                f"{['move', 'lift', 'place', 'open', 'close'][index % 5]} "
                f"object {index % 7} near zone {index % 3}"
                for index in range(row_count)
            ],
            "benchmark": "synthetic",
            "scene_family": np.arange(row_count) % 3,
            "task_phase": "initial",
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
    comparisons = _paired_comparison_table(
        selected,
        rows,
        targets,
        {
            "column": "split",
            "train_value": "train",
            "selection_value": "selection",
            "test_value": "test",
        },
        bootstrap_samples=100,
    )
    assert len(comparisons) == 16
    assert set(comparisons["model"]) == {"linear"}
    assert comparisons["candidate"].str.endswith("__linear").all()
    baselines = _fit_no_activation_baselines(
        rows,
        targets,
        {
            "column": "split",
            "train_value": "train",
            "selection_value": "selection",
            "test_value": "test",
        },
        ridge_alphas=[0.1, 1.0],
        min_train_episodes=1,
        context_columns=["benchmark", "scene_family", "task_phase"],
    )
    assert {
        (item.record["baseline"], item.record["target"]) for item in baselines
    } == {
        ("training_frequency", "scene_identity"),
        ("per_object_training_mean", "object_position"),
        ("prompt_and_scene_context", "scene_identity"),
        ("prompt_and_scene_context", "object_position"),
    }
    baseline_comparisons = _activation_baseline_comparison_table(
        selected,
        baselines,
        rows,
        targets,
        {
            "column": "split",
            "train_value": "train",
            "selection_value": "selection",
            "test_value": "test",
        },
        bootstrap_samples=100,
    )
    assert len(baseline_comparisons) == 8


def test_object_conditioned_mlp_position_heads_replay_without_sklearn(tmp_path):
    rng = np.random.default_rng(11)
    X = rng.normal(size=(90, 4))
    position = np.full((90, 2, 3), np.nan, dtype=np.float64)
    position[:, 0, 0] = X[:, 0] * X[:, 1]
    position[:, 0, 1] = X[:, 2] ** 2
    position[:, 0, 2] = X[:, 3]
    position[:, 1] = position[:, 0] * 0.5
    rows = pd.DataFrame(
        {"trace_id": [f"episode_{index}" for index in range(len(X))]}
    )

    decoder = fit_scene_decoder(
        X,
        position,
        rows,
        np.arange(len(X)) < 70,
        np.array([True, True]),
        target="object_position",
        alpha=1e-4,
        min_train_episodes=1,
        model="mlp",
        mlp_hidden_layer_sizes=(16,),
        mlp_max_iter=500,
    )

    assert isinstance(decoder, SceneMLPDecoder)
    assert all(
        network is not None and network.n_iter > 0 and np.isfinite(network.final_loss)
        for network in decoder.networks
    )
    prediction = decoder.predict(X[70:])
    assert prediction.shape == (20, 2, 3)
    assert np.isfinite(prediction).all()
    probe_error = float(np.mean(np.abs(prediction - position[70:])))
    train_mean = np.nanmean(position[:70], axis=0)
    baseline_error = float(np.mean(np.abs(position[70:] - train_mean)))
    assert probe_error < baseline_error
    fitted = FittedSceneRepresentation(
        record={
            "representation": "tokenwise",
            "structure": "single_layer",
            "model": "mlp",
            "target": "object_position",
        },
        decoder=decoder,
        prediction=decoder.predict(X),
        layer_weights=np.array([1.0]),
    )
    parameters = _decoder_parameter_table([fitted])
    parameters.to_parquet(tmp_path / "decoder_parameters.parquet", index=False)
    assert set(parameters["parameter_kind"]) == {"standardizer", "mlp_layer"}
    assert parameters["n_iter"].dropna().gt(0).all()
    assert parameters["converged"].dropna().isin([True, False]).all()
