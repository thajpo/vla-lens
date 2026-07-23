from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from tests._support.object_flow_dataset import object_flow_dataset
from vla_lens.probes.geometry_study import (
    GeometryTarget,
    _align_labeled_geometry_rows,
    _apply_split_contract,
    _decode_orientation,
    _encode_orientation,
    _fit_feature_study,
    _geometry_confidence_table,
    _limit_rows_by_episode,
    _normalize_spec,
    _save_geometry_study,
    _select_geometry_targets,
    _target_metrics,
    geometry_target_table,
    predict_geometry_readout,
)
from vla_lens.synthetic import create_synthetic_trace_dataset


def test_orientation_representations_round_trip_on_so3():
    quaternions = Rotation.from_euler(
        "xyz",
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, -0.4, 1.1],
                [-1.2, 0.7, -2.4],
            ]
        ),
    ).as_quat()

    for basis in ["quaternion", "rotation_6d", "rotation_vector", "euler_sincos"]:
        encoded = _encode_orientation(quaternions, basis)
        decoded = _decode_orientation(encoded, basis)
        relative = Rotation.from_quat(decoded).inv() * Rotation.from_quat(quaternions)
        np.testing.assert_allclose(relative.magnitude(), 0.0, atol=1e-7)


def test_position_metric_weights_episodes_equally():
    truth = np.zeros((4, 3), dtype=np.float64)
    predicted = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    target = GeometryTarget(
        name="position",
        kind="position",
        basis="xyz",
        values=truth,
        truth=truth,
        baseline_values={},
    )

    metrics = _target_metrics(
        target,
        truth,
        predicted,
        trace_ids=np.array(["long", "long", "long", "short"]),
    )

    assert metrics["row_mean_error"] == 1.5
    assert metrics["episode_mean_error"] == 2.0
    assert metrics["error_unit"] == "meters"


def test_geometry_target_table_aligns_pose_frames_and_reuses_cache(tmp_path):
    dataset = object_flow_dataset(tmp_path / "dataset")
    rows = pd.DataFrame(
        {
            "trace_id": ["flow_trace", "flow_trace"],
            "timestep": [0, 3],
            "primary_target_object": ["red_cube_1", "red_cube_1"],
        }
    )

    first = geometry_target_table(
        dataset,
        rows,
        object_column="primary_target_object",
        cache=True,
    )
    second = geometry_target_table(
        dataset,
        rows,
        object_column="primary_target_object",
        cache=True,
    )

    assert len(first) == 2
    np.testing.assert_allclose(first.iloc[0]["position_initial_delta"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(first.iloc[1]["position_initial_delta"], [0.06, 0.0, 0.0])
    np.testing.assert_allclose(first.iloc[1]["position_previous_delta"], [0.06, 0.0, 0.0])
    np.testing.assert_allclose(first.iloc[1]["eef_position_previous_delta"], [0.0, 0.0, 0.0])
    assert first.iloc[1]["executed_action_mean"] == [0.0]
    assert not bool(first.iloc[1]["is_first_policy_call"])
    np.testing.assert_allclose(
        first.iloc[1]["orientation_initial_relative_quat"], [0.0, 0.0, 0.0, 1.0]
    )
    pd.testing.assert_frame_equal(first, second)


def test_geometry_rows_keep_original_feature_indices_after_missing_label(tmp_path):
    dataset = object_flow_dataset(tmp_path / "dataset")
    rows = pd.DataFrame(
        {
            "trace_id": ["flow_trace"] * 3,
            "timestep": [0, 1, 3],
            "primary_target_object": ["red_cube_1", None, "red_cube_1"],
            "__feature_row_index": [0, 1, 2],
        }
    )
    features = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)

    aligned_rows, aligned_features = _align_labeled_geometry_rows(
        dataset,
        rows,
        features,
        object_column="primary_target_object",
    )

    assert aligned_rows["timestep"].tolist() == [0, 3]
    np.testing.assert_array_equal(aligned_features[:, 0], [10.0, 30.0])


def test_within_task_split_keeps_episodes_intact_and_represents_every_split():
    rows = pd.DataFrame(
        {
            "trace_id": [f"task_a_{index}" for index in range(10)]
            + [f"task_b_{index}" for index in range(10)],
            "task_id": ["a"] * 10 + ["b"] * 10,
        }
    )
    split = {
        "kind": "within_task_episode",
        "group_column": "task_id",
        "column": "geometry_split",
        "train_value": "train",
        "selection_value": "val",
        "test_value": "test",
        "seed": 7,
    }

    assigned = _apply_split_contract(rows, split)

    for _, group in assigned.groupby("task_id"):
        assert group["geometry_split"].value_counts().to_dict() == {
            "train": 6,
            "val": 2,
            "test": 2,
        }


def test_episode_limit_rejects_fewer_episodes_than_required_splits():
    rows = pd.DataFrame(
        {
            "trace_id": ["train_0", "val_0", "test_0"],
            "split": ["train", "val", "test"],
        }
    )
    features = np.arange(3, dtype=np.float32)[:, None]

    with pytest.raises(ValueError, match="cannot cover all 3 required splits"):
        _limit_rows_by_episode(
            rows,
            features,
            2,
            required_split_values=("train", "val", "test"),
        )


def test_episode_limit_preserves_every_required_split():
    rows = pd.DataFrame(
        {
            "trace_id": ["train_0", "train_1", "val_0", "test_0"],
            "split": ["train", "train", "val", "test"],
        }
    )
    features = np.arange(4, dtype=np.float32)[:, None]

    limited_rows, limited_features = _limit_rows_by_episode(
        rows,
        features,
        3,
        required_split_values=("train", "val", "test"),
    )

    assert set(limited_rows["split"]) == {"train", "val", "test"}
    np.testing.assert_array_equal(limited_features[:, 0], [0.0, 2.0, 3.0])


def _nonlinear_geometry_problem(seed: int = 4):
    rng = np.random.default_rng(seed)
    row_count = 180
    features = rng.uniform(-2.0, 2.0, size=(row_count, 8))
    truth = np.column_stack(
        [
            features[:, 0] ** 2,
            np.sin(features[:, 1]),
            features[:, 2] * features[:, 3],
        ]
    )
    rows = pd.DataFrame(
        {
            "trace_id": [f"episode_{index:03d}" for index in range(row_count)],
            "timestep": np.arange(row_count),
            "split": ["train"] * 100 + ["validation"] * 40 + ["test"] * 40,
            "task_id": [f"task_{index // 10:02d}" for index in range(row_count)],
        }
    )
    target = GeometryTarget(
        name="position_world",
        kind="position",
        basis="xyz",
        values=truth,
        truth=truth,
        baseline_values={"physical_zero": np.zeros_like(truth)},
    )
    return features, rows, target


def _fit_nonlinear_problem(features, rows, target, *, models=("ridge", "mlp")):
    return _fit_feature_study(
        features,
        rows,
        [target],
        feature_id="synthetic",
        split_column="split",
        train_value="train",
        selection_value="validation",
        test_value="test",
        sweep_columns=[],
        pca_dims=[4, 8],
        ridge_alphas=[1.0],
        models=models,
        mlp_seed=0,
        mlp_max_iter=300,
        bootstrap_samples=200,
        bootstrap_group_column="task_id",
        baseline_columns=[],
    )


def test_geometry_spec_keeps_ridge_default_and_validates_target_allowlist():
    normalized = _normalize_spec(
        {
            "features": [{"name": "site"}],
            "probe": {"pca_dims": [64], "ridge_alphas": [1.0]},
            "targets": ["position_world", "orientation_world_rotation_6d"],
        }
    )

    assert normalized["probe"]["models"] == ["ridge"]
    assert normalized["probe"]["mlp_hidden_units"] == 64
    assert normalized["probe"]["mlp_alpha"] == 1e-4
    assert normalized["probe"]["mlp_max_iter"] == 300
    assert normalized["targets"] == [
        "position_world",
        "orientation_world_rotation_6d",
    ]
    with pytest.raises(ValueError, match="Unknown geometry targets"):
        _normalize_spec(
            {"features": [{"name": "site"}], "targets": ["not_a_pose_target"]}
        )


def test_target_allowlist_preserves_requested_order():
    _, rows, target = _nonlinear_geometry_problem()
    other = GeometryTarget(
        "position_previous_delta",
        "position",
        "xyz",
        target.values,
        target.truth,
        target.baseline_values,
    )

    selected = _select_geometry_targets(
        [target, other], ["position_previous_delta", "position_world"]
    )

    assert [item.name for item in selected] == [
        "position_previous_delta",
        "position_world",
    ]
    assert len(rows) == 180


def test_mlp_is_validation_selected_gated_and_reconstructable():
    features, rows, target = _nonlinear_geometry_problem()

    candidates, selections, predictions, states = _fit_nonlinear_problem(
        features, rows, target
    )

    selection_frame = pd.DataFrame.from_records(selections)
    mlp = selection_frame.loc[selection_frame["model"] == "mlp"].iloc[0]
    ridge = selection_frame.loc[selection_frame["model"] == "ridge"].iloc[0]
    assert mlp["selection_error"] < mlp["selection_baseline_error"]
    assert bool(mlp["validation_gate_passed"])
    assert bool(mlp["test_reported"])
    assert bool(mlp["promoted"])
    assert not bool(ridge["promoted"])
    assert mlp["n_iter"] <= 300
    assert mlp["selection_confidence_group_column"] == "task_id"
    assert mlp["selection_confidence_group_count"] == 4
    assert {record["model"] for record in predictions} == {"ridge", "mlp"}
    assert all(record["baseline_value"] is not None for record in predictions)
    assert all(record["model"] in {"ridge", "mlp"} for record in candidates)

    state_by_id = {state.readout_id: state for state in states}
    for selection in selections:
        state = state_by_id[selection["readout_id"]]
        reconstructed = predict_geometry_readout(state.contract, state.arrays, features)
        assert reconstructed.shape == target.values.shape
        np.testing.assert_allclose(
            state.arrays["feature_mean"], features[:100].mean(axis=0), atol=1e-12
        )
        matching = next(
            record
            for record in predictions
            if record["readout_id"] == selection["readout_id"]
            and record["split"] == "validation"
        )
        source_index = int(str(matching["trace_id"]).split("_")[-1])
        np.testing.assert_allclose(
            reconstructed[source_index], matching["prediction_representation"], atol=1e-10
        )


def test_geometry_reuses_one_train_fitted_pca_for_all_dims(monkeypatch):
    features, rows, target = _nonlinear_geometry_problem()
    calls: list[tuple[int, int]] = []
    from sklearn.decomposition import PCA

    original = PCA.fit_transform

    def recorded_fit(projector, values, targets=None):
        calls.append(values.shape)
        return original(projector, values, targets)

    monkeypatch.setattr(PCA, "fit_transform", recorded_fit)

    first = _fit_nonlinear_problem(features, rows, target)
    second = _fit_nonlinear_problem(features, rows, target)

    assert calls == [(100, 8), (100, 8)]
    first_mlp = next(record for record in first[1] if record["model"] == "mlp")
    second_mlp = next(record for record in second[1] if record["model"] == "mlp")
    assert first_mlp["readout_id"] == second_mlp["readout_id"]
    assert first_mlp["selection_error"] == second_mlp["selection_error"]


def test_mlp_that_misses_validation_baseline_has_no_test_report():
    features, rows, target = _nonlinear_geometry_problem()
    zeros = np.zeros_like(target.values)
    zero_target = GeometryTarget(
        target.name,
        target.kind,
        target.basis,
        zeros,
        zeros,
        {"physical_zero": zeros},
    )

    _, selections, predictions, _ = _fit_nonlinear_problem(features, rows, zero_target)

    selected = next(record for record in selections if record["model"] == "mlp")
    assert not selected["validation_gate_passed"]
    assert not selected["test_reported"]
    assert not selected["promoted"]
    assert selected["test_error"] is None
    assert {
        record["split"] for record in predictions if record["model"] == "mlp"
    } == {"validation"}
    intervals = _geometry_confidence_table(pd.DataFrame.from_records(selections))
    assert set(intervals.loc[intervals["model"] == "mlp", "split"]) == {"validation"}


def test_ridge_only_candidates_keep_legacy_test_reporting():
    features, rows, target = _nonlinear_geometry_problem()

    candidates, selections, predictions, _ = _fit_nonlinear_problem(
        features, rows, target, models=("ridge",)
    )

    assert len(candidates) == 2
    assert all(record["model"] == "ridge" for record in candidates)
    assert all(record["test_error"] is not None for record in candidates)
    assert selections[0]["test_reported"]
    assert {record["split"] for record in predictions} == {"validation", "test"}


def test_saved_geometry_readout_reconstructs_predictions(tmp_path):
    features, rows, target = _nonlinear_geometry_problem()
    candidates, selections, predictions, states = _fit_nonlinear_problem(
        features, rows, target
    )
    dataset = create_synthetic_trace_dataset(
        tmp_path / "dataset", num_episodes=1, timesteps=4
    )
    spec = _normalize_spec(
        {
            "name": "replayable nonlinear geometry",
            "object_column": "primary_target_object",
            "features": [{"name": "synthetic"}],
            "targets": ["position_world"],
            "probe": {
                "models": ["ridge", "mlp"],
                "pca_dims": [4, 8],
                "ridge_alphas": [1.0],
            },
        }
    )

    artifact = _save_geometry_study(
        dataset,
        spec,
        pd.DataFrame.from_records(candidates),
        pd.DataFrame.from_records(selections),
        pd.DataFrame.from_records(predictions),
        states,
        {"total_seconds": 0.1},
        [dataset.bundles[0].manifest.trace_id],
    )

    artifact_dir = dataset._dataset_artifact_root() / "artifacts" / artifact.artifact_id
    manifest = json.loads(
        (artifact_dir / "fitted_readouts.json").read_text(encoding="utf-8")
    )
    assert all(
        value.startswith("sha256:")
        for value in manifest[0]["array_fingerprints"].values()
    )
    with np.load(artifact_dir / "fitted_arrays.npz") as saved_arrays:
        replayed = predict_geometry_readout(manifest[0], saved_arrays, features)
    local_state = {state.readout_id: state for state in states}[manifest[0]["readout_id"]]
    expected = predict_geometry_readout(
        local_state.contract, local_state.arrays, features
    )
    np.testing.assert_allclose(replayed, expected, atol=1e-12)
    assert (artifact_dir / "confidence_intervals.parquet").exists()
    assert set(pd.read_parquet(artifact_dir / "confidence_intervals.parquet")["model"]) == {
        "ridge",
        "mlp",
    }
