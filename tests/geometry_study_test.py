from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from tests._support.object_flow_dataset import object_flow_dataset
from vla_lens.probes.geometry_study import (
    GeometryTarget,
    _apply_split_contract,
    _decode_orientation,
    _encode_orientation,
    _target_metrics,
    geometry_target_table,
)


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
