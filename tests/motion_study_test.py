from __future__ import annotations

import numpy as np
import pandas as pd

from vla_lens.probes.motion_study import (
    _comparison_table,
    _geometry_row_diagnostics,
    _normalize_motion_spec,
)


def test_position_diagnostics_separate_distance_and_direction():
    truth = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    predicted = np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    result = _geometry_row_diagnostics("position", truth, predicted)

    np.testing.assert_allclose(result["vector_error"], [1.0, np.sqrt(18.0)])
    np.testing.assert_allclose(result["magnitude_error"], [1.0, 0.0])
    np.testing.assert_allclose(result["direction_error"], [0.0, 90.0])


def test_rotation_diagnostics_use_physical_angles():
    truth = np.deg2rad(np.array([[0.0, 0.0, 30.0]]))
    predicted = np.deg2rad(np.array([[0.0, 0.0, 20.0]]))

    result = _geometry_row_diagnostics("rotation", truth, predicted)

    np.testing.assert_allclose(result["vector_error"], [10.0], atol=1e-8)
    np.testing.assert_allclose(result["magnitude_error"], [10.0], atol=1e-8)
    np.testing.assert_allclose(result["direction_error"], [0.0], atol=1e-8)


def test_comparison_table_uses_paired_task_advantage():
    rows = []
    for task, activation, comparison in [("a", 1.0, 3.0), ("b", 2.0, 4.0)]:
        for model, error in [("activation", activation), ("context", comparison)]:
            rows.append(
                {
                    "selected_feature_id": "expert",
                    "analysis": "moving_geometry",
                    "target": "position",
                    "threshold": 0.1,
                    "split": "test",
                    "model": model,
                    "trace_id": task,
                    "timestep": 1,
                    "task_key": task,
                    "vector_error": error,
                }
            )

    result = _comparison_table(pd.DataFrame.from_records(rows))

    assert len(result) == 1
    assert result.iloc[0]["activation_advantage"] == 2.0
    assert result.iloc[0]["ci_low"] == 2.0
    assert result.iloc[0]["ci_high"] == 2.0


def test_motion_spec_has_locked_physical_thresholds():
    normalized = _normalize_motion_spec(
        {
            "features": [{"id": "x", "name": "site"}],
            "baseline": ["benchmark", "task_id"],
        }
    )

    assert normalized["movement"] == {
        "position": [0.01, 0.10],
        "rotation": [1.0, 15.0],
    }
    assert normalized["baseline_columns"] == ["benchmark", "task_id"]
