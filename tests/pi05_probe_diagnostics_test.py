from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import report_pi05_probe_diagnostics as diagnostics


def test_readout_battery_exports_target_scoped_episode_rows():
    prepared = _prepared_probe_data()

    result = diagnostics.readout_battery_from_prepared(
        prepared,
        target_names=["next_manipulated_object", "task_phase"],
        max_iter=100,
        seed=0,
        top_k=3,
        model_name="ridge",
    )
    browser = diagnostics._error_browser_frame(result.predictions)

    assert set(result.metrics["target"]) == {"next_manipulated_object", "task_phase"}
    assert "trained_probe_id" in result.metrics.columns
    assert result.metrics["trained_probe_id"].str.contains("NMO-L").any()
    assert result.metrics["trained_probe_id"].str.contains("TPH-L").any()
    assert set(result.layer_metrics["target"]) == {"next_manipulated_object", "task_phase"}
    assert set(result.per_class["target"]) == {"next_manipulated_object", "task_phase"}
    assert set(result.confusion["target"]) == {"next_manipulated_object", "task_phase"}
    assert set(result.lead_time["target"]) == {"next_manipulated_object", "task_phase"}
    assert set(result.supports["policy_call_support_by_class_split"]["target"]) == {
        "next_manipulated_object",
        "task_phase",
    }
    assert set(browser["target"]) == {"next_manipulated_object", "task_phase"}
    assert "trained_probe_id" in browser.columns
    assert "readout_id" in browser.columns
    assert browser["trained_probe_id"].str.contains("NMO-L").any()
    assert browser["trained_probe_id"].str.contains("TPH-L").any()
    assert browser["readout_id"].str.contains("target=").sum() == 0
    assert browser["readout_id"].str.contains("next_manipulated_object").any()
    assert browser["readout_id"].str.contains("task_phase").any()


def test_selection_aware_null_exports_target_scoped_controls():
    prepared = _prepared_probe_data()

    null_frame = diagnostics.selection_aware_null(
        prepared,
        target_names=["next_manipulated_object", "task_phase"],
        shuffles=2,
        max_iter=100,
        seed=0,
        top_k=3,
        model_name="ridge",
    )

    assert set(null_frame["target"]) == {"next_manipulated_object", "task_phase"}
    assert null_frame.groupby("target")["run"].nunique().to_dict() == {
        "next_manipulated_object": 2,
        "task_phase": 2,
    }


def _prepared_probe_data() -> diagnostics.PreparedProbeData:
    policy_calls = [
        ("episode-0", 0, "train", "cup", "approach"),
        ("episode-1", 1, "train", "cube", "grasp"),
        ("episode-2", 2, "train", "cup", "approach"),
        ("episode-3", 3, "train", "cube", "grasp"),
        ("episode-4", 4, "validation", "cup", "approach"),
        ("episode-5", 5, "test", "cube", "grasp"),
    ]
    rows = []
    features = []
    for trace_id, call, split, obj, phase in policy_calls:
        for layer in [0, 1]:
            rows.append(
                {
                    "trace_id": trace_id,
                    "episode_id": trace_id,
                    "task_id": "synthetic",
                    "prompt": "pick the object",
                    "timestep": call * 10,
                    "observation_timestep": call * 10,
                    "policy_call_index": call,
                    "layer": layer,
                    "split": split,
                    "next_manipulated_object": obj,
                    "task_phase": phase,
                    "first_contact_time_next_object": call * 10 + 20,
                    "first_motion_time_next_object": call * 10 + 10,
                }
            )
            obj_feature = 1.0 if obj == "cup" else -1.0
            phase_feature = 1.0 if phase == "approach" else -1.0
            features.append([obj_feature, phase_feature, float(layer)])
    return diagnostics.PreparedProbeData(
        dataset=None,
        X=np.asarray(features, dtype=np.float32),
        rows=pd.DataFrame.from_records(rows),
        target="next_manipulated_object",
        train_value="train",
        selection_value="validation",
        test_value="test",
        eval_values=["validation", "test"],
        split_column="split",
        filter_summary={},
        missing_summary={},
        cache_key="synthetic",
    )
