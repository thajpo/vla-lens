from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _export(module_name: str, name: str):
    return getattr(importlib.import_module(module_name), name)


def test_public_backend_import_paths_remain_stable():
    import vla_lens
    from vla_lens import TraceBundle as PublicTraceBundle
    from vla_lens import TraceDataset as PublicTraceDataset
    from vla_lens import write_lerobot_trace_record as public_write_lerobot_trace_record
    from vla_lens.lerobot_dataset import (
        LeRobotEpisodeBundle,
        is_lerobot_dataset_root,
        open_lerobot_dataset,
        write_lerobot_trace_record,
    )
    from vla_lens.pi05.capture import main, parse_args, run_pi05_capture_task
    from vla_lens.probes.workflow import (
        SavedProbeSuite,
        dump_probe_spec,
        load_probe_spec,
        normalize_probe_spec,
        train_probe_artifact,
        train_probe_artifact_from_spec,
    )
    from vla_lens.server import run_dashboard_server
    from vla_lens.traces import TraceBundle, TraceDataset

    assert vla_lens.TraceDataset is PublicTraceDataset is TraceDataset
    assert vla_lens.TraceBundle is PublicTraceBundle is TraceBundle
    assert public_write_lerobot_trace_record is write_lerobot_trace_record
    assert callable(run_dashboard_server)
    assert callable(is_lerobot_dataset_root)
    assert callable(open_lerobot_dataset)
    assert callable(main)
    assert callable(parse_args)
    assert callable(run_pi05_capture_task)
    assert callable(train_probe_artifact)
    assert callable(train_probe_artifact_from_spec)
    assert callable(normalize_probe_spec)
    assert callable(load_probe_spec)
    assert callable(dump_probe_spec)
    assert hasattr(LeRobotEpisodeBundle, "array")
    assert hasattr(SavedProbeSuite, "__dataclass_fields__")


@pytest.mark.parametrize(
    ("public_module", "public_name", "canonical_module", "canonical_name"),
    [
        (
            "vla_lens.probes",
            "normalize_probe_spec",
            "vla_lens.probes.workflow_spec",
            "normalize_probe_spec",
        ),
        (
            "vla_lens.probes",
            "train_probe_artifact_from_spec",
            "vla_lens.probes.workflow_training",
            "train_probe_artifact_from_spec",
        ),
        (
            "vla_lens.probes.workflow",
            "SavedProbeSuite",
            "vla_lens.probes.workflow_types",
            "SavedProbeSuite",
        ),
        (
            "vla_lens.probes.workflow",
            "baseline_columns",
            "vla_lens.probes.workflow_spec",
            "baseline_columns",
        ),
        (
            "vla_lens.probes.workflow",
            "normalize_probe_spec",
            "vla_lens.probes.workflow_spec",
            "normalize_probe_spec",
        ),
        (
            "vla_lens.probes.workflow",
            "train_probe_artifact",
            "vla_lens.probes.workflow_training",
            "train_probe_artifact",
        ),
        (
            "vla_lens.probes.workflow",
            "train_probe_artifact_from_spec",
            "vla_lens.probes.workflow_training",
            "train_probe_artifact_from_spec",
        ),
    ],
)
def test_probe_public_facades_alias_canonical_implementations(
    public_module: str,
    public_name: str,
    canonical_module: str,
    canonical_name: str,
):
    assert _export(public_module, public_name) is _export(canonical_module, canonical_name)


def test_probe_workflow_facade_does_not_reexport_private_helpers():
    import vla_lens.probes.workflow as workflow

    assert not hasattr(workflow, "_apply_missing_policy")
    assert not hasattr(workflow, "_resolve_probe_target")


def test_probe_workflow_helper_contracts_used_by_server_modules():
    from vla_lens.probes.workflow_artifacts import _value_counts
    from vla_lens.probes.workflow_prepare import (
        _apply_missing_policy,
        _apply_row_filters,
        _ensure_split,
    )
    from vla_lens.probes.workflow_spec import baseline_columns, normalize_probe_spec
    from vla_lens.probes.workflow_targets import (
        _normalize_target_spec,
        _target_name,
    )

    spec = normalize_probe_spec(
        {
            "target": "outcome",
            "features": {"reduce_tokens": "mean"},
            "probe": "linear",
            "baseline": ["majority_class", "benchmark", "benchmark", "env"],
        }
    )
    assert spec["target"] == {"kind": "outcome"}
    assert spec["features"]["reduction"] == "mean"
    assert spec["features"]["policy_calls"] == "all"
    assert spec["features"]["dtype"] == "float32"
    assert spec["probe"]["models"] == ["linear"]
    assert baseline_columns(spec["baseline"]) == ["benchmark", "env_id"]

    row_target = _normalize_target_spec("outcome")
    assert row_target == {
        "kind": "outcome",
        "name": "outcome",
        "source": "row",
        "column": "outcome",
    }
    assert _target_name({"source": "evaluation", "metric_name": "reward"}) == "evaluation"

    rows = pd.DataFrame(
        {
            "trace_id": ["a", "b", "c"],
            "task_id": [0, 0, 1],
            "score": [0.1, 0.7, 0.9],
            "label": ["keep", "drop", "keep"],
            "target": [1.0, np.nan, ""],
        }
    )
    split = _ensure_split(
        rows,
        "split",
        train_value="train",
        test_value="test",
        split_kind="random_episode",
    )
    assert split["split"].tolist() == ["train", "train", "test"]

    X = np.arange(12, dtype=np.float32).reshape(3, 4)
    filtered_X, filtered_rows, filter_summary = _apply_row_filters(
        X,
        rows,
        {
            "all": [
                {"column": "label", "op": "==", "value": "keep"},
                {"column": "score", "op": ">=", "value": 0.5},
            ]
        },
    )
    assert filtered_X.tolist() == [X[2].tolist()]
    assert filtered_rows["trace_id"].tolist() == ["c"]
    assert filter_summary["output_rows"] == 1

    kept_X, kept_rows, missing_summary = _apply_missing_policy(
        X,
        rows,
        "target",
        policy="drop",
    )
    assert kept_X.tolist() == [X[0].tolist()]
    assert kept_rows["trace_id"].tolist() == ["a"]
    assert missing_summary == {
        "policy": "drop",
        "missing_rows": 2,
        "input_rows": 3,
        "output_rows": 1,
    }
    with pytest.raises(ValueError, match="Probe skipped"):
        _apply_missing_policy(X, rows, "target", policy="skip_probe")
    assert _value_counts(pd.Series(["success", "failure", "success"])) == {
        "success": 2,
        "failure": 1,
    }


def test_context_capture_public_extractors_are_composable():
    from vla_lens.pi05.context_capture import (
        ContextCaptureResult,
        extract_camera_context,
        extract_env_metadata,
        extract_object_context,
        extract_robot_arrays,
    )

    observations = [
        {
            "robot0_joint_pos": np.array([[0.0, 0.1]], dtype=np.float32),
            "robot0_eef_quat": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
            "mug_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "mug_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "mug_grasped": np.array([1.0], dtype=np.float32),
            "front_image": np.zeros((1, 8, 9, 3), dtype=np.uint8),
        },
        {
            "robot0_joint_pos": np.array([[0.2, 0.3]], dtype=np.float32),
            "robot0_eef_quat": np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
            "mug_pos": np.array([0.4, 0.5, 0.6], dtype=np.float32),
            "mug_quat": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "mug_grasped": np.array([0.0], dtype=np.float32),
            "front_image": np.zeros((1, 8, 9, 3), dtype=np.uint8),
        },
    ]

    class Env:
        task_name = "pick mug"
        task_description = "pick the mug"
        camera_names = ["front"]
        camera_heights = {"front": 8}
        camera_widths = {"front": 9}
        object_names = ["mug"]

    robot_arrays = extract_robot_arrays(observations)
    env_table, env_arrays = extract_env_metadata(Env())
    object_table, object_arrays = extract_object_context(observations, Env())
    camera_table, camera_arrays = extract_camera_context(observations, Env())

    assert robot_arrays["robot_joint_pos"].array.shape == (2, 2)
    assert robot_arrays["eef_mat"].metadata["source"] == "derived:eef_quat"
    assert env_table.loc[env_table["field"].eq("task_name"), "available"].item() is True
    assert env_arrays == {}
    assert object_table["object_name"].tolist() == ["mug"]
    assert object_arrays["scene_object_pos"].array.shape == (2, 1, 3)
    assert object_arrays["scene_predicates"].metadata["predicate_names"] == ["mug_grasped"]
    assert camera_table["camera_name"].tolist() == ["front"]
    assert camera_arrays["camera_resolution"].array.tolist() == [[8, 9]]

    result = ContextCaptureResult(
        arrays={**robot_arrays, **object_arrays, **camera_arrays},
        tables={
            "context_availability": pd.DataFrame(
                {
                    "component": ["robot", "camera"],
                    "field": ["robot_joint_pos", "intrinsics"],
                    "available": [True, False],
                }
            )
        },
    )
    assert result.unavailable["field"].tolist() == ["intrinsics"]
