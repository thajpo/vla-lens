from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from vla_lens import ArraySpec, TraceDataset, TraceManifest, validate_lerobot_v3_dataset
from vla_lens.traces import ModelSiteSpec, TraceBundle
from vla_lens.vlatrace_migration import copy_dataset_level_state, migrate_vlatrace_bundle


def test_migrate_vlatrace_bundle_writes_lerobot_root_and_pruned_overlay(tmp_path):
    source_root = tmp_path / "legacy"
    source = _legacy_bundle(source_root / "trace-a.vlatrace")
    output = tmp_path / "lerobot"

    result = migrate_vlatrace_bundle(source.path, output, source_root=source_root)
    dataset = TraceDataset.open(output)
    opened = dataset.bundle("trace-a")
    validation = validate_lerobot_v3_dataset(output)
    overlay_path = output / "vla_lens" / "episodes" / "episode_000000"
    overlay_arrays = pd.read_parquet(overlay_path / TraceBundle.ARRAY_INDEX)

    assert validation.valid, validation.to_dict()
    assert result.episode_index == 0
    assert opened.actions().shape == (3, 2)
    assert opened.array("observation.state").shape == (3, 3)
    assert opened.frames("main").shape[:3] == (3, 16, 16)
    assert opened.action_chunks().shape == (1, 2, 2)
    assert opened.model_site("model.layer0.hidden").shape == (1, 2, 4)
    assert "executed_actions" not in set(overlay_arrays["name"].astype(str))
    assert "frames.main" not in set(overlay_arrays["name"].astype(str))
    assert "action_chunks" in set(overlay_arrays["name"].astype(str))
    assert not (overlay_path / "arrays" / "action" / "executed_actions.zarr").exists()
    assert not (overlay_path / "media" / "frames" / "main").exists()
    assert (overlay_path / TraceBundle.FINGERPRINTS).exists()


def test_copy_dataset_level_state_moves_artifacts_and_workbench_under_overlay(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    (source / "tables").mkdir(parents=True)
    (source / "artifacts" / "artifact-a").mkdir(parents=True)
    (source / "workbench" / "analysis_runs").mkdir(parents=True)
    pd.DataFrame.from_records(
        [{"artifact_id": "artifact-a", "path": "artifacts/artifact-a/artifact.json"}]
    ).to_parquet(source / TraceBundle.ARTIFACT_INDEX, index=False)
    (source / "artifacts" / "artifact-a" / "artifact.json").write_text("{}", encoding="utf-8")
    (source / "workbench" / "analysis_runs" / "run.json").write_text(
        json.dumps({"run_id": "run"}),
        encoding="utf-8",
    )

    copy_dataset_level_state(source, output)

    assert (output / "vla_lens" / TraceBundle.ARTIFACT_INDEX).exists()
    assert (output / "vla_lens" / "artifacts" / "artifact-a" / "artifact.json").exists()
    assert (output / "vla_lens" / "workbench" / "analysis_runs" / "run.json").exists()


def _legacy_bundle(path: Path) -> TraceBundle:
    length = 3
    manifest = TraceManifest(
        trace_id="trace-a",
        episode_id="trace-a",
        task_id="task-a",
        prompt="pick the cube",
        model_id="test-model",
        env_id="test-env",
        robot_id="test-robot",
        outcome="success",
        length=length,
        metadata={
            "capture_profile": "mechanistic_sampled",
            "task_name": "pick the cube",
            "action_space": {"action_names": ["x", "y"]},
        },
    )
    frames = np.zeros((length, 16, 16, 3), dtype=np.uint8)
    frames[-1, :, :, 0] = 255
    return TraceBundle.create(
        path,
        manifest=manifest,
        timesteps=pd.DataFrame(
            {
                "timestep": np.arange(length, dtype=np.int64),
                "reward": np.linspace(0.0, 1.0, length, dtype=np.float32),
                "done": [False, False, True],
            }
        ),
        episode_arrays={
            "executed_actions": ArraySpec(
                np.asarray([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]], dtype=np.float32),
                ["timestep", "action_dim"],
            ),
            "frames.main": ArraySpec(
                frames,
                ["timestep", "height", "width", "channel"],
            ),
            "robot_joint_pos": ArraySpec(
                np.zeros((length, 3), dtype=np.float32),
                ["timestep", "joint"],
            ),
            "action_chunks": ArraySpec(
                np.zeros((1, 2, 2), dtype=np.float32),
                ["policy_call", "horizon", "action_dim"],
            ),
        },
        model_arrays=[
            ModelSiteSpec(
                name="model.layer0.hidden",
                array=np.zeros((1, 2, 4), dtype=np.float32),
                axes=["policy_call", "token", "channel"],
                module="model.layer0",
                layer=0,
            )
        ],
        policy_calls=pd.DataFrame(
            {
                "policy_call_index": [0],
                "observation_timestep": [0],
                "env_timestep_start": [0],
                "env_timestep_end": [2],
            }
        ),
        capture_report={"missing_model_sites": []},
    )
