from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from vla_lens.capture import (
    LEROBOT_INFO_PATH,
    LEROBOT_TASKS_JSONL_PATH,
    VLA_LENS_OVERLAY_REFERENCES,
    is_lerobot_robot_field,
    lerobot_required_metadata_paths,
    lerobot_task_metadata_paths,
    validate_lerobot_v3_dataset,
    vla_lens_overlay_path,
)


def test_lerobot_v3_contract_paths_and_canonical_fields(tmp_path):
    assert LEROBOT_INFO_PATH == Path("meta/info.json")
    assert LEROBOT_TASKS_JSONL_PATH in lerobot_task_metadata_paths()
    assert Path("meta/stats.json") in lerobot_required_metadata_paths()
    assert vla_lens_overlay_path(tmp_path, "tables", "model_sites.parquet") == (
        tmp_path / "vla_lens" / "tables" / "model_sites.parquet"
    )

    assert is_lerobot_robot_field("action")
    assert is_lerobot_robot_field("observation.state")
    assert is_lerobot_robot_field("observation.images.main")
    assert not is_lerobot_robot_field("executed_actions")
    assert not is_lerobot_robot_field("frames.main")
    assert not is_lerobot_robot_field("action_chunks")


def test_valid_minimal_lerobot_v3_root_passes(tmp_path):
    _write_minimal_lerobot_v3_root(tmp_path)

    result = validate_lerobot_v3_dataset(tmp_path)

    assert result.valid, result.to_dict()


def test_missing_lerobot_info_fails(tmp_path):
    _write_minimal_lerobot_v3_root(tmp_path)
    (tmp_path / LEROBOT_INFO_PATH).unlink()

    result = validate_lerobot_v3_dataset(tmp_path)

    assert not result.valid
    assert "missing_metadata" in _error_codes(result)


def test_missing_action_field_fails(tmp_path):
    _write_minimal_lerobot_v3_root(tmp_path, include_action=False)

    result = validate_lerobot_v3_dataset(tmp_path)

    assert not result.valid
    assert "missing_step_field" in _error_codes(result)


def test_overlay_references_unknown_episode_fail(tmp_path):
    _write_minimal_lerobot_v3_root(tmp_path)
    table_path = tmp_path / VLA_LENS_OVERLAY_REFERENCES
    table_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "overlay_id": ["model-site-0"],
            "episode_index": [99],
            "frame_index": [0],
        }
    ).to_parquet(table_path, index=False)

    result = validate_lerobot_v3_dataset(tmp_path)

    assert not result.valid
    assert "overlay_unknown_episode" in _error_codes(result)


def test_overlay_references_out_of_range_frame_fail(tmp_path):
    _write_minimal_lerobot_v3_root(tmp_path)
    table_path = tmp_path / VLA_LENS_OVERLAY_REFERENCES
    table_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "overlay_id": ["policy-call-0"],
            "episode_index": [0],
            "frame_index": [2],
        }
    ).to_parquet(table_path, index=False)

    result = validate_lerobot_v3_dataset(tmp_path)

    assert not result.valid
    assert "overlay_unknown_frame" in _error_codes(result)


def _write_minimal_lerobot_v3_root(root: Path, *, include_action: bool = True) -> None:
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "main" / "chunk-000").mkdir(parents=True)

    features: dict[str, object] = {
        "episode_index": {"dtype": "int64"},
        "frame_index": {"dtype": "int64"},
        "timestamp": {"dtype": "float32"},
        "task_index": {"dtype": "int64"},
        "observation.state": {"dtype": "float32", "shape": [2]},
        "observation.images.main": {
            "dtype": "video",
            "shape": [64, 64, 3],
            "names": ["height", "width", "channel"],
        },
    }
    if include_action:
        features["action"] = {"dtype": "float32", "shape": [1]}

    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 30,
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": (
                    "videos/{video_key}/chunk-{episode_chunk:03d}/"
                    "episode_{episode_index:06d}.mp4"
                ),
                "features": features,
            }
        ),
        encoding="utf-8",
    )
    (root / "meta" / "stats.json").write_text("{}", encoding="utf-8")
    (root / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "pick the cube"}) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "episode_index": [0],
            "length": [2],
            "task_index": [0],
        }
    ).to_parquet(
        root / "meta" / "episodes" / "chunk-000" / "episode_000000.parquet",
        index=False,
    )

    rows = {
        "episode_index": [0, 0],
        "frame_index": [0, 1],
        "timestamp": [0.0, 1.0 / 30.0],
        "task_index": [0, 0],
        "observation.state": [[0.0, 0.0], [0.1, 0.1]],
    }
    if include_action:
        rows["action"] = [[0.0], [0.1]]
    pd.DataFrame(rows).to_parquet(
        root / "data" / "chunk-000" / "episode_000000.parquet",
        index=False,
    )
    (root / "videos" / "main" / "chunk-000" / "episode_000000.mp4").write_bytes(b"")


def _error_codes(result) -> set[str]:
    return {issue.code for issue in result.errors}
