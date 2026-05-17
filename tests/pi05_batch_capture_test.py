from __future__ import annotations

import csv

from vla_lens.pi05.batch_capture import (
    _capture_commands,
    _episode_rows_from_config,
    _read_episode_plan,
    _write_plan_files,
)


def _config(tmp_path):
    return {
        "name": "ignored-name",
        "dataset_id": "pi05-test-dataset",
        "output_root": str(tmp_path),
        "model_id": "lerobot/pi05_libero_finetuned",
        "python_executable": "python",
        "pythonpath": "src",
        "capture_profile": "mechanistic_sampled",
        "storage_dtype": "float16",
        "obs_size": 256,
        "device": "cuda",
        "dtype": "bfloat16",
        "start_seed": 10,
        "seeds_per_task": 2,
        "estimated_mb_per_episode": 1,
        "minimum_free_gb_after_capture": 0,
        "benchmarks": ["libero_object"],
        "task_ids": {"train_seen_task": [0]},
        "seed_splits_for_train_seen_tasks": {"train": [10], "test": [11]},
    }


def test_batch_capture_generates_dataset_id_episode_plan(tmp_path):
    config = _config(tmp_path)

    rows = _episode_rows_from_config(config)
    commands = _capture_commands(config, tmp_path, rows)
    _write_plan_files(tmp_path, config=config, rows=rows)

    assert [row.dataset_id for row in rows] == ["pi05-test-dataset", "pi05-test-dataset"]
    assert [row.seed for row in rows] == [10, 11]
    assert len(commands) == 1
    assert "--dataset-id" in commands[0].command
    assert "--batch-id" not in commands[0].command
    assert str(commands[0].output_root).endswith(
        "traces/pi05-test-dataset/mechanistic_sampled/libero_object/task_00"
    )

    with (tmp_path / "episode_plan.csv").open(newline="", encoding="utf-8") as handle:
        plan_rows = list(csv.DictReader(handle))
    assert plan_rows[0]["dataset_id"] == "pi05-test-dataset"
    assert plan_rows[0]["expected_trace_path"].endswith(
        "pi05_mechanistic_sampled_libero_object_task0_seed10.vlatrace"
    )

    with (tmp_path / "probe_splits.csv").open(newline="", encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))
    assert [row["split"] for row in split_rows] == ["train", "test"]


def test_batch_capture_reads_explicit_episode_plan_csv(tmp_path):
    plan = tmp_path / "episodes.csv"
    plan.write_text(
        "\n".join(
            [
                "dataset_id,benchmark,task_id,seed,split,capture_profile",
                "dataset-a,libero_goal,1,42,train,rollout",
                "dataset-a,libero_goal,1,44,test,rollout",
            ]
        ),
        encoding="utf-8",
    )

    rows = _read_episode_plan(plan)
    commands = _capture_commands(_config(tmp_path), tmp_path, rows)

    assert [row.seed for row in rows] == [42, 44]
    assert len(commands) == 2
    assert {command.start_seed for command in commands} == {42, 44}
