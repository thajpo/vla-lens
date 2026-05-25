from __future__ import annotations

import csv
from pathlib import Path

from vla_lens.pi05.batch_capture import (
    _capture_commands,
    _episode_rows_from_config,
    _read_episode_plan,
    _write_plan_files,
)
from vla_lens.pi05.plan_capture import RECYCLE_EXIT_CODE, capture_args_from_command, parse_args


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
        "traces/pi05-test-dataset/mechanistic_sampled/libero_object/task_00"
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


def test_batch_capture_can_group_noncontiguous_seed_list(tmp_path):
    plan = tmp_path / "episodes.csv"
    plan.write_text(
        "\n".join(
            [
                "dataset_id,benchmark,task_id,seed,split,capture_profile",
                "dataset-a,libero_goal,1,1000,train,mechanistic_sampled",
                "dataset-a,libero_goal,1,2000,test,mechanistic_sampled",
                "dataset-a,libero_goal,1,3000,test,mechanistic_sampled",
            ]
        ),
        encoding="utf-8",
    )

    config = {**_config(tmp_path), "group_seed_list": True}
    rows = _read_episode_plan(plan)
    commands = _capture_commands(config, tmp_path, rows)

    assert len(commands) == 1
    assert commands[0].episodes == 3
    assert commands[0].start_seed == 1000
    assert "--seed-list" in commands[0].command
    assert "1000,2000,3000" in commands[0].command


def test_batch_capture_runtime_env_overrides_stale_config_values(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setenv("VLA_LENS_CAPTURE_PYTHON", "/tmp/pi05/bin/python")
    monkeypatch.setenv("VLA_LENS_CAPTURE_PYTHONPATH", "/tmp/vla-lens/src")
    monkeypatch.setenv("VLA_LENS_CAPTURE_DEVICE", "mps")
    monkeypatch.setenv("VLA_LENS_CAPTURE_DTYPE", "float32")

    rows = _episode_rows_from_config(config)
    command = _capture_commands(config, tmp_path, rows)[0].command

    assert command[:3] == ("env", "PYTHONPATH=/tmp/vla-lens/src", "/tmp/pi05/bin/python")
    assert command[command.index("--device") + 1] == "mps"
    assert command[command.index("--dtype") + 1] == "float32"


def test_plan_capture_reuses_batch_capture_command_args(tmp_path):
    plan = tmp_path / "episodes.csv"
    plan.write_text(
        "\n".join(
            [
                "dataset_id,benchmark,task_id,seed,split,capture_profile",
                "dataset-a,libero_goal,1,1000,train,mechanistic_sampled",
                "dataset-a,libero_goal,1,2000,test,mechanistic_sampled",
            ]
        ),
        encoding="utf-8",
    )

    config = {**_config(tmp_path), "group_seed_list": True}
    rows = _read_episode_plan(plan)
    command = _capture_commands(config, tmp_path, rows)[0]
    args = capture_args_from_command(command.command)

    assert args.model_id == "lerobot/pi05_libero_finetuned"
    assert args.benchmark == "libero_goal"
    assert args.task_id == 1
    assert args.episodes == 2
    assert args.seed_list == "1000,2000"
    assert args.capture_profile == "mechanistic_sampled"
    assert str(args.vlatrace_out_root).endswith(
        "traces/dataset-a/mechanistic_sampled/libero_goal/task_01"
    )


def test_plan_capture_accepts_recycle_limit():
    args = parse_args(
        [
            "--episode-plan",
            "episodes.csv",
            "--max-executed-commands",
            "100",
        ]
    )

    assert args.max_executed_commands == 100
    assert RECYCLE_EXIT_CODE == 75


def test_rocm_plan_supervisor_restarts_only_for_recycle_exit():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts/pi05_supervise_plan_capture_rocm.sh"
    )
    script = script_path.read_text(encoding="utf-8")

    assert 'RECYCLE_EXIT_CODE="${PI05_CAPTURE_RECYCLE_EXIT_CODE:-75}"' in script
    assert 'if [[ "$status" != "$RECYCLE_EXIT_CODE" ]]; then' in script
    assert "not restarting" in script
    assert 'exit "$status"' in script


def test_batch_capture_reads_paired_counterfactual_plan(tmp_path):
    plan = tmp_path / "episodes.csv"
    plan.write_text(
        "\n".join(
            [
                ",".join(
                    [
                        "dataset_id",
                        "benchmark",
                        "task_id",
                        "seed",
                        "split",
                        "capture_profile",
                        "counterfactual_group_id",
                        "counterfactual_role",
                        "counterfactual_type",
                        "changed_fields",
                        "matched_fields",
                        "target_object_id",
                    ]
                ),
                (
                    "dataset-a,libero_goal,1,42,train,mechanistic_sampled,"
                    "group-1,clean,prompt_target_swap,prompt.target_object,"
                    '"benchmark,task_id,seed",mug'
                ),
                (
                    "dataset-a,libero_goal,1,42,train,mechanistic_sampled,"
                    "group-1,corrupt,prompt_target_swap,prompt.target_object,"
                    '"benchmark,task_id,seed",bowl'
                ),
            ]
        ),
        encoding="utf-8",
    )

    rows = _read_episode_plan(plan)
    commands = _capture_commands(_config(tmp_path), tmp_path, rows)
    _write_plan_files(tmp_path, config=_config(tmp_path), rows=rows)

    assert [row.expected_trace_id for row in rows] == [
        "pi05_mechanistic_sampled_libero_goal_task1_seed42_clean",
        "pi05_mechanistic_sampled_libero_goal_task1_seed42_corrupt",
    ]
    assert rows[0].paired_trace_id == rows[1].expected_trace_id
    assert rows[1].paired_trace_id == rows[0].expected_trace_id
    assert rows[0].capture_design == "paired_counterfactual"
    assert len(commands) == 2
    clean_command = commands[0].command
    assert "--capture-design" in clean_command
    assert "paired_counterfactual" in clean_command
    assert "--trace-variant" in clean_command
    assert "clean" in clean_command
    assert "--paired-trace-id" in clean_command
    assert rows[1].expected_trace_id in clean_command

    with (tmp_path / "episode_plan.csv").open(newline="", encoding="utf-8") as handle:
        plan_rows = list(csv.DictReader(handle))
    assert plan_rows[0]["trace_variant"] == "clean"
    assert plan_rows[0]["counterfactual_role"] == "clean"
    assert plan_rows[0]["expected_trace_path"].endswith(
        "traces/dataset-a/mechanistic_sampled/libero_goal/task_01"
    )
