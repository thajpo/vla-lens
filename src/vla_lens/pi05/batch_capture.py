"""Package-native PI0.5 batch capture runner.

The invariant here is simple: one command runs capture, and the variable part is
data.  A run can be generated from a small YAML matrix or driven directly by an
``episode_plan.csv`` with one row per intended episode.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

DEFAULT_CONFIG = Path("configs/pi05_diverse_500.yaml")
PLAN_COLUMNS = (
    "dataset_id",
    "benchmark",
    "task_id",
    "seed",
    "split",
    "capture_profile",
    "capture_design",
    "trace_variant",
    "counterfactual_group_id",
    "counterfactual_role",
    "counterfactual_type",
    "pair_index",
    "paired_trace_id",
    "changed_fields",
    "matched_fields",
    "target_object_id",
    "counterfactual_target_object_id",
    "expected_trace_id",
    "expected_trace_path",
)
PROBE_SPLIT_COLUMNS = (
    "dataset_id",
    "trace_id",
    "benchmark",
    "task_id",
    "seed",
    "split",
    "capture_profile",
    "capture_design",
    "trace_variant",
    "counterfactual_group_id",
    "counterfactual_role",
    "counterfactual_type",
    "pair_index",
    "paired_trace_id",
    "changed_fields",
    "matched_fields",
    "target_object_id",
    "counterfactual_target_object_id",
)


@dataclass(frozen=True, slots=True)
class EpisodePlanRow:
    dataset_id: str
    benchmark: str
    task_id: int
    seed: int
    split: str
    capture_profile: str
    capture_design: str = "single_trace"
    trace_variant: str = ""
    counterfactual_group_id: str = ""
    counterfactual_role: str = ""
    counterfactual_type: str = ""
    pair_index: int | None = None
    paired_trace_id: str = ""
    changed_fields: tuple[str, ...] = ()
    matched_fields: tuple[str, ...] = ()
    target_object_id: str = ""
    counterfactual_target_object_id: str = ""

    @property
    def expected_trace_id(self) -> str:
        base = f"pi05_{self.capture_profile}_{self.benchmark}_task{self.task_id}_seed{self.seed}"
        suffix = self.trace_suffix
        return f"{base}_{suffix}" if suffix else base

    @property
    def trace_suffix(self) -> str:
        return _trace_variant_suffix(self.trace_variant or self.counterfactual_role)


@dataclass(frozen=True, slots=True)
class CaptureCommand:
    dataset_id: str
    benchmark: str
    task_id: int
    start_seed: int
    episodes: int
    capture_profile: str
    output_root: Path
    expected_paths: tuple[Path, ...]
    command: tuple[str, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--episode-plan",
        type=Path,
        help=(
            "CSV with one row per episode. Required columns: dataset_id, benchmark, "
            "task_id, seed, split, capture_profile. If omitted, the plan is generated "
            "from the YAML matrix config."
        ),
    )
    parser.add_argument("--output-root", type=Path, help="Override config output_root.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute capture. Omit this to write/print the plan only.",
    )
    parser.add_argument(
        "--limit-commands",
        type=int,
        help="Only run/print the first N grouped capture commands.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run commands even if all expected trace directories already exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = _load_config(args.config)
    if args.output_root:
        config["output_root"] = str(args.output_root)

    output_root = Path(str(config["output_root"])).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    rows = (
        _read_episode_plan(args.episode_plan)
        if args.episode_plan is not None
        else _episode_rows_from_config(config)
    )
    rows = _infer_counterfactual_pair_links(rows)
    if not rows:
        raise SystemExit("episode plan is empty")

    commands = _capture_commands(config, output_root, rows)
    if args.limit_commands is not None:
        commands = commands[: max(0, args.limit_commands)]

    _write_plan_files(output_root, config=config, rows=rows)
    _preflight_storage(config, output_root, episode_count=len(rows))

    print(f"dataset_id={_dataset_ids(rows)}")
    print(f"episode_plan={output_root / 'episode_plan.csv'}")
    print(f"probe_splits={output_root / 'probe_splits.csv'}")
    print(f"episodes={len(rows)}")
    print(f"capture_commands={len(commands)}")
    print(f"mode={'run' if args.run else 'dry-run'}")

    if not args.run:
        print("Dry run only. Re-run with --run to execute capture.")
        return

    status_path = output_root / "capture_status.jsonl"
    for index, item in enumerate(commands, start=1):
        expected_exists = item.expected_paths and all(path.exists() for path in item.expected_paths)
        if expected_exists and not args.force:
            _append_status(
                status_path,
                {
                    "status": "skipped_existing",
                    "index": index,
                    "dataset_id": item.dataset_id,
                    "benchmark": item.benchmark,
                    "task_id": item.task_id,
                    "capture_profile": item.capture_profile,
                    "expected_traces": len(item.expected_paths),
                },
            )
            print(f"[{index}/{len(commands)}] skip existing {item.benchmark} task={item.task_id}")
            continue

        _preflight_storage(config, output_root, episode_count=0)
        print(f"[{index}/{len(commands)}] run {' '.join(item.command)}", flush=True)
        result = subprocess.run(list(item.command), cwd=Path.cwd(), check=False)
        _append_status(
            status_path,
            {
                "status": "completed" if result.returncode == 0 else "failed",
                "index": index,
                "dataset_id": item.dataset_id,
                "benchmark": item.benchmark,
                "task_id": item.task_id,
                "capture_profile": item.capture_profile,
                "returncode": result.returncode,
            },
        )
        if result.returncode != 0:
            raise SystemExit(result.returncode)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping config at {path}")
    return payload


def _episode_rows_from_config(config: Mapping[str, Any]) -> list[EpisodePlanRow]:
    dataset_id = _dataset_id(config)
    profile = str(config["capture_profile"])
    start_seed = int(config["start_seed"])
    seeds_per_task = int(config["seeds_per_task"])
    all_seeds = tuple(range(start_seed, start_seed + seeds_per_task))
    seed_splits = {
        int(seed): str(split)
        for split, seeds in dict(config["seed_splits_for_train_seen_tasks"]).items()
        for seed in seeds
    }
    rows: list[EpisodePlanRow] = []
    for benchmark in _as_str_list(config["benchmarks"]):
        for task_group, task_ids in dict(config["task_ids"]).items():
            for task_id in _as_int_list(task_ids):
                for seed in all_seeds:
                    rows.append(
                        EpisodePlanRow(
                            dataset_id=dataset_id,
                            benchmark=benchmark,
                            task_id=task_id,
                            seed=seed,
                            split=(
                                seed_splits[seed]
                                if task_group == "train_seen_task"
                                else str(task_group)
                            ),
                            capture_profile=profile,
                        )
                    )
    return rows


def _read_episode_plan(path: Path) -> list[EpisodePlanRow]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = {
            "dataset_id",
            "benchmark",
            "task_id",
            "seed",
            "split",
            "capture_profile",
        } - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows = [_episode_plan_row_from_record(record) for record in reader]
    return _infer_counterfactual_pair_links(rows)


def _episode_plan_row_from_record(record: Mapping[str, Any]) -> EpisodePlanRow:
    counterfactual_group_id = str(record.get("counterfactual_group_id") or "").strip()
    counterfactual_role = str(record.get("counterfactual_role") or "").strip()
    capture_design = str(record.get("capture_design") or "").strip()
    if not capture_design:
        capture_design = "paired_counterfactual" if counterfactual_group_id else "single_trace"
    return EpisodePlanRow(
        dataset_id=str(record["dataset_id"]).strip(),
        benchmark=str(record["benchmark"]).strip(),
        task_id=int(record["task_id"]),
        seed=int(record["seed"]),
        split=str(record["split"]).strip(),
        capture_profile=str(record["capture_profile"]).strip(),
        capture_design=capture_design,
        trace_variant=str(record.get("trace_variant") or "").strip(),
        counterfactual_group_id=counterfactual_group_id,
        counterfactual_role=counterfactual_role,
        counterfactual_type=str(record.get("counterfactual_type") or "").strip(),
        pair_index=_optional_int(record.get("pair_index")),
        paired_trace_id=str(record.get("paired_trace_id") or "").strip(),
        changed_fields=_parse_list_field(record.get("changed_fields")),
        matched_fields=_parse_list_field(record.get("matched_fields")),
        target_object_id=str(record.get("target_object_id") or "").strip(),
        counterfactual_target_object_id=str(
            record.get("counterfactual_target_object_id") or ""
        ).strip(),
    )


def _capture_commands(
    config: Mapping[str, Any],
    output_root: Path,
    rows: Sequence[EpisodePlanRow],
) -> list[CaptureCommand]:
    commands: list[CaptureCommand] = []
    python_executable = str(
        os.environ.get("VLA_LENS_CAPTURE_PYTHON")
        or config.get("python_executable")
        or sys.executable
    )
    pythonpath = str(
        os.environ.get("VLA_LENS_CAPTURE_PYTHONPATH") or config.get("pythonpath") or ""
    )
    device = str(os.environ.get("VLA_LENS_CAPTURE_DEVICE") or config["device"])
    dtype = str(os.environ.get("VLA_LENS_CAPTURE_DTYPE") or config["dtype"])
    command_prefix = (*_env_prefix(pythonpath),) if pythonpath else ()

    sorted_rows = sorted(
        rows,
        key=lambda row: (*_capture_command_group_key(row), row.seed),
    )
    for _group_key, group_iter in groupby(
        sorted_rows,
        key=_capture_command_group_key,
    ):
        group_rows = list(group_iter)
        first_row = group_rows[0]
        dataset_id = first_row.dataset_id
        profile = first_row.capture_profile
        benchmark = first_row.benchmark
        task_id = first_row.task_id
        if config.get("group_seed_list"):
            seed_groups = [sorted({int(row.seed) for row in group_rows})]
        else:
            seed_groups = _contiguous_seed_groups([row.seed for row in group_rows])
        for seeds in seed_groups:
            seed_set = set(seeds)
            seed_rows = [row for row in group_rows if row.seed in seed_set]
            task_root = _trace_root(output_root, seed_rows[0])
            expected_paths = tuple(_expected_trace_path(output_root, row) for row in seed_rows)
            command = (
                *command_prefix,
                python_executable,
                "-m",
                "vla_lens.pi05.capture",
                "--model-id",
                str(config["model_id"]),
                "--benchmark",
                benchmark,
                "--task-id",
                str(task_id),
                "--episodes",
                str(len(seeds)),
                "--start-seed",
                str(seeds[0]),
                "--capture-profile",
                profile,
                "--storage-dtype",
                str(config["storage_dtype"]),
                "--obs-size",
                str(config["obs_size"]),
                "--device",
                device,
                "--dtype",
                dtype,
                "--vlatrace-out-root",
                str(task_root),
                "--dataset-id",
                dataset_id,
                *_counterfactual_command_args(seed_rows[0]),
            )
            if config.get("group_seed_list"):
                command = (
                    *command,
                    "--seed-list",
                    ",".join(str(seed) for seed in seeds),
                )
            commands.append(
                CaptureCommand(
                    dataset_id=dataset_id,
                    benchmark=benchmark,
                    task_id=int(task_id),
                    start_seed=int(seeds[0]),
                    episodes=len(seeds),
                    capture_profile=profile,
                    output_root=task_root,
                    expected_paths=expected_paths,
                    command=command,
                )
            )
    return commands


def _capture_command_group_key(row: EpisodePlanRow) -> tuple[Any, ...]:
    return (
        row.dataset_id,
        row.capture_profile,
        row.benchmark,
        row.task_id,
        row.capture_design,
        row.trace_variant,
        row.counterfactual_group_id,
        row.counterfactual_role,
        row.counterfactual_type,
        row.pair_index if row.pair_index is not None else -1,
        row.paired_trace_id,
        row.changed_fields,
        row.matched_fields,
        row.target_object_id,
        row.counterfactual_target_object_id,
    )


def _counterfactual_command_args(row: EpisodePlanRow) -> tuple[str, ...]:
    args: list[str] = []
    if row.capture_design and row.capture_design != "single_trace":
        args.extend(["--capture-design", row.capture_design])
    if row.trace_variant:
        args.extend(["--trace-variant", row.trace_variant])
    elif row.counterfactual_role:
        args.extend(["--trace-variant", row.counterfactual_role])
    if row.counterfactual_group_id:
        args.extend(["--counterfactual-group-id", row.counterfactual_group_id])
    if row.counterfactual_role:
        args.extend(["--counterfactual-role", row.counterfactual_role])
    if row.counterfactual_type:
        args.extend(["--counterfactual-type", row.counterfactual_type])
    if row.pair_index is not None:
        args.extend(["--pair-index", str(row.pair_index)])
    if row.paired_trace_id:
        args.extend(["--paired-trace-id", row.paired_trace_id])
    if row.changed_fields:
        args.extend(["--changed-fields", ",".join(row.changed_fields)])
    if row.matched_fields:
        args.extend(["--matched-fields", ",".join(row.matched_fields)])
    if row.target_object_id:
        args.extend(["--target-object-id", row.target_object_id])
    if row.counterfactual_target_object_id:
        args.extend(["--counterfactual-target-object-id", row.counterfactual_target_object_id])
    return tuple(args)


def _write_plan_files(
    output_root: Path,
    *,
    config: Mapping[str, Any],
    rows: Sequence[EpisodePlanRow],
) -> None:
    (output_root / "logs").mkdir(parents=True, exist_ok=True)
    with (output_root / "capture_config.resolved.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(config), handle, indent=2, sort_keys=True)

    plan_rows = [_episode_plan_record(output_root, row) for row in rows]
    _write_csv(output_root / "episode_plan.csv", plan_rows, fieldnames=PLAN_COLUMNS)
    _write_csv(
        output_root / "probe_splits.csv",
        [
            {
                "dataset_id": row.dataset_id,
                "trace_id": row.expected_trace_id,
                "benchmark": row.benchmark,
                "task_id": row.task_id,
                "seed": row.seed,
                "split": row.split,
                "capture_profile": row.capture_profile,
                "capture_design": row.capture_design,
                "trace_variant": row.trace_variant,
                "counterfactual_group_id": row.counterfactual_group_id,
                "counterfactual_role": row.counterfactual_role,
                "counterfactual_type": row.counterfactual_type,
                "pair_index": row.pair_index if row.pair_index is not None else "",
                "paired_trace_id": row.paired_trace_id,
                "changed_fields": _list_cell(row.changed_fields),
                "matched_fields": _list_cell(row.matched_fields),
                "target_object_id": row.target_object_id,
                "counterfactual_target_object_id": row.counterfactual_target_object_id,
            }
            for row in rows
        ],
        fieldnames=PROBE_SPLIT_COLUMNS,
    )
    with (output_root / "episode_plan.json").open("w", encoding="utf-8") as handle:
        json.dump(plan_rows, handle, indent=2)


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _episode_plan_record(output_root: Path, row: EpisodePlanRow) -> dict[str, Any]:
    return {
        "dataset_id": row.dataset_id,
        "benchmark": row.benchmark,
        "task_id": row.task_id,
        "seed": row.seed,
        "split": row.split,
        "capture_profile": row.capture_profile,
        "capture_design": row.capture_design,
        "trace_variant": row.trace_variant,
        "counterfactual_group_id": row.counterfactual_group_id,
        "counterfactual_role": row.counterfactual_role,
        "counterfactual_type": row.counterfactual_type,
        "pair_index": row.pair_index if row.pair_index is not None else "",
        "paired_trace_id": row.paired_trace_id,
        "changed_fields": _list_cell(row.changed_fields),
        "matched_fields": _list_cell(row.matched_fields),
        "target_object_id": row.target_object_id,
        "counterfactual_target_object_id": row.counterfactual_target_object_id,
        "expected_trace_id": row.expected_trace_id,
        "expected_trace_path": str(_expected_trace_path(output_root, row)),
    }


def _trace_root(output_root: Path, row: EpisodePlanRow) -> Path:
    return (
        output_root
        / "traces"
        / row.dataset_id
        / row.capture_profile
        / row.benchmark
        / f"task_{row.task_id:02d}"
    )


def _expected_trace_path(output_root: Path, row: EpisodePlanRow) -> Path:
    return _trace_root(output_root, row) / f"{row.expected_trace_id}.vlatrace"


def _contiguous_seed_groups(seeds: Sequence[int]) -> list[list[int]]:
    sorted_seeds = sorted(set(int(seed) for seed in seeds))
    if not sorted_seeds:
        return []
    groups: list[list[int]] = [[sorted_seeds[0]]]
    for seed in sorted_seeds[1:]:
        if seed == groups[-1][-1] + 1:
            groups[-1].append(seed)
        else:
            groups.append([seed])
    return groups


def _infer_counterfactual_pair_links(rows: Sequence[EpisodePlanRow]) -> list[EpisodePlanRow]:
    normalized = [_normalize_pair_row(row) for row in rows]
    groups: dict[str, list[int]] = {}
    for index, row in enumerate(normalized):
        if row.counterfactual_group_id:
            groups.setdefault(row.counterfactual_group_id, []).append(index)

    result = list(normalized)
    for indices in groups.values():
        if len(indices) == 2:
            left, right = indices
            if not result[left].paired_trace_id:
                result[left] = replace(
                    result[left],
                    paired_trace_id=result[right].expected_trace_id,
                )
            if not result[right].paired_trace_id:
                result[right] = replace(
                    result[right],
                    paired_trace_id=result[left].expected_trace_id,
                )
        for pair_index, row_index in enumerate(indices):
            if result[row_index].pair_index is None:
                result[row_index] = replace(result[row_index], pair_index=pair_index)
    return result


def _normalize_pair_row(row: EpisodePlanRow) -> EpisodePlanRow:
    capture_design = (row.capture_design or "").strip()
    if not capture_design:
        capture_design = "single_trace"
    if row.counterfactual_group_id and capture_design == "single_trace":
        capture_design = "paired_counterfactual"
    if capture_design not in {"single_trace", "paired_counterfactual"}:
        raise ValueError(f"Unknown capture_design: {capture_design!r}")
    return replace(
        row,
        capture_design=capture_design,
        trace_variant=_trace_variant_suffix(row.trace_variant or row.counterfactual_role),
        counterfactual_role=_trace_variant_suffix(row.counterfactual_role),
    )


def _trace_variant_suffix(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _optional_int(value: Any) -> int | None:
    text = str(value or "").strip()
    return int(text) if text else None


def _parse_list_field(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    if not text:
        return ()
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected list field, got {value!r}")
        return tuple(str(item).strip() for item in parsed if str(item).strip())
    return tuple(item.strip() for item in text.split(",") if item.strip())


def _list_cell(values: Sequence[str]) -> str:
    return json.dumps(list(values))


def _preflight_storage(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    episode_count: int,
) -> None:
    usage = shutil.disk_usage(output_root)
    free_gb = usage.free / (1024**3)
    minimum_free_gb = float(config.get("minimum_free_gb_after_capture", 25))
    estimate_gb = episode_count * float(config.get("estimated_mb_per_episode", 320)) / 1024
    if free_gb - estimate_gb < minimum_free_gb:
        raise SystemExit(
            "Not enough free space for planned capture: "
            f"free={free_gb:.1f}GB estimated_needed={estimate_gb:.1f}GB "
            f"minimum_remaining={minimum_free_gb:.1f}GB output_root={output_root}"
        )
    print(
        "storage preflight: "
        f"free={free_gb:.1f}GB estimated_needed={estimate_gb:.1f}GB "
        f"estimated_remaining={free_gb - estimate_gb:.1f}GB"
    )


def _append_status(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True))
        handle.write("\n")


def _dataset_id(config: Mapping[str, Any]) -> str:
    value = config.get("dataset_id") or config.get("name")
    if not value:
        raise ValueError("capture config must define dataset_id")
    return str(value)


def _dataset_ids(rows: Sequence[EpisodePlanRow]) -> str:
    return ",".join(sorted({row.dataset_id for row in rows}))


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise ValueError(f"Expected list of strings, got {value!r}")
    return [str(item) for item in value]


def _as_int_list(value: Any) -> list[int]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        raise ValueError(f"Expected list of ints, got {value!r}")
    return [int(item) for item in value]


def _env_prefix(pythonpath: str) -> tuple[str, str]:
    return ("env", f"PYTHONPATH={pythonpath}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
