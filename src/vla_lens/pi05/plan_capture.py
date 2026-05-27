"""Single-process PI0.5 plan capture runner.

This runner keeps the normal batch data model, but avoids reloading the PI0.5
policy for every benchmark/task command. It reads the same YAML/CSV inputs as
``batch_capture`` and executes each grouped task command in one Python process.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from vla_lens.capture.lerobot_v3 import validate_lerobot_v3_dataset
from vla_lens.pi05 import capture
from vla_lens.pi05.batch_capture import (
    CaptureCommand,
    _capture_commands,
    _dataset_ids,
    _expected_trace_exists,
    _load_config,
    _preflight_storage,
    _read_episode_plan,
    _write_plan_files,
)
from vla_lens.traces import TraceDataset
from vla_lens.validation import validate_trace_dataset

DEFAULT_CONFIG = Path("configs/pi05_broad_1000_mech_light.yaml")
RECYCLE_EXIT_CODE = 75


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episode-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, help="Override config output_root.")
    parser.add_argument("--run", action="store_true", help="Execute capture.")
    parser.add_argument("--limit-commands", type=int)
    parser.add_argument(
        "--max-executed-commands",
        type=int,
        help=(
            "Exit with a restart-friendly code after this many non-skipped commands. "
            "Use this to recycle ROCm/PyTorch state periodically."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = _load_config(args.config)
    if args.output_root:
        config["output_root"] = str(args.output_root)
    output_root = Path(str(config["output_root"])).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    rows = _read_episode_plan(args.episode_plan)
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
    print("execution=single-process")

    if not args.run:
        print("Dry run only. Re-run with --run to execute capture.")
        return

    status_path = output_root / "capture_status.jsonl"
    runtime = None
    runtime_key: tuple[str, str, str] | None = None
    executed_commands = 0
    for index, item in enumerate(commands, start=1):
        expected_exists = _command_expected_traces_exist(item)
        if expected_exists and not args.force:
            _append_status(status_path, _status_payload("skipped_existing", index, item))
            print(f"[{index}/{len(commands)}] skip existing {item.benchmark} task={item.task_id}")
            continue

        _preflight_storage(config, output_root, episode_count=0)
        capture_args = capture_args_from_command(item.command)
        key = _runtime_key(capture_args)
        if runtime is None or key != runtime_key:
            print(
                "loading PI0.5 policy once: "
                f"model_id={capture_args.model_id} device={capture_args.device} "
                f"dtype={capture_args.dtype}",
                flush=True,
            )
            runtime = capture.load_pi05_capture_runtime(capture_args)
            runtime_key = key

        plan = capture._resolve_capture_plan(capture_args)
        if index == 1:
            print(f"capture plan: {plan.to_metadata()}", flush=True)
        print(
            f"[{index}/{len(commands)}] run {item.benchmark} task={item.task_id} "
            f"episodes={item.episodes} start_seed={item.start_seed}",
            flush=True,
        )
        try:
            capture.run_pi05_capture_task(capture_args, runtime=runtime, plan=plan)
            _validate_task_root(capture_args.output_root)
        except Exception as exc:
            _append_status(
                status_path,
                {
                    **_status_payload("failed", index, item),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
        _append_status(status_path, _status_payload("completed", index, item))
        executed_commands += 1
        if (
            args.max_executed_commands is not None
            and executed_commands >= max(0, args.max_executed_commands)
        ):
            print(
                f"recycle requested after {executed_commands} executed commands; "
                f"restart to continue from existing traces",
                flush=True,
            )
            raise SystemExit(RECYCLE_EXIT_CODE)


def capture_args_from_command(command: Sequence[str]) -> argparse.Namespace:
    args = list(command)
    try:
        module_index = args.index("vla_lens.pi05.capture")
    except ValueError as exc:
        raise ValueError(f"Command does not invoke vla_lens.pi05.capture: {command}") from exc
    return capture.parse_args(args[module_index + 1 :])


def _runtime_key(args: argparse.Namespace) -> tuple[str, str, str]:
    return (str(args.model_id), str(args.device), str(args.dtype))


def _command_expected_traces_exist(item: CaptureCommand) -> bool:
    return bool(item.expected_trace_ids) and all(
        _expected_trace_exists(item.output_root, trace_id) for trace_id in item.expected_trace_ids
    )


def _validate_task_root(path: Path) -> None:
    if (path / "meta" / "info.json").exists() and (path / "data").exists():
        validation = validate_lerobot_v3_dataset(path)
        if not validation.valid:
            raise ValueError(validation.to_dict())
        print(f"validated {len(TraceDataset.open(path).bundles)} LeRobot traces in {path}")
        return
    dataset = TraceDataset.open(path)
    validation = validate_trace_dataset(dataset)
    if not validation.valid:
        raise ValueError(validation.to_dict())
    print(f"validated {len(dataset.bundles)} traces in {path}")


def _status_payload(status: str, index: int, item: CaptureCommand) -> dict[str, Any]:
    return {
        "status": status,
        "index": index,
        "dataset_id": item.dataset_id,
        "benchmark": item.benchmark,
        "task_id": item.task_id,
        "capture_profile": item.capture_profile,
        "expected_traces": len(item.expected_paths),
    }


def _append_status(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


if __name__ == "__main__":
    main()
