"""Run a small capture matrix that writes LeRobot roots plus VLA overlays.

This script is intentionally an orchestrator.  The model/env runner is supplied
as a command template so the smoke test can survive changes to the concrete
PI0.5 entrypoint while keeping the regeneration workflow stable.  The command
must write dataset roots directly.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from string import Formatter

from vla_lens import TraceDataset, validate_lerobot_v3_dataset

PROFILE_ORDER = ("rollout", "features", "mechanistic_sampled", "mechanistic_all", "audit_windowed")
ALLOWED_PROFILES = (
    *PROFILE_ORDER,
    "internals_sampled",
    "audit_sampled",
    "audit_full",
    "custom",
)


@dataclass(frozen=True)
class ProfileResult:
    profile: str
    traces_root: str
    capture_returncode: int | None
    trace_count: int
    validation_valid: bool
    validation_errors: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--profiles", default=",".join(PROFILE_ORDER))
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--vlatrace-root", type=Path, default=Path("runs/profile_smoke_vlatraces"))
    parser.add_argument(
        "--dataset-id",
        help=(
            "Dataset id to pass through the capture-command template. "
            "Defaults to output root name."
        ),
    )
    parser.add_argument("--delete-existing", action="store_true")
    parser.add_argument("--skip-capture", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--capture-command",
        default="",
        help=(
            "Command template that writes LeRobot dataset roots. Supported fields: "
            "{model_id}, {profile}, {episodes}, {start_seed}, {traces_root}, {dataset_id}. "
            "Example: 'scripts/pi05_capture.sh --backend rocm --model-id {model_id} "
            "--episodes {episodes} --start-seed {start_seed} --capture-profile {profile} "
            "--dataset-id {dataset_id} --vlatrace-out-root {traces_root}'"
        ),
    )
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = _parse_profiles(args.profiles)
    dataset_id = args.dataset_id or args.vlatrace_root.name
    if not args.skip_capture and not args.capture_command:
        raise SystemExit(
            "--capture-command is required unless --skip-capture is set. "
            "Pass the PI0.5 runner as a template so this script owns the profile matrix."
        )
    _validate_template(args.capture_command)

    if args.delete_existing and not args.dry_run:
        _delete_roots(args.vlatrace_root)

    results: list[ProfileResult] = []
    for index, profile in enumerate(profiles):
        traces_root = args.vlatrace_root / profile
        seed = args.start_seed + index * args.episodes
        print(f"== profile={profile} episodes={args.episodes} seed={seed}")

        returncode: int | None = None
        if not args.skip_capture:
            command = _format_command(
                args.capture_command,
                model_id=args.model_id,
                profile=profile,
                episodes=args.episodes,
                start_seed=seed,
                traces_root=traces_root,
                dataset_id=dataset_id,
            )
            print("+ " + " ".join(command))
            if not args.dry_run:
                returncode = subprocess.run(command, check=False).returncode
                if returncode != 0:
                    raise SystemExit(
                        f"capture failed for profile={profile} returncode={returncode}"
                    )

        trace_count = 0
        valid = False
        errors: list[dict] = []
        if not args.dry_run:
            dataset = TraceDataset.open(traces_root)
            validation = validate_lerobot_v3_dataset(traces_root)
            trace_count = len(dataset.bundles)
            valid = validation.valid
            errors = [issue.to_dict() for issue in validation.errors]
            if not valid:
                raise SystemExit(f"validation failed for profile={profile}: {errors}")
            _assert_trace_count(dataset, args.episodes, profile)

        results.append(
            ProfileResult(
                profile=profile,
                traces_root=str(traces_root),
                capture_returncode=returncode,
                trace_count=trace_count,
                validation_valid=valid,
                validation_errors=errors,
            )
        )

    summary = {
        "dataset_id": dataset_id,
        "model_id": args.model_id,
        "episodes_per_profile": args.episodes,
        "profiles": [asdict(result) for result in results],
    }
    print(json.dumps(summary, indent=2))
    if args.summary_json and not args.dry_run:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _parse_profiles(value: str) -> list[str]:
    profiles = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(profiles) - set(ALLOWED_PROFILES))
    if unknown:
        raise SystemExit(f"unknown profiles: {', '.join(unknown)}")
    return profiles


def _validate_template(template: str) -> None:
    if not template:
        return
    allowed = {"model_id", "profile", "episodes", "start_seed", "traces_root", "dataset_id"}
    fields = {field for _, field, _, _ in Formatter().parse(template) if field}
    unknown = sorted(fields - allowed)
    if unknown:
        raise SystemExit(f"unknown capture-command fields: {', '.join(unknown)}")


def _format_command(
    template: str,
    *,
    model_id: str,
    profile: str,
    episodes: int,
    start_seed: int,
    traces_root: Path,
    dataset_id: str,
) -> list[str]:
    formatted = template.format(
        model_id=model_id,
        profile=profile,
        episodes=episodes,
        start_seed=start_seed,
        traces_root=str(traces_root),
        dataset_id=dataset_id,
    )
    return shlex.split(formatted)


def _delete_roots(*roots: Path) -> None:
    for root in roots:
        if root.exists():
            print(f"deleting {root}")
            shutil.rmtree(root)


def _assert_trace_count(dataset: TraceDataset, expected: int, profile: str) -> None:
    if len(dataset.bundles) != expected:
        raise SystemExit(
            f"profile={profile} expected {expected} traces, found {len(dataset.bundles)}"
        )


if __name__ == "__main__":
    main()
