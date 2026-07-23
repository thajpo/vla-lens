"""Run the motion-aware follow-up to a vector geometry probe study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from vla_lens.probes import run_motion_probe_study
from vla_lens.traces import TraceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="VLA Lens dataset root")
    parser.add_argument("--spec", type=Path, required=True, help="Motion study YAML")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Motion study spec must be a mapping")
    spec: dict[str, Any] = dict(payload)
    if args.limit_episodes is not None:
        spec["limit_episodes"] = args.limit_episodes
    if not args.run:
        print(json.dumps(spec, indent=2, sort_keys=True))
        return
    result = run_motion_probe_study(
        TraceDataset.open(args.root),
        spec,
        save=not args.no_save,
    )
    print(
        "artifact_id="
        + (result.artifact.artifact_id if result.artifact is not None else "not_saved")
    )
    print("timings_seconds=" + json.dumps(result.timings, sort_keys=True))
    print(f"models={len(result.models)} predictions={len(result.predictions)}")
    print(f"object_motion_rows={len(result.object_motion)}")
    if not result.comparisons.empty:
        test = result.comparisons.loc[
            result.comparisons["split"].astype(str).str.startswith("test")
        ]
        columns = [
            "feature_id",
            "analysis",
            "target",
            "threshold",
            "comparison_model",
            "activation_advantage",
            "ci_low",
            "ci_high",
            "two_sided_p_value",
        ]
        print(test[columns].to_string(index=False))


if __name__ == "__main__":
    main()
