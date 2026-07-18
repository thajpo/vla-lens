"""Run a vector-aware object position and orientation probe study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from vla_lens.probes import run_geometry_probe_study
from vla_lens.traces import TraceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="VLA Lens dataset root")
    parser.add_argument("--spec", type=Path, required=True, help="Geometry study YAML")
    parser.add_argument(
        "--limit-episodes",
        type=int,
        default=None,
        help="Smoke-test cap applied before fitting; omit for the full study",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute and save the study; omit to print the normalized request",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run diagnostics without adding an artifact to the dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Geometry study spec must be a mapping")
    spec: dict[str, Any] = dict(payload)
    if args.limit_episodes is not None:
        spec["limit_episodes"] = args.limit_episodes
    if not args.run:
        print(json.dumps(spec, indent=2, sort_keys=True))
        return

    dataset = TraceDataset.open(args.root)
    result = run_geometry_probe_study(dataset, spec, save=not args.no_save)
    if result.artifact is not None:
        print(f"artifact_id={result.artifact.artifact_id}")
        print(f"artifact_path={result.artifact.path}")
    else:
        print("artifact_id=not_saved")
    print(f"candidates={len(result.candidates)}")
    print(f"selections={len(result.selections)}")
    print("timings_seconds=" + json.dumps(result.timings, sort_keys=True))
    if not result.selections.empty:
        columns = [
            "feature_id",
            "feature_group",
            "target",
            "pca_dim",
            "ridge_alpha",
            "selection_error",
            "selection_baseline_error",
            "test_error",
            "test_baseline_error",
            "error_unit",
        ]
        print(result.selections[columns].to_string(index=False))


if __name__ == "__main__":
    main()
