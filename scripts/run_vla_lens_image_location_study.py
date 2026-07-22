"""Test whether captured visual tokens explicitly encode named-object image locations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from vla_lens.probes import run_image_location_probe_study
from vla_lens.traces import TraceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="VLA Lens dataset root")
    parser.add_argument("--spec", type=Path, required=True, help="Study YAML")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Image-location study spec must be a mapping")
    spec: dict[str, Any] = dict(payload)
    if not args.run:
        print(json.dumps(spec, indent=2, sort_keys=True))
        return
    result = run_image_location_probe_study(
        TraceDataset.open(args.root), spec, save=not args.no_save
    )
    if result.artifact is not None:
        print(f"artifact_id={result.artifact.artifact_id}")
        print(f"artifact_path={result.artifact.path}")
    print("selected_probes")
    print(result.selections.to_string(index=False))
    if not result.capacity_checks.empty:
        print("nonlinear_capacity_checks")
        print(result.capacity_checks.to_string(index=False))
    print("held_out_comparisons")
    print(result.comparisons.to_string(index=False))
    if not result.capacity_comparisons.empty:
        print("nonlinear_capacity_comparisons")
        print(result.capacity_comparisons.to_string(index=False))
    print("shuffled_scene_controls")
    print(result.shuffled_controls.groupby("study_part").mean(numeric_only=True))
    print("objects")
    print(result.object_summary.to_string(index=False))
    print("reconstruction_check")
    print(result.reconstruction_check.to_string(index=False))
    print("timings_seconds=" + json.dumps(result.timings, sort_keys=True))


if __name__ == "__main__":
    main()
