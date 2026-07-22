"""Replay a saved identity probe and test whether its evidence covers the object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from vla_lens.probes import run_identity_localization_study
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
        raise TypeError("Identity localization study spec must be a mapping")
    spec: dict[str, Any] = dict(payload)
    if not args.run:
        print(json.dumps(spec, indent=2, sort_keys=True))
        return
    result = run_identity_localization_study(
        TraceDataset.open(args.root), spec, save=not args.no_save
    )
    if result.artifact is not None:
        print(f"artifact_id={result.artifact.artifact_id}")
        print(f"artifact_path={result.artifact.path}")
    print("episode_localization")
    print(result.summary.to_string(index=False))
    print("object_localization")
    print(
        result.object_summary.sort_values("static_lift", ascending=False).to_string(
            index=False
        )
    )
    if not result.matched_pair_summary.empty:
        print("matched_scene_localization")
        print(result.matched_pair_summary.to_string(index=False))
    print("reconstruction_check")
    print(result.reconstruction_check.to_string(index=False))
    print("timings_seconds=" + json.dumps(result.timings, sort_keys=True))


if __name__ == "__main__":
    main()
