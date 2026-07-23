"""Run a joint all-object identity and location probe study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from vla_lens.probes import run_scene_map_probe_study
from vla_lens.traces import TraceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="VLA Lens dataset root")
    parser.add_argument("--spec", type=Path, required=True, help="Scene-map study YAML")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Scene-map study spec must be a mapping")
    spec: dict[str, Any] = dict(payload)
    if args.limit_episodes is not None:
        spec["limit_episodes"] = args.limit_episodes
    if not args.run:
        print(json.dumps(spec, indent=2, sort_keys=True))
        return
    result = run_scene_map_probe_study(TraceDataset.open(args.root), spec, save=not args.no_save)
    if result.artifact is not None:
        print(f"artifact_id={result.artifact.artifact_id}")
        print(f"artifact_path={result.artifact.path}")
    else:
        print("artifact_id=not_saved")
    print(f"candidates={len(result.candidates)}")
    print(f"selections={len(result.selections)}")
    print(f"comparisons={len(result.comparisons)}")
    print(f"predictions={len(result.predictions)}")
    print("timings_seconds=" + json.dumps(result.timings, sort_keys=True))
    if not result.comparisons.empty:
        identity_columns = [
            "feature_id",
            "feature_group",
            "target",
            "model_name",
            "test_scene_jaccard",
            "test_full_scene_jaccard",
            "test_f1",
            "test_exact_scene_rate",
            "test_full_exact_scene_rate",
        ]
        location_columns = [
            "feature_id",
            "feature_group",
            "target",
            "model_name",
            "test_error_m",
            "test_visible_error_m",
            "test_moved_10cm_error_m",
        ]
        identity = result.comparisons.loc[
            result.comparisons["target"].astype(str) != "object_position"
        ]
        location = result.comparisons.loc[
            result.comparisons["target"].astype(str) == "object_position"
        ]
        if not identity.empty:
            print("identity_results")
            print(identity[identity_columns].to_string(index=False))
        if not location.empty:
            print("location_results")
            print(location[location_columns].to_string(index=False))


if __name__ == "__main__":
    main()
