"""Run matched pooled, tokenwise, and learned-layer scene-object probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from vla_lens.probes import run_token_scene_probe_study
from vla_lens.traces import TraceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="VLA Lens dataset root")
    parser.add_argument("--spec", type=Path, required=True, help="Study YAML")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Token scene study spec must be a mapping")
    spec: dict[str, Any] = dict(payload)
    if args.limit_episodes is not None:
        spec["limit_episodes"] = args.limit_episodes
    if not args.run:
        print(json.dumps(spec, indent=2, sort_keys=True))
        return
    result = run_token_scene_probe_study(
        TraceDataset.open(args.root), spec, save=not args.no_save
    )
    if result.artifact is not None:
        print(f"artifact_id={result.artifact.artifact_id}")
        print(f"artifact_path={result.artifact.path}")
    else:
        print("artifact_id=not_saved")
    print(f"candidates={len(result.candidates)}")
    print(f"selections={len(result.selections)}")
    print(f"predictions={len(result.predictions)}")
    print("timings_seconds=" + json.dumps(result.timings, sort_keys=True))
    columns = [
        "representation",
        "structure",
        "model",
        "target",
        "selected_layer",
        "readout_dim",
        "ridge_alpha",
        "test_scene_jaccard",
        "test_macro_average_precision",
        "test_error_m",
        "test_moved_10cm_error_m",
    ]
    available = [column for column in columns if column in result.selections]
    print(result.selections[available].to_string(index=False))


if __name__ == "__main__":
    main()
