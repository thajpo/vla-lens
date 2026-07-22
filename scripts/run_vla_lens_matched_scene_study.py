"""Run matched-scene visual patch localization over saved PI0.5 traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from vla_lens.probes import run_matched_scene_localization_study
from vla_lens.traces import TraceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="VLA Lens dataset root")
    parser.add_argument("--spec", type=Path, required=True, help="Study YAML")
    parser.add_argument("--limit-pairs-per-split", type=int, default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = yaml.safe_load(args.spec.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Matched-scene study spec must be a mapping")
    spec: dict[str, Any] = dict(payload)
    if args.limit_pairs_per_split is not None:
        spec["limit_pairs_per_split"] = args.limit_pairs_per_split
    if not args.run:
        print(json.dumps(spec, indent=2, sort_keys=True))
        return
    result = run_matched_scene_localization_study(
        TraceDataset.open(args.root), spec, save=not args.no_save
    )
    print(
        f"artifact_id={result.artifact.artifact_id}"
        if result.artifact is not None
        else "artifact_id=not_saved"
    )
    print(f"pairs={len(result.pairs)}")
    print(f"patch_scores={len(result.patch_scores)}")
    print("timings_seconds=" + json.dumps(result.timings, sort_keys=True))
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
