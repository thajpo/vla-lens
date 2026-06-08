"""Build local Parquet indexes used by the VLA Lens dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_lens.dataset import build_dataset_index
from vla_lens.probes import refresh_all_probe_score_caches
from vla_lens.traces import TraceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="LeRobot root or nested batch output root")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild all index tables instead of appending new episode rows.",
    )
    parser.add_argument(
        "--refresh-probe-scores",
        action="store_true",
        help="Refresh mutable probe score caches before rebuilding dashboard indexes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_results = []
    if args.refresh_probe_scores:
        dataset = TraceDataset.open(args.root)
        score_results = [
            result.to_dict() for result in refresh_all_probe_score_caches(dataset)
        ]
    result = build_dataset_index(args.root, overwrite=args.overwrite)
    payload = result.to_dict()
    if args.refresh_probe_scores:
        payload["probe_score_caches"] = score_results
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
