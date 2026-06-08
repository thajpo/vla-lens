"""Refresh mutable probe score caches for a VLA Lens dataset."""

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
        "--artifact-id",
        action="append",
        default=[],
        help="Probe artifact ID to refresh. Can be supplied more than once.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild dashboard indexes after refreshing probe scores.",
    )
    parser.add_argument(
        "--overwrite-index",
        action="store_true",
        help="Use a full index rebuild instead of append mode when --rebuild-index is set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = TraceDataset.open(args.root)
    results = refresh_all_probe_score_caches(
        dataset,
        artifact_ids=args.artifact_id or None,
    )
    payload = {
        "root": str(args.root),
        "probe_score_caches": [result.to_dict() for result in results],
    }
    if args.rebuild_index:
        payload["index"] = build_dataset_index(
            args.root,
            overwrite=args.overwrite_index,
        ).to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
