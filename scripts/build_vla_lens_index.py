"""Build local Parquet indexes used by the VLA Lens dashboard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_lens.dataset import build_dataset_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="LeRobot root or nested batch output root")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild all index tables instead of appending new episode rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_dataset_index(args.root, overwrite=args.overwrite)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
