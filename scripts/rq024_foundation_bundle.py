#!/usr/bin/env python
"""Build or validate the deterministic RQ-024 FOUNDATION catalog bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_lens.rq024_foundation import validate_bundle, write_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument(
        "--bundle-root",
        type=Path,
        default=Path("configs/campaigns/rq024/foundation-r1"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = (
        write_bundle(args.bundle_root)
        if args.command == "build"
        else validate_bundle(args.bundle_root)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
