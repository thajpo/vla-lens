"""Serve a live VLA-lens dashboard backend for a trace dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_lens.server import run_dashboard_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="LeRobot v3 dataset root, trace dataset root, or one .vlatrace bundle",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dashboard_server(args.root, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
