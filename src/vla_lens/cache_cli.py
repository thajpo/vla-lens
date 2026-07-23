"""Command-line cache management for iterative VLA Lens research."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import yaml

from vla_lens.cache import CacheManager
from vla_lens.campaigns import prepare_feature_campaign
from vla_lens.traces import TraceDataset

GIB = 1024**3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="VLA Lens dataset root")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("status", help="List recorded cache entries without scanning captures")

    prepare = commands.add_parser("prepare", help="Build deduplicated campaign precomputes")
    prepare.add_argument("--campaign", type=Path, required=True, help="YAML or JSON campaign file")

    for command in ("pin", "unpin"):
        pin = commands.add_parser(command, help=f"{command.title()} one cache entry")
        pin.add_argument("entry", help="Entry as namespace/key, for example features/abc123")

    prune = commands.add_parser("prune", help="Plan safe least-recently-used cleanup")
    prune.add_argument("--max-gib", type=float, default=10.0, help="Maximum managed cache size")
    prune.add_argument(
        "--min-free-gib", type=float, default=25.0, help="Required free disk space after cleanup"
    )
    prune.add_argument("--apply", action="store_true", help="Actually remove selected entries")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manager = CacheManager(args.root / ".vla_cache")
    if args.command == "status":
        entries = manager.entries()
        payload: Any = {
            "entry_count": len(entries),
            "size_bytes": sum(item.size_bytes for item in entries),
            "pinned_count": sum(item.pinned for item in entries),
            "entries": [item.to_dict() for item in entries],
        }
    elif args.command == "prepare":
        campaign = yaml.safe_load(args.campaign.read_text(encoding="utf-8"))
        if not isinstance(campaign, dict):
            raise ValueError("Campaign file must contain a mapping")
        payload = asdict(prepare_feature_campaign(TraceDataset.open(args.root), campaign))
    elif args.command in {"pin", "unpin"}:
        namespace, key = _entry_parts(args.entry)
        payload = manager.set_pinned(namespace, key, pinned=args.command == "pin").to_dict()
    else:
        payload = asdict(
            manager.prune(
                max_bytes=int(args.max_gib * GIB),
                min_free_bytes=int(args.min_free_gib * GIB),
                apply=bool(args.apply),
            )
        )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human(args.command, payload)
    return 0


def _entry_parts(value: str) -> tuple[str, str]:
    parts = value.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("Cache entry must be written as namespace/key")
    return parts[0], parts[1]


def _print_human(command: str, payload: dict[str, Any]) -> None:
    if command == "status":
        print(
            f"{payload['entry_count']} entries, {_format_bytes(payload['size_bytes'])}, "
            f"{payload['pinned_count']} pinned"
        )
        for item in payload["entries"]:
            marker = "pinned" if item["pinned"] else "disposable"
            print(
                f"  {item['namespace']}/{item['key']}  "
                f"{_format_bytes(item['size_bytes'])}  {marker}"
            )
        return
    if command == "prepare":
        print(
            f"Prepared {payload['unique_count']} unique feature matrices "
            f"for {payload['requested_count']} requests."
        )
        for item in payload["features"]:
            action = "built" if item["built"] else "reused"
            print(f"  features/{item['cache_key']}  {action}  shape={tuple(item['shape'])}")
        return
    if command in {"pin", "unpin"}:
        state = "pinned" if payload["pinned"] else "disposable"
        print(f"{payload['namespace']}/{payload['key']} is now {state}.")
        return
    mode = "Removed" if payload["apply"] else "Would remove"
    print(
        f"{mode} {len(payload['entries'])} entries and reclaim "
        f"{_format_bytes(payload['reclaimed_bytes'])}."
    )
    if payload["blocked_by_pins"]:
        print("Pinned entries prevent the requested size or free-space target.")
    if not payload["apply"]:
        print("Dry run only. Pass --apply to remove these entries.")


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
