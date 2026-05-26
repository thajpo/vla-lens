#!/usr/bin/env python3
"""Migrate legacy .vlatrace datasets to LeRobot v3 + VLA Lens overlay."""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_lens.migration.vlatrace import (
    discover_vlatrace_bundles,
    migrate_vlatrace_bundle,
    migrate_vlatrace_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--limit", type=int, help="Migrate only the first N bundles.")
    parser.add_argument(
        "--overwrite-root",
        action="store_true",
        help="Delete OUTPUT_ROOT before migrating.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip trace_ids already present in OUTPUT_ROOT.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N migrated bundles.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_paths = discover_vlatrace_bundles(args.source_root)
    if args.limit is not None:
        bundle_paths = bundle_paths[: max(0, args.limit)]
    print(
        f"found {len(bundle_paths)} legacy bundle(s) under {args.source_root}",
        flush=True,
    )
    if args.overwrite_root or args.no_resume or args.progress_every <= 0:
        results = migrate_vlatrace_dataset(
            args.source_root,
            args.output_root,
            limit=args.limit,
            overwrite_root=args.overwrite_root,
            resume=not args.no_resume,
        )
        print(f"migrated {len(results)} bundle(s) to {args.output_root}", flush=True)
        return

    migrated = 0
    for index, path in enumerate(bundle_paths, start=1):
        try:
            result = migrate_vlatrace_bundle(
                path,
                args.output_root,
                source_root=args.source_root,
                overwrite=False,
            )
        except FileExistsError:
            continue
        migrated += 1
        if migrated == 1 or migrated % args.progress_every == 0:
            print(
                f"migrated {migrated}/{len(bundle_paths)} "
                f"(source #{index}, episode {result.episode_index}, {result.trace_id})",
                flush=True,
            )
    migrate_vlatrace_dataset(
        args.source_root,
        args.output_root,
        limit=0,
        overwrite_root=False,
        resume=True,
    )
    print(f"migrated {migrated} new bundle(s) to {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
