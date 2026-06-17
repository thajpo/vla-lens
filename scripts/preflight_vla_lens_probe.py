"""Review a VLA-lens probe spec before spending compute on training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_lens.probes import (
    format_probe_preflight_markdown,
    load_probe_spec,
    probe_preflight_report,
)
from vla_lens.traces import TraceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Trace dataset root")
    parser.add_argument("--spec", required=True, help="YAML probe spec path. Use '-' for stdin.")
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format for the review packet.",
    )
    parser.add_argument(
        "--min-class-support",
        type=int,
        default=20,
        help="Warn when an eval-split class has fewer rows than this threshold.",
    )
    parser.add_argument(
        "--large-sweep-readouts",
        type=int,
        default=100,
        help="Warn when planned readouts exceed this count.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output file path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = TraceDataset.open(args.root)
    spec = load_probe_spec(args.spec)
    report = probe_preflight_report(
        dataset,
        spec,
        min_class_support=args.min_class_support,
        large_sweep_readouts=args.large_sweep_readouts,
    )
    if args.format == "json":
        payload = json.dumps(report, indent=2, sort_keys=True)
    else:
        payload = format_probe_preflight_markdown(report)
    if args.output is None:
        print(payload)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
