"""Explain, replay, or apply a fitted VLA Lens probe artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vla_lens.probes import format_experiment_card_markdown, load_probe_artifact
from vla_lens.traces import TraceDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Trace dataset root")
    parser.add_argument("artifact_id", help="Saved probe artifact ID")
    parser.add_argument("action", choices=["explain", "replay", "use"])
    parser.add_argument(
        "--features",
        type=Path,
        help="For 'use': a two-dimensional NumPy .npy feature matrix.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="For 'use': optional .npy path for predictions.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format for 'explain'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe = load_probe_artifact(TraceDataset.open(args.root), args.artifact_id)
    if args.action == "explain":
        explanation = probe.explain()
        if args.format == "json":
            print(json.dumps(explanation, indent=2, sort_keys=True))
        else:
            print(format_experiment_card_markdown(explanation["experiment_card"]))
        return
    if args.action == "replay":
        result = probe.replay()
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        if not result.matched:
            raise SystemExit(2)
        return
    if args.features is None:
        raise SystemExit("'use' requires --features PATH.npy")
    predictions = probe.predict(np.load(args.features, allow_pickle=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, predictions, allow_pickle=False)
        print(f"wrote={args.output}")
        return
    print(json.dumps(np.asarray(predictions).tolist()))


if __name__ == "__main__":
    main()
