#!/usr/bin/env python
"""Create the immutable repo-side index for external RQ-024 FOUNDATION evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_lens.rq024_foundation_evidence import create_foundation_evidence_index

DEFAULT_CHILD = Path("configs/campaigns/rq024/foundation-r1/child.yaml")
DEFAULT_TRIALS = Path("configs/campaigns/rq024/foundation-r1/trials.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child", type=Path, default=DEFAULT_CHILD)
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument(
        "--external-root",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="Approve an external artifact root; repeat for multiple roots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    payload, created = create_foundation_evidence_index(
        repo_root=repo_root,
        child_path=args.child,
        trial_manifest_path=args.trials,
        event_root=args.event_root,
        output_path=args.output,
        external_roots=_external_roots(args.external_root),
    )
    summary = {
        "created": created,
        "output": str(args.output),
        "trial_count": payload["trial_count"],
        "ledger_tip": payload["attempt_ledger"]["tip_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _external_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        root_id, separator, raw_path = value.partition("=")
        if not separator or not root_id or not raw_path or root_id in roots:
            raise ValueError("--external-root must be a unique ID=PATH pair")
        roots[root_id] = Path(raw_path)
    return roots


if __name__ == "__main__":
    main()
