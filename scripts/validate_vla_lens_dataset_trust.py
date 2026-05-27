#!/usr/bin/env python
"""Check a local VLA Lens dataset root for research/probe trust gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_lens.research_guardrails import check_dataset_trust


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="LeRobot v3 dataset root or top-level directory containing nested LeRobot v3 roots.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument(
        "--allow-missing-splits",
        action="store_true",
        help="Do not fail when probe_splits.csv is absent.",
    )
    parser.add_argument(
        "--allow-missing-artifacts",
        action="store_true",
        help="Do not fail when no saved VLA Lens artifacts are present.",
    )
    parser.add_argument(
        "--allow-weak-outcome-balance",
        action="store_true",
        help="Do not fail when outcomes are missing or single-class.",
    )
    parser.add_argument(
        "--allow-missing-activations",
        action="store_true",
        help="Do not fail when no model-site activation overlay is present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = check_dataset_trust(
        args.root,
        require_splits=not args.allow_missing_splits,
        require_activation_coverage=not args.allow_missing_activations,
        require_artifacts=not args.allow_missing_artifacts,
        require_outcome_balance=not args.allow_weak_outcome_balance,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        _print_report(report.to_dict())
    raise SystemExit(0 if report.valid else 1)


def _print_report(report: dict[str, object]) -> None:
    print(f"{report['name']}: {'ok' if report['valid'] else 'failed'}")
    summary = report.get("summary")
    if isinstance(summary, dict):
        for key in ("root", "episodes", "activation_site_rows", "activation_coverage_ratio"):
            if key in summary:
                print(f"{key}={summary[key]}")
        for key in ("outcomes", "splits", "artifacts"):
            if key in summary:
                print(f"{key}={summary[key]}")
    issues = report.get("issues")
    if not isinstance(issues, list):
        return
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        location = f" {issue.get('path')}" if issue.get("path") else ""
        print(
            f"{issue.get('severity', 'error').upper()} {issue.get('code')}{location}: "
            f"{issue.get('message')}"
        )


if __name__ == "__main__":
    main()
