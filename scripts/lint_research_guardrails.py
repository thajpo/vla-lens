#!/usr/bin/env python
"""Lint research configs, episode plans, and audit contracts without writing outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_lens.research_guardrails import lint_research_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root containing configs/.",
    )
    parser.add_argument(
        "--episode-plan",
        action="append",
        default=[],
        type=Path,
        help="Optional episode_plan.csv to lint.",
    )
    parser.add_argument(
        "--audit-contract",
        action="append",
        default=[],
        type=Path,
        help="Optional audit/circuit capture contract YAML to lint.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = lint_research_configs(
        args.root,
        episode_plans=args.episode_plan,
        audit_contracts=args.audit_contract,
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
        for key, value in summary.items():
            print(f"{key}={value}")
    issues = report.get("issues")
    if not isinstance(issues, list):
        return
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        location = f" {issue.get('path')}" if issue.get("path") else ""
        details = issue.get("details")
        detail_text = ""
        if isinstance(details, dict) and details:
            detail_text = " " + " ".join(f"{key}={value}" for key, value in details.items())
        print(
            f"{issue.get('severity', 'error').upper()} {issue.get('code')}{location}: "
            f"{issue.get('message')}{detail_text}"
        )


if __name__ == "__main__":
    main()
