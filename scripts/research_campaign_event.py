#!/usr/bin/env python
"""Append or verify hash-chained autonomous-research campaign events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_lens.research_events import (
    EVENT_TYPES,
    append_research_event,
    format_event_ledger_markdown,
    verify_research_event_ledger,
)
from vla_lens.research_io import load_research_mapping
from vla_lens.research_plan import load_research_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify", help="Verify and summarize a ledger.")
    _common(verify)
    verify.add_argument("--json", action="store_true")
    append = subparsers.add_parser("append", help="Append one immutable event.")
    _common(append)
    append.add_argument("--event-id", required=True)
    append.add_argument("--event-type", choices=sorted(EVENT_TYPES), required=True)
    append.add_argument("--actor-id", required=True)
    append.add_argument("--subject-id", required=True)
    append.add_argument("--subject-fingerprint", required=True)
    append.add_argument("--payload", type=Path, required=True, help="Strict JSON/YAML payload.")
    return parser.parse_args()


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--event-root", type=Path, required=True)


def main() -> None:
    args = parse_args()
    program = load_research_plan(args.program)
    if args.command == "append":
        path, fingerprint = append_research_event(
            args.event_root,
            program,
            event_id=args.event_id,
            event_type=args.event_type,
            actor_id=args.actor_id,
            subject_id=args.subject_id,
            subject_fingerprint=args.subject_fingerprint,
            payload=load_research_mapping(args.payload),
        )
        print(f"Created `{path}` ({fingerprint}).")
        return
    check = verify_research_event_ledger(args.event_root, program)
    if args.json:
        print(json.dumps(check.to_dict(), indent=2, sort_keys=True))
    else:
        print(format_event_ledger_markdown(check), end="")
    raise SystemExit(0 if check.valid else 1)


if __name__ == "__main__":
    main()
