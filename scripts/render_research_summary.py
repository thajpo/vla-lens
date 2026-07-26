#!/usr/bin/env python
"""Validate and render one structured research result card."""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_lens.research_events import verify_research_event_ledger
from vla_lens.research_io import (
    canonical_research_fingerprint,
    file_sha256,
    load_research_mapping,
    write_bytes_create_only,
)
from vla_lens.research_summary import format_research_result_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("card", type=Path, help="Result-card JSON or YAML file.")
    parser.add_argument("--program", type=Path, required=True, help="Immutable program YAML.")
    parser.add_argument("--child-plan", type=Path, required=True, help="Locked child-plan YAML.")
    parser.add_argument("--child-lock", type=Path, required=True, help="Exact child-lock receipt.")
    parser.add_argument("--audit-report", type=Path, required=True, help="Exact audit report.")
    parser.add_argument(
        "--analysis-package", type=Path, required=True, help="Exact analysis result package."
    )
    parser.add_argument(
        "--authorization-receipt", type=Path, required=True, help="Exact start authorization."
    )
    parser.add_argument(
        "--attempt-ledger", type=Path, required=True, help="Exact append-only attempt ledger."
    )
    parser.add_argument("--budget-record", type=Path, required=True, help="Exact resource record.")
    parser.add_argument(
        "--event-root",
        type=Path,
        help="Campaign event root; defaults to the program's canonical event_root.",
    )
    parser.add_argument(
        "--result-event-id",
        required=True,
        help="Exact accepted result_recorded event that authorizes this summary.",
    )
    parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    card = load_research_mapping(args.card)
    program = load_research_mapping(args.program)
    child_plan = load_research_mapping(args.child_plan)
    child_lock = load_research_mapping(args.child_lock)
    analysis_package = load_research_mapping(args.analysis_package)
    repo_root = Path(__file__).resolve().parents[1]
    event_root = args.event_root or repo_root / str(program["protocol_defaults"]["event_root"])
    ledger = verify_research_event_ledger(
        event_root,
        program,
        repo_root=repo_root,
        verify_artifacts=True,
    )
    if not ledger.valid:
        raise ValueError("Refusing to render from an invalid campaign event ledger")
    result_event = ledger.state.events_by_id.get(args.result_event_id)
    if result_event is None or result_event.get("event_type") != "result_recorded":
        raise ValueError("Result event is absent or is not a result_recorded event")
    result_payload = result_event["payload"]
    if (
        result_payload["result_ref"]["content_fingerprint"] != canonical_research_fingerprint(card)
        or result_payload["analysis_ref"]["sha256"] != file_sha256(args.analysis_package)
        or result_payload["attempt_range"]["ledger_tip_before_result"]
        != card["ledger_tip_before_result"]
        or {
            "first_sequence": result_payload["attempt_range"]["first_sequence"],
            "last_sequence": result_payload["attempt_range"]["last_sequence"],
        }
        != card["attempt_event_range"]
    ):
        raise ValueError("Result card or analysis differs from the accepted result event")
    rendered = format_research_result_markdown(
        card,
        program=program,
        child_plan=child_plan,
        child_lock=child_lock,
        analysis_package=analysis_package,
        lock_receipt_sha256=file_sha256(args.child_lock),
        audit_report_sha256=file_sha256(args.audit_report),
        analysis_package_sha256=file_sha256(args.analysis_package),
        authorization_receipt_sha256=file_sha256(args.authorization_receipt),
        attempt_ledger_sha256=file_sha256(args.attempt_ledger),
        budget_record_sha256=file_sha256(args.budget_record),
    )
    if args.output:
        protected = {
            path.resolve()
            for path in (
                args.card,
                args.program,
                args.child_plan,
                args.child_lock,
                args.analysis_package,
                args.audit_report,
                args.authorization_receipt,
                args.attempt_ledger,
                args.budget_record,
            )
        }
        if args.output.resolve() in protected:
            raise ValueError("Refusing to overwrite an immutable input with a summary")
        write_bytes_create_only(args.output, rendered.encode("utf-8"))
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
