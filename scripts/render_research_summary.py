#!/usr/bin/env python
"""Validate and render one structured research result card."""

from __future__ import annotations

import argparse
from pathlib import Path

from vla_lens.research_io import file_sha256, load_research_mapping, write_bytes_create_only
from vla_lens.research_summary import format_research_result_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("card", type=Path, help="Result-card JSON or YAML file.")
    parser.add_argument("--program", type=Path, required=True, help="Immutable program YAML.")
    parser.add_argument("--child-plan", type=Path, required=True, help="Locked child-plan YAML.")
    parser.add_argument("--audit-report", type=Path, required=True, help="Exact audit report.")
    parser.add_argument(
        "--analysis-package", type=Path, required=True, help="Exact analysis result package."
    )
    parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    card = load_research_mapping(args.card)
    program = load_research_mapping(args.program)
    child_plan = load_research_mapping(args.child_plan)
    rendered = format_research_result_markdown(
        card,
        program=program,
        child_plan=child_plan,
        audit_report_sha256=file_sha256(args.audit_report),
        analysis_package_sha256=file_sha256(args.analysis_package),
    )
    if args.output:
        protected = {path.resolve() for path in (args.card, args.program, args.child_plan)}
        if args.output.resolve() in protected:
            raise ValueError("Refusing to overwrite an immutable input with a summary")
        write_bytes_create_only(args.output, rendered.encode("utf-8"))
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
