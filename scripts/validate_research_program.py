#!/usr/bin/env python
"""Check, fingerprint, and optionally snapshot one research program plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from vla_lens.research_io import canonical_research_fingerprint, write_bytes_create_only
from vla_lens.research_plan import (
    check_research_plan,
    format_research_plan_markdown,
    load_research_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Research program YAML to check.")
    parser.add_argument("--json", action="store_true", help="Print the check snapshot as JSON.")
    parser.add_argument(
        "--expect-fingerprint",
        help="Fail if the plan does not have this exact sha256 fingerprint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Atomically save the plan snapshot and schema check as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        payload = load_research_plan(args.plan)
    except (OSError, ValueError) as exc:
        print(f"Research plan load failed: {exc}")
        raise SystemExit(1) from exc

    check = check_research_plan(payload, path=args.plan)
    fingerprint_matches = (
        args.expect_fingerprint is None or check.fingerprint == args.expect_fingerprint
    )
    snapshot = _snapshot(args.plan, payload, check.to_dict(), fingerprint_matches)

    if args.output:
        if args.output.resolve() == args.plan.resolve():
            print("Refusing to overwrite the research program with its check snapshot.")
            raise SystemExit(1)
        if not check.valid or not fingerprint_matches:
            print("Refusing to save lock evidence for an invalid or mismatched program.")
            raise SystemExit(1)
        _write_json_atomic(args.output, snapshot)
    if args.json:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
    else:
        print(format_research_plan_markdown(payload, check), end="")
        if not fingerprint_matches:
            print(
                "\nFingerprint mismatch: "
                f"expected `{args.expect_fingerprint}`, observed `{check.fingerprint}`."
            )
        if args.output:
            print(f"\nSaved schema-check snapshot: `{args.output}`")
    raise SystemExit(0 if check.valid and fingerprint_matches else 1)


def _snapshot(
    path: Path,
    payload: object,
    check: dict[str, object],
    fingerprint_matches: bool,
) -> dict[str, object]:
    checker_path = Path(__file__).resolve().parents[1] / "src/vla_lens/research_plan.py"
    snapshot = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        "plan_path": str(path.resolve()),
        "plan": payload,
        "check": check,
        "expected_fingerprint_matches": fingerprint_matches,
        "checker_sha256": _sha256_file(checker_path),
        "git": _git_identity(path.resolve().parent),
        "limits": {
            "schema_check_only": True,
            "execution_readiness_checked": False,
            "scientific_validity_checked": False,
        },
    }
    snapshot["snapshot_payload_fingerprint"] = canonical_research_fingerprint(snapshot)
    return snapshot


def _git_identity(workdir: Path) -> dict[str, object]:
    root = _run_git(workdir, "rev-parse", "--show-toplevel")
    if not root:
        return {"available": False}
    repo = Path(root)
    status = _run_git(repo, "status", "--porcelain=v1")
    return {
        "available": True,
        "root": str(repo),
        "commit": _run_git(repo, "rev-parse", "HEAD"),
        "dirty": bool(status),
        "status": status.splitlines() if status else [],
    }


def _run_git(workdir: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=workdir,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _sha256_file(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _write_json_atomic(path: Path, payload: object) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_bytes_create_only(path, content)


if __name__ == "__main__":
    main()
