#!/usr/bin/env python
"""Validate one locked research child and optional authorization receipt."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from vla_lens.research_child import (
    check_research_child,
    check_research_child_lock,
    load_research_child,
)
from vla_lens.research_io import (
    canonical_research_fingerprint,
    load_research_mapping,
    write_bytes_create_only,
)
from vla_lens.research_plan import check_research_plan, load_research_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("child", type=Path, help="Immutable child-plan YAML or JSON.")
    parser.add_argument("--program", type=Path, required=True, help="Parent program YAML.")
    parser.add_argument("--lock-receipt", type=Path, help="Post-audit child lock receipt.")
    parser.add_argument(
        "--verify-files",
        action="store_true",
        help="Verify every locked input/audit file and Git lock evidence.",
    )
    parser.add_argument("--json", action="store_true", help="Print the machine check.")
    parser.add_argument("--output", type=Path, help="Create-only JSON check snapshot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = _repo_root(args.child.parent)
    child = load_research_child(args.child)
    program = load_research_plan(args.program)
    program_check = check_research_plan(program, path=args.program)
    child_check = check_research_child(
        child,
        program,
        repo_root=repo_root,
        verify_files=args.verify_files,
    )
    receipt: Mapping[str, Any] | None = None
    lock_check = None
    if args.lock_receipt:
        receipt = load_research_mapping(args.lock_receipt)
        lock_check = check_research_child_lock(
            receipt,
            child,
            program,
            repo_root=repo_root,
            verify_files=args.verify_files,
        )
    git_check = _git_lock_check(args, child, receipt, repo_root) if args.verify_files else None
    storage_check = _storage_check(child, repo_root) if args.verify_files else None
    authorized = bool(
        program_check.valid
        and child_check.valid
        and child_check.files_verified
        and lock_check is not None
        and lock_check.valid
        and lock_check.audit_files_verified
        and git_check
        and git_check["valid"]
        and storage_check
        and storage_check["valid"]
    )
    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "program_check": program_check.to_dict(),
        "child_check": child_check.to_dict(),
        "lock_check": None if lock_check is None else lock_check.to_dict(),
        "git_lock_check": git_check,
        "storage_check": storage_check,
        "authorized_to_start_child": authorized,
        "limits": {
            "model_or_simulator_started": False,
            "capture_wrapper_runtime_check_still_required": True,
            "scientific_truth_evaluated": False,
        },
    }
    snapshot["snapshot_payload_fingerprint"] = canonical_research_fingerprint(snapshot)
    if args.output:
        if args.output.resolve() in {args.child.resolve(), args.program.resolve()}:
            raise SystemExit("Refusing to overwrite an immutable plan with a check snapshot")
        _write_snapshot(args.output, snapshot)
    if args.json:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
    else:
        _print_human(args, snapshot)
    raise SystemExit(0 if (authorized if args.verify_files else child_check.valid) else 1)


def _print_human(args: argparse.Namespace, snapshot: Mapping[str, Any]) -> None:
    child = snapshot["child_check"]
    lock = snapshot.get("lock_check")
    print(f"Child plan: {'VALID' if child['valid'] else 'INVALID'}")
    print(f"Child fingerprint: {child['fingerprint']}")
    print(f"Locked input files verified: {child['files_verified']}")
    print(f"Independent lock: {bool(lock and lock['valid'])}")
    print(f"Authorized to start child: {snapshot['authorized_to_start_child']}")
    if not args.verify_files:
        print(
            "Execution authorization was not evaluated; rerun with --verify-files "
            "and --lock-receipt."
        )
    for section in ("program_check", "child_check", "lock_check"):
        check = snapshot.get(section)
        if isinstance(check, Mapping):
            for issue in check.get("issues", []):
                print(f"- {section}: {issue['code']} — {issue['message']}")


def _git_lock_check(
    args: argparse.Namespace,
    child: Mapping[str, Any],
    receipt: Mapping[str, Any] | None,
    repo_root: Path,
) -> dict[str, Any]:
    paths = [args.child, args.program]
    if args.lock_receipt:
        paths.append(args.lock_receipt)
    for dotted in (
        "cohort.manifest.path",
        "cohort.exposure_log.path",
        "trials.manifest.path",
        "runtime.environment.package_receipt.path",
        "runtime.runner.config.path",
    ):
        value = _get(child, dotted)
        if value:
            paths.append(repo_root / str(value))
    relative: list[str] = []
    errors: list[str] = []
    for path in paths:
        try:
            item = str(path.resolve().relative_to(repo_root))
        except ValueError:
            errors.append(f"outside_repo:{path}")
            continue
        relative.append(item)
        if _git(repo_root, "ls-files", "--error-unmatch", "--", item).returncode != 0:
            errors.append(f"untracked:{item}")
        if _git(repo_root, "diff", "--quiet", "HEAD", "--", item).returncode != 0:
            errors.append(f"dirty:{item}")
    if receipt is not None:
        observed = _git(
            repo_root,
            "log",
            "-1",
            "--format=%H",
            "--",
            str(args.child.resolve().relative_to(repo_root)),
        ).stdout.strip()
        if observed != receipt.get("manifest_commit"):
            errors.append("manifest_commit_does_not_match_child_file")
    return {"valid": not errors, "checked_paths": relative, "errors": errors}


def _storage_check(child: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    output_root = Path(str(_get(child, "output.root") or repo_root))
    probe = output_root if output_root.exists() else output_root.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1_000_000_000
    required_gb = float(_get(child, "budget.min_free_space_gb") or 0)
    return {
        "valid": free_gb >= required_gb,
        "filesystem_probe": str(probe),
        "free_gb": free_gb,
        "required_free_gb": required_gb,
    }


def _repo_root(path: Path) -> Path:
    result = _git(path, "rev-parse", "--show-toplevel")
    if result.returncode != 0:
        raise ValueError("Research child must be prepared inside a Git repository")
    return Path(result.stdout.strip()).resolve()


def _git(workdir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=workdir,
        check=False,
        capture_output=True,
        text=True,
    )


def _get(payload: Mapping[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _write_snapshot(path: Path, payload: Mapping[str, Any]) -> None:
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_bytes_create_only(path, content)


if __name__ == "__main__":
    main()
