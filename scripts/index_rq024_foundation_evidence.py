#!/usr/bin/env python
"""Verify and index immutable external RQ-024 FOUNDATION evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla_lens.rq024_foundation_evidence import (
    build_foundation_evidence_index,
    load_verified_foundation_ledger,
    write_foundation_evidence_index,
)

DEFAULT_PROGRAM = Path("configs/campaigns/rq024_controlled_scene_to_behavior.yaml")
DEFAULT_CHILD = Path("configs/campaigns/rq024/foundation-r1/child.yaml")
DEFAULT_TRIALS = Path("configs/campaigns/rq024/foundation-r1/trials.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--child", type=Path, default=DEFAULT_CHILD)
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--event-root", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        metavar="ID=PATH",
        help="Approve an artifact root; repeat for multiple roots.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    program_path = _repo_path(repo_root, args.program)
    child_path = _repo_path(repo_root, args.child)
    trials_path = _repo_path(repo_root, args.trials)
    ledger = load_verified_foundation_ledger(
        program_path=program_path,
        event_root=args.event_root,
        repo_root=repo_root,
    )
    roots = {"repo": repo_root, **_artifact_roots(args.artifact_root)}
    manifest = build_foundation_evidence_index(
        child_path=child_path,
        trial_manifest_path=trials_path,
        campaign_state=ledger.state,
        accepted_events=ledger.event_documents,
        artifact_roots=roots,
    )
    output_path = _repo_output(repo_root, args.output)
    created = write_foundation_evidence_index(output_path, manifest)
    print(
        json.dumps(
            {
                "created": created,
                "output": str(output_path),
                "trial_count": len(manifest["trials"]),
                "ledger_tip": ledger.ledger_tip,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _repo_path(repo_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else repo_root / value


def _repo_output(repo_root: Path, value: Path) -> Path:
    candidate = _repo_path(repo_root, value)
    parent = candidate.parent.resolve()
    try:
        parent.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("--output must be inside the repository") from exc
    if any(path.is_symlink() for path in (candidate, *candidate.absolute().parents)):
        raise ValueError("--output must not traverse symlinks")
    return parent / candidate.name


def _artifact_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        root_id, separator, raw_path = value.partition("=")
        if not separator or not root_id or not raw_path or root_id in roots or root_id == "repo":
            raise ValueError("--artifact-root must be a unique non-repo ID=PATH pair")
        roots[root_id] = Path(raw_path)
    return roots


if __name__ == "__main__":
    main()
