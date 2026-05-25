#!/usr/bin/env python3
"""Report source files over a configured line-count limit."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_ROOTS = ("src", "scripts", "frontend/src")
SOURCE_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".css"}
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
}


@dataclass(frozen=True, slots=True)
class SourceFileSize:
    path: Path
    lines: int


def oversized_source_files(
    roots: Sequence[Path],
    *,
    max_lines: int,
    cwd: Path | None = None,
) -> list[SourceFileSize]:
    """Return source files whose line count is above ``max_lines``."""

    base = Path.cwd() if cwd is None else cwd
    offenders: list[SourceFileSize] = []
    for source in iter_source_files(roots, cwd=base):
        lines = line_count(source)
        if lines > max_lines:
            offenders.append(SourceFileSize(_display_path(source, base), lines))
    return sorted(offenders, key=lambda item: (-item.lines, str(item.path)))


def iter_source_files(roots: Sequence[Path], *, cwd: Path) -> Iterable[Path]:
    """Yield tracked-source-like files under the configured roots."""

    for root in roots:
        full_root = root if root.is_absolute() else cwd / root
        if not full_root.exists():
            continue
        if full_root.is_file():
            if _is_source_file(full_root):
                yield full_root
            continue
        for path in full_root.rglob("*"):
            if path.is_file() and _is_source_file(path):
                yield path


def line_count(path: Path) -> int:
    """Count lines without requiring source files to be valid UTF-8."""

    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        dest="roots",
        action="append",
        type=Path,
        help="Source root to scan. May be passed multiple times.",
    )
    parser.add_argument("--max-lines", type=int, default=1000)
    parser.add_argument(
        "--fail",
        action="store_true",
        help="Exit nonzero when oversized files are found.",
    )
    args = parser.parse_args(argv)

    roots = tuple(args.roots or (Path(root) for root in DEFAULT_ROOTS))
    offenders = oversized_source_files(roots, max_lines=args.max_lines)
    if not offenders:
        print(f"source-size check: no source files over {args.max_lines} lines")
        return 0

    print(f"source-size check: {len(offenders)} source files over {args.max_lines} lines")
    for offender in offenders:
        print(f"{offender.lines:>6} {offender.path}")
    return 1 if args.fail else 0


def _is_source_file(path: Path) -> bool:
    return path.suffix in SOURCE_SUFFIXES and not _is_excluded(path)


def _display_path(path: Path, cwd: Path) -> Path:
    try:
        return path.relative_to(cwd)
    except ValueError:
        return path


def _is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


if __name__ == "__main__":
    raise SystemExit(main())
