#!/usr/bin/env python3
"""Catch frontend copy that leaks LLM caveats or ambiguous internal terms.

This is intentionally a small, explicit phrase check. It is not a style linter
and should not ban normal technical language like "policy call".
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_SRC = ROOT / "frontend" / "src"

ALLOW_PATH_PARTS = {
    "frontend/src/copy/researchCopy.ts",
    "frontend/src/pages/probeDisplayCopy.ts",
    "frontend/src/pages/episodes/InspectorDebugPanels.tsx",
}

BANNED_PHRASES = {
    "artifact readout": "Use a user-facing noun such as 'Probe readout' outside debug/provenance panels.",
    "boundary cases": "Name the action or state directly, e.g. 'review suspicious failures'.",
    "claim should": "Avoid epistemic footnotes in primary UI; use a trust/status label instead.",
    "debug sanity": "Use concrete status copy, e.g. 'training episodes only'.",
    "indexed result map": "Use task language such as 'Probe results'.",
    "indexed split map": "Use task language such as 'Split coverage'.",
    "model site": "Use a more specific phrase such as 'layer', 'activation source', or the actual tensor/module name.",
    "not semantic-labeled": "Use plain language such as 'unlabeled dimension'.",
    "single source cell": "Use concrete count language such as 'one scored row'.",
    "useful for": "Avoid LLM-style justifications; prefer 'Best use:' or a concrete next action.",
}

EXTENSIONS = {".ts", ".tsx"}


def is_allowed(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    return rel in ALLOW_PATH_PARTS


def main() -> int:
    failures: list[str] = []
    for path in sorted(FRONTEND_SRC.rglob("*")):
        if path.suffix not in EXTENSIONS or is_allowed(path):
            continue
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        for phrase, guidance in BANNED_PHRASES.items():
            if phrase in lowered:
                failures.append(f"{path.relative_to(ROOT)}: banned copy phrase '{phrase}'. {guidance}")

    if failures:
        print("Frontend copy check failed:\n")
        print("\n".join(failures))
        return 1
    print("Frontend copy check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
