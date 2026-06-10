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
    "frontend/src/pages/episodes/InspectorDebugPanels.tsx",
}

ALLOW_LINE_SUBSTRINGS = {
    "model sites",
    "selected model sites",
}

BANNED_PHRASES = {
    "artifact readout": "Use a user-facing noun such as 'Probe readout' outside debug/provenance panels.",
    "artifact exists": "Use the user-facing object name, usually 'probe'.",
    "artifact family": "Use the user-facing object name, usually 'lens'.",
    "best use:": "Avoid rubric prose in primary UI; use a short state label and action.",
    "boundary cases": "Name the action or state directly, e.g. 'review suspicious failures'.",
    "before drawing conclusions": "Avoid advisory footnotes; show a concrete trust state.",
    "before treating": "Avoid advisory footnotes; show the required evidence state.",
    "claim should": "Avoid epistemic footnotes in primary UI; use a trust/status label instead.",
    "compatible scored": "Use direct count language such as 'scored'.",
    "dataset-level bundle": "Use task language such as 'dataset view'.",
    "discovery artifact": "Use 'lens', 'probe', or 'saved analysis' in primary UI.",
    "debug sanity": "Use concrete status copy, e.g. 'training episodes only'.",
    "evidence moment": "Use task language such as 'episode' or 'policy call'.",
    "feature-level readout": "Use concrete detail language such as 'feature details'.",
    "indexed result map": "Use task language such as 'Probe results'.",
    "indexed split map": "Use task language such as 'Split coverage'.",
    "model locus": "Use a concrete phrase such as 'layer' or 'activation source'.",
    "model site": "Use a more specific phrase such as 'layer', 'activation tensor', or the actual tensor/module name.",
    "not semantic-labeled": "Use plain language such as 'unlabeled dimension'.",
    "per-call prediction rows": "Use task language such as 'probe predictions'.",
    "policy-call readout": "Use task language such as 'Probe across policy calls'.",
    "probe read sources": "Use task language such as 'Probe input layers'.",
    "read source": "Use 'Probe input'.",
    "readout uses": "Use direct action language such as 'used by this probe result'.",
    "row metadata": "Use the actual concept, e.g. 'input metadata'.",
    "sanity check": "Use concrete split/evidence language.",
    "selected source": "Use a concrete phrase such as 'selected input'.",
    "single source cell": "Use concrete count language such as 'one scored row'.",
    "side-panel evidence": "Use the concrete panel or concept name.",
    "source cell": "Use concrete count language such as 'scored row'.",
    "source row": "Use concrete prediction language.",
    "selected probe artifact": "Use 'selected probe' unless the UI is explicitly showing provenance.",
    "treat it as": "Avoid advisory footnotes; show the trust state directly.",
    "trust:": "Avoid rubric prose in primary UI; render trust as a label/state.",
    "use probe site": "Use concrete action language such as 'Inspect probe input'.",
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
        for line_number, line in enumerate(text.splitlines(), start=1):
            lowered = line.lower()
            if any(allowed in lowered for allowed in ALLOW_LINE_SUBSTRINGS):
                continue
            for phrase, guidance in BANNED_PHRASES.items():
                if phrase in lowered:
                    failures.append(
                        f"{path.relative_to(ROOT)}:{line_number}: banned copy phrase '{phrase}'. {guidance}"
                    )

    if failures:
        print("Frontend copy check failed:\n")
        print("\n".join(failures))
        return 1
    print("Frontend copy check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
