from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CANONICAL_EVIDENCE_MODULES = [
    REPO_ROOT / "frontend" / "src" / "types" / "probeEvidence.ts",
    REPO_ROOT / "frontend" / "src" / "pages" / "evidencePinsModel.ts",
    REPO_ROOT / "frontend" / "src" / "pages" / "workbench" / "episodeRouteModel.ts",
    REPO_ROOT / "src" / "vla_lens" / "probe_evidence.py",
    REPO_ROOT / "src" / "vla_lens" / "probe_evidence_adapter.py",
]

RESEARCH_UI_ROOTS = [
    REPO_ROOT / "frontend" / "src" / "components" / "interventions",
    REPO_ROOT / "frontend" / "src" / "components" / "panels",
    REPO_ROOT / "frontend" / "src" / "components" / "workflows",
    REPO_ROOT / "frontend" / "src" / "pages",
    REPO_ROOT / "frontend" / "src" / "pages" / "capabilityGating.ts",
    REPO_ROOT / "frontend" / "src" / "pages" / "EvidencePage.tsx",
    REPO_ROOT / "frontend" / "src" / "pages" / "EpisodesPage.tsx",
    REPO_ROOT / "frontend" / "src" / "pages" / "workbench",
    REPO_ROOT / "frontend" / "src" / "pages" / "episodes",
]

RAW_SCHEMA_BRIDGE_ALLOWLIST = {
    "frontend/src/components/interventions/InterventionLab.tsx": (
        "intervention episode/artifact picker bridge"
    ),
    "frontend/src/pages/EpisodesPage.tsx": "episode microscope data-loading bridge",
    "frontend/src/pages/episodes/CameraTimeline.tsx": "camera/object overlay rendering bridge",
    "frontend/src/pages/episodes/EpisodeNavigation.tsx": "episode navigation bridge",
    "frontend/src/pages/episodes/EpisodeProbePanel.tsx": "legacy demoted probe panel bridge",
    "frontend/src/pages/episodes/ExpertPanels.tsx": "expert-token payload rendering bridge",
    "frontend/src/pages/episodes/FramePlaybackControls.tsx": "policy-call playback bridge",
    "frontend/src/pages/episodes/InspectorDebugPanels.tsx": "explicit debug panel exception",
    "frontend/src/pages/episodes/InspectorPanels.tsx": "episode inspector composition bridge",
    "frontend/src/pages/episodes/InspectorTables.tsx": "activation-slice table bridge",
    "frontend/src/pages/episodes/InteractionSummary.tsx": "episode interaction bridge",
    "frontend/src/pages/episodes/LensInspectorPanels.tsx": "EpisodeLensView bridge",
    "frontend/src/pages/episodes/MetricPlots.tsx": "episode metric plot bridge",
    "frontend/src/pages/episodes/PipelineMap.tsx": "activation-site pipeline bridge",
    "frontend/src/pages/episodes/episodeData.ts": "episode list normalization bridge",
    "frontend/src/pages/episodes/episodeLensModel.ts": "legacy lens view to evidence bridge",
    "frontend/src/pages/episodes/episodeProbeModel.ts": "legacy demoted probe readout bridge",
    "frontend/src/pages/episodes/formatters.ts": "dataset overlay formatting bridge",
    "frontend/src/pages/episodes/interventionSeed.ts": "intervention seed bridge",
    "frontend/src/pages/episodes/pipelineModel.ts": "activation-site pipeline model bridge",
    "frontend/src/pages/episodes/shared.ts": "shared episode raw payload type aliases",
    "frontend/src/pages/episodes/siteModel.ts": "activation-site selection bridge",
    "frontend/src/pages/episodes/useEpisodeInspectorModel.ts": "episode inspector data bridge",
    "frontend/src/pages/episodes/useEpisodePrefetch.ts": "episode prefetch bridge",
    "frontend/src/pages/episodes/useEpisodeRouteContext.ts": "policy-call route bridge",
    "frontend/src/pages/episodes/useProbeEvidenceLensContext.ts": "probe evidence fetch bridge",
    "frontend/src/pages/workbench/DatasetBrowser.tsx": "dataset browser API bridge",
    "frontend/src/pages/workbench/datasetBrowserModel.ts": (
        "dataset rows to evidence row model bridge"
    ),
}

FORBIDDEN_FRONTEND_IMPORTS = (
    "../../types/dataset",
    "../types/dataset",
    "./types/dataset",
    "types/dataset",
)
FORBIDDEN_PYTHON_IMPORTS = (
    "vla_lens.traces",
    "vla_lens.server.dataset",
    "vla_lens.server.fastapi_app",
)
IMPORT_RE = re.compile(r"^\s*import(?:\s+type)?\s+.*?['\"](?P<module>[^'\"]+)['\"]", re.MULTILINE)


def test_canonical_evidence_modules_do_not_import_raw_capture_or_page_schemas():
    offenders: list[str] = []

    for path in CANONICAL_EVIDENCE_MODULES:
        text = path.read_text(encoding="utf-8")
        forbidden = (
            FORBIDDEN_FRONTEND_IMPORTS
            if path.suffix in {".ts", ".tsx"}
            else FORBIDDEN_PYTHON_IMPORTS
        )
        for marker in forbidden:
            if marker in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)} imports {marker}")

    assert offenders == []


def test_research_ui_raw_schema_imports_are_explicitly_allowlisted():
    offenders: list[str] = []
    seen: set[str] = set()

    for path in _research_ui_files():
        text = path.read_text(encoding="utf-8")
        imports_raw_schema = any(marker in text for marker in FORBIDDEN_FRONTEND_IMPORTS)
        if not imports_raw_schema:
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        seen.add(relative)
        if relative not in RAW_SCHEMA_BRIDGE_ALLOWLIST:
            offenders.append(f"{relative} imports raw dataset schema without allowlist rationale")

    stale = sorted(set(RAW_SCHEMA_BRIDGE_ALLOWLIST) - seen)
    offenders.extend(f"{path} no longer imports raw dataset schema" for path in stale)
    offenders.extend(
        f"{path} bridge allowlist lacks rationale"
        for path, reason in RAW_SCHEMA_BRIDGE_ALLOWLIST.items()
        if not reason.strip()
    )

    assert offenders == []


def _research_ui_files() -> list[Path]:
    files: set[Path] = set()
    for root in RESEARCH_UI_ROOTS:
        if root.is_file():
            files.add(root)
        else:
            files.update(path for path in root.rglob("*") if path.suffix in {".ts", ".tsx"})
    return sorted(files)
