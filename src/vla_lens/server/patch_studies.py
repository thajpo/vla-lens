"""Read compact patch-study analyses for the research workbench."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from starlette.responses import Response

from vla_lens.server.http import NO_STORE_CACHE_CONTROL, _handle_json


def register_patch_study_routes(app: FastAPI) -> None:
    """Register the compact analysis endpoint outside the main route module."""

    @app.get("/api/patch-studies")
    async def patch_studies_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: patch_studies_payload(state.root),
            cache_control=NO_STORE_CACHE_CONTROL,
        )


def patch_studies_payload(root: Path) -> dict[str, Any]:
    studies_root = root / "vla_lens" / "patch_studies"
    analyses: list[tuple[float, dict[str, Any]]] = []
    for path in studies_root.glob("*/analysis.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        analyses.append((path.stat().st_mtime, payload))
    studies = [payload for _mtime, payload in sorted(analyses, reverse=True)]
    return {"patch_studies": studies, "total": len(studies)}


__all__ = ["patch_studies_payload", "register_patch_study_routes"]
