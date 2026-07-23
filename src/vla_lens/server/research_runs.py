"""Read-only ResearchRun dashboard routes."""

from __future__ import annotations

from fastapi import FastAPI, Request
from starlette.responses import Response

from vla_lens.server.http import NO_STORE_CACHE_CONTROL, _handle_json
from vla_lens.server.workbench_payloads import _research_run_payload, _research_runs_payload


def register_research_run_routes(app: FastAPI) -> None:
    """Register lifecycle-list and detail endpoints on the dashboard app."""

    @app.get("/api/research-runs")
    async def research_runs_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: _research_runs_payload(state.dataset),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/research-runs/{run_id}")
    async def research_run_endpoint(request: Request, run_id: str) -> Response:
        return _handle_json(
            request,
            lambda state, _query: _research_run_payload(state.dataset, run_id),
            cache_control=NO_STORE_CACHE_CONTROL,
        )
