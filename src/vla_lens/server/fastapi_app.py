"""FastAPI dashboard server for VLA Lens datasets."""

from __future__ import annotations

import io
import json
import mimetypes
import threading
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from time import monotonic
from typing import Any

import imageio.v2 as imageio
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import FileResponse, Response

import vla_lens.server as legacy
from vla_lens.traces import TraceBundle, TraceDataset
from vla_lens.workbench import (
    resolve_workspace,
    spatial_overlay_contracts,
    validate_workbench_contracts,
)

JSON_CACHE_CONTROL = "private, max-age=2"
MEDIA_CACHE_CONTROL = "public, max-age=31536000, immutable"
NO_STORE_CACHE_CONTROL = "no-store"
OPENAPI_URL = "/api/openapi.json"
DOCS_URL = "/api/docs"
REDOC_URL = "/api/redoc"


class DashboardState:
    """Shared dataset state for the FastAPI server.

    The legacy local server refreshes the dataset when files under the root
    change. Keeping the same behavior here preserves the dashboard workflow
    where analyses write artifacts while the browser is already open.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.dataset = TraceDataset.open(self.root)
        self.dataset_signature = legacy._dataset_signature(self.root)
        self.dataset_signature_checked_at = monotonic()
        self.dataset_signature_check_interval_s = 2.0
        self.payload_cache: dict[str, tuple[tuple[int, int], Any]] = {}
        self.dataset_lock = threading.RLock()
        self.replay_renderers: dict[str, Any] = {}
        self.replay_lock = threading.RLock()

    def refresh_dataset_if_needed(self) -> None:
        now = monotonic()
        with self.dataset_lock:
            if now - self.dataset_signature_checked_at < self.dataset_signature_check_interval_s:
                return
            self.dataset_signature_checked_at = now
        signature = legacy._dataset_signature(self.root)
        with self.dataset_lock:
            if self.dataset_signature == signature:
                return
            self.dataset = TraceDataset.open(self.root)
            self.dataset_signature = signature
            self.payload_cache.clear()
            self.replay_renderers.clear()

    def cached_payload(
        self,
        key: str,
        build: Callable[[TraceDataset], dict[str, Any]],
    ) -> dict[str, Any]:
        signature = self.dataset_signature or legacy._dataset_signature(self.root)
        with self.dataset_lock:
            cached = self.payload_cache.get(key)
            if cached is not None and cached[0] == signature:
                return cached[1]
            payload = build(self.dataset)
            self.payload_cache[key] = (signature, payload)
            return payload

    def clear_payload_cache(self) -> None:
        with self.dataset_lock:
            self.payload_cache.clear()

    def bundle_from_query(self, query: dict[str, list[str]]) -> TraceBundle:
        return self.dataset.bundle(legacy._query_one(query, "trace_id"))

    def frame_bytes(self, query: dict[str, list[str]]) -> bytes:
        bundle = self.bundle_from_query(query)
        camera = legacy._query_one(query, "camera")
        timestep = int(legacy._query_one(query, "timestep"))
        source = query.get("source", ["auto"])[0]
        timestep = max(0, min(timestep, bundle.manifest.length - 1))
        if source == "trace":
            frame_path = legacy._trace_frame_file_path(bundle, camera=camera, timestep=timestep)
            if frame_path is not None:
                return frame_path.read_bytes()
        frame = self._frame(bundle, camera=camera, timestep=timestep, source=source)
        buffer = io.BytesIO()
        imageio.imwrite(buffer, np.asarray(frame), format="jpg", quality=90)
        return buffer.getvalue()

    def frame_file_path(self, query: dict[str, list[str]]) -> Path | None:
        bundle = self.bundle_from_query(query)
        camera = legacy._query_one(query, "camera")
        timestep = int(legacy._query_one(query, "timestep"))
        source = query.get("source", ["auto"])[0]
        timestep = max(0, min(timestep, bundle.manifest.length - 1))
        if source != "trace":
            return None
        return legacy._trace_frame_file_path(bundle, camera=camera, timestep=timestep)

    def episode_video_bytes(self, query: dict[str, list[str]]) -> bytes:
        return self.episode_video_path(query).read_bytes()

    def episode_video_path(self, query: dict[str, list[str]]) -> Path:
        bundle = self.bundle_from_query(query)
        camera = query.get("camera", ["all"])[0]
        fps = int(query.get("fps", ["10"])[0])
        max_width = int(query.get("max_width", ["320"])[0])
        return legacy._episode_video_path(
            bundle,
            camera=camera,
            fps=fps,
            max_width=max_width,
        )

    def _frame(self, bundle: TraceBundle, *, camera: str, timestep: int, source: str) -> np.ndarray:
        if source in {"auto", "sparse"} and legacy.read_sparse_image is not None:
            try:
                sparse = legacy.read_sparse_image(bundle, camera=camera, timestep=timestep)
                if sparse is not None:
                    return sparse
            except Exception:
                if source == "sparse":
                    raise
        if source in {"auto", "replay"} and legacy.PI05LiberoReplayRenderer is not None:
            try:
                return self._replay_renderer(bundle).render(camera=camera, timestep=timestep)
            except Exception:
                if source == "replay":
                    raise
        if source == "auto":
            frame_reader = getattr(bundle, "frame", None)
            if callable(frame_reader):
                return np.asarray(frame_reader(camera, timestep))
        frames = bundle.frames(camera, mmap=True)
        return np.asarray(frames[timestep])

    def _replay_renderer(self, bundle: TraceBundle) -> Any:
        with self.replay_lock:
            renderer = self.replay_renderers.get(bundle.manifest.trace_id)
            if renderer is None:
                if legacy.PI05LiberoReplayRenderer is None:
                    raise RuntimeError("PI0.5 LIBERO replay dependencies are not available")
                renderer = legacy.PI05LiberoReplayRenderer(bundle)
                self.replay_renderers[bundle.manifest.trace_id] = renderer
            return renderer


def create_dashboard_app(root: str | Path) -> FastAPI:
    """Create the FastAPI dashboard app for a local dataset root."""

    app = FastAPI(
        title="VLA Lens Dashboard API",
        version="0.1.0",
        openapi_url=OPENAPI_URL,
        docs_url=DOCS_URL,
        redoc_url=REDOC_URL,
    )
    app.router.redirect_slashes = False
    app.state.dashboard = DashboardState(root)

    @app.get("/")
    async def root_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: {
                "service": "vla-lens-backend",
                "api": "/api/dataset",
                "frontend": "Run the React workbench from frontend/.",
            },
        )

    @app.get("/api/health")
    async def health_endpoint(request: Request) -> Response:
        return _handle_health(request)

    @app.get("/api/dataset")
    async def dataset_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: state.cached_payload(
                "dataset",
                lambda dataset: legacy._dataset_payload(dataset, include_workbench=False),
            ),
        )

    @app.get("/api/counterfactual-pairs")
    async def counterfactual_pairs_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: state.cached_payload(
                "counterfactual-pairs",
                legacy._counterfactual_pairs_response,
            ),
        )

    @app.get("/api/observational-comparisons")
    async def observational_comparisons_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, query: legacy._observational_comparisons_payload(
                state.dataset,
                query,
            ),
        )

    @app.get("/api/workbench")
    async def workbench_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: state.cached_payload("workbench", legacy._workbench_payload),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/workbench/validate")
    async def workbench_validate_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: validate_workbench_contracts(state.dataset),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/spatial-overlays")
    async def spatial_overlays_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: {"overlays": spatial_overlay_contracts(state.dataset)},
        )

    @app.get("/api/lens-arrays")
    async def lens_arrays_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: legacy._lens_arrays_payload(state.dataset),
        )

    @app.get("/api/lens-arrays/{array_id}")
    async def lens_array_meta_endpoint(request: Request, array_id: str) -> Response:
        return _handle_json(
            request,
            lambda state, _query: legacy._lens_array_meta_payload(state.dataset, array_id),
        )

    @app.get("/api/cohorts")
    async def cohorts_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: legacy._cohorts_payload(state.dataset),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/analysis-runs")
    async def analysis_runs_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: legacy._analysis_runs_payload(state.dataset),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/workspaces")
    async def workspaces_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: legacy._workspaces_payload(state.dataset),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/workspaces/{workspace_id}/resolve")
    async def workspace_resolve_endpoint(request: Request, workspace_id: str) -> Response:
        return _handle_json(
            request,
            lambda state, _query: resolve_workspace(state.dataset, workspace_id),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/intervention-runs")
    async def intervention_runs_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: legacy._intervention_runs_payload(state.dataset),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/unit-profile")
    async def unit_profile_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, query: legacy._unit_profile_payload(state.dataset, query),
        )

    @app.get("/api/dataset-diagnostics")
    async def dataset_diagnostics_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: legacy._dataset_diagnostics_payload(state.dataset),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/episode-annotations")
    async def episode_annotations_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, query: legacy._episode_annotations_payload(
                state.root,
                trace_id=query.get("trace_id", [None])[0],
            ),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/dataset-diagnostics/run")
    async def run_dataset_diagnostics_get_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: legacy._run_dataset_diagnostics_payload(state.dataset),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/artifacts")
    async def artifacts_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: legacy._artifacts_payload(state.dataset),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/artifacts/{artifact_id:path}")
    async def artifact_detail_endpoint(request: Request, artifact_id: str) -> Response:
        return _handle_json(
            request,
            lambda state, _query: legacy._artifact_detail_payload(state.dataset, artifact_id),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/episodes/{trace_id}")
    async def episode_endpoint(request: Request, trace_id: str) -> Response:
        return _handle_json(
            request,
            lambda state, _query: legacy._episode_payload(state.dataset.bundle(trace_id)),
        )

    @app.get("/api/frame")
    async def frame_endpoint(request: Request) -> Response:
        return _handle_optional_file(
            request,
            lambda state, query: state.frame_file_path(query),
            lambda state, query: state.frame_bytes(query),
            media_type=mimetypes.types_map.get(".jpg", "image/jpeg"),
        )

    @app.get("/api/episode-video")
    async def episode_video_endpoint(request: Request) -> Response:
        return _handle_file(
            request,
            lambda state, query: state.episode_video_path(query),
            media_type="video/mp4",
        )

    @app.get("/api/policy-calls")
    async def policy_calls_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, _query: legacy._policy_calls_payload(bundle),
        )

    @app.get("/api/action-norm")
    async def action_norm_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, _query: legacy._action_norm_payload(bundle),
        )

    @app.get("/api/generation-commitment")
    async def generation_commitment_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, _query: legacy._generation_commitment_payload(bundle),
        )

    @app.get("/api/episode-metrics")
    async def episode_metrics_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, _query: legacy._episode_metrics_payload(bundle),
        )

    @app.get("/api/episode-interactions")
    async def episode_interactions_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, query: legacy._episode_interactions_payload(state.dataset, query),
        )

    @app.get("/api/episode-probes")
    async def episode_probes_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, query: legacy._episode_probes_payload(state.dataset, query),
        )

    @app.get("/api/probe-index")
    async def probe_index_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: legacy._probe_index_payload(state.dataset),
        )

    @app.get("/api/activation-sites")
    async def activation_sites_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, _query: legacy._activation_sites_payload(bundle),
        )

    @app.get("/api/activation-slice")
    async def activation_slice_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, query: legacy._activation_slice_payload(bundle, query),
        )

    @app.get("/api/image-token-map")
    async def image_token_map_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, query: legacy._image_token_map_payload(bundle, query),
        )

    @app.get("/api/object-camera-overlay")
    async def object_camera_overlay_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, query: legacy._object_camera_overlay_payload(bundle, query),
        )

    @app.get("/api/attention-map")
    async def attention_map_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, query: legacy._attention_map_payload(bundle, query),
        )

    @app.get("/api/patch-features")
    async def patch_features_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, query: legacy._patch_features_payload(bundle, query),
        )

    @app.get("/api/prompt-attention")
    async def prompt_attention_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, query: legacy._prompt_attention_payload(bundle, query),
        )

    @app.get("/api/prompt-feature-map")
    async def prompt_feature_map_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, query: legacy._prompt_feature_map_payload(bundle, query),
        )

    @app.get("/api/expert-token-activations")
    async def expert_token_activations_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, query: legacy._expert_token_model_sites_payload(bundle, query),
        )

    @app.get("/api/expert-token-details")
    async def expert_token_details_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, query: legacy._expert_token_details_payload(bundle, query),
        )

    @app.post("/api/dataset-diagnostics/run")
    async def run_dataset_diagnostics_post_endpoint(request: Request) -> Response:
        return _handle_post_json(
            request,
            lambda state, _body: legacy._run_dataset_diagnostics_payload(state.dataset),
        )

    @app.post("/api/episode-annotations")
    async def save_episode_annotation_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: legacy._save_episode_annotation_payload(state.root, body),
        )

    @app.post("/api/selections/resolve")
    async def resolve_selection_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: legacy._resolve_selection_payload(state.dataset, body),
        )

    @app.post("/api/cohorts")
    async def save_cohort_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: legacy._save_cohort_payload(state.dataset, body),
        )

    @app.post("/api/cohorts/from-selection")
    async def save_cohort_from_selection_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: legacy._save_cohort_from_selection_payload(state.dataset, body),
        )

    @app.post("/api/cohorts/compare")
    async def cohort_compare_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: legacy._cohort_compare_payload(state.dataset, body),
        )

    @app.post("/api/analysis-runs")
    async def save_analysis_run_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: legacy._save_analysis_run_payload(state.dataset, body),
        )

    @app.post("/api/intervention-runs")
    async def save_intervention_run_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: legacy._save_intervention_run_payload(state.dataset, body),
        )

    @app.post("/api/workspaces")
    async def save_workspace_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: legacy._save_workspace_payload(state.dataset, body),
        )

    @app.post("/api/projection")
    async def projection_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: legacy._projection_payload(state.dataset, body),
        )

    @app.post("/api/graph")
    async def graph_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: legacy._graph_payload(state.dataset, body),
        )

    @app.post("/api/tables/query")
    async def table_query_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: legacy._table_query_payload(state.dataset, body),
        )

    @app.post("/api/lens-arrays/{array_id}/slice")
    async def lens_array_slice_endpoint(request: Request, array_id: str) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: legacy._lens_array_slice_payload(state.dataset, array_id, body),
        )

    @app.post("/api/artifacts/create/outcome-probe")
    async def create_outcome_probe_endpoint(request: Request) -> Response:
        return _handle_post_json(
            request,
            lambda state, _body: legacy._create_outcome_probe_payload(state.dataset),
        )

    @app.post("/api/artifacts/create/target-object-probe")
    async def create_target_object_probe_endpoint(request: Request) -> Response:
        return _handle_post_json(
            request,
            lambda state, _body: legacy._create_target_object_probe_payload(state.dataset),
        )

    @app.post("/api/artifacts/create/action-generation")
    async def create_action_generation_endpoint(request: Request) -> Response:
        return _handle_post_json(
            request,
            lambda state, _body: legacy._create_action_generation_payload(state.dataset),
        )

    @app.get("/api", include_in_schema=False)
    async def unknown_api_get_endpoint() -> Response:
        return _error_response(HTTPStatus.NOT_FOUND, "Unknown route: /api")

    @app.get("/api/{path:path}", include_in_schema=False)
    async def unknown_get_endpoint(path: str) -> Response:
        return _error_response(HTTPStatus.NOT_FOUND, f"Unknown route: /api/{path}")

    @app.post("/api", include_in_schema=False)
    async def unknown_api_post_endpoint() -> Response:
        return _error_response(HTTPStatus.NOT_FOUND, "Unknown route: /api")

    @app.post("/api/{path:path}", include_in_schema=False)
    async def unknown_post_endpoint(path: str) -> Response:
        return _error_response(HTTPStatus.NOT_FOUND, f"Unknown route: /api/{path}")

    return app


def run_dashboard_fastapi_server(
    root: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> uvicorn.Server:
    """Serve a trace dataset dashboard from local disk with FastAPI."""

    app = create_dashboard_app(root)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    print(f"vla-lens dashboard: http://{host}:{port}", flush=True)
    server.run()
    return server


def _state(request: Request) -> DashboardState:
    return request.app.state.dashboard


def _query(request: Request) -> dict[str, list[str]]:
    query: dict[str, list[str]] = {}
    for key, value in request.query_params.multi_items():
        if value == "":
            continue
        query.setdefault(key, []).append(value)
    return query


def _handle_health(request: Request) -> Response:
    state = _state(request)
    try:
        dataset = state.dataset
        return _json_response(
            {
                "status": "ok",
                "service": "vla-lens-backend",
                "api": "/api/dataset",
                "dataset": {
                    "root": str(state.root),
                    "episodes": len(dataset.bundles),
                    "activation_sites": int(len(dataset.model_site_index)),
                },
            },
            cache_control=NO_STORE_CACHE_CONTROL,
        )
    except (BrokenPipeError, ConnectionResetError):  # pragma: no cover - client boundary
        raise
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        return _api_exception_response(exc)
    except Exception as exc:  # pragma: no cover - defensive server boundary
        return _error_response(HTTPStatus.INTERNAL_SERVER_ERROR, repr(exc))


def _handle_json(
    request: Request,
    build: Callable[[DashboardState, dict[str, list[str]]], Any],
    *,
    cache_control: str = JSON_CACHE_CONTROL,
) -> Response:
    state = _state(request)
    state.refresh_dataset_if_needed()
    try:
        return _json_response(build(state, _query(request)), cache_control=cache_control)
    except (BrokenPipeError, ConnectionResetError):  # pragma: no cover - client boundary
        raise
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        return _api_exception_response(exc)
    except Exception as exc:  # pragma: no cover - defensive server boundary
        return _error_response(HTTPStatus.INTERNAL_SERVER_ERROR, repr(exc))


def _handle_bundle_json(
    request: Request,
    build: Callable[[TraceBundle, dict[str, list[str]]], Any],
) -> Response:
    return _handle_json(
        request,
        lambda state, query: build(state.bundle_from_query(query), query),
    )


def _handle_binary(
    request: Request,
    build: Callable[[DashboardState, dict[str, list[str]]], bytes],
    *,
    media_type: str,
) -> Response:
    state = _state(request)
    query = _query(request)
    if _media_requires_dataset_refresh(query):
        state.refresh_dataset_if_needed()
    try:
        return Response(
            content=build(state, query),
            media_type=media_type,
            headers={"Cache-Control": _media_cache_control(query)},
        )
    except (BrokenPipeError, ConnectionResetError):  # pragma: no cover - client boundary
        raise
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        return _api_exception_response(exc)
    except Exception as exc:  # pragma: no cover - defensive server boundary
        return _error_response(HTTPStatus.INTERNAL_SERVER_ERROR, repr(exc))


def _handle_file(
    request: Request,
    build_path: Callable[[DashboardState, dict[str, list[str]]], Path],
    *,
    media_type: str,
) -> Response:
    return _handle_optional_file(
        request,
        lambda state, query: build_path(state, query),
        None,
        media_type=media_type,
    )


def _handle_optional_file(
    request: Request,
    build_path: Callable[[DashboardState, dict[str, list[str]]], Path | None],
    build_bytes: Callable[[DashboardState, dict[str, list[str]]], bytes] | None,
    *,
    media_type: str,
) -> Response:
    state = _state(request)
    query = _query(request)
    if _media_requires_dataset_refresh(query):
        state.refresh_dataset_if_needed()
    headers = {"Cache-Control": _media_cache_control(query)}
    try:
        path = build_path(state, query)
        if path is not None:
            return FileResponse(path, media_type=media_type, headers=headers)
        if build_bytes is None:
            raise FileNotFoundError("No file response is available.")
        return Response(
            content=build_bytes(state, query),
            media_type=media_type,
            headers=headers,
        )
    except (BrokenPipeError, ConnectionResetError):  # pragma: no cover - client boundary
        raise
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        return _api_exception_response(exc)
    except Exception as exc:  # pragma: no cover - defensive server boundary
        return _error_response(HTTPStatus.INTERNAL_SERVER_ERROR, repr(exc))


def _handle_post_json(
    request: Request,
    build: Callable[[DashboardState, dict[str, Any]], Any],
) -> Response:
    state = _state(request)
    state.refresh_dataset_if_needed()
    try:
        response = _json_response(build(state, {}), cache_control=NO_STORE_CACHE_CONTROL)
        state.clear_payload_cache()
        return response
    except (BrokenPipeError, ConnectionResetError):  # pragma: no cover - client boundary
        raise
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        return _api_exception_response(exc)
    except Exception as exc:  # pragma: no cover - defensive server boundary
        return _error_response(HTTPStatus.INTERNAL_SERVER_ERROR, repr(exc))


async def _handle_post_body_json(
    request: Request,
    build: Callable[[DashboardState, dict[str, Any]], Any],
) -> Response:
    state = _state(request)
    state.refresh_dataset_if_needed()
    try:
        body = await _read_json_body(request)
        response = _json_response(build(state, body), cache_control=NO_STORE_CACHE_CONTROL)
        state.clear_payload_cache()
        return response
    except (BrokenPipeError, ConnectionResetError):  # pragma: no cover - client boundary
        raise
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        return _api_exception_response(exc)
    except Exception as exc:  # pragma: no cover - defensive server boundary
        return _error_response(HTTPStatus.INTERNAL_SERVER_ERROR, repr(exc))


async def _read_json_body(request: Request) -> dict[str, Any]:
    payload = await request.body()
    if not payload:
        return {}
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("JSON request body must be an object")
    return value


def _json_response(
    value: Any,
    *,
    status: HTTPStatus = HTTPStatus.OK,
    cache_control: str = JSON_CACHE_CONTROL,
) -> Response:
    return Response(
        content=json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8"),
        status_code=int(status),
        media_type="application/json; charset=utf-8",
        headers={"Cache-Control": cache_control},
    )


def _api_exception_response(exc: Exception) -> Response:
    return _error_response(legacy._api_exception_status(exc), legacy._api_exception_message(exc))


def _error_response(status: HTTPStatus, message: str) -> Response:
    return _json_response(
        {"error": status.phrase, "message": message},
        status=status,
        cache_control=NO_STORE_CACHE_CONTROL,
    )


def _media_cache_control(query: dict[str, list[str]]) -> str:
    version = query.get("v", [""])[0]
    if version:
        return MEDIA_CACHE_CONTROL
    return "private, max-age=60"


def _media_requires_dataset_refresh(query: dict[str, list[str]]) -> bool:
    return not query.get("v", [""])[0]


__all__ = [
    "DashboardState",
    "DOCS_URL",
    "OPENAPI_URL",
    "REDOC_URL",
    "create_dashboard_app",
    "run_dashboard_fastapi_server",
]
