"""FastAPI dashboard server for VLA Lens datasets."""

from __future__ import annotations

import mimetypes
from http import HTTPStatus
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import Response

import vla_lens.server.activation as activation_api
import vla_lens.server.attention as attention_api
import vla_lens.server.dataset as dataset_api
import vla_lens.server.indexed as indexed_api
import vla_lens.server.indexed_probes as indexed_probes_api
import vla_lens.server.interventions as interventions_api
import vla_lens.server.metrics as metrics_api
import vla_lens.server.probes as probes_api
import vla_lens.server.spatial as spatial_api
from vla_lens.server.http import (
    NO_STORE_CACHE_CONTROL,
    _error_response,
    _handle_bundle_json,
    _handle_file,
    _handle_health,
    _handle_json,
    _handle_optional_file,
    _handle_post_body_json,
    _handle_post_json,
)
from vla_lens.server.openapi import (
    API_DESCRIPTION,
    DOCS_URL,
    OPENAPI_URL,
    REDOC_URL,
    _openapi_schema,
)
from vla_lens.server.state import DashboardState
from vla_lens.workbench import (
    resolve_workspace,
    spatial_overlay_contracts,
    validate_workbench_contracts,
)


def create_dashboard_app(root: str | Path) -> FastAPI:
    """Create the FastAPI dashboard app for a local dataset root."""

    app = FastAPI(
        title="VLA Lens Dashboard API",
        version="0.1.0",
        description=API_DESCRIPTION,
        openapi_url=OPENAPI_URL,
        docs_url=DOCS_URL,
        redoc_url=REDOC_URL,
    )
    app.router.redirect_slashes = False
    app.state.dashboard = DashboardState(root)
    app.openapi = lambda: _openapi_schema(app)  # type: ignore[method-assign]

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
                lambda _dataset: indexed_api.indexed_dataset_payload(
                    state.root,
                    state.index_manifest,
                ),
            ),
        )

    @app.get("/api/counterfactual-pairs")
    async def counterfactual_pairs_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: state.cached_payload(
                "counterfactual-pairs",
                lambda _dataset: indexed_api.counterfactual_pairs_from_index(state.root),
            ),
        )

    @app.get("/api/observational-comparisons")
    async def observational_comparisons_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, query: dataset_api._observational_comparisons_payload(
                state.dataset,
                query,
            ),
        )

    @app.get("/api/workbench")
    async def workbench_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: state.cached_payload("workbench", dataset_api._workbench_payload),
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
            lambda state, _query: dataset_api._lens_arrays_payload(state.dataset),
        )

    @app.get("/api/lens-arrays/{array_id}")
    async def lens_array_meta_endpoint(request: Request, array_id: str) -> Response:
        return _handle_json(
            request,
            lambda state, _query: dataset_api._lens_array_meta_payload(state.dataset, array_id),
        )

    @app.get("/api/cohorts")
    async def cohorts_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: dataset_api._cohorts_payload(state.dataset),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/analysis-runs")
    async def analysis_runs_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: dataset_api._analysis_runs_payload(state.dataset),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/workspaces")
    async def workspaces_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: dataset_api._workspaces_payload(state.dataset),
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
            lambda state, _query: dataset_api._intervention_runs_payload(state.dataset),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/intervention-runs/{run_id}")
    async def intervention_run_endpoint(request: Request, run_id: str) -> Response:
        return _handle_json(
            request,
            lambda state, _query: dataset_api._intervention_run_payload(state.dataset, run_id),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/unit-profile")
    async def unit_profile_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, query: dataset_api._unit_profile_payload(state.dataset, query),
        )

    @app.get("/api/dataset-diagnostics")
    async def dataset_diagnostics_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: dataset_api._dataset_diagnostics_payload(state.dataset),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/episode-annotations")
    async def episode_annotations_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, query: dataset_api._episode_annotations_payload(
                state.root,
                trace_id=query.get("trace_id", [None])[0],
            ),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/dataset-diagnostics/run")
    async def run_dataset_diagnostics_get_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: dataset_api._run_dataset_diagnostics_payload(state.dataset),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/artifacts")
    async def artifacts_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: indexed_api.indexed_artifacts_payload(state.root),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/artifacts/{artifact_id:path}")
    async def artifact_detail_endpoint(request: Request, artifact_id: str) -> Response:
        return _handle_json(
            request,
            lambda state, _query: dataset_api._artifact_detail_payload(state.dataset, artifact_id),
            cache_control=NO_STORE_CACHE_CONTROL,
        )

    @app.get("/api/episodes")
    async def episodes_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, query: indexed_api.indexed_episodes_payload(state.root, query),
        )

    @app.get("/api/episodes/{trace_id}/neighbors")
    async def episode_neighbors_endpoint(request: Request, trace_id: str) -> Response:
        return _handle_json(
            request,
            lambda state, _query: indexed_api.indexed_episode_neighbors_payload(
                state.root,
                trace_id,
            ),
        )

    @app.get("/api/episodes/{trace_id}")
    async def episode_endpoint(request: Request, trace_id: str) -> Response:
        return _handle_json(
            request,
            lambda state, _query: dataset_api._episode_payload(state.dataset.bundle(trace_id)),
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
            lambda bundle, _query: metrics_api._policy_calls_payload(bundle),
        )

    @app.get("/api/action-norm")
    async def action_norm_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, _query: metrics_api._action_norm_payload(bundle),
        )

    @app.get("/api/generation-commitment")
    async def generation_commitment_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, _query: metrics_api._generation_commitment_payload(bundle),
        )

    @app.get("/api/episode-metrics")
    async def episode_metrics_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, _query: metrics_api._episode_metrics_payload(bundle),
        )

    @app.get("/api/episode-interactions")
    async def episode_interactions_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, query: probes_api._episode_interactions_payload(state.dataset, query),
        )

    @app.get("/api/episode-probes")
    async def episode_probes_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, query: indexed_probes_api.indexed_episode_probes_payload(
                state.root,
                query,
            ),
        )

    @app.get("/api/probe-index")
    async def probe_index_endpoint(request: Request) -> Response:
        return _handle_json(
            request,
            lambda state, _query: indexed_probes_api.indexed_probe_index_payload(state.root),
        )

    @app.get("/api/probes/{probe_id}/evidence")
    async def probe_evidence_endpoint(request: Request, probe_id: str) -> Response:
        return _handle_json(
            request,
            lambda state, query: indexed_probes_api.indexed_probe_evidence_payload(
                state.root,
                probe_id,
                query,
            ),
        )

    @app.get("/api/activation-sites")
    async def activation_sites_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, _query: activation_api._activation_sites_payload(bundle),
        )

    @app.get("/api/activation-slice")
    async def activation_slice_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, query: activation_api._activation_slice_payload(bundle, query),
        )

    @app.get("/api/image-token-map")
    async def image_token_map_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, query: activation_api._image_token_map_payload(bundle, query),
        )

    @app.get("/api/object-camera-overlay")
    async def object_camera_overlay_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, query: spatial_api._object_camera_overlay_payload(bundle, query),
        )

    @app.get("/api/attention-map")
    async def attention_map_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, query: attention_api._attention_map_payload(bundle, query),
        )

    @app.get("/api/patch-features")
    async def patch_features_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, query: attention_api._patch_features_payload(bundle, query),
        )

    @app.get("/api/prompt-attention")
    async def prompt_attention_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, query: attention_api._prompt_attention_payload(bundle, query),
        )

    @app.get("/api/prompt-feature-map")
    async def prompt_feature_map_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, query: activation_api._prompt_feature_map_payload(bundle, query),
        )

    @app.get("/api/expert-token-activations")
    async def expert_token_activations_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, query: attention_api._expert_token_model_sites_payload(bundle, query),
        )

    @app.get("/api/expert-token-details")
    async def expert_token_details_endpoint(request: Request) -> Response:
        return _handle_bundle_json(
            request,
            lambda bundle, query: attention_api._expert_token_details_payload(bundle, query),
        )

    @app.post("/api/dataset-diagnostics/run")
    async def run_dataset_diagnostics_post_endpoint(request: Request) -> Response:
        return _handle_post_json(
            request,
            lambda state, _body: dataset_api._run_dataset_diagnostics_payload(state.dataset),
        )

    @app.post("/api/episode-annotations")
    async def save_episode_annotation_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: dataset_api._save_episode_annotation_payload(state.root, body),
        )

    @app.post("/api/selections/resolve")
    async def resolve_selection_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: dataset_api._resolve_selection_payload(state.dataset, body),
        )

    @app.post("/api/cohorts")
    async def save_cohort_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: dataset_api._save_cohort_payload(state.dataset, body),
        )

    @app.post("/api/cohorts/from-selection")
    async def save_cohort_from_selection_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: dataset_api._save_cohort_from_selection_payload(
                state.dataset,
                body,
            ),
        )

    @app.post("/api/cohorts/compare")
    async def cohort_compare_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: dataset_api._cohort_compare_payload(state.dataset, body),
        )

    @app.post("/api/analysis-runs")
    async def save_analysis_run_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: dataset_api._save_analysis_run_payload(state.dataset, body),
        )

    @app.post("/api/intervention-runs")
    async def save_intervention_run_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: dataset_api._save_intervention_run_payload(state.dataset, body),
        )

    @app.post("/api/interventions/preflight")
    async def intervention_preflight_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: interventions_api._intervention_preflight_payload(
                state.dataset,
                body,
            ),
        )

    @app.post("/api/workspaces")
    async def save_workspace_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: dataset_api._save_workspace_payload(state.dataset, body),
        )

    @app.post("/api/projection")
    async def projection_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: dataset_api._projection_payload(state.dataset, body),
        )

    @app.post("/api/graph")
    async def graph_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: dataset_api._graph_payload(state.dataset, body),
        )

    @app.post("/api/tables/query")
    async def table_query_endpoint(request: Request) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: dataset_api._table_query_payload(state.dataset, body),
        )

    @app.post("/api/lens-arrays/{array_id}/slice")
    async def lens_array_slice_endpoint(request: Request, array_id: str) -> Response:
        return await _handle_post_body_json(
            request,
            lambda state, body: dataset_api._lens_array_slice_payload(
                state.dataset,
                array_id,
                body,
            ),
        )

    @app.post("/api/artifacts/create/outcome-probe")
    async def create_outcome_probe_endpoint(request: Request) -> Response:
        return _handle_post_json(
            request,
            lambda state, _body: dataset_api._create_outcome_probe_payload(state.dataset),
        )

    @app.post("/api/artifacts/create/target-object-probe")
    async def create_target_object_probe_endpoint(request: Request) -> Response:
        return _handle_post_json(
            request,
            lambda state, _body: dataset_api._create_target_object_probe_payload(state.dataset),
        )

    @app.post("/api/artifacts/create/action-generation")
    async def create_action_generation_endpoint(request: Request) -> Response:
        return _handle_post_json(
            request,
            lambda state, _body: dataset_api._create_action_generation_payload(state.dataset),
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
    """Serve a LeRobot-backed VLA Lens dashboard from local disk with FastAPI."""

    app = create_dashboard_app(root)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    print(f"vla-lens dashboard: http://{host}:{port}", flush=True)
    server.run()
    return server


__all__ = [
    "create_dashboard_app",
    "run_dashboard_fastapi_server",
]
