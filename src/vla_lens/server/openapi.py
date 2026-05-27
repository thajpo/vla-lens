"""OpenAPI descriptions for the dashboard API."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

OPENAPI_URL = "/api/openapi.json"
DOCS_URL = "/api/docs"
REDOC_URL = "/api/redoc"

REDOC_URL = "/api/redoc"

API_DESCRIPTION = """
Local dashboard API for one VLA Lens dataset root.

The server is intentionally file-backed and single-user oriented: normal GET
routes inspect the current LeRobot v3 + vla_lens overlay dataset, and POST
routes persist workbench state or derived artifacts back under the same root.
"""


def _query_param(
    name: str,
    description: str,
    *,
    required: bool = False,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "in": "query",
        "required": required,
        "schema": schema or {"type": "string"},
        "description": description,
    }


def _json_body(description: str) -> dict[str, Any]:
    return {
        "required": False,
        "description": description,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "additionalProperties": True,
                },
            },
        },
    }


TRACE_ID_PARAM = _query_param(
    "trace_id",
    "Episode trace ID from `/api/dataset` or `/api/episodes/{trace_id}`.",
    required=True,
)
OPTIONAL_TRACE_ID_PARAM = _query_param(
    "trace_id",
    "Optional episode trace ID used to narrow the response.",
)
CALL_PARAM = _query_param(
    "call",
    "Policy-call index. Defaults to the first available call when omitted.",
    schema={"type": "integer", "minimum": 0},
)
NAME_PARAM = _query_param(
    "name",
    "Model-site name from `/api/activation-sites`.",
)
FEATURE_PARAM = _query_param(
    "feature",
    "Channel or feature index to inspect.",
    schema={"type": "integer", "minimum": 0},
)
GENERATION_STEP_PARAM = _query_param(
    "generation_step",
    "Denoising/action-generation step for tensors with a generation_step axis.",
    schema={"type": "integer", "minimum": 0},
)
CAMERA_PARAM = _query_param(
    "camera",
    "Camera name from the episode detail payload, or `all` where supported.",
    required=True,
)
TIMESTEP_PARAM = _query_param(
    "timestep",
    "Environment timestep/frame index.",
    required=True,
    schema={"type": "integer", "minimum": 0},
)
VERSION_PARAM = _query_param(
    "v",
    "Optional cache-busting dataset/media version fingerprint.",
)
HEAD_PARAM = _query_param(
    "head",
    "Attention-head index. Omit to average across heads.",
    schema={"type": "integer", "minimum": 0},
)
QUERY_TOKEN_PARAM = _query_param(
    "query_token",
    "Attention query-token index. Omit to average across query tokens.",
    schema={"type": "integer", "minimum": 0},
)
KIND_PARAM = _query_param(
    "kind",
    "Attention family to inspect, usually `expert` or `vlm`.",
)

API_ROUTE_DOCS: dict[tuple[str, str], dict[str, Any]] = {
    ("get", "/"): {
        "summary": "Service pointer",
        "description": "Returns a minimal pointer to the dashboard API and frontend.",
    },
    ("get", "/api/health"): {
        "summary": "Health and dataset count",
        "description": "Checks that the dataset opened and reports episode/model-site counts.",
    },
    ("get", "/api/dataset"): {
        "summary": "Dataset overview",
        "description": "Returns episodes, capability flags, artifact counts, and dataset metadata.",
    },
    ("get", "/api/counterfactual-pairs"): {
        "summary": "Counterfactual pair groups",
        "description": "Lists clean/corrupt or related trace groups declared in episode metadata.",
    },
    ("get", "/api/observational-comparisons"): {
        "summary": "Observational comparison candidates",
        "description": "Ranks comparable episodes for inspection; this is not causal evidence.",
        "parameters": [
            OPTIONAL_TRACE_ID_PARAM,
            _query_param(
                "probe_id",
                "Optional probe artifact used to score comparison candidates.",
            ),
            _query_param("limit", "Maximum candidates to return.", schema={"type": "integer"}),
        ],
    },
    ("get", "/api/workbench"): {
        "summary": "Workbench manifest",
        "description": "Returns axes, lens arrays, tables, panels, workflows, and saved state.",
    },
    ("get", "/api/workbench/validate"): {
        "summary": "Workbench contract validation",
        "description": "Runs lightweight validation over manifest references and panel contracts.",
    },
    ("get", "/api/spatial-overlays"): {
        "summary": "Spatial overlay contracts",
        "description": "Lists camera/object overlay contracts available for the current dataset.",
    },
    ("get", "/api/lens-arrays"): {
        "summary": "Lens array catalog",
        "description": "Lists tensor/table/image/video arrays addressable by workbench panels.",
    },
    ("get", "/api/lens-arrays/{array_id}"): {
        "summary": "Lens array metadata",
        "description": "Returns metadata for one lens array without reading the full payload.",
    },
    ("get", "/api/cohorts"): {
        "summary": "Saved cohorts",
        "description": "Lists saved episode/timestep/unit cohorts.",
    },
    ("get", "/api/analysis-runs"): {
        "summary": "Saved analysis runs",
        "description": "Lists saved analysis run records.",
    },
    ("get", "/api/workspaces"): {
        "summary": "Saved workspaces",
        "description": "Lists saved dashboard/workbench layouts and selections.",
    },
    ("get", "/api/workspaces/{workspace_id}/resolve"): {
        "summary": "Resolve workspace",
        "description": "Expands a saved workspace into the current dataset context.",
    },
    ("get", "/api/intervention-runs"): {
        "summary": "Saved intervention runs",
        "description": "Lists recorded intervention or replay run metadata.",
    },
    ("get", "/api/unit-profile"): {
        "summary": "Unit profile",
        "description": "Returns available evidence for a selected model unit or direction.",
    },
    ("get", "/api/dataset-diagnostics"): {
        "summary": "Dataset diagnostics",
        "description": "Returns the latest saved dataset analyzer result when present.",
    },
    ("get", "/api/episode-annotations"): {
        "summary": "Episode annotations",
        "description": "Lists saved notes/stars, optionally for one trace.",
        "parameters": [OPTIONAL_TRACE_ID_PARAM],
    },
    ("get", "/api/dataset-diagnostics/run"): {
        "summary": "Run dataset diagnostics",
        "description": "Runs the dataset analyzer and returns its current recommendation payload.",
    },
    ("get", "/api/artifacts"): {
        "summary": "Artifact index",
        "description": "Lists saved probes, videos, reports, and action-generation artifacts.",
    },
    ("get", "/api/artifacts/{artifact_id}"): {
        "summary": "Artifact detail",
        "description": "Returns one artifact record plus lightweight array previews.",
    },
    ("get", "/api/episodes/{trace_id}"): {
        "summary": "Episode detail",
        "description": "Returns per-episode cameras, model arrays, artifacts, and metadata.",
    },
    ("get", "/api/frame"): {
        "summary": "Episode frame JPEG",
        "description": (
            "Returns one encoded camera frame from trace media, replay, or LeRobot video."
        ),
        "parameters": [
            TRACE_ID_PARAM,
            CAMERA_PARAM,
            TIMESTEP_PARAM,
            _query_param("source", "`auto`, `trace`, `sparse`, or `replay` frame source."),
            VERSION_PARAM,
        ],
    },
    ("get", "/api/episode-video"): {
        "summary": "Episode MP4",
        "description": (
            "Builds or returns a cached MP4 for one camera or a stitched all-camera view."
        ),
        "parameters": [
            TRACE_ID_PARAM,
            _query_param("camera", "Camera name or `all`.", schema={"type": "string"}),
            _query_param("fps", "Output frames per second.", schema={"type": "integer"}),
            _query_param("max_width", "Maximum output frame width.", schema={"type": "integer"}),
            VERSION_PARAM,
        ],
    },
    ("get", "/api/policy-calls"): {
        "summary": "Policy calls",
        "description": "Lists model-policy calls aligned to environment timesteps.",
        "parameters": [TRACE_ID_PARAM],
    },
    ("get", "/api/action-norm"): {
        "summary": "Action norm series",
        "description": "Returns executed-action magnitude series for one episode.",
        "parameters": [TRACE_ID_PARAM],
    },
    ("get", "/api/generation-commitment"): {
        "summary": "Generation commitment series",
        "description": "Summarizes how iterative action generation moves toward the final chunk.",
        "parameters": [TRACE_ID_PARAM],
    },
    ("get", "/api/episode-metrics"): {
        "summary": "Episode metric series",
        "description": "Returns action, reward, robot-state, and generation metric series.",
        "parameters": [TRACE_ID_PARAM],
    },
    ("get", "/api/episode-interactions"): {
        "summary": "Episode interaction labels",
        "description": (
            "Returns object interaction and movement/lift/contact labels for an episode."
        ),
        "parameters": [OPTIONAL_TRACE_ID_PARAM],
    },
    ("get", "/api/episode-probes"): {
        "summary": "Episode probe evidence",
        "description": "Returns probe predictions and source rows linked to an episode.",
        "parameters": [TRACE_ID_PARAM],
    },
    ("get", "/api/probe-index"): {
        "summary": "Probe artifact index",
        "description": "Returns probe artifacts summarized by trace and split.",
    },
    ("get", "/api/activation-sites"): {
        "summary": "Activation site catalog",
        "description": "Lists model sites captured for one episode.",
        "parameters": [TRACE_ID_PARAM],
    },
    ("get", "/api/activation-slice"): {
        "summary": "Activation vector slice",
        "description": "Returns a ranked feature/channel slice for one model site and call.",
        "parameters": [
            TRACE_ID_PARAM,
            NAME_PARAM,
            CALL_PARAM,
            FEATURE_PARAM,
            GENERATION_STEP_PARAM,
        ],
    },
    ("get", "/api/image-token-map"): {
        "summary": "Image token heatmap",
        "description": "Maps a model-site feature over image patch tokens and cameras.",
        "parameters": [
            TRACE_ID_PARAM,
            NAME_PARAM,
            CALL_PARAM,
            FEATURE_PARAM,
            GENERATION_STEP_PARAM,
        ],
    },
    ("get", "/api/object-camera-overlay"): {
        "summary": "Object camera overlay",
        "description": (
            "Projects object positions or boxes into one camera frame when context exists."
        ),
        "parameters": [TRACE_ID_PARAM, CAMERA_PARAM, TIMESTEP_PARAM],
    },
    ("get", "/api/attention-map"): {
        "summary": "Attention image map",
        "description": "Maps attention key mass over image patch tokens.",
        "parameters": [
            TRACE_ID_PARAM,
            KIND_PARAM,
            CALL_PARAM,
            NAME_PARAM,
            GENERATION_STEP_PARAM,
            HEAD_PARAM,
            QUERY_TOKEN_PARAM,
        ],
    },
    ("get", "/api/patch-features"): {
        "summary": "Patch feature detail",
        "description": "Returns feature details for one camera patch row/column.",
        "parameters": [
            TRACE_ID_PARAM,
            NAME_PARAM,
            CALL_PARAM,
            CAMERA_PARAM,
            _query_param("row", "Patch row.", schema={"type": "integer"}),
            _query_param("col", "Patch column.", schema={"type": "integer"}),
            FEATURE_PARAM,
            GENERATION_STEP_PARAM,
        ],
    },
    ("get", "/api/prompt-attention"): {
        "summary": "Prompt attention detail",
        "description": (
            "Summarizes attention mass over prompt text, image patches, and action suffix."
        ),
        "parameters": [
            TRACE_ID_PARAM,
            KIND_PARAM,
            CALL_PARAM,
            NAME_PARAM,
            GENERATION_STEP_PARAM,
            HEAD_PARAM,
            QUERY_TOKEN_PARAM,
        ],
    },
    ("get", "/api/prompt-feature-map"): {
        "summary": "Prompt feature map",
        "description": "Maps a selected feature over prompt token rows.",
        "parameters": [
            TRACE_ID_PARAM,
            NAME_PARAM,
            CALL_PARAM,
            FEATURE_PARAM,
            GENERATION_STEP_PARAM,
        ],
    },
    ("get", "/api/expert-token-activations"): {
        "summary": "Expert token activations",
        "description": "Returns feature values across action/noise tokens for one expert site.",
        "parameters": [
            TRACE_ID_PARAM,
            NAME_PARAM,
            CALL_PARAM,
            FEATURE_PARAM,
            GENERATION_STEP_PARAM,
        ],
    },
    ("get", "/api/expert-token-details"): {
        "summary": "Expert token detail",
        "description": "Returns feature, action, and attention context for one expert token.",
        "parameters": [
            TRACE_ID_PARAM,
            NAME_PARAM,
            CALL_PARAM,
            FEATURE_PARAM,
            _query_param("token_index", "Action/noise token index.", schema={"type": "integer"}),
            GENERATION_STEP_PARAM,
        ],
    },
    ("post", "/api/dataset-diagnostics/run"): {
        "summary": "Run dataset diagnostics",
        "description": "Runs the dataset analyzer. No request body is required.",
    },
    ("post", "/api/episode-annotations"): {
        "summary": "Save episode annotation",
        "description": "Saves note/star state for one episode.",
        "requestBody": _json_body(
            "Episode annotation object with trace_id, notes, and starred fields."
        ),
    },
    ("post", "/api/selections/resolve"): {
        "summary": "Resolve selection",
        "description": (
            "Resolves a linked workbench selection into episodes, examples, arrays, and panels."
        ),
        "requestBody": _json_body("SelectionState-shaped object."),
    },
    ("post", "/api/cohorts"): {
        "summary": "Save cohort",
        "description": "Persists a reusable cohort definition.",
        "requestBody": _json_body("CohortSpec-shaped object."),
    },
    ("post", "/api/cohorts/from-selection"): {
        "summary": "Save cohort from selection",
        "description": "Creates a cohort from the current linked selection.",
        "requestBody": _json_body("SelectionState plus optional cohort label/metadata."),
    },
    ("post", "/api/cohorts/compare"): {
        "summary": "Compare cohorts",
        "description": "Computes lightweight differences between saved or selected cohorts.",
        "requestBody": _json_body("Cohort comparison request."),
    },
    ("post", "/api/analysis-runs"): {
        "summary": "Save analysis run",
        "description": "Persists an analysis run record.",
        "requestBody": _json_body("AnalysisRunSpec-shaped object."),
    },
    ("post", "/api/intervention-runs"): {
        "summary": "Save intervention run",
        "description": "Persists an intervention/replay run record.",
        "requestBody": _json_body("InterventionRunSpec-shaped object."),
    },
    ("post", "/api/workspaces"): {
        "summary": "Save workspace",
        "description": "Persists dashboard panel layout, linked selection, and run references.",
        "requestBody": _json_body("SavedWorkspace-shaped object."),
    },
    ("post", "/api/projection"): {
        "summary": "Projection points",
        "description": "Computes or returns projection points for a selection.",
        "requestBody": _json_body("Projection request with selection and array/site references."),
    },
    ("post", "/api/graph"): {
        "summary": "Selection graph",
        "description": "Builds a graph view around a linked selection.",
        "requestBody": _json_body("Graph request with selection and optional graph options."),
    },
    ("post", "/api/tables/query"): {
        "summary": "Query table",
        "description": "Returns filtered table rows for a workbench table.",
        "requestBody": _json_body("Table query request with table_id, filters, and limit."),
    },
    ("post", "/api/lens-arrays/{array_id}/slice"): {
        "summary": "Slice lens array",
        "description": "Reads a bounded JSON preview from one lens array.",
        "requestBody": _json_body("Axis selection object used to slice the lens array."),
    },
    ("post", "/api/artifacts/create/outcome-probe"): {
        "summary": "Create outcome probe artifact",
        "description": "Creates the default outcome probe artifact when the dataset supports it.",
    },
    ("post", "/api/artifacts/create/target-object-probe"): {
        "summary": "Create target-object probe artifact",
        "description": "Creates the default target-object encoding artifact when supported.",
    },
    ("post", "/api/artifacts/create/action-generation"): {
        "summary": "Create action-generation artifact",
        "description": "Creates the default action-generation summary artifact when supported.",
    },
}

def _openapi_schema(app: FastAPI) -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    _annotate_openapi_schema(schema)
    app.openapi_schema = schema
    return schema


def _annotate_openapi_schema(schema: dict[str, Any]) -> None:
    for path, methods in (schema.get("paths") or {}).items():
        if not isinstance(methods, dict):
            continue
        for method, operation in methods.items():
            if not isinstance(operation, dict):
                continue
            docs = API_ROUTE_DOCS.get((str(method).lower(), str(path)))
            if docs is None:
                continue
            if "summary" in docs:
                operation["summary"] = docs["summary"]
            if "description" in docs:
                operation["description"] = docs["description"]
            if docs.get("parameters"):
                _append_openapi_parameters(operation, docs["parameters"])
            if "requestBody" in docs:
                operation["requestBody"] = docs["requestBody"]


def _append_openapi_parameters(
    operation: dict[str, Any],
    parameters: list[dict[str, Any]],
) -> None:
    existing = operation.setdefault("parameters", [])
    seen = {
        (str(item.get("in")), str(item.get("name")))
        for item in existing
        if isinstance(item, dict)
    }
    for parameter in parameters:
        key = (str(parameter.get("in")), str(parameter.get("name")))
        if key not in seen:
            existing.append(parameter)
            seen.add(key)
