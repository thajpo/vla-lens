"""HTTP response and request helpers for the dashboard API."""

from __future__ import annotations

import json
from collections.abc import Callable
from http import HTTPStatus
from pathlib import Path
from typing import Any

from fastapi import Request
from starlette.responses import FileResponse, Response

from vla_lens.server.common import _api_exception_message, _api_exception_status
from vla_lens.server.state import DashboardState
from vla_lens.traces import TraceBundle

JSON_CACHE_CONTROL = "private, max-age=2"
MEDIA_CACHE_CONTROL = "public, max-age=31536000, immutable"
NO_STORE_CACHE_CONTROL = "no-store"


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
    try:
        state.refresh_dataset_if_needed()
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
    return _error_response(_api_exception_status(exc), _api_exception_message(exc))


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
