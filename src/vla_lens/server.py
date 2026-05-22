"""Local slice-based dashboard server for VLA-lens trace datasets."""

from __future__ import annotations

import io
import json
import mimetypes
import re
import threading
import urllib.parse
from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import monotonic
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np
import pandas as pd

from vla_lens.action_generation import save_action_generation_artifact
from vla_lens.analyzer import diagnostics_status, run_dataset_diagnostics
from vla_lens.artifacts import LensArtifact
from vla_lens.probes.workflow import train_probe_artifact_from_spec
from vla_lens.target_object import save_target_object_encoding_artifact
from vla_lens.traces import TraceBundle, TraceDataset
from vla_lens.workbench import (
    AnalysisRunSpec,
    InterventionRunSpec,
    SavedWorkspace,
    SelectionState,
    UnitRef,
    cohort_from_selection,
    compare_cohorts,
    graph_from_selection,
    lens_array_catalog,
    lens_array_meta,
    list_analysis_runs,
    list_cohorts,
    list_intervention_runs,
    list_workspaces,
    projection_points,
    query_table,
    resolve_selection,
    resolve_workspace,
    save_analysis_run,
    save_cohort,
    save_intervention_run,
    save_workspace,
    slice_lens_array,
    spatial_overlay_contracts,
    unit_profile,
    validate_workbench_contracts,
    workbench_manifest,
)

try:  # Replay dependencies are optional outside the PI0.5/LIBERO environment.
    from vla_lens.pi05.replay import PI05LiberoReplayRenderer, read_sparse_image
except Exception:  # pragma: no cover - optional dependency boundary
    PI05LiberoReplayRenderer = None  # type: ignore[assignment]
    read_sparse_image = None  # type: ignore[assignment]


def run_dashboard_server(
    root: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """Serve a trace dataset dashboard from local disk."""
    dataset = TraceDataset.open(root)

    class Handler(TraceDashboardHandler):
        pass

    Handler.dataset = dataset
    Handler.dataset_signature = _dataset_signature(Path(root))
    Handler.dataset_signature_checked_at = monotonic()
    Handler.root = Path(root)
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"vla-lens dashboard: http://{host}:{port}", flush=True)
    server.serve_forever()
    return server


class TraceDashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler that reads only requested trace slices."""

    dataset: TraceDataset
    dataset_signature: tuple[int, int] | None = None
    dataset_signature_checked_at = 0.0
    dataset_signature_check_interval_s = 2.0
    payload_cache: dict[str, tuple[tuple[int, int], Any]] = {}
    dataset_lock = threading.RLock()
    root: Path
    replay_renderers: dict[str, Any] = {}
    replay_lock = threading.RLock()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)
        self._refresh_dataset_if_needed()

        try:
            if path == "/":
                self._send_json(
                    {
                        "service": "vla-lens-backend",
                        "api": "/api/dataset",
                        "frontend": "Run the React workbench from frontend/.",
                    }
                )
            elif path == "/api/dataset":
                self._send_json(
                    self._cached_payload(
                        "dataset",
                        lambda dataset: _dataset_payload(dataset, include_workbench=False),
                    )
                )
            elif path == "/api/counterfactual-pairs":
                self._send_json(
                    self._cached_payload("counterfactual-pairs", _counterfactual_pairs_response)
                )
            elif path == "/api/workbench":
                self._send_json(self._cached_payload("workbench", _workbench_payload))
            elif path == "/api/workbench/validate":
                self._send_json(validate_workbench_contracts(self.dataset))
            elif path == "/api/spatial-overlays":
                self._send_json({"overlays": spatial_overlay_contracts(self.dataset)})
            elif path == "/api/lens-arrays":
                self._send_json(_lens_arrays_payload(self.dataset))
            elif path.startswith("/api/lens-arrays/"):
                array_id = urllib.parse.unquote(path.removeprefix("/api/lens-arrays/"))
                self._send_json(_lens_array_meta_payload(self.dataset, array_id))
            elif path == "/api/cohorts":
                self._send_json(_cohorts_payload(self.dataset))
            elif path == "/api/analysis-runs":
                self._send_json(_analysis_runs_payload(self.dataset))
            elif path == "/api/workspaces":
                self._send_json(_workspaces_payload(self.dataset))
            elif path.startswith("/api/workspaces/") and path.endswith("/resolve"):
                workspace_id = urllib.parse.unquote(
                    path.removeprefix("/api/workspaces/").removesuffix("/resolve")
                )
                self._send_json(resolve_workspace(self.dataset, workspace_id))
            elif path == "/api/intervention-runs":
                self._send_json(_intervention_runs_payload(self.dataset))
            elif path == "/api/unit-profile":
                self._send_json(_unit_profile_payload(self.dataset, query))
            elif path == "/api/dataset-diagnostics":
                self._send_json(_dataset_diagnostics_payload(self.dataset))
            elif path == "/api/episode-annotations":
                self._send_json(
                    _episode_annotations_payload(
                        self.root,
                        trace_id=query.get("trace_id", [None])[0],
                    )
                )
            elif path == "/api/dataset-diagnostics/run":
                self._send_json(_run_dataset_diagnostics_payload(self.dataset))
            elif path == "/api/artifacts":
                self._send_json(_artifacts_payload(self.dataset))
            elif path.startswith("/api/artifacts/"):
                artifact_id = urllib.parse.unquote(path.removeprefix("/api/artifacts/"))
                self._send_json(_artifact_detail_payload(self.dataset, artifact_id))
            elif path.startswith("/api/episodes/"):
                trace_id = urllib.parse.unquote(path.removeprefix("/api/episodes/"))
                self._send_json(_episode_payload(self.dataset.bundle(trace_id)))
            elif path == "/api/frame":
                self._send_frame(query)
            elif path == "/api/episode-video":
                self._send_episode_video(query)
            elif path == "/api/policy-calls":
                self._send_json(_policy_calls_payload(self._bundle_from_query(query)))
            elif path == "/api/action-norm":
                self._send_json(_action_norm_payload(self._bundle_from_query(query)))
            elif path == "/api/generation-commitment":
                self._send_json(_generation_commitment_payload(self._bundle_from_query(query)))
            elif path == "/api/episode-metrics":
                self._send_json(_episode_metrics_payload(self._bundle_from_query(query)))
            elif path == "/api/episode-interactions":
                self._send_json(_episode_interactions_payload(self.dataset, query))
            elif path == "/api/activation-sites":
                self._send_json(_activation_sites_payload(self._bundle_from_query(query)))
            elif path == "/api/activation-slice":
                self._send_json(_activation_slice_payload(self._bundle_from_query(query), query))
            elif path == "/api/image-token-map":
                self._send_json(_image_token_map_payload(self._bundle_from_query(query), query))
            elif path == "/api/object-camera-overlay":
                self._send_json(
                    _object_camera_overlay_payload(self._bundle_from_query(query), query)
                )
            elif path == "/api/attention-map":
                self._send_json(_attention_map_payload(self._bundle_from_query(query), query))
            elif path == "/api/patch-features":
                self._send_json(_patch_features_payload(self._bundle_from_query(query), query))
            elif path == "/api/prompt-attention":
                self._send_json(_prompt_attention_payload(self._bundle_from_query(query), query))
            elif path == "/api/prompt-feature-map":
                self._send_json(_prompt_feature_map_payload(self._bundle_from_query(query), query))
            elif path == "/api/expert-token-activations":
                self._send_json(
                    _expert_token_model_sites_payload(self._bundle_from_query(query), query)
                )
            elif path == "/api/expert-token-details":
                self._send_json(
                    _expert_token_details_payload(self._bundle_from_query(query), query)
                )
            else:
                self.send_error(HTTPStatus.NOT_FOUND, f"Unknown route: {path}")
        except (BrokenPipeError, ConnectionResetError):  # browser navigated or scrubbed away
            return
        except Exception as exc:  # pragma: no cover - defensive server boundary
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, repr(exc))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        self._refresh_dataset_if_needed()
        try:
            if path == "/api/dataset-diagnostics/run":
                self._send_json(_run_dataset_diagnostics_payload(self.dataset))
            elif path == "/api/episode-annotations":
                self._send_json(_save_episode_annotation_payload(self.root, self._read_json_body()))
            elif path == "/api/selections/resolve":
                self._send_json(_resolve_selection_payload(self.dataset, self._read_json_body()))
            elif path == "/api/cohorts":
                self._send_json(_save_cohort_payload(self.dataset, self._read_json_body()))
            elif path == "/api/cohorts/from-selection":
                self._send_json(
                    _save_cohort_from_selection_payload(self.dataset, self._read_json_body())
                )
            elif path == "/api/cohorts/compare":
                self._send_json(_cohort_compare_payload(self.dataset, self._read_json_body()))
            elif path == "/api/analysis-runs":
                self._send_json(_save_analysis_run_payload(self.dataset, self._read_json_body()))
            elif path == "/api/intervention-runs":
                self._send_json(
                    _save_intervention_run_payload(self.dataset, self._read_json_body())
                )
            elif path == "/api/workspaces":
                self._send_json(_save_workspace_payload(self.dataset, self._read_json_body()))
            elif path == "/api/projection":
                self._send_json(_projection_payload(self.dataset, self._read_json_body()))
            elif path == "/api/graph":
                self._send_json(_graph_payload(self.dataset, self._read_json_body()))
            elif path == "/api/tables/query":
                self._send_json(_table_query_payload(self.dataset, self._read_json_body()))
            elif path.startswith("/api/lens-arrays/") and path.endswith("/slice"):
                array_id = urllib.parse.unquote(
                    path.removeprefix("/api/lens-arrays/").removesuffix("/slice")
                )
                self._send_json(
                    _lens_array_slice_payload(self.dataset, array_id, self._read_json_body())
                )
            elif path == "/api/artifacts/create/outcome-probe":
                self._send_json(_create_outcome_probe_payload(self.dataset))
            elif path == "/api/artifacts/create/target-object-probe":
                self._send_json(_create_target_object_probe_payload(self.dataset))
            elif path == "/api/artifacts/create/action-generation":
                self._send_json(_create_action_generation_payload(self.dataset))
            else:
                self.send_error(HTTPStatus.NOT_FOUND, f"Unknown route: {path}")
        except (BrokenPipeError, ConnectionResetError):  # browser navigated or scrubbed away
            return
        except Exception as exc:  # pragma: no cover - defensive server boundary
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, repr(exc))

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _bundle_from_query(self, query: dict[str, list[str]]) -> TraceBundle:
        return self.dataset.bundle(_query_one(query, "trace_id"))

    def _refresh_dataset_if_needed(self) -> None:
        cls = type(self)
        now = monotonic()
        with cls.dataset_lock:
            if now - cls.dataset_signature_checked_at < cls.dataset_signature_check_interval_s:
                return
            cls.dataset_signature_checked_at = now
        signature = _dataset_signature(self.root)
        with cls.dataset_lock:
            if cls.dataset_signature == signature:
                return
            cls.dataset = TraceDataset.open(self.root)
            cls.dataset_signature = signature
            cls.payload_cache.clear()
            cls.replay_renderers.clear()

    def _cached_payload(
        self,
        key: str,
        build: Callable[[TraceDataset], dict[str, Any]],
    ) -> dict[str, Any]:
        cls = type(self)
        signature = cls.dataset_signature or _dataset_signature(self.root)
        with cls.dataset_lock:
            cached = cls.payload_cache.get(key)
            if cached is not None and cached[0] == signature:
                return cached[1]
            payload = build(cls.dataset)
            cls.payload_cache[key] = (signature, payload)
            return payload

    def _send_json(self, value: Any) -> None:
        payload = json.dumps(value, allow_nan=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise TypeError("JSON request body must be an object")
        return value

    def _send_frame(self, query: dict[str, list[str]]) -> None:
        bundle = self._bundle_from_query(query)
        camera = _query_one(query, "camera")
        timestep = int(_query_one(query, "timestep"))
        source = query.get("source", ["auto"])[0]
        timestep = max(0, min(timestep, bundle.manifest.length - 1))
        if source == "trace":
            frame_path = _trace_frame_file_path(bundle, camera=camera, timestep=timestep)
            if frame_path is not None:
                payload = frame_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mimetypes.types_map.get(".jpg", "image/jpeg"))
                self.send_header("Cache-Control", "public, max-age=3600, immutable")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
        frame = self._frame(bundle, camera=camera, timestep=timestep, source=source)
        buffer = io.BytesIO()
        imageio.imwrite(buffer, np.asarray(frame), format="jpg", quality=90)
        payload = buffer.getvalue()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.types_map.get(".jpg", "image/jpeg"))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_episode_video(self, query: dict[str, list[str]]) -> None:
        bundle = self._bundle_from_query(query)
        camera = query.get("camera", ["all"])[0]
        fps = int(query.get("fps", ["10"])[0])
        max_width = int(query.get("max_width", ["320"])[0])
        video_path = _episode_video_path(
            bundle,
            camera=camera,
            fps=fps,
            max_width=max_width,
        )
        payload = video_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _frame(self, bundle: TraceBundle, *, camera: str, timestep: int, source: str) -> np.ndarray:
        if source in {"auto", "sparse"} and read_sparse_image is not None:
            try:
                sparse = read_sparse_image(bundle, camera=camera, timestep=timestep)
                if sparse is not None:
                    return sparse
            except Exception:
                if source == "sparse":
                    raise
        if source in {"auto", "replay"} and PI05LiberoReplayRenderer is not None:
            try:
                return self._replay_renderer(bundle).render(camera=camera, timestep=timestep)
            except Exception:
                if source == "replay":
                    raise
        frames = bundle.frames(camera, mmap=True)
        return np.asarray(frames[timestep])

    def _replay_renderer(self, bundle: TraceBundle) -> Any:
        with self.replay_lock:
            renderer = self.replay_renderers.get(bundle.manifest.trace_id)
            if renderer is None:
                if PI05LiberoReplayRenderer is None:
                    raise RuntimeError("PI0.5 LIBERO replay dependencies are not available")
                renderer = PI05LiberoReplayRenderer(bundle)
                self.replay_renderers[bundle.manifest.trace_id] = renderer
            return renderer


def _dataset_payload(dataset: TraceDataset, *, include_workbench: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "root": str(dataset.root),
        "episodes": [_manifest_payload(bundle) for bundle in dataset.bundles],
        "counterfactual_pairs": _counterfactual_pairs_payload(dataset),
    }
    if include_workbench:
        workbench = workbench_manifest(dataset)
        payload.update(
            {
                "activation_sites": int(len(dataset.model_site_index)),
                "artifacts": _artifact_summary(dataset),
                "workbench": workbench,
            }
        )
    else:
        payload.update(
            {
                "activation_sites": 0,
                "artifacts": {"total": 0, "counts": {}},
            }
        )
    return payload


def _counterfactual_pairs_response(dataset: TraceDataset) -> dict[str, Any]:
    pairs = _counterfactual_pairs_payload(dataset)
    return {"pairs": pairs, "count": len(pairs)}


def _counterfactual_pairs_payload(dataset: TraceDataset) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    group_metadata: dict[str, dict[str, Any]] = {}
    for bundle in dataset.bundles:
        metadata = dict(bundle.manifest.metadata or {})
        counterfactual = _counterfactual_metadata_from_manifest(metadata)
        group_id = str(counterfactual.get("group_id") or "").strip()
        if not group_id:
            continue
        member = {
            "trace_id": bundle.manifest.trace_id,
            "episode_id": bundle.manifest.episode_id,
            "role": str(counterfactual.get("role") or ""),
            "pair_index": _json_scalar(counterfactual.get("pair_index")),
            "paired_trace_id": str(counterfactual.get("paired_trace_id") or ""),
            "target_object_id": str(counterfactual.get("target_object_id") or ""),
            "counterfactual_target_object_id": str(
                counterfactual.get("counterfactual_target_object_id") or ""
            ),
            "outcome": bundle.manifest.outcome,
            "prompt": bundle.manifest.prompt,
        }
        grouped.setdefault(group_id, []).append(member)
        group_metadata.setdefault(
            group_id,
            {
                "group_id": group_id,
                "type": str(counterfactual.get("type") or ""),
                "changed_fields": _string_list(counterfactual.get("changed_fields")),
                "matched_fields": _string_list(counterfactual.get("matched_fields")),
            },
        )
    pairs: list[dict[str, Any]] = []
    for group_id, members in grouped.items():
        members.sort(key=_counterfactual_member_sort_key)
        pairs.append({**group_metadata[group_id], "members": members})
    pairs.sort(key=lambda pair: str(pair.get("group_id") or ""))
    return pairs


def _counterfactual_metadata_from_manifest(metadata: Mapping[str, Any]) -> dict[str, Any]:
    nested = metadata.get("counterfactual")
    counterfactual = dict(nested) if isinstance(nested, Mapping) else {}
    aliases = {
        "counterfactual_group_id": "group_id",
        "counterfactual_role": "role",
        "counterfactual_type": "type",
        "paired_trace_id": "paired_trace_id",
        "pair_index": "pair_index",
        "changed_fields": "changed_fields",
        "matched_fields": "matched_fields",
        "target_object_id": "target_object_id",
        "counterfactual_target_object_id": "counterfactual_target_object_id",
    }
    for source, target in aliases.items():
        if target not in counterfactual and source in metadata:
            counterfactual[target] = metadata[source]
    return counterfactual


def _counterfactual_member_sort_key(member: Mapping[str, Any]) -> tuple[int, int, str]:
    role_order = {"clean": 0, "control": 1, "corrupt": 2, "intervention": 3}
    role = str(member.get("role") or "")
    try:
        pair_index = int(member.get("pair_index"))
    except (TypeError, ValueError):
        pair_index = 10_000
    return (pair_index, role_order.get(role, 100), str(member.get("trace_id") or ""))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item)]
    return [item.strip() for item in text.split(",") if item.strip()]


def _workbench_payload(dataset: TraceDataset) -> dict[str, Any]:
    return workbench_manifest(dataset)


def _episode_annotations_payload(root: Path, *, trace_id: str | None = None) -> dict[str, Any]:
    annotations = _read_episode_annotations(root)
    if trace_id:
        return {
            "annotation": annotations.get(
                trace_id,
                _empty_episode_annotation(trace_id),
            )
        }
    return {"annotations": annotations}


def _save_episode_annotation_payload(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    trace_id = str(payload.get("trace_id") or "").strip()
    if not trace_id:
        raise ValueError("Episode annotation requires trace_id")
    annotations = _read_episode_annotations(root)
    current = annotations.get(trace_id, _empty_episode_annotation(trace_id))
    annotation = {
        **current,
        "trace_id": trace_id,
        "starred": bool(payload.get("starred", current.get("starred", False))),
        "notes": str(payload.get("notes", current.get("notes", ""))),
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    annotations[trace_id] = annotation
    path = _episode_annotations_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(annotations, indent=2, sort_keys=True), encoding="utf-8")
    return {"annotation": annotation}


def _read_episode_annotations(root: Path) -> dict[str, dict[str, Any]]:
    path = _episode_annotations_path(root)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    annotations: dict[str, dict[str, Any]] = {}
    for trace_id, value in payload.items():
        if isinstance(value, Mapping):
            annotations[str(trace_id)] = {
                **_empty_episode_annotation(str(trace_id)),
                **dict(value),
                "trace_id": str(trace_id),
            }
    return annotations


def _episode_annotations_path(root: Path) -> Path:
    return root / "annotations" / "episode_annotations.json"


def _empty_episode_annotation(trace_id: str) -> dict[str, Any]:
    return {"trace_id": trace_id, "starred": False, "notes": "", "updated_utc": None}


def _dataset_signature(root: Path) -> tuple[int, int]:
    """Cheap cache key for dataset-level metadata endpoints."""
    if (root / TraceBundle.MANIFEST).exists():
        paths = [root / TraceBundle.MANIFEST, root / TraceBundle.ARTIFACT_INDEX]
    else:
        paths = [
            candidate
            for bundle in root.rglob("*.vlatrace")
            if bundle.is_dir() and (bundle / TraceBundle.MANIFEST).exists()
            for candidate in (bundle / TraceBundle.MANIFEST, bundle / TraceBundle.ARTIFACT_INDEX)
        ]
        paths.append(root / TraceBundle.ARTIFACT_INDEX)
    existing = [path for path in paths if path.exists()]
    latest_mtime = max((path.stat().st_mtime_ns for path in existing), default=0)
    trace_count = sum(1 for path in existing if path.name == TraceBundle.MANIFEST)
    return trace_count, latest_mtime


def _lens_arrays_payload(dataset: TraceDataset) -> dict[str, Any]:
    arrays = [array.to_dict() for array in lens_array_catalog(dataset)]
    return {"lens_arrays": arrays, "total": len(arrays)}


def _lens_array_meta_payload(dataset: TraceDataset, array_id: str) -> dict[str, Any]:
    return lens_array_meta(dataset, array_id).to_dict()


def _lens_array_slice_payload(
    dataset: TraceDataset,
    array_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    max_values = int(payload.get("max_values", 4096))
    selection = payload.get("selection") or payload.get("axis_values") or {}
    if not isinstance(selection, dict):
        raise TypeError("LensArray slice selection must be an object")
    return slice_lens_array(dataset, array_id, selection=selection, max_values=max_values)


def _cohorts_payload(dataset: TraceDataset) -> dict[str, Any]:
    cohorts = [cohort.to_dict() for cohort in list_cohorts(dataset)]
    return {"cohorts": cohorts, "total": len(cohorts)}


def _save_cohort_payload(dataset: TraceDataset, payload: dict[str, Any]) -> dict[str, Any]:
    from vla_lens.workbench import CohortSpec

    cohort_payload = payload.get("cohort", payload)
    if not isinstance(cohort_payload, dict):
        raise TypeError("Cohort payload must be an object")
    cohort = save_cohort(dataset, CohortSpec.from_dict(cohort_payload))
    return {"cohort": cohort.to_dict(), **_cohorts_payload(dataset)}


def _save_cohort_from_selection_payload(
    dataset: TraceDataset,
    payload: dict[str, Any],
) -> dict[str, Any]:
    selection_payload = payload.get("selection", payload)
    if not isinstance(selection_payload, dict):
        raise TypeError("Selection payload must be an object")
    cohort = cohort_from_selection(
        dataset,
        SelectionState.from_dict(selection_payload),
        label=payload.get("label"),
        cohort_id=payload.get("cohort_id"),
    )
    saved = save_cohort(dataset, cohort)
    return {"cohort": saved.to_dict(), **_cohorts_payload(dataset)}


def _analysis_runs_payload(dataset: TraceDataset) -> dict[str, Any]:
    runs = [run.to_dict() for run in list_analysis_runs(dataset)]
    return {"analysis_runs": runs, "total": len(runs)}


def _intervention_runs_payload(dataset: TraceDataset) -> dict[str, Any]:
    runs = [run.to_dict() for run in list_intervention_runs(dataset)]
    return {"intervention_runs": runs, "total": len(runs)}


def _save_analysis_run_payload(dataset: TraceDataset, payload: dict[str, Any]) -> dict[str, Any]:
    run_payload = payload.get("analysis_run", payload.get("run", payload))
    if not isinstance(run_payload, dict):
        raise TypeError("Analysis run payload must be an object")
    run = save_analysis_run(dataset, AnalysisRunSpec.from_dict(run_payload))
    return {"analysis_run": run.to_dict(), **_analysis_runs_payload(dataset)}


def _save_intervention_run_payload(
    dataset: TraceDataset,
    payload: dict[str, Any],
) -> dict[str, Any]:
    run_payload = payload.get("intervention_run", payload.get("run", payload))
    if not isinstance(run_payload, dict):
        raise TypeError("Intervention run payload must be an object")
    run = save_intervention_run(dataset, InterventionRunSpec.from_dict(run_payload))
    return {"intervention_run": run.to_dict(), **_intervention_runs_payload(dataset)}


def _workspaces_payload(dataset: TraceDataset) -> dict[str, Any]:
    workspaces = [workspace.to_dict() for workspace in list_workspaces(dataset)]
    return {"workspaces": workspaces, "total": len(workspaces)}


def _cohort_compare_payload(dataset: TraceDataset, payload: dict[str, Any]) -> dict[str, Any]:
    left = str(payload.get("left") or payload.get("left_cohort_id"))
    right = str(payload.get("right") or payload.get("right_cohort_id"))
    return compare_cohorts(dataset, left, right)


def _unit_profile_payload(dataset: TraceDataset, query: dict[str, list[str]]) -> dict[str, Any]:
    unit = UnitRef(
        kind=query.get("kind", ["neuron"])[0],
        site_id=query.get("site_id", [None])[0],
        index=int(query.get("unit", query.get("index", ["0"]))[0]),
    )
    selection: dict[str, Any] = {"axis_values": {}}
    for axis in ["layer", "token_kind", "module", "tensor_type", "episode"]:
        if axis in query:
            selection["axis_values"][axis] = query[axis]
    return unit_profile(
        dataset,
        unit,
        selection=selection,
        top_k=int(query.get("top_k", ["12"])[0]),
    )


def _projection_payload(dataset: TraceDataset, payload: dict[str, Any]) -> dict[str, Any]:
    selection = payload.get("selection") or {"selection_id": "projection", "axis_values": {}}
    if not isinstance(selection, dict):
        raise TypeError("Projection selection must be an object")
    return projection_points(dataset, selection=selection, limit=int(payload.get("limit", 500)))


def _graph_payload(dataset: TraceDataset, payload: dict[str, Any]) -> dict[str, Any]:
    selection = payload.get("selection", payload)
    if not isinstance(selection, dict):
        raise TypeError("Graph selection must be an object")
    return graph_from_selection(dataset, selection)


def _table_query_payload(dataset: TraceDataset, payload: dict[str, Any]) -> dict[str, Any]:
    table = str(payload.get("table") or "episodes")
    filters = payload.get("filters") or {}
    columns = payload.get("columns") or None
    if not isinstance(filters, dict):
        raise TypeError("Table query filters must be an object")
    if columns is not None and not isinstance(columns, list):
        raise TypeError("Table query columns must be a list")
    return query_table(
        dataset,
        table=table,
        filters=filters,
        columns=columns,
        limit=int(payload.get("limit", 200)),
    )


def _save_workspace_payload(dataset: TraceDataset, payload: dict[str, Any]) -> dict[str, Any]:
    workspace_payload = payload.get("workspace", payload)
    if not isinstance(workspace_payload, dict):
        raise TypeError("Workspace payload must be an object")
    workspace = save_workspace(dataset, SavedWorkspace.from_dict(workspace_payload))
    return {"workspace": workspace.to_dict(), **_workspaces_payload(dataset)}


def _resolve_selection_payload(dataset: TraceDataset, payload: dict[str, Any]) -> dict[str, Any]:
    selection_payload = payload.get("selection", payload)
    request = payload.get("request", ())
    selection = SelectionState.from_dict(selection_payload)
    return resolve_selection(dataset, selection, request=request)


def _dataset_diagnostics_payload(dataset: TraceDataset) -> dict[str, Any]:
    status = diagnostics_status(dataset)
    return _diagnostics_payload(status)


def _run_dataset_diagnostics_payload(dataset: TraceDataset) -> dict[str, Any]:
    artifact = run_dataset_diagnostics(dataset)
    status = diagnostics_status(dataset)
    payload = _diagnostics_payload(status)
    payload["artifact"] = _jsonable(artifact.to_dict())
    return payload


def _create_outcome_probe_payload(dataset: TraceDataset) -> dict[str, Any]:
    saved = train_probe_artifact_from_spec(dataset, _default_outcome_probe_spec(dataset))
    return {
        "artifact": _jsonable(saved.artifact.to_dict()),
        "results": _jsonable(saved.results.to_dict("records")),
        "artifacts": _artifacts_payload(dataset),
    }


def _create_target_object_probe_payload(dataset: TraceDataset) -> dict[str, Any]:
    saved = save_target_object_encoding_artifact(dataset, name="Dashboard target-object encoding")
    return {
        "artifact": _jsonable(saved.artifact.to_dict()),
        "arrays": {
            "metric_cube": list(saved.metric_cube.shape),
            "baseline_cube": list(saved.baseline_cube.shape),
            "delta_cube": list(saved.delta_cube.shape),
        },
        "artifacts": _artifacts_payload(dataset),
    }


def _create_action_generation_payload(dataset: TraceDataset) -> dict[str, Any]:
    saved = save_action_generation_artifact(dataset, name="Dashboard action generation")
    return {
        "artifact": _jsonable(saved.artifact.to_dict()),
        "artifacts": _artifacts_payload(dataset),
    }


def _default_target_object_probe_spec(dataset: TraceDataset) -> dict[str, Any]:
    spec = _default_outcome_probe_spec(dataset)
    spec["name"] = "Dashboard target-object encoding probe"
    spec["target"] = {"kind": "target_object"}
    spec["split"] = {"kind": "random_episode"}
    spec["baseline"] = ["majority_class", "benchmark", "task"]
    spec["sweep"] = "layer"
    return spec


def _default_outcome_probe_spec(dataset: TraceDataset) -> dict[str, Any]:
    model_sites = dataset.model_site_index
    module = "pi05.expert.layers.*"
    tensor_type = "hidden_mean"
    token_kind = "action"
    if not model_sites.empty and "module" in model_sites:
        modules = model_sites["module"].astype(str)
        if modules.str.contains("pi05.expert", regex=False).any():
            module = "pi05.expert.layers.*"
            tensor_type = "hidden_mean"
            token_kind = "action"
        elif modules.str.contains("action_head", regex=False).any():
            module = "action_head.layers.*.resid"
            tensor_type = "resid"
            token_kind = "action"
        else:
            module = str(modules.iloc[0])
            if "tensor_type" in model_sites:
                tensor_type = str(model_sites["tensor_type"].astype(str).iloc[0])
            token_kind = None
    return {
        "name": "Dashboard outcome probe",
        "target": {"kind": "outcome"},
        "features": {
            "module": module,
            "tensor_type": tensor_type,
            "token_kind": token_kind,
            "layers": None,
            "timesteps": "all",
            "generation_step": None,
            "reduction": "mean",
        },
        "split": {"kind": "heldout_benchmark"},
        "baseline": ["majority_class", "benchmark", "target_object", "task"],
        "sweep": "layer",
    }


def _diagnostics_payload(status: dict[str, Any]) -> dict[str, Any]:
    latest = status.get("latest")
    return {
        "fingerprint": status.get("fingerprint"),
        "stale": bool(status.get("stale", True)),
        "latest": _jsonable(latest) if latest else None,
    }


def _episode_payload(bundle: TraceBundle) -> dict[str, Any]:
    return {
        **_manifest_payload(bundle),
        "cameras": bundle.cameras(),
        "artifacts": (
            bundle.artifact_index.to_dict("records") if not bundle.artifact_index.empty else []
        ),
        "arrays": bundle.array_index.to_dict("records") if not bundle.array_index.empty else [],
    }


def _artifacts_payload(dataset: TraceDataset) -> dict[str, Any]:
    artifacts = [_artifact_record_payload(record) for record in _artifact_records(dataset)]
    counts: dict[str, int] = {}
    for artifact in artifacts:
        key = str(artifact.get("artifact_type") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return {
        "artifacts": artifacts,
        "counts": counts,
        "total": len(artifacts),
    }


def _artifact_detail_payload(dataset: TraceDataset, artifact_id: str) -> dict[str, Any]:
    artifact = dataset.load_artifact(artifact_id)
    arrays: list[dict[str, Any]] = []
    for name, path in artifact.arrays.items():
        array = dataset.load_artifact_array(artifact, name, mmap=True)
        arrays.append(
            {
                "name": name,
                "path": path,
                "shape": [int(item) for item in array.shape],
                "dtype": str(array.dtype),
                "summary": _array_summary(array),
                "preview": _array_preview(array),
            }
        )
    return {
        "artifact": _jsonable(artifact.to_dict()),
        "arrays": arrays,
    }


def _artifact_summary(dataset: TraceDataset) -> dict[str, Any]:
    records = _artifact_records(dataset)
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get("artifact_type") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return {"total": len(records), "counts": counts}


def _artifact_records(dataset: TraceDataset) -> list[dict[str, Any]]:
    table = dataset.artifact_index
    if table.empty:
        return []
    return table.sort_values("created_utc", ascending=False, na_position="last").to_dict("records")


def _artifact_record_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = {str(key): _json_scalar(value) for key, value in record.items()}
    for key in ["selector", "method", "metrics", "arrays", "display", "tags", "source_trace_ids"]:
        payload[key] = _jsonable(_json_parse(payload.get(key)))
    return payload


def _json_parse(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array)
    if value.size == 0 or not np.issubdtype(value.dtype, np.number):
        return {}
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        return {}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _array_preview(array: np.ndarray) -> Any:
    value = np.asarray(array)
    if value.size > 5000:
        slices = tuple(slice(0, min(64, size)) for size in value.shape)
        value = value[slices]
    return _round(value)


def _manifest_payload(bundle: TraceBundle) -> dict[str, Any]:
    manifest = bundle.manifest.to_dict()
    return _jsonable(manifest)


def _action_norm_payload(bundle: TraceBundle) -> dict[str, Any]:
    actions = np.asarray(bundle.actions(mmap=True), dtype=np.float32)
    return {"values": _round(np.linalg.norm(actions, axis=-1))}


def _policy_calls_payload(bundle: TraceBundle) -> dict[str, Any]:
    calls = _policy_calls(bundle)
    return {
        "calls": calls,
        "count": len(calls),
        "env_length": int(bundle.manifest.length),
    }


def _policy_calls(bundle: TraceBundle) -> list[dict[str, Any]]:
    call_rows = bundle.policy_calls.copy()
    if call_rows.empty:
        return []
    calls: list[dict[str, Any]] = []
    call_rows = call_rows.sort_values("policy_call_index").reset_index(drop=True)
    for index, row in enumerate(call_rows.to_dict("records")):
        call_index = int(row.get("policy_call_index", index))
        timestep = int(row.get("observation_timestep", row.get("env_timestep_start", 0)))
        segment_start = int(row.get("env_timestep_start", timestep))
        segment_end = int(
            row.get(
                "env_timestep_end",
                call_rows.iloc[index + 1]["env_timestep_start"] - 1
                if index + 1 < len(call_rows)
                else bundle.manifest.length - 1,
            )
        )
        calls.append(
            {
                "index": index,
                "model_call_index": call_index,
                "env_timestep": timestep,
                "segment_start": segment_start,
                "segment_end": max(segment_start, segment_end),
                "segment_length": max(1, segment_end - segment_start + 1),
            }
        )
    return calls


def _generation_commitment_payload(bundle: TraceBundle) -> dict[str, Any]:
    generation = np.asarray(bundle.generation_actions(mmap=True), dtype=np.float32)
    final = generation[:, -1:, :, :]
    commitment = np.linalg.norm(generation - final, axis=(-1, -2))
    return {"values": _round(commitment)}


def _action_metric_metadata(bundle: TraceBundle) -> dict[int, dict[str, str]]:
    table = bundle.action_normalization
    if table.empty:
        return {}
    row = table.iloc[0]
    names = _json_parse(row.get("action_dim_names"))
    metadata = _json_parse(row.get("metadata")) or {}
    labels = metadata.get("action_labels") if isinstance(metadata, dict) else None
    units = metadata.get("action_units") if isinstance(metadata, dict) else None
    if not isinstance(names, list):
        names = metadata.get("action_names") if isinstance(metadata, dict) else None
    if not isinstance(names, list):
        return {}
    result: dict[int, dict[str, str]] = {}
    for index, name in enumerate(names):
        label = (
            labels[index]
            if isinstance(labels, list) and index < len(labels) and labels[index]
            else _label_from_metric_name(str(name))
        )
        unit = (
            units[index]
            if isinstance(units, list) and index < len(units) and units[index]
            else "normalized controller units"
        )
        result[index] = {"name": str(name), "label": str(label), "unit": str(unit)}
    return result


def _label_from_metric_name(name: str) -> str:
    overrides = {
        "eef": "EEF",
        "x": "x",
        "y": "y",
        "z": "z",
    }
    parts = name.replace("-", "_").split("_")
    return " ".join(overrides.get(part, part.capitalize()) for part in parts if part)


def _policy_call_x_values(bundle: TraceBundle, count: int) -> np.ndarray:
    calls = bundle.policy_calls
    if calls.empty or "observation_timestep" not in calls:
        return np.arange(count, dtype=np.float32)
    values = (
        calls.sort_values("policy_call_index")["observation_timestep"]
        .to_numpy(dtype=np.float32)
        .reshape(-1)
    )
    if values.size < count:
        return np.arange(count, dtype=np.float32)
    return values[:count]


def _domain_x_label(domain: str) -> str:
    if domain == "call":
        return "Policy call timestep"
    if domain == "generation":
        return "Generation step"
    return "Environment timestep"


def _episode_metrics_payload(bundle: TraceBundle) -> dict[str, Any]:
    """Return plot-ready episode metrics from logged state/action arrays."""
    metrics: list[dict[str, Any]] = []
    action_metadata = _action_metric_metadata(bundle)

    def add_metric(
        key: str,
        label: str,
        values: Any,
        *,
        domain: str = "time",
        kind: str = "line",
        description: str = "",
        x_values: Any | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        y_unit: str | None = None,
    ) -> None:
        array = np.asarray(values, dtype=np.float32)
        if array.ndim != 1 or array.size == 0:
            return
        if x_values is None:
            x_array = np.arange(array.size, dtype=np.float32)
        else:
            x_array = np.asarray(x_values, dtype=np.float32)
            if x_array.ndim != 1 or x_array.size != array.size:
                x_array = np.arange(array.size, dtype=np.float32)
        metrics.append(
            {
                "key": key,
                "label": label,
                "domain": domain,
                "kind": kind,
                "description": description,
                "values": _round(array),
                "x_values": _round(x_array),
                "x_label": x_label or _domain_x_label(domain),
                "y_label": y_label or label,
                "y_unit": y_unit,
            }
        )

    try:
        actions = np.asarray(bundle.actions(mmap=True), dtype=np.float32)
        add_metric(
            "action_norm",
            "Action norm",
            np.linalg.norm(actions, axis=-1),
            description="Executed action magnitude by environment timestep.",
            y_label="Action norm",
            y_unit="normalized controller units",
        )
        for dim in range(min(actions.shape[-1], 8)):
            dim_info = action_metadata.get(dim, {})
            dim_label = str(dim_info.get("label") or f"Action dim {dim}")
            dim_name = str(dim_info.get("name") or f"dim_{dim}")
            dim_unit = str(dim_info.get("unit") or "normalized controller units")
            add_metric(
                f"action_dim_{dim}",
                dim_label,
                actions[:, dim],
                description=(
                    f"Executed action dimension {dim} ({dim_name}) by environment timestep."
                ),
                y_label=dim_label,
                y_unit=dim_unit,
            )
    except KeyError:
        pass

    for name, label, description in [
        ("gripper_open_signal", "Gripper open", "Logged gripper open/close signal."),
        ("rewards", "Reward", "Environment reward by timestep."),
    ]:
        try:
            add_metric(name, label, bundle.array(name, mmap=True), description=description)
        except KeyError:
            pass

    try:
        eef = np.asarray(bundle.array("eef_pos", mmap=True), dtype=np.float32)
        add_metric("eef_x", "EEF x", eef[:, 0], description="End-effector x position.")
        add_metric("eef_y", "EEF y", eef[:, 1], description="End-effector y position.")
        add_metric("eef_z", "EEF z", eef[:, 2], description="End-effector z position.")
        if eef.shape[0] > 1:
            speed = np.concatenate([[0.0], np.linalg.norm(np.diff(eef, axis=0), axis=-1)])
            add_metric(
                "eef_speed",
                "EEF speed",
                speed,
                description="End-effector step-to-step movement.",
                y_label="EEF speed",
                y_unit="position units / timestep",
            )
    except KeyError:
        pass

    try:
        gripper = np.asarray(bundle.array("gripper_qpos", mmap=True), dtype=np.float32)
        add_metric(
            "gripper_qpos_mean",
            "Gripper qpos mean",
            gripper.mean(axis=-1),
            description="Mean gripper joint position.",
        )
    except KeyError:
        pass

    try:
        generation = np.asarray(bundle.generation_actions(mmap=True), dtype=np.float32)
        final = generation[:, -1:, :, :]
        commitment = np.linalg.norm(generation - final, axis=(-1, -2))
        if commitment.ndim == 2:
            call_x = _policy_call_x_values(bundle, commitment.shape[0])
            add_metric(
                "generation_start",
                "Generation start delta",
                commitment[:, 0],
                domain="call",
                description="First generation-step distance from final sampled action.",
                x_values=call_x,
                y_label="Generation delta",
                y_unit="action L2",
            )
            add_metric(
                "generation_end",
                "Generation end delta",
                commitment[:, -1],
                domain="call",
                description="Final generation-step distance from sampled action.",
                x_values=call_x,
                y_label="Generation delta",
                y_unit="action L2",
            )
            add_metric(
                "generation_delta",
                "Generation convergence",
                commitment[:, 0] - commitment[:, -1],
                domain="call",
                description="Start-to-end generation commitment change per timestep.",
                x_values=call_x,
                y_label="Generation convergence",
                y_unit="action L2",
            )
    except KeyError:
        pass

    return {
        "domains": [
            {"key": "time", "label": "Time"},
            {"key": "call", "label": "Policy call"},
            {"key": "generation", "label": "Generation step"},
        ],
        "metrics": metrics,
    }


def _episode_interactions_payload(
    dataset: TraceDataset,
    query: dict[str, list[str]],
) -> dict[str, Any]:
    trace_id = query.get("trace_id", [""])[0]
    if not trace_id:
        return {
            "available": False,
            "reason": "Missing trace_id.",
            "trace_id": "",
            "objects": [],
        }
    artifact = _latest_interaction_metrics_artifact(dataset)
    if artifact is None:
        return {
            "available": False,
            "reason": "No pi05_interaction_metrics artifact found.",
            "trace_id": trace_id,
            "objects": [],
        }

    episode_table = _interaction_metrics_table(dataset, artifact, "episode_labels")
    object_table = _interaction_metrics_table(dataset, artifact, "object_metrics")
    if episode_table.empty or "trace_id" not in episode_table:
        return {
            "available": False,
            "reason": "Interaction metrics artifact has no episode label table.",
            "trace_id": trace_id,
            "artifact_id": artifact.artifact_id,
            "objects": [],
        }
    episode_rows = episode_table[episode_table["trace_id"].astype(str) == trace_id]
    if episode_rows.empty:
        return {
            "available": False,
            "reason": "Trace is not present in the interaction metrics artifact.",
            "trace_id": trace_id,
            "artifact_id": artifact.artifact_id,
            "objects": [],
        }

    episode_row = episode_rows.iloc[0].to_dict()
    object_rows = (
        object_table[object_table["trace_id"].astype(str) == trace_id]
        if not object_table.empty and "trace_id" in object_table
        else pd.DataFrame()
    )
    objects = [_interaction_object_payload(row) for row in object_rows.to_dict("records")]
    objects = sorted(
        objects,
        key=lambda item: (
            not bool(item["is_target_object"]),
            not (bool(item["moved"]) or bool(item["lifted"]) or bool(item["contacted"])),
            str(item["object_name"]),
        ),
    )
    return {
        "available": True,
        "trace_id": trace_id,
        "artifact_id": artifact.artifact_id,
        "episode": _interaction_episode_payload(episode_row),
        "quality": _interaction_quality_payload(episode_row),
        "objects": objects,
    }


def _latest_interaction_metrics_artifact(dataset: TraceDataset) -> LensArtifact | None:
    table = dataset.artifact_index
    if table.empty or "artifact_type" not in table:
        return None
    matches = table[table["artifact_type"].astype(str) == "pi05_interaction_metrics"]
    if matches.empty:
        return None
    if "created_utc" in matches:
        matches = matches.sort_values("created_utc", ascending=False, na_position="last")
    try:
        return dataset.load_artifact(str(matches.iloc[0]["artifact_id"]))
    except (FileNotFoundError, KeyError, ValueError):
        return None


def _interaction_metrics_table(
    dataset: TraceDataset,
    artifact: LensArtifact,
    key: str,
) -> pd.DataFrame:
    outputs = artifact.method.get("outputs") if isinstance(artifact.method, Mapping) else None
    relative_path = outputs.get(key) if isinstance(outputs, Mapping) else None
    if not relative_path:
        return pd.DataFrame()
    path = dataset.root / str(relative_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _interaction_episode_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    target_objects = _json_parse(row.get("target_objects"))
    if not isinstance(target_objects, list):
        target_objects = []
    parsed_targets = [_optional_text(value) for value in target_objects]
    return {
        "primary_target_object": _optional_text(row.get("primary_target_object")),
        "target_objects": [value for value in parsed_targets if value],
        "target_parse_status": _optional_text(row.get("target_parse_status")),
        "first_moved_object": _optional_text(row.get("first_moved_object")),
        "first_moved_timestep": _optional_int(row.get("first_moved_timestep")),
        "first_moved_is_target": _optional_bool(row.get("first_moved_is_target")),
        "first_lifted_object": _optional_text(row.get("first_lifted_object")),
        "first_lifted_timestep": _optional_int(row.get("first_lifted_timestep")),
        "first_lifted_is_target": _optional_bool(row.get("first_lifted_is_target")),
        "first_contacted_object": _optional_text(row.get("first_contacted_object")),
        "first_contact_timestep": _optional_int(row.get("first_contact_timestep")),
        "scene_family": _optional_text(row.get("scene_family")),
        "task_verb": _optional_text(row.get("task_verb")),
    }


def _interaction_quality_payload(row: Mapping[str, Any]) -> dict[str, bool]:
    keys = [
        "target_parse_failed",
        "multi_target_task",
        "no_object_moved",
        "ambiguous_first_moved",
        "no_object_lifted",
        "ambiguous_first_lifted",
    ]
    return {key: _optional_bool(row.get(key)) for key in keys}


def _interaction_object_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "object_name": _optional_text(row.get("object_name")),
        "object_base_name": _optional_text(row.get("object_base_name")),
        "object_kind": _optional_text(row.get("object_kind")),
        "is_target_object": _optional_bool(row.get("is_target_object")),
        "moved": _optional_bool(row.get("moved")),
        "lifted": _optional_bool(row.get("lifted")),
        "contacted": _optional_bool(row.get("contacted")),
        "movement_onset_timestep": _optional_int(row.get("movement_onset_timestep")),
        "lift_onset_timestep": _optional_int(row.get("lift_onset_timestep")),
        "contact_onset_timestep": _optional_int(row.get("contact_onset_timestep")),
        "max_displacement": _optional_float(row.get("max_displacement")),
        "max_z_delta": _optional_float(row.get("max_z_delta")),
    }


def _optional_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_bool(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return bool(value)


def _activation_sites_payload(bundle: TraceBundle) -> dict[str, Any]:
    if bundle.model_sites.empty:
        return {"sites": [], "runtime_collections": [], "architecture": {}}
    rows = []
    for record in bundle.model_sites.to_dict("records"):
        axes = json.loads(str(record.get("axes") or "[]"))
        metadata = json.loads(str(record.get("metadata") or "{}"))
        rows.append(
            {
                "name": str(record["name"]),
                "site_id": str(record.get("site_id") or record["name"]),
                "module": str(record.get("module") or ""),
                "layer": _json_scalar(record.get("layer")),
                "tensor_type": str(record.get("tensor_type") or ""),
                "token_kind": _json_scalar(record.get("token_kind")),
                "family": _json_scalar(record.get("family")),
                "role": _json_scalar(record.get("role")),
                "segment": _json_scalar(record.get("segment")),
                "materialization": _json_scalar(record.get("materialization")),
                "exactness": _json_scalar(record.get("exactness")),
                "token_space_id": _json_scalar(record.get("token_space_id")),
                "query_token_space_id": _json_scalar(record.get("query_token_space_id")),
                "key_token_space_id": _json_scalar(record.get("key_token_space_id")),
                "parent_site_id": _json_scalar(record.get("parent_site_id")),
                "summary_type": _json_scalar(record.get("summary_type")),
                "capture_family": _json_scalar(record.get("capture_family")),
                "view_kind": _json_scalar(record.get("view_kind")),
                "capture_role": _json_scalar(record.get("capture_role")),
                "default_view": _json_scalar(record.get("default_view")),
                "derived_from": _json_list(record.get("derived_from")),
                "derivation": _json_scalar(record.get("derivation")),
                "axes": axes,
                "shape": json.loads(str(record.get("shape") or "[]")),
                "dtype": str(record.get("dtype") or ""),
                "metadata": metadata,
            }
        )
    return {
        "sites": rows,
        "runtime_collections": _activation_runtime_collections(rows),
        "architecture": _activation_architecture(rows),
    }


def _activation_runtime_collections(sites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kv_members = []
    for site in sites:
        name = str(site.get("name") or "")
        role = str(site.get("role") or "")
        if ".vlm.layers." not in name or ".kv_cache." not in name:
            continue
        component = "key" if role.endswith("key") or name.endswith(".key") else "value"
        kv_members.append(
            {
                "layer": site.get("layer"),
                "component": component,
                "site_name": name,
            }
        )
    if not kv_members:
        return []
    kv_members.sort(key=lambda item: (int(item["layer"] or 0), str(item["component"])))
    return [
        {
            "id": "pi05.vlm.past_key_values",
            "label": "Layer-wise prefix K/V",
            "kind": "runtime_collection",
            "materialized": False,
            "aggregation": "none",
            "members": kv_members,
        }
    ]


def _activation_architecture(sites: list[dict[str, Any]]) -> dict[str, Any]:
    if not any(str(site.get("name") or "").startswith("pi05.") for site in sites):
        return {}

    vlm_layers = sorted(_captured_layers(sites, stack="vlm"))
    expert_layers = sorted(_captured_layers(sites, stack="expert"))
    nodes = [
        {
            "id": "pi05.vlm.prefix",
            "label": "Inputs",
            "kind": "inputs",
            "stage": "prefix",
            "captured": any(
                str(site.get("name") or "").startswith("pi05.vlm.prefix") for site in sites
            ),
        }
    ]
    nodes.extend(
        {
            "id": f"pi05.vlm.layers.{layer}",
            "label": f"VLM L{layer}",
            "kind": "vlm_layer",
            "stage": "prefix",
            "layer": layer,
            "captured": True,
        }
        for layer in vlm_layers
    )
    nodes.append(
        {
            "id": "pi05.expert.by_step.input_embeddings",
            "label": "x_t",
            "kind": "denoise_state",
            "stage": "action_denoiser",
            "captured": any(
                str(site.get("name") or "") == "pi05.expert.by_step.input_embeddings"
                for site in sites
            ),
        }
    )
    nodes.extend(
        {
            "id": f"pi05.expert.layers.{layer}",
            "label": f"Expert L{layer}",
            "kind": "expert_layer",
            "stage": "action_denoiser",
            "layer": layer,
            "captured": True,
        }
        for layer in expert_layers
    )
    nodes.extend(
        [
            {
                "id": "pi05.action_head",
                "label": "Head",
                "kind": "action_head",
                "stage": "output",
                "captured": any(
                    str(site.get("name") or "").startswith("pi05.action_head")
                    and str(site.get("role") or "") != "action_head_output"
                    for site in sites
                ),
            },
            {
                "id": "pi05.action_output",
                "label": "Action",
                "kind": "action_output",
                "stage": "output",
                "captured": any(
                    str(site.get("role") or "") == "action_head_output" for site in sites
                ),
            },
        ]
    )
    edges = _activation_architecture_edges(sites)
    return {"nodes": nodes, "edges": edges} if nodes or edges else {}


def _activation_architecture_edges(sites: list[dict[str, Any]]) -> list[dict[str, Any]]:
    vlm_kv = _vlm_kv_sites_by_layer(sites)
    expert_layers = _captured_layers(sites, stack="expert")
    attention_by_layer = {
        int(site["layer"]): site
        for site in sites
        if _site_layer(site) is not None
        and ".expert.layers." in str(site.get("name") or "")
        and (
            str(site.get("role") or "") == "attention_probs"
            or str(site.get("tensor_type") or "") == "attention"
        )
    }
    edges = []
    for layer in sorted(set(vlm_kv) & expert_layers):
        source_sites = vlm_kv[layer]
        if not {"key", "value"}.issubset(source_sites):
            continue
        attention_site = attention_by_layer.get(layer)
        query_token_space = (
            attention_site.get("query_token_space_id") if attention_site else "pi05.action_suffix"
        )
        key_token_space = (
            attention_site.get("key_token_space_id") if attention_site else "pi05.expert_context"
        )
        edges.append(
            {
                "id": f"pi05.vlm.layers.{layer}.kv_to_expert.layers.{layer}",
                "kind": "per_layer_kv_conditioning",
                "source": f"pi05.vlm.layers.{layer}",
                "target": f"pi05.expert.layers.{layer}",
                "layer": layer,
                "source_sites": [source_sites["key"], source_sites["value"]],
                "target_site_family": (
                    attention_site.get("name")
                    if attention_site
                    else f"pi05.expert.layers.{layer}.by_step.attention"
                ),
                "source_token_space": "pi05.prefix",
                "query_token_space": query_token_space,
                "key_token_space": key_token_space,
                "runtime_collection": "pi05.vlm.past_key_values",
                "materialized": False,
            }
        )
    return edges


def _vlm_kv_sites_by_layer(sites: list[dict[str, Any]]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for site in sites:
        layer = _site_layer(site)
        name = str(site.get("name") or "")
        if layer is None or ".vlm.layers." not in name or ".kv_cache." not in name:
            continue
        role = str(site.get("role") or "")
        if role.endswith("key") or name.endswith(".key"):
            component = "key"
        elif role.endswith("value") or name.endswith(".value"):
            component = "value"
        else:
            continue
        out.setdefault(layer, {})[component] = name
    return out


def _captured_layers(sites: list[dict[str, Any]], *, stack: str) -> set[int]:
    marker = f".{stack}.layers."
    layers = set()
    for site in sites:
        name = str(site.get("name") or "")
        layer = _site_layer(site)
        if layer is not None and marker in name:
            layers.add(layer)
    return layers


def _site_layer(site: dict[str, Any]) -> int | None:
    layer = site.get("layer")
    try:
        if layer is None:
            return None
        if isinstance(layer, float) and not np.isfinite(layer):
            return None
        return int(layer)
    except (TypeError, ValueError, OverflowError):
        return None


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and not np.isfinite(value):
        return []
    if isinstance(value, np.generic):
        return _json_list(value.item())
    if isinstance(value, list):
        return value
    text = str(value or "[]")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _activation_slice_payload(bundle: TraceBundle, query: dict[str, list[str]]) -> dict[str, Any]:
    name = _query_one(query, "name")
    calls = _policy_calls(bundle)
    if not calls:
        return {"name": name, "values": [], "top_abs": [], "selected": None}
    call = calls[_query_call_index(query)]
    generation_step = query.get("generation_step", [""])[0]
    matches = bundle.model_sites.loc[bundle.model_sites["name"].astype(str) == name]
    if matches.empty:
        raise KeyError(name)
    record = matches.iloc[0]
    axes = json.loads(str(record.get("axes") or "[]"))
    array = np.asarray(bundle.model_site(name, mmap=True), dtype=np.float32)
    value, remaining_axes = _take_policy_call_value(array, axes, call)
    if "generation_step" in remaining_axes:
        step = int(generation_step) if generation_step not in {"", None} else 0
        value = _take_axis_value(value, remaining_axes, "generation_step", step)
        remaining_axes = [axis for axis in remaining_axes if axis != "generation_step"]
    value_array = np.asarray(value, dtype=np.float32)
    if "channel" not in remaining_axes:
        return {
            "name": name,
            "selected": call,
            "axes": remaining_axes,
            "shape": [int(item) for item in array.shape],
            "feature_count": 0,
            "feature": 0,
            "feature_value": None,
            "top_abs": [],
            "reason": "Selected site has no channel feature axis.",
        }
    channel_axis = remaining_axes.index("channel")
    channel_count = int(value_array.shape[channel_axis])
    channel_matrix = np.moveaxis(value_array, channel_axis, -1).reshape(-1, channel_count)
    vector = np.nanmean(channel_matrix, axis=0)
    remaining_axes = [axis for axis in remaining_axes if axis != "channel"]
    clip_percent = _query_float(query, "clip_percent", 0.0)
    clip_percent = min(20.0, max(0.0, clip_percent))
    try:
        top_k = int(query.get("top_k", ["12"])[0])
    except (TypeError, ValueError):
        top_k = 12
    top_k = max(1, min(top_k, 256))
    order, clip_info = _rank_feature_vector(vector, clip_percent=clip_percent, limit=top_k)
    feature = int(query.get("feature", ["0"])[0])
    feature = max(0, min(feature, max(0, vector.shape[0] - 1)))
    return {
        "name": name,
        "selected": call,
        "axes": remaining_axes,
        "shape": [int(item) for item in array.shape],
        "feature_count": int(vector.shape[0]),
        "feature": feature,
        "feature_value": _json_scalar(float(vector[feature])) if vector.size else None,
        "clip_percent": clip_percent,
        "clip": clip_info,
        "top_abs": [
            {"index": int(index), "value": _json_scalar(float(vector[index]))} for index in order
        ],
    }


def _rank_feature_vector(
    vector: np.ndarray,
    *,
    clip_percent: float = 0.0,
    limit: int = 24,
) -> tuple[np.ndarray, dict[str, Any]]:
    finite_mask = np.isfinite(vector)
    finite_values = vector[finite_mask]
    if finite_values.size == 0:
        return np.asarray([], dtype=np.int64), {
            "enabled": clip_percent > 0,
            "kept": 0,
            "total": int(vector.size),
        }

    lower: float | None = None
    upper: float | None = None
    keep_mask = finite_mask.copy()
    if clip_percent > 0:
        lower = float(np.percentile(finite_values, clip_percent))
        upper = float(np.percentile(finite_values, 100.0 - clip_percent))
        keep_mask &= vector >= lower
        keep_mask &= vector <= upper

    candidates = np.flatnonzero(keep_mask)
    if candidates.size == 0:
        return candidates, {
            "enabled": clip_percent > 0,
            "lower": _json_scalar(lower),
            "upper": _json_scalar(upper),
            "kept": 0,
            "total": int(vector.size),
        }
    candidate_values = np.nan_to_num(vector[candidates], nan=0.0, posinf=0.0, neginf=0.0)
    ranked = candidates[np.argsort(np.abs(candidate_values))[::-1][:limit]]
    return ranked, {
        "enabled": clip_percent > 0,
        "lower": _json_scalar(lower),
        "upper": _json_scalar(upper),
        "kept": int(candidates.size),
        "total": int(vector.size),
    }


def _image_token_map_payload(bundle: TraceBundle, query: dict[str, list[str]]) -> dict[str, Any]:
    name = _query_one(query, "name")
    calls = _policy_calls(bundle)
    if not calls:
        return {"available": False, "reason": "No policy calls are available."}
    call = calls[_query_call_index(query)]
    feature = int(query.get("feature", ["0"])[0])
    matches = bundle.model_sites.loc[bundle.model_sites["name"].astype(str) == name]
    if matches.empty:
        raise KeyError(name)
    record = matches.iloc[0]
    axes = json.loads(str(record.get("axes") or "[]"))
    if "token" not in axes:
        return _not_captured_in_profile(
            f"Image heatmaps require model_sites with a token axis. {name!r} has axes={axes!r}.",
            name=name,
            token_kind=_json_scalar(record.get("token_kind")),
            axes=axes,
        )
    token_matrix = _activation_token_matrix(bundle, name, call, query)
    feature = max(0, min(feature, token_matrix.shape[1] - 1))
    image_rows = _image_token_rows_for_site(bundle, record, call, token_matrix.shape[0])
    if not image_rows.empty:
        maps, layout = _camera_patch_maps_from_token_rows(
            bundle,
            image_rows,
            token_matrix,
            feature,
        )
        prefix_rows = _token_rows_for_space(bundle, call, str(record.get("token_space_id") or ""))
        text_tokens = (
            int((prefix_rows.get("token_kind", "").astype(str) == "language").sum())
            if not prefix_rows.empty
            else 0
        )
        return {
            "available": True,
            "name": name,
            "feature": feature,
            "feature_count": int(token_matrix.shape[1]),
            "call": call,
            "source": "vlatrace",
            "grid_size": layout["grid_size"],
            "grid_height": layout["grid_height"],
            "grid_width": layout["grid_width"],
            "patches_per_image": layout["patches_per_image"],
            "image_tokens": int(len(image_rows)),
            "text_tokens": text_tokens,
            "image_slots": len(maps),
            "maps": maps,
            "note": (
                "Mapped image-token rows from token layout. This site is a mixed prefix sequence, "
                "so language tokens are excluded from the camera heatmap."
            ),
        }
    if str(record.get("token_kind") or "") != "image":
        return _not_captured_in_profile(
            "Image heatmaps require either an image-token site or token layout rows that mark "
            f"the image-token subset. {name!r} is token_kind={record.get('token_kind')!r}.",
            name=name,
            token_kind=_json_scalar(record.get("token_kind")),
            token_space_id=_json_scalar(record.get("token_space_id")),
        )
    text_tokens = 0
    layout = _camera_patch_layout_from_record(
        bundle,
        record,
        token_matrix.shape[0],
        text_tokens=text_tokens,
    )
    image_tokens = int(layout["image_tokens"])
    patches_per_image = int(layout["patches_per_image"])
    grid_height = int(layout["grid_height"])
    grid_width = int(layout["grid_width"])
    maps: dict[str, Any] = {}
    cameras = bundle.cameras()
    image_slots = image_tokens // patches_per_image if patches_per_image else 0
    for camera_index, camera in enumerate(cameras):
        if camera_index >= image_slots:
            continue
        start = camera_index * patches_per_image
        end = start + patches_per_image
        values = token_matrix[start:end, feature].reshape(grid_height, grid_width)
        maps[camera] = {
            "values": _round(values),
            "token_start": start,
            "token_end": end - 1,
            "active_tokens": None,
            "min": _json_scalar(float(np.nanmin(values))),
            "max": _json_scalar(float(np.nanmax(values))),
        }
    return {
        "available": True,
        "name": name,
        "feature": feature,
        "feature_count": int(token_matrix.shape[1]),
        "call": call,
        "source": "vlatrace",
        "grid_size": grid_height if grid_height == grid_width else None,
        "grid_height": grid_height,
        "grid_width": grid_width,
        "patches_per_image": patches_per_image,
        "image_tokens": image_tokens,
        "text_tokens": text_tokens,
        "image_slots": image_slots,
        "maps": maps,
        "note": (
            f"Inferred PI0.5/PaliGemma prefix layout: {image_slots} image slots x "
            f"{grid_height}x{grid_width} patches, followed by {text_tokens} text token slots."
        ),
    }


def _prompt_feature_map_payload(bundle: TraceBundle, query: dict[str, list[str]]) -> dict[str, Any]:
    name = _query_one(query, "name")
    calls = _policy_calls(bundle)
    if not calls:
        return {"available": False, "reason": "No policy calls are available."}
    call = calls[_query_call_index(query)]
    feature = int(query.get("feature", ["0"])[0])
    matches = bundle.model_sites.loc[bundle.model_sites["name"].astype(str) == name]
    if matches.empty:
        raise KeyError(name)
    record = matches.iloc[0]
    axes = json.loads(str(record.get("axes") or "[]"))
    if "token" not in axes or "channel" not in axes:
        return _not_captured_in_profile(
            "Prompt feature maps require a model site with token and channel axes.",
            name=name,
            axes=axes,
        )
    if str(record.get("role") or "") == "image_prefix_hidden_tokens":
        return _not_captured_in_profile(
            "This site stores image-prefix tokens only, "
            "so it has no aligned prompt-token features.",
            name=name,
            role=_json_scalar(record.get("role")),
        )
    token_space_id = str(record.get("token_space_id") or "")
    prefix_rows = _token_rows_for_space(bundle, call, token_space_id)
    if prefix_rows.empty:
        return _not_captured_in_profile(
            "Prompt feature maps require token layout rows for the selected token space.",
            name=name,
            token_space_id=token_space_id,
        )
    token_matrix = _activation_token_matrix(bundle, name, call, query)
    feature = max(0, min(feature, token_matrix.shape[1] - 1))
    text_rows = prefix_rows.loc[prefix_rows.get("token_kind", "").astype(str) == "language"].copy()
    if text_rows.empty:
        return _not_captured_in_profile(
            "The selected token space has no language-token rows.",
            name=name,
            token_space_id=token_space_id,
        )
    values = np.full((_token_count(prefix_rows),), np.nan, dtype=np.float32)
    limit = min(values.shape[0], token_matrix.shape[0])
    values[:limit] = token_matrix[:limit, feature]
    _top, prompt_mass, prompt, prompt_tokens = _prompt_attention_from_prefix_rows(
        prefix_rows, values
    )
    active_text_tokens = len(prompt_tokens)
    if "attention_mask" in text_rows:
        active_text_tokens = int(text_rows.get("attention_mask", []).astype(bool).sum())
    return {
        "available": True,
        "kind": "feature",
        "name": name,
        "call": call,
        "feature": feature,
        "feature_count": int(token_matrix.shape[1]),
        "prompt": bundle.manifest.prompt or prompt,
        "active_text_tokens": active_text_tokens,
        "allocated_text_slots": int(len(text_rows)),
        "expert_coarse": {"prompt": _json_scalar(float(prompt_mass))},
        "top_text_tokens": sorted(
            prompt_tokens,
            key=lambda item: abs(float(item.get("attention") or 0.0)),
            reverse=True,
        )[:24],
        "prompt_tokens": prompt_tokens,
        "top_image_patches": [],
        "note": (
            "Prompt tokens are colored by the selected hidden feature value, not attention mass."
        ),
    }


def _object_camera_overlay_payload(
    bundle: TraceBundle,
    query: dict[str, list[str]],
) -> dict[str, Any]:
    camera = _query_one(query, "camera")
    timestep = int(query.get("timestep", ["0"])[0] or 0)
    timestep = max(0, min(timestep, max(0, bundle.manifest.length - 1)))
    include_sites = str(query.get("include_sites", ["false"])[0]).lower() in {
        "1",
        "true",
        "yes",
    }
    try:
        object_pos = bundle.array("scene_object_pos", mmap=True)
    except KeyError:
        return {
            "available": False,
            "reason": "object_positions_unavailable",
            "detail": "scene_object_pos is not captured in this trace.",
            "camera": camera,
            "timestep": timestep,
            "objects": [],
        }

    object_quat = None
    try:
        object_quat = bundle.array("scene_object_quat", mmap=True)
    except KeyError:
        pass
    object_geom_center = _optional_array(bundle, "scene_object_geom_center")
    object_bbox = _optional_array(bundle, "scene_object_bbox_world")
    camera_object_bbox = _optional_array(bundle, "camera_object_bbox")
    camera_object_visible = _optional_array(bundle, "camera_object_visible")

    height, width = _camera_frame_size(bundle, camera)
    calibration = _camera_projection_calibration(bundle, camera, timestep)
    object_rows = _scene_object_rows(bundle, include_sites=include_sites)
    if not object_rows:
        return {
            "available": False,
            "reason": "object_metadata_unavailable",
            "detail": "scene_state has no object rows to map onto camera frames.",
            "camera": camera,
            "timestep": timestep,
            "objects": [],
        }
    if calibration is None:
        return {
            "available": False,
            "reason": "camera_calibration_unavailable",
            "detail": "camera_intrinsics/camera_extrinsics are required for projection.",
            "camera": camera,
            "timestep": timestep,
            "objects": [_jsonable(row) for row in object_rows],
        }

    intrinsic, extrinsic, calibration_camera = calibration
    camera_to_pixel = _camera_to_pixel_transform(intrinsic, extrinsic)
    objects: list[dict[str, Any]] = []
    for row in object_rows:
        object_index = row.get("object_index")
        if object_index is None:
            continue
        index = int(object_index)
        pos = _object_position_at(object_pos, index, timestep)
        if pos is None:
            continue
        geom_center = (
            _object_position_at(object_geom_center, index, timestep)
            if object_geom_center is not None
            else None
        )
        bbox = _object_bbox_at(object_bbox, index, timestep) if object_bbox is not None else None
        projection_kind = "object_pose_center"
        projection = _project_world_point(pos, camera_to_pixel, width=width, height=height)
        bbox_projection = None
        if geom_center is not None:
            projection = _project_world_point(
                geom_center,
                camera_to_pixel,
                width=width,
                height=height,
            )
            projection_kind = "object_geometry_center"
        if bbox is not None:
            bbox_projection = _project_world_bbox(
                bbox,
                camera_to_pixel,
                width=width,
                height=height,
            )
            if bbox_projection is not None:
                projection = {
                    **projection,
                    "pixel_x": bbox_projection["center_pixel_x"],
                    "pixel_y": bbox_projection["center_pixel_y"],
                    "x": bbox_projection["center_x"],
                    "y": bbox_projection["center_y"],
                    "in_frame": bbox_projection["in_frame"],
                }
                projection_kind = "object_geometry_bbox"
        camera_bbox_projection = _camera_object_bbox_projection(
            bundle,
            camera_object_bbox,
            camera_object_visible,
            camera=camera,
            object_name=str(row.get("object_name") or ""),
            object_index=index,
            timestep=timestep,
            width=width,
            height=height,
        )
        if camera_bbox_projection is not None:
            bbox_projection = camera_bbox_projection
            projection = {
                **projection,
                "pixel_x": bbox_projection["center_pixel_x"],
                "pixel_y": bbox_projection["center_pixel_y"],
                "x": bbox_projection["center_x"],
                "y": bbox_projection["center_y"],
                "in_frame": bbox_projection["in_frame"],
            }
            projection_kind = "camera_segmentation_bbox"
        quat = _object_quat_at(object_quat, index, timestep) if object_quat is not None else None
        objects.append(
            {
                **row,
                "position_world": _round(pos),
                "geometry_center_world": _round(geom_center) if geom_center is not None else None,
                "quaternion_xyzw": _round(quat) if quat is not None else None,
                "bbox": bbox_projection,
                "pixel_x": projection.get("pixel_x"),
                "pixel_y": projection.get("pixel_y"),
                "x": projection.get("x"),
                "y": projection.get("y"),
                "depth": projection.get("depth"),
                "in_frame": projection.get("in_frame"),
                "approximate": True,
                "projection_kind": projection_kind,
            }
        )

    visible = [item for item in objects if item.get("in_frame")]
    return {
        "available": bool(objects),
        "camera": camera,
        "calibration_camera": calibration_camera,
        "timestep": timestep,
        "width": width,
        "height": height,
        "include_sites": include_sites,
        "approximate": True,
        "projection_kind": (
            "camera_segmentation_bbox"
            if camera_object_bbox is not None
            else "object_geometry_bbox"
            if object_bbox is not None
            else "object_pose_center"
        ),
        "visible_count": len(visible),
        "objects": _jsonable(objects),
        "note": (
            "Object labels use captured world-frame object geometry bounds when available, "
            "falling back to object pose centers."
        ),
    }


def _scene_object_rows(bundle: TraceBundle, *, include_sites: bool) -> list[dict[str, Any]]:
    table = bundle.scene_state
    if table.empty:
        return []
    if "context_kind" in table:
        table = table.loc[table["context_kind"].astype(str) == "object"]
    if "object_index" not in table or "object_name" not in table:
        return []
    rows: list[dict[str, Any]] = []
    for raw in table.sort_values("object_index").to_dict("records"):
        object_index = _optional_int(raw.get("object_index"))
        object_name = raw.get("object_name")
        if object_index is None or object_name is None or str(object_name) == "nan":
            continue
        object_kind = str(raw.get("object_kind") or "object")
        if object_kind == "site" and not include_sites:
            continue
        rows.append(
            {
                "object_index": object_index,
                "object_name": str(object_name),
                "object_kind": object_kind,
                "body_id": _optional_int(raw.get("body_id")),
                "body_name": str(raw.get("body_name") or ""),
                "site_name": str(raw.get("site_name") or ""),
                "source": str(raw.get("source") or ""),
            }
        )
    return rows


def _optional_array(bundle: TraceBundle, name: str) -> np.ndarray | None:
    try:
        return bundle.array(name, mmap=True)
    except KeyError:
        return None


def _camera_frame_size(bundle: TraceBundle, camera: str) -> tuple[int, int]:
    try:
        frames = bundle.frames(camera, mmap=True)
        if frames.ndim >= 3:
            return int(frames.shape[1]), int(frames.shape[2])
    except KeyError:
        pass
    resolution = _camera_resolution_from_context(bundle, camera)
    if resolution is not None:
        return resolution
    return 1, 1


def _camera_resolution_from_context(bundle: TraceBundle, camera: str) -> tuple[int, int] | None:
    try:
        resolution = bundle.array("camera_resolution", mmap=True)
    except KeyError:
        resolution = None
    camera_index = _camera_index_for_array(bundle, "camera_resolution", camera)
    if resolution is not None and camera_index is not None and resolution.ndim == 2:
        pair = np.asarray(resolution[camera_index]).astype(int)
        if pair.size >= 2 and pair[0] > 0 and pair[1] > 0:
            return int(pair[0]), int(pair[1])
    table = bundle.camera_state
    if table.empty:
        return None
    names = _camera_aliases(camera)
    for row in table.to_dict("records"):
        row_name = str(row.get("camera_name") or row.get("name") or row.get("camera_id") or "")
        if row_name in names:
            height = _optional_int(row.get("height"))
            width = _optional_int(row.get("width"))
            if height and width:
                return height, width
    return None


def _camera_projection_calibration(
    bundle: TraceBundle,
    camera: str,
    timestep: int,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    try:
        intrinsics = bundle.array("camera_intrinsics", mmap=True)
        extrinsics = bundle.array("camera_extrinsics", mmap=True)
    except KeyError:
        return None
    camera_index = _camera_index_for_array(bundle, "camera_intrinsics", camera)
    if camera_index is None:
        camera_index = _camera_index_for_array(bundle, "camera_extrinsics", camera)
    if camera_index is None:
        return None
    intrinsic = np.asarray(intrinsics[camera_index], dtype=np.float32)
    extrinsic = _camera_extrinsic_at(extrinsics, camera_index, timestep)
    if intrinsic.shape != (3, 3) or extrinsic is None or extrinsic.shape != (4, 4):
        return None
    camera_names = _camera_names_for_array(bundle, "camera_intrinsics")
    calibration_camera = camera_names[camera_index] if camera_index < len(camera_names) else camera
    return intrinsic, extrinsic, calibration_camera


def _camera_extrinsic_at(
    extrinsics: np.ndarray,
    camera_index: int,
    timestep: int,
) -> np.ndarray | None:
    value = np.asarray(extrinsics)
    if value.ndim == 3:
        extrinsic = np.asarray(value[camera_index], dtype=np.float32)
        return extrinsic if np.all(np.isfinite(extrinsic)) else None
    if value.ndim == 4:
        step = max(0, min(timestep, value.shape[0] - 1))
        extrinsic = np.asarray(value[step, camera_index], dtype=np.float32)
        return extrinsic if np.all(np.isfinite(extrinsic)) else None
    return None


def _camera_index_for_array(bundle: TraceBundle, array_name: str, camera: str) -> int | None:
    names = _camera_names_for_array(bundle, array_name)
    aliases = _camera_aliases(camera)
    for index, name in enumerate(names):
        if name in aliases:
            return index
    return None


def _camera_names_for_array(bundle: TraceBundle, array_name: str) -> list[str]:
    return _metadata_list_for_array(bundle, array_name, "camera_names")


def _object_names_for_array(bundle: TraceBundle, array_name: str) -> list[str]:
    return _metadata_list_for_array(bundle, array_name, "object_names")


def _metadata_list_for_array(bundle: TraceBundle, array_name: str, key: str) -> list[str]:
    if bundle.array_index.empty or "name" not in bundle.array_index:
        return []
    rows = bundle.array_index.loc[bundle.array_index["name"].astype(str) == array_name]
    if rows.empty:
        return []
    metadata = _json_parse(rows.iloc[0].get("metadata"))
    if isinstance(metadata, dict):
        names = metadata.get(key)
        if isinstance(names, list):
            return [str(name) for name in names]
    return []


def _camera_aliases(camera: str) -> set[str]:
    aliases = {camera, camera.removesuffix("_image")}
    if camera == "main":
        aliases.update({"agentview", "agentview_image", "image"})
    if camera == "wrist":
        aliases.update({"robot0_eye_in_hand", "robot0_eye_in_hand_image", "image2"})
    if camera == "agentview":
        aliases.update({"main", "agentview_image", "image"})
    if camera == "robot0_eye_in_hand":
        aliases.update({"wrist", "robot0_eye_in_hand_image", "image2"})
    return aliases


def _camera_to_pixel_transform(intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    expanded = np.eye(4, dtype=np.float32)
    expanded[:3, :3] = np.asarray(intrinsic, dtype=np.float32)
    return expanded @ np.linalg.inv(np.asarray(extrinsic, dtype=np.float32))


def _object_position_at(
    object_pos: np.ndarray, object_index: int, timestep: int
) -> np.ndarray | None:
    value = np.asarray(object_pos)
    try:
        if value.ndim == 3:
            step = max(0, min(timestep, value.shape[0] - 1))
            pos = np.asarray(value[step, object_index], dtype=np.float32)
        elif value.ndim == 2:
            pos = np.asarray(value[object_index], dtype=np.float32)
        else:
            return None
    except IndexError:
        return None
    if pos.shape[-1] < 3 or not np.all(np.isfinite(pos[:3])):
        return None
    return pos[:3]


def _object_quat_at(
    object_quat: np.ndarray | None,
    object_index: int,
    timestep: int,
) -> np.ndarray | None:
    if object_quat is None:
        return None
    value = np.asarray(object_quat)
    try:
        if value.ndim == 3:
            step = max(0, min(timestep, value.shape[0] - 1))
            quat = np.asarray(value[step, object_index], dtype=np.float32)
        elif value.ndim == 2:
            quat = np.asarray(value[object_index], dtype=np.float32)
        else:
            return None
    except IndexError:
        return None
    if quat.shape[-1] < 4 or not np.all(np.isfinite(quat[:4])):
        return None
    return quat[:4]


def _object_bbox_at(
    object_bbox: np.ndarray,
    object_index: int,
    timestep: int,
) -> np.ndarray | None:
    value = np.asarray(object_bbox)
    try:
        if value.ndim == 4:
            step = max(0, min(timestep, value.shape[0] - 1))
            bbox = np.asarray(value[step, object_index], dtype=np.float32)
        elif value.ndim == 3:
            bbox = np.asarray(value[object_index], dtype=np.float32)
        else:
            return None
    except IndexError:
        return None
    if bbox.shape != (2, 3) or not np.all(np.isfinite(bbox)):
        return None
    return bbox


def _project_world_point(
    point: np.ndarray,
    camera_to_pixel: np.ndarray,
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    homogeneous = np.asarray([point[0], point[1], point[2], 1.0], dtype=np.float32)
    projected = camera_to_pixel @ homogeneous
    depth = float(projected[2])
    if not np.isfinite(depth) or abs(depth) <= 1e-8:
        return {"in_frame": False, "depth": _json_scalar(depth)}
    pixel_x = float(projected[0] / depth)
    pixel_y = float(projected[1] / depth)
    x = pixel_x / max(1, width)
    y = pixel_y / max(1, height)
    in_frame = bool(depth > 0 and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
    return {
        "pixel_x": _json_scalar(pixel_x),
        "pixel_y": _json_scalar(pixel_y),
        "x": _json_scalar(x),
        "y": _json_scalar(y),
        "depth": _json_scalar(depth),
        "in_frame": in_frame,
    }


def _project_world_bbox(
    bbox: np.ndarray,
    camera_to_pixel: np.ndarray,
    *,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    bounds = np.asarray(bbox, dtype=np.float32)
    mins = np.minimum(bounds[0], bounds[1])
    maxs = np.maximum(bounds[0], bounds[1])
    corners = np.asarray(
        [
            [x, y, z]
            for x in (mins[0], maxs[0])
            for y in (mins[1], maxs[1])
            for z in (mins[2], maxs[2])
        ],
        dtype=np.float32,
    )
    projections = [
        _project_world_point(corner, camera_to_pixel, width=width, height=height)
        for corner in corners
    ]
    visible_points = [
        item
        for item in projections
        if item.get("pixel_x") is not None
        and item.get("pixel_y") is not None
        and item.get("depth") is not None
        and float(item["depth"]) > 0.0
    ]
    if not visible_points:
        return None
    xs = np.asarray([float(item["x"]) for item in visible_points], dtype=np.float32)
    ys = np.asarray([float(item["y"]) for item in visible_points], dtype=np.float32)
    x0_raw = float(np.nanmin(xs))
    x1_raw = float(np.nanmax(xs))
    y0_raw = float(np.nanmin(ys))
    y1_raw = float(np.nanmax(ys))
    in_frame = bool(x1_raw >= 0.0 and x0_raw <= 1.0 and y1_raw >= 0.0 and y0_raw <= 1.0)
    x0 = min(1.0, max(0.0, x0_raw))
    x1 = min(1.0, max(0.0, x1_raw))
    y0 = min(1.0, max(0.0, y0_raw))
    y1 = min(1.0, max(0.0, y1_raw))
    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5
    return {
        "x0": _json_scalar(x0),
        "y0": _json_scalar(y0),
        "x1": _json_scalar(x1),
        "y1": _json_scalar(y1),
        "raw_x0": _json_scalar(x0_raw),
        "raw_y0": _json_scalar(y0_raw),
        "raw_x1": _json_scalar(x1_raw),
        "raw_y1": _json_scalar(y1_raw),
        "center_x": _json_scalar(center_x),
        "center_y": _json_scalar(center_y),
        "center_pixel_x": _json_scalar(center_x * max(1, width)),
        "center_pixel_y": _json_scalar(center_y * max(1, height)),
        "in_frame": in_frame,
    }


def _camera_object_bbox_projection(
    bundle: TraceBundle,
    camera_object_bbox: np.ndarray | None,
    camera_object_visible: np.ndarray | None,
    *,
    camera: str,
    object_name: str,
    object_index: int,
    timestep: int,
    width: int,
    height: int,
) -> dict[str, Any] | None:
    if camera_object_bbox is None:
        return None
    camera_index = _camera_index_for_array(bundle, "camera_object_bbox", camera)
    if camera_index is None:
        return None
    object_names = _object_names_for_array(bundle, "camera_object_bbox")
    if object_name in object_names:
        array_object_index = object_names.index(object_name)
    else:
        array_object_index = object_index
    value = np.asarray(camera_object_bbox)
    try:
        if value.ndim == 4:
            step = max(0, min(timestep, value.shape[0] - 1))
            bbox = np.asarray(value[step, camera_index, array_object_index], dtype=np.float32)
        elif value.ndim == 3:
            bbox = np.asarray(value[camera_index, array_object_index], dtype=np.float32)
        else:
            return None
    except IndexError:
        return None
    if bbox.shape[-1] < 4 or not np.all(np.isfinite(bbox[:4])):
        return None
    visible = True
    if camera_object_visible is not None:
        visible_value = np.asarray(camera_object_visible)
        try:
            if visible_value.ndim == 3:
                step = max(0, min(timestep, visible_value.shape[0] - 1))
                visible = bool(visible_value[step, camera_index, array_object_index])
            elif visible_value.ndim == 2:
                visible = bool(visible_value[camera_index, array_object_index])
        except IndexError:
            visible = False
    x0_raw = float(bbox[0]) / max(1, width)
    y0_raw = float(bbox[1]) / max(1, height)
    x1_raw = float(bbox[2]) / max(1, width)
    y1_raw = float(bbox[3]) / max(1, height)
    in_frame = bool(visible and x1_raw >= 0.0 and x0_raw <= 1.0 and y1_raw >= 0.0 and y0_raw <= 1.0)
    x0 = min(1.0, max(0.0, min(x0_raw, x1_raw)))
    x1 = min(1.0, max(0.0, max(x0_raw, x1_raw)))
    y0 = min(1.0, max(0.0, min(y0_raw, y1_raw)))
    y1 = min(1.0, max(0.0, max(y0_raw, y1_raw)))
    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5
    return {
        "x0": _json_scalar(x0),
        "y0": _json_scalar(y0),
        "x1": _json_scalar(x1),
        "y1": _json_scalar(y1),
        "raw_x0": _json_scalar(x0_raw),
        "raw_y0": _json_scalar(y0_raw),
        "raw_x1": _json_scalar(x1_raw),
        "raw_y1": _json_scalar(y1_raw),
        "center_x": _json_scalar(center_x),
        "center_y": _json_scalar(center_y),
        "center_pixel_x": _json_scalar(center_x * max(1, width)),
        "center_pixel_y": _json_scalar(center_y * max(1, height)),
        "in_frame": in_frame,
        "source": "camera_segmentation",
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return int(numeric)


def _attention_map_payload(bundle: TraceBundle, query: dict[str, list[str]]) -> dict[str, Any]:
    calls = _policy_calls(bundle)
    if not calls:
        return {"available": False, "reason": "No policy calls are available."}
    kind = query.get("kind", ["expert"])[0]
    call = calls[_query_call_index(query)]
    generation_step = int(query.get("generation_step", ["0"])[0] or 0)
    name = query.get("name", [""])[0] or None
    head = query.get("head", [""])[0]
    head_index = int(head) if head not in {"", None} else None
    query_token = query.get("query_token", [""])[0]
    query_token_index = int(query_token) if query_token not in {"", None} else None
    try:
        key_mass, selected_site, axis_selection = _attention_key_mass_from_trace(
            bundle,
            kind,
            call,
            generation_step,
            name=name,
            head=head_index,
            query_token=query_token_index,
        )
    except KeyError:
        return _not_captured_in_profile(
            f"{kind} attention maps require attention arrays stored in the .vlatrace bundle.",
            kind=kind,
            generation_step=generation_step,
            name=name,
            head=head_index,
            query_token=query_token_index,
        )
    except ValueError as error:
        return {
            "available": False,
            "reason": "selected_axis_unavailable",
            "detail": str(error),
            "kind": kind,
            "generation_step": generation_step,
            "name": name,
            "head": head_index,
            "query_token": query_token_index,
        }
    layout = _attention_camera_layout(bundle, key_mass.shape[0])
    return {
        "available": True,
        "kind": kind,
        "call": call,
        "generation_step": generation_step,
        **axis_selection,
        "site": selected_site,
        "source": "vlatrace",
        **layout,
        "coarse": {
            "image": _json_scalar(float(np.nansum(key_mass[: int(layout["image_tokens"])]))),
            "prompt": None,
            "action_suffix": None,
        },
        "maps": _camera_maps_from_trace_key_mass(bundle, key_mass, layout),
    }


def _expert_token_attention_payload(
    bundle: TraceBundle,
    source_name: str,
    call: dict[str, Any],
    generation_step: int,
    token_index: int,
) -> dict[str, Any] | None:
    """Return action-query attention over prefix image/text tokens for one expert token."""
    attention = _expert_attention_for_token(
        bundle,
        source_name,
        call,
        generation_step,
        token_index,
    )
    if attention is None:
        return None
    key_mass, site_name = attention
    prefix_rows = _token_rows_for_space(bundle, call, "pi05.prefix")
    if prefix_rows.empty:
        return None

    prefix_count = _token_count(prefix_rows)
    prefix_mass = np.asarray(key_mass[:prefix_count], dtype=np.float32)
    action_mass = np.asarray(key_mass[prefix_count:], dtype=np.float32)
    maps, top_image_patches, image_mass = _image_attention_from_prefix_rows(
        bundle,
        prefix_rows,
        prefix_mass,
    )
    top_prompt_tokens, prompt_mass, prompt, prompt_tokens = _prompt_attention_from_prefix_rows(
        prefix_rows,
        prefix_mass,
    )
    return {
        "attention_site": site_name,
        "attention_coarse": {
            "image": _json_scalar(float(image_mass)),
            "prompt": _json_scalar(float(prompt_mass)),
            "action_suffix": (
                _json_scalar(float(np.nansum(action_mass))) if action_mass.size else 0.0
            ),
        },
        "top_prompt_tokens": top_prompt_tokens,
        "prompt_tokens": prompt_tokens,
        "top_image_patches": top_image_patches,
        "maps": maps,
        "prompt": bundle.manifest.prompt or prompt,
    }


def _expert_attention_for_token(
    bundle: TraceBundle,
    source_name: str,
    call: dict[str, Any],
    generation_step: int,
    token_index: int,
) -> tuple[np.ndarray, str] | None:
    matches = bundle.model_sites.loc[bundle.model_sites["name"].astype(str) == source_name]
    if matches.empty:
        return None
    source = matches.iloc[0]
    layer = source.get("layer")
    candidates = _expert_attention_site_candidates(bundle, layer)
    if candidates.empty:
        return None
    record = candidates.iloc[0]
    name = str(record["name"])
    axes = json.loads(str(record.get("axes") or "[]"))
    array = np.asarray(bundle.model_site(name, mmap=True), dtype=np.float32)
    value, remaining_axes = _take_policy_call_value(array, axes, call)
    if "generation_step" in remaining_axes:
        value = _take_axis_value(value, remaining_axes, "generation_step", generation_step)
        remaining_axes = [axis for axis in remaining_axes if axis != "generation_step"]
    value_array = np.asarray(value, dtype=np.float32)
    if "head" in remaining_axes:
        value_array = np.nanmean(value_array, axis=remaining_axes.index("head"))
        remaining_axes = [axis for axis in remaining_axes if axis != "head"]
    if "query_token" in remaining_axes:
        value_array = _take_axis_value(value_array, remaining_axes, "query_token", token_index)
        remaining_axes = [axis for axis in remaining_axes if axis != "query_token"]
    if "key_token" in remaining_axes:
        key_axis = remaining_axes.index("key_token")
        value_array = np.moveaxis(value_array, key_axis, -1)
    return np.asarray(value_array, dtype=np.float32).reshape(-1), name


def _expert_attention_site_candidates(bundle: TraceBundle, layer: Any) -> Any:
    if bundle.model_sites.empty:
        return bundle.model_sites
    table = bundle.model_sites.copy()
    names = table["name"].astype(str)
    table = table.loc[names.str.contains(".expert.layers.", regex=False)].copy()
    if table.empty:
        return table
    if layer is not None and str(layer) != "nan":
        numeric_layer = float(layer)
        table = table.loc[table.get("layer").astype(float) == numeric_layer].copy()
    if table.empty:
        return table
    axes = table.get("axes", "").astype(str)
    roles = table.get("role", "").astype(str)
    tensor_types = table.get("tensor_type", "").astype(str)
    names = table["name"].astype(str)
    table = table.loc[
        axes.str.contains("query_token")
        & axes.str.contains("key_token")
        & (
            (roles == "attention_probs")
            | (tensor_types == "attention_probs")
            | names.str.endswith(".by_step.attention")
        )
    ].copy()
    if table.empty:
        return table
    table["_priority"] = np.select(
        [
            table["name"].astype(str).str.endswith(".attention.attention_probs"),
            table["name"].astype(str).str.endswith(".by_step.attention"),
        ],
        [0, 1],
        default=2,
    )
    return table.sort_values(["_priority", "name"])


def _attention_key_mass_from_trace(
    bundle: TraceBundle,
    kind: str,
    call: dict[str, Any],
    generation_step: int,
    *,
    name: str | None = None,
    head: int | None = None,
    query_token: int | None = None,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    matches = _attention_site_matches(bundle, kind, name=name)
    if matches.empty:
        raise KeyError(kind)
    record = matches.iloc[-1]
    name = str(record["name"])
    axes = json.loads(str(record.get("axes") or "[]"))
    array = np.asarray(bundle.model_site(name, mmap=True), dtype=np.float32)
    value, remaining_axes = _take_policy_call_value(array, axes, call)
    if "generation_step" in remaining_axes:
        value = _take_axis_value(value, remaining_axes, "generation_step", generation_step)
        remaining_axes = [axis for axis in remaining_axes if axis != "generation_step"]
    value_array = np.asarray(value, dtype=np.float32)
    axis_selection: dict[str, Any] = {
        "head": None,
        "head_mode": "average",
        "query_token": None,
        "query_mode": "average",
    }
    if "head" in remaining_axes:
        head_axis = remaining_axes.index("head")
        if head is None:
            value_array = np.nanmean(value_array, axis=head_axis)
            axis_selection["head_mode"] = "average"
        else:
            value_array = _take_axis_value(value_array, remaining_axes, "head", head)
            axis_selection["head"] = int(head)
            axis_selection["head_mode"] = "selected"
        remaining_axes = [axis for axis in remaining_axes if axis != "head"]
    elif head is not None:
        raise ValueError("Selected head is not available for this attention capture.")
    if "query_token" in remaining_axes:
        query_axis = remaining_axes.index("query_token")
        if query_token is None:
            value_array = np.nanmean(value_array, axis=query_axis)
            axis_selection["query_mode"] = "average"
        else:
            value_array = _take_axis_value(value_array, remaining_axes, "query_token", query_token)
            axis_selection["query_token"] = int(query_token)
            axis_selection["query_mode"] = "selected"
        remaining_axes = [axis for axis in remaining_axes if axis != "query_token"]
    elif query_token is not None:
        raise ValueError("Selected looking slot is not available for this attention capture.")
    return value_array.reshape(-1), name, axis_selection


def _attention_site_matches(bundle: TraceBundle, kind: str, *, name: str | None = None) -> Any:
    if bundle.model_sites.empty:
        return bundle.model_sites
    table = bundle.model_sites.copy()
    names = table["name"].astype(str)
    if name:
        table = table.loc[names == name].copy()
        names = table["name"].astype(str)
    roles = table.get("role", "").astype(str)
    tensor_types = table.get("tensor_type", "").astype(str)
    table = table.loc[
        (tensor_types.isin({"attention", "attention_probs"}) | (roles == "attention_probs"))
        & names.str.contains(f"pi05.{kind}.", regex=False)
    ].copy()
    if table.empty:
        return table
    table["_layer_sort"] = table.get("layer", 0).fillna(0).astype(float)
    table["_key_mass_sort"] = names.loc[table.index].str.contains("attention_key_mass").astype(int)
    return table.sort_values(["_key_mass_sort", "_layer_sort", "name"])


def _camera_maps_from_trace_key_mass(
    bundle: TraceBundle,
    key_mass: np.ndarray,
    layout: dict[str, int],
) -> dict[str, Any]:
    maps: dict[str, Any] = {}
    grid_size = int(layout["grid_size"])
    patches_per_image = int(layout["patches_per_image"])
    image_slots = int(layout["image_slots"])
    for camera_index, camera in enumerate(bundle.cameras()):
        if camera_index >= image_slots:
            continue
        start = camera_index * patches_per_image
        end = start + patches_per_image
        if end > key_mass.shape[0]:
            continue
        values = key_mass[start:end].reshape(grid_size, grid_size)
        maps[camera] = {
            "values": _round(values),
            "token_start": start,
            "token_end": end - 1,
            "min": _json_scalar(float(np.nanmin(values))),
            "max": _json_scalar(float(np.nanmax(values))),
        }
    return maps


def _attention_camera_layout(bundle: TraceBundle, key_count: int) -> dict[str, int]:
    if not bundle.model_sites.empty:
        matches = bundle.model_sites.loc[
            bundle.model_sites["name"].astype(str) == "pi05.vlm.prefix.image_hidden_tokens"
        ]
        if not matches.empty:
            metadata = json.loads(str(matches.iloc[0].get("metadata") or "{}"))
            image_tokens = int(metadata.get("patches_per_image") or 0) * int(
                metadata.get("image_slots") or 0
            )
            if image_tokens > 0 and key_count >= image_tokens:
                return _camera_patch_layout_from_record(
                    bundle,
                    matches.iloc[0],
                    key_count,
                    text_tokens=key_count - image_tokens,
                )
    return _camera_patch_layout(bundle, key_count, text_tokens=0)


def _patch_features_payload(bundle: TraceBundle, query: dict[str, list[str]]) -> dict[str, Any]:
    name = _query_one(query, "name")
    calls = _policy_calls(bundle)
    if not calls:
        return {"available": False, "reason": "No policy calls are available."}
    call = calls[_query_call_index(query)]
    camera = _query_one(query, "camera")
    row = int(_query_one(query, "row"))
    col = int(_query_one(query, "col"))
    feature = int(query.get("feature", ["0"])[0])
    matches = bundle.model_sites.loc[bundle.model_sites["name"].astype(str) == name]
    if matches.empty:
        raise KeyError(name)
    record = matches.iloc[0]
    token_matrix = _activation_token_matrix(bundle, name, call, query)
    cameras = bundle.cameras()
    if camera not in cameras:
        raise KeyError(f"Unknown camera {camera!r}; available={cameras}")
    token_index, row, col = _image_token_index_for_patch(
        bundle,
        record,
        call,
        token_matrix.shape[0],
        camera,
        row,
        col,
    )
    if token_index is None:
        layout = _camera_patch_layout(bundle, token_matrix.shape[0], text_tokens=0)
        camera_index = cameras.index(camera)
        grid_size = int(layout["grid_size"])
        row = max(0, min(row, grid_size - 1))
        col = max(0, min(col, grid_size - 1))
        token_index = camera_index * int(layout["patches_per_image"]) + row * grid_size + col
    token_index = max(0, min(int(token_index), token_matrix.shape[0] - 1))
    vector = token_matrix[token_index]
    safe = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(np.abs(safe))[::-1]
    positive = np.argsort(safe)[::-1]
    negative = np.argsort(safe)
    feature = max(0, min(feature, vector.shape[0] - 1))
    feature_rank = int(np.where(order == feature)[0][0]) + 1 if vector.size else None
    return {
        "available": True,
        "name": name,
        "call": call,
        "camera": camera,
        "patch_row": row,
        "patch_col": col,
        "token_index": int(token_index),
        "feature": feature,
        "feature_value": _json_scalar(float(vector[feature])),
        "feature_rank_by_abs": feature_rank,
        "feature_count": int(vector.shape[0]),
        "top_abs": [
            {"index": int(index), "value": _json_scalar(float(vector[index]))}
            for index in order[:32]
        ],
        "top_positive": [
            {"index": int(index), "value": _json_scalar(float(vector[index]))}
            for index in positive[:16]
        ],
        "top_negative": [
            {"index": int(index), "value": _json_scalar(float(vector[index]))}
            for index in negative[:16]
        ],
    }


def _activation_token_matrix(
    bundle: TraceBundle,
    name: str,
    call: dict[str, Any],
    query: dict[str, list[str]],
) -> np.ndarray:
    matches = bundle.model_sites.loc[bundle.model_sites["name"].astype(str) == name]
    if matches.empty:
        raise KeyError(name)
    record = matches.iloc[0]
    axes = json.loads(str(record.get("axes") or "[]"))
    array = np.asarray(bundle.model_site(name, mmap=True), dtype=np.float32)
    value, remaining_axes = _take_policy_call_value(array, axes, call)
    if "generation_step" in remaining_axes:
        generation_step = query.get("generation_step", [""])[0]
        step = int(generation_step) if generation_step not in {"", None} else 0
        value = _take_axis_value(value, remaining_axes, "generation_step", step)
        remaining_axes = [axis for axis in remaining_axes if axis != "generation_step"]
    matrix = np.asarray(value, dtype=np.float32)
    if "token" in remaining_axes:
        token_axis = remaining_axes.index("token")
        matrix = np.moveaxis(matrix, token_axis, 0)
        if matrix.ndim == 1:
            matrix = matrix.reshape(matrix.shape[0], 1)
        else:
            matrix = matrix.reshape(matrix.shape[0], -1)
    elif matrix.ndim != 2:
        matrix = matrix.reshape(-1, matrix.shape[-1])
    if matrix.ndim != 2:
        raise ValueError(f"Expected token x channel activation for {name!r}, got {matrix.shape}")
    return matrix


def _camera_patch_layout(
    bundle: TraceBundle,
    token_count: int,
    *,
    text_tokens: int,
) -> dict[str, int]:
    image_tokens = max(0, token_count - text_tokens)
    camera_count = max(1, len(bundle.cameras()))
    candidate = image_tokens // camera_count if image_tokens % camera_count == 0 else image_tokens
    root = int(round(float(np.sqrt(max(1, candidate)))))
    patches_per_image = candidate if root * root == candidate else _patches_per_image(image_tokens)
    grid_size = int(round(float(np.sqrt(patches_per_image)))) if patches_per_image else 0
    image_slots = image_tokens // patches_per_image if patches_per_image else 0
    return {
        "grid_size": grid_size,
        "grid_height": grid_size,
        "grid_width": grid_size,
        "patches_per_image": patches_per_image,
        "image_tokens": image_tokens,
        "text_tokens": text_tokens,
        "image_slots": min(image_slots, len(bundle.cameras())),
    }


def _camera_patch_layout_from_record(
    bundle: TraceBundle,
    record: Any,
    token_count: int,
    *,
    text_tokens: int,
) -> dict[str, int]:
    metadata = json.loads(str(record.get("metadata") or "{}"))
    patches_per_image = int(metadata.get("patches_per_image") or 0)
    grid_height = int(metadata.get("grid_height") or metadata.get("grid_size") or 0)
    grid_width = int(metadata.get("grid_width") or metadata.get("grid_size") or 0)
    if patches_per_image > 0 and grid_height > 0 and grid_width > 0:
        image_tokens = max(0, token_count - text_tokens)
        return {
            "grid_size": grid_height if grid_height == grid_width else 0,
            "grid_height": grid_height,
            "grid_width": grid_width,
            "patches_per_image": patches_per_image,
            "image_tokens": image_tokens,
            "text_tokens": text_tokens,
            "image_slots": min(image_tokens // patches_per_image, len(bundle.cameras())),
        }
    return _camera_patch_layout(bundle, token_count, text_tokens=text_tokens)


def _token_rows_for_space(
    bundle: TraceBundle,
    call: dict[str, Any],
    token_space_id: str,
) -> Any:
    rows = bundle.tokens
    if rows.empty or "token_space_id" not in rows:
        return rows.iloc[0:0].copy()
    rows = rows.loc[rows["token_space_id"].astype(str) == token_space_id].copy()
    if rows.empty:
        return rows
    if "policy_call_index" in rows:
        call_index = int(call.get("model_call_index", call.get("index", 0)))
        call_rows = rows.loc[rows["policy_call_index"].astype(int) == call_index].copy()
        if not call_rows.empty:
            rows = call_rows
    return rows.sort_values("token_index").reset_index(drop=True)


def _token_count(rows: Any) -> int:
    if rows.empty or "token_index" not in rows:
        return 0
    return int(rows["token_index"].max()) + 1


def _image_token_rows_for_site(
    bundle: TraceBundle,
    record: Any,
    call: dict[str, Any],
    token_count: int,
) -> Any:
    token_space_id = str(record.get("token_space_id") or "")
    if not token_space_id or token_space_id.lower() == "nan":
        return bundle.tokens.iloc[0:0].copy()
    rows = _token_rows_for_space(bundle, call, token_space_id)
    if rows.empty:
        return rows
    token_kind = rows.get("token_kind", "").astype(str)
    image_rows = rows.loc[token_kind == "image"].copy()
    if image_rows.empty or "token_index" not in image_rows:
        return image_rows
    image_rows = image_rows.loc[image_rows["token_index"].astype(int) < token_count].copy()
    return image_rows.sort_values("token_index").reset_index(drop=True)


def _camera_patch_maps_from_token_rows(
    bundle: TraceBundle,
    image_rows: Any,
    token_matrix: np.ndarray,
    feature: int,
) -> tuple[dict[str, Any], dict[str, int | None]]:
    maps: dict[str, Any] = {}
    grid_heights: list[int] = []
    grid_widths: list[int] = []
    patch_counts: list[int] = []
    for camera in bundle.cameras():
        camera_rows = image_rows.loc[image_rows.get("camera_id", "").astype(str) == camera].copy()
        if camera_rows.empty:
            continue
        grid_height = int(camera_rows.get("patch_row", 0).max()) + 1
        grid_width = int(camera_rows.get("patch_col", 0).max()) + 1
        values = np.full((grid_height, grid_width), np.nan, dtype=np.float32)
        for row in camera_rows.to_dict("records"):
            token_index = int(row.get("token_index", 0))
            if token_index >= token_matrix.shape[0]:
                continue
            patch_row = int(row.get("patch_row", 0))
            patch_col = int(row.get("patch_col", 0))
            if patch_row < grid_height and patch_col < grid_width:
                values[patch_row, patch_col] = float(token_matrix[token_index, feature])
        grid_heights.append(grid_height)
        grid_widths.append(grid_width)
        patch_counts.append(int(len(camera_rows)))
        finite_values = np.nan_to_num(values, nan=0.0)
        maps[camera] = {
            "values": _round(finite_values),
            "token_start": int(camera_rows["token_index"].min()),
            "token_end": int(camera_rows["token_index"].max()),
            "active_tokens": int(len(camera_rows)),
            "min": _json_scalar(float(np.nanmin(values))),
            "max": _json_scalar(float(np.nanmax(values))),
        }
    grid_height = grid_heights[0] if grid_heights and len(set(grid_heights)) == 1 else None
    grid_width = grid_widths[0] if grid_widths and len(set(grid_widths)) == 1 else None
    patches_per_image = patch_counts[0] if patch_counts and len(set(patch_counts)) == 1 else 0
    return maps, {
        "grid_size": grid_height if grid_height is not None and grid_height == grid_width else None,
        "grid_height": grid_height or 0,
        "grid_width": grid_width or 0,
        "patches_per_image": patches_per_image,
    }


def _image_token_index_for_patch(
    bundle: TraceBundle,
    record: Any,
    call: dict[str, Any],
    token_count: int,
    camera: str,
    row: int,
    col: int,
) -> tuple[int | None, int, int]:
    image_rows = _image_token_rows_for_site(bundle, record, call, token_count)
    if image_rows.empty:
        return None, row, col
    camera_rows = image_rows.loc[image_rows.get("camera_id", "").astype(str) == camera].copy()
    if camera_rows.empty:
        return None, row, col
    max_row = int(camera_rows.get("patch_row", 0).max())
    max_col = int(camera_rows.get("patch_col", 0).max())
    row = max(0, min(row, max_row))
    col = max(0, min(col, max_col))
    matches = camera_rows.loc[
        (camera_rows.get("patch_row", 0).astype(int) == row)
        & (camera_rows.get("patch_col", 0).astype(int) == col)
    ]
    if matches.empty:
        return None, row, col
    return int(matches.iloc[0].get("token_index", 0)), row, col


def _image_attention_from_prefix_rows(
    bundle: TraceBundle,
    prefix_rows: Any,
    prefix_mass: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    image_rows = prefix_rows.loc[prefix_rows.get("token_kind", "").astype(str) == "image"].copy()
    if image_rows.empty:
        return {}, [], 0.0
    cameras = bundle.cameras()
    maps: dict[str, Any] = {}
    patch_rows: list[dict[str, Any]] = []
    image_mass = 0.0
    for camera in cameras:
        camera_rows = image_rows.loc[image_rows.get("camera_id", "").astype(str) == camera].copy()
        if camera_rows.empty:
            continue
        max_row = int(camera_rows.get("patch_row", 0).max())
        max_col = int(camera_rows.get("patch_col", 0).max())
        values = np.full((max_row + 1, max_col + 1), np.nan, dtype=np.float32)
        for row in camera_rows.to_dict("records"):
            token_index = int(row.get("token_index", 0))
            if token_index >= prefix_mass.shape[0]:
                continue
            patch_row = int(row.get("patch_row", 0))
            patch_col = int(row.get("patch_col", 0))
            attention = float(prefix_mass[token_index])
            values[patch_row, patch_col] = attention
            image_mass += attention
            patch_rows.append(
                {
                    "camera": camera,
                    "row": patch_row,
                    "col": patch_col,
                    "token_index": token_index,
                    "attention": _json_scalar(attention),
                }
            )
        if values.size:
            maps[camera] = {
                "values": _round(np.nan_to_num(values, nan=0.0)),
                "token_start": int(camera_rows["token_index"].min()),
                "token_end": int(camera_rows["token_index"].max()),
                "min": _json_scalar(float(np.nanmin(values))),
                "max": _json_scalar(float(np.nanmax(values))),
            }
    patch_rows.sort(key=lambda item: float(item.get("attention") or 0.0), reverse=True)
    return maps, patch_rows[:24], image_mass


def _prompt_attention_from_prefix_rows(
    prefix_rows: Any,
    prefix_mass: np.ndarray,
) -> tuple[list[dict[str, Any]], float, str, list[dict[str, Any]]]:
    text_rows = prefix_rows.loc[prefix_rows.get("token_kind", "").astype(str) == "language"].copy()
    if text_rows.empty:
        return [], 0.0, "", []
    if "attention_mask" in text_rows:
        active = text_rows["attention_mask"].astype(bool)
        active_rows = text_rows.loc[active].copy()
    else:
        active_rows = text_rows
    if active_rows.empty:
        return [], 0.0, "", []
    start = int(text_rows["token_index"].min())
    token_records: list[dict[str, Any]] = []
    prompt_pieces: list[str] = []
    prompt_mass = 0.0
    for row in active_rows.to_dict("records"):
        token_index = int(row.get("token_index", 0))
        if token_index >= prefix_mass.shape[0]:
            continue
        attention = float(prefix_mass[token_index])
        prompt_mass += attention
        token_piece = _display_token_piece(row)
        prompt_pieces.append(token_piece)
        token_records.append(
            {
                "local_index": token_index - start,
                "prefix_index": token_index,
                "token_id": _json_scalar(row.get("token_id")),
                "token_piece": token_piece,
                "attention": _json_scalar(attention),
            }
        )
    top_records = sorted(
        token_records,
        key=lambda item: float(item.get("attention") or 0.0),
        reverse=True,
    )
    return (
        top_records[:24],
        prompt_mass,
        _join_token_pieces(prompt_pieces),
        token_records,
    )


_NUMERIC_TOKEN_RE = re.compile(r"^-?\d+(?:\.0)?$")


def _display_token_piece(row: Mapping[str, Any]) -> str:
    """Return a human-readable token piece for numeric tokenizer rows."""

    raw_piece = row.get("token_piece")
    token_id = _optional_int(row.get("token_id"))
    piece = "" if raw_piece is None else str(raw_piece)
    if token_id is not None and (not piece or _NUMERIC_TOKEN_RE.match(piece)):
        decoded = _decode_paligemma_token(token_id)
        if decoded:
            piece = decoded
    return _clean_token_piece(piece)


@lru_cache(maxsize=4096)
def _decode_paligemma_token(token_id: int) -> str:
    tokenizer = _paligemma_tokenizer()
    if tokenizer is None:
        return ""
    try:
        piece = tokenizer.convert_ids_to_tokens([int(token_id)])
    except Exception:
        return ""
    if isinstance(piece, str):
        return piece
    if piece:
        return str(piece[0])
    return ""


@lru_cache(maxsize=1)
def _paligemma_tokenizer() -> Any | None:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            "google/paligemma-3b-pt-224",
            local_files_only=True,
        )
    except Exception:
        return None


def _clean_token_piece(piece: str) -> str:
    text = str(piece)
    text = text.replace("<0x0A>", "\n")
    text = text.replace("Ċ", "\n")
    return text


def _join_token_pieces(pieces: list[str]) -> str:
    text = "".join(piece.replace("▁", " ") for piece in pieces)
    text = text.replace("  ", " ")
    return text.strip()


def _not_captured_in_profile(reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "available": False,
        "reason": "not_captured_in_profile",
        "detail": reason,
        **extra,
    }


def _prompt_attention_payload(bundle: TraceBundle, query: dict[str, list[str]]) -> dict[str, Any]:
    calls = _policy_calls(bundle)
    if not calls:
        return {"available": False, "reason": "No policy calls are available."}
    kind = query.get("kind", ["expert"])[0]
    call = calls[_query_call_index(query)]
    generation_step = int(query.get("generation_step", ["0"])[0] or 0)
    name = query.get("name", [""])[0] or None
    head = query.get("head", [""])[0]
    head_index = int(head) if head not in {"", None} else None
    query_token = query.get("query_token", [""])[0]
    query_token_index = int(query_token) if query_token not in {"", None} else None
    try:
        key_mass, selected_site, axis_selection = _attention_key_mass_from_trace(
            bundle,
            kind,
            call,
            generation_step,
            name=name,
            head=head_index,
            query_token=query_token_index,
        )
    except KeyError:
        return _not_captured_in_profile(
            f"Prompt attention requires {kind} attention arrays stored in the .vlatrace bundle.",
            kind=kind,
            prompt=bundle.manifest.prompt,
            generation_step=generation_step,
            name=name,
            head=head_index,
            query_token=query_token_index,
        )
    except ValueError as error:
        return {
            "available": False,
            "reason": "selected_axis_unavailable",
            "detail": str(error),
            "kind": kind,
            "prompt": bundle.manifest.prompt,
            "generation_step": generation_step,
            "name": name,
            "head": head_index,
            "query_token": query_token_index,
        }
    prefix_rows = _token_rows_for_space(bundle, call, "pi05.prefix")
    if prefix_rows.empty:
        return _not_captured_in_profile(
            "Prompt attention requires token layout rows for pi05.prefix.",
            prompt=bundle.manifest.prompt,
            kind=kind,
            generation_step=generation_step,
            attention_site=selected_site,
        )
    prefix_count = _token_count(prefix_rows)
    prefix_mass = np.asarray(key_mass[:prefix_count], dtype=np.float32)
    _maps, top_image_patches, image_mass = _image_attention_from_prefix_rows(
        bundle,
        prefix_rows,
        prefix_mass,
    )
    top_text_tokens, prompt_mass, prompt, prompt_tokens = _prompt_attention_from_prefix_rows(
        prefix_rows,
        prefix_mass,
    )
    action_mass = np.asarray(key_mass[prefix_count:], dtype=np.float32)
    active_text_tokens = len(top_text_tokens)
    if "attention_mask" in prefix_rows:
        text_rows = prefix_rows.loc[prefix_rows.get("token_kind", "").astype(str) == "language"]
        active_text_tokens = int(text_rows.get("attention_mask", []).astype(bool).sum())
    return {
        "available": True,
        "kind": kind,
        "call": call,
        "generation_step": generation_step,
        **axis_selection,
        "attention_site": selected_site,
        "prompt": bundle.manifest.prompt or prompt,
        "active_text_tokens": active_text_tokens,
        "allocated_text_slots": int(
            (prefix_rows.get("token_kind", "").astype(str) == "language").sum()
        ),
        "expert_coarse": {
            "image": _json_scalar(float(image_mass)),
            "prompt": _json_scalar(float(prompt_mass)),
            "action_suffix": (
                _json_scalar(float(np.nansum(action_mass))) if action_mass.size else 0.0
            ),
        },
        "top_text_tokens": top_text_tokens,
        "prompt_tokens": prompt_tokens,
        "top_image_patches": top_image_patches,
    }


def _expert_token_model_sites_payload(
    bundle: TraceBundle,
    query: dict[str, list[str]],
) -> dict[str, Any]:
    name = _query_one(query, "name")
    calls = _policy_calls(bundle)
    if not calls:
        return {"available": False, "reason": "No policy calls are available."}
    call = calls[_query_call_index(query)]
    feature = int(query.get("feature", ["0"])[0])
    generation_step = int(query.get("generation_step", ["0"])[0] or 0)
    token_matrix = _activation_token_matrix(bundle, name, call, query)
    feature = max(0, min(feature, token_matrix.shape[1] - 1))
    values = token_matrix[:, feature]
    return {
        "available": True,
        "name": name,
        "call": call,
        "generation_step": generation_step,
        "feature": feature,
        "feature_count": int(token_matrix.shape[1]),
        "values": _round(values),
        "min": _json_scalar(float(np.nanmin(values))),
        "max": _json_scalar(float(np.nanmax(values))),
        "note": "Expert model_sites live on action/noise tokens, not image patch tokens.",
    }


def _expert_token_details_payload(
    bundle: TraceBundle,
    query: dict[str, list[str]],
) -> dict[str, Any]:
    name = _query_one(query, "name")
    calls = _policy_calls(bundle)
    if not calls:
        return {"available": False, "reason": "No policy calls are available."}
    call = calls[_query_call_index(query)]
    feature = int(query.get("feature", ["0"])[0])
    token_index = int(query.get("token_index", ["0"])[0])
    generation_step = int(query.get("generation_step", ["0"])[0] or 0)

    token_matrix = _activation_token_matrix(bundle, name, call, query)
    if token_matrix.ndim != 2:
        raise ValueError(f"Expected action-token x channel tensor, got {token_matrix.shape}")
    token_index = max(0, min(token_index, token_matrix.shape[0] - 1))
    feature = max(0, min(feature, token_matrix.shape[1] - 1))
    vector = token_matrix[token_index]
    safe = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(np.abs(safe))[::-1]
    feature_rank = int(np.where(order == feature)[0][0]) + 1 if vector.size else None

    action = _action_vector_for_token(bundle, call, token_index)
    attention = _expert_token_attention_payload(
        bundle,
        name,
        call,
        generation_step,
        token_index,
    )
    return {
        "available": True,
        "name": name,
        "call": call,
        "generation_step": generation_step,
        "token_index": token_index,
        "token_count": int(token_matrix.shape[0]),
        "feature": feature,
        "feature_value": _json_scalar(float(vector[feature])),
        "feature_rank_by_abs": feature_rank,
        "feature_count": int(vector.shape[0]),
        "top_abs": [
            {"index": int(index), "value": _json_scalar(float(vector[index]))}
            for index in order[:24]
        ],
        "attention_site": attention.get("attention_site") if attention else None,
        "attention_coarse": attention.get("attention_coarse") if attention else None,
        "top_prompt_tokens": attention.get("top_prompt_tokens", []) if attention else [],
        "prompt_tokens": attention.get("prompt_tokens", []) if attention else [],
        "top_image_patches": attention.get("top_image_patches", []) if attention else [],
        "maps": attention.get("maps", {}) if attention else {},
        "action": action,
        "note": (
            "This is one expert query/action token from the .vlatrace activation store. "
            "Image and prompt rows are attention mass from the matching expert layer/query token."
            if attention
            else "Attention details are unavailable unless captured into .vlatrace."
        ),
    }


def _action_vector_for_token(
    bundle: TraceBundle,
    call: dict[str, Any],
    token_index: int,
) -> dict[str, Any] | None:
    try:
        array = np.asarray(bundle.action_chunks(mmap=True)[int(call["index"])], dtype=np.float32)
    except KeyError:
        return None
    if array.ndim < 2 or array.shape[0] <= 0:
        return None
    token_index = max(0, min(int(token_index), array.shape[0] - 1))
    vector = np.asarray(array[token_index], dtype=np.float32).reshape(-1)
    safe = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    order = np.argsort(np.abs(safe))[::-1][:10]
    return {
        "source": "vlatrace.action_chunks",
        "dim": int(vector.shape[0]),
        "norm": _json_scalar(float(np.linalg.norm(safe))),
        "top_abs": [
            {"index": int(index), "value": _json_scalar(float(vector[index]))} for index in order
        ],
    }


def _episode_video_path(
    bundle: TraceBundle,
    *,
    camera: str,
    fps: int,
    max_width: int,
) -> Path:
    cameras = bundle.cameras()
    selected_cameras = cameras if camera == "all" else [camera]
    missing = [name for name in selected_cameras if name not in cameras]
    if missing:
        raise KeyError(f"Unknown camera(s): {missing}; available={cameras}")

    fps = max(1, min(int(fps), 30))
    max_width = max(64, min(int(max_width), 960))
    frame_timesteps = _video_frame_timesteps(bundle)
    video_dir = bundle.path / "artifacts" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    cache_name = (
        f"{_cache_part(bundle.manifest.trace_id)}__{_cache_part(camera)}"
        f"__full_episode__fps{fps}__w{max_width}.mp4"
    )
    video_path = video_dir / cache_name
    input_paths = _episode_frame_array_paths(bundle, selected_cameras)
    if video_path.exists() and not _video_cache_stale(video_path, input_paths):
        _ensure_episode_video_artifact(
            bundle,
            video_path=video_path,
            camera=camera,
            fps=fps,
            max_width=max_width,
            timesteps=frame_timesteps,
        )
        return video_path

    tmp_path = video_path.with_suffix(".tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        _write_episode_video(
            bundle,
            selected_cameras,
            tmp_path,
            fps=fps,
            max_width=max_width,
            timesteps=frame_timesteps,
        )
        tmp_path.replace(video_path)
        _ensure_episode_video_artifact(
            bundle,
            video_path=video_path,
            camera=camera,
            fps=fps,
            max_width=max_width,
            timesteps=frame_timesteps,
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return video_path


def _write_episode_video(
    bundle: TraceBundle,
    cameras: list[str],
    output_path: Path,
    *,
    fps: int,
    max_width: int,
    timesteps: list[int],
) -> None:
    if not cameras:
        raise ValueError("No cameras are available for this episode.")

    frame_arrays = {camera: bundle.frames(camera, mmap=True) for camera in cameras}
    frame_count = min(
        [int(bundle.manifest.length), *(int(frames.shape[0]) for frames in frame_arrays.values())]
    )
    if frame_count <= 0:
        raise ValueError(f"Episode {bundle.manifest.trace_id} has no frames.")
    selected_timesteps = [timestep for timestep in timesteps if 0 <= timestep < frame_count]
    if not selected_timesteps:
        selected_timesteps = list(range(frame_count))

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        macro_block_size=16,
        ffmpeg_params=[
            "-preset",
            "veryfast",
            "-crf",
            "34",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ],
    )
    try:
        for timestep in selected_timesteps:
            images = [
                _prepare_video_frame(frame_arrays[camera][timestep], max_width=max_width)
                for camera in cameras
            ]
            writer.append_data(np.asarray(_tile_video_frames(images)))
    finally:
        writer.close()


def _video_frame_timesteps(bundle: TraceBundle) -> list[int]:
    calls = bundle.policy_calls
    if not calls.empty:
        column = "observation_timestep" if "observation_timestep" in calls else "env_timestep_start"
        if column in calls:
            return [int(value) for value in calls[column].dropna().tolist()]
    table = bundle.timesteps
    if not table.empty and "timestep" in table:
        return [int(value) for value in table["timestep"].tolist()]
    return list(range(int(bundle.manifest.length)))


def _ensure_episode_video_artifact(
    bundle: TraceBundle,
    *,
    video_path: Path,
    camera: str,
    fps: int,
    max_width: int,
    timesteps: list[int],
) -> None:
    artifact_id = (
        f"episode_video-{_cache_part(bundle.manifest.trace_id)}-{_cache_part(camera)}"
        f"-full-episode-fps{fps}-w{max_width}"
    )
    relative_path = video_path.relative_to(bundle.path)
    artifact = LensArtifact(
        artifact_id=artifact_id,
        artifact_type="episode_video",
        name=f"Full episode video ({camera})",
        group_id="episode_videos",
        scope="bundle",
        selector={
            "trace_id": bundle.manifest.trace_id,
            "camera": camera,
            "timesteps": timesteps,
            "source": "trace_frames",
        },
        method={
            "codec": "libx264",
            "crf": 34,
            "preset": "veryfast",
            "fps": fps,
            "max_width": max_width,
            "layout": "stitched_cameras" if camera == "all" else "single_camera",
        },
        metrics={
            "frame_count": len(timesteps),
            "file_size_bytes": video_path.stat().st_size if video_path.exists() else 0,
        },
        display={
            "kind": "episode_video",
            "media_type": "video/mp4",
            "relative_path": str(relative_path),
        },
        tags=("video", "episode", "full_episode"),
        source_trace_ids=(bundle.manifest.trace_id,),
        path=str(Path("artifacts") / artifact_id / "artifact.json"),
    )
    bundle.save_artifact(artifact)


def _prepare_video_frame(frame: np.ndarray, *, max_width: int) -> Any:
    from PIL import Image

    array = np.asarray(frame)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        image = Image.fromarray(array, mode="L").convert("RGB")
    else:
        image = Image.fromarray(array[..., :3]).convert("RGB")

    scale = min(1.0, float(max_width) / max(1, image.width))
    width = max(2, int(round(image.width * scale)))
    height = max(2, int(round(image.height * scale)))
    width -= width % 2
    height -= height % 2
    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return image


def _tile_video_frames(images: list[Any]) -> Any:
    from PIL import Image

    if len(images) == 1:
        return _pad_video_frame(images[0])

    gap = 2
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    x = 0
    for image in images:
        y = (height - image.height) // 2
        canvas.paste(image, (x, y))
        x += image.width + gap
    return _pad_video_frame(canvas)


def _pad_video_frame(image: Any, *, multiple: int = 16) -> Any:
    from PIL import Image

    width = ((image.width + multiple - 1) // multiple) * multiple
    height = ((image.height + multiple - 1) // multiple) * multiple
    if (width, height) == image.size:
        return image
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(image, (0, 0))
    return canvas


def _episode_frame_array_paths(bundle: TraceBundle, cameras: list[str]) -> list[Path]:
    if bundle.array_index.empty:
        return []
    paths: list[Path] = []
    for camera in cameras:
        name = f"frames.{camera}"
        matches = bundle.array_index.loc[bundle.array_index["name"].astype(str) == name]
        if matches.empty:
            continue
        paths.append(bundle.path / str(matches.iloc[0]["relative_path"]))
    return paths


def _trace_frame_file_path(bundle: TraceBundle, *, camera: str, timestep: int) -> Path | None:
    if bundle.array_index.empty:
        return None
    name = f"frames.{camera}"
    matches = bundle.array_index.loc[bundle.array_index["name"].astype(str) == name]
    if matches.empty:
        return None
    frame_dir = bundle.path / str(matches.iloc[0]["relative_path"])
    if not frame_dir.is_dir():
        return None
    path = frame_dir / f"{timestep:06d}.jpg"
    return path if path.exists() else None


def _video_cache_stale(video_path: Path, input_paths: list[Path]) -> bool:
    if not input_paths:
        return True
    video_mtime = video_path.stat().st_mtime
    return any(path.exists() and path.stat().st_mtime > video_mtime for path in input_paths)


def _cache_part(value: str) -> str:
    safe = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(safe).strip("_")[:96] or "item"


def _take_axis_value(array: np.ndarray, axes: list[str], axis_name: str, index: int) -> np.ndarray:
    if axis_name not in axes:
        return array
    axis = axes.index(axis_name)
    limit = array.shape[axis]
    clipped = max(0, min(int(index), limit - 1))
    return np.take(array, clipped, axis=axis)


def _take_policy_call_value(
    array: np.ndarray,
    axes: list[str],
    call: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    if "policy_call" in axes:
        index = int(call.get("index", call.get("model_call_index", 0)))
        return (
            _take_axis_value(array, axes, "policy_call", index),
            [axis for axis in axes if axis != "policy_call"],
        )
    if "timestep" in axes:
        index = int(call["env_timestep"])
        return (
            _take_axis_value(array, axes, "timestep", index),
            [axis for axis in axes if axis != "timestep"],
        )
    return array, list(axes)


def _site_family(name: str) -> str:
    if ".vlm." in name:
        return "vlm"
    if ".expert." in name:
        return "expert"
    return "other"


def _patches_per_image(image_tokens: int) -> int:
    if image_tokens >= 256 and image_tokens % 256 == 0:
        return 256
    root = int(round(float(np.sqrt(max(1, image_tokens)))))
    if root * root == image_tokens:
        return image_tokens
    return 256


def _query_one(query: dict[str, list[str]], name: str) -> str:
    values = query.get(name)
    if not values:
        raise KeyError(f"Missing query parameter: {name}")
    return values[0]


def _query_call_index(query: dict[str, list[str]]) -> int:
    return int(_query_one(query, "call_index"))


def _query_float(query: dict[str, list[str]], name: str, default: float) -> float:
    values = query.get(name)
    if not values or values[0] in {"", None}:
        return default
    return float(values[0])


def _round(array: np.ndarray) -> Any:
    value = np.round(np.asarray(array, dtype=np.float32), 4)
    return _jsonable(value.tolist())


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return _json_scalar(value)


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return _json_scalar(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


__all__ = ["run_dashboard_server"]
