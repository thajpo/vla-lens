"""Local slice-based dashboard server for VLA-lens trace datasets."""

from __future__ import annotations

import io
import json
import mimetypes
import threading
import urllib.parse
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import monotonic
from typing import Any

import imageio.v2 as imageio
import numpy as np

from vla_lens.server_helpers import (
    _action_metric_metadata,
    _action_norm_payload,
    _action_vector_for_token,
    _activation_architecture,
    _activation_architecture_edges,
    _activation_runtime_collections,
    _activation_sites_payload,
    _activation_slice_payload,
    _activation_token_feature_vector,
    _activation_token_matrix,
    _analysis_runs_payload,
    _api_exception_message,
    _api_exception_status,
    _array_preview,
    _array_summary,
    _artifact_detail_payload,
    _artifact_record_payload,
    _artifact_records,
    _artifact_summary,
    _artifacts_payload,
    _attention_camera_layout,
    _attention_key_mass_from_trace,
    _attention_map_payload,
    _attention_site_matches,
    _best_probe_rows,
    _cache_part,
    _camera_aliases,
    _camera_extrinsic_at,
    _camera_frame_size,
    _camera_index_for_array,
    _camera_maps_from_trace_key_mass,
    _camera_names_for_array,
    _camera_object_bbox_projection,
    _camera_patch_layout,
    _camera_patch_layout_from_record,
    _camera_patch_maps_from_token_rows,
    _camera_projection_calibration,
    _camera_resolution_from_context,
    _camera_to_pixel_transform,
    _captured_layers,
    _clean_token_piece,
    _cohort_compare_payload,
    _cohorts_payload,
    _comparison_episode_payload,
    _counterfactual_member_sort_key,
    _counterfactual_metadata_from_manifest,
    _counterfactual_pairs_payload,
    _counterfactual_pairs_response,
    _create_action_generation_payload,
    _create_outcome_probe_payload,
    _create_target_object_probe_payload,
    _dataset_diagnostics_payload,
    _dataset_payload,
    _dataset_signature,
    _dataset_trace_count_hint,
    _decode_paligemma_token,
    _dedupe_reasons,
    _default_outcome_probe_spec,
    _default_target_object_probe_spec,
    _diagnostics_payload,
    _display_token_piece,
    _domain_x_label,
    _dominant_value,
    _empty_episode_annotation,
    _ensure_episode_video_artifact,
    _episode_annotations_path,
    _episode_annotations_payload,
    _episode_frame_array_paths,
    _episode_interactions_payload,
    _episode_metrics_payload,
    _episode_payload,
    _episode_probe_prediction_path,
    _episode_probes_payload,
    _episode_video_path,
    _episode_video_path_locked,
    _expert_attention_for_token,
    _expert_attention_site_candidates,
    _expert_token_attention_payload,
    _expert_token_details_payload,
    _expert_token_model_sites_payload,
    _filter_rows_for_probe_feature,
    _generation_commitment_payload,
    _graph_payload,
    _image_attention_from_prefix_rows,
    _image_token_index_for_patch,
    _image_token_map_payload,
    _image_token_rows_for_site,
    _interaction_episode_payload,
    _interaction_metrics_table,
    _interaction_object_payload,
    _interaction_quality_payload,
    _intervention_runs_payload,
    _is_missing_scalar,
    _join_token_pieces,
    _json_list,
    _json_parse,
    _json_scalar,
    _jsonable,
    _label_from_metric_name,
    _latest_interaction_metrics_artifact,
    _lens_array_meta_payload,
    _lens_array_slice_payload,
    _lens_arrays_payload,
    _lerobot_signature_paths,
    _linear_probe_predictions,
    _manifest_payload,
    _mean_numeric,
    _metadata_list_for_array,
    _metadata_text,
    _not_captured_in_profile,
    _object_bbox_at,
    _object_camera_overlay_payload,
    _object_names_for_array,
    _object_position_at,
    _object_quat_at,
    _observational_candidate_score,
    _observational_comparison_artifact_id,
    _observational_comparisons_payload,
    _optional_array,
    _optional_bool,
    _optional_float,
    _optional_int,
    _optional_text,
    _pad_video_frame,
    _paligemma_tokenizer,
    _patch_features_payload,
    _patches_per_image,
    _policy_call_axis_selection,
    _policy_call_x_values,
    _policy_calls,
    _policy_calls_payload,
    _prepare_video_frame,
    _probe_episode_summary,
    _probe_index_artifact_payload,
    _probe_index_by_trace,
    _probe_index_payload,
    _probe_index_prediction_summary,
    _probe_index_split,
    _probe_index_split_summary,
    _probe_prediction_rows,
    _probe_prediction_table,
    _probe_split_category,
    _probe_split_sidecar,
    _probe_trace_record,
    _project_world_bbox,
    _project_world_point,
    _projection_payload,
    _prompt_attention_from_prefix_rows,
    _prompt_attention_payload,
    _prompt_feature_map_payload,
    _query_call_index,
    _query_float,
    _query_int_value,
    _query_one,
    _rank_feature_vector,
    _read_episode_annotations,
    _record_bool,
    _record_float,
    _record_text,
    _resolve_selection_payload,
    _round,
    _run_dataset_diagnostics_payload,
    _safe_filename,
    _save_analysis_run_payload,
    _save_cohort_from_selection_payload,
    _save_cohort_payload,
    _save_episode_annotation_payload,
    _save_intervention_run_payload,
    _save_workspace_payload,
    _saved_episode_probe_predictions,
    _saved_probe_prediction_tables,
    _scene_object_rows,
    _score_and_save_episode_probe,
    _site_family,
    _site_layer,
    _string_list,
    _table_query_payload,
    _take_axis_value,
    _take_axis_values,
    _take_policy_call_value,
    _tile_video_frames,
    _token_count,
    _token_rows_for_space,
    _trace_frame_file_path,
    _unique_paths,
    _unit_profile_payload,
    _video_cache_stale,
    _video_frame_timesteps,
    _vlm_kv_sites_by_layer,
    _workbench_payload,
    _workbench_signature_paths,
    _workspaces_payload,
    _write_episode_video,
)
from vla_lens.traces import TraceBundle, TraceDataset
from vla_lens.workbench import (
    resolve_workspace,
    spatial_overlay_contracts,
    validate_workbench_contracts,
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
            elif path == "/api/observational-comparisons":
                self._send_json(_observational_comparisons_payload(self.dataset, query))
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
            elif path == "/api/episode-probes":
                self._send_json(_episode_probes_payload(self.dataset, query))
            elif path == "/api/probe-index":
                self._send_json(_probe_index_payload(self.dataset))
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
                self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown route: {path}")
        except (BrokenPipeError, ConnectionResetError):  # browser navigated or scrubbed away
            return
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, FileNotFoundError) as exc:
            self._send_api_exception(exc)
        except Exception as exc:  # pragma: no cover - defensive server boundary
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, repr(exc))

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
                self._send_error_json(HTTPStatus.NOT_FOUND, f"Unknown route: {path}")
            self._clear_payload_cache()
        except (BrokenPipeError, ConnectionResetError):  # browser navigated or scrubbed away
            return
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, FileNotFoundError) as exc:
            self._send_api_exception(exc)
        except Exception as exc:  # pragma: no cover - defensive server boundary
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, repr(exc))

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

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        payload = json.dumps(
            {"error": status.phrase, "message": message},
            allow_nan=False,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_api_exception(self, exc: Exception) -> None:
        status = _api_exception_status(exc)
        self._send_error_json(status, _api_exception_message(exc))

    def _clear_payload_cache(self) -> None:
        cls = type(self)
        with cls.dataset_lock:
            cls.payload_cache.clear()

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
                self.send_header("Cache-Control", "no-store")
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
                if PI05LiberoReplayRenderer is None:
                    raise RuntimeError("PI0.5 LIBERO replay dependencies are not available")
                renderer = PI05LiberoReplayRenderer(bundle)
                self.replay_renderers[bundle.manifest.trace_id] = renderer
            return renderer




__all__ = [
    "run_dashboard_server",
    "_api_exception_status",
    "_api_exception_message",
    "_dataset_payload",
    "_counterfactual_pairs_response",
    "_counterfactual_pairs_payload",
    "_counterfactual_metadata_from_manifest",
    "_counterfactual_member_sort_key",
    "_observational_comparisons_payload",
    "_observational_comparison_artifact_id",
    "_query_int_value",
    "_probe_trace_record",
    "_comparison_episode_payload",
    "_observational_candidate_score",
    "_metadata_text",
    "_record_bool",
    "_record_float",
    "_record_text",
    "_dedupe_reasons",
    "_string_list",
    "_workbench_payload",
    "_episode_annotations_payload",
    "_save_episode_annotation_payload",
    "_read_episode_annotations",
    "_episode_annotations_path",
    "_empty_episode_annotation",
    "_dataset_signature",
    "_lerobot_signature_paths",
    "_workbench_signature_paths",
    "_dataset_trace_count_hint",
    "_unique_paths",
    "_lens_arrays_payload",
    "_lens_array_meta_payload",
    "_lens_array_slice_payload",
    "_cohorts_payload",
    "_save_cohort_payload",
    "_save_cohort_from_selection_payload",
    "_analysis_runs_payload",
    "_intervention_runs_payload",
    "_save_analysis_run_payload",
    "_save_intervention_run_payload",
    "_workspaces_payload",
    "_cohort_compare_payload",
    "_unit_profile_payload",
    "_projection_payload",
    "_graph_payload",
    "_table_query_payload",
    "_save_workspace_payload",
    "_resolve_selection_payload",
    "_dataset_diagnostics_payload",
    "_run_dataset_diagnostics_payload",
    "_create_outcome_probe_payload",
    "_create_target_object_probe_payload",
    "_create_action_generation_payload",
    "_default_target_object_probe_spec",
    "_default_outcome_probe_spec",
    "_diagnostics_payload",
    "_episode_payload",
    "_artifacts_payload",
    "_artifact_detail_payload",
    "_artifact_summary",
    "_artifact_records",
    "_artifact_record_payload",
    "_json_parse",
    "_array_summary",
    "_array_preview",
    "_manifest_payload",
    "_action_norm_payload",
    "_policy_calls_payload",
    "_policy_calls",
    "_generation_commitment_payload",
    "_action_metric_metadata",
    "_label_from_metric_name",
    "_policy_call_x_values",
    "_domain_x_label",
    "_episode_metrics_payload",
    "_episode_interactions_payload",
    "_latest_interaction_metrics_artifact",
    "_interaction_metrics_table",
    "_episode_probes_payload",
    "_probe_index_payload",
    "_probe_index_artifact_payload",
    "_probe_split_sidecar",
    "_saved_probe_prediction_tables",
    "_probe_index_by_trace",
    "_best_probe_rows",
    "_probe_index_split",
    "_probe_split_category",
    "_is_missing_scalar",
    "_probe_index_split_summary",
    "_probe_index_prediction_summary",
    "_probe_prediction_table",
    "_saved_episode_probe_predictions",
    "_score_and_save_episode_probe",
    "_episode_probe_prediction_path",
    "_filter_rows_for_probe_feature",
    "_linear_probe_predictions",
    "_safe_filename",
    "_probe_prediction_rows",
    "_probe_episode_summary",
    "_dominant_value",
    "_mean_numeric",
    "_interaction_episode_payload",
    "_interaction_quality_payload",
    "_interaction_object_payload",
    "_optional_text",
    "_optional_int",
    "_optional_float",
    "_optional_bool",
    "_activation_sites_payload",
    "_activation_runtime_collections",
    "_activation_architecture",
    "_activation_architecture_edges",
    "_vlm_kv_sites_by_layer",
    "_captured_layers",
    "_site_layer",
    "_json_list",
    "_activation_slice_payload",
    "_rank_feature_vector",
    "_image_token_map_payload",
    "_prompt_feature_map_payload",
    "_object_camera_overlay_payload",
    "_scene_object_rows",
    "_optional_array",
    "_camera_frame_size",
    "_camera_resolution_from_context",
    "_camera_projection_calibration",
    "_camera_extrinsic_at",
    "_camera_index_for_array",
    "_camera_names_for_array",
    "_object_names_for_array",
    "_metadata_list_for_array",
    "_camera_aliases",
    "_camera_to_pixel_transform",
    "_object_position_at",
    "_object_quat_at",
    "_object_bbox_at",
    "_project_world_point",
    "_project_world_bbox",
    "_camera_object_bbox_projection",
    "_attention_map_payload",
    "_expert_token_attention_payload",
    "_expert_attention_for_token",
    "_expert_attention_site_candidates",
    "_attention_key_mass_from_trace",
    "_attention_site_matches",
    "_camera_maps_from_trace_key_mass",
    "_attention_camera_layout",
    "_patch_features_payload",
    "_activation_token_matrix",
    "_activation_token_feature_vector",
    "_camera_patch_layout",
    "_camera_patch_layout_from_record",
    "_token_rows_for_space",
    "_token_count",
    "_image_token_rows_for_site",
    "_camera_patch_maps_from_token_rows",
    "_image_token_index_for_patch",
    "_image_attention_from_prefix_rows",
    "_prompt_attention_from_prefix_rows",
    "_display_token_piece",
    "_decode_paligemma_token",
    "_paligemma_tokenizer",
    "_clean_token_piece",
    "_join_token_pieces",
    "_not_captured_in_profile",
    "_prompt_attention_payload",
    "_expert_token_model_sites_payload",
    "_expert_token_details_payload",
    "_action_vector_for_token",
    "_episode_video_path",
    "_episode_video_path_locked",
    "_write_episode_video",
    "_video_frame_timesteps",
    "_ensure_episode_video_artifact",
    "_prepare_video_frame",
    "_tile_video_frames",
    "_pad_video_frame",
    "_episode_frame_array_paths",
    "_trace_frame_file_path",
    "_video_cache_stale",
    "_cache_part",
    "_take_axis_value",
    "_take_axis_values",
    "_policy_call_axis_selection",
    "_take_policy_call_value",
    "_site_family",
    "_patches_per_image",
    "_query_one",
    "_query_call_index",
    "_query_float",
    "_round",
    "_jsonable",
    "_json_scalar",
]
