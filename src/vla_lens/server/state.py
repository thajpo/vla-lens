"""Shared FastAPI dashboard state."""

from __future__ import annotations

import io
import threading
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Any

import imageio.v2 as imageio
import numpy as np

from vla_lens.dataset import validate_dataset_index
from vla_lens.server.common import _query_one
from vla_lens.server.dataset import _dataset_signature
from vla_lens.server.video import _episode_video_path, _trace_frame_file_path
from vla_lens.traces import TraceBundle, TraceDataset

try:  # Replay dependencies are optional outside the PI0.5/LIBERO environment.
    from vla_lens.pi05.replay import PI05LiberoReplayRenderer, read_sparse_image
except Exception:  # pragma: no cover - optional dependency boundary
    PI05LiberoReplayRenderer = None  # type: ignore[assignment]
    read_sparse_image = None  # type: ignore[assignment]


class DashboardState:
    """Shared dataset state for the FastAPI server.

    The legacy local server refreshes the dataset when files under the root
    change. Keeping the same behavior here preserves the dashboard workflow
    where analyses write artifacts while the browser is already open.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.index_manifest = validate_dataset_index(self.root)
        self.dataset = TraceDataset.open(self.root)
        self.dataset_signature = _dataset_signature(self.root)
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
        signature = _dataset_signature(self.root)
        with self.dataset_lock:
            if self.dataset_signature == signature:
                return
            self.index_manifest = validate_dataset_index(self.root)
            self.dataset = TraceDataset.open(self.root)
            self.dataset_signature = signature
            self.payload_cache.clear()
            self.replay_renderers.clear()

    def cached_payload(
        self,
        key: str,
        build: Callable[[TraceDataset], dict[str, Any]],
    ) -> dict[str, Any]:
        signature = self.dataset_signature or _dataset_signature(self.root)
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
        return self.dataset.bundle(_query_one(query, "trace_id"))

    def frame_bytes(self, query: dict[str, list[str]]) -> bytes:
        bundle = self.bundle_from_query(query)
        camera = _query_one(query, "camera")
        timestep = int(_query_one(query, "timestep"))
        source = query.get("source", ["auto"])[0]
        timestep = max(0, min(timestep, bundle.manifest.length - 1))
        if source in {"auto", "trace"}:
            frame_path = self.single_frame_file_path(bundle, camera=camera, timestep=timestep)
            if frame_path is not None:
                return frame_path.read_bytes()
        frame = self.read_single_frame(bundle, camera=camera, timestep=timestep, source=source)
        buffer = io.BytesIO()
        imageio.imwrite(buffer, np.asarray(frame), format="jpg", quality=90)
        return buffer.getvalue()

    def frame_file_path(self, query: dict[str, list[str]]) -> Path | None:
        bundle = self.bundle_from_query(query)
        camera = _query_one(query, "camera")
        timestep = int(_query_one(query, "timestep"))
        source = query.get("source", ["auto"])[0]
        timestep = max(0, min(timestep, bundle.manifest.length - 1))
        if source not in {"auto", "trace"}:
            return None
        return self.single_frame_file_path(bundle, camera=camera, timestep=timestep)

    def episode_video_bytes(self, query: dict[str, list[str]]) -> bytes:
        return self.episode_video_path(query).read_bytes()

    def episode_video_path(self, query: dict[str, list[str]]) -> Path:
        bundle = self.bundle_from_query(query)
        camera = query.get("camera", ["all"])[0]
        fps = int(query.get("fps", ["10"])[0])
        max_width = int(query.get("max_width", ["320"])[0])
        return _episode_video_path(
            bundle,
            camera=camera,
            fps=fps,
            max_width=max_width,
        )

    def single_frame_file_path(
        self,
        bundle: TraceBundle,
        *,
        camera: str,
        timestep: int,
    ) -> Path | None:
        return _trace_frame_file_path(bundle, camera=camera, timestep=timestep)

    def read_single_frame(
        self,
        bundle: TraceBundle,
        *,
        camera: str,
        timestep: int,
        source: str,
    ) -> np.ndarray:
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
        if source == "trace":
            frame_reader = getattr(bundle, "frame", None)
            if callable(frame_reader):
                return np.asarray(frame_reader(camera, timestep))
        # Legacy fallback for old array-backed traces. The intended path is a
        # single-frame file or bundle.frame() reader.
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
