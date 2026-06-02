from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import vla_lens.server.state as server_state
from vla_lens import create_synthetic_trace_dataset
from vla_lens.server.state import DashboardState


@pytest.mark.parametrize("source", ["auto", "trace"])
def test_dashboard_frame_file_path_uses_direct_trace_frame_for_auto_and_trace(
    tmp_path,
    source,
):
    bundle = _FrameDirectoryBundle(tmp_path)
    state = _state_for_bundle(bundle)

    path = state.frame_file_path(_frame_query(source=source))

    assert path == tmp_path / "frames" / "main" / "000000.jpg"


@pytest.mark.parametrize("source", ["auto", "trace"])
def test_dashboard_frame_bytes_returns_direct_trace_frame_without_array_load(
    tmp_path,
    source,
):
    bundle = _FrameDirectoryBundle(tmp_path)
    state = _state_for_bundle(bundle)

    payload = state.frame_bytes(_frame_query(source=source))

    assert payload == _FRAME_BYTES


@pytest.mark.parametrize("source", ["auto", "trace"])
def test_dashboard_frame_bytes_uses_single_frame_reader_before_full_frame_array(
    tmp_path,
    monkeypatch,
    source,
):
    monkeypatch.setattr(server_state, "read_sparse_image", None)
    monkeypatch.setattr(server_state, "PI05LiberoReplayRenderer", None)
    dataset = create_synthetic_trace_dataset(tmp_path / "demo", num_episodes=1, timesteps=2)
    state = DashboardState(dataset.root)
    bundle = state.dataset.bundle("synthetic_000")
    calls: list[tuple[str, int]] = []

    def frame(camera: str, timestep: int) -> np.ndarray:
        calls.append((camera, timestep))
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def frames(camera: str, *, mmap: bool = False) -> np.ndarray:
        raise AssertionError("frames() should not be called for single-frame reads")

    monkeypatch.setattr(bundle, "frame", frame)
    monkeypatch.setattr(bundle, "frames", frames)

    payload = state.frame_bytes(_frame_query(source=source, trace_id="synthetic_000"))

    assert payload.startswith(b"\xff\xd8")
    assert calls == [("main", 0)]


def test_dashboard_frame_bytes_keeps_legacy_full_frame_array_fallback(monkeypatch):
    monkeypatch.setattr(server_state, "read_sparse_image", None)
    monkeypatch.setattr(server_state, "PI05LiberoReplayRenderer", None)
    bundle = _LegacyFramesBundle()
    state = _state_for_bundle(bundle)

    payload = state.frame_bytes(_frame_query(source="auto"))

    assert payload.startswith(b"\xff\xd8")
    assert bundle.frames_called == 1


_FRAME_BYTES = b"\xff\xd8direct-frame\xff\xd9"


def _frame_query(*, source: str, trace_id: str = "trace-a") -> dict[str, list[str]]:
    return {
        "trace_id": [trace_id],
        "camera": ["main"],
        "timestep": ["0"],
        "source": [source],
    }


def _state_for_bundle(bundle: object) -> DashboardState:
    state = object.__new__(DashboardState)
    state.dataset = _SingleBundleDataset(bundle)
    return state


class _SingleBundleDataset:
    def __init__(self, bundle: object):
        self._bundle = bundle

    def bundle(self, trace_id: str) -> object:
        assert trace_id == "trace-a"
        return self._bundle


class _FrameDirectoryBundle:
    def __init__(self, root):
        self.path = root
        self.manifest = SimpleNamespace(length=2)
        frame_dir = root / "frames" / "main"
        frame_dir.mkdir(parents=True)
        (frame_dir / "000000.jpg").write_bytes(_FRAME_BYTES)
        self.array_index = pd.DataFrame(
            [
                {
                    "name": "frames.main",
                    "relative_path": "frames/main",
                }
            ]
        )

    def frame(self, camera: str, timestep: int) -> np.ndarray:
        raise AssertionError("frame() should not be called when a direct file exists")

    def frames(self, camera: str, *, mmap: bool = False) -> np.ndarray:
        raise AssertionError("frames() should not be called when a direct file exists")


class _LegacyFramesBundle:
    path = ""
    manifest = SimpleNamespace(length=2)
    array_index = pd.DataFrame()

    def __init__(self):
        self.frames_called = 0

    def frames(self, camera: str, *, mmap: bool = False) -> np.ndarray:
        self.frames_called += 1
        return np.zeros((2, 8, 8, 3), dtype=np.uint8)
