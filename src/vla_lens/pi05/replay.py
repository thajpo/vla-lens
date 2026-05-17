"""Replay rendering for PI0.5 LIBERO traces.

Replay uses only metadata and action arrays stored in the ``.vlatrace`` bundle.
Recorded RGB frames, when available, are also read from the bundle rather than
from the original capture directory.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from vla_lens.traces import TraceBundle

DEFAULT_CAMERAS = "agentview_image,robot0_eye_in_hand_image"


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    benchmark: str
    task_id: int
    layout_id: int
    seed: int
    horizon: int
    obs_size: int = 256
    camera_name: str = DEFAULT_CAMERAS
    control_mode: str = "relative"


class PI05LiberoReplayRenderer:
    """Render camera frames by replaying saved PI0.5 actions in LIBERO."""

    def __init__(self, bundle: TraceBundle):
        self.bundle = bundle
        self.config = replay_config_from_bundle(bundle)
        self._lock = threading.RLock()
        self._base_env: Any | None = None
        self._vec_env: Any | None = None
        self._raw_obs: dict[str, Any] | None = None
        self._last_rendered_timestep = -1
        self._done = False

    def close(self) -> None:
        with self._lock:
            if self._vec_env is not None:
                self._vec_env.close()
            self._vec_env = None
            self._base_env = None
            self._raw_obs = None
            self._last_rendered_timestep = -1
            self._done = False

    def render(self, *, camera: str, timestep: int) -> np.ndarray:
        """Return the post-action observation frame for ``timestep``."""
        with self._lock:
            timestep = max(0, min(int(timestep), self.bundle.manifest.length - 1))
            if self._raw_obs is None or timestep <= self._last_rendered_timestep:
                self._reset()
            actions = self.bundle.actions(mmap=True)
            while self._last_rendered_timestep < timestep and not self._done:
                next_step = self._last_rendered_timestep + 1
                self._step(np.asarray(actions[next_step], dtype=np.float32))
                self._last_rendered_timestep = next_step
            if self._raw_obs is None:
                raise RuntimeError("Replay did not produce an observation")
            pixels = self._raw_obs.get("pixels", {})
            if camera not in pixels:
                raise KeyError(f"Camera {camera!r} not available; cameras={sorted(pixels)}")
            return np.asarray(pixels[camera])

    def _reset(self) -> None:
        if self._vec_env is not None:
            self._vec_env.close()
        self._vec_env, self._base_env = _make_base_env(self.config)
        self._base_env.episode_index = self.config.layout_id
        self._base_env.init_state_id = self.config.layout_id
        self._raw_obs, _ = self._base_env.reset(seed=self.config.seed)
        self._last_rendered_timestep = -1
        self._done = False

    def _step(self, action: np.ndarray) -> None:
        if self._base_env is None:
            self._reset()
        self._raw_obs, _reward, terminated, truncated, _info = self._base_env.step(action)
        self._done = bool(terminated or truncated)


def replay_config_from_bundle(bundle: TraceBundle) -> ReplayConfig:
    metadata = bundle.manifest.metadata
    benchmark = str(metadata.get("benchmark") or bundle.manifest.env_id)
    if not benchmark:
        raise ValueError(f"Trace {bundle.manifest.trace_id} is missing LIBERO benchmark metadata")
    task_id = _task_id(bundle)
    layout_id = int(metadata.get("layout_episode_index") or 0)
    seed = int(metadata.get("env_seed") or metadata.get("policy_seed") or 0)
    obs_size = int(metadata.get("obs_size") or _infer_obs_size(bundle) or 256)
    return ReplayConfig(
        benchmark=benchmark,
        task_id=task_id,
        layout_id=layout_id,
        seed=seed,
        horizon=int(bundle.manifest.length),
        obs_size=obs_size,
    )


def sparse_image_path(bundle: TraceBundle, *, camera: str, timestep: int) -> Path | None:
    matches = bundle.array_index.loc[bundle.array_index["name"].astype(str) == f"frames.{camera}"]
    if matches.empty:
        return None
    path = bundle.path / str(matches.iloc[0]["relative_path"]) / f"{int(timestep):06d}.jpg"
    return path if path.exists() else None


def read_sparse_image(bundle: TraceBundle, *, camera: str, timestep: int) -> np.ndarray | None:
    path = sparse_image_path(bundle, camera=camera, timestep=timestep)
    if path is None:
        return None
    return np.asarray(imageio.imread(path))


def _make_base_env(config: ReplayConfig) -> tuple[Any, Any]:
    from lerobot.envs.factory import make_env, make_env_config

    env_cfg = make_env_config(
        "libero",
        task=config.benchmark,
        task_ids=[config.task_id],
        episode_length=config.horizon,
        observation_height=config.obs_size,
        observation_width=config.obs_size,
        camera_name=config.camera_name,
        control_mode=config.control_mode,
    )
    env_dict = make_env(env_cfg, n_envs=1, use_async_envs=False)
    vec_env = env_dict[config.benchmark][config.task_id]
    return vec_env, vec_env.envs[0]


def _task_id(bundle: TraceBundle) -> int:
    metadata = bundle.manifest.metadata
    task_id = metadata.get("task_id")
    if task_id is not None:
        return int(task_id)
    raise ValueError(f"Trace {bundle.manifest.trace_id} is missing task_id metadata")


def _infer_obs_size(bundle: TraceBundle) -> int | None:
    for camera in bundle.cameras():
        try:
            frames = bundle.frames(camera, mmap=True)
        except KeyError:
            continue
        if frames.ndim >= 3:
            return int(frames.shape[1])
    return None


__all__ = [
    "PI05LiberoReplayRenderer",
    "ReplayConfig",
    "read_sparse_image",
    "replay_config_from_bundle",
    "sparse_image_path",
]
