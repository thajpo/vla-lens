"""Replay rendering for PI0.5 LIBERO traces.

Replay uses only metadata and action arrays stored in the VLA Lens overlay.
Recorded RGB frames, when available, are also read from the dataset rather than
from the original capture directory.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np

from vla_lens.pi05.scene_mutation import (
    apply_scene_mutation,
    scene_mutation_from_metadata,
)
from vla_lens.traces import TraceBundle

DEFAULT_CAMERAS = "agentview_image,robot0_eye_in_hand_image"


@dataclass(frozen=True, slots=True)
class ReplayConfig:
    benchmark: str
    task_id: int
    layout_id: int | None
    seed: int
    horizon: int
    obs_size: int = 256
    camera_name: str = DEFAULT_CAMERAS
    control_mode: str = "relative"
    scene_mutation: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PolicyCallReplayInputs:
    """Runtime-free inputs required to reproduce one captured PI0.5 policy call."""

    config: ReplayConfig
    trace_id: str
    policy_call_index: int
    observation_timestep: int
    policy_call: Mapping[str, Any]
    stored_action_chunk: np.ndarray
    initial_noise: np.ndarray
    initial_noise_ref: str
    initial_noise_exactness: str

    def summary(self) -> dict[str, Any]:
        """Return JSON-safe provenance without embedding the replay tensors."""
        return {
            "trace_id": self.trace_id,
            "policy_call_index": self.policy_call_index,
            "observation_timestep": self.observation_timestep,
            "environment": {
                "benchmark": self.config.benchmark,
                "task_id": self.config.task_id,
                "layout_id": self.config.layout_id,
                "seed": self.config.seed,
                "obs_size": self.config.obs_size,
                "camera_name": self.config.camera_name,
                "control_mode": self.config.control_mode,
                "scene_mutation": dict(self.config.scene_mutation),
            },
            "stored_action_chunk": {
                "shape": list(self.stored_action_chunk.shape),
                "dtype": str(self.stored_action_chunk.dtype),
            },
            "initial_noise": {
                "ref": self.initial_noise_ref,
                "exactness": self.initial_noise_exactness,
                "shape": list(self.initial_noise.shape),
                "dtype": str(self.initial_noise.dtype),
            },
        }


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
        if self.config.layout_id is not None:
            self._base_env.episode_index = self.config.layout_id
            self._base_env.init_state_id = self.config.layout_id
        self._raw_obs, _ = self._base_env.reset(seed=self.config.seed)
        mutation = scene_mutation_from_metadata(self.config.scene_mutation)
        if mutation is not None:
            self._raw_obs, _report = apply_scene_mutation(self._base_env, mutation)
        self._last_rendered_timestep = -1
        self._done = False

    def _step(self, action: np.ndarray) -> None:
        if self._base_env is None:
            self._reset()
        self._raw_obs, _reward, terminated, truncated, _info = self._base_env.step(action)
        self._done = bool(terminated or truncated)


def replay_config_from_bundle(bundle: TraceBundle) -> ReplayConfig:
    metadata = bundle.manifest.metadata
    environment = _mapping(metadata.get("environment"))
    benchmark = str(
        _first_present(
            environment.get("benchmark"),
            metadata.get("benchmark"),
            bundle.manifest.env_id,
        )
        or ""
    )
    if not benchmark:
        raise ValueError(f"Trace {bundle.manifest.trace_id} is missing LIBERO benchmark metadata")
    task_id = _task_id(bundle)
    layout_value = _first_present(
        environment.get("layout_id"),
        metadata.get("layout_episode_index"),
        metadata.get("layout_id"),
    )
    layout_id = int(layout_value) if layout_value is not None else None
    seed = int(
        _first_present(
            environment.get("seed"),
            metadata.get("env_seed"),
            metadata.get("seed"),
            metadata.get("policy_seed"),
            0,
        )
    )
    obs_size = int(
        _first_present(
            environment.get("obs_size"),
            metadata.get("obs_size"),
            _infer_obs_size(bundle),
            256,
        )
    )
    return ReplayConfig(
        benchmark=benchmark,
        task_id=task_id,
        layout_id=layout_id,
        seed=seed,
        horizon=int(bundle.manifest.length),
        obs_size=obs_size,
        scene_mutation=dict(_mapping(environment.get("scene_mutation"))),
    )


def policy_call_replay_inputs(
    bundle: TraceBundle,
    policy_call_index: int,
) -> PolicyCallReplayInputs:
    """Resolve stored action and initial flow noise without loading PI0.5 dependencies."""

    call_index = int(policy_call_index)
    policy_call = _policy_call_record(bundle, call_index)
    observation_timestep = _required_int(
        _first_present(
            policy_call.get("observation_timestep"),
            policy_call.get("env_timestep_start"),
        ),
        field=f"policy call {call_index} observation timestep",
    )
    stored_actions = bundle.action_chunks(mmap=True)
    if call_index >= len(stored_actions):
        raise IndexError(
            f"Trace {bundle.manifest.trace_id} policy call {call_index} has no stored action chunk"
        )
    initial_noise, initial_noise_ref, exactness = _initial_noise(bundle, call_index)
    return PolicyCallReplayInputs(
        config=replay_config_from_bundle(bundle),
        trace_id=bundle.manifest.trace_id,
        policy_call_index=call_index,
        observation_timestep=observation_timestep,
        policy_call=policy_call,
        stored_action_chunk=np.asarray(stored_actions[call_index]),
        initial_noise=initial_noise,
        initial_noise_ref=initial_noise_ref,
        initial_noise_exactness=exactness,
    )


def sparse_image_path(bundle: TraceBundle, *, camera: str, timestep: int) -> Path | None:
    names = {f"frames.{camera}", f"observation.images.{camera}"}
    matches = bundle.array_index.loc[bundle.array_index["name"].astype(str).isin(names)]
    if matches.empty:
        return None
    frame_root = bundle.path / str(matches.iloc[0]["relative_path"])
    if not frame_root.is_dir():
        return None
    path = frame_root / f"{int(timestep):06d}.jpg"
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
    environment = _mapping(metadata.get("environment"))
    task_id = _first_present(
        environment.get("task_id"),
        bundle.manifest.task_id,
        metadata.get("task_id"),
    )
    if task_id is not None and str(task_id).strip():
        return int(task_id)
    raise ValueError(f"Trace {bundle.manifest.trace_id} is missing task_id metadata")


def _policy_call_record(bundle: TraceBundle, policy_call_index: int) -> dict[str, Any]:
    if bundle.policy_calls.empty:
        raise KeyError(f"Trace {bundle.manifest.trace_id} has no policy calls")
    for record in bundle.policy_calls.to_dict("records"):
        value = record.get("policy_call_index")
        if value is not None and int(value) == policy_call_index:
            return dict(record)
    raise KeyError(f"Trace {bundle.manifest.trace_id} has no policy call {policy_call_index}")


def _initial_noise(bundle: TraceBundle, policy_call_index: int) -> tuple[np.ndarray, str, str]:
    try:
        exact_noise = bundle.array("flow_initial_noise", mmap=True)
    except KeyError:
        exact_noise = None
    if exact_noise is not None:
        if policy_call_index >= len(exact_noise):
            raise IndexError(
                f"Trace {bundle.manifest.trace_id} policy call {policy_call_index} "
                "has no exact flow initial noise"
            )
        return (
            np.asarray(exact_noise[policy_call_index]),
            f"flow_initial_noise[{policy_call_index}]",
            "exact",
        )

    generation_actions = bundle.generation_actions(mmap=True)
    if policy_call_index >= len(generation_actions) or generation_actions.shape[1] == 0:
        raise IndexError(
            f"Trace {bundle.manifest.trace_id} policy call {policy_call_index} "
            "has no generation-step-zero initial noise fallback"
        )
    initial_noise = np.asarray(generation_actions[policy_call_index, 0])
    exactness = (
        "quantized"
        if initial_noise.dtype.itemsize < np.dtype(np.float32).itemsize
        else "exact"
    )
    return (
        initial_noise,
        f"generation_actions[{policy_call_index},0]",
        exactness,
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(*values: Any) -> Any:
    return next((value for value in values if value is not None and value != ""), None)


def _required_int(value: Any, *, field: str) -> int:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Missing {field}")
    return int(value)


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
    "PolicyCallReplayInputs",
    "ReplayConfig",
    "policy_call_replay_inputs",
    "read_sparse_image",
    "replay_config_from_bundle",
    "sparse_image_path",
]
