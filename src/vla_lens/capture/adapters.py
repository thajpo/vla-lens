"""Adapter protocols for robotics datasets and model capture."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from vla_lens.capture.records import (
    EnvironmentDescriptor,
    EpisodeRecord,
    ModelDescriptor,
    ModelTraceRecord,
)


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    """Stable identity for a robotics episode source."""

    dataset_family: str
    dataset_id: str
    storage_format: str
    root: Path | None = None
    robot_id: str | None = None
    env_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EnvironmentSupport:
    """Names common robot environment families and their adapter expectations."""

    family: str
    simulator: str | None
    replay_supported: bool
    state_available: bool
    notes: str


@dataclass(frozen=True, slots=True)
class RoboticsDatasetSupport:
    """Names common robotics dataset families without coupling imports to them."""

    family: str
    common_formats: tuple[str, ...]
    replay_supported: bool
    notes: str


@dataclass(frozen=True, slots=True)
class CaptureSite:
    """One hookable model site declared by an adapter."""

    site_id: str
    family: str
    role: str
    segment: str
    axes: tuple[str, ...]
    expensive: bool = True
    profiles: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TransformerSegmentSpec:
    """Named transformer block group with its token streams and dimensions."""

    name: str
    layers: int
    token_spaces: tuple[str, ...]
    heads: int | None = None
    hidden_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActionGeneratorSpec:
    """How an action policy emits chunks or iterative generation states."""

    kind: str
    horizon: int | None = None
    action_dim: int | None = None
    generation_steps: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CaptureSpec:
    """Adapter-declared capture contract for a supported model family."""

    model_family: str
    architecture_class: str
    streams: tuple[str, ...]
    transformer_segments: tuple[TransformerSegmentSpec, ...] = ()
    action_generator: ActionGeneratorSpec | None = None
    sites: tuple[CaptureSite, ...] = ()
    profiles: tuple[str, ...] = (
        "rollout",
        "features",
        "mechanistic_sampled",
        "mechanistic_all",
        "internals_sampled",
        "audit_sampled",
        "audit_full",
        "custom",
    )
    metadata: dict[str, Any] = field(default_factory=dict)


COMMON_ROBOTICS_ENVIRONMENTS: tuple[EnvironmentSupport, ...] = (
    EnvironmentSupport(
        family="libero",
        simulator="robosuite",
        replay_supported=True,
        state_available=True,
        notes="Language-conditioned robosuite manipulation tasks with reset/layout metadata.",
    ),
    EnvironmentSupport(
        family="robosuite",
        simulator="mujoco",
        replay_supported=True,
        state_available=True,
        notes="MuJoCo simulated manipulation environments with object and robot state.",
    ),
    EnvironmentSupport(
        family="gymnasium",
        simulator=None,
        replay_supported=False,
        state_available=False,
        notes="Generic API family; capabilities depend on the concrete environment.",
    ),
    EnvironmentSupport(
        family="real_robot",
        simulator=None,
        replay_supported=False,
        state_available=False,
        notes="Logged real robot episodes usually require timestamp and calibration adapters.",
    ),
)


COMMON_ROBOTICS_DATASETS: tuple[RoboticsDatasetSupport, ...] = (
    RoboticsDatasetSupport(
        family="libero",
        common_formats=("npz", "hdf5", "custom_capture"),
        replay_supported=True,
        notes=(
            "Simulation episodes with language, actions, object state, and optional camera frames."
        ),
    ),
    RoboticsDatasetSupport(
        family="rlds",
        common_formats=("tensorflow_dataset", "rlds"),
        replay_supported=False,
        notes="Common real/sim robot dataset container; schemas vary by dataset.",
    ),
    RoboticsDatasetSupport(
        family="robomimic",
        common_formats=("hdf5",),
        replay_supported=False,
        notes="HDF5 demonstrations with observations, actions, rewards, and metadata.",
    ),
    RoboticsDatasetSupport(
        family="lerobot",
        common_formats=("parquet", "safetensors", "video"),
        replay_supported=False,
        notes="LeRobot dataset layout with tabular episode metadata and external media.",
    ),
)


class DatasetEpisodeAdapter(Protocol):
    """Normalize dataset episodes into trace-ready evidence."""

    descriptor: DatasetDescriptor

    def episode_ids(self) -> Sequence[str]: ...

    def load_episode(self, episode_id: str) -> EpisodeRecord: ...


class EnvironmentAdapter(Protocol):
    """Describe or replay the environment that produced an episode."""

    descriptor: EnvironmentDescriptor

    def metadata(self) -> dict[str, Any]: ...


class ModelCaptureAdapter(Protocol):
    """Normalize captured model internals for one episode."""

    descriptor: ModelDescriptor
    capture_spec: CaptureSpec

    def capture_episode(self, episode: EpisodeRecord) -> ModelTraceRecord: ...


__all__ = [
    "COMMON_ROBOTICS_ENVIRONMENTS",
    "ActionGeneratorSpec",
    "CaptureSite",
    "CaptureSpec",
    "COMMON_ROBOTICS_DATASETS",
    "DatasetDescriptor",
    "DatasetEpisodeAdapter",
    "EnvironmentAdapter",
    "EnvironmentSupport",
    "ModelCaptureAdapter",
    "RoboticsDatasetSupport",
    "TransformerSegmentSpec",
]
