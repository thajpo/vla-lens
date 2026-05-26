"""Tiny in-memory adapters used to prove the generic capture boundary.

These adapters intentionally avoid PI0.5, Torch, LeRobot runtime imports, and
simulators. They create normal capture records that the existing LeRobot writer
can persist as robot data plus a ``vla_lens/`` overlay.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from vla_lens.capture.adapters import (
    ActionGeneratorSpec,
    CaptureSite,
    CaptureSpec,
    DatasetDescriptor,
    TransformerSegmentSpec,
)
from vla_lens.capture.records import (
    EnvironmentDescriptor,
    EpisodeRecord,
    ModelDescriptor,
    ModelTraceRecord,
    PolicyCallRecord,
    merge_episode_and_model_trace,
)
from vla_lens.traces import ArraySpec, ModelSiteSpec, TraceDataset, TraceManifest


@dataclass(frozen=True, slots=True)
class FakeDatasetEpisodeAdapter:
    """A minimal robot-dataset adapter with deterministic in-memory episodes."""

    dataset_id: str = "fake-adapter-dataset"
    episode_count: int = 2
    length: int = 4
    action_dim: int = 2
    state_dim: int = 3
    seed: int = 7

    @property
    def descriptor(self) -> DatasetDescriptor:
        return DatasetDescriptor(
            dataset_family="fake_robot",
            dataset_id=self.dataset_id,
            storage_format="in_memory",
            robot_id="fake-2dof",
            env_id="fake-tabletop",
            metadata={"purpose": "adapter_compliance"},
        )

    def episode_ids(self) -> Sequence[str]:
        return tuple(f"fake_{index:03d}" for index in range(self.episode_count))

    def load_episode(self, episode_id: str) -> EpisodeRecord:
        index = self.episode_ids().index(episode_id)
        rng = np.random.default_rng(self.seed + index)
        timesteps = pd.DataFrame(
            {
                "timestep": np.arange(self.length, dtype=np.int32),
                "reward": np.linspace(0.0, 1.0 if index % 2 == 0 else 0.2, self.length),
                "done": [False] * (self.length - 1) + [True],
                "policy_call_index": np.arange(self.length, dtype=np.int32) // 2,
                "horizon_index": np.arange(self.length, dtype=np.int32) % 2,
            }
        )
        action = np.stack(
            [
                np.linspace(0.0, 0.3 + 0.1 * index, self.length),
                np.linspace(0.2, -0.1, self.length),
            ],
            axis=1,
        ).astype(np.float32)
        state = np.stack(
            [
                np.linspace(0.0, 1.0, self.length),
                np.linspace(float(index), float(index) + 0.5, self.length),
                np.ones(self.length),
            ],
            axis=1,
        ).astype(np.float32)
        frames = _fake_frames(self.length, index=index, rng=rng)
        manifest = TraceManifest(
            trace_id=episode_id,
            episode_id=episode_id,
            task_id="fake_pick",
            prompt="move the fake cube",
            model_id="",
            env_id="fake-tabletop",
            robot_id="fake-2dof",
            outcome="success" if index % 2 == 0 else "failure",
            length=self.length,
            metadata={
                "dataset_id": self.dataset_id,
                "split": "train" if index == 0 else "test",
                "adapter_family": self.descriptor.dataset_family,
                "seed": self.seed + index,
            },
        )
        return EpisodeRecord(
            manifest=manifest,
            timesteps=timesteps,
            episode_arrays={
                "action": ArraySpec(action, ["timestep", "action_dim"]),
                "observation.state": ArraySpec(state, ["timestep", "state_dim"]),
                "observation.images.main": ArraySpec(
                    frames,
                    ["timestep", "height", "width", "rgb"],
                ),
            },
            environment=FakeEnvironmentAdapter().descriptor,
        )


@dataclass(frozen=True, slots=True)
class FakeEnvironmentAdapter:
    """A minimal environment adapter descriptor."""

    env_id: str = "fake-tabletop"

    @property
    def descriptor(self) -> EnvironmentDescriptor:
        return EnvironmentDescriptor(
            env_family="fake_env",
            env_id=self.env_id,
            simulator=None,
            benchmark="fake-benchmark",
            task_id="fake_pick",
            replay_supported=False,
            state_available=True,
            metadata={"purpose": "adapter_compliance"},
        )

    def metadata(self) -> dict[str, object]:
        return self.descriptor.to_metadata()


@dataclass(frozen=True, slots=True)
class FakeModelCaptureAdapter:
    """A minimal model adapter that emits non-PI0.5 model-site overlays."""

    model_id: str = "fake-vla-tiny"
    hidden_size: int = 6
    token_count: int = 3
    horizon: int = 2
    action_dim: int = 2
    generation_steps: int = 3

    @property
    def descriptor(self) -> ModelDescriptor:
        return ModelDescriptor(
            model_family="fake_vla",
            model_id=self.model_id,
            supported_profiles=("features", "mechanistic_sampled"),
            metadata={"adapter_family": "fake_model"},
        )

    @property
    def capture_spec(self) -> CaptureSpec:
        return CaptureSpec(
            model_family="fake_vla",
            architecture_class="fake_transformer_policy",
            streams=("fake.state", "fake.action"),
            transformer_segments=(
                TransformerSegmentSpec(
                    name="fake.backbone",
                    layers=1,
                    token_spaces=("fake.state_tokens",),
                    hidden_size=self.hidden_size,
                ),
            ),
            action_generator=ActionGeneratorSpec(
                kind="direct_chunk",
                horizon=self.horizon,
                action_dim=self.action_dim,
                generation_steps=self.generation_steps,
            ),
            sites=(
                CaptureSite(
                    site_id="fake.backbone.layers.0.hidden",
                    family="activation",
                    role="hidden_state",
                    segment="backbone",
                    axes=("policy_call", "token", "feature"),
                    expensive=False,
                    profiles=("features", "mechanistic_sampled"),
                ),
                CaptureSite(
                    site_id="fake.action_head.output",
                    family="action",
                    role="action_output",
                    segment="action_head",
                    axes=("policy_call", "horizon", "action_dim"),
                    expensive=False,
                    profiles=("mechanistic_sampled",),
                ),
            ),
        )

    def capture_episode(self, episode: EpisodeRecord) -> ModelTraceRecord:
        call_count = max(1, int(np.ceil(episode.manifest.length / self.horizon)))
        rng = np.random.default_rng(abs(hash((self.model_id, episode.manifest.trace_id))) % 2**32)
        hidden = rng.normal(
            0.0,
            0.03,
            size=(call_count, self.token_count, self.hidden_size),
        ).astype(np.float32)
        hidden[..., 0] += np.linspace(0.0, 1.0, call_count, dtype=np.float32)[:, None]
        action_chunks = np.zeros((call_count, self.horizon, self.action_dim), dtype=np.float32)
        for call_index in range(call_count):
            action_chunks[call_index, :, 0] = 0.1 * (call_index + 1)
            action_chunks[call_index, :, 1] = -0.05 * (call_index + 1)
        generation_actions = np.stack(
            [
                action_chunks * ((step + 1) / self.generation_steps)
                for step in range(self.generation_steps)
            ],
            axis=1,
        ).astype(np.float32)
        policy_calls = tuple(
            PolicyCallRecord(
                call_index=call_index,
                env_timestep=min(call_index * self.horizon, episode.manifest.length - 1),
                metadata={
                    "env_timestep_end": min(
                        call_index * self.horizon + self.horizon - 1,
                        episode.manifest.length - 1,
                    ),
                    "model_id": self.descriptor.model_id,
                    "model_family": self.descriptor.model_family,
                    "model_call_kind": "policy_action_chunk",
                    "action_generator_kind": self.capture_spec.action_generator.kind,
                    "action_horizon": self.horizon,
                    "action_dim": self.action_dim,
                },
            )
            for call_index in range(call_count)
        )
        return ModelTraceRecord(
            descriptor=self.descriptor,
            model_arrays=(
                ModelSiteSpec(
                    name="fake.backbone.layers.0.hidden",
                    array=hidden,
                    axes=("policy_call", "token", "feature"),
                    module="fake.backbone.layers.0",
                    layer=0,
                    tensor_type="hidden_tokens",
                    token_kind="state",
                    family="activation",
                    role="hidden_state",
                    segment="backbone",
                    token_space_id="fake.state_tokens",
                    default_view=True,
                ),
                ModelSiteSpec(
                    name="fake.action_head.output",
                    array=action_chunks,
                    axes=("policy_call", "horizon", "action_dim"),
                    module="fake.action_head",
                    tensor_type="action",
                    token_kind="action",
                    family="action",
                    role="action_output",
                    segment="action_head",
                    default_view=False,
                ),
            ),
            episode_arrays={
                "action_chunks": ArraySpec(
                    action_chunks,
                    ["policy_call", "horizon", "action_dim"],
                ),
                "generation_actions": ArraySpec(
                    generation_actions,
                    ["policy_call", "generation_step", "horizon", "action_dim"],
                ),
            },
            streams=_fake_streams(),
            token_spaces=_fake_token_spaces(self.token_count),
            tokens=_fake_tokens(self.token_count),
            policy_calls=policy_calls,
            capture_request={"requested_profile": "mechanistic_sampled"},
            capture_plan={"actual_profile": "mechanistic_sampled", "complete": True},
            capture_report={
                "adapter": "fake_vla",
                "actual_profile": "mechanistic_sampled",
                "complete": True,
                "missing_model_sites": [],
            },
            metadata={"capture_adapter": "fake_model"},
        )


def write_fake_adapter_lerobot_dataset(
    root: str | Path,
    *,
    episode_count: int = 2,
    length: int = 4,
    overwrite: bool = False,
) -> TraceDataset:
    """Write a tiny generic-adapter dataset as LeRobot v3 plus VLA Lens overlay."""

    dataset_root = Path(root)
    if overwrite and dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_adapter = FakeDatasetEpisodeAdapter(episode_count=episode_count, length=length)
    model_adapter = FakeModelCaptureAdapter()

    from vla_lens.dataset import write_lerobot_trace_record

    for episode_id in dataset_adapter.episode_ids():
        episode = dataset_adapter.load_episode(episode_id)
        model_trace = model_adapter.capture_episode(episode)
        record = merge_episode_and_model_trace(episode, model_trace)
        write_lerobot_trace_record(record, dataset_root, overwrite=overwrite)
    return TraceDataset.open(dataset_root)


def _fake_frames(length: int, *, index: int, rng: np.random.Generator) -> np.ndarray:
    frames = np.zeros((length, 16, 16, 3), dtype=np.uint8)
    for timestep in range(length):
        frames[timestep, :, :, 0] = 48 + index * 24
        frames[timestep, :, :, 1] = 64 + timestep * 20
        frames[timestep, :, :, 2] = 120
        row = min(14, 2 + timestep)
        col = min(14, 3 + index + timestep)
        frames[timestep, row : row + 2, col : col + 2] = np.asarray([230, 80, 60], dtype=np.uint8)
    noise = rng.integers(0, 8, size=frames.shape, dtype=np.uint8)
    return np.clip(frames + noise, 0, 255).astype(np.uint8)


def _fake_streams() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "stream_id": "fake.state",
                "name": "fake state tokens",
                "modality": "state",
                "camera_id": "",
                "description": "Tiny non-PI0.5 state token stream.",
            },
            {
                "stream_id": "fake.action",
                "name": "fake action tokens",
                "modality": "action",
                "camera_id": "",
                "description": "Tiny non-PI0.5 action token stream.",
            },
        ]
    )


def _fake_token_spaces(token_count: int) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "token_space_id": "fake.state_tokens",
                "policy_call_index": -1,
                "segment": "state_tokens",
                "stream_id": "fake.state",
                "token_count": int(token_count),
            },
            {
                "token_space_id": "fake.action_tokens",
                "policy_call_index": -1,
                "segment": "action_tokens",
                "stream_id": "fake.action",
                "token_count": 2,
            },
        ]
    )


def _fake_tokens(token_count: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for token_index in range(token_count):
        records.append(
            {
                "token_space_id": "fake.state_tokens",
                "token_index": token_index,
                "modality": "state",
                "segment": "state_tokens",
                "token_kind": "state",
                "token_type": "low_dim_state",
                "stream_id": "fake.state",
                "is_padding": False,
                "attention_mask": True,
            }
        )
    for token_index in range(2):
        records.append(
            {
                "token_space_id": "fake.action_tokens",
                "token_index": token_index,
                "modality": "action",
                "segment": "action_tokens",
                "token_kind": "action",
                "token_type": "continuous_action",
                "stream_id": "fake.action",
                "action_horizon_index": token_index,
                "is_padding": False,
                "attention_mask": True,
            }
        )
    return pd.DataFrame.from_records(records)


__all__ = [
    "FakeDatasetEpisodeAdapter",
    "FakeEnvironmentAdapter",
    "FakeModelCaptureAdapter",
    "write_fake_adapter_lerobot_dataset",
]
