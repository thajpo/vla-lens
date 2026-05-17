"""Normalized capture records used to assemble sealed ``.vlatrace`` bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from vla_lens.artifacts import LensArtifact
from vla_lens.traces import ArraySpec, ModelSiteSpec, TraceBundle, TraceManifest
from vla_lens.validation import validate_trace_bundle


@dataclass(frozen=True, slots=True)
class PolicyCallRecord:
    """One model policy call aligned to an environment timestep."""

    call_index: int
    env_timestep: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "policy_call_index": int(self.call_index),
            "observation_timestep": int(self.env_timestep),
            "env_timestep_start": int(self.env_timestep),
            **dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class EnvironmentDescriptor:
    """Stable environment identity and replay/state capabilities."""

    env_family: str
    env_id: str
    simulator: str | None = None
    benchmark: str | None = None
    task_id: str | int | None = None
    layout_id: str | int | None = None
    seed: int | None = None
    replay_supported: bool = False
    state_available: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "env_family": self.env_family,
            "env_id": self.env_id,
            "replay_supported": self.replay_supported,
            "state_available": self.state_available,
        }
        for key, value in {
            "simulator": self.simulator,
            "benchmark": self.benchmark,
            "task_id": self.task_id,
            "layout_id": self.layout_id,
            "seed": self.seed,
        }.items():
            if value is not None:
                payload[key] = value
        payload.update(dict(self.metadata))
        return payload


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """Dataset-normalized episode evidence independent of model internals."""

    manifest: TraceManifest
    timesteps: pd.DataFrame
    episode_arrays: Mapping[str, ArraySpec] = field(default_factory=dict)
    environment: EnvironmentDescriptor | None = None
    tokens: pd.DataFrame | None = None
    generation_steps: pd.DataFrame | None = None
    streams: pd.DataFrame | None = None
    token_spaces: pd.DataFrame | None = None
    robot_state: pd.DataFrame | None = None
    scene_state: pd.DataFrame | None = None
    camera_state: pd.DataFrame | None = None
    evaluation: pd.DataFrame | None = None
    image_preprocessing: pd.DataFrame | None = None
    prompt_metadata: pd.DataFrame | None = None
    action_normalization: pd.DataFrame | None = None
    capture_request: Mapping[str, Any] = field(default_factory=dict)
    capture_plan: Mapping[str, Any] = field(default_factory=dict)
    capture_report: Mapping[str, Any] = field(default_factory=dict)
    policy_calls: Sequence[PolicyCallRecord] = ()
    artifacts: Sequence[LensArtifact] = ()


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Stable model identity and capability metadata for trace provenance."""

    model_family: str
    model_id: str
    checkpoint_sha: str | None = None
    supported_profiles: tuple[str, ...] = (
        "rollout",
        "features",
        "mechanistic_sampled",
        "mechanistic_all",
        "internals_sampled",
        "audit_full",
        "custom",
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        payload = {
            "model_family": self.model_family,
            "model_id": self.model_id,
            "supported_profiles": list(self.supported_profiles),
        }
        if self.checkpoint_sha:
            payload["checkpoint_sha"] = self.checkpoint_sha
        payload.update(dict(self.metadata))
        return payload


@dataclass(frozen=True, slots=True)
class ModelTraceRecord:
    """Model-normalized internals captured for one episode."""

    descriptor: ModelDescriptor
    model_arrays: Sequence[ModelSiteSpec] = ()
    episode_arrays: Mapping[str, ArraySpec] = field(default_factory=dict)
    tokens: pd.DataFrame | None = None
    generation_steps: pd.DataFrame | None = None
    streams: pd.DataFrame | None = None
    token_spaces: pd.DataFrame | None = None
    robot_state: pd.DataFrame | None = None
    scene_state: pd.DataFrame | None = None
    camera_state: pd.DataFrame | None = None
    evaluation: pd.DataFrame | None = None
    image_preprocessing: pd.DataFrame | None = None
    prompt_metadata: pd.DataFrame | None = None
    action_normalization: pd.DataFrame | None = None
    capture_request: Mapping[str, Any] = field(default_factory=dict)
    capture_plan: Mapping[str, Any] = field(default_factory=dict)
    capture_report: Mapping[str, Any] = field(default_factory=dict)
    policy_calls: Sequence[PolicyCallRecord] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TraceRecord:
    """Complete normalized record ready to write as one ``.vlatrace`` bundle."""

    manifest: TraceManifest
    timesteps: pd.DataFrame
    episode_arrays: Mapping[str, ArraySpec] = field(default_factory=dict)
    model_arrays: Sequence[ModelSiteSpec] = ()
    tokens: pd.DataFrame | None = None
    policy_calls: pd.DataFrame | None = None
    generation_steps: pd.DataFrame | None = None
    streams: pd.DataFrame | None = None
    token_spaces: pd.DataFrame | None = None
    robot_state: pd.DataFrame | None = None
    scene_state: pd.DataFrame | None = None
    camera_state: pd.DataFrame | None = None
    evaluation: pd.DataFrame | None = None
    image_preprocessing: pd.DataFrame | None = None
    prompt_metadata: pd.DataFrame | None = None
    action_normalization: pd.DataFrame | None = None
    capture_request: Mapping[str, Any] = field(default_factory=dict)
    capture_plan: Mapping[str, Any] = field(default_factory=dict)
    capture_report: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Sequence[LensArtifact] = ()


def merge_episode_and_model_trace(
    episode: EpisodeRecord,
    model_trace: ModelTraceRecord | None,
) -> TraceRecord:
    """Combine dataset evidence and model internals without leaking source paths."""
    if model_trace is None:
        manifest = _manifest_with_environment(episode.manifest, episode.environment)
        return TraceRecord(
            manifest=manifest,
            timesteps=episode.timesteps,
            episode_arrays=episode.episode_arrays,
            tokens=episode.tokens,
            policy_calls=_policy_call_frame(episode.policy_calls),
            generation_steps=episode.generation_steps,
            streams=episode.streams,
            token_spaces=episode.token_spaces,
            robot_state=episode.robot_state,
            scene_state=episode.scene_state,
            camera_state=episode.camera_state,
            evaluation=episode.evaluation,
            image_preprocessing=episode.image_preprocessing,
            prompt_metadata=episode.prompt_metadata,
            action_normalization=episode.action_normalization,
            capture_request=episode.capture_request,
            capture_plan=episode.capture_plan,
            capture_report=episode.capture_report,
            artifacts=episode.artifacts,
        )

    base_manifest = _manifest_with_environment(episode.manifest, episode.environment)
    metadata = dict(base_manifest.metadata or {})
    metadata.update(model_trace.descriptor.to_metadata())
    metadata.update(dict(model_trace.metadata))
    manifest = TraceManifest(
        trace_id=base_manifest.trace_id,
        episode_id=base_manifest.episode_id,
        task_id=base_manifest.task_id,
        prompt=base_manifest.prompt,
        model_id=model_trace.descriptor.model_id or base_manifest.model_id,
        env_id=base_manifest.env_id,
        robot_id=base_manifest.robot_id,
        outcome=base_manifest.outcome,
        length=base_manifest.length,
        schema_version=base_manifest.schema_version,
        metadata=metadata,
    )
    episode_arrays = {**dict(episode.episode_arrays), **dict(model_trace.episode_arrays)}
    tokens = model_trace.tokens if model_trace.tokens is not None else episode.tokens
    capture_request = {**dict(episode.capture_request), **dict(model_trace.capture_request)}
    capture_plan = {**dict(episode.capture_plan), **dict(model_trace.capture_plan)}
    capture_report = {**dict(episode.capture_report), **dict(model_trace.capture_report)}
    return TraceRecord(
        manifest=manifest,
        timesteps=episode.timesteps,
        episode_arrays=episode_arrays,
        model_arrays=tuple(model_trace.model_arrays),
        tokens=tokens,
        policy_calls=_policy_call_frame((*episode.policy_calls, *model_trace.policy_calls)),
        generation_steps=model_trace.generation_steps
        if model_trace.generation_steps is not None
        else episode.generation_steps,
        streams=model_trace.streams if model_trace.streams is not None else episode.streams,
        token_spaces=model_trace.token_spaces
        if model_trace.token_spaces is not None
        else episode.token_spaces,
        robot_state=model_trace.robot_state
        if model_trace.robot_state is not None
        else episode.robot_state,
        scene_state=model_trace.scene_state
        if model_trace.scene_state is not None
        else episode.scene_state,
        camera_state=model_trace.camera_state
        if model_trace.camera_state is not None
        else episode.camera_state,
        evaluation=model_trace.evaluation
        if model_trace.evaluation is not None
        else episode.evaluation,
        image_preprocessing=model_trace.image_preprocessing
        if model_trace.image_preprocessing is not None
        else episode.image_preprocessing,
        prompt_metadata=model_trace.prompt_metadata
        if model_trace.prompt_metadata is not None
        else episode.prompt_metadata,
        action_normalization=model_trace.action_normalization
        if model_trace.action_normalization is not None
        else episode.action_normalization,
        capture_request=capture_request,
        capture_plan=capture_plan,
        capture_report=capture_report,
        artifacts=episode.artifacts,
    )


def _manifest_with_environment(
    manifest: TraceManifest,
    environment: EnvironmentDescriptor | None,
) -> TraceManifest:
    if environment is None:
        return manifest
    metadata = dict(manifest.metadata or {})
    metadata["environment"] = environment.to_metadata()
    return TraceManifest(
        trace_id=manifest.trace_id,
        episode_id=manifest.episode_id,
        task_id=manifest.task_id,
        prompt=manifest.prompt,
        model_id=manifest.model_id,
        env_id=manifest.env_id,
        robot_id=manifest.robot_id,
        outcome=manifest.outcome,
        length=manifest.length,
        schema_version=manifest.schema_version,
        metadata=metadata,
    )


def write_trace_record(
    record: TraceRecord,
    path: str | Path,
    *,
    overwrite: bool = False,
    validate: bool = True,
) -> TraceBundle:
    """Write one normalized record as a sealed trace bundle."""
    bundle = TraceBundle.create(
        path,
        manifest=record.manifest,
        timesteps=record.timesteps,
        episode_arrays=record.episode_arrays,
        model_arrays=record.model_arrays,
        policy_calls=record.policy_calls,
        generation_steps=record.generation_steps,
        streams=record.streams,
        token_spaces=record.token_spaces,
        tokens=record.tokens,
        robot_state=record.robot_state,
        scene_state=record.scene_state,
        camera_state=record.camera_state,
        evaluation=record.evaluation,
        image_preprocessing=record.image_preprocessing,
        prompt_metadata=record.prompt_metadata,
        action_normalization=record.action_normalization,
        capture_request=record.capture_request,
        capture_plan=record.capture_plan,
        capture_report=record.capture_report,
        artifacts=record.artifacts,
        overwrite=overwrite,
    )
    if validate:
        result = validate_trace_bundle(bundle)
        if not result.valid:
            messages = "; ".join(str(error) for error in result.errors)
            raise ValueError(f"Trace failed validation: {messages}")
    return bundle


def _policy_call_frame(calls: Sequence[PolicyCallRecord]) -> pd.DataFrame:
    if not calls:
        return pd.DataFrame()
    rows = [call.to_row() for call in calls]
    frame = pd.DataFrame.from_records(rows).drop_duplicates(
        subset=["policy_call_index"],
        keep="last",
    )
    return frame.sort_values("policy_call_index").reset_index(drop=True)


__all__ = [
    "EpisodeRecord",
    "EnvironmentDescriptor",
    "ModelDescriptor",
    "ModelTraceRecord",
    "PolicyCallRecord",
    "TraceRecord",
    "merge_episode_and_model_trace",
    "write_trace_record",
]
