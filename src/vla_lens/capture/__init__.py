"""Capture adapter contracts for building sealed ``.vlatrace`` bundles."""

from __future__ import annotations

from vla_lens.capture.adapters import (
    COMMON_ROBOTICS_DATASETS,
    COMMON_ROBOTICS_ENVIRONMENTS,
    ActionGeneratorSpec,
    CaptureSite,
    CaptureSpec,
    DatasetDescriptor,
    DatasetEpisodeAdapter,
    EnvironmentAdapter,
    EnvironmentSupport,
    ModelCaptureAdapter,
    RoboticsDatasetSupport,
    TransformerSegmentSpec,
)
from vla_lens.capture.records import (
    EnvironmentDescriptor,
    EpisodeRecord,
    ModelDescriptor,
    ModelTraceRecord,
    PolicyCallRecord,
    TraceRecord,
    merge_episode_and_model_trace,
    write_trace_record,
)

__all__ = [
    "COMMON_ROBOTICS_DATASETS",
    "COMMON_ROBOTICS_ENVIRONMENTS",
    "ActionGeneratorSpec",
    "CaptureSite",
    "CaptureSpec",
    "DatasetDescriptor",
    "DatasetEpisodeAdapter",
    "EpisodeRecord",
    "EnvironmentAdapter",
    "EnvironmentDescriptor",
    "EnvironmentSupport",
    "ModelCaptureAdapter",
    "ModelDescriptor",
    "ModelTraceRecord",
    "PolicyCallRecord",
    "RoboticsDatasetSupport",
    "TransformerSegmentSpec",
    "TraceRecord",
    "merge_episode_and_model_trace",
    "write_trace_record",
]
