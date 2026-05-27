"""Dataset and episode-bundle primitives.

The canonical public storage shape is a LeRobot v3 robot-data root plus an
optional ``vla_lens/`` overlay. This package provides the common query surface
used by selectors, probes, artifacts, and the dashboard.
"""

from vla_lens.traces.bundle import TraceBundle
from vla_lens.traces.dataset import TraceDataset, TraceDatasetStats
from vla_lens.traces.types import ActivationSpec, ArraySpec, ModelSiteSpec, TraceManifest

__all__ = [
    "ActivationSpec",
    "ArraySpec",
    "ModelSiteSpec",
    "TraceBundle",
    "TraceDataset",
    "TraceDatasetStats",
    "TraceManifest",
]
