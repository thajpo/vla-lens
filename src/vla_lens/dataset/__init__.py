"""LeRobot v3 robot data plus VLA Lens overlay storage.

The normal VLA Lens app reads and writes the LeRobot v3 directory contract
directly.  It intentionally does not import ``lerobot`` because policy/runtime
dependencies belong to capture environments, not the dashboard/test stack.
"""

from __future__ import annotations

from vla_lens.dataset.bundle import LeRobotEpisodeBundle
from vla_lens.dataset.index import (
    DatasetIndexError,
    IndexBuildResult,
    build_dataset_index,
    index_manifest_path,
    index_root,
    validate_dataset_index,
)
from vla_lens.dataset.reader import is_lerobot_dataset_root, open_lerobot_dataset
from vla_lens.dataset.writer import write_lerobot_trace_record

__all__ = [
    "DatasetIndexError",
    "IndexBuildResult",
    "LeRobotEpisodeBundle",
    "build_dataset_index",
    "index_manifest_path",
    "index_root",
    "is_lerobot_dataset_root",
    "open_lerobot_dataset",
    "validate_dataset_index",
    "write_lerobot_trace_record",
]
