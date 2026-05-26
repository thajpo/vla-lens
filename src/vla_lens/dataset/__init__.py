"""LeRobot v3 robot data plus VLA Lens overlay storage.

The normal VLA Lens app reads and writes the LeRobot v3 directory contract
directly.  It intentionally does not import ``lerobot`` because policy/runtime
dependencies belong to capture environments, not the dashboard/test stack.
"""

from __future__ import annotations

from vla_lens.dataset.bundle import LeRobotEpisodeBundle
from vla_lens.dataset.reader import is_lerobot_dataset_root, open_lerobot_dataset
from vla_lens.dataset.writer import write_lerobot_trace_record

__all__ = [
    "LeRobotEpisodeBundle",
    "is_lerobot_dataset_root",
    "open_lerobot_dataset",
    "write_lerobot_trace_record",
]
