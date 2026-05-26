"""Compatibility facade for the LeRobot-backed dataset API.

New storage internals live under :mod:`vla_lens.dataset`; this module keeps the
short public import path used by scripts, tests, and notebooks.
"""

from __future__ import annotations

from vla_lens.dataset import (
    LeRobotEpisodeBundle,
    is_lerobot_dataset_root,
    open_lerobot_dataset,
    write_lerobot_trace_record,
)

__all__ = [
    "LeRobotEpisodeBundle",
    "is_lerobot_dataset_root",
    "open_lerobot_dataset",
    "write_lerobot_trace_record",
]
