"""Read LeRobot v3 dataset roots through the TraceDataset API."""

from __future__ import annotations

from pathlib import Path

from vla_lens.capture.lerobot_v3 import (
    LEROBOT_DATA_DIR,
    LEROBOT_INFO_PATH,
    VLA_LENS_OVERLAY_REFERENCES,
    validate_lerobot_v3_dataset,
)
from vla_lens.dataset.bundle import LeRobotEpisodeBundle
from vla_lens.dataset.common import (
    _read_episode_metadata,
    _read_json,
    _read_table,
    _read_tasks,
)
from vla_lens.dataset.overlay import _overlay_bundle_for_episode
from vla_lens.traces import TraceDataset


def is_lerobot_dataset_root(root: str | Path) -> bool:
    """Return whether ``root`` looks like a LeRobot v3 robot dataset root."""

    path = Path(root)
    return (path / LEROBOT_INFO_PATH).exists() and (path / LEROBOT_DATA_DIR).exists()


def open_lerobot_dataset(root: str | Path, *, trace_id_prefix: str | None = None) -> TraceDataset:
    """Open a LeRobot v3 root as the existing VLA Lens ``TraceDataset`` API."""

    dataset_root = Path(root)
    result = validate_lerobot_v3_dataset(dataset_root)
    if not result.valid:
        messages = "; ".join(issue.message for issue in result.errors)
        raise ValueError(f"Invalid LeRobot v3 dataset root: {messages}")

    info = _read_json(dataset_root / LEROBOT_INFO_PATH)
    tasks = _read_tasks(dataset_root)
    refs = _read_table(dataset_root / VLA_LENS_OVERLAY_REFERENCES)
    episodes = _read_episode_metadata(dataset_root)
    bundles = [
        LeRobotEpisodeBundle(
            dataset_root,
            episode_row=row,
            info=info,
            tasks=tasks,
            trace_id_prefix=trace_id_prefix,
            overlay_bundle=_overlay_bundle_for_episode(
                dataset_root,
                refs,
                int(row["episode_index"]),
            ),
        )
        for row in episodes.to_dict("records")
    ]
    return TraceDataset(dataset_root, bundles)  # type: ignore[arg-type]
