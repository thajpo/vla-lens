"""LeRobot dataset video read/write helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np

from vla_lens.capture.lerobot_v3 import (
    LEROBOT_EPISODE_INDEX,
    LEROBOT_IMAGE_PREFIX,
    LEROBOT_INFO_PATH,
)
from vla_lens.capture.records import TraceRecord
from vla_lens.dataset.common import (
    LEROBOT_VIDEO_PATH_TEMPLATE,
    _chunk_file_index,
    _format_lerobot_path,
    _read_episode_metadata,
    _read_json,
    _uint8_rgb,
)


def _write_videos(
    root: Path,
    record: TraceRecord,
    *,
    episode_index: int,
    frame_arrays: Mapping[str, np.ndarray],
    fps: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    chunk_index, file_index = _chunk_file_index(episode_index)
    duration = float(record.manifest.length) / float(max(1, fps))
    for camera, frames in frame_arrays.items():
        video_key = f"{LEROBOT_IMAGE_PREFIX}{camera}"
        path = root / LEROBOT_VIDEO_PATH_TEMPLATE.format(
            video_key=video_key,
            chunk_index=chunk_index,
            file_index=file_index,
        )
        _write_video(path, frames, fps=fps)
        metadata.update(
            {
                f"videos/{video_key}/chunk_index": int(chunk_index),
                f"videos/{video_key}/file_index": int(file_index),
                f"videos/{video_key}/from_timestamp": 0.0,
                f"videos/{video_key}/to_timestamp": duration,
            }
        )
    return metadata


def _remove_existing_episode_media(root: Path, episode_index: int) -> None:
    episodes = _read_episode_metadata(root)
    if episodes.empty or LEROBOT_EPISODE_INDEX not in episodes:
        return
    matches = episodes.loc[episodes[LEROBOT_EPISODE_INDEX].astype(int) == int(episode_index)]
    if matches.empty:
        return
    row = dict(matches.iloc[-1])
    info = _read_json(root / LEROBOT_INFO_PATH) if (root / LEROBOT_INFO_PATH).exists() else {}
    template = str(info.get("video_path") or LEROBOT_VIDEO_PATH_TEMPLATE)
    prefixes = sorted(
        key.removesuffix("/chunk_index")
        for key in row
        if str(key).startswith("videos/") and str(key).endswith("/chunk_index")
    )
    for prefix in prefixes:
        video_key = prefix.removeprefix("videos/")
        path = root / _format_lerobot_path(
            template,
            row,
            int(episode_index),
            prefix=prefix,
            video_key=video_key,
        )
        if path.exists():
            path.unlink()


def _write_video(path: Path, frames: np.ndarray, *, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    writer = imageio.get_writer(
        path,
        fps=max(1, int(fps)),
        codec="libx264",
        macro_block_size=16,
        ffmpeg_params=["-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p"],
    )
    try:
        for frame in frames:
            writer.append_data(_uint8_rgb(frame))
    finally:
        writer.close()


def _read_video_frames(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    reader = imageio.get_reader(path)
    try:
        return np.stack([_uint8_rgb(frame) for frame in reader], axis=0)
    finally:
        reader.close()
