"""Video dashboard server helpers."""


from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from vla_lens.artifacts import LensArtifact
from vla_lens.server_common import (
    _cache_part,
)
from vla_lens.traces import TraceBundle

_EPISODE_VIDEO_CACHE_LOCK = threading.RLock()

def _episode_video_path(
    bundle: TraceBundle,
    *,
    camera: str,
    fps: int,
    max_width: int,
) -> Path:
    with _EPISODE_VIDEO_CACHE_LOCK:
        return _episode_video_path_locked(
            bundle,
            camera=camera,
            fps=fps,
            max_width=max_width,
        )

def _episode_video_path_locked(
    bundle: TraceBundle,
    *,
    camera: str,
    fps: int,
    max_width: int,
) -> Path:
    cameras = bundle.cameras()
    selected_cameras = cameras if camera == "all" else [camera]
    missing = [name for name in selected_cameras if name not in cameras]
    if missing:
        raise KeyError(f"Unknown camera(s): {missing}; available={cameras}")

    fps = max(1, min(int(fps), 30))
    max_width = max(64, min(int(max_width), 960))
    frame_timesteps = _video_frame_timesteps(bundle)
    video_dir = bundle.path / "artifacts" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    cache_name = (
        f"{_cache_part(bundle.manifest.trace_id)}__{_cache_part(camera)}"
        f"__full_episode__fps{fps}__w{max_width}.mp4"
    )
    video_path = video_dir / cache_name
    input_paths = _episode_frame_array_paths(bundle, selected_cameras)
    if video_path.exists() and not _video_cache_stale(video_path, input_paths):
        _ensure_episode_video_artifact(
            bundle,
            video_path=video_path,
            camera=camera,
            fps=fps,
            max_width=max_width,
            timesteps=frame_timesteps,
        )
        return video_path

    tmp_path = video_path.with_suffix(".tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        _write_episode_video(
            bundle,
            selected_cameras,
            tmp_path,
            fps=fps,
            max_width=max_width,
            timesteps=frame_timesteps,
        )
        tmp_path.replace(video_path)
        _ensure_episode_video_artifact(
            bundle,
            video_path=video_path,
            camera=camera,
            fps=fps,
            max_width=max_width,
            timesteps=frame_timesteps,
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return video_path

def _write_episode_video(
    bundle: TraceBundle,
    cameras: list[str],
    output_path: Path,
    *,
    fps: int,
    max_width: int,
    timesteps: list[int],
) -> None:
    if not cameras:
        raise ValueError("No cameras are available for this episode.")

    frame_arrays = {camera: bundle.frames(camera, mmap=True) for camera in cameras}
    frame_count = min(
        [int(bundle.manifest.length), *(int(frames.shape[0]) for frames in frame_arrays.values())]
    )
    if frame_count <= 0:
        raise ValueError(f"Episode {bundle.manifest.trace_id} has no frames.")
    selected_timesteps = [timestep for timestep in timesteps if 0 <= timestep < frame_count]
    if not selected_timesteps:
        selected_timesteps = list(range(frame_count))

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        macro_block_size=16,
        ffmpeg_params=[
            "-preset",
            "veryfast",
            "-crf",
            "34",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ],
    )
    try:
        for timestep in selected_timesteps:
            images = [
                _prepare_video_frame(frame_arrays[camera][timestep], max_width=max_width)
                for camera in cameras
            ]
            writer.append_data(np.asarray(_tile_video_frames(images)))
    finally:
        writer.close()

def _video_frame_timesteps(bundle: TraceBundle) -> list[int]:
    calls = bundle.policy_calls
    if not calls.empty:
        column = "observation_timestep" if "observation_timestep" in calls else "env_timestep_start"
        if column in calls:
            return [int(value) for value in calls[column].dropna().tolist()]
    table = bundle.timesteps
    if not table.empty and "timestep" in table:
        return [int(value) for value in table["timestep"].tolist()]
    return list(range(int(bundle.manifest.length)))

def _ensure_episode_video_artifact(
    bundle: TraceBundle,
    *,
    video_path: Path,
    camera: str,
    fps: int,
    max_width: int,
    timesteps: list[int],
) -> None:
    artifact_id = (
        f"episode_video-{_cache_part(bundle.manifest.trace_id)}-{_cache_part(camera)}"
        f"-full-episode-fps{fps}-w{max_width}"
    )
    relative_path = video_path.relative_to(bundle.path)
    artifact = LensArtifact(
        artifact_id=artifact_id,
        artifact_type="episode_video",
        name=f"Full episode video ({camera})",
        group_id="episode_videos",
        scope="bundle",
        selector={
            "trace_id": bundle.manifest.trace_id,
            "camera": camera,
            "timesteps": timesteps,
            "source": "trace_frames",
        },
        method={
            "codec": "libx264",
            "crf": 34,
            "preset": "veryfast",
            "fps": fps,
            "max_width": max_width,
            "layout": "stitched_cameras" if camera == "all" else "single_camera",
        },
        metrics={
            "frame_count": len(timesteps),
            "file_size_bytes": video_path.stat().st_size if video_path.exists() else 0,
        },
        display={
            "kind": "episode_video",
            "media_type": "video/mp4",
            "relative_path": str(relative_path),
        },
        tags=("video", "episode", "full_episode"),
        source_trace_ids=(bundle.manifest.trace_id,),
        path=str(Path("artifacts") / artifact_id / "artifact.json"),
    )
    bundle.save_artifact(artifact)

def _prepare_video_frame(frame: np.ndarray, *, max_width: int) -> Any:
    from PIL import Image

    array = np.asarray(frame)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        image = Image.fromarray(array, mode="L").convert("RGB")
    else:
        image = Image.fromarray(array[..., :3]).convert("RGB")

    scale = min(1.0, float(max_width) / max(1, image.width))
    width = max(2, int(round(image.width * scale)))
    height = max(2, int(round(image.height * scale)))
    width -= width % 2
    height -= height % 2
    if (width, height) != image.size:
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return image

def _tile_video_frames(images: list[Any]) -> Any:
    from PIL import Image

    if len(images) == 1:
        return _pad_video_frame(images[0])

    gap = 2
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    x = 0
    for image in images:
        y = (height - image.height) // 2
        canvas.paste(image, (x, y))
        x += image.width + gap
    return _pad_video_frame(canvas)

def _pad_video_frame(image: Any, *, multiple: int = 16) -> Any:
    from PIL import Image

    width = ((image.width + multiple - 1) // multiple) * multiple
    height = ((image.height + multiple - 1) // multiple) * multiple
    if (width, height) == image.size:
        return image
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(image, (0, 0))
    return canvas

def _episode_frame_array_paths(bundle: TraceBundle, cameras: list[str]) -> list[Path]:
    if bundle.array_index.empty:
        return []
    paths: list[Path] = []
    for camera in cameras:
        names = {f"frames.{camera}", f"observation.images.{camera}"}
        matches = bundle.array_index.loc[bundle.array_index["name"].astype(str).isin(names)]
        if matches.empty:
            continue
        paths.append(bundle.path / str(matches.iloc[0]["relative_path"]))
    return paths

def _trace_frame_file_path(bundle: TraceBundle, *, camera: str, timestep: int) -> Path | None:
    if bundle.array_index.empty:
        return None
    names = {f"frames.{camera}", f"observation.images.{camera}"}
    matches = bundle.array_index.loc[bundle.array_index["name"].astype(str).isin(names)]
    if matches.empty:
        return None
    frame_dir = bundle.path / str(matches.iloc[0]["relative_path"])
    if not frame_dir.is_dir():
        return None
    path = frame_dir / f"{timestep:06d}.jpg"
    return path if path.exists() else None

def _video_cache_stale(video_path: Path, input_paths: list[Path]) -> bool:
    if not input_paths:
        return True
    video_mtime = video_path.stat().st_mtime
    return any(path.exists() and path.stat().st_mtime > video_mtime for path in input_paths)
