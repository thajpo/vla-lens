"""Shared LeRobot dataset storage constants and low-level IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.capture.lerobot_v3 import (
    LEROBOT_ACTION,
    LEROBOT_EPISODE_INDEX,
    LEROBOT_EPISODES_DIR,
    LEROBOT_IMAGE_PREFIX,
    LEROBOT_OBSERVATION_STATE,
    LEROBOT_TASKS_JSONL_PATH,
    LEROBOT_TASKS_PARQUET_PATH,
    LEROBOT_TIMESTAMP,
    VLA_LENS_OVERLAY_DIR,
)
from vla_lens.capture.records import TraceRecord

DEFAULT_CHUNKS_SIZE = 1000


DEFAULT_DATA_FILE_SIZE_IN_MB = 100


DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200


DEFAULT_FPS = 30


LEROBOT_DATA_PATH_TEMPLATE = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"


LEROBOT_VIDEO_PATH_TEMPLATE = (
    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
)


LEROBOT_EPISODE_PATH_TEMPLATE = (
    "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
)


OVERLAY_SCHEMA_VERSION = "0.1.0"


OVERLAY_EPISODE_DIR = VLA_LENS_OVERLAY_DIR / "episodes"


OVERLAY_ROOT_ARRAY_NAMES = {
    LEROBOT_ACTION,
    LEROBOT_OBSERVATION_STATE,
}


OVERLAY_ROOT_ARRAY_PREFIXES = (
    LEROBOT_IMAGE_PREFIX,
    "frames.",
)


LEGACY_ACTION_ARRAY = "executed_actions"


LEGACY_FRAME_PREFIX = "frames."


def _read_episode_metadata(root: Path) -> pd.DataFrame:
    paths = sorted((root / LEROBOT_EPISODES_DIR).rglob("*.parquet"))
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    episodes = pd.concat(frames, ignore_index=True)
    return episodes.sort_values(LEROBOT_EPISODE_INDEX).reset_index(drop=True)


def _read_tasks(root: Path) -> pd.DataFrame:
    parquet_path = root / LEROBOT_TASKS_PARQUET_PATH
    if parquet_path.exists():
        tasks = pd.read_parquet(parquet_path)
        if "task" not in tasks.columns and tasks.index.name == "task":
            tasks = tasks.reset_index()
        return tasks.reset_index(drop=True)
    jsonl_path = root / LEROBOT_TASKS_JSONL_PATH
    if not jsonl_path.exists():
        return pd.DataFrame(columns=["task_index", "task"])
    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame.from_records(rows)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _chunk_file_index(episode_index: int) -> tuple[int, int]:
    return int(episode_index) // DEFAULT_CHUNKS_SIZE, int(episode_index) % DEFAULT_CHUNKS_SIZE


def _task_text(record: TraceRecord) -> str:
    task = str(record.manifest.metadata.get("task_name") or record.manifest.prompt or "").strip()
    return task or str(record.manifest.task_id)


def _task_for_index(tasks: pd.DataFrame, task_index: int) -> str:
    if tasks.empty or "task_index" not in tasks or "task" not in tasks:
        return str(task_index)
    matches = tasks.loc[tasks["task_index"].astype(int) == int(task_index)]
    if matches.empty:
        return str(task_index)
    return str(matches.iloc[-1]["task"])


def _feature_signature(features: Mapping[str, Any]) -> str:
    return json.dumps(_jsonable(features), sort_keys=True, separators=(",", ":"))


def _stack_non_null_column(values: pd.Series) -> np.ndarray:
    arrays = [np.asarray(value) for value in values if not _is_missing_cell(value)]
    if not arrays:
        return np.asarray([])
    return np.stack(arrays, axis=0)


def _is_missing_cell(value: Any) -> bool:
    if value is None:
        return True
    if np.isscalar(value):
        try:
            return bool(pd.isna(value))
        except (TypeError, ValueError):
            return False
    return False


def _pad_or_trim(array: np.ndarray, *, length: int) -> np.ndarray:
    value = np.asarray(array)
    if value.shape[0] == length:
        return value
    if value.shape[0] > length:
        return value[:length]
    if value.shape[0] == 0:
        raise ValueError("Cannot pad an empty time array")
    pad = np.repeat(value[-1:,...], length - value.shape[0], axis=0)
    return np.concatenate([value, pad], axis=0)


def _timestamps(timesteps: pd.DataFrame, *, length: int, fps: int) -> np.ndarray:
    if LEROBOT_TIMESTAMP in timesteps:
        return _pad_or_trim(
            np.asarray(timesteps[LEROBOT_TIMESTAMP], dtype=np.float32),
            length=length,
        )
    return (np.arange(length, dtype=np.float32) / float(max(1, fps))).astype(np.float32)


def _column_or_default(
    frame: pd.DataFrame,
    column: str,
    *,
    length: int,
    default: Any,
    dtype: Any,
) -> np.ndarray:
    if column in frame:
        return _pad_or_trim(np.asarray(frame[column], dtype=dtype), length=length)
    return np.full(length, default, dtype=dtype)


def _fps(record: TraceRecord) -> int:
    for source in (
        record.manifest.metadata,
        record.capture_request,
        record.capture_plan,
        record.capture_report,
    ):
        value = source.get("fps") if isinstance(source, Mapping) else None
        if value is not None:
            return max(1, int(value))
    return DEFAULT_FPS


def _action_dim_names(record: TraceRecord, action_dim: int) -> list[str]:
    metadata = record.manifest.metadata.get("action_space")
    if isinstance(metadata, Mapping):
        names = metadata.get("action_names")
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
            values = [str(item) for item in names]
            if len(values) == action_dim:
                return values
    return [f"action_{idx}" for idx in range(action_dim)]


def _stats_payload(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array, dtype=np.float32)
    if value.ndim == 1:
        value = value.reshape(-1, 1)
    flat = value.reshape(-1, value.shape[-1])
    return {
        "mean": np.nanmean(flat, axis=0).tolist(),
        "std": np.nanstd(flat, axis=0).tolist(),
        "min": np.nanmin(flat, axis=0).tolist(),
        "max": np.nanmax(flat, axis=0).tolist(),
    }


def _flatten_stats(payload: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            rows.update(_flatten_stats(value, full_key))
        else:
            rows[full_key] = json.dumps(_jsonable(value))
    return rows


def _format_lerobot_path(
    template: str,
    row: Mapping[str, Any],
    episode_index: int,
    *,
    prefix: str,
    video_key: str | None = None,
) -> Path:
    chunk_index = int(row.get(f"{prefix}/chunk_index", row.get("data/chunk_index", 0)) or 0)
    file_index = int(
        row.get(f"{prefix}/file_index", row.get("data/file_index", episode_index)) or 0
    )
    return Path(
        template.format(
            chunk_index=chunk_index,
            file_index=file_index,
            episode_chunk=chunk_index,
            episode_index=episode_index,
            video_key=video_key or "",
        )
    )


def _stack_column(values: pd.Series) -> np.ndarray:
    if values.empty:
        return np.asarray([])
    first = values.iloc[0]
    if isinstance(first, np.ndarray):
        return np.stack([np.asarray(value) for value in values], axis=0)
    if isinstance(first, (list, tuple)):
        return np.stack([np.asarray(value) for value in values], axis=0)
    return values.to_numpy()


def _array_record(
    *,
    name: str,
    relative_path: Path,
    array: np.ndarray,
    axes: Sequence[str],
    storage_format: str,
    compression: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "relative_path": str(relative_path),
        "storage_format": storage_format,
        "chunks": json.dumps([]),
        "compression": compression,
        "array_type": "episode",
        "shape": json.dumps([int(item) for item in np.asarray(array).shape]),
        "dtype": str(np.asarray(array).dtype),
        "axes": json.dumps(list(axes)),
        "metadata": json.dumps(dict(metadata), sort_keys=True),
    }


def _prefix_table_paths(table: pd.DataFrame, prefix: Path) -> pd.DataFrame:
    out = table.copy()
    for column in ("relative_path", "path"):
        if column in out:
            out[column] = [_prefix_path_value(value, prefix) for value in out[column]]
    return out


def _prefix_path_value(value: object, prefix: Path) -> str:
    text = str(value)
    if not text or Path(text).is_absolute():
        return text
    return str(prefix / text)


def _uint8_rgb(frame: np.ndarray) -> np.ndarray:
    value = np.asarray(frame)
    if value.dtype != np.uint8:
        if np.issubdtype(value.dtype, np.floating) and np.nanmax(value) <= 1.0:
            value = value * 255.0
        value = np.clip(value, 0, 255).astype(np.uint8)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=-1)
    if value.shape[-1] == 1:
        value = np.repeat(value, 3, axis=-1)
    return np.ascontiguousarray(value[..., :3])


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    return value


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))
