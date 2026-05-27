"""Write TraceRecord captures into LeRobot v3 dataset roots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from vla_lens.capture.lerobot_v3 import (
    LEROBOT_ACTION,
    LEROBOT_DATA_DIR,
    LEROBOT_EPISODE_INDEX,
    LEROBOT_EPISODES_DIR,
    LEROBOT_FRAME_INDEX,
    LEROBOT_IMAGE_PREFIX,
    LEROBOT_INFO_PATH,
    LEROBOT_OBSERVATION_STATE,
    LEROBOT_STATS_PATH,
    LEROBOT_TASK_INDEX,
    LEROBOT_TASKS_JSONL_PATH,
    LEROBOT_TASKS_PARQUET_PATH,
    LEROBOT_TIMESTAMP,
    LEROBOT_V3_VERSION,
    VLA_LENS_OVERLAY_REFERENCES,
)
from vla_lens.capture.records import TraceRecord
from vla_lens.dataset.bundle import LeRobotEpisodeBundle
from vla_lens.dataset.common import (
    DEFAULT_CHUNKS_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    LEROBOT_DATA_PATH_TEMPLATE,
    LEROBOT_EPISODE_PATH_TEMPLATE,
    LEROBOT_VIDEO_PATH_TEMPLATE,
    TRACE_ACTION_ARRAY,
    TRACE_FRAME_PREFIX,
    _action_dim_names,
    _chunk_file_index,
    _column_or_default,
    _feature_signature,
    _flatten_stats,
    _fps,
    _jsonable,
    _pad_or_trim,
    _read_episode_metadata,
    _read_json,
    _read_table,
    _read_tasks,
    _stack_non_null_column,
    _stats_payload,
    _task_text,
    _timestamps,
    _write_json,
    _write_table,
)
from vla_lens.dataset.media import _remove_existing_episode_media, _write_videos
from vla_lens.dataset.overlay import _write_overlay_bundle, _write_overlay_root


def write_lerobot_trace_record(
    record: TraceRecord,
    root: str | Path,
    *,
    overwrite: bool = False,
) -> "LeRobotEpisodeBundle":
    """Write one captured episode as LeRobot v3 robot data plus VLA Lens overlay."""

    dataset_root = Path(root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    length = int(record.manifest.length)
    refs = _read_table(dataset_root / VLA_LENS_OVERLAY_REFERENCES)
    existing_ref = _matching_ref(refs, record.manifest.trace_id)
    if existing_ref is not None and not overwrite:
        raise FileExistsError(f"Episode already exists in LeRobot root: {record.manifest.trace_id}")

    if existing_ref is not None:
        episode_index = int(existing_ref["episode_index"])
        _reject_length_changing_overwrite(dataset_root, episode_index, length)
        dataset_from_index = _episode_dataset_from_index(dataset_root, episode_index)
    else:
        episode_index = _next_episode_index(dataset_root)
        dataset_from_index = _next_dataset_frame_index(dataset_root)

    chunk_index, file_index = _chunk_file_index(episode_index)
    task_index = _task_index_for_record(dataset_root, record)
    data_path = dataset_root / LEROBOT_DATA_PATH_TEMPLATE.format(
        chunk_index=chunk_index,
        file_index=file_index,
    )
    episode_path = dataset_root / LEROBOT_EPISODE_PATH_TEMPLATE.format(
        chunk_index=chunk_index,
        file_index=file_index,
    )

    action = _required_episode_array(record, TRACE_ACTION_ARRAY, LEROBOT_ACTION, length=length)
    observation_state = _observation_state_array(record, length=length)
    frame_arrays = _frame_arrays(record, length=length)
    features = _features_for_record(
        record,
        action=action,
        observation_state=observation_state,
        frame_arrays=frame_arrays,
    )
    _assert_lerobot_feature_schema(dataset_root, features, episode_index=episode_index)
    _write_info(dataset_root, record, features=features)
    _write_tasks(dataset_root, task_index=task_index, task=_task_text(record))
    _write_robot_data(
        data_path,
        record,
        episode_index=episode_index,
        task_index=task_index,
        dataset_from_index=dataset_from_index,
        action=action,
        observation_state=observation_state,
    )
    if existing_ref is not None:
        _remove_existing_episode_media(dataset_root, episode_index)
    video_metadata = _write_videos(
        dataset_root,
        record,
        episode_index=episode_index,
        frame_arrays=frame_arrays,
        fps=_fps(record),
    )
    _write_episode_metadata(
        episode_path,
        record,
        episode_index=episode_index,
        task_index=task_index,
        dataset_from_index=dataset_from_index,
        video_metadata=video_metadata,
    )
    _write_stats(dataset_root)
    overlay_bundle = _write_overlay_bundle(
        dataset_root,
        record,
        episode_index=episode_index,
        data_path=data_path.relative_to(dataset_root),
        overwrite=overwrite,
    )
    _write_overlay_root(
        dataset_root,
        record,
        episode_index=episode_index,
        overlay_bundle=overlay_bundle,
    )
    _refresh_info_counts(dataset_root)

    return LeRobotEpisodeBundle(
        dataset_root,
        episode_row=_episode_row_for_index(dataset_root, episode_index),
        info=_read_json(dataset_root / LEROBOT_INFO_PATH),
        tasks=_read_tasks(dataset_root),
        overlay_bundle=overlay_bundle,
    )


def _matching_ref(refs: pd.DataFrame, trace_id: str) -> pd.Series | None:
    if refs.empty or "trace_id" not in refs:
        return None
    matches = refs.loc[refs["trace_id"].astype(str) == str(trace_id)]
    if matches.empty:
        return None
    return matches.iloc[-1]


def _next_episode_index(root: Path) -> int:
    episodes = (
        _read_episode_metadata(root)
        if (root / LEROBOT_EPISODES_DIR).exists()
        else pd.DataFrame()
    )
    if episodes.empty or LEROBOT_EPISODE_INDEX not in episodes:
        return 0
    return int(episodes[LEROBOT_EPISODE_INDEX].max()) + 1


def _next_dataset_frame_index(root: Path) -> int:
    episodes = (
        _read_episode_metadata(root)
        if (root / LEROBOT_EPISODES_DIR).exists()
        else pd.DataFrame()
    )
    if episodes.empty or "dataset_to_index" not in episodes:
        return 0
    return int(pd.to_numeric(episodes["dataset_to_index"], errors="coerce").fillna(0).max())


def _episode_dataset_from_index(root: Path, episode_index: int) -> int:
    episodes = _read_episode_metadata(root)
    matches = episodes.loc[episodes[LEROBOT_EPISODE_INDEX].astype(int) == int(episode_index)]
    if matches.empty:
        return _next_dataset_frame_index(root)
    return int(matches.iloc[-1].get("dataset_from_index") or 0)


def _reject_length_changing_overwrite(root: Path, episode_index: int, length: int) -> None:
    episodes = _read_episode_metadata(root)
    if episodes.empty or LEROBOT_EPISODE_INDEX not in episodes:
        return
    matches = episodes.loc[episodes[LEROBOT_EPISODE_INDEX].astype(int) == int(episode_index)]
    if matches.empty:
        return
    row = matches.iloc[-1]
    existing_length = row.get("length", row.get("num_frames"))
    if existing_length is None or pd.isna(existing_length):
        return
    if int(existing_length) != int(length):
        raise ValueError(
            "Cannot overwrite an existing LeRobot episode with a different length: "
            f"episode_index={episode_index} existing_length={int(existing_length)} "
            f"new_length={int(length)}"
        )


def _episode_row_for_index(root: Path, episode_index: int) -> dict[str, Any]:
    episodes = _read_episode_metadata(root)
    matches = episodes.loc[episodes[LEROBOT_EPISODE_INDEX].astype(int) == int(episode_index)]
    if matches.empty:
        raise KeyError(f"Unknown episode_index {episode_index}")
    return dict(matches.iloc[-1])


def _task_index_for_record(root: Path, record: TraceRecord) -> int:
    tasks = _read_tasks(root)
    task = _task_text(record)
    if not tasks.empty and "task" in tasks and "task_index" in tasks:
        matches = tasks.loc[tasks["task"].astype(str) == task]
        if not matches.empty:
            return int(matches.iloc[-1]["task_index"])
        return int(pd.to_numeric(tasks["task_index"], errors="coerce").fillna(-1).max()) + 1
    return 0


def _write_tasks(root: Path, *, task_index: int, task: str) -> None:
    tasks = _read_tasks(root)
    row = pd.DataFrame.from_records([{"task_index": int(task_index), "task": task}])
    if tasks.empty:
        tasks = row
    else:
        tasks = pd.concat([tasks, row], ignore_index=True)
        tasks = tasks.drop_duplicates(subset=["task"], keep="last")
    tasks = tasks.sort_values("task_index").reset_index(drop=True)
    parquet_path = root / LEROBOT_TASKS_PARQUET_PATH
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    tasks.set_index("task").to_parquet(parquet_path)
    jsonl_path = root / LEROBOT_TASKS_JSONL_PATH
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text(
        "".join(json.dumps(_jsonable(row)) + "\n" for row in tasks.to_dict("records")),
        encoding="utf-8",
    )


def _write_info(root: Path, record: TraceRecord, *, features: Mapping[str, Any]) -> None:
    existing = _read_json(root / LEROBOT_INFO_PATH) if (root / LEROBOT_INFO_PATH).exists() else {}
    payload = {
        "codebase_version": LEROBOT_V3_VERSION,
        "fps": _fps(record),
        "features": dict(features),
        "total_episodes": int(existing.get("total_episodes") or 0),
        "total_frames": int(existing.get("total_frames") or 0),
        "total_tasks": int(existing.get("total_tasks") or 0),
        "chunks_size": int(existing.get("chunks_size") or DEFAULT_CHUNKS_SIZE),
        "data_files_size_in_mb": int(
            existing.get("data_files_size_in_mb") or DEFAULT_DATA_FILE_SIZE_IN_MB
        ),
        "video_files_size_in_mb": int(
            existing.get("video_files_size_in_mb") or DEFAULT_VIDEO_FILE_SIZE_IN_MB
        ),
        "data_path": str(existing.get("data_path") or LEROBOT_DATA_PATH_TEMPLATE),
        "video_path": str(existing.get("video_path") or LEROBOT_VIDEO_PATH_TEMPLATE),
        "robot_type": str(
            existing.get("robot_type")
            or record.manifest.robot_id
            or record.manifest.metadata.get("robot_type")
            or ""
        ),
        "splits": dict(existing.get("splits") or {}),
    }
    _write_json(root / LEROBOT_INFO_PATH, payload)


def _assert_lerobot_feature_schema(
    root: Path,
    features: Mapping[str, Any],
    *,
    episode_index: int,
) -> None:
    info_path = root / LEROBOT_INFO_PATH
    if not info_path.exists():
        return
    existing = _read_json(info_path)
    existing_features = existing.get("features")
    if not isinstance(existing_features, Mapping):
        return
    episodes = (
        _read_episode_metadata(root)
        if (root / LEROBOT_EPISODES_DIR).exists()
        else pd.DataFrame()
    )
    if episodes.empty or LEROBOT_EPISODE_INDEX not in episodes:
        return
    other_episodes = episodes.loc[episodes[LEROBOT_EPISODE_INDEX].astype(int) != int(episode_index)]
    if other_episodes.empty:
        return
    if _feature_signature(existing_features) == _feature_signature(features):
        return
    raise ValueError(
        "Cannot mix LeRobot robot feature schemas in one dataset root. "
        "Write captures with different robot observation/action fields to separate roots."
    )


def _refresh_info_counts(root: Path) -> None:
    info = _read_json(root / LEROBOT_INFO_PATH)
    episodes = _read_episode_metadata(root)
    tasks = _read_tasks(root)
    info["total_episodes"] = int(len(episodes))
    if not episodes.empty and "dataset_to_index" in episodes:
        info["total_frames"] = int(pd.to_numeric(episodes["dataset_to_index"]).max())
    else:
        info["total_frames"] = 0
    info["total_tasks"] = int(len(tasks))
    info["splits"] = {"train": f"0:{len(episodes)}"}
    _write_json(root / LEROBOT_INFO_PATH, info)


def _features_for_record(
    record: TraceRecord,
    *,
    action: np.ndarray,
    observation_state: np.ndarray | None,
    frame_arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "index": {"dtype": "int64", "shape": [1], "names": None},
        LEROBOT_EPISODE_INDEX: {"dtype": "int64", "shape": [1], "names": None},
        LEROBOT_FRAME_INDEX: {"dtype": "int64", "shape": [1], "names": None},
        LEROBOT_TIMESTAMP: {"dtype": "float32", "shape": [1], "names": None},
        LEROBOT_TASK_INDEX: {"dtype": "int64", "shape": [1], "names": None},
        LEROBOT_ACTION: {
            "dtype": str(action.dtype),
            "shape": [int(action.shape[-1])],
            "names": _action_dim_names(record, int(action.shape[-1])),
        },
        "reward": {"dtype": "float32", "shape": [1], "names": None},
        "done": {"dtype": "bool", "shape": [1], "names": None},
        "truncated": {"dtype": "bool", "shape": [1], "names": None},
        "is_first": {"dtype": "bool", "shape": [1], "names": None},
        "is_last": {"dtype": "bool", "shape": [1], "names": None},
        "is_terminal": {"dtype": "bool", "shape": [1], "names": None},
    }
    if observation_state is not None:
        features[LEROBOT_OBSERVATION_STATE] = {
            "dtype": str(observation_state.dtype),
            "shape": [int(observation_state.shape[-1])],
            "names": [f"state_{idx}" for idx in range(int(observation_state.shape[-1]))],
        }
    for camera, frames in frame_arrays.items():
        height, width, channels = (int(frames.shape[1]), int(frames.shape[2]), int(frames.shape[3]))
        features[f"{LEROBOT_IMAGE_PREFIX}{camera}"] = {
            "dtype": "video",
            "shape": [channels, height, width],
            "names": ["channel", "height", "width"],
        }
    return features


def _write_stats(root: Path) -> None:
    data = _read_all_robot_data(root)
    stats: dict[str, Any] = {}
    for name in (LEROBOT_ACTION, LEROBOT_OBSERVATION_STATE):
        if name in data:
            array = _stack_non_null_column(data[name])
            if array.size:
                stats[name] = _stats_payload(array)
    _write_json(root / LEROBOT_STATS_PATH, stats)


def _read_all_robot_data(root: Path) -> pd.DataFrame:
    paths = sorted((root / LEROBOT_DATA_DIR).rglob("*.parquet"))
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _write_robot_data(
    path: Path,
    record: TraceRecord,
    *,
    episode_index: int,
    task_index: int,
    dataset_from_index: int,
    action: np.ndarray,
    observation_state: np.ndarray | None,
) -> None:
    length = int(record.manifest.length)
    timesteps = record.timesteps.copy()
    frame_index = np.arange(length, dtype=np.int64)
    rows: dict[str, Any] = {
        "index": np.arange(dataset_from_index, dataset_from_index + length, dtype=np.int64),
        LEROBOT_EPISODE_INDEX: np.full(length, int(episode_index), dtype=np.int64),
        LEROBOT_FRAME_INDEX: frame_index,
        LEROBOT_TIMESTAMP: _timestamps(timesteps, length=length, fps=_fps(record)),
        LEROBOT_TASK_INDEX: np.full(length, int(task_index), dtype=np.int64),
        LEROBOT_ACTION: [np.asarray(row, dtype=action.dtype) for row in action],
        "reward": _column_or_default(
            timesteps,
            "reward",
            length=length,
            default=0.0,
            dtype=np.float32,
        ),
        "done": _column_or_default(timesteps, "done", length=length, default=False, dtype=np.bool_),
        "truncated": _column_or_default(
            timesteps,
            "truncated",
            length=length,
            default=False,
            dtype=np.bool_,
        ),
    }
    done = np.asarray(rows["done"], dtype=np.bool_)
    truncated = np.asarray(rows["truncated"], dtype=np.bool_)
    rows["is_first"] = frame_index == 0
    rows["is_last"] = frame_index == length - 1
    rows["is_terminal"] = done | truncated
    if observation_state is not None:
        rows[LEROBOT_OBSERVATION_STATE] = [
            np.asarray(row, dtype=observation_state.dtype) for row in observation_state
        ]
    frame = pd.DataFrame(rows)
    if path.exists():
        path.unlink()
    _write_table(path, frame)


def _write_episode_metadata(
    path: Path,
    record: TraceRecord,
    *,
    episode_index: int,
    task_index: int,
    dataset_from_index: int,
    video_metadata: Mapping[str, Any],
) -> None:
    length = int(record.manifest.length)
    row = {
        LEROBOT_EPISODE_INDEX: int(episode_index),
        LEROBOT_TASK_INDEX: int(task_index),
        "tasks": [_task_text(record)],
        "length": length,
        "dataset_from_index": int(dataset_from_index),
        "dataset_to_index": int(dataset_from_index + length),
        "data/chunk_index": _chunk_file_index(episode_index)[0],
        "data/file_index": _chunk_file_index(episode_index)[1],
        **dict(video_metadata),
        **_flatten_stats(
            {
                "stats": {
                    LEROBOT_ACTION: _stats_payload(
                        _required_episode_array(
                            record,
                            TRACE_ACTION_ARRAY,
                            LEROBOT_ACTION,
                            length=length,
                        )
                    )
                }
            }
        ),
    }
    if path.exists():
        path.unlink()
    _write_table(path, pd.DataFrame.from_records([row]))


def _required_episode_array(
    record: TraceRecord,
    *names: str,
    length: int,
) -> np.ndarray:
    for name in names:
        spec = record.episode_arrays.get(name)
        if spec is not None:
            return _pad_or_trim(np.asarray(spec.array), length=length)
    raise KeyError(f"TraceRecord is missing required action array; expected one of {names}")


def _observation_state_array(record: TraceRecord, *, length: int) -> np.ndarray | None:
    for name in (
        LEROBOT_OBSERVATION_STATE,
        "robot_joint_pos",
        "eef_pos",
        "gripper_qpos",
    ):
        spec = record.episode_arrays.get(name)
        if spec is not None:
            return _pad_or_trim(np.asarray(spec.array, dtype=np.float32), length=length)
    return None


def _frame_arrays(record: TraceRecord, *, length: int) -> dict[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {}
    for name, spec in record.episode_arrays.items():
        text = str(name)
        if text.startswith(LEROBOT_IMAGE_PREFIX):
            camera = text.removeprefix(LEROBOT_IMAGE_PREFIX)
        elif text.startswith(TRACE_FRAME_PREFIX):
            camera = text.removeprefix(TRACE_FRAME_PREFIX)
        else:
            continue
        frames[camera] = _pad_or_trim(np.asarray(spec.array), length=length)
    return frames
