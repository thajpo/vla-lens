"""LeRobot v3 robot data plus VLA Lens overlay storage.

The normal VLA Lens app reads and writes the LeRobot v3 directory contract
directly.  It intentionally does not import ``lerobot`` because policy/runtime
dependencies belong to capture environments, not the dashboard/test stack.
"""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np
import pandas as pd

from vla_lens.artifacts import LensArtifact
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
    VLA_LENS_OVERLAY_DIR,
    VLA_LENS_OVERLAY_MANIFEST,
    VLA_LENS_OVERLAY_REFERENCES,
    validate_lerobot_v3_dataset,
)
from vla_lens.capture.records import TraceRecord
from vla_lens.traces import ArraySpec, TraceBundle, TraceDataset, TraceManifest

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

    action = _required_episode_array(record, LEGACY_ACTION_ARRAY, LEROBOT_ACTION, length=length)
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


class LeRobotEpisodeBundle:
    """Episode-level view over LeRobot robot data and one optional VLA overlay."""

    def __init__(
        self,
        root: str | Path,
        *,
        episode_row: Mapping[str, Any],
        info: Mapping[str, Any],
        tasks: pd.DataFrame,
        overlay_bundle: TraceBundle | None = None,
        trace_id_prefix: str | None = None,
    ):
        self.root = Path(root)
        self.path = self.root
        self.episode_row = dict(episode_row)
        self.info = dict(info)
        self.tasks = tasks
        self.overlay_bundle = overlay_bundle
        self.trace_id_prefix = str(trace_id_prefix or "").strip()
        self.episode_index = int(self.episode_row[LEROBOT_EPISODE_INDEX])
        self._frame_cache: dict[str, np.ndarray] = {}

    @cached_property
    def manifest(self) -> TraceManifest:
        if self.overlay_bundle is not None:
            overlay_manifest = self.overlay_bundle.manifest
            metadata = dict(overlay_manifest.metadata)
            metadata["lerobot_episode_index"] = self.episode_index
            metadata["robot_dataset_format"] = "lerobot_v3"
            return TraceManifest(
                trace_id=overlay_manifest.trace_id,
                episode_id=overlay_manifest.episode_id,
                task_id=overlay_manifest.task_id,
                prompt=overlay_manifest.prompt,
                model_id=overlay_manifest.model_id,
                env_id=overlay_manifest.env_id,
                robot_id=overlay_manifest.robot_id,
                outcome=overlay_manifest.outcome,
                length=self._length(),
                schema_version=overlay_manifest.schema_version,
                metadata=metadata,
            )

        task_index = int(self.episode_row.get(LEROBOT_TASK_INDEX, 0))
        task = _task_for_index(self.tasks, task_index)
        episode_id = f"episode_{self.episode_index:06d}"
        trace_id = (
            f"{self.trace_id_prefix}__{episode_id}" if self.trace_id_prefix else episode_id
        )
        return TraceManifest(
            trace_id=trace_id,
            episode_id=episode_id,
            task_id=str(task_index),
            prompt=task,
            model_id="",
            env_id=str(self.info.get("robot_type") or ""),
            robot_id=str(self.info.get("robot_type") or ""),
            outcome="unknown",
            length=self._length(),
            metadata={
                "robot_dataset_format": "lerobot_v3",
                "lerobot_episode_index": self.episode_index,
                "lerobot_trace_id_prefix": self.trace_id_prefix,
            },
        )

    @cached_property
    def timesteps(self) -> pd.DataFrame:
        data = self._data_frame()
        frame = pd.DataFrame(
            {
                "timestep": data.get(LEROBOT_FRAME_INDEX, pd.Series(dtype=np.int64)),
                LEROBOT_FRAME_INDEX: data.get(LEROBOT_FRAME_INDEX, pd.Series(dtype=np.int64)),
                LEROBOT_TIMESTAMP: data.get(LEROBOT_TIMESTAMP, pd.Series(dtype=np.float32)),
                LEROBOT_EPISODE_INDEX: data.get(LEROBOT_EPISODE_INDEX, self.episode_index),
                LEROBOT_TASK_INDEX: data.get(LEROBOT_TASK_INDEX, 0),
            }
        )
        for column in ("reward", "done", "truncated", "is_first", "is_last", "is_terminal"):
            if column in data:
                frame[column] = data[column].to_numpy()
        if self.overlay_bundle is not None and not self.overlay_bundle.timesteps.empty:
            overlay = self.overlay_bundle.timesteps.copy()
            if "timestep" in overlay:
                merged = frame.merge(
                    overlay,
                    on="timestep",
                    how="left",
                    suffixes=("", "_overlay"),
                )
                for column in tuple(merged.columns):
                    if column.endswith("_overlay"):
                        base = column.removesuffix("_overlay")
                        if base not in frame:
                            merged[base] = merged[column]
                        merged = merged.drop(columns=[column])
                return merged
        return frame

    @cached_property
    def policy_calls(self) -> pd.DataFrame:
        return _overlay_table(self.overlay_bundle, "policy_calls")

    @cached_property
    def generation_steps(self) -> pd.DataFrame:
        return _overlay_table(self.overlay_bundle, "generation_steps")

    @cached_property
    def streams(self) -> pd.DataFrame:
        return _overlay_table(self.overlay_bundle, "streams")

    @cached_property
    def token_spaces(self) -> pd.DataFrame:
        return _overlay_table(self.overlay_bundle, "token_spaces")

    @cached_property
    def tokens(self) -> pd.DataFrame:
        return _overlay_table(self.overlay_bundle, "tokens")

    @cached_property
    def robot_state(self) -> pd.DataFrame:
        return _overlay_table(self.overlay_bundle, "robot_state")

    @cached_property
    def scene_state(self) -> pd.DataFrame:
        return _overlay_table(self.overlay_bundle, "scene_state")

    @cached_property
    def camera_state(self) -> pd.DataFrame:
        return _overlay_table(self.overlay_bundle, "camera_state")

    @cached_property
    def evaluation(self) -> pd.DataFrame:
        overlay = _overlay_table(self.overlay_bundle, "evaluation")
        if not overlay.empty:
            return overlay
        rows = []
        for _, row in self.timesteps.iterrows():
            if "reward" in row:
                rows.append(
                    {
                        "timestep": int(row["timestep"]),
                        "metric_name": "reward",
                        "metric_value": float(row["reward"]),
                        "source": "lerobot.action_loop",
                    }
                )
        return pd.DataFrame.from_records(rows)

    @cached_property
    def image_preprocessing(self) -> pd.DataFrame:
        return _overlay_table(self.overlay_bundle, "image_preprocessing")

    @cached_property
    def prompt_metadata(self) -> pd.DataFrame:
        return _overlay_table(self.overlay_bundle, "prompt_metadata")

    @cached_property
    def action_normalization(self) -> pd.DataFrame:
        return _overlay_table(self.overlay_bundle, "action_normalization")

    @cached_property
    def capture_report(self) -> dict[str, Any]:
        if self.overlay_bundle is None:
            return {
                "dataset_format": "lerobot_v3",
                "captured_cheap_fields": [
                    LEROBOT_ACTION,
                    *[f"{LEROBOT_IMAGE_PREFIX}{camera}" for camera in self.cameras()],
                ],
            }
        payload = dict(self.overlay_bundle.capture_report)
        payload["dataset_format"] = "lerobot_v3"
        payload["lerobot_episode_index"] = self.episode_index
        return payload

    @cached_property
    def fingerprints(self) -> dict[str, Any]:
        return self.overlay_bundle.fingerprints if self.overlay_bundle is not None else {}

    @cached_property
    def array_index(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        data = self._data_frame()
        for name, axes in (
            (LEROBOT_ACTION, ("timestep", "action_dim")),
            (LEROBOT_OBSERVATION_STATE, ("timestep", "state_dim")),
        ):
            if name not in data:
                continue
            array = _stack_column(data[name])
            rows.append(
                _array_record(
                    name=name,
                    relative_path=self._data_path().relative_to(self.root),
                    array=array,
                    axes=axes,
                    storage_format="parquet_column",
                    compression="snappy",
                    metadata={"column": name, "robot_dataset_format": "lerobot_v3"},
                )
            )
        for camera in self.cameras():
            video_key = f"{LEROBOT_IMAGE_PREFIX}{camera}"
            shape = self._frame_shape(camera)
            rows.append(
                {
                    "name": video_key,
                    "relative_path": str(self._video_path(video_key).relative_to(self.root)),
                    "storage_format": "mp4",
                    "chunks": json.dumps([]),
                    "compression": "h264",
                    "array_type": "episode",
                    "shape": json.dumps([self._length(), *shape]),
                    "dtype": "uint8",
                    "axes": json.dumps(["timestep", "height", "width", "channel"]),
                    "metadata": json.dumps(
                        {"video_key": video_key, "robot_dataset_format": "lerobot_v3"},
                        sort_keys=True,
                    ),
                }
            )
        if self.overlay_bundle is not None and not self.overlay_bundle.array_index.empty:
            rows.extend(
                _prefix_table_paths(
                    self.overlay_bundle.array_index,
                    self.overlay_bundle.path.relative_to(self.root),
                ).to_dict("records")
            )
        return pd.DataFrame.from_records(rows)

    @cached_property
    def model_sites(self) -> pd.DataFrame:
        if self.overlay_bundle is None or self.overlay_bundle.model_sites.empty:
            return pd.DataFrame()
        return _prefix_table_paths(
            self.overlay_bundle.model_sites,
            self.overlay_bundle.path.relative_to(self.root),
        )

    @cached_property
    def artifact_index(self) -> pd.DataFrame:
        return _overlay_table(self.overlay_bundle, "artifact_index")

    def episode_record(self) -> dict[str, Any]:
        record = self.manifest.to_dict()
        record["path"] = str(self.root)
        record["episode_index"] = self.episode_index
        record["task_index"] = self.episode_row.get(LEROBOT_TASK_INDEX)
        record["robot_dataset_format"] = "lerobot_v3"
        for key, value in self.manifest.metadata.items():
            if _is_scalar(value) and key not in record:
                record[key] = value
        return record

    def array(self, name: str, *, mmap: bool = False) -> np.ndarray:
        del mmap
        if name == LEROBOT_ACTION:
            return _stack_column(self._data_frame()[LEROBOT_ACTION])
        if name == LEROBOT_OBSERVATION_STATE:
            return _stack_column(self._data_frame()[LEROBOT_OBSERVATION_STATE])
        if name.startswith(LEROBOT_IMAGE_PREFIX):
            return self.frames(name.removeprefix(LEROBOT_IMAGE_PREFIX))
        if self.overlay_bundle is None:
            raise KeyError(f"Unknown episode array '{name}' in {self.root}")
        return self.overlay_bundle.array(name)

    def model_site(self, name: str, *, mmap: bool = False) -> np.ndarray:
        if self.overlay_bundle is None:
            raise KeyError(f"Unknown model site '{name}' in {self.root}")
        return self.overlay_bundle.model_site(name, mmap=mmap)

    def frames(self, camera: str, *, mmap: bool = False) -> np.ndarray:
        del mmap
        if camera not in self._frame_cache:
            self._frame_cache[camera] = _read_video_frames(
                self._video_path(f"{LEROBOT_IMAGE_PREFIX}{camera}")
            )
        return self._frame_cache[camera]

    def frame(self, camera: str, timestep: int) -> np.ndarray:
        if camera in self._frame_cache:
            return np.asarray(self._frame_cache[camera][timestep])
        reader = imageio.get_reader(self._video_path(f"{LEROBOT_IMAGE_PREFIX}{camera}"))
        try:
            return np.asarray(reader.get_data(int(timestep)))
        finally:
            reader.close()

    def cameras(self) -> list[str]:
        features = self.info.get("features")
        if not isinstance(features, Mapping):
            return []
        return sorted(
            str(key).removeprefix(LEROBOT_IMAGE_PREFIX)
            for key, value in features.items()
            if str(key).startswith(LEROBOT_IMAGE_PREFIX)
            and isinstance(value, Mapping)
            and str(value.get("dtype")) == "video"
        )

    def actions(self, *, mmap: bool = False) -> np.ndarray:
        return self.array(LEROBOT_ACTION, mmap=mmap)

    def action_chunks(self, *, mmap: bool = False) -> np.ndarray:
        if self.overlay_bundle is None:
            raise KeyError("action_chunks")
        return self.overlay_bundle.action_chunks(mmap=mmap)

    def generation_actions(self, *, mmap: bool = False) -> np.ndarray:
        if self.overlay_bundle is None:
            raise KeyError("generation_actions")
        return self.overlay_bundle.generation_actions(mmap=mmap)

    def save_artifact(
        self,
        artifact: LensArtifact,
        arrays: Mapping[str, np.ndarray] | None = None,
    ) -> LensArtifact:
        bundle = self._ensure_overlay_bundle()
        return bundle.save_artifact(artifact, arrays=arrays)

    def load_artifact(self, artifact_id: str) -> LensArtifact:
        if self.overlay_bundle is None:
            raise KeyError(f"Unknown artifact '{artifact_id}'")
        return self.overlay_bundle.load_artifact(artifact_id)

    def load_artifact_array(
        self,
        artifact: LensArtifact,
        name: str,
        *,
        mmap: bool = False,
    ) -> np.ndarray:
        if self.overlay_bundle is None:
            raise KeyError(f"Unknown artifact '{artifact.artifact_id}'")
        return self.overlay_bundle.load_artifact_array(artifact, name, mmap=mmap)

    def _ensure_overlay_bundle(self) -> TraceBundle:
        if self.overlay_bundle is not None:
            return self.overlay_bundle
        overlay_path = self.root / OVERLAY_EPISODE_DIR / f"episode_{self.episode_index:06d}"
        self.overlay_bundle = TraceBundle.create(
            overlay_path,
            manifest=self.manifest,
            timesteps=self.timesteps,
            overwrite=False,
        )
        _write_overlay_root(
            self.root,
            _record_from_manifest(self.manifest),
            episode_index=self.episode_index,
            overlay_bundle=self.overlay_bundle,
        )
        return self.overlay_bundle

    def _data_frame(self) -> pd.DataFrame:
        if "data_frame" not in self.__dict__:
            frame = pd.read_parquet(self._data_path())
            if LEROBOT_EPISODE_INDEX in frame:
                frame = frame.loc[frame[LEROBOT_EPISODE_INDEX].astype(int) == self.episode_index]
            self.__dict__["data_frame"] = frame.reset_index(drop=True)
        return self.__dict__["data_frame"]

    def _data_path(self) -> Path:
        return self.root / _format_lerobot_path(
            str(self.info.get("data_path") or LEROBOT_DATA_PATH_TEMPLATE),
            self.episode_row,
            self.episode_index,
            prefix="data",
        )

    def _video_path(self, video_key: str) -> Path:
        template = str(self.info.get("video_path") or LEROBOT_VIDEO_PATH_TEMPLATE)
        return self.root / _format_lerobot_path(
            template,
            self.episode_row,
            self.episode_index,
            prefix=f"videos/{video_key}",
            video_key=video_key,
        )

    def _length(self) -> int:
        return int(
            self.episode_row.get("length")
            or self.episode_row.get("num_frames")
            or len(self._data_frame())
        )

    def _frame_shape(self, camera: str) -> list[int]:
        features = self.info.get("features")
        key = f"{LEROBOT_IMAGE_PREFIX}{camera}"
        if isinstance(features, Mapping):
            feature = features.get(key)
            if isinstance(feature, Mapping):
                shape = [int(item) for item in feature.get("shape", [])]
                if len(shape) == 3:
                    if shape[0] in {1, 3, 4}:
                        return [shape[1], shape[2], shape[0]]
                    return shape
        try:
            frames = self.frames(camera)
            return [int(frames.shape[1]), int(frames.shape[2]), int(frames.shape[3])]
        except Exception:
            return [0, 0, 3]


def _record_from_manifest(manifest: TraceManifest) -> TraceRecord:
    return TraceRecord(
        manifest=manifest,
        timesteps=pd.DataFrame({"timestep": np.arange(manifest.length, dtype=np.int32)}),
    )


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


def _overlay_bundle_for_episode(
    root: Path,
    refs: pd.DataFrame,
    episode_index: int,
) -> TraceBundle | None:
    overlay_path: Path | None = None
    if not refs.empty and LEROBOT_EPISODE_INDEX in refs:
        matches = refs.loc[refs[LEROBOT_EPISODE_INDEX].astype(int) == int(episode_index)]
        if not matches.empty and "overlay_path" in matches:
            overlay_path = root / str(matches.iloc[-1]["overlay_path"])
    if overlay_path is None:
        candidate = root / OVERLAY_EPISODE_DIR / f"episode_{episode_index:06d}"
        overlay_path = candidate if (candidate / TraceBundle.MANIFEST).exists() else None
    if overlay_path is None or not (overlay_path / TraceBundle.MANIFEST).exists():
        return None
    return TraceBundle.open(overlay_path)


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


def _chunk_file_index(episode_index: int) -> tuple[int, int]:
    return int(episode_index) // DEFAULT_CHUNKS_SIZE, int(episode_index) % DEFAULT_CHUNKS_SIZE


def _task_index_for_record(root: Path, record: TraceRecord) -> int:
    tasks = _read_tasks(root)
    task = _task_text(record)
    if not tasks.empty and "task" in tasks and "task_index" in tasks:
        matches = tasks.loc[tasks["task"].astype(str) == task]
        if not matches.empty:
            return int(matches.iloc[-1]["task_index"])
        return int(pd.to_numeric(tasks["task_index"], errors="coerce").fillna(-1).max()) + 1
    return 0


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


def _feature_signature(features: Mapping[str, Any]) -> str:
    return json.dumps(_jsonable(features), sort_keys=True, separators=(",", ":"))


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
                            LEGACY_ACTION_ARRAY,
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


def _write_overlay_bundle(
    root: Path,
    record: TraceRecord,
    *,
    episode_index: int,
    data_path: Path,
    overwrite: bool,
) -> TraceBundle:
    path = root / OVERLAY_EPISODE_DIR / f"episode_{episode_index:06d}"
    manifest = TraceManifest(
        trace_id=record.manifest.trace_id,
        episode_id=record.manifest.episode_id,
        task_id=record.manifest.task_id,
        prompt=record.manifest.prompt,
        model_id=record.manifest.model_id,
        env_id=record.manifest.env_id,
        robot_id=record.manifest.robot_id,
        outcome=record.manifest.outcome,
        length=record.manifest.length,
        schema_version=record.manifest.schema_version,
        metadata={
            **dict(record.manifest.metadata),
            "robot_dataset_format": "lerobot_v3",
            "lerobot_episode_index": int(episode_index),
            "lerobot_data_path": str(data_path),
        },
    )
    return TraceBundle.create(
        path,
        manifest=manifest,
        timesteps=record.timesteps,
        episode_arrays=_overlay_episode_arrays(record),
        model_arrays=record.model_arrays,
        policy_calls=record.policy_calls,
        generation_steps=record.generation_steps,
        streams=record.streams,
        token_spaces=record.token_spaces,
        tokens=record.tokens,
        robot_state=record.robot_state,
        scene_state=record.scene_state,
        camera_state=record.camera_state,
        evaluation=record.evaluation,
        image_preprocessing=record.image_preprocessing,
        prompt_metadata=record.prompt_metadata,
        action_normalization=_canonical_action_normalization(record.action_normalization),
        capture_request=record.capture_request,
        capture_plan=record.capture_plan,
        capture_report={
            **dict(record.capture_report),
            "dataset_format": "lerobot_v3_plus_vla_lens_overlay",
            "lerobot_episode_index": int(episode_index),
        },
        artifacts=record.artifacts,
        overwrite=overwrite,
    )


def _write_overlay_root(
    root: Path,
    record: TraceRecord,
    *,
    episode_index: int,
    overlay_bundle: TraceBundle,
) -> None:
    overlay_root = root / VLA_LENS_OVERLAY_DIR
    refs = _read_table(root / VLA_LENS_OVERLAY_REFERENCES)
    row = {
        LEROBOT_EPISODE_INDEX: int(episode_index),
        "trace_id": record.manifest.trace_id,
        "episode_id": record.manifest.episode_id,
        "length": int(record.manifest.length),
        "overlay_path": str(overlay_bundle.path.relative_to(root)),
    }
    if refs.empty:
        refs = pd.DataFrame.from_records([row])
    else:
        refs = refs.loc[refs[LEROBOT_EPISODE_INDEX].astype(int) != int(episode_index)]
        refs = pd.concat([refs, pd.DataFrame.from_records([row])], ignore_index=True)
    refs = refs.sort_values(LEROBOT_EPISODE_INDEX).reset_index(drop=True)
    _write_table(root / VLA_LENS_OVERLAY_REFERENCES, refs)
    _write_json(
        root / VLA_LENS_OVERLAY_MANIFEST,
        {
            "overlay_schema_version": OVERLAY_SCHEMA_VERSION,
            "robot_dataset_format": "lerobot_v3",
            "overlay_root": str(VLA_LENS_OVERLAY_DIR),
            "episodes": int(len(refs)),
        },
    )
    overlay_root.mkdir(parents=True, exist_ok=True)


def _overlay_episode_arrays(record: TraceRecord) -> dict[str, ArraySpec]:
    arrays: dict[str, ArraySpec] = {}
    for name, spec in record.episode_arrays.items():
        if name == LEGACY_ACTION_ARRAY:
            continue
        if str(name).startswith(LEGACY_FRAME_PREFIX):
            continue
        if name in OVERLAY_ROOT_ARRAY_NAMES:
            continue
        if str(name).startswith(OVERLAY_ROOT_ARRAY_PREFIXES):
            continue
        arrays[str(name)] = spec
    return arrays


def _canonical_action_normalization(frame: pd.DataFrame | None) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return frame
    out = frame.copy()
    if "unnormalized_action_array_ref" in out:
        out["unnormalized_action_array_ref"] = LEROBOT_ACTION
    return out


def _overlay_table(bundle: TraceBundle | None, name: str) -> pd.DataFrame:
    if bundle is None:
        return pd.DataFrame()
    return getattr(bundle, name)


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
        elif text.startswith(LEGACY_FRAME_PREFIX):
            camera = text.removeprefix(LEGACY_FRAME_PREFIX)
        else:
            continue
        frames[camera] = _pad_or_trim(np.asarray(spec.array), length=length)
    return frames


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
    if "relative_path" in out:
        out["relative_path"] = [
            str(prefix / str(value)) if str(value) else str(value)
            for value in out["relative_path"]
        ]
    return out


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


__all__ = [
    "LeRobotEpisodeBundle",
    "is_lerobot_dataset_root",
    "open_lerobot_dataset",
    "write_lerobot_trace_record",
]
