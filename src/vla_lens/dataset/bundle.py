"""Episode-level VLA Lens view over a LeRobot dataset root."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping

import imageio.v2 as imageio
import numpy as np
import pandas as pd

from vla_lens.artifacts import LensArtifact
from vla_lens.capture.lerobot_v3 import (
    LEROBOT_ACTION,
    LEROBOT_EPISODE_INDEX,
    LEROBOT_FRAME_INDEX,
    LEROBOT_IMAGE_PREFIX,
    LEROBOT_OBSERVATION_STATE,
    LEROBOT_TASK_INDEX,
    LEROBOT_TIMESTAMP,
)
from vla_lens.dataset.common import (
    LEROBOT_DATA_PATH_TEMPLATE,
    LEROBOT_VIDEO_PATH_TEMPLATE,
    OVERLAY_EPISODE_DIR,
    _array_record,
    _format_lerobot_path,
    _is_scalar,
    _prefix_table_paths,
    _stack_column,
    _task_for_index,
)
from vla_lens.dataset.media import _read_video_frames
from vla_lens.dataset.overlay import (
    _overlay_table,
    _record_from_manifest,
    _write_overlay_root,
)
from vla_lens.traces import TraceBundle, TraceManifest


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
