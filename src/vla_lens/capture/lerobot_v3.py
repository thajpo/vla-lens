"""LeRobot v3 robot-data contract and VLA Lens overlay helpers.

This module intentionally does not import ``lerobot``.  It defines the robot
dataset layer VLA Lens expects, plus a small validator for dataset roots that
look like LeRobotDataset v3 exports.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow.parquet as pq

LEROBOT_V3_VERSION = "v3.0"

LEROBOT_META_DIR = Path("meta")
LEROBOT_DATA_DIR = Path("data")
LEROBOT_VIDEOS_DIR = Path("videos")
LEROBOT_INFO_PATH = LEROBOT_META_DIR / "info.json"
LEROBOT_STATS_PATH = LEROBOT_META_DIR / "stats.json"
LEROBOT_TASKS_JSONL_PATH = LEROBOT_META_DIR / "tasks.jsonl"
LEROBOT_TASKS_PARQUET_PATH = LEROBOT_META_DIR / "tasks.parquet"
LEROBOT_EPISODES_DIR = LEROBOT_META_DIR / "episodes"

VLA_LENS_OVERLAY_DIR = Path("vla_lens")
VLA_LENS_OVERLAY_MANIFEST = VLA_LENS_OVERLAY_DIR / "overlay.json"
VLA_LENS_OVERLAY_TABLES_DIR = VLA_LENS_OVERLAY_DIR / "tables"
VLA_LENS_OVERLAY_ARRAYS_DIR = VLA_LENS_OVERLAY_DIR / "arrays"
VLA_LENS_OVERLAY_ARTIFACTS_DIR = VLA_LENS_OVERLAY_DIR / "artifacts"
VLA_LENS_OVERLAY_REFERENCES = VLA_LENS_OVERLAY_TABLES_DIR / "episode_refs.parquet"

LEROBOT_EPISODE_INDEX = "episode_index"
LEROBOT_FRAME_INDEX = "frame_index"
LEROBOT_TIMESTAMP = "timestamp"
LEROBOT_TASK_INDEX = "task_index"
LEROBOT_ACTION = "action"
LEROBOT_OBSERVATION_STATE = "observation.state"
LEROBOT_IMAGE_PREFIX = "observation.images."

LEROBOT_REQUIRED_STEP_FIELDS = (
    LEROBOT_EPISODE_INDEX,
    LEROBOT_FRAME_INDEX,
    LEROBOT_TIMESTAMP,
    LEROBOT_TASK_INDEX,
    LEROBOT_ACTION,
)
LEROBOT_CANONICAL_ROBOT_FIELDS = (
    *LEROBOT_REQUIRED_STEP_FIELDS,
    LEROBOT_OBSERVATION_STATE,
    f"{LEROBOT_IMAGE_PREFIX}<camera>",
)


@dataclass(frozen=True, slots=True)
class ContractIssue:
    """One validation issue for a LeRobot v3 root or VLA Lens overlay."""

    code: str
    message: str
    path: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True, slots=True)
class LeRobotV3ValidationResult:
    """Validation result for the robot-data layer and optional overlay."""

    root: Path
    valid: bool
    errors: tuple[ContractIssue, ...] = ()
    warnings: tuple[ContractIssue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "valid": self.valid,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


def lerobot_required_metadata_paths() -> tuple[Path, ...]:
    """Required LeRobot v3 metadata files with a stable path."""

    return (LEROBOT_INFO_PATH, LEROBOT_STATS_PATH)


def lerobot_task_metadata_paths() -> tuple[Path, ...]:
    """Accepted task metadata paths.

    VLA Lens prefers ``meta/tasks.jsonl``.  ``meta/tasks.parquet`` is accepted
    because LeRobot v3 documentation and tooling have used both forms.
    """

    return (LEROBOT_TASKS_JSONL_PATH, LEROBOT_TASKS_PARQUET_PATH)


def vla_lens_overlay_path(root: str | Path, *parts: str | Path) -> Path:
    """Return a path inside the VLA Lens overlay for a LeRobot dataset root."""

    return Path(root) / VLA_LENS_OVERLAY_DIR / Path(*parts)


def is_lerobot_image_field(name: str) -> bool:
    """Return whether a field is a LeRobot image observation."""

    return str(name).startswith(LEROBOT_IMAGE_PREFIX)


def is_lerobot_robot_field(name: str) -> bool:
    """Return whether a field belongs to the canonical LeRobot robot-data layer."""

    field = str(name)
    return field in {
        LEROBOT_EPISODE_INDEX,
        LEROBOT_FRAME_INDEX,
        LEROBOT_TIMESTAMP,
        LEROBOT_TASK_INDEX,
        LEROBOT_ACTION,
        LEROBOT_OBSERVATION_STATE,
    } or is_lerobot_image_field(field)


def lerobot_episode_metadata_paths(root: str | Path) -> tuple[Path, ...]:
    """Return sorted episode metadata shards under ``meta/episodes``."""

    return tuple(sorted((Path(root) / LEROBOT_EPISODES_DIR).rglob("*.parquet")))


def lerobot_data_paths(root: str | Path) -> tuple[Path, ...]:
    """Return sorted low-dimensional data shards under ``data``."""

    return tuple(sorted((Path(root) / LEROBOT_DATA_DIR).rglob("*.parquet")))


def lerobot_video_paths(root: str | Path) -> tuple[Path, ...]:
    """Return sorted MP4 video shards under ``videos``."""

    return tuple(sorted((Path(root) / LEROBOT_VIDEOS_DIR).rglob("*.mp4")))


def validate_lerobot_v3_dataset(
    root: str | Path,
    *,
    validate_overlay: bool = True,
) -> LeRobotV3ValidationResult:
    """Validate the dependency-free shape of a LeRobot v3 dataset root."""

    dataset_root = Path(root)
    errors: list[ContractIssue] = []
    warnings: list[ContractIssue] = []

    if not dataset_root.exists():
        errors.append(_issue("missing_root", "Dataset root does not exist", path=dataset_root))
        return _result(dataset_root, errors, warnings)

    info = _read_json(dataset_root / LEROBOT_INFO_PATH, errors)
    for required_path in lerobot_required_metadata_paths():
        _require_path(dataset_root / required_path, errors, "missing_metadata")

    task_path = _first_existing(dataset_root / path for path in lerobot_task_metadata_paths())
    if task_path is None:
        errors.append(
            _issue(
                "missing_task_metadata",
                "Expected meta/tasks.jsonl or meta/tasks.parquet",
                path=dataset_root / LEROBOT_TASKS_JSONL_PATH,
            )
        )

    episode_paths = lerobot_episode_metadata_paths(dataset_root)
    if not episode_paths:
        errors.append(
            _issue(
                "missing_episode_metadata",
                "Expected at least one episode metadata parquet under meta/episodes",
                path=dataset_root / LEROBOT_EPISODES_DIR,
            )
        )
    data_paths = lerobot_data_paths(dataset_root)
    if not data_paths:
        errors.append(
            _issue(
                "missing_data_shards",
                "Expected at least one low-dimensional data parquet under data",
                path=dataset_root / LEROBOT_DATA_DIR,
            )
        )

    feature_names = _feature_names(info)
    data_column_sets = {path: _parquet_columns(path, warnings) for path in data_paths}
    data_columns = set().union(*data_column_sets.values()) if data_column_sets else set()
    for field_name in LEROBOT_REQUIRED_STEP_FIELDS:
        if field_name not in feature_names:
            errors.append(
                _issue(
                    "missing_step_field",
                    f"LeRobot v3 feature metadata is missing required field '{field_name}'",
                    details={"field": field_name},
                )
            )
        for data_path, columns in data_column_sets.items():
            if field_name in columns:
                continue
            errors.append(
                _issue(
                    "missing_data_field",
                    f"LeRobot v3 data shard is missing required field '{field_name}'",
                    path=data_path,
                    details={"field": field_name},
                )
            )

    if info:
        codebase_version = str(info.get("codebase_version", ""))
        if not codebase_version.startswith("v3"):
            errors.append(
                _issue(
                    "unsupported_lerobot_version",
                    "Expected a LeRobot v3 dataset root",
                    path=dataset_root / LEROBOT_INFO_PATH,
                    details={"codebase_version": codebase_version},
                )
            )
        if "data_path" not in info:
            warnings.append(
                _issue(
                    "missing_data_path_template",
                    "meta/info.json should include the LeRobot data_path template",
                    path=dataset_root / LEROBOT_INFO_PATH,
                )
            )
        if "features" not in info:
            errors.append(
                _issue(
                    "missing_features",
                    "meta/info.json should define LeRobot feature metadata",
                    path=dataset_root / LEROBOT_INFO_PATH,
                )
            )

    available_fields = feature_names | data_columns
    image_fields = {field for field in available_fields if is_lerobot_image_field(field)}
    if image_fields and not lerobot_video_paths(dataset_root):
        errors.append(
            _issue(
                "missing_video_shards",
                "Image features require MP4 video shards under videos",
                path=dataset_root / LEROBOT_VIDEOS_DIR,
                details={"image_fields": sorted(image_fields)},
            )
        )
    if LEROBOT_OBSERVATION_STATE not in available_fields and not image_fields:
        warnings.append(
            _issue(
                "missing_observation_source",
                "Expected observation.state and/or observation.images.<camera>",
            )
        )

    episode_lengths = _episode_lengths(dataset_root, episode_paths, errors, warnings)
    if validate_overlay:
        _validate_overlay_references(dataset_root, episode_lengths, errors, warnings)

    return _result(dataset_root, errors, warnings)


def _validate_overlay_references(
    root: Path,
    episode_lengths: Mapping[int, int | None],
    errors: list[ContractIssue],
    warnings: list[ContractIssue],
) -> None:
    overlay_root = root / VLA_LENS_OVERLAY_DIR
    if not overlay_root.exists():
        return

    reference_tables = sorted((overlay_root / "tables").rglob("*.parquet"))
    if not reference_tables:
        warnings.append(
            _issue(
                "empty_overlay",
                "VLA Lens overlay exists but has no parquet reference tables",
                path=overlay_root,
            )
        )
        return

    for table_path in reference_tables:
        columns = _parquet_columns(table_path, warnings)
        if LEROBOT_EPISODE_INDEX not in columns:
            continue
        read_columns = [LEROBOT_EPISODE_INDEX]
        if LEROBOT_FRAME_INDEX in columns:
            read_columns.append(LEROBOT_FRAME_INDEX)
        try:
            table = pd.read_parquet(table_path, columns=read_columns)
        except Exception as exc:  # pragma: no cover - exercised by integration data corruption
            errors.append(
                _issue(
                    "unreadable_overlay_table",
                    "Could not read VLA Lens overlay reference table",
                    path=table_path,
                    details={"error": str(exc)},
                )
            )
            continue
        _validate_reference_rows(table_path, table, episode_lengths, errors)


def _validate_reference_rows(
    table_path: Path,
    table: pd.DataFrame,
    episode_lengths: Mapping[int, int | None],
    errors: list[ContractIssue],
) -> None:
    known_episodes = set(episode_lengths)
    for row_index, row in table.iterrows():
        episode_index = _maybe_int(row.get(LEROBOT_EPISODE_INDEX))
        if episode_index is None or episode_index not in known_episodes:
            errors.append(
                _issue(
                    "overlay_unknown_episode",
                    "Overlay row references an unknown LeRobot episode_index",
                    path=table_path,
                    details={
                        "row": int(row_index),
                        "episode_index": row.get(LEROBOT_EPISODE_INDEX),
                    },
                )
            )
            continue
        if LEROBOT_FRAME_INDEX not in table.columns:
            continue
        frame_index = _maybe_int(row.get(LEROBOT_FRAME_INDEX))
        length = episode_lengths.get(episode_index)
        if frame_index is None or length is None:
            continue
        if frame_index < 0 or frame_index >= length:
            errors.append(
                _issue(
                    "overlay_unknown_frame",
                    "Overlay row references a frame_index outside the episode length",
                    path=table_path,
                    details={
                        "row": int(row_index),
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "episode_length": length,
                    },
                )
            )


def _episode_lengths(
    root: Path,
    episode_paths: Sequence[Path],
    errors: list[ContractIssue],
    warnings: list[ContractIssue],
) -> dict[int, int | None]:
    del root
    lengths: dict[int, int | None] = {}
    for path in episode_paths:
        columns = _parquet_columns(path, warnings)
        if LEROBOT_EPISODE_INDEX not in columns:
            errors.append(
                _issue(
                    "missing_episode_index",
                    "Episode metadata requires episode_index",
                    path=path,
                )
            )
            continue
        length_column = _first_existing_name(columns, ("length", "num_frames", "episode_length"))
        read_columns = [LEROBOT_EPISODE_INDEX]
        if length_column is not None:
            read_columns.append(length_column)
        try:
            table = pd.read_parquet(path, columns=read_columns)
        except Exception as exc:  # pragma: no cover - exercised by integration data corruption
            errors.append(
                _issue(
                    "unreadable_episode_metadata",
                    "Could not read episode metadata parquet",
                    path=path,
                    details={"error": str(exc)},
                )
            )
            continue
        if length_column is None:
            warnings.append(
                _issue(
                    "missing_episode_length",
                    "Episode metadata should include length or num_frames",
                    path=path,
                )
            )
        for _, row in table.iterrows():
            episode_index = _maybe_int(row.get(LEROBOT_EPISODE_INDEX))
            if episode_index is None:
                continue
            length = _maybe_int(row.get(length_column)) if length_column is not None else None
            lengths[episode_index] = length
    return lengths


def _read_json(path: Path, errors: list[ContractIssue]) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(
            _issue(
                "invalid_json",
                "Could not parse JSON metadata",
                path=path,
                details={"error": str(exc)},
            )
        )
        return {}
    if not isinstance(payload, dict):
        errors.append(_issue("invalid_json", "JSON metadata should be an object", path=path))
        return {}
    return payload


def _feature_names(info: Mapping[str, Any]) -> set[str]:
    features = info.get("features")
    if not isinstance(features, Mapping):
        return set()
    return {str(name) for name in features}


def _parquet_columns(path: Path, warnings: list[ContractIssue]) -> set[str]:
    try:
        return set(pq.read_schema(path).names)
    except Exception as exc:  # pragma: no cover - exercised by integration data corruption
        warnings.append(
            _issue(
                "unreadable_parquet_schema",
                "Could not read parquet schema",
                path=path,
                details={"error": str(exc)},
            )
        )
        return set()


def _require_path(path: Path, errors: list[ContractIssue], code: str) -> None:
    if not path.exists():
        errors.append(_issue(code, "Expected required LeRobot v3 path", path=path))


def _first_existing(paths: Sequence[Path] | Any) -> Path | None:
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return None


def _first_existing_name(names: set[str], candidates: Sequence[str]) -> str | None:
    for name in candidates:
        if name in names:
            return name
    return None


def _maybe_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _issue(
    code: str,
    message: str,
    *,
    path: str | Path | None = None,
    details: Mapping[str, Any] | None = None,
) -> ContractIssue:
    return ContractIssue(
        code=code,
        message=message,
        path=None if path is None else str(path),
        details=dict(details or {}),
    )


def _result(
    root: Path,
    errors: list[ContractIssue],
    warnings: list[ContractIssue],
) -> LeRobotV3ValidationResult:
    return LeRobotV3ValidationResult(
        root=root,
        valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


__all__ = [
    "ContractIssue",
    "LEROBOT_ACTION",
    "LEROBOT_CANONICAL_ROBOT_FIELDS",
    "LEROBOT_DATA_DIR",
    "LEROBOT_EPISODE_INDEX",
    "LEROBOT_EPISODES_DIR",
    "LEROBOT_FRAME_INDEX",
    "LEROBOT_IMAGE_PREFIX",
    "LEROBOT_INFO_PATH",
    "LEROBOT_META_DIR",
    "LEROBOT_OBSERVATION_STATE",
    "LEROBOT_REQUIRED_STEP_FIELDS",
    "LEROBOT_STATS_PATH",
    "LEROBOT_TASKS_JSONL_PATH",
    "LEROBOT_TASKS_PARQUET_PATH",
    "LEROBOT_TASK_INDEX",
    "LEROBOT_TIMESTAMP",
    "LEROBOT_V3_VERSION",
    "LEROBOT_VIDEOS_DIR",
    "LeRobotV3ValidationResult",
    "VLA_LENS_OVERLAY_ARRAYS_DIR",
    "VLA_LENS_OVERLAY_ARTIFACTS_DIR",
    "VLA_LENS_OVERLAY_DIR",
    "VLA_LENS_OVERLAY_MANIFEST",
    "VLA_LENS_OVERLAY_REFERENCES",
    "VLA_LENS_OVERLAY_TABLES_DIR",
    "is_lerobot_image_field",
    "is_lerobot_robot_field",
    "lerobot_data_paths",
    "lerobot_episode_metadata_paths",
    "lerobot_required_metadata_paths",
    "lerobot_task_metadata_paths",
    "lerobot_video_paths",
    "validate_lerobot_v3_dataset",
    "vla_lens_overlay_path",
]
