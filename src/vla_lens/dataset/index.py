"""Local Parquet indexes for dashboard browsing.

Raw LeRobot/VLA Lens traces remain the source of truth.  These indexes are
rebuildable materialized views used by the local dashboard so list/search routes
do not serialize or scan full episode payloads during normal browsing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.artifacts import LensArtifact
from vla_lens.capture.lerobot_v3 import LEROBOT_INFO_PATH
from vla_lens.traces import TraceDataset

INDEX_SCHEMA_VERSION = "0.1.1"
INDEX_TABLE_DIR = Path("vla_lens") / "tables"
INDEX_MANIFEST = INDEX_TABLE_DIR / "index_manifest.json"
EPISODE_INDEX = INDEX_TABLE_DIR / "episode_index.parquet"
MODEL_SITE_INDEX = INDEX_TABLE_DIR / "model_site_index.parquet"
ARTIFACT_INDEX = INDEX_TABLE_DIR / "dashboard_artifact_index.parquet"
PROBE_PREDICTIONS = INDEX_TABLE_DIR / "probe_predictions.parquet"

REQUIRED_EPISODE_COLUMNS = (
    "trace_id",
    "episode_id",
    "episode_index",
    "task_id",
    "prompt",
    "outcome",
    "length",
    "dataset_id",
    "benchmark",
    "profile",
    "seed",
    "path",
    "camera_names",
    "model_site_count",
    "artifact_count",
)

EPISODE_COLUMNS = (
    *REQUIRED_EPISODE_COLUMNS,
    "model_id",
    "env_id",
    "robot_id",
    "schema_version",
    "metadata",
    "array_names",
    "policy_call_count",
    "token_space_count",
)
MODEL_SITE_COLUMNS = (
    "trace_id",
    "episode_id",
    "site_id",
    "name",
    "module",
    "layer",
    "tensor_type",
    "token_kind",
    "axes",
    "shape",
    "dtype",
    "relative_path",
    "family",
    "role",
    "segment",
)
ARTIFACT_COLUMNS = (
    "artifact_id",
    "artifact_type",
    "name",
    "group_id",
    "scope",
    "artifact_scope",
    "trace_id",
    "episode_id",
    "created_utc",
    "path",
    "selector",
    "method",
    "metrics",
    "arrays",
    "display",
    "tags",
    "source_trace_ids",
)
PROBE_PREDICTION_COLUMNS = (
    "probe_id",
    "probe_name",
    "target",
    "trace_id",
    "episode_id",
    "task_id",
    "split",
    "split_category",
    "actual",
    "predicted",
    "confidence",
    "correct",
    "correct_rate",
    "model",
    "feature",
    "policy_call_index",
    "timestep",
    "generation_step",
)


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    root: Path
    manifest_path: Path
    episode_count: int
    model_site_count: int
    artifact_count: int
    probe_prediction_count: int
    dataset_fingerprint: str
    mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "manifest_path": str(self.manifest_path),
            "episode_count": self.episode_count,
            "model_site_count": self.model_site_count,
            "artifact_count": self.artifact_count,
            "probe_prediction_count": self.probe_prediction_count,
            "dataset_fingerprint": self.dataset_fingerprint,
            "mode": self.mode,
        }


class DatasetIndexError(ValueError):
    """Raised when a dashboard dataset index is missing or invalid."""


def index_root(root: str | Path) -> Path:
    return Path(root) / INDEX_TABLE_DIR


def index_manifest_path(root: str | Path) -> Path:
    return Path(root) / INDEX_MANIFEST


def build_dataset_index(root: str | Path, *, overwrite: bool = False) -> IndexBuildResult:
    """Build or append local dashboard indexes for a dataset root."""

    dataset = TraceDataset.open(root)
    root_path = Path(root)
    tables_dir = index_root(root_path)
    tables_dir.mkdir(parents=True, exist_ok=True)

    existing_episode_index = _read_existing_table(root_path / EPISODE_INDEX, EPISODE_COLUMNS)
    append_mode = bool(not overwrite and not existing_episode_index.empty)
    existing_trace_ids = (
        set(existing_episode_index["trace_id"].astype(str).tolist()) if append_mode else set()
    )
    new_bundles = [
        bundle for bundle in dataset.bundles if bundle.manifest.trace_id not in existing_trace_ids
    ]
    source_bundles = new_bundles if append_mode else dataset.bundles

    episode_rows = [
        _episode_index_row(dataset, bundle, idx)
        for idx, bundle in enumerate(source_bundles)
    ]
    episode_index = _merge_append_table(
        existing_episode_index if append_mode else _empty_table(EPISODE_COLUMNS),
        pd.DataFrame.from_records(episode_rows),
        key_columns=("trace_id",),
        columns=EPISODE_COLUMNS,
    )

    model_rows = [
        row
        for bundle in source_bundles
        for row in _model_site_rows(bundle)
    ]
    model_site_index = _merge_append_table(
        _read_existing_table(root_path / MODEL_SITE_INDEX, MODEL_SITE_COLUMNS)
        if append_mode
        else _empty_table(MODEL_SITE_COLUMNS),
        pd.DataFrame.from_records(model_rows),
        key_columns=("trace_id", "name"),
        columns=MODEL_SITE_COLUMNS,
    )

    artifact_index = _artifact_index_table(dataset)
    probe_predictions = _probe_prediction_index_table(dataset, artifact_index)
    dataset_fingerprint = _dataset_fingerprint(root_path, episode_index)

    _write_table(root_path / EPISODE_INDEX, episode_index, EPISODE_COLUMNS)
    _write_table(root_path / MODEL_SITE_INDEX, model_site_index, MODEL_SITE_COLUMNS)
    _write_table(root_path / ARTIFACT_INDEX, artifact_index, ARTIFACT_COLUMNS)
    _write_table(root_path / PROBE_PREDICTIONS, probe_predictions, PROBE_PREDICTION_COLUMNS)
    manifest = _index_manifest(
        root_path,
        dataset_fingerprint=dataset_fingerprint,
        episode_index=episode_index,
        model_site_index=model_site_index,
        artifact_index=artifact_index,
        probe_predictions=probe_predictions,
    )
    _write_json(root_path / INDEX_MANIFEST, manifest)
    return IndexBuildResult(
        root=root_path,
        manifest_path=root_path / INDEX_MANIFEST,
        episode_count=int(len(episode_index)),
        model_site_count=int(len(model_site_index)),
        artifact_count=int(len(artifact_index)),
        probe_prediction_count=int(len(probe_predictions)),
        dataset_fingerprint=dataset_fingerprint,
        mode="append" if append_mode else "rebuild",
    )


def validate_dataset_index(root: str | Path) -> dict[str, Any]:
    """Validate that a dataset has the required local dashboard index."""

    root_path = Path(root)
    manifest_path = root_path / INDEX_MANIFEST
    if not manifest_path.exists():
        raise DatasetIndexError(_missing_index_message(root_path))
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise DatasetIndexError(
            _rebuild_message(
                root_path,
                f"Dataset index schema mismatch: {manifest.get('schema_version')!r}",
            )
    )
    episode_index = _validate_manifest_tables(root_path, manifest)["episode_index"]
    indexed_count = int(len(episode_index))
    if episode_index["trace_id"].astype(str).duplicated().any():
        raise DatasetIndexError(
            _rebuild_message(root_path, "Episode index has duplicate trace IDs.")
        )
    if str(manifest.get("dataset_fingerprint") or "") != _dataset_fingerprint(
        root_path,
        episode_index,
    ):
        raise DatasetIndexError(_rebuild_message(root_path, "Dataset index fingerprint is stale."))
    declared_count = _declared_lerobot_episode_count(root_path)
    if declared_count is not None and declared_count != indexed_count:
        raise DatasetIndexError(
            _rebuild_message(
                root_path,
                f"Dataset has {declared_count} declared episode(s), index has {indexed_count}.",
            )
        )
    return manifest


def _episode_index_row(dataset: TraceDataset, bundle: Any, fallback_index: int) -> dict[str, Any]:
    manifest = bundle.manifest
    metadata = dict(manifest.metadata or {})
    cameras = bundle.cameras()
    array_names = _table_column_values(bundle.array_index, "name")
    episode_index = _optional_int(
        metadata.get("lerobot_episode_index")
        or getattr(bundle, "episode_index", None)
        or fallback_index
    )
    dataset_id = _first_text(metadata, ("dataset_id", "batch_id")) or dataset.root.name
    benchmark = _first_text(metadata, ("benchmark", "task_suite", "suite"))
    profile = _first_text(metadata, ("capture_profile", "actual_profile", "requested_profile"))
    seed = _first_text(metadata, ("seed", "episode_seed", "start_seed"))
    return {
        "trace_id": manifest.trace_id,
        "episode_id": manifest.episode_id,
        "episode_index": episode_index,
        "task_id": manifest.task_id,
        "prompt": manifest.prompt,
        "outcome": manifest.outcome,
        "length": int(manifest.length),
        "dataset_id": str(dataset_id or ""),
        "benchmark": str(benchmark or manifest.env_id or ""),
        "profile": str(profile or ""),
        "seed": str(seed or ""),
        "path": str(bundle.path),
        "camera_names": _json_dumps(cameras),
        "model_site_count": int(len(bundle.model_sites)),
        "artifact_count": int(len(bundle.artifact_index)),
        "model_id": manifest.model_id,
        "env_id": manifest.env_id,
        "robot_id": manifest.robot_id,
        "schema_version": manifest.schema_version,
        "metadata": _json_dumps(metadata),
        "array_names": _json_dumps(array_names),
        "policy_call_count": int(len(bundle.policy_calls)),
        "token_space_count": int(len(bundle.token_spaces)),
    }


def _model_site_rows(bundle: Any) -> list[dict[str, Any]]:
    if bundle.model_sites.empty:
        return []
    rows: list[dict[str, Any]] = []
    for record in bundle.model_sites.to_dict("records"):
        name = str(record.get("name") or "")
        rows.append(
            {
                "trace_id": bundle.manifest.trace_id,
                "episode_id": bundle.manifest.episode_id,
                "site_id": str(record.get("site_id") or name),
                "name": name,
                "module": str(record.get("module") or ""),
                "layer": _optional_int(record.get("layer")),
                "tensor_type": _optional_str(record.get("tensor_type")),
                "token_kind": _optional_str(record.get("token_kind")),
                "axes": str(record.get("axes") or "[]"),
                "shape": str(record.get("shape") or "[]"),
                "dtype": str(record.get("dtype") or ""),
                "relative_path": str(record.get("relative_path") or ""),
                "family": _optional_str(record.get("family")),
                "role": _optional_str(record.get("role")),
                "segment": _optional_str(record.get("segment")),
            }
        )
    return rows


def _artifact_index_table(dataset: TraceDataset) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dataset_table = _source_dataset_artifact_table(dataset)
    for record in dataset_table.to_dict("records"):
        enriched = dict(record)
        enriched["trace_id"] = None
        enriched["episode_id"] = None
        enriched["artifact_scope"] = "dataset"
        rows.append({column: _jsonable_scalar(enriched.get(column)) for column in ARTIFACT_COLUMNS})
    for bundle in dataset.bundles:
        table = bundle.artifact_index
        if table.empty:
            continue
        for record in table.to_dict("records"):
            enriched = dict(record)
            enriched["trace_id"] = bundle.manifest.trace_id
            enriched["episode_id"] = bundle.manifest.episode_id
            enriched["artifact_scope"] = "bundle"
            rows.append(
                {column: _jsonable_scalar(enriched.get(column)) for column in ARTIFACT_COLUMNS}
            )
    return _coerce_columns(pd.DataFrame.from_records(rows), ARTIFACT_COLUMNS)


def _source_dataset_artifact_table(dataset: TraceDataset) -> pd.DataFrame:
    table = dataset.dataset_artifact_index.copy()
    if table.empty or "artifact_scope" not in table:
        return table
    artifact_scope = table["artifact_scope"].fillna("").astype(str).str.strip().str.lower()
    return table.loc[(artifact_scope == "") | (artifact_scope == "dataset")].copy()


def _probe_prediction_index_table(dataset: TraceDataset, artifacts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if artifacts.empty:
        return _empty_table(PROBE_PREDICTION_COLUMNS)
    probe_rows = artifacts.loc[artifacts.get("artifact_type", "").astype(str) == "probe_suite"]
    for record in probe_rows.to_dict("records"):
        artifact_id = str(record.get("artifact_id") or "")
        if not artifact_id:
            continue
        try:
            artifact = dataset.load_artifact(artifact_id)
        except (FileNotFoundError, KeyError, ValueError):
            continue
        predictions = _probe_prediction_table(dataset, artifact)
        if predictions.empty:
            continue
        for row in predictions.to_dict("records"):
            rows.append(_probe_prediction_row(artifact, row))
    return _coerce_columns(pd.DataFrame.from_records(rows), PROBE_PREDICTION_COLUMNS)


def _probe_prediction_row(artifact: LensArtifact, row: Mapping[str, Any]) -> dict[str, Any]:
    confidence = _optional_float(row.get("confidence"))
    correct = _optional_bool(row.get("correct"))
    split = _optional_str(row.get("split") or row.get("eval_split")) or ""
    target = str(artifact.metrics.get("target") or artifact.display.get("target") or "")
    return {
        "probe_id": artifact.artifact_id,
        "probe_name": artifact.name,
        "target": target,
        "trace_id": str(row.get("trace_id") or ""),
        "episode_id": _optional_str(row.get("episode_id")),
        "task_id": _optional_str(row.get("task_id")),
        "split": split,
        "split_category": _probe_split_category(split),
        "actual": _jsonable_scalar(row.get("actual", row.get("target_value"))),
        "predicted": _jsonable_scalar(row.get("predicted", row.get("prediction_value"))),
        "confidence": confidence,
        "correct": correct,
        "correct_rate": None if correct is None else float(bool(correct)),
        "model": _optional_str(row.get("model")),
        "feature": _optional_str(row.get("feature")),
        "policy_call_index": _optional_int(row.get("policy_call_index")),
        "timestep": _optional_int(row.get("timestep", row.get("target_timestep"))),
        "generation_step": _jsonable_scalar(row.get("generation_step")),
    }


def _probe_prediction_table(dataset: TraceDataset, artifact: LensArtifact) -> pd.DataFrame:
    outputs = artifact.method.get("outputs") if isinstance(artifact.method, Mapping) else None
    relative_path = outputs.get("predictions") if isinstance(outputs, Mapping) else None
    if not relative_path:
        return pd.DataFrame()
    path = dataset.root / str(relative_path)
    if not path.exists():
        path = dataset._dataset_artifact_root() / str(relative_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _index_manifest(
    root: Path,
    *,
    dataset_fingerprint: str,
    episode_index: pd.DataFrame,
    model_site_index: pd.DataFrame,
    artifact_index: pd.DataFrame,
    probe_predictions: pd.DataFrame,
) -> dict[str, Any]:
    tables = {
        "episode_index": {"path": str(EPISODE_INDEX), "rows": int(len(episode_index))},
        "model_site_index": {"path": str(MODEL_SITE_INDEX), "rows": int(len(model_site_index))},
        "artifact_index": {"path": str(ARTIFACT_INDEX), "rows": int(len(artifact_index))},
        "probe_predictions": {"path": str(PROBE_PREDICTIONS), "rows": int(len(probe_predictions))},
    }
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "root": str(root),
        "dataset_fingerprint": dataset_fingerprint,
        "indexed_episode_count": int(len(episode_index)),
        "created_utc": datetime.now(UTC).isoformat(),
        "updated_utc": datetime.now(UTC).isoformat(),
        "tables": tables,
    }


def _dataset_fingerprint(root: Path, episode_index: pd.DataFrame) -> str:
    payload = {
        "root": str(root),
        "episodes": episode_index[["trace_id", "episode_index", "length"]].to_dict("records")
        if not episode_index.empty
        else [],
        "source": _source_file_signature(root),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_file_signature(root: Path) -> list[dict[str, Any]]:
    paths: list[Path] = []
    for pattern in (
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.jsonl",
        "meta/tasks.parquet",
        "meta/episodes/**/*.parquet",
        "vla_lens/overlay.json",
        "vla_lens/tables/episode_refs.parquet",
        "vla_lens/episodes/*/manifest.json",
        "*/meta/info.json",
        "*/meta/stats.json",
        "*/meta/tasks.jsonl",
        "*/meta/tasks.parquet",
        "*/meta/episodes/**/*.parquet",
        "*/vla_lens/overlay.json",
        "*/vla_lens/tables/episode_refs.parquet",
        "*/vla_lens/episodes/*/manifest.json",
    ):
        paths.extend(root.glob(pattern))
    records = []
    for path in sorted(set(paths)):
        if not path.exists() or path.is_dir():
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        records.append(
            {
                "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return records


def _declared_lerobot_episode_count(root: Path) -> int | None:
    info_path = root / LEROBOT_INFO_PATH
    if not info_path.exists():
        return None
    try:
        info = _read_json(info_path)
    except Exception:
        return None
    for key in ("total_episodes", "episodes"):
        value = info.get(key)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _validate_manifest_tables(root: Path, manifest: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    tables = manifest.get("tables")
    if not isinstance(tables, Mapping):
        raise DatasetIndexError(_rebuild_message(root, "Dataset index manifest has no tables map."))
    expected_tables: Mapping[str, tuple[Path, Sequence[str]]] = {
        "episode_index": (EPISODE_INDEX, REQUIRED_EPISODE_COLUMNS),
        "model_site_index": (MODEL_SITE_INDEX, MODEL_SITE_COLUMNS),
        "artifact_index": (ARTIFACT_INDEX, ARTIFACT_COLUMNS),
        "probe_predictions": (PROBE_PREDICTIONS, PROBE_PREDICTION_COLUMNS),
    }
    frames: dict[str, pd.DataFrame] = {}
    for table_name, (expected_path, required_columns) in expected_tables.items():
        table_info = tables.get(table_name)
        if not isinstance(table_info, Mapping):
            raise DatasetIndexError(
                _rebuild_message(root, f"Dataset index manifest missing {table_name}.")
            )
        manifest_path = Path(str(table_info.get("path") or ""))
        if manifest_path != expected_path:
            raise DatasetIndexError(
                _rebuild_message(
                    root,
                    f"Dataset index table path mismatch for {table_name}: {manifest_path}",
                )
            )
        table_path = root / manifest_path
        if not table_path.exists():
            raise DatasetIndexError(
                _rebuild_message(root, f"Missing index table: {manifest_path}.")
            )
        try:
            frame = pd.read_parquet(table_path)
        except Exception as exc:
            raise DatasetIndexError(
                _rebuild_message(root, f"Unreadable index table: {manifest_path}.")
            ) from exc
        missing = [column for column in required_columns if column not in frame]
        if missing:
            raise DatasetIndexError(
                _rebuild_message(
                    root,
                    f"{table_name} missing columns: {', '.join(missing)}",
                )
            )
        try:
            manifest_rows = int(table_info.get("rows", -1))
        except (TypeError, ValueError):
            manifest_rows = -1
        actual_rows = int(len(frame))
        if manifest_rows != actual_rows:
            reason = (
                f"{table_name} row count mismatch: "
                f"manifest={manifest_rows} actual={actual_rows}"
            )
            raise DatasetIndexError(
                _rebuild_message(
                    root,
                    reason,
                )
            )
        frames[table_name] = frame
    return frames


def _merge_append_table(
    existing: pd.DataFrame,
    new: pd.DataFrame,
    *,
    key_columns: Sequence[str],
    columns: Sequence[str],
) -> pd.DataFrame:
    existing = _coerce_columns(existing, columns)
    new = _coerce_columns(new, columns)
    if new.empty:
        if existing.empty:
            return existing
        return existing.sort_values(list(key_columns)).reset_index(drop=True)
    if existing.empty:
        return new.sort_values(list(key_columns)).reset_index(drop=True)
    out = pd.concat([existing, new], ignore_index=True, sort=False)
    out = out.drop_duplicates(subset=list(key_columns), keep="last")
    return _coerce_columns(out, columns).sort_values(list(key_columns)).reset_index(drop=True)


def _write_table(path: Path, frame: pd.DataFrame, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _coerce_columns(frame, columns).to_parquet(path, index=False)


def _read_existing_table(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        return _empty_table(columns)
    try:
        return _coerce_columns(pd.read_parquet(path), columns)
    except Exception:
        return _empty_table(columns)


def _coerce_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy() if frame is not None and not frame.empty else pd.DataFrame()
    for column in columns:
        if column not in out:
            out[column] = pd.Series(dtype=object)
    return out.loc[:, list(columns)].copy()


def _empty_table(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype=object) for column in columns})


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_dumps(value: Any) -> str:
    return json.dumps(_jsonable_scalar(value), sort_keys=True, separators=(",", ":"))


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _jsonable_scalar(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable_scalar(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable_scalar(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if pd.isna(value) if np.isscalar(value) and not isinstance(value, str) else False:
        return None
    return value


def _table_column_values(frame: pd.DataFrame, column: str) -> list[str]:
    if frame.empty or column not in frame:
        return []
    return sorted(str(value) for value in frame[column].dropna().unique() if str(value).strip())


def _first_text(metadata: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    text = str(value)
    return None if text.lower() in {"nan", "none", "null", ""} else text


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    if isinstance(value, float) and np.isfinite(value):
        return bool(value)
    return None


def _probe_split_category(split: str) -> str:
    text = str(split or "").strip().lower().replace("-", "_")
    if text in {"train", "training"}:
        return "train"
    if text.startswith("test"):
        return "test"
    if text.startswith("val") or text in {"valid", "validation"}:
        return "validation"
    if "heldout" in text or "held_out" in text:
        return "validation"
    return "unknown"


def _missing_index_message(root: Path) -> str:
    return _rebuild_message(root, "Dataset index is missing.")


def _rebuild_message(root: Path, reason: str) -> str:
    return (
        f"{reason} Build the local dashboard index with: "
        f"uv run python scripts/build_vla_lens_index.py {root} --overwrite"
    )


__all__ = [
    "ARTIFACT_INDEX",
    "DatasetIndexError",
    "EPISODE_INDEX",
    "INDEX_MANIFEST",
    "INDEX_SCHEMA_VERSION",
    "INDEX_TABLE_DIR",
    "IndexBuildResult",
    "MODEL_SITE_INDEX",
    "PROBE_PREDICTIONS",
    "REQUIRED_EPISODE_COLUMNS",
    "build_dataset_index",
    "index_manifest_path",
    "index_root",
    "validate_dataset_index",
]
