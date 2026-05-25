"""One-time migration helpers for legacy ``.vlatrace`` datasets."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.capture.lerobot_v3 import (
    LEROBOT_ACTION,
    LEROBOT_IMAGE_PREFIX,
    LEROBOT_OBSERVATION_STATE,
    validate_lerobot_v3_dataset,
)
from vla_lens.capture.records import TraceRecord
from vla_lens.lerobot_dataset import write_lerobot_trace_record
from vla_lens.lerobot_dataset_common import (
    LEGACY_ACTION_ARRAY,
    LEGACY_FRAME_PREFIX,
    LEROBOT_DATA_PATH_TEMPLATE,
    _chunk_file_index,
    _read_table,
)
from vla_lens.traces import (
    ArraySpec,
    TraceBundle,
    TraceDataset,
    TraceManifest,
    _compute_trace_fingerprints,
)

MIGRATION_VERSION = 1
MIGRATION_STATUS_PATH = Path("vla_lens") / "migration" / "vlatrace_to_lerobot.jsonl"


@dataclass(frozen=True, slots=True)
class MigrationResult:
    """Summary of a migrated legacy trace."""

    episode_index: int
    output_root: Path
    source_path: Path
    trace_id: str


def discover_vlatrace_bundles(source: str | Path) -> list[Path]:
    """Return legacy bundle directories below ``source`` in deterministic order."""

    root = Path(source)
    if (root / TraceBundle.MANIFEST).exists() and root.suffix == ".vlatrace":
        return [root]
    return sorted(
        path
        for path in root.rglob("*.vlatrace")
        if path.is_dir() and (path / TraceBundle.MANIFEST).exists()
    )


def trace_bundle_to_robot_record(
    bundle: TraceBundle,
    *,
    source_root: str | Path | None = None,
) -> TraceRecord:
    """Build the robot-layer ``TraceRecord`` needed by the LeRobot writer.

    The returned record intentionally contains only arrays required to write the
    canonical LeRobot robot layer. The interpretability overlay is copied from
    the source bundle separately so large activation arrays are not rewritten.
    """

    source_root_path = Path(source_root).resolve() if source_root is not None else None
    metadata = dict(bundle.manifest.metadata)
    metadata["source_format"] = "vlatrace"
    metadata["migration_version"] = MIGRATION_VERSION
    metadata["migration_tool"] = "vlatrace_to_lerobot"
    metadata["migrated_utc"] = datetime.now(UTC).isoformat()
    if source_root_path is not None:
        try:
            relative_source_path = bundle.path.resolve().relative_to(source_root_path)
            metadata["source_vlatrace_path"] = str(relative_source_path)
        except ValueError:
            metadata["source_vlatrace_path"] = str(bundle.path)
    else:
        metadata["source_vlatrace_path"] = str(bundle.path)

    manifest = TraceManifest(
        trace_id=bundle.manifest.trace_id,
        episode_id=bundle.manifest.episode_id,
        task_id=bundle.manifest.task_id,
        prompt=bundle.manifest.prompt,
        model_id=bundle.manifest.model_id,
        env_id=bundle.manifest.env_id,
        robot_id=bundle.manifest.robot_id,
        outcome=bundle.manifest.outcome,
        length=bundle.manifest.length,
        schema_version=bundle.manifest.schema_version,
        metadata=metadata,
    )
    return TraceRecord(
        manifest=manifest,
        timesteps=bundle.timesteps,
        episode_arrays=_robot_layer_arrays(bundle),
        capture_request=_read_json(bundle.path / TraceBundle.CAPTURE_REQUEST),
        capture_plan=_read_json(bundle.path / TraceBundle.CAPTURE_PLAN),
        capture_report={
            **_read_json(bundle.path / TraceBundle.CAPTURE_REPORT),
            "source_format": "vlatrace",
            "migration_version": MIGRATION_VERSION,
        },
    )


def migrate_vlatrace_bundle(
    source_bundle_path: str | Path,
    output_root: str | Path,
    *,
    source_root: str | Path | None = None,
    overwrite: bool = False,
) -> MigrationResult:
    """Migrate one legacy bundle into a LeRobot v3 root plus VLA overlay."""

    bundle = TraceBundle.open(source_bundle_path)
    record = trace_bundle_to_robot_record(bundle, source_root=source_root)
    written = write_lerobot_trace_record(record, output_root, overwrite=overwrite)
    episode_index = int(written.episode_index)
    overlay_bundle = written.overlay_bundle
    if overlay_bundle is None:
        raise RuntimeError(f"LeRobot writer did not create an overlay for {bundle.path}")
    chunk_index, file_index = _chunk_file_index(episode_index)
    data_path = Path(
        LEROBOT_DATA_PATH_TEMPLATE.format(chunk_index=chunk_index, file_index=file_index)
    )
    _copy_pruned_overlay_bundle(
        bundle,
        overlay_bundle.path,
        episode_index=episode_index,
        data_path=data_path,
    )
    result = MigrationResult(
        episode_index=episode_index,
        output_root=Path(output_root),
        source_path=Path(source_bundle_path),
        trace_id=bundle.manifest.trace_id,
    )
    _append_migration_status(Path(output_root), result)
    return result


def migrate_vlatrace_dataset(
    source_root: str | Path,
    output_root: str | Path,
    *,
    limit: int | None = None,
    overwrite_root: bool = False,
    resume: bool = True,
) -> list[MigrationResult]:
    """Migrate a legacy dataset root into one LeRobot v3 dataset root."""

    source = Path(source_root)
    output = Path(output_root)
    if overwrite_root and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    bundle_paths = discover_vlatrace_bundles(source)
    if limit is not None:
        bundle_paths = bundle_paths[: max(0, int(limit))]
    completed = _completed_trace_ids(output) if resume else set()
    results: list[MigrationResult] = []
    for path in bundle_paths:
        trace_id = TraceBundle.open(path).manifest.trace_id
        if trace_id in completed:
            continue
        results.append(migrate_vlatrace_bundle(path, output, source_root=source))
    copy_dataset_level_state(source, output)
    validation = validate_lerobot_v3_dataset(output)
    if not validation.valid:
        messages = "; ".join(issue.message for issue in validation.errors)
        raise ValueError(f"Migrated dataset failed LeRobot validation: {messages}")
    TraceDataset.open(output)
    return results


def copy_dataset_level_state(source_root: str | Path, output_root: str | Path) -> None:
    """Copy dataset-level artifacts and workbench state into ``vla_lens/``."""

    source = Path(source_root)
    overlay_root = Path(output_root) / "vla_lens"
    overlay_root.mkdir(parents=True, exist_ok=True)
    for name in ("artifacts", "workbench"):
        src = source / name
        if src.exists():
            _copytree_hardlink_or_copy(src, overlay_root / name)
    src_artifact_index = source / TraceBundle.ARTIFACT_INDEX
    if src_artifact_index.exists():
        dst_artifact_index = overlay_root / TraceBundle.ARTIFACT_INDEX
        dst_artifact_index.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_artifact_index, dst_artifact_index)


def _robot_layer_arrays(bundle: TraceBundle) -> dict[str, ArraySpec]:
    arrays: dict[str, ArraySpec] = {}
    for row in bundle.array_index.to_dict("records"):
        name = str(row.get("name") or "")
        if not _belongs_in_robot_record(name):
            continue
        arrays[name] = ArraySpec(
            array=bundle.array(name),
            axes=_json_sequence(row.get("axes")),
            metadata=_json_mapping(row.get("metadata")),
        )
    return arrays


def _belongs_in_robot_record(name: str) -> bool:
    return (
        name in {LEGACY_ACTION_ARRAY, LEROBOT_ACTION, LEROBOT_OBSERVATION_STATE}
        or name.startswith(LEGACY_FRAME_PREFIX)
        or name.startswith(LEROBOT_IMAGE_PREFIX)
        or name in {"robot_joint_pos", "eef_pos", "gripper_qpos"}
    )


def _copy_pruned_overlay_bundle(
    source: TraceBundle,
    destination: Path,
    *,
    episode_index: int,
    data_path: Path,
) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    _copytree_hardlink_or_copy(source.path, destination)
    _prune_robot_owned_overlay_arrays(destination)
    _update_overlay_manifest(destination, episode_index=episode_index, data_path=data_path)
    _update_overlay_capture_report(destination, episode_index=episode_index)
    _refresh_overlay_fingerprints(destination)


def _prune_robot_owned_overlay_arrays(bundle_path: Path) -> None:
    array_index_path = bundle_path / TraceBundle.ARRAY_INDEX
    array_index = _read_table(array_index_path)
    if array_index.empty or "name" not in array_index:
        return
    keep_rows = []
    for row in array_index.to_dict("records"):
        name = str(row.get("name") or "")
        relative_path = Path(str(row.get("relative_path") or ""))
        if _is_robot_owned_overlay_array(name):
            _remove_path(bundle_path / relative_path)
        else:
            keep_rows.append(row)
    if array_index_path.exists():
        array_index_path.unlink()
    pd.DataFrame.from_records(keep_rows).to_parquet(array_index_path, index=False)


def _is_robot_owned_overlay_array(name: str) -> bool:
    return (
        name in {LEGACY_ACTION_ARRAY, LEROBOT_ACTION, LEROBOT_OBSERVATION_STATE}
        or name.startswith(LEGACY_FRAME_PREFIX)
        or name.startswith(LEROBOT_IMAGE_PREFIX)
    )


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    for parent in path.parents:
        if parent.name in {"arrays", "media", "artifacts", "tables"}:
            break
        try:
            parent.rmdir()
        except OSError:
            break


def _copytree_hardlink_or_copy(source: Path, destination: Path) -> None:
    for root, dirs, files in os.walk(source):
        root_path = Path(root)
        relative = root_path.relative_to(source)
        target_root = destination / relative
        target_root.mkdir(parents=True, exist_ok=True)
        for directory in dirs:
            (target_root / directory).mkdir(exist_ok=True)
        for filename in files:
            src = root_path / filename
            dst = target_root / filename
            if dst.exists():
                dst.unlink()
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)


def _update_overlay_manifest(bundle_path: Path, *, episode_index: int, data_path: Path) -> None:
    path = bundle_path / TraceBundle.MANIFEST
    payload = _read_json(path)
    metadata = dict(payload.get("metadata") or {})
    metadata.update(
        {
            "robot_dataset_format": "lerobot_v3",
            "lerobot_episode_index": int(episode_index),
            "lerobot_data_path": str(data_path),
            "source_format": "vlatrace",
            "migration_version": MIGRATION_VERSION,
            "migration_tool": "vlatrace_to_lerobot",
        }
    )
    payload["metadata"] = metadata
    _write_json(path, payload)


def _update_overlay_capture_report(bundle_path: Path, *, episode_index: int) -> None:
    path = bundle_path / TraceBundle.CAPTURE_REPORT
    payload = _read_json(path)
    payload.update(
        {
            "dataset_format": "lerobot_v3_plus_vla_lens_overlay",
            "lerobot_episode_index": int(episode_index),
            "source_format": "vlatrace",
            "migration_version": MIGRATION_VERSION,
        }
    )
    _write_json(path, payload)


def _refresh_overlay_fingerprints(bundle_path: Path) -> None:
    manifest = TraceBundle.open(bundle_path).manifest
    fingerprints = _compute_trace_fingerprints(bundle_path, manifest=manifest)
    _write_json(bundle_path / TraceBundle.FINGERPRINTS, fingerprints)
    manifest_path = bundle_path / TraceBundle.MANIFEST
    manifest_payload = _read_json(manifest_path)
    manifest_metadata = dict(manifest_payload.get("metadata") or {})
    manifest_metadata["fingerprints"] = fingerprints
    manifest_payload["metadata"] = manifest_metadata
    _write_json(manifest_path, manifest_payload)
    report_path = bundle_path / TraceBundle.CAPTURE_REPORT
    report_payload = _read_json(report_path)
    report_payload["fingerprints"] = fingerprints
    _write_json(report_path, report_payload)


def _completed_trace_ids(output_root: Path) -> set[str]:
    refs = _read_table(output_root / "vla_lens" / "tables" / "episode_refs.parquet")
    if refs.empty or "trace_id" not in refs:
        return set()
    return {str(value) for value in refs["trace_id"]}


def _append_migration_status(output_root: Path, result: MigrationResult) -> None:
    path = output_root / MIGRATION_STATUS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episode_index": result.episode_index,
        "source_path": str(result.source_path),
        "trace_id": result.trace_id,
        "version": MIGRATION_VERSION,
        "written_utc": datetime.now(UTC).isoformat(),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _json_sequence(value: Any) -> Sequence[str]:
    decoded = _decode_json_cell(value, default=[])
    if isinstance(decoded, Sequence) and not isinstance(decoded, (str, bytes)):
        return [str(item) for item in decoded]
    return []


def _json_mapping(value: Any) -> Mapping[str, Any]:
    decoded = _decode_json_cell(value, default={})
    return decoded if isinstance(decoded, Mapping) else {}


def _decode_json_cell(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    if isinstance(value, (Mapping, list, tuple)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return default
    return default


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


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
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


__all__ = [
    "MIGRATION_STATUS_PATH",
    "MIGRATION_VERSION",
    "MigrationResult",
    "copy_dataset_level_state",
    "discover_vlatrace_bundles",
    "migrate_vlatrace_bundle",
    "migrate_vlatrace_dataset",
    "trace_bundle_to_robot_record",
]
