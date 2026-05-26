"""Validation workbench primitives."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import zarr

from vla_lens.traces import TraceDataset
from vla_lens.validation import validate_trace_dataset
from vla_lens.workbench.schema import (
    CohortSpec,
    LensArraySpec,
    StorageRef,
    normalize_axis_name,
)
from vla_lens.workbench.tables import (
    _filter_table,
)
from vla_lens.workbench.utils import (
    _first_scalar,
    _jsonable_scalar,
    _optional_str,
    _parse_shape,
)


def validate_workbench_contracts(dataset: TraceDataset) -> dict[str, Any]:
    """Validate core workbench composition contracts."""
    import vla_lens.workbench as public_workbench

    axes = {axis.name for axis in public_workbench.axis_registry(dataset)}
    arrays = public_workbench.lens_array_catalog(dataset)
    panels = public_workbench.panel_registry()
    invalid_array_dims = [
        {"array_id": array.array_id, "dims": sorted(set(array.dims) - axes)}
        for array in arrays
        if set(array.dims) - axes
    ]
    invalid_panel_axes: list[dict[str, Any]] = []
    for panel_type, entry in panels.items():
        for field_name, values in [
            ("emits", entry.recipe.emits),
            ("responds_to", entry.recipe.responds_to),
            ("selection_axes", entry.selection_axes),
        ]:
            axes_in_field = {
                normalize_axis_name(str(value).removeprefix("selection.")) for value in values
            }
            missing = sorted(axes_in_field - axes)
            if missing:
                invalid_panel_axes.append(
                    {"panel_type": panel_type, "field": field_name, "axes": missing}
                )
    invalid_workflow_panels = [
        {"workflow_id": workflow["workflow_id"], "panels": missing}
        for workflow in public_workbench.workflow_presets(dataset)
        if (
            missing := sorted(set(str(panel) for panel in workflow.get("panels", ())) - set(panels))
        )
    ]
    invalid_storage = _invalid_storage_refs(dataset, arrays)
    invalid_tables = _invalid_table_refs(dataset)
    invalid_media = _invalid_media_refs(dataset)
    invalid_analysis_outputs = _invalid_analysis_run_outputs(dataset)
    trace_validation = _storage_contract_validation(dataset)
    resolver_keys = {
        "examples",
        "lens_arrays",
        "suggested_panels",
        "provenance",
        "valid_references",
    }
    return {
        "valid": not (
            invalid_array_dims
            or invalid_panel_axes
            or invalid_workflow_panels
            or invalid_storage
            or invalid_tables
            or invalid_media
            or invalid_analysis_outputs
            or not trace_validation["valid"]
        ),
        "invalid_array_dims": invalid_array_dims,
        "invalid_panel_axes": invalid_panel_axes,
        "invalid_workflow_panels": invalid_workflow_panels,
        "invalid_storage": invalid_storage,
        "invalid_tables": invalid_tables,
        "invalid_media": invalid_media,
        "invalid_analysis_outputs": invalid_analysis_outputs,
        "trace_validation": trace_validation,
        "resolver_required_keys": sorted(resolver_keys),
    }

def _storage_contract_validation(dataset: TraceDataset) -> dict[str, Any]:
    lerobot_roots: set[Path] = set()
    for bundle in dataset.bundles:
        if dict(bundle.manifest.metadata or {}).get("robot_dataset_format") != "lerobot_v3":
            continue
        if hasattr(bundle, "root"):
            lerobot_roots.add(Path(bundle.root))
    if lerobot_roots:
        from vla_lens.capture.lerobot_v3 import validate_lerobot_v3_dataset

        results = [validate_lerobot_v3_dataset(root).to_dict() for root in sorted(lerobot_roots)]
        return {
            "valid": all(bool(result["valid"]) for result in results),
            "format": "lerobot_v3",
            "datasets": results,
        }
    return validate_trace_dataset(dataset).to_dict()

def _cohort_episode_frame(dataset: TraceDataset, cohort: CohortSpec) -> pd.DataFrame:
    frame = dataset.episode_index.copy()
    trace_ids = set(cohort.members.get("trace_id", ()))
    if trace_ids:
        return frame.loc[frame["trace_id"].astype(str).isin(trace_ids)].copy()
    filters = cohort.definition.get("filters") or cohort.filters
    return _filter_table(frame, filters if isinstance(filters, Mapping) else {})

def _cohort_delta_table(
    left: pd.DataFrame,
    right: pd.DataFrame,
    column: str,
) -> list[dict[str, Any]]:
    if column not in left and column not in right:
        return []
    left_counts = (
        left[column].astype(str).value_counts() if column in left else pd.Series(dtype=int)
    )
    right_counts = (
        right[column].astype(str).value_counts() if column in right else pd.Series(dtype=int)
    )
    keys = sorted(set(left_counts.index.astype(str)) | set(right_counts.index.astype(str)))
    rows: list[dict[str, Any]] = []
    for key in keys:
        left_count = int(left_counts.get(key, 0))
        right_count = int(right_counts.get(key, 0))
        rows.append(
            {
                column: key,
                "left_count": left_count,
                "right_count": right_count,
                "delta_count": left_count - right_count,
            }
        )
    return rows

def _graph_edge(source: str, target: str, edge_type: str) -> dict[str, Any]:
    import vla_lens.workbench as public_workbench

    causal = next(
        (
            item["causal"]
            for item in public_workbench.graph_edge_types()
            if item["edge_type"] == edge_type
        ),
        False,
    )
    return {
        "source": source,
        "target": target,
        "edge_type": edge_type,
        "causal": bool(causal),
    }

def _invalid_storage_refs(
    dataset: TraceDataset,
    arrays: Sequence[LensArraySpec],
) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    for array in arrays:
        if array.kind in {"tensor", "artifact_array"}:
            if array.storage.format == "parquet_column":
                path = _storage_path(dataset, array)
                if path is None or not path.exists():
                    invalid.append(
                        {
                            "array_id": array.array_id,
                            "reason": "missing_storage",
                            "uri": array.storage.uri,
                        }
                    )
                continue
            if array.storage.format != "zarr":
                invalid.append(
                    {
                        "array_id": array.array_id,
                        "reason": "dense_array_not_zarr",
                        "format": array.storage.format,
                    }
                )
                continue
            path = _storage_path(dataset, array)
            if path is None or not path.exists():
                invalid.append(
                    {
                        "array_id": array.array_id,
                        "reason": "missing_storage",
                        "uri": array.storage.uri,
                    }
                )
                continue
            try:
                zarr.open_array(str(path), mode="r")
            except Exception as exc:  # pragma: no cover - defensive validation detail
                invalid.append(
                    {
                        "array_id": array.array_id,
                        "reason": "unreadable_zarr",
                        "error": repr(exc),
                    }
                )
        elif array.kind == "image_sequence":
            if array.storage.format == "mp4":
                path = _storage_path(dataset, array)
                if path is None or not path.exists():
                    invalid.append(
                        {
                            "array_id": array.array_id,
                            "reason": "missing_storage",
                            "uri": array.storage.uri,
                        }
                    )
                continue
            if array.storage.format != "jpeg":
                invalid.append(
                    {
                        "array_id": array.array_id,
                        "reason": "image_sequence_not_jpeg",
                        "format": array.storage.format,
                    }
                )
                continue
            path = _storage_path(dataset, array)
            if path is None or not path.exists():
                invalid.append(
                    {
                        "array_id": array.array_id,
                        "reason": "missing_storage",
                        "uri": array.storage.uri,
                    }
                )
                continue
            if not list(path.glob("*.jpg")):
                invalid.append(
                    {
                        "array_id": array.array_id,
                        "reason": "empty_jpeg_sequence",
                        "uri": array.storage.uri,
                    }
                )
    return invalid

def _invalid_analysis_run_outputs(dataset: TraceDataset) -> list[dict[str, Any]]:
    from vla_lens.workbench.api import list_analysis_runs

    invalid: list[dict[str, Any]] = []
    for run in list_analysis_runs(dataset):
        artifact_id = str(run.provenance.get("artifact_id") or run.run_id)
        try:
            artifact = dataset.load_artifact(artifact_id)
        except (FileNotFoundError, KeyError, ValueError):
            continue
        missing = sorted(set(run.outputs) - set(artifact.arrays))
        if missing:
            invalid.append({"run_id": run.run_id, "missing_outputs": missing})
    return invalid

def _invalid_table_refs(dataset: TraceDataset) -> list[dict[str, Any]]:
    import vla_lens.workbench as public_workbench

    invalid: list[dict[str, Any]] = []
    for table in public_workbench.table_catalog(dataset):
        if table.storage.format != "parquet":
            invalid.append(
                {
                    "table_id": table.table_id,
                    "reason": "table_not_parquet",
                    "format": table.storage.format,
                }
            )
            continue
        if table.storage.relative_to == "dataset":
            paths = list(dataset.root.glob(table.storage.uri))
            if not paths:
                invalid.append(
                    {
                        "table_id": table.table_id,
                        "reason": "missing_table",
                        "uri": table.storage.uri,
                    }
                )
        elif table.storage.relative_to == "bundle":
            if not any((bundle.path / table.storage.uri).exists() for bundle in dataset.bundles):
                invalid.append(
                    {
                        "table_id": table.table_id,
                        "reason": "missing_table",
                        "uri": table.storage.uri,
                    }
                )
    return invalid

def _invalid_media_refs(dataset: TraceDataset) -> list[dict[str, Any]]:
    import vla_lens.workbench as public_workbench

    invalid: list[dict[str, Any]] = []
    for frame in public_workbench.image_frame_catalog(dataset):
        if frame.storage.format == "mp4":
            path = _storage_path_for_ref(dataset, frame.storage, trace_id=frame.trace_id)
            if path is None or not path.exists():
                invalid.append(
                    {
                        "frame_id": frame.frame_id,
                        "reason": "missing_frame_storage",
                        "uri": frame.storage.uri,
                    }
                )
            continue
        if frame.storage.format != "jpeg":
            invalid.append(
                {
                    "frame_id": frame.frame_id,
                    "reason": "image_frame_not_jpeg",
                    "format": frame.storage.format,
                }
            )
            continue
        path = _storage_path_for_ref(dataset, frame.storage, trace_id=frame.trace_id)
        if path is None or not path.exists():
            invalid.append(
                {
                    "frame_id": frame.frame_id,
                    "reason": "missing_frame_storage",
                    "uri": frame.storage.uri,
                }
            )
            continue
        if not list(path.glob("*.jpg")):
            invalid.append(
                {
                    "frame_id": frame.frame_id,
                    "reason": "empty_jpeg_sequence",
                    "uri": frame.storage.uri,
                }
            )
    for media in public_workbench.media_catalog(dataset):
        if media.storage.format not in {"jpeg", "mp4"}:
            invalid.append(
                {
                    "media_id": media.media_id,
                    "reason": "unsupported_media_format",
                    "format": media.storage.format,
                }
            )
            continue
        path = _storage_path_for_ref(
            dataset,
            media.storage,
            trace_id=_optional_str(media.provenance.get("trace_id")),
        )
        if path is None or not path.exists():
            invalid.append(
                {
                    "media_id": media.media_id,
                    "reason": "missing_media_storage",
                    "uri": media.storage.uri,
                }
            )
    return invalid

def _storage_path(dataset: TraceDataset, array: LensArraySpec) -> Path | None:
    return _storage_path_for_ref(
        dataset,
        array.storage,
        trace_id=_optional_str(array.provenance.get("trace_id")),
    )

def _storage_path_for_ref(
    dataset: TraceDataset,
    storage: StorageRef,
    *,
    trace_id: str | None = None,
) -> Path | None:
    if storage.relative_to == "bundle":
        if trace_id:
            return dataset.bundle(str(trace_id)).path / storage.uri
        return None
    dataset_path = dataset.root / storage.uri
    if dataset_path.exists():
        return dataset_path
    artifact_root_path = dataset._dataset_artifact_root() / storage.uri
    if artifact_root_path.exists():
        return artifact_root_path
    return dataset_path

def _storage_ref_from_row(row: Mapping[str, Any]) -> StorageRef:
    return StorageRef(
        format=str(row.get("storage_format") or "zarr"),
        uri=str(row.get("relative_path")),
        relative_to="bundle",
        chunks=tuple(_parse_shape(row.get("chunks"))),
        compression=_optional_str(row.get("compression")) or "zstd",
    )

def _axis_index(value: Any, *, default: int) -> int:
    scalar = _first_scalar(value)
    if scalar is None:
        return default
    if isinstance(scalar, str) and not scalar.lstrip("-").isdigit():
        return default
    return max(0, int(scalar))

def _array_value(array: np.ndarray, *indexes: int) -> Any:
    if len(indexes) != array.ndim:
        return None
    bounded = tuple(max(0, min(array.shape[axis] - 1, index)) for axis, index in enumerate(indexes))
    return _jsonable_scalar(array[bounded])
