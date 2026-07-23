"""Utils workbench primitives."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.traces import TraceBundle, TraceDataset
from vla_lens.workbench.schema import (
    LensArraySpec,
    LensDataKind,
)


def _array_names(dataset: TraceDataset) -> set[str]:
    names: set[str] = set()
    for bundle in dataset.bundles:
        table = bundle.array_index
        if not table.empty and "name" in table:
            names.update(str(value) for value in table["name"].dropna())
    return names

def _array_episode_count(dataset: TraceDataset, name: str) -> int:
    count = 0
    for bundle in dataset.bundles:
        table = bundle.array_index
        if not table.empty and "name" in table:
            count += int((table["name"].astype(str) == name).any())
    return count

def _activation_axes(index: pd.DataFrame) -> list[str]:
    axes: set[str] = set()
    if index.empty or "axes" not in index:
        return []
    for value in index["axes"].dropna():
        axes.update(str(item) for item in _parse_axes(value))
    return sorted(axes)

def _label_columns(
    episode_index: pd.DataFrame,
    timestep_index: pd.DataFrame,
    array_names: set[str],
) -> dict[str, list[str]]:
    episode_candidates = [
        "task_id",
        "prompt",
        "outcome",
        "success",
        "target_object",
        "object_label",
        "benchmark",
        "env_id",
        "robot_id",
        "scene_id",
        "layout_id",
    ]
    timestep_candidates = [
        "phase",
        "contact",
        "grasp",
        "lift",
        "reward",
        "done",
        "policy_call_index",
    ]
    array_label_names = sorted(
        name
        for name in array_names
        if any(part in name for part in ["object", "contact", "phase", "reward", "done"])
    )
    return {
        "episode": [column for column in episode_candidates if column in episode_index],
        "timestep": [column for column in timestep_candidates if column in timestep_index]
        + array_label_names,
    }

def _unique_column(frame: pd.DataFrame, column: str) -> list[Any]:
    if frame.empty or column not in frame:
        return []
    return sorted(str(value) for value in frame[column].dropna().unique())

def _selection_to_slices(
    spec: LensArraySpec,
    shape: Sequence[int],
    selection: Mapping[str, Any],
) -> tuple[Any, ...]:
    slices: list[Any] = []
    for dim, size in zip(spec.dims, shape, strict=False):
        if dim not in selection:
            slices.append(slice(None))
            continue
        slices.append(_axis_selector(selection[dim], int(size), coords=spec.coords.get(dim)))
    return tuple(slices)

def _axis_selector(value: Any, size: int, *, coords: Any = None) -> Any:
    if size <= 0:
        return slice(0, 0)
    if isinstance(value, Mapping):
        start = _coord_index(value.get("start", 0), size, coords, default=0)
        end = _coord_index(value.get("end", value.get("start", start)), size, coords, default=start)
        step = int(value.get("step", 1))
        return slice(max(0, start), min(size, end + 1), max(1, step))
    if isinstance(value, (list, tuple)):
        indexes = [_coord_index(item, size, coords, default=0) for item in value]
        return [max(0, min(size - 1, item)) for item in indexes]
    index = _coord_index(value, size, coords, default=0)
    return max(0, min(size - 1, index))

def _coord_index(value: Any, size: int, coords: Any, *, default: int) -> int:
    """Map semantic coordinate values to positional indexes when coords exist."""
    if value is None:
        return default
    if isinstance(coords, Sequence) and not isinstance(coords, (str, bytes, bytearray)):
        for index, coord in enumerate(coords):
            if str(coord) == str(value):
                return index
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _slice_payload(slices: Sequence[Any]) -> list[Any]:
    payload: list[Any] = []
    for item in slices:
        if isinstance(item, slice):
            payload.append({"start": item.start, "stop": item.stop, "step": item.step})
        elif isinstance(item, list):
            payload.append(item)
        else:
            payload.append(int(item))
    return payload

def _preview_slices(shape: Sequence[int], *, max_values: int) -> tuple[slice, ...]:
    if not shape:
        return ()
    remaining = max(1, int(max_values))
    slices: list[slice] = []
    for size in shape:
        width = min(int(size), max(1, remaining))
        slices.append(slice(0, width))
        remaining = max(1, remaining // max(1, width))
    return tuple(slices)

def _numeric_summary(value: np.ndarray) -> dict[str, Any]:
    if value.size == 0 or not np.issubdtype(value.dtype, np.number):
        return {}
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        return {}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }

def _jsonable_array(value: np.ndarray) -> Any:
    return np.asarray(value).tolist()

def _workbench_dir(dataset: TraceDataset, name: str, *, create: bool) -> Path:
    base = (
        dataset.root / "vla_lens"
        if (dataset.root / "meta" / "info.json").exists() and (dataset.root / "data").exists()
        else dataset.root
    )
    root = base / "workbench" / name
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root

def _safe_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    return safe.strip("._") or "workbench_item"

def _merge_axis_value(existing: Any, incoming: Any) -> Any:
    if existing == incoming:
        return existing
    if isinstance(existing, Mapping) or isinstance(incoming, Mapping):
        return incoming
    values: list[Any] = []
    for item in (existing, incoming):
        if isinstance(item, (list, tuple, set, frozenset)):
            values.extend(item)
        else:
            values.append(item)
    deduped: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key not in seen:
            seen.add(key)
            deduped.append(value)
    return deduped

def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Workbench record must be an object: {path}")
    return payload

def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace one JSON record atomically so readers never observe partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

def _kind_for_episode_array(name: str) -> LensDataKind:
    if name.startswith("frames.") or name.startswith("observation.images."):
        return "image_sequence"
    return "tensor"

def _axis_names_for_array(axes: Sequence[str]) -> list[str]:
    mapping = {
        "time": "timestep",
        "timestep": "timestep",
        "call": "policy_call",
        "call_index": "policy_call",
        "camera": "camera",
        "patch": "image_patch",
        "token": "token",
        "layer": "layer",
        "generation_step": "generation_step",
        "horizon": "action_horizon",
        "action_dim": "action_dim",
        "dim": "action_dim",
        "state": "state_component",
        "state_dim": "state_component",
        "feature": "unit",
        "hidden": "unit",
        "channel": "unit",
        "object": "object",
    }
    return [mapping.get(str(axis), str(axis)) for axis in axes]

def _axis_names() -> set[str]:
    return {
        "episode",
        "timestep",
        "policy_call",
        "camera",
        "image_patch",
        "height",
        "width",
        "rgb",
        "xyz",
        "quat",
        "pose_component",
        "matrix_row",
        "matrix_col",
        "joint",
        "gripper_joint",
        "gripper_component",
        "state_component",
        "predicate",
        "module",
        "layer",
        "token_kind",
        "token",
        "unit",
        "generation_step",
        "action_horizon",
        "action_dim",
        "object",
        "label",
        "cohort",
        "metric",
        "prediction_status",
        "example",
        "cell",
        "axis_range",
        "image_xy",
        "point",
        "node",
        "edge",
        "projection_x",
        "projection_y",
        "analysis_run",
    }

def _coords_for_array(
    bundle: TraceBundle,
    dims: Sequence[str],
    shape: Sequence[int],
) -> dict[str, Any]:
    coords: dict[str, Any] = {}
    for dim, size in zip(dims, shape, strict=False):
        if dim == "episode":
            coords[dim] = [bundle.manifest.trace_id]
        elif size <= 256:
            coords[dim] = list(range(int(size)))
        else:
            coords[dim] = {"start": 0, "stop": int(size), "step": 1}
    return coords

def _artifact_array_dims(name: str) -> list[str]:
    if name in {"metric_cube", "baseline_cube", "delta_cube"}:
        return ["layer", "timestep", "token_kind"]
    if name in {"delta_to_final", "step_delta"}:
        return ["episode", "policy_call", "generation_step", "action_horizon"]
    if name == "final_vs_executed":
        return ["episode", "policy_call", "action_horizon", "action_dim"]
    if "commitment" in name:
        return ["episode", "policy_call", "generation_step"]
    if "executed" in name or "predicted" in name:
        return ["episode", "policy_call"]
    if "margin" in name or "score" in name:
        return ["layer", "timestep"]
    return []

def _artifact_array_coords(artifact: Any, name: str, shape: Sequence[int]) -> dict[str, Any]:
    dims = _artifact_array_dims(name)
    coords: dict[str, Any] = {}
    display = getattr(artifact, "display", {}) or {}
    axes = display.get("axes") if isinstance(display, Mapping) else None
    if isinstance(axes, Mapping):
        for dim, size in zip(dims, shape, strict=False):
            values = axes.get(dim)
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
                coords[dim] = list(values)[: int(size)]
    for dim, size in zip(dims, shape, strict=False):
        coords.setdefault(
            dim,
            list(range(int(size)))
            if int(size) <= 256
            else {
                "start": 0,
                "stop": int(size),
                "step": 1,
            },
        )
    return coords

def _parse_axes(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []

def _parse_shape(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [int(item) for item in parsed]
    return []

def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _optional_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value)
    return None if text == "nan" else text

def _json_loads(value: Any, *, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default

def _jsonable_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable_scalar(value) for key, value in record.items()}

def _jsonable_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return _jsonable_scalar(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value

def _as_set(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        if "start" in value and "end" in value:
            return {str(value["start"])}
        return {str(item) for item in value.values()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(item) for item in value}
    return {str(value)}

def _first_scalar(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("start") if "start" in value else next(iter(value.values()), None)
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value
