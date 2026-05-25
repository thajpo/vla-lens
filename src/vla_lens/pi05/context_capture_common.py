"""Shared PI0.5 context capture utilities."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class _Status:
    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def available(
        self,
        component: str,
        field: str,
        source: str,
        *,
        shape: Sequence[int] | None = None,
    ) -> None:
        self._records.append(
            {
                "component": component,
                "field": field,
                "available": True,
                "reason": "",
                "source": source,
                "shape": _shape_text(shape),
            }
        )

    def missing(self, component: str, field: str, reason: str) -> None:
        self._records.append(
            {
                "component": component,
                "field": field,
                "available": False,
                "reason": reason,
                "source": "",
                "shape": "",
            }
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(self._records)


def _observation_sequence(
    observations: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    if observations is None:
        return []
    if isinstance(observations, Mapping):
        return [observations]
    return [item for item in observations if isinstance(item, Mapping)]


def _stack_observation_field(
    observations: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
) -> tuple[np.ndarray | None, str | None]:
    source: str | None = None
    values: list[np.ndarray] = []
    for obs in observations:
        found = None
        for key in keys:
            value = _lookup_mapping_path(obs, key)
            if value is not None:
                found = value
                source = key
                break
        if found is None:
            return None, None
        array = _numeric_array(found)
        if array is None:
            return None, None
        values.append(_squeeze_single_env(array))
    try:
        return np.stack(values, axis=0), source
    except ValueError:
        return None, None


def _lookup_mapping_path(mapping: Mapping[str, Any], path: str) -> Any | None:
    if path in mapping:
        return mapping[path]
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _numeric_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.dtype.kind not in {"b", "i", "u", "f"}:
        return None
    return array


def _squeeze_single_env(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim >= 2 and value.shape[0] == 1:
        return value[0]
    return value


def _robot_axes(field: str, values: np.ndarray) -> list[str]:
    if field == "eef_mat":
        return ["timestep", "row", "col"]
    if field in {"eef_pos"}:
        return ["timestep", "xyz"]
    if field == "eef_quat":
        return ["timestep", "xyzw"]
    if field.startswith("gripper_"):
        return ["timestep", "gripper_joint"]
    return ["timestep", "joint"]


def _quat_array_to_matrix(quats: np.ndarray) -> np.ndarray:
    quats = np.asarray(quats, dtype=np.float64)
    flat = quats.reshape(-1, quats.shape[-1])
    mats = np.stack([_quat_to_matrix(quat) for quat in flat], axis=0)
    return mats.reshape(*quats.shape[:-1], 3, 3)


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    if quat.shape[-1] != 4:
        return np.full((3, 3), np.nan, dtype=np.float32)
    x, y, z, w = [float(item) for item in quat]
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return np.full((3, 3), np.nan, dtype=np.float32)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _env_candidates(env: Any | None) -> list[Any]:
    if env is None:
        return []
    candidates: list[Any] = []
    seen: set[int] = set()
    stack = [env]
    while stack:
        item = stack.pop(0)
        if item is None or id(item) in seen:
            continue
        seen.add(id(item))
        candidates.append(item)
        for attr in ("unwrapped", "env", "_env", "gym_env", "base_env"):
            try:
                child = getattr(item, attr)
            except Exception:
                continue
            if child is not item:
                stack.append(child)
        for attr in ("envs",):
            try:
                children = getattr(item, attr)
            except Exception:
                continue
            if isinstance(children, Mapping):
                stack.extend(children.values())
            elif isinstance(children, Sequence) and not isinstance(children, (str, bytes)):
                stack.extend(children)
    return candidates


def _first_attr(candidates: Sequence[Any], names: Sequence[str]) -> tuple[Any, str] | None:
    for candidate in candidates:
        for name in names:
            try:
                value = getattr(candidate, name)
            except Exception:
                continue
            if value is not None and not callable(value):
                return value, f"{type(candidate).__name__}.{name}"
    return None


def _first_existing_attr(candidates: Sequence[Any], names: Sequence[str]) -> Any | None:
    found = _first_attr(candidates, names)
    return None if found is None else found[0]


def _metadata_row(
    field: str,
    *,
    available: bool,
    source: str = "",
    value: Any = None,
    reason: str = "",
    array_name: str = "",
    shape: Sequence[int] | None = None,
) -> dict[str, Any]:
    return {
        "field": field,
        "available": available,
        "source": source,
        "value": _scalar_text(value),
        "array_name": array_name,
        "shape": _shape_text(shape),
        "reason": reason,
    }


def _scene_snapshot_sequence(
    scene_snapshots: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    if scene_snapshots is None:
        return []
    return [snapshot for snapshot in scene_snapshots if isinstance(snapshot, Mapping)]


def _camera_snapshot_sequence(
    camera_snapshots: Sequence[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    if camera_snapshots is None:
        return []
    return [snapshot for snapshot in camera_snapshots if isinstance(snapshot, Mapping)]


def _names_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [str(key) for key in value]
    if isinstance(value, (str, bytes)):
        return [item.strip() for item in str(value).split(",") if item.strip()]
    if isinstance(value, Sequence):
        names: list[str] = []
        for item in value:
            if isinstance(item, (str, bytes)):
                names.append(str(item))
            else:
                name = getattr(item, "name", None) or getattr(item, "object_name", None)
                if name is not None:
                    names.append(str(name))
        return names
    return []


def _flatten_mapping_keys(mapping: Mapping[str, Any], prefix: str = "") -> list[str]:
    keys: list[str] = []
    for key, value in mapping.items():
        text = f"{prefix}.{key}" if prefix else str(key)
        keys.append(text)
        if isinstance(value, Mapping):
            keys.extend(_flatten_mapping_keys(value, text))
    return keys


def _named_data_vector(data: Any, method_name: str, name: str, size: int) -> np.ndarray | None:
    if not name:
        return None
    method = getattr(data, method_name, None)
    if not callable(method):
        return None
    try:
        return _numeric_vector(method(name), size)
    except Exception:
        return None


def _indexed_data_vector(data: Any, attr_name: str, index: int, size: int) -> np.ndarray | None:
    try:
        values = getattr(data, attr_name)
        return _numeric_vector(values[int(index)], size)
    except Exception:
        return None


def _named_data_matrix(data: Any, method_name: str, name: str) -> np.ndarray | None:
    if not name:
        return None
    method = getattr(data, method_name, None)
    if not callable(method):
        return None
    try:
        return _numeric_matrix(method(name), 3, 3)
    except Exception:
        return None


def _indexed_data_matrix(data: Any, attr_name: str, index: int) -> np.ndarray | None:
    try:
        values = getattr(data, attr_name)
        return _numeric_matrix(values[int(index)], 3, 3)
    except Exception:
        return None


def _numeric_vector(value: Any, size: int) -> np.ndarray | None:
    array = _numeric_array(value)
    if array is None:
        return None
    flat = np.ravel(array).astype(np.float32, copy=False)
    if flat.size < size:
        return None
    return flat[:size].copy()


def _numeric_matrix(value: Any, rows: int, cols: int) -> np.ndarray | None:
    array = _numeric_array(value)
    if array is None:
        return None
    try:
        return np.asarray(array, dtype=np.float32).reshape(rows, cols)
    except ValueError:
        return None


def _resolve_body_id(sim: Any, value: Any) -> int | None:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if value is None:
        return None
    model = getattr(sim, "model", None)
    method = getattr(model, "body_name2id", None)
    if callable(method):
        try:
            return int(method(str(value)))
        except Exception:
            return None
    return None


def _resolve_site_id(model: Any, value: Any) -> int | None:
    if model is None or value is None:
        return None
    method = getattr(model, "site_name2id", None)
    if callable(method):
        try:
            return int(method(str(value)))
        except Exception:
            return None
    return None


def _body_name_from_value(sim: Any, value: Any) -> str:
    if isinstance(value, str):
        return value
    body_id = _resolve_body_id(sim, value)
    if body_id is None:
        return ""
    model = getattr(sim, "model", None)
    method = getattr(model, "body_id2name", None)
    if callable(method):
        try:
            name = method(body_id)
            return "" if name is None else str(name)
        except Exception:
            pass
    names = getattr(model, "body_names", None)
    if isinstance(names, Sequence) and 0 <= body_id < len(names):
        return str(names[body_id])
    return ""


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mujoco_quat_to_xyzw(quat: np.ndarray) -> np.ndarray:
    value = _numeric_vector(quat, 4)
    if value is None:
        return np.full(4, np.nan, dtype=np.float32)
    return np.asarray([value[1], value[2], value[3], value[0]], dtype=np.float32)


def _mat_to_quat_xyzw(mat: np.ndarray) -> np.ndarray:
    matrix = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    quat = np.asarray([x, y, z, w], dtype=np.float32)
    norm = np.linalg.norm(quat)
    if not np.isfinite(norm) or norm <= 1e-12:
        return np.full(4, np.nan, dtype=np.float32)
    return quat / norm


def _generic_axes(array: np.ndarray, *, trailing_prefix: str) -> list[str]:
    if array.ndim == 0:
        return []
    if array.ndim == 1:
        return [f"{trailing_prefix}_dim"]
    return [f"axis_{index}" for index in range(array.ndim)]


def _shape_text(shape: Sequence[int] | None) -> str:
    if shape is None:
        return ""
    return "x".join(str(int(item)) for item in shape)


def _scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    array = _numeric_array(value)
    if array is not None and array.size == 1:
        return str(np.ravel(array)[0].item())
    return repr(value)
