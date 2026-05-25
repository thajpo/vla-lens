"""Common dashboard server helpers."""


from __future__ import annotations

import json
import re
from http import HTTPStatus
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.traces import TraceBundle


def _api_exception_status(exc: Exception) -> HTTPStatus:
    if isinstance(exc, json.JSONDecodeError):
        return HTTPStatus.BAD_REQUEST
    if isinstance(exc, KeyError):
        message = _api_exception_message(exc)
        if message.startswith("Missing query parameter:"):
            return HTTPStatus.BAD_REQUEST
        return HTTPStatus.NOT_FOUND
    if isinstance(exc, FileNotFoundError):
        return HTTPStatus.NOT_FOUND
    if isinstance(exc, (TypeError, ValueError)):
        return HTTPStatus.BAD_REQUEST
    return HTTPStatus.INTERNAL_SERVER_ERROR

def _api_exception_message(exc: Exception) -> str:
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)

def _query_int_value(
    query: Mapping[str, list[str]],
    name: str,
    default: int,
) -> int:
    raw = (query.get(name) or [None])[0]
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return default

def _metadata_text(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    return "" if _is_missing_scalar(value) else str(value)

def _record_bool(record: Mapping[str, Any] | None, key: str) -> bool | None:
    if not record:
        return None
    value = record.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes"}:
            return True
        if text in {"false", "0", "no"}:
            return False
    return None

def _record_float(record: Mapping[str, Any] | None, key: str) -> float | None:
    if not record:
        return None
    value = record.get(key)
    if _is_missing_scalar(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None

def _record_text(record: Mapping[str, Any] | None, key: str) -> str:
    if not record:
        return ""
    value = record.get(key)
    return "" if _is_missing_scalar(value) else str(value)

def _dedupe_reasons(reasons: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for reason in reasons:
        if reason and reason not in seen:
            seen.add(reason)
            out.append(reason)
    return out

def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item)]
    return [item.strip() for item in text.split(",") if item.strip()]

def _json_parse(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value

def _array_summary(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array)
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

def _array_preview(array: np.ndarray) -> Any:
    value = np.asarray(array)
    if value.size > 5000:
        slices = tuple(slice(0, min(64, size)) for size in value.shape)
        value = value[slices]
    return _round(value)

def _label_from_metric_name(name: str) -> str:
    overrides = {
        "eef": "EEF",
        "x": "x",
        "y": "y",
        "z": "z",
    }
    parts = name.replace("-", "_").split("_")
    return " ".join(overrides.get(part, part.capitalize()) for part in parts if part)

def _domain_x_label(domain: str) -> str:
    if domain == "call":
        return "Policy call timestep"
    if domain == "generation":
        return "Generation step"
    return "Environment timestep"

def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and not np.isfinite(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}:
        return True
    return False

def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)

def _dominant_value(values: pd.Series) -> Any:
    if values.empty:
        return None
    nonnull = values.dropna()
    if nonnull.empty:
        return None
    counts = nonnull.astype(str).value_counts()
    return counts.index[0] if not counts.empty else nonnull.iloc[0]

def _mean_numeric(values: pd.Series) -> float | None:
    if values.empty:
        return None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())

def _optional_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)

def _optional_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)

def _optional_bool(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return bool(value)

def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, float) and not np.isfinite(value):
        return []
    if isinstance(value, np.generic):
        return _json_list(value.item())
    if isinstance(value, list):
        return value
    text = str(value or "[]")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []

def _optional_array(bundle: TraceBundle, name: str) -> np.ndarray | None:
    try:
        return bundle.array(name, mmap=True)
    except KeyError:
        return None

def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return int(numeric)

def _cache_part(value: str) -> str:
    safe = [char if char.isalnum() or char in {"-", "_"} else "_" for char in value]
    return "".join(safe).strip("_")[:96] or "item"

def _take_axis_value(array: Any, axes: list[str], axis_name: str, index: int) -> Any:
    if axis_name not in axes:
        return array
    axis = axes.index(axis_name)
    limit = array.shape[axis]
    clipped = max(0, min(int(index), limit - 1))
    selection: list[Any] = [slice(None)] * len(array.shape)
    selection[axis] = clipped
    return array[tuple(selection)]

def _take_axis_values(
    array: Any,
    axes: list[str],
    selections: Mapping[str, int],
) -> tuple[Any, list[str]]:
    if not selections:
        return array, list(axes)
    indexer: list[Any] = [slice(None)] * len(axes)
    remaining_axes: list[str] = []
    for axis, axis_name in enumerate(axes):
        if axis_name in selections:
            limit = array.shape[axis]
            indexer[axis] = max(0, min(int(selections[axis_name]), limit - 1))
        else:
            remaining_axes.append(axis_name)
    return array[tuple(indexer)], remaining_axes

def _policy_call_axis_selection(axes: list[str], call: dict[str, Any]) -> dict[str, int]:
    if "policy_call" in axes:
        return {"policy_call": int(call.get("index", call.get("model_call_index", 0)))}
    if "timestep" in axes:
        return {"timestep": int(call["env_timestep"])}
    return {}

def _take_policy_call_value(
    array: Any,
    axes: list[str],
    call: dict[str, Any],
) -> tuple[Any, list[str]]:
    return _take_axis_values(array, axes, _policy_call_axis_selection(axes, call))

def _site_family(name: str) -> str:
    if ".vlm." in name:
        return "vlm"
    if ".expert." in name:
        return "expert"
    return "other"

def _patches_per_image(image_tokens: int) -> int:
    if image_tokens >= 256 and image_tokens % 256 == 0:
        return 256
    root = int(round(float(np.sqrt(max(1, image_tokens)))))
    if root * root == image_tokens:
        return image_tokens
    return 256

def _query_one(query: dict[str, list[str]], name: str) -> str:
    values = query.get(name)
    if not values:
        raise KeyError(f"Missing query parameter: {name}")
    return values[0]

def _query_call_index(query: dict[str, list[str]]) -> int:
    return int(_query_one(query, "call_index"))

def _query_float(query: dict[str, list[str]], name: str, default: float) -> float:
    values = query.get(name)
    if not values or values[0] in {"", None}:
        return default
    return float(values[0])

def _round(array: np.ndarray) -> Any:
    value = np.round(np.asarray(array, dtype=np.float32), 4)
    return _jsonable(value.tolist())

def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return _json_scalar(value)

def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return _json_scalar(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
