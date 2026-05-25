"""PI0.5 capture utils helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from vla_lens.pi05.capture_schema import (
    CaptureCall,
    EpisodeBuffer,
)
from vla_lens.traces import ArraySpec, ModelSiteSpec


def _stack_optional_calls(calls: list[CaptureCall], attr: str) -> np.ndarray | None:
    sample = next((getattr(call, attr) for call in calls if getattr(call, attr) is not None), None)
    if sample is None:
        return None
    sample_array = np.asarray(sample)
    out = np.full((len(calls), *sample_array.shape), np.nan, dtype=sample_array.dtype)
    for index, call in enumerate(calls):
        value = getattr(call, attr)
        if value is not None:
            out[index] = np.asarray(value, dtype=sample_array.dtype)
    return out

def _stack_layer_calls(calls: list[CaptureCall], attr: str, layer: int) -> np.ndarray | None:
    sample = next(
        (
            getattr(call, attr).get(layer)
            for call in calls
            if getattr(call, attr).get(layer) is not None
        ),
        None,
    )
    if sample is None:
        return None
    sample_array = np.asarray(sample)
    out = np.full((len(calls), *sample_array.shape), np.nan, dtype=sample_array.dtype)
    for index, call in enumerate(calls):
        value = getattr(call, attr).get(layer)
        if value is not None:
            out[index] = np.asarray(value, dtype=sample_array.dtype)
    return out

def _array_size_summary(
    episode_arrays: dict[str, ArraySpec],
    model_arrays: list[ModelSiteSpec],
) -> dict[str, Any]:
    episode_bytes = {
        name: int(np.asarray(spec.array).nbytes) for name, spec in episode_arrays.items()
    }
    model_bytes = {spec.name: int(np.asarray(spec.array).nbytes) for spec in model_arrays}
    total_episode = int(sum(episode_bytes.values()))
    total_model = int(sum(model_bytes.values()))
    return {
        "episode_bytes": total_episode,
        "model_bytes": total_model,
        "total_uncompressed_bytes": total_episode + total_model,
        "episode_arrays": episode_bytes,
        "model_arrays": model_bytes,
    }

def _stack_call_arrays(calls: list[CaptureCall], attr: str) -> np.ndarray:
    sample = np.asarray(getattr(calls[0], attr), dtype=np.float32)
    out = np.full((len(calls), *sample.shape), np.nan, dtype=np.float32)
    for index, call in enumerate(calls):
        value = getattr(call, attr)
        if value is not None:
            out[index] = np.asarray(value, dtype=np.float32)
    return out

def _scatter_calls(calls: list[CaptureCall], length: int, attr: str) -> np.ndarray:
    sample = np.asarray(getattr(calls[0], attr), dtype=np.float32)
    out = np.full((length, *sample.shape), np.nan, dtype=np.float32)
    for call in calls:
        value = getattr(call, attr)
        if value is None:
            continue
        out[call.env_timestep] = np.asarray(value, dtype=np.float32)
    return out

def _scatter_optional_calls(calls: list[CaptureCall], length: int, attr: str) -> np.ndarray | None:
    sample = next((getattr(call, attr) for call in calls if getattr(call, attr) is not None), None)
    if sample is None:
        return None
    out = np.full((length, *np.asarray(sample, dtype=np.float32).shape), np.nan, dtype=np.float32)
    for call in calls:
        value = getattr(call, attr)
        if value is not None:
            out[call.env_timestep] = np.asarray(value, dtype=np.float32)
    return out

def _trace_cameras(buffer: EpisodeBuffer) -> list[str]:
    cameras = []
    if buffer.frames:
        cameras.append("main")
    if buffer.wrist_frames:
        cameras.append("wrist")
    return cameras

def _patch_grid_shape(patches_per_image: int) -> tuple[int, int]:
    width = int(np.ceil(np.sqrt(max(1, patches_per_image))))
    while width > 1 and patches_per_image % width != 0:
        width -= 1
    height = patches_per_image // width
    return height, width

def _pad_time(values: list[np.ndarray], length: int) -> np.ndarray:
    if not values:
        return np.zeros((length, 0), dtype=np.float32)
    sample = np.asarray(values[0], dtype=np.float32)
    out = np.full((length, *sample.shape), np.nan, dtype=np.float32)
    for idx, value in enumerate(values[:length]):
        out[idx] = np.asarray(value, dtype=np.float32)
    return out

def _pad_bool(values: list[bool], length: int) -> np.ndarray:
    out = np.zeros(length, dtype=bool)
    for index, value in enumerate(values[:length]):
        out[index] = bool(value)
    return out

def _call_mask(length: int, calls: list[CaptureCall]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for call in calls:
        if call.env_timestep < length:
            mask[call.env_timestep] = True
    return mask

def _call_indices(length: int, calls: list[CaptureCall]) -> np.ndarray:
    indices = np.full(length, np.nan, dtype=np.float32)
    for call in calls:
        if call.env_timestep < length:
            indices[call.env_timestep] = call.call_index
    return indices

def _extract_frames(observation: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not isinstance(observation, dict):
        return None, None
    main = _find_image(observation, ("agentview", "image"))
    wrist = _find_image(observation, ("eye_in_hand", "image2", "wrist"))
    return main, wrist

def _find_image(observation: dict[str, Any], needles: tuple[str, ...]) -> np.ndarray | None:
    for key, value in observation.items():
        text = str(key).lower()
        if any(needle in text for needle in needles):
            image = _as_image(value)
            if image is not None:
                return image
    for value in observation.values():
        if isinstance(value, dict):
            image = _find_image(value, needles)
            if image is not None:
                return image
    return None

def _as_image(value: Any) -> np.ndarray | None:
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        return None
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] > 3:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255 if array.max(initial=0) > 1.0 else 1.0)
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = array.astype(np.uint8)
    return np.ascontiguousarray(array)

def _task_prompt(task: Any) -> str:
    for attr in ("language", "description", "name"):
        value = getattr(task, attr, None)
        if value:
            return str(value)
    return str(task)

def _language_metadata_from_observation(obs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    input_ids = _lookup_nested_value(
        obs,
        (
            "observation.language.tokens",
            "observation.language.input_ids",
            "language.tokens",
            "language.input_ids",
            "input_ids",
        ),
    )
    if input_ids is None:
        return None
    payload: dict[str, Any] = {"input_ids": input_ids}
    attention_mask = _lookup_nested_value(
        obs,
        (
            "observation.language.attention_mask",
            "language.attention_mask",
            "attention_mask",
        ),
    )
    if attention_mask is not None:
        payload["attention_mask"] = attention_mask
    special_mask = _lookup_nested_value(
        obs,
        (
            "observation.language.special_tokens_mask",
            "language.special_tokens_mask",
            "special_tokens_mask",
        ),
    )
    if special_mask is not None:
        payload["special_tokens_mask"] = special_mask
    return payload

def _lookup_nested_value(payload: Mapping[str, Any], candidates: tuple[str, ...]) -> Any | None:
    flat: dict[str, Any] = {}
    _flatten_mapping(payload, prefix="", out=flat)
    normalized = {key.lower(): value for key, value in flat.items()}
    for candidate in candidates:
        key = candidate.lower()
        if key in normalized:
            return normalized[key]
    for candidate in candidates:
        suffix = candidate.lower()
        for key, value in normalized.items():
            if key.endswith(suffix):
                return value
    return None

def _flatten_mapping(payload: Mapping[str, Any], *, prefix: str, out: dict[str, Any]) -> None:
    for key, value in payload.items():
        text = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            _flatten_mapping(value, prefix=text, out=out)
        else:
            out[text] = value

def _first_info(info: Any) -> Any:
    if isinstance(info, Mapping):
        out: dict[str, Any] = {}
        for key, value in info.items():
            if isinstance(value, Mapping):
                out[str(key)] = _first_info(value)
                continue
            out[str(key)] = _first_batch_value(value)
        return out
    if isinstance(info, (list, tuple)) and info:
        return info[0]
    return info

def _first_batch_value(value: Any) -> Any:
    if isinstance(value, (str, bytes)) or value is None:
        return value
    try:
        array = np.asarray(value)
    except Exception:
        return value
    if not hasattr(array, "ndim"):
        return array
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    if not hasattr(array, "ndim"):
        return array
    if array.ndim == 0:
        return array.item()
    return array

def _jsonable_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return {"value": _jsonable(value)}

def _json_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True)

def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

def _info_bool(info: Mapping[str, Any], key: str) -> bool | None:
    if key in info:
        value = info[key]
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "success"}
        if value is None:
            return None
        return bool(value)
    for value in info.values():
        if isinstance(value, Mapping):
            found = _info_bool(value, key)
            if found is not None:
                return found
    return None

def _to_numpy(value: Any, *, dtype: np.dtype | str | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "to"):
        value = value.to("cpu")
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    return array.astype(np.dtype("float32") if dtype is None else np.dtype(dtype), copy=False)
