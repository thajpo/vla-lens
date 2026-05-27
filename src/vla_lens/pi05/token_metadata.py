"""PI0.5 token and preprocessing metadata helpers.

These helpers intentionally accept plain Python mappings and array-like values
so unit tests and offline capture code do not need to import LeRobot.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

PREFIX_TOKEN_SPACE_ID = "pi05.prefix"
ACTION_SUFFIX_TOKEN_SPACE_ID = "pi05.action_suffix"
EXPERT_CONTEXT_TOKEN_SPACE_ID = "pi05.expert_context"
LANGUAGE_STREAM_ID = "language"
ACTION_SUFFIX_STREAM_ID = "action_suffix"
PREFIX_STREAM_ID = "prefix"
EXPERT_CONTEXT_STREAM_ID = "expert_context"

STREAM_COLUMNS = [
    "stream_id",
    "name",
    "modality",
    "camera_id",
    "is_present",
    "is_empty_camera_slot",
    "metadata",
]
TOKEN_SPACE_COLUMNS = [
    "token_space_id",
    "segment",
    "stream_id",
    "modality",
    "token_count",
    "metadata",
]
TOKEN_COLUMNS = [
    "token_space_id",
    "token_index",
    "token_kind",
    "token_type",
    "modality",
    "stream_id",
    "token_id",
    "token_piece",
    "attention_mask",
    "special_token_mask",
    "prefix_mask",
    "token_value_type",
    "camera_id",
    "camera_slot_index",
    "is_empty_camera_slot",
    "patch_index",
    "patch_row",
    "patch_col",
    "pixel_y0",
    "pixel_y1",
    "pixel_x0",
    "pixel_x1",
    "action_horizon_index",
    "action_dim",
    "metadata",
]
POLICY_CALL_COLUMNS = [
    "policy_call_index",
    "observation_timestep",
    "env_timestep_start",
    "env_timestep_end",
    "prompt",
    "metadata",
]


@dataclass(frozen=True, slots=True)
class NormalizedLanguageTokens:
    """Language-token metadata normalized from tokenizer outputs."""

    input_ids: tuple[int, ...]
    token_pieces: tuple[str, ...]
    attention_mask: tuple[bool, ...]
    special_tokens_mask: tuple[bool, ...]
    prompt: str | None = None

    @property
    def token_count(self) -> int:
        return len(self.input_ids)

    @property
    def active_token_count(self) -> int:
        return sum(1 for item in self.attention_mask if item)

    def records(
        self,
        *,
        token_space_id: str = PREFIX_TOKEN_SPACE_ID,
        start_index: int = 0,
        stream_id: str = LANGUAGE_STREAM_ID,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for offset, token_id in enumerate(self.input_ids):
            records.append(
                {
                    "token_space_id": token_space_id,
                    "token_index": start_index + offset,
                    "token_kind": "language",
                    "token_type": "text",
                    "modality": "language",
                    "stream_id": stream_id,
                    "token_id": int(token_id),
                    "token_piece": self.token_pieces[offset],
                    "attention_mask": bool(self.attention_mask[offset]),
                    "special_token_mask": bool(self.special_tokens_mask[offset]),
                    "prefix_mask": bool(self.attention_mask[offset]),
                    "token_value_type": "token_id",
                    "metadata": "{}",
                }
            )
        return records

    def metadata(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "token_count": self.token_count,
            "active_token_count": self.active_token_count,
            "input_ids": list(self.input_ids),
            "token_pieces": list(self.token_pieces),
            "attention_mask": list(self.attention_mask),
            "special_tokens_mask": list(self.special_tokens_mask),
        }


@dataclass(frozen=True, slots=True)
class ImagePrefixLayout:
    """Image-prefix token layout, including masked empty camera slots."""

    camera_order: tuple[str, ...]
    camera_present: tuple[bool, ...]
    camera_prefix_mask: tuple[bool, ...]
    patches_per_image: int
    grid_height: int
    grid_width: int
    image_size: tuple[int, int] | None = None

    @property
    def image_slots(self) -> int:
        return len(self.camera_order)

    @property
    def token_count(self) -> int:
        return self.image_slots * self.patches_per_image

    def records(
        self,
        *,
        token_space_id: str = PREFIX_TOKEN_SPACE_ID,
        start_index: int = 0,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for camera_index, camera_id in enumerate(self.camera_order):
            is_present = bool(self.camera_present[camera_index])
            prefix_mask = bool(self.camera_prefix_mask[camera_index])
            is_empty = not is_present or camera_id.startswith("empty_camera")
            stream_id = image_stream_id(camera_id)
            for patch_index in range(self.patches_per_image):
                patch_row = patch_index // max(1, self.grid_width)
                patch_col = patch_index % max(1, self.grid_width)
                record = {
                    "token_space_id": token_space_id,
                    "token_index": start_index
                    + camera_index * self.patches_per_image
                    + patch_index,
                    "token_kind": "image",
                    "token_type": "image_patch",
                    "modality": "image",
                    "stream_id": stream_id,
                    "prefix_mask": prefix_mask,
                    "token_value_type": "embedding",
                    "camera_id": camera_id,
                    "camera_slot_index": camera_index,
                    "is_empty_camera_slot": is_empty,
                    "patch_index": patch_index,
                    "patch_row": patch_row,
                    "patch_col": patch_col,
                    "metadata": "{}",
                }
                record.update(
                    _patch_pixel_bounds(
                        self.image_size,
                        self.grid_height,
                        self.grid_width,
                        patch_row,
                        patch_col,
                    )
                )
                records.append(record)
        return records

    def stream_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for camera_id, is_present, prefix_mask in zip(
            self.camera_order,
            self.camera_present,
            self.camera_prefix_mask,
            strict=True,
        ):
            is_empty = not bool(is_present) or camera_id.startswith("empty_camera")
            records.append(
                {
                    "stream_id": image_stream_id(camera_id),
                    "name": camera_id,
                    "modality": "image",
                    "camera_id": camera_id,
                    "is_present": bool(is_present),
                    "is_empty_camera_slot": is_empty,
                    "metadata": _json_dumps({"prefix_mask": bool(prefix_mask)}),
                }
            )
        return records

    def metadata(self) -> dict[str, Any]:
        return {
            "camera_order": list(self.camera_order),
            "camera_present": list(self.camera_present),
            "camera_prefix_mask": list(self.camera_prefix_mask),
            "patches_per_image": self.patches_per_image,
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "image_slots": self.image_slots,
            "image_tokens": self.token_count,
            "image_size": list(self.image_size) if self.image_size is not None else None,
        }


@dataclass(frozen=True, slots=True)
class ActionSuffixLayout:
    """Continuous PI0.5 action-suffix token layout."""

    horizon: int
    action_dim: int | None = None

    @property
    def token_count(self) -> int:
        return self.horizon

    def records(
        self,
        *,
        token_space_id: str = ACTION_SUFFIX_TOKEN_SPACE_ID,
        stream_id: str = ACTION_SUFFIX_STREAM_ID,
    ) -> list[dict[str, Any]]:
        return [
            {
                "token_space_id": token_space_id,
                "token_index": horizon_index,
                "token_kind": "action",
                "token_type": "continuous_action",
                "modality": "action",
                "stream_id": stream_id,
                "prefix_mask": True,
                "token_value_type": "continuous",
                "action_horizon_index": horizon_index,
                "action_dim": self.action_dim,
                "metadata": "{}",
            }
            for horizon_index in range(self.horizon)
        ]

    def metadata(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "token_value_type": "continuous",
            "action_tokens_are_token_ids": False,
        }


@dataclass(frozen=True, slots=True)
class PI05TokenMetadata:
    """DataFrames and per-call metadata ready for the VLA Lens overlay."""

    streams: pd.DataFrame
    token_spaces: pd.DataFrame
    tokens: pd.DataFrame
    policy_calls: pd.DataFrame
    policy_call_metadata: dict[str, Any]


def normalize_language_tokens(
    language: Mapping[str, Any] | Sequence[int] | str | None = None,
    *,
    tokenizer: Any | None = None,
    prompt: str | None = None,
) -> NormalizedLanguageTokens:
    """Normalize token IDs, pieces, and masks from plain mappings or fake tokenizers."""

    source: Any = language
    if isinstance(language, str):
        prompt = language
        source = None
    if isinstance(source, Mapping) and "prompt" in source and prompt is None:
        prompt = str(source["prompt"])
    should_tokenize = source is None or (isinstance(source, Mapping) and "input_ids" not in source)
    if should_tokenize and tokenizer:
        if prompt is None:
            raise ValueError("prompt is required when tokenizing language metadata")
        tokenized = tokenizer(prompt)
        source = _mapping_from_tokenized(tokenized)
    if source is None:
        return NormalizedLanguageTokens((), (), (), (), prompt=prompt)

    if isinstance(source, Mapping):
        input_ids = _flat_list(source.get("input_ids", ()), "input_ids")
        raw_attention = source.get("attention_mask", source.get("prefix_mask"))
        raw_pieces = source.get("token_pieces", source.get("pieces", source.get("tokens")))
        raw_special = source.get("special_tokens_mask")
    else:
        input_ids = _flat_list(source, "input_ids")
        raw_attention = None
        raw_pieces = None
        raw_special = None

    ids = tuple(int(item) for item in input_ids)
    attention_mask = _bool_mask(raw_attention, len(ids), default=True, name="attention_mask")
    special_tokens_mask = _bool_mask(
        raw_special,
        len(ids),
        default=False,
        name="special_tokens_mask",
    )
    pieces = tuple(_token_pieces(ids, raw_pieces, tokenizer))
    return NormalizedLanguageTokens(
        input_ids=ids,
        token_pieces=pieces,
        attention_mask=attention_mask,
        special_tokens_mask=special_tokens_mask,
        prompt=prompt,
    )


def image_prefix_token_layout(
    cameras: Sequence[str | Mapping[str, Any]] | None = None,
    *,
    image_slots: int | None = None,
    patches_per_image: int = 256,
    grid_shape: Sequence[int] | None = None,
    image_size: Sequence[int] | None = None,
    camera_masks: Mapping[str, bool] | Sequence[bool] | None = None,
) -> ImagePrefixLayout:
    """Build the PI0.5 image-prefix layout, preserving masked empty slots."""

    camera_specs = [_camera_spec(item) for item in cameras or ()]
    if image_slots is None:
        image_slots = len(camera_specs)
    if image_slots < len(camera_specs):
        raise ValueError("image_slots cannot be smaller than the number of camera specs")
    for empty_index in range(image_slots - len(camera_specs)):
        camera_specs.append(
            {
                "camera_id": f"empty_camera_{empty_index}",
                "is_present": False,
                "prefix_mask": False,
                "is_empty_camera_slot": True,
            }
        )
    grid_height, grid_width = _grid_shape(patches_per_image, grid_shape)
    normalized_size = _image_size(image_size)
    camera_order: list[str] = []
    camera_present: list[bool] = []
    camera_prefix_mask: list[bool] = []
    for index, spec in enumerate(camera_specs):
        camera_id = str(spec["camera_id"])
        present = bool(spec.get("is_present", not camera_id.startswith("empty_camera")))
        prefix_mask = bool(spec.get("prefix_mask", present))
        if camera_masks is not None:
            prefix_mask = _camera_mask(camera_masks, camera_id, index)
        if bool(spec.get("is_empty_camera_slot", False)):
            present = False
            prefix_mask = False
        camera_order.append(camera_id)
        camera_present.append(present)
        camera_prefix_mask.append(prefix_mask)
    return ImagePrefixLayout(
        camera_order=tuple(camera_order),
        camera_present=tuple(camera_present),
        camera_prefix_mask=tuple(camera_prefix_mask),
        patches_per_image=int(patches_per_image),
        grid_height=grid_height,
        grid_width=grid_width,
        image_size=normalized_size,
    )


def action_suffix_token_layout(
    action: Any | None = None,
    *,
    horizon: int | None = None,
    action_dim: int | None = None,
) -> ActionSuffixLayout:
    """Build the continuous action-suffix token layout.

    A PI0.5 suffix token represents one horizon step of a continuous action
    chunk. It is not a generated text-token ID.
    """

    if action is not None:
        shape = tuple(int(item) for item in _to_numpy(action).shape)
        if len(shape) == 1:
            horizon = horizon or 1
            action_dim = action_dim or shape[0]
        elif len(shape) >= 2:
            horizon = horizon or shape[-2]
            action_dim = action_dim or shape[-1]
    if horizon is None:
        raise ValueError("horizon is required when action is not provided")
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    normalized_action_dim = None if action_dim is None else int(action_dim)
    return ActionSuffixLayout(horizon=int(horizon), action_dim=normalized_action_dim)


def prompt_metadata(
    prompt: str | None,
    language_tokens: NormalizedLanguageTokens | Mapping[str, Any] | None = None,
    *,
    prompt_template: str | None = None,
) -> dict[str, Any]:
    normalized = (
        language_tokens
        if isinstance(language_tokens, NormalizedLanguageTokens)
        else normalize_language_tokens(language_tokens, prompt=prompt)
        if language_tokens is not None
        else None
    )
    return {
        "prompt": prompt,
        "prompt_template": prompt_template,
        "prompt_length_chars": len(prompt or ""),
        "language_token_count": normalized.token_count if normalized is not None else None,
        "language_active_token_count": (
            normalized.active_token_count if normalized is not None else None
        ),
    }


def image_preprocessing_metadata(
    metadata: Mapping[str, Any] | None = None,
    *,
    image_size: Sequence[int] | None = None,
    resize_size: Sequence[int] | None = None,
    crop_size: Sequence[int] | None = None,
    patch_size: int | None = None,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Normalize image preprocessing metadata into JSON-safe values."""

    payload = dict(metadata or {})
    for key, value in {
        "image_size": image_size,
        "resize_size": resize_size,
        "crop_size": crop_size,
        "patch_size": patch_size,
        "mean": mean,
        "std": std,
    }.items():
        if value is not None:
            payload[key] = value
    return _jsonable(payload)


def action_normalization_metadata(
    stats: Mapping[str, Any] | None = None,
    *,
    action_dim: int | None = None,
    normalization_type: str | None = None,
    unnormalize_key: str | None = None,
    action_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Normalize action un/normalization stats into JSON-safe metadata."""

    payload = dict(stats or {})
    if action_dim is not None:
        payload["action_dim"] = int(action_dim)
    if normalization_type is not None:
        payload["normalization_type"] = normalization_type
    if unnormalize_key is not None:
        payload["unnormalize_key"] = unnormalize_key
    if action_names is not None:
        payload["action_names"] = list(action_names)
    if "mask" in payload:
        payload["mask"] = [bool(item) for item in _flat_list(payload["mask"], "mask")]
    payload.setdefault("normalized_range", [-1.0, 1.0])
    payload.setdefault(
        "formula",
        "unnormalized = where(mask, 0.5 * (normalized + 1) * (high - low) + low, normalized)",
    )
    return _jsonable(payload)


def build_pi05_token_metadata(
    *,
    prompt: str | None = None,
    language: Mapping[str, Any] | Sequence[int] | str | None = None,
    tokenizer: Any | None = None,
    cameras: Sequence[str | Mapping[str, Any]] | None = None,
    image_slots: int | None = None,
    patches_per_image: int = 256,
    grid_shape: Sequence[int] | None = None,
    image_size: Sequence[int] | None = None,
    camera_masks: Mapping[str, bool] | Sequence[bool] | None = None,
    image_preprocessing: Mapping[str, Any] | None = None,
    action: Any | None = None,
    action_horizon: int | None = None,
    action_dim: int | None = None,
    action_normalization: Mapping[str, Any] | None = None,
    policy_calls: pd.DataFrame | Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    policy_call_index: int | None = None,
    observation_timestep: int | None = None,
    env_timestep_start: int | None = None,
    env_timestep_end: int | None = None,
) -> PI05TokenMetadata:
    """Build streams, token_spaces, tokens, and policy-call metadata tables."""

    language_tokens = normalize_language_tokens(language, tokenizer=tokenizer, prompt=prompt)
    image_layout = image_prefix_token_layout(
        cameras,
        image_slots=image_slots,
        patches_per_image=patches_per_image,
        grid_shape=grid_shape,
        image_size=image_size,
        camera_masks=camera_masks,
    )
    action_layout = action_suffix_token_layout(
        action,
        horizon=action_horizon,
        action_dim=action_dim,
    )
    image_preprocess = image_preprocessing_metadata(image_preprocessing, image_size=image_size)
    action_norm = action_normalization_metadata(
        action_normalization,
        action_dim=action_layout.action_dim,
    )
    call_metadata = {
        "model_family": "pi05",
        "prompt": prompt,
        "prompt_metadata": prompt_metadata(prompt, language_tokens),
        "language_tokens": language_tokens.metadata(),
        "image_prefix": image_layout.metadata(),
        "image_preprocessing_metadata": image_preprocess,
        "action_suffix": action_layout.metadata(),
        "expert_context": {
            "segments": [PREFIX_TOKEN_SPACE_ID, ACTION_SUFFIX_TOKEN_SPACE_ID],
            "token_count": image_layout.token_count
            + language_tokens.token_count
            + action_layout.token_count,
        },
        "action_normalization_metadata": action_norm,
        "token_space_ids": [
            PREFIX_TOKEN_SPACE_ID,
            ACTION_SUFFIX_TOKEN_SPACE_ID,
            EXPERT_CONTEXT_TOKEN_SPACE_ID,
        ],
    }

    streams = _dataframe(
        [
            {
                "stream_id": PREFIX_STREAM_ID,
                "name": "vlm_prefix",
                "modality": "multimodal",
                "metadata": "{}",
            },
            {
                "stream_id": LANGUAGE_STREAM_ID,
                "name": "language",
                "modality": "language",
                "metadata": "{}",
            },
            {
                "stream_id": ACTION_SUFFIX_STREAM_ID,
                "name": "action_suffix",
                "modality": "action",
                "metadata": "{}",
            },
            {
                "stream_id": EXPERT_CONTEXT_STREAM_ID,
                "name": "expert_context",
                "modality": "multimodal_action",
                "metadata": _json_dumps(
                    {"segments": [PREFIX_TOKEN_SPACE_ID, ACTION_SUFFIX_TOKEN_SPACE_ID]}
                ),
            },
            *image_layout.stream_records(),
        ],
        STREAM_COLUMNS,
    )
    prefix_count = image_layout.token_count + language_tokens.token_count
    token_spaces = _dataframe(
        [
            {
                "token_space_id": PREFIX_TOKEN_SPACE_ID,
                "segment": "vlm_prefix",
                "stream_id": PREFIX_STREAM_ID,
                "modality": "multimodal",
                "token_count": prefix_count,
                "metadata": _json_dumps(
                    {
                        "image_tokens": image_layout.token_count,
                        "language_tokens": language_tokens.token_count,
                        "image_prefix": image_layout.metadata(),
                        "prompt_metadata": call_metadata["prompt_metadata"],
                    }
                ),
            },
            {
                "token_space_id": ACTION_SUFFIX_TOKEN_SPACE_ID,
                "segment": "action_expert",
                "stream_id": ACTION_SUFFIX_STREAM_ID,
                "modality": "action",
                "token_count": action_layout.token_count,
                "metadata": _json_dumps(action_layout.metadata()),
            },
            {
                "token_space_id": EXPERT_CONTEXT_TOKEN_SPACE_ID,
                "segment": "expert_context",
                "stream_id": EXPERT_CONTEXT_STREAM_ID,
                "modality": "multimodal_action",
                "token_count": prefix_count + action_layout.token_count,
                "metadata": _json_dumps(
                    {
                        "kind": "composite",
                        "segments": [PREFIX_TOKEN_SPACE_ID, ACTION_SUFFIX_TOKEN_SPACE_ID],
                        "prefix_tokens": prefix_count,
                        "action_tokens": action_layout.token_count,
                    }
                ),
            },
        ],
        TOKEN_SPACE_COLUMNS,
    )
    token_records = [
        *image_layout.records(),
        *language_tokens.records(start_index=image_layout.token_count),
        *action_layout.records(),
    ]
    tokens = _dataframe(token_records, TOKEN_COLUMNS)
    calls = policy_calls_dataframe(
        policy_calls,
        metadata=call_metadata,
        prompt=prompt,
        policy_call_index=policy_call_index,
        observation_timestep=observation_timestep,
        env_timestep_start=env_timestep_start,
        env_timestep_end=env_timestep_end,
    )
    return PI05TokenMetadata(
        streams=streams,
        token_spaces=token_spaces,
        tokens=tokens,
        policy_calls=calls,
        policy_call_metadata=call_metadata,
    )


def policy_calls_dataframe(
    policy_calls: pd.DataFrame | Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    *,
    metadata: Mapping[str, Any] | None = None,
    prompt: str | None = None,
    policy_call_index: int | None = None,
    observation_timestep: int | None = None,
    env_timestep_start: int | None = None,
    env_timestep_end: int | None = None,
) -> pd.DataFrame:
    """Attach JSON policy-call metadata to existing or single-call records."""

    if policy_calls is None:
        if policy_call_index is None and observation_timestep is None:
            return _dataframe([], POLICY_CALL_COLUMNS)
        records: list[dict[str, Any]] = [
            {
                "policy_call_index": 0 if policy_call_index is None else int(policy_call_index),
                "observation_timestep": observation_timestep,
                "env_timestep_start": env_timestep_start,
                "env_timestep_end": env_timestep_end,
            }
        ]
    elif isinstance(policy_calls, pd.DataFrame):
        records = policy_calls.to_dict("records")
    elif isinstance(policy_calls, Mapping):
        records = [dict(policy_calls)]
    else:
        records = [dict(item) for item in policy_calls]

    metadata_json = _json_dumps(metadata or {})
    for index, record in enumerate(records):
        record.setdefault("policy_call_index", index)
        record.setdefault("prompt", prompt)
        existing_metadata = record.get("metadata")
        if _metadata_present(existing_metadata):
            merged = _json_loads(existing_metadata)
            merged.update(metadata or {})
            record["metadata"] = _json_dumps(merged)
        else:
            record["metadata"] = metadata_json
    return _dataframe(records, POLICY_CALL_COLUMNS)


def image_stream_id(camera_id: str) -> str:
    return f"image_{camera_id}"


def _mapping_from_tokenized(tokenized: Any) -> Mapping[str, Any]:
    if isinstance(tokenized, Mapping):
        return tokenized
    keys = ["input_ids", "attention_mask", "special_tokens_mask"]
    return {key: getattr(tokenized, key) for key in keys if hasattr(tokenized, key)}


def _flat_list(value: Any, name: str) -> list[Any]:
    array = _to_numpy(value)
    if array.ndim == 0:
        return [array.item()]
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim > 1:
        raise ValueError(f"{name} must be one-dimensional or a single-row batch")
    return array.tolist()


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _bool_mask(value: Any, length: int, *, default: bool, name: str) -> tuple[bool, ...]:
    if value is None:
        return tuple(default for _ in range(length))
    items = [bool(item) for item in _flat_list(value, name)]
    if len(items) != length:
        raise ValueError(f"{name} length {len(items)} does not match input_ids length {length}")
    return tuple(items)


def _token_pieces(ids: tuple[int, ...], raw_pieces: Any, tokenizer: Any | None) -> list[str]:
    if raw_pieces is not None:
        pieces = [str(item) for item in _flat_list(raw_pieces, "token_pieces")]
        if len(pieces) != len(ids):
            raise ValueError("token_pieces length does not match input_ids length")
        return pieces
    if tokenizer is not None and hasattr(tokenizer, "convert_ids_to_tokens"):
        pieces = tokenizer.convert_ids_to_tokens(list(ids))
        if isinstance(pieces, str):
            pieces = [pieces]
        return [str(item) for item in pieces]
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        pieces = []
        for token_id in ids:
            try:
                pieces.append(str(tokenizer.decode([token_id], skip_special_tokens=False)))
            except TypeError:
                pieces.append(str(tokenizer.decode([token_id])))
        return pieces
    return [str(token_id) for token_id in ids]


def _camera_spec(value: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, str):
        return {"camera_id": value}
    camera_id = (
        value.get("camera_id") or value.get("name") or value.get("stream_id") or value.get("id")
    )
    if camera_id is None:
        raise ValueError("camera specs require camera_id, name, stream_id, or id")
    return {
        **dict(value),
        "camera_id": str(camera_id),
    }


def _camera_mask(
    camera_masks: Mapping[str, bool] | Sequence[bool],
    camera_id: str,
    index: int,
) -> bool:
    if isinstance(camera_masks, Mapping):
        return bool(camera_masks.get(camera_id, True))
    if index >= len(camera_masks):
        return True
    return bool(camera_masks[index])


def _grid_shape(patches_per_image: int, grid_shape: Sequence[int] | None) -> tuple[int, int]:
    if patches_per_image <= 0:
        raise ValueError("patches_per_image must be positive")
    if grid_shape is not None:
        if len(grid_shape) != 2:
            raise ValueError("grid_shape must be (height, width)")
        height, width = int(grid_shape[0]), int(grid_shape[1])
        if height * width != patches_per_image:
            raise ValueError("grid_shape product must equal patches_per_image")
        return height, width
    width = int(np.ceil(np.sqrt(patches_per_image)))
    while width > 1 and patches_per_image % width != 0:
        width -= 1
    return patches_per_image // width, width


def _image_size(value: Sequence[int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError("image_size must be (height, width)")
    return int(value[0]), int(value[1])


def _patch_pixel_bounds(
    image_size: tuple[int, int] | None,
    grid_height: int,
    grid_width: int,
    patch_row: int,
    patch_col: int,
) -> dict[str, int | None]:
    if image_size is None:
        return {"pixel_y0": None, "pixel_y1": None, "pixel_x0": None, "pixel_x1": None}
    height, width = image_size
    y_edges = np.linspace(0, height, grid_height + 1, dtype=np.int32)
    x_edges = np.linspace(0, width, grid_width + 1, dtype=np.int32)
    return {
        "pixel_y0": int(y_edges[patch_row]),
        "pixel_y1": int(y_edges[patch_row + 1]),
        "pixel_x0": int(x_edges[patch_col]),
        "pixel_x1": int(x_edges[patch_col + 1]),
    }


def _dataframe(records: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(records)
    for column in columns:
        if column not in frame:
            frame[column] = pd.NA
    return frame.loc[:, list(columns)]


def _json_dumps(value: Mapping[str, Any]) -> str:
    return json.dumps(_jsonable(value), sort_keys=True)


def _json_loads(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None or value == "":
        return {}
    return dict(json.loads(str(value)))


def _metadata_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, str):
        return value != ""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return True
    if isinstance(missing, bool):
        return not missing
    return True


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


__all__ = [
    "ACTION_SUFFIX_TOKEN_SPACE_ID",
    "LANGUAGE_STREAM_ID",
    "PREFIX_TOKEN_SPACE_ID",
    "ACTION_SUFFIX_STREAM_ID",
    "ActionSuffixLayout",
    "ImagePrefixLayout",
    "NormalizedLanguageTokens",
    "PI05TokenMetadata",
    "action_normalization_metadata",
    "action_suffix_token_layout",
    "build_pi05_token_metadata",
    "image_prefix_token_layout",
    "image_preprocessing_metadata",
    "image_stream_id",
    "normalize_language_tokens",
    "policy_calls_dataframe",
    "prompt_metadata",
]
