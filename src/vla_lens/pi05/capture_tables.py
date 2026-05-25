"""PI0.5 capture tables helpers."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from vla_lens.pi05.capture_schema import (
    LIBERO_ACTION_DIM_LABELS,
    LIBERO_ACTION_DIM_NAMES,
    LIBERO_ACTION_DIM_UNITS,
    CaptureCall,
    EpisodeBuffer,
)
from vla_lens.pi05.capture_utils import (
    _info_bool,
    _json_dumps,
    _language_metadata_from_observation,
    _pad_bool,
    _patch_grid_shape,
    _trace_cameras,
)
from vla_lens.pi05.context_capture import (
    ContextCaptureResult,
)
from vla_lens.pi05.token_metadata import (
    EXPERT_CONTEXT_STREAM_ID,
    EXPERT_CONTEXT_TOKEN_SPACE_ID,
    build_pi05_token_metadata,
)


def _timesteps_table(buffer: EpisodeBuffer, length: int) -> pd.DataFrame:
    policy_call_for_timestep = np.full(length, np.nan, dtype=np.float32)
    horizon_index = np.full(length, np.nan, dtype=np.float32)
    sorted_calls = sorted(buffer.calls, key=lambda call: call.env_timestep)
    for index, call in enumerate(sorted_calls):
        next_start = (
            sorted_calls[index + 1].env_timestep if index + 1 < len(sorted_calls) else length
        )
        end = min(length, next_start)
        for timestep in range(max(0, call.env_timestep), end):
            policy_call_for_timestep[timestep] = call.call_index
            horizon_index[timestep] = timestep - call.env_timestep
    done = _pad_bool(buffer.terminated, length)
    truncated = _pad_bool(buffer.truncated, length)
    if length and not done.any() and not truncated.any():
        done[-1] = True
    return pd.DataFrame(
        {
            "timestep": np.arange(length, dtype=np.int32),
            "reward": np.asarray(buffer.rewards, dtype=np.float32),
            "done": done,
            "truncated": truncated,
            "policy_call_index": policy_call_for_timestep,
            "horizon_index": horizon_index,
        }
    )

def _generation_steps_table(buffer: EpisodeBuffer) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for call in buffer.calls:
        steps = int(call.denoising_actions.shape[0])
        for generation_step in range(steps):
            records.append(
                {
                    "policy_call_index": int(call.call_index),
                    "generation_step": int(generation_step),
                    "process_kind": "flow_matching",
                    "scheduler_index": int(generation_step),
                }
            )
    return pd.DataFrame.from_records(records)

def _attach_token_metadata(call: CaptureCall, obs: dict[str, Any], buffer: EpisodeBuffer) -> None:
    language = _language_metadata_from_observation(obs)
    cameras = [{"camera_id": camera} for camera in _trace_cameras(buffer)]
    first_frame = buffer.frames[0] if buffer.frames else None
    image_size = (
        (int(first_frame.shape[0]), int(first_frame.shape[1])) if first_frame is not None else None
    )
    metadata = build_pi05_token_metadata(
        prompt=buffer.prompt,
        language=language,
        tokenizer=_pi05_language_tokenizer(),
        cameras=cameras,
        image_slots=call.prefix_image_slots or len(cameras),
        patches_per_image=max(1, call.prefix_patches_per_image or 1),
        image_size=image_size,
        image_preprocessing={
            "raw_shape": list(first_frame.shape) if first_frame is not None else None,
            "processed_shape": list(first_frame.shape) if first_frame is not None else None,
            "resize_mode": "lerobot_policy_preprocessor",
            "value_range": "policy_checkpoint",
        },
        action=call.final_action_chunk,
        action_normalization={
            "normalization_type": "checkpoint",
            "stats_ref": "policy_preprocessor",
            **_libero_action_space_metadata(
                action_dim=int(np.asarray(call.final_action_chunk).shape[-1])
            ),
        },
        policy_call_index=call.call_index,
        observation_timestep=call.env_timestep,
        env_timestep_start=call.env_timestep,
    )
    call.token_metadata = metadata
    call.policy_call_metadata = dict(metadata.policy_call_metadata)

@lru_cache(maxsize=1)
def _pi05_language_tokenizer() -> Any | None:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            "google/paligemma-3b-pt-224",
            local_files_only=True,
        )
    except Exception:
        return None

def _buffer_action_dim(buffer: EpisodeBuffer) -> int | None:
    for action in buffer.executed_actions:
        array = np.asarray(action)
        if array.ndim:
            return int(array.shape[-1])
    for call in buffer.calls:
        array = np.asarray(call.final_action_chunk)
        if array.ndim:
            return int(array.shape[-1])
    return None

def _libero_action_space_metadata(action_dim: int | None = None) -> dict[str, Any]:
    if action_dim != len(LIBERO_ACTION_DIM_NAMES):
        return {"action_dim": action_dim}
    return {
        "action_dim": action_dim,
        "action_names": list(LIBERO_ACTION_DIM_NAMES),
        "action_labels": list(LIBERO_ACTION_DIM_LABELS),
        "action_units": list(LIBERO_ACTION_DIM_UNITS),
        "controller": "robosuite OSC_POSE",
        "control_mode": "relative",
        "position_scale_m": 0.05,
        "rotation_scale_rad": 0.5,
        "orientation_parameterization": "rotation_vector",
    }

def _token_tables(buffer: EpisodeBuffer) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calls_with_metadata = [call for call in buffer.calls if call.token_metadata is not None]
    if calls_with_metadata:
        streams = pd.concat(
            [call.token_metadata.streams for call in calls_with_metadata if call.token_metadata],
            ignore_index=True,
        )
        if "stream_id" in streams:
            streams = streams.drop_duplicates(subset=["stream_id"], keep="first")
        token_spaces = []
        tokens = []
        for call in calls_with_metadata:
            if call.token_metadata is None:
                continue
            call_token_spaces = call.token_metadata.token_spaces.copy()
            call_tokens = call.token_metadata.tokens.copy()
            call_token_spaces["policy_call_index"] = call.call_index
            call_tokens["policy_call_index"] = call.call_index
            token_spaces.append(call_token_spaces)
            tokens.append(call_tokens)
        return (
            streams.reset_index(drop=True),
            pd.concat(token_spaces, ignore_index=True),
            pd.concat(tokens, ignore_index=True),
        )

    cameras = _trace_cameras(buffer)
    streams = [
        {"stream_id": "prefix", "name": "vlm_prefix", "modality": "multimodal"},
        {"stream_id": "language", "name": "language", "modality": "language"},
        {"stream_id": "action_suffix", "name": "action_suffix", "modality": "action"},
        {
            "stream_id": EXPERT_CONTEXT_STREAM_ID,
            "name": "expert_context",
            "modality": "multimodal_action",
        },
    ]
    streams.extend(
        {
            "stream_id": f"image_{camera}",
            "name": camera,
            "modality": "image",
            "camera_id": camera,
        }
        for camera in cameras
    )
    token_spaces: list[dict[str, Any]] = []
    tokens: list[dict[str, Any]] = []
    if buffer.calls:
        sample_call = buffer.calls[0]
        action_tokens = int(sample_call.final_action_chunk.shape[0])
        token_spaces.append(
            {
                "token_space_id": "pi05.action_suffix",
                "policy_call_index": -1,
                "segment": "action_expert",
                "stream_id": "action_suffix",
                "token_count": action_tokens,
            }
        )
        for token_index in range(action_tokens):
            tokens.append(
                {
                    "token_space_id": "pi05.action_suffix",
                    "token_index": token_index,
                    "modality": "action",
                    "token_type": "action_horizon",
                    "stream_id": "action_suffix",
                    "action_horizon_index": token_index,
                }
            )
        if sample_call.prefix_image_hidden is not None:
            patches_per_image = int(
                sample_call.prefix_patches_per_image
                or sample_call.prefix_image_hidden.shape[0] // max(1, len(cameras))
            )
            grid_height, grid_width = _patch_grid_shape(patches_per_image)
            token_spaces.append(
                {
                    "token_space_id": "pi05.prefix",
                    "policy_call_index": -1,
                    "segment": "vlm_prefix",
                    "stream_id": "prefix",
                    "token_count": int(sample_call.prefix_image_hidden.shape[0]),
                }
            )
            token_spaces.append(
                {
                    "token_space_id": EXPERT_CONTEXT_TOKEN_SPACE_ID,
                    "policy_call_index": -1,
                    "segment": "expert_context",
                    "stream_id": EXPERT_CONTEXT_STREAM_ID,
                    "token_count": int(sample_call.prefix_image_hidden.shape[0]) + action_tokens,
                    "metadata": json.dumps(
                        {
                            "kind": "composite",
                            "segments": ["pi05.prefix", "pi05.action_suffix"],
                        }
                    ),
                }
            )
            token_index = 0
            for camera in cameras:
                for patch in range(patches_per_image):
                    row = patch // max(1, grid_width)
                    col = patch % max(1, grid_width)
                    tokens.append(
                        {
                            "token_space_id": "pi05.prefix",
                            "token_index": token_index,
                            "modality": "image",
                            "token_type": "image_patch",
                            "stream_id": f"image_{camera}",
                            "camera_id": camera,
                            "patch_row": row,
                            "patch_col": col,
                        }
                    )
                    token_index += 1
    return (
        pd.DataFrame.from_records(streams),
        pd.DataFrame.from_records(token_spaces),
        pd.DataFrame.from_records(tokens),
    )

def _prompt_metadata_table(buffer: EpisodeBuffer) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for call in buffer.calls:
        metadata = call.policy_call_metadata.get("prompt_metadata")
        if not isinstance(metadata, dict):
            metadata = (
                call.token_metadata.policy_call_metadata.get("prompt_metadata")
                if call.token_metadata is not None
                else {}
            )
        if not isinstance(metadata, dict):
            metadata = {}
        records.append(
            {
                "policy_call_index": int(call.call_index),
                "prompt_id": "prompt.default",
                "raw_task": buffer.prompt,
                "cleaned_task": str(metadata.get("prompt") or buffer.prompt),
                "formatted_prompt": str(
                    metadata.get("formatted_prompt") or metadata.get("prompt") or ""
                ),
                "state_bin_count": metadata.get("state_bin_count"),
                "metadata": _json_dumps(metadata),
            }
        )
    return pd.DataFrame.from_records(records)

def _image_preprocessing_table(buffer: EpisodeBuffer) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for call in buffer.calls:
        metadata = call.policy_call_metadata.get("image_preprocessing_metadata")
        if not isinstance(metadata, dict) and call.token_metadata is not None:
            metadata = call.token_metadata.policy_call_metadata.get("image_preprocessing_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        cameras = _trace_cameras(buffer)
        for slot_index, camera_id in enumerate(cameras):
            records.append(
                {
                    "policy_call_index": int(call.call_index),
                    "slot_index": slot_index,
                    "feature_key": camera_id,
                    "camera_id": camera_id,
                    "present": True,
                    "raw_shape": _json_dumps(metadata.get("raw_shape")),
                    "processed_shape": _json_dumps(metadata.get("processed_shape")),
                    "resize_mode": str(metadata.get("resize_mode") or ""),
                    "value_range": str(metadata.get("value_range") or ""),
                    "metadata": _json_dumps(metadata),
                }
            )
    return pd.DataFrame.from_records(records)

def _action_normalization_table(buffer: EpisodeBuffer) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for call in buffer.calls:
        metadata = call.policy_call_metadata.get("action_normalization_metadata")
        if not isinstance(metadata, dict) and call.token_metadata is not None:
            metadata = call.token_metadata.policy_call_metadata.get("action_normalization_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        records.append(
            {
                "policy_call_index": int(call.call_index),
                "normalization_id": "lerobot.pi05.action",
                "mode": str(metadata.get("normalization_type") or "checkpoint"),
                "stats_ref": str(metadata.get("stats_ref") or ""),
                "action_dim_names": _json_dumps(metadata.get("action_names")),
                "normalized_action_array_ref": "action_chunks",
                "unnormalized_action_array_ref": "action",
                "metadata": _json_dumps(metadata),
            }
        )
    return pd.DataFrame.from_records(records)

def _evaluation_table(buffer: EpisodeBuffer, length: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    done = _pad_bool(buffer.terminated, length)
    truncated = _pad_bool(buffer.truncated, length)
    for timestep, reward in enumerate(buffer.rewards[:length]):
        info = buffer.infos[timestep] if timestep < len(buffer.infos) else {}
        is_success = _info_bool(info, "is_success")
        records.append(
            {
                "timestep": timestep,
                "metric_name": "reward",
                "metric_value": float(reward),
                "threshold": np.nan,
                "passed": bool(reward > 0.0),
                "source": "env.step",
                "metadata": _json_dumps(info),
            }
        )
        records.append(
            {
                "timestep": timestep,
                "metric_name": "done",
                "metric_value": float(done[timestep]),
                "threshold": 1.0,
                "passed": bool(done[timestep]),
                "source": "env.step",
                "metadata": _json_dumps(info),
            }
        )
        records.append(
            {
                "timestep": timestep,
                "metric_name": "truncated",
                "metric_value": float(truncated[timestep]),
                "threshold": 1.0,
                "passed": bool(truncated[timestep]),
                "source": "env.step",
                "metadata": _json_dumps(info),
            }
        )
        if is_success is not None:
            records.append(
                {
                    "timestep": timestep,
                    "metric_name": "success",
                    "metric_value": float(is_success),
                    "threshold": 1.0,
                    "passed": bool(is_success),
                    "source": "env.step.info",
                    "metadata": _json_dumps(info),
                }
            )
    return pd.DataFrame.from_records(records)

def _context_table(context: ContextCaptureResult, component: str) -> pd.DataFrame:
    availability = context.availability
    if availability.empty or "component" not in availability:
        return pd.DataFrame()
    return availability.loc[availability["component"].astype(str) == component].reset_index(
        drop=True
    )

def _scene_state_table(context: ContextCaptureResult) -> pd.DataFrame:
    frames = []
    if "objects" in context.tables and not context.tables["objects"].empty:
        frame = context.tables["objects"].copy()
        frame["context_kind"] = "object"
        frames.append(frame)
    if "episode_context" in context.tables and not context.tables["episode_context"].empty:
        frame = context.tables["episode_context"].copy()
        frame["context_kind"] = "episode"
        frames.append(frame)
    object_availability = _context_table(context, "object")
    if not object_availability.empty:
        object_availability["context_kind"] = "availability"
        frames.append(object_availability)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

def _camera_state_table(context: ContextCaptureResult) -> pd.DataFrame:
    if "cameras" in context.tables:
        return context.tables["cameras"].copy()
    return _context_table(context, "camera")

def _context_availability_fields(
    context: ContextCaptureResult,
) -> tuple[list[str], list[dict[str, Any]]]:
    availability = context.availability
    if availability.empty:
        return [], []
    captured: list[str] = []
    unavailable: list[dict[str, Any]] = []
    for row in availability.to_dict("records"):
        field = f"{row.get('component')}.{row.get('field')}"
        if bool(row.get("available")):
            captured.append(field)
        else:
            unavailable.append(
                {
                    "field": field,
                    "reason": str(row.get("reason") or ""),
                }
            )
    return captured, unavailable
