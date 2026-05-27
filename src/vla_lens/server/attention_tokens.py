"""Token layout and attention-display helpers."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Mapping

import numpy as np

from vla_lens.server.common import _json_scalar, _optional_int, _patches_per_image, _round
from vla_lens.traces import TraceBundle

_NUMERIC_TOKEN_RE = re.compile(r"^-?\d+(?:\.0)?$")


def _camera_patch_layout(
    bundle: TraceBundle,
    token_count: int,
    *,
    text_tokens: int,
) -> dict[str, int]:
    image_tokens = max(0, token_count - text_tokens)
    camera_count = max(1, len(bundle.cameras()))
    candidate = image_tokens // camera_count if image_tokens % camera_count == 0 else image_tokens
    root = int(round(float(np.sqrt(max(1, candidate)))))
    patches_per_image = candidate if root * root == candidate else _patches_per_image(image_tokens)
    grid_size = int(round(float(np.sqrt(patches_per_image)))) if patches_per_image else 0
    image_slots = image_tokens // patches_per_image if patches_per_image else 0
    return {
        "grid_size": grid_size,
        "grid_height": grid_size,
        "grid_width": grid_size,
        "patches_per_image": patches_per_image,
        "image_tokens": image_tokens,
        "text_tokens": text_tokens,
        "image_slots": min(image_slots, len(bundle.cameras())),
    }

def _camera_patch_layout_from_record(
    bundle: TraceBundle,
    record: Any,
    token_count: int,
    *,
    text_tokens: int,
) -> dict[str, int]:
    metadata = json.loads(str(record.get("metadata") or "{}"))
    patches_per_image = int(metadata.get("patches_per_image") or 0)
    grid_height = int(metadata.get("grid_height") or metadata.get("grid_size") or 0)
    grid_width = int(metadata.get("grid_width") or metadata.get("grid_size") or 0)
    if patches_per_image > 0 and grid_height > 0 and grid_width > 0:
        image_tokens = max(0, token_count - text_tokens)
        return {
            "grid_size": grid_height if grid_height == grid_width else 0,
            "grid_height": grid_height,
            "grid_width": grid_width,
            "patches_per_image": patches_per_image,
            "image_tokens": image_tokens,
            "text_tokens": text_tokens,
            "image_slots": min(image_tokens // patches_per_image, len(bundle.cameras())),
        }
    return _camera_patch_layout(bundle, token_count, text_tokens=text_tokens)

def _token_rows_for_space(
    bundle: TraceBundle,
    call: dict[str, Any],
    token_space_id: str,
) -> Any:
    rows = bundle.tokens
    if rows.empty or "token_space_id" not in rows:
        return rows.iloc[0:0].copy()
    rows = rows.loc[rows["token_space_id"].astype(str) == token_space_id].copy()
    if rows.empty:
        return rows
    if "policy_call_index" in rows:
        call_index = int(call.get("model_call_index", call.get("index", 0)))
        call_rows = rows.loc[rows["policy_call_index"].astype(int) == call_index].copy()
        if not call_rows.empty:
            rows = call_rows
    return rows.sort_values("token_index").reset_index(drop=True)

def _token_count(rows: Any) -> int:
    if rows.empty or "token_index" not in rows:
        return 0
    return int(rows["token_index"].max()) + 1

def _image_token_rows_for_site(
    bundle: TraceBundle,
    record: Any,
    call: dict[str, Any],
    token_count: int,
) -> Any:
    token_space_id = str(record.get("token_space_id") or "")
    if not token_space_id or token_space_id.lower() == "nan":
        return bundle.tokens.iloc[0:0].copy()
    rows = _token_rows_for_space(bundle, call, token_space_id)
    if rows.empty:
        return rows
    token_kind = rows.get("token_kind", "").astype(str)
    image_rows = rows.loc[token_kind == "image"].copy()
    if image_rows.empty or "token_index" not in image_rows:
        return image_rows
    image_rows = image_rows.loc[image_rows["token_index"].astype(int) < token_count].copy()
    return image_rows.sort_values("token_index").reset_index(drop=True)

def _camera_patch_maps_from_token_rows(
    bundle: TraceBundle,
    image_rows: Any,
    feature_values: np.ndarray,
) -> tuple[dict[str, Any], dict[str, int | None]]:
    maps: dict[str, Any] = {}
    grid_heights: list[int] = []
    grid_widths: list[int] = []
    patch_counts: list[int] = []
    for camera in bundle.cameras():
        camera_rows = image_rows.loc[image_rows.get("camera_id", "").astype(str) == camera].copy()
        if camera_rows.empty:
            continue
        grid_height = int(camera_rows.get("patch_row", 0).max()) + 1
        grid_width = int(camera_rows.get("patch_col", 0).max()) + 1
        values = np.full((grid_height, grid_width), np.nan, dtype=np.float32)
        for row in camera_rows.to_dict("records"):
            token_index = int(row.get("token_index", 0))
            if token_index >= feature_values.shape[0]:
                continue
            patch_row = int(row.get("patch_row", 0))
            patch_col = int(row.get("patch_col", 0))
            if patch_row < grid_height and patch_col < grid_width:
                values[patch_row, patch_col] = float(feature_values[token_index])
        grid_heights.append(grid_height)
        grid_widths.append(grid_width)
        patch_counts.append(int(len(camera_rows)))
        finite_values = np.nan_to_num(values, nan=0.0)
        maps[camera] = {
            "values": _round(finite_values),
            "token_start": int(camera_rows["token_index"].min()),
            "token_end": int(camera_rows["token_index"].max()),
            "active_tokens": int(len(camera_rows)),
            "min": _json_scalar(float(np.nanmin(values))),
            "max": _json_scalar(float(np.nanmax(values))),
        }
    grid_height = grid_heights[0] if grid_heights and len(set(grid_heights)) == 1 else None
    grid_width = grid_widths[0] if grid_widths and len(set(grid_widths)) == 1 else None
    patches_per_image = patch_counts[0] if patch_counts and len(set(patch_counts)) == 1 else 0
    return maps, {
        "grid_size": grid_height if grid_height is not None and grid_height == grid_width else None,
        "grid_height": grid_height or 0,
        "grid_width": grid_width or 0,
        "patches_per_image": patches_per_image,
    }

def _image_token_index_for_patch(
    bundle: TraceBundle,
    record: Any,
    call: dict[str, Any],
    token_count: int,
    camera: str,
    row: int,
    col: int,
) -> tuple[int | None, int, int]:
    image_rows = _image_token_rows_for_site(bundle, record, call, token_count)
    if image_rows.empty:
        return None, row, col
    camera_rows = image_rows.loc[image_rows.get("camera_id", "").astype(str) == camera].copy()
    if camera_rows.empty:
        return None, row, col
    max_row = int(camera_rows.get("patch_row", 0).max())
    max_col = int(camera_rows.get("patch_col", 0).max())
    row = max(0, min(row, max_row))
    col = max(0, min(col, max_col))
    matches = camera_rows.loc[
        (camera_rows.get("patch_row", 0).astype(int) == row)
        & (camera_rows.get("patch_col", 0).astype(int) == col)
    ]
    if matches.empty:
        return None, row, col
    return int(matches.iloc[0].get("token_index", 0)), row, col

def _image_attention_from_prefix_rows(
    bundle: TraceBundle,
    prefix_rows: Any,
    prefix_mass: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    image_rows = prefix_rows.loc[prefix_rows.get("token_kind", "").astype(str) == "image"].copy()
    if image_rows.empty:
        return {}, [], 0.0
    cameras = bundle.cameras()
    maps: dict[str, Any] = {}
    patch_rows: list[dict[str, Any]] = []
    image_mass = 0.0
    for camera in cameras:
        camera_rows = image_rows.loc[image_rows.get("camera_id", "").astype(str) == camera].copy()
        if camera_rows.empty:
            continue
        max_row = int(camera_rows.get("patch_row", 0).max())
        max_col = int(camera_rows.get("patch_col", 0).max())
        values = np.full((max_row + 1, max_col + 1), np.nan, dtype=np.float32)
        for row in camera_rows.to_dict("records"):
            token_index = int(row.get("token_index", 0))
            if token_index >= prefix_mass.shape[0]:
                continue
            patch_row = int(row.get("patch_row", 0))
            patch_col = int(row.get("patch_col", 0))
            attention = float(prefix_mass[token_index])
            values[patch_row, patch_col] = attention
            image_mass += attention
            patch_rows.append(
                {
                    "camera": camera,
                    "row": patch_row,
                    "col": patch_col,
                    "token_index": token_index,
                    "attention": _json_scalar(attention),
                }
            )
        if values.size:
            maps[camera] = {
                "values": _round(np.nan_to_num(values, nan=0.0)),
                "token_start": int(camera_rows["token_index"].min()),
                "token_end": int(camera_rows["token_index"].max()),
                "min": _json_scalar(float(np.nanmin(values))),
                "max": _json_scalar(float(np.nanmax(values))),
            }
    patch_rows.sort(key=lambda item: float(item.get("attention") or 0.0), reverse=True)
    return maps, patch_rows[:24], image_mass

def _prompt_attention_from_prefix_rows(
    prefix_rows: Any,
    prefix_mass: np.ndarray,
) -> tuple[list[dict[str, Any]], float, str, list[dict[str, Any]]]:
    text_rows = prefix_rows.loc[prefix_rows.get("token_kind", "").astype(str) == "language"].copy()
    if text_rows.empty:
        return [], 0.0, "", []
    if "attention_mask" in text_rows:
        active = text_rows["attention_mask"].astype(bool)
        active_rows = text_rows.loc[active].copy()
    else:
        active_rows = text_rows
    if active_rows.empty:
        return [], 0.0, "", []
    start = int(text_rows["token_index"].min())
    token_records: list[dict[str, Any]] = []
    prompt_pieces: list[str] = []
    prompt_mass = 0.0
    for row in active_rows.to_dict("records"):
        token_index = int(row.get("token_index", 0))
        if token_index >= prefix_mass.shape[0]:
            continue
        attention = float(prefix_mass[token_index])
        prompt_mass += attention
        token_piece = _display_token_piece(row)
        prompt_pieces.append(token_piece)
        token_records.append(
            {
                "local_index": token_index - start,
                "prefix_index": token_index,
                "token_id": _json_scalar(row.get("token_id")),
                "token_piece": token_piece,
                "attention": _json_scalar(attention),
            }
        )
    top_records = sorted(
        token_records,
        key=lambda item: float(item.get("attention") or 0.0),
        reverse=True,
    )
    return (
        top_records[:24],
        prompt_mass,
        _join_token_pieces(prompt_pieces),
        token_records,
    )

def _display_token_piece(row: Mapping[str, Any]) -> str:
    """Return a human-readable token piece for numeric tokenizer rows."""

    raw_piece = row.get("token_piece")
    token_id = _optional_int(row.get("token_id"))
    piece = "" if raw_piece is None else str(raw_piece)
    if token_id is not None and (not piece or _NUMERIC_TOKEN_RE.match(piece)):
        decoded = _decode_paligemma_token(token_id)
        if decoded:
            piece = decoded
    return _clean_token_piece(piece)

@lru_cache(maxsize=4096)
def _decode_paligemma_token(token_id: int) -> str:
    tokenizer = _paligemma_tokenizer()
    if tokenizer is None:
        return ""
    try:
        piece = tokenizer.convert_ids_to_tokens([int(token_id)])
    except Exception:
        return ""
    if isinstance(piece, str):
        return piece
    if piece:
        return str(piece[0])
    return ""

@lru_cache(maxsize=1)
def _paligemma_tokenizer() -> Any | None:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            "google/paligemma-3b-pt-224",
            local_files_only=True,
        )
    except Exception:
        return None

def _clean_token_piece(piece: str) -> str:
    text = str(piece)
    text = text.replace("<0x0A>", "\n")
    text = text.replace("Ċ", "\n")
    return text

def _join_token_pieces(pieces: list[str]) -> str:
    text = "".join(piece.replace("▁", " ") for piece in pieces)
    text = text.replace("  ", " ")
    return text.strip()

def _not_captured_in_profile(reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "available": False,
        "reason": "not_captured_in_profile",
        "detail": reason,
        **extra,
    }
