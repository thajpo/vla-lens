"""Low-level trace table, array, and media IO helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc

from vla_lens.artifacts import slugify
from vla_lens.traces.layout import ARRAY_COMPRESSION


def _array_record(
    *,
    name: str,
    relative_path: Path,
    array: np.ndarray,
    axes: Sequence[str],
    array_type: str,
    metadata: Mapping[str, Any],
    chunks: Sequence[int],
    storage_format: str,
    compression: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "relative_path": str(relative_path),
        "storage_format": storage_format,
        "chunks": json.dumps([int(item) for item in chunks]),
        "compression": compression,
        "array_type": array_type,
        "shape": json.dumps([int(item) for item in array.shape]),
        "dtype": str(array.dtype),
        "axes": json.dumps(list(axes)),
        "metadata": json.dumps(dict(metadata), sort_keys=True),
    }


def _write_table(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _table_or_empty(frame: pd.DataFrame | None) -> pd.DataFrame:
    return frame if frame is not None else pd.DataFrame()


def _bundle_index_by_trace_id(bundles: Sequence[Any]) -> dict[str, Any]:
    index: dict[str, Any] = {}
    duplicates: dict[str, list[str]] = {}
    for bundle in bundles:
        trace_id = str(bundle.manifest.trace_id)
        if trace_id in index:
            duplicates.setdefault(trace_id, [str(index[trace_id].path)]).append(str(bundle.path))
            continue
        index[trace_id] = bundle
    if duplicates:
        details = "; ".join(
            f"{trace_id}: {', '.join(paths)}" for trace_id, paths in sorted(duplicates.items())
        )
        raise ValueError(f"Duplicate trace_id values in dataset: {details}")
    return index


def _nested_lerobot_dataset_roots(root: Path, is_dataset_root: Any) -> tuple[Path, ...]:
    if not root.exists() or not root.is_dir():
        return ()
    roots: list[Path] = []
    seen: set[Path] = set()
    for info_path in root.rglob("meta/info.json"):
        dataset_root = info_path.parent.parent
        if dataset_root == root or dataset_root in seen:
            continue
        if is_dataset_root(dataset_root):
            seen.add(dataset_root)
            roots.append(dataset_root)
    return tuple(sorted(roots))


def _nested_lerobot_trace_id_prefix(root: Path, dataset_root: Path) -> str:
    try:
        relative = dataset_root.relative_to(root)
    except ValueError:
        relative = dataset_root
    return slugify(str(relative), fallback="lerobot")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True), encoding="utf-8")


def _write_zarr_array(path: Path, array: np.ndarray) -> tuple[int, ...]:
    value = np.asarray(array)
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = _default_chunks(value.shape)
    z = zarr.open_array(
        str(path),
        mode="w",
        shape=value.shape,
        dtype=value.dtype,
        chunks=chunks,
        compressor=Blosc(cname=ARRAY_COMPRESSION, clevel=3, shuffle=Blosc.BITSHUFFLE),
    )
    z[:] = value
    return tuple(int(item) for item in z.chunks)


def _read_zarr_array(path: Path) -> zarr.Array:
    if not path.exists():
        raise FileNotFoundError(path)
    return zarr.open_array(str(path), mode="r")


def _write_frame_sequence(path: Path, frames: np.ndarray) -> None:
    value = np.asarray(frames)
    if value.ndim != 4 or value.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"Frame arrays must have shape T x H x W x C, got {value.shape}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(value):
        imageio.imwrite(path / f"{index:06d}.jpg", np.asarray(frame), format="jpg", quality=90)


def _read_frame_sequence(path: Path) -> np.ndarray:
    files = sorted(path.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No JPEG frames found in {path}")
    return np.stack([np.asarray(imageio.imread(file)) for file in files])


def _is_frame_array(name: str) -> bool:
    return str(name).startswith("frames.")


def _frame_camera(name: str) -> str:
    text = str(name)
    return text.split(".", 1)[1] if "." in text else text


def _episode_array_path(name: str) -> Path:
    text = str(name)
    if text in {"executed_actions", "action_chunks", "generation_actions", "generation_velocities"}:
        group = "action"
    elif text.startswith(("robot_", "scene_", "camera_", "evaluation_")):
        group = "context"
    else:
        group = "episode"
    return Path("arrays") / group / f"{slugify(text)}.zarr"


def _default_chunks(shape: Sequence[int]) -> tuple[int, ...]:
    if not shape:
        return ()
    # Keep leading axes reasonably small for episode/time selections and cap
    # feature/image chunk sizes so previews can read bounded regions cheaply.
    chunks: list[int] = []
    for axis, size in enumerate(shape):
        if axis == 0:
            chunks.append(min(int(size), 16))
        elif axis <= 2:
            chunks.append(min(int(size), 64))
        else:
            chunks.append(min(int(size), 128))
    return tuple(max(1, item) for item in chunks)


def _concat_or_empty(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _filter_frame(frame: pd.DataFrame, filters: Mapping[str, Any]) -> pd.DataFrame:
    out = frame
    for column, expected in filters.items():
        if column not in out:
            return out.iloc[0:0].copy()
        if callable(expected):
            out = out.loc[out[column].map(expected)]
        elif isinstance(expected, (set, list, tuple, frozenset)):
            out = out.loc[out[column].isin(expected)]
        else:
            out = out.loc[out[column] == expected]
    return out.copy()


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))

def _validate_artifact_id(artifact_id: str) -> None:
    value = str(artifact_id)
    if not value or Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"Invalid artifact_id: {artifact_id!r}")
    if value in {".", ".."}:
        raise ValueError(f"Invalid artifact_id: {artifact_id!r}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
