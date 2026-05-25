"""Axis-aware activation selection and feature caching."""

from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import zarr

from vla_lens.traces import TraceBundle, TraceDataset

TokenReduction = Literal["mean", "flat", "none"]


@dataclass(frozen=True, slots=True)
class ActivationQuery:
    """Describe one slice of activation data across a ``TraceDataset``."""

    episodes: Mapping[str, Any] = field(default_factory=dict)
    name: str | None = None
    module: str | None = None
    layers: Sequence[int] | None = None
    tensor_type: str | None = None
    token_kind: str | None = None
    timesteps: Any = "all"
    policy_calls: Any = "all"
    generation_step: int | str | None = None
    reduce_tokens: TokenReduction = "mean"
    dtype: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["layers"] = list(self.layers) if self.layers is not None else None
        payload["timesteps"] = _jsonable_timesteps(self.timesteps)
        payload["policy_calls"] = _jsonable_timesteps(self.policy_calls)
        return payload


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """Materialized selector output."""

    X: np.ndarray
    rows: pd.DataFrame
    selector: ActivationQuery
    cache_key: str


class FeatureView:
    """Lazy view from a dataset selector to a probe-ready feature matrix."""

    def __init__(self, dataset: TraceDataset, selector: ActivationQuery):
        self.dataset = dataset
        self.selector = selector

    def to_matrix(
        self,
        *,
        cache: bool = True,
        mmap: bool = False,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        matrix = self.materialize(cache=cache, mmap=mmap)
        return matrix.X, matrix.rows

    def materialize(self, *, cache: bool = True, mmap: bool = False) -> FeatureMatrix:
        key = self.cache_key()
        cache_path = self._cache_path(key)
        rows_path = cache_path / "rows.parquet"
        x_path = cache_path / "X.zarr"
        if cache and rows_path.exists() and x_path.exists():
            del mmap
            return FeatureMatrix(
                X=np.asarray(zarr.open_array(str(x_path), mode="r")),
                rows=pd.read_parquet(rows_path),
                selector=self.selector,
                cache_key=key,
            )

        X, rows = self._compute()
        if cache:
            cache_path.mkdir(parents=True, exist_ok=True)
            store = zarr.open_array(
                str(x_path),
                mode="w",
                shape=X.shape,
                dtype=X.dtype,
                chunks=_cache_chunks(X.shape),
            )
            store[...] = X
            rows.to_parquet(rows_path, index=False)
        return FeatureMatrix(X=X, rows=rows, selector=self.selector, cache_key=key)

    def cache_key(self) -> str:
        payload = {
            "selector": self.selector.to_dict(),
            "episodes": self.dataset.episode_index[["trace_id", "length"]].to_dict("records"),
            "model_sites": _cache_activation_records(self.dataset.model_site_index),
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:20]

    def _cache_path(self, key: str) -> Path:
        return self.dataset.cache_dir() / "features" / key

    def _compute(self) -> tuple[np.ndarray, pd.DataFrame]:
        model_sites = self._matching_model_sites()
        vectors: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        expected_shape: tuple[int, ...] | None = None

        for row in model_sites.to_dict("records"):
            bundle = self.dataset.bundle(str(row["trace_id"]))
            array = bundle.model_site(str(row["name"]), mmap=True)
            axes = json.loads(row["axes"])
            samples = _resolve_samples(
                bundle,
                array,
                axes,
                timesteps=self.selector.timesteps,
                policy_calls=self.selector.policy_calls,
            )
            token_indices = _token_indices(bundle, self.selector.token_kind)
            if token_indices is not None and len(token_indices) == 0:
                continue

            for sample_axis, sample_index in samples:
                value, remaining_axes = _select_value(
                    array=array,
                    axes=axes,
                    sample_axis=sample_axis,
                    sample_index=sample_index,
                    token_indices=token_indices,
                    generation_step=self.selector.generation_step,
                )
                vector = _reduce_value(value, remaining_axes, self.selector.reduce_tokens)
                vector = vector.astype(self.selector.dtype, copy=False).reshape(-1)
                if vector.size and not np.isfinite(vector).any():
                    continue
                if expected_shape is None:
                    expected_shape = vector.shape
                elif vector.shape != expected_shape:
                    raise ValueError(
                        "Selected model_sites do not share a feature shape: "
                        f"expected {expected_shape}, got {vector.shape} for {row['name']}"
                    )

                vectors.append(vector)
                policy_call = sample_index if sample_axis == "policy_call" else None
                timestep = (
                    _policy_call_timestep(bundle, int(policy_call))
                    if policy_call is not None
                    else sample_index
                )
                if policy_call is None and sample_axis == "timestep" and sample_index is not None:
                    policy_call = _policy_call_for_timestep(bundle, int(sample_index))
                records.append(
                    {
                        "input_row_index": len(records),
                        "trace_id": row["trace_id"],
                        "episode_id": row.get("episode_id"),
                        "timestep": timestep,
                        "policy_call": policy_call,
                        "policy_call_index": policy_call,
                        "activation": row["name"],
                        "model_site_id": row.get("site_id") or row["name"],
                        "token_space_id": row.get("token_space_id"),
                        "axes": row.get("axes"),
                        "dtype": row.get("dtype"),
                        "module": row.get("module"),
                        "layer": row.get("layer"),
                        "tensor_type": row.get("tensor_type"),
                        "token_kind": self.selector.token_kind or row.get("token_kind"),
                        "generation_step": self.selector.generation_step,
                        "selection_reduction": self.selector.reduce_tokens,
                        "feature_dim": int(vector.shape[0]),
                    }
                )

        if not vectors:
            return np.empty((0, 0), dtype=self.selector.dtype), pd.DataFrame()
        return np.stack(vectors), pd.DataFrame.from_records(records)

    def _matching_model_sites(self) -> pd.DataFrame:
        index = self.dataset.model_site_index
        if index.empty:
            return index

        episode_ids = set(self.dataset.episodes(**dict(self.selector.episodes))["trace_id"])
        index = index.loc[index["trace_id"].isin(episode_ids)]

        if self.selector.name is not None:
            index = index.loc[index["name"].astype(str).map(_matches(self.selector.name))]
        if self.selector.module is not None and "module" in index:
            index = index.loc[index["module"].astype(str).map(_matches(self.selector.module))]
        if self.selector.layers is not None and "layer" in index:
            layers = {int(layer) for layer in self.selector.layers}
            index = index.loc[index["layer"].map(_maybe_int).isin(layers)]
        if self.selector.tensor_type is not None and "tensor_type" in index:
            index = index.loc[index["tensor_type"].astype(str) == self.selector.tensor_type]
        if self.selector.token_kind is not None and "token_kind" in index:
            token_kind = self.selector.token_kind
            token_column = index["token_kind"]
            index = index.loc[token_column.isna() | (token_column.astype(str) == token_kind)]
        if self.selector.generation_step is not None and "generation_step" in index:
            axes_column = index.get("axes", pd.Series("", index=index.index)).astype(str)
            has_generation_axis = axes_column.str.contains('"generation_step"', regex=False)
            index = index.loc[
                has_generation_axis
                | index["generation_step"].map(_matches_value(self.selector.generation_step))
            ]
        return index.reset_index(drop=True)


def _select_value(
    *,
    array: np.ndarray,
    axes: Sequence[str],
    sample_axis: str | None,
    sample_index: int | None,
    token_indices: np.ndarray | None,
    generation_step: int | str | None,
) -> tuple[np.ndarray, list[str]]:
    value = array
    remaining_axes = list(axes)

    if sample_axis is not None and sample_index is not None and sample_axis in remaining_axes:
        axis = remaining_axes.index(sample_axis)
        value = np.take(value, int(sample_index), axis=axis)
        remaining_axes.pop(axis)

    if generation_step is not None and "generation_step" in remaining_axes:
        axis = remaining_axes.index("generation_step")
        index = _generation_step_index(int(value.shape[axis]), generation_step)
        value = np.take(value, index, axis=axis)
        remaining_axes.pop(axis)

    if token_indices is not None and "token" in remaining_axes:
        axis = remaining_axes.index("token")
        value = np.take(value, token_indices, axis=axis)

    return np.asarray(value), remaining_axes


def _generation_step_index(count: int, generation_step: int | str) -> int:
    if str(generation_step) == "final":
        return max(0, count - 1)
    index = int(generation_step)
    if index < 0:
        return max(0, count + index)
    return min(index, max(0, count - 1))


def _reduce_value(
    value: np.ndarray,
    axes: Sequence[str],
    reduction: TokenReduction,
) -> np.ndarray:
    reduced = np.asarray(value)
    remaining_axes = list(axes)
    if reduction == "mean" and "token" in remaining_axes:
        axis = remaining_axes.index("token")
        reduced = reduced.mean(axis=axis)
        remaining_axes.pop(axis)
    if reduction == "flat":
        return reduced.reshape(-1)
    if reduction == "none":
        if reduced.ndim != 1:
            raise ValueError(f"Reduction 'none' requires a vector, got shape {reduced.shape}")
        return reduced
    if reduction == "mean":
        return reduced.reshape(-1)
    raise ValueError(f"Unknown token reduction: {reduction}")


def _resolve_samples(
    bundle: TraceBundle,
    array: np.ndarray,
    axes: Sequence[str],
    *,
    timesteps: Any,
    policy_calls: Any,
) -> list[tuple[str | None, int | None]]:
    if "timestep" in axes:
        count = int(array.shape[axes.index("timestep")])
        return [("timestep", item) for item in _axis_indices(count, timesteps)]
    if "policy_call" in axes:
        count = int(array.shape[axes.index("policy_call")])
        if policy_calls != "all":
            return [
                ("policy_call", item)
                for item in _axis_indices(count, policy_calls)
                if 0 <= item < count
            ]
        if timesteps != "all":
            indices = _policy_calls_for_timesteps(bundle, timesteps)
            return [("policy_call", item) for item in indices if 0 <= item < count]
        return [("policy_call", item) for item in range(count)]
    return [(None, None)]


def _axis_indices(count: int, value: Any) -> list[int]:
    if value == "all":
        return list(range(count))
    if isinstance(value, int):
        return [value]
    if isinstance(value, slice):
        return list(range(count))[value]
    return [int(item) for item in value]


def _policy_calls_for_timesteps(bundle: TraceBundle, timesteps: Any) -> list[int]:
    requested = set(_axis_indices(max(1, bundle.manifest.length), timesteps))
    table = bundle.timesteps
    if table.empty or "policy_call_index" not in table:
        return []
    out: list[int] = []
    rows = table.loc[table["timestep"].astype(int).isin(requested)]
    for row in rows.to_dict("records"):
        if int(row.get("timestep", -1)) not in requested:
            continue
        value = row.get("policy_call_index")
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        out.append(int(value))
    return sorted(set(out))


def _policy_call_timestep(bundle: TraceBundle, policy_call: int) -> int | None:
    calls = bundle.policy_calls
    if not calls.empty and "policy_call_index" in calls:
        matches = calls.loc[calls["policy_call_index"].astype(int) == int(policy_call)]
        if not matches.empty:
            row = matches.iloc[0]
            return int(row.get("observation_timestep", row.get("env_timestep_start", 0)))
    table = bundle.timesteps
    if table.empty or "policy_call_index" not in table:
        return None
    matches = table.loc[table["policy_call_index"].fillna(-1).astype(int) == int(policy_call)]
    if not matches.empty:
        return int(matches.iloc[0]["timestep"])
    return None


def _policy_call_for_timestep(bundle: TraceBundle, timestep: int) -> int | None:
    table = bundle.timesteps
    if table.empty or "policy_call_index" not in table or "timestep" not in table:
        return None
    matches = table.loc[table["timestep"].astype(int) == int(timestep)]
    if matches.empty:
        return None
    value = matches.iloc[0].get("policy_call_index")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return int(value)


def _token_indices(bundle: TraceBundle, token_kind: str | None) -> np.ndarray | None:
    if token_kind is None:
        return None
    metadata = bundle.tokens
    if metadata.empty or "token_index" not in metadata or "token_kind" not in metadata:
        return None
    selected = metadata.loc[metadata["token_kind"].astype(str) == token_kind, "token_index"]
    if selected.empty:
        return np.array([], dtype=np.int64)
    return selected.astype(np.int64).to_numpy()


def _matches(pattern: str):
    def inner(value: str) -> bool:
        return fnmatch.fnmatchcase(value, pattern)

    return inner


def _matches_value(expected: int | str):
    def inner(value: Any) -> bool:
        if expected == "final":
            return str(value) == "final"
        try:
            return int(value) == int(expected)
        except (TypeError, ValueError):
            return str(value) == str(expected)

    return inner


def _maybe_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return int(value)


def _jsonable_timesteps(value: Any) -> Any:
    if isinstance(value, slice):
        return {"slice": [value.start, value.stop, value.step]}
    if isinstance(value, range):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _cache_chunks(shape: Sequence[int]) -> tuple[int, ...]:
    if not shape:
        return (1,)
    if len(shape) == 1:
        return (max(1, min(int(shape[0]), 4096)),)
    return (
        max(1, min(int(shape[0]), 1024)),
        max(1, min(int(shape[1]), 1024)),
        *tuple(max(1, int(item)) for item in shape[2:]),
    )


def _cache_activation_records(index: pd.DataFrame) -> list[dict[str, Any]]:
    if index.empty:
        return []
    columns = [
        column
        for column in ["trace_id", "name", "module", "layer", "tensor_type", "shape", "dtype"]
        if column in index
    ]
    return index[columns].to_dict("records")


__all__ = ["ActivationQuery", "FeatureMatrix", "FeatureView"]
