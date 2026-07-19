"""Memory-bounded token-preserving feature preparation for probe studies.

The capture already owns the large raw activation arrays.  This module reads
those arrays in episode-sized blocks, learns a small channel projection from
training rows only, and immediately reduces each token.  It never writes a
flattened copy of the raw token tensors.  The reusable cache contains only the
final fixed-width readouts plus the projections required to trace coefficients
back to token positions.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import zarr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from vla_lens.probes.geometry_study import (
    _activation_query,
    _apply_split_contract,
    _geometry_metadata_rows,
    _limited_episode_ids,
)
from vla_lens.selectors import (
    _compact_indices,
    _generation_step_index,
    _policy_call_for_timestep,
    _policy_call_timestep,
    _resolve_samples,
    _token_indices,
)
from vla_lens.traces import TraceBundle, TraceDataset

TOKEN_REPRESENTATION_CACHE_SCHEMA_VERSION = 5


@dataclass(frozen=True, slots=True)
class ProjectionState:
    """A standardization plus PCA transform that can be replayed exactly."""

    input_center: np.ndarray
    input_scale: np.ndarray
    pca_center: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[0])

    def transform(self, values: np.ndarray) -> np.ndarray:
        scaled = (np.asarray(values, dtype=np.float32) - self.input_center) / self.input_scale
        return np.asarray((scaled - self.pca_center) @ self.components.T, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class LayerTokenReadouts:
    """Aligned fixed-width readouts for pooled and token-preserving probes."""

    rows: pd.DataFrame
    source_sites: pd.DataFrame
    token_metadata: pd.DataFrame
    layers: tuple[int, ...]
    pooled: np.ndarray
    tokenwise: np.ndarray
    channel_projection: ProjectionState
    pooled_projection: ProjectionState
    tokenwise_projection: ProjectionState
    token_count: int
    channel_dim: int
    cache_key: str


def build_layer_token_readouts(
    dataset: TraceDataset,
    feature_spec: Mapping[str, Any],
    split: Mapping[str, Any],
    *,
    readout_dim: int,
    token_channel_dim: int = 16,
    channel_sample_count: int = 50_000,
    projection_fit_rows: int = 10_000,
    io_workers: int = 8,
    limit_episodes: int | None = None,
    cache: bool = True,
) -> LayerTokenReadouts:
    """Build matched layer readouts without saving a raw flattened tensor cache."""

    if readout_dim < 1 or token_channel_dim < 1:
        raise ValueError("Projection dimensions must be positive")
    query = _activation_query({**dict(feature_spec), "reduction": "mean"})
    view = dataset.select_model_sites(query)
    sites = view._matching_model_sites()  # noqa: SLF001 - shared selector contract
    limited_ids = _limited_episode_ids(dataset, limit_episodes)
    if limited_ids is not None:
        sites = sites.loc[sites["trace_id"].astype(str).isin(limited_ids)].copy()
    sites = _token_site_rows(sites)
    if query.generation_step is None and sites["axes"].astype(str).str.contains(
        '"generation_step"', regex=False
    ).any():
        raise ValueError(
            "Token-preserving studies require an explicit generation_step when the "
            "captured arrays retain that axis"
        )
    layers = _selected_layers(sites, feature_spec.get("layers"))
    rows, source_sites = _source_rows(dataset, sites, layers, query)
    rows = _geometry_metadata_rows(dataset, rows, cache=True)
    rows = _apply_split_contract(rows, split).reset_index(drop=True)
    rows["representation_row_index"] = np.arange(len(rows), dtype=np.int64)
    if rows.empty:
        raise ValueError("Token selector produced no complete cross-layer scene rows")
    split_column = str(split["column"])
    train_value = str(split["train_value"])
    if split_column not in rows:
        raise KeyError(f"Token representation rows are missing split column {split_column!r}")
    train_mask = rows[split_column].astype(str).to_numpy() == train_value
    if not train_mask.any():
        raise ValueError("Token representation has no training rows")

    token_metadata, token_indices, topology_key = _common_token_topology(
        dataset,
        rows,
        source_sites,
        query.token_kind,
    )
    if len(token_indices) == 0:
        raise ValueError(f"No tokens matched token kind {query.token_kind!r}")

    cache_key = _cache_key(
        view.cache_key(),
        rows,
        layers,
        readout_dim=readout_dim,
        token_channel_dim=token_channel_dim,
        channel_sample_count=channel_sample_count,
        projection_fit_rows=projection_fit_rows,
        io_workers=io_workers,
        token_topology=topology_key,
    )
    cache_path = dataset.cache_dir() / "token_representations" / cache_key
    if cache:
        cached = _load_cache(cache_path, cache_key)
        if cached is not None:
            return cached

    channel_samples = _sample_channel_vectors(
        dataset,
        rows,
        source_sites,
        layers,
        token_indices,
        generation_step=query.generation_step,
        train_mask=train_mask,
        sample_count=channel_sample_count,
        io_workers=io_workers,
    )
    channel_projection = _fit_projection(
        channel_samples,
        min(token_channel_dim, channel_samples.shape[1]),
        random_state=0,
    )
    pooled_raw, compressed_tokens = _read_and_compress(
        dataset,
        rows,
        source_sites,
        layers,
        token_indices,
        generation_step=query.generation_step,
        channel_projection=channel_projection,
        io_workers=io_workers,
    )
    token_count = int(compressed_tokens.shape[2])
    pooled_projection, pooled = _fit_layer_projection(
        pooled_raw,
        train_mask,
        readout_dim,
        projection_fit_rows,
        random_state=1,
    )
    tokenwise_projection, tokenwise = _fit_layer_projection(
        compressed_tokens.reshape(len(rows), len(layers), -1),
        train_mask,
        readout_dim,
        projection_fit_rows,
        random_state=2,
    )
    result = LayerTokenReadouts(
        rows=rows,
        source_sites=source_sites,
        token_metadata=token_metadata,
        layers=layers,
        pooled=pooled,
        tokenwise=tokenwise,
        channel_projection=channel_projection,
        pooled_projection=pooled_projection,
        tokenwise_projection=tokenwise_projection,
        token_count=token_count,
        channel_dim=channel_projection.output_dim,
        cache_key=cache_key,
    )
    if cache:
        _save_cache(cache_path, result)
    return result


def _token_site_rows(sites: pd.DataFrame) -> pd.DataFrame:
    if sites.empty:
        raise ValueError("Token selector matched no model sites")
    token_rows = sites.loc[
        sites["axes"].astype(str).str.contains('"token"', regex=False)
        & sites["axes"].astype(str).str.contains('"channel"', regex=False)
    ].copy()
    token_rows["layer"] = pd.to_numeric(token_rows["layer"], errors="coerce")
    token_rows = token_rows.loc[token_rows["layer"].notna()].copy()
    token_rows["layer"] = token_rows["layer"].astype(int)
    if token_rows.empty:
        raise ValueError("Token selector matched no layered token/channel arrays")
    return token_rows.reset_index(drop=True)


def _selected_layers(sites: pd.DataFrame, requested: Any) -> tuple[int, ...]:
    available = tuple(sorted(int(value) for value in sites["layer"].unique()))
    if requested is None:
        layers = available
    else:
        wanted = {int(value) for value in requested}
        layers = tuple(value for value in available if value in wanted)
    if len(layers) < 2:
        raise ValueError("Layer comparisons require at least two captured layers")
    return layers


def _source_rows(
    dataset: TraceDataset,
    sites: pd.DataFrame,
    layers: tuple[int, ...],
    query: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_records: list[dict[str, Any]] = []
    row_records: list[dict[str, Any]] = []
    for trace_id, trace_sites in sites.groupby("trace_id", sort=True):
        by_layer: dict[int, dict[str, Any]] = {}
        for layer in layers:
            matches = trace_sites.loc[trace_sites["layer"].astype(int) == layer]
            if len(matches) != 1:
                break
            by_layer[layer] = dict(matches.iloc[0])
        if len(by_layer) != len(layers):
            continue
        bundle = dataset.bundle(str(trace_id))
        common: set[tuple[str | None, int | None]] | None = None
        for layer in layers:
            record = by_layer[layer]
            array = bundle.model_site(str(record["name"]), mmap=True)
            samples = set(
                _resolve_samples(
                    bundle,
                    array,
                    _axes(record["axes"]),
                    timesteps=query.timesteps,
                    policy_calls=query.policy_calls,
                )
            )
            common = samples if common is None else common & samples
            source_records.append(
                {
                    "trace_id": str(trace_id),
                    "layer": int(layer),
                    "name": str(record["name"]),
                    "module": record.get("module"),
                    "tensor_type": record.get("tensor_type"),
                    "token_kind": query.token_kind or record.get("token_kind"),
                    "token_space_id": record.get("token_space_id"),
                    "axes": record["axes"],
                    "shape": record["shape"],
                    "dtype": record.get("dtype"),
                }
            )
        for sample_axis, sample_index in sorted(
            common or (), key=lambda value: (-1 if value[1] is None else int(value[1]))
        ):
            if sample_axis == "policy_call" and sample_index is not None:
                policy_call = int(sample_index)
                timestep = _policy_call_timestep(bundle, policy_call)
            else:
                timestep = int(sample_index or 0)
                policy_call = _policy_call_for_timestep(bundle, timestep)
            row_records.append(
                {
                    "trace_id": str(trace_id),
                    "episode_id": bundle.manifest.episode_id,
                    "timestep": int(timestep or 0),
                    "policy_call": policy_call,
                    "policy_call_index": policy_call,
                    "sample_axis": sample_axis,
                    "sample_index": sample_index,
                }
            )
    return pd.DataFrame.from_records(row_records), pd.DataFrame.from_records(source_records)


def _selected_token_metadata(
    bundle: TraceBundle,
    token_kind: str | None,
    token_space_id: str | None = None,
) -> pd.DataFrame:
    metadata = bundle.tokens.copy()
    if token_space_id is not None and "token_space_id" in metadata:
        metadata = metadata.loc[
            metadata["token_space_id"].astype(str) == token_space_id
        ].copy()
    if token_kind is not None and "token_kind" in metadata:
        metadata = metadata.loc[metadata["token_kind"].astype(str) == token_kind].copy()
    if "token_index" in metadata:
        metadata = (
            metadata.sort_values("token_index")
            .drop_duplicates("token_index", keep="first")
            .reset_index(drop=True)
        )
    return metadata


def _common_token_topology(
    dataset: TraceDataset,
    rows: pd.DataFrame,
    source_sites: pd.DataFrame,
    token_kind: str | None,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    reference_metadata: pd.DataFrame | None = None
    reference_indices: np.ndarray | None = None
    reference_token_count: int | None = None
    reference_token_space: str | None = None
    trace_ids = sorted(str(value) for value in rows["trace_id"].unique())
    for trace_id in trace_ids:
        bundle = dataset.bundle(trace_id)
        trace_sites = source_sites.loc[source_sites["trace_id"].astype(str) == trace_id]
        token_counts = {
            _shape(site["shape"])[_axes(site["axes"]).index("token")]
            for site in trace_sites.to_dict("records")
        }
        token_spaces = {
            str(value)
            for value in trace_sites["token_space_id"].dropna()
            if str(value)
        }
        if len(token_counts) != 1:
            raise ValueError(
                "Token-preserving studies require one token count per trace; "
                f"trace {trace_id!r} has {sorted(token_counts)}"
            )
        if len(token_spaces) > 1:
            raise ValueError(
                "Token-preserving studies require one token space per trace; "
                f"trace {trace_id!r} has {sorted(token_spaces)}"
            )
        token_count = int(next(iter(token_counts)))
        token_space = next(iter(token_spaces), None)
        indices = _token_indices(bundle, token_kind)
        if indices is None:
            indices = np.arange(token_count, dtype=np.int64)
        else:
            indices = np.unique(np.asarray(indices, dtype=np.int64))
        if len(indices) and (int(indices.min()) < 0 or int(indices.max()) >= token_count):
            raise ValueError(
                f"Token indices for trace {trace_id!r} exceed its token axis of "
                f"length {token_count}"
            )
        metadata = _selected_token_metadata(bundle, token_kind, token_space)
        if "token_index" in metadata:
            metadata_indices = metadata["token_index"].astype(np.int64).to_numpy()
            if not np.array_equal(metadata_indices, indices):
                raise ValueError(
                    f"Token metadata for trace {trace_id!r} does not describe the "
                    "selected token indices"
                )
        if reference_indices is None:
            reference_indices = indices
            reference_metadata = metadata
            reference_token_count = token_count
            reference_token_space = token_space
            continue
        if (
            token_count != reference_token_count
            or token_space != reference_token_space
            or not np.array_equal(indices, reference_indices)
            or _token_metadata_records(metadata)
            != _token_metadata_records(reference_metadata)
        ):
            raise ValueError(
                "Token-preserving studies require identical token counts, indices, "
                f"and metadata across traces; trace {trace_id!r} differs from "
                f"trace {trace_ids[0]!r}"
            )
    if reference_indices is None or reference_metadata is None:
        raise ValueError("Token-preserving studies found no trace token topology")
    topology_payload = {
        "token_count": reference_token_count,
        "token_space_id": reference_token_space,
        "indices": reference_indices.tolist(),
        "metadata": _token_metadata_records(reference_metadata),
    }
    topology_key = hashlib.sha256(
        json.dumps(topology_payload, sort_keys=True).encode()
    ).hexdigest()[:20]
    return reference_metadata, reference_indices, topology_key


def _token_metadata_records(metadata: pd.DataFrame) -> list[dict[str, Any]]:
    stable = metadata.drop(columns=["policy_call_index"], errors="ignore")
    return json.loads(stable.to_json(orient="records", date_format="iso"))


def _sample_channel_vectors(
    dataset: TraceDataset,
    rows: pd.DataFrame,
    source_sites: pd.DataFrame,
    layers: tuple[int, ...],
    token_indices: np.ndarray,
    *,
    generation_step: int | str | None,
    train_mask: np.ndarray,
    sample_count: int,
    io_workers: int,
) -> np.ndarray:
    tokens_per_scene = min(8, len(token_indices))
    scene_count = max(1, sample_count // max(1, len(layers) * tokens_per_scene))
    eligible = np.flatnonzero(train_mask)
    selected_rows = _even_sample(eligible, scene_count)
    selected_tokens = _even_sample(np.asarray(token_indices, dtype=np.int64), tokens_per_scene)
    selected_trace_ids = rows.iloc[selected_rows]["trace_id"].astype(str).to_numpy()

    def read_trace(trace_id: str) -> list[np.ndarray]:
        trace_row_indices = selected_rows[selected_trace_ids == trace_id]
        if not len(trace_row_indices):
            return []
        bundle = dataset.bundle(trace_id)
        trace_blocks: list[np.ndarray] = []
        trace_sites = source_sites.loc[source_sites["trace_id"].astype(str) == trace_id]
        for layer in layers:
            site = trace_sites.loc[trace_sites["layer"].astype(int) == int(layer)]
            if site.empty:
                continue
            record = site.iloc[0]
            block = _read_block(
                bundle.model_site(str(record["name"]), mmap=True),
                _axes(record["axes"]),
                rows.iloc[trace_row_indices],
                selected_tokens,
                generation_step,
            )
            trace_blocks.append(block.reshape(-1, block.shape[-1]))
        return trace_blocks

    trace_ids = sorted(set(selected_trace_ids))
    with ThreadPoolExecutor(max_workers=max(1, int(io_workers))) as executor:
        nested = executor.map(read_trace, trace_ids)
        blocks = [block for trace_blocks in nested for block in trace_blocks]
    if not blocks:
        raise ValueError("No training token vectors were available for channel PCA")
    values = np.concatenate(blocks, axis=0).astype(np.float32, copy=False)
    if len(values) > sample_count:
        values = values[_even_sample(np.arange(len(values)), sample_count)]
    return values


def _read_and_compress(
    dataset: TraceDataset,
    rows: pd.DataFrame,
    source_sites: pd.DataFrame,
    layers: tuple[int, ...],
    token_indices: np.ndarray,
    *,
    generation_step: int | str | None,
    channel_projection: ProjectionState,
    io_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(rows)
    layer_lookup = {layer: index for index, layer in enumerate(layers)}
    pooled = np.empty(
        (n_rows, len(layers), channel_projection.input_center.size), dtype=np.float32
    )
    compressed = np.empty(
        (n_rows, len(layers), len(token_indices), channel_projection.output_dim),
        dtype=np.float32,
    )
    row_trace_ids = rows["trace_id"].astype(str).to_numpy()

    def read_trace(
        trace_id: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row_indices = np.flatnonzero(row_trace_ids == trace_id)
        local_pooled = np.empty(
            (len(row_indices), len(layers), channel_projection.input_center.size),
            dtype=np.float32,
        )
        local_compressed = np.empty(
            (
                len(row_indices),
                len(layers),
                len(token_indices),
                channel_projection.output_dim,
            ),
            dtype=np.float32,
        )
        bundle = dataset.bundle(trace_id)
        trace_sites = source_sites.loc[source_sites["trace_id"].astype(str) == trace_id]
        for layer in layers:
            site = trace_sites.loc[trace_sites["layer"].astype(int) == int(layer)]
            record = site.iloc[0]
            block = _read_block(
                bundle.model_site(str(record["name"]), mmap=True),
                _axes(record["axes"]),
                rows.iloc[row_indices],
                token_indices,
                generation_step,
            ).astype(np.float32, copy=False)
            layer_index = layer_lookup[int(layer)]
            local_pooled[:, layer_index] = block.mean(axis=1)
            projected = channel_projection.transform(block.reshape(-1, block.shape[-1]))
            local_compressed[:, layer_index] = projected.reshape(
                len(row_indices), len(token_indices), -1
            )
        return row_indices, local_pooled, local_compressed

    trace_ids = sorted(set(row_trace_ids))
    with ThreadPoolExecutor(max_workers=max(1, int(io_workers))) as executor:
        for row_indices, local_pooled, local_compressed in executor.map(
            read_trace, trace_ids
        ):
            pooled[row_indices] = local_pooled
            compressed[row_indices] = local_compressed
    return pooled, compressed


def _read_block(
    array: Any,
    axes: Sequence[str],
    sample_rows: pd.DataFrame,
    token_indices: np.ndarray,
    generation_step: int | str | None,
) -> np.ndarray:
    sample_axes = set(sample_rows["sample_axis"].tolist())
    if len(sample_axes) != 1:
        raise ValueError("One activation block cannot mix sample axes")
    sample_axis = next(iter(sample_axes))
    if sample_axis not in axes or "token" not in axes or "channel" not in axes:
        raise ValueError(f"Expected sample/token/channel axes, got {list(axes)}")
    selection: list[Any] = [slice(None)] * len(axes)
    selection[axes.index(str(sample_axis))] = _compact_indices(
        sample_rows["sample_index"].astype(int).to_numpy()
    )
    selection[axes.index("token")] = _compact_indices(
        np.asarray(token_indices, dtype=np.int64)
    )
    if "generation_step" in axes:
        if generation_step is None:
            raise ValueError("A generation_step is required for this token array")
        selection[axes.index("generation_step")] = _generation_step_index(
            int(array.shape[axes.index("generation_step")]), generation_step
        )
    selected = np.asarray(array.oindex[tuple(selection)])
    remaining = [
        axis
        for axis, indexer in zip(axes, selection, strict=True)
        if not isinstance(indexer, (int, np.integer))
    ]
    expected = {str(sample_axis), "token", "channel"}
    if set(remaining) != expected:
        raise ValueError(f"Token block retained unexpected axes: {remaining}")
    return np.moveaxis(
        selected,
        [remaining.index(str(sample_axis)), remaining.index("token"), remaining.index("channel")],
        [0, 1, 2],
    )


def _fit_layer_projection(
    values: np.ndarray,
    train_mask: np.ndarray,
    output_dim: int,
    fit_rows: int,
    *,
    random_state: int,
) -> tuple[ProjectionState, np.ndarray]:
    train = values[train_mask].reshape(-1, values.shape[-1])
    train = train[_even_sample(np.arange(len(train)), fit_rows)]
    projection = _fit_projection(
        train,
        min(output_dim, train.shape[0] - 1, train.shape[1]),
        random_state=random_state,
    )
    flat = values.reshape(-1, values.shape[-1])
    transformed = np.empty((len(flat), projection.output_dim), dtype=np.float32)
    for start in range(0, len(flat), 2048):
        stop = min(len(flat), start + 2048)
        transformed[start:stop] = projection.transform(flat[start:stop])
    return projection, transformed.reshape(*values.shape[:-1], -1)


def _fit_projection(
    values: np.ndarray,
    output_dim: int,
    *,
    random_state: int,
) -> ProjectionState:
    if output_dim < 1:
        raise ValueError("Not enough rows or columns to fit a projection")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(np.asarray(values, dtype=np.float32))
    projector = PCA(
        n_components=int(output_dim),
        svd_solver="randomized",
        iterated_power=2,
        random_state=random_state,
    )
    projector.fit(scaled)
    scale = np.asarray(scaler.scale_, dtype=np.float32)
    scale[scale == 0] = 1.0
    return ProjectionState(
        input_center=np.asarray(scaler.mean_, dtype=np.float32),
        input_scale=scale,
        pca_center=np.asarray(projector.mean_, dtype=np.float32),
        components=np.asarray(projector.components_, dtype=np.float32),
        explained_variance_ratio=np.asarray(
            projector.explained_variance_ratio_, dtype=np.float32
        ),
    )


def _even_sample(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values)
    if len(values) <= count:
        return values
    indices = np.linspace(0, len(values) - 1, num=count, dtype=np.int64)
    return values[indices]


def _cache_key(
    selector_key: str,
    rows: pd.DataFrame,
    layers: Sequence[int],
    **settings: Any,
) -> str:
    columns = [
        column
        for column in ["trace_id", "timestep", "policy_call_index", "sample_index"]
        if column in rows
    ]
    row_hash = pd.util.hash_pandas_object(rows[columns], index=False).to_numpy(dtype=np.uint64)
    payload = {
        "schema": TOKEN_REPRESENTATION_CACHE_SCHEMA_VERSION,
        "selector": selector_key,
        "rows": hashlib.sha256(row_hash.tobytes()).hexdigest(),
        "layers": list(layers),
        **settings,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def _save_cache(path: Path, result: LayerTokenReadouts) -> None:
    path.mkdir(parents=True, exist_ok=True)
    result.rows.to_parquet(path / "rows.parquet", index=False)
    result.source_sites.to_parquet(path / "source_sites.parquet", index=False)
    result.token_metadata.to_parquet(path / "token_metadata.parquet", index=False)
    features = zarr.open_array(
        str(path / "readouts.zarr"),
        mode="w",
        shape=(2, *result.pooled.shape),
        chunks=(1, min(512, len(result.rows)), 1, result.pooled.shape[-1]),
        dtype="float32",
    )
    features[0] = result.pooled
    features[1] = result.tokenwise
    temporary = path / "projections.tmp.npz"
    np.savez_compressed(
        temporary,
        **_projection_arrays("channel", result.channel_projection),
        **_projection_arrays("pooled", result.pooled_projection),
        **_projection_arrays("tokenwise", result.tokenwise_projection),
    )
    os.replace(temporary, path / "projections.npz")
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": TOKEN_REPRESENTATION_CACHE_SCHEMA_VERSION,
                "cache_key": result.cache_key,
                "layers": list(result.layers),
                "token_count": result.token_count,
                "channel_dim": result.channel_dim,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _load_cache(path: Path, cache_key: str) -> LayerTokenReadouts | None:
    required = [
        path / "metadata.json",
        path / "rows.parquet",
        path / "source_sites.parquet",
        path / "token_metadata.parquet",
        path / "readouts.zarr",
        path / "projections.npz",
    ]
    if not all(value.exists() for value in required):
        return None
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("cache_key") != cache_key:
        return None
    readouts = np.asarray(zarr.open_array(str(path / "readouts.zarr"), mode="r"))
    with np.load(path / "projections.npz") as arrays:
        channel = _projection_from_arrays(arrays, "channel")
        pooled = _projection_from_arrays(arrays, "pooled")
        tokenwise = _projection_from_arrays(arrays, "tokenwise")
    return LayerTokenReadouts(
        rows=pd.read_parquet(path / "rows.parquet"),
        source_sites=pd.read_parquet(path / "source_sites.parquet"),
        token_metadata=pd.read_parquet(path / "token_metadata.parquet"),
        layers=tuple(int(value) for value in metadata["layers"]),
        pooled=readouts[0],
        tokenwise=readouts[1],
        channel_projection=channel,
        pooled_projection=pooled,
        tokenwise_projection=tokenwise,
        token_count=int(metadata["token_count"]),
        channel_dim=int(metadata["channel_dim"]),
        cache_key=cache_key,
    )


def _projection_arrays(prefix: str, state: ProjectionState) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_input_center": state.input_center,
        f"{prefix}_input_scale": state.input_scale,
        f"{prefix}_pca_center": state.pca_center,
        f"{prefix}_components": state.components,
        f"{prefix}_explained_variance_ratio": state.explained_variance_ratio,
    }


def _projection_from_arrays(arrays: Any, prefix: str) -> ProjectionState:
    return ProjectionState(
        input_center=np.asarray(arrays[f"{prefix}_input_center"]),
        input_scale=np.asarray(arrays[f"{prefix}_input_scale"]),
        pca_center=np.asarray(arrays[f"{prefix}_pca_center"]),
        components=np.asarray(arrays[f"{prefix}_components"]),
        explained_variance_ratio=np.asarray(
            arrays[f"{prefix}_explained_variance_ratio"]
        ),
    )


def _axes(value: Any) -> list[str]:
    if isinstance(value, str):
        return [str(item) for item in json.loads(value)]
    return [str(item) for item in value]


def _shape(value: Any) -> list[int]:
    if isinstance(value, str):
        return [int(item) for item in json.loads(value)]
    return [int(item) for item in value]
