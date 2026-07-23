"""Trace dataset object and dataset-level artifact storage."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.artifacts import LensArtifact, slugify
from vla_lens.cache import InterProcessFileLock, atomic_replace_file
from vla_lens.traces.bundle import TraceBundle
from vla_lens.traces.io import (
    _bundle_index_by_trace_id,
    _concat_or_empty,
    _filter_frame,
    _nested_lerobot_dataset_roots,
    _nested_lerobot_trace_id_prefix,
    _read_table,
    _read_zarr_array,
    _validate_artifact_id,
    _write_table,
    _write_zarr_array,
)

if TYPE_CHECKING:
    from vla_lens.selectors import ActivationQuery, FeatureView


class TraceDataset:
    """Queryable collection of LeRobot episodes and optional VLA overlays."""

    def __init__(self, root: str | Path, bundles: Sequence[TraceBundle]):
        self.root = Path(root)
        self.bundles = list(bundles)
        self._bundle_by_trace_id = _bundle_index_by_trace_id(self.bundles)

    @classmethod
    def open(cls, root: str | Path) -> "TraceDataset":
        """Open a LeRobot root or nested batch output."""
        root = Path(root)
        from vla_lens.dataset import is_lerobot_dataset_root, open_lerobot_dataset

        if is_lerobot_dataset_root(root):
            return open_lerobot_dataset(root)

        lerobot_roots = _nested_lerobot_dataset_roots(root, is_lerobot_dataset_root)
        if lerobot_roots:
            bundles: list[Any] = []
            for dataset_root in lerobot_roots:
                bundles.extend(
                    open_lerobot_dataset(
                        dataset_root,
                        trace_id_prefix=_nested_lerobot_trace_id_prefix(root, dataset_root),
                    ).bundles
                )
            return cls(root, bundles)

        raise FileNotFoundError(f"No LeRobot v3 dataset root found in {root}")

    @cached_property
    def episode_index(self) -> pd.DataFrame:
        return pd.DataFrame.from_records([bundle.episode_record() for bundle in self.bundles])

    @cached_property
    def timestep_index(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for bundle in self.bundles:
            table = bundle.timesteps.copy()
            table["trace_id"] = bundle.manifest.trace_id
            table["episode_id"] = bundle.manifest.episode_id
            table["bundle_path"] = str(bundle.path)
            frames.append(table)
        return _concat_or_empty(frames)

    @cached_property
    def model_site_index(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for bundle in self.bundles:
            table = bundle.model_sites.copy()
            if table.empty:
                continue
            table["trace_id"] = bundle.manifest.trace_id
            table["episode_id"] = bundle.manifest.episode_id
            table["bundle_path"] = str(bundle.path)
            frames.append(table)
        return _concat_or_empty(frames)

    @cached_property
    def artifact_index(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        dataset_table = self.dataset_artifact_index.copy()
        if not dataset_table.empty:
            dataset_table["trace_id"] = None
            dataset_table["episode_id"] = None
            dataset_table["bundle_path"] = None
            dataset_table["dataset_path"] = str(self._dataset_artifact_root())
            dataset_table["artifact_scope"] = "dataset"
            frames.append(dataset_table)
        for bundle in self.bundles:
            table = bundle.artifact_index.copy()
            if table.empty:
                continue
            table["trace_id"] = bundle.manifest.trace_id
            table["episode_id"] = bundle.manifest.episode_id
            table["bundle_path"] = str(bundle.path)
            table["dataset_path"] = str(self.root)
            table["artifact_scope"] = "bundle"
            frames.append(table)
        return _concat_or_empty(frames)

    @cached_property
    def dataset_artifact_index(self) -> pd.DataFrame:
        return _read_table(self._dataset_artifact_root() / TraceBundle.ARTIFACT_INDEX)

    @property
    def stats(self) -> "TraceDatasetStats":
        return TraceDatasetStats(self)

    def bundle(self, trace_id: str) -> TraceBundle:
        return self._bundle_by_trace_id[trace_id]

    def episodes(self, **filters: Any) -> pd.DataFrame:
        return _filter_frame(self.episode_index, filters)

    def select_model_sites(self, selector: "ActivationQuery") -> "FeatureView":
        from vla_lens.selectors import FeatureView

        return FeatureView(self, selector)

    def save_artifact(
        self,
        artifact: LensArtifact,
        arrays: Mapping[str, np.ndarray] | None = None,
    ) -> LensArtifact:
        """Persist a dataset-level artifact and optionally its arrays.

        Cross-episode artifacts are stored at the dataset artifact root so probe
        suites and reports do not appear to belong to an arbitrary episode.
        """
        _validate_artifact_id(artifact.artifact_id)
        artifact_root = self._dataset_artifact_root()
        lock_path = artifact_root / ".locks" / "dataset-artifacts.lock"
        with InterProcessFileLock(lock_path):
            saved = self._save_artifact_locked(artifact, arrays=arrays)
        self.__dict__.pop("dataset_artifact_index", None)
        self.__dict__.pop("artifact_index", None)
        return saved

    def _save_artifact_locked(
        self,
        artifact: LensArtifact,
        *,
        arrays: Mapping[str, np.ndarray] | None,
    ) -> LensArtifact:
        artifact_root = self._dataset_artifact_root()
        artifact_dir = artifact_root / "artifacts" / artifact.artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        array_paths = dict(artifact.arrays)
        for name, array in (arrays or {}).items():
            relative_path = Path("artifacts") / artifact.artifact_id / f"{slugify(name)}.zarr"
            _write_zarr_array(artifact_root / relative_path, array)
            array_paths[name] = str(relative_path)

        saved = LensArtifact(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            name=artifact.name,
            group_id=artifact.group_id,
            scope="dataset",
            selector=artifact.selector,
            method=artifact.method,
            metrics=artifact.metrics,
            arrays=array_paths,
            display=artifact.display,
            tags=artifact.tags,
            created_utc=artifact.created_utc,
            source_trace_ids=artifact.source_trace_ids
            or tuple(bundle.manifest.trace_id for bundle in self.bundles),
            path=str(Path("artifacts") / artifact.artifact_id / "artifact.json"),
        )
        atomic_replace_file(
            artifact_dir / "artifact.json",
            lambda temporary: temporary.write_text(
                json.dumps(saved.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
            ),
        )

        existing = _read_table(artifact_root / TraceBundle.ARTIFACT_INDEX)
        updated = pd.concat(
            [existing, pd.DataFrame.from_records([saved.to_record()])],
            ignore_index=True,
        )
        updated = updated.drop_duplicates(subset=["artifact_id"], keep="last")
        atomic_replace_file(
            artifact_root / TraceBundle.ARTIFACT_INDEX,
            lambda temporary: _write_table(temporary, updated),
        )
        return saved

    def load_artifact(self, artifact_id: str) -> LensArtifact:
        _validate_artifact_id(artifact_id)
        artifact_root = self._dataset_artifact_root()
        dataset_path = artifact_root / "artifacts" / artifact_id / "artifact.json"
        if dataset_path.exists():
            payload = json.loads(dataset_path.read_text(encoding="utf-8"))
            return LensArtifact.from_dict(payload)
        for bundle in self.bundles:
            try:
                return bundle.load_artifact(artifact_id)
            except (FileNotFoundError, KeyError):
                continue
        raise KeyError(f"Unknown artifact '{artifact_id}'")

    def load_artifact_array(
        self,
        artifact: LensArtifact,
        name: str,
        *,
        mmap: bool = False,
    ) -> np.ndarray:
        if name not in artifact.arrays:
            raise KeyError(f"Artifact '{artifact.name}' has no array named '{name}'")
        relative_path = artifact.arrays[name]
        dataset_path = self._dataset_artifact_root() / relative_path
        if artifact.scope == "dataset" and dataset_path.exists():
            return _read_zarr_array(dataset_path)
        for bundle in self.bundles:
            try:
                return bundle.load_artifact_array(artifact, name, mmap=mmap)
            except (FileNotFoundError, KeyError):
                continue
        raise FileNotFoundError(relative_path)

    def cache_dir(self) -> Path:
        path = self.root / ".vla_cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _dataset_artifact_root(self) -> Path:
        if (self.root / "meta" / "info.json").exists() and (self.root / "data").exists():
            return self.root / "vla_lens"
        return self.root


class TraceDatasetStats:
    """Cheap dataset summaries used before choosing probes or dashboard filters."""

    def __init__(self, dataset: TraceDataset):
        self.dataset = dataset

    def by_task(self) -> pd.DataFrame:
        index = self.dataset.episode_index
        if index.empty:
            return pd.DataFrame()
        columns = [col for col in ["task_id", "outcome"] if col in index]
        return index.groupby(columns, dropna=False).size().reset_index(name="episodes")

    def activation_coverage(self) -> pd.DataFrame:
        index = self.dataset.model_site_index
        if index.empty:
            return pd.DataFrame()
        columns = [
            col
            for col in ["module", "layer", "tensor_type", "token_kind", "generation_step"]
            if col in index
        ]
        return index.groupby(columns, dropna=False).size().reset_index(name="traces")

    def action_ranges(self) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for bundle in self.dataset.bundles:
            try:
                actions = np.asarray(bundle.actions(mmap=True), dtype=np.float32)
            except KeyError:
                continue
            if actions.size == 0:
                continue
            flat = actions.reshape(-1, actions.shape[-1])
            for dim in range(flat.shape[-1]):
                values = flat[:, dim]
                records.append(
                    {
                        "trace_id": bundle.manifest.trace_id,
                        "action_dim": dim,
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                    }
                )
        return pd.DataFrame.from_records(records)
