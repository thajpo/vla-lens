"""Trace bundle and dataset primitives.

The generic VLA-lens substrate is intentionally trace-first.  A bundle is one
episode-aligned model trace on disk; a dataset is a queryable collection of
bundles.  Model/environment adapters can create bundles, but downstream probes,
attribution, dashboards, and intervention records should only need this layer.
"""

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc

from vla_lens.artifacts import LensArtifact, slugify

if TYPE_CHECKING:
    from vla_lens.selectors import ActivationQuery, FeatureView

SCHEMA_VERSION = "0.3.0"
ARRAY_STORAGE_FORMAT = "zarr"
ARRAY_COMPRESSION = "zstd"
FRAME_STORAGE_FORMAT = "jpeg"
FRAME_COMPRESSION = "jpeg"


@dataclass(frozen=True, slots=True)
class TraceManifest:
    """Rollout-level metadata for one trace bundle."""

    trace_id: str
    episode_id: str
    task_id: str
    prompt: str
    model_id: str
    env_id: str
    robot_id: str
    outcome: str
    length: int
    schema_version: str = SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TraceManifest":
        return cls(
            trace_id=str(payload["trace_id"]),
            episode_id=str(payload.get("episode_id", payload["trace_id"])),
            task_id=str(payload.get("task_id", "")),
            prompt=str(payload.get("prompt", "")),
            model_id=str(payload.get("model_id", "")),
            env_id=str(payload.get("env_id", "")),
            robot_id=str(payload.get("robot_id", "")),
            outcome=str(payload.get("outcome", "unknown")),
            length=int(payload.get("length", 0)),
            schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class ArraySpec:
    """Array payload plus axis metadata used when writing a bundle."""

    array: np.ndarray
    axes: Sequence[str]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelSiteSpec:
    """Model tensor payload plus the semantic site it came from."""

    name: str
    array: np.ndarray
    axes: Sequence[str]
    module: str
    layer: int | None = None
    tensor_type: str = "activation"
    token_kind: str | None = None
    generation_step: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    family: str | None = None
    role: str | None = None
    segment: str | None = None
    materialization: str = "raw"
    exactness: str = "exact"
    token_space_id: str | None = None
    query_token_space_id: str | None = None
    key_token_space_id: str | None = None
    parent_site_id: str | None = None
    summary_type: str | None = None
    capture_family: str | None = None
    view_kind: str | None = None
    capture_role: str | None = None
    default_view: bool | None = None
    derived_from: Sequence[str] | None = None
    derivation: str | None = None


ActivationSpec = ModelSiteSpec


class TraceBundle:
    """One ``.vlatrace`` folder containing an episode and aligned internals."""

    MANIFEST = "manifest.json"
    TABLES_DIR = "tables"
    TIMESTEPS = "tables/timesteps.parquet"
    POLICY_CALLS = "tables/policy_calls.parquet"
    GENERATION_STEPS = "tables/generation_steps.parquet"
    STREAMS = "tables/streams.parquet"
    TOKEN_SPACES = "tables/token_spaces.parquet"
    TOKENS = "tables/tokens.parquet"
    ROBOT_STATE = "tables/robot_state.parquet"
    SCENE_STATE = "tables/scene_state.parquet"
    CAMERA_STATE = "tables/camera_state.parquet"
    EVALUATION = "tables/evaluation.parquet"
    IMAGE_PREPROCESSING = "tables/image_preprocessing.parquet"
    PROMPT_METADATA = "tables/prompt_metadata.parquet"
    ACTION_NORMALIZATION = "tables/action_normalization.parquet"
    MODEL_SITES = "tables/model_sites.parquet"
    ARRAY_INDEX = "tables/array_index.parquet"
    ARTIFACT_INDEX = "tables/artifact_index.parquet"
    CAPTURE_REQUEST = "tables/capture_request.json"
    CAPTURE_PLAN = "tables/capture_plan.json"
    CAPTURE_REPORT = "tables/capture_report.json"
    FINGERPRINTS = "tables/fingerprints.json"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @classmethod
    def open(cls, path: str | Path) -> "TraceBundle":
        bundle = cls(path)
        bundle._require_file(bundle.path / cls.MANIFEST)
        return bundle

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        manifest: TraceManifest,
        timesteps: pd.DataFrame | None = None,
        episode_arrays: Mapping[str, ArraySpec] | None = None,
        model_arrays: Sequence[ModelSiteSpec] | None = None,
        policy_calls: pd.DataFrame | None = None,
        generation_steps: pd.DataFrame | None = None,
        streams: pd.DataFrame | None = None,
        token_spaces: pd.DataFrame | None = None,
        tokens: pd.DataFrame | None = None,
        robot_state: pd.DataFrame | None = None,
        scene_state: pd.DataFrame | None = None,
        camera_state: pd.DataFrame | None = None,
        evaluation: pd.DataFrame | None = None,
        image_preprocessing: pd.DataFrame | None = None,
        prompt_metadata: pd.DataFrame | None = None,
        action_normalization: pd.DataFrame | None = None,
        capture_request: Mapping[str, Any] | None = None,
        capture_plan: Mapping[str, Any] | None = None,
        capture_report: Mapping[str, Any] | None = None,
        artifacts: Sequence[LensArtifact] | None = None,
        overwrite: bool = False,
    ) -> "TraceBundle":
        path = Path(path)
        if path.exists() and overwrite:
            shutil.rmtree(path)
        if path.exists() and any(path.iterdir()):
            raise FileExistsError(f"Trace bundle already exists: {path}")

        path.mkdir(parents=True, exist_ok=True)
        (path / cls.TABLES_DIR).mkdir(parents=True, exist_ok=True)
        (path / "arrays" / "episode").mkdir(parents=True, exist_ok=True)
        (path / "arrays" / "action").mkdir(parents=True, exist_ok=True)
        (path / "arrays" / "context").mkdir(parents=True, exist_ok=True)
        (path / "arrays" / "model").mkdir(parents=True, exist_ok=True)
        (path / "arrays" / "derived_trace_local").mkdir(parents=True, exist_ok=True)
        (path / "artifacts").mkdir(parents=True, exist_ok=True)

        _write_json(path / cls.MANIFEST, manifest.to_dict())

        if timesteps is None:
            timesteps = pd.DataFrame({"timestep": np.arange(manifest.length, dtype=np.int32)})
        _write_table(path / cls.TIMESTEPS, timesteps)
        _write_table(path / cls.POLICY_CALLS, _table_or_empty(policy_calls))
        _write_table(path / cls.GENERATION_STEPS, _table_or_empty(generation_steps))
        _write_table(path / cls.STREAMS, _table_or_empty(streams))
        _write_table(path / cls.TOKEN_SPACES, _table_or_empty(token_spaces))
        _write_table(path / cls.TOKENS, _table_or_empty(tokens))
        _write_table(path / cls.ROBOT_STATE, _table_or_empty(robot_state))
        _write_table(path / cls.SCENE_STATE, _table_or_empty(scene_state))
        _write_table(path / cls.CAMERA_STATE, _table_or_empty(camera_state))
        _write_table(path / cls.EVALUATION, _table_or_empty(evaluation))
        _write_table(path / cls.IMAGE_PREPROCESSING, _table_or_empty(image_preprocessing))
        _write_table(path / cls.PROMPT_METADATA, _table_or_empty(prompt_metadata))
        _write_table(path / cls.ACTION_NORMALIZATION, _table_or_empty(action_normalization))
        capture_report_payload = dict(capture_report or {})
        _write_json(path / cls.CAPTURE_REQUEST, capture_request or {})
        _write_json(path / cls.CAPTURE_PLAN, capture_plan or {})
        _write_json(path / cls.CAPTURE_REPORT, capture_report_payload)

        array_records: list[dict[str, Any]] = []
        for name, spec in (episode_arrays or {}).items():
            if _is_frame_array(name):
                relative_path = Path("media") / "frames" / slugify(_frame_camera(name))
                chunks: tuple[int, ...] = ()
                storage_format = FRAME_STORAGE_FORMAT
                compression = FRAME_COMPRESSION
                _write_frame_sequence(path / relative_path, spec.array)
            else:
                relative_path = _episode_array_path(name)
                chunks = _write_zarr_array(path / relative_path, spec.array)
                storage_format = ARRAY_STORAGE_FORMAT
                compression = ARRAY_COMPRESSION
            array_records.append(
                _array_record(
                    name=name,
                    relative_path=relative_path,
                    array=spec.array,
                    axes=spec.axes,
                    array_type="episode",
                    metadata=spec.metadata,
                    chunks=chunks,
                    storage_format=storage_format,
                    compression=compression,
                )
            )
        _write_table(path / cls.ARRAY_INDEX, pd.DataFrame.from_records(array_records))

        model_records: list[dict[str, Any]] = []
        for spec in model_arrays or ():
            relative_path = Path("arrays") / "model" / f"{slugify(spec.name)}.zarr"
            chunks = _write_zarr_array(path / relative_path, spec.array)
            record = _array_record(
                name=spec.name,
                relative_path=relative_path,
                array=spec.array,
                axes=spec.axes,
                array_type="model",
                metadata=spec.metadata,
                chunks=chunks,
                storage_format=ARRAY_STORAGE_FORMAT,
                compression=ARRAY_COMPRESSION,
            )
            record.update(
                {
                    "module": spec.module,
                    "layer": spec.layer,
                    "tensor_type": spec.tensor_type,
                    "token_kind": spec.token_kind,
                    "generation_step": spec.generation_step,
                    "site_id": spec.name,
                    "array_id": spec.name,
                    "model_path": spec.module,
                    "family": spec.family,
                    "role": spec.role or spec.tensor_type,
                    "segment": spec.segment,
                    "layer_index": spec.layer,
                    "dtype_original": str(spec.array.dtype),
                    "dtype_saved": str(spec.array.dtype),
                    "materialization": spec.materialization,
                    "exactness": spec.exactness,
                    "token_space_id": spec.token_space_id,
                    "query_token_space_id": spec.query_token_space_id,
                    "key_token_space_id": spec.key_token_space_id,
                    "parent_site_id": spec.parent_site_id,
                    "summary_type": spec.summary_type,
                    "capture_family": spec.capture_family,
                    "view_kind": spec.view_kind,
                    "capture_role": spec.capture_role,
                    "default_view": spec.default_view,
                    "derived_from": json.dumps(list(spec.derived_from or ())),
                    "derivation": spec.derivation,
                }
            )
            model_records.append(record)
        _write_table(path / cls.MODEL_SITES, pd.DataFrame.from_records(model_records))

        artifact_records: list[dict[str, Any]] = []
        for artifact in artifacts or ():
            saved = cls(path).save_artifact(artifact, arrays=None)
            artifact_records.append(saved.to_record())
        if not artifact_records:
            _write_table(path / cls.ARTIFACT_INDEX, pd.DataFrame())

        fingerprints = _compute_trace_fingerprints(path, manifest=manifest)
        _write_json(path / cls.FINGERPRINTS, fingerprints)
        capture_report_payload["fingerprints"] = fingerprints
        _write_json(path / cls.CAPTURE_REPORT, capture_report_payload)
        manifest_payload = manifest.to_dict()
        manifest_metadata = dict(manifest_payload.get("metadata") or {})
        manifest_metadata["fingerprints"] = fingerprints
        manifest_payload["metadata"] = manifest_metadata
        _write_json(path / cls.MANIFEST, manifest_payload)

        return cls(path)

    @cached_property
    def manifest(self) -> TraceManifest:
        payload = json.loads((self.path / self.MANIFEST).read_text(encoding="utf-8"))
        return TraceManifest.from_dict(payload)

    @cached_property
    def timesteps(self) -> pd.DataFrame:
        return _read_table(self.path / self.TIMESTEPS)

    @cached_property
    def policy_calls(self) -> pd.DataFrame:
        return _read_table(self.path / self.POLICY_CALLS)

    @cached_property
    def generation_steps(self) -> pd.DataFrame:
        return _read_table(self.path / self.GENERATION_STEPS)

    @cached_property
    def streams(self) -> pd.DataFrame:
        return _read_table(self.path / self.STREAMS)

    @cached_property
    def token_spaces(self) -> pd.DataFrame:
        return _read_table(self.path / self.TOKEN_SPACES)

    @cached_property
    def tokens(self) -> pd.DataFrame:
        return _read_table(self.path / self.TOKENS)

    @cached_property
    def robot_state(self) -> pd.DataFrame:
        return _read_table(self.path / self.ROBOT_STATE)

    @cached_property
    def scene_state(self) -> pd.DataFrame:
        return _read_table(self.path / self.SCENE_STATE)

    @cached_property
    def camera_state(self) -> pd.DataFrame:
        return _read_table(self.path / self.CAMERA_STATE)

    @cached_property
    def evaluation(self) -> pd.DataFrame:
        return _read_table(self.path / self.EVALUATION)

    @cached_property
    def image_preprocessing(self) -> pd.DataFrame:
        return _read_table(self.path / self.IMAGE_PREPROCESSING)

    @cached_property
    def prompt_metadata(self) -> pd.DataFrame:
        return _read_table(self.path / self.PROMPT_METADATA)

    @cached_property
    def action_normalization(self) -> pd.DataFrame:
        return _read_table(self.path / self.ACTION_NORMALIZATION)

    @cached_property
    def capture_report(self) -> dict[str, Any]:
        path = self.path / self.CAPTURE_REPORT
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @cached_property
    def fingerprints(self) -> dict[str, Any]:
        path = self.path / self.FINGERPRINTS
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        value = self.manifest.metadata.get("fingerprints")
        return dict(value) if isinstance(value, Mapping) else {}

    @cached_property
    def array_index(self) -> pd.DataFrame:
        return _read_table(self.path / self.ARRAY_INDEX)

    @cached_property
    def model_sites(self) -> pd.DataFrame:
        return _read_table(self.path / self.MODEL_SITES)

    @cached_property
    def artifact_index(self) -> pd.DataFrame:
        return _read_table(self.path / self.ARTIFACT_INDEX)

    def episode_record(self) -> dict[str, Any]:
        record = self.manifest.to_dict()
        record["path"] = str(self.path)
        for key, value in self.manifest.metadata.items():
            if _is_scalar(value) and key not in record:
                record[key] = value
        return record

    def array(self, name: str, *, mmap: bool = False) -> np.ndarray:
        record = self._one_record(self.array_index, name, "episode array")
        return self._load_relative_array(str(record["relative_path"]), mmap=mmap)

    def model_site(self, name: str, *, mmap: bool = False) -> np.ndarray:
        record = self._one_record(self.model_sites, name, "model site")
        return self._load_relative_array(str(record["relative_path"]), mmap=mmap)

    def frames(self, camera: str, *, mmap: bool = False) -> np.ndarray:
        return self.array(f"frames.{camera}", mmap=mmap)

    def cameras(self) -> list[str]:
        if self.array_index.empty or "name" not in self.array_index:
            return []
        prefix = "frames."
        return sorted(
            str(name)[len(prefix) :]
            for name in self.array_index["name"]
            if str(name).startswith(prefix)
        )

    def actions(self, *, mmap: bool = False) -> np.ndarray:
        return self.array("executed_actions", mmap=mmap)

    def action_chunks(self, *, mmap: bool = False) -> np.ndarray:
        return self.array("action_chunks", mmap=mmap)

    def generation_actions(self, *, mmap: bool = False) -> np.ndarray:
        return self.array("generation_actions", mmap=mmap)

    def save_artifact(
        self,
        artifact: LensArtifact,
        arrays: Mapping[str, np.ndarray] | None = None,
    ) -> LensArtifact:
        """Persist an artifact and optionally its arrays inside this bundle."""
        artifact_dir = self.path / "artifacts" / artifact.artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        array_paths = dict(artifact.arrays)
        for name, array in (arrays or {}).items():
            relative_path = Path("artifacts") / artifact.artifact_id / f"{slugify(name)}.zarr"
            _write_zarr_array(self.path / relative_path, array)
            array_paths[name] = str(relative_path)

        saved = LensArtifact(
            artifact_id=artifact.artifact_id,
            artifact_type=artifact.artifact_type,
            name=artifact.name,
            group_id=artifact.group_id,
            scope=artifact.scope,
            selector=artifact.selector,
            method=artifact.method,
            metrics=artifact.metrics,
            arrays=array_paths,
            display=artifact.display,
            tags=artifact.tags,
            created_utc=artifact.created_utc,
            source_trace_ids=artifact.source_trace_ids or (self.manifest.trace_id,),
            path=str(Path("artifacts") / artifact.artifact_id / "artifact.json"),
        )
        (artifact_dir / "artifact.json").write_text(
            json.dumps(saved.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        existing = _read_table(self.path / self.ARTIFACT_INDEX)
        updated = pd.concat(
            [existing, pd.DataFrame.from_records([saved.to_record()])],
            ignore_index=True,
        )
        updated = updated.drop_duplicates(subset=["artifact_id"], keep="last")
        _write_table(self.path / self.ARTIFACT_INDEX, updated)
        self.__dict__.pop("artifact_index", None)
        return saved

    def load_artifact(self, artifact_id: str) -> LensArtifact:
        path = self.path / "artifacts" / artifact_id / "artifact.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return LensArtifact.from_dict(payload)

    def load_artifact_array(
        self,
        artifact: LensArtifact,
        name: str,
        *,
        mmap: bool = False,
    ) -> np.ndarray:
        if name not in artifact.arrays:
            raise KeyError(f"Artifact '{artifact.name}' has no array named '{name}'")
        return self._load_relative_array(artifact.arrays[name], mmap=mmap)

    def _load_relative_array(self, relative_path: str, *, mmap: bool = False) -> np.ndarray:
        del mmap
        path = self.path / relative_path
        if path.is_dir() and path.suffix != ".zarr":
            return _read_frame_sequence(path)
        return _read_zarr_array(path)

    def _one_record(self, table: pd.DataFrame, name: str, kind: str) -> pd.Series:
        if table.empty or "name" not in table:
            raise KeyError(f"No {kind} records in {self.path}")
        matches = table.loc[table["name"].astype(str) == name]
        if matches.empty:
            raise KeyError(f"Unknown {kind} '{name}' in {self.path}")
        return matches.iloc[0]

    @staticmethod
    def _require_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)


class TraceDataset:
    """Queryable collection of trace bundles."""

    def __init__(self, root: str | Path, bundles: Sequence[TraceBundle]):
        self.root = Path(root)
        self.bundles = list(bundles)
        self._bundle_by_trace_id = _bundle_index_by_trace_id(self.bundles)

    @classmethod
    def open(cls, root: str | Path) -> "TraceDataset":
        root = Path(root)
        from vla_lens.lerobot_dataset import is_lerobot_dataset_root, open_lerobot_dataset

        if is_lerobot_dataset_root(root):
            return open_lerobot_dataset(root)
        if (root / TraceBundle.MANIFEST).exists():
            return cls(root, [TraceBundle.open(root)])

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

        bundle_paths = sorted(
            path
            for path in root.rglob("*.vlatrace")
            if path.is_dir()
            and path.suffix == ".vlatrace"
            and (path / TraceBundle.MANIFEST).exists()
        )
        if not bundle_paths:
            raise FileNotFoundError(
                f"No LeRobot v3 dataset roots or .vlatrace bundles found in {root}"
            )
        return cls(root, [TraceBundle.open(path) for path in bundle_paths])

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
            dataset_table["dataset_path"] = str(self.root)
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
        if (self.root / TraceBundle.MANIFEST).exists():
            return pd.DataFrame()
        return _read_table(self.root / TraceBundle.ARTIFACT_INDEX)

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

        A single-bundle dataset stores artifacts inside that bundle. Multi-bundle
        datasets store cross-episode artifacts at the dataset root so probe
        suites and reports do not appear to belong to an arbitrary episode.
        """
        if (self.root / TraceBundle.MANIFEST).exists() and len(self.bundles) == 1:
            saved = self.bundles[0].save_artifact(artifact, arrays=arrays)
            self.__dict__.pop("artifact_index", None)
            return saved

        artifact_dir = self.root / "artifacts" / artifact.artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        array_paths = dict(artifact.arrays)
        for name, array in (arrays or {}).items():
            relative_path = Path("artifacts") / artifact.artifact_id / f"{slugify(name)}.zarr"
            _write_zarr_array(self.root / relative_path, array)
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
        (artifact_dir / "artifact.json").write_text(
            json.dumps(saved.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        existing = _read_table(self.root / TraceBundle.ARTIFACT_INDEX)
        updated = pd.concat(
            [existing, pd.DataFrame.from_records([saved.to_record()])],
            ignore_index=True,
        )
        updated = updated.drop_duplicates(subset=["artifact_id"], keep="last")
        _write_table(self.root / TraceBundle.ARTIFACT_INDEX, updated)
        self.__dict__.pop("dataset_artifact_index", None)
        self.__dict__.pop("artifact_index", None)
        return saved

    def load_artifact(self, artifact_id: str) -> LensArtifact:
        dataset_path = self.root / "artifacts" / artifact_id / "artifact.json"
        if dataset_path.exists() and not (self.root / TraceBundle.MANIFEST).exists():
            payload = json.loads(dataset_path.read_text(encoding="utf-8"))
            return LensArtifact.from_dict(payload)
        for bundle in self.bundles:
            path = bundle.path / "artifacts" / artifact_id / "artifact.json"
            if path.exists():
                return bundle.load_artifact(artifact_id)
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
        dataset_path = self.root / relative_path
        if artifact.scope == "dataset" and dataset_path.exists():
            return _read_zarr_array(dataset_path)
        for bundle in self.bundles:
            path = bundle.path / relative_path
            if path.exists():
                return bundle.load_artifact_array(artifact, name, mmap=mmap)
        raise FileNotFoundError(relative_path)

    def cache_dir(self) -> Path:
        path = self.root / ".vla_cache"
        path.mkdir(parents=True, exist_ok=True)
        return path


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


def _compute_trace_fingerprints(path: Path, *, manifest: TraceManifest) -> dict[str, Any]:
    array_index = _read_table(path / TraceBundle.ARRAY_INDEX)
    trajectory_payload = {
        "tables": {
            "timesteps": _table_fingerprint_payload(_read_table(path / TraceBundle.TIMESTEPS)),
            "policy_calls": _table_fingerprint_payload(
                _read_table(path / TraceBundle.POLICY_CALLS)
            ),
        },
        "arrays": _fingerprint_arrays(
            path,
            array_index,
            names={
                "executed_actions",
                "action_chunks",
                "generation_actions",
                "generation_velocities",
            },
        ),
    }
    context_payload = {
        "manifest_context": {
            "trace_id": manifest.trace_id,
            "episode_id": manifest.episode_id,
            "task_id": manifest.task_id,
            "prompt": manifest.prompt,
            "model_id": manifest.model_id,
            "env_id": manifest.env_id,
            "robot_id": manifest.robot_id,
            "outcome": manifest.outcome,
            "length": manifest.length,
        },
        "tables": {
            "robot_state": _table_fingerprint_payload(_read_table(path / TraceBundle.ROBOT_STATE)),
            "scene_state": _table_fingerprint_payload(_read_table(path / TraceBundle.SCENE_STATE)),
            "camera_state": _table_fingerprint_payload(
                _read_table(path / TraceBundle.CAMERA_STATE)
            ),
            "evaluation": _table_fingerprint_payload(_read_table(path / TraceBundle.EVALUATION)),
            "image_preprocessing": _table_fingerprint_payload(
                _read_table(path / TraceBundle.IMAGE_PREPROCESSING)
            ),
            "prompt_metadata": _table_fingerprint_payload(
                _read_table(path / TraceBundle.PROMPT_METADATA)
            ),
            "action_normalization": _table_fingerprint_payload(
                _read_table(path / TraceBundle.ACTION_NORMALIZATION)
            ),
        },
        "arrays": _fingerprint_arrays(
            path,
            array_index,
            prefixes=("robot_", "scene_", "camera_", "evaluation_"),
        ),
    }
    trace_schema_payload = {
        "manifest": _without_fingerprint_fields(manifest.to_dict()),
        "capture_request": _without_fingerprint_fields(
            _read_json(path / TraceBundle.CAPTURE_REQUEST)
        ),
        "capture_plan": _without_fingerprint_fields(_read_json(path / TraceBundle.CAPTURE_PLAN)),
        "capture_report": _without_fingerprint_fields(
            _read_json(path / TraceBundle.CAPTURE_REPORT)
        ),
        "tables": {
            "array_index": _table_fingerprint_payload(array_index),
            "model_sites": _table_fingerprint_payload(_read_table(path / TraceBundle.MODEL_SITES)),
            "streams": _table_fingerprint_payload(_read_table(path / TraceBundle.STREAMS)),
            "token_spaces": _table_fingerprint_payload(
                _read_table(path / TraceBundle.TOKEN_SPACES)
            ),
            "tokens": _table_fingerprint_payload(_read_table(path / TraceBundle.TOKENS)),
            "policy_calls": _table_fingerprint_payload(
                _read_table(path / TraceBundle.POLICY_CALLS)
            ),
            "generation_steps": _table_fingerprint_payload(
                _read_table(path / TraceBundle.GENERATION_STEPS)
            ),
        },
    }

    trajectory_fingerprint = _hash_json_payload(trajectory_payload)
    context_fingerprint = _hash_json_payload(context_payload)
    trace_schema_fingerprint = _hash_json_payload(trace_schema_payload)
    component_payload = {
        "trajectory_fingerprint": trajectory_fingerprint,
        "context_fingerprint": context_fingerprint,
        "trace_schema_fingerprint": trace_schema_fingerprint,
    }
    return {
        "fingerprint_schema_version": 1,
        "algorithm": "sha256",
        **component_payload,
        "trace_fingerprint": _hash_json_payload(component_payload),
        "components": {
            "trajectory": _fingerprint_component_summary(trajectory_payload),
            "context": _fingerprint_component_summary(context_payload),
            "trace_schema": _fingerprint_component_summary(trace_schema_payload),
        },
    }


def _fingerprint_component_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    tables = payload.get("tables")
    if isinstance(tables, Mapping):
        summary["tables"] = {
            str(name): {
                "rows": value.get("rows"),
                "columns": value.get("columns"),
                "fingerprint": value.get("fingerprint"),
            }
            for name, value in tables.items()
            if isinstance(value, Mapping)
        }
    arrays = payload.get("arrays")
    if isinstance(arrays, Mapping):
        summary["arrays"] = {
            str(name): {
                "shape": value.get("shape"),
                "dtype": value.get("dtype"),
                "fingerprint": value.get("fingerprint"),
            }
            for name, value in arrays.items()
            if isinstance(value, Mapping)
        }
    return summary


def _table_fingerprint_payload(frame: pd.DataFrame) -> dict[str, Any]:
    columns = sorted(str(column) for column in frame.columns)
    if frame.empty:
        records: list[dict[str, Any]] = []
    else:
        records = [
            {str(column): _jsonable_cell(row[column]) for column in columns}
            for _, row in frame.loc[:, columns].iterrows()
        ]
    payload = {"columns": columns, "rows": int(len(frame)), "records": records}
    return {
        "columns": columns,
        "rows": int(len(frame)),
        "fingerprint": _hash_json_payload(payload),
    }


def _fingerprint_arrays(
    bundle_path: Path,
    array_index: pd.DataFrame,
    *,
    names: set[str] | None = None,
    prefixes: tuple[str, ...] = (),
) -> dict[str, dict[str, Any]]:
    if array_index.empty or "name" not in array_index:
        return {}
    records: dict[str, dict[str, Any]] = {}
    for row in array_index.to_dict("records"):
        name = str(row.get("name") or "")
        if names is not None and name not in names:
            continue
        if prefixes and not name.startswith(prefixes):
            continue
        relative_path = Path(str(row.get("relative_path") or ""))
        if relative_path.suffix != ".zarr":
            continue
        records[name] = _array_fingerprint_payload(bundle_path / relative_path)
    return records


def _array_fingerprint_payload(path: Path) -> dict[str, Any]:
    array = zarr.open_array(str(path), mode="r")
    value = np.asarray(array[:])
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(value)
    digest.update(json.dumps([int(item) for item in contiguous.shape]).encode("utf-8"))
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(contiguous.view(np.uint8))
    return {
        "shape": [int(item) for item in contiguous.shape],
        "dtype": str(contiguous.dtype),
        "fingerprint": f"sha256:{digest.hexdigest()}",
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _without_fingerprint_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_fingerprint_fields(item)
            for key, item in value.items()
            if str(key) != "fingerprints"
        }
    if isinstance(value, list):
        return [_without_fingerprint_fields(item) for item in value]
    return value


def _hash_json_payload(payload: Any) -> str:
    encoded = json.dumps(_jsonable_cell(payload), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _jsonable_cell(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable_cell(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_cell(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable_cell(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable_cell(value.item())
    if pd.isna(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "ActivationSpec",
    "ArraySpec",
    "ModelSiteSpec",
    "TraceBundle",
    "TraceDataset",
    "TraceDatasetStats",
    "TraceManifest",
]
