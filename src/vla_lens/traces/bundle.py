"""Trace bundle object and bundle-level artifact storage."""

from __future__ import annotations

import json
import shutil
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.artifacts import LensArtifact, slugify
from vla_lens.traces.fingerprints import _compute_trace_fingerprints
from vla_lens.traces.io import (
    _array_record,
    _episode_array_path,
    _frame_camera,
    _is_frame_array,
    _is_scalar,
    _read_frame_sequence,
    _read_table,
    _read_zarr_array,
    _table_or_empty,
    _validate_artifact_id,
    _write_frame_sequence,
    _write_json,
    _write_table,
    _write_zarr_array,
)
from vla_lens.traces.layout import (
    ARRAY_COMPRESSION,
    ARRAY_STORAGE_FORMAT,
    FRAME_COMPRESSION,
    FRAME_STORAGE_FORMAT,
)
from vla_lens.traces.types import ArraySpec, ModelSiteSpec, TraceManifest


class TraceBundle:
    """Episode overlay bundle used under ``vla_lens/episodes/...``."""

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
        """Open an existing overlay bundle directory."""
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
        """Create one overlay bundle with tables, arrays, and artifacts.

        This low-level writer is used for VLA Lens overlay episodes.
        Dataset-scale robot data should be written through the LeRobot writer,
        not by treating this bundle as the primary dataset root.
        """
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
        _validate_artifact_id(artifact.artifact_id)
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
        _validate_artifact_id(artifact_id)
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
