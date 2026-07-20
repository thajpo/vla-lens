"""Versioned, replayable probe-run artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.artifacts import LensArtifact
from vla_lens.probes.workflow_artifacts import (
    _array_fingerprint,
    _bundle_fingerprint,
    _hash_json,
)
from vla_lens.probes.workflow_prepare import latest_loadable_artifact
from vla_lens.probes.workflow_types import (
    INTERACTION_METRICS_ARTIFACT_TYPE,
    OBJECT_FLOW_ARTIFACT_TYPE,
    POLICY_CALL_LABELS_ARTIFACT_TYPE,
)
from vla_lens.selectors import ActivationQuery
from vla_lens.traces import TraceDataset

PROBE_RUN_CONTRACT_KEY = "probe_run_contract"
PROBE_RUN_CONTRACT_SCHEMA_VERSION = 2
SOURCE_FEATURE_ROW_INDEX = "source_feature_row_index"


class ProbeArtifactError(ValueError):
    """Base error for invalid or incompatible probe artifacts."""


class NonReplayableProbeError(ProbeArtifactError):
    """Raised when a legacy or incomplete artifact cannot replay predictions."""


@dataclass(frozen=True, slots=True)
class ProbeReplayResult:
    artifact_id: str
    matched: bool
    row_count: int
    mismatch_count: int
    max_absolute_difference: float | None
    absolute_tolerance: float | None
    relative_tolerance: float | None
    feature_matrix_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "matched": self.matched,
            "row_count": self.row_count,
            "mismatch_count": self.mismatch_count,
            "max_absolute_difference": self.max_absolute_difference,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "feature_matrix_fingerprint": self.feature_matrix_fingerprint,
        }


@dataclass(slots=True)
class LoadedProbeArtifact:
    """A saved probe that can explain itself and, for current artifacts, run again."""

    dataset: TraceDataset
    artifact: LensArtifact

    @property
    def contract(self) -> dict[str, Any] | None:
        value = self.artifact.method.get(PROBE_RUN_CONTRACT_KEY)
        return dict(value) if isinstance(value, Mapping) else None

    @property
    def capabilities(self) -> dict[str, Any]:
        contract = self.contract
        if contract is None:
            return {
                "explain": True,
                "replay": False,
                "use": False,
                "legacy": True,
                "reason": "Artifact predates the replayable probe-run contract.",
            }
        try:
            schema_version = int(contract.get("schema_version", -1))
        except (TypeError, ValueError):
            schema_version = -1
        if schema_version != PROBE_RUN_CONTRACT_SCHEMA_VERSION:
            reason = (
                "Probe-run schema version 1 predates fitted-model integrity checks."
                if schema_version == 1
                else f"Unsupported probe-run schema version {contract.get('schema_version')!r}."
            )
            return {
                "explain": True,
                "replay": False,
                "use": False,
                "legacy": schema_version < PROBE_RUN_CONTRACT_SCHEMA_VERSION,
                "reason": reason,
            }
        return dict(contract.get("capabilities") or {})

    def explain(self) -> dict[str, Any]:
        if self.contract is not None:
            return {
                "artifact_id": self.artifact.artifact_id,
                "artifact_type": self.artifact.artifact_type,
                "replayable": bool(self.capabilities.get("replay")),
                "experiment_card": dict(self.contract.get("experiment_card") or {}),
                "uncertainty": dict(self.contract.get("uncertainty") or {}),
                "capabilities": self.capabilities,
            }
        return {
            "artifact_id": self.artifact.artifact_id,
            "artifact_type": self.artifact.artifact_type,
            "replayable": False,
            "experiment_card": {
                "name": self.artifact.name,
                "question": dict(self.artifact.method.get("research") or {}).get("question"),
                "stage": "legacy",
            },
            "uncertainty": {"status": "unknown"},
            "capabilities": self.capabilities,
        }

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Apply the fitted probe to a compatible feature matrix."""

        contract = self._require_replayable()
        model = dict(contract["model"])
        arrays = self._validated_model_arrays(model)
        return self._predict(features, model=model, arrays=arrays)

    def _predict(
        self,
        features: np.ndarray,
        *,
        model: Mapping[str, Any],
        arrays: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        array_names = dict(model["array_names"])
        feature_mean = arrays[array_names["feature_mean"]]
        feature_scale = arrays[array_names["feature_scale"]]
        X = np.asarray(features)
        if X.ndim != 2:
            raise ProbeArtifactError(f"Probe features must be 2D, got shape {X.shape}")
        expected_dim = int(model["feature_dim"])
        if X.shape[1] != expected_dim:
            raise ProbeArtifactError(
                f"Probe expects {expected_dim} features, received {X.shape[1]}"
            )
        # StandardScaler transforms floating inputs in place when possible, so a
        # float32 capture stays float32 even though the fitted statistics are
        # float64. Match that behavior to reproduce sklearn predictions exactly.
        if np.issubdtype(X.dtype, np.complexfloating):
            raise ProbeArtifactError("Probe features must be real-valued numeric data")
        if np.issubdtype(X.dtype, np.floating):
            normalized = X.copy()
        else:
            normalization_dtype = np.result_type(feature_mean.dtype, feature_scale.dtype)
            try:
                normalized = X.astype(normalization_dtype, copy=True)
            except (TypeError, ValueError) as error:
                raise ProbeArtifactError(
                    "Probe features must be real-valued numeric data"
                ) from error
        if not np.isfinite(normalized).all():
            raise ProbeArtifactError("Probe features must contain only finite values")
        normalized -= feature_mean
        normalized /= feature_scale
        model_format = str(model["format"])
        if model_format == "standardized_linear_v1":
            weights = np.asarray(arrays[array_names["weights"]])
            bias = np.asarray(arrays[array_names["bias"]]).reshape(-1)
            if weights.ndim == 1:
                scores = normalized @ weights + bias[0]
            else:
                scores = normalized @ weights.T + bias
        elif model_format == "standardized_mlp_v1":
            scores = normalized
            weight_names = list(array_names.get("layer_weights") or [])
            bias_names = list(array_names.get("layer_biases") or [])
            if not weight_names or len(weight_names) != len(bias_names):
                raise ProbeArtifactError("MLP probe is missing fitted layer arrays")
            for index, (weight_name, bias_name) in enumerate(
                zip(weight_names, bias_names, strict=True)
            ):
                scores = scores @ arrays[weight_name] + arrays[bias_name]
                if index < len(weight_names) - 1:
                    scores = _activation(scores, str(model.get("activation") or "relu"))
            scores = _output_activation(
                np.asarray(scores), str(model.get("out_activation") or "identity")
            )
        else:
            raise ProbeArtifactError(f"Unsupported fitted probe format {model_format!r}")
        if str(model["probe_type"]) == "regression":
            if scores.ndim == 1:
                return scores
            return scores[:, 0] if scores.shape[1] == 1 else scores
        classes = _prediction_classes(model.get("classes") or [])
        if len(classes) < 2:
            raise ProbeArtifactError("Classification probe is missing its fitted classes")
        if scores.ndim == 1 or (scores.ndim == 2 and scores.shape[1] == 1):
            threshold = 0.5 if model_format == "standardized_mlp_v1" else 0.0
            indices = (scores.reshape(-1) > threshold).astype(np.int64)
        else:
            indices = np.argmax(scores, axis=1)
        return classes[indices]

    def replay(self) -> ProbeReplayResult:
        """Rebuild saved features from the capture and reproduce saved predictions."""

        contract = self._require_replayable()
        model = dict(contract["model"])
        arrays = self._validated_model_arrays(model)
        source = dict(contract["source"])
        _validate_trace_fingerprints(self.dataset, source)
        try:
            selector = ActivationQuery.from_dict(dict(source["selector"]))
        except (TypeError, ValueError) as error:
            raise ProbeArtifactError(
                "Probe artifact contains an invalid activation selector"
            ) from error
        matrix = self.dataset.select_model_sites(selector).materialize(cache=False)
        saved_sites = _read_replay_table(
            self.dataset,
            source["source_sites_path"],
            "source-site rows",
        )
        saved_sites_fingerprint = dataframe_fingerprint(saved_sites)
        expected_sites_fingerprint = str(source["source_sites_fingerprint"])
        if saved_sites_fingerprint != expected_sites_fingerprint:
            raise ProbeArtifactError(
                "Saved source-site rows changed after training: "
                f"expected {expected_sites_fingerprint}, got {saved_sites_fingerprint}"
            )
        actual_sites_fingerprint = dataframe_fingerprint(matrix.rows)
        if actual_sites_fingerprint != expected_sites_fingerprint:
            raise ProbeArtifactError(
                "Selected model-site rows changed since this probe was trained: "
                f"expected {expected_sites_fingerprint}, got {actual_sites_fingerprint}"
            )
        rows = _read_replay_table(
            self.dataset,
            source["source_rows_path"],
            "source rows",
        )
        actual_rows_fingerprint = dataframe_fingerprint(rows)
        expected_rows_fingerprint = str(source["source_rows_fingerprint"])
        if actual_rows_fingerprint != expected_rows_fingerprint:
            raise ProbeArtifactError(
                "Saved probe rows changed after training: "
                f"expected {expected_rows_fingerprint}, got {actual_rows_fingerprint}"
            )
        if SOURCE_FEATURE_ROW_INDEX not in rows:
            raise ProbeArtifactError(
                f"Saved source rows do not contain {SOURCE_FEATURE_ROW_INDEX!r}"
            )
        indices = rows[SOURCE_FEATURE_ROW_INDEX].to_numpy(dtype=np.int64)
        if len(indices) and (indices.min() < 0 or indices.max() >= len(matrix.X)):
            raise ProbeArtifactError("Saved source-row indices exceed the selected feature matrix")
        features = np.asarray(matrix.X[indices])
        feature_fingerprint = _array_fingerprint(features)
        expected_feature_fingerprint = str(source["feature_matrix_fingerprint"])
        if feature_fingerprint != expected_feature_fingerprint:
            raise ProbeArtifactError(
                "Prepared probe features changed since training: "
                f"expected {expected_feature_fingerprint}, got {feature_fingerprint}"
            )
        predictions = self._predict(features, model=model, arrays=arrays)
        saved = _read_replay_table(
            self.dataset,
            source["scored_predictions_path"],
            "scored predictions",
        )
        saved_predictions_fingerprint = dataframe_fingerprint(saved)
        expected_predictions_fingerprint = str(source["scored_predictions_fingerprint"])
        if saved_predictions_fingerprint != expected_predictions_fingerprint:
            raise ProbeArtifactError(
                "Saved scored predictions changed after training: "
                f"expected {expected_predictions_fingerprint}, "
                f"got {saved_predictions_fingerprint}"
            )
        if len(saved) != len(predictions):
            raise ProbeArtifactError(
                f"Saved predictions have {len(saved)} rows, replay produced {len(predictions)}"
            )
        return _compare_predictions(
            artifact_id=self.artifact.artifact_id,
            probe_type=str(model["probe_type"]),
            replayed=predictions,
            saved=saved["prediction_value"].to_numpy(),
            feature_fingerprint=feature_fingerprint,
            prediction_tolerance=model.get("prediction_tolerance"),
        )

    def _array(self, name: str) -> np.ndarray:
        if name not in self.artifact.arrays:
            raise ProbeArtifactError(f"Probe artifact is missing fitted array {name!r}")
        return np.asarray(self.dataset.load_artifact_array(self.artifact, name))

    def _validated_model_arrays(self, model: Mapping[str, Any]) -> dict[str, np.ndarray]:
        fingerprints = dict(model["array_fingerprints"])
        arrays: dict[str, np.ndarray] = {}
        for name, expected in fingerprints.items():
            value = self._array(str(name))
            actual = _array_fingerprint(value)
            if actual != str(expected):
                raise ProbeArtifactError(
                    f"Fitted probe array {name!r} changed after training: "
                    f"expected {expected}, got {actual}"
                )
            arrays[str(name)] = value
        return arrays

    def _require_replayable(self) -> dict[str, Any]:
        contract = self.contract
        if contract is None or not bool(self.capabilities.get("replay")):
            reason = self.capabilities.get("reason") or "Artifact is not replayable."
            raise NonReplayableProbeError(str(reason))
        validate_probe_run_contract(contract)
        return contract


def load_probe_artifact(dataset: TraceDataset, artifact_id: str) -> LoadedProbeArtifact:
    """Load a probe while preserving readable legacy behavior."""

    artifact = dataset.load_artifact(artifact_id)
    if artifact.artifact_type != "probe_suite":
        raise ProbeArtifactError(
            f"Artifact {artifact_id!r} has type {artifact.artifact_type!r}, not 'probe_suite'"
        )
    return LoadedProbeArtifact(dataset=dataset, artifact=artifact)


def make_probe_run_contract(
    *,
    experiment_card: Mapping[str, Any],
    run_spec: Mapping[str, Any],
    selector: Mapping[str, Any],
    source_rows_path: str,
    source_rows: pd.DataFrame,
    source_sites_path: str,
    source_sites: pd.DataFrame,
    scored_predictions_path: str,
    scored_predictions: pd.DataFrame,
    feature_matrix: np.ndarray,
    source_trace_fingerprints: Mapping[str, str],
    label_sources: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    uncertainty: Mapping[str, Any],
) -> dict[str, Any]:
    contract = {
        "schema_version": PROBE_RUN_CONTRACT_SCHEMA_VERSION,
        "status": "replayable",
        "capabilities": {
            "explain": True,
            "replay": True,
            "use": True,
            "legacy": False,
        },
        "experiment_card": dict(experiment_card),
        "run_spec": dict(run_spec),
        "source": {
            "selector": dict(selector),
            "source_rows_path": source_rows_path,
            "source_rows_fingerprint": dataframe_fingerprint(source_rows),
            "source_sites_path": source_sites_path,
            "source_sites_fingerprint": dataframe_fingerprint(source_sites),
            "scored_predictions_path": scored_predictions_path,
            "scored_predictions_fingerprint": dataframe_fingerprint(scored_predictions),
            "feature_matrix_fingerprint": _array_fingerprint(feature_matrix),
            "trace_fingerprints": dict(source_trace_fingerprints),
            "label_sources": [dict(value) for value in label_sources],
        },
        "model": dict(model),
        "uncertainty": dict(uncertainty),
    }
    validate_probe_run_contract(contract)
    return contract


def validate_probe_run_contract(contract: Mapping[str, Any]) -> None:
    """Fail clearly when a runner claims replayability without required state."""

    if int(contract.get("schema_version", -1)) != PROBE_RUN_CONTRACT_SCHEMA_VERSION:
        raise ProbeArtifactError(
            f"Unsupported probe-run schema version {contract.get('schema_version')!r}"
        )
    for key in ["capabilities", "experiment_card", "run_spec", "source", "model", "uncertainty"]:
        if not isinstance(contract.get(key), Mapping):
            raise ProbeArtifactError(f"Probe-run contract is missing mapping {key!r}")
    capabilities = dict(contract["capabilities"])
    if not all(bool(capabilities.get(action)) for action in ["explain", "replay", "use"]):
        raise ProbeArtifactError("Replayable probe contract must support explain, replay, and use")
    source = dict(contract["source"])
    if not isinstance(source.get("selector"), Mapping):
        raise ProbeArtifactError("Probe-run source selector must be a mapping")
    for key in [
        "selector",
        "source_rows_path",
        "source_rows_fingerprint",
        "source_sites_path",
        "source_sites_fingerprint",
        "scored_predictions_path",
        "scored_predictions_fingerprint",
        "feature_matrix_fingerprint",
        "trace_fingerprints",
        "label_sources",
    ]:
        if key not in source:
            raise ProbeArtifactError(f"Probe-run source is missing {key!r}")
    model = dict(contract["model"])
    for key in [
        "format",
        "probe_type",
        "feature_dim",
        "array_names",
        "array_fingerprints",
    ]:
        if key not in model:
            raise ProbeArtifactError(f"Probe-run model is missing {key!r}")
    array_names = dict(model.get("array_names") or {})
    for key in ["feature_mean", "feature_scale"]:
        if key not in array_names:
            raise ProbeArtifactError(f"Probe-run model arrays are missing {key!r}")
    model_format = str(model["format"])
    if model_format == "standardized_linear_v1":
        for key in ["weights", "bias"]:
            if key not in array_names:
                raise ProbeArtifactError(f"Linear probe arrays are missing {key!r}")
    elif model_format == "standardized_mlp_v1":
        if not array_names.get("layer_weights") or not array_names.get("layer_biases"):
            raise ProbeArtifactError("MLP probe arrays are missing fitted layers")
    else:
        raise ProbeArtifactError(f"Unsupported fitted probe format {model_format!r}")
    fitted_names = _fitted_array_names(array_names)
    fingerprints = model.get("array_fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise ProbeArtifactError("Probe-run model array fingerprints must be a mapping")
    fingerprint_names = {str(name) for name in fingerprints}
    if fingerprint_names != fitted_names:
        missing = sorted(fitted_names - fingerprint_names)
        unexpected = sorted(fingerprint_names - fitted_names)
        raise ProbeArtifactError(
            "Probe-run model array fingerprints do not match fitted arrays: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if any(not str(value).startswith("sha256:") for value in fingerprints.values()):
        raise ProbeArtifactError("Probe-run model contains an invalid array fingerprint")
    if str(model["probe_type"]) == "regression":
        tolerance = model.get("prediction_tolerance")
        if not isinstance(tolerance, Mapping):
            raise ProbeArtifactError("Regression probe prediction tolerance must be a mapping")
        for key in ["absolute", "relative"]:
            try:
                value = float(tolerance[key])
            except (KeyError, TypeError, ValueError) as error:
                raise ProbeArtifactError(
                    f"Regression probe prediction tolerance is missing numeric {key!r}"
                ) from error
            if not np.isfinite(value) or value < 0:
                raise ProbeArtifactError(
                    f"Regression probe prediction tolerance {key!r} must be finite and nonnegative"
                )


def _fitted_array_names(array_names: Mapping[str, Any]) -> set[str]:
    names: set[str] = set()
    for value in array_names.values():
        if isinstance(value, str):
            names.add(value)
        elif isinstance(value, Sequence):
            names.update(str(item) for item in value)
    return names


def probe_label_sources(dataset: TraceDataset) -> list[dict[str, Any]]:
    """Record external label artifacts that may contribute probe metadata."""

    records: list[dict[str, Any]] = []
    for artifact_type in [
        INTERACTION_METRICS_ARTIFACT_TYPE,
        OBJECT_FLOW_ARTIFACT_TYPE,
        POLICY_CALL_LABELS_ARTIFACT_TYPE,
    ]:
        artifact = latest_loadable_artifact(dataset, artifact_type)
        if artifact is None:
            continue
        records.append(
            {
                "artifact_id": artifact.artifact_id,
                "artifact_type": artifact.artifact_type,
                "created_utc": artifact.created_utc,
                "fingerprint": _hash_json(artifact.to_dict()),
            }
        )
    return records


def source_trace_fingerprint_map(dataset: TraceDataset, rows: pd.DataFrame) -> dict[str, str]:
    trace_ids = sorted(str(value) for value in rows["trace_id"].dropna().unique())
    fingerprints: dict[str, str] = {}
    for trace_id in trace_ids:
        bundle = dataset.bundle(trace_id)
        fingerprints[trace_id] = str(
            bundle.fingerprints.get("trace_fingerprint") or _bundle_fingerprint(bundle)
        )
    return fingerprints


def dataframe_fingerprint(frame: pd.DataFrame) -> str:
    """Stable fingerprint for replay tables, including values and row order."""

    columns = [str(column) for column in frame.columns]
    digest = hashlib.sha256()
    _update_dataframe_digest(
        digest,
        {"columns": columns, "row_count": int(len(frame))},
    )
    for row in frame.itertuples(index=False, name=None):
        _update_dataframe_digest(digest, [_stable_cell(value) for value in row])
    return f"sha256:{digest.hexdigest()}"


def _update_dataframe_digest(digest: Any, value: Any) -> None:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, byteorder="big"))
    digest.update(encoded)


def _stable_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return _stable_cell(value.item())
    if isinstance(value, np.ndarray):
        return [_stable_cell(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _stable_cell(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stable_cell(item) for item in value]
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if np.isposinf(value):
            return {"__vla_lens_float__": "positive_infinity"}
        if np.isneginf(value):
            return {"__vla_lens_float__": "negative_infinity"}
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _prediction_classes(values: Sequence[Any]) -> np.ndarray:
    classes = np.asarray(values)
    if classes.dtype.hasobject:
        classes = np.asarray([str(value) for value in values], dtype=np.str_)
    return classes


def _validate_trace_fingerprints(dataset: TraceDataset, source: Mapping[str, Any]) -> None:
    expected = dict(source.get("trace_fingerprints") or {})
    missing = sorted(set(expected) - set(dataset.episode_index["trace_id"].astype(str)))
    if missing:
        raise ProbeArtifactError(f"Capture is missing source traces: {missing[:5]}")
    changed = []
    for trace_id, fingerprint in expected.items():
        bundle = dataset.bundle(str(trace_id))
        actual = str(bundle.fingerprints.get("trace_fingerprint") or _bundle_fingerprint(bundle))
        if actual != str(fingerprint):
            changed.append(str(trace_id))
    if changed:
        raise ProbeArtifactError(f"Source trace fingerprints changed: {changed[:5]}")


def _compare_predictions(
    *,
    artifact_id: str,
    probe_type: str,
    replayed: np.ndarray,
    saved: np.ndarray,
    feature_fingerprint: str,
    prediction_tolerance: Any,
) -> ProbeReplayResult:
    if probe_type == "classification":
        mismatch = np.asarray(replayed).astype(str) != np.asarray(saved).astype(str)
        return ProbeReplayResult(
            artifact_id=artifact_id,
            matched=not bool(mismatch.any()),
            row_count=int(len(replayed)),
            mismatch_count=int(mismatch.sum()),
            max_absolute_difference=None,
            absolute_tolerance=None,
            relative_tolerance=None,
            feature_matrix_fingerprint=feature_fingerprint,
        )
    actual = np.asarray(replayed, dtype=np.float64).reshape(-1)
    expected = pd.to_numeric(pd.Series(saved), errors="coerce").to_numpy(dtype=np.float64)
    difference = np.abs(actual - expected)
    tolerance = dict(prediction_tolerance or {})
    absolute_tolerance = float(tolerance.get("absolute") or 0.0)
    relative_tolerance = float(tolerance.get("relative") or 0.0)
    mismatch = ~np.isclose(
        actual,
        expected,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    return ProbeReplayResult(
        artifact_id=artifact_id,
        matched=not bool(mismatch.any()),
        row_count=int(len(actual)),
        mismatch_count=int(mismatch.sum()),
        max_absolute_difference=float(difference.max()) if len(difference) else 0.0,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        feature_matrix_fingerprint=feature_fingerprint,
    )


def _artifact_output_path(dataset: TraceDataset, value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        raise ProbeArtifactError("Probe artifact output paths must be relative to the dataset")
    roots = [dataset.root.resolve(), dataset._dataset_artifact_root().resolve()]
    candidates: list[Path] = []
    for root in dict.fromkeys(roots):
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root):
            raise ProbeArtifactError("Probe artifact output path leaves the dataset directory")
        candidates.append(candidate)
        if candidate.exists():
            return candidate
    return candidates[0]


def _read_replay_table(dataset: TraceDataset, value: Any, label: str) -> pd.DataFrame:
    path = _artifact_output_path(dataset, value)
    try:
        return pd.read_parquet(path)
    except Exception as error:
        raise ProbeArtifactError(f"Could not read saved probe {label} at {path}") from error


def _activation(values: np.ndarray, name: str) -> np.ndarray:
    if name == "relu":
        return np.maximum(values, 0)
    if name == "tanh":
        return np.tanh(values)
    if name == "logistic":
        return 1.0 / (1.0 + np.exp(-values))
    if name == "identity":
        return values
    raise ProbeArtifactError(f"Unsupported MLP activation {name!r}")


def _output_activation(values: np.ndarray, name: str) -> np.ndarray:
    if name == "identity":
        return values
    if name == "logistic":
        return 1.0 / (1.0 + np.exp(-values))
    if name == "softmax":
        shifted = values - np.max(values, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)
    raise ProbeArtifactError(f"Unsupported MLP output activation {name!r}")
