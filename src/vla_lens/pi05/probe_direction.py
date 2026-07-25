"""Resolve saved object-identity probes into PI0.5 VLM steering directions.

This module intentionally depends only on the normal VLA Lens runtime.  It
loads saved NumPy/pandas evidence, validates the artifact and request against
one another, and returns arrays that the dedicated PI0.5 executor can convert
to Torch tensors later.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from vla_lens.interventions import TargetSpec
from vla_lens.traces import TraceDataset

OBJECT_ROI_ARTIFACT_TYPE = "object_roi_identity_study"
OBJECT_ROI_METHOD = "object_roi"
PI05_VLM_HIDDEN_DIM = 2048


@dataclass(frozen=True, slots=True)
class ResolvedProbeDirection:
    """A validated linear identity contrast and its exact raw-space mapping."""

    artifact_id: str
    artifact_type: str
    method: str
    layer: int
    model_site: str
    token_indices: tuple[int, ...]
    instance_index: int
    trace_id: str
    policy_call_index: int
    target_class: int
    contrast_class: int | str
    target_name: str
    contrast_name: str
    classes: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    classifier_weights: np.ndarray
    classifier_bias: np.ndarray
    channel_input_center: np.ndarray
    channel_input_scale: np.ndarray
    channel_pca_center: np.ndarray
    channel_components: np.ndarray
    compact_direction: np.ndarray
    raw_delta_per_strength: np.ndarray
    inverse_projection: np.ndarray
    provenance: Mapping[str, Any]

    @property
    def hidden_dim(self) -> int:
        return int(self.channel_input_center.size)

    @property
    def feature_dim(self) -> int:
        return int(self.feature_mean.size)

    def compact_features(self, raw_hidden: np.ndarray) -> np.ndarray:
        """Replay the artifact's raw-hidden to compact-feature transform."""
        values = np.asarray(raw_hidden, dtype=np.float64)
        if values.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Raw hidden width {values.shape[-1]} does not match {self.hidden_dim}"
            )
        scaled = (values - self.channel_input_center) / self.channel_input_scale
        return (scaled - self.channel_pca_center) @ self.channel_components.T

    def standardized_features(self, raw_hidden: np.ndarray) -> np.ndarray:
        compact = self.compact_features(raw_hidden)
        return (compact - self.feature_mean) / self.feature_scale

    def score_difference(self, raw_hidden: np.ndarray) -> np.ndarray:
        """Return the exact target-minus-contrast linear logit."""
        standardized = self.standardized_features(raw_hidden)
        target_column = _class_column(self.classes, self.target_class)
        contrast_weights, contrast_bias = _contrast_parameters(
            self.classes,
            self.classifier_weights,
            self.classifier_bias,
            self.contrast_class,
        )
        weights = self.classifier_weights[:, target_column] - contrast_weights
        bias = self.classifier_bias[target_column] - contrast_bias
        return standardized @ weights + bias

    def direction_coordinate(self, raw_hidden: np.ndarray) -> np.ndarray:
        """Return the standardized feature coordinate along the applied direction."""
        return self.standardized_features(raw_hidden) @ self.compact_direction

    def raw_delta(self, standardized_delta: np.ndarray) -> np.ndarray:
        """Map a compact standardized delta to the minimum-norm raw delta."""
        delta = np.asarray(standardized_delta, dtype=np.float64)
        if delta.shape != (self.feature_dim,):
            raise ValueError(
                f"Standardized delta shape {delta.shape} does not match {(self.feature_dim,)}"
            )
        compact_delta = delta * self.feature_scale
        return np.asarray(compact_delta @ self.inverse_projection, dtype=np.float64)

    def add_delta(self, strength: float) -> np.ndarray:
        """Return the raw delta for a signed distance in standardized probe space."""
        return np.asarray(float(strength) * self.raw_delta_per_strength, dtype=np.float64)

    def project_out_delta(self, raw_roi_mean: np.ndarray, strength: float) -> np.ndarray:
        """Remove a fraction of the ROI mean's component along this contrast."""
        standardized = np.asarray(
            self.standardized_features(np.asarray(raw_roi_mean, dtype=np.float64)),
            dtype=np.float64,
        )
        if standardized.shape != (self.feature_dim,):
            raise ValueError(
                "Project-out requires one ROI-mean hidden vector, found "
                f"shape {standardized.shape}"
            )
        coordinate = float(standardized @ self.compact_direction)
        return self.raw_delta(-float(strength) * coordinate * self.compact_direction)

    def with_class_pair(
        self,
        target_class: int,
        contrast_class: int | str,
        *,
        purpose: str,
    ) -> "ResolvedProbeDirection":
        """Resolve a control class pair using the same validated artifact/request."""
        target = int(target_class)
        contrast = _class_value(contrast_class)
        direction = _normalized_class_direction(
            self.classes,
            self.classifier_weights,
            target,
            contrast,
        )
        raw_delta = self.raw_delta(direction)
        match_scale = float(np.linalg.norm(self.raw_delta_per_strength)) / float(
            np.linalg.norm(raw_delta)
        )
        raw_delta = raw_delta * match_scale
        names = dict(self.provenance.get("class_names") or {})
        provenance = {
            **dict(self.provenance),
            "purpose": purpose,
            "target_class": target,
            "contrast_class": contrast,
            "target_name": str(names.get(str(target), target)),
            "contrast_name": str(names.get(str(contrast), contrast)),
            "raw_norm_match_scale": match_scale,
        }
        provenance["compact_direction_sha256"] = array_sha256(direction)
        provenance["raw_delta_per_strength_sha256"] = array_sha256(raw_delta)
        return ResolvedProbeDirection(
            artifact_id=self.artifact_id,
            artifact_type=self.artifact_type,
            method=self.method,
            layer=self.layer,
            model_site=self.model_site,
            token_indices=self.token_indices,
            instance_index=self.instance_index,
            trace_id=self.trace_id,
            policy_call_index=self.policy_call_index,
            target_class=target,
            contrast_class=contrast,
            target_name=str(provenance["target_name"]),
            contrast_name=str(provenance["contrast_name"]),
            classes=self.classes,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            classifier_weights=self.classifier_weights,
            classifier_bias=self.classifier_bias,
            channel_input_center=self.channel_input_center,
            channel_input_scale=self.channel_input_scale,
            channel_pca_center=self.channel_pca_center,
            channel_components=self.channel_components,
            compact_direction=direction,
            raw_delta_per_strength=raw_delta,
            inverse_projection=self.inverse_projection,
            provenance=provenance,
        )

    def random_control(self, seed: int) -> "ResolvedProbeDirection":
        """Return a deterministic matched-norm direction in standardized space."""
        rng = np.random.default_rng(int(seed))
        direction = rng.normal(size=self.feature_dim)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            raise RuntimeError("Random control direction unexpectedly has zero norm")
        direction = np.asarray(direction / norm, dtype=np.float64)
        direction = direction - float(direction @ self.compact_direction) * self.compact_direction
        orthogonal_norm = float(np.linalg.norm(direction))
        if orthogonal_norm == 0.0:
            raise RuntimeError("Random control collapsed during probe-direction orthogonalization")
        direction = direction / orthogonal_norm
        raw_delta = self.raw_delta(direction)
        match_scale = float(np.linalg.norm(self.raw_delta_per_strength)) / float(
            np.linalg.norm(raw_delta)
        )
        raw_delta = raw_delta * match_scale
        provenance = {
            **dict(self.provenance),
            "purpose": "matched_random_control",
            "random_seed": int(seed),
            "raw_norm_match_scale": match_scale,
            "compact_cosine_to_probe": float(direction @ self.compact_direction),
            "compact_direction_sha256": array_sha256(direction),
            "raw_delta_per_strength_sha256": array_sha256(raw_delta),
        }
        return ResolvedProbeDirection(
            artifact_id=self.artifact_id,
            artifact_type=self.artifact_type,
            method=self.method,
            layer=self.layer,
            model_site=self.model_site,
            token_indices=self.token_indices,
            instance_index=self.instance_index,
            trace_id=self.trace_id,
            policy_call_index=self.policy_call_index,
            target_class=self.target_class,
            contrast_class=self.contrast_class,
            target_name=self.target_name,
            contrast_name=self.contrast_name,
            classes=self.classes,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            classifier_weights=self.classifier_weights,
            classifier_bias=self.classifier_bias,
            channel_input_center=self.channel_input_center,
            channel_input_scale=self.channel_input_scale,
            channel_pca_center=self.channel_pca_center,
            channel_components=self.channel_components,
            compact_direction=direction,
            raw_delta_per_strength=raw_delta,
            inverse_projection=self.inverse_projection,
            provenance=provenance,
        )


def resolve_object_roi_probe_direction(
    dataset: TraceDataset,
    target: TargetSpec,
    *,
    trace_id: str,
    policy_call_index: int,
) -> ResolvedProbeDirection:
    """Resolve and fail-closed validate an object-ROI linear probe target."""
    if target.kind not in {"probe_direction", "contrast_direction"}:
        raise ValueError(
            "Object-ROI direction requires target.kind=probe_direction or contrast_direction"
        )
    artifact_id = str(target.source_artifact_id or "")
    if not artifact_id:
        raise ValueError("Object-ROI direction requires target.source_artifact_id")
    artifact = dataset.load_artifact(artifact_id)
    if artifact.artifact_type != OBJECT_ROI_ARTIFACT_TYPE:
        raise ValueError(
            f"Expected artifact_type={OBJECT_ROI_ARTIFACT_TYPE!r}, found "
            f"{artifact.artifact_type!r}"
        )
    if target.source_artifact_type not in {None, OBJECT_ROI_ARTIFACT_TYPE}:
        raise ValueError("target.source_artifact_type disagrees with the saved artifact")
    if int(_mapping(artifact.method).get("schema_version", -1)) != 1:
        raise ValueError("Only object ROI artifact schema_version=1 is supported")

    representation = dict(target.representation)
    if representation.get("method") != OBJECT_ROI_METHOD:
        raise ValueError("Object-ROI intervention requires representation.method='object_roi'")
    instance_index = _required_int(representation, "instance_index")
    target_class = _required_int(representation, "target_class")
    contrast_class = _required_contrast(representation, "contrast_class")
    token_indices = _explicit_token_indices(target)

    models = _mapping(artifact.method.get("models"))
    model = _mapping(models.get(OBJECT_ROI_METHOD))
    if model.get("model") != "linear":
        raise ValueError("Object-ROI intervention requires a saved linear model")
    layer = int(model.get("layer", -1))
    prefix = str(model.get("prefix") or "")
    if layer < 0 or not prefix:
        raise ValueError("Saved object_roi model is missing its layer or array prefix")
    expected_site = f"pi05.vlm.layers.{layer}.prefix.hidden_tokens"
    if target.layer != layer or target.model_site != expected_site:
        raise ValueError(
            "Requested layer/site disagrees with the saved object_roi model: "
            f"expected layer={layer}, model_site={expected_site!r}"
        )
    if target.token_space != "pi05.prefix":
        raise ValueError("Object-ROI intervention requires token_space='pi05.prefix'")
    if target.model_family not in {None, "pi05"}:
        raise ValueError("Object-ROI intervention requires model_family='pi05'")
    runtime_model_id = str(dataset.bundle(trace_id).manifest.model_id or "")
    if target.model_id is not None and str(target.model_id) != runtime_model_id:
        raise ValueError("Requested model_id disagrees with the selected trace checkpoint")

    array_names = {
        "classes": f"{prefix}_classes",
        "feature_mean": f"{prefix}_feature_mean",
        "feature_scale": f"{prefix}_feature_scale",
        "weights": f"{prefix}_weights_0",
        "bias": f"{prefix}_biases_0",
        "input_center": "channel_input_center",
        "input_scale": "channel_input_scale",
        "pca_center": "channel_pca_center",
        "components": "channel_components",
    }
    missing = sorted(set(array_names.values()) - set(artifact.arrays))
    if missing:
        raise ValueError(f"Object-ROI artifact is missing required arrays: {missing}")
    arrays = {
        key: np.asarray(dataset.load_artifact_array(artifact, name), dtype=np.float64)
        for key, name in array_names.items()
    }
    classes = np.asarray(
        dataset.load_artifact_array(artifact, array_names["classes"])
    ).reshape(-1)
    arrays["classes"] = classes
    _validate_arrays(arrays, expected_feature_dim=int(model.get("feature_dim", -1)))

    tables, table_hashes = _load_evidence_tables(dataset, artifact)
    class_names = _class_names(tables["vocabulary"])
    instance = _validated_instance(
        tables["instances"],
        instance_index=instance_index,
        trace_id=trace_id,
        target_class=target_class,
        token_indices=token_indices,
    )
    source_row = _validate_source_row(
        tables["source_rows"],
        instance=instance,
        trace_id=trace_id,
        policy_call_index=int(policy_call_index),
    )
    if str(source_row.get("model_id") or "") != runtime_model_id:
        raise ValueError("Probe source row and replay trace use different model checkpoints")
    _validate_prediction(
        tables["predictions"],
        instance_index=instance_index,
        layer=layer,
        target_class=target_class,
    )
    _validate_token_metadata(tables["token_metadata"], token_indices)
    raw_wrong_roi = instance.get("wrong_roi_patch_indices")
    wrong_roi_token_indices = tuple(
        int(value) for value in (() if raw_wrong_roi is None else raw_wrong_roi)
    )
    if not wrong_roi_token_indices:
        raise ValueError("Saved instance does not provide a wrong-ROI token control")
    _validate_token_metadata(tables["token_metadata"], wrong_roi_token_indices)
    source_shape = _validate_source_site(
        tables["source_sites"],
        trace_id=trace_id,
        policy_call_index=int(policy_call_index),
        model_site=expected_site,
        hidden_dim=PI05_VLM_HIDDEN_DIM,
        token_indices=token_indices,
    )
    if str(instance.get("object_name")) != str(class_names.get(target_class)):
        raise ValueError("Instance object name disagrees with the artifact vocabulary")
    requested_target_name = representation.get("target_name")
    if requested_target_name is not None and str(requested_target_name) != str(
        class_names.get(target_class)
    ):
        raise ValueError("Requested target_name disagrees with the artifact vocabulary")
    requested_contrast_name = representation.get("contrast_name")
    contrast_name = (
        "class_mean"
        if contrast_class == "class_mean"
        else str(class_names.get(int(contrast_class)))
    )
    if requested_contrast_name is not None and str(requested_contrast_name) != contrast_name:
        raise ValueError("Requested contrast_name disagrees with the artifact vocabulary")

    direction = _normalized_class_direction(
        classes, arrays["weights"], target_class, contrast_class
    )
    raw_to_compact = (
        arrays["components"] / arrays["input_scale"][None, :]
    ).T
    rank = int(np.linalg.matrix_rank(raw_to_compact))
    if rank != arrays["feature_mean"].size:
        raise ValueError(
            f"Saved channel projection rank {rank} cannot reconstruct all compact features"
        )
    inverse = np.linalg.pinv(raw_to_compact)
    reconstruction_error = float(
        np.max(np.abs(inverse @ raw_to_compact - np.eye(inverse.shape[0])))
    )
    if reconstruction_error > 1e-8:
        raise ValueError(
            "Saved channel projection does not support an exact compact-space replay; "
            f"max error={reconstruction_error:.3g}"
        )
    raw_delta = np.asarray((direction * arrays["feature_scale"]) @ inverse)
    array_hashes = {
        array_names[key]: array_sha256(value) for key, value in arrays.items()
    }
    provenance = {
        "schema_kind": "vla_lens.pi05_probe_direction_resolution",
        "schema_version": 1,
        "artifact_id": artifact.artifact_id,
        "artifact_type": artifact.artifact_type,
        "artifact_created_utc": artifact.created_utc,
        "method": OBJECT_ROI_METHOD,
        "model": "linear",
        "layer": layer,
        "model_site": expected_site,
        "instance_index": instance_index,
        "trace_id": trace_id,
        "policy_call_index": int(policy_call_index),
        "target_class": target_class,
        "contrast_class": contrast_class,
        "target_name": str(class_names[target_class]),
        "contrast_name": contrast_name,
        "class_names": {str(key): value for key, value in class_names.items()},
        "token_indices": list(token_indices),
        "wrong_roi_token_indices": list(wrong_roi_token_indices),
        "model_id": runtime_model_id,
        "source_tensor_shape": list(source_shape),
        "strength_units": "standardized_probe_feature_l2",
        "mapping": (
            "standardized_delta -> multiply by saved feature_scale -> "
            "minimum-norm inverse of saved channel PCA delta transform"
        ),
        "projection_rank": rank,
        "projection_reconstruction_max_abs": reconstruction_error,
        "array_sha256": array_hashes,
        "evidence_table_sha256": table_hashes,
        "compact_direction_sha256": array_sha256(direction),
        "raw_delta_per_strength_sha256": array_sha256(raw_delta),
        "raw_delta_per_strength_l2": float(np.linalg.norm(raw_delta)),
        "raw_delta_per_strength_rms": float(np.sqrt(np.mean(np.square(raw_delta)))),
    }
    return ResolvedProbeDirection(
        artifact_id=artifact.artifact_id,
        artifact_type=artifact.artifact_type,
        method=OBJECT_ROI_METHOD,
        layer=layer,
        model_site=expected_site,
        token_indices=token_indices,
        instance_index=instance_index,
        trace_id=trace_id,
        policy_call_index=int(policy_call_index),
        target_class=target_class,
        contrast_class=contrast_class,
        target_name=str(class_names[target_class]),
        contrast_name=contrast_name,
        classes=classes,
        feature_mean=arrays["feature_mean"],
        feature_scale=arrays["feature_scale"],
        classifier_weights=arrays["weights"],
        classifier_bias=arrays["bias"],
        channel_input_center=arrays["input_center"],
        channel_input_scale=arrays["input_scale"],
        channel_pca_center=arrays["pca_center"],
        channel_components=arrays["components"],
        compact_direction=direction,
        raw_delta_per_strength=raw_delta,
        inverse_projection=inverse,
        provenance=provenance,
    )


def array_sha256(value: np.ndarray) -> str:
    """Hash array dtype, shape, and contiguous bytes for audit provenance."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_arrays(arrays: Mapping[str, np.ndarray], *, expected_feature_dim: int) -> None:
    classes = np.asarray(arrays["classes"]).reshape(-1)
    feature_mean = np.asarray(arrays["feature_mean"]).reshape(-1)
    feature_scale = np.asarray(arrays["feature_scale"]).reshape(-1)
    weights = np.asarray(arrays["weights"])
    bias = np.asarray(arrays["bias"]).reshape(-1)
    input_center = np.asarray(arrays["input_center"]).reshape(-1)
    input_scale = np.asarray(arrays["input_scale"]).reshape(-1)
    pca_center = np.asarray(arrays["pca_center"]).reshape(-1)
    components = np.asarray(arrays["components"])
    feature_dim = int(feature_mean.size)
    if feature_dim != expected_feature_dim or feature_dim < 2:
        raise ValueError("Saved object_roi feature_dim does not match its arrays")
    if feature_scale.shape != (feature_dim,) or np.any(feature_scale <= 0.0):
        raise ValueError("Saved object_roi feature_scale must be positive and feature-aligned")
    if len(classes) < 2 or len(np.unique(classes)) != len(classes):
        raise ValueError("Saved object_roi classes must be unique")
    if weights.shape != (feature_dim, len(classes)) or bias.shape != (len(classes),):
        raise ValueError("Saved linear classifier weights/bias have incompatible shapes")
    if input_center.shape != (PI05_VLM_HIDDEN_DIM,):
        raise ValueError(
            f"Saved channel projection must accept {PI05_VLM_HIDDEN_DIM} raw channels"
        )
    if input_scale.shape != input_center.shape or np.any(input_scale <= 0.0):
        raise ValueError("Saved channel input_scale must be positive and hidden-width aligned")
    if pca_center.shape != input_center.shape:
        raise ValueError("Saved PCA center is not aligned to the raw hidden width")
    if components.shape != (feature_dim, PI05_VLM_HIDDEN_DIM):
        raise ValueError("Saved channel PCA components have incompatible shape")
    for name, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Saved array {name!r} contains non-finite values")


def _normalized_class_direction(
    classes: np.ndarray,
    weights: np.ndarray,
    target_class: int,
    contrast_class: int | str,
) -> np.ndarray:
    contrast = _class_value(contrast_class)
    if contrast != "class_mean" and int(target_class) == int(contrast):
        raise ValueError("target_class and contrast_class must differ")
    target_column = _class_column(classes, target_class)
    contrast_weights, _ = _contrast_parameters(
        classes,
        weights,
        np.zeros(weights.shape[1], dtype=np.float64),
        contrast,
    )
    direction = np.asarray(weights[:, target_column] - contrast_weights, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("Selected target-vs-contrast classifier direction has zero norm")
    return direction / norm


def _class_column(classes: np.ndarray, class_value: int) -> int:
    matches = np.flatnonzero(np.asarray(classes).astype(int) == int(class_value))
    if len(matches) != 1:
        raise ValueError(f"Class {class_value} is not uniquely present in the saved classifier")
    return int(matches[0])


def _contrast_parameters(
    classes: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    contrast_class: int | str,
) -> tuple[np.ndarray, float]:
    contrast = _class_value(contrast_class)
    if contrast == "class_mean":
        return np.asarray(weights, dtype=np.float64).mean(axis=1), float(
            np.asarray(bias, dtype=np.float64).mean()
        )
    column = _class_column(classes, int(contrast))
    return np.asarray(weights[:, column], dtype=np.float64), float(bias[column])


def _class_value(value: Any) -> int | str:
    if str(value) == "class_mean":
        return "class_mean"
    if isinstance(value, bool):
        raise ValueError("Class values must be integer IDs or 'class_mean'")
    return int(value)


def _explicit_token_indices(target: TargetSpec) -> tuple[int, ...]:
    raw = target.token_selector.get("indices")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("Object-ROI intervention requires explicit token_selector.indices")
    indices = tuple(int(value) for value in raw)
    if len(set(indices)) != len(indices) or any(value < 0 for value in indices):
        raise ValueError("token_selector.indices must be unique non-negative integers")
    return indices


def _required_int(values: Mapping[str, Any], field: str) -> int:
    if field not in values or isinstance(values[field], bool):
        raise ValueError(f"target.representation.{field} is required")
    return int(values[field])


def _required_contrast(values: Mapping[str, Any], field: str) -> int | str:
    if field not in values:
        raise ValueError(f"target.representation.{field} is required")
    return _class_value(values[field])


def _load_evidence_tables(
    dataset: TraceDataset, artifact: Any
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    outputs = _mapping(artifact.method.get("outputs"))
    required = (
        "instances",
        "predictions",
        "vocabulary",
        "token_metadata",
        "source_sites",
        "source_rows",
    )
    root = dataset._dataset_artifact_root().resolve()
    tables: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    for name in required:
        relative = outputs.get(name)
        if not relative:
            raise ValueError(f"Object-ROI artifact is missing evidence table {name!r}")
        path = (root / str(relative)).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError(f"Object-ROI evidence table path is invalid: {relative!r}")
        payload = path.read_bytes()
        hashes[name] = hashlib.sha256(payload).hexdigest()
        tables[name] = pd.read_parquet(path)
    return tables, hashes


def _class_names(vocabulary: pd.DataFrame) -> dict[int, str]:
    required = {"object_index", "object_name"}
    if not required.issubset(vocabulary.columns):
        raise ValueError("Artifact vocabulary is missing object_index/object_name")
    names = {
        int(row.object_index): str(row.object_name)
        for row in vocabulary.itertuples(index=False)
    }
    if len(names) != len(vocabulary):
        raise ValueError("Artifact vocabulary contains duplicate object indices")
    return names


def _validated_instance(
    instances: pd.DataFrame,
    *,
    instance_index: int,
    trace_id: str,
    target_class: int,
    token_indices: tuple[int, ...],
) -> Mapping[str, Any]:
    if not 0 <= instance_index < len(instances):
        raise ValueError(f"instance_index {instance_index} is outside the saved instance table")
    row = instances.iloc[int(instance_index)]
    if str(row.get("trace_id")) != str(trace_id):
        raise ValueError("Requested trace_id disagrees with the saved object instance")
    if int(row.get("object_index")) != int(target_class):
        raise ValueError("Requested target_class disagrees with the saved object instance")
    saved_tokens = tuple(int(value) for value in row.get("roi_patch_indices"))
    if token_indices != saved_tokens:
        raise ValueError("Requested token indices disagree with the saved object ROI")
    return row.to_dict()


def _validate_prediction(
    predictions: pd.DataFrame,
    *,
    instance_index: int,
    layer: int,
    target_class: int,
) -> None:
    selected = predictions.loc[
        (predictions["instance_index"].astype(int) == int(instance_index))
        & (predictions["method"].astype(str) == OBJECT_ROI_METHOD)
    ]
    if len(selected) != 1:
        raise ValueError("Artifact does not contain one object_roi prediction for the instance")
    row = selected.iloc[0]
    if (
        str(row.get("model")) != "linear"
        or int(row.get("layer")) != int(layer)
        or int(row.get("object_index")) != int(target_class)
    ):
        raise ValueError("Saved instance prediction disagrees with the selected linear model")
    if not bool(row.get("correct")):
        raise ValueError("First intervention recipient must be a correctly decoded ROI instance")


def _validate_source_row(
    source_rows: pd.DataFrame,
    *,
    instance: Mapping[str, Any],
    trace_id: str,
    policy_call_index: int,
) -> Mapping[str, Any]:
    row_index = int(instance.get("source_row_index", -1))
    if not 0 <= row_index < len(source_rows):
        raise ValueError("Saved object instance has an invalid source_row_index")
    row = source_rows.iloc[row_index]
    if str(row.get("trace_id")) != str(trace_id):
        raise ValueError("Object instance source row disagrees with the requested trace")
    recorded_call = row.get("policy_call_index", row.get("sample_index"))
    if recorded_call is None or int(recorded_call) != int(policy_call_index):
        raise ValueError(
            "Object instance source row disagrees with the requested policy call"
        )
    return row.to_dict()


def _validate_token_metadata(metadata: pd.DataFrame, token_indices: tuple[int, ...]) -> None:
    selected = metadata.loc[metadata["token_index"].astype(int).isin(token_indices)]
    if len(selected) != len(token_indices):
        raise ValueError("Artifact token metadata does not contain every requested ROI token")
    if set(selected["token_space_id"].astype(str)) != {"pi05.prefix"}:
        raise ValueError("Requested ROI tokens are not all in pi05.prefix")
    if set(selected["camera_id"].astype(str)) != {"main"}:
        raise ValueError("RQ-015 object ROI tokens must all belong to the main camera")
    if set(selected["camera_slot_index"].astype(int)) != {0}:
        raise ValueError("RQ-015 object ROI tokens must belong to main-camera slot 0")
    if set(selected["token_kind"].astype(str)) != {"image"}:
        raise ValueError("RQ-015 object ROI tokens must be image tokens")
    if not selected["prefix_mask"].astype(bool).all():
        raise ValueError("RQ-015 object ROI tokens must all be prefix tokens")
    if not np.array_equal(
        selected.sort_values("token_index")["token_index"].to_numpy(dtype=int),
        selected.sort_values("token_index")["patch_index"].to_numpy(dtype=int),
    ):
        raise ValueError("Artifact patch indices do not map exactly to global prefix tokens")


def _validate_source_site(
    source_sites: pd.DataFrame,
    *,
    trace_id: str,
    policy_call_index: int,
    model_site: str,
    hidden_dim: int,
    token_indices: tuple[int, ...],
) -> tuple[int, ...]:
    selected = source_sites.loc[
        (source_sites["trace_id"].astype(str) == str(trace_id))
        & (source_sites["name"].astype(str) == str(model_site))
    ]
    if len(selected) != 1:
        raise ValueError("Artifact does not declare exactly one matching runtime source site")
    row = selected.iloc[0]
    shape = tuple(int(value) for value in json.loads(str(row["shape"])))
    axes = tuple(str(value) for value in json.loads(str(row["axes"])))
    if axes != ("policy_call", "token", "channel") or len(shape) != 3:
        raise ValueError("Saved source site does not have policy_call/token/channel axes")
    if shape[-1] != int(hidden_dim):
        raise ValueError("Saved source site hidden width disagrees with PI0.5")
    if not 0 <= int(policy_call_index) < shape[0]:
        raise ValueError("Requested policy call is outside the saved source tensor")
    if max(token_indices) >= shape[1]:
        raise ValueError("Requested ROI token is outside the saved source tensor")
    return shape


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "OBJECT_ROI_ARTIFACT_TYPE",
    "OBJECT_ROI_METHOD",
    "PI05_VLM_HIDDEN_DIM",
    "ResolvedProbeDirection",
    "array_sha256",
    "resolve_object_roi_probe_direction",
]
