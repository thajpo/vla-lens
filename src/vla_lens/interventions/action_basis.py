"""Runtime-free action basis provenance and delta helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from vla_lens.interventions.serialization import (
    jsonable,
    mapping_from,
    optional_str,
    require_literal,
    require_nonempty,
    required_mapping,
    tuple_from,
)
from vla_lens.interventions.specs import ActionBasisRequest

BASIS_STATUSES = {"ok", "partial", "failed"}
RESULT_STATUSES = {"ok", "partial", "failed"}
SUPPORTED_BASES = {"raw", "gripper", "eef_delta_xyz", "rotation", "speed"}


@dataclass(frozen=True, slots=True)
class ActionSchemaRef:
    """Stable reference to the saved action schema used for basis resolution."""

    trace_id: str | None = None
    action_array_ref: str = "action_chunks"
    normalization_id: str | None = None
    action_dim_count: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "action_array_ref": self.action_array_ref,
            "normalization_id": self.normalization_id,
            "action_dim_count": self.action_dim_count,
            "metadata": jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionSchemaRef":
        data = required_mapping(payload, field="ActionSchemaRef")
        count = data.get("action_dim_count")
        return cls(
            trace_id=optional_str(data.get("trace_id")),
            action_array_ref=str(data.get("action_array_ref") or "action_chunks"),
            normalization_id=optional_str(data.get("normalization_id")),
            action_dim_count=None if count is None else int(count),
            metadata=mapping_from(data.get("metadata"), field="metadata"),
        )


@dataclass(frozen=True, slots=True)
class ActionNormalizationSpec:
    """Action normalization/schema row normalized into JSON-safe fields."""

    normalization_id: str | None = None
    mode: str | None = None
    stats_ref: str | None = None
    action_dim_names: tuple[str, ...] = ()
    action_labels: tuple[str, ...] = ()
    action_units: tuple[str, ...] = ()
    normalized_action_array_ref: str | None = None
    unnormalized_action_array_ref: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalization_id": self.normalization_id,
            "mode": self.mode,
            "stats_ref": self.stats_ref,
            "action_dim_names": list(self.action_dim_names),
            "action_labels": list(self.action_labels),
            "action_units": list(self.action_units),
            "normalized_action_array_ref": self.normalized_action_array_ref,
            "unnormalized_action_array_ref": self.unnormalized_action_array_ref,
            "metadata": jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionNormalizationSpec":
        data = required_mapping(payload, field="ActionNormalizationSpec")
        return cls(
            normalization_id=optional_str(data.get("normalization_id")),
            mode=optional_str(data.get("mode")),
            stats_ref=optional_str(data.get("stats_ref")),
            action_dim_names=tuple_from(
                data.get("action_dim_names"),
                cast=str,
                field="action_dim_names",
            ),
            action_labels=tuple_from(
                data.get("action_labels"),
                cast=str,
                field="action_labels",
            ),
            action_units=tuple_from(
                data.get("action_units"),
                cast=str,
                field="action_units",
            ),
            normalized_action_array_ref=optional_str(data.get("normalized_action_array_ref")),
            unnormalized_action_array_ref=optional_str(
                data.get("unnormalized_action_array_ref"),
            ),
            metadata=mapping_from(data.get("metadata"), field="metadata"),
        )


@dataclass(frozen=True, slots=True)
class ActionBasisResolution:
    """Resolution record for one requested action basis."""

    basis_name: str
    status: str
    action_schema_ref: ActionSchemaRef
    basis_resolution: Mapping[str, Any] = field(default_factory=dict)
    units: Mapping[str, Any] = field(default_factory=dict)
    sign_convention: Mapping[str, Any] = field(default_factory=dict)
    source_dimensions: Mapping[str, Any] = field(default_factory=dict)
    normalization: Mapping[str, Any] = field(default_factory=dict)
    coordinate_frame: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_nonempty(self.basis_name, field="basis_name")
        require_literal(self.status, BASIS_STATUSES, field="action basis status")

    @property
    def available(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis_name": self.basis_name,
            "status": self.status,
            "available": self.available,
            "action_schema_ref": self.action_schema_ref.to_dict(),
            "basis_resolution": jsonable(self.basis_resolution),
            "units": jsonable(self.units),
            "sign_convention": jsonable(self.sign_convention),
            "source_dimensions": jsonable(self.source_dimensions),
            "normalization": jsonable(self.normalization),
            "coordinate_frame": self.coordinate_frame,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionBasisResolution":
        data = required_mapping(payload, field="ActionBasisResolution")
        return cls(
            basis_name=str(data["basis_name"]),
            status=str(data["status"]),
            action_schema_ref=ActionSchemaRef.from_dict(
                required_mapping(data.get("action_schema_ref"), field="action_schema_ref"),
            ),
            basis_resolution=mapping_from(
                data.get("basis_resolution"),
                field="basis_resolution",
            ),
            units=mapping_from(data.get("units"), field="units"),
            sign_convention=mapping_from(
                data.get("sign_convention"),
                field="sign_convention",
            ),
            source_dimensions=mapping_from(
                data.get("source_dimensions"),
                field="source_dimensions",
            ),
            normalization=mapping_from(data.get("normalization"), field="normalization"),
            coordinate_frame=optional_str(data.get("coordinate_frame")),
            warnings=tuple_from(data.get("warnings"), cast=str, field="warnings"),
        )


@dataclass(frozen=True, slots=True)
class ActionBasisResult:
    """Resolution result for all requested action bases."""

    status: str
    action_schema_ref: ActionSchemaRef
    normalization: ActionNormalizationSpec
    requested: tuple[str, ...]
    resolutions: tuple[ActionBasisResolution, ...]
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_literal(self.status, RESULT_STATUSES, field="action basis result status")

    @property
    def available(self) -> tuple[str, ...]:
        return tuple(
            resolution.basis_name for resolution in self.resolutions if resolution.available
        )

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(
            resolution.basis_name for resolution in self.resolutions if not resolution.available
        )

    def resolution(self, basis_name: str) -> ActionBasisResolution | None:
        for resolution in self.resolutions:
            if resolution.basis_name == basis_name:
                return resolution
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "action_schema_ref": self.action_schema_ref.to_dict(),
            "normalization": self.normalization.to_dict(),
            "requested": list(self.requested),
            "available": list(self.available),
            "missing": list(self.missing),
            "resolutions": [resolution.to_dict() for resolution in self.resolutions],
            "basis_status": {
                resolution.basis_name: resolution.to_dict()
                for resolution in self.resolutions
            },
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionBasisResult":
        data = required_mapping(payload, field="ActionBasisResult")
        return cls(
            status=str(data["status"]),
            action_schema_ref=ActionSchemaRef.from_dict(
                required_mapping(data.get("action_schema_ref"), field="action_schema_ref"),
            ),
            normalization=ActionNormalizationSpec.from_dict(
                required_mapping(data.get("normalization"), field="normalization"),
            ),
            requested=tuple_from(data.get("requested"), cast=str, field="requested"),
            resolutions=tuple(
                ActionBasisResolution.from_dict(
                    required_mapping(item, field="resolution"),
                )
                for item in data.get("resolutions", ())
            ),
            warnings=tuple_from(data.get("warnings"), cast=str, field="warnings"),
            errors=tuple_from(data.get("errors"), cast=str, field="errors"),
        )


def resolve_action_basis(
    bundle: Any,
    request: ActionBasisRequest | Mapping[str, Any] | Sequence[str] | None = None,
) -> ActionBasisResult:
    """Resolve requested action bases from saved action schema metadata."""

    basis_request = _basis_request(request)
    normalization = _normalization_spec(bundle)
    action_schema_ref = ActionSchemaRef(
        trace_id=_trace_id(bundle),
        action_array_ref=normalization.normalized_action_array_ref or "action_chunks",
        normalization_id=normalization.normalization_id,
        action_dim_count=_action_dim_count(bundle, normalization),
        metadata={"source": "action_normalization"},
    )
    requested = tuple(dict.fromkeys(basis_request.basis or ("raw",)))
    resolutions = tuple(
        _resolve_one_basis(
            basis_name=basis,
            bundle=bundle,
            schema_ref=action_schema_ref,
            normalization=normalization,
        )
        for basis in requested
    )
    missing = tuple(resolution.basis_name for resolution in resolutions if not resolution.available)
    errors = tuple(
        warning
        for resolution in resolutions
        if resolution.status == "failed"
        for warning in resolution.warnings
    )
    warnings = tuple(
        warning
        for resolution in resolutions
        if resolution.status == "partial"
        for warning in resolution.warnings
    )
    if any(resolution.status == "failed" for resolution in resolutions):
        status = "failed" if len(missing) == len(resolutions) else "partial"
    elif missing:
        status = "partial"
    else:
        status = "ok"
    return ActionBasisResult(
        status=status,
        action_schema_ref=action_schema_ref,
        normalization=normalization,
        requested=requested,
        resolutions=resolutions,
        warnings=_dedupe(warnings),
        errors=_dedupe(errors),
    )


def action_delta_metrics(
    *,
    stored_original: Any,
    intervened: Any,
    basis_result: ActionBasisResult,
    noop: Any | None = None,
    intended_basis: str | None = None,
) -> dict[str, dict[str, float]]:
    """Compute basis-aware action delta metrics from saved action arrays."""

    stored = _action_matrix(stored_original)
    changed = _action_matrix(intervened)
    noop_matrix = _action_matrix(noop) if noop is not None else None
    if stored.shape != changed.shape:
        raise ValueError("stored_original and intervened actions must have matching shapes")
    if noop_matrix is not None and noop_matrix.shape != stored.shape:
        raise ValueError("noop actions must match stored_original shape")
    intended_dims = _resolution_indices(basis_result.resolution(intended_basis or ""))
    all_dims = tuple(range(stored.shape[-1]))
    result: dict[str, dict[str, float]] = {}
    for resolution in basis_result.resolutions:
        if not resolution.available:
            continue
        dims = _resolution_indices(resolution)
        if not dims:
            continue
        metrics = {
            "raw_delta": _delta_norm(changed, stored, dims),
            "intervened_minus_stored_original": _delta_norm(changed, stored, dims),
        }
        if noop_matrix is not None:
            metrics["noop_delta"] = _delta_norm(noop_matrix, stored, dims)
            metrics["intervened_minus_noop"] = _delta_norm(changed, noop_matrix, dims)
        if basis_result.normalization.normalization_id or basis_result.normalization.mode:
            metrics["normalized_delta"] = metrics["intervened_minus_stored_original"]
        if intended_dims:
            metrics["side_effect_score"] = _side_effect_score(
                changed,
                stored,
                all_dims=all_dims,
                intended_dims=intended_dims,
                basis_dims=dims,
            )
        result[resolution.basis_name] = metrics
    return result


def _resolve_one_basis(
    *,
    basis_name: str,
    bundle: Any,
    schema_ref: ActionSchemaRef,
    normalization: ActionNormalizationSpec,
) -> ActionBasisResolution:
    if basis_name not in SUPPORTED_BASES:
        return _missing_resolution(
            basis_name,
            schema_ref,
            normalization,
            f"Action basis {basis_name!r} is not supported in v0.",
        )
    if basis_name == "raw":
        return _raw_resolution(bundle, schema_ref, normalization)
    if basis_name == "gripper":
        return _named_dimension_resolution(
            basis_name,
            schema_ref,
            normalization,
            aliases=(("gripper",),),
            default_unit="normalized gripper command",
            coordinate_frame="robot_action",
        )
    if basis_name == "eef_delta_xyz":
        return _named_dimension_resolution(
            basis_name,
            schema_ref,
            normalization,
            aliases=(("eef", "delta", "x"), ("x",), ("eef_delta_x",)),
            required_components=("x", "y", "z"),
            default_unit="normalized end-effector delta",
            coordinate_frame="end_effector",
        )
    if basis_name == "rotation":
        return _named_dimension_resolution(
            basis_name,
            schema_ref,
            normalization,
            aliases=(("roll",), ("pitch",), ("yaw",)),
            required_components=("roll", "pitch", "yaw"),
            default_unit="normalized rotation command",
            coordinate_frame="end_effector",
        )
    return _speed_resolution(schema_ref, normalization)


def _raw_resolution(
    bundle: Any,
    schema_ref: ActionSchemaRef,
    normalization: ActionNormalizationSpec,
) -> ActionBasisResolution:
    action_dim_count = schema_ref.action_dim_count
    if not _array_declared(bundle, schema_ref.action_array_ref) or not action_dim_count:
        return ActionBasisResolution(
            basis_name="raw",
            status="failed",
            action_schema_ref=schema_ref,
            basis_resolution={"kind": "raw_action_vector"},
            units={"kind": "native_saved_action_units"},
            sign_convention={"kind": "native_saved_action_sign"},
            source_dimensions={"indices": []},
            normalization=normalization.to_dict(),
            coordinate_frame="robot_action",
            warnings=("Raw action basis requires a declared action_chunks array.",),
        )
    return ActionBasisResolution(
        basis_name="raw",
        status="ok",
        action_schema_ref=schema_ref,
        basis_resolution={"kind": "raw_action_vector"},
        units={"kind": "native_saved_action_units"},
        sign_convention={"kind": "native_saved_action_sign"},
        source_dimensions={
            "indices": list(range(action_dim_count)),
            "names": list(_dimension_names(normalization, action_dim_count)),
        },
        normalization=normalization.to_dict(),
        coordinate_frame="robot_action",
    )


def _named_dimension_resolution(
    basis_name: str,
    schema_ref: ActionSchemaRef,
    normalization: ActionNormalizationSpec,
    *,
    aliases: tuple[tuple[str, ...], ...],
    required_components: tuple[str, ...] = (),
    default_unit: str,
    coordinate_frame: str,
) -> ActionBasisResolution:
    names = _dimension_names(normalization, schema_ref.action_dim_count or 0)
    if required_components:
        indices = []
        missing = []
        for component in required_components:
            index = _component_index(component, names)
            if index is None:
                missing.append(component)
            else:
                indices.append(index)
        if missing:
            return _missing_resolution(
                basis_name,
                schema_ref,
                normalization,
                f"Missing action dimensions for {basis_name}: {', '.join(missing)}.",
            )
    else:
        indices = [
            index
            for index, name in enumerate(names)
            if any(_matches_alias(name, alias) for alias in aliases)
        ]
        if not indices:
            return _missing_resolution(
                basis_name,
                schema_ref,
                normalization,
                f"Missing action dimension metadata for {basis_name}.",
            )
    return ActionBasisResolution(
        basis_name=basis_name,
        status="ok",
        action_schema_ref=schema_ref,
        basis_resolution={"kind": "named_action_dimensions"},
        units={
            "kind": "per_dimension",
            "values": _units_for_indices(normalization, indices, default_unit),
        },
        sign_convention=_sign_convention(normalization, basis_name),
        source_dimensions={
            "indices": indices,
            "names": [names[index] for index in indices],
        },
        normalization=normalization.to_dict(),
        coordinate_frame=coordinate_frame,
    )


def _speed_resolution(
    schema_ref: ActionSchemaRef,
    normalization: ActionNormalizationSpec,
) -> ActionBasisResolution:
    xyz = _named_dimension_resolution(
        "speed",
        schema_ref,
        normalization,
        aliases=(),
        required_components=("x", "y", "z"),
        default_unit="normalized speed command",
        coordinate_frame="end_effector",
    )
    if not xyz.available:
        return xyz
    return ActionBasisResolution(
        basis_name="speed",
        status="ok",
        action_schema_ref=schema_ref,
        basis_resolution={"kind": "l2_norm", "source_basis": "eef_delta_xyz"},
        units=xyz.units,
        sign_convention={"kind": "nonnegative_l2_norm"},
        source_dimensions=xyz.source_dimensions,
        normalization=normalization.to_dict(),
        coordinate_frame="end_effector",
    )


def _missing_resolution(
    basis_name: str,
    schema_ref: ActionSchemaRef,
    normalization: ActionNormalizationSpec,
    warning: str,
) -> ActionBasisResolution:
    return ActionBasisResolution(
        basis_name=basis_name,
        status="partial",
        action_schema_ref=schema_ref,
        basis_resolution={"kind": "unresolved"},
        units={},
        sign_convention={},
        source_dimensions={"indices": []},
        normalization=normalization.to_dict(),
        coordinate_frame=None,
        warnings=(warning,),
    )


def _normalization_spec(bundle: Any) -> ActionNormalizationSpec:
    table = getattr(bundle, "action_normalization", pd.DataFrame())
    if table is None or table.empty:
        return ActionNormalizationSpec()
    row = table.iloc[0].to_dict()
    metadata = _mapping(_json_parse(row.get("metadata")))
    names = _string_tuple(
        _json_parse(row.get("action_dim_names"))
        or metadata.get("action_names")
        or metadata.get("action_dim_names"),
    )
    labels = _string_tuple(metadata.get("action_labels"))
    units = _string_tuple(metadata.get("action_units"))
    return ActionNormalizationSpec(
        normalization_id=_optional_text(row.get("normalization_id")),
        mode=_optional_text(row.get("mode") or metadata.get("normalization_type")),
        stats_ref=_optional_text(row.get("stats_ref")),
        action_dim_names=names,
        action_labels=labels,
        action_units=units,
        normalized_action_array_ref=_optional_text(
            row.get("normalized_action_array_ref") or "action_chunks",
        ),
        unnormalized_action_array_ref=_optional_text(row.get("unnormalized_action_array_ref")),
        metadata=metadata,
    )


def _basis_request(
    request: ActionBasisRequest | Mapping[str, Any] | Sequence[str] | None,
) -> ActionBasisRequest:
    if isinstance(request, ActionBasisRequest):
        return request
    if request is None:
        return ActionBasisRequest(basis=("raw",))
    if isinstance(request, Mapping):
        return ActionBasisRequest.from_dict(request)
    if isinstance(request, str):
        return ActionBasisRequest(basis=(request,))
    return ActionBasisRequest(basis=tuple(str(item) for item in request))


def _action_dim_count(bundle: Any, normalization: ActionNormalizationSpec) -> int | None:
    if normalization.action_dim_names:
        return len(normalization.action_dim_names)
    record = _array_record(bundle, normalization.normalized_action_array_ref or "action_chunks")
    shape = _json_parse(record.get("shape"))
    axes = _json_parse(record.get("axes"))
    if isinstance(shape, list) and isinstance(axes, list) and "action_dim" in axes:
        return int(shape[int(axes.index("action_dim"))])
    if isinstance(shape, list) and shape:
        return int(shape[-1])
    return None


def _array_declared(bundle: Any, array_name: str) -> bool:
    if bundle is None:
        return False
    table = getattr(bundle, "array_index", pd.DataFrame())
    if table is None or table.empty or "name" not in table:
        return False
    return bool((table["name"].astype(str) == array_name).any())


def _array_record(bundle: Any, array_name: str) -> dict[str, Any]:
    if bundle is None:
        return {}
    table = getattr(bundle, "array_index", pd.DataFrame())
    if table is None or table.empty or "name" not in table:
        return {}
    matches = table.loc[table["name"].astype(str) == array_name]
    if matches.empty:
        return {}
    return dict(matches.iloc[0].to_dict())


def _dimension_names(
    normalization: ActionNormalizationSpec,
    action_dim_count: int,
) -> tuple[str, ...]:
    if normalization.action_dim_names:
        return normalization.action_dim_names
    return tuple(f"dim_{index}" for index in range(action_dim_count))


def _component_index(component: str, names: Sequence[str]) -> int | None:
    component = component.lower()
    aliases = {
        "x": ("x", "eef_delta_x", "delta_x", "eef_x"),
        "y": ("y", "eef_delta_y", "delta_y", "eef_y"),
        "z": ("z", "eef_delta_z", "delta_z", "eef_z"),
        "roll": ("roll", "rx", "rot_x", "rotation_x"),
        "pitch": ("pitch", "ry", "rot_y", "rotation_y"),
        "yaw": ("yaw", "rz", "rot_z", "rotation_z"),
    }.get(component, (component,))
    for index, name in enumerate(names):
        normalized = _normalize_name(name)
        if normalized in aliases or any(normalized.endswith(f"_{alias}") for alias in aliases):
            return index
    return None


def _matches_alias(name: str, alias: tuple[str, ...]) -> bool:
    normalized = _normalize_name(name)
    return all(part in normalized for part in alias)


def _units_for_indices(
    normalization: ActionNormalizationSpec,
    indices: Sequence[int],
    default_unit: str,
) -> dict[str, str]:
    values: dict[str, str] = {}
    for index in indices:
        name = (
            normalization.action_dim_names[index]
            if index < len(normalization.action_dim_names)
            else f"dim_{index}"
        )
        unit = (
            normalization.action_units[index]
            if index < len(normalization.action_units) and normalization.action_units[index]
            else default_unit
        )
        values[str(name)] = str(unit)
    return values


def _sign_convention(
    normalization: ActionNormalizationSpec,
    basis_name: str,
) -> dict[str, Any]:
    convention = normalization.metadata.get("sign_convention")
    if isinstance(convention, Mapping):
        value = convention.get(basis_name)
        if isinstance(value, Mapping):
            return dict(value)
        if value:
            return {"description": str(value)}
    return {
        "kind": "declared_dimension_sign",
        "description": "Positive and negative directions follow the saved action schema.",
    }


def _resolution_indices(resolution: ActionBasisResolution | None) -> tuple[int, ...]:
    if resolution is None:
        return ()
    indices = resolution.source_dimensions.get("indices")
    if not isinstance(indices, Sequence) or isinstance(indices, str):
        return ()
    return tuple(int(index) for index in indices)


def _action_matrix(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError("action array must have at least one dimension")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array.reshape(-1, array.shape[-1])


def _delta_norm(left: np.ndarray, right: np.ndarray, dims: Sequence[int]) -> float:
    delta = left[:, tuple(dims)] - right[:, tuple(dims)]
    return float(np.linalg.norm(delta))


def _side_effect_score(
    changed: np.ndarray,
    stored: np.ndarray,
    *,
    all_dims: Sequence[int],
    intended_dims: Sequence[int],
    basis_dims: Sequence[int],
) -> float:
    if set(basis_dims).issubset(set(intended_dims)):
        return 0.0
    outside = tuple(dim for dim in all_dims if dim not in set(intended_dims))
    total = _delta_norm(changed, stored, all_dims)
    if total <= 1e-12:
        return 0.0
    return float(_delta_norm(changed, stored, outside) / total)


def _trace_id(bundle: Any) -> str | None:
    manifest = getattr(bundle, "manifest", None)
    return optional_str(getattr(manifest, "trace_id", None))


def _json_parse(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _string_tuple(value: Any) -> tuple[str, ...]:
    parsed = _json_parse(value)
    if isinstance(parsed, Mapping):
        parsed = parsed.values()
    if isinstance(parsed, Sequence) and not isinstance(parsed, str):
        return tuple(str(item) for item in parsed if not _missing(item))
    return ()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_text(value: Any) -> str | None:
    if _missing(value):
        return None
    return str(value)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null", "nan"}
    if isinstance(value, float):
        return not math.isfinite(value)
    return False


def _normalize_name(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(".", "_")


def _dedupe(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


__all__ = [
    "ActionBasisRequest",
    "ActionBasisResolution",
    "ActionBasisResult",
    "ActionNormalizationSpec",
    "ActionSchemaRef",
    "action_delta_metrics",
    "resolve_action_basis",
]
