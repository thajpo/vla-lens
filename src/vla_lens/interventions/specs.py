"""Runtime-free intervention request and capability contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from vla_lens.interventions.serialization import (
    jsonable,
    mapping_from,
    optional_float,
    optional_int,
    optional_str,
    require_literal,
    require_nonempty,
    required_mapping,
    tuple_from,
)

SCHEMA_VERSION = "0.1.0"

RUN_STATUSES = {"inspected_only", "ok", "partial", "failed"}
PREFLIGHT_STATUSES = RUN_STATUSES
CHECK_STATUSES = {"ok", "partial", "failed", "skipped", "unavailable"}
TARGET_KINDS = {
    "probe_direction",
    "contrast_direction",
    "activation_slice",
    "feature",
    "subspace",
    "head",
    "edge",
    "path",
    "manual",
}
ARTIFACT_DERIVED_TARGET_KINDS = {
    "probe_direction",
    "contrast_direction",
    "feature",
    "subspace",
    "head",
    "edge",
    "path",
}
OPERATOR_KINDS = {
    "add_direction",
    "add_decoder_direction",
    "attention_patch",
    "project_out_direction",
    "replace",
    "ablate",
    "clamp",
    "feature_ablate",
    "feature_boost",
    "feature_clamp",
    "head_ablate",
    "path_patch",
    "source_patch",
    "mean_replace",
}
CONTROL_KINDS = {
    "noop_rerun",
    "random_direction",
    "wrong_layer",
    "wrong_time",
    "wrong_token",
    "wrong_feature",
    "wrong_head",
    "wrong_edge",
    "shuffled_donor",
    "matched_cohort",
    "placebo_target",
    "heldout_split",
    "strength_sweep",
    "source_patch",
    "manual",
}
OUTCOME_KINDS = {
    "action",
    "rollout",
    "token",
    "probe",
    "metric",
    "activation",
    "attention",
    "pathway",
}


@dataclass(frozen=True, slots=True)
class TraceRef:
    """Stable address of a trace/episode inside a dataset."""

    trace_id: str
    dataset_id: str | None = None
    dataset_root_id: str | None = None
    episode_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_nonempty(self.trace_id, field="trace_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "dataset_id": self.dataset_id,
            "dataset_root_id": self.dataset_root_id,
            "episode_id": self.episode_id,
            "metadata": jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TraceRef":
        data = required_mapping(payload, field="TraceRef")
        return cls(
            trace_id=str(data["trace_id"]),
            dataset_id=optional_str(data.get("dataset_id")),
            dataset_root_id=optional_str(data.get("dataset_root_id")),
            episode_id=optional_str(data.get("episode_id")),
            metadata=mapping_from(data.get("metadata"), field="metadata"),
        )


@dataclass(frozen=True, slots=True)
class PolicyCallRef:
    """Address of one model invocation inside a trace."""

    trace_id: str
    policy_call_index: int
    timestep: int | None = None
    frame_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_nonempty(self.trace_id, field="trace_id")
        if self.policy_call_index < 0:
            raise ValueError("policy_call_index must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "policy_call_index": self.policy_call_index,
            "timestep": self.timestep,
            "frame_index": self.frame_index,
            "metadata": jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PolicyCallRef":
        data = required_mapping(payload, field="PolicyCallRef")
        return cls(
            trace_id=str(data["trace_id"]),
            policy_call_index=int(data["policy_call_index"]),
            timestep=optional_int(data.get("timestep")),
            frame_index=optional_int(data.get("frame_index")),
            metadata=mapping_from(data.get("metadata"), field="metadata"),
        )


@dataclass(frozen=True, slots=True)
class ContextSpec:
    """Display and identity context for an intervention request or run."""

    dataset_id: str | None = None
    dataset_root_id: str | None = None
    dataset_fingerprint: str | None = None
    trace_id: str | None = None
    episode_id: str | None = None
    policy_call_index: int | None = None
    timestep: int | None = None
    frame_index: int | None = None
    instruction: str | None = None
    task: str | None = None
    preview_media: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate_persisted_identity(self) -> None:
        if not (self.dataset_id or self.dataset_root_id):
            raise ValueError("dataset_id or dataset_root_id is required")
        require_nonempty(self.dataset_fingerprint, field="dataset_fingerprint")
        require_nonempty(self.trace_id, field="trace_id")
        if self.policy_call_index is None:
            raise ValueError("policy_call_index is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "dataset_root_id": self.dataset_root_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "trace_id": self.trace_id,
            "episode_id": self.episode_id,
            "policy_call_index": self.policy_call_index,
            "timestep": self.timestep,
            "frame_index": self.frame_index,
            "instruction": self.instruction,
            "task": self.task,
            "preview_media": jsonable(self.preview_media),
            "metadata": jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ContextSpec":
        data = required_mapping(payload, field="ContextSpec")
        return cls(
            dataset_id=optional_str(data.get("dataset_id")),
            dataset_root_id=optional_str(data.get("dataset_root_id")),
            dataset_fingerprint=optional_str(data.get("dataset_fingerprint")),
            trace_id=optional_str(data.get("trace_id")),
            episode_id=optional_str(data.get("episode_id")),
            policy_call_index=optional_int(data.get("policy_call_index")),
            timestep=optional_int(data.get("timestep")),
            frame_index=optional_int(data.get("frame_index")),
            instruction=optional_str(data.get("instruction")),
            task=optional_str(data.get("task")),
            preview_media=mapping_from(data.get("preview_media"), field="preview_media"),
            metadata=mapping_from(data.get("metadata"), field="metadata"),
        )


@dataclass(frozen=True, slots=True)
class RecipientSpec:
    """Runtime recipient of an intervention."""

    trace: TraceRef
    policy_call: PolicyCallRef | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace": self.trace.to_dict(),
            "policy_call": self.policy_call.to_dict() if self.policy_call else None,
            "metadata": jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RecipientSpec":
        data = required_mapping(payload, field="RecipientSpec")
        policy_call = data.get("policy_call")
        return cls(
            trace=TraceRef.from_dict(required_mapping(data.get("trace"), field="trace")),
            policy_call=(
                PolicyCallRef.from_dict(required_mapping(policy_call, field="policy_call"))
                if policy_call is not None
                else None
            ),
            metadata=mapping_from(data.get("metadata"), field="metadata"),
        )


@dataclass(frozen=True, slots=True)
class DonorSpec:
    """Optional source for patching/replacement-style interventions."""

    trace: TraceRef | None = None
    policy_call: PolicyCallRef | None = None
    activation_ref: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace": self.trace.to_dict() if self.trace else None,
            "policy_call": self.policy_call.to_dict() if self.policy_call else None,
            "activation_ref": self.activation_ref,
            "metadata": jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DonorSpec":
        data = required_mapping(payload, field="DonorSpec")
        trace = data.get("trace")
        policy_call = data.get("policy_call")
        return cls(
            trace=(
                TraceRef.from_dict(required_mapping(trace, field="trace"))
                if trace is not None
                else None
            ),
            policy_call=(
                PolicyCallRef.from_dict(required_mapping(policy_call, field="policy_call"))
                if policy_call is not None
                else None
            ),
            activation_ref=optional_str(data.get("activation_ref")),
            metadata=mapping_from(data.get("metadata"), field="metadata"),
        )


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """Normalized description of the internal object to manipulate."""

    kind: str
    source_artifact_id: str | None = None
    source_artifact_type: str | None = None
    model_id: str | None = None
    model_family: str | None = None
    model_site: str | None = None
    site_id: str | None = None
    module_path: str | None = None
    layer: int | None = None
    tensor_type: str | None = None
    token_space: str | None = None
    token_selector: Mapping[str, Any] = field(default_factory=dict)
    generation_step_selector: Mapping[str, Any] = field(default_factory=dict)
    reduction: str | None = None
    representation: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_literal(self.kind, TARGET_KINDS, field="target.kind")
        self.validate_source()

    def validate_source(self) -> None:
        artifact_derived = self.kind in ARTIFACT_DERIVED_TARGET_KINDS or self.source_artifact_type
        if artifact_derived and not self.source_artifact_id:
            raise ValueError("source_artifact_id is required for artifact-derived targets")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source_artifact_id": self.source_artifact_id,
            "source_artifact_type": self.source_artifact_type,
            "model_id": self.model_id,
            "model_family": self.model_family,
            "model_site": self.model_site,
            "site_id": self.site_id,
            "module_path": self.module_path,
            "layer": self.layer,
            "tensor_type": self.tensor_type,
            "token_space": self.token_space,
            "token_selector": jsonable(self.token_selector),
            "generation_step_selector": jsonable(self.generation_step_selector),
            "reduction": self.reduction,
            "representation": jsonable(self.representation),
            "metadata": jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TargetSpec":
        data = required_mapping(payload, field="TargetSpec")
        return cls(
            kind=str(data["kind"]),
            source_artifact_id=optional_str(data.get("source_artifact_id")),
            source_artifact_type=optional_str(data.get("source_artifact_type")),
            model_id=optional_str(data.get("model_id")),
            model_family=optional_str(data.get("model_family")),
            model_site=optional_str(data.get("model_site")),
            site_id=optional_str(data.get("site_id")),
            module_path=optional_str(data.get("module_path")),
            layer=optional_int(data.get("layer")),
            tensor_type=optional_str(data.get("tensor_type")),
            token_space=optional_str(data.get("token_space")),
            token_selector=mapping_from(data.get("token_selector"), field="token_selector"),
            generation_step_selector=mapping_from(
                data.get("generation_step_selector"),
                field="generation_step_selector",
            ),
            reduction=optional_str(data.get("reduction")),
            representation=mapping_from(data.get("representation"), field="representation"),
            metadata=mapping_from(data.get("metadata"), field="metadata"),
        )


@dataclass(frozen=True, slots=True)
class InterventionScheduleSpec:
    """When and where an intervention operator is applied."""

    policy_calls: tuple[int, ...] | str = "selected"
    timesteps: Mapping[str, Any] = field(default_factory=dict)
    generation_steps: Mapping[str, Any] | str = "all"
    action_horizon: Mapping[str, Any] | str = "full_chunk"
    tokens: Mapping[str, Any] | str = "target_tokens"
    condition: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.policy_calls, tuple) and any(item < 0 for item in self.policy_calls):
            raise ValueError("policy_calls must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_calls": jsonable(self.policy_calls),
            "timesteps": jsonable(self.timesteps),
            "generation_steps": jsonable(self.generation_steps),
            "action_horizon": jsonable(self.action_horizon),
            "tokens": jsonable(self.tokens),
            "condition": jsonable(self.condition),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InterventionScheduleSpec":
        data = required_mapping(payload, field="InterventionScheduleSpec")
        raw_policy_calls = data.get("policy_calls", "selected")
        policy_calls: tuple[int, ...] | str
        if isinstance(raw_policy_calls, str):
            policy_calls = raw_policy_calls
        else:
            policy_calls = tuple_from(raw_policy_calls, cast=int, field="policy_calls")
        return cls(
            policy_calls=policy_calls,
            timesteps=mapping_from(data.get("timesteps"), field="timesteps"),
            generation_steps=_mapping_or_str(data.get("generation_steps", "all")),
            action_horizon=_mapping_or_str(data.get("action_horizon", "full_chunk")),
            tokens=_mapping_or_str(data.get("tokens", "target_tokens")),
            condition=mapping_from(data.get("condition"), field="condition"),
        )


@dataclass(frozen=True, slots=True)
class InterventionOperatorSpec:
    """Operator and parameters for the target manipulation."""

    operator: str
    strength: float | None = None
    strengths: tuple[float, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_literal(self.operator, OPERATOR_KINDS, field="operator")

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "strength": self.strength,
            "strengths": list(self.strengths),
            "parameters": jsonable(self.parameters),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InterventionOperatorSpec":
        data = required_mapping(payload, field="InterventionOperatorSpec")
        return cls(
            operator=str(data["operator"]),
            strength=optional_float(data.get("strength")),
            strengths=tuple_from(data.get("strengths"), cast=float, field="strengths"),
            parameters=mapping_from(data.get("parameters"), field="parameters"),
        )


@dataclass(frozen=True, slots=True)
class ControlSpec:
    """Planned control condition for an intervention request."""

    kind: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    expected_effect: str | None = None

    def __post_init__(self) -> None:
        require_literal(self.kind, CONTROL_KINDS, field="control.kind")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "parameters": jsonable(self.parameters),
            "expected_effect": self.expected_effect,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlSpec":
        data = required_mapping(payload, field="ControlSpec")
        return cls(
            kind=str(data["kind"]),
            parameters=mapping_from(data.get("parameters"), field="parameters"),
            expected_effect=optional_str(data.get("expected_effect")),
        )


@dataclass(frozen=True, slots=True)
class OutcomeSpec:
    """Requested outcome measurement for an intervention."""

    kind: str
    basis: tuple[str, ...] = ("raw",)
    horizon: str | Mapping[str, Any] = "full_chunk"
    metrics: tuple[str, ...] = ("raw_delta", "normalized_delta")
    compare_to: str = "noop_if_available_else_stored_original"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_literal(self.kind, OUTCOME_KINDS, field="outcome.kind")
        if not self.basis:
            raise ValueError("outcome basis must include at least one basis")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "basis": list(self.basis),
            "horizon": jsonable(self.horizon),
            "metrics": list(self.metrics),
            "compare_to": self.compare_to,
            "parameters": jsonable(self.parameters),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OutcomeSpec":
        data = required_mapping(payload, field="OutcomeSpec")
        return cls(
            kind=str(data["kind"]),
            basis=tuple_from(data.get("basis", ("raw",)), cast=str, field="basis"),
            horizon=_mapping_or_str(data.get("horizon", "full_chunk")),
            metrics=tuple_from(
                data.get("metrics", ("raw_delta", "normalized_delta")),
                cast=str,
                field="metrics",
            ),
            compare_to=str(data.get("compare_to", "noop_if_available_else_stored_original")),
            parameters=mapping_from(data.get("parameters"), field="parameters"),
        )


@dataclass(frozen=True, slots=True)
class ActionBasisRequest:
    """Requested action coordinate systems plus available provenance hints."""

    basis: tuple[str, ...] = ("raw",)
    action_schema_ref: str | None = None
    basis_resolution: Mapping[str, Any] = field(default_factory=dict)
    units: Mapping[str, Any] = field(default_factory=dict)
    sign_convention: Mapping[str, Any] = field(default_factory=dict)
    source_dimensions: Mapping[str, Any] = field(default_factory=dict)
    normalization: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.basis:
            raise ValueError("action basis request must include at least one basis")

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis": list(self.basis),
            "action_schema_ref": self.action_schema_ref,
            "basis_resolution": jsonable(self.basis_resolution),
            "units": jsonable(self.units),
            "sign_convention": jsonable(self.sign_convention),
            "source_dimensions": jsonable(self.source_dimensions),
            "normalization": jsonable(self.normalization),
            "metadata": jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionBasisRequest":
        data = required_mapping(payload, field="ActionBasisRequest")
        return cls(
            basis=tuple_from(data.get("basis", ("raw",)), cast=str, field="basis"),
            action_schema_ref=optional_str(data.get("action_schema_ref")),
            basis_resolution=mapping_from(data.get("basis_resolution"), field="basis_resolution"),
            units=mapping_from(data.get("units"), field="units"),
            sign_convention=mapping_from(data.get("sign_convention"), field="sign_convention"),
            source_dimensions=mapping_from(
                data.get("source_dimensions"),
                field="source_dimensions",
            ),
            normalization=mapping_from(data.get("normalization"), field="normalization"),
            metadata=mapping_from(data.get("metadata"), field="metadata"),
        )


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    """One runtime capability check result."""

    name: str
    status: str
    message: str = ""
    ok: bool | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_nonempty(self.name, field="preflight check name")
        require_literal(self.status, CHECK_STATUSES, field="preflight check status")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "ok": self.ok,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "metadata": jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PreflightCheck":
        data = required_mapping(payload, field="PreflightCheck")
        return cls(
            name=str(data["name"]),
            status=str(data["status"]),
            message=str(data.get("message", "")),
            ok=data.get("ok") if data.get("ok") is None else bool(data.get("ok")),
            warnings=tuple_from(data.get("warnings"), cast=str, field="warnings"),
            errors=tuple_from(data.get("errors"), cast=str, field="errors"),
            metadata=mapping_from(data.get("metadata"), field="metadata"),
        )


@dataclass(frozen=True, slots=True)
class RuntimePreflightResult:
    """Runtime-agnostic answer to whether an intervention can run."""

    status: str
    checks: tuple[PreflightCheck, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    runtime_resolution: Mapping[str, Any] = field(default_factory=dict)
    missing_capabilities: tuple[str, ...] = ()
    capability_status: Mapping[str, bool] = field(default_factory=dict)
    target_resolution: Mapping[str, Any] = field(default_factory=dict)
    action_basis_status: Mapping[str, Any] = field(default_factory=dict)
    runtime_environment: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_literal(self.status, PREFLIGHT_STATUSES, field="preflight status")

    @property
    def ok(self) -> bool:
        return self.status == "ok" and not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ok": self.ok,
            "checks": [check.to_dict() for check in self.checks],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "runtime_resolution": jsonable(self.runtime_resolution),
            "missing_capabilities": list(self.missing_capabilities),
            "capability_status": dict(self.capability_status),
            "target_resolution": jsonable(self.target_resolution),
            "action_basis_status": jsonable(self.action_basis_status),
            "runtime_environment": jsonable(self.runtime_environment),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimePreflightResult":
        data = required_mapping(payload, field="RuntimePreflightResult")
        return cls(
            status=str(data["status"]),
            checks=tuple(
                PreflightCheck.from_dict(required_mapping(item, field="check"))
                for item in data.get("checks", ())
            ),
            warnings=tuple_from(data.get("warnings"), cast=str, field="warnings"),
            errors=tuple_from(data.get("errors"), cast=str, field="errors"),
            runtime_resolution=mapping_from(
                data.get("runtime_resolution"),
                field="runtime_resolution",
            ),
            missing_capabilities=tuple_from(
                data.get("missing_capabilities"),
                cast=str,
                field="missing_capabilities",
            ),
            capability_status={
                str(key): bool(value)
                for key, value in mapping_from(
                    data.get("capability_status"),
                    field="capability_status",
                ).items()
            },
            target_resolution=mapping_from(
                data.get("target_resolution"),
                field="target_resolution",
            ),
            action_basis_status=mapping_from(
                data.get("action_basis_status"),
                field="action_basis_status",
            ),
            runtime_environment=mapping_from(
                data.get("runtime_environment"),
                field="runtime_environment",
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeResolution:
    """Concrete runtime hook selected for a generic target."""

    adapter: str
    model_family: str
    requested_target: Mapping[str, Any]
    resolved_hook: Mapping[str, Any]
    model_id: str | None = None
    model_checkpoint: str | None = None
    call_index: int | None = None
    generation_step_mapping: Mapping[str, Any] = field(default_factory=dict)
    token_selector_mapping: Mapping[str, Any] = field(default_factory=dict)
    resolved_tensor_shape: tuple[int, ...] = ()
    resolved_dtype: str | None = None
    resolved_device: str | None = None
    runtime_environment: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_nonempty(self.adapter, field="runtime adapter")
        require_nonempty(self.model_family, field="model_family")

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "model_family": self.model_family,
            "model_id": self.model_id,
            "model_checkpoint": self.model_checkpoint,
            "call_index": self.call_index,
            "requested_target": jsonable(self.requested_target),
            "resolved_hook": jsonable(self.resolved_hook),
            "generation_step_mapping": jsonable(self.generation_step_mapping),
            "token_selector_mapping": jsonable(self.token_selector_mapping),
            "resolved_tensor_shape": list(self.resolved_tensor_shape),
            "resolved_dtype": self.resolved_dtype,
            "resolved_device": self.resolved_device,
            "runtime_environment": jsonable(self.runtime_environment),
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuntimeResolution":
        data = required_mapping(payload, field="RuntimeResolution")
        return cls(
            adapter=str(data["adapter"]),
            model_family=str(data["model_family"]),
            requested_target=mapping_from(data.get("requested_target"), field="requested_target"),
            resolved_hook=mapping_from(data.get("resolved_hook"), field="resolved_hook"),
            model_id=optional_str(data.get("model_id")),
            model_checkpoint=optional_str(data.get("model_checkpoint")),
            call_index=optional_int(data.get("call_index")),
            generation_step_mapping=mapping_from(
                data.get("generation_step_mapping"),
                field="generation_step_mapping",
            ),
            token_selector_mapping=mapping_from(
                data.get("token_selector_mapping"),
                field="token_selector_mapping",
            ),
            resolved_tensor_shape=tuple_from(
                data.get("resolved_tensor_shape"),
                cast=int,
                field="resolved_tensor_shape",
            ),
            resolved_dtype=optional_str(data.get("resolved_dtype")),
            resolved_device=optional_str(data.get("resolved_device")),
            runtime_environment=mapping_from(
                data.get("runtime_environment"),
                field="runtime_environment",
            ),
            warnings=tuple_from(data.get("warnings"), cast=str, field="warnings"),
        )


def _mapping_or_str(value: Any) -> Mapping[str, Any] | str:
    if isinstance(value, str):
        return value
    return mapping_from(value, field="mapping_or_str")
