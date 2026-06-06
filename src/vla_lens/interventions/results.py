"""Runtime-free intervention result and saved evidence contracts."""

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
    tuple_of_mappings,
    utc_now_iso,
)
from vla_lens.interventions.specs import (
    RUN_STATUSES,
    SCHEMA_VERSION,
    ContextSpec,
    RuntimePreflightResult,
    RuntimeResolution,
    TargetSpec,
)

TRIAL_KINDS = {
    "stored_original",
    "noop",
    "noop_rerun",
    "intervention",
    "control",
    "random_direction_control",
    "wrong_layer_control",
    "wrong_time_control",
    "wrong_token_control",
    "source_patch_control",
    "manual",
}
TRIAL_STATUSES = {"ok", "partial", "failed", "skipped", "inspected_only"}
CONTROL_RESULT_STATUSES = {"ok", "partial", "failed", "skipped"}


@dataclass(frozen=True, slots=True)
class InterventionTrial:
    """One inspected or executed intervention attempt."""

    trial_id: str
    trial_kind: str
    control_kind: str | None = None
    strength: float | None = None
    seed: int | None = None
    target_override: Mapping[str, Any] = field(default_factory=dict)
    operator_override: Mapping[str, Any] = field(default_factory=dict)
    schedule_override: Mapping[str, Any] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    status: str = "ok"
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_nonempty(self.trial_id, field="trial_id")
        require_literal(self.trial_kind, TRIAL_KINDS, field="trial_kind")
        require_literal(self.status, TRIAL_STATUSES, field="trial status")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "trial_kind": self.trial_kind,
            "control_kind": self.control_kind,
            "strength": self.strength,
            "seed": self.seed,
            "target_override": jsonable(self.target_override),
            "operator_override": jsonable(self.operator_override),
            "schedule_override": jsonable(self.schedule_override),
            "outputs": jsonable(self.outputs),
            "metrics": jsonable(self.metrics),
            "runtime": jsonable(self.runtime),
            "status": self.status,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InterventionTrial":
        data = required_mapping(payload, field="InterventionTrial")
        return cls(
            trial_id=str(data["trial_id"]),
            trial_kind=str(data["trial_kind"]),
            control_kind=optional_str(data.get("control_kind")),
            strength=optional_float(data.get("strength")),
            seed=optional_int(data.get("seed")),
            target_override=mapping_from(data.get("target_override"), field="target_override"),
            operator_override=mapping_from(
                data.get("operator_override"),
                field="operator_override",
            ),
            schedule_override=mapping_from(
                data.get("schedule_override"),
                field="schedule_override",
            ),
            outputs=mapping_from(data.get("outputs"), field="outputs"),
            metrics=mapping_from(data.get("metrics"), field="metrics"),
            runtime=mapping_from(data.get("runtime"), field="runtime"),
            status=str(data.get("status", "ok")),
            warnings=tuple_from(data.get("warnings"), cast=str, field="warnings"),
            errors=tuple_from(data.get("errors"), cast=str, field="errors"),
        )


@dataclass(frozen=True, slots=True)
class ActionOutcomeResult:
    """Action-delta result for one baseline/intervention comparison."""

    basis: str
    horizon: Mapping[str, Any] | str
    baseline_trial_id: str
    intervention_trial_id: str
    action_ref_baseline: str | None = None
    action_ref_intervened: str | None = None
    delta_ref: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    summaries: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_nonempty(self.basis, field="action outcome basis")
        require_nonempty(self.baseline_trial_id, field="baseline_trial_id")
        require_nonempty(self.intervention_trial_id, field="intervention_trial_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "action",
            "basis": self.basis,
            "horizon": jsonable(self.horizon),
            "baseline_trial_id": self.baseline_trial_id,
            "intervention_trial_id": self.intervention_trial_id,
            "action_ref_baseline": self.action_ref_baseline,
            "action_ref_intervened": self.action_ref_intervened,
            "delta_ref": self.delta_ref,
            "metrics": jsonable(self.metrics),
            "summaries": jsonable(self.summaries),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionOutcomeResult":
        data = required_mapping(payload, field="ActionOutcomeResult")
        return cls(
            basis=str(data["basis"]),
            horizon=data.get("horizon", "full_chunk")
            if isinstance(data.get("horizon", "full_chunk"), str)
            else mapping_from(data.get("horizon"), field="horizon"),
            baseline_trial_id=str(data["baseline_trial_id"]),
            intervention_trial_id=str(data["intervention_trial_id"]),
            action_ref_baseline=optional_str(data.get("action_ref_baseline")),
            action_ref_intervened=optional_str(data.get("action_ref_intervened")),
            delta_ref=optional_str(data.get("delta_ref")),
            metrics=mapping_from(data.get("metrics"), field="metrics"),
            summaries=mapping_from(data.get("summaries"), field="summaries"),
        )


@dataclass(frozen=True, slots=True)
class RolloutOutcomeResult:
    """Closed-loop rollout outcome shell."""

    rollout_ref: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    summaries: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "rollout",
            "rollout_ref": self.rollout_ref,
            "metrics": jsonable(self.metrics),
            "summaries": jsonable(self.summaries),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RolloutOutcomeResult":
        data = required_mapping(payload, field="RolloutOutcomeResult")
        return cls(
            rollout_ref=optional_str(data.get("rollout_ref")),
            metrics=mapping_from(data.get("metrics"), field="metrics"),
            summaries=mapping_from(data.get("summaries"), field="summaries"),
        )


@dataclass(frozen=True, slots=True)
class TokenOutcomeResult:
    """Token/logit/image-token outcome shell."""

    token_ref: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    summaries: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "token",
            "token_ref": self.token_ref,
            "metrics": jsonable(self.metrics),
            "summaries": jsonable(self.summaries),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TokenOutcomeResult":
        data = required_mapping(payload, field="TokenOutcomeResult")
        return cls(
            token_ref=optional_str(data.get("token_ref")),
            metrics=mapping_from(data.get("metrics"), field="metrics"),
            summaries=mapping_from(data.get("summaries"), field="summaries"),
        )


@dataclass(frozen=True, slots=True)
class ControlResult:
    """Executed or skipped control summary."""

    control_kind: str
    status: str
    trial_ids: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_nonempty(self.control_kind, field="control_kind")
        require_literal(self.status, CONTROL_RESULT_STATUSES, field="control result status")

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_kind": self.control_kind,
            "status": self.status,
            "trial_ids": list(self.trial_ids),
            "metrics": jsonable(self.metrics),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ControlResult":
        data = required_mapping(payload, field="ControlResult")
        return cls(
            control_kind=str(data["control_kind"]),
            status=str(data["status"]),
            trial_ids=tuple_from(data.get("trial_ids"), cast=str, field="trial_ids"),
            metrics=mapping_from(data.get("metrics"), field="metrics"),
            warnings=tuple_from(data.get("warnings"), cast=str, field="warnings"),
            errors=tuple_from(data.get("errors"), cast=str, field="errors"),
        )


@dataclass(frozen=True, slots=True)
class SweepAxis:
    """One dimension varied across a sweep."""

    name: str
    values: tuple[Any, ...]
    source: str = "request"
    path: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_nonempty(self.name, field="sweep axis name")
        if not self.values:
            raise ValueError("sweep axis values are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "values": [jsonable(value) for value in self.values],
            "source": self.source,
            "path": list(self.path),
            "metadata": jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SweepAxis":
        data = required_mapping(payload, field="SweepAxis")
        return cls(
            name=str(data["name"]),
            values=tuple_from(data.get("values"), cast=jsonable, field="values"),
            source=str(data.get("source", "request")),
            path=tuple_from(data.get("path"), cast=str, field="path"),
            metadata=mapping_from(data.get("metadata"), field="metadata"),
        )


@dataclass(frozen=True, slots=True)
class AggregateOutcomeResult:
    """Numeric aggregate over one metric collected from multiple runs."""

    metric: str
    count: int
    mean: float | None = None
    median: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    std: float | None = None
    coverage: float | None = None
    monotonicity: str | None = None
    values: tuple[float, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_nonempty(self.metric, field="aggregate metric")
        if self.count < 0:
            raise ValueError("aggregate count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "std": self.std,
            "coverage": self.coverage,
            "monotonicity": self.monotonicity,
            "values": list(self.values),
            "metadata": jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AggregateOutcomeResult":
        data = required_mapping(payload, field="AggregateOutcomeResult")
        return cls(
            metric=str(data["metric"]),
            count=int(data.get("count", 0)),
            mean=optional_float(data.get("mean")),
            median=optional_float(data.get("median")),
            minimum=optional_float(data.get("minimum")),
            maximum=optional_float(data.get("maximum")),
            std=optional_float(data.get("std")),
            coverage=optional_float(data.get("coverage")),
            monotonicity=optional_str(data.get("monotonicity")),
            values=tuple_from(data.get("values"), cast=float, field="values"),
            metadata=mapping_from(data.get("metadata"), field="metadata"),
        )


@dataclass(frozen=True, slots=True)
class CohortInterventionRequest:
    """Runtime-free request shell for applying one intervention spec over a cohort."""

    request_id: str
    base_run_id: str | None = None
    cohort: Mapping[str, Any] = field(default_factory=dict)
    axes: tuple[SweepAxis, ...] = ()
    controls: tuple[Mapping[str, Any], ...] = ()
    splits: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_nonempty(self.request_id, field="cohort intervention request_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "base_run_id": self.base_run_id,
            "cohort": jsonable(self.cohort),
            "axes": [axis.to_dict() for axis in self.axes],
            "controls": [jsonable(control) for control in self.controls],
            "splits": jsonable(self.splits),
            "provenance": jsonable(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CohortInterventionRequest":
        data = required_mapping(payload, field="CohortInterventionRequest")
        return cls(
            request_id=str(data["request_id"]),
            base_run_id=optional_str(data.get("base_run_id")),
            cohort=mapping_from(data.get("cohort"), field="cohort"),
            axes=tuple(
                SweepAxis.from_dict(required_mapping(item, field="axis"))
                for item in data.get("axes", ())
            ),
            controls=tuple_of_mappings(data.get("controls"), field="controls"),
            splits=mapping_from(data.get("splits"), field="splits"),
            provenance=mapping_from(data.get("provenance"), field="provenance"),
        )


@dataclass(frozen=True, slots=True)
class InterventionRun:
    """Canonical saved evidence payload for one intervention record."""

    run_id: str
    title: str
    status: str
    context: ContextSpec
    target: TargetSpec
    request: Mapping[str, Any]
    preflight: RuntimePreflightResult
    schema_version: str = SCHEMA_VERSION
    created_utc: str = field(default_factory=utc_now_iso)
    runtime_resolution: RuntimeResolution | None = None
    trials: tuple[InterventionTrial, ...] = ()
    outcomes: tuple[Mapping[str, Any], ...] = ()
    controls: tuple[Mapping[str, Any], ...] = ()
    outputs: tuple[str, ...] = ()
    display: Mapping[str, Any] = field(default_factory=dict)
    claim: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_nonempty(self.run_id, field="run_id")
        require_nonempty(self.title, field="title")
        require_literal(self.status, RUN_STATUSES, field="run status")
        self.context.validate_persisted_identity()
        self.target.validate_source()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "title": self.title,
            "status": self.status,
            "created_utc": self.created_utc,
            "context": self.context.to_dict(),
            "target": self.target.to_dict(),
            "request": jsonable(self.request),
            "preflight": self.preflight.to_dict(),
            "runtime_resolution": (
                self.runtime_resolution.to_dict() if self.runtime_resolution else None
            ),
            "trials": [trial.to_dict() for trial in self.trials],
            "outcomes": [jsonable(outcome) for outcome in self.outcomes],
            "controls": [jsonable(control) for control in self.controls],
            "outputs": list(self.outputs),
            "display": jsonable(self.display),
            "claim": jsonable(self.claim),
            "provenance": jsonable(self.provenance),
        }

    def to_workbench_spec(self):
        """Map this typed run into the current workbench compatibility shell."""
        from vla_lens.workbench.schema import InterventionRunSpec

        provenance = {
            **jsonable(self.provenance),
            "schema_kind": "vla_lens.intervention_run",
            "schema_version": self.schema_version,
            "dataset_id": self.context.dataset_id,
            "dataset_root_id": self.context.dataset_root_id,
            "dataset_fingerprint": self.context.dataset_fingerprint,
            "trace_id": self.context.trace_id,
            "episode_id": self.context.episode_id,
            "policy_call_index": self.context.policy_call_index,
            "source_artifact_id": self.target.source_artifact_id,
            "created_utc": self.created_utc,
        }
        return InterventionRunSpec(
            run_id=self.run_id,
            intervention_type="intervention_record",
            target=self.target.to_dict(),
            baseline={"context": self.context.to_dict()},
            intervention={"request": jsonable(self.request)},
            readouts={
                "schema_version": self.schema_version,
                "title": self.title,
                "status": self.status,
                "created_utc": self.created_utc,
                "preflight": self.preflight.to_dict(),
                "runtime_resolution": (
                    self.runtime_resolution.to_dict() if self.runtime_resolution else None
                ),
                "trials": [trial.to_dict() for trial in self.trials],
                "outcomes": [jsonable(outcome) for outcome in self.outcomes],
                "controls": [jsonable(control) for control in self.controls],
                "display": jsonable(self.display),
                "claim": jsonable(self.claim),
            },
            outputs=self.outputs,
            provenance=provenance,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InterventionRun":
        data = required_mapping(payload, field="InterventionRun")
        runtime_resolution = data.get("runtime_resolution")
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            run_id=str(data["run_id"]),
            title=str(data["title"]),
            status=str(data["status"]),
            created_utc=str(data["created_utc"]),
            context=ContextSpec.from_dict(required_mapping(data.get("context"), field="context")),
            target=TargetSpec.from_dict(required_mapping(data.get("target"), field="target")),
            request=mapping_from(data.get("request"), field="request"),
            preflight=RuntimePreflightResult.from_dict(
                required_mapping(data.get("preflight"), field="preflight")
            ),
            runtime_resolution=(
                RuntimeResolution.from_dict(
                    required_mapping(runtime_resolution, field="runtime_resolution")
                )
                if runtime_resolution is not None
                else None
            ),
            trials=tuple(
                InterventionTrial.from_dict(required_mapping(item, field="trial"))
                for item in data.get("trials", ())
            ),
            outcomes=tuple_of_mappings(data.get("outcomes"), field="outcomes"),
            controls=tuple_of_mappings(data.get("controls"), field="controls"),
            outputs=tuple_from(data.get("outputs"), cast=str, field="outputs"),
            display=mapping_from(data.get("display"), field="display"),
            claim=mapping_from(data.get("claim"), field="claim"),
            provenance=mapping_from(data.get("provenance"), field="provenance"),
        )

    @classmethod
    def from_workbench_spec(cls, spec: Any) -> "InterventionRun":
        """Reconstruct a typed run from an `InterventionRunSpec` shell."""
        if spec.intervention_type != "intervention_record":
            raise ValueError("Only intervention_record workbench specs are supported")
        readouts = mapping_from(spec.readouts, field="readouts")
        baseline = mapping_from(spec.baseline, field="baseline")
        intervention = mapping_from(spec.intervention, field="intervention")
        context = baseline.get("context")
        preflight = readouts.get("preflight")
        runtime_resolution = readouts.get("runtime_resolution")
        return cls(
            schema_version=str(
                readouts.get("schema_version")
                or spec.provenance.get("schema_version")
                or SCHEMA_VERSION
            ),
            run_id=spec.run_id,
            title=str(readouts.get("title") or spec.run_id),
            status=str(readouts["status"]),
            created_utc=str(readouts.get("created_utc") or spec.provenance.get("created_utc")),
            context=ContextSpec.from_dict(required_mapping(context, field="baseline.context")),
            target=TargetSpec.from_dict(spec.target),
            request=mapping_from(intervention.get("request"), field="intervention.request"),
            preflight=RuntimePreflightResult.from_dict(
                required_mapping(preflight, field="readouts.preflight")
            ),
            runtime_resolution=(
                RuntimeResolution.from_dict(
                    required_mapping(runtime_resolution, field="runtime_resolution")
                )
                if runtime_resolution is not None
                else None
            ),
            trials=tuple(
                InterventionTrial.from_dict(required_mapping(item, field="trial"))
                for item in readouts.get("trials", ())
            ),
            outcomes=tuple_of_mappings(readouts.get("outcomes"), field="outcomes"),
            controls=tuple_of_mappings(readouts.get("controls"), field="controls"),
            outputs=spec.outputs,
            display=mapping_from(readouts.get("display"), field="display"),
            claim=mapping_from(readouts.get("claim"), field="claim"),
            provenance=mapping_from(spec.provenance, field="provenance"),
        )


@dataclass(frozen=True, slots=True)
class InterventionSweep:
    """Typed shell for a collection of related intervention runs."""

    sweep_id: str
    run_ids: tuple[str, ...]
    schema_version: str = SCHEMA_VERSION
    axes: tuple[SweepAxis, ...] | Mapping[str, Any] = field(default_factory=dict)
    aggregate_outcomes: tuple[AggregateOutcomeResult, ...] = ()
    controls: tuple[Mapping[str, Any], ...] = ()
    cohort: Mapping[str, Any] = field(default_factory=dict)
    summary: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_nonempty(self.sweep_id, field="sweep_id")
        if not self.run_ids:
            raise ValueError("sweep run_ids are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sweep_id": self.sweep_id,
            "run_ids": list(self.run_ids),
            "axes": _axes_to_dict(self.axes),
            "aggregate_outcomes": [
                aggregate.to_dict() for aggregate in self.aggregate_outcomes
            ],
            "controls": [jsonable(control) for control in self.controls],
            "cohort": jsonable(self.cohort),
            "summary": jsonable(self.summary),
            "provenance": jsonable(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InterventionSweep":
        data = required_mapping(payload, field="InterventionSweep")
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            sweep_id=str(data["sweep_id"]),
            run_ids=tuple_from(data.get("run_ids"), cast=str, field="run_ids"),
            axes=_axes_from_payload(data.get("axes")),
            aggregate_outcomes=tuple(
                AggregateOutcomeResult.from_dict(
                    required_mapping(item, field="aggregate outcome")
                )
                for item in data.get("aggregate_outcomes", ())
            ),
            controls=tuple_of_mappings(data.get("controls"), field="controls"),
            cohort=mapping_from(data.get("cohort"), field="cohort"),
            summary=mapping_from(data.get("summary"), field="summary"),
            provenance=mapping_from(data.get("provenance"), field="provenance"),
        )


@dataclass(frozen=True, slots=True)
class InterventionStudy:
    """Typed shell for sweeps and controls over a cohort."""

    study_id: str
    sweep_ids: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()
    request_ids: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION
    cohort: Mapping[str, Any] = field(default_factory=dict)
    controls: tuple[Mapping[str, Any], ...] = ()
    aggregate_outcomes: tuple[AggregateOutcomeResult, ...] = ()
    summary: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_nonempty(self.study_id, field="study_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "sweep_ids": list(self.sweep_ids),
            "run_ids": list(self.run_ids),
            "request_ids": list(self.request_ids),
            "cohort": jsonable(self.cohort),
            "controls": [jsonable(control) for control in self.controls],
            "aggregate_outcomes": [
                aggregate.to_dict() for aggregate in self.aggregate_outcomes
            ],
            "summary": jsonable(self.summary),
            "provenance": jsonable(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InterventionStudy":
        data = required_mapping(payload, field="InterventionStudy")
        return cls(
            schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
            study_id=str(data["study_id"]),
            sweep_ids=tuple_from(data.get("sweep_ids"), cast=str, field="sweep_ids"),
            run_ids=tuple_from(data.get("run_ids"), cast=str, field="run_ids"),
            request_ids=tuple_from(
                data.get("request_ids"),
                cast=str,
                field="request_ids",
            ),
            cohort=mapping_from(data.get("cohort"), field="cohort"),
            controls=tuple_of_mappings(data.get("controls"), field="controls"),
            aggregate_outcomes=tuple(
                AggregateOutcomeResult.from_dict(
                    required_mapping(item, field="aggregate outcome")
                )
                for item in data.get("aggregate_outcomes", ())
            ),
            summary=mapping_from(data.get("summary"), field="summary"),
            provenance=mapping_from(data.get("provenance"), field="provenance"),
        )


def _axes_to_dict(axes: tuple[SweepAxis, ...] | Mapping[str, Any]) -> Any:
    if isinstance(axes, Mapping):
        return jsonable(axes)
    return [axis.to_dict() for axis in axes]


def _axes_from_payload(value: Any) -> tuple[SweepAxis, ...] | dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return mapping_from(value, field="axes")
    return tuple(
        SweepAxis.from_dict(required_mapping(item, field="axis"))
        for item in value or ()
    )
