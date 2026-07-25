"""Runtime-free contracts and measurements for counterfactual patch studies.

The hardware runtime writes these records, but this module intentionally depends
only on NumPy and the lightweight intervention contracts. That keeps planning,
evaluation, artifact inspection, and the dashboard in the normal repo environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

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
from vla_lens.interventions.specs import DonorSpec, RecipientSpec

COUNTERFACTUAL_SCHEMA_VERSION = "0.1.0"
RECIPE_KINDS = {"pose_exchange", "object_swap", "appearance_swap", "manual"}
ACTION_ROLES = {"recipient", "donor", "patched", "noop", "control"}
PATCH_TRIAL_KINDS = {"recipient", "donor", "patched", "noop", "control"}
PATCH_TRIAL_STATUSES = {"planned", "ok", "partial", "failed", "skipped"}
PATCH_VERDICTS = {
    "insufficient_data",
    "pair_invalid",
    "replay_invalid",
    "hook_invalid",
    "natural_effect_absent",
    "nonspecific",
    "localized_transfer",
    "specific_action_transfer",
    "confirmation_failed",
    "confirmation_passed",
}


@dataclass(frozen=True, slots=True)
class CounterfactualRecipe:
    """The intended scene change and everything that should remain fixed."""

    kind: str
    target_object: str
    distractor_object: str | None = None
    changed_variables: tuple[str, ...] = ()
    held_fixed: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_literal(self.kind, RECIPE_KINDS, field="counterfactual recipe kind")
        require_nonempty(self.target_object, field="target_object")
        if not self.changed_variables:
            raise ValueError("counterfactual recipe must declare changed_variables")
        if not self.held_fixed:
            raise ValueError("counterfactual recipe must declare held_fixed variables")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target_object": self.target_object,
            "distractor_object": self.distractor_object,
            "changed_variables": list(self.changed_variables),
            "held_fixed": jsonable(self.held_fixed),
            "parameters": jsonable(self.parameters),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CounterfactualRecipe":
        data = required_mapping(payload, field="CounterfactualRecipe")
        return cls(
            kind=str(data["kind"]),
            target_object=str(data["target_object"]),
            distractor_object=optional_str(data.get("distractor_object")),
            changed_variables=tuple_from(
                data.get("changed_variables"), cast=str, field="changed_variables"
            ),
            held_fixed=mapping_from(data.get("held_fixed"), field="held_fixed"),
            parameters=mapping_from(data.get("parameters"), field="parameters"),
        )


@dataclass(frozen=True, slots=True)
class CounterfactualPairManifest:
    """One recipient/donor pair plus the checks that make it interpretable."""

    pair_id: str
    recipe: CounterfactualRecipe
    recipient: RecipientSpec
    donor: DonorSpec
    compatibility: Mapping[str, Any] = field(default_factory=dict)
    validation: Mapping[str, Any] = field(default_factory=dict)
    media: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_nonempty(self.pair_id, field="pair_id")
        if self.donor.trace is None and self.donor.activation_ref is None:
            raise ValueError("counterfactual donor must reference a trace or activation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "recipe": self.recipe.to_dict(),
            "recipient": self.recipient.to_dict(),
            "donor": self.donor.to_dict(),
            "compatibility": jsonable(self.compatibility),
            "validation": jsonable(self.validation),
            "media": jsonable(self.media),
            "provenance": jsonable(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CounterfactualPairManifest":
        data = required_mapping(payload, field="CounterfactualPairManifest")
        return cls(
            pair_id=str(data["pair_id"]),
            recipe=CounterfactualRecipe.from_dict(
                required_mapping(data.get("recipe"), field="recipe")
            ),
            recipient=RecipientSpec.from_dict(
                required_mapping(data.get("recipient"), field="recipient")
            ),
            donor=DonorSpec.from_dict(required_mapping(data.get("donor"), field="donor")),
            compatibility=mapping_from(
                data.get("compatibility"), field="compatibility"
            ),
            validation=mapping_from(data.get("validation"), field="validation"),
            media=mapping_from(data.get("media"), field="media"),
            provenance=mapping_from(data.get("provenance"), field="provenance"),
        )


@dataclass(frozen=True, slots=True)
class ActionArrayRef:
    """A reconstructable action chunk with named axes rather than a flat vector."""

    array_ref: str
    role: str
    shape: tuple[int, int]
    dims: tuple[str, str] = ("action_horizon", "action_dim")
    dtype: str = "float32"
    sha256: str | None = None
    coordinates: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_nonempty(self.array_ref, field="action array_ref")
        require_literal(self.role, ACTION_ROLES, field="action role")
        if len(self.shape) != 2 or any(size <= 0 for size in self.shape):
            raise ValueError(
                "action arrays must have positive [action_horizon, action_dim] shape"
            )
        if self.dims != ("action_horizon", "action_dim"):
            raise ValueError("action array dims must be action_horizon, action_dim")

    def to_dict(self) -> dict[str, Any]:
        return {
            "array_ref": self.array_ref,
            "role": self.role,
            "shape": list(self.shape),
            "dims": list(self.dims),
            "dtype": self.dtype,
            "sha256": self.sha256,
            "coordinates": jsonable(self.coordinates),
            "metadata": jsonable(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionArrayRef":
        data = required_mapping(payload, field="ActionArrayRef")
        raw_shape = tuple_from(data.get("shape"), cast=int, field="shape")
        raw_dims = tuple_from(
            data.get("dims", ("action_horizon", "action_dim")),
            cast=str,
            field="dims",
        )
        if len(raw_shape) != 2 or len(raw_dims) != 2:
            raise ValueError("action arrays require two shape values and two dims")
        return cls(
            array_ref=str(data["array_ref"]),
            role=str(data["role"]),
            shape=(raw_shape[0], raw_shape[1]),
            dims=(raw_dims[0], raw_dims[1]),
            dtype=str(data.get("dtype", "float32")),
            sha256=optional_str(data.get("sha256")),
            coordinates=mapping_from(data.get("coordinates"), field="coordinates"),
            metadata=mapping_from(data.get("metadata"), field="metadata"),
        )


@dataclass(frozen=True, slots=True)
class PatchTrialManifest:
    """The complete recipe and result address for one patching trial."""

    trial_id: str
    pair_id: str
    trial_kind: str
    action: ActionArrayRef
    noise_ref: str
    target: Mapping[str, Any] = field(default_factory=dict)
    operation: Mapping[str, Any] = field(default_factory=dict)
    control_kind: str | None = None
    token_indices: tuple[int, ...] = ()
    token_mapping_ref: str | None = None
    token_mapping_sha256: str | None = None
    hook_calls: int | None = None
    status: str = "ok"
    metrics: Mapping[str, Any] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_nonempty(self.trial_id, field="patch trial_id")
        require_nonempty(self.pair_id, field="patch pair_id")
        require_nonempty(self.noise_ref, field="patch noise_ref")
        require_literal(self.trial_kind, PATCH_TRIAL_KINDS, field="patch trial_kind")
        require_literal(self.status, PATCH_TRIAL_STATUSES, field="patch trial status")
        if self.hook_calls is not None and self.hook_calls < 0:
            raise ValueError("hook_calls must be non-negative")
        if any(index < 0 for index in self.token_indices):
            raise ValueError("token_indices must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "pair_id": self.pair_id,
            "trial_kind": self.trial_kind,
            "action": self.action.to_dict(),
            "noise_ref": self.noise_ref,
            "target": jsonable(self.target),
            "operation": jsonable(self.operation),
            "control_kind": self.control_kind,
            "token_indices": list(self.token_indices),
            "token_mapping_ref": self.token_mapping_ref,
            "token_mapping_sha256": self.token_mapping_sha256,
            "hook_calls": self.hook_calls,
            "status": self.status,
            "metrics": jsonable(self.metrics),
            "errors": list(self.errors),
            "provenance": jsonable(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PatchTrialManifest":
        data = required_mapping(payload, field="PatchTrialManifest")
        return cls(
            trial_id=str(data["trial_id"]),
            pair_id=str(data["pair_id"]),
            trial_kind=str(data["trial_kind"]),
            action=ActionArrayRef.from_dict(
                required_mapping(data.get("action"), field="action")
            ),
            noise_ref=str(data["noise_ref"]),
            target=mapping_from(data.get("target"), field="target"),
            operation=mapping_from(data.get("operation"), field="operation"),
            control_kind=optional_str(data.get("control_kind")),
            token_indices=tuple_from(
                data.get("token_indices"), cast=int, field="token_indices"
            ),
            token_mapping_ref=optional_str(data.get("token_mapping_ref")),
            token_mapping_sha256=optional_str(data.get("token_mapping_sha256")),
            hook_calls=optional_int(data.get("hook_calls")),
            status=str(data.get("status", "ok")),
            metrics=mapping_from(data.get("metrics"), field="metrics"),
            errors=tuple_from(data.get("errors"), cast=str, field="errors"),
            provenance=mapping_from(data.get("provenance"), field="provenance"),
        )


@dataclass(frozen=True, slots=True)
class PatchDecisionThresholds:
    """Predeclared gates for turning action movement into a scientific verdict."""

    minimum_natural_delta_norm: float = 1e-6
    minimum_direction_agreement: float = 0.5
    minimum_transfer_fraction: float = 0.1
    maximum_donor_gap_remaining: float = 0.95
    minimum_control_margin: float = 0.05

    def __post_init__(self) -> None:
        if self.minimum_natural_delta_norm < 0:
            raise ValueError("minimum_natural_delta_norm must be non-negative")
        if not -1 <= self.minimum_direction_agreement <= 1:
            raise ValueError("minimum_direction_agreement must be between -1 and 1")
        if self.maximum_donor_gap_remaining < 0:
            raise ValueError("maximum_donor_gap_remaining must be non-negative")
        if self.minimum_control_margin < 0:
            raise ValueError("minimum_control_margin must be non-negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "minimum_natural_delta_norm": self.minimum_natural_delta_norm,
            "minimum_direction_agreement": self.minimum_direction_agreement,
            "minimum_transfer_fraction": self.minimum_transfer_fraction,
            "maximum_donor_gap_remaining": self.maximum_donor_gap_remaining,
            "minimum_control_margin": self.minimum_control_margin,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PatchDecisionThresholds":
        data = required_mapping(payload, field="PatchDecisionThresholds")
        return cls(
            minimum_natural_delta_norm=float(
                data.get("minimum_natural_delta_norm", 1e-6)
            ),
            minimum_direction_agreement=float(
                data.get("minimum_direction_agreement", 0.5)
            ),
            minimum_transfer_fraction=float(data.get("minimum_transfer_fraction", 0.1)),
            maximum_donor_gap_remaining=float(
                data.get("maximum_donor_gap_remaining", 0.95)
            ),
            minimum_control_margin=float(data.get("minimum_control_margin", 0.05)),
        )


@dataclass(frozen=True, slots=True)
class PatchStudySpec:
    """A reconstructable scientific question and its planned comparisons."""

    study_id: str
    question: str
    hypothesis: str
    pair_ids: tuple[str, ...]
    sites: tuple[Mapping[str, Any], ...]
    controls: tuple[str, ...]
    shared_noise_refs: tuple[str, ...]
    thresholds: PatchDecisionThresholds = field(default_factory=PatchDecisionThresholds)
    axes: Mapping[str, Any] = field(default_factory=dict)
    confounds: tuple[str, ...] = ()
    stopping_rule: str = ""
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_nonempty(self.study_id, field="patch study_id")
        require_nonempty(self.question, field="patch study question")
        require_nonempty(self.hypothesis, field="patch study hypothesis")
        if not self.pair_ids:
            raise ValueError("patch study requires pair_ids")
        if not self.sites:
            raise ValueError("patch study requires sites")
        if not self.shared_noise_refs:
            raise ValueError("patch study requires shared_noise_refs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "question": self.question,
            "hypothesis": self.hypothesis,
            "pair_ids": list(self.pair_ids),
            "sites": [jsonable(site) for site in self.sites],
            "controls": list(self.controls),
            "shared_noise_refs": list(self.shared_noise_refs),
            "thresholds": self.thresholds.to_dict(),
            "axes": jsonable(self.axes),
            "confounds": list(self.confounds),
            "stopping_rule": self.stopping_rule,
            "provenance": jsonable(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PatchStudySpec":
        data = required_mapping(payload, field="PatchStudySpec")
        return cls(
            study_id=str(data["study_id"]),
            question=str(data["question"]),
            hypothesis=str(data["hypothesis"]),
            pair_ids=tuple_from(data.get("pair_ids"), cast=str, field="pair_ids"),
            sites=tuple_of_mappings(data.get("sites"), field="sites"),
            controls=tuple_from(data.get("controls"), cast=str, field="controls"),
            shared_noise_refs=tuple_from(
                data.get("shared_noise_refs"), cast=str, field="shared_noise_refs"
            ),
            thresholds=PatchDecisionThresholds.from_dict(
                required_mapping(data.get("thresholds", {}), field="thresholds")
            ),
            axes=mapping_from(data.get("axes"), field="axes"),
            confounds=tuple_from(data.get("confounds"), cast=str, field="confounds"),
            stopping_rule=str(data.get("stopping_rule", "")),
            provenance=mapping_from(data.get("provenance"), field="provenance"),
        )


@dataclass(frozen=True, slots=True)
class CounterfactualMetrics:
    """How far a patched action moved along and away from the natural change."""

    natural_delta_norm: float
    patch_delta_norm: float
    recipient_to_donor_norm: float
    patched_to_donor_norm: float
    direction_agreement: float | None
    transfer_fraction: float | None
    donor_gap_remaining: float | None
    donor_recovery: float | None
    off_direction_norm: float | None
    off_direction_fraction: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "natural_delta_norm": self.natural_delta_norm,
            "patch_delta_norm": self.patch_delta_norm,
            "recipient_to_donor_norm": self.recipient_to_donor_norm,
            "patched_to_donor_norm": self.patched_to_donor_norm,
            "direction_agreement": self.direction_agreement,
            "transfer_fraction": self.transfer_fraction,
            "donor_gap_remaining": self.donor_gap_remaining,
            "donor_recovery": self.donor_recovery,
            "off_direction_norm": self.off_direction_norm,
            "off_direction_fraction": self.off_direction_fraction,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CounterfactualMetrics":
        data = required_mapping(payload, field="CounterfactualMetrics")
        return cls(
            natural_delta_norm=float(data["natural_delta_norm"]),
            patch_delta_norm=float(data["patch_delta_norm"]),
            recipient_to_donor_norm=float(data["recipient_to_donor_norm"]),
            patched_to_donor_norm=float(data["patched_to_donor_norm"]),
            direction_agreement=optional_float(data.get("direction_agreement")),
            transfer_fraction=optional_float(data.get("transfer_fraction")),
            donor_gap_remaining=optional_float(data.get("donor_gap_remaining")),
            donor_recovery=optional_float(data.get("donor_recovery")),
            off_direction_norm=optional_float(data.get("off_direction_norm")),
            off_direction_fraction=optional_float(data.get("off_direction_fraction")),
        )


@dataclass(frozen=True, slots=True)
class EvaluationDecision:
    """A plain machine verdict with the gates and evidence that produced it."""

    verdict: str
    summary: str
    supports_specificity: bool = False
    passed_gates: tuple[str, ...] = ()
    failed_gates: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_literal(self.verdict, PATCH_VERDICTS, field="patch verdict")
        require_nonempty(self.summary, field="patch verdict summary")

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "summary": self.summary,
            "supports_specificity": self.supports_specificity,
            "passed_gates": list(self.passed_gates),
            "failed_gates": list(self.failed_gates),
            "metrics": jsonable(self.metrics),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvaluationDecision":
        data = required_mapping(payload, field="EvaluationDecision")
        return cls(
            verdict=str(data["verdict"]),
            summary=str(data["summary"]),
            supports_specificity=bool(data.get("supports_specificity", False)),
            passed_gates=tuple_from(
                data.get("passed_gates"), cast=str, field="passed_gates"
            ),
            failed_gates=tuple_from(
                data.get("failed_gates"), cast=str, field="failed_gates"
            ),
            metrics=mapping_from(data.get("metrics"), field="metrics"),
        )


@dataclass(frozen=True, slots=True)
class PatchStudyArtifact:
    """Permanent study record; full activation caches are explicitly disposable."""

    study: PatchStudySpec
    pairs: tuple[CounterfactualPairManifest, ...]
    trials: tuple[PatchTrialManifest, ...]
    action_arrays: tuple[ActionArrayRef, ...]
    decisions: tuple[EvaluationDecision, ...]
    permanent_outputs: tuple[str, ...]
    disposable_cache_refs: tuple[str, ...] = ()
    schema_version: str = COUNTERFACTUAL_SCHEMA_VERSION
    created_utc: str = field(default_factory=utc_now_iso)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pair_ids = {pair.pair_id for pair in self.pairs}
        if pair_ids != set(self.study.pair_ids):
            raise ValueError("artifact pairs must exactly match study pair_ids")
        if any(trial.pair_id not in pair_ids for trial in self.trials):
            raise ValueError("every trial must reference an artifact pair")
        if not self.permanent_outputs:
            raise ValueError("patch study artifact requires permanent_outputs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_utc": self.created_utc,
            "study": self.study.to_dict(),
            "pairs": [pair.to_dict() for pair in self.pairs],
            "trials": [trial.to_dict() for trial in self.trials],
            "action_arrays": [array.to_dict() for array in self.action_arrays],
            "decisions": [decision.to_dict() for decision in self.decisions],
            "permanent_outputs": list(self.permanent_outputs),
            "disposable_cache_refs": list(self.disposable_cache_refs),
            "provenance": jsonable(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PatchStudyArtifact":
        data = required_mapping(payload, field="PatchStudyArtifact")
        return cls(
            study=PatchStudySpec.from_dict(
                required_mapping(data.get("study"), field="study")
            ),
            pairs=tuple(
                CounterfactualPairManifest.from_dict(
                    required_mapping(item, field="pair")
                )
                for item in data.get("pairs", ())
            ),
            trials=tuple(
                PatchTrialManifest.from_dict(required_mapping(item, field="trial"))
                for item in data.get("trials", ())
            ),
            action_arrays=tuple(
                ActionArrayRef.from_dict(required_mapping(item, field="action_array"))
                for item in data.get("action_arrays", ())
            ),
            decisions=tuple(
                EvaluationDecision.from_dict(required_mapping(item, field="decision"))
                for item in data.get("decisions", ())
            ),
            permanent_outputs=tuple_from(
                data.get("permanent_outputs"), cast=str, field="permanent_outputs"
            ),
            disposable_cache_refs=tuple_from(
                data.get("disposable_cache_refs"),
                cast=str,
                field="disposable_cache_refs",
            ),
            schema_version=str(data.get("schema_version", COUNTERFACTUAL_SCHEMA_VERSION)),
            created_utc=str(data.get("created_utc", utc_now_iso())),
            provenance=mapping_from(data.get("provenance"), field="provenance"),
        )


def counterfactual_action_metrics(
    recipient: Any,
    donor: Any,
    patched: Any,
    *,
    epsilon: float = 1e-12,
) -> CounterfactualMetrics:
    """Compare a patched action with the natural recipient-to-donor change."""
    recipient_array = _action_matrix(recipient, name="recipient")
    donor_array = _action_matrix(donor, name="donor")
    patched_array = _action_matrix(patched, name="patched")
    if recipient_array.shape != donor_array.shape or recipient_array.shape != patched_array.shape:
        raise ValueError("recipient, donor, and patched action chunks must have matching shapes")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    natural_delta = (donor_array - recipient_array).reshape(-1)
    patch_delta = (patched_array - recipient_array).reshape(-1)
    natural_norm = float(np.linalg.norm(natural_delta))
    patch_norm = float(np.linalg.norm(patch_delta))
    patched_to_donor = float(np.linalg.norm((patched_array - donor_array).reshape(-1)))

    if natural_norm <= epsilon:
        direction_agreement = None
        transfer_fraction = None
        donor_gap_remaining = None
        donor_recovery = None
        off_direction_norm = None
        off_direction_fraction = None
    else:
        transfer_fraction = float(np.dot(patch_delta, natural_delta) / (natural_norm**2))
        donor_gap_remaining = patched_to_donor / natural_norm
        donor_recovery = 1.0 - donor_gap_remaining
        off_direction = patch_delta - transfer_fraction * natural_delta
        off_direction_norm = float(np.linalg.norm(off_direction))
        off_direction_fraction = off_direction_norm / natural_norm
        direction_agreement = (
            float(np.dot(patch_delta, natural_delta) / (patch_norm * natural_norm))
            if patch_norm > epsilon
            else None
        )

    return CounterfactualMetrics(
        natural_delta_norm=natural_norm,
        patch_delta_norm=patch_norm,
        recipient_to_donor_norm=natural_norm,
        patched_to_donor_norm=patched_to_donor,
        direction_agreement=direction_agreement,
        transfer_fraction=transfer_fraction,
        donor_gap_remaining=donor_gap_remaining,
        donor_recovery=donor_recovery,
        off_direction_norm=off_direction_norm,
        off_direction_fraction=off_direction_fraction,
    )


def evaluate_patch_trial(
    metrics: CounterfactualMetrics,
    *,
    pair_valid: bool = True,
    replay_valid: bool = True,
    hook_valid: bool = True,
    controls: Sequence[CounterfactualMetrics] = (),
    thresholds: PatchDecisionThresholds | None = None,
    confirmation: bool = False,
) -> EvaluationDecision:
    """Apply predeclared gates without treating control execution as success."""
    gates = thresholds or PatchDecisionThresholds()
    metric_payload = metrics.to_dict()
    if not pair_valid:
        return _decision("pair_invalid", "The recipient and donor pair failed validation.", metrics)
    if not replay_valid:
        return _decision(
            "replay_invalid",
            "The action replay was not stable enough to compare.",
            metrics,
        )
    if not hook_valid:
        return _decision(
            "hook_invalid",
            "The runtime hook did not fire exactly as planned.",
            metrics,
        )
    if metrics.natural_delta_norm < gates.minimum_natural_delta_norm:
        return _decision(
            "natural_effect_absent",
            "Recipient and donor actions were too similar to define a transfer direction.",
            metrics,
        )
    if (
        metrics.direction_agreement is None
        or metrics.transfer_fraction is None
        or metrics.donor_gap_remaining is None
    ):
        return _decision(
            "insufficient_data",
            "The saved actions do not support all required transfer measurements.",
            metrics,
        )

    transfer_gates = {
        "direction_agreement": metrics.direction_agreement
        >= gates.minimum_direction_agreement,
        "transfer_fraction": metrics.transfer_fraction >= gates.minimum_transfer_fraction,
        "donor_gap": metrics.donor_gap_remaining <= gates.maximum_donor_gap_remaining,
    }
    passed = tuple(name for name, ok in transfer_gates.items() if ok)
    failed = tuple(name for name, ok in transfer_gates.items() if not ok)
    if failed:
        verdict = "confirmation_failed" if confirmation else "nonspecific"
        return EvaluationDecision(
            verdict=verdict,
            summary="The patch did not move the action toward the donor strongly enough.",
            passed_gates=passed,
            failed_gates=failed,
            metrics=metric_payload,
        )

    if not controls:
        return EvaluationDecision(
            verdict="localized_transfer",
            summary="The patch moved the action toward the donor; specificity controls remain.",
            passed_gates=passed,
            metrics=metric_payload,
        )

    control_transfer = [
        item.transfer_fraction
        for item in controls
        if item.transfer_fraction is not None
    ]
    if len(control_transfer) != len(controls):
        return EvaluationDecision(
            verdict="insufficient_data",
            summary="At least one specificity control lacks a measurable transfer fraction.",
            passed_gates=passed,
            failed_gates=("control_measurements",),
            metrics={**metric_payload, "control_transfer_fractions": control_transfer},
        )
    strongest_control = max(control_transfer, default=float("-inf"))
    specificity_margin = metrics.transfer_fraction - strongest_control
    metric_payload = {
        **metric_payload,
        "control_transfer_fractions": control_transfer,
        "specificity_margin": specificity_margin,
    }
    if specificity_margin < gates.minimum_control_margin:
        verdict = "confirmation_failed" if confirmation else "nonspecific"
        return EvaluationDecision(
            verdict=verdict,
            summary="A control transferred nearly as much action change as the intended patch.",
            passed_gates=passed,
            failed_gates=("control_margin",),
            metrics=metric_payload,
        )

    verdict = "confirmation_passed" if confirmation else "specific_action_transfer"
    return EvaluationDecision(
        verdict=verdict,
        summary="The intended patch transferred donor-directed action change beyond controls.",
        supports_specificity=True,
        passed_gates=(*passed, "control_margin"),
        metrics=metric_payload,
    )


def _action_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} action must have [action_horizon, action_dim] shape")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} action contains non-finite values")
    return array


def _decision(
    verdict: str,
    summary: str,
    metrics: CounterfactualMetrics,
) -> EvaluationDecision:
    return EvaluationDecision(
        verdict=verdict,
        summary=summary,
        failed_gates=(verdict,),
        metrics=metrics.to_dict(),
    )
