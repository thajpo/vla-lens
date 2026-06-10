"""Probe evidence contracts for capability-gated research UI surfaces.

This module is the v1 narrow waist from ``docs/probe_evidence_contract_phased.md``.
It deliberately models probe evidence only. Generic lens families are extension
seams, not implemented branches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal, Mapping, Sequence, TypeAlias, TypeVar

LensCapability: TypeAlias = Literal[
    "score_series",
    "ranked_moments",
    "thresholding",
    "prediction",
    "uncertainty",
    "contribution_breakdown",
    "model_locus_view",
    "visual_heatmap",
    "cohort_summary",
    "failure_cases",
    "comparison",
]

T = TypeVar("T")

TemporalScope: TypeAlias = Literal[
    "episode",
    "timestep",
    "window",
    "event",
    "token",
    "frame",
    "policy_call",
]

OutputKind: TypeAlias = Literal[
    "scalar",
    "class_label",
    "class_distribution",
    "vector",
    "ranked_features",
    "attribution_map",
    "heatmap",
]

InputBasis: TypeAlias = Literal[
    "layer_activation",
    "pooled_layer_activation",
    "sae_feature",
    "attention_head_output",
    "token_state",
    "image_patch",
    "action_state",
    "custom",
]

LocusKind: TypeAlias = Literal[
    "none",
    "model_locus",
    "visual_locus",
    "action_locus",
    "token_locus",
    "mixed_locus",
]

EvidenceClaimLevel: TypeAlias = Literal[
    "numeric_only",
    "grouped_model_locus",
    "human_labeled_feature",
    "semantic_hypothesis",
]

EvidencePrimitiveKind: TypeAlias = Literal[
    "provenance",
    "score_series",
    "ranked_moments",
    "prediction",
    "contribution",
    "model_locus",
    "cohort_summary",
    "failure_case",
]

UnavailableReasonCode: TypeAlias = Literal[
    "missing_scores",
    "missing_labels",
    "missing_contribution_basis",
    "pooled_representation",
    "missing_model_locus",
    "unsupported_probe_type",
    "not_computed",
]

RankingKind: TypeAlias = Literal[
    "top",
    "bottom",
    "uncertain",
    "false_positive",
    "false_negative",
    "largest_delta",
]

TIME_AXES = {"timestep", "frame", "token", "window", "policy_call"}
CAPABILITIES = set(LensCapability.__args__)  # type: ignore[attr-defined]
TEMPORAL_SCOPES = set(TemporalScope.__args__)  # type: ignore[attr-defined]
OUTPUT_KINDS = set(OutputKind.__args__)  # type: ignore[attr-defined]
INPUT_BASES = set(InputBasis.__args__)  # type: ignore[attr-defined]
LOCUS_KINDS = set(LocusKind.__args__)  # type: ignore[attr-defined]
CLAIM_LEVELS = set(EvidenceClaimLevel.__args__)  # type: ignore[attr-defined]
PRIMITIVE_KINDS = set(EvidencePrimitiveKind.__args__)  # type: ignore[attr-defined]
UNAVAILABLE_REASONS = set(UnavailableReasonCode.__args__)  # type: ignore[attr-defined]
RANKING_KINDS = set(RankingKind.__args__)  # type: ignore[attr-defined]
CONTRIBUTION_BASES = {
    "raw_activation_dimension",
    "sae_feature",
    "attention_head_output",
    "token_state",
    "action_dimension",
    "custom",
}
CONTRIBUTION_SIGNS = {"positive", "negative"}
COHORT_SOURCES = {"ranking", "filter", "manual", "saved"}
FAILURE_RANKINGS = {"false_positive", "false_negative", "high_confidence_wrong"}
PREDICTION_SPLITS = {"train", "validation", "test", "missing"}
PRIMITIVE_CAPABILITIES: dict[str, str] = {
    "score_series": "score_series",
    "ranked_moments": "ranked_moments",
    "prediction": "prediction",
    "contribution": "contribution_breakdown",
    "model_locus": "model_locus_view",
    "cohort_summary": "cohort_summary",
    "failure_case": "failure_cases",
}


class ProbeEvidenceContractError(ValueError):
    """Raised when a probe evidence contract object violates the v1 schema."""


def _ensure_choice(field_name: str, value: str, allowed: set[str]) -> None:
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ProbeEvidenceContractError(f"{field_name}={value!r} must be one of: {choices}")


def _ensure_choices(field_name: str, values: Sequence[str], allowed: set[str]) -> None:
    for value in values:
        _ensure_choice(field_name, str(value), allowed)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ArrayRef:
    """Lazy reference to a score or tensor array."""

    uri: str
    format: str = "array_ref"
    shape: tuple[int, ...] = ()
    dtype: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class ModelLocusRef:
    """Stable model location used by probe evidence and inspector panels."""

    model_site_id: str | None = None
    layer: int | None = None
    module: str | None = None
    stream: str | None = None
    head_index: int | None = None
    token_index: int | None = None
    channel_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class ProbeLensArtifact:
    """Probe-specific read model over the existing durable artifact record."""

    lens_id: str
    lens_version: str
    name: str
    lens_type: str = "probe"
    target: str | None = None
    source_model: Mapping[str, Any] = field(default_factory=dict)
    source: Mapping[str, Any] = field(default_factory=dict)
    training: Mapping[str, Any] = field(default_factory=dict)
    created_at: str | None = None

    def __post_init__(self) -> None:
        if self.lens_type != "probe":
            raise ProbeEvidenceContractError(
                "ProbeEvidenceBundle v1 only supports lens_type='probe'"
            )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class LensRun:
    """Application of one lens artifact to one dataset/result version."""

    lens_run_id: str
    lens_id: str
    lens_version: str
    dataset_id: str
    computed_at: str
    result_version: str
    status: Literal["complete", "partial", "failed"] = "complete"
    episode_ids: tuple[str, ...] = ()
    capture_profile_id: str | None = None
    evidence_bundle_id: str | None = None

    def __post_init__(self) -> None:
        _ensure_choice("status", self.status, {"complete", "partial", "failed"})

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class LensGeometry:
    """Geometry and capability declaration for a probe run."""

    temporal_scope: TemporalScope
    output_kind: OutputKind
    input_basis: InputBasis
    locus_kind: LocusKind
    capabilities: tuple[LensCapability, ...] = ()

    def __post_init__(self) -> None:
        _ensure_choice("temporal_scope", self.temporal_scope, TEMPORAL_SCOPES)
        _ensure_choice("output_kind", self.output_kind, OUTPUT_KINDS)
        _ensure_choice("input_basis", self.input_basis, INPUT_BASES)
        _ensure_choice("locus_kind", self.locus_kind, LOCUS_KINDS)
        _ensure_choices("capabilities", self.capabilities, CAPABILITIES)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class LensProvenanceEvidence:
    kind: Literal["provenance"]
    lens_id: str
    lens_run_id: str
    fields: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class ScoreSeriesEvidence:
    kind: Literal["score_series"]
    lens_id: str
    lens_run_id: str
    episode_id: str
    time_axis: Literal["timestep", "frame", "token", "window", "policy_call"]
    values_ref: ArrayRef
    summary: Mapping[str, float]
    threshold: float | None = None

    def __post_init__(self) -> None:
        _ensure_choice("time_axis", self.time_axis, TIME_AXES)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class RankedMoment:
    episode_id: str
    timestep: int | None = None
    policy_call: int | None = None
    frame_idx: int | None = None
    score: float | None = None
    prediction: str | bool | int | float | None = None
    label: str | bool | int | float | None = None
    confidence: float | None = None
    thumbnail_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class RankedMomentsEvidence:
    kind: Literal["ranked_moments"]
    lens_id: str
    lens_run_id: str
    ranking: RankingKind
    moments: tuple[RankedMoment, ...] = ()

    def __post_init__(self) -> None:
        _ensure_choice("ranking", self.ranking, RANKING_KINDS)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class PredictionEvidence:
    kind: Literal["prediction"]
    lens_id: str
    lens_run_id: str
    episode_id: str
    prediction: str | bool | int | float
    timestep: int | None = None
    policy_call: int | None = None
    label: str | bool | int | float | None = None
    confidence: float | None = None
    correct: bool | None = None
    split: Literal["train", "validation", "test", "missing"] | None = None

    def __post_init__(self) -> None:
        if self.split is not None:
            _ensure_choice("split", self.split, PREDICTION_SPLITS)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class ContributionItem:
    key: str
    value: float
    rank: int
    sign: Literal["positive", "negative"] | None = None
    model_locus: ModelLocusRef | None = None
    label: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if self.sign is not None:
            _ensure_choice("sign", self.sign, CONTRIBUTION_SIGNS)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class ContributionEvidence:
    kind: Literal["contribution"]
    lens_id: str
    lens_run_id: str
    episode_id: str
    basis: Literal[
        "raw_activation_dimension",
        "sae_feature",
        "attention_head_output",
        "token_state",
        "action_dimension",
        "custom",
    ]
    claim_level: EvidenceClaimLevel
    items: tuple[ContributionItem, ...] = ()
    timestep: int | None = None
    policy_call: int | None = None

    def __post_init__(self) -> None:
        _ensure_choice("basis", self.basis, CONTRIBUTION_BASES)
        _ensure_choice("claim_level", self.claim_level, CLAIM_LEVELS)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class ModelLocusEvidence:
    kind: Literal["model_locus"]
    lens_id: str
    lens_run_id: str
    locus: ModelLocusRef
    episode_id: str | None = None
    timestep: int | None = None
    policy_call: int | None = None
    source_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class ResearchSelectionState:
    """Shared research cursor for probe evidence surfaces."""

    dataset_id: str | None = None
    lens_id: str | None = None
    lens_run_id: str | None = None
    episode_id: str | None = None
    timestep: int | None = None
    policy_call: int | None = None
    time_window: Mapping[str, int] | None = None
    ranking: RankingKind | None = None
    cohort_id: str | None = None
    model_locus: ModelLocusRef | None = None
    feature_id: str | None = None

    def __post_init__(self) -> None:
        if self.ranking is not None:
            _ensure_choice("ranking", self.ranking, RANKING_KINDS)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class EvidenceCohortRef:
    cohort_id: str
    source: Literal["ranking", "filter", "manual", "saved"]
    selection: ResearchSelectionState
    count: int

    def __post_init__(self) -> None:
        _ensure_choice("source", self.source, COHORT_SOURCES)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class CohortSummaryEvidence:
    kind: Literal["cohort_summary"]
    lens_id: str
    lens_run_id: str
    cohort: EvidenceCohortRef
    summary: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class FailureCaseEvidence:
    kind: Literal["failure_case"]
    lens_id: str
    lens_run_id: str
    ranking: Literal["false_positive", "false_negative", "high_confidence_wrong"]
    moments: tuple[RankedMoment, ...] = ()

    def __post_init__(self) -> None:
        _ensure_choice("ranking", self.ranking, FAILURE_RANKINGS)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


EvidencePrimitive: TypeAlias = (
    LensProvenanceEvidence
    | ScoreSeriesEvidence
    | RankedMomentsEvidence
    | PredictionEvidence
    | ContributionEvidence
    | ModelLocusEvidence
    | CohortSummaryEvidence
    | FailureCaseEvidence
)


@dataclass(frozen=True, slots=True)
class UnavailableReason:
    capability: LensCapability
    reason: UnavailableReasonCode
    message: str
    panel_id: str | None = None

    def __post_init__(self) -> None:
        _ensure_choice("capability", self.capability, CAPABILITIES)
        _ensure_choice("reason", self.reason, UNAVAILABLE_REASONS)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class ProbeEvidenceBundle:
    bundle_id: str
    artifact: ProbeLensArtifact
    run: LensRun
    geometry: LensGeometry
    capabilities: tuple[LensCapability, ...]
    primitives: tuple[EvidencePrimitive, ...] = ()
    unavailable: tuple[UnavailableReason, ...] = ()
    family: Literal["probe"] = "probe"

    def __post_init__(self) -> None:
        if self.family != "probe":
            raise ProbeEvidenceContractError("ProbeEvidenceBundle v1 only supports family='probe'")
        if self.artifact.lens_id != self.run.lens_id:
            raise ProbeEvidenceContractError("artifact.lens_id must match run.lens_id")
        if self.artifact.lens_version != self.run.lens_version:
            raise ProbeEvidenceContractError("artifact.lens_version must match run.lens_version")
        _ensure_choices("capabilities", self.capabilities, CAPABILITIES)
        missing = set(self.capabilities) - set(self.geometry.capabilities)
        if missing:
            raise ProbeEvidenceContractError(
                "bundle capabilities must be declared by geometry: " + ", ".join(sorted(missing))
            )
        for primitive in self.primitives:
            self._validate_primitive(primitive)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    def _validate_primitive(self, primitive: EvidencePrimitive) -> None:
        if primitive.lens_id != self.artifact.lens_id:
            raise ProbeEvidenceContractError(
                f"{primitive.kind} primitive lens_id={primitive.lens_id!r} "
                f"does not match bundle lens_id={self.artifact.lens_id!r}"
            )
        if primitive.lens_run_id != self.run.lens_run_id:
            raise ProbeEvidenceContractError(
                f"{primitive.kind} primitive lens_run_id={primitive.lens_run_id!r} "
                f"does not match bundle lens_run_id={self.run.lens_run_id!r}"
            )
        required_capability = PRIMITIVE_CAPABILITIES.get(str(primitive.kind))
        if required_capability and required_capability not in self.capabilities:
            raise ProbeEvidenceContractError(
                f"{primitive.kind} primitive requires capability {required_capability!r}"
            )


@dataclass(frozen=True, slots=True)
class PanelSpec:
    panel_id: str
    consumes: tuple[EvidencePrimitiveKind, ...]
    requires_capabilities: tuple[LensCapability, ...] = ()
    requires_geometry: Mapping[str, str] = field(default_factory=dict)
    unavailable_copy: str = "Panel unavailable for this probe evidence bundle."

    def __post_init__(self) -> None:
        _ensure_choices("consumes", self.consumes, PRIMITIVE_KINDS)
        _ensure_choices("requires_capabilities", self.requires_capabilities, CAPABILITIES)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class PanelAvailability:
    panel_id: str
    available: bool
    reason: str | None = None
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class CurrentMomentEvidence:
    """Evidence primitives narrowed to the active research cursor."""

    selection: ResearchSelectionState
    score_series: tuple[ScoreSeriesEvidence, ...] = ()
    ranked_moments: tuple[RankedMoment, ...] = ()
    predictions: tuple[PredictionEvidence, ...] = ()
    contributions: tuple[ContributionEvidence, ...] = ()
    model_loci: tuple[ModelLocusEvidence, ...] = ()
    failure_moments: tuple[RankedMoment, ...] = ()
    unavailable: tuple[UnavailableReason, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class LensFeatureContribution:
    """Episode microscope row derived from contribution evidence."""

    key: str
    value: float
    rank: int
    claim_level: EvidenceClaimLevel
    basis: str
    sign: Literal["positive", "negative"] | None = None
    model_locus: ModelLocusRef | None = None
    label: str | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class PipelineLensAnnotation:
    """Model-pipeline annotation for the active probe lens."""

    annotation_id: str
    source: Literal["model_locus", "contribution"]
    label: str
    model_locus: ModelLocusRef
    episode_id: str | None = None
    timestep: int | None = None
    policy_call: int | None = None
    claim_level: EvidenceClaimLevel | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True, slots=True)
class LensTemporalRow:
    """Timeline marker derived from ranked moments, predictions, or failures."""

    row_id: str
    source: Literal["ranked_moment", "prediction", "failure_case"]
    episode_id: str
    timestep: int | None = None
    policy_call: int | None = None
    ranking: str | None = None
    score: float | None = None
    confidence: float | None = None
    prediction: str | bool | int | float | None = None
    label: str | bool | int | float | None = None

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


def primitive_kinds(bundle: ProbeEvidenceBundle) -> set[str]:
    return {str(primitive.kind) for primitive in bundle.primitives}


def primitives_by_kind(
    bundle: ProbeEvidenceBundle,
    kind: EvidencePrimitiveKind,
) -> tuple[EvidencePrimitive, ...]:
    _ensure_choice("kind", kind, PRIMITIVE_KINDS)
    return tuple(primitive for primitive in bundle.primitives if primitive.kind == kind)


def ranked_moments(
    bundle: ProbeEvidenceBundle,
    ranking: RankingKind,
) -> tuple[RankedMoment, ...]:
    _ensure_choice("ranking", ranking, RANKING_KINDS)
    for primitive in bundle.primitives:
        if isinstance(primitive, RankedMomentsEvidence) and primitive.ranking == ranking:
            return primitive.moments
    return ()


def select_top_moments(
    bundle: ProbeEvidenceBundle,
    ranking: RankingKind = "top",
    *,
    limit: int | None = None,
) -> tuple[RankedMoment, ...]:
    """Return ranked moments for one ranking, preserving adapter order."""
    return _limit_tuple(ranked_moments(bundle, ranking), limit)


def select_current_moment_evidence(
    bundle: ProbeEvidenceBundle,
    selection: ResearchSelectionState,
) -> CurrentMomentEvidence:
    """Narrow a bundle to evidence relevant to the active episode/time cursor."""
    ranked: list[RankedMoment] = []
    failure_moments: list[RankedMoment] = []
    for primitive in bundle.primitives:
        if isinstance(primitive, RankedMomentsEvidence):
            if selection.ranking is not None and primitive.ranking != selection.ranking:
                continue
            ranked.extend(
                moment
                for moment in primitive.moments
                if _moment_matches_selection(moment, selection)
            )
        elif isinstance(primitive, FailureCaseEvidence):
            failure_moments.extend(
                moment
                for moment in primitive.moments
                if _moment_matches_selection(moment, selection)
            )
    score_series = tuple(
        primitive
        for primitive in primitives_by_kind(bundle, "score_series")
        if isinstance(primitive, ScoreSeriesEvidence)
        and _evidence_matches_selection(primitive, selection)
    )
    predictions = tuple(
        primitive
        for primitive in primitives_by_kind(bundle, "prediction")
        if isinstance(primitive, PredictionEvidence)
        and _evidence_matches_selection(primitive, selection)
    )
    contributions = tuple(
        primitive
        for primitive in primitives_by_kind(bundle, "contribution")
        if isinstance(primitive, ContributionEvidence)
        and _evidence_matches_selection(primitive, selection)
    )
    model_loci = tuple(
        primitive
        for primitive in primitives_by_kind(bundle, "model_locus")
        if isinstance(primitive, ModelLocusEvidence)
        and _evidence_matches_selection(primitive, selection)
    )
    return CurrentMomentEvidence(
        selection=selection,
        score_series=score_series,
        ranked_moments=tuple(ranked),
        predictions=predictions,
        contributions=contributions,
        model_loci=model_loci,
        failure_moments=tuple(failure_moments),
        unavailable=select_unavailable_reasons(bundle),
    )


def select_contribution_rows(
    bundle: ProbeEvidenceBundle,
    selection: ResearchSelectionState | None = None,
    *,
    limit: int | None = None,
) -> tuple[ContributionItem, ...]:
    """Return contribution rows visible for the active cursor without inventing rows."""
    rows: list[ContributionItem] = []
    for primitive in primitives_by_kind(bundle, "contribution"):
        if not isinstance(primitive, ContributionEvidence):
            continue
        if selection is not None and not _evidence_matches_selection(primitive, selection):
            continue
        for item in primitive.items:
            if selection is not None and selection.feature_id is not None:
                if item.key != selection.feature_id:
                    continue
            rows.append(item)
    rows.sort(key=lambda item: item.rank)
    return _limit_tuple(tuple(rows), limit)


def select_contribution_claim_level(
    bundle: ProbeEvidenceBundle,
    selection: ResearchSelectionState | None = None,
) -> EvidenceClaimLevel | None:
    """Return the conservative claim level for visible contribution evidence."""
    levels: list[EvidenceClaimLevel] = []
    for primitive in primitives_by_kind(bundle, "contribution"):
        if not isinstance(primitive, ContributionEvidence):
            continue
        if selection is not None and not _evidence_matches_selection(primitive, selection):
            continue
        if selection is not None and selection.feature_id is not None:
            if not any(item.key == selection.feature_id for item in primitive.items):
                continue
        levels.append(primitive.claim_level)
    if not levels:
        return None
    return min(levels, key=lambda level: _CLAIM_LEVEL_STRENGTH[level])


def select_unavailable_reasons(
    bundle: ProbeEvidenceBundle,
    *,
    panel_id: str | None = None,
    capability: LensCapability | None = None,
) -> tuple[UnavailableReason, ...]:
    """Return explicit unavailable reasons, optionally narrowed by panel or capability."""
    if capability is not None:
        _ensure_choice("capability", capability, CAPABILITIES)
    return tuple(
        reason
        for reason in bundle.unavailable
        if (panel_id is None or reason.panel_id == panel_id)
        and (capability is None or reason.capability == capability)
    )


def select_available_panels(
    bundle: ProbeEvidenceBundle,
    panel_specs: Sequence[PanelSpec],
) -> tuple[PanelAvailability, ...]:
    present = primitive_kinds(bundle)
    bundle_capabilities = set(bundle.capabilities)
    availability: list[PanelAvailability] = []
    for spec in panel_specs:
        missing_capabilities = [
            capability
            for capability in spec.requires_capabilities
            if capability not in bundle_capabilities
        ]
        missing_primitives = [kind for kind in spec.consumes if kind not in present]
        geometry_error = _first_geometry_mismatch(bundle.geometry, spec.requires_geometry)
        if not missing_capabilities and not missing_primitives and geometry_error is None:
            availability.append(PanelAvailability(panel_id=spec.panel_id, available=True))
            continue
        reason = (
            f"missing capability: {missing_capabilities[0]}"
            if missing_capabilities
            else f"missing evidence primitive: {missing_primitives[0]}"
            if missing_primitives
            else geometry_error
        )
        unavailable = _matching_unavailable(bundle, spec, missing_capabilities)
        availability.append(
            PanelAvailability(
                panel_id=spec.panel_id,
                available=False,
                reason=unavailable.reason if unavailable else reason,
                message=unavailable.message if unavailable else spec.unavailable_copy,
            )
        )
    return tuple(availability)


def _first_geometry_mismatch(
    geometry: LensGeometry,
    required: Mapping[str, str],
) -> str | None:
    for key, expected in required.items():
        actual = getattr(geometry, key, None)
        if actual != expected:
            return f"geometry {key}={actual!r} does not match required {expected!r}"
    return None


def _matching_unavailable(
    bundle: ProbeEvidenceBundle,
    spec: PanelSpec,
    missing_capabilities: Sequence[str],
) -> UnavailableReason | None:
    for reason in bundle.unavailable:
        if reason.panel_id == spec.panel_id:
            return reason
    for reason in bundle.unavailable:
        if reason.capability in missing_capabilities:
            return reason
    return None


def default_probe_panel_specs() -> tuple[PanelSpec, ...]:
    return (
        PanelSpec(
            panel_id="probe_provenance",
            consumes=("provenance",),
            unavailable_copy="Probe provenance is unavailable for this run.",
        ),
        PanelSpec(
            panel_id="score_series",
            consumes=("score_series",),
            requires_capabilities=("score_series",),
            unavailable_copy="Score series is unavailable for this probe run.",
        ),
        PanelSpec(
            panel_id="ranked_moments",
            consumes=("ranked_moments",),
            requires_capabilities=("ranked_moments",),
            unavailable_copy="Ranked moments are unavailable for this probe run.",
        ),
        PanelSpec(
            panel_id="prediction",
            consumes=("prediction",),
            requires_capabilities=("prediction",),
            unavailable_copy="Predictions are unavailable for this probe run.",
        ),
        PanelSpec(
            panel_id="contribution",
            consumes=("contribution",),
            requires_capabilities=("contribution_breakdown",),
            unavailable_copy="Contribution breakdown is unavailable for this probe run.",
        ),
        PanelSpec(
            panel_id="model_locus",
            consumes=("model_locus",),
            requires_capabilities=("model_locus_view",),
            requires_geometry={"locus_kind": "model_locus"},
            unavailable_copy="Model locus is unavailable for this probe run.",
        ),
        PanelSpec(
            panel_id="failure_cases",
            consumes=("failure_case",),
            requires_capabilities=("failure_cases",),
            unavailable_copy="Failure cases are unavailable for this probe run.",
        ),
        PanelSpec(
            panel_id="unavailable_reasons",
            consumes=(),
            unavailable_copy="Unavailable reason explanations are unavailable.",
        ),
    )


@dataclass(frozen=True, slots=True)
class EpisodeLensAdapter:
    """Concrete v1 bridge from ProbeEvidenceBundle into the episode microscope."""

    family: Literal["probe"] = "probe"

    def default_selection(self, bundle: ProbeEvidenceBundle) -> ResearchSelectionState:
        ranked_default = _first_available_moment(bundle)
        base = {
            "dataset_id": bundle.run.dataset_id,
            "lens_id": bundle.artifact.lens_id,
            "lens_run_id": bundle.run.lens_run_id,
        }
        if ranked_default is not None:
            ranking, moment = ranked_default
            return ResearchSelectionState(
                **base,
                episode_id=moment.episode_id,
                timestep=moment.timestep,
                policy_call=moment.policy_call,
                ranking=ranking,
            )
        prediction = next(
            (
                primitive
                for primitive in primitives_by_kind(bundle, "prediction")
                if isinstance(primitive, PredictionEvidence)
            ),
            None,
        )
        if prediction is not None:
            return ResearchSelectionState(
                **base,
                episode_id=prediction.episode_id,
                timestep=prediction.timestep,
                policy_call=prediction.policy_call,
            )
        model_locus = next(
            (
                primitive
                for primitive in primitives_by_kind(bundle, "model_locus")
                if isinstance(primitive, ModelLocusEvidence)
            ),
            None,
        )
        if model_locus is not None:
            return ResearchSelectionState(
                **base,
                episode_id=model_locus.episode_id,
                timestep=model_locus.timestep,
                policy_call=model_locus.policy_call,
                model_locus=model_locus.locus,
            )
        return ResearchSelectionState(**base)

    def pipeline_annotations(
        self,
        bundle: ProbeEvidenceBundle,
        selection: ResearchSelectionState | None = None,
    ) -> tuple[PipelineLensAnnotation, ...]:
        out: list[PipelineLensAnnotation] = []
        for primitive in primitives_by_kind(bundle, "model_locus"):
            if not isinstance(primitive, ModelLocusEvidence):
                continue
            if selection is not None and not _evidence_matches_selection(primitive, selection):
                continue
            label = primitive.source_label or _model_locus_label(primitive.locus)
            out.append(
                PipelineLensAnnotation(
                    annotation_id=f"model_locus:{len(out)}",
                    source="model_locus",
                    label=label,
                    model_locus=primitive.locus,
                    episode_id=primitive.episode_id,
                    timestep=primitive.timestep,
                    policy_call=primitive.policy_call,
                )
            )
        for primitive in primitives_by_kind(bundle, "contribution"):
            if not isinstance(primitive, ContributionEvidence):
                continue
            if selection is not None and not _evidence_matches_selection(primitive, selection):
                continue
            for item in primitive.items:
                if item.model_locus is None:
                    continue
                if selection is not None and selection.feature_id is not None:
                    if item.key != selection.feature_id:
                        continue
                out.append(
                    PipelineLensAnnotation(
                        annotation_id=f"contribution:{item.key}",
                        source="contribution",
                        label=item.label or item.key,
                        model_locus=item.model_locus,
                        episode_id=primitive.episode_id,
                        timestep=primitive.timestep,
                        policy_call=primitive.policy_call,
                        claim_level=primitive.claim_level,
                    )
                )
        return tuple(out)

    def channel_ranking(
        self,
        bundle: ProbeEvidenceBundle,
        selection: ResearchSelectionState,
    ) -> tuple[LensFeatureContribution, ...]:
        rows: list[LensFeatureContribution] = []
        for primitive in primitives_by_kind(bundle, "contribution"):
            if not isinstance(primitive, ContributionEvidence):
                continue
            if not _evidence_matches_selection(primitive, selection):
                continue
            for item in primitive.items:
                if selection.feature_id is not None and item.key != selection.feature_id:
                    continue
                rows.append(
                    LensFeatureContribution(
                        key=item.key,
                        value=item.value,
                        rank=item.rank,
                        sign=item.sign,
                        model_locus=item.model_locus,
                        label=item.label,
                        description=item.description,
                        claim_level=primitive.claim_level,
                        basis=primitive.basis,
                    )
                )
        rows.sort(key=lambda item: item.rank)
        return tuple(rows)

    def timeline_rows(self, bundle: ProbeEvidenceBundle) -> tuple[LensTemporalRow, ...]:
        rows: list[LensTemporalRow] = []
        for primitive in bundle.primitives:
            if isinstance(primitive, RankedMomentsEvidence):
                for index, moment in enumerate(primitive.moments):
                    rows.append(
                        LensTemporalRow(
                            row_id=f"ranked:{primitive.ranking}:{index}",
                            source="ranked_moment",
                            episode_id=moment.episode_id,
                            timestep=moment.timestep,
                            policy_call=moment.policy_call,
                            ranking=primitive.ranking,
                            score=moment.score,
                            confidence=moment.confidence,
                            prediction=moment.prediction,
                            label=moment.label,
                        )
                    )
            elif isinstance(primitive, PredictionEvidence):
                rows.append(
                    LensTemporalRow(
                        row_id=f"prediction:{len(rows)}",
                        source="prediction",
                        episode_id=primitive.episode_id,
                        timestep=primitive.timestep,
                        policy_call=primitive.policy_call,
                        confidence=primitive.confidence,
                        prediction=primitive.prediction,
                        label=primitive.label,
                    )
                )
            elif isinstance(primitive, FailureCaseEvidence):
                for index, moment in enumerate(primitive.moments):
                    rows.append(
                        LensTemporalRow(
                            row_id=f"failure:{primitive.ranking}:{index}",
                            source="failure_case",
                            episode_id=moment.episode_id,
                            timestep=moment.timestep,
                            policy_call=moment.policy_call,
                            ranking=primitive.ranking,
                            score=moment.score,
                            confidence=moment.confidence,
                            prediction=moment.prediction,
                            label=moment.label,
                        )
                    )
        return tuple(rows)

    def intervention_seed(
        self,
        bundle: ProbeEvidenceBundle,
        selection: ResearchSelectionState,
    ) -> None:
        del bundle, selection
        return None


probe_episode_lens_adapter = EpisodeLensAdapter()


_CLAIM_LEVEL_STRENGTH: dict[EvidenceClaimLevel, int] = {
    "numeric_only": 0,
    "grouped_model_locus": 1,
    "human_labeled_feature": 2,
    "semantic_hypothesis": 3,
}


def _limit_tuple(items: tuple[T, ...], limit: int | None) -> tuple[T, ...]:
    if limit is None:
        return items
    if limit < 0:
        raise ProbeEvidenceContractError("limit must be non-negative")
    return items[:limit]


def _first_available_moment(bundle: ProbeEvidenceBundle) -> tuple[RankingKind, RankedMoment] | None:
    for ranking in ("top", "uncertain", "bottom"):
        moments = ranked_moments(bundle, ranking)
        if moments:
            return ranking, moments[0]
    return None


def _moment_matches_selection(moment: RankedMoment, selection: ResearchSelectionState) -> bool:
    if selection.episode_id is not None and moment.episode_id != selection.episode_id:
        return False
    return _time_matches_selection(
        selection,
        timestep=moment.timestep,
        policy_call=moment.policy_call,
    )


def _evidence_matches_selection(
    evidence: EvidencePrimitive,
    selection: ResearchSelectionState,
) -> bool:
    episode_id = getattr(evidence, "episode_id", None)
    if selection.episode_id is not None and episode_id is not None:
        if episode_id != selection.episode_id:
            return False
    return _time_matches_selection(
        selection,
        timestep=getattr(evidence, "timestep", None),
        policy_call=getattr(evidence, "policy_call", None),
    )


def _time_matches_selection(
    selection: ResearchSelectionState,
    *,
    timestep: int | None,
    policy_call: int | None,
) -> bool:
    if selection.timestep is not None and timestep is not None:
        if timestep != selection.timestep:
            return False
    if selection.policy_call is not None and policy_call is not None:
        if policy_call != selection.policy_call:
            return False
    return True


def _model_locus_label(locus: ModelLocusRef) -> str:
    if locus.model_site_id:
        return locus.model_site_id
    if locus.layer is not None:
        return f"layer {locus.layer}"
    return "model locus"


def scalar_timestep_probe_bundle() -> ProbeEvidenceBundle:
    artifact, run = _artifact_and_run("probe-target-contacted", "target contacted")
    geometry = LensGeometry(
        temporal_scope="timestep",
        output_kind="scalar",
        input_basis="pooled_layer_activation",
        locus_kind="none",
        capabilities=(
            "score_series",
            "ranked_moments",
            "thresholding",
            "prediction",
            "uncertainty",
        ),
    )
    primitives: tuple[EvidencePrimitive, ...] = (
        LensProvenanceEvidence(
            kind="provenance",
            lens_id=artifact.lens_id,
            lens_run_id=run.lens_run_id,
            fields={"Prediction": "Target contacted", "Input": "Pooled hidden states"},
        ),
        ScoreSeriesEvidence(
            kind="score_series",
            lens_id=artifact.lens_id,
            lens_run_id=run.lens_run_id,
            episode_id="episode-1",
            time_axis="timestep",
            values_ref=ArrayRef(uri="arrays/probe-target-contacted/scores.zarr", shape=(12,)),
            summary={"min": 0.1, "max": 0.91, "mean": 0.42},
            threshold=0.5,
        ),
        RankedMomentsEvidence(
            kind="ranked_moments",
            lens_id=artifact.lens_id,
            lens_run_id=run.lens_run_id,
            ranking="top",
            moments=(RankedMoment(episode_id="episode-1", timestep=7, score=0.91),),
        ),
        RankedMomentsEvidence(
            kind="ranked_moments",
            lens_id=artifact.lens_id,
            lens_run_id=run.lens_run_id,
            ranking="bottom",
            moments=(RankedMoment(episode_id="episode-2", timestep=1, score=0.1),),
        ),
        PredictionEvidence(
            kind="prediction",
            lens_id=artifact.lens_id,
            lens_run_id=run.lens_run_id,
            episode_id="episode-1",
            timestep=7,
            prediction=True,
            confidence=0.91,
            split="test",
        ),
    )
    unavailable = (
        UnavailableReason(
            capability="contribution_breakdown",
            panel_id="contribution",
            reason="missing_contribution_basis",
            message=(
                "Contribution breakdown unavailable because this probe exposes scores "
                "but not decomposed inputs."
            ),
        ),
        UnavailableReason(
            capability="model_locus_view",
            panel_id="model_locus",
            reason="pooled_representation",
            message="Model locus unavailable because this probe uses pooled layer activations.",
        ),
        UnavailableReason(
            capability="failure_cases",
            panel_id="failure_cases",
            reason="missing_labels",
            message="Failure cases unavailable because no labels or proxy targets exist.",
        ),
    )
    return ProbeEvidenceBundle(
        bundle_id="bundle-probe-target-contacted",
        artifact=artifact,
        run=run,
        geometry=geometry,
        capabilities=geometry.capabilities,
        primitives=primitives,
        unavailable=unavailable,
    )


def pooled_no_contribution_probe_bundle() -> ProbeEvidenceBundle:
    return scalar_timestep_probe_bundle()


def raw_layer_contribution_probe_bundle() -> ProbeEvidenceBundle:
    artifact, run = _artifact_and_run("probe-grasp-intent", "grasp intent")
    geometry = LensGeometry(
        temporal_scope="policy_call",
        output_kind="scalar",
        input_basis="layer_activation",
        locus_kind="model_locus",
        capabilities=(
            "score_series",
            "ranked_moments",
            "prediction",
            "contribution_breakdown",
            "model_locus_view",
        ),
    )
    locus = ModelLocusRef(model_site_id="action_head.layers.8.resid", layer=8, stream="residual")
    primitives: tuple[EvidencePrimitive, ...] = (
        LensProvenanceEvidence(
            kind="provenance",
            lens_id=artifact.lens_id,
            lens_run_id=run.lens_run_id,
            fields={"Prediction": "Grasp intent", "Input": "Layer 8 residual stream"},
        ),
        ScoreSeriesEvidence(
            kind="score_series",
            lens_id=artifact.lens_id,
            lens_run_id=run.lens_run_id,
            episode_id="episode-1",
            time_axis="policy_call",
            values_ref=ArrayRef(uri="arrays/probe-grasp-intent/scores.zarr", shape=(4,)),
            summary={"min": 0.2, "max": 0.88, "mean": 0.6},
        ),
        RankedMomentsEvidence(
            kind="ranked_moments",
            lens_id=artifact.lens_id,
            lens_run_id=run.lens_run_id,
            ranking="top",
            moments=(RankedMoment(episode_id="episode-1", policy_call=3, score=0.88),),
        ),
        PredictionEvidence(
            kind="prediction",
            lens_id=artifact.lens_id,
            lens_run_id=run.lens_run_id,
            episode_id="episode-1",
            policy_call=3,
            prediction=True,
            confidence=0.88,
            correct=True,
            split="validation",
        ),
        ModelLocusEvidence(
            kind="model_locus",
            lens_id=artifact.lens_id,
            lens_run_id=run.lens_run_id,
            episode_id="episode-1",
            policy_call=3,
            locus=locus,
            source_label="Layer 8 residual stream",
        ),
        ContributionEvidence(
            kind="contribution",
            lens_id=artifact.lens_id,
            lens_run_id=run.lens_run_id,
            episode_id="episode-1",
            policy_call=3,
            basis="raw_activation_dimension",
            claim_level="numeric_only",
            items=(
                ContributionItem(
                    key="dim_42",
                    rank=1,
                    value=0.42,
                    sign="positive",
                    model_locus=ModelLocusRef(**{**locus.to_dict(), "channel_index": 42}),
                ),
            ),
        ),
    )
    return ProbeEvidenceBundle(
        bundle_id="bundle-probe-grasp-intent",
        artifact=artifact,
        run=run,
        geometry=geometry,
        capabilities=geometry.capabilities,
        primitives=primitives,
    )


def sae_feature_contribution_probe_bundle() -> ProbeEvidenceBundle:
    return _contribution_variant(
        lens_id="probe-sae-target-contact",
        name="target contact SAE probe",
        input_basis="sae_feature",
        basis="sae_feature",
        claim_level="human_labeled_feature",
        key="sae_17",
        label="contact-like feature",
    )


def attention_head_grouped_probe_bundle() -> ProbeEvidenceBundle:
    return _contribution_variant(
        lens_id="probe-head-grasp-intent",
        name="attention head grasp intent probe",
        input_basis="attention_head_output",
        basis="attention_head_output",
        claim_level="grouped_model_locus",
        key="head_3",
        label="Layer 6 head 3",
    )


def _contribution_variant(
    *,
    lens_id: str,
    name: str,
    input_basis: InputBasis,
    basis: str,
    claim_level: EvidenceClaimLevel,
    key: str,
    label: str,
) -> ProbeEvidenceBundle:
    artifact, run = _artifact_and_run(lens_id, name)
    geometry = LensGeometry(
        temporal_scope="policy_call",
        output_kind="scalar",
        input_basis=input_basis,
        locus_kind="model_locus",
        capabilities=("contribution_breakdown", "model_locus_view"),
    )
    locus = ModelLocusRef(model_site_id="action_head.layers.6.attn", layer=6, stream="attention")
    primitives: tuple[EvidencePrimitive, ...] = (
        LensProvenanceEvidence(
            kind="provenance",
            lens_id=artifact.lens_id,
            lens_run_id=run.lens_run_id,
            fields={"Prediction": name, "Input": input_basis},
        ),
        ModelLocusEvidence(
            kind="model_locus",
            lens_id=artifact.lens_id,
            lens_run_id=run.lens_run_id,
            locus=locus,
            source_label=label,
        ),
        ContributionEvidence(
            kind="contribution",
            lens_id=artifact.lens_id,
            lens_run_id=run.lens_run_id,
            episode_id="episode-1",
            policy_call=2,
            basis=basis,
            claim_level=claim_level,
            items=(
                ContributionItem(
                    key=key,
                    value=0.5,
                    rank=1,
                    sign="positive",
                    model_locus=locus,
                    label=label,
                ),
            ),
        ),
    )
    return ProbeEvidenceBundle(
        bundle_id=f"bundle-{lens_id}",
        artifact=artifact,
        run=run,
        geometry=geometry,
        capabilities=geometry.capabilities,
        primitives=primitives,
    )


def _artifact_and_run(lens_id: str, name: str) -> tuple[ProbeLensArtifact, LensRun]:
    artifact = ProbeLensArtifact(
        lens_id=lens_id,
        lens_version="v1",
        name=name,
        target=name,
        source_model={"model_id": "pi05"},
        source={"module": "action_head", "token_scope": "pooled"},
        training={"dataset_id": "demo", "objective": "logistic regression"},
        created_at="2026-06-09T00:00:00+00:00",
    )
    run = LensRun(
        lens_run_id=f"run-{lens_id}",
        lens_id=artifact.lens_id,
        lens_version=artifact.lens_version,
        dataset_id="demo",
        computed_at="2026-06-09T00:00:00+00:00",
        result_version="probe_evidence.v1",
        status="complete",
        evidence_bundle_id=f"bundle-{lens_id}",
    )
    return artifact, run


__all__ = [
    "ArrayRef",
    "CohortSummaryEvidence",
    "ContributionEvidence",
    "ContributionItem",
    "CurrentMomentEvidence",
    "EvidenceClaimLevel",
    "EvidenceCohortRef",
    "EvidencePrimitive",
    "EvidencePrimitiveKind",
    "EpisodeLensAdapter",
    "FailureCaseEvidence",
    "LensCapability",
    "LensFeatureContribution",
    "LensGeometry",
    "LensProvenanceEvidence",
    "LensRun",
    "LensTemporalRow",
    "ModelLocusEvidence",
    "ModelLocusRef",
    "PanelAvailability",
    "PanelSpec",
    "PipelineLensAnnotation",
    "PredictionEvidence",
    "ProbeEvidenceBundle",
    "ProbeEvidenceContractError",
    "ProbeLensArtifact",
    "RankedMoment",
    "RankedMomentsEvidence",
    "ResearchSelectionState",
    "ScoreSeriesEvidence",
    "UnavailableReason",
    "attention_head_grouped_probe_bundle",
    "default_probe_panel_specs",
    "primitive_kinds",
    "primitives_by_kind",
    "probe_episode_lens_adapter",
    "pooled_no_contribution_probe_bundle",
    "ranked_moments",
    "raw_layer_contribution_probe_bundle",
    "sae_feature_contribution_probe_bundle",
    "scalar_timestep_probe_bundle",
    "select_available_panels",
    "select_contribution_claim_level",
    "select_contribution_rows",
    "select_current_moment_evidence",
    "select_top_moments",
    "select_unavailable_reasons",
]
