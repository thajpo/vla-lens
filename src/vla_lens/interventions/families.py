"""Discovery-artifact family contracts for future intervention targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from vla_lens.artifacts import LensArtifact
from vla_lens.interventions.serialization import jsonable, mapping_from, required_mapping
from vla_lens.interventions.specs import (
    ControlSpec,
    InterventionOperatorSpec,
    OutcomeSpec,
    TargetSpec,
)


@dataclass(frozen=True, slots=True)
class ArtifactFamilyContract:
    """How one discovery artifact family plugs into the intervention spine."""

    artifact_type: str
    target_kind: str
    operators: tuple[str, ...]
    outcomes: tuple[str, ...]
    required_controls: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    representation_kind: str = "metadata"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.artifact_type:
            raise ValueError("artifact_type is required")
        # Reuse core validators so future-family contracts cannot drift.
        TargetSpec(
            kind=self.target_kind,
            source_artifact_id="contract-smoke",
            source_artifact_type=self.artifact_type,
        )
        for outcome in self.outcomes:
            OutcomeSpec(kind=outcome)
        for operator in self.operators:
            InterventionOperatorSpec(operator=operator)
        for controls in self.required_controls.values():
            for control in controls:
                ControlSpec(kind=control)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": self.artifact_type,
            "target_kind": self.target_kind,
            "operators": list(self.operators),
            "outcomes": list(self.outcomes),
            "required_controls": {
                str(label): list(controls)
                for label, controls in self.required_controls.items()
            },
            "representation_kind": self.representation_kind,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ArtifactFamilyContract":
        data = required_mapping(payload, field="ArtifactFamilyContract")
        required_controls = mapping_from(
            data.get("required_controls"),
            field="required_controls",
        )
        return cls(
            artifact_type=str(data["artifact_type"]),
            target_kind=str(data["target_kind"]),
            operators=tuple(str(item) for item in data.get("operators", ())),
            outcomes=tuple(str(item) for item in data.get("outcomes", ())),
            required_controls={
                str(label): tuple(str(item) for item in controls)
                for label, controls in required_controls.items()
                if isinstance(controls, (list, tuple))
            },
            representation_kind=str(data.get("representation_kind", "metadata")),
            description=str(data.get("description", "")),
        )


_FAMILY_CONTRACTS = (
    ArtifactFamilyContract(
        artifact_type="probe_suite",
        target_kind="probe_direction",
        operators=("add_direction", "project_out_direction"),
        outcomes=("action", "rollout", "token"),
        required_controls={
            "causal_local": ("noop_rerun", "random_direction"),
            "specific": ("wrong_layer", "wrong_time", "wrong_token"),
        },
        representation_kind="direction_vector",
        description="Probe coefficient or direction artifact.",
    ),
    ArtifactFamilyContract(
        artifact_type="contrast_direction",
        target_kind="contrast_direction",
        operators=("add_direction", "project_out_direction"),
        outcomes=("action", "rollout", "activation"),
        required_controls={
            "causal_local": ("noop_rerun", "random_direction"),
            "specific": ("wrong_layer", "wrong_time"),
        },
        representation_kind="direction_vector",
        description="Mean-difference or contrast direction artifact.",
    ),
    ArtifactFamilyContract(
        artifact_type="activation_cluster",
        target_kind="contrast_direction",
        operators=("add_direction", "project_out_direction"),
        outcomes=("action", "rollout", "activation"),
        required_controls={
            "causal_local": ("noop_rerun", "random_direction"),
            "specific": ("wrong_layer", "wrong_time"),
        },
        representation_kind="cluster_direction",
        description="Activation cluster converted to a manipulable direction.",
    ),
    ArtifactFamilyContract(
        artifact_type="sae_feature",
        target_kind="feature",
        operators=("feature_boost", "feature_clamp", "feature_ablate", "add_decoder_direction"),
        outcomes=("activation", "action", "rollout", "token"),
        required_controls={
            "causal_local": ("noop_rerun", "wrong_feature"),
            "specific": ("wrong_feature", "wrong_layer", "placebo_target"),
            "causal_cohort": ("heldout_split", "matched_cohort"),
        },
        representation_kind="feature_index",
        description="Sparse autoencoder feature target.",
    ),
    ArtifactFamilyContract(
        artifact_type="transcoder_feature",
        target_kind="path",
        operators=("feature_clamp", "path_patch", "source_patch", "project_out_direction"),
        outcomes=("pathway", "activation", "action", "rollout"),
        required_controls={
            "causal_local": ("noop_rerun", "wrong_feature"),
            "specific": ("wrong_feature", "wrong_layer", "source_patch"),
            "causal_cohort": ("heldout_split", "matched_cohort"),
        },
        representation_kind="pathway_feature",
        description="Transcoder feature or learned pathway target.",
    ),
    ArtifactFamilyContract(
        artifact_type="crosscoder_feature",
        target_kind="path",
        operators=("feature_clamp", "path_patch", "source_patch", "project_out_direction"),
        outcomes=("pathway", "activation", "action", "rollout"),
        required_controls={
            "causal_local": ("noop_rerun", "wrong_feature"),
            "specific": ("wrong_feature", "wrong_layer", "source_patch"),
            "causal_cohort": ("heldout_split", "matched_cohort"),
        },
        representation_kind="cross_model_pathway",
        description="Shared or differential crosscoder feature target.",
    ),
    ArtifactFamilyContract(
        artifact_type="attention_map",
        target_kind="edge",
        operators=("attention_patch", "head_ablate"),
        outcomes=("attention", "action", "rollout"),
        required_controls={
            "causal_local": ("noop_rerun", "wrong_head"),
            "specific": ("wrong_head", "wrong_token", "wrong_edge"),
            "causal_cohort": ("heldout_split", "matched_cohort"),
        },
        representation_kind="attention_edge",
        description="Attention or attribution edge candidate.",
    ),
    ArtifactFamilyContract(
        artifact_type="attention_edge",
        target_kind="edge",
        operators=("attention_patch", "head_ablate"),
        outcomes=("attention", "action", "rollout"),
        required_controls={
            "causal_local": ("noop_rerun", "wrong_head"),
            "specific": ("wrong_head", "wrong_token", "wrong_edge"),
            "causal_cohort": ("heldout_split", "matched_cohort"),
        },
        representation_kind="attention_edge",
        description="Explicit attention edge candidate.",
    ),
)

_ALIASES = {
    "mean_difference_direction": "contrast_direction",
    "attribution_map": "attention_map",
    "sae": "sae_feature",
    "transcoder": "transcoder_feature",
    "crosscoder": "crosscoder_feature",
}
_REGISTRY = {contract.artifact_type: contract for contract in _FAMILY_CONTRACTS}


def artifact_family_registry() -> tuple[ArtifactFamilyContract, ...]:
    """Return built-in discovery artifact family contracts."""
    return _FAMILY_CONTRACTS


def artifact_family_for_type(artifact_type: str) -> ArtifactFamilyContract:
    """Return the family contract for a discovery artifact type."""
    normalized = _ALIASES.get(artifact_type, artifact_type)
    try:
        return _REGISTRY[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown intervention artifact family '{artifact_type}'") from exc


def target_from_discovery_artifact(
    artifact: LensArtifact | Mapping[str, Any],
    *,
    model_site: str | None = None,
    token_space: str | None = None,
    representation: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TargetSpec:
    """Normalize a discovery artifact into a TargetSpec candidate."""
    payload = artifact.to_dict() if isinstance(artifact, LensArtifact) else dict(artifact)
    artifact_type = str(payload["artifact_type"])
    artifact_id = str(payload["artifact_id"])
    contract = artifact_family_for_type(artifact_type)
    selector = _mapping(payload.get("selector"))
    method = _mapping(payload.get("method"))
    display = _mapping(payload.get("display"))
    return TargetSpec(
        kind=contract.target_kind,
        source_artifact_id=artifact_id,
        source_artifact_type=artifact_type,
        model_site=model_site
        or _first_str(
            _nested(selector, ("target", "model_site")),
            selector.get("model_site"),
            method.get("model_site"),
            _nested(display, ("target", "model_site")),
        ),
        layer=_optional_int(
            _nested(selector, ("target", "layer")),
            selector.get("layer"),
            method.get("layer"),
        ),
        tensor_type=_first_str(
            _nested(selector, ("target", "tensor_type")),
            selector.get("tensor_type"),
            method.get("tensor_type"),
        ),
        token_space=token_space
        or _first_str(
            _nested(selector, ("target", "token_space")),
            selector.get("token_space"),
            method.get("token_space"),
        ),
        token_selector=_mapping(
            _nested(selector, ("target", "token_selector"))
            or selector.get("token_selector")
        ),
        reduction=_first_str(
            _nested(selector, ("target", "reduction")),
            selector.get("reduction"),
            method.get("reduction"),
        ),
        representation=dict(
            representation
            or _representation_from_artifact(contract, selector, method, payload)
        ),
        metadata={
            "artifact_family": contract.artifact_type,
            "legal_operators": list(contract.operators),
            "legal_outcomes": list(contract.outcomes),
            **jsonable(metadata or {}),
        },
    )


def legal_operators_for_artifact(artifact_type: str) -> tuple[str, ...]:
    return artifact_family_for_type(artifact_type).operators


def legal_outcomes_for_artifact(artifact_type: str) -> tuple[str, ...]:
    return artifact_family_for_type(artifact_type).outcomes


def required_controls_for_artifact_claim(
    artifact_type: str,
    claim_label: str,
) -> tuple[str, ...]:
    contract = artifact_family_for_type(artifact_type)
    return tuple(contract.required_controls.get(claim_label, ()))


def _representation_from_artifact(
    contract: ArtifactFamilyContract,
    selector: Mapping[str, Any],
    method: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    target = _mapping(selector.get("target"))
    representation = _mapping(target.get("representation"))
    if representation:
        return dict(representation)
    arrays = _mapping(payload.get("arrays"))
    return {
        "kind": contract.representation_kind,
        "feature_index": _first_value(
            target.get("feature_index"),
            selector.get("feature_index"),
            method.get("feature_index"),
        ),
        "array_ref": _first_str(
            arrays.get("decoder_vector"),
            arrays.get("direction"),
            arrays.get("basis"),
            arrays.get("primary_array"),
        ),
        "edge": jsonable(_mapping(target.get("edge") or selector.get("edge"))),
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(value: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _first_str(*values: Any) -> str | None:
    for value in values:
        if value is not None and str(value) != "":
            return str(value)
    return None


def _first_value(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _optional_int(*values: Any) -> int | None:
    for value in values:
        if value is not None and str(value) != "":
            return int(value)
    return None
