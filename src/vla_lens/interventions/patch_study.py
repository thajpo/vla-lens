"""Deterministic planning for counterfactual activation-patch studies."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from vla_lens.interventions.counterfactuals import (
    CounterfactualPairManifest,
    PatchStudySpec,
)
from vla_lens.interventions.serialization import jsonable


@dataclass(frozen=True, slots=True)
class PlannedPatchTrial:
    """One fully resolved pair, layer, and token-region comparison."""

    trial_id: str
    study_id: str
    pair_id: str
    site_index: int
    layer: int
    model_site: str
    token_region: str
    recipient_token_indices: tuple[int, ...]
    donor_token_indices: tuple[int, ...]
    wrong_recipient_token_indices: tuple[int, ...] = ()
    wrong_donor_token_indices: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "study_id": self.study_id,
            "pair_id": self.pair_id,
            "site_index": self.site_index,
            "layer": self.layer,
            "model_site": self.model_site,
            "token_region": self.token_region,
            "recipient_token_indices": list(self.recipient_token_indices),
            "donor_token_indices": list(self.donor_token_indices),
            "wrong_recipient_token_indices": list(self.wrong_recipient_token_indices),
            "wrong_donor_token_indices": list(self.wrong_donor_token_indices),
        }


def expand_patch_study(
    study: PatchStudySpec,
    pairs: Sequence[CounterfactualPairManifest],
) -> tuple[PlannedPatchTrial, ...]:
    """Expand the declared pair, site, and token axes in a stable order."""
    pairs_by_id = {pair.pair_id: pair for pair in pairs}
    if set(pairs_by_id) != set(study.pair_ids):
        raise ValueError("study pairs must exactly match study.pair_ids")
    token_regions = _string_values(study.axes.get("token_regions"), "token_regions")
    if not token_regions:
        raise ValueError("patch study axes.token_regions must name at least one region")
    wrong_by_region = _mapping(study.axes.get("wrong_region_by_region"))
    default_wrong = str(study.axes.get("wrong_region") or "").strip()

    planned: list[PlannedPatchTrial] = []
    for pair_id in study.pair_ids:
        pair = pairs_by_id[pair_id]
        regions = _mapping(pair.validation.get("token_regions"))
        for site_index, site in enumerate(study.sites):
            layer = _required_int(site.get("layer"), field="site layer")
            model_site = str(
                site.get("model_site")
                or f"pi05.vlm.layers.{layer}.prefix.hidden_tokens"
            ).format(layer=layer)
            for region_name in token_regions:
                recipient_indices, donor_indices = _region_indices(regions, region_name)
                wrong_name = str(wrong_by_region.get(region_name) or default_wrong).strip()
                wrong_recipient: tuple[int, ...] = ()
                wrong_donor: tuple[int, ...] = ()
                if "wrong_region" in study.controls:
                    if not wrong_name:
                        raise ValueError(
                            f"study must declare a wrong region for token region {region_name!r}"
                        )
                    wrong_recipient, wrong_donor = _region_indices(regions, wrong_name)
                    if len(wrong_recipient) != len(recipient_indices):
                        raise ValueError(
                            f"wrong region {wrong_name!r} must contain the same number of "
                            f"tokens as {region_name!r} for pair {pair_id}"
                        )
                identity = {
                    "study_id": study.study_id,
                    "pair_id": pair_id,
                    "site_index": site_index,
                    "layer": layer,
                    "model_site": model_site,
                    "token_region": region_name,
                    "recipient": recipient_indices,
                    "donor": donor_indices,
                    "wrong_recipient": wrong_recipient,
                    "wrong_donor": wrong_donor,
                }
                suffix = _canonical_sha256(identity)[:10]
                trial_id = _safe_id(
                    f"{study.study_id}-{pair_id}-l{layer}-{region_name}-{suffix}"
                )
                planned.append(
                    PlannedPatchTrial(
                        trial_id=trial_id,
                        study_id=study.study_id,
                        pair_id=pair_id,
                        site_index=site_index,
                        layer=layer,
                        model_site=model_site,
                        token_region=region_name,
                        recipient_token_indices=recipient_indices,
                        donor_token_indices=donor_indices,
                        wrong_recipient_token_indices=wrong_recipient,
                        wrong_donor_token_indices=wrong_donor,
                    )
                )
    return tuple(planned)


def build_patch_trial_request(
    template: Mapping[str, Any],
    *,
    study: PatchStudySpec,
    pair: CounterfactualPairManifest,
    trial: PlannedPatchTrial,
) -> dict[str, Any]:
    """Resolve a reusable source-patch request template for one planned trial."""
    payload = copy.deepcopy(dict(template))
    payload["run_id"] = trial.trial_id
    payload["title"] = (
        f"{study.study_id}: {pair.pair_id}, layer {trial.layer}, "
        f"{trial.token_region} tokens"
    )
    payload["counterfactual"] = {
        "study_id": study.study_id,
        "pair_id": pair.pair_id,
        "token_region": trial.token_region,
        "site_index": trial.site_index,
    }
    recipient_context = _policy_context(pair.recipient.to_dict(), role="recipient")
    payload["baseline"] = {"context": recipient_context}
    payload["donor"] = pair.donor.to_dict()

    target = _required_mutable_mapping(payload, "target")
    target["layer"] = trial.layer
    target["model_site"] = trial.model_site
    target["token_selector"] = {
        "kind": "indices",
        "indices": list(trial.recipient_token_indices),
        "region": trial.token_region,
    }

    request = _nested_request(payload)
    operator = _required_mutable_mapping(request, "operator")
    parameters = dict(_mapping(operator.get("parameters")))
    parameters["mode"] = "donor_source_patch"
    parameters["donor_token_indices"] = list(trial.donor_token_indices)
    operator["parameters"] = parameters
    outcome = dict(_mapping(request.get("outcome")))
    outcome_parameters = dict(_mapping(outcome.get("parameters")))
    outcome_parameters["patch_decision_thresholds"] = study.thresholds.to_dict()
    outcome["parameters"] = outcome_parameters
    request["outcome"] = outcome
    request["controls"] = _resolved_controls(request, study, trial)
    return payload


def patch_study_request_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a complete study job so resume cannot mix different plans."""
    return _canonical_sha256(payload)


def _resolved_controls(
    request: Mapping[str, Any],
    study: PatchStudySpec,
    trial: PlannedPatchTrial,
) -> list[dict[str, Any]]:
    existing = request.get("controls") or ()
    if isinstance(existing, Mapping | str):
        existing = (existing,)
    by_kind: dict[str, dict[str, Any]] = {}
    for item in existing if isinstance(existing, Sequence) else ():
        control = {"kind": item} if isinstance(item, str) else dict(_mapping(item))
        kind = str(control.get("kind") or "").strip()
        if kind:
            by_kind[kind] = control
    controls: list[dict[str, Any]] = []
    for kind in study.controls:
        control = copy.deepcopy(by_kind.get(kind, {"kind": kind}))
        if kind == "wrong_region":
            parameters = dict(_mapping(control.get("parameters")))
            parameters.update(
                {
                    "recipient_indices": list(trial.wrong_recipient_token_indices),
                    "donor_indices": list(trial.wrong_donor_token_indices),
                }
            )
            control["parameters"] = parameters
        controls.append(control)
    return controls


def _policy_context(spec: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    trace = _mapping(spec.get("trace"))
    policy_call = _mapping(spec.get("policy_call"))
    trace_id = str(policy_call.get("trace_id") or trace.get("trace_id") or "").strip()
    if not trace_id:
        raise ValueError(f"counterfactual {role} must declare a trace_id")
    return {
        "trace_id": trace_id,
        "policy_call_index": int(policy_call.get("policy_call_index") or 0),
    }


def _region_indices(
    regions: Mapping[str, Any],
    name: str,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    region = regions.get(name)
    if isinstance(region, Mapping):
        recipient = _integer_values(region.get("recipient"), f"{name} recipient tokens")
        donor = _integer_values(region.get("donor"), f"{name} donor tokens")
    else:
        recipient = _integer_values(region, f"{name} tokens")
        donor = recipient
    if not recipient or not donor:
        raise ValueError(f"pair token region {name!r} must contain recipient and donor tokens")
    if len(recipient) != len(donor):
        raise ValueError(f"pair token region {name!r} recipient/donor counts must match")
    return recipient, donor


def _nested_request(payload: dict[str, Any]) -> dict[str, Any]:
    intervention = _required_mutable_mapping(payload, "intervention")
    request = intervention.get("request")
    if not isinstance(request, Mapping):
        raise ValueError("patch template requires intervention.request")
    mutable = dict(request)
    intervention["request"] = mutable
    return mutable


def _required_mutable_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"patch template requires {key}")
    mutable = dict(value)
    payload[key] = mutable
    return mutable


def _string_values(value: Any, field: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, Sequence):
        return ()
    values = tuple(str(item).strip() for item in value if str(item).strip())
    if len(values) != len(set(values)):
        raise ValueError(f"{field} values must be unique")
    return values


def _integer_values(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    values = tuple(int(item) for item in value)
    if any(item < 0 for item in values):
        raise ValueError(f"{field} must be non-negative")
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must be unique")
    return values


def _required_int(value: Any, *, field: str) -> int:
    if value is None:
        raise ValueError(f"{field} is required")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{field} must be non-negative")
    return resolved


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        jsonable(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_id(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = [
    "PlannedPatchTrial",
    "build_patch_trial_request",
    "expand_patch_study",
    "patch_study_request_sha256",
]
