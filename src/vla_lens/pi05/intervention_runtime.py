"""PI0.5 intervention runtime contract.

The live PI0.5 executor is injected so this module can be imported and tested in
the normal repo environment. A real executor must be constructed only in the
dedicated PI0.5 capture environment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from vla_lens.analyzer import dataset_fingerprint
from vla_lens.interventions import (
    ActionBasisRequest,
    ActionInterventionExecutor,
    ActionOutcomeResult,
    ContextSpec,
    ControlResult,
    CounterfactualMetrics,
    InterventionRun,
    InterventionTrial,
    PatchDecisionThresholds,
    RuntimeResolution,
    RuntimeTrialOutput,
    TargetSpec,
    action_delta_metrics,
    counterfactual_action_metrics,
    evaluate_patch_trial,
    intervention_run_to_lens_artifact,
    resolve_action_basis,
)
from vla_lens.pi05.intervention_preflight import pi05_intervention_preflight
from vla_lens.traces import TraceDataset
from vla_lens.workbench import dataset_id, save_intervention_run


@dataclass(frozen=True, slots=True)
class PI05InterventionRunResult:
    """Saved PI0.5 intervention run plus index artifact metadata."""

    run: InterventionRun
    artifact_id: str | None = None
    arrays: Mapping[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run.to_dict(),
            "artifact_id": self.artifact_id,
            "arrays": {
                name: {"shape": list(array.shape), "dtype": str(array.dtype)}
                for name, array in self.arrays.items()
            },
        }


def run_pi05_intervention(
    dataset: TraceDataset,
    payload: Mapping[str, Any],
    *,
    executor: ActionInterventionExecutor,
    save: bool = True,
    claim_gate: Mapping[str, Any] | None = None,
    noop_output: RuntimeTrialOutput | None = None,
) -> PI05InterventionRunResult:
    """Run a PI0.5 action intervention through an injected runtime executor."""

    preflight = pi05_intervention_preflight(dataset, payload, runtime_available=True)
    context = _context(dataset, payload)
    target = TargetSpec.from_dict(_target_payload(payload))
    request = _request_payload(payload)
    bundle = dataset.bundle(str(context.trace_id))
    call_index = int(context.policy_call_index or 0)
    stored_original = np.asarray(bundle.action_chunks(mmap=True)[call_index], dtype=np.float32)

    noop = noop_output or executor.run_noop(payload)
    intervention = executor.run_intervention(payload)
    noop_action = _trial_action(noop)
    intervened_action = _trial_action(intervention)
    controls, control_arrays, control_action_refs = _run_controls(executor, payload, request)
    intervention_arrays = _array_outputs(intervention)

    basis_result = resolve_action_basis(
        bundle,
        ActionBasisRequest(basis=_outcome_basis(request)),
    )
    delta_metrics = action_delta_metrics(
        stored_original=stored_original,
        noop=noop_action,
        intervened=intervened_action,
        basis_result=basis_result,
        intended_basis=_intended_basis(request, target),
    )
    delta = intervened_action - noop_action
    arrays: dict[str, np.ndarray] = {
        "stored_original": stored_original,
        "noop": noop_action,
        "intervened": intervened_action,
        "intervened_minus_noop": delta,
        **intervention_arrays,
        **control_arrays,
    }
    source_patch_metrics, source_patch_decision, source_patch_thresholds = (
        _source_patch_evidence(
        noop_action,
        intervened_action,
        controls,
        arrays,
        intervention,
        request,
        claim_gate=claim_gate,
    )
    )
    trials = (
        InterventionTrial(
            trial_id="trial_stored_original",
            trial_kind="stored_original",
            outputs={"action_ref": "stored_original"},
            status="ok",
        ),
        _trial_from_output(
            noop,
            default_trial_id="trial_noop",
            default_kind="noop_rerun",
            action_ref="noop",
        ),
        _trial_from_output(
            intervention,
            default_trial_id="trial_intervention",
            default_kind="intervention",
            action_ref="intervened",
        ),
        *(
            _trial_from_output(
                control,
                default_trial_id=control.trial_id,
                default_kind=control.trial_kind,
                action_ref=action_ref,
            )
            for control, action_ref in zip(controls, control_action_refs, strict=True)
        ),
    )
    status = _run_status(preflight.status, basis_result.status, trials)
    run = InterventionRun(
        run_id=_run_id(payload, context, request),
        title=_title(payload, target),
        status=status,
        context=context,
        target=target,
        request=request,
        preflight=preflight,
        runtime_resolution=_runtime_resolution(
            payload=payload,
            request=request,
            target=target,
            context=context,
            action=intervened_action,
            preflight_target=preflight.target_resolution,
            intervention_runtime=intervention.runtime,
        ),
        trials=trials,
        outcomes=tuple(
            _action_outcomes(
                delta_metrics,
                source_patch_metrics=source_patch_metrics,
                source_patch_decision=source_patch_decision,
            )
        ),
        controls=tuple(_control_results(controls, noop_action)),
        outputs=tuple(arrays),
        display={
            "summary": "PI0.5 runtime produced no-op and intervened action chunks.",
            "action_basis": basis_result.to_dict(),
            "specificity_summary": _specificity_summary(
                noop_action,
                intervened_action,
                controls,
            ),
            **(
                {
                    "counterfactual_transfer": {
                        "metrics": source_patch_metrics.to_dict(),
                        "decision": source_patch_decision.to_dict(),
                        "thresholds": source_patch_thresholds.to_dict(),
                    }
                }
                if source_patch_metrics is not None and source_patch_decision is not None
                else {}
            ),
        },
        claim=_claim_record(
            status,
            trials,
            claim_gate=claim_gate,
            source_patch_decision=source_patch_decision,
        ),
        provenance={
            "schema_kind": "vla_lens.intervention_run",
            "runtime_adapter": "pi05",
            "runtime_surface": "injected_executor",
            "source": "pi05_intervention_runtime",
            "counterfactual": dict(_mapping(payload.get("counterfactual"))),
            "donor": dict(_mapping(payload.get("donor"))),
        },
    )
    artifact_id = None
    if save:
        save_intervention_run(dataset, run.to_workbench_spec())
        artifact = dataset.save_artifact(intervention_run_to_lens_artifact(run), arrays=arrays)
        artifact_id = artifact.artifact_id
    return PI05InterventionRunResult(run=run, artifact_id=artifact_id, arrays=arrays)


def _run_controls(
    executor: ActionInterventionExecutor,
    payload: Mapping[str, Any],
    request: Mapping[str, Any],
) -> tuple[tuple[RuntimeTrialOutput, ...], dict[str, np.ndarray], tuple[str, ...]]:
    controls: list[RuntimeTrialOutput] = []
    arrays: dict[str, np.ndarray] = {}
    action_refs: list[str] = []
    for control_kind in _control_kinds(request):
        output = executor.run_control(payload, control_kind=control_kind)
        controls.append(output)
        action_ref = f"control_{control_kind}"
        arrays[action_ref] = _trial_action(output)
        for name, value in _array_outputs(output).items():
            existing = arrays.get(name)
            if existing is not None and not np.array_equal(existing, value):
                raise ValueError(f"Runtime trials produced conflicting array output {name!r}")
            arrays.setdefault(name, value)
        action_refs.append(action_ref)
    return tuple(controls), arrays, tuple(action_refs)


def _trial_from_output(
    output: RuntimeTrialOutput,
    *,
    default_trial_id: str,
    default_kind: str,
    action_ref: str,
) -> InterventionTrial:
    return InterventionTrial(
        trial_id=output.trial_id or default_trial_id,
        trial_kind=output.trial_kind or default_kind,
        control_kind=output.control_kind,
        outputs={
            "action_ref": action_ref,
            **{f"{name}_ref": name for name in output.array_outputs},
        },
        metrics=output.metrics,
        runtime=output.runtime,
        status=output.status,
        warnings=output.warnings,
        errors=output.errors,
    )


def _action_outcomes(
    delta_metrics: Mapping[str, Mapping[str, float]],
    *,
    source_patch_metrics: CounterfactualMetrics | None = None,
    source_patch_decision: Any | None = None,
) -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    for basis, metrics in delta_metrics.items():
        outcomes.append(
            ActionOutcomeResult(
                basis=basis,
                horizon="full_chunk",
                baseline_trial_id="trial_noop",
                intervention_trial_id="trial_intervention",
                action_ref_baseline="noop",
                action_ref_intervened="intervened",
                delta_ref="intervened_minus_noop",
                metrics=metrics,
            ).to_dict()
        )
    if source_patch_metrics is not None:
        outcomes.append(
            ActionOutcomeResult(
                basis="counterfactual_donor_direction",
                horizon="full_chunk",
                baseline_trial_id="trial_noop",
                intervention_trial_id="trial_intervention",
                action_ref_baseline="noop",
                action_ref_intervened="intervened",
                delta_ref="intervened_minus_noop",
                metrics=source_patch_metrics.to_dict(),
                summaries=(
                    {"decision": source_patch_decision.to_dict()}
                    if source_patch_decision is not None
                    else {}
                ),
            ).to_dict()
        )
    return outcomes


def _control_results(
    controls: tuple[RuntimeTrialOutput, ...],
    noop_action: np.ndarray,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for control in controls:
        action = _trial_action(control)
        results.append(
            ControlResult(
                control_kind=control.control_kind or control.trial_kind,
                status=(
                    control.status
                    if control.status in {"ok", "partial", "failed"}
                    else "partial"
                ),
                trial_ids=(control.trial_id,),
                metrics={"delta_from_noop": float(np.linalg.norm(action - noop_action))},
                warnings=control.warnings,
                errors=control.errors,
            ).to_dict()
        )
    return results


def _runtime_resolution(
    *,
    payload: Mapping[str, Any],
    request: Mapping[str, Any],
    target: TargetSpec,
    context: ContextSpec,
    action: np.ndarray,
    preflight_target: Mapping[str, Any],
    intervention_runtime: Mapping[str, Any],
) -> RuntimeResolution:
    target_metadata = dict(target.metadata)
    direction_resolution = _mapping(intervention_runtime.get("direction_resolution"))
    hook_shape = intervention_runtime.get("resolved_tensor_shape")
    resolved_shape = (
        tuple(int(value) for value in hook_shape)
        if isinstance(hook_shape, (list, tuple))
        else tuple(int(dim) for dim in action.shape)
    )
    return RuntimeResolution(
        adapter="pi05",
        model_family="pi05",
        model_id=target.model_id or _text(payload.get("model_id")),
        model_checkpoint=_text(target_metadata.get("model_checkpoint")),
        call_index=context.policy_call_index,
        requested_target=target.to_dict(),
        resolved_hook={
            "model_site": target.model_site or target.site_id,
            "site_record": preflight_target.get("target_site_record"),
            "layer": direction_resolution.get("layer"),
            "token_indices": intervention_runtime.get("token_indices"),
            "direction_resolution": direction_resolution,
        },
        generation_step_mapping=dict(_mapping(_mapping(request.get("schedule")).get("generation_steps"))),
        token_selector_mapping=dict(target.token_selector),
        resolved_tensor_shape=resolved_shape,
        resolved_dtype=_text(intervention_runtime.get("runtime_dtype")) or str(action.dtype),
        resolved_device=_text(intervention_runtime.get("runtime_device"))
        or _text(payload.get("device"))
        or "unknown",
        runtime_environment={
            "adapter": "pi05",
            "executor": "injected",
            "artifact_id": direction_resolution.get("artifact_id"),
            "array_sha256": direction_resolution.get("array_sha256"),
            "evidence_table_sha256": direction_resolution.get("evidence_table_sha256"),
        },
    )


def _context(dataset: TraceDataset, payload: Mapping[str, Any]) -> ContextSpec:
    context_payload = _context_payload(payload)
    trace_id = _text(context_payload.get("trace_id"))
    if not trace_id:
        raise ValueError("PI0.5 intervention runtime requires context.trace_id")
    bundle = dataset.bundle(trace_id)
    return ContextSpec(
        dataset_id=_text(context_payload.get("dataset_id")) or dataset_id(dataset),
        dataset_root_id=_text(context_payload.get("dataset_root_id")) or str(dataset.root),
        dataset_fingerprint=_text(context_payload.get("dataset_fingerprint"))
        or dataset_fingerprint(dataset),
        trace_id=trace_id,
        episode_id=_text(context_payload.get("episode_id")) or bundle.manifest.episode_id,
        policy_call_index=_int_or_none(context_payload.get("policy_call_index")) or 0,
        timestep=_int_or_none(context_payload.get("timestep")),
        frame_index=_int_or_none(context_payload.get("frame_index")),
        instruction=_text(context_payload.get("instruction")) or bundle.manifest.prompt,
        task=_text(context_payload.get("task")) or bundle.manifest.task_id,
        metadata={"runtime_adapter": "pi05"},
    )


def _run_status(
    preflight_status: str,
    basis_status: str,
    trials: tuple[InterventionTrial, ...],
) -> str:
    if any(trial.status == "failed" for trial in trials):
        return "failed"
    if preflight_status == "failed":
        return "failed"
    if preflight_status == "partial" or basis_status == "partial":
        return "partial"
    return "ok"


def _claim_record(
    status: str,
    trials: tuple[InterventionTrial, ...],
    *,
    claim_gate: Mapping[str, Any] | None,
    source_patch_decision: Any | None = None,
) -> dict[str, Any]:
    """Separate a valid causal method from a positive scientific result."""
    if status != "ok":
        return {"claim_strength": []}
    if any(trial.runtime.get("claim_eligible") is False for trial in trials):
        return {"claim_strength": []}
    if source_patch_decision is not None:
        method_eligible = bool(_mapping(claim_gate).get("passed")) and (
            source_patch_decision.verdict
            not in {"pair_invalid", "replay_invalid", "hook_invalid", "insufficient_data"}
        )
        supports_specificity = bool(source_patch_decision.supports_specificity)
        return {
            "claim_strength": (
                ["causal_local", "action_level"] if supports_specificity else []
            ),
            "method_eligible": method_eligible,
            "scientific_verdict": source_patch_decision.verdict,
            "supports_specificity": supports_specificity,
            "replay_gate": dict(claim_gate or {}),
            "limitations": [
                "Action-level donor transfer does not establish closed-loop behavior.",
                "A cohort and held-out confirmation are required beyond one pair.",
            ],
        }
    artifact_trial = next(
        (
            trial
            for trial in trials
            if trial.runtime.get("purpose") == "artifact_probe_direction"
        ),
        None,
    )
    if artifact_trial is None:
        return {"claim_strength": []}
    required = {"matched_random", "wrong_identity", "wrong_roi"}
    present = {
        str(trial.control_kind)
        for trial in trials
        if trial.control_kind is not None and trial.status == "ok"
    }
    replay_passed = bool(_mapping(claim_gate).get("passed"))
    method_eligible = replay_passed and required.issubset(present)
    return {
        "claim_strength": (["causal_local", "action_level"] if method_eligible else []),
        "method_eligible": method_eligible,
        "scientific_verdict": "not_evaluated_from_execution_alone",
        "required_controls": sorted(required),
        "completed_controls": sorted(present),
        "replay_gate": dict(claim_gate or {}),
        "limitations": [
            "A claim-eligible action-level run does not by itself show that identity is causal.",
            "Behavioral conclusions require rollouts and repeated recipients.",
        ],
    }


def _specificity_summary(
    noop_action: np.ndarray,
    intervened_action: np.ndarray,
    controls: tuple[RuntimeTrialOutput, ...],
) -> Mapping[str, Any]:
    main_l2 = float(np.linalg.norm(intervened_action - noop_action))
    rows = []
    for control in controls:
        control_l2 = float(np.linalg.norm(_trial_action(control) - noop_action))
        rows.append(
            {
                "control_kind": control.control_kind or control.trial_kind,
                "action_delta_l2": control_l2,
                "main_minus_control_l2": main_l2 - control_l2,
                "control_to_main_ratio": (
                    control_l2 / main_l2 if main_l2 > 0.0 else None
                ),
            }
        )
    return {
        "main_action_delta_l2": main_l2,
        "controls": rows,
        "interpretation": (
            "Descriptive action deltas only; no positive identity-mechanism verdict is "
            "assigned automatically."
        ),
    }


def _trial_action(output: RuntimeTrialOutput) -> np.ndarray:
    return np.asarray(output.action_chunk, dtype=np.float32)


def _array_outputs(output: RuntimeTrialOutput) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, value in output.array_outputs.items():
        array = np.asarray(value, dtype=np.float32)
        if array.ndim != 2 or not np.all(np.isfinite(array)):
            raise ValueError(
                f"Runtime array output {name!r} must be a finite action_horizon by "
                "action_dim array"
            )
        arrays[str(name)] = array
    return arrays


def _source_patch_evidence(
    noop_action: np.ndarray,
    intervened_action: np.ndarray,
    controls: tuple[RuntimeTrialOutput, ...],
    arrays: Mapping[str, np.ndarray],
    intervention: RuntimeTrialOutput,
    request: Mapping[str, Any],
    *,
    claim_gate: Mapping[str, Any] | None,
) -> tuple[
    CounterfactualMetrics | None,
    Any | None,
    PatchDecisionThresholds | None,
]:
    donor_action = arrays.get("donor_shared_noise")
    if donor_action is None or intervention.runtime.get("purpose") != "donor_source_patch":
        return None, None, None
    thresholds = _source_patch_thresholds(request)
    main = counterfactual_action_metrics(noop_action, donor_action, intervened_action)
    negative_controls = []
    for control in controls:
        if control.control_kind not in {
            "alpha_zero",
            "shuffled_donor",
            "random_matched_norm",
            "wrong_region",
        }:
            continue
        negative_controls.append(
            counterfactual_action_metrics(
                noop_action,
                donor_action,
                _trial_action(control),
            )
        )
    runtime_pair = _mapping(intervention.runtime.get("pair_compatibility"))
    pair_valid = bool(runtime_pair) and all(
        bool(value)
        for key, value in runtime_pair.items()
        if key
        in {
            "different_trace",
            "model_id",
            "prompt",
            "benchmark",
            "task_id",
            "observation_shape",
            "stored_action_shape",
            "noise_shape",
        }
    )
    decision = evaluate_patch_trial(
        main,
        pair_valid=pair_valid,
        replay_valid=bool(_mapping(claim_gate).get("passed")),
        hook_valid=bool(
            intervention.runtime.get(
                "hook_valid",
                int(intervention.runtime.get("hook_calls") or 0) == 1,
            )
        ),
        controls=tuple(negative_controls),
        thresholds=thresholds,
    )
    return main, decision, thresholds


def _source_patch_thresholds(
    request: Mapping[str, Any],
) -> PatchDecisionThresholds:
    outcome = _mapping(request.get("outcome"))
    parameters = _mapping(outcome.get("parameters"))
    payload = _mapping(parameters.get("patch_decision_thresholds"))
    return PatchDecisionThresholds.from_dict(payload)


def _control_kinds(request: Mapping[str, Any]) -> tuple[str, ...]:
    controls = request.get("controls") or request.get("control")
    if controls is None:
        return ()
    if isinstance(controls, Mapping):
        controls = (controls,)
    if isinstance(controls, str):
        controls = ({"kind": controls},)
    out: list[str] = []
    for item in controls if isinstance(controls, (list, tuple)) else ():
        if isinstance(item, Mapping):
            kind = _text(item.get("kind"))
        else:
            kind = _text(item)
        if kind == "random_direction":
            kind = "random_direction_control"
        if kind:
            out.append(kind)
    return tuple(out)


def _outcome_basis(request: Mapping[str, Any]) -> tuple[str, ...]:
    outcome = _mapping(request.get("outcome"))
    basis = outcome.get("basis", ("raw",))
    if isinstance(basis, str):
        return (basis,)
    if isinstance(basis, (list, tuple)):
        return tuple(str(item) for item in basis if str(item))
    return ("raw",)


def _intended_basis(request: Mapping[str, Any], target: TargetSpec) -> str | None:
    outcome = _mapping(request.get("outcome"))
    value = outcome.get("intended_basis") or target.metadata.get("intended_basis")
    return _text(value) or None


def _run_id(
    payload: Mapping[str, Any],
    context: ContextSpec,
    request: Mapping[str, Any],
) -> str:
    explicit = _text(payload.get("run_id"))
    if explicit:
        return explicit
    encoded = json.dumps(request, sort_keys=True, default=str).encode("utf-8")
    suffix = hashlib.sha256(encoded).hexdigest()[:10]
    return f"pi05_intervention_{context.trace_id}_{context.policy_call_index}_{suffix}"


def _title(payload: Mapping[str, Any], target: TargetSpec) -> str:
    return (
        _text(payload.get("title"))
        or f"PI0.5 intervention at {target.model_site or target.site_id or target.kind}"
    )


def _context_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    baseline = _mapping(payload.get("baseline"))
    return _mapping(payload.get("context")) or _mapping(baseline.get("context"))


def _target_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    target = _mapping(payload.get("target"))
    if not target:
        raise ValueError("PI0.5 intervention runtime requires target")
    return target


def _request_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    intervention = _mapping(payload.get("intervention"))
    request = _mapping(payload.get("request")) or _mapping(intervention.get("request"))
    if not request:
        raise ValueError("PI0.5 intervention runtime requires intervention.request")
    return request


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _int_or_none(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


__all__ = ["PI05InterventionRunResult", "run_pi05_intervention"]
