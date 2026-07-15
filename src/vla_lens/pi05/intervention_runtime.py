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
    InterventionRun,
    InterventionTrial,
    RuntimeResolution,
    RuntimeTrialOutput,
    TargetSpec,
    action_delta_metrics,
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
) -> PI05InterventionRunResult:
    """Run a PI0.5 action intervention through an injected runtime executor."""

    preflight = pi05_intervention_preflight(dataset, payload, runtime_available=True)
    context = _context(dataset, payload)
    target = TargetSpec.from_dict(_target_payload(payload))
    request = _request_payload(payload)
    bundle = dataset.bundle(str(context.trace_id))
    call_index = int(context.policy_call_index or 0)
    stored_original = np.asarray(bundle.action_chunks(mmap=True)[call_index], dtype=np.float32)

    noop = executor.run_noop(payload)
    intervention = executor.run_intervention(payload)
    noop_action = _trial_action(noop)
    intervened_action = _trial_action(intervention)
    controls, control_arrays, control_action_refs = _run_controls(executor, payload, request)

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
        **control_arrays,
    }
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
        ),
        trials=trials,
        outcomes=tuple(_action_outcomes(delta_metrics)),
        controls=tuple(_control_results(controls, noop_action)),
        outputs=tuple(arrays),
        display={
            "summary": "PI0.5 runtime produced no-op and intervened action chunks.",
            "action_basis": basis_result.to_dict(),
        },
        claim={"claim_strength": _claim_strength(status, trials)},
        provenance={
            "schema_kind": "vla_lens.intervention_run",
            "runtime_adapter": "pi05",
            "runtime_surface": "injected_executor",
            "source": "pi05_intervention_runtime",
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
        outputs={"action_ref": action_ref},
        metrics=output.metrics,
        runtime=output.runtime,
        status=output.status,
        warnings=output.warnings,
        errors=output.errors,
    )


def _action_outcomes(delta_metrics: Mapping[str, Mapping[str, float]]) -> list[dict[str, Any]]:
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
) -> RuntimeResolution:
    target_metadata = dict(target.metadata)
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
        },
        generation_step_mapping=dict(_mapping(_mapping(request.get("schedule")).get("generation_steps"))),
        token_selector_mapping=dict(target.token_selector),
        resolved_tensor_shape=tuple(int(dim) for dim in action.shape),
        resolved_dtype=str(action.dtype),
        resolved_device=_text(payload.get("device")) or "unknown",
        runtime_environment={"adapter": "pi05", "executor": "injected"},
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


def _claim_strength(status: str, trials: tuple[InterventionTrial, ...]) -> list[str]:
    """Prevent engineering-only hook smokes from being promoted to causal evidence."""
    if status != "ok":
        return []
    if any(trial.runtime.get("claim_eligible") is False for trial in trials):
        return []
    return ["causal_local", "action_level"]


def _trial_action(output: RuntimeTrialOutput) -> np.ndarray:
    return np.asarray(output.action_chunk, dtype=np.float32)


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
