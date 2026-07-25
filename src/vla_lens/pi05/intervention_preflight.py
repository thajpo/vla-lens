"""PI0.5 intervention preflight boundary.

This module is safe to import in the normal repo environment. It does not load
Torch, LeRobot, LIBERO, or model checkpoints.
"""

from __future__ import annotations

from typing import Any, Mapping

from vla_lens.interventions.preflight import intervention_preflight
from vla_lens.interventions.specs import PreflightCheck, RuntimePreflightResult
from vla_lens.traces import TraceDataset


def pi05_intervention_preflight(
    dataset: TraceDataset,
    payload: Mapping[str, Any],
    *,
    runtime_available: bool = False,
) -> RuntimePreflightResult:
    """Run generic preflight and resolve PI0.5 runtime availability."""

    base = intervention_preflight(dataset, payload)
    checks = tuple(
        [
            *_replace_model_runtime_check(
                base.checks,
                runtime_available=runtime_available,
            ),
            *_source_patch_checks(dataset, payload),
        ]
    )
    runtime_resolution = {
        **dict(base.runtime_resolution),
        "adapter": "pi05",
        "model_family": "pi05",
        "runtime_available": runtime_available,
    }
    return RuntimePreflightResult(
        status=_status_from_checks(checks),
        checks=checks,
        warnings=_warnings_from_checks(checks),
        errors=_errors_from_checks(checks),
        runtime_resolution=runtime_resolution,
        missing_capabilities=tuple(
            check.name
            for check in checks
            if check.status in {"failed", "partial", "unavailable"}
        ),
        capability_status={check.name: check.status == "ok" for check in checks},
        target_resolution=base.target_resolution,
        action_basis_status=base.action_basis_status,
        runtime_environment={
            **dict(base.runtime_environment),
            "pi05_runtime_available": runtime_available,
        },
    )


def _replace_model_runtime_check(
    checks: tuple[PreflightCheck, ...],
    *,
    runtime_available: bool,
) -> list[PreflightCheck]:
    out: list[PreflightCheck] = []
    for check in checks:
        if check.name != "model_runtime_available":
            out.append(check)
            continue
        if runtime_available:
            out.append(
                PreflightCheck(
                    name="model_runtime_available",
                    status="ok",
                    ok=True,
                    message="PI0.5 runtime executor is available for this run.",
                    metadata={"adapter": "pi05", "can_rerun": True},
                )
            )
        else:
            out.append(
                PreflightCheck(
                    name="model_runtime_available",
                    status="unavailable",
                    ok=False,
                    message=(
                        "PI0.5 runtime executor is unavailable in this process. "
                        "Run inside the dedicated PI0.5 capture environment."
                    ),
                    warnings=("No PI0.5 executor was provided.",),
                    metadata={"adapter": "pi05", "can_rerun": False},
                )
            )
    return out


def _status_from_checks(checks: tuple[PreflightCheck, ...]) -> str:
    fatal_names = {
        "policy_call_exists",
        "stored_action_exists",
        "stored_action_chunk_exists",
        "source_artifact_exists",
        "target_site_declared_in_model_site_index",
        "token_space_declared",
        "counterfactual_pair_compatible",
    }
    failed = {check.name for check in checks if check.status == "failed"}
    if failed & fatal_names:
        return "failed"
    if any(check.status in {"failed", "partial"} for check in checks):
        return "partial"
    if any(
        check.name == "model_runtime_available" and check.status == "unavailable"
        for check in checks
    ):
        return "inspected_only"
    return "ok"


def _warnings_from_checks(checks: tuple[PreflightCheck, ...]) -> tuple[str, ...]:
    return _dedupe(
        tuple(
            warning
            for check in checks
            if check.status in {"partial", "unavailable"}
            for warning in (*check.warnings, check.message)
            if warning
        )
    )


def _errors_from_checks(checks: tuple[PreflightCheck, ...]) -> tuple[str, ...]:
    return _dedupe(
        tuple(
            error
            for check in checks
            if check.status == "failed"
            for error in (check.errors or (check.message,))
            if error
        )
    )


def _dedupe(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


def _source_patch_checks(
    dataset: TraceDataset,
    payload: Mapping[str, Any],
) -> tuple[PreflightCheck, ...]:
    if _execution_mode(payload) != "donor_source_patch":
        return ()
    donor = _mapping(payload.get("donor"))
    donor_trace = _mapping(donor.get("trace"))
    donor_call = _mapping(donor.get("policy_call"))
    donor_trace_id = str(donor_trace.get("trace_id") or "").strip()
    context = _mapping(payload.get("context"))
    if not context:
        context = _mapping(_mapping(payload.get("baseline")).get("context"))
    recipient_trace_id = str(context.get("trace_id") or "").strip()
    failures: list[str] = []
    if not donor_trace_id:
        failures.append("donor.trace.trace_id is missing")
    if donor_trace_id == recipient_trace_id and donor_trace_id:
        failures.append("recipient and donor must be different traces")
    try:
        recipient_bundle = dataset.bundle(recipient_trace_id)
    except Exception:
        recipient_bundle = None
        failures.append(f"recipient trace {recipient_trace_id!r} is unavailable")
    try:
        donor_bundle = dataset.bundle(donor_trace_id)
    except Exception:
        donor_bundle = None
        failures.append(f"donor trace {donor_trace_id!r} is unavailable")
    donor_call_index = int(donor_call.get("policy_call_index") or 0)
    if recipient_bundle is not None and donor_bundle is not None:
        if donor_call_index >= int(donor_bundle.action_chunks(mmap=True).shape[0]):
            failures.append(f"donor policy call {donor_call_index} is unavailable")
        comparisons = {
            "model checkpoint": (
                recipient_bundle.manifest.model_id == donor_bundle.manifest.model_id
            ),
            "prompt": recipient_bundle.manifest.prompt == donor_bundle.manifest.prompt,
            "task": recipient_bundle.manifest.task_id == donor_bundle.manifest.task_id,
            "action shape": (
                tuple(recipient_bundle.action_chunks(mmap=True).shape[1:])
                == tuple(donor_bundle.action_chunks(mmap=True).shape[1:])
            ),
        }
        failures.extend(name for name, matches in comparisons.items() if not matches)
    if failures:
        return (
            PreflightCheck(
                name="counterfactual_pair_compatible",
                status="failed",
                ok=False,
                message="Recipient and donor cannot be compared: " + ", ".join(failures),
                errors=tuple(failures),
                metadata={
                    "recipient_trace_id": recipient_trace_id,
                    "donor_trace_id": donor_trace_id,
                    "donor_policy_call_index": donor_call_index,
                },
            ),
        )
    return (
        PreflightCheck(
            name="counterfactual_pair_compatible",
            status="ok",
            ok=True,
            message="Recipient and donor trace metadata support source patching.",
            metadata={
                "recipient_trace_id": recipient_trace_id,
                "donor_trace_id": donor_trace_id,
                "donor_policy_call_index": donor_call_index,
            },
        ),
    )


def _execution_mode(payload: Mapping[str, Any]) -> str:
    intervention = _mapping(payload.get("intervention"))
    request = _mapping(payload.get("request")) or _mapping(intervention.get("request"))
    operator = _mapping(request.get("operator"))
    return str(_mapping(operator.get("parameters")).get("mode") or "")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["pi05_intervention_preflight"]
