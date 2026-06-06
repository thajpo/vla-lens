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
    checks = tuple(_replace_model_runtime_check(base.checks, runtime_available=runtime_available))
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


__all__ = ["pi05_intervention_preflight"]
