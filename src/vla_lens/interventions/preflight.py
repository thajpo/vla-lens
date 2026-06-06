"""Runtime-free intervention preflight checks.

The functions in this module inspect saved trace metadata, model-site indexes,
artifact records, and action schema hints. They deliberately do not import or
load PI0.5, Torch, LeRobot, LIBERO, or simulator runtimes.
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any, Mapping

import pandas as pd

from vla_lens.interventions.specs import (
    ARTIFACT_DERIVED_TARGET_KINDS,
    PreflightCheck,
    RuntimePreflightResult,
)
from vla_lens.traces import TraceBundle, TraceDataset

HEAVY_RUNTIME_MODULES = ("torch", "lerobot", "libero", "robosuite")

PREFLIGHT_CHECK_NAMES = (
    "policy_call_exists",
    "stored_action_exists",
    "stored_action_chunk_exists",
    "source_artifact_exists",
    "target_site_declared_in_model_site_index",
    "token_space_declared",
    "action_decoder_metadata_available",
    "action_basis_metadata_available",
    "runtime_adapter_declared",
    "model_runtime_available",
    "runtime_environment_safe",
)

_FATAL_CHECKS = {
    "policy_call_exists",
    "stored_action_exists",
    "stored_action_chunk_exists",
    "source_artifact_exists",
    "target_site_declared_in_model_site_index",
    "token_space_declared",
}


def intervention_preflight(
    dataset: TraceDataset,
    payload: Mapping[str, Any],
) -> RuntimePreflightResult:
    """Return metadata-only readiness checks for an intervention request."""

    context = _context_payload(payload)
    target = _target_payload(payload)
    request = _request_payload(payload)
    outcome = _outcome_payload(payload, request=request)
    schedule = _mapping(request.get("schedule"))

    trace_id = _trace_id(payload, context)
    policy_call_index = _policy_call_index(payload, context, schedule)
    bundle = _bundle_or_none(dataset, trace_id)

    policy_call_record, policy_call_check = _policy_call_check(
        bundle=bundle,
        trace_id=trace_id,
        policy_call_index=policy_call_index,
    )
    stored_action_check = _stored_action_check(bundle)
    stored_action_chunk_check = _stored_action_chunk_check(bundle)
    source_artifact_check = _source_artifact_check(dataset, payload, target)
    target_site_record, target_site_check = _target_site_check(dataset, trace_id, target)
    token_space, token_space_check = _token_space_check(bundle, target, target_site_record)
    decoder_check = _action_decoder_check(
        bundle=bundle,
        policy_call_record=policy_call_record,
        action_chunk_available=stored_action_chunk_check.status == "ok",
    )
    action_basis_status, basis_check = _action_basis_check(
        bundle=bundle,
        outcome=outcome,
        action_chunk_available=stored_action_chunk_check.status == "ok",
    )
    runtime_resolution, runtime_adapter_check = _runtime_adapter_check(
        payload=payload,
        request=request,
        target=target,
        policy_call_record=policy_call_record,
    )
    model_runtime_check = _model_runtime_check(runtime_resolution)
    runtime_environment, runtime_environment_check = _runtime_environment_check()

    checks = (
        policy_call_check,
        stored_action_check,
        stored_action_chunk_check,
        source_artifact_check,
        target_site_check,
        token_space_check,
        decoder_check,
        basis_check,
        runtime_adapter_check,
        model_runtime_check,
        runtime_environment_check,
    )
    check_by_name = {check.name: check for check in checks}
    missing_capabilities = tuple(
        check.name
        for check in checks
        if check.status in {"failed", "partial", "unavailable"}
    )
    capability_status = {name: check_by_name[name].status == "ok" for name in PREFLIGHT_CHECK_NAMES}
    status = _preflight_status(checks)
    warnings = tuple(
        message
        for check in checks
        for message in (
            (*check.warnings, check.message) if check.status != "ok" else check.warnings
        )
        if message and check.status in {"partial", "unavailable"}
    )
    errors = tuple(
        error or check.message
        for check in checks
        if check.status == "failed"
        for error in (check.errors or (check.message,))
        if error
    )

    return RuntimePreflightResult(
        status=status,
        checks=checks,
        warnings=_dedupe(warnings),
        errors=_dedupe(errors),
        runtime_resolution=runtime_resolution,
        missing_capabilities=missing_capabilities,
        capability_status=capability_status,
        target_resolution={
            "trace_id": trace_id,
            "policy_call_index": policy_call_index,
            "source_artifact_id": _source_artifact_id(payload, target),
            "target_site": _target_site_address(target),
            "token_space": token_space,
            "target_site_record": target_site_record,
        },
        action_basis_status=action_basis_status,
        runtime_environment=runtime_environment,
    )


def _preflight_status(checks: tuple[PreflightCheck, ...]) -> str:
    failed = {check.name for check in checks if check.status == "failed"}
    if failed & _FATAL_CHECKS:
        return "failed"
    if any(check.status in {"failed", "partial"} for check in checks):
        return "partial"
    if any(
        check.name == "model_runtime_available" and check.status == "unavailable"
        for check in checks
    ):
        return "inspected_only"
    return "ok"


def _policy_call_check(
    *,
    bundle: TraceBundle | None,
    trace_id: str | None,
    policy_call_index: int | None,
) -> tuple[dict[str, Any] | None, PreflightCheck]:
    if not trace_id:
        return None, _check(
            "policy_call_exists",
            "failed",
            "Request must include context.trace_id before a policy call can be checked.",
            errors=("Missing context.trace_id.",),
        )
    if bundle is None:
        return None, _check(
            "policy_call_exists",
            "failed",
            f"Trace {trace_id!r} does not exist in this dataset.",
            errors=(f"Unknown trace_id {trace_id!r}.",),
            metadata={"trace_id": trace_id},
        )
    if policy_call_index is None:
        return None, _check(
            "policy_call_exists",
            "failed",
            "Request must include context.policy_call_index or schedule.policy_calls.",
            errors=("Missing policy_call_index.",),
            metadata={"trace_id": trace_id},
        )
    calls = bundle.policy_calls
    if calls.empty or "policy_call_index" not in calls:
        return None, _check(
            "policy_call_exists",
            "failed",
            f"Trace {trace_id!r} has no policy_calls table rows.",
            errors=("No policy_calls table rows.",),
            metadata={"trace_id": trace_id, "policy_call_index": policy_call_index},
        )
    call_ids = pd.to_numeric(calls["policy_call_index"], errors="coerce")
    matches = calls.loc[call_ids == policy_call_index]
    if matches.empty:
        return None, _check(
            "policy_call_exists",
            "failed",
            f"Policy call {policy_call_index} is not recorded for trace {trace_id!r}.",
            errors=(f"Unknown policy_call_index {policy_call_index}.",),
            metadata={"trace_id": trace_id, "policy_call_index": policy_call_index},
        )
    record = _jsonable_record(matches.iloc[0].to_dict())
    return record, _check(
        "policy_call_exists",
        "ok",
        f"Policy call {policy_call_index} exists for trace {trace_id!r}.",
        metadata={
            "trace_id": trace_id,
            "policy_call_index": policy_call_index,
            "policy_call": record,
        },
    )


def _stored_action_check(bundle: TraceBundle | None) -> PreflightCheck:
    names = _array_names(bundle)
    available = sorted({"executed_actions", "action"} & names)
    if available:
        return _check(
            "stored_action_exists",
            "ok",
            f"Stored action array exists: {available[0]}.",
            metadata={"array_names": available},
        )
    return _check(
        "stored_action_exists",
        "failed",
        "No stored executed action array is declared for this trace.",
        errors=("Missing executed_actions/action array.",),
    )


def _stored_action_chunk_check(bundle: TraceBundle | None) -> PreflightCheck:
    if "action_chunks" in _array_names(bundle):
        return _check(
            "stored_action_chunk_exists",
            "ok",
            "Stored action chunk array exists.",
            metadata={
                "array_name": "action_chunks",
                "array_record": _array_record(bundle, "action_chunks"),
            },
        )
    return _check(
        "stored_action_chunk_exists",
        "failed",
        "No stored action_chunks array is declared for this trace.",
        errors=("Missing action_chunks array.",),
    )


def _source_artifact_check(
    dataset: TraceDataset,
    payload: Mapping[str, Any],
    target: Mapping[str, Any],
) -> PreflightCheck:
    source_artifact_id = _source_artifact_id(payload, target)
    target_kind = _text(target.get("kind"))
    artifact_required = bool(source_artifact_id) or target_kind in ARTIFACT_DERIVED_TARGET_KINDS
    if not artifact_required:
        return _check(
            "source_artifact_exists",
            "skipped",
            "Target does not require a source artifact.",
            ok=None,
            metadata={"target_kind": target_kind},
        )
    if not source_artifact_id:
        return _check(
            "source_artifact_exists",
            "failed",
            f"Target kind {target_kind!r} requires source_artifact_id.",
            errors=("Missing source_artifact_id.",),
            metadata={"target_kind": target_kind},
        )
    try:
        artifact = dataset.load_artifact(source_artifact_id)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return _check(
            "source_artifact_exists",
            "failed",
            f"Source artifact {source_artifact_id!r} is not present in this dataset.",
            errors=(str(exc),),
            metadata={"source_artifact_id": source_artifact_id},
        )
    return _check(
        "source_artifact_exists",
        "ok",
        f"Source artifact {source_artifact_id!r} exists.",
        metadata={
            "source_artifact_id": source_artifact_id,
            "artifact_type": artifact.artifact_type,
            "scope": artifact.scope,
            "source_trace_ids": list(artifact.source_trace_ids),
        },
    )


def _target_site_check(
    dataset: TraceDataset,
    trace_id: str | None,
    target: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, PreflightCheck]:
    target_site = _target_site_address(target)
    if not target_site:
        return None, _check(
            "target_site_declared_in_model_site_index",
            "failed",
            "Target must include model_site, site_id, or module_path.",
            errors=("Missing target model-site address.",),
        )

    model_sites = dataset.model_site_index
    if model_sites.empty:
        return None, _check(
            "target_site_declared_in_model_site_index",
            "failed",
            "Dataset has no model_site_index rows.",
            errors=("Missing model_site_index rows.",),
            metadata={"target_site": target_site},
        )
    candidates = model_sites
    if trace_id and "trace_id" in candidates:
        trace_rows = candidates.loc[candidates["trace_id"].astype(str) == trace_id]
        if not trace_rows.empty:
            candidates = trace_rows
    matches = _target_site_matches(candidates, target_site)
    if matches.empty:
        return None, _check(
            "target_site_declared_in_model_site_index",
            "failed",
            f"Target site {target_site!r} is not declared in model_site_index.",
            errors=(f"Unknown target site {target_site!r}.",),
            metadata={"target_site": target_site, "trace_id": trace_id},
        )
    record = _jsonable_record(matches.iloc[0].to_dict())
    return record, _check(
        "target_site_declared_in_model_site_index",
        "ok",
        f"Target site {target_site!r} is declared in model_site_index.",
        metadata={"target_site": target_site, "trace_id": trace_id, "model_site": record},
    )


def _target_site_matches(model_sites: pd.DataFrame, target_site: str) -> pd.DataFrame:
    masks = []
    for column in ("site_id", "name", "model_site", "model_path", "module"):
        if column in model_sites:
            masks.append(model_sites[column].astype(str) == target_site)
    if not masks:
        return model_sites.iloc[0:0]
    mask = masks[0]
    for item in masks[1:]:
        mask = mask | item
    return model_sites.loc[mask]


def _token_space_check(
    bundle: TraceBundle | None,
    target: Mapping[str, Any],
    target_site_record: Mapping[str, Any] | None,
) -> tuple[str | None, PreflightCheck]:
    token_space = _target_token_space(target, target_site_record)
    if not token_space:
        return None, _check(
            "token_space_declared",
            "failed",
            "Target site does not declare token_space/token_space_id.",
            errors=("Missing target token space.",),
        )
    if bundle is None:
        return token_space, _check(
            "token_space_declared",
            "failed",
            f"Token space {token_space!r} cannot be checked because the trace is missing.",
            errors=("Missing trace bundle.",),
            metadata={"token_space": token_space},
        )
    spaces = bundle.token_spaces
    if spaces.empty or "token_space_id" not in spaces:
        return token_space, _check(
            "token_space_declared",
            "failed",
            f"Token space {token_space!r} is requested, but this trace has no token_spaces rows.",
            errors=("Missing token_spaces table rows.",),
            metadata={"token_space": token_space},
        )
    matches = spaces.loc[spaces["token_space_id"].astype(str) == token_space]
    if matches.empty:
        return token_space, _check(
            "token_space_declared",
            "failed",
            f"Token space {token_space!r} is not declared for this trace.",
            errors=(f"Unknown token_space_id {token_space!r}.",),
            metadata={
                "token_space": token_space,
                "available_token_spaces": sorted(spaces["token_space_id"].astype(str).unique()),
            },
        )
    return token_space, _check(
        "token_space_declared",
        "ok",
        f"Token space {token_space!r} is declared.",
        metadata={
            "token_space": token_space,
            "token_space_record": _jsonable_record(matches.iloc[0].to_dict()),
        },
    )


def _action_decoder_check(
    *,
    bundle: TraceBundle | None,
    policy_call_record: Mapping[str, Any] | None,
    action_chunk_available: bool,
) -> PreflightCheck:
    fields = {}
    for key in ("model_call_kind", "action_generator_kind", "action_horizon", "action_dim"):
        value = (policy_call_record or {}).get(key)
        if not _missing(value):
            fields[key] = value
    if fields.get("action_horizon") and fields.get("action_dim"):
        return _check(
            "action_decoder_metadata_available",
            "ok",
            "Policy-call decoder metadata declares action horizon and action dimension.",
            metadata={"decoder_metadata": fields},
        )
    chunk_record = _array_record(bundle, "action_chunks")
    if action_chunk_available and chunk_record:
        return _check(
            "action_decoder_metadata_available",
            "partial",
            "Raw action chunks are present, but policy-call decoder metadata is incomplete.",
            warnings=("Missing action_horizon/action_dim metadata on policy_call.",),
            metadata={"decoder_metadata": fields, "action_chunk_record": chunk_record},
        )
    return _check(
        "action_decoder_metadata_available",
        "failed",
        "No action decoder metadata or action_chunks array is available.",
        errors=("Missing action decoder metadata.",),
        metadata={"decoder_metadata": fields},
    )


def _action_basis_check(
    *,
    bundle: TraceBundle | None,
    outcome: Mapping[str, Any],
    action_chunk_available: bool,
) -> tuple[dict[str, Any], PreflightCheck]:
    requested = _requested_basis(outcome)
    metadata = _action_basis_metadata(bundle)
    basis_status: dict[str, Any] = {}
    missing_named: list[str] = []

    for basis in requested:
        if basis in {"raw", "normalized", "action_chunks"}:
            available = action_chunk_available
            basis_status[basis] = {
                "available": available,
                "source": "action_chunks" if available else None,
            }
            if not available:
                missing_named.append(basis)
            continue
        available = basis.lower() in metadata["labels"]
        basis_status[basis] = {
            "available": available,
            "source": "action_normalization" if available else None,
            "available_labels": sorted(metadata["labels"]),
        }
        if not available:
            missing_named.append(basis)

    if not missing_named:
        return basis_status, _check(
            "action_basis_metadata_available",
            "ok",
            "Requested action basis metadata is available.",
            metadata={"basis": requested, "basis_status": basis_status},
        )
    if any(basis in {"raw", "normalized", "action_chunks"} for basis in missing_named):
        return basis_status, _check(
            "action_basis_metadata_available",
            "failed",
            "Raw action basis is unavailable because action_chunks are missing.",
            errors=("Missing raw action basis.",),
            metadata={"basis": requested, "basis_status": basis_status},
        )
    return basis_status, _check(
        "action_basis_metadata_available",
        "partial",
        "Raw actions are inspectable, but requested named action basis metadata is incomplete.",
        warnings=(f"Missing named action basis metadata: {', '.join(missing_named)}.",),
        metadata={"basis": requested, "basis_status": basis_status},
    )


def _runtime_adapter_check(
    *,
    payload: Mapping[str, Any],
    request: Mapping[str, Any],
    target: Mapping[str, Any],
    policy_call_record: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], PreflightCheck]:
    runtime_payload = _mapping(payload.get("runtime")) or _mapping(request.get("runtime"))
    target_metadata = _mapping(target.get("metadata"))
    adapter = _first_text(
        payload.get("runtime_adapter"),
        request.get("runtime_adapter"),
        runtime_payload.get("adapter"),
        target_metadata.get("runtime_adapter"),
        target.get("runtime_adapter"),
    )
    model_family = _first_text(
        target.get("model_family"),
        payload.get("model_family"),
        request.get("model_family"),
        runtime_payload.get("model_family"),
        (policy_call_record or {}).get("model_family"),
    )
    model_id = _first_text(
        target.get("model_id"),
        payload.get("model_id"),
        request.get("model_id"),
        runtime_payload.get("model_id"),
        (policy_call_record or {}).get("model_id"),
    )
    runtime_resolution = {
        "adapter": adapter,
        "model_family": model_family,
        "model_id": model_id,
        "can_rerun": False,
        "reason": "Runtime-free preflight does not import or instantiate model runtimes.",
    }
    if adapter:
        return runtime_resolution, _check(
            "runtime_adapter_declared",
            "ok",
            f"Runtime adapter {adapter!r} is declared.",
            metadata=runtime_resolution,
        )
    if model_family or model_id:
        return runtime_resolution, _check(
            "runtime_adapter_declared",
            "partial",
            "Model identity is recorded, but no runtime_adapter is declared.",
            warnings=("Declare runtime_adapter before attempting a live rerun.",),
            metadata=runtime_resolution,
        )
    return runtime_resolution, _check(
        "runtime_adapter_declared",
        "failed",
        "No runtime_adapter, model_family, or model_id is declared.",
        errors=("Missing runtime adapter declaration.",),
        metadata=runtime_resolution,
    )


def _model_runtime_check(runtime_resolution: Mapping[str, Any]) -> PreflightCheck:
    adapter = _text(runtime_resolution.get("adapter")) or "unknown"
    return _check(
        "model_runtime_available",
        "unavailable",
        (
            f"Model runtime for adapter {adapter!r} is not loaded in the dashboard environment. "
            "Use the PI0.5 capture wrappers or matching capture virtualenv for live reruns."
        ),
        ok=False,
        warnings=(
            "Runtime-free preflight intentionally avoids "
            "PI0.5/Torch/LeRobot/LIBERO imports.",
        ),
        metadata={"adapter": adapter, "can_rerun": False},
    )


def _runtime_environment_check() -> tuple[dict[str, Any], PreflightCheck]:
    loaded = sorted(name for name in HEAVY_RUNTIME_MODULES if name in sys.modules)
    environment = {
        "mode": "metadata_preflight",
        "heavy_runtime_modules_checked": list(HEAVY_RUNTIME_MODULES),
        "heavy_runtime_modules_loaded": loaded,
        "safe_for_dashboard": True,
    }
    return environment, _check(
        "runtime_environment_safe",
        "ok",
        "Preflight stayed on the metadata-only dashboard path.",
        warnings=(
            (f"Heavy modules were already loaded elsewhere: {', '.join(loaded)}.",)
            if loaded
            else ()
        ),
        metadata=environment,
    )


def _bundle_or_none(dataset: TraceDataset, trace_id: str | None) -> TraceBundle | None:
    if not trace_id:
        return None
    try:
        return dataset.bundle(trace_id)
    except KeyError:
        return None


def _context_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    baseline = _mapping(payload.get("baseline"))
    return (
        _mapping(payload.get("context"))
        or _mapping(baseline.get("context"))
        or _mapping(payload.get("provenance"))
    )


def _target_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(payload.get("target"))


def _request_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    intervention = _mapping(payload.get("intervention"))
    return _mapping(payload.get("request")) or _mapping(intervention.get("request"))


def _outcome_payload(
    payload: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
) -> Mapping[str, Any]:
    return _mapping(payload.get("outcome")) or _mapping(request.get("outcome"))


def _trace_id(payload: Mapping[str, Any], context: Mapping[str, Any]) -> str | None:
    recipient = _mapping(payload.get("recipient"))
    recipient_trace = _mapping(recipient.get("trace"))
    return _first_text(
        context.get("trace_id"),
        payload.get("trace_id"),
        _mapping(payload.get("provenance")).get("trace_id"),
        recipient_trace.get("trace_id"),
    )


def _policy_call_index(
    payload: Mapping[str, Any],
    context: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> int | None:
    value = _first_present(
        context.get("policy_call_index"),
        payload.get("policy_call_index"),
        _mapping(payload.get("provenance")).get("policy_call_index"),
    )
    if value is not None:
        return _int_or_none(value)
    policy_calls = schedule.get("policy_calls")
    if isinstance(policy_calls, (list, tuple)) and policy_calls:
        return _int_or_none(policy_calls[0])
    return None


def _source_artifact_id(payload: Mapping[str, Any], target: Mapping[str, Any]) -> str | None:
    return _first_text(
        target.get("source_artifact_id"),
        payload.get("source_artifact_id"),
        _mapping(payload.get("provenance")).get("source_artifact_id"),
    )


def _target_site_address(target: Mapping[str, Any]) -> str | None:
    return _first_text(
        target.get("site_id"),
        target.get("model_site"),
        target.get("name"),
        target.get("model_path"),
        target.get("module_path"),
    )


def _target_token_space(
    target: Mapping[str, Any],
    target_site_record: Mapping[str, Any] | None,
) -> str | None:
    token_selector = _mapping(target.get("token_selector"))
    return _first_text(
        target.get("token_space"),
        target.get("token_space_id"),
        token_selector.get("token_space"),
        token_selector.get("token_space_id"),
        (target_site_record or {}).get("token_space_id"),
        (target_site_record or {}).get("token_space"),
    )


def _requested_basis(outcome: Mapping[str, Any]) -> tuple[str, ...]:
    basis = outcome.get("basis", ("raw",))
    if isinstance(basis, str):
        return (basis,)
    if isinstance(basis, (list, tuple)):
        out = tuple(_text(item) for item in basis if _text(item))
        return out or ("raw",)
    return ("raw",)


def _action_basis_metadata(bundle: TraceBundle | None) -> dict[str, Any]:
    labels: set[str] = set()
    if bundle is None:
        return {"labels": labels}
    table = bundle.action_normalization
    if table.empty:
        return {"labels": labels}
    for record in table.to_dict("records"):
        for key in ("action_dim_names", "action_labels", "basis", "basis_names"):
            labels.update(_label_tokens(record.get(key)))
        metadata = _json_parse(record.get("metadata"))
        if isinstance(metadata, Mapping):
            for key in ("action_dim_names", "action_labels", "basis", "basis_names"):
                labels.update(_label_tokens(metadata.get(key)))
    return {"labels": {label.lower() for label in labels if label}}


def _label_tokens(value: Any) -> set[str]:
    parsed = _json_parse(value)
    if isinstance(parsed, Mapping):
        values = parsed.values()
    elif isinstance(parsed, (list, tuple, set)):
        values = parsed
    elif parsed is None or _missing(parsed):
        values = ()
    else:
        values = (parsed,)
    labels: set[str] = set()
    for item in values:
        text = _text(item).lower()
        if not text:
            continue
        labels.add(text)
        labels.update(part for part in text.replace("-", "_").split("_") if part)
    return labels


def _array_names(bundle: TraceBundle | None) -> set[str]:
    if bundle is None or bundle.array_index.empty or "name" not in bundle.array_index:
        return set()
    return set(bundle.array_index["name"].astype(str))


def _array_record(bundle: TraceBundle | None, name: str) -> dict[str, Any]:
    if bundle is None or bundle.array_index.empty or "name" not in bundle.array_index:
        return {}
    matches = bundle.array_index.loc[bundle.array_index["name"].astype(str) == name]
    if matches.empty:
        return {}
    return _jsonable_record(matches.iloc[0].to_dict())


def _check(
    name: str,
    status: str,
    message: str,
    *,
    ok: bool | None = None,
    warnings: tuple[str, ...] = (),
    errors: tuple[str, ...] = (),
    metadata: Mapping[str, Any] | None = None,
) -> PreflightCheck:
    resolved_ok = status == "ok" if ok is None and status != "skipped" else ok
    return PreflightCheck(
        name=name,
        status=status,
        message=message,
        ok=resolved_ok,
        warnings=warnings,
        errors=errors,
        metadata=metadata or {},
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if not _missing(value):
            return value
    return None


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = _text(value)
        if text:
            return text
    return None


def _text(value: Any) -> str:
    if _missing(value):
        return ""
    return str(value).strip()


def _int_or_none(value: Any) -> int | None:
    if _missing(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null", "nan"}
    if isinstance(value, float):
        return not math.isfinite(value)
    return False


def _json_parse(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _jsonable_record(record: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in record.items():
        if _missing(value):
            out[str(key)] = None
        elif hasattr(value, "item"):
            out[str(key)] = value.item()
        else:
            out[str(key)] = _json_parse(value)
    return out


def _dedupe(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return tuple(out)


__all__ = [
    "HEAVY_RUNTIME_MODULES",
    "PREFLIGHT_CHECK_NAMES",
    "intervention_preflight",
]
