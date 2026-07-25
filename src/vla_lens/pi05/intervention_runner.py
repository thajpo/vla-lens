"""One-shot, replay-gated PI0.5 intervention command."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from vla_lens.interventions import ActionInterventionExecutor, TargetSpec
from vla_lens.pi05.intervention_executor import build_pi05_action_intervention_executor
from vla_lens.pi05.intervention_preflight import pi05_intervention_preflight
from vla_lens.pi05.intervention_runtime import run_pi05_intervention
from vla_lens.pi05.probe_direction import resolve_object_roi_probe_direction
from vla_lens.pi05.replay import policy_call_replay_inputs
from vla_lens.traces import TraceDataset

ExecutorFactory = Callable[..., ActionInterventionExecutor]


def main(argv: list[str] | None = None) -> None:
    """Measure replay drift, then run an explicitly gated engineering hook smoke."""
    args = parse_args(argv)
    try:
        report, exit_code = run_job(args)
    except Exception as exc:
        report = {
            "schema_kind": "vla_lens.pi05_intervention_report",
            "schema_version": 1,
            "status": "failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        if args.output is not None:
            _write_report(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(2) from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    if exit_code:
        raise SystemExit(exit_code)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--request", type=Path, required=True, help="Intervention request JSON")
    parser.add_argument(
        "--output",
        type=Path,
        help="Report JSON path (default: dataset vla_lens/intervention_reports)",
    )
    parser.add_argument("--model-id", help="Override the model checkpoint recorded by the trace")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--noop-repeats", type=int, default=3)
    parser.add_argument(
        "--run-intervention",
        action="store_true",
        help="Run the requested synthetic or artifact-derived hook after replay passes",
    )
    parser.add_argument("--max-noop-l2", type=float)
    parser.add_argument("--max-noop-max-abs", type=float)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect request, trace, noise, and preflight without loading PI0.5",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save intervention artifact")
    return parser.parse_args(argv)


def run_job(
    args: argparse.Namespace,
    *,
    executor_factory: ExecutorFactory = build_pi05_action_intervention_executor,
) -> tuple[dict[str, Any], int]:
    """Execute one request and return its persisted report and process exit code."""
    if int(args.noop_repeats) < 1:
        raise ValueError("--noop-repeats must be at least 1")
    _validate_tolerance("--max-noop-l2", args.max_noop_l2)
    _validate_tolerance("--max-noop-max-abs", args.max_noop_max_abs)
    payload = _read_request(args.request)
    request_sha256 = _request_sha256(payload)
    payload = dict(payload)
    payload.setdefault("device", str(args.device))
    if args.model_id:
        payload.setdefault("model_id", str(args.model_id))

    dataset = TraceDataset.open(args.dataset_root)
    trace_id, policy_call_index = _context_selection(payload)
    bundle = dataset.bundle(trace_id)
    replay_inputs = policy_call_replay_inputs(bundle, policy_call_index)
    preflight = pi05_intervention_preflight(
        dataset,
        payload,
        runtime_available=not bool(args.dry_run),
    )
    direction_resolution = None
    if _execution_mode(payload) == "artifact_probe_direction":
        target = TargetSpec.from_dict(_mapping(payload.get("target")))
        direction_resolution = resolve_object_roi_probe_direction(
            dataset,
            target,
            trace_id=trace_id,
            policy_call_index=policy_call_index,
        ).provenance
    report: dict[str, Any] = {
        "schema_kind": "vla_lens.pi05_intervention_report",
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "inspected" if args.dry_run else "running",
        "mode": "dry_run" if args.dry_run else "replay_measurement",
        "dataset_root": str(dataset.root),
        "request": {
            "path": str(args.request),
            "sha256": request_sha256,
        },
        "replay_inputs": replay_inputs.summary(),
        "preflight": preflight.to_dict(),
        "runtime": {
            "device": str(args.device),
            "dtype": str(args.dtype),
            "model_id_override": args.model_id,
        },
    }
    if direction_resolution is not None:
        report["probe_direction_resolution"] = dict(direction_resolution)
    report_path = args.output or _default_report_path(dataset, request_sha256)
    report["report_path"] = str(report_path)
    if args.dry_run:
        _write_report(report_path, report)
        return report, 0 if preflight.status != "failed" else 2

    executor: ActionInterventionExecutor | None = None
    try:
        executor = executor_factory(
            dataset,
            payload,
            device=str(args.device),
            dtype=str(args.dtype),
            model_id=args.model_id,
        )
        noops = [
            np.asarray(executor.run_noop(payload).action_chunk, dtype=np.float32)
            for _ in range(int(args.noop_repeats))
        ]
        stored = np.asarray(replay_inputs.stored_action_chunk, dtype=np.float32)
        replay_trials = _replay_drift_records(stored, noops)
        report["noop_replay"] = {
            "repeats": len(noops),
            "trials": replay_trials,
            "deterministic_across_repeats": all(
                trial["delta_from_first"]["exact_match"] for trial in replay_trials
            ),
        }
        gate = _intervention_gate(
            args,
            replay_inputs.initial_noise_exactness,
            preflight.status,
            replay_trials,
        )
        report["intervention_gate"] = gate
        if not args.run_intervention:
            report["status"] = "replay_measured"
            _write_report(report_path, report)
            return report, 0
        if not gate["passed"]:
            report["status"] = "blocked_by_replay_gate"
            report["mode"] = "intervention_blocked"
            _write_report(report_path, report)
            return report, 3

        result = run_pi05_intervention(
            dataset,
            payload,
            executor=executor,
            save=not bool(args.no_save),
            claim_gate=gate,
        )
        report["status"] = "completed"
        claim = dict(result.run.claim)
        report["mode"] = _execution_mode(payload)
        report["claim_eligible"] = bool(claim.get("method_eligible", False))
        report["scientific_verdict"] = claim.get(
            "scientific_verdict", "not_eligible"
        )
        report["intervention_result"] = result.to_dict()
        _write_report(report_path, report)
        return report, 0
    finally:
        close = getattr(executor, "close", None)
        if callable(close):
            close()


def _replay_drift_records(
    stored: np.ndarray,
    noops: list[np.ndarray],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    first = noops[0]
    for index, action in enumerate(noops):
        if action.shape != stored.shape:
            raise ValueError(
                f"No-op action shape {action.shape} does not match stored chunk {stored.shape}"
            )
        records.append(
            {
                "repeat": index,
                "delta_from_stored": _drift(stored, action),
                "delta_from_first": _drift(first, action),
            }
        )
    return records


def _drift(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    if not np.all(np.isfinite(reference)):
        raise ValueError("Stored action chunk contains non-finite values")
    if not np.all(np.isfinite(candidate)):
        raise ValueError("No-op replay action chunk contains non-finite values")
    delta = candidate - reference
    return {
        "l2": float(np.linalg.norm(delta)),
        "max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "mean_abs": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "exact_match": bool(np.array_equal(reference, candidate)),
    }


def _validate_tolerance(name: str, value: float | None) -> None:
    if value is None:
        return
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")


def _intervention_gate(
    args: argparse.Namespace,
    noise_exactness: str,
    preflight_status: str,
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    if not args.run_intervention:
        reasons.append("Intervention was not requested; report contains replay measurement only.")
    if preflight_status != "ok":
        reasons.append(f"Preflight status must be ok, found {preflight_status}.")
    if noise_exactness != "exact":
        reasons.append(
            f"Intervention requires exact captured initial noise, found {noise_exactness}."
        )
    if args.max_noop_l2 is None or args.max_noop_max_abs is None:
        reasons.append(
            "Intervention requires explicit --max-noop-l2 and --max-noop-max-abs tolerances."
        )
    if args.max_noop_l2 is not None and any(
        trial["delta_from_stored"]["l2"] > float(args.max_noop_l2) for trial in trials
    ):
        reasons.append("At least one no-op replay exceeded the L2 tolerance.")
    if args.max_noop_max_abs is not None and any(
        trial["delta_from_stored"]["max_abs"] > float(args.max_noop_max_abs)
        for trial in trials
    ):
        reasons.append("At least one no-op replay exceeded the max-absolute tolerance.")
    return {
        "requested": bool(args.run_intervention),
        "passed": bool(args.run_intervention) and not reasons,
        "thresholds": {
            "max_noop_l2": args.max_noop_l2,
            "max_noop_max_abs": args.max_noop_max_abs,
        },
        "reasons": reasons,
    }


def _read_request(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Intervention request JSON must contain an object")
    return payload


def _request_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _context_selection(payload: Mapping[str, Any]) -> tuple[str, int]:
    baseline = _mapping(payload.get("baseline"))
    context = _mapping(payload.get("context")) or _mapping(baseline.get("context"))
    trace_id = str(context.get("trace_id") or "").strip()
    if not trace_id:
        raise ValueError("PI0.5 intervention requires context.trace_id")
    return trace_id, int(context.get("policy_call_index") or 0)


def _execution_mode(payload: Mapping[str, Any]) -> str:
    intervention = _mapping(payload.get("intervention"))
    request = _mapping(payload.get("request")) or _mapping(intervention.get("request"))
    operator = _mapping(request.get("operator"))
    parameters = _mapping(operator.get("parameters"))
    return str(parameters.get("mode") or "unspecified")


def _default_report_path(dataset: TraceDataset, request_sha256: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return (
        dataset.root
        / "vla_lens"
        / "intervention_reports"
        / f"pi05_{timestamp}_{request_sha256[:10]}.json"
    )


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["main", "parse_args", "run_job"]
