"""Run a resumable PI0.5 counterfactual patch study with one model load."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

import numpy as np

from vla_lens.interventions import (
    CounterfactualPairManifest,
    PatchStudySpec,
    PatchStudyStore,
    build_intervention_sweep,
    build_patch_trial_request,
    expand_patch_study,
    intervention_sweep_to_lens_artifact,
    patch_study_request_sha256,
)
from vla_lens.interventions.patch_study_analysis import save_patch_study_analysis
from vla_lens.pi05.intervention_executor import (
    PI05ActionInterventionExecutor,
    build_pi05_action_intervention_executor,
)
from vla_lens.pi05.intervention_preflight import pi05_intervention_preflight
from vla_lens.pi05.intervention_runner import (
    _intervention_gate,
    _replay_drift_records,
    _validate_tolerance,
)
from vla_lens.pi05.intervention_runtime import run_pi05_intervention
from vla_lens.traces import TraceDataset
from vla_lens.workbench import save_intervention_run

ExecutorFactory = Callable[..., PI05ActionInterventionExecutor]


class ReplayGateError(RuntimeError):
    """The study did not have a trustworthy recipient replay baseline."""


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        report, exit_code = run_patch_study_job(args)
    except Exception as exc:
        report = {
            "schema_kind": "vla_lens.pi05_patch_study_report",
            "schema_version": 1,
            "status": "failed",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        if args.output is not None:
            _write_json_atomic(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(2) from exc
    print(json.dumps(report, indent=2, sort_keys=True))
    if exit_code:
        raise SystemExit(exit_code)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--study", type=Path, required=True, help="Patch-study job JSON")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output", type=Path, help="Optional summary report JSON")
    parser.add_argument("--model-id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--noop-repeats", type=int, default=3)
    parser.add_argument("--max-noop-l2", type=float)
    parser.add_argument("--max-noop-max-abs", type=float)
    parser.add_argument(
        "--run-study",
        action="store_true",
        help="Load PI0.5 and execute; without this flag only inspect the plan",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry trials already recorded as failures",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--no-workbench",
        action="store_true",
        help="Do not add completed runs and the sweep to dataset indexes",
    )
    return parser.parse_args(argv)


def run_patch_study_job(
    args: argparse.Namespace,
    *,
    executor_factory: ExecutorFactory = build_pi05_action_intervention_executor,
) -> tuple[dict[str, Any], int]:
    """Inspect or execute one study, checkpointing every materialized trial."""
    if int(args.noop_repeats) < 1:
        raise ValueError("--noop-repeats must be at least 1")
    _validate_tolerance("--max-noop-l2", args.max_noop_l2)
    _validate_tolerance("--max-noop-max-abs", args.max_noop_max_abs)
    job = _read_json(args.study)
    study = PatchStudySpec.from_dict(_required_mapping(job.get("study"), "study"))
    pairs = tuple(
        CounterfactualPairManifest.from_dict(_required_mapping(item, "pair"))
        for item in job.get("pairs", ())
    )
    template = _required_mapping(job.get("request_template"), "request_template")
    plan = expand_patch_study(study, pairs)
    pairs_by_id = {pair.pair_id: pair for pair in pairs}
    dataset = TraceDataset.open(args.dataset_root)
    model_ids = _study_model_ids(dataset, pairs)
    if args.model_id is None and len(model_ids) != 1:
        raise ValueError(
            "one-load patch studies require every pair to use the same recorded model_id"
        )
    requests = tuple(
        build_patch_trial_request(
            template,
            study=study,
            pair=pairs_by_id[trial.pair_id],
            trial=trial,
        )
        for trial in plan
    )
    preflights = tuple(
        pi05_intervention_preflight(
            dataset,
            request,
            runtime_available=True,
        )
        for request in requests
    )
    failed_preflights = [
        plan[index].trial_id
        for index, preflight in enumerate(preflights)
        if preflight.status == "failed"
    ]
    report: dict[str, Any] = {
        "schema_kind": "vla_lens.pi05_patch_study_report",
        "schema_version": 1,
        "study_id": study.study_id,
        "status": "inspected",
        "request_sha256": patch_study_request_sha256(job),
        "planned_trial_count": len(plan),
        "pair_count": len(pairs),
        "layers": sorted({trial.layer for trial in plan}),
        "token_regions": list(dict.fromkeys(trial.token_region for trial in plan)),
        "controls": list(study.controls),
        "model_ids": sorted(model_ids),
        "failed_preflight_trial_ids": failed_preflights,
        "model_load_strategy": "one load for the study; one donor cache fill per pair",
        "permanent_data": [
            "recipient/donor/patched/control actions",
            "trial and pair tables",
            "decisions, failures, hashes, and reconstruction plan",
        ],
        "disposable_data": "donor hidden-state caches remain in memory only",
    }
    if args.output is not None:
        _write_json_atomic(args.output, report)
    if not args.run_study:
        return report, 0 if not failed_preflights else 2
    if failed_preflights:
        raise ValueError(
            f"patch study has {len(failed_preflights)} failed preflight trial(s)"
        )
    if args.max_noop_l2 is None or args.max_noop_max_abs is None:
        raise ValueError(
            "--run-study requires explicit --max-noop-l2 and --max-noop-max-abs"
        )

    output_dir = args.output_dir or (
        dataset.root / "vla_lens" / "patch_studies" / _safe_id(study.study_id)
    )
    store = PatchStudyStore(
        output_dir,
        study=study,
        pairs=pairs,
        plan=plan,
        request_sha256=patch_study_request_sha256(job),
    )
    store.prepare()
    request_by_id = {
        trial.trial_id: request
        for trial, request in zip(plan, requests, strict=True)
    }
    runtime = None
    last_executor: PI05ActionInterventionExecutor | None = None
    pair_gates: dict[str, Any] = {}
    try:
        for pair_id in study.pair_ids:
            pending = [
                trial
                for trial in plan
                if trial.pair_id == pair_id
                and not store.is_completed(trial.trial_id)
                and (args.retry_failed or not store.is_failed(trial.trial_id))
            ]
            if not pending:
                continue
            executor: PI05ActionInterventionExecutor | None = None
            try:
                executor = executor_factory(
                    dataset,
                    request_by_id[pending[0].trial_id],
                    device=str(args.device),
                    dtype=str(args.dtype),
                    model_id=args.model_id,
                    runtime=runtime,
                )
                last_executor = executor
                if runtime is None:
                    runtime = executor.runtime
                noop_outputs = [
                    executor.run_noop(request_by_id[pending[0].trial_id])
                    for _ in range(int(args.noop_repeats))
                ]
                stored = np.asarray(
                    executor.replay_inputs.stored_action_chunk, dtype=np.float32
                )
                replay_trials = _replay_drift_records(
                    stored,
                    [np.asarray(output.action_chunk, dtype=np.float32) for output in noop_outputs],
                )
                gate = _intervention_gate(
                    SimpleNamespace(
                        run_intervention=True,
                        max_noop_l2=args.max_noop_l2,
                        max_noop_max_abs=args.max_noop_max_abs,
                    ),
                    executor.replay_inputs.initial_noise_exactness,
                    "ok",
                    replay_trials,
                )
                gate = {**gate, "trials": replay_trials}
                pair_gates[pair_id] = gate
                if not gate["passed"]:
                    raise ReplayGateError("; ".join(gate["reasons"]))
                prime_cache = getattr(executor, "prime_donor_cache", None)
                if not callable(prime_cache):
                    raise TypeError("patch-study executor must support prime_donor_cache")
                prime_cache(sorted({trial.layer for trial in pending}))
                for trial in pending:
                    try:
                        result = run_pi05_intervention(
                            dataset,
                            request_by_id[trial.trial_id],
                            executor=executor,
                            save=False,
                            claim_gate=gate,
                            noop_output=noop_outputs[0],
                        )
                        store.record_run(trial, result.run, result.arrays)
                        if not args.no_workbench:
                            save_intervention_run(dataset, result.run.to_workbench_spec())
                    except Exception as exc:
                        store.record_failure(trial, exc)
                        if args.fail_fast:
                            raise
            except Exception as exc:
                unresolved = [trial for trial in pending if not store.is_completed(trial.trial_id)]
                for trial in unresolved:
                    store.record_failure(trial, exc)
                if args.fail_fast:
                    raise
            finally:
                close = getattr(executor, "close", None)
                if callable(close):
                    close()
                if executor is last_executor:
                    last_executor = None
    finally:
        close = getattr(last_executor, "close", None)
        if callable(close):
            close()

    artifact = store.finalize()
    analysis = save_patch_study_analysis(store.root)
    runs = store.intervention_runs()
    sweep_artifact_id = None
    if runs and not args.no_workbench:
        decisions = [decision.to_dict() for decision in artifact.decisions]
        sweep = build_intervention_sweep(
            sweep_id=f"sweep-{study.study_id}",
            runs=runs,
            axes=study.axes,
            cohort={"pair_ids": list(study.pair_ids)},
            controls=decisions,
            provenance={
                "patch_study_id": study.study_id,
                "patch_study_root": str(store.root),
                "request_sha256": store.request_sha256,
                "source_trace_ids": [
                    trace_id
                    for pair in pairs
                    for trace_id in _pair_trace_ids(pair)
                ],
            },
        )
        _write_json_atomic(store.root / "sweep.json", sweep.to_dict())
        sweep_artifact_id = dataset.save_artifact(
            intervention_sweep_to_lens_artifact(sweep)
        ).artifact_id
    verdicts = Counter(decision.verdict for decision in artifact.decisions)
    progress = store.progress()
    report.update(
        {
            "status": progress.status,
            "study_root": str(store.root),
            "completed_trial_count": len(progress.completed_trial_ids),
            "failed_trial_count": len(progress.failed_trial_ids),
            "completed_trial_ids": list(progress.completed_trial_ids),
            "failed_trial_ids": list(progress.failed_trial_ids),
            "verdict_counts": dict(sorted(verdicts.items())),
            "pair_replay_gates": pair_gates,
            "sweep_artifact_id": sweep_artifact_id,
            "analysis_path": str(store.root / "analysis.json"),
            "analysis_headline": analysis.get("headline"),
            "model_loaded_once": runtime is not None,
        }
    )
    if args.output is not None:
        _write_json_atomic(args.output, report)
    exit_code = 0 if progress.status == "completed" else 4
    return report, exit_code


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _required_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"patch study job requires {field}")
    return value


def _study_model_ids(
    dataset: TraceDataset,
    pairs: tuple[CounterfactualPairManifest, ...],
) -> set[str]:
    model_ids = {
        str(dataset.bundle(trace_id).manifest.model_id or "").strip()
        for pair in pairs
        for trace_id in _pair_trace_ids(pair)
    }
    if "" in model_ids:
        raise ValueError("every counterfactual trace must record model_id")
    return model_ids


def _pair_trace_ids(pair: CounterfactualPairManifest) -> tuple[str, str]:
    recipient = pair.recipient.to_dict()
    donor = pair.donor.to_dict()
    recipient_trace = str(_required_mapping(recipient.get("trace"), "recipient.trace")["trace_id"])
    donor_trace = str(_required_mapping(donor.get("trace"), "donor.trace")["trace_id"])
    return recipient_trace, donor_trace


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _safe_id(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    )


__all__ = ["main", "parse_args", "run_patch_study_job"]


if __name__ == "__main__":
    main()
