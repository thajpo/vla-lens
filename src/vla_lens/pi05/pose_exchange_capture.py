"""Capture replayable recipient/donor LIBERO pose-exchange pairs with one model load."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from vla_lens.capture import validate_lerobot_v3_dataset
from vla_lens.dataset import build_dataset_index
from vla_lens.pi05.capture_runner import (
    _resolve_capture_plan,
    _trace_id_for_seed,
    load_pi05_capture_runtime,
    namespace_for_capture_args,
    run_pi05_capture_task,
)
from vla_lens.pi05.pose_exchange_pairs import (
    build_pose_exchange_pair_manifests,
    save_pose_exchange_pair_manifests,
)
from vla_lens.traces import TraceBundle, TraceDataset

RuntimeLoader = Callable[[argparse.Namespace], Any]
CaptureTask = Callable[..., None]


@dataclass(frozen=True, slots=True)
class PoseExchangeCaptureRow:
    """One role-specific capture resolved from a pair job."""

    pair_id: str
    pair_index: int
    role: str
    trace_id: str
    paired_trace_id: str
    seed: int
    layout_id: int
    target_object: str
    distractor_object: str
    scene_mutation: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "pair_index": self.pair_index,
            "role": self.role,
            "trace_id": self.trace_id,
            "paired_trace_id": self.paired_trace_id,
            "seed": self.seed,
            "layout_id": self.layout_id,
            "target_object": self.target_object,
            "distractor_object": self.distractor_object,
            "scene_mutation": dict(self.scene_mutation),
        }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        report, exit_code = run_pose_exchange_capture_job(args)
    except Exception as exc:
        report = {
            "schema_kind": "vla_lens.pi05_pose_exchange_capture_report",
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
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional capture report JSON")
    parser.add_argument("--model-id")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--run-capture",
        action="store_true",
        help="Load PI0.5 and capture; without this flag only inspect the plan",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recapture trace IDs that already have manifests",
    )
    return parser.parse_args(argv)


def expand_pose_exchange_capture_job(job: Mapping[str, Any]) -> tuple[PoseExchangeCaptureRow, ...]:
    """Resolve stable recipient/donor trace identities for every declared pair."""
    benchmark = str(job.get("benchmark") or "libero_90")
    task_id = int(job["task_id"])
    profile = str(job.get("capture_profile") or "rollout")
    pairs = job.get("pairs")
    if not isinstance(pairs, Sequence) or isinstance(pairs, str) or not pairs:
        raise ValueError("pose-exchange capture job requires pairs")
    rows: list[PoseExchangeCaptureRow] = []
    seen_pairs: set[str] = set()
    for pair_index, raw_pair in enumerate(pairs):
        pair = _required_mapping(raw_pair, "pair")
        pair_id = str(pair.get("pair_id") or f"pose-exchange-{pair_index:02d}").strip()
        if not pair_id or pair_id in seen_pairs:
            raise ValueError("pose-exchange pair_id values must be non-empty and unique")
        seen_pairs.add(pair_id)
        seed = int(pair["seed"])
        layout_id = int(pair["layout_id"])
        target = str(pair["target_object"]).strip()
        distractor = str(pair["distractor_object"]).strip()
        if not target or not distractor or target == distractor:
            raise ValueError("pose-exchange pairs require two different named objects")
        role_args = {}
        for role in ("recipient", "donor"):
            role_args[role] = namespace_for_capture_args(
                capture_profile=profile,
                benchmark=benchmark,
                task_id=task_id,
                start_seed=seed,
                trace_variant=f"{pair_id}-{role}",
                counterfactual_role=role,
            )
        trace_ids = {
            role: _trace_id_for_seed(role_args[role], seed)
            for role in ("recipient", "donor")
        }
        for role in ("recipient", "donor"):
            mutation = (
                {"kind": "identity", "objects": []}
                if role == "recipient"
                else {"kind": "pose_exchange", "objects": [target, distractor]}
            )
            rows.append(
                PoseExchangeCaptureRow(
                    pair_id=pair_id,
                    pair_index=pair_index,
                    role=role,
                    trace_id=trace_ids[role],
                    paired_trace_id=trace_ids["donor" if role == "recipient" else "recipient"],
                    seed=seed,
                    layout_id=layout_id,
                    target_object=target,
                    distractor_object=distractor,
                    scene_mutation=mutation,
                )
            )
    return tuple(rows)


def run_pose_exchange_capture_job(
    args: argparse.Namespace,
    *,
    runtime_loader: RuntimeLoader = load_pi05_capture_runtime,
    capture_task: CaptureTask = run_pi05_capture_task,
) -> tuple[dict[str, Any], int]:
    """Inspect or execute the capture rows while loading PI0.5 at most once."""
    job = _read_json(args.job)
    rows = expand_pose_exchange_capture_job(job)
    selected_model_id = str(
        args.model_id or job.get("model_id") or "lerobot/pi05_libero_finetuned"
    )
    report: dict[str, Any] = {
        "schema_kind": "vla_lens.pi05_pose_exchange_capture_report",
        "schema_version": 1,
        "status": "inspected",
        "output_root": str(args.output_root),
        "model_id": selected_model_id,
        "pair_count": len(rows) // 2,
        "trace_count": len(rows),
        "rows": [row.to_dict() for row in rows],
        "model_load_strategy": "one PI0.5 load for every recipient and donor trace",
    }
    if args.output is not None:
        _write_json_atomic(args.output, report)
    if not args.run_capture:
        return report, 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    existing_trace_ids = set() if args.no_resume else _existing_trace_ids(args.output_root)
    pending_rows = [
        row
        for row in rows
        if args.no_resume or row.trace_id not in existing_trace_ids
    ]
    runtime = None
    if pending_rows:
        capture_args = _capture_args(job, pending_rows[0], args, selected_model_id)
        runtime = runtime_loader(capture_args)
    completed: list[str] = []
    skipped: list[str] = []
    for row in rows:
        if row not in pending_rows:
            skipped.append(row.trace_id)
            continue
        row_args = _capture_args(job, row, args, selected_model_id)
        capture_task(row_args, runtime=runtime, plan=_resolve_capture_plan(row_args))
        completed.append(row.trace_id)
    build_dataset_index(args.output_root, overwrite=True)
    validation = validate_lerobot_v3_dataset(args.output_root)
    if not validation.valid:
        raise ValueError(f"captured pose-exchange dataset is invalid: {validation.to_dict()}")
    dataset = TraceDataset.open(args.output_root)
    captured_trace_ids = {bundle.manifest.trace_id for bundle in dataset.bundles}
    missing = [row.trace_id for row in rows if row.trace_id not in captured_trace_ids]
    if missing:
        raise ValueError(f"pose-exchange capture is missing traces: {missing}")
    pair_manifests = build_pose_exchange_pair_manifests(dataset, rows)
    pair_manifest_path = save_pose_exchange_pair_manifests(
        args.output_root, pair_manifests
    )
    invalid_pairs = [
        pair.pair_id for pair in pair_manifests if not pair.validation.get("pair_valid")
    ]
    if invalid_pairs:
        raise ValueError(f"pose-exchange validation failed for pairs: {invalid_pairs}")
    report.update(
        {
            "status": "completed",
            "completed_trace_ids": completed,
            "skipped_trace_ids": skipped,
            "dataset_trace_count": len(dataset.bundles),
            "model_load_count": 1 if pending_rows else 0,
            "validation": validation.to_dict(),
            "pair_manifest_path": str(pair_manifest_path),
            "valid_pair_count": len(pair_manifests),
            "natural_action_delta_l2": {
                pair.pair_id: pair.validation[
                    "natural_action_delta_l2_separate_saved_noise"
                ]
                for pair in pair_manifests
            },
        }
    )
    if args.output is not None:
        _write_json_atomic(args.output, report)
    return report, 0


def _capture_args(
    job: Mapping[str, Any],
    row: PoseExchangeCaptureRow,
    cli_args: argparse.Namespace,
    model_id: str,
) -> argparse.Namespace:
    matched_fields = (
        "model_id,prompt,camera,robot_state,task_id,layout_id,seed,action_shape"
    )
    return namespace_for_capture_args(
        model_id=model_id,
        benchmark=str(job.get("benchmark") or "libero_90"),
        task_id=int(job["task_id"]),
        episodes=1,
        start_seed=row.seed,
        seed_list=str(row.seed),
        capture_profile=str(job.get("capture_profile") or "rollout"),
        output_root=cli_args.output_root,
        dataset_id=str(job.get("dataset_id") or "rq019_pose_exchange_pairs"),
        capture_design="paired_counterfactual",
        trace_variant=f"{row.pair_id}-{row.role}",
        counterfactual_group_id=row.pair_id,
        counterfactual_role=row.role,
        counterfactual_type="pose_exchange",
        pair_index=row.pair_index,
        paired_trace_id=row.paired_trace_id,
        changed_fields=(
            f"{row.target_object}.pose,{row.distractor_object}.pose"
            if row.role == "donor"
            else ""
        ),
        matched_fields=matched_fields,
        target_object_id=row.target_object,
        counterfactual_target_object_id=row.distractor_object,
        obs_size=int(job.get("obs_size") or 256),
        max_steps=int(job.get("max_steps") or 1),
        layout_id=row.layout_id,
        scene_mutation_json=json.dumps(row.scene_mutation, separators=(",", ":")),
        device=str(cli_args.device),
        dtype=str(cli_args.dtype),
        delete_existing=False,
    )


def _existing_trace_ids(root: Path) -> set[str]:
    """Return only trace IDs whose overlay bundle is complete enough to reuse."""
    trace_ids: set[str] = set()
    episodes_root = root / "vla_lens" / "episodes"
    for path in episodes_root.glob("*/manifest.json"):
        try:
            payload = _read_json(path)
            bundle = TraceBundle.open(path.parent)
            _validate_resumable_bundle(bundle)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        trace_id = str(payload.get("trace_id") or "").strip()
        if trace_id:
            trace_ids.add(trace_id)
    return trace_ids


def _validate_resumable_bundle(bundle: TraceBundle) -> None:
    required_scene_columns = {"object_index", "object_name"}
    if not required_scene_columns <= set(bundle.scene_state.columns):
        raise ValueError("incomplete scene state")
    if bundle.scene_state.empty or bundle.tokens.empty or bundle.policy_calls.empty:
        raise ValueError("incomplete replay tables")
    for array_name in (
        "action_chunks",
        "flow_initial_noise",
        "scene_object_pos",
        "scene_object_quat",
        "camera_object_bbox",
    ):
        bundle.array(array_name, mmap=True)


def _required_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"pose-exchange capture job requires {field}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


__all__ = [
    "PoseExchangeCaptureRow",
    "expand_pose_exchange_capture_job",
    "main",
    "parse_args",
    "run_pose_exchange_capture_job",
]


if __name__ == "__main__":
    main()
