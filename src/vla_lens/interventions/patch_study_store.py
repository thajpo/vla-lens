"""Incremental, resumable storage for counterfactual patch studies."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import zarr

from vla_lens.interventions.counterfactuals import (
    ActionArrayRef,
    CounterfactualPairManifest,
    EvaluationDecision,
    PatchStudyArtifact,
    PatchStudySpec,
    PatchTrialManifest,
)
from vla_lens.interventions.patch_study import PlannedPatchTrial
from vla_lens.interventions.results import InterventionRun
from vla_lens.interventions.serialization import jsonable, utc_now_iso


@dataclass(frozen=True, slots=True)
class PatchStudyProgress:
    """Compact state needed to explain and resume a study."""

    study_id: str
    request_sha256: str
    planned_trial_count: int
    completed_trial_ids: tuple[str, ...] = ()
    failed_trial_ids: tuple[str, ...] = ()
    status: str = "planned"
    updated_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_kind": "vla_lens.patch_study_progress",
            "schema_version": 1,
            "study_id": self.study_id,
            "request_sha256": self.request_sha256,
            "planned_trial_count": self.planned_trial_count,
            "completed_trial_ids": list(self.completed_trial_ids),
            "failed_trial_ids": list(self.failed_trial_ids),
            "status": self.status,
            "updated_utc": self.updated_utc or utc_now_iso(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PatchStudyProgress":
        return cls(
            study_id=str(payload["study_id"]),
            request_sha256=str(payload["request_sha256"]),
            planned_trial_count=int(payload["planned_trial_count"]),
            completed_trial_ids=tuple(
                str(value) for value in payload.get("completed_trial_ids", ())
            ),
            failed_trial_ids=tuple(
                str(value) for value in payload.get("failed_trial_ids", ())
            ),
            status=str(payload.get("status", "planned")),
            updated_utc=str(payload.get("updated_utc", "")),
        )


class PatchStudyStore:
    """Write each finished trial before advancing to the next expensive call."""

    def __init__(
        self,
        root: Path,
        *,
        study: PatchStudySpec,
        pairs: Sequence[CounterfactualPairManifest],
        plan: Sequence[PlannedPatchTrial],
        request_sha256: str,
    ) -> None:
        self.root = Path(root)
        self.study = study
        self.pairs = tuple(pairs)
        self.plan = tuple(plan)
        self.request_sha256 = request_sha256
        self.state_path = self.root / "state.json"
        self.plan_path = self.root / "plan.json"
        self.runs_dir = self.root / "runs"
        self.failures_dir = self.root / "failures"
        self.actions_path = self.root / "actions.zarr"

    def prepare(self) -> PatchStudyProgress:
        """Create a study directory or verify that an existing one is resumable."""
        self.root.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)
        self.failures_dir.mkdir(exist_ok=True)
        if self.state_path.exists():
            progress = PatchStudyProgress.from_dict(_read_json(self.state_path))
            if progress.study_id != self.study.study_id:
                raise ValueError("existing patch-study state belongs to another study")
            if progress.request_sha256 != self.request_sha256:
                raise ValueError(
                    "existing patch-study state was created from a different request; "
                    "use a new study_id"
                )
            if progress.planned_trial_count != len(self.plan):
                raise ValueError("existing patch-study state has a different trial plan")
            return progress
        _write_json_atomic(
            self.plan_path,
            {
                "schema_kind": "vla_lens.patch_study_plan",
                "schema_version": 1,
                "study": self.study.to_dict(),
                "pairs": [pair.to_dict() for pair in self.pairs],
                "trials": [trial.to_dict() for trial in self.plan],
                "request_sha256": self.request_sha256,
            },
        )
        progress = PatchStudyProgress(
            study_id=self.study.study_id,
            request_sha256=self.request_sha256,
            planned_trial_count=len(self.plan),
            updated_utc=utc_now_iso(),
        )
        self._write_progress(progress)
        return progress

    def progress(self) -> PatchStudyProgress:
        if not self.state_path.exists():
            return self.prepare()
        return PatchStudyProgress.from_dict(_read_json(self.state_path))

    def is_completed(self, trial_id: str) -> bool:
        return trial_id in set(self.progress().completed_trial_ids)

    def is_failed(self, trial_id: str) -> bool:
        return trial_id in set(self.progress().failed_trial_ids)

    def record_run(
        self,
        trial: PlannedPatchTrial,
        run: InterventionRun,
        arrays: Mapping[str, np.ndarray],
    ) -> None:
        """Save actions, the full run, and typed trial records, then mark complete."""
        if run.run_id != trial.trial_id:
            raise ValueError("intervention run_id must match the planned patch trial_id")
        action_refs, trial_manifests = self._save_action_records(trial, run, arrays)
        decisions = _run_decisions(run)
        _write_json_atomic(
            self.runs_dir / f"{trial.trial_id}.json",
            {
                "schema_kind": "vla_lens.patch_study_trial_result",
                "schema_version": 1,
                "planned_trial": trial.to_dict(),
                "run": run.to_dict(),
                "patch_trials": [item.to_dict() for item in trial_manifests],
                "action_arrays": [item.to_dict() for item in action_refs],
                "decisions": [item.to_dict() for item in decisions],
            },
        )
        (self.failures_dir / f"{trial.trial_id}.json").unlink(missing_ok=True)
        progress = self.progress()
        completed = tuple(dict.fromkeys((*progress.completed_trial_ids, trial.trial_id)))
        failed = tuple(value for value in progress.failed_trial_ids if value != trial.trial_id)
        self._write_progress(
            PatchStudyProgress(
                study_id=self.study.study_id,
                request_sha256=self.request_sha256,
                planned_trial_count=len(self.plan),
                completed_trial_ids=completed,
                failed_trial_ids=failed,
                status="running",
                updated_utc=utc_now_iso(),
            )
        )

    def record_failure(self, trial: PlannedPatchTrial, error: BaseException) -> None:
        """Record one bounded failure without losing earlier successful work."""
        _write_json_atomic(
            self.failures_dir / f"{trial.trial_id}.json",
            {
                "trial_id": trial.trial_id,
                "planned_trial": trial.to_dict(),
                "error_type": type(error).__name__,
                "error": str(error),
                "recorded_utc": utc_now_iso(),
            },
        )
        progress = self.progress()
        failed = tuple(dict.fromkeys((*progress.failed_trial_ids, trial.trial_id)))
        self._write_progress(
            PatchStudyProgress(
                study_id=self.study.study_id,
                request_sha256=self.request_sha256,
                planned_trial_count=len(self.plan),
                completed_trial_ids=progress.completed_trial_ids,
                failed_trial_ids=failed,
                status="partial",
                updated_utc=utc_now_iso(),
            )
        )

    def finalize(self) -> PatchStudyArtifact:
        """Materialize compact tables and the reconstructable permanent artifact."""
        records = [_read_json(path) for path in sorted(self.runs_dir.glob("*.json"))]
        failures = [_read_json(path) for path in sorted(self.failures_dir.glob("*.json"))]
        action_refs = tuple(
            ActionArrayRef.from_dict(item)
            for record in records
            for item in record.get("action_arrays", ())
        )
        trials = tuple(
            PatchTrialManifest.from_dict(item)
            for record in records
            for item in record.get("patch_trials", ())
        )
        decisions = tuple(
            EvaluationDecision.from_dict(item)
            for record in records
            for item in record.get("decisions", ())
        )
        self._write_tables(trials, failures)
        progress = self.progress()
        completed_all = len(progress.completed_trial_ids) == len(self.plan)
        status = "completed" if completed_all and not failures else "partial"
        final_progress = PatchStudyProgress(
            study_id=self.study.study_id,
            request_sha256=self.request_sha256,
            planned_trial_count=len(self.plan),
            completed_trial_ids=progress.completed_trial_ids,
            failed_trial_ids=progress.failed_trial_ids,
            status=status,
            updated_utc=utc_now_iso(),
        )
        self._write_progress(final_progress)
        artifact = PatchStudyArtifact(
            study=self.study,
            pairs=self.pairs,
            trials=trials,
            action_arrays=action_refs,
            decisions=decisions,
            permanent_outputs=(
                "artifact.json",
                "plan.json",
                "state.json",
                "pairs.parquet",
                "trials.parquet",
                "failures.parquet",
                "actions.zarr",
                "runs/",
            ),
            disposable_cache_refs=tuple(
                f"memory://donor/{pair_id}/layers"
                for pair_id in self.study.pair_ids
            ),
            provenance={
                "request_sha256": self.request_sha256,
                "status": status,
                "planned_trial_count": len(self.plan),
                "completed_trial_count": len(progress.completed_trial_ids),
                "failed_trial_count": len(failures),
            },
        )
        _write_json_atomic(self.root / "artifact.json", artifact.to_dict())
        return artifact

    def intervention_runs(self) -> tuple[InterventionRun, ...]:
        return tuple(
            InterventionRun.from_dict(_read_json(path)["run"])
            for path in sorted(self.runs_dir.glob("*.json"))
        )

    def _save_action_records(
        self,
        trial: PlannedPatchTrial,
        run: InterventionRun,
        arrays: Mapping[str, np.ndarray],
    ) -> tuple[tuple[ActionArrayRef, ...], tuple[PatchTrialManifest, ...]]:
        group = zarr.open_group(str(self.actions_path), mode="a")
        refs: list[ActionArrayRef] = []
        manifests: list[PatchTrialManifest] = []
        for name, role, trial_kind, control_kind in _saved_action_roles(run, arrays):
            array = np.asarray(arrays[name], dtype=np.float32)
            if array.ndim != 2 or not np.all(np.isfinite(array)):
                raise ValueError(f"study action {name!r} must be a finite 2D array")
            array_key = f"{trial.trial_id}/{name}"
            _save_zarr_array(group, array_key, array)
            sha256 = _array_sha256(array)
            action_ref = ActionArrayRef(
                array_ref=f"actions.zarr/{array_key}",
                role=role,
                shape=(int(array.shape[0]), int(array.shape[1])),
                dtype=str(array.dtype),
                sha256=sha256,
                coordinates={
                    "action_dim": _action_dim_names(array.shape[1]),
                    "planned_trial_id": trial.trial_id,
                },
                metadata={"array_name": name},
            )
            runtime_trial = _runtime_trial(run, control_kind=control_kind, role=role)
            token_indices = tuple(
                int(value)
                for value in runtime_trial.get("runtime", {}).get(
                    "recipient_token_indices", trial.recipient_token_indices
                )
            )
            manifests.append(
                PatchTrialManifest(
                    trial_id=f"{trial.trial_id}--{name}",
                    pair_id=trial.pair_id,
                    trial_kind=trial_kind,
                    action=action_ref,
                    noise_ref=str(
                        runtime_trial.get("runtime", {}).get("shared_noise_ref")
                        or self.study.shared_noise_refs[0]
                    ),
                    target=run.target.to_dict(),
                    operation=dict(run.request.get("operator") or {}),
                    control_kind=control_kind,
                    token_indices=token_indices,
                    token_mapping_sha256=runtime_trial.get("runtime", {}).get(
                        "token_mapping_sha256"
                    ),
                    hook_calls=_optional_int(
                        runtime_trial.get("runtime", {}).get("hook_calls")
                    ),
                    status=(run.status if run.status in {"ok", "partial", "failed"} else "partial"),
                    metrics=dict(runtime_trial.get("metrics") or {}),
                    provenance={
                        "planned_trial_id": trial.trial_id,
                        "layer": trial.layer,
                        "token_region": trial.token_region,
                    },
                )
            )
            refs.append(action_ref)
        return tuple(refs), tuple(manifests)

    def _write_tables(
        self,
        trials: Sequence[PatchTrialManifest],
        failures: Sequence[Mapping[str, Any]],
    ) -> None:
        pair_rows = [
            {
                "pair_id": pair.pair_id,
                "recipe_kind": pair.recipe.kind,
                "target_object": pair.recipe.target_object,
                "distractor_object": pair.recipe.distractor_object,
                "recipient_trace_id": _trace_id(pair.recipient.to_dict()),
                "donor_trace_id": _trace_id(pair.donor.to_dict()),
                "manifest_json": _canonical_json(pair.to_dict()),
            }
            for pair in self.pairs
        ]
        trial_rows = [
            {
                "trial_id": trial.trial_id,
                "pair_id": trial.pair_id,
                "trial_kind": trial.trial_kind,
                "control_kind": trial.control_kind,
                "status": trial.status,
                "array_ref": trial.action.array_ref,
                "array_sha256": trial.action.sha256,
                "layer": trial.target.get("layer"),
                "token_indices_json": _canonical_json(list(trial.token_indices)),
                "metrics_json": _canonical_json(trial.metrics),
                "manifest_json": _canonical_json(trial.to_dict()),
            }
            for trial in trials
        ]
        failure_rows = [
            {
                "trial_id": failure.get("trial_id"),
                "error_type": failure.get("error_type"),
                "error": failure.get("error"),
                "recorded_utc": failure.get("recorded_utc"),
                "failure_json": _canonical_json(failure),
            }
            for failure in failures
        ]
        pd.DataFrame(pair_rows).to_parquet(self.root / "pairs.parquet", index=False)
        pd.DataFrame(
            trial_rows,
            columns=(
                "trial_id",
                "pair_id",
                "trial_kind",
                "control_kind",
                "status",
                "array_ref",
                "array_sha256",
                "layer",
                "token_indices_json",
                "metrics_json",
                "manifest_json",
            ),
        ).to_parquet(self.root / "trials.parquet", index=False)
        pd.DataFrame(
            failure_rows,
            columns=(
                "trial_id",
                "error_type",
                "error",
                "recorded_utc",
                "failure_json",
            ),
        ).to_parquet(self.root / "failures.parquet", index=False)

    def _write_progress(self, progress: PatchStudyProgress) -> None:
        _write_json_atomic(self.state_path, progress.to_dict())


def _saved_action_roles(
    run: InterventionRun,
    arrays: Mapping[str, np.ndarray],
) -> tuple[tuple[str, str, str, str | None], ...]:
    roles: list[tuple[str, str, str, str | None]] = []
    for name, role, kind in (
        ("noop", "recipient", "recipient"),
        ("donor_shared_noise", "donor", "donor"),
        ("intervened", "patched", "patched"),
    ):
        if name in arrays:
            roles.append((name, role, kind, None))
    for control in run.controls:
        control_kind = str(control.get("control_kind") or control.get("kind") or "")
        name = f"control_{control_kind}"
        if control_kind and name in arrays:
            roles.append((name, "control", "control", control_kind))
    return tuple(roles)


def _runtime_trial(
    run: InterventionRun,
    *,
    control_kind: str | None,
    role: str,
) -> Mapping[str, Any]:
    if control_kind:
        return next(
            (
                trial.to_dict()
                for trial in run.trials
                if trial.control_kind == control_kind
            ),
            {},
        )
    if role in {"patched", "donor"}:
        return next(
            (
                trial.to_dict()
                for trial in run.trials
                if trial.trial_kind == "intervention"
            ),
            {},
        )
    return next(
        (
            trial.to_dict()
            for trial in run.trials
            if trial.trial_kind == "noop_rerun"
        ),
        {},
    )


def _run_decisions(run: InterventionRun) -> tuple[EvaluationDecision, ...]:
    transfer = run.display.get("counterfactual_transfer")
    if not isinstance(transfer, Mapping) or not isinstance(transfer.get("decision"), Mapping):
        return ()
    return (EvaluationDecision.from_dict(transfer["decision"]),)


def _save_zarr_array(group: Any, key: str, array: np.ndarray) -> None:
    if key in group:
        existing = np.asarray(group[key])
        if existing.shape != array.shape or not np.array_equal(existing, array):
            raise ValueError(f"existing study action {key!r} does not match resumed result")
        return
    group.create_dataset(key, data=array, chunks=array.shape, overwrite=False)


def _action_dim_names(size: int) -> list[str]:
    if size == 7:
        return ["x", "y", "z", "rx", "ry", "rz", "gripper"]
    return [f"action_{index}" for index in range(size)]


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _trace_id(spec: Mapping[str, Any]) -> str | None:
    trace = spec.get("trace")
    return str(trace.get("trace_id")) if isinstance(trace, Mapping) else None


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = ["PatchStudyProgress", "PatchStudyStore"]
