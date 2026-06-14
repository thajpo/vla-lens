"""Policy-call-grain labels derived from PI0.5 object-flow artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from vla_lens.analyzer import dataset_fingerprint
from vla_lens.artifacts import LensArtifact
from vla_lens.dataset import build_dataset_index
from vla_lens.pi05.object_flow import ARTIFACT_TYPE as OBJECT_FLOW_ARTIFACT_TYPE
from vla_lens.traces import TraceBundle, TraceDataset
from vla_lens.workbench import AnalysisRunSpec, save_analysis_run

ARTIFACT_TYPE = "pi05_policy_call_labels"
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PolicyCallLabelsArtifact:
    artifact: LensArtifact
    policy_call_labels: pd.DataFrame


def save_pi05_policy_call_labels_artifact(
    dataset: TraceDataset,
    *,
    name: str = "PI0.5 policy-call labels",
    object_flow_artifact_id: str | None = None,
    rebuild_index: bool = True,
) -> PolicyCallLabelsArtifact:
    """Save one label row per policy call from the latest object-flow artifact."""
    object_flow = _load_object_flow_artifact(dataset, object_flow_artifact_id)
    outputs = dict(object_flow.method.get("outputs") or {})
    timestep_labels = _read_output_table(dataset, outputs, "timestep_labels")
    flow_steps = _read_output_table(dataset, outputs, "flow_steps")
    object_roles = _read_output_table(dataset, outputs, "object_roles")
    labels = build_policy_call_labels(
        dataset,
        timestep_labels=timestep_labels,
        flow_steps=flow_steps,
        object_roles=object_roles,
    )
    artifact = LensArtifact.create(
        artifact_type=ARTIFACT_TYPE,
        name=name,
        group_id=ARTIFACT_TYPE,
        scope="dataset",
        selector={
            "episodes": "all",
            "source_artifact_id": object_flow.artifact_id,
            "source_artifact_type": object_flow.artifact_type,
        },
        method={
            "workflow": "save_pi05_policy_call_labels_artifact",
            "schema_version": SCHEMA_VERSION,
            "dataset_fingerprint": dataset_fingerprint(dataset),
            "source_artifact_id": object_flow.artifact_id,
            "outputs": {"policy_call_labels": ""},
        },
        metrics=_summary_metrics(labels),
        display={
            "kind": ARTIFACT_TYPE,
            "summary": _summary_text(labels),
            "primary_table": "policy_call_labels",
            "interpretation_notes": [
                "Rows are aligned to policy-call observation timesteps.",
                "Prompt, task, geometry, and object metadata remain row metadata or baselines; "
                "they are not activation-probe inputs unless explicitly configured.",
            ],
        },
        tags=("pi05", "policy_call", "object_flow", "labels", "postprocess"),
        source_trace_ids=tuple(
            sorted(str(value) for value in labels.get("trace_id", pd.Series()).unique())
        ),
    )
    saved = dataset.save_artifact(artifact)
    artifact_dir = (dataset._dataset_artifact_root() / str(saved.path or "")).parent
    path = artifact_dir / "policy_call_labels.parquet"
    labels.to_parquet(path, index=False)
    updated = LensArtifact(
        artifact_id=saved.artifact_id,
        artifact_type=saved.artifact_type,
        name=saved.name,
        group_id=saved.group_id,
        scope=saved.scope,
        selector=saved.selector,
        method={
            **dict(saved.method),
            "outputs": {"policy_call_labels": str(path.relative_to(dataset.root))},
        },
        metrics=saved.metrics,
        arrays=saved.arrays,
        display=saved.display,
        tags=saved.tags,
        created_utc=saved.created_utc,
        source_trace_ids=saved.source_trace_ids,
        path=saved.path,
    )
    (artifact_dir / "artifact.json").write_text(
        json.dumps(updated.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _update_dataset_artifact_index(dataset, updated)
    save_analysis_run(
        dataset,
        AnalysisRunSpec(
            run_id=updated.artifact_id,
            workflow=ARTIFACT_TYPE,
            inputs=updated.selector,
            outputs=(),
            provenance={"artifact_id": updated.artifact_id},
        ),
    )
    if rebuild_index:
        build_dataset_index(dataset.root, overwrite=True)
    return PolicyCallLabelsArtifact(artifact=updated, policy_call_labels=labels)


def build_policy_call_labels(
    dataset: TraceDataset,
    *,
    timestep_labels: pd.DataFrame,
    flow_steps: pd.DataFrame,
    object_roles: pd.DataFrame,
) -> pd.DataFrame:
    """Return one object-flow label row per policy call in ``dataset``."""
    label_lookup = _timestep_label_lookup(timestep_labels)
    flow_by_trace = {
        str(trace_id): group.copy()
        for trace_id, group in flow_steps.groupby(flow_steps.get("trace_id", pd.Series(dtype=str)))
    } if not flow_steps.empty and "trace_id" in flow_steps else {}
    objects_by_trace = _objects_by_trace(object_roles)
    rows: list[dict[str, Any]] = []
    for bundle in dataset.bundles:
        calls = bundle.policy_calls.copy()
        if calls.empty or "policy_call_index" not in calls:
            continue
        for call in calls.to_dict("records"):
            policy_call_index = _optional_int(call.get("policy_call_index"))
            if policy_call_index is None:
                continue
            observation_timestep = _policy_observation_timestep(bundle, call)
            label = label_lookup.get((bundle.manifest.trace_id, observation_timestep), {})
            next_object = str(label.get("next_manipulated_object") or "")
            step = _matching_next_step(
                flow_by_trace.get(bundle.manifest.trace_id, pd.DataFrame()),
                next_object,
                observation_timestep,
            )
            candidate_objects = objects_by_trace.get(bundle.manifest.trace_id, [])
            visible_objects = _visible_objects(bundle, observation_timestep)
            rows.append(
                {
                    "trace_id": bundle.manifest.trace_id,
                    "episode_id": bundle.manifest.episode_id,
                    "task_id": bundle.manifest.task_id,
                    "prompt": bundle.manifest.prompt,
                    "policy_call_index": policy_call_index,
                    "policy_call_id": policy_call_index,
                    "observation_timestep": observation_timestep,
                    "env_timestep_start": _optional_int(call.get("env_timestep_start")),
                    "env_timestep_end": _optional_int(call.get("env_timestep_end")),
                    "policy_call_label_timestep": observation_timestep,
                    "split": label.get("split", ""),
                    "benchmark": label.get("benchmark", bundle.manifest.env_id),
                    "dataset_id": label.get("dataset_id", ""),
                    "task_phase": label.get("task_phase", ""),
                    "next_manipulated_object": next_object,
                    "active_manipulated_object": label.get("active_manipulated_object", ""),
                    "active_receptacle_object": label.get("active_receptacle_object", ""),
                    "current_contact_object": label.get("current_contact_object", ""),
                    "current_moved_object": label.get("current_moved_object", ""),
                    "current_lifted_object": label.get("current_lifted_object", ""),
                    "next_flow_step_index": label.get("next_flow_step_index", np.nan),
                    "active_flow_step_index": label.get("active_flow_step_index", np.nan),
                    "next_object_flow_step_index": step.get("flow_step_index", np.nan),
                    "first_contact_time_next_object": step.get(
                        "contact_onset_timestep",
                        np.nan,
                    ),
                    "first_motion_time_next_object": step.get(
                        "movement_onset_timestep",
                        np.nan,
                    ),
                    "first_lift_time_next_object": step.get("lift_onset_timestep", np.nan),
                    "is_pre_contact": _is_before(
                        observation_timestep,
                        step.get("contact_onset_timestep", np.nan),
                        next_object,
                    ),
                    "is_pre_motion": _is_before(
                        observation_timestep,
                        step.get("movement_onset_timestep", np.nan),
                        next_object,
                    ),
                    "is_pre_lift": _is_before(
                        observation_timestep,
                        step.get("lift_onset_timestep", np.nan),
                        next_object,
                    ),
                    "candidate_objects": json.dumps(candidate_objects),
                    "visible_candidate_objects": json.dumps(visible_objects),
                    "visible_candidate_count": len(visible_objects),
                }
            )
    return pd.DataFrame.from_records(rows)


def _load_object_flow_artifact(
    dataset: TraceDataset,
    artifact_id: str | None,
) -> LensArtifact:
    if artifact_id:
        artifact = dataset.load_artifact(artifact_id)
        if artifact.artifact_type != OBJECT_FLOW_ARTIFACT_TYPE:
            raise ValueError(
                f"Artifact {artifact_id!r} is {artifact.artifact_type!r}, "
                f"expected {OBJECT_FLOW_ARTIFACT_TYPE!r}"
            )
        return artifact
    table = dataset.artifact_index
    if table.empty or "artifact_type" not in table:
        raise FileNotFoundError("Dataset has no object-flow artifact index")
    matches = table.loc[
        table["artifact_type"].astype(str) == OBJECT_FLOW_ARTIFACT_TYPE
    ].copy()
    if matches.empty:
        raise FileNotFoundError("Dataset has no pi05_object_flow artifact")
    matches = matches.sort_values("created_utc", ascending=False, na_position="last")
    for candidate_id in matches["artifact_id"].astype(str):
        try:
            return dataset.load_artifact(candidate_id)
        except (FileNotFoundError, KeyError):
            continue
    raise FileNotFoundError("No readable pi05_object_flow artifact found")


def _read_output_table(
    dataset: TraceDataset,
    outputs: Mapping[str, Any],
    key: str,
) -> pd.DataFrame:
    path = outputs.get(key)
    if not path:
        raise FileNotFoundError(f"Object-flow artifact has no {key!r} output")
    table_path = _artifact_output_path(dataset, str(path))
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    return pd.read_parquet(table_path)


def _artifact_output_path(dataset: TraceDataset, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    dataset_path = dataset.root / path
    if dataset_path.exists():
        return dataset_path
    return dataset._dataset_artifact_root() / path


def _update_dataset_artifact_index(dataset: TraceDataset, artifact: LensArtifact) -> None:
    artifact_index_path = dataset._dataset_artifact_root() / TraceBundle.ARTIFACT_INDEX
    existing = (
        pd.read_parquet(artifact_index_path)
        if artifact_index_path.exists()
        else pd.DataFrame()
    )
    updated_index = pd.concat(
        [existing, pd.DataFrame.from_records([artifact.to_record()])],
        ignore_index=True,
    ).drop_duplicates(subset=["artifact_id"], keep="last")
    artifact_index_path.parent.mkdir(parents=True, exist_ok=True)
    updated_index.to_parquet(artifact_index_path, index=False)
    dataset.__dict__.pop("dataset_artifact_index", None)
    dataset.__dict__.pop("artifact_index", None)


def _timestep_label_lookup(timestep_labels: pd.DataFrame) -> dict[tuple[str, int], dict[str, Any]]:
    if timestep_labels.empty or not {"trace_id", "timestep"}.issubset(timestep_labels.columns):
        return {}
    return {
        (str(row["trace_id"]), int(row["timestep"])): row
        for row in timestep_labels.to_dict("records")
    }


def _objects_by_trace(object_roles: pd.DataFrame) -> dict[str, list[str]]:
    if object_roles.empty or not {"trace_id", "object_name"}.issubset(object_roles.columns):
        return {}
    out: dict[str, list[str]] = {}
    for trace_id, group in object_roles.groupby(object_roles["trace_id"].astype(str)):
        out[str(trace_id)] = sorted(str(value) for value in group["object_name"].dropna().unique())
    return out


def _policy_observation_timestep(bundle: TraceBundle, call: Mapping[str, Any]) -> int:
    for key in ["observation_timestep", "env_timestep_start"]:
        value = _optional_int(call.get(key))
        if value is not None:
            return value
    policy_call = _optional_int(call.get("policy_call_index"))
    if policy_call is not None and not bundle.timesteps.empty:
        table = bundle.timesteps
        if {"policy_call_index", "timestep"}.issubset(table.columns):
            matches = table.loc[table["policy_call_index"].fillna(-1).astype(int) == policy_call]
            if not matches.empty:
                return int(matches.iloc[0]["timestep"])
    return 0


def _matching_next_step(
    flow_steps: pd.DataFrame,
    object_name: str,
    timestep: int,
) -> dict[str, Any]:
    if not object_name or flow_steps.empty or "object_name" not in flow_steps:
        return {}
    candidates = flow_steps.loc[
        (flow_steps["step_type"].astype(str) == "manipulate_object")
        & (flow_steps["object_name"].astype(str) == object_name)
    ].copy()
    if candidates.empty:
        return {}
    if "observed_start_timestep" in candidates:
        starts = pd.to_numeric(candidates["observed_start_timestep"], errors="coerce")
        future = candidates.loc[starts.isna() | (starts >= int(timestep))]
        if not future.empty:
            candidates = future
        candidates = candidates.assign(
            __sort_start=pd.to_numeric(
                candidates["observed_start_timestep"],
                errors="coerce",
            ).fillna(float("inf"))
        )
        candidates = candidates.sort_values(["__sort_start", "flow_step_index"], kind="mergesort")
    return candidates.iloc[0].drop(labels=["__sort_start"], errors="ignore").to_dict()


def _visible_objects(bundle: TraceBundle, timestep: int) -> list[str]:
    try:
        visible = np.asarray(bundle.array("camera_object_visible", mmap=True))
    except KeyError:
        return []
    if visible.ndim < 3 or timestep >= visible.shape[0]:
        return []
    names = _array_object_names(bundle, "camera_object_visible")
    if not names:
        names = [
            str(row.get("object_name") or "")
            for row in bundle.scene_state.to_dict("records")
            if row.get("object_name")
        ]
    mask = np.asarray(visible[int(timestep)]).astype(bool)
    if mask.ndim == 2:
        object_mask = mask.any(axis=0)
    elif mask.ndim == 1:
        object_mask = mask
    else:
        object_mask = mask.reshape(-1, mask.shape[-1]).any(axis=0)
    return [
        str(name)
        for index, name in enumerate(names)
        if index < len(object_mask) and bool(object_mask[index])
    ]


def _array_object_names(bundle: TraceBundle, array_name: str) -> list[str]:
    index = bundle.array_index
    if index.empty or "name" not in index:
        return []
    rows = index.loc[index["name"].astype(str) == array_name]
    if rows.empty or "metadata" not in rows:
        return []
    metadata = rows.iloc[0].get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if not isinstance(metadata, Mapping):
        return []
    names = metadata.get("object_names")
    if not isinstance(names, list):
        return []
    return [str(name) for name in names]


def _is_before(timestep: int, onset: Any, object_name: str) -> bool:
    if not object_name or pd.isna(onset):
        return False
    return int(timestep) < int(float(onset))


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _summary_metrics(labels: pd.DataFrame) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "policy_call_count": int(len(labels)),
        "episode_count": int(labels["trace_id"].nunique()) if "trace_id" in labels else 0,
        "next_manipulated_nonempty_count": _nonempty_count(labels, "next_manipulated_object"),
        "pre_contact_count": _truthy_count(labels, "is_pre_contact"),
        "pre_motion_count": _truthy_count(labels, "is_pre_motion"),
    }


def _summary_text(labels: pd.DataFrame) -> str:
    return (
        f"{len(labels)} policy call(s); "
        f"{_nonempty_count(labels, 'next_manipulated_object')} with next object; "
        f"{_truthy_count(labels, 'is_pre_contact')} pre-contact."
    )


def _nonempty_count(labels: pd.DataFrame, column: str) -> int:
    if column not in labels:
        return 0
    return int((labels[column].fillna("").astype(str) != "").sum())


def _truthy_count(labels: pd.DataFrame, column: str) -> int:
    if column not in labels:
        return 0
    return int(labels[column].fillna(False).astype(bool).sum())


__all__ = [
    "ARTIFACT_TYPE",
    "PolicyCallLabelsArtifact",
    "build_policy_call_labels",
    "save_pi05_policy_call_labels_artifact",
]
