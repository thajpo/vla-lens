"""Dataset dashboard server helpers."""


from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from vla_lens.action_generation import save_action_generation_artifact
from vla_lens.analyzer import diagnostics_status, run_dataset_diagnostics
from vla_lens.probes.workflow import (
    train_probe_artifact_from_spec,
)
from vla_lens.server_common import (
    _array_preview,
    _array_summary,
    _dedupe_reasons,
    _is_missing_scalar,
    _json_parse,
    _json_scalar,
    _jsonable,
    _metadata_text,
    _query_int_value,
    _query_one,
    _record_bool,
    _record_float,
    _record_text,
    _string_list,
)
from vla_lens.server_metrics import (
    _manifest_payload,
)
from vla_lens.target_object import save_target_object_encoding_artifact
from vla_lens.traces import TraceBundle, TraceDataset
from vla_lens.workbench import (
    AnalysisRunSpec,
    InterventionRunSpec,
    SavedWorkspace,
    SelectionState,
    UnitRef,
    cohort_from_selection,
    compare_cohorts,
    graph_from_selection,
    lens_array_catalog,
    lens_array_meta,
    list_analysis_runs,
    list_cohorts,
    list_intervention_runs,
    list_workspaces,
    projection_points,
    query_table,
    resolve_selection,
    save_analysis_run,
    save_cohort,
    save_intervention_run,
    save_workspace,
    slice_lens_array,
    unit_profile,
    workbench_manifest,
)


def _dataset_payload(dataset: TraceDataset, *, include_workbench: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "root": str(dataset.root),
        "episodes": [_manifest_payload(bundle) for bundle in dataset.bundles],
        "counterfactual_pairs": _counterfactual_pairs_payload(dataset),
    }
    if include_workbench:
        workbench = workbench_manifest(dataset)
        payload.update(
            {
                "activation_sites": int(len(dataset.model_site_index)),
                "artifacts": _artifact_summary(dataset),
                "workbench": workbench,
            }
        )
    else:
        payload.update(
            {
                "activation_sites": 0,
                "artifacts": {"total": 0, "counts": {}},
            }
        )
    return payload

def _counterfactual_pairs_response(dataset: TraceDataset) -> dict[str, Any]:
    pairs = _counterfactual_pairs_payload(dataset)
    return {"pairs": pairs, "count": len(pairs)}

def _counterfactual_pairs_payload(dataset: TraceDataset) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    group_metadata: dict[str, dict[str, Any]] = {}
    for bundle in dataset.bundles:
        metadata = dict(bundle.manifest.metadata or {})
        counterfactual = _counterfactual_metadata_from_manifest(metadata)
        group_id = str(counterfactual.get("group_id") or "").strip()
        if not group_id:
            continue
        member = {
            "trace_id": bundle.manifest.trace_id,
            "episode_id": bundle.manifest.episode_id,
            "role": str(counterfactual.get("role") or ""),
            "pair_index": _json_scalar(counterfactual.get("pair_index")),
            "paired_trace_id": str(counterfactual.get("paired_trace_id") or ""),
            "target_object_id": str(counterfactual.get("target_object_id") or ""),
            "counterfactual_target_object_id": str(
                counterfactual.get("counterfactual_target_object_id") or ""
            ),
            "outcome": bundle.manifest.outcome,
            "prompt": bundle.manifest.prompt,
        }
        grouped.setdefault(group_id, []).append(member)
        group_metadata.setdefault(
            group_id,
            {
                "group_id": group_id,
                "type": str(counterfactual.get("type") or ""),
                "changed_fields": _string_list(counterfactual.get("changed_fields")),
                "matched_fields": _string_list(counterfactual.get("matched_fields")),
            },
        )
    pairs: list[dict[str, Any]] = []
    for group_id, members in grouped.items():
        members.sort(key=_counterfactual_member_sort_key)
        pairs.append({**group_metadata[group_id], "members": members})
    pairs.sort(key=lambda pair: str(pair.get("group_id") or ""))
    return pairs

def _counterfactual_metadata_from_manifest(metadata: Mapping[str, Any]) -> dict[str, Any]:
    nested = metadata.get("counterfactual")
    counterfactual = dict(nested) if isinstance(nested, Mapping) else {}
    aliases = {
        "counterfactual_group_id": "group_id",
        "counterfactual_role": "role",
        "counterfactual_type": "type",
        "paired_trace_id": "paired_trace_id",
        "pair_index": "pair_index",
        "changed_fields": "changed_fields",
        "matched_fields": "matched_fields",
        "target_object_id": "target_object_id",
        "counterfactual_target_object_id": "counterfactual_target_object_id",
    }
    for source, target in aliases.items():
        if target not in counterfactual and source in metadata:
            counterfactual[target] = metadata[source]
    return counterfactual

def _counterfactual_member_sort_key(member: Mapping[str, Any]) -> tuple[int, int, str]:
    role_order = {"clean": 0, "control": 1, "corrupt": 2, "intervention": 3}
    role = str(member.get("role") or "")
    try:
        pair_index = int(member.get("pair_index"))
    except (TypeError, ValueError):
        pair_index = 10_000
    return (pair_index, role_order.get(role, 100), str(member.get("trace_id") or ""))

def _observational_comparisons_payload(
    dataset: TraceDataset,
    query: Mapping[str, list[str]],
) -> dict[str, Any]:
    from vla_lens.server_probes import _probe_index_artifact_payload

    trace_id = _query_one(dict(query), "trace_id")
    probe_id = (query.get("probe_id") or query.get("probe") or [""])[0]
    limit = _query_int_value(query, "limit", 6)
    source = dataset.bundle(trace_id)
    probe = _probe_index_artifact_payload(dataset, probe_id) if probe_id else None
    source_probe = _probe_trace_record(probe, trace_id)
    candidates: list[dict[str, Any]] = []
    for candidate in dataset.bundles:
        if candidate.manifest.trace_id == trace_id:
            continue
        candidate_probe = _probe_trace_record(probe, candidate.manifest.trace_id)
        score, reasons, metrics = _observational_candidate_score(
            source,
            candidate,
            source_probe,
            candidate_probe,
            has_probe=probe is not None,
        )
        candidates.append(
            {
                "trace_id": candidate.manifest.trace_id,
                "score": round(score, 3),
                "reasons": reasons,
                "episode": _comparison_episode_payload(candidate),
                "probe": _jsonable(candidate_probe) if candidate_probe else None,
                "metrics": metrics,
                "contract": {
                    "source_trace_id": trace_id,
                    "comparison_trace_id": candidate.manifest.trace_id,
                    "method": "nearest_neighbor_existing_trace",
                    "causal": False,
                    "requires_live_intervention": False,
                },
            }
        )
    candidates.sort(key=lambda item: (-float(item["score"]), str(item["trace_id"])))
    candidates = candidates[: max(1, min(limit, 24))]
    return {
        "artifact_type": "observational_counterfactual_comparison",
        "artifact_id": _observational_comparison_artifact_id(trace_id, probe_id),
        "name": "Observational comparison candidates",
        "causal": False,
        "comparison_kind": "nearest_neighbor_existing_trace",
        "source_trace_id": trace_id,
        "probe_id": probe_id or None,
        "probe_name": probe.get("name") if probe else None,
        "source": {
            "episode": _comparison_episode_payload(source),
            "probe": _jsonable(source_probe) if source_probe else None,
        },
        "candidates": candidates,
        "total_candidates": max(0, len(dataset.bundles) - 1),
        "limit": limit,
        "notes": (
            "Existing traces only. This is a comparison queue for inspection, "
            "not evidence that an activation change caused the behavior change."
        ),
    }

def _observational_comparison_artifact_id(trace_id: str, probe_id: str) -> str:
    suffix = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{trace_id}.{probe_id or 'episode'}").strip("_")
    return f"observational_comparison.{suffix[:160]}"

def _probe_trace_record(
    probe: Mapping[str, Any] | None,
    trace_id: str,
) -> Mapping[str, Any] | None:
    if not probe:
        return None
    by_trace = probe.get("by_trace")
    if not isinstance(by_trace, Mapping):
        return None
    record = by_trace.get(trace_id)
    return record if isinstance(record, Mapping) else None

def _comparison_episode_payload(bundle: TraceBundle) -> dict[str, Any]:
    metadata = dict(bundle.manifest.metadata or {})
    compact_metadata = {
        key: _json_scalar(metadata.get(key))
        for key in (
            "benchmark",
            "capture_profile",
            "seed",
            "split",
            "suite",
            "target_object",
            "task_name",
        )
        if key in metadata and not _is_missing_scalar(metadata.get(key))
    }
    return {
        "trace_id": bundle.manifest.trace_id,
        "episode_id": bundle.manifest.episode_id,
        "task_id": bundle.manifest.task_id,
        "prompt": bundle.manifest.prompt,
        "model_id": bundle.manifest.model_id,
        "env_id": bundle.manifest.env_id,
        "outcome": bundle.manifest.outcome,
        "length": bundle.manifest.length,
        "metadata": compact_metadata,
    }

def _observational_candidate_score(
    source: TraceBundle,
    candidate: TraceBundle,
    source_probe: Mapping[str, Any] | None,
    candidate_probe: Mapping[str, Any] | None,
    *,
    has_probe: bool,
) -> tuple[float, list[str], dict[str, Any]]:
    source_manifest = source.manifest
    candidate_manifest = candidate.manifest
    source_metadata = dict(source_manifest.metadata or {})
    candidate_metadata = dict(candidate_manifest.metadata or {})
    reasons: list[str] = []
    score = 0.0

    same_task = bool(
        source_manifest.task_id and source_manifest.task_id == candidate_manifest.task_id
    )
    same_prompt = bool(
        source_manifest.prompt and source_manifest.prompt == candidate_manifest.prompt
    )
    same_env = bool(source_manifest.env_id and source_manifest.env_id == candidate_manifest.env_id)
    same_model = bool(
        source_manifest.model_id and source_manifest.model_id == candidate_manifest.model_id
    )
    source_target = _metadata_text(source_metadata, "target_object")
    candidate_target = _metadata_text(candidate_metadata, "target_object")
    same_target_object = bool(source_target and source_target == candidate_target)
    different_outcome = bool(
        source_manifest.outcome
        and candidate_manifest.outcome
        and source_manifest.outcome != candidate_manifest.outcome
    )
    length_delta = int(candidate_manifest.length) - int(source_manifest.length)

    if same_task:
        score += 220
        reasons.append("same task")
    if same_prompt:
        score += 60
    if same_target_object:
        score += 70
        reasons.append("same target")
    if same_env:
        score += 20
    if same_model:
        score += 20
    if different_outcome:
        score += 180
        reasons.append("different outcome")
    else:
        reasons.append("same outcome")
    score -= min(90, abs(length_delta) * 2.0)

    source_correct = _record_bool(source_probe, "correct")
    candidate_correct = _record_bool(candidate_probe, "correct")
    source_confidence = _record_float(source_probe, "confidence")
    candidate_confidence = _record_float(candidate_probe, "confidence")
    source_split = _record_text(source_probe, "split_category")
    candidate_split = _record_text(candidate_probe, "split_category")
    confidence_delta = (
        None
        if source_confidence is None or candidate_confidence is None
        else round(candidate_confidence - source_confidence, 4)
    )

    if has_probe:
        if candidate_probe and candidate_probe.get("available"):
            score += 80
            reasons.append("probe scored")
        else:
            score -= 120
            reasons.append("probe unscored")
        if candidate_split in {"test", "validation"}:
            score += 130 if candidate_split == "test" else 95
            reasons.append(f"{candidate_split} probe record")
        elif candidate_split == "train":
            score -= 180
            reasons.append("training-set probe record")
        if source_correct is not None and candidate_correct is not None:
            if source_correct != candidate_correct:
                score += 150
                reasons.append("probe result differs")
            elif candidate_correct is False:
                score += 95
                reasons.append("probe also misses")
            else:
                score -= 35
                reasons.append("probe also correct")
        if candidate_confidence is not None:
            score += min(40, max(0.0, candidate_confidence) * 24)
        if confidence_delta is not None and abs(confidence_delta) >= 0.2:
            score += min(45, abs(confidence_delta) * 60)
            reasons.append("confidence shift")

    metrics = {
        "same_task": same_task,
        "same_prompt": same_prompt,
        "same_target_object": same_target_object,
        "different_outcome": different_outcome,
        "length_delta": length_delta,
        "source_outcome": source_manifest.outcome,
        "candidate_outcome": candidate_manifest.outcome,
        "source_probe_correct": source_correct,
        "candidate_probe_correct": candidate_correct,
        "source_split_category": source_split or None,
        "candidate_split_category": candidate_split or None,
        "source_confidence": source_confidence,
        "candidate_confidence": candidate_confidence,
        "confidence_delta": confidence_delta,
    }
    return score, _dedupe_reasons(reasons), _jsonable(metrics)

def _workbench_payload(dataset: TraceDataset) -> dict[str, Any]:
    return workbench_manifest(dataset)

def _episode_annotations_payload(root: Path, *, trace_id: str | None = None) -> dict[str, Any]:
    annotations = _read_episode_annotations(root)
    if trace_id:
        return {
            "annotation": annotations.get(
                trace_id,
                _empty_episode_annotation(trace_id),
            )
        }
    return {"annotations": annotations}

def _save_episode_annotation_payload(root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    trace_id = str(payload.get("trace_id") or "").strip()
    if not trace_id:
        raise ValueError("Episode annotation requires trace_id")
    annotations = _read_episode_annotations(root)
    current = annotations.get(trace_id, _empty_episode_annotation(trace_id))
    annotation = {
        **current,
        "trace_id": trace_id,
        "starred": bool(payload.get("starred", current.get("starred", False))),
        "notes": str(payload.get("notes", current.get("notes", ""))),
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    annotations[trace_id] = annotation
    path = _episode_annotations_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(annotations, indent=2, sort_keys=True), encoding="utf-8")
    return {"annotation": annotation}

def _read_episode_annotations(root: Path) -> dict[str, dict[str, Any]]:
    path = _episode_annotations_path(root)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    annotations: dict[str, dict[str, Any]] = {}
    for trace_id, value in payload.items():
        if isinstance(value, Mapping):
            annotations[str(trace_id)] = {
                **_empty_episode_annotation(str(trace_id)),
                **dict(value),
                "trace_id": str(trace_id),
            }
    return annotations

def _episode_annotations_path(root: Path) -> Path:
    return root / "annotations" / "episode_annotations.json"

def _empty_episode_annotation(trace_id: str) -> dict[str, Any]:
    return {"trace_id": trace_id, "starred": False, "notes": "", "updated_utc": None}

def _dataset_signature(root: Path) -> tuple[int, int]:
    """Cheap cache key for dataset-level metadata endpoints."""
    if (root / TraceBundle.MANIFEST).exists():
        paths = [root / TraceBundle.MANIFEST, root / TraceBundle.ARTIFACT_INDEX]
        trace_count = 1
    else:
        paths = [
            root / TraceBundle.ARTIFACT_INDEX,
            root / "vla_lens" / TraceBundle.ARTIFACT_INDEX,
            root / "episode_plan.csv",
            root / "episode_plan.json",
            root / "capture_status.jsonl",
            *list(_lerobot_signature_paths(root)),
            *_workbench_signature_paths(root),
        ]
        trace_count = _dataset_trace_count_hint(root)
    existing = [path for path in paths if path.exists()]
    latest_mtime = max((path.stat().st_mtime_ns for path in existing), default=0)
    return trace_count, latest_mtime

def _lerobot_signature_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pattern in (
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.jsonl",
        "meta/tasks.parquet",
        "meta/episodes/**/*.parquet",
        "vla_lens/overlay.json",
        "vla_lens/tables/*.parquet",
        "*/meta/info.json",
        "*/meta/stats.json",
        "*/meta/tasks.jsonl",
        "*/meta/tasks.parquet",
        "*/meta/episodes/**/*.parquet",
        "*/vla_lens/overlay.json",
        "*/vla_lens/tables/*.parquet",
        "**/vla_lens/episodes/*/manifest.json",
    ):
        paths.extend(root.glob(pattern))
    return paths

def _workbench_signature_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for directory in (root / "workbench", root / "vla_lens" / "workbench"):
        if directory.exists():
            paths.extend(directory.rglob("*.json"))
    return paths

def _dataset_trace_count_hint(root: Path) -> int:
    refs_paths = _unique_paths(
        [
            root / "vla_lens" / "tables" / "episode_refs.parquet",
            *list(root.glob("**/vla_lens/tables/episode_refs.parquet")),
        ]
    )
    refs_count = 0
    for path in refs_paths:
        if not path.exists():
            continue
        try:
            refs_count += int(len(pd.read_parquet(path, columns=["trace_id"])))
        except Exception:
            continue
    if refs_count:
        return refs_count
    episode_plan = root / "episode_plan.csv"
    if episode_plan.exists():
        try:
            with episode_plan.open("r", encoding="utf-8") as handle:
                return max(0, sum(1 for _line in handle) - 1)
        except OSError:
            return 0
    return 0

def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        resolved = path.resolve() if path.exists() else path
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out

def _lens_arrays_payload(dataset: TraceDataset) -> dict[str, Any]:
    arrays = [array.to_dict() for array in lens_array_catalog(dataset)]
    return {"lens_arrays": arrays, "total": len(arrays)}

def _lens_array_meta_payload(dataset: TraceDataset, array_id: str) -> dict[str, Any]:
    return lens_array_meta(dataset, array_id).to_dict()

def _lens_array_slice_payload(
    dataset: TraceDataset,
    array_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    max_values = int(payload.get("max_values", 4096))
    selection = payload.get("selection") or payload.get("axis_values") or {}
    if not isinstance(selection, dict):
        raise TypeError("LensArray slice selection must be an object")
    return slice_lens_array(dataset, array_id, selection=selection, max_values=max_values)

def _cohorts_payload(dataset: TraceDataset) -> dict[str, Any]:
    cohorts = [cohort.to_dict() for cohort in list_cohorts(dataset)]
    return {"cohorts": cohorts, "total": len(cohorts)}

def _save_cohort_payload(dataset: TraceDataset, payload: dict[str, Any]) -> dict[str, Any]:
    from vla_lens.workbench import CohortSpec

    cohort_payload = payload.get("cohort", payload)
    if not isinstance(cohort_payload, dict):
        raise TypeError("Cohort payload must be an object")
    cohort = save_cohort(dataset, CohortSpec.from_dict(cohort_payload))
    return {"cohort": cohort.to_dict(), **_cohorts_payload(dataset)}

def _save_cohort_from_selection_payload(
    dataset: TraceDataset,
    payload: dict[str, Any],
) -> dict[str, Any]:
    selection_payload = payload.get("selection", payload)
    if not isinstance(selection_payload, dict):
        raise TypeError("Selection payload must be an object")
    cohort = cohort_from_selection(
        dataset,
        SelectionState.from_dict(selection_payload),
        label=payload.get("label"),
        cohort_id=payload.get("cohort_id"),
    )
    saved = save_cohort(dataset, cohort)
    return {"cohort": saved.to_dict(), **_cohorts_payload(dataset)}

def _analysis_runs_payload(dataset: TraceDataset) -> dict[str, Any]:
    runs = [run.to_dict() for run in list_analysis_runs(dataset)]
    return {"analysis_runs": runs, "total": len(runs)}

def _intervention_runs_payload(dataset: TraceDataset) -> dict[str, Any]:
    runs = [run.to_dict() for run in list_intervention_runs(dataset)]
    return {"intervention_runs": runs, "total": len(runs)}

def _save_analysis_run_payload(dataset: TraceDataset, payload: dict[str, Any]) -> dict[str, Any]:
    run_payload = payload.get("analysis_run", payload.get("run", payload))
    if not isinstance(run_payload, dict):
        raise TypeError("Analysis run payload must be an object")
    run = save_analysis_run(dataset, AnalysisRunSpec.from_dict(run_payload))
    return {"analysis_run": run.to_dict(), **_analysis_runs_payload(dataset)}

def _save_intervention_run_payload(
    dataset: TraceDataset,
    payload: dict[str, Any],
) -> dict[str, Any]:
    run_payload = payload.get("intervention_run", payload.get("run", payload))
    if not isinstance(run_payload, dict):
        raise TypeError("Intervention run payload must be an object")
    run = save_intervention_run(dataset, InterventionRunSpec.from_dict(run_payload))
    return {"intervention_run": run.to_dict(), **_intervention_runs_payload(dataset)}

def _workspaces_payload(dataset: TraceDataset) -> dict[str, Any]:
    workspaces = [workspace.to_dict() for workspace in list_workspaces(dataset)]
    return {"workspaces": workspaces, "total": len(workspaces)}

def _cohort_compare_payload(dataset: TraceDataset, payload: dict[str, Any]) -> dict[str, Any]:
    left = str(payload.get("left") or payload.get("left_cohort_id"))
    right = str(payload.get("right") or payload.get("right_cohort_id"))
    return compare_cohorts(dataset, left, right)

def _unit_profile_payload(dataset: TraceDataset, query: dict[str, list[str]]) -> dict[str, Any]:
    unit = UnitRef(
        kind=query.get("kind", ["neuron"])[0],
        site_id=query.get("site_id", [None])[0],
        index=int(query.get("unit", query.get("index", ["0"]))[0]),
    )
    selection: dict[str, Any] = {"axis_values": {}}
    for axis in ["layer", "token_kind", "module", "tensor_type", "episode"]:
        if axis in query:
            selection["axis_values"][axis] = query[axis]
    return unit_profile(
        dataset,
        unit,
        selection=selection,
        top_k=int(query.get("top_k", ["12"])[0]),
    )

def _projection_payload(dataset: TraceDataset, payload: dict[str, Any]) -> dict[str, Any]:
    selection = payload.get("selection") or {"selection_id": "projection", "axis_values": {}}
    if not isinstance(selection, dict):
        raise TypeError("Projection selection must be an object")
    return projection_points(dataset, selection=selection, limit=int(payload.get("limit", 500)))

def _graph_payload(dataset: TraceDataset, payload: dict[str, Any]) -> dict[str, Any]:
    selection = payload.get("selection", payload)
    if not isinstance(selection, dict):
        raise TypeError("Graph selection must be an object")
    return graph_from_selection(dataset, selection)

def _table_query_payload(dataset: TraceDataset, payload: dict[str, Any]) -> dict[str, Any]:
    table = str(payload.get("table") or "episodes")
    filters = payload.get("filters") or {}
    columns = payload.get("columns") or None
    if not isinstance(filters, dict):
        raise TypeError("Table query filters must be an object")
    if columns is not None and not isinstance(columns, list):
        raise TypeError("Table query columns must be a list")
    return query_table(
        dataset,
        table=table,
        filters=filters,
        columns=columns,
        limit=int(payload.get("limit", 200)),
    )

def _save_workspace_payload(dataset: TraceDataset, payload: dict[str, Any]) -> dict[str, Any]:
    workspace_payload = payload.get("workspace", payload)
    if not isinstance(workspace_payload, dict):
        raise TypeError("Workspace payload must be an object")
    workspace = save_workspace(dataset, SavedWorkspace.from_dict(workspace_payload))
    return {"workspace": workspace.to_dict(), **_workspaces_payload(dataset)}

def _resolve_selection_payload(dataset: TraceDataset, payload: dict[str, Any]) -> dict[str, Any]:
    selection_payload = payload.get("selection", payload)
    request = payload.get("request", ())
    selection = SelectionState.from_dict(selection_payload)
    return resolve_selection(dataset, selection, request=request)

def _dataset_diagnostics_payload(dataset: TraceDataset) -> dict[str, Any]:
    status = diagnostics_status(dataset)
    return _diagnostics_payload(status)

def _run_dataset_diagnostics_payload(dataset: TraceDataset) -> dict[str, Any]:
    artifact = run_dataset_diagnostics(dataset)
    status = diagnostics_status(dataset)
    payload = _diagnostics_payload(status)
    payload["artifact"] = _jsonable(artifact.to_dict())
    return payload

def _create_outcome_probe_payload(dataset: TraceDataset) -> dict[str, Any]:
    saved = train_probe_artifact_from_spec(dataset, _default_outcome_probe_spec(dataset))
    return {
        "artifact": _jsonable(saved.artifact.to_dict()),
        "results": _jsonable(saved.results.to_dict("records")),
        "artifacts": _artifacts_payload(dataset),
    }

def _create_target_object_probe_payload(dataset: TraceDataset) -> dict[str, Any]:
    saved = save_target_object_encoding_artifact(dataset, name="Dashboard target-object encoding")
    return {
        "artifact": _jsonable(saved.artifact.to_dict()),
        "arrays": {
            "metric_cube": list(saved.metric_cube.shape),
            "baseline_cube": list(saved.baseline_cube.shape),
            "delta_cube": list(saved.delta_cube.shape),
        },
        "artifacts": _artifacts_payload(dataset),
    }

def _create_action_generation_payload(dataset: TraceDataset) -> dict[str, Any]:
    saved = save_action_generation_artifact(dataset, name="Dashboard action generation")
    return {
        "artifact": _jsonable(saved.artifact.to_dict()),
        "artifacts": _artifacts_payload(dataset),
    }

def _default_target_object_probe_spec(dataset: TraceDataset) -> dict[str, Any]:
    spec = _default_outcome_probe_spec(dataset)
    spec["name"] = "Dashboard target-object encoding probe"
    spec["target"] = {"kind": "target_object"}
    spec["split"] = {"kind": "random_episode"}
    spec["baseline"] = ["majority_class", "benchmark", "task"]
    spec["sweep"] = "layer"
    return spec

def _default_outcome_probe_spec(dataset: TraceDataset) -> dict[str, Any]:
    model_sites = dataset.model_site_index
    module = "pi05.expert.layers.*"
    tensor_type = "hidden_mean"
    token_kind = "action"
    if not model_sites.empty and "module" in model_sites:
        modules = model_sites["module"].astype(str)
        if modules.str.contains("pi05.expert", regex=False).any():
            module = "pi05.expert.layers.*"
            tensor_type = "hidden_mean"
            token_kind = "action"
        elif modules.str.contains("action_head", regex=False).any():
            module = "action_head.layers.*.resid"
            tensor_type = "resid"
            token_kind = "action"
        else:
            module = str(modules.iloc[0])
            if "tensor_type" in model_sites:
                tensor_type = str(model_sites["tensor_type"].astype(str).iloc[0])
            token_kind = None
    return {
        "name": "Dashboard outcome probe",
        "target": {"kind": "outcome"},
        "features": {
            "module": module,
            "tensor_type": tensor_type,
            "token_kind": token_kind,
            "layers": None,
            "timesteps": "all",
            "generation_step": None,
            "reduction": "mean",
        },
        "split": {"kind": "heldout_benchmark"},
        "baseline": ["majority_class", "benchmark", "target_object", "task"],
        "sweep": "layer",
    }

def _diagnostics_payload(status: dict[str, Any]) -> dict[str, Any]:
    latest = status.get("latest")
    return {
        "fingerprint": status.get("fingerprint"),
        "stale": bool(status.get("stale", True)),
        "latest": _jsonable(latest) if latest else None,
    }

def _episode_payload(bundle: TraceBundle) -> dict[str, Any]:
    return {
        **_manifest_payload(bundle),
        "cameras": bundle.cameras(),
        "artifacts": (
            bundle.artifact_index.to_dict("records") if not bundle.artifact_index.empty else []
        ),
        "arrays": bundle.array_index.to_dict("records") if not bundle.array_index.empty else [],
    }

def _artifacts_payload(dataset: TraceDataset) -> dict[str, Any]:
    artifacts = [_artifact_record_payload(record) for record in _artifact_records(dataset)]
    counts: dict[str, int] = {}
    for artifact in artifacts:
        key = str(artifact.get("artifact_type") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return {
        "artifacts": artifacts,
        "counts": counts,
        "total": len(artifacts),
    }

def _artifact_detail_payload(dataset: TraceDataset, artifact_id: str) -> dict[str, Any]:
    artifact = dataset.load_artifact(artifact_id)
    arrays: list[dict[str, Any]] = []
    for name, path in artifact.arrays.items():
        array = dataset.load_artifact_array(artifact, name, mmap=True)
        arrays.append(
            {
                "name": name,
                "path": path,
                "shape": [int(item) for item in array.shape],
                "dtype": str(array.dtype),
                "summary": _array_summary(array),
                "preview": _array_preview(array),
            }
        )
    return {
        "artifact": _jsonable(artifact.to_dict()),
        "arrays": arrays,
    }

def _artifact_summary(dataset: TraceDataset) -> dict[str, Any]:
    records = _artifact_records(dataset)
    counts: dict[str, int] = {}
    for record in records:
        key = str(record.get("artifact_type") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return {"total": len(records), "counts": counts}

def _artifact_records(dataset: TraceDataset) -> list[dict[str, Any]]:
    table = dataset.artifact_index
    if table.empty:
        return []
    return table.sort_values("created_utc", ascending=False, na_position="last").to_dict("records")

def _artifact_record_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = {str(key): _json_scalar(value) for key, value in record.items()}
    for key in ["selector", "method", "metrics", "arrays", "display", "tags", "source_trace_ids"]:
        payload[key] = _jsonable(_json_parse(payload.get(key)))
    return payload
