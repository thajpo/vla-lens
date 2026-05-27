"""Workbench-oriented dataset payload helpers."""

from __future__ import annotations

from typing import Any

from vla_lens.traces import TraceDataset
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


def _workbench_payload(dataset: TraceDataset) -> dict[str, Any]:
    return workbench_manifest(dataset)


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
