"""Selection workbench primitives."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from vla_lens.traces import TraceDataset
from vla_lens.workbench.catalog import (
    default_panel_recipes,
    lens_array_catalog,
    model_site_catalog,
)
from vla_lens.workbench.schema import (
    LensArraySpec,
    ModelSiteSpec,
    PanelRecipe,
    SelectionState,
    UnitRef,
    normalize_selection,
)
from vla_lens.workbench.utils import (
    _as_set,
    _first_scalar,
    _jsonable_record,
    _jsonable_scalar,
    _numeric_summary,
)
from vla_lens.workbench.validation import (
    _array_value,
    _axis_index,
)


def unit_profile(
    dataset: TraceDataset,
    unit: UnitRef | Mapping[str, Any],
    *,
    selection: SelectionState | Mapping[str, Any] | None = None,
    top_k: int = 12,
) -> dict[str, Any]:
    """Resolve a neuron/feature-style handle to linked episode examples.

    The profile is deliberately example-first: it tells the UI where the unit
    fires and which arrays/sites can inspect it, without claiming what the unit
    means or that the association is causal.
    """
    unit_ref = unit if isinstance(unit, UnitRef) else UnitRef.from_dict(unit)
    state = _unit_selection(unit_ref, selection)
    arrays = [
        array
        for array in _matching_lens_arrays(dataset, state)
        if array.kind == "tensor" and "unit" in array.dims
    ]
    rows: list[dict[str, Any]] = []
    for spec in arrays:
        rows.extend(_unit_examples_for_array(dataset, spec, unit_ref, top_k=top_k))
    rows = sorted(rows, key=lambda row: abs(float(row["value"])), reverse=True)[:top_k]
    values = np.array([row["value"] for row in rows], dtype=np.float32)
    histograms = _unit_axis_histograms(rows)
    probe_associations = _unit_probe_associations(dataset, unit_ref, state)
    action_associations = _unit_action_associations(dataset, state)
    return {
        "unit_ref": unit_ref.to_dict(),
        "selection": state.to_dict(),
        "top_examples": rows,
        "axis_histograms": histograms,
        "probe_associations": probe_associations,
        "action_associations": action_associations,
        "causal": False,
        "association_kind": "observational_activation",
        "summary": _numeric_summary(values),
        "lens_arrays": [array.to_dict() for array in arrays[:50]],
        "model_sites": [site.to_dict() for site in _matching_model_sites(dataset, state)[:50]],
        "suggested_panels": [
            panel.to_dict()
            for panel in _compatible_panels(arrays, _matching_model_sites(dataset, state), state)
        ],
    }

def resolve_selection(
    dataset: TraceDataset,
    selection: SelectionState | Mapping[str, Any],
    *,
    request: Sequence[str] = (),
) -> dict[str, Any]:
    """Resolve a typed selection into episodes, arrays, sites, and compatible panels."""
    del request
    state = normalize_selection(selection)
    episodes = _matching_episode_records(dataset, state)
    arrays = _matching_lens_arrays(dataset, state)
    sites = _matching_model_sites(dataset, state)
    panels = _compatible_panels(arrays, sites, state)
    examples = _target_object_examples(dataset, state) or _representative_examples(episodes, state)
    action_cell = _action_stabilization_cell(dataset, state)
    if action_cell and not examples.get("matching"):
        examples = _action_stabilization_examples(action_cell)
    response = {
        "selection": state.to_dict(),
        "episodes": episodes[:50],
        "examples": examples,
        "lens_arrays": [array.to_dict() for array in arrays[:50]],
        "model_sites": [site.to_dict() for site in sites[:50]],
        "suggested_panels": [panel.to_dict() for panel in panels],
        "provenance": _selection_provenance(dataset, state, arrays, sites),
        "valid_references": _valid_references(episodes, state),
    }
    cell = _target_object_cell(dataset, state)
    if cell:
        response["target_object_cell"] = cell
    if action_cell:
        response["action_stabilization_cell"] = action_cell
    return response

def _matching_episode_records(
    dataset: TraceDataset,
    selection: SelectionState,
) -> list[dict[str, Any]]:
    frame = dataset.episode_index.copy()
    axes = selection.axis_values
    if "episode" in axes:
        allowed = _as_set(axes["episode"])
        frame = frame.loc[
            frame["trace_id"].astype(str).isin(allowed)
            | frame["episode_id"].astype(str).isin(allowed)
        ]
    for axis, column in [
        ("object", "target_object"),
        ("label", "outcome"),
        ("outcome", "outcome"),
        ("task", "task_id"),
    ]:
        if axis in axes and column in frame:
            frame = frame.loc[frame[column].astype(str).isin(_as_set(axes[axis]))]
    return [_jsonable_record(record) for record in frame.head(50).to_dict("records")]

def _matching_lens_arrays(
    dataset: TraceDataset,
    selection: SelectionState,
) -> list[LensArraySpec]:
    arrays = list(lens_array_catalog(dataset))
    axes = selection.axis_values
    analysis_allowed: set[str] = set()
    if "analysis_run" in axes:
        analysis_allowed = _as_set(axes["analysis_run"])
        arrays = [
            array
            for array in arrays
            if str(array.provenance.get("analysis_run_id")) in analysis_allowed
            or str(array.provenance.get("artifact_id")) in analysis_allowed
            or array.kind != "artifact_array"
        ]
        arrays = sorted(
            arrays,
            key=lambda array: (
                0
                if array.kind == "artifact_array"
                and (
                    str(array.provenance.get("analysis_run_id")) in analysis_allowed
                    or str(array.provenance.get("artifact_id")) in analysis_allowed
                )
                else 1,
                array.array_id,
            ),
        )
    if "episode" in axes:
        allowed = _as_set(axes["episode"])
        arrays = [
            array
            for array in arrays
            if str(array.provenance.get("trace_id")) in allowed
            or str(array.provenance.get("episode_id")) in allowed
            or array.kind == "artifact_array"
        ]
    if "layer" in axes:
        allowed = {int(value) for value in _as_set(axes["layer"]) if str(value).isdigit()}
        arrays = [
            array
            for array in arrays
            if array.provenance.get("layer") in allowed or "layer" not in array.provenance
        ]
    if "token_kind" in axes:
        allowed = _as_set(axes["token_kind"])
        arrays = [
            array
            for array in arrays
            if str(array.provenance.get("token_kind")) in allowed
            or "token_kind" not in array.provenance
        ]
    for axis in [
        "timestep",
        "policy_call",
        "camera",
        "image_patch",
        "unit",
        "generation_step",
        "action_horizon",
        "action_dim",
    ]:
        if axis in axes:
            arrays = [array for array in arrays if axis in array.dims]
    if analysis_allowed:
        arrays = sorted(
            arrays,
            key=lambda array: (
                not (
                    array.kind == "artifact_array"
                    and (
                        str(array.provenance.get("analysis_run_id")) in analysis_allowed
                        or str(array.provenance.get("artifact_id")) in analysis_allowed
                    )
                ),
                array.array_id,
            ),
        )
    return arrays

def _matching_model_sites(
    dataset: TraceDataset,
    selection: SelectionState,
) -> list[ModelSiteSpec]:
    sites = list(model_site_catalog(dataset))
    axes = selection.axis_values
    if "layer" in axes:
        allowed = {int(value) for value in _as_set(axes["layer"]) if str(value).isdigit()}
        sites = [site for site in sites if site.layer in allowed]
    if "token_kind" in axes:
        allowed = _as_set(axes["token_kind"])
        sites = [site for site in sites if str(site.token_kind) in allowed]
    if "module" in axes:
        allowed = _as_set(axes["module"])
        sites = [site for site in sites if site.module in allowed or site.site_id in allowed]
    if "site_id" in axes:
        allowed = _as_set(axes["site_id"])
        sites = [site for site in sites if site.site_id in allowed]
    if "tensor_type" in axes:
        allowed = _as_set(axes["tensor_type"])
        sites = [site for site in sites if str(site.tensor_type) in allowed]
    return sites

def _compatible_panels(
    arrays: Sequence[LensArraySpec],
    sites: Sequence[ModelSiteSpec],
    selection: SelectionState,
) -> tuple[PanelRecipe, ...]:
    del sites
    kinds = {array.kind for array in arrays}
    dims = {dim for array in arrays for dim in array.dims}
    panels: list[PanelRecipe] = []
    for panel in default_panel_recipes():
        if panel.panel_type == "episode.viewer" and {"image_sequence", "tensor"} & kinds:
            panels.append(panel)
        elif panel.panel_type == "heatmap" and {"layer", "timestep"} & dims:
            panels.append(panel)
        elif panel.panel_type == "inspector":
            panels.append(panel)
        elif panel.panel_type == "confusion_matrix" and "analysis_run" in selection.axis_values:
            panels.append(panel)
        elif panel.panel_type == "unit.profile" and (
            "unit" in selection.axis_values or selection.unit_refs
        ):
            panels.append(panel)
        elif panel.panel_type == "image.patch_overlay" and "image_patch" in dims:
            panels.append(panel)
        elif panel.panel_type == "action.horizon_heatmap" and "action_horizon" in dims:
            panels.append(panel)
        elif panel.panel_type == "projection.scatter" and "projection_x" in dims:
            panels.append(panel)
        elif panel.panel_type == "graph.explorer" and (
            "unit" in selection.axis_values or "analysis_run" in selection.axis_values
        ):
            panels.append(panel)
        elif panel.panel_type == "examples.table":
            panels.append(panel)
    return tuple(panels)

def _representative_examples(
    episodes: Sequence[Mapping[str, Any]],
    selection: SelectionState,
) -> dict[str, list[dict[str, Any]]]:
    timestep = selection.axis_values.get("timestep")
    selected_timestep = _first_scalar(timestep)
    rows = []
    for episode in episodes[:12]:
        rows.append(
            {
                "example_id": f"{episode.get('trace_id')}:{selected_timestep}",
                "trace_id": episode.get("trace_id"),
                "episode_id": episode.get("episode_id"),
                "timestep": selected_timestep,
                "task_id": episode.get("task_id"),
                "target_object": episode.get("target_object"),
                "outcome": episode.get("outcome"),
            }
        )
    return {"matching": rows}

def _target_object_cell(
    dataset: TraceDataset,
    selection: SelectionState,
) -> dict[str, Any] | None:
    artifact = _selection_analysis_artifact(dataset, selection)
    if artifact is None or artifact.display.get("kind") != "target_object_encoding":
        return None
    layer = _first_scalar(selection.axis_values.get("layer"))
    timestep = _first_scalar(selection.axis_values.get("timestep"))
    token_kind = _first_scalar(selection.axis_values.get("token_kind"))
    if layer is None or timestep is None or token_kind is None:
        return None
    for cell in artifact.display.get("cell_details") or ():
        if (
            str(cell.get("layer")) == str(layer)
            and str(cell.get("timestep")) == str(timestep)
            and str(cell.get("token_kind")) == str(token_kind)
        ):
            return dict(cell)
    records = artifact.display.get("records") or ()
    for record in records:
        if (
            str(record.get("layer")) == str(layer)
            and str(record.get("timestep")) == str(timestep)
            and str(record.get("token_kind")) == str(token_kind)
        ):
            return {
                **dict(record),
                "split_summary": artifact.display.get("split_summary") or {},
                "confusion_matrix": artifact.display.get("confusion_matrix") or [],
                "linked_examples": artifact.display.get("linked_examples") or [],
            }
    return None

def _target_object_examples(
    dataset: TraceDataset,
    selection: SelectionState,
) -> dict[str, list[dict[str, Any]]]:
    cell = _target_object_cell(dataset, selection)
    if not cell:
        return {}
    rows = [
        _target_example_row(example, selection)
        for example in cell.get("linked_examples") or ()
        if isinstance(example, Mapping)
    ]
    if not rows:
        return {}
    buckets: dict[str, list[dict[str, Any]]] = {
        "matching": rows,
        "correct": [],
        "false_positive": [],
        "false_negative": [],
        "low_margin": [],
    }
    for row in rows:
        actual = str(row.get("actual") or row.get("target_object") or "")
        predicted = str(row.get("predicted") or "")
        status = str(row.get("prediction_status") or "")
        if status == "correct" or (actual and predicted and actual == predicted):
            buckets["correct"].append(row)
        elif status == "false_negative":
            buckets["false_negative"].append(row)
        elif status == "false_positive" or (actual and predicted and actual != predicted):
            buckets["false_positive"].append(row)
        margin = row.get("margin")
        if margin is not None and abs(float(margin)) <= 0.1:
            buckets["low_margin"].append(row)
    return {key: value for key, value in buckets.items() if value}

def _target_example_row(
    example: Mapping[str, Any],
    selection: SelectionState,
) -> dict[str, Any]:
    timestep = example.get("timestep", _first_scalar(selection.axis_values.get("timestep")))
    trace_id = example.get("trace_id")
    actual = example.get("actual") or example.get("target_object")
    predicted = example.get("predicted")
    status = example.get("prediction_status")
    if status is None and actual is not None and predicted is not None:
        status = "correct" if str(actual) == str(predicted) else "false_positive"
    return {
        **dict(example),
        "example_id": str(example.get("example_id") or f"{trace_id}:{timestep}"),
        "trace_id": trace_id,
        "episode_id": example.get("episode_id"),
        "timestep": timestep,
        "target_object": actual,
        "actual": actual,
        "predicted": predicted,
        "prediction_status": status,
    }

def _action_stabilization_cell(
    dataset: TraceDataset,
    selection: SelectionState,
) -> dict[str, Any] | None:
    artifact = _selection_analysis_artifact(dataset, selection)
    if artifact is None or artifact.display.get("kind") != "action_generation":
        return None
    episode_index = _axis_index(selection.axis_values.get("episode"), default=0)
    policy_call = _axis_index(selection.axis_values.get("policy_call"), default=0)
    generation_step = _axis_index(selection.axis_values.get("generation_step"), default=0)
    horizon = _axis_index(selection.axis_values.get("action_horizon"), default=0)
    action_dim = _axis_index(selection.axis_values.get("action_dim"), default=0)
    cell: dict[str, Any] = {
        "episode_index": episode_index,
        "policy_call": policy_call,
        "generation_step": generation_step,
        "action_horizon": horizon,
        "action_dim": action_dim,
        "analysis_run": artifact.artifact_id,
    }
    for name in ["delta_to_final", "step_delta", "final_vs_executed"]:
        try:
            array = np.asarray(dataset.load_artifact_array(artifact, name, mmap=True))
        except KeyError:
            continue
        if name in {"delta_to_final", "step_delta"} and array.ndim == 4:
            cell[name] = _array_value(array, episode_index, policy_call, generation_step, horizon)
        elif name == "final_vs_executed" and array.ndim == 4:
            cell[name] = _array_value(array, episode_index, policy_call, horizon, action_dim)
    episodes = artifact.display.get("episodes") or []
    if episode_index < len(episodes):
        cell["episode"] = dict(episodes[episode_index])
        cell["trace_id"] = episodes[episode_index].get("trace_id")
        unstable = episodes[episode_index].get("unstable_calls") or []
        cell["unstable_call"] = next(
            (dict(row) for row in unstable if int(row.get("call_index", -1)) == policy_call),
            None,
        )
    return cell

def _action_stabilization_examples(cell: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    trace_id = cell.get("trace_id")
    if not trace_id:
        return {"matching": []}
    timestep = None
    unstable = cell.get("unstable_call")
    if isinstance(unstable, Mapping):
        timestep = unstable.get("timestep")
    return {
        "matching": [
            {
                "example_id": f"{trace_id}:{cell.get('policy_call')}:{cell.get('generation_step')}",
                "trace_id": trace_id,
                "episode_id": cell.get("episode", {}).get("episode_id")
                if isinstance(cell.get("episode"), Mapping)
                else None,
                "timestep": timestep,
                "policy_call": cell.get("policy_call"),
                "generation_step": cell.get("generation_step"),
                "action_horizon": cell.get("action_horizon"),
                "delta_to_final": cell.get("delta_to_final"),
                "step_delta": cell.get("step_delta"),
                "final_vs_executed": cell.get("final_vs_executed"),
            }
        ]
    }

def _selection_analysis_artifact(
    dataset: TraceDataset,
    selection: SelectionState,
) -> Any | None:
    run_id = _first_scalar(selection.axis_values.get("analysis_run"))
    if not run_id:
        return None
    try:
        return dataset.load_artifact(str(run_id))
    except (FileNotFoundError, KeyError, ValueError):
        return None

def _selection_provenance(
    dataset: TraceDataset,
    selection: SelectionState,
    arrays: Sequence[LensArraySpec],
    sites: Sequence[ModelSiteSpec],
) -> dict[str, Any]:
    artifact = _selection_analysis_artifact(dataset, selection)
    artifact_id = artifact.artifact_id if artifact is not None else None
    return {
        "source_panel_id": selection.source_panel_id,
        "intent": selection.intent,
        "analysis_run": _first_scalar(selection.axis_values.get("analysis_run")),
        "artifact_id": artifact_id,
        "artifact_type": artifact.artifact_type if artifact is not None else None,
        "array_ids": [array.array_id for array in arrays[:50]],
        "model_site_ids": [site.site_id for site in sites[:50]],
    }

def _valid_references(
    episodes: Sequence[Mapping[str, Any]],
    selection: SelectionState,
) -> dict[str, Any]:
    timestep = _first_scalar(selection.axis_values.get("timestep"))
    return {
        "episodes": [
            {"trace_id": episode.get("trace_id"), "episode_id": episode.get("episode_id")}
            for episode in episodes[:50]
            if episode.get("trace_id") or episode.get("episode_id")
        ],
        "timestep": timestep,
        "policy_call": _first_scalar(selection.axis_values.get("policy_call")),
    }

def _load_lens_array(dataset: TraceDataset, spec: LensArraySpec) -> np.ndarray:
    source = spec.provenance.get("source")
    trace_id = spec.provenance.get("trace_id")
    if source == "trace_bundle" and trace_id:
        return dataset.bundle(str(trace_id)).array(spec.label, mmap=True)
    if source == "model_sites" and trace_id:
        return dataset.bundle(str(trace_id)).model_site(spec.label, mmap=True)
    if spec.kind == "artifact_array":
        artifact_id = str(spec.provenance.get("artifact_id"))
        artifact = dataset.load_artifact(artifact_id)
        return dataset.load_artifact_array(artifact, spec.label, mmap=True)
    raise KeyError(f"LensArray '{spec.array_id}' cannot be loaded by the local data plane")

def _unit_selection(
    unit: UnitRef,
    selection: SelectionState | Mapping[str, Any] | None,
) -> SelectionState:
    if selection is None:
        axes: dict[str, Any] = {}
        if unit.index is not None:
            axes["unit"] = [unit.index]
        return SelectionState(
            selection_id=f"unit_{unit.kind}_{unit.index or unit.name or 'selected'}",
            axis_values=axes,
            unit_refs=(unit,),
            source_panel_id="unit.explorer",
        )
    state = (
        selection if isinstance(selection, SelectionState) else SelectionState.from_dict(selection)
    )
    axes = dict(state.axis_values)
    if unit.index is not None and "unit" not in axes:
        axes["unit"] = [unit.index]
    return SelectionState(
        selection_id=state.selection_id,
        axis_values=axes,
        unit_refs=tuple([*state.unit_refs, unit]),
        cohort_refs=state.cohort_refs,
        source_panel_id=state.source_panel_id,
        intent=state.intent,
    )

def _unit_examples_for_array(
    dataset: TraceDataset,
    spec: LensArraySpec,
    unit: UnitRef,
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    if unit.site_id:
        sites = [site for site in model_site_catalog(dataset) if site.site_id == unit.site_id]
        if sites and not _array_matches_site(spec, sites[0]):
            return []
    array = np.asarray(_load_lens_array(dataset, spec), dtype=np.float32)
    if not spec.dims or "unit" not in spec.dims or unit.index is None:
        return []
    unit_axis = spec.dims.index("unit")
    index = max(0, min(int(unit.index), array.shape[unit_axis] - 1))
    values = np.take(array, index, axis=unit_axis)
    flat = np.nan_to_num(values.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if flat.size == 0:
        return []
    order = np.argsort(np.abs(flat))[::-1][:top_k]
    reduced_dims = [dim for dim in spec.dims if dim != "unit"]
    reduced_shape = [size for i, size in enumerate(values.shape)]
    records: list[dict[str, Any]] = []
    for flat_index in order:
        coord = np.unravel_index(int(flat_index), tuple(reduced_shape)) if reduced_shape else ()
        axis_values = {dim: int(value) for dim, value in zip(reduced_dims, coord, strict=False)}
        trace_id = str(spec.provenance.get("trace_id") or "")
        episode = dataset.bundle(trace_id).manifest if trace_id else None
        records.append(
            {
                "array_id": spec.array_id,
                "trace_id": trace_id or None,
                "episode_id": episode.episode_id if episode else None,
                "task_id": episode.task_id if episode else None,
                "outcome": episode.outcome if episode else None,
                "axis_values": axis_values,
                "unit": index,
                "value": _jsonable_scalar(float(values[coord] if coord else values)),
            }
        )
    return records

def _unit_axis_histograms(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    histograms: dict[str, dict[str, int]] = {}
    for row in rows:
        for key in ["task_id", "outcome", "target_object", "trace_id"]:
            value = row.get(key)
            if value is None:
                continue
            bucket = histograms.setdefault(key, {})
            text = str(value)
            bucket[text] = bucket.get(text, 0) + 1
    return histograms

def _unit_probe_associations(
    dataset: TraceDataset,
    unit: UnitRef,
    selection: SelectionState,
) -> list[dict[str, Any]]:
    from vla_lens.workbench.api import list_analysis_runs

    del unit
    axes = selection.axis_values
    layer = _first_scalar(axes.get("layer"))
    token_kind = _first_scalar(axes.get("token_kind"))
    associations: list[dict[str, Any]] = []
    for run in list_analysis_runs(dataset):
        if run.workflow != "target_object_encoding":
            continue
        artifact_id = str(run.provenance.get("artifact_id") or run.run_id)
        try:
            artifact = dataset.load_artifact(artifact_id)
        except (FileNotFoundError, KeyError, ValueError):
            continue
        for record in artifact.display.get("records") or ():
            if layer is not None and str(record.get("layer")) != str(layer):
                continue
            if token_kind is not None and str(record.get("token_kind")) != str(token_kind):
                continue
            associations.append(
                {
                    "analysis_run": run.run_id,
                    "workflow": run.workflow,
                    "layer": record.get("layer"),
                    "timestep": record.get("timestep"),
                    "token_kind": record.get("token_kind"),
                    "score": record.get("score"),
                    "delta": record.get("delta"),
                    "causal": False,
                    "association_kind": "probe_diagnostic",
                }
            )
    return sorted(
        associations,
        key=lambda item: abs(float(item.get("delta") or 0.0)),
        reverse=True,
    )[:20]

def _unit_action_associations(
    dataset: TraceDataset,
    selection: SelectionState,
) -> list[dict[str, Any]]:
    action_arrays = [
        array
        for array in _matching_lens_arrays(dataset, selection)
        if "action_horizon" in array.dims
    ]
    return [
        {
            "array_id": array.array_id,
            "dims": list(array.dims),
            "causal": False,
            "association_kind": "shared_selection_axes",
        }
        for array in action_arrays[:20]
    ]

def _array_matches_site(array: LensArraySpec, site: ModelSiteSpec) -> bool:
    return (
        array.provenance.get("module") == site.module
        and array.provenance.get("layer") == site.layer
        and array.provenance.get("tensor_type") == site.tensor_type
        and array.provenance.get("token_kind") == site.token_kind
    )
