"""Api workbench primitives."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from vla_lens.traces import TraceDataset
from vla_lens.workbench.catalog import (
    axis_registry,
    dataset_id,
    default_panel_recipes,
    graph_edge_types,
    image_frame_catalog,
    lens_array_catalog,
    media_catalog,
    model_site_catalog,
    overlay_score_types,
    panel_registry,
    table_catalog,
    workflow_presets,
)
from vla_lens.workbench.schema import (
    MAX_JSON_SLICE_VALUES,
    AnalysisRunSpec,
    CohortSpec,
    GraphNodeRef,
    InterventionRunSpec,
    LensArraySpec,
    ResearchRunSpec,
    SavedWorkspace,
    SelectionState,
    normalize_axis_values,
    normalize_selection,
)
from vla_lens.workbench.selection import (
    _load_lens_array,
    _matching_episode_records,
    _matching_lens_arrays,
    _representative_examples,
    _target_object_examples,
    resolve_selection,
)
from vla_lens.workbench.utils import (
    _first_scalar,
    _jsonable_array,
    _jsonable_scalar,
    _numeric_summary,
    _preview_slices,
    _read_json,
    _safe_id,
    _selection_to_slices,
    _slice_payload,
    _workbench_dir,
    _write_json,
    _write_json_atomic,
)
from vla_lens.workbench.validation import (
    _cohort_delta_table,
    _cohort_episode_frame,
    _graph_edge,
    validate_workbench_contracts,
)

CAUSAL_CLAIM_LABELS = {"causal_local", "causal_cohort", "behavioral"}
INTERVENTION_TRIAL_KINDS = {"intervention"}
SUCCESSFUL_TRIAL_STATUSES = {"ok", "partial"}
CAUSAL_RUN_STATUSES = {"ok", "partial"}


def workbench_manifest(dataset: TraceDataset) -> dict[str, Any]:
    """Return the axis-native contract consumed by visual workbench frontends."""
    arrays = lens_array_catalog(dataset)
    sites = model_site_catalog(dataset)
    frames = image_frame_catalog(dataset)
    media = media_catalog(dataset)
    return {
        "schema_version": "0.1.0",
        "dataset_id": dataset_id(dataset),
        "axes": {axis.name: axis.to_dict() for axis in axis_registry(dataset)},
        "lens_arrays": [array.to_dict() for array in arrays],
        "tables": [table.to_dict() for table in table_catalog(dataset)],
        "image_frames": [frame.to_dict() for frame in frames],
        "media": [item.to_dict() for item in media],
        "model_sites": [site.to_dict() for site in sites],
        "panel_registry": {
            entry.recipe.panel_type: entry.to_dict() for entry in panel_registry().values()
        },
        "panel_recipes": [panel.to_dict() for panel in default_panel_recipes()],
        "workflow_presets": workflow_presets(dataset),
        "overlay_score_types": [score.to_dict() for score in overlay_score_types()],
        "graph_edge_types": graph_edge_types(),
        "cohorts": [cohort.to_dict() for cohort in list_cohorts(dataset)],
        "analysis_runs": [run.to_dict() for run in list_analysis_runs(dataset)],
        "research_runs": [run.to_dict() for run in list_research_runs(dataset)],
        "intervention_runs": [run.to_dict() for run in list_intervention_runs(dataset)],
        "saved_workspaces": [workspace.to_dict() for workspace in list_workspaces(dataset)],
        "contract_validation": validate_workbench_contracts(dataset),
    }

def list_cohorts(dataset: TraceDataset) -> tuple[CohortSpec, ...]:
    """Return saved cohorts for this dataset."""
    root = _workbench_dir(dataset, "cohorts", create=False)
    if not root.exists():
        return ()
    return tuple(CohortSpec.from_dict(_read_json(path)) for path in sorted(root.glob("*.json")))

def save_cohort(dataset: TraceDataset, cohort: CohortSpec) -> CohortSpec:
    """Persist a cohort in the dataset-local workbench store."""
    path = _workbench_dir(dataset, "cohorts", create=True) / f"{_safe_id(cohort.cohort_id)}.json"
    _write_json(path, cohort.to_dict())
    return cohort

def list_analysis_runs(dataset: TraceDataset) -> tuple[AnalysisRunSpec, ...]:
    """Return saved analysis-run records for this dataset."""
    root = _workbench_dir(dataset, "analysis_runs", create=False)
    if not root.exists():
        return ()
    return tuple(
        AnalysisRunSpec.from_dict(_read_json(path)) for path in sorted(root.glob("*.json"))
    )

def save_analysis_run(dataset: TraceDataset, run: AnalysisRunSpec) -> AnalysisRunSpec:
    """Persist computation provenance outside the artifact-browser UX."""
    path = _workbench_dir(dataset, "analysis_runs", create=True) / f"{_safe_id(run.run_id)}.json"
    _write_json(path, run.to_dict())
    return run

def list_research_runs(dataset: TraceDataset) -> tuple[ResearchRunSpec, ...]:
    """Return human-facing research lifecycle records, newest update first."""
    root = _workbench_dir(dataset, "research_runs", create=False)
    if not root.exists():
        return ()
    runs = [ResearchRunSpec.from_dict(_read_json(path)) for path in root.glob("*.json")]
    return tuple(sorted(runs, key=lambda run: (run.updated_utc, run.run_id), reverse=True))

def get_research_run(dataset: TraceDataset, run_id: str) -> ResearchRunSpec:
    """Return one research lifecycle record by its stable run id."""
    path = _workbench_dir(dataset, "research_runs", create=False) / f"{_safe_id(run_id)}.json"
    if not path.exists():
        raise KeyError(f"Unknown research run '{run_id}'")
    run = ResearchRunSpec.from_dict(_read_json(path))
    if run.run_id != run_id:
        raise KeyError(f"Unknown research run '{run_id}'")
    return run

def save_research_run(dataset: TraceDataset, run: ResearchRunSpec) -> ResearchRunSpec:
    """Atomically persist one lifecycle record independently of result artifacts."""
    path = _workbench_dir(dataset, "research_runs", create=True) / f"{_safe_id(run.run_id)}.json"
    _write_json_atomic(path, run.to_dict())
    return run

def list_intervention_runs(dataset: TraceDataset) -> tuple[InterventionRunSpec, ...]:
    """Return saved intervention/ablation readout records."""
    root = _workbench_dir(dataset, "intervention_runs", create=False)
    if not root.exists():
        return ()
    return tuple(
        InterventionRunSpec.from_dict(_read_json(path)) for path in sorted(root.glob("*.json"))
    )

def save_intervention_run(
    dataset: TraceDataset,
    run: InterventionRunSpec,
) -> InterventionRunSpec:
    """Persist a saved intervention readout without executing the intervention."""
    path = (
        _workbench_dir(dataset, "intervention_runs", create=True) / f"{_safe_id(run.run_id)}.json"
    )
    _write_json(path, run.to_dict())
    analysis_run = AnalysisRunSpec(
        run_id=run.run_id,
        workflow="intervention_readout",
        inputs={
            "intervention_type": run.intervention_type,
            "target": dict(run.target),
            "baseline": dict(run.baseline),
            "intervention": dict(run.intervention),
        },
        outputs=run.outputs,
        provenance={
            **dict(run.provenance),
            "causal_evidence": _intervention_causal_evidence(run),
        },
    )
    save_analysis_run(dataset, analysis_run)
    return run

def _intervention_causal_evidence(run: InterventionRunSpec) -> bool:
    """Conservatively interpret whether a saved record contains causal evidence."""
    readouts = dict(run.readouts)
    if str(readouts.get("status", "")) not in CAUSAL_RUN_STATUSES:
        return False
    if not (_claim_labels(readouts) & CAUSAL_CLAIM_LABELS):
        return False
    trials = readouts.get("trials", ())
    outcomes = readouts.get("outcomes") or readouts.get("outcome_results") or ()
    has_intervention_trial = any(
        isinstance(trial, Mapping)
        and str(trial.get("trial_kind")) in INTERVENTION_TRIAL_KINDS
        and str(trial.get("status", "ok")) in SUCCESSFUL_TRIAL_STATUSES
        for trial in trials
    )
    return has_intervention_trial and bool(outcomes)

def _claim_labels(readouts: Mapping[str, Any]) -> set[str]:
    claim = readouts.get("claim")
    labels: list[Any] = []
    if isinstance(claim, Mapping):
        raw = claim.get("claim_strength", claim.get("claim_strengths", ()))
        labels.extend(raw if isinstance(raw, (list, tuple, set, frozenset)) else [raw])
    raw_readout_labels = readouts.get("claim_strengths", ())
    labels.extend(
        raw_readout_labels
        if isinstance(raw_readout_labels, (list, tuple, set, frozenset))
        else [raw_readout_labels]
    )
    return {str(label) for label in labels if label}

def cohort_from_selection(
    dataset: TraceDataset,
    selection: SelectionState | Mapping[str, Any],
    *,
    label: str | None = None,
    cohort_id: str | None = None,
) -> CohortSpec:
    """Create a reusable episode cohort from a linked selection."""
    state = normalize_selection(selection)
    episodes = _matching_episode_records(dataset, state)
    traces = tuple(str(row["trace_id"]) for row in episodes if row.get("trace_id"))
    examples = _target_object_examples(dataset, state) or _representative_examples(episodes, state)
    example_rows = [row for rows in examples.values() for row in rows]
    example_ids = tuple(
        str(row.get("example_id") or f"{row.get('trace_id')}:{row.get('timestep')}")
        for row in example_rows
        if row.get("trace_id")
    )
    analysis_run = _first_scalar(state.axis_values.get("analysis_run"))
    resolved_id = cohort_id or _safe_id(label or f"selection_{state.selection_id}")
    definition = {
        "source": "selection",
        "filters": dict(state.axis_values),
        "analysis_run": analysis_run,
        "prediction_status": _first_scalar(state.axis_values.get("prediction_status")),
        "selected_axes": sorted(state.axis_values),
    }
    return CohortSpec(
        cohort_id=resolved_id,
        label=label or resolved_id.replace("_", " "),
        definition=definition,
        filters=dict(state.axis_values),
        members={"trace_id": traces, "example_id": example_ids},
        provenance={
            "source": "selection",
            "selection": state.to_dict(),
            "analysis_run": analysis_run,
            "member_count": len(traces),
            "example_count": len(example_ids),
        },
    )

def list_workspaces(dataset: TraceDataset) -> tuple[SavedWorkspace, ...]:
    """Return saved visual workspaces for this dataset."""
    root = _workbench_dir(dataset, "workspaces", create=False)
    if not root.exists():
        return ()
    return tuple(SavedWorkspace.from_dict(_read_json(path)) for path in sorted(root.glob("*.json")))

def save_workspace(dataset: TraceDataset, workspace: SavedWorkspace) -> SavedWorkspace:
    """Persist a saved workspace in the dataset-local workbench store."""
    path = (
        _workbench_dir(dataset, "workspaces", create=True)
        / f"{_safe_id(workspace.workspace_id)}.json"
    )
    _write_json(path, workspace.to_dict())
    return workspace

def resolve_workspace(dataset: TraceDataset, workspace_id: str) -> dict[str, Any]:
    """Load a saved workspace and re-resolve its linked selection."""
    workspace = next(
        (item for item in list_workspaces(dataset) if item.workspace_id == workspace_id),
        None,
    )
    if workspace is None:
        raise KeyError(f"Unknown workspace '{workspace_id}'")
    resolved = (
        resolve_selection(dataset, workspace.selection) if workspace.selection is not None else None
    )
    panel_types = {
        str(panel.get("panel_type")) for panel in workspace.panels if panel.get("panel_type")
    }
    registry = panel_registry()
    return {
        "workspace": workspace.to_dict(),
        "resolved_selection": resolved,
        "panel_registry": {
            key: entry.to_dict()
            for key, entry in registry.items()
            if not panel_types or key in panel_types
        },
        "cohorts": [
            cohort.to_dict()
            for cohort in list_cohorts(dataset)
            if cohort.cohort_id in workspace.cohorts
        ],
        "analysis_runs": [
            run.to_dict()
            for run in list_analysis_runs(dataset)
            if run.run_id in workspace.analysis_runs
        ],
    }

def lens_array_meta(dataset: TraceDataset, array_id: str) -> LensArraySpec:
    """Return metadata for a stable LensArray id."""
    for array in lens_array_catalog(dataset):
        if array.array_id == array_id:
            return array
    raise KeyError(f"Unknown LensArray '{array_id}'")

def slice_lens_array(
    dataset: TraceDataset,
    array_id: str,
    *,
    selection: Mapping[str, Any] | None = None,
    max_values: int = MAX_JSON_SLICE_VALUES,
) -> dict[str, Any]:
    """Return a bounded JSON preview for one LensArray selection.

    This is intentionally a plot-preview API, not a tensor transport protocol.
    Large arrays should later be served through Zarr/Arrow/media routes, while
    this endpoint gives panels enough shape, summary, and small slices to link
    selections without accidentally serializing a full trace.
    """
    spec = lens_array_meta(dataset, array_id)
    array = _load_lens_array(dataset, spec)
    normalized_selection = normalize_axis_values(selection or {})
    slices = _selection_to_slices(spec, array.shape, normalized_selection)
    value = np.asarray(array[slices])
    summary = _numeric_summary(value)
    response: dict[str, Any] = {
        "array": spec.to_dict(),
        "selection": normalized_selection,
        "resolved_slices": _slice_payload(slices),
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "summary": summary,
        "truncated": bool(value.size > max_values),
    }
    if value.size <= max_values:
        response["values"] = _jsonable_array(value)
    else:
        preview_slices = _preview_slices(value.shape, max_values=max_values)
        preview = np.asarray(value[preview_slices])
        response["preview_shape"] = [int(item) for item in preview.shape]
        response["preview"] = _jsonable_array(preview)
    return response

def compare_cohorts(
    dataset: TraceDataset,
    left: str | CohortSpec,
    right: str | CohortSpec,
) -> dict[str, Any]:
    """Compare two materialized cohorts as reusable analytical objects."""
    cohorts = {cohort.cohort_id: cohort for cohort in list_cohorts(dataset)}
    left_cohort = left if isinstance(left, CohortSpec) else cohorts[str(left)]
    right_cohort = right if isinstance(right, CohortSpec) else cohorts[str(right)]
    left_frame = _cohort_episode_frame(dataset, left_cohort)
    right_frame = _cohort_episode_frame(dataset, right_cohort)
    return {
        "left": left_cohort.to_dict(),
        "right": right_cohort.to_dict(),
        "members": {
            "left_trace_id": left_cohort.members.get("trace_id", ()),
            "right_trace_id": right_cohort.members.get("trace_id", ()),
        },
        "tables": {
            "outcome": _cohort_delta_table(left_frame, right_frame, "outcome"),
            "task": _cohort_delta_table(left_frame, right_frame, "task_id"),
            "object": _cohort_delta_table(left_frame, right_frame, "target_object"),
        },
        "summary": {
            "left_count": int(len(left_frame)),
            "right_count": int(len(right_frame)),
            "delta_count": int(len(left_frame) - len(right_frame)),
        },
    }

def projection_points(
    dataset: TraceDataset,
    *,
    selection: SelectionState | Mapping[str, Any] | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    """Return a deterministic PCA-like projection table for cohort discovery."""
    state = normalize_selection(selection or {"selection_id": "projection"})
    rows: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for array in _matching_lens_arrays(dataset, state):
        if array.kind != "tensor" or "unit" not in array.dims:
            continue
        loaded = np.asarray(_load_lens_array(dataset, array), dtype=np.float32)
        if loaded.size == 0:
            continue
        unit_axis = array.dims.index("unit")
        moved = np.moveaxis(loaded, unit_axis, -1)
        flat = moved.reshape(-1, moved.shape[-1])
        take = min(limit - len(vectors), flat.shape[0])
        if take <= 0:
            break
        for index in range(take):
            vectors.append(flat[index])
            trace_id = str(array.provenance.get("trace_id") or "")
            episode = dataset.bundle(trace_id).manifest if trace_id else None
            rows.append(
                {
                    "point_id": f"{array.array_id}:{index}",
                    "array_id": array.array_id,
                    "trace_id": trace_id,
                    "episode_id": episode.episode_id if episode else None,
                    "task_id": episode.task_id if episode else None,
                    "outcome": episode.outcome if episode else None,
                    "target_object": episode.metadata.get("target_object") if episode else None,
                }
            )
    if vectors:
        X = np.nan_to_num(np.vstack(vectors), nan=0.0)
        X = X - X.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        coords = X @ vt[:2].T if vt.shape[0] >= 2 else np.c_[X[:, 0], np.zeros(len(X))]
    else:
        coords = np.zeros((0, 2), dtype=np.float32)
    for row, coord in zip(rows, coords, strict=False):
        row["x"] = _jsonable_scalar(float(coord[0]))
        row["y"] = _jsonable_scalar(float(coord[1]))
    return {
        "selection": state.to_dict(),
        "method": "pca_preview",
        "points": rows,
        "columns": ["point_id", "x", "y", "trace_id", "task_id", "outcome", "target_object"],
        "semantic_role": "cohort_discovery_not_explanation",
    }

def graph_from_selection(
    dataset: TraceDataset,
    selection: SelectionState | Mapping[str, Any],
) -> dict[str, Any]:
    """Build a typed graph view from a resolved selection."""
    resolved = resolve_selection(dataset, selection)
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def add_node(kind: str, node_id: str, label: str | None = None) -> str:
        key = f"{kind}:{node_id}"
        nodes[key] = GraphNodeRef(kind=kind, id=node_id, label=label).to_dict()
        return key

    for episode in resolved["episodes"][:12]:
        ep_key = add_node("episode", str(episode.get("trace_id")), episode.get("task_id"))
        if episode.get("target_object"):
            object_key = add_node("object", str(episode["target_object"]))
            edges.append(_graph_edge(ep_key, object_key, "same_example"))
        if episode.get("outcome"):
            outcome_key = add_node("outcome", str(episode["outcome"]))
            edges.append(_graph_edge(ep_key, outcome_key, "same_example"))
    for site in resolved["model_sites"][:12]:
        site_key = add_node("site", str(site["site_id"]), site.get("module"))
        for unit in resolved["selection"].get("unit_refs", []):
            unit_key = add_node(str(unit.get("kind") or "unit"), str(unit.get("index")))
            edges.append(_graph_edge(site_key, unit_key, "activation_similarity"))
    return {
        "selection": resolved["selection"],
        "nodes": list(nodes.values()),
        "edges": edges,
        "edge_types": graph_edge_types(),
        "provenance": resolved["provenance"],
    }

def spatial_overlay_contracts(dataset: TraceDataset) -> list[dict[str, Any]]:
    """Return available spatial overlay contracts without generic importance labels."""
    arrays = lens_array_catalog(dataset)
    image_arrays = [array for array in arrays if array.kind == "image_sequence"]
    score_arrays = [
        array for array in arrays if "image_patch" in array.dims or "height" in array.dims
    ]
    contracts: list[dict[str, Any]] = []
    for score in overlay_score_types():
        if score.score_type in {"attention_weight", "activation_similarity"}:
            contracts.append(
                {
                    "score_type": score.score_type,
                    "label": score.label,
                    "causal": score.causal,
                    "image_arrays": [array.array_id for array in image_arrays[:20]],
                    "score_arrays": [array.array_id for array in score_arrays[:20]],
                    "emits": ["image_patch", "camera", "timestep", "layer", "token_kind"],
                }
            )
    return contracts
