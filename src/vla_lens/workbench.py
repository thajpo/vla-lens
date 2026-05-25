"""Axis-native workbench primitives for linked VLA interpretability views.

The objects in this module are intentionally renderer-neutral.  They describe
addressable data, model sites, selections, cohorts, panels, and saved
workspaces.  A dashboard can render them with React, Plotly, ECharts, canvas,
or any other frontend without changing the trace/analysis contract.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd
import zarr

from vla_lens.traces import TraceBundle, TraceDataset
from vla_lens.validation import validate_trace_dataset

LensDataKind = Literal["tensor", "table", "image_sequence", "video", "artifact_array"]
MAX_JSON_SLICE_VALUES = 4096
AXIS_ALIASES: dict[str, str] = {
    "step": "timestep",
    "time": "timestep",
    "t": "timestep",
    "frame": "timestep",
    "frame_idx": "timestep",
    "call": "policy_call",
    "call_index": "policy_call",
    "model_call": "policy_call",
    "model_call_index": "policy_call",
    "patch": "image_patch",
    "feature": "unit",
    "channel": "unit",
    "neuron": "unit",
    "horizon": "action_horizon",
    "dim": "action_dim",
    "object_label": "object",
    "target_object": "object",
    "run": "analysis_run",
    "analysis_run_id": "analysis_run",
}


@dataclass(frozen=True, slots=True)
class AxisSpec:
    """Canonical semantic dimension used by selections and visual panels."""

    name: str
    kind: str
    label: str
    unit: str | None = None
    aliases: tuple[str, ...] = ()
    values: tuple[Any, ...] = ()
    alignments: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["aliases"] = list(self.aliases)
        payload["values"] = list(self.values)
        payload["alignments"] = list(self.alignments)
        return payload


@dataclass(frozen=True, slots=True)
class StorageRef:
    """Lazy reference to local or future remote data."""

    format: str
    uri: str
    relative_to: str = "dataset"
    chunks: tuple[int, ...] = ()
    compression: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["chunks"] = list(self.chunks)
        return payload


@dataclass(frozen=True, slots=True)
class LensArraySpec:
    """Typed reference to tensor/table/image/video data with axis metadata."""

    array_id: str
    kind: LensDataKind
    label: str
    storage: StorageRef
    dims: tuple[str, ...]
    shape: tuple[int, ...] = ()
    dtype: str | None = None
    coords: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    summary: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["storage"] = self.storage.to_dict()
        payload["dims"] = list(self.dims)
        payload["shape"] = list(self.shape)
        return payload


@dataclass(frozen=True, slots=True)
class TableSpec:
    """Typed reference to a Parquet-backed metadata/index table."""

    table_id: str
    label: str
    storage: StorageRef
    columns: tuple[str, ...]
    row_count: int
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["storage"] = self.storage.to_dict()
        payload["columns"] = list(self.columns)
        return payload


@dataclass(frozen=True, slots=True)
class ImageFrameSpec:
    """Typed reference to an encoded frame sequence for one camera stream."""

    frame_id: str
    trace_id: str
    episode_id: str
    camera: str
    storage: StorageRef
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str | None = None
    frame_count: int = 0
    uri_template: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["storage"] = self.storage.to_dict()
        payload["dims"] = list(self.dims)
        payload["shape"] = list(self.shape)
        return payload


@dataclass(frozen=True, slots=True)
class MediaSpec:
    """Typed reference to encoded media files or sequences."""

    media_id: str
    kind: str
    label: str
    storage: StorageRef
    dims: tuple[str, ...] = ()
    shape: tuple[int, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["storage"] = self.storage.to_dict()
        payload["dims"] = list(self.dims)
        payload["shape"] = list(self.shape)
        return payload


@dataclass(frozen=True, slots=True)
class ModelSiteSpec:
    """Stable handle for a model activation site."""

    site_id: str
    module: str
    site_type: str
    axes: tuple[str, ...]
    layer: int | None = None
    token_kind: str | None = None
    tensor_type: str | None = None
    family: str | None = None
    role: str | None = None
    segment: str | None = None
    materialization: str | None = None
    exactness: str | None = None
    token_space_id: str | None = None
    query_token_space_id: str | None = None
    key_token_space_id: str | None = None
    parent_site_id: str | None = None
    summary_type: str | None = None
    refs: Mapping[str, Any] = field(default_factory=dict)
    summary: Mapping[str, Any] = field(default_factory=dict)
    shape: tuple[int, ...] = ()
    source_trace_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["axes"] = list(self.axes)
        payload["shape"] = list(self.shape)
        return payload


TRACE_TABLE_SPECS: tuple[tuple[str, str, str, tuple[str, ...], bool], ...] = (
    ("timesteps", "Timesteps", TraceBundle.TIMESTEPS, ("timestep_index",), False),
    ("policy_calls", "Policy Calls", TraceBundle.POLICY_CALLS, ("policy_call_index",), False),
    ("generation_steps", "Generation Steps", TraceBundle.GENERATION_STEPS, (), False),
    ("streams", "Streams", TraceBundle.STREAMS, (), False),
    ("token_spaces", "Token Spaces", TraceBundle.TOKEN_SPACES, (), False),
    ("tokens", "Tokens", TraceBundle.TOKENS, (), False),
    ("array_index", "Array Index", TraceBundle.ARRAY_INDEX, ("arrays",), False),
    ("model_sites", "Model Sites", TraceBundle.MODEL_SITES, (), False),
    ("artifact_index", "Artifact Index", TraceBundle.ARTIFACT_INDEX, ("artifacts",), False),
    ("robot_state", "Robot State", TraceBundle.ROBOT_STATE, (), True),
    ("scene_state", "Scene State", TraceBundle.SCENE_STATE, (), True),
    ("camera_state", "Camera State", TraceBundle.CAMERA_STATE, (), True),
    ("evaluation", "Evaluation", TraceBundle.EVALUATION, (), True),
    ("image_preprocessing", "Image Preprocessing", TraceBundle.IMAGE_PREPROCESSING, (), True),
    ("prompt_metadata", "Prompt Metadata", TraceBundle.PROMPT_METADATA, (), True),
    ("action_normalization", "Action Normalization", TraceBundle.ACTION_NORMALIZATION, (), True),
)
TRACE_TABLE_PATHS: dict[str, str] = {
    table_id: path for table_id, _, path, _, _ in TRACE_TABLE_SPECS
}
TRACE_TABLE_ALIASES: dict[str, str] = {
    alias: table_id for table_id, _, _, aliases, _ in TRACE_TABLE_SPECS for alias in aliases
}
CONTEXT_TABLE_IDS: tuple[str, ...] = tuple(
    table_id for table_id, _, _, _, is_context in TRACE_TABLE_SPECS if is_context
)


@dataclass(frozen=True, slots=True)
class UnitRef:
    """Address of an interpretable unit or direction inside a model/site."""

    kind: str
    site_id: str | None = None
    index: int | None = None
    name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UnitRef":
        return cls(
            kind=str(payload.get("kind") or "unit"),
            site_id=_optional_str(payload.get("site_id")),
            index=_optional_int(payload.get("index")),
            name=_optional_str(payload.get("name")),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class SelectionState:
    """Typed linked selection emitted by panels and resolved by the backend."""

    selection_id: str
    axis_values: Mapping[str, Any] = field(default_factory=dict)
    unit_refs: tuple[UnitRef, ...] = ()
    cohort_refs: tuple[str, ...] = ()
    source_panel_id: str | None = None
    intent: str = "inspect"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SelectionState":
        return cls(
            selection_id=str(payload.get("selection_id") or "selection"),
            axis_values=normalize_axis_values(
                payload.get("axis_values") or payload.get("selection") or {}
            ),
            unit_refs=tuple(
                UnitRef.from_dict(item)
                for item in payload.get("unit_refs", ())
                if isinstance(item, Mapping)
            ),
            cohort_refs=tuple(str(item) for item in payload.get("cohort_refs", ())),
            source_panel_id=payload.get("source_panel_id"),
            intent=str(payload.get("intent") or "inspect"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_id": self.selection_id,
            "axis_values": normalize_axis_values(self.axis_values),
            "unit_refs": [unit.to_dict() for unit in self.unit_refs],
            "cohort_refs": list(self.cohort_refs),
            "source_panel_id": self.source_panel_id,
            "intent": self.intent,
        }


@dataclass(frozen=True, slots=True)
class CohortSpec:
    """Reusable subset of episodes/timesteps/units."""

    cohort_id: str
    label: str
    definition: Mapping[str, Any] = field(default_factory=dict)
    filters: Mapping[str, Any] = field(default_factory=dict)
    members: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["members"] = {key: list(values) for key, values in self.members.items()}
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CohortSpec":
        members = payload.get("members") or {}
        return cls(
            cohort_id=str(payload["cohort_id"]),
            label=str(payload.get("label") or payload["cohort_id"]),
            definition=dict(payload.get("definition") or {}),
            filters=dict(payload.get("filters") or {}),
            members={
                str(key): tuple(str(item) for item in values)
                for key, values in dict(members).items()
            },
            provenance=dict(payload.get("provenance") or {}),
        )


@dataclass(frozen=True, slots=True)
class PanelRecipe:
    """Renderer-neutral declaration for a linked visual panel."""

    panel_type: str
    label: str
    accepts: Mapping[str, Any]
    emits: tuple[str, ...]
    responds_to: tuple[str, ...] = ()
    preferred_axes: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["emits"] = list(self.emits)
        payload["responds_to"] = list(self.responds_to)
        return payload


@dataclass(frozen=True, slots=True)
class PanelRegistryEntry:
    """Workbench panel contract plus renderer-neutral behavior metadata."""

    recipe: PanelRecipe
    selection_axes: tuple[str, ...]
    renderer: str
    workflow_families: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_type": self.recipe.panel_type,
            "recipe": self.recipe.to_dict(),
            "selection_axes": list(self.selection_axes),
            "renderer": self.renderer,
            "workflow_families": list(self.workflow_families),
        }


@dataclass(frozen=True, slots=True)
class GraphNodeRef:
    """Typed node handle for graph/circuit views."""

    kind: str
    id: str
    label: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GraphEdgeSpec:
    """Typed edge whose semantics are explicit enough to avoid false equivalence."""

    source: GraphNodeRef
    target: GraphNodeRef
    edge_type: str
    score: float | None = None
    score_units: str | None = None
    analysis_run_id: str | None = None
    cohort_id: str | None = None
    examples: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source"] = self.source.to_dict()
        payload["target"] = self.target.to_dict()
        payload["examples"] = list(self.examples)
        return payload


@dataclass(frozen=True, slots=True)
class OverlayScoreSpec:
    """Semantic contract for spatial overlays."""

    score_type: str
    label: str
    causal: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnalysisRunSpec:
    """Computation provenance that produces LensArrays, tables, or effects."""

    run_id: str
    workflow: str
    inputs: Mapping[str, Any]
    outputs: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = list(self.outputs)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AnalysisRunSpec":
        return cls(
            run_id=str(payload["run_id"]),
            workflow=str(payload["workflow"]),
            inputs=dict(payload.get("inputs") or {}),
            outputs=tuple(str(item) for item in payload.get("outputs", ())),
            provenance=dict(payload.get("provenance") or {}),
        )


@dataclass(frozen=True, slots=True)
class InterventionRunSpec:
    """Saved intervention/ablation readout record, not a live execution request."""

    run_id: str
    intervention_type: str
    target: Mapping[str, Any]
    baseline: Mapping[str, Any] = field(default_factory=dict)
    intervention: Mapping[str, Any] = field(default_factory=dict)
    readouts: Mapping[str, Any] = field(default_factory=dict)
    outputs: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["outputs"] = list(self.outputs)
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InterventionRunSpec":
        return cls(
            run_id=str(payload["run_id"]),
            intervention_type=str(payload.get("intervention_type") or "intervention_delta"),
            target=dict(payload.get("target") or {}),
            baseline=dict(payload.get("baseline") or {}),
            intervention=dict(payload.get("intervention") or {}),
            readouts=dict(payload.get("readouts") or {}),
            outputs=tuple(str(item) for item in payload.get("outputs", ())),
            provenance=dict(payload.get("provenance") or {}),
        )


@dataclass(frozen=True, slots=True)
class SavedWorkspace:
    """Saved layout plus selections, cohorts, panels, and analysis runs."""

    workspace_id: str
    dataset_id: str
    panels: tuple[Mapping[str, Any], ...]
    selection: SelectionState | None = None
    cohorts: tuple[str, ...] = ()
    analysis_runs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "dataset_id": self.dataset_id,
            "panels": [dict(panel) for panel in self.panels],
            "selection": self.selection.to_dict() if self.selection else None,
            "cohorts": list(self.cohorts),
            "analysis_runs": list(self.analysis_runs),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SavedWorkspace":
        selection = payload.get("selection")
        workspace_selection = (
            SelectionState.from_dict(selection) if isinstance(selection, Mapping) else None
        )
        return cls(
            workspace_id=str(payload["workspace_id"]),
            dataset_id=str(payload.get("dataset_id") or ""),
            panels=tuple(dict(panel) for panel in payload.get("panels", ())),
            selection=workspace_selection,
            cohorts=tuple(str(item) for item in payload.get("cohorts", ())),
            analysis_runs=tuple(str(item) for item in payload.get("analysis_runs", ())),
        )


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
        "intervention_runs": [run.to_dict() for run in list_intervention_runs(dataset)],
        "saved_workspaces": [workspace.to_dict() for workspace in list_workspaces(dataset)],
        "contract_validation": validate_workbench_contracts(dataset),
    }


def normalize_axis_name(axis: str) -> str:
    """Return the canonical workbench axis name for a panel/backend alias."""
    text = str(axis)
    return AXIS_ALIASES.get(text, text)


def normalize_axis_values(axis_values: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a SelectionState axis map without changing value shapes."""
    normalized: dict[str, Any] = {}
    for raw_axis, value in dict(axis_values).items():
        axis = normalize_axis_name(str(raw_axis))
        if axis in normalized:
            normalized[axis] = _merge_axis_value(normalized[axis], value)
        else:
            normalized[axis] = value
    return normalized


def normalize_selection(selection: SelectionState | Mapping[str, Any]) -> SelectionState:
    """Normalize a SelectionState or dict to canonical axis names."""
    state = (
        selection if isinstance(selection, SelectionState) else SelectionState.from_dict(selection)
    )
    axes = normalize_axis_values(state.axis_values)
    if axes == state.axis_values:
        return state
    return SelectionState(
        selection_id=state.selection_id,
        axis_values=axes,
        unit_refs=state.unit_refs,
        cohort_refs=state.cohort_refs,
        source_panel_id=state.source_panel_id,
        intent=state.intent,
    )


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
            "causal_evidence": run.intervention_type
            in {"intervention_delta", "ablation_effect", "patch_ablation_delta"},
        },
    )
    save_analysis_run(dataset, analysis_run)
    return run


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


def query_table(
    dataset: TraceDataset,
    *,
    table: str,
    filters: Mapping[str, Any] | None = None,
    columns: Sequence[str] | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    """Query metadata/index tables through DuckDB.

    Physical bundle indexes are read from Parquet. Dataset-level episode rows
    are registered as a small in-memory DuckDB relation because their source of
    truth is the JSON manifest per trace.
    """
    filtered = _query_table_duckdb(
        dataset,
        table=table,
        filters=filters or {},
        columns=columns,
        limit=limit,
    )
    if filtered is None:
        raise KeyError(f"Unknown metadata table '{table}'")
    total = int(filtered.attrs.get("total", len(filtered)))
    limited = filtered
    if columns:
        requested = [column for column in columns if column in limited]
        limited = limited.loc[:, requested]
    return {
        "table": table,
        "total": total,
        "returned": int(len(limited)),
        "columns": [str(column) for column in limited.columns],
        "rows": [_jsonable_record(row) for row in limited.to_dict("records")],
    }


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


def validate_workbench_contracts(dataset: TraceDataset) -> dict[str, Any]:
    """Validate core workbench composition contracts."""
    axes = {axis.name for axis in axis_registry(dataset)}
    arrays = lens_array_catalog(dataset)
    panels = panel_registry()
    invalid_array_dims = [
        {"array_id": array.array_id, "dims": sorted(set(array.dims) - axes)}
        for array in arrays
        if set(array.dims) - axes
    ]
    invalid_panel_axes: list[dict[str, Any]] = []
    for panel_type, entry in panels.items():
        for field_name, values in [
            ("emits", entry.recipe.emits),
            ("responds_to", entry.recipe.responds_to),
            ("selection_axes", entry.selection_axes),
        ]:
            axes_in_field = {
                normalize_axis_name(str(value).removeprefix("selection.")) for value in values
            }
            missing = sorted(axes_in_field - axes)
            if missing:
                invalid_panel_axes.append(
                    {"panel_type": panel_type, "field": field_name, "axes": missing}
                )
    invalid_workflow_panels = [
        {"workflow_id": workflow["workflow_id"], "panels": missing}
        for workflow in workflow_presets(dataset)
        if (
            missing := sorted(set(str(panel) for panel in workflow.get("panels", ())) - set(panels))
        )
    ]
    invalid_storage = _invalid_storage_refs(dataset, arrays)
    invalid_tables = _invalid_table_refs(dataset)
    invalid_media = _invalid_media_refs(dataset)
    invalid_analysis_outputs = _invalid_analysis_run_outputs(dataset)
    trace_validation = validate_trace_dataset(dataset).to_dict()
    resolver_keys = {
        "examples",
        "lens_arrays",
        "suggested_panels",
        "provenance",
        "valid_references",
    }
    return {
        "valid": not (
            invalid_array_dims
            or invalid_panel_axes
            or invalid_workflow_panels
            or invalid_storage
            or invalid_tables
            or invalid_media
            or invalid_analysis_outputs
            or not trace_validation["valid"]
        ),
        "invalid_array_dims": invalid_array_dims,
        "invalid_panel_axes": invalid_panel_axes,
        "invalid_workflow_panels": invalid_workflow_panels,
        "invalid_storage": invalid_storage,
        "invalid_tables": invalid_tables,
        "invalid_media": invalid_media,
        "invalid_analysis_outputs": invalid_analysis_outputs,
        "trace_validation": trace_validation,
        "resolver_required_keys": sorted(resolver_keys),
    }


def _cohort_episode_frame(dataset: TraceDataset, cohort: CohortSpec) -> pd.DataFrame:
    frame = dataset.episode_index.copy()
    trace_ids = set(cohort.members.get("trace_id", ()))
    if trace_ids:
        return frame.loc[frame["trace_id"].astype(str).isin(trace_ids)].copy()
    filters = cohort.definition.get("filters") or cohort.filters
    return _filter_table(frame, filters if isinstance(filters, Mapping) else {})


def _cohort_delta_table(
    left: pd.DataFrame,
    right: pd.DataFrame,
    column: str,
) -> list[dict[str, Any]]:
    if column not in left and column not in right:
        return []
    left_counts = (
        left[column].astype(str).value_counts() if column in left else pd.Series(dtype=int)
    )
    right_counts = (
        right[column].astype(str).value_counts() if column in right else pd.Series(dtype=int)
    )
    keys = sorted(set(left_counts.index.astype(str)) | set(right_counts.index.astype(str)))
    rows: list[dict[str, Any]] = []
    for key in keys:
        left_count = int(left_counts.get(key, 0))
        right_count = int(right_counts.get(key, 0))
        rows.append(
            {
                column: key,
                "left_count": left_count,
                "right_count": right_count,
                "delta_count": left_count - right_count,
            }
        )
    return rows


def _graph_edge(source: str, target: str, edge_type: str) -> dict[str, Any]:
    causal = next(
        (item["causal"] for item in graph_edge_types() if item["edge_type"] == edge_type),
        False,
    )
    return {
        "source": source,
        "target": target,
        "edge_type": edge_type,
        "causal": bool(causal),
    }


def _invalid_storage_refs(
    dataset: TraceDataset,
    arrays: Sequence[LensArraySpec],
) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    for array in arrays:
        if array.kind in {"tensor", "artifact_array"}:
            if array.storage.format != "zarr":
                invalid.append(
                    {
                        "array_id": array.array_id,
                        "reason": "dense_array_not_zarr",
                        "format": array.storage.format,
                    }
                )
                continue
            path = _storage_path(dataset, array)
            if path is None or not path.exists():
                invalid.append(
                    {
                        "array_id": array.array_id,
                        "reason": "missing_storage",
                        "uri": array.storage.uri,
                    }
                )
                continue
            try:
                zarr.open_array(str(path), mode="r")
            except Exception as exc:  # pragma: no cover - defensive validation detail
                invalid.append(
                    {
                        "array_id": array.array_id,
                        "reason": "unreadable_zarr",
                        "error": repr(exc),
                    }
                )
        elif array.kind == "image_sequence":
            if array.storage.format != "jpeg":
                invalid.append(
                    {
                        "array_id": array.array_id,
                        "reason": "image_sequence_not_jpeg",
                        "format": array.storage.format,
                    }
                )
                continue
            path = _storage_path(dataset, array)
            if path is None or not path.exists():
                invalid.append(
                    {
                        "array_id": array.array_id,
                        "reason": "missing_storage",
                        "uri": array.storage.uri,
                    }
                )
                continue
            if not list(path.glob("*.jpg")):
                invalid.append(
                    {
                        "array_id": array.array_id,
                        "reason": "empty_jpeg_sequence",
                        "uri": array.storage.uri,
                    }
                )
    return invalid


def _invalid_analysis_run_outputs(dataset: TraceDataset) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    for run in list_analysis_runs(dataset):
        artifact_id = str(run.provenance.get("artifact_id") or run.run_id)
        try:
            artifact = dataset.load_artifact(artifact_id)
        except (FileNotFoundError, KeyError, ValueError):
            continue
        missing = sorted(set(run.outputs) - set(artifact.arrays))
        if missing:
            invalid.append({"run_id": run.run_id, "missing_outputs": missing})
    return invalid


def _invalid_table_refs(dataset: TraceDataset) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    for table in table_catalog(dataset):
        if table.storage.format != "parquet":
            invalid.append(
                {
                    "table_id": table.table_id,
                    "reason": "table_not_parquet",
                    "format": table.storage.format,
                }
            )
            continue
        if table.storage.relative_to == "dataset":
            paths = list(dataset.root.glob(table.storage.uri))
            if not paths:
                invalid.append(
                    {
                        "table_id": table.table_id,
                        "reason": "missing_table",
                        "uri": table.storage.uri,
                    }
                )
        elif table.storage.relative_to == "bundle":
            if not any((bundle.path / table.storage.uri).exists() for bundle in dataset.bundles):
                invalid.append(
                    {
                        "table_id": table.table_id,
                        "reason": "missing_table",
                        "uri": table.storage.uri,
                    }
                )
    return invalid


def _invalid_media_refs(dataset: TraceDataset) -> list[dict[str, Any]]:
    invalid: list[dict[str, Any]] = []
    for frame in image_frame_catalog(dataset):
        if frame.storage.format != "jpeg":
            invalid.append(
                {
                    "frame_id": frame.frame_id,
                    "reason": "image_frame_not_jpeg",
                    "format": frame.storage.format,
                }
            )
            continue
        path = _storage_path_for_ref(dataset, frame.storage, trace_id=frame.trace_id)
        if path is None or not path.exists():
            invalid.append(
                {
                    "frame_id": frame.frame_id,
                    "reason": "missing_frame_storage",
                    "uri": frame.storage.uri,
                }
            )
            continue
        if not list(path.glob("*.jpg")):
            invalid.append(
                {
                    "frame_id": frame.frame_id,
                    "reason": "empty_jpeg_sequence",
                    "uri": frame.storage.uri,
                }
            )
    for media in media_catalog(dataset):
        if media.storage.format not in {"jpeg", "mp4"}:
            invalid.append(
                {
                    "media_id": media.media_id,
                    "reason": "unsupported_media_format",
                    "format": media.storage.format,
                }
            )
            continue
        path = _storage_path_for_ref(
            dataset,
            media.storage,
            trace_id=_optional_str(media.provenance.get("trace_id")),
        )
        if path is None or not path.exists():
            invalid.append(
                {
                    "media_id": media.media_id,
                    "reason": "missing_media_storage",
                    "uri": media.storage.uri,
                }
            )
    return invalid


def _storage_path(dataset: TraceDataset, array: LensArraySpec) -> Path | None:
    return _storage_path_for_ref(
        dataset,
        array.storage,
        trace_id=_optional_str(array.provenance.get("trace_id")),
    )


def _storage_path_for_ref(
    dataset: TraceDataset,
    storage: StorageRef,
    *,
    trace_id: str | None = None,
) -> Path | None:
    if storage.relative_to == "bundle":
        if trace_id:
            return dataset.bundle(str(trace_id)).path / storage.uri
        return None
    return dataset.root / storage.uri


def _storage_ref_from_row(row: Mapping[str, Any]) -> StorageRef:
    return StorageRef(
        format=str(row.get("storage_format") or "zarr"),
        uri=str(row.get("relative_path")),
        relative_to="bundle",
        chunks=tuple(_parse_shape(row.get("chunks"))),
        compression=_optional_str(row.get("compression")) or "zstd",
    )


def _axis_index(value: Any, *, default: int) -> int:
    scalar = _first_scalar(value)
    if scalar is None:
        return default
    if isinstance(scalar, str) and not scalar.lstrip("-").isdigit():
        return default
    return max(0, int(scalar))


def _array_value(array: np.ndarray, *indexes: int) -> Any:
    if len(indexes) != array.ndim:
        return None
    bounded = tuple(max(0, min(array.shape[axis] - 1, index)) for axis, index in enumerate(indexes))
    return _jsonable_scalar(array[bounded])


def dataset_id(dataset: TraceDataset) -> str:
    if dataset.bundles:
        return str(dataset.root.name or dataset.bundles[0].manifest.trace_id)
    return str(dataset.root.name)


def _workbench_capabilities(dataset: TraceDataset) -> dict[str, dict[str, Any]]:
    """Capability records derived from workbench-native indexes and arrays."""
    episode_index = dataset.episode_index
    model_sites = dataset.model_site_index
    timestep_index = dataset.timestep_index
    array_names = _array_names(dataset)
    cameras = sorted({camera for bundle in dataset.bundles for camera in bundle.cameras()})
    labels = _label_columns(episode_index, timestep_index, array_names)
    return {
        "episodes": {
            "available": len(episode_index) > 0,
            "count": int(len(episode_index)),
            "detail": {},
        },
        "frames": {
            "available": bool(cameras),
            "count": len(cameras),
            "detail": {"cameras": cameras},
        },
        "actions": {
            "available": bool({"action", "executed_actions"} & array_names),
            "count": _array_episode_count(dataset, "action")
            or _array_episode_count(dataset, "executed_actions"),
            "detail": {},
        },
        "action_chunks": {
            "available": "action_chunks" in array_names,
            "count": _array_episode_count(dataset, "action_chunks"),
            "detail": {},
        },
        "generation_actions": {
            "available": "generation_actions" in array_names,
            "count": _array_episode_count(dataset, "generation_actions"),
            "detail": {},
        },
        "model_sites": {
            "available": not model_sites.empty,
            "count": int(len(model_sites)),
            "detail": {
                "modules": _unique_column(model_sites, "module"),
                "token_kinds": _unique_column(model_sites, "token_kind"),
                "axes": _activation_axes(model_sites),
            },
        },
        "tokens": {
            "available": any(not bundle.tokens.empty for bundle in dataset.bundles),
            "count": sum(int(len(bundle.tokens)) for bundle in dataset.bundles),
            "detail": {},
        },
        "episode_labels": {
            "available": bool(labels["episode"]),
            "count": len(labels["episode"]),
            "detail": {"columns": labels["episode"]},
        },
        "timestep_labels": {
            "available": bool(labels["timestep"]),
            "count": len(labels["timestep"]),
            "detail": {"columns": labels["timestep"]},
        },
        "capture_adapter": {"available": False, "count": 0, "detail": {}},
        "model_adapter": {"available": False, "count": 0, "detail": {}},
        "env_replay": {"available": False, "count": 0, "detail": {}},
    }


def _axis_value_catalog(dataset: TraceDataset) -> dict[str, tuple[Any, ...]]:
    episode_index = dataset.episode_index
    model_sites = dataset.model_site_index
    timestep_index = dataset.timestep_index
    capabilities = _workbench_capabilities(dataset)
    timesteps: tuple[Any, ...] = ()
    if not timestep_index.empty and "timestep" in timestep_index:
        max_timestep = int(pd.to_numeric(timestep_index["timestep"], errors="coerce").max())
        timesteps = (0, max_timestep) if max_timestep else (0,)
    layers = tuple(
        sorted(
            {
                int(value)
                for value in pd.to_numeric(
                    model_sites.get("layer", pd.Series(dtype=float)),
                    errors="coerce",
                )
                .dropna()
                .tolist()
            }
        )
    )
    return {
        "timestep": timesteps,
        "layer": layers,
        "module": tuple(capabilities["model_sites"]["detail"].get("modules", ())),
        "token_kind": tuple(capabilities["model_sites"]["detail"].get("token_kinds", ())),
        "camera": tuple(capabilities["frames"]["detail"].get("cameras", ())),
        "label": tuple(capabilities["episode_labels"]["detail"].get("columns", ())),
        "object": tuple(_unique_column(episode_index, "target_object")),
    }


def axis_registry(dataset: TraceDataset) -> tuple[AxisSpec, ...]:
    axes = _axis_value_catalog(dataset)
    return (
        AxisSpec("episode", "categorical", "Episode"),
        AxisSpec(
            "timestep",
            "ordered_index",
            "Timestep",
            unit="environment_step",
            aliases=("step",),
            values=tuple(axes.get("timestep", ())),
            alignments=("raw", "policy_call", "phase_normalized"),
        ),
        AxisSpec("policy_call", "ordered_index", "Policy Call", alignments=("call_index",)),
        AxisSpec(
            "camera",
            "categorical",
            "Camera",
            values=tuple(axes.get("camera", ())),
        ),
        AxisSpec("image_patch", "spatial_2d", "Image Patch", aliases=("patch",)),
        AxisSpec("height", "spatial_index", "Image Height"),
        AxisSpec("width", "spatial_index", "Image Width"),
        AxisSpec("rgb", "channel", "RGB Channel"),
        AxisSpec("xyz", "coordinate", "XYZ Coordinate"),
        AxisSpec("quat", "coordinate", "Quaternion Coordinate"),
        AxisSpec("pose_component", "coordinate", "Pose Component"),
        AxisSpec("matrix_row", "spatial_index", "Matrix Row"),
        AxisSpec("matrix_col", "spatial_index", "Matrix Column"),
        AxisSpec("joint", "robot_axis", "Robot Joint"),
        AxisSpec("gripper_joint", "robot_axis", "Gripper Joint"),
        AxisSpec("gripper_component", "robot_axis", "Gripper Component"),
        AxisSpec("state_component", "state_axis", "State Component"),
        AxisSpec("predicate", "categorical", "Predicate"),
        AxisSpec(
            "module",
            "categorical",
            "Module",
            values=tuple(axes.get("module", ())),
        ),
        AxisSpec(
            "layer",
            "ordered_index",
            "Layer",
            values=tuple(axes.get("layer", ())),
        ),
        AxisSpec(
            "token_kind",
            "categorical",
            "Token Kind",
            values=tuple(axes.get("token_kind", ())),
        ),
        AxisSpec("token", "ordered_index", "Token"),
        AxisSpec("unit", "ordered_index", "Unit / Channel"),
        AxisSpec("generation_step", "ordered_index", "Generation Step"),
        AxisSpec("action_horizon", "ordered_index", "Action Horizon"),
        AxisSpec("action_dim", "ordered_index", "Action Dimension"),
        AxisSpec(
            "object",
            "categorical",
            "Object",
            values=tuple(axes.get("object", ())),
        ),
        AxisSpec(
            "label",
            "categorical",
            "Label",
            values=tuple(axes.get("label", ())),
        ),
        AxisSpec("cohort", "categorical", "Cohort"),
        AxisSpec("metric", "categorical", "Metric"),
        AxisSpec("prediction_status", "categorical", "Prediction Status"),
        AxisSpec("example", "categorical", "Example"),
        AxisSpec("cell", "categorical", "Panel Cell"),
        AxisSpec("axis_range", "range", "Axis Range"),
        AxisSpec("image_xy", "spatial_2d", "Image XY"),
        AxisSpec("point", "categorical", "Projection Point"),
        AxisSpec("node", "categorical", "Graph Node"),
        AxisSpec("edge", "categorical", "Graph Edge"),
        AxisSpec("projection_x", "continuous", "Projection X"),
        AxisSpec("projection_y", "continuous", "Projection Y"),
        AxisSpec("analysis_run", "categorical", "Analysis Run"),
    )


def lens_array_catalog(dataset: TraceDataset) -> tuple[LensArraySpec, ...]:
    arrays: list[LensArraySpec] = []
    for bundle in dataset.bundles:
        arrays.extend(_episode_lens_arrays(bundle))
        arrays.extend(_activation_lens_arrays(bundle))
    arrays.extend(_artifact_lens_arrays(dataset))
    return tuple(arrays)


def image_frame_catalog(dataset: TraceDataset) -> tuple[ImageFrameSpec, ...]:
    """Return first-class encoded frame-stream specs."""
    frames: list[ImageFrameSpec] = []
    for bundle in dataset.bundles:
        table = bundle.array_index
        if table.empty:
            continue
        for row in table.to_dict("records"):
            name = str(row.get("name") or "")
            if not (name.startswith("frames.") or name.startswith("observation.images.")):
                continue
            dims = tuple(_axis_names_for_array(_parse_axes(row.get("axes"))))
            shape = tuple(_parse_shape(row.get("shape")))
            camera = (
                name.removeprefix("observation.images.")
                if name.startswith("observation.images.")
                else name.removeprefix("frames.")
            )
            storage = _storage_ref_from_row(row)
            frame_count = int(shape[0]) if shape else 0
            frames.append(
                ImageFrameSpec(
                    frame_id=f"trace.{bundle.manifest.trace_id}.observation.images.{camera}",
                    trace_id=bundle.manifest.trace_id,
                    episode_id=bundle.manifest.episode_id,
                    camera=camera,
                    storage=storage,
                    dims=dims,
                    shape=shape,
                    dtype=_optional_str(row.get("dtype")),
                    frame_count=frame_count,
                    uri_template=f"{storage.uri}/{{timestep:06d}}.jpg",
                    provenance={
                        "trace_id": bundle.manifest.trace_id,
                        "episode_id": bundle.manifest.episode_id,
                        "source": "trace_bundle",
                        "field": name,
                    },
                )
            )
    return tuple(frames)


def media_catalog(dataset: TraceDataset) -> tuple[MediaSpec, ...]:
    """Return encoded media refs used by frame/video panels."""
    media: list[MediaSpec] = []
    for frame in image_frame_catalog(dataset):
        media.append(
            MediaSpec(
                media_id=frame.frame_id,
                kind="jpeg_sequence",
                label=f"{frame.trace_id} {frame.camera}",
                storage=frame.storage,
                dims=frame.dims,
                shape=frame.shape,
                provenance=frame.provenance,
            )
        )
    for artifact in dataset.artifact_index.to_dict("records"):
        display = _json_loads(artifact.get("display"), default={})
        if not isinstance(display, Mapping):
            continue
        relative_path = _optional_str(display.get("relative_path"))
        if not relative_path or not relative_path.endswith(".mp4"):
            continue
        artifact_scope = str(artifact.get("artifact_scope") or "dataset")
        trace_id = _optional_str(artifact.get("trace_id"))
        media.append(
            MediaSpec(
                media_id=f"artifact.{artifact.get('artifact_id')}.video",
                kind="video",
                label=str(artifact.get("name") or artifact.get("artifact_id") or "video"),
                storage=StorageRef(
                    format="mp4",
                    uri=relative_path,
                    relative_to="bundle" if artifact_scope == "bundle" else "dataset",
                    compression="h264",
                ),
                provenance={
                    "artifact_id": str(artifact.get("artifact_id")),
                    "artifact_type": str(artifact.get("artifact_type")),
                    "artifact_scope": artifact_scope,
                    "trace_id": trace_id,
                },
            )
        )
    return tuple(media)


def table_catalog(dataset: TraceDataset) -> tuple[TableSpec, ...]:
    """Return first-class Parquet-backed table contracts for metadata queries."""
    specs: list[TableSpec] = []
    for table_id, label, bundle_uri, aliases, is_context in TRACE_TABLE_SPECS:
        uri = f"**/*.vlatrace/{bundle_uri}"
        try:
            summary = query_table(dataset, table=table_id, limit=0)
        except KeyError:
            if not list(dataset.root.glob(uri)):
                continue
            summary = {"total": 0, "columns": []}
        if summary["total"] == 0 and not summary["columns"] and not list(dataset.root.glob(uri)):
            continue
        specs.append(
            TableSpec(
                table_id=table_id,
                label=label,
                storage=StorageRef(format="parquet", uri=uri, relative_to="dataset"),
                columns=tuple(str(column) for column in summary["columns"]),
                row_count=int(summary["total"]),
                provenance={
                    "source": "trace_bundle_indexes",
                    "query_table": table_id,
                    "aliases": list(aliases),
                    "category": "context" if is_context else "core",
                },
            )
        )
    context_uri = "**/*.vlatrace/tables/*.parquet"
    try:
        context_summary = query_table(dataset, table="context", limit=0)
    except KeyError:
        context_summary = {"total": 0, "columns": []}
    if context_summary["total"] or any(
        list(dataset.root.glob(f"**/*.vlatrace/{TRACE_TABLE_PATHS[table_id]}"))
        for table_id in CONTEXT_TABLE_IDS
    ):
        specs.append(
            TableSpec(
                table_id="context",
                label="Context Tables",
                storage=StorageRef(format="parquet", uri=context_uri, relative_to="dataset"),
                columns=tuple(str(column) for column in context_summary["columns"]),
                row_count=int(context_summary["total"]),
                provenance={
                    "source": "trace_bundle_context_tables",
                    "query_table": "context",
                    "context_tables": list(CONTEXT_TABLE_IDS),
                    "category": "context",
                    "virtual_union": True,
                },
            )
        )
    return tuple(specs)


def model_site_catalog(dataset: TraceDataset) -> tuple[ModelSiteSpec, ...]:
    index = dataset.model_site_index
    if index.empty:
        return ()
    records: list[ModelSiteSpec] = []
    group_columns = [
        column
        for column in [
            "site_id",
            "module",
            "layer",
            "tensor_type",
            "token_kind",
            "family",
            "role",
            "segment",
            "materialization",
            "exactness",
            "token_space_id",
            "query_token_space_id",
            "key_token_space_id",
            "parent_site_id",
            "summary_type",
        ]
        if column in index
    ]
    for keys, group in index.groupby(group_columns, dropna=False, sort=True):
        values = keys if isinstance(keys, tuple) else (keys,)
        meta = dict(zip(group_columns, values, strict=False))
        module = str(meta.get("module") or "unknown")
        layer = _optional_int(meta.get("layer"))
        tensor_type = _optional_str(meta.get("tensor_type"))
        token_kind = _optional_str(meta.get("token_kind"))
        family = _optional_str(meta.get("family"))
        role = _optional_str(meta.get("role"))
        segment = _optional_str(meta.get("segment"))
        materialization = _optional_str(meta.get("materialization"))
        exactness = _optional_str(meta.get("exactness"))
        token_space_id = _optional_str(meta.get("token_space_id"))
        query_token_space_id = _optional_str(meta.get("query_token_space_id"))
        key_token_space_id = _optional_str(meta.get("key_token_space_id"))
        parent_site_id = _optional_str(meta.get("parent_site_id"))
        summary_type = _optional_str(meta.get("summary_type"))
        axes = _parse_axes(group.iloc[0].get("axes"))
        shape = _parse_shape(group.iloc[0].get("shape"))
        site_id = _optional_str(meta.get("site_id")) or ".".join(
            part
            for part in [
                module,
                f"layer{layer}" if layer is not None else None,
                tensor_type,
                token_kind,
                segment,
                token_space_id,
            ]
            if part and part != "nan"
        )
        refs = {
            key: value
            for key, value in {
                "token_space_id": token_space_id,
                "query_token_space_id": query_token_space_id,
                "key_token_space_id": key_token_space_id,
                "parent_site_id": parent_site_id,
            }.items()
            if value
        }
        summary = _model_site_summary(group)
        records.append(
            ModelSiteSpec(
                site_id=site_id,
                module=module,
                site_type=role or tensor_type or "activation",
                axes=tuple(_axis_names_for_array(axes)),
                layer=layer,
                token_kind=token_kind,
                tensor_type=tensor_type,
                family=family,
                role=role,
                segment=segment,
                materialization=materialization,
                exactness=exactness,
                token_space_id=token_space_id,
                query_token_space_id=query_token_space_id,
                key_token_space_id=key_token_space_id,
                parent_site_id=parent_site_id,
                summary_type=summary_type,
                refs=refs,
                summary=summary,
                shape=tuple(shape),
                source_trace_count=int(group["trace_id"].nunique())
                if "trace_id" in group
                else int(len(group)),
            )
        )
    return tuple(records)


def _model_site_summary(group: pd.DataFrame) -> dict[str, Any]:
    first = group.iloc[0]
    payload: dict[str, Any] = {
        "row_count": int(len(group)),
    }
    for column in ["dtype", "dtype_original", "dtype_saved", "storage_format", "compression"]:
        if column in group:
            value = _optional_str(first.get(column))
            if value:
                payload[column] = value
    for column in ["summary", "metadata"]:
        if column not in group:
            continue
        parsed = _json_loads(first.get(column), default=None)
        if isinstance(parsed, Mapping):
            if column == "summary":
                payload.update({str(key): _jsonable_scalar(value) for key, value in parsed.items()})
            elif "metadata" not in payload:
                payload["metadata"] = {
                    str(key): _jsonable_scalar(value)
                    for key, value in parsed.items()
                    if isinstance(value, (str, int, float, bool)) or value is None
                }
    return payload


def default_panel_recipes() -> tuple[PanelRecipe, ...]:
    return tuple(entry.recipe for entry in panel_registry().values())


def panel_registry() -> dict[str, PanelRegistryEntry]:
    """Typed registry for renderer-neutral workbench panels."""
    entries = [
        PanelRegistryEntry(
            recipe=PanelRecipe(
                panel_type="episode.viewer",
                label="Episode Viewer",
                accepts={"kinds": ["image_sequence", "tensor", "table"]},
                emits=("selection.episode", "selection.timestep", "selection.camera"),
                responds_to=("selection.episode", "selection.timestep", "selection.policy_call"),
                preferred_axes={"x": "timestep", "facet": "camera"},
            ),
            selection_axes=("episode", "timestep", "policy_call", "camera"),
            renderer="media",
            workflow_families=("target_object_encoding", "action_stabilization", "unit_explorer"),
        ),
        PanelRegistryEntry(
            recipe=PanelRecipe(
                panel_type="heatmap",
                label="Axis Heatmap",
                accepts={"kind": "tensor", "required_dims": ["x", "y"]},
                emits=("selection.cell", "selection.axis_range"),
                responds_to=("selection.cohort", "selection.metric", "selection.layer"),
                preferred_axes={"x": "timestep", "y": "layer", "color": "value"},
            ),
            selection_axes=("layer", "timestep", "token_kind", "metric"),
            renderer="heatmap",
            workflow_families=("target_object_encoding", "unit_explorer"),
        ),
        PanelRegistryEntry(
            recipe=PanelRecipe(
                panel_type="inspector",
                label="Selection Inspector",
                accepts={"kind": "resolved_selection"},
                emits=("selection.cohort",),
                responds_to=("selection.cell", "selection.unit", "selection.edge"),
            ),
            selection_axes=tuple(_axis_names()),
            renderer="inspector",
            workflow_families=("target_object_encoding", "action_stabilization", "unit_explorer"),
        ),
        PanelRegistryEntry(
            recipe=PanelRecipe(
                panel_type="confusion_matrix",
                label="Confusion Matrix",
                accepts={"kind": "table", "required_columns": ["actual", "predicted", "count"]},
                emits=("selection.label", "selection.prediction_status"),
                responds_to=("selection.analysis_run", "selection.cell"),
            ),
            selection_axes=("label", "prediction_status", "analysis_run"),
            renderer="table",
            workflow_families=("target_object_encoding",),
        ),
        PanelRegistryEntry(
            recipe=PanelRecipe(
                panel_type="examples.table",
                label="Linked Examples",
                accepts={"kind": "table"},
                emits=("selection.episode", "selection.timestep", "selection.example"),
                responds_to=("selection.cell", "selection.cohort", "selection.analysis_run"),
            ),
            selection_axes=("episode", "timestep", "example"),
            renderer="table",
            workflow_families=(
                "target_object_encoding",
                "action_stabilization",
                "unit_explorer",
                "representation_projection",
                "graph_explorer",
            ),
        ),
        PanelRegistryEntry(
            recipe=PanelRecipe(
                panel_type="unit.profile",
                label="Unit Profile",
                accepts={"kind": "unit_ref"},
                emits=("selection.unit", "selection.episode", "selection.timestep"),
                responds_to=("selection.unit", "selection.layer", "selection.module"),
            ),
            selection_axes=("unit", "layer", "module", "timestep"),
            renderer="unit_profile",
            workflow_families=("unit_explorer",),
        ),
        PanelRegistryEntry(
            recipe=PanelRecipe(
                panel_type="image.patch_overlay",
                label="Image Patch Overlay",
                accepts={"kinds": ["image_sequence", "tensor"], "dims": ["image_patch"]},
                emits=("selection.patch", "selection.image_xy"),
                responds_to=("selection.episode", "selection.timestep", "selection.layer"),
                preferred_axes={"x": "image_patch", "color": "score"},
            ),
            selection_axes=("image_patch", "camera", "timestep", "layer", "token_kind"),
            renderer="image_overlay",
            workflow_families=("spatial_correspondence", "unit_explorer"),
        ),
        PanelRegistryEntry(
            recipe=PanelRecipe(
                panel_type="action.horizon_heatmap",
                label="Action Horizon Heatmap",
                accepts={"kind": "tensor", "dims": ["action_horizon", "generation_step"]},
                emits=("selection.generation_step", "selection.action_horizon"),
                responds_to=("selection.episode", "selection.timestep", "selection.policy_call"),
                preferred_axes={"x": "action_horizon", "y": "generation_step", "color": "value"},
            ),
            selection_axes=(
                "episode",
                "policy_call",
                "generation_step",
                "action_horizon",
                "action_dim",
            ),
            renderer="heatmap",
            workflow_families=("action_stabilization",),
        ),
        PanelRegistryEntry(
            recipe=PanelRecipe(
                panel_type="projection.scatter",
                label="Projection Scatter",
                accepts={"kind": "table", "required_columns": ["x", "y"]},
                emits=("selection.point", "selection.cohort"),
                responds_to=("selection.label", "selection.cohort"),
                preferred_axes={"x": "projection_x", "y": "projection_y", "color": "label"},
            ),
            selection_axes=("episode", "timestep", "label", "cohort"),
            renderer="scatter",
            workflow_families=("representation_projection",),
        ),
        PanelRegistryEntry(
            recipe=PanelRecipe(
                panel_type="graph.explorer",
                label="Graph Explorer",
                accepts={"kind": "table", "edge_semantics": True},
                emits=("selection.node", "selection.edge", "selection.unit"),
                responds_to=("selection.unit", "selection.cohort", "selection.analysis_run"),
            ),
            selection_axes=("unit", "cohort", "analysis_run", "edge", "node"),
            renderer="graph",
            workflow_families=("graph_explorer", "unit_explorer"),
        ),
    ]
    return {entry.recipe.panel_type: entry for entry in entries}


def overlay_score_types() -> tuple[OverlayScoreSpec, ...]:
    return (
        OverlayScoreSpec(
            "attention_weight",
            "Attention Weight",
            causal=False,
            notes="Routing mass; useful for inspection but not causal by itself.",
        ),
        OverlayScoreSpec(
            "gradient_attribution",
            "Gradient Attribution",
            causal=False,
            notes="Local sensitivity around the selected example.",
        ),
        OverlayScoreSpec(
            "activation_similarity",
            "Activation Similarity",
            causal=False,
            notes="Similarity between internal states, patches, tokens, or examples.",
        ),
        OverlayScoreSpec(
            "probe_contribution",
            "Probe Contribution",
            causal=False,
            notes="Contribution under a trained diagnostic probe.",
        ),
        OverlayScoreSpec(
            "patch_ablation_delta",
            "Patch Ablation Delta",
            causal=True,
            notes="Output change after removing or replacing a spatial patch.",
        ),
        OverlayScoreSpec(
            "intervention_delta",
            "Intervention Delta",
            causal=True,
            notes="Output or behavior change after an explicit model intervention.",
        ),
        OverlayScoreSpec(
            "ablation_effect",
            "Ablation Effect",
            causal=True,
            notes="Output or behavior change after removing a typed model component.",
        ),
    )


def graph_edge_types() -> list[dict[str, Any]]:
    return [
        {"edge_type": "activation_similarity", "causal": False},
        {"edge_type": "correlation", "causal": False},
        {"edge_type": "linear_probe_weight", "causal": False},
        {"edge_type": "gradient_attribution", "causal": False},
        {"edge_type": "attention_weight", "causal": False},
        {"edge_type": "patch_ablation_delta", "causal": True},
        {"edge_type": "activation_patch_delta", "causal": True},
        {"edge_type": "intervention_delta", "causal": True},
        {"edge_type": "ablation_effect", "causal": True},
        {"edge_type": "temporal_precedes", "causal": False},
        {"edge_type": "same_example", "causal": False},
        {"edge_type": "same_cohort", "causal": False},
    ]


def workflow_presets(dataset: TraceDataset) -> list[dict[str, Any]]:
    capability = _workbench_capabilities(dataset)

    def available(key: str) -> bool:
        return bool(capability.get(key, {}).get("available"))

    return [
        {
            "workflow_id": "probe_suite",
            "label": "Probe Suites",
            "enabled": available("artifacts"),
            "panels": ["heatmap", "inspector", "confusion_matrix", "examples.table"],
            "primary_axes": ["layer", "policy_call", "metric", "analysis_run"],
            "outputs": ["weights", "bias", "normalizer_feature_mean", "normalizer_feature_scale"],
            "run_spec": {
                "label": {"level": "row", "source": "probe_artifact_target"},
                "split": {"unit": "episode", "kind": "artifact_defined"},
                "metrics": ["balanced_accuracy", "macro_f1", "delta_vs_metadata_baseline"],
            },
        },
        {
            "workflow_id": "target_object_encoding",
            "label": "Target Object Encoding",
            "enabled": available("model_sites") and available("episode_labels"),
            "panels": ["heatmap", "examples.table", "episode.viewer", "image.patch_overlay"],
            "primary_axes": ["layer", "timestep", "token_kind", "object"],
            "outputs": ["metric_cube", "confusion_matrix", "example_index"],
            "run_spec": {
                "label": {"name": "target_object", "level": "episode"},
                "split": {"unit": "episode", "kind": "random_episode"},
                "metrics": [
                    "balanced_accuracy",
                    "macro_f1",
                    "margin",
                    "per_class_accuracy",
                    "confusion_matrix",
                ],
            },
        },
        {
            "workflow_id": "action_stabilization",
            "label": "Action Stabilization",
            "enabled": available("action_chunks"),
            "panels": ["action.horizon_heatmap", "episode.viewer", "examples.table"],
            "primary_axes": ["timestep", "generation_step", "action_horizon", "action_dim"],
            "outputs": ["delta_to_final", "step_delta", "final_vs_executed"],
            "measures": {
                "delta_to_final": "||a[k,h,:] - a[K,h,:]||",
                "step_delta": "||a[k,h,:] - a[k-1,h,:]||",
                "final_vs_executed": "a[K,h,d] - executed[t+h,d]",
            },
        },
        {
            "workflow_id": "spatial_correspondence",
            "label": "Spatial Correspondence",
            "enabled": available("frames") and available("model_sites"),
            "panels": ["image.patch_overlay", "heatmap", "episode.viewer"],
            "primary_axes": ["camera", "image_patch", "layer", "token_kind", "timestep"],
            "outputs": ["patch_score_overlay", "linked_tokens"],
            "score_types": [score.score_type for score in overlay_score_types()],
        },
        {
            "workflow_id": "representation_projection",
            "label": "Representation Projection",
            "enabled": available("model_sites"),
            "panels": ["projection.scatter", "examples.table", "episode.viewer"],
            "primary_axes": ["episode", "timestep", "layer", "label"],
            "outputs": ["projection_points", "cohort_selection"],
        },
        {
            "workflow_id": "unit_explorer",
            "label": "Unit Explorer",
            "enabled": available("model_sites"),
            "panels": ["examples.table", "heatmap", "episode.viewer", "graph.explorer"],
            "primary_axes": ["module", "layer", "unit", "timestep", "object"],
            "outputs": ["top_examples", "unit_correlations", "probe_associations"],
            "unit_kinds": ["neuron", "sae_feature", "probe_direction", "attention_head"],
        },
    ]


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


def _episode_lens_arrays(bundle: TraceBundle) -> list[LensArraySpec]:
    arrays: list[LensArraySpec] = []
    table = bundle.array_index
    if table.empty:
        return arrays
    for row in table.to_dict("records"):
        name = str(row["name"])
        dims = tuple(_axis_names_for_array(_parse_axes(row.get("axes"))))
        shape = tuple(_parse_shape(row.get("shape")))
        arrays.append(
            LensArraySpec(
                array_id=f"trace.{bundle.manifest.trace_id}.episode.{name}",
                kind=_kind_for_episode_array(name),
                label=name,
                storage=_storage_ref_from_row(row),
                dims=dims,
                shape=shape,
                dtype=_optional_str(row.get("dtype")),
                coords=_coords_for_array(bundle, dims, shape),
                provenance={
                    "trace_id": bundle.manifest.trace_id,
                    "episode_id": bundle.manifest.episode_id,
                    "source": "trace_bundle",
                },
                summary={"array_type": "episode"},
            )
        )
    return arrays


def _activation_lens_arrays(bundle: TraceBundle) -> list[LensArraySpec]:
    arrays: list[LensArraySpec] = []
    table = bundle.model_sites
    if table.empty:
        return arrays
    for row in table.to_dict("records"):
        dims = tuple(_axis_names_for_array(_parse_axes(row.get("axes"))))
        shape = tuple(_parse_shape(row.get("shape")))
        arrays.append(
            LensArraySpec(
                array_id=f"trace.{bundle.manifest.trace_id}.model_site.{row['name']}",
                kind="tensor",
                label=str(row["name"]),
                storage=_storage_ref_from_row(row),
                dims=dims,
                shape=shape,
                dtype=_optional_str(row.get("dtype")),
                coords=_coords_for_array(bundle, dims, shape),
                provenance={
                    "trace_id": bundle.manifest.trace_id,
                    "module": _optional_str(row.get("module")),
                    "layer": _optional_int(row.get("layer")),
                    "tensor_type": _optional_str(row.get("tensor_type")),
                    "token_kind": _optional_str(row.get("token_kind")),
                    "source": "model_sites",
                },
                summary={"array_type": "model_site"},
            )
        )
    return arrays


def _artifact_lens_arrays(dataset: TraceDataset) -> list[LensArraySpec]:
    arrays: list[LensArraySpec] = []
    table = dataset.artifact_index
    if table.empty:
        return arrays
    for row in table.to_dict("records"):
        artifact_arrays = _json_loads(row.get("arrays"), default={})
        if not isinstance(artifact_arrays, Mapping):
            continue
        for name, path in artifact_arrays.items():
            shape: tuple[int, ...] = ()
            dtype: str | None = None
            coords: dict[str, Any] = {}
            chunks: tuple[int, ...] = ()
            try:
                artifact = dataset.load_artifact(str(row.get("artifact_id")))
                array = dataset.load_artifact_array(artifact, str(name), mmap=True)
                shape = tuple(int(item) for item in array.shape)
                dtype = str(array.dtype)
                coords = _artifact_array_coords(artifact, str(name), shape)
                chunks = tuple(int(item) for item in getattr(array, "chunks", ()) or ())
            except (FileNotFoundError, KeyError, ValueError, TypeError):
                pass
            arrays.append(
                LensArraySpec(
                    array_id=f"artifact.{row.get('artifact_id')}.{name}",
                    kind="artifact_array",
                    label=str(name),
                    storage=StorageRef(
                        format="zarr",
                        uri=str(path),
                        relative_to=str(row.get("artifact_scope") or "dataset"),
                        chunks=chunks,
                        compression="zstd",
                    ),
                    dims=tuple(_artifact_array_dims(str(name))),
                    shape=shape,
                    dtype=dtype,
                    coords=coords,
                    provenance={
                        "artifact_id": str(row.get("artifact_id")),
                        "artifact_type": str(row.get("artifact_type")),
                        "analysis_run_id": str(row.get("artifact_id")),
                        "artifact_scope": str(row.get("artifact_scope") or "dataset"),
                        "trace_id": _optional_str(row.get("trace_id")),
                    },
                    summary={"array_type": "artifact"},
                )
            )
    return arrays


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


def _table_frame(dataset: TraceDataset, table: str) -> pd.DataFrame:
    table_name = TRACE_TABLE_ALIASES.get(str(table), str(table))
    if table_name in {"episodes", "episode_index"}:
        return dataset.episode_index.copy()
    if table_name in {"timesteps", "timestep_index"}:
        return dataset.timestep_index.copy()
    if table_name == "artifact_index":
        return dataset.artifact_index.copy()
    if table_name == "context":
        frames: list[pd.DataFrame] = []
        for context_table in CONTEXT_TABLE_IDS:
            frame = _table_frame(dataset, context_table)
            if not frame.empty:
                frames.append(frame)
        return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if table_name in TRACE_TABLE_PATHS and table_name != "timesteps":
        frames = []
        for bundle in dataset.bundles:
            frame = _bundle_trace_table(bundle, table_name).copy()
            if frame.empty:
                continue
            frame["trace_id"] = bundle.manifest.trace_id
            frame["episode_id"] = bundle.manifest.episode_id
            frame["bundle_path"] = str(bundle.path)
            if table_name in CONTEXT_TABLE_IDS:
                frame["context_table"] = table_name
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    raise KeyError(f"Unknown metadata table '{table}'")


def _query_table_duckdb(
    dataset: TraceDataset,
    *,
    table: str,
    filters: Mapping[str, Any],
    columns: Sequence[str] | None,
    limit: int,
) -> pd.DataFrame | None:
    con = duckdb.connect(database=":memory:")
    try:
        source = _duckdb_source_sql(dataset, table, con)
        if source is None:
            return None
        available_columns = set(con.execute(f"SELECT * FROM ({source}) AS source LIMIT 0").df())
        if any(column not in available_columns for column in filters):
            empty_columns = list(columns) if columns else sorted(available_columns)
            frame = pd.DataFrame(columns=empty_columns)
            frame.attrs["total"] = 0
            return frame
        where_sql, params = _duckdb_where(filters)
        total = con.execute(
            f"SELECT count(*) AS total FROM ({source}) AS source {where_sql}",
            params,
        ).fetchone()[0]
        selected_columns = "*"
        if columns:
            selected = [column for column in columns if column in available_columns]
            selected_columns = ", ".join(_quote_identifier(column) for column in selected) or "*"
        frame = con.execute(
            f"""
            SELECT {selected_columns}
            FROM ({source}) AS source
            {where_sql}
            LIMIT ?
            """,
            [*params, max(0, int(limit))],
        ).df()
        frame.attrs["total"] = int(total)
        return frame
    finally:
        con.close()


def _duckdb_source_sql(
    dataset: TraceDataset,
    table: str,
    con: duckdb.DuckDBPyConnection,
) -> str | None:
    selects: list[str] = []
    table_name = TRACE_TABLE_ALIASES.get(str(table), str(table))
    if table_name in {"episodes", "episode_index"}:
        con.register("episode_index_source", dataset.episode_index.copy())
        return "SELECT * FROM episode_index_source"
    if table_name == "context":
        for context_table in CONTEXT_TABLE_IDS:
            bundle_path = TRACE_TABLE_PATHS[context_table]
            for bundle in dataset.bundles:
                selects.append(
                    _parquet_select(
                        bundle.path / bundle_path,
                        {
                            "trace_id": bundle.manifest.trace_id,
                            "episode_id": bundle.manifest.episode_id,
                            "bundle_path": str(bundle.path),
                            "context_table": context_table,
                        },
                    )
                )
    elif table_name in {"artifacts", "artifact_index"}:
        dataset_index = dataset.root / TraceBundle.ARTIFACT_INDEX
        if dataset_index.exists() and not (dataset.root / TraceBundle.MANIFEST).exists():
            selects.append(
                _parquet_select(
                    dataset_index,
                    {
                        "trace_id": None,
                        "episode_id": None,
                        "bundle_path": None,
                        "dataset_path": str(dataset.root),
                        "artifact_scope": "dataset",
                    },
                )
            )
        for bundle in dataset.bundles:
            selects.append(
                _parquet_select(
                    bundle.path / TraceBundle.ARTIFACT_INDEX,
                    {
                        "trace_id": bundle.manifest.trace_id,
                        "episode_id": bundle.manifest.episode_id,
                        "bundle_path": str(bundle.path),
                        "dataset_path": str(dataset.root),
                        "artifact_scope": "bundle",
                    },
                )
            )
    elif table_name in TRACE_TABLE_PATHS:
        bundle_path = TRACE_TABLE_PATHS[table_name]
        for bundle in dataset.bundles:
            selects.append(
                _parquet_select(
                    bundle.path / bundle_path,
                    {
                        "trace_id": bundle.manifest.trace_id,
                        "episode_id": bundle.manifest.episode_id,
                        "bundle_path": str(bundle.path),
                    },
                )
            )
    else:
        return None
    selects = [select for select in selects if select]
    if not selects:
        return None
    return "\nUNION ALL BY NAME\n".join(selects)


def _bundle_trace_table(bundle: TraceBundle, table_name: str) -> pd.DataFrame:
    readers = {
        "policy_calls": bundle.policy_calls,
        "generation_steps": bundle.generation_steps,
        "streams": bundle.streams,
        "token_spaces": bundle.token_spaces,
        "tokens": bundle.tokens,
        "array_index": bundle.array_index,
        "model_sites": bundle.model_sites,
        "artifact_index": bundle.artifact_index,
        "robot_state": bundle.robot_state,
        "scene_state": bundle.scene_state,
        "camera_state": bundle.camera_state,
        "evaluation": bundle.evaluation,
        "image_preprocessing": bundle.image_preprocessing,
        "prompt_metadata": bundle.prompt_metadata,
        "action_normalization": bundle.action_normalization,
    }
    if table_name not in readers:
        raise KeyError(f"Unknown metadata table '{table_name}'")
    return readers[table_name]


def _parquet_select(path: Path, constants: Mapping[str, Any]) -> str:
    if not path.exists():
        return ""
    try:
        columns = pd.read_parquet(path).columns
        if columns.empty:
            return ""
    except Exception:
        return ""
    extra = []
    existing_columns = {str(column) for column in columns}
    for key, value in constants.items():
        if key in existing_columns:
            continue
        if value is None:
            extra.append(f"NULL AS {_quote_identifier(key)}")
        else:
            extra.append(f"{_quote_literal(str(value))} AS {_quote_identifier(key)}")
    suffix = ", " + ", ".join(extra) if extra else ""
    return f"SELECT *{suffix} FROM read_parquet({_quote_literal(str(path))})"


def _duckdb_where(filters: Mapping[str, Any]) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for column, expected in filters.items():
        ident = _quote_identifier(str(column))
        if isinstance(expected, Mapping):
            if "start" in expected:
                clauses.append(f"try_cast({ident} AS DOUBLE) >= ?")
                params.append(float(expected["start"]))
            if "end" in expected:
                clauses.append(f"try_cast({ident} AS DOUBLE) <= ?")
                params.append(float(expected["end"]))
        elif isinstance(expected, (list, tuple, set, frozenset)):
            values = [str(item) for item in expected]
            if values:
                clauses.append(f"cast({ident} AS VARCHAR) IN ({', '.join('?' for _ in values)})")
                params.extend(values)
            else:
                clauses.append("FALSE")
        else:
            clauses.append(f"cast({ident} AS VARCHAR) = ?")
            params.append(str(expected))
    if not clauses:
        return "", []
    return "WHERE " + " AND ".join(clauses), params


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _filter_table(frame: pd.DataFrame, filters: Mapping[str, Any]) -> pd.DataFrame:
    out = frame
    for column, expected in filters.items():
        if column not in out:
            return out.iloc[0:0].copy()
        if isinstance(expected, Mapping):
            out = _filter_table_range(out, column, expected)
        elif isinstance(expected, (list, tuple, set, frozenset)):
            allowed = {str(item) for item in expected}
            out = out.loc[out[column].astype(str).isin(allowed)]
        else:
            out = out.loc[out[column].astype(str) == str(expected)]
    return out


def _filter_table_range(
    frame: pd.DataFrame,
    column: str,
    expected: Mapping[str, Any],
) -> pd.DataFrame:
    values = pd.to_numeric(frame[column], errors="coerce")
    mask = pd.Series(True, index=frame.index)
    if "start" in expected:
        mask = mask & (values >= float(expected["start"]))
    if "end" in expected:
        mask = mask & (values <= float(expected["end"]))
    return frame.loc[mask]


def _array_names(dataset: TraceDataset) -> set[str]:
    names: set[str] = set()
    for bundle in dataset.bundles:
        table = bundle.array_index
        if not table.empty and "name" in table:
            names.update(str(value) for value in table["name"].dropna())
    return names


def _array_episode_count(dataset: TraceDataset, name: str) -> int:
    count = 0
    for bundle in dataset.bundles:
        table = bundle.array_index
        if not table.empty and "name" in table:
            count += int((table["name"].astype(str) == name).any())
    return count


def _activation_axes(index: pd.DataFrame) -> list[str]:
    axes: set[str] = set()
    if index.empty or "axes" not in index:
        return []
    for value in index["axes"].dropna():
        axes.update(str(item) for item in _parse_axes(value))
    return sorted(axes)


def _label_columns(
    episode_index: pd.DataFrame,
    timestep_index: pd.DataFrame,
    array_names: set[str],
) -> dict[str, list[str]]:
    episode_candidates = [
        "task_id",
        "prompt",
        "outcome",
        "success",
        "target_object",
        "object_label",
        "benchmark",
        "env_id",
        "robot_id",
        "scene_id",
        "layout_id",
    ]
    timestep_candidates = [
        "phase",
        "contact",
        "grasp",
        "lift",
        "reward",
        "done",
        "policy_call_index",
    ]
    array_label_names = sorted(
        name
        for name in array_names
        if any(part in name for part in ["object", "contact", "phase", "reward", "done"])
    )
    return {
        "episode": [column for column in episode_candidates if column in episode_index],
        "timestep": [column for column in timestep_candidates if column in timestep_index]
        + array_label_names,
    }


def _unique_column(frame: pd.DataFrame, column: str) -> list[Any]:
    if frame.empty or column not in frame:
        return []
    return sorted(str(value) for value in frame[column].dropna().unique())


def _selection_to_slices(
    spec: LensArraySpec,
    shape: Sequence[int],
    selection: Mapping[str, Any],
) -> tuple[Any, ...]:
    slices: list[Any] = []
    for dim, size in zip(spec.dims, shape, strict=False):
        if dim not in selection:
            slices.append(slice(None))
            continue
        slices.append(_axis_selector(selection[dim], int(size), coords=spec.coords.get(dim)))
    return tuple(slices)


def _axis_selector(value: Any, size: int, *, coords: Any = None) -> Any:
    if size <= 0:
        return slice(0, 0)
    if isinstance(value, Mapping):
        start = _coord_index(value.get("start", 0), size, coords, default=0)
        end = _coord_index(value.get("end", value.get("start", start)), size, coords, default=start)
        step = int(value.get("step", 1))
        return slice(max(0, start), min(size, end + 1), max(1, step))
    if isinstance(value, (list, tuple)):
        indexes = [_coord_index(item, size, coords, default=0) for item in value]
        return [max(0, min(size - 1, item)) for item in indexes]
    index = _coord_index(value, size, coords, default=0)
    return max(0, min(size - 1, index))


def _coord_index(value: Any, size: int, coords: Any, *, default: int) -> int:
    """Map semantic coordinate values to positional indexes when coords exist."""
    if value is None:
        return default
    if isinstance(coords, Sequence) and not isinstance(coords, (str, bytes, bytearray)):
        for index, coord in enumerate(coords):
            if str(coord) == str(value):
                return index
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _slice_payload(slices: Sequence[Any]) -> list[Any]:
    payload: list[Any] = []
    for item in slices:
        if isinstance(item, slice):
            payload.append({"start": item.start, "stop": item.stop, "step": item.step})
        elif isinstance(item, list):
            payload.append(item)
        else:
            payload.append(int(item))
    return payload


def _preview_slices(shape: Sequence[int], *, max_values: int) -> tuple[slice, ...]:
    if not shape:
        return ()
    remaining = max(1, int(max_values))
    slices: list[slice] = []
    for size in shape:
        width = min(int(size), max(1, remaining))
        slices.append(slice(0, width))
        remaining = max(1, remaining // max(1, width))
    return tuple(slices)


def _numeric_summary(value: np.ndarray) -> dict[str, Any]:
    if value.size == 0 or not np.issubdtype(value.dtype, np.number):
        return {}
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        return {}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
    }


def _jsonable_array(value: np.ndarray) -> Any:
    return np.asarray(value).tolist()


def _workbench_dir(dataset: TraceDataset, name: str, *, create: bool) -> Path:
    root = dataset.root / "workbench" / name
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    return safe.strip("._") or "workbench_item"


def _merge_axis_value(existing: Any, incoming: Any) -> Any:
    if existing == incoming:
        return existing
    if isinstance(existing, Mapping) or isinstance(incoming, Mapping):
        return incoming
    values: list[Any] = []
    for item in (existing, incoming):
        if isinstance(item, (list, tuple, set, frozenset)):
            values.extend(item)
        else:
            values.append(item)
    deduped: list[Any] = []
    seen: set[str] = set()
    for value in values:
        key = str(value)
        if key not in seen:
            seen.add(key)
            deduped.append(value)
    return deduped


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Workbench record must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _kind_for_episode_array(name: str) -> LensDataKind:
    if name.startswith("frames.") or name.startswith("observation.images."):
        return "image_sequence"
    return "tensor"


def _axis_names_for_array(axes: Sequence[str]) -> list[str]:
    mapping = {
        "time": "timestep",
        "timestep": "timestep",
        "call": "policy_call",
        "call_index": "policy_call",
        "camera": "camera",
        "patch": "image_patch",
        "token": "token",
        "layer": "layer",
        "generation_step": "generation_step",
        "horizon": "action_horizon",
        "action_dim": "action_dim",
        "dim": "action_dim",
        "feature": "unit",
        "hidden": "unit",
        "channel": "unit",
        "object": "object",
    }
    return [mapping.get(str(axis), str(axis)) for axis in axes]


def _axis_names() -> set[str]:
    return {
        "episode",
        "timestep",
        "policy_call",
        "camera",
        "image_patch",
        "height",
        "width",
        "rgb",
        "xyz",
        "quat",
        "joint",
        "gripper_joint",
        "module",
        "layer",
        "token_kind",
        "token",
        "unit",
        "generation_step",
        "action_horizon",
        "action_dim",
        "object",
        "label",
        "cohort",
        "metric",
        "prediction_status",
        "example",
        "cell",
        "axis_range",
        "image_xy",
        "point",
        "node",
        "edge",
        "projection_x",
        "projection_y",
        "analysis_run",
    }


def _coords_for_array(
    bundle: TraceBundle,
    dims: Sequence[str],
    shape: Sequence[int],
) -> dict[str, Any]:
    coords: dict[str, Any] = {}
    for dim, size in zip(dims, shape, strict=False):
        if dim == "episode":
            coords[dim] = [bundle.manifest.trace_id]
        elif size <= 256:
            coords[dim] = list(range(int(size)))
        else:
            coords[dim] = {"start": 0, "stop": int(size), "step": 1}
    return coords


def _artifact_array_dims(name: str) -> list[str]:
    if name in {"metric_cube", "baseline_cube", "delta_cube"}:
        return ["layer", "timestep", "token_kind"]
    if name in {"delta_to_final", "step_delta"}:
        return ["episode", "policy_call", "generation_step", "action_horizon"]
    if name == "final_vs_executed":
        return ["episode", "policy_call", "action_horizon", "action_dim"]
    if "commitment" in name:
        return ["episode", "policy_call", "generation_step"]
    if "executed" in name or "predicted" in name:
        return ["episode", "policy_call"]
    if "margin" in name or "score" in name:
        return ["layer", "timestep"]
    return []


def _artifact_array_coords(artifact: Any, name: str, shape: Sequence[int]) -> dict[str, Any]:
    dims = _artifact_array_dims(name)
    coords: dict[str, Any] = {}
    display = getattr(artifact, "display", {}) or {}
    axes = display.get("axes") if isinstance(display, Mapping) else None
    if isinstance(axes, Mapping):
        for dim, size in zip(dims, shape, strict=False):
            values = axes.get(dim)
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
                coords[dim] = list(values)[: int(size)]
    for dim, size in zip(dims, shape, strict=False):
        coords.setdefault(
            dim,
            list(range(int(size)))
            if int(size) <= 256
            else {
                "start": 0,
                "stop": int(size),
                "step": 1,
            },
        )
    return coords


def _parse_axes(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return []


def _parse_shape(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [int(item) for item in parsed]
    return []


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value)
    return None if text == "nan" else text


def _json_loads(value: Any, *, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _jsonable_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable_scalar(value) for key, value in record.items()}


def _jsonable_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        return _jsonable_scalar(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _as_set(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        if "start" in value and "end" in value:
            return {str(value["start"])}
        return {str(item) for item in value.values()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return {str(item) for item in value}
    return {str(value)}


def _first_scalar(value: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get("start") if "start" in value else next(iter(value.values()), None)
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value
