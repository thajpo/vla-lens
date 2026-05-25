"""Schema workbench primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

from vla_lens.traces import TraceBundle

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


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text == "nan" else text


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
