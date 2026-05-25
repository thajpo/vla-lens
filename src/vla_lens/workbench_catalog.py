"""Catalog workbench primitives."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from vla_lens.traces import TraceBundle, TraceDataset
from vla_lens.workbench_schema import (
    CONTEXT_TABLE_IDS,
    TRACE_TABLE_PATHS,
    TRACE_TABLE_SPECS,
    AxisSpec,
    ImageFrameSpec,
    LensArraySpec,
    MediaSpec,
    ModelSiteSpec,
    OverlayScoreSpec,
    PanelRecipe,
    PanelRegistryEntry,
    StorageRef,
    TableSpec,
)
from vla_lens.workbench_tables import (
    query_table,
)
from vla_lens.workbench_utils import (
    _activation_axes,
    _array_episode_count,
    _array_names,
    _artifact_array_coords,
    _artifact_array_dims,
    _axis_names,
    _axis_names_for_array,
    _coords_for_array,
    _json_loads,
    _jsonable_scalar,
    _kind_for_episode_array,
    _label_columns,
    _optional_int,
    _optional_str,
    _parse_axes,
    _parse_shape,
    _unique_column,
)


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
    from vla_lens.workbench_validation import _storage_ref_from_row

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
        uri = _table_storage_uri(dataset, bundle_uri)
        try:
            summary = query_table(dataset, table=table_id, limit=0)
        except KeyError:
            if not _table_storage_exists(dataset, bundle_uri):
                continue
            summary = {"total": 0, "columns": []}
        if summary["total"] == 0 and not summary["columns"] and not _table_storage_exists(
            dataset, bundle_uri
        ):
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
    context_uri = _context_table_storage_uri(dataset)
    try:
        context_summary = query_table(dataset, table="context", limit=0)
    except KeyError:
        context_summary = {"total": 0, "columns": []}
    if context_summary["total"] or any(
        _table_storage_exists(dataset, TRACE_TABLE_PATHS[table_id])
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

def _table_storage_uri(dataset: TraceDataset, bundle_uri: str) -> str:
    for uri in _table_storage_uri_candidates(bundle_uri):
        if list(dataset.root.glob(uri)):
            return uri
    return f"**/*.vlatrace/{bundle_uri}"

def _context_table_storage_uri(dataset: TraceDataset) -> str:
    for uri in (
        "vla_lens/episodes/*/tables/*.parquet",
        "**/vla_lens/episodes/*/tables/*.parquet",
        "**/*.vlatrace/tables/*.parquet",
    ):
        if list(dataset.root.glob(uri)):
            return uri
    return "**/*.vlatrace/tables/*.parquet"

def _table_storage_exists(dataset: TraceDataset, bundle_uri: str) -> bool:
    return any(list(dataset.root.glob(uri)) for uri in _table_storage_uri_candidates(bundle_uri))

def _table_storage_uri_candidates(bundle_uri: str) -> tuple[str, ...]:
    return (
        f"vla_lens/episodes/*/{bundle_uri}",
        f"**/vla_lens/episodes/*/{bundle_uri}",
        f"**/*.vlatrace/{bundle_uri}",
    )

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

def _episode_lens_arrays(bundle: TraceBundle) -> list[LensArraySpec]:
    from vla_lens.workbench_validation import _storage_ref_from_row

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
    from vla_lens.workbench_validation import _storage_ref_from_row

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
