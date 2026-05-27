# ruff: noqa: F401
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import vla_lens.workbench as workbench_module
from vla_lens import (
    FULL_REQUIRED_MODEL_SITE_ROLES,
    ActivationQuery,
    ActivationSpec,
    AnalysisRunSpec,
    ArraySpec,
    InterventionRunSpec,
    SavedWorkspace,
    SelectionState,
    TraceBundle,
    TraceDataset,
    TraceManifest,
    UnitRef,
    cohort_from_selection,
    compare_cohorts,
    create_synthetic_trace_dataset,
    graph_from_selection,
    list_analysis_runs,
    list_cohorts,
    list_workspaces,
    normalize_axis_values,
    projection_points,
    query_table,
    resolve_selection,
    resolve_workspace,
    save_analysis_run,
    save_cohort,
    save_intervention_run,
    save_pi05_interaction_metrics_artifact,
    save_workspace,
    spatial_overlay_contracts,
    table_catalog,
    unit_profile,
    validate_trace_bundle,
    validate_workbench_contracts,
    workbench_manifest,
)
from vla_lens.action_generation import save_action_generation_artifact
from vla_lens.analyzer import diagnostics_status, run_dataset_diagnostics
from vla_lens.pi05.capture import AUDIT_WINDOWED_LAYERS
from vla_lens.probes import dump_probe_spec, train_probe_artifact_from_spec
from vla_lens.server import (
    _activation_sites_payload,
    _artifact_detail_payload,
    _artifacts_payload,
    _attention_map_payload,
    _create_action_generation_payload,
    _create_outcome_probe_payload,
    _create_target_object_probe_payload,
    _dataset_diagnostics_payload,
    _dataset_payload,
    _episode_interactions_payload,
    _episode_metrics_payload,
    _episode_probes_payload,
    _episode_video_path,
    _expert_token_details_payload,
    _lens_array_meta_payload,
    _lens_array_slice_payload,
    _lens_arrays_payload,
    _object_camera_overlay_payload,
    _observational_comparisons_payload,
    _probe_index_payload,
    _prompt_attention_payload,
    _resolve_selection_payload,
    _run_dataset_diagnostics_payload,
    _save_analysis_run_payload,
    _save_cohort_from_selection_payload,
    _save_intervention_run_payload,
    _save_workspace_payload,
    _table_query_payload,
    _unit_profile_payload,
    _workbench_payload,
)
from vla_lens.target_object import save_target_object_encoding_artifact
from vla_lens.traces import ModelSiteSpec as TraceModelSiteSpec
from vla_lens.workbench import ImageFrameSpec, LensArraySpec, StorageRef, TableSpec


def _make_minimal_trace(
    path,
    *,
    profile: str = "rollout",
    model_sites: list[TraceModelSiteSpec] | None = None,
    streams: dict[str, list[object]] | None = None,
    token_spaces: dict[str, list[object]] | None = None,
    tokens: dict[str, list[object]] | None = None,
    include_frames: bool = False,
    action_normalization: dict[str, list[object]] | None = None,
    extra_episode_arrays: dict[str, ArraySpec] | None = None,
    scene_state: pd.DataFrame | None = None,
    camera_state: pd.DataFrame | None = None,
    metadata: dict[str, object] | None = None,
) -> TraceBundle:
    length = 2
    manifest = TraceManifest(
        trace_id=path.stem,
        episode_id=path.stem,
        task_id="minimal",
        prompt="minimal",
        model_id="minimal-model",
        env_id="minimal-env",
        robot_id="minimal-robot",
        outcome="unknown",
        length=length,
        metadata={"capture_profile": profile, **(metadata or {})},
    )
    timesteps = {
        "timestep": [0, 1],
        "reward": [0.0, 0.0],
        "policy_call_index": [0, 0],
        "horizon_index": [0, 1],
    }
    policy_calls = {
        "policy_call_index": [0],
        "episode_id": [path.stem],
        "observation_timestep": [0],
        "env_timestep_start": [0],
        "env_timestep_end": [1],
    }
    episode_arrays = {
        "executed_actions": ArraySpec(
            np.zeros((length, 1), dtype=np.float32),
            ["timestep", "action_dim"],
        ),
        "action_chunks": ArraySpec(
            np.zeros((1, length, 1), dtype=np.float32),
            ["policy_call", "horizon", "action_dim"],
        ),
        "generation_actions": ArraySpec(
            np.zeros((1, 1, length, 1), dtype=np.float32),
            ["policy_call", "generation_step", "horizon", "action_dim"],
        ),
    }
    if include_frames:
        episode_arrays["frames.main"] = ArraySpec(
            np.zeros((length, 16, 16, 3), dtype=np.uint8),
            ["timestep", "height", "width", "channel"],
        )
    if extra_episode_arrays:
        episode_arrays.update(extra_episode_arrays)
    return TraceBundle.create(
        path,
        manifest=manifest,
        timesteps=pd.DataFrame(timesteps),
        policy_calls=pd.DataFrame(policy_calls),
        generation_steps=pd.DataFrame({"policy_call_index": [0], "generation_step": [0]}),
        streams=pd.DataFrame(
            streams or {"stream_id": ["action"], "name": ["action"], "modality": ["action"]}
        ),
        token_spaces=pd.DataFrame(
            token_spaces
            or {"token_space_id": ["action"], "stream_id": ["action"], "token_count": [2]}
        ),
        tokens=pd.DataFrame(
            tokens or {"token_space_id": ["action"], "token_index": [0], "token_kind": ["action"]}
        ),
        action_normalization=pd.DataFrame(
            action_normalization
            or {
                "normalization_id": ["identity"],
                "mode": ["identity"],
                "stats_ref": [""],
            }
        ),
        scene_state=scene_state,
        camera_state=camera_state,
        episode_arrays=episode_arrays,
        model_arrays=model_sites or (),
        capture_report={"missing_model_sites": []},
    )

__all__ = [name for name in globals() if not name.startswith("__")]
