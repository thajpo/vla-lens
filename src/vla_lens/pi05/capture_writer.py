"""PI0.5 capture writer helpers."""

from __future__ import annotations

import argparse
from typing import Any

from vla_lens.capture import (
    EnvironmentDescriptor,
    EpisodeRecord,
    ModelDescriptor,
    ModelTraceRecord,
    PolicyCallRecord,
    merge_episode_and_model_trace,
)
from vla_lens.lerobot_dataset import write_lerobot_trace_record
from vla_lens.pi05.capture_arrays import (
    _episode_arrays,
    _is_audit_sampled_site,
    _model_arrays,
)
from vla_lens.pi05.capture_metadata import (
    _capture_design_metadata,
    _capture_design_request_metadata,
)
from vla_lens.pi05.capture_schema import (
    CaptureCall,
    CapturePlan,
    EpisodeBuffer,
    canonical_profile,
)
from vla_lens.pi05.capture_tables import (
    _action_normalization_table,
    _buffer_action_dim,
    _camera_state_table,
    _context_availability_fields,
    _context_table,
    _evaluation_table,
    _generation_steps_table,
    _image_preprocessing_table,
    _libero_action_space_metadata,
    _prompt_metadata_table,
    _scene_state_table,
    _timesteps_table,
    _token_tables,
)
from vla_lens.pi05.capture_utils import (
    _array_size_summary,
    _info_bool,
)
from vla_lens.pi05.context_capture import (
    ContextCaptureResult,
    capture_libero_context,
)
from vla_lens.pi05.full_capture import (
    missing_pi05_full_sites,
    pi05_full_site_declarations,
    required_pi05_full_site_names,
)
from vla_lens.traces import ModelSiteSpec, TraceManifest


def _write_episode(
    buffer: EpisodeBuffer,
    args: argparse.Namespace,
    policy: Any,
    plan: CapturePlan,
    *,
    env: Any | None = None,
) -> None:
    length = len(buffer.executed_actions)
    context = capture_libero_context(
        buffer.observations,
        env,
        scene_snapshots=buffer.scene_snapshots,
        camera_snapshots=buffer.camera_snapshots,
    )
    episode_arrays = {**_episode_arrays(buffer, length), **dict(context.arrays)}
    model_arrays = _model_arrays(buffer, plan)
    size_summary = _array_size_summary(episode_arrays, model_arrays)
    timesteps = _timesteps_table(buffer, length)
    generation_steps = _generation_steps_table(buffer)
    streams, token_spaces, tokens = _token_tables(buffer)
    prompt_metadata = _prompt_metadata_table(buffer)
    image_preprocessing = _image_preprocessing_table(buffer)
    action_normalization = _action_normalization_table(buffer)
    evaluation = _evaluation_table(buffer, length)
    capture_plan = plan.to_metadata()
    capture_report = _capture_report(buffer, plan, context, model_arrays=model_arrays)
    metadata = {
        "capture_profile": args.capture_profile,
        "requested_profile": args.capture_profile,
        "actual_profile": canonical_profile(plan.profile),
        "complete": bool(capture_report.get("complete", True)),
        "capture_plan": capture_plan,
        "capture_size": size_summary,
        "action_space": _libero_action_space_metadata(action_dim=_buffer_action_dim(buffer)),
        "task_name": buffer.task_name,
        "seed": buffer.seed,
        "capture_capabilities": {
            "raw_capture_fallback": False,
            "trace_native_capture": True,
            "policy_call_axis": True,
            "vlm_hidden": plan.vlm_hidden,
            "vlm_attention": plan.vlm_attention,
            "expert_hidden": plan.expert_hidden,
            "expert_attention": plan.expert_attention,
        },
    }
    dataset_id = _capture_dataset_id(args)
    if dataset_id:
        metadata["dataset_id"] = dataset_id
    metadata.update(_capture_design_metadata(args))
    manifest = TraceManifest(
        trace_id=buffer.trace_id,
        episode_id=buffer.trace_id,
        task_id=str(buffer.task_id),
        prompt=buffer.prompt,
        model_id=args.model_id,
        env_id=args.benchmark,
        robot_id="libero_panda",
        outcome="success" if buffer.success else "failure",
        length=length,
        metadata=metadata,
    )
    episode = EpisodeRecord(
        manifest=manifest,
        timesteps=timesteps,
        episode_arrays=episode_arrays,
        environment=EnvironmentDescriptor(
            env_family="libero",
            env_id=args.benchmark,
            simulator="robosuite",
            benchmark=args.benchmark,
            task_id=args.task_id,
            seed=buffer.seed,
            replay_supported=True,
            state_available=True,
            metadata={"obs_size": args.obs_size},
        ),
        tokens=tokens,
        generation_steps=generation_steps,
        streams=streams,
        token_spaces=token_spaces,
        robot_state=_context_table(context, "robot"),
        scene_state=_scene_state_table(context),
        camera_state=_camera_state_table(context),
        evaluation=evaluation,
        image_preprocessing=image_preprocessing,
        prompt_metadata=prompt_metadata,
        action_normalization=action_normalization,
        capture_request={
            "requested_profile": args.capture_profile,
            "model_id": args.model_id,
            "capture_design": str(args.capture_design or "single_trace"),
            **_capture_design_request_metadata(args),
            **({"dataset_id": dataset_id} if dataset_id else {}),
        },
        capture_plan=capture_plan,
        capture_report=capture_report,
    )
    model_trace = ModelTraceRecord(
        descriptor=ModelDescriptor(
            model_family="pi05",
            model_id=args.model_id,
            metadata={
                "device": str(policy.config.device),
                "profile": canonical_profile(plan.profile),
                "requested_profile": args.capture_profile,
                "capture_design": str(args.capture_design or "single_trace"),
                **_capture_design_request_metadata(args),
                **({"dataset_id": dataset_id} if dataset_id else {}),
            },
        ),
        model_arrays=model_arrays,
        policy_calls=[
            PolicyCallRecord(
                call.call_index,
                call.env_timestep,
                {
                    "episode_id": buffer.trace_id,
                    "env_timestep_end": _call_end_timestep(buffer.calls, call, length),
                    "prompt_id": "prompt.default",
                    "model_id": args.model_id,
                    "model_family": "pi05",
                    "model_call_kind": "policy_action_chunk",
                    "action_generator_kind": "flow_matching",
                    "action_horizon": int(call.final_action_chunk.shape[0]),
                    "action_dim": int(call.final_action_chunk.shape[-1]),
                    "preprocess_id": "lerobot.default",
                    "postprocess_id": "lerobot.default",
                    **call.policy_call_metadata,
                },
            )
            for call in buffer.calls
        ],
        metadata={"capture_profile": args.capture_profile, "capture_plan": capture_plan},
    )
    record = merge_episode_and_model_trace(episode, model_trace)
    write_lerobot_trace_record(
        record,
        args.vlatrace_out_root,
        overwrite=True,
    )


def _capture_dataset_id(args: argparse.Namespace) -> str:
    return str(args.dataset_id or "").strip()


def _capture_report(
    buffer: EpisodeBuffer,
    plan: CapturePlan,
    context: ContextCaptureResult,
    *,
    model_arrays: list[ModelSiteSpec] | None = None,
) -> dict[str, Any]:
    declared_model_sites = _declared_pi05_sites(plan)
    captured_model_sites = (
        [spec.name for spec in model_arrays] if model_arrays is not None else declared_model_sites
    )
    required_model_sites = (
        list(
            required_pi05_full_site_names(
                vlm_layers=plan.vlm_layers,
                expert_layers=plan.expert_layers,
            )
        )
        if plan.capture_audit_full_sites
        else []
    )
    if plan.capture_audit_full_sites:
        missing_model_sites = _pi05_true_full_missing_sites(buffer, plan)
    else:
        captured_names = set(captured_model_sites)
        missing_model_sites = sorted(
            name for name in declared_model_sites if name not in captured_names
        )
    captured_context, unavailable_context = _context_availability_fields(context)
    return {
        "requested_profile": plan.profile,
        "actual_profile": canonical_profile(plan.profile),
        "complete": not missing_model_sites,
        "captured_cheap_fields": sorted(
            item
            for item in [
                "task_id",
                "task_name",
                "prompt",
                "timesteps",
                "policy_calls",
                "reward",
                "done",
                "action",
                "action_chunks",
                "generation_actions",
                "observation.images.main" if buffer.frames else None,
                "observation.images.wrist" if buffer.wrist_frames else None,
                "tokens",
                "token_spaces",
                "streams",
                *captured_context,
            ]
            if item
        ),
        "unavailable_cheap_fields": unavailable_context,
        "captured_model_sites": captured_model_sites,
        "declared_model_sites": declared_model_sites,
        "required_model_sites": required_model_sites,
        "missing_model_sites": missing_model_sites,
    }

def _declared_pi05_sites(plan: CapturePlan) -> list[str]:
    profile = canonical_profile(plan.profile)
    if profile == "rollout":
        return []
    if profile == "audit_full":
        return list(
            required_pi05_full_site_names(
                vlm_layers=plan.vlm_layers,
                expert_layers=plan.expert_layers,
            )
        )
    declared_raw_sites: list[str] = []
    if profile in {"audit_sampled", "audit_windowed"}:
        declared_raw_sites = [
            declaration.name
            for declaration in pi05_full_site_declarations(
                vlm_layers=plan.vlm_layers,
                expert_layers=plan.expert_layers,
            )
            if _is_audit_sampled_site(declaration)
        ]
    sites = ["pi05.vlm.prefix.image_hidden_tokens"]
    for layer in plan.vlm_layers:
        if plan.vlm_hidden != "none":
            sites.append(f"pi05.vlm.layers.{layer}.prefix.hidden_{plan.vlm_hidden}")
        if plan.vlm_attention != "none":
            sites.append(f"pi05.vlm.layers.{layer}.prefix.attention")
        if plan.capture_bridge_sites:
            sites.extend(
                [
                    f"pi05.vlm.layers.{layer}.kv_cache.key",
                    f"pi05.vlm.layers.{layer}.kv_cache.value",
                ]
            )
    for layer in plan.expert_layers:
        if plan.expert_hidden != "none":
            sites.append(f"pi05.expert.layers.{layer}.by_step.hidden_{plan.expert_hidden}")
        if plan.expert_attention != "none":
            sites.append(f"pi05.expert.layers.{layer}.by_step.attention")
    if plan.capture_bridge_sites:
        sites.extend(
            [
                "pi05.expert.by_step.input_embeddings",
                "pi05.action_head.input",
                "pi05.action_head.output",
            ]
        )
    return list(dict.fromkeys([*sites, *declared_raw_sites]))

def _pi05_true_full_missing_sites(buffer: EpisodeBuffer, plan: CapturePlan) -> list[str]:
    captured = {name for call in buffer.calls for name in call.full_site_arrays}
    return list(
        missing_pi05_full_sites(
            captured,
            vlm_layers=plan.vlm_layers,
            expert_layers=plan.expert_layers,
        )
    )

def _call_end_timestep(calls: list[CaptureCall], call: CaptureCall, length: int) -> int:
    ordered = sorted(calls, key=lambda item: item.env_timestep)
    for index, item in enumerate(ordered):
        if item.call_index != call.call_index:
            continue
        next_start = ordered[index + 1].env_timestep if index + 1 < len(ordered) else length
        return max(call.env_timestep, next_start - 1)
    return max(call.env_timestep, length - 1)

def _episode_success(buffer: EpisodeBuffer) -> bool:
    saw_success_signal = False
    for info in buffer.infos:
        is_success = _info_bool(info, "is_success")
        if is_success is not None:
            saw_success_signal = True
            if is_success:
                return True
    if saw_success_signal:
        return False
    return bool(buffer.rewards and max(buffer.rewards) > 0.0)
