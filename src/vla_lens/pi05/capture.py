"""Trace-native PI0.5/LIBERO capture runner."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vla_lens.capture import (
    EnvironmentDescriptor,
    EpisodeRecord,
    ModelDescriptor,
    ModelTraceRecord,
    PolicyCallRecord,
    merge_episode_and_model_trace,
    write_trace_record,
)
from vla_lens.pi05.context_capture import (
    ContextCaptureResult,
    capture_camera_snapshot,
    capture_libero_context,
    capture_scene_snapshot,
)
from vla_lens.pi05.full_capture import (
    IncompletePI05FullCaptureError,
    missing_pi05_full_sites,
    pi05_full_site_declarations,
    required_pi05_full_site_names,
)
from vla_lens.pi05.token_metadata import (
    EXPERT_CONTEXT_STREAM_ID,
    EXPERT_CONTEXT_TOKEN_SPACE_ID,
    PI05TokenMetadata,
    build_pi05_token_metadata,
)
from vla_lens.traces import ArraySpec, ModelSiteSpec, TraceDataset, TraceManifest
from vla_lens.validation import validate_trace_dataset

LANDMARK_5_LAYERS = (0, 4, 8, 12, 17)
ALL_PI05_LAYERS = tuple(range(18))
PROFILE_ALIASES = {
    "representation": "features",
    "mechanistic_light": "mechanistic_sampled",
    "mechanistic_heavy": "mechanistic_all",
    "full": "audit_full",
}
CANONICAL_PROFILES = (
    "rollout",
    "features",
    "mechanistic_sampled",
    "mechanistic_all",
    "internals_sampled",
    "audit_full",
    "custom",
)
PROFILE_CHOICES = (*CANONICAL_PROFILES, *PROFILE_ALIASES)
PROFILE_LAYERS = {
    "rollout": (),
    "features": LANDMARK_5_LAYERS,
    "mechanistic_sampled": LANDMARK_5_LAYERS,
    "mechanistic_all": ALL_PI05_LAYERS,
    "internals_sampled": LANDMARK_5_LAYERS,
    "audit_full": ALL_PI05_LAYERS,
    "custom": LANDMARK_5_LAYERS,
}
PROFILE_VLM_LAYERS = PROFILE_LAYERS
PROFILE_EXPERT_LAYERS = PROFILE_LAYERS

HIDDEN_RESOLUTIONS = ("profile", "none", "mean", "tokens")
ATTENTION_RESOLUTIONS = ("profile", "none", "key_mass", "full")
STORAGE_DTYPES = ("float16", "float32")
LIBERO_ACTION_DIM_NAMES = (
    "eef_delta_x",
    "eef_delta_y",
    "eef_delta_z",
    "eef_rotvec_delta_x",
    "eef_rotvec_delta_y",
    "eef_rotvec_delta_z",
    "gripper",
)
LIBERO_ACTION_DIM_LABELS = (
    "EEF delta x",
    "EEF delta y",
    "EEF delta z",
    "EEF rotvec delta x",
    "EEF rotvec delta y",
    "EEF rotvec delta z",
    "Gripper command",
)
LIBERO_ACTION_DIM_UNITS = (
    "normalized OSC command",
    "normalized OSC command",
    "normalized OSC command",
    "normalized OSC command",
    "normalized OSC command",
    "normalized OSC command",
    "normalized gripper command",
)


@dataclass(frozen=True, slots=True)
class CapturePlan:
    profile: str
    vlm_layers: tuple[int, ...]
    expert_layers: tuple[int, ...]
    vlm_hidden: str
    vlm_attention: str
    expert_hidden: str
    expert_attention: str
    storage_dtype: str

    @property
    def np_dtype(self) -> np.dtype:
        return np.dtype(self.storage_dtype)

    @property
    def capture_bridge_sites(self) -> bool:
        return canonical_profile(self.profile) in {
            "mechanistic_sampled",
            "mechanistic_all",
            "internals_sampled",
            "audit_full",
        }

    @property
    def capture_audit_full_sites(self) -> bool:
        return canonical_profile(self.profile) == "audit_full"

    @property
    def capture_internals_sites(self) -> bool:
        return canonical_profile(self.profile) in {"internals_sampled", "audit_full"}

    def to_metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["profile"] = canonical_profile(self.profile)
        if self.profile != payload["profile"]:
            payload["requested_profile"] = self.profile
            payload["profile_alias"] = self.profile
        payload["layers"] = sorted(set(self.vlm_layers) | set(self.expert_layers))
        payload["vlm_layers"] = list(self.vlm_layers)
        payload["expert_layers"] = list(self.expert_layers)
        payload["profile_dimensions"] = _plan_dimensions(self)
        payload["captures_bridge_sites"] = self.capture_bridge_sites
        payload["bridge_sites"] = (
            [
                "vlm_kv_cache_key_value",
                "expert_generation_input_embeddings",
                "action_head_input_output",
            ]
            if self.capture_bridge_sites
            else []
        )
        payload["runtime_collections"] = (
            [_past_key_values_collection_metadata(self)] if self.capture_bridge_sites else []
        )
        payload["axis_strategy"] = "policy_call"
        payload["attention_full_semantics"] = "head x query_token x key_token"
        payload["attention_key_mass_semantics"] = "head x key_token, mean over query_token"
        return payload


def _past_key_values_collection_metadata(plan: "CapturePlan") -> dict[str, Any]:
    return {
        "id": "pi05.vlm.past_key_values",
        "label": "Layer-wise prefix K/V",
        "kind": "runtime_collection",
        "materialized": False,
        "aggregation": "none",
        "members": [
            {
                "layer": int(layer),
                "component": component,
                "site_name": f"pi05.vlm.layers.{layer}.kv_cache.{component}",
            }
            for layer in plan.vlm_layers
            for component in ("key", "value")
        ],
    }


def canonical_profile(profile: str) -> str:
    return PROFILE_ALIASES.get(str(profile), str(profile))


def profile_dimensions(profile: str) -> dict[str, Any]:
    profile = canonical_profile(profile)
    if profile == "rollout":
        return {
            "layer_coverage": {"vlm": "none", "expert": "none"},
            "families": {
                "representations": "none",
                "attention": "none",
                "cache": "none",
                "action_head": "none",
                "internals": "none",
                "state_setup": "none",
            },
        }
    if profile == "features":
        return {
            "layer_coverage": {"vlm": "landmark_5", "expert": "landmark_5"},
            "families": {
                "representations": "tokens",
                "attention": "none",
                "cache": "none",
                "action_head": "none",
                "internals": "none",
                "state_setup": "none",
            },
        }
    if profile == "mechanistic_all":
        layer_coverage = {"vlm": "all", "expert": "all"}
    elif profile == "audit_full":
        layer_coverage = {"vlm": "all", "expert": "all"}
    elif profile == "custom":
        return {
            "layer_coverage": {"vlm": "sampled_5", "expert": "sampled_5"},
            "families": {
                "representations": "tokens",
                "attention": "none",
                "cache": "none",
                "action_head": "none",
                "internals": "none",
                "state_setup": "none",
            },
        }
    else:
        layer_coverage = {"vlm": "sampled_5", "expert": "sampled_5"}
    return {
        "layer_coverage": layer_coverage,
        "families": {
            "representations": "tokens",
            "attention": "full_probs",
            "cache": "layer_kv",
            "action_head": "io",
            "internals": "full_raw"
            if profile == "audit_full"
            else ("selected_ops" if profile == "internals_sampled" else "none"),
            "state_setup": "full_raw" if profile == "audit_full" else "none",
        },
    }


def _plan_dimensions(plan: CapturePlan) -> dict[str, Any]:
    profile = canonical_profile(plan.profile)

    def coverage_label(layers: tuple[int, ...]) -> str | list[int]:
        if not layers:
            return "none"
        if layers == ALL_PI05_LAYERS:
            return "all"
        if layers == LANDMARK_5_LAYERS:
            return "landmark_5" if profile == "features" else "sampled_5"
        return [int(layer) for layer in layers]

    def attention_label(vlm_attention: str, expert_attention: str) -> str | dict[str, str]:
        labels = {
            "none": "none",
            "key_mass": "key_mass",
            "full": "full_probs",
        }
        vlm = labels.get(vlm_attention, vlm_attention)
        expert = labels.get(expert_attention, expert_attention)
        if vlm == expert:
            return vlm
        return {"vlm": vlm, "expert": expert}

    def representation_label(vlm_hidden: str, expert_hidden: str) -> str | dict[str, str]:
        if vlm_hidden == expert_hidden:
            return vlm_hidden
        return {"vlm": vlm_hidden, "expert": expert_hidden}

    return {
        "layer_coverage": {
            "vlm": coverage_label(plan.vlm_layers),
            "expert": coverage_label(plan.expert_layers),
        },
        "families": {
            "representations": representation_label(plan.vlm_hidden, plan.expert_hidden),
            "attention": attention_label(plan.vlm_attention, plan.expert_attention),
            "cache": "layer_kv" if plan.capture_bridge_sites else "none",
            "action_head": "io" if plan.capture_bridge_sites else "none",
            "internals": "full_raw"
            if plan.capture_audit_full_sites
            else ("selected_ops" if profile == "internals_sampled" else "none"),
            "state_setup": "full_raw" if plan.capture_audit_full_sites else "none",
        },
    }


@dataclass
class CaptureCall:
    call_index: int
    env_timestep: int
    final_action_chunk: np.ndarray
    denoising_actions: np.ndarray
    suffix_hidden: np.ndarray
    prefix_image_hidden: np.ndarray | None = None
    prefix_patches_per_image: int | None = None
    prefix_image_slots: int | None = None
    attention_mass: np.ndarray | None = None
    denoise_velocities: np.ndarray | None = None
    vlm_hidden_by_layer: dict[int, np.ndarray] = field(default_factory=dict)
    vlm_attention_by_layer: dict[int, np.ndarray] = field(default_factory=dict)
    expert_hidden_by_layer: dict[int, np.ndarray] = field(default_factory=dict)
    expert_attention_by_layer: dict[int, np.ndarray] = field(default_factory=dict)
    vlm_kv_key_by_layer: dict[int, np.ndarray] = field(default_factory=dict)
    vlm_kv_value_by_layer: dict[int, np.ndarray] = field(default_factory=dict)
    generation_input_embeddings: np.ndarray | None = None
    action_head_input: np.ndarray | None = None
    action_head_output: np.ndarray | None = None
    token_metadata: PI05TokenMetadata | None = None
    policy_call_metadata: dict[str, Any] = field(default_factory=dict)
    full_site_arrays: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class EpisodeBuffer:
    trace_id: str
    task_id: int
    task_name: str
    prompt: str
    seed: int
    frames: list[np.ndarray] = field(default_factory=list)
    wrist_frames: list[np.ndarray] = field(default_factory=list)
    executed_actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    calls: list[CaptureCall] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    scene_snapshots: list[dict[str, Any]] = field(default_factory=list)
    camera_snapshots: list[dict[str, Any]] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    success: bool = False


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.delete_existing and args.vlatrace_out_root.exists():
        shutil.rmtree(args.vlatrace_out_root)
    args.vlatrace_out_root.mkdir(parents=True, exist_ok=True)

    _run_capture(args)
    dataset = TraceDataset.open(args.vlatrace_out_root)
    validation = validate_trace_dataset(dataset)
    if not validation.valid:
        raise SystemExit(validation.to_dict())
    print(f"wrote {len(dataset.bundles)} traces to {args.vlatrace_out_root}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--benchmark", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument(
        "--capture-profile",
        choices=PROFILE_CHOICES,
        default="mechanistic_sampled",
    )
    parser.add_argument(
        "--vlm-hidden-resolution",
        choices=HIDDEN_RESOLUTIONS,
        default="profile",
        help="VLM transformer hidden capture resolution. 'profile' derives from --capture-profile.",
    )
    parser.add_argument(
        "--vlm-attention-resolution",
        choices=ATTENTION_RESOLUTIONS,
        default="profile",
        help="VLM attention capture resolution. 'full' stores head x query x key weights.",
    )
    parser.add_argument(
        "--expert-hidden-resolution",
        choices=HIDDEN_RESOLUTIONS,
        default="profile",
        help="Diffusion expert hidden capture resolution. 'tokens' stores token x channel states.",
    )
    parser.add_argument(
        "--expert-attention-resolution",
        choices=ATTENTION_RESOLUTIONS,
        default="profile",
        help=(
            "Diffusion expert attention capture resolution. "
            "'full' stores head x query x key weights."
        ),
    )
    parser.add_argument(
        "--storage-dtype",
        choices=STORAGE_DTYPES,
        default="float16",
        help="Numeric dtype used for captured model internals in .vlatrace arrays.",
    )
    parser.add_argument("--vlatrace-out-root", type=Path, default=Path("runs/pi05_golden"))
    parser.add_argument(
        "--dataset-id",
        help="Dataset identifier stored in every trace manifest metadata.",
    )
    parser.add_argument("--obs-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--delete-existing", action="store_true")
    return parser.parse_args(argv)


def _run_capture(args: argparse.Namespace) -> None:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy
    from libero.libero.benchmark import get_benchmark

    plan = _resolve_capture_plan(args)
    print(f"capture plan: {plan.to_metadata()}", flush=True)

    benchmark = get_benchmark(args.benchmark)(task_order_index=0)
    task = benchmark.get_task(args.task_id)
    task_name = str(task.name)

    policy_cfg = PreTrainedConfig.from_pretrained(
        args.model_id,
        cli_overrides=[
            f"--device={args.device}",
            "--compile_model=false",
            f"--dtype={args.dtype}",
        ],
    )
    policy_cfg.pretrained_path = Path(args.model_id)
    policy = PI05Policy.from_pretrained(args.model_id, config=policy_cfg)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.model_id,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    env_cfg = make_env_config(
        "libero",
        task=args.benchmark,
        task_ids=[args.task_id],
        observation_height=args.obs_size,
        observation_width=args.obs_size,
        camera_name="agentview_image,robot0_eye_in_hand_image",
        control_mode="relative",
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[args.benchmark][args.task_id]
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=policy_cfg,
    )

    try:
        for episode_index in range(args.episodes):
            seed = args.start_seed + episode_index
            trace_id = f"pi05_{args.capture_profile}_{args.benchmark}_task{args.task_id}_seed{seed}"
            buffer = EpisodeBuffer(
                trace_id=trace_id,
                task_id=args.task_id,
                task_name=task_name,
                prompt=_task_prompt(task),
                seed=seed,
            )
            policy.reset()
            observation, _ = env.reset(seed=[seed])
            done = np.array([False])
            step = 0
            max_steps = args.max_steps or int(env.call("_max_episode_steps")[0])
            action_iter: Iterator[np.ndarray] = iter(())

            while not np.all(done) and step < max_steps:
                buffer.observations.append(dict(observation))
                buffer.scene_snapshots.append(capture_scene_snapshot(env))
                buffer.camera_snapshots.append(capture_camera_snapshot(env, observation))
                frame, wrist = _extract_frames(observation)
                if frame is not None:
                    buffer.frames.append(frame)
                if wrist is not None:
                    buffer.wrist_frames.append(wrist)

                try:
                    action_numpy = next(action_iter)
                except StopIteration:
                    obs = preprocess_observation(observation)
                    obs = add_envs_task(env, obs)
                    obs = env_preprocessor(obs)
                    obs = preprocessor(obs)
                    call = _predict_action_chunk(policy, obs, len(buffer.calls), step, plan)
                    _attach_token_metadata(call, obs, buffer)
                    buffer.calls.append(call)
                    actions = postprocessor(torch.as_tensor(call.final_action_chunk))
                    action_transition = env_postprocessor({"action": actions})
                    action_chunk = action_transition["action"].detach().cpu().numpy()
                    action_iter = iter(action_chunk)
                    action_numpy = next(action_iter)

                env_action = np.expand_dims(action_numpy, axis=0)
                observation, reward, terminated, truncated, info = env.step(env_action)
                buffer.executed_actions.append(np.asarray(action_numpy, dtype=np.float32))
                buffer.rewards.append(float(np.asarray(reward).reshape(-1)[0]))
                buffer.terminated.append(bool(np.asarray(terminated).reshape(-1)[0]))
                buffer.truncated.append(bool(np.asarray(truncated).reshape(-1)[0]))
                buffer.infos.append(_jsonable_mapping(_first_info(info)))
                done = terminated | truncated | done
                step += 1

            buffer.success = _episode_success(buffer)
            _write_episode(buffer, args, policy, plan, env=env)
            print(f"{trace_id} steps={step} calls={len(buffer.calls)} success={buffer.success}")
    finally:
        env.close()


def _resolve_capture_plan(args: argparse.Namespace) -> CapturePlan:
    requested_profile = str(args.capture_profile)
    profile = canonical_profile(requested_profile)
    vlm_layers = tuple(PROFILE_VLM_LAYERS[profile])
    expert_layers = tuple(PROFILE_EXPERT_LAYERS[profile])

    def hidden(value: str, *, family: str) -> str:
        if value != "profile":
            return value
        if profile == "rollout":
            return "none"
        if profile in {
            "features",
            "mechanistic_sampled",
            "mechanistic_all",
            "internals_sampled",
            "custom",
        }:
            return "tokens"
        if profile == "audit_full":
            return "tokens"
        return "tokens"

    def attention(value: str) -> str:
        if value != "profile":
            return value
        return (
            "full"
            if profile
            in {
                "mechanistic_sampled",
                "mechanistic_all",
                "internals_sampled",
                "audit_full",
            }
            else "none"
        )

    return CapturePlan(
        profile=profile,
        vlm_layers=vlm_layers,
        expert_layers=expert_layers,
        vlm_hidden=hidden(str(args.vlm_hidden_resolution), family="vlm"),
        vlm_attention=attention(str(args.vlm_attention_resolution)),
        expert_hidden=hidden(str(args.expert_hidden_resolution), family="expert"),
        expert_attention=attention(str(args.expert_attention_resolution)),
        storage_dtype=str(args.storage_dtype),
    )


def _capture_dataset_id(args: argparse.Namespace) -> str:
    return str(args.dataset_id or "").strip()


def _predict_action_chunk(
    policy: Any,
    obs: dict[str, Any],
    call_index: int,
    step: int,
    plan: CapturePlan,
) -> CaptureCall:
    import torch
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks
    from transformers.models.gemma import modeling_gemma

    model = policy.model
    full_recorder = _PI05FullSiteRecorder(plan) if plan.capture_internals_sites else None
    capture: dict[str, list[np.ndarray]] = {
        "x_t": [],
        "denoise_velocities": [],
        "prefix_image_hidden": [],
        "generation_input_embeddings": [],
        "action_head_input": [],
        "action_head_output": [],
    }
    current_denoise_step: dict[str, int | None] = {"index": None}
    vlm_hidden_by_layer: dict[int, np.ndarray] = {}
    vlm_attention_by_layer: dict[int, np.ndarray] = {}
    expert_hidden_by_layer: dict[int, list[np.ndarray]] = defaultdict(list)
    expert_attention_by_layer: dict[int, list[np.ndarray]] = defaultdict(list)
    vlm_kv_key_by_layer: dict[int, np.ndarray] = {}
    vlm_kv_value_by_layer: dict[int, np.ndarray] = {}

    original_denoise = model.denoise_step
    original_embed_prefix = model.embed_prefix
    original_embed_suffix = model.embed_suffix
    original_action_out_forward = model.action_out_proj.forward
    original_embed_image = model.paligemma_with_expert.embed_image
    original_attention = modeling_gemma.eager_attention_forward
    vlm_model = model.paligemma_with_expert.paligemma.model.language_model
    expert_model = model.paligemma_with_expert.gemma_expert.model
    original_vlm_forward = vlm_model.forward
    original_expert_forward = expert_model.forward
    original_vlm_rotary_forward = vlm_model.rotary_emb.forward
    original_expert_rotary_forward = expert_model.rotary_emb.forward
    full_hook_handles: list[Any] = []
    patched_mlps: list[tuple[Any, Any]] = []
    if full_recorder is not None:
        full_hook_handles, patched_mlps = _install_full_layer_hooks(
            full_recorder,
            plan,
            current_denoise_step,
            vlm_model=vlm_model,
            expert_model=expert_model,
        )
    vlm_attention_modules = {
        id(layer.self_attn): layer_idx for layer_idx, layer in enumerate(vlm_model.layers)
    }
    expert_attention_modules = {
        id(layer.self_attn): layer_idx for layer_idx, layer in enumerate(expert_model.layers)
    }

    def embed_image_wrapper(image: Any) -> Any:
        out = original_embed_image(image)
        capture["prefix_image_hidden"].append(_to_numpy(out, dtype=plan.np_dtype).squeeze(0))
        return out

    def embed_prefix_wrapper(*args: Any, **kwargs: Any) -> Any:
        out = original_embed_prefix(*args, **kwargs)
        if full_recorder is not None:
            prefix_embs, prefix_pad_masks, prefix_att_masks = out
            full_recorder.capture(
                "pi05.vlm.prefix.input_embeddings",
                prefix_embs,
                dtype=plan.np_dtype,
            )
            full_recorder.capture(
                "pi05.inputs.attention_mask",
                prefix_pad_masks,
                dtype=np.bool_,
            )
            full_recorder.capture(
                "pi05.inputs.causal_mask",
                make_att_2d_masks(prefix_pad_masks, prefix_att_masks),
                dtype=np.bool_,
            )
            full_recorder.capture(
                "pi05.inputs.position_ids",
                torch.cumsum(prefix_pad_masks, dim=1) - 1,
                dtype=np.int64,
            )
            full_recorder.capture(
                "pi05.inputs.rope.metadata",
                _rope_metadata_array(vlm_model),
                dtype=np.float32,
                squeeze_batch=False,
            )
        return out

    def embed_suffix_wrapper(*args: Any, **kwargs: Any) -> Any:
        out = original_embed_suffix(*args, **kwargs)
        suffix_embs = out[0]
        if plan.capture_bridge_sites and current_denoise_step["index"] is not None:
            capture["generation_input_embeddings"].append(
                _to_numpy(suffix_embs, dtype=plan.np_dtype).squeeze(0)
            )
        if full_recorder is not None and current_denoise_step["index"] is not None:
            full_recorder.capture(
                "pi05.expert.by_step.input_embeddings",
                suffix_embs,
                dtype=plan.np_dtype,
                generation_step=current_denoise_step["index"],
            )
        return out

    def vlm_rotary_wrapper(*args: Any, **kwargs: Any) -> Any:
        out = original_vlm_rotary_forward(*args, **kwargs)
        if full_recorder is not None:
            cos, sin = out
            full_recorder.capture("pi05.inputs.rope.cos", cos, dtype=plan.np_dtype)
            full_recorder.capture("pi05.inputs.rope.sin", sin, dtype=plan.np_dtype)
        return out

    def expert_rotary_wrapper(*args: Any, **kwargs: Any) -> Any:
        out = original_expert_rotary_forward(*args, **kwargs)
        if full_recorder is not None and current_denoise_step["index"] is not None:
            cos, sin = out
            full_recorder.capture(
                "pi05.expert.by_step.rope.cos",
                cos,
                dtype=plan.np_dtype,
                generation_step=current_denoise_step["index"],
            )
            full_recorder.capture(
                "pi05.expert.by_step.rope.sin",
                sin,
                dtype=plan.np_dtype,
                generation_step=current_denoise_step["index"],
            )
        return out

    def action_out_forward_wrapper(*args: Any, **kwargs: Any) -> Any:
        out = original_action_out_forward(*args, **kwargs)
        if plan.capture_bridge_sites and current_denoise_step["index"] is not None and args:
            capture["action_head_input"].append(_to_numpy(args[0], dtype=plan.np_dtype).squeeze(0))
            capture["action_head_output"].append(_to_numpy(out, dtype=plan.np_dtype).squeeze(0))
        if full_recorder is not None and current_denoise_step["index"] is not None and args:
            step_index = current_denoise_step["index"]
            full_recorder.capture(
                "pi05.action_head.input",
                args[0],
                dtype=plan.np_dtype,
                generation_step=step_index,
            )
            full_recorder.capture(
                "pi05.action_head.output",
                out,
                dtype=plan.np_dtype,
                generation_step=step_index,
            )
        return out

    def vlm_forward_wrapper(*args: Any, **kwargs: Any) -> Any:
        if plan.vlm_hidden != "none":
            kwargs["output_hidden_states"] = True
        out = original_vlm_forward(*args, **kwargs)
        if plan.vlm_hidden != "none":
            _capture_hidden_layers(
                getattr(out, "hidden_states", None),
                layers=plan.vlm_layers,
                resolution=plan.vlm_hidden,
                dtype=plan.np_dtype,
                target=vlm_hidden_by_layer,
                append=False,
            )
        return out

    def expert_forward_wrapper(*args: Any, **kwargs: Any) -> Any:
        if plan.expert_hidden != "none":
            kwargs["output_hidden_states"] = True
        out = original_expert_forward(*args, **kwargs)
        if plan.expert_hidden != "none" and current_denoise_step["index"] is not None:
            _capture_hidden_layers(
                getattr(out, "hidden_states", None),
                layers=plan.expert_layers,
                resolution=plan.expert_hidden,
                dtype=plan.np_dtype,
                target=expert_hidden_by_layer,
                append=True,
            )
        return out

    def attention_wrapper(*args: Any, **kwargs: Any) -> Any:
        if len(args) >= 1:
            module = args[0]
        else:
            module = kwargs.get("module")
        can_capture_attention = (
            full_recorder is not None and len(args) >= 5 and (len(args) >= 6 or "scaling" in kwargs)
        )
        if can_capture_attention:
            query, key, value, attention_mask = args[1:5]
            scaling = args[5] if len(args) >= 6 else kwargs["scaling"]
            dropout = kwargs.get("dropout", args[6] if len(args) > 6 else 0.0)
            key_states = modeling_gemma.repeat_kv(key, module.num_key_value_groups)
            value_states = modeling_gemma.repeat_kv(value, module.num_key_value_groups)
            pre_mask_scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
            post_mask_logits = (
                pre_mask_scores if attention_mask is None else pre_mask_scores + attention_mask
            )
            attn_weights = torch.nn.functional.softmax(
                post_mask_logits,
                dim=-1,
                dtype=torch.float32,
            ).to(query.dtype)
            attn_weights = torch.nn.functional.dropout(
                attn_weights,
                p=float(dropout),
                training=module.training,
            )
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            out = (attn_output, attn_weights)
        else:
            out = original_attention(*args, **kwargs)
        if not isinstance(out, tuple) or len(out) < 2:
            return out
        attn_weights = out[1]
        module_id = id(module)
        if module_id in vlm_attention_modules and plan.vlm_attention != "none":
            layer = vlm_attention_modules[module_id]
            if layer in plan.vlm_layers:
                if plan.capture_bridge_sites and len(args) >= 4:
                    query, key, value = args[1:4]
                    vlm_kv_key_by_layer[layer] = _to_numpy(key, dtype=plan.np_dtype).squeeze(0)
                    vlm_kv_value_by_layer[layer] = _to_numpy(value, dtype=plan.np_dtype).squeeze(0)
                if can_capture_attention:
                    query, key, value = args[1:4]
                    _capture_full_attention_sites(
                        full_recorder,
                        plan,
                        stack="vlm",
                        layer=layer,
                        generation_step=None,
                        query=query,
                        key=key,
                        value=value,
                        pre_mask_scores=pre_mask_scores,
                        post_mask_logits=post_mask_logits,
                        attention_probs=attn_weights,
                        attn_output=out[0],
                    )
                vlm_attention_by_layer[layer] = _attention_to_numpy(
                    attn_weights,
                    resolution=plan.vlm_attention,
                    dtype=plan.np_dtype,
                )
        elif (
            module_id in expert_attention_modules
            and plan.expert_attention != "none"
            and current_denoise_step["index"] is not None
        ):
            layer = expert_attention_modules[module_id]
            if layer in plan.expert_layers:
                if can_capture_attention:
                    query, key, value = args[1:4]
                    _capture_full_attention_sites(
                        full_recorder,
                        plan,
                        stack="expert",
                        layer=layer,
                        generation_step=current_denoise_step["index"],
                        query=query,
                        key=key,
                        value=value,
                        pre_mask_scores=pre_mask_scores,
                        post_mask_logits=post_mask_logits,
                        attention_probs=attn_weights,
                        attn_output=out[0],
                    )
                expert_attention_by_layer[layer].append(
                    _attention_to_numpy(
                        attn_weights,
                        resolution=plan.expert_attention,
                        dtype=plan.np_dtype,
                    )
                )
        return out

    def denoise_wrapper(*args: Any, **kwargs: Any) -> Any:
        x_t = kwargs.get("x_t") if "x_t" in kwargs else args[2]
        prefix_pad_masks = (
            kwargs.get("prefix_pad_masks") if "prefix_pad_masks" in kwargs else args[0]
        )
        denoise_index = len(capture["x_t"])
        current_denoise_step["index"] = denoise_index
        capture["x_t"].append(_to_numpy(x_t, dtype=plan.np_dtype).squeeze(0))
        if full_recorder is not None:
            _capture_expert_step_inputs(
                full_recorder,
                make_att_2d_masks,
                prefix_pad_masks=prefix_pad_masks,
                x_t=x_t,
                generation_step=denoise_index,
            )
        out = original_denoise(*args, **kwargs)
        capture["denoise_velocities"].append(_to_numpy(out, dtype=plan.np_dtype).squeeze(0))
        current_denoise_step["index"] = None
        return out

    model.paligemma_with_expert.embed_image = embed_image_wrapper
    model.embed_prefix = embed_prefix_wrapper
    model.embed_suffix = embed_suffix_wrapper
    model.denoise_step = denoise_wrapper
    model.action_out_proj.forward = action_out_forward_wrapper
    vlm_model.forward = vlm_forward_wrapper
    expert_model.forward = expert_forward_wrapper
    vlm_model.rotary_emb.forward = vlm_rotary_wrapper
    expert_model.rotary_emb.forward = expert_rotary_wrapper
    modeling_gemma.eager_attention_forward = attention_wrapper
    try:
        with torch.no_grad():
            chunk = policy.predict_action_chunk(obs)
    finally:
        model.denoise_step = original_denoise
        model.embed_prefix = original_embed_prefix
        model.embed_suffix = original_embed_suffix
        model.action_out_proj.forward = original_action_out_forward
        model.paligemma_with_expert.embed_image = original_embed_image
        vlm_model.forward = original_vlm_forward
        expert_model.forward = original_expert_forward
        vlm_model.rotary_emb.forward = original_vlm_rotary_forward
        expert_model.rotary_emb.forward = original_expert_rotary_forward
        modeling_gemma.eager_attention_forward = original_attention
        for handle in full_hook_handles:
            handle.remove()
        for mlp, original_forward in patched_mlps:
            mlp.forward = original_forward

    final_chunk = _to_numpy(chunk, dtype=plan.np_dtype).squeeze(0)
    denoising = np.stack(capture["x_t"], axis=0).astype(plan.np_dtype)
    velocities = np.stack(capture["denoise_velocities"], axis=0).astype(plan.np_dtype)
    generation_input_embeddings = (
        np.stack(capture["generation_input_embeddings"], axis=0).astype(plan.np_dtype)
        if capture["generation_input_embeddings"]
        else None
    )
    action_head_input = (
        np.stack(capture["action_head_input"], axis=0).astype(plan.np_dtype)
        if capture["action_head_input"]
        else None
    )
    action_head_output = (
        np.stack(capture["action_head_output"], axis=0).astype(plan.np_dtype)
        if capture["action_head_output"]
        else None
    )
    prefix_image_hidden = (
        np.concatenate(capture["prefix_image_hidden"], axis=0).astype(plan.np_dtype)
        if capture["prefix_image_hidden"]
        else None
    )
    prefix_patches_per_image = (
        int(capture["prefix_image_hidden"][0].shape[0]) if capture["prefix_image_hidden"] else None
    )
    prefix_image_slots = (
        len(capture["prefix_image_hidden"]) if capture["prefix_image_hidden"] else None
    )
    attention = _expert_attention_key_mass(expert_attention_by_layer, dtype=plan.np_dtype)
    full_site_arrays: dict[str, np.ndarray] = {}
    if full_recorder is not None:
        full_site_arrays = full_recorder.finalized_arrays(generation_steps=denoising.shape[0])
        missing = full_recorder.missing_names(full_site_arrays)
        if missing and plan.capture_audit_full_sites:
            preview = ", ".join(missing[:12])
            suffix = "..." if len(missing) > 12 else ""
            raise IncompletePI05FullCaptureError(
                f"PI0.5 full capture missed {len(missing)} required raw sites: {preview}{suffix}"
            )
    return CaptureCall(
        call_index=call_index,
        env_timestep=step,
        final_action_chunk=final_chunk.astype(plan.np_dtype),
        denoising_actions=denoising,
        suffix_hidden=velocities,
        prefix_image_hidden=prefix_image_hidden,
        prefix_patches_per_image=prefix_patches_per_image,
        prefix_image_slots=prefix_image_slots,
        attention_mass=attention,
        denoise_velocities=velocities,
        vlm_hidden_by_layer=vlm_hidden_by_layer,
        vlm_attention_by_layer=vlm_attention_by_layer,
        expert_hidden_by_layer={
            layer: np.stack(values, axis=0).astype(plan.np_dtype)
            for layer, values in expert_hidden_by_layer.items()
            if values
        },
        expert_attention_by_layer={
            layer: np.stack(values, axis=0).astype(plan.np_dtype)
            for layer, values in expert_attention_by_layer.items()
            if values
        },
        vlm_kv_key_by_layer=vlm_kv_key_by_layer,
        vlm_kv_value_by_layer=vlm_kv_value_by_layer,
        generation_input_embeddings=generation_input_embeddings,
        action_head_input=action_head_input,
        action_head_output=action_head_output,
        full_site_arrays=full_site_arrays,
    )


def _capture_hidden_layers(
    hidden_states: Any,
    *,
    layers: tuple[int, ...],
    resolution: str,
    dtype: np.dtype,
    target: dict[int, Any],
    append: bool,
) -> None:
    if resolution == "none" or hidden_states is None:
        return
    hidden_tuple = tuple(hidden_states)
    if not hidden_tuple:
        return
    for layer in layers:
        hidden_index = min(layer + 1, len(hidden_tuple) - 1)
        array = _hidden_to_numpy(hidden_tuple[hidden_index], resolution=resolution, dtype=dtype)
        if append:
            target[layer].append(array)
        else:
            target[layer] = array


def _hidden_to_numpy(value: Any, *, resolution: str, dtype: np.dtype) -> np.ndarray:
    array = _to_numpy(value, dtype=dtype).squeeze(0)
    if resolution == "mean":
        return np.nanmean(array, axis=0).astype(dtype)
    return array.astype(dtype)


def _attention_to_numpy(value: Any, *, resolution: str, dtype: np.dtype) -> np.ndarray:
    array = _to_numpy(value, dtype=np.float32).squeeze(0)
    if resolution == "key_mass":
        array = np.nanmean(array, axis=-2)
    return array.astype(dtype)


def _expert_attention_key_mass(
    attention_by_layer: dict[int, list[np.ndarray]],
    *,
    dtype: np.dtype,
) -> np.ndarray | None:
    if not attention_by_layer:
        return None
    final_layer = max(attention_by_layer)
    values = attention_by_layer.get(final_layer) or []
    if not values:
        return None
    array = np.stack(values, axis=0).astype(np.float32)
    if array.ndim == 4:
        # denoise_step x head x query_token x key_token
        return np.nanmean(array, axis=(1, 2)).astype(dtype)
    if array.ndim == 3:
        # denoise_step x head x key_token
        return np.nanmean(array, axis=1).astype(dtype)
    return None


class _PI05FullSiteRecorder:
    """Per-policy-call recorder for exact raw PI0.5 full-capture sites."""

    def __init__(self, plan: CapturePlan):
        self.declarations = pi05_full_site_declarations(
            vlm_layers=plan.vlm_layers,
            expert_layers=plan.expert_layers,
        )
        self._declarations_by_name = {item.name: item for item in self.declarations}
        self._arrays: dict[str, np.ndarray] = {}
        self._step_arrays: dict[str, dict[int, np.ndarray]] = defaultdict(dict)

    def capture(
        self,
        name: str,
        value: Any,
        *,
        dtype: np.dtype | str | None = None,
        generation_step: int | None = None,
        squeeze_batch: bool = True,
    ) -> None:
        if name not in self._declarations_by_name:
            raise KeyError(f"Unknown PI0.5 full capture site: {name}")
        array = _capture_numpy(value, dtype=dtype)
        if squeeze_batch and array.ndim > 0 and array.shape[0] == 1:
            array = np.squeeze(array, axis=0)
        if generation_step is None:
            self._arrays[name] = array
        else:
            self._step_arrays[name][int(generation_step)] = array

    def finalized_arrays(self, *, generation_steps: int) -> dict[str, np.ndarray]:
        arrays = dict(self._arrays)
        for name, by_step in self._step_arrays.items():
            if not by_step:
                continue
            sample = np.asarray(next(iter(by_step.values())))
            out = _empty_step_array(generation_steps, sample)
            for step, value in by_step.items():
                if 0 <= step < generation_steps:
                    out[step] = np.asarray(value, dtype=sample.dtype)
            arrays[name] = out
        return arrays

    def missing_names(self, arrays: Mapping[str, np.ndarray]) -> tuple[str, ...]:
        captured = set(arrays)
        return tuple(item.name for item in self.declarations if item.name not in captured)


def _empty_step_array(generation_steps: int, sample: np.ndarray) -> np.ndarray:
    if np.issubdtype(sample.dtype, np.floating):
        fill_value: float | int | bool = np.nan
    elif np.issubdtype(sample.dtype, np.bool_):
        fill_value = False
    else:
        fill_value = -1
    return np.full((generation_steps, *sample.shape), fill_value, dtype=sample.dtype)


def _capture_numpy(value: Any, *, dtype: np.dtype | str | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if dtype is not None and np.dtype(dtype).kind == "f" and hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "to"):
        value = value.to("cpu")
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if dtype is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            array = array.astype(np.dtype(dtype), copy=False)
    return np.ascontiguousarray(array)


def _capture_step(
    current_denoise_step: Mapping[str, int | None],
    stack: str,
) -> int | None:
    if stack == "expert":
        return current_denoise_step.get("index")
    return None


def _full_site_prefix(stack: str, layer: int) -> str:
    return f"pi05.{stack}.layers.{layer}"


def _capture_full_tensor(
    recorder: _PI05FullSiteRecorder | None,
    plan: CapturePlan,
    current_denoise_step: Mapping[str, int | None],
    *,
    stack: str,
    name: str,
    value: Any,
    dtype: np.dtype | str | None = None,
    squeeze_batch: bool = True,
) -> None:
    if recorder is None:
        return
    generation_step = _capture_step(current_denoise_step, stack)
    if stack == "expert" and generation_step is None:
        return
    recorder.capture(
        name,
        value,
        dtype=plan.np_dtype if dtype is None else dtype,
        generation_step=generation_step,
        squeeze_batch=squeeze_batch,
    )


def _register_forward_pre_hook(module: Any, hook: Any) -> Any:
    try:
        return module.register_forward_pre_hook(hook, with_kwargs=True)
    except TypeError:
        return module.register_forward_pre_hook(lambda mod, args: hook(mod, args, {}))


def _register_forward_hook(module: Any, hook: Any) -> Any:
    try:
        return module.register_forward_hook(hook, with_kwargs=True)
    except TypeError:
        return module.register_forward_hook(lambda mod, args, output: hook(mod, args, {}, output))


def _install_full_layer_hooks(
    recorder: _PI05FullSiteRecorder,
    plan: CapturePlan,
    current_denoise_step: Mapping[str, int | None],
    *,
    vlm_model: Any,
    expert_model: Any,
) -> tuple[list[Any], list[tuple[Any, Any]]]:
    handles: list[Any] = []
    patched_mlps: list[tuple[Any, Any]] = []

    for stack, model in (("vlm", vlm_model), ("expert", expert_model)):
        stack_layers = plan.vlm_layers if stack == "vlm" else plan.expert_layers
        for layer_idx, layer in enumerate(getattr(model, "layers", ())):
            if layer_idx not in stack_layers:
                continue
            prefix = _full_site_prefix(stack, int(layer_idx))
            handles.extend(
                _install_full_single_layer_hooks(
                    recorder,
                    plan,
                    current_denoise_step,
                    stack=stack,
                    layer=layer,
                    prefix=prefix,
                )
            )
            patched_mlps.append((layer.mlp, layer.mlp.forward))
            layer.mlp.forward = _make_full_mlp_forward(
                layer.mlp,
                recorder,
                plan,
                current_denoise_step,
                stack=stack,
                prefix=prefix,
            )

    return handles, patched_mlps


def _install_full_single_layer_hooks(
    recorder: _PI05FullSiteRecorder,
    plan: CapturePlan,
    current_denoise_step: Mapping[str, int | None],
    *,
    stack: str,
    layer: Any,
    prefix: str,
) -> list[Any]:
    handles: list[Any] = []

    def capture(name: str, value: Any, *, dtype: np.dtype | str | None = None) -> None:
        _capture_full_tensor(
            recorder,
            plan,
            current_denoise_step,
            stack=stack,
            name=name,
            value=value,
            dtype=dtype,
        )

    def self_attn_pre_hook(_module: Any, args: tuple[Any, ...], _kwargs: Mapping[str, Any]) -> None:
        if args:
            capture(f"{prefix}.residual_pre_attention", args[0])

    def capture_adarms(
        module: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        normed: Any,
        *,
        norm_site: str,
    ) -> None:
        if stack != "expert":
            return
        cond = kwargs.get("cond") if isinstance(kwargs, Mapping) else None
        if cond is None and len(args) > 1:
            cond = args[1]
        dense = getattr(module, "dense", None)
        if cond is None or dense is None:
            return
        modulation = dense(cond)
        x = args[0] if args else normed
        if len(getattr(x, "shape", ())) == 3:
            modulation = modulation.unsqueeze(1)
        scale, shift, gate = modulation.chunk(3, dim=-1)
        capture(f"{prefix}.{norm_site}.scale", scale)
        capture(f"{prefix}.{norm_site}.shift", shift)
        capture(f"{prefix}.{norm_site}.gate", gate)

    def input_norm_hook(
        module: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        output: Any,
    ) -> None:
        normed = output[0] if isinstance(output, tuple) else output
        capture(f"{prefix}.attention_norm_output", normed)
        capture_adarms(module, args, kwargs, normed, norm_site="attention_adarms")

    def post_norm_pre_hook(_module: Any, args: tuple[Any, ...], _kwargs: Mapping[str, Any]) -> None:
        if not args:
            return
        residual = args[0]
        capture(f"{prefix}.residual_post_attention", residual)
        capture(f"{prefix}.residual_pre_mlp", residual)

    def post_norm_hook(
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: Mapping[str, Any],
        output: Any,
    ) -> None:
        normed = output[0] if isinstance(output, tuple) else output
        capture(f"{prefix}.mlp_norm_output", normed)
        capture_adarms(_module, _args, _kwargs, normed, norm_site="mlp_adarms")

    def o_proj_hook(
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: Mapping[str, Any],
        output: Any,
    ) -> None:
        capture(f"{prefix}.attention.o_proj", output)

    def layer_hook(
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: Mapping[str, Any],
        output: Any,
    ) -> None:
        value = output[0] if isinstance(output, tuple) else output
        capture(f"{prefix}.residual_post_mlp", value)

    handles.append(_register_forward_pre_hook(layer.self_attn, self_attn_pre_hook))
    handles.append(_register_forward_hook(layer.input_layernorm, input_norm_hook))
    handles.append(_register_forward_pre_hook(layer.post_attention_layernorm, post_norm_pre_hook))
    handles.append(_register_forward_hook(layer.post_attention_layernorm, post_norm_hook))
    handles.append(_register_forward_hook(layer.self_attn.o_proj, o_proj_hook))
    handles.append(_register_forward_hook(layer, layer_hook))
    return handles


def _make_full_mlp_forward(
    mlp: Any,
    recorder: _PI05FullSiteRecorder,
    plan: CapturePlan,
    current_denoise_step: Mapping[str, int | None],
    *,
    stack: str,
    prefix: str,
) -> Any:
    def capture(name: str, value: Any) -> None:
        _capture_full_tensor(
            recorder,
            plan,
            current_denoise_step,
            stack=stack,
            name=name,
            value=value,
        )

    def mlp_forward(x: Any) -> Any:
        gate = mlp.gate_proj(x)
        up = mlp.up_proj(x)
        intermediate = mlp.act_fn(gate) * up
        down = mlp.down_proj(intermediate)
        capture(f"{prefix}.mlp.gate", gate)
        capture(f"{prefix}.mlp.up", up)
        capture(f"{prefix}.mlp.intermediate", intermediate)
        capture(f"{prefix}.mlp.down", down)
        capture(f"{prefix}.mlp.output", down)
        return down

    return mlp_forward


def _capture_full_attention_sites(
    recorder: _PI05FullSiteRecorder,
    plan: CapturePlan,
    *,
    stack: str,
    layer: int,
    generation_step: int | None,
    query: Any,
    key: Any,
    value: Any,
    pre_mask_scores: Any,
    post_mask_logits: Any,
    attention_probs: Any,
    attn_output: Any,
) -> None:
    prefix = _full_site_prefix(stack, int(layer))
    step_kwargs = {} if generation_step is None else {"generation_step": generation_step}
    recorder.capture(
        f"{prefix}.attention.q",
        query,
        dtype=plan.np_dtype,
        **step_kwargs,
    )
    recorder.capture(
        f"{prefix}.attention.k",
        key,
        dtype=plan.np_dtype,
        **step_kwargs,
    )
    recorder.capture(
        f"{prefix}.attention.v",
        value,
        dtype=plan.np_dtype,
        **step_kwargs,
    )
    recorder.capture(
        f"{prefix}.attention.pre_mask_scores",
        pre_mask_scores,
        dtype=plan.np_dtype,
        **step_kwargs,
    )
    recorder.capture(
        f"{prefix}.attention.post_mask_logits",
        post_mask_logits,
        dtype=plan.np_dtype,
        **step_kwargs,
    )
    recorder.capture(
        f"{prefix}.attention.attention_probs",
        attention_probs,
        dtype=plan.np_dtype,
        **step_kwargs,
    )
    recorder.capture(
        f"{prefix}.attention.attn_output_pre_o_proj",
        _flatten_attention_output(attn_output),
        dtype=plan.np_dtype,
        **step_kwargs,
    )
    recorder.capture(
        f"{prefix}.kv_cache.key",
        key,
        dtype=plan.np_dtype,
        **step_kwargs,
    )
    recorder.capture(
        f"{prefix}.kv_cache.value",
        value,
        dtype=plan.np_dtype,
        **step_kwargs,
    )


def _capture_expert_step_inputs(
    recorder: _PI05FullSiteRecorder,
    make_att_2d_masks_fn: Any,
    *,
    prefix_pad_masks: Any,
    x_t: Any,
    generation_step: int,
) -> None:
    import torch

    suffix_len = int(x_t.shape[1])
    batch_size = int(prefix_pad_masks.shape[0])
    prefix_len = int(prefix_pad_masks.shape[1])
    suffix_pad_masks = torch.ones(
        batch_size,
        suffix_len,
        dtype=torch.bool,
        device=x_t.device,
    )
    suffix_att_masks = torch.tensor(
        [1, *([0] * max(0, suffix_len - 1))],
        dtype=torch.bool,
        device=x_t.device,
    )[None, :].expand(batch_size, suffix_len)
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
        batch_size,
        suffix_len,
        prefix_len,
    )
    suffix_att_2d_masks = make_att_2d_masks_fn(suffix_pad_masks, suffix_att_masks)
    full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    recorder.capture(
        "pi05.expert.by_step.attention_mask",
        suffix_pad_masks,
        dtype=np.bool_,
        generation_step=generation_step,
    )
    recorder.capture(
        "pi05.expert.by_step.causal_mask",
        full_att_2d_masks,
        dtype=np.bool_,
        generation_step=generation_step,
    )
    recorder.capture(
        "pi05.expert.by_step.position_ids",
        position_ids,
        dtype=np.int64,
        generation_step=generation_step,
    )


def _flatten_attention_output(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) != 4:
        return value
    return value.reshape(shape[0], shape[1], shape[2] * shape[3]).contiguous()


def _rope_metadata_array(model: Any) -> np.ndarray:
    config = getattr(model, "config", None)
    rotary = getattr(model, "rotary_emb", None)
    values = [
        getattr(config, "head_dim", np.nan),
        getattr(config, "max_position_embeddings", np.nan),
        getattr(config, "rope_theta", np.nan),
        getattr(rotary, "base", np.nan),
    ]
    return np.asarray(
        [float(value) if value is not None else np.nan for value in values],
        dtype=np.float32,
    )


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
    write_trace_record(
        record,
        args.vlatrace_out_root / f"{buffer.trace_id}.vlatrace",
        overwrite=True,
        validate=True,
    )


def _timesteps_table(buffer: EpisodeBuffer, length: int) -> pd.DataFrame:
    policy_call_for_timestep = np.full(length, np.nan, dtype=np.float32)
    horizon_index = np.full(length, np.nan, dtype=np.float32)
    sorted_calls = sorted(buffer.calls, key=lambda call: call.env_timestep)
    for index, call in enumerate(sorted_calls):
        next_start = (
            sorted_calls[index + 1].env_timestep if index + 1 < len(sorted_calls) else length
        )
        end = min(length, next_start)
        for timestep in range(max(0, call.env_timestep), end):
            policy_call_for_timestep[timestep] = call.call_index
            horizon_index[timestep] = timestep - call.env_timestep
    done = _pad_bool(buffer.terminated, length)
    truncated = _pad_bool(buffer.truncated, length)
    if length and not done.any() and not truncated.any():
        done[-1] = True
    return pd.DataFrame(
        {
            "timestep": np.arange(length, dtype=np.int32),
            "reward": np.asarray(buffer.rewards, dtype=np.float32),
            "done": done,
            "truncated": truncated,
            "policy_call_index": policy_call_for_timestep,
            "horizon_index": horizon_index,
        }
    )


def _generation_steps_table(buffer: EpisodeBuffer) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for call in buffer.calls:
        steps = int(call.denoising_actions.shape[0])
        for generation_step in range(steps):
            records.append(
                {
                    "policy_call_index": int(call.call_index),
                    "generation_step": int(generation_step),
                    "process_kind": "flow_matching",
                    "scheduler_index": int(generation_step),
                }
            )
    return pd.DataFrame.from_records(records)


def _attach_token_metadata(call: CaptureCall, obs: dict[str, Any], buffer: EpisodeBuffer) -> None:
    language = _language_metadata_from_observation(obs)
    cameras = [{"camera_id": camera} for camera in _trace_cameras(buffer)]
    first_frame = buffer.frames[0] if buffer.frames else None
    image_size = (
        (int(first_frame.shape[0]), int(first_frame.shape[1])) if first_frame is not None else None
    )
    metadata = build_pi05_token_metadata(
        prompt=buffer.prompt,
        language=language,
        tokenizer=_pi05_language_tokenizer(),
        cameras=cameras,
        image_slots=call.prefix_image_slots or len(cameras),
        patches_per_image=max(1, call.prefix_patches_per_image or 1),
        image_size=image_size,
        image_preprocessing={
            "raw_shape": list(first_frame.shape) if first_frame is not None else None,
            "processed_shape": list(first_frame.shape) if first_frame is not None else None,
            "resize_mode": "lerobot_policy_preprocessor",
            "value_range": "policy_checkpoint",
        },
        action=call.final_action_chunk,
        action_normalization={
            "normalization_type": "checkpoint",
            "stats_ref": "policy_preprocessor",
            **_libero_action_space_metadata(
                action_dim=int(np.asarray(call.final_action_chunk).shape[-1])
            ),
        },
        policy_call_index=call.call_index,
        observation_timestep=call.env_timestep,
        env_timestep_start=call.env_timestep,
    )
    call.token_metadata = metadata
    call.policy_call_metadata = dict(metadata.policy_call_metadata)


@lru_cache(maxsize=1)
def _pi05_language_tokenizer() -> Any | None:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            "google/paligemma-3b-pt-224",
            local_files_only=True,
        )
    except Exception:
        return None


def _buffer_action_dim(buffer: EpisodeBuffer) -> int | None:
    for action in buffer.executed_actions:
        array = np.asarray(action)
        if array.ndim:
            return int(array.shape[-1])
    for call in buffer.calls:
        array = np.asarray(call.final_action_chunk)
        if array.ndim:
            return int(array.shape[-1])
    return None


def _libero_action_space_metadata(action_dim: int | None = None) -> dict[str, Any]:
    if action_dim != len(LIBERO_ACTION_DIM_NAMES):
        return {"action_dim": action_dim}
    return {
        "action_dim": action_dim,
        "action_names": list(LIBERO_ACTION_DIM_NAMES),
        "action_labels": list(LIBERO_ACTION_DIM_LABELS),
        "action_units": list(LIBERO_ACTION_DIM_UNITS),
        "controller": "robosuite OSC_POSE",
        "control_mode": "relative",
        "position_scale_m": 0.05,
        "rotation_scale_rad": 0.5,
        "orientation_parameterization": "rotation_vector",
    }


def _token_tables(buffer: EpisodeBuffer) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calls_with_metadata = [call for call in buffer.calls if call.token_metadata is not None]
    if calls_with_metadata:
        streams = pd.concat(
            [call.token_metadata.streams for call in calls_with_metadata if call.token_metadata],
            ignore_index=True,
        )
        if "stream_id" in streams:
            streams = streams.drop_duplicates(subset=["stream_id"], keep="first")
        token_spaces = []
        tokens = []
        for call in calls_with_metadata:
            if call.token_metadata is None:
                continue
            call_token_spaces = call.token_metadata.token_spaces.copy()
            call_tokens = call.token_metadata.tokens.copy()
            call_token_spaces["policy_call_index"] = call.call_index
            call_tokens["policy_call_index"] = call.call_index
            token_spaces.append(call_token_spaces)
            tokens.append(call_tokens)
        return (
            streams.reset_index(drop=True),
            pd.concat(token_spaces, ignore_index=True),
            pd.concat(tokens, ignore_index=True),
        )

    cameras = _trace_cameras(buffer)
    streams = [
        {"stream_id": "prefix", "name": "vlm_prefix", "modality": "multimodal"},
        {"stream_id": "language", "name": "language", "modality": "language"},
        {"stream_id": "action_suffix", "name": "action_suffix", "modality": "action"},
        {
            "stream_id": EXPERT_CONTEXT_STREAM_ID,
            "name": "expert_context",
            "modality": "multimodal_action",
        },
    ]
    streams.extend(
        {
            "stream_id": f"image_{camera}",
            "name": camera,
            "modality": "image",
            "camera_id": camera,
        }
        for camera in cameras
    )
    token_spaces: list[dict[str, Any]] = []
    tokens: list[dict[str, Any]] = []
    if buffer.calls:
        sample_call = buffer.calls[0]
        action_tokens = int(sample_call.final_action_chunk.shape[0])
        token_spaces.append(
            {
                "token_space_id": "pi05.action_suffix",
                "policy_call_index": -1,
                "segment": "action_expert",
                "stream_id": "action_suffix",
                "token_count": action_tokens,
            }
        )
        for token_index in range(action_tokens):
            tokens.append(
                {
                    "token_space_id": "pi05.action_suffix",
                    "token_index": token_index,
                    "modality": "action",
                    "token_type": "action_horizon",
                    "stream_id": "action_suffix",
                    "action_horizon_index": token_index,
                }
            )
        if sample_call.prefix_image_hidden is not None:
            patches_per_image = int(
                sample_call.prefix_patches_per_image
                or sample_call.prefix_image_hidden.shape[0] // max(1, len(cameras))
            )
            grid_height, grid_width = _patch_grid_shape(patches_per_image)
            token_spaces.append(
                {
                    "token_space_id": "pi05.prefix",
                    "policy_call_index": -1,
                    "segment": "vlm_prefix",
                    "stream_id": "prefix",
                    "token_count": int(sample_call.prefix_image_hidden.shape[0]),
                }
            )
            token_spaces.append(
                {
                    "token_space_id": EXPERT_CONTEXT_TOKEN_SPACE_ID,
                    "policy_call_index": -1,
                    "segment": "expert_context",
                    "stream_id": EXPERT_CONTEXT_STREAM_ID,
                    "token_count": int(sample_call.prefix_image_hidden.shape[0]) + action_tokens,
                    "metadata": json.dumps(
                        {
                            "kind": "composite",
                            "segments": ["pi05.prefix", "pi05.action_suffix"],
                        }
                    ),
                }
            )
            token_index = 0
            for camera in cameras:
                for patch in range(patches_per_image):
                    row = patch // max(1, grid_width)
                    col = patch % max(1, grid_width)
                    tokens.append(
                        {
                            "token_space_id": "pi05.prefix",
                            "token_index": token_index,
                            "modality": "image",
                            "token_type": "image_patch",
                            "stream_id": f"image_{camera}",
                            "camera_id": camera,
                            "patch_row": row,
                            "patch_col": col,
                        }
                    )
                    token_index += 1
    return (
        pd.DataFrame.from_records(streams),
        pd.DataFrame.from_records(token_spaces),
        pd.DataFrame.from_records(tokens),
    )


def _prompt_metadata_table(buffer: EpisodeBuffer) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for call in buffer.calls:
        metadata = call.policy_call_metadata.get("prompt_metadata")
        if not isinstance(metadata, dict):
            metadata = (
                call.token_metadata.policy_call_metadata.get("prompt_metadata")
                if call.token_metadata is not None
                else {}
            )
        if not isinstance(metadata, dict):
            metadata = {}
        records.append(
            {
                "policy_call_index": int(call.call_index),
                "prompt_id": "prompt.default",
                "raw_task": buffer.prompt,
                "cleaned_task": str(metadata.get("prompt") or buffer.prompt),
                "formatted_prompt": str(
                    metadata.get("formatted_prompt") or metadata.get("prompt") or ""
                ),
                "state_bin_count": metadata.get("state_bin_count"),
                "metadata": _json_dumps(metadata),
            }
        )
    return pd.DataFrame.from_records(records)


def _image_preprocessing_table(buffer: EpisodeBuffer) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for call in buffer.calls:
        metadata = call.policy_call_metadata.get("image_preprocessing_metadata")
        if not isinstance(metadata, dict) and call.token_metadata is not None:
            metadata = call.token_metadata.policy_call_metadata.get("image_preprocessing_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        cameras = _trace_cameras(buffer)
        for slot_index, camera_id in enumerate(cameras):
            records.append(
                {
                    "policy_call_index": int(call.call_index),
                    "slot_index": slot_index,
                    "feature_key": camera_id,
                    "camera_id": camera_id,
                    "present": True,
                    "raw_shape": _json_dumps(metadata.get("raw_shape")),
                    "processed_shape": _json_dumps(metadata.get("processed_shape")),
                    "resize_mode": str(metadata.get("resize_mode") or ""),
                    "value_range": str(metadata.get("value_range") or ""),
                    "metadata": _json_dumps(metadata),
                }
            )
    return pd.DataFrame.from_records(records)


def _action_normalization_table(buffer: EpisodeBuffer) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for call in buffer.calls:
        metadata = call.policy_call_metadata.get("action_normalization_metadata")
        if not isinstance(metadata, dict) and call.token_metadata is not None:
            metadata = call.token_metadata.policy_call_metadata.get("action_normalization_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        records.append(
            {
                "policy_call_index": int(call.call_index),
                "normalization_id": "lerobot.pi05.action",
                "mode": str(metadata.get("normalization_type") or "checkpoint"),
                "stats_ref": str(metadata.get("stats_ref") or ""),
                "action_dim_names": _json_dumps(metadata.get("action_names")),
                "normalized_action_array_ref": "action_chunks",
                "unnormalized_action_array_ref": "executed_actions",
                "metadata": _json_dumps(metadata),
            }
        )
    return pd.DataFrame.from_records(records)


def _evaluation_table(buffer: EpisodeBuffer, length: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    done = _pad_bool(buffer.terminated, length)
    truncated = _pad_bool(buffer.truncated, length)
    for timestep, reward in enumerate(buffer.rewards[:length]):
        info = buffer.infos[timestep] if timestep < len(buffer.infos) else {}
        is_success = _info_bool(info, "is_success")
        records.append(
            {
                "timestep": timestep,
                "metric_name": "reward",
                "metric_value": float(reward),
                "threshold": np.nan,
                "passed": bool(reward > 0.0),
                "source": "env.step",
                "metadata": _json_dumps(info),
            }
        )
        records.append(
            {
                "timestep": timestep,
                "metric_name": "done",
                "metric_value": float(done[timestep]),
                "threshold": 1.0,
                "passed": bool(done[timestep]),
                "source": "env.step",
                "metadata": _json_dumps(info),
            }
        )
        records.append(
            {
                "timestep": timestep,
                "metric_name": "truncated",
                "metric_value": float(truncated[timestep]),
                "threshold": 1.0,
                "passed": bool(truncated[timestep]),
                "source": "env.step",
                "metadata": _json_dumps(info),
            }
        )
        if is_success is not None:
            records.append(
                {
                    "timestep": timestep,
                    "metric_name": "success",
                    "metric_value": float(is_success),
                    "threshold": 1.0,
                    "passed": bool(is_success),
                    "source": "env.step.info",
                    "metadata": _json_dumps(info),
                }
            )
    return pd.DataFrame.from_records(records)


def _context_table(context: ContextCaptureResult, component: str) -> pd.DataFrame:
    availability = context.availability
    if availability.empty or "component" not in availability:
        return pd.DataFrame()
    return availability.loc[availability["component"].astype(str) == component].reset_index(
        drop=True
    )


def _scene_state_table(context: ContextCaptureResult) -> pd.DataFrame:
    frames = []
    if "objects" in context.tables and not context.tables["objects"].empty:
        frame = context.tables["objects"].copy()
        frame["context_kind"] = "object"
        frames.append(frame)
    if "episode_context" in context.tables and not context.tables["episode_context"].empty:
        frame = context.tables["episode_context"].copy()
        frame["context_kind"] = "episode"
        frames.append(frame)
    object_availability = _context_table(context, "object")
    if not object_availability.empty:
        object_availability["context_kind"] = "availability"
        frames.append(object_availability)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _camera_state_table(context: ContextCaptureResult) -> pd.DataFrame:
    if "cameras" in context.tables:
        return context.tables["cameras"].copy()
    return _context_table(context, "camera")


def _context_availability_fields(
    context: ContextCaptureResult,
) -> tuple[list[str], list[dict[str, Any]]]:
    availability = context.availability
    if availability.empty:
        return [], []
    captured: list[str] = []
    unavailable: list[dict[str, Any]] = []
    for row in availability.to_dict("records"):
        field = f"{row.get('component')}.{row.get('field')}"
        if bool(row.get("available")):
            captured.append(field)
        else:
            unavailable.append(
                {
                    "field": field,
                    "reason": str(row.get("reason") or ""),
                }
            )
    return captured, unavailable


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
                "executed_actions",
                "action_chunks",
                "generation_actions",
                "frames.main" if buffer.frames else None,
                "frames.wrist" if buffer.wrist_frames else None,
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
    return sites


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


def _episode_arrays(buffer: EpisodeBuffer, length: int) -> dict[str, ArraySpec]:
    arrays: dict[str, ArraySpec] = {
        "executed_actions": ArraySpec(
            _pad_time(buffer.executed_actions, length),
            ["timestep", "action_dim"],
        )
    }
    if buffer.frames:
        arrays["frames.main"] = ArraySpec(
            np.stack(buffer.frames),
            ["timestep", "height", "width", "rgb"],
        )
    if buffer.wrist_frames:
        arrays["frames.wrist"] = ArraySpec(
            np.stack(buffer.wrist_frames),
            ["timestep", "height", "width", "rgb"],
        )
    if buffer.calls:
        arrays["action_chunks"] = ArraySpec(
            _stack_call_arrays(buffer.calls, "final_action_chunk"),
            ["policy_call", "horizon", "action_dim"],
        )
        arrays["generation_actions"] = ArraySpec(
            _stack_call_arrays(buffer.calls, "denoising_actions"),
            ["policy_call", "generation_step", "horizon", "action_dim"],
        )
        velocities = _stack_optional_calls(buffer.calls, "denoise_velocities")
        if velocities is not None:
            arrays["generation_velocities"] = ArraySpec(
                velocities.astype(np.float32),
                ["policy_call", "generation_step", "horizon", "action_dim"],
            )
    return arrays


def _model_arrays(
    buffer: EpisodeBuffer,
    plan: CapturePlan,
) -> list[ModelSiteSpec]:
    if canonical_profile(plan.profile) == "rollout" or not buffer.calls:
        return []
    specs: list[ModelSiteSpec] = []
    image_hidden = _stack_optional_calls(buffer.calls, "prefix_image_hidden")
    if image_hidden is not None:
        patches_per_image = next(
            (
                int(call.prefix_patches_per_image)
                for call in buffer.calls
                if call.prefix_patches_per_image
            ),
            image_hidden.shape[1] // max(1, len(_trace_cameras(buffer))),
        )
        image_slots = next(
            (int(call.prefix_image_slots) for call in buffer.calls if call.prefix_image_slots),
            image_hidden.shape[1] // max(1, patches_per_image),
        )
        grid_height, grid_width = _patch_grid_shape(patches_per_image)
        specs.append(
            ModelSiteSpec(
                name="pi05.vlm.prefix.image_hidden_tokens",
                array=image_hidden,
                axes=["policy_call", "token", "channel"],
                module="pi05.vlm.prefix",
                tensor_type="hidden_tokens",
                token_kind="image",
                family="representation",
                role="image_prefix_hidden_tokens",
                segment="vlm_prefix",
                token_space_id="pi05.prefix",
                metadata={
                    "camera_order": _trace_cameras(buffer),
                    "patches_per_image": patches_per_image,
                    "grid_height": grid_height,
                    "grid_width": grid_width,
                    "image_slots": image_slots,
                },
                capture_family="representation",
                view_kind="features",
                capture_role="primary",
                default_view=True,
            )
        )

    for layer in sorted(set(plan.vlm_layers) | set(plan.expert_layers)):
        vlm_hidden = _stack_layer_calls(buffer.calls, "vlm_hidden_by_layer", layer)
        if vlm_hidden is not None:
            vlm_axes = ["policy_call", "channel"]
            if plan.vlm_hidden == "tokens":
                vlm_axes = ["policy_call", "token", "channel"]
            specs.append(
                ModelSiteSpec(
                    name=f"pi05.vlm.layers.{layer}.prefix.hidden_{plan.vlm_hidden}",
                    array=vlm_hidden,
                    axes=vlm_axes,
                    module=f"pi05.vlm.layers.{layer}",
                    layer=layer,
                    tensor_type=f"hidden_{plan.vlm_hidden}",
                    token_kind="prefix",
                    family="representation",
                    role="hidden_state",
                    segment="vlm_prefix",
                    token_space_id="pi05.prefix",
                    capture_family="representation",
                    view_kind="features",
                    capture_role="primary",
                    default_view=True,
                )
            )

        vlm_attention = _stack_layer_calls(buffer.calls, "vlm_attention_by_layer", layer)
        if vlm_attention is not None:
            specs.append(
                _attention_spec(
                    family="vlm",
                    layer=layer,
                    array=vlm_attention,
                    resolution=plan.vlm_attention,
                    by_step=False,
                    token_kind="prefix",
                    segment="vlm_prefix",
                )
            )
        vlm_kv_key = _stack_layer_calls(buffer.calls, "vlm_kv_key_by_layer", layer)
        if vlm_kv_key is not None:
            specs.append(
                ModelSiteSpec(
                    name=f"pi05.vlm.layers.{layer}.kv_cache.key",
                    array=vlm_kv_key,
                    axes=["policy_call", "kv_head", "cached_token", "head_channel"],
                    module=f"pi05.vlm.layers.{layer}.attention",
                    layer=layer,
                    tensor_type="kv_cache",
                    token_kind="prefix",
                    family="cache",
                    role="kv_cache_key",
                    segment="vlm_prefix",
                    materialization="raw",
                    exactness="exact",
                    token_space_id="pi05.prefix",
                    metadata={
                        "capture_scope": "mechanistic_bridge",
                        "included_in_profile": plan.profile,
                    },
                    capture_family="cache",
                    view_kind="cache",
                    capture_role="primary",
                    default_view=False,
                )
            )
        vlm_kv_value = _stack_layer_calls(buffer.calls, "vlm_kv_value_by_layer", layer)
        if vlm_kv_value is not None:
            specs.append(
                ModelSiteSpec(
                    name=f"pi05.vlm.layers.{layer}.kv_cache.value",
                    array=vlm_kv_value,
                    axes=["policy_call", "kv_head", "cached_token", "head_channel"],
                    module=f"pi05.vlm.layers.{layer}.attention",
                    layer=layer,
                    tensor_type="kv_cache",
                    token_kind="prefix",
                    family="cache",
                    role="kv_cache_value",
                    segment="vlm_prefix",
                    materialization="raw",
                    exactness="exact",
                    token_space_id="pi05.prefix",
                    metadata={
                        "capture_scope": "mechanistic_bridge",
                        "included_in_profile": plan.profile,
                    },
                    capture_family="cache",
                    view_kind="cache",
                    capture_role="primary",
                    default_view=False,
                )
            )

        expert_hidden = _stack_layer_calls(buffer.calls, "expert_hidden_by_layer", layer)
        if expert_hidden is not None:
            expert_axes = ["policy_call", "generation_step", "channel"]
            if plan.expert_hidden == "tokens":
                expert_axes = ["policy_call", "generation_step", "token", "channel"]
            specs.append(
                ModelSiteSpec(
                    name=f"pi05.expert.layers.{layer}.by_step.hidden_{plan.expert_hidden}",
                    array=expert_hidden,
                    axes=expert_axes,
                    module=f"pi05.expert.layers.{layer}",
                    layer=layer,
                    tensor_type=f"hidden_{plan.expert_hidden}",
                    token_kind="action",
                    family="representation",
                    role="hidden_state",
                    segment="action_expert",
                    token_space_id="pi05.action_suffix",
                    capture_family="representation",
                    view_kind="features",
                    capture_role="primary",
                    default_view=True,
                )
            )

        expert_attention = _stack_layer_calls(buffer.calls, "expert_attention_by_layer", layer)
        if expert_attention is not None:
            specs.append(
                _attention_spec(
                    family="expert",
                    layer=layer,
                    array=expert_attention,
                    resolution=plan.expert_attention,
                    by_step=True,
                    token_kind="action",
                    segment="action_expert",
                )
            )

    attention = _stack_optional_calls(buffer.calls, "attention_mass")
    if attention is not None and plan.expert_attention != "full":
        specs.append(
            ModelSiteSpec(
                name="pi05.expert.by_step.attention_key_mass",
                array=attention,
                axes=["policy_call", "generation_step", "key_token"],
                module="pi05.expert",
                tensor_type="attention",
                token_kind="action",
                family="derived",
                role="attention_key_mass_summary",
                segment="action_expert",
                materialization="summary",
                exactness="lossy_summary",
                metadata={
                    "attention_resolution": "key_mass",
                    "source": "final captured expert layer averaged over heads and queries",
                },
                capture_family="attention",
                view_kind="attention",
                capture_role="derived_summary",
                default_view=False,
                derived_from=tuple(
                    f"pi05.expert.layers.{layer}.by_step.attention" for layer in plan.expert_layers
                ),
                derivation="mean_over_heads_and_queries",
            )
        )
    generation_input_embeddings = _stack_optional_calls(buffer.calls, "generation_input_embeddings")
    if generation_input_embeddings is not None:
        specs.append(
            ModelSiteSpec(
                name="pi05.expert.by_step.input_embeddings",
                array=generation_input_embeddings,
                axes=["policy_call", "generation_step", "token", "channel"],
                module="pi05.expert",
                tensor_type="embedding",
                token_kind="action",
                family="embedding",
                role="input_embeddings",
                segment="action_expert",
                materialization="raw",
                exactness="exact",
                token_space_id="pi05.action_suffix",
                metadata={
                    "capture_scope": "mechanistic_bridge",
                    "included_in_profile": plan.profile,
                },
                capture_family="representation",
                view_kind="features",
                capture_role="primary",
                default_view=True,
            )
        )
    action_head_input = _stack_optional_calls(buffer.calls, "action_head_input")
    if action_head_input is not None:
        specs.append(
            ModelSiteSpec(
                name="pi05.action_head.input",
                array=action_head_input,
                axes=["policy_call", "generation_step", "token", "channel"],
                module="pi05.action_head",
                tensor_type="action_head",
                token_kind="action",
                family="action_head",
                role="action_head_input",
                segment="action_head",
                materialization="raw",
                exactness="exact",
                token_space_id="pi05.action_suffix",
                metadata={
                    "capture_scope": "mechanistic_bridge",
                    "included_in_profile": plan.profile,
                },
                capture_family="action_head",
                view_kind="action",
                capture_role="primary",
                default_view=True,
            )
        )
    action_head_output = _stack_optional_calls(buffer.calls, "action_head_output")
    if action_head_output is not None:
        specs.append(
            ModelSiteSpec(
                name="pi05.action_head.output",
                array=action_head_output,
                axes=["policy_call", "generation_step", "horizon", "action_dim"],
                module="pi05.action_head",
                tensor_type="action_head",
                token_kind="action",
                family="action_head",
                role="action_head_output",
                segment="action_head",
                materialization="raw",
                exactness="exact",
                token_space_id="pi05.action_suffix",
                metadata={
                    "capture_scope": "mechanistic_bridge",
                    "included_in_profile": plan.profile,
                },
                capture_family="action_head",
                view_kind="action",
                capture_role="primary",
                default_view=True,
            )
        )
    if plan.capture_internals_sites:
        existing_names = {spec.name for spec in specs}
        specs.extend(
            spec for spec in _full_model_site_specs(buffer, plan) if spec.name not in existing_names
        )
    return specs


def _full_model_site_specs(
    buffer: EpisodeBuffer,
    plan: CapturePlan,
) -> list[ModelSiteSpec]:
    profile = canonical_profile(plan.profile)
    declarations = {
        item.name: item
        for item in pi05_full_site_declarations(
            vlm_layers=plan.vlm_layers,
            expert_layers=plan.expert_layers,
        )
    }
    specs: list[ModelSiteSpec] = []
    for name, declaration in declarations.items():
        if profile == "internals_sampled" and not _is_selected_internal_site(declaration):
            continue
        stacked = _stack_full_site_calls(buffer.calls, name)
        if stacked is None:
            continue
        specs.append(
            declaration.spec(
                stacked,
                metadata={
                    "numeric_lossy": str(stacked.dtype) != "float32"
                    and np.issubdtype(stacked.dtype, np.floating),
                    "semantic_lossy": False,
                    "summary_lossy": False,
                },
            )
        )
    return specs


def _is_selected_internal_site(declaration: Any) -> bool:
    selected_roles = {
        "q",
        "k",
        "v",
        "attention_probs",
        "o_proj",
        "mlp_gate",
        "mlp_up",
        "mlp_intermediate",
        "mlp_down",
        "adarms_gate",
        "kv_cache_key",
        "kv_cache_value",
    }
    return str(declaration.role) in selected_roles


def _stack_full_site_calls(calls: list[CaptureCall], site_name: str) -> np.ndarray | None:
    sample = next(
        (
            call.full_site_arrays.get(site_name)
            for call in calls
            if call.full_site_arrays.get(site_name) is not None
        ),
        None,
    )
    if sample is None:
        return None
    sample_array = np.asarray(sample)
    out = _empty_step_array(len(calls), sample_array)
    for index, call in enumerate(calls):
        value = call.full_site_arrays.get(site_name)
        if value is not None:
            out[index] = np.asarray(value, dtype=sample_array.dtype)
    return out


def _attention_spec(
    *,
    family: str,
    layer: int,
    array: np.ndarray,
    resolution: str,
    by_step: bool,
    token_kind: str,
    segment: str,
) -> ModelSiteSpec:
    suffix = "attention" if resolution == "full" else "attention_key_mass"
    axes = ["policy_call"]
    if by_step:
        axes.append("generation_step")
    axes.append("head")
    if resolution == "full":
        axes.extend(["query_token", "key_token"])
    else:
        axes.append("key_token")
    return ModelSiteSpec(
        name=f"pi05.{family}.layers.{layer}.{'by_step.' if by_step else 'prefix.'}{suffix}",
        array=array,
        axes=axes,
        module=f"pi05.{family}.layers.{layer}",
        layer=layer,
        tensor_type="attention",
        token_kind=token_kind,
        family="attention",
        role="attention_probs" if resolution == "full" else "attention_key_mass_summary",
        segment=segment,
        materialization="raw" if resolution == "full" else "summary",
        exactness="exact" if resolution == "full" else "lossy_summary",
        metadata={"attention_resolution": resolution},
        token_space_id="pi05.prefix" if family == "vlm" else EXPERT_CONTEXT_TOKEN_SPACE_ID,
        query_token_space_id="pi05.prefix" if family == "vlm" else "pi05.action_suffix",
        key_token_space_id="pi05.prefix" if family == "vlm" else EXPERT_CONTEXT_TOKEN_SPACE_ID,
        capture_family="attention",
        view_kind="attention",
        capture_role="primary" if resolution == "full" else "derived_summary",
        default_view=True,
    )


def _stack_optional_calls(calls: list[CaptureCall], attr: str) -> np.ndarray | None:
    sample = next((getattr(call, attr) for call in calls if getattr(call, attr) is not None), None)
    if sample is None:
        return None
    sample_array = np.asarray(sample)
    out = np.full((len(calls), *sample_array.shape), np.nan, dtype=sample_array.dtype)
    for index, call in enumerate(calls):
        value = getattr(call, attr)
        if value is not None:
            out[index] = np.asarray(value, dtype=sample_array.dtype)
    return out


def _stack_layer_calls(calls: list[CaptureCall], attr: str, layer: int) -> np.ndarray | None:
    sample = next(
        (
            getattr(call, attr).get(layer)
            for call in calls
            if getattr(call, attr).get(layer) is not None
        ),
        None,
    )
    if sample is None:
        return None
    sample_array = np.asarray(sample)
    out = np.full((len(calls), *sample_array.shape), np.nan, dtype=sample_array.dtype)
    for index, call in enumerate(calls):
        value = getattr(call, attr).get(layer)
        if value is not None:
            out[index] = np.asarray(value, dtype=sample_array.dtype)
    return out


def _array_size_summary(
    episode_arrays: dict[str, ArraySpec],
    model_arrays: list[ModelSiteSpec],
) -> dict[str, Any]:
    episode_bytes = {
        name: int(np.asarray(spec.array).nbytes) for name, spec in episode_arrays.items()
    }
    model_bytes = {spec.name: int(np.asarray(spec.array).nbytes) for spec in model_arrays}
    total_episode = int(sum(episode_bytes.values()))
    total_model = int(sum(model_bytes.values()))
    return {
        "episode_bytes": total_episode,
        "model_bytes": total_model,
        "total_uncompressed_bytes": total_episode + total_model,
        "episode_arrays": episode_bytes,
        "model_arrays": model_bytes,
    }


def _stack_call_arrays(calls: list[CaptureCall], attr: str) -> np.ndarray:
    sample = np.asarray(getattr(calls[0], attr), dtype=np.float32)
    out = np.full((len(calls), *sample.shape), np.nan, dtype=np.float32)
    for index, call in enumerate(calls):
        value = getattr(call, attr)
        if value is not None:
            out[index] = np.asarray(value, dtype=np.float32)
    return out


def _scatter_calls(calls: list[CaptureCall], length: int, attr: str) -> np.ndarray:
    sample = np.asarray(getattr(calls[0], attr), dtype=np.float32)
    out = np.full((length, *sample.shape), np.nan, dtype=np.float32)
    for call in calls:
        value = getattr(call, attr)
        if value is None:
            continue
        out[call.env_timestep] = np.asarray(value, dtype=np.float32)
    return out


def _scatter_optional_calls(calls: list[CaptureCall], length: int, attr: str) -> np.ndarray | None:
    sample = next((getattr(call, attr) for call in calls if getattr(call, attr) is not None), None)
    if sample is None:
        return None
    out = np.full((length, *np.asarray(sample, dtype=np.float32).shape), np.nan, dtype=np.float32)
    for call in calls:
        value = getattr(call, attr)
        if value is not None:
            out[call.env_timestep] = np.asarray(value, dtype=np.float32)
    return out


def _trace_cameras(buffer: EpisodeBuffer) -> list[str]:
    cameras = []
    if buffer.frames:
        cameras.append("main")
    if buffer.wrist_frames:
        cameras.append("wrist")
    return cameras


def _patch_grid_shape(patches_per_image: int) -> tuple[int, int]:
    width = int(np.ceil(np.sqrt(max(1, patches_per_image))))
    while width > 1 and patches_per_image % width != 0:
        width -= 1
    height = patches_per_image // width
    return height, width


def _pad_time(values: list[np.ndarray], length: int) -> np.ndarray:
    if not values:
        return np.zeros((length, 0), dtype=np.float32)
    sample = np.asarray(values[0], dtype=np.float32)
    out = np.full((length, *sample.shape), np.nan, dtype=np.float32)
    for idx, value in enumerate(values[:length]):
        out[idx] = np.asarray(value, dtype=np.float32)
    return out


def _pad_bool(values: list[bool], length: int) -> np.ndarray:
    out = np.zeros(length, dtype=bool)
    for index, value in enumerate(values[:length]):
        out[index] = bool(value)
    return out


def _call_mask(length: int, calls: list[CaptureCall]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for call in calls:
        if call.env_timestep < length:
            mask[call.env_timestep] = True
    return mask


def _call_indices(length: int, calls: list[CaptureCall]) -> np.ndarray:
    indices = np.full(length, np.nan, dtype=np.float32)
    for call in calls:
        if call.env_timestep < length:
            indices[call.env_timestep] = call.call_index
    return indices


def _extract_frames(observation: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not isinstance(observation, dict):
        return None, None
    main = _find_image(observation, ("agentview", "image"))
    wrist = _find_image(observation, ("eye_in_hand", "image2", "wrist"))
    return main, wrist


def _find_image(observation: dict[str, Any], needles: tuple[str, ...]) -> np.ndarray | None:
    for key, value in observation.items():
        text = str(key).lower()
        if any(needle in text for needle in needles):
            image = _as_image(value)
            if image is not None:
                return image
    for value in observation.values():
        if isinstance(value, dict):
            image = _find_image(value, needles)
            if image is not None:
                return image
    return None


def _as_image(value: Any) -> np.ndarray | None:
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        return None
    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] > 3:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255 if array.max(initial=0) > 1.0 else 1.0)
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = array.astype(np.uint8)
    return np.ascontiguousarray(array)


def _task_prompt(task: Any) -> str:
    for attr in ("language", "description", "name"):
        value = getattr(task, attr, None)
        if value:
            return str(value)
    return str(task)


def _language_metadata_from_observation(obs: Mapping[str, Any]) -> Mapping[str, Any] | None:
    input_ids = _lookup_nested_value(
        obs,
        (
            "observation.language.tokens",
            "observation.language.input_ids",
            "language.tokens",
            "language.input_ids",
            "input_ids",
        ),
    )
    if input_ids is None:
        return None
    payload: dict[str, Any] = {"input_ids": input_ids}
    attention_mask = _lookup_nested_value(
        obs,
        (
            "observation.language.attention_mask",
            "language.attention_mask",
            "attention_mask",
        ),
    )
    if attention_mask is not None:
        payload["attention_mask"] = attention_mask
    special_mask = _lookup_nested_value(
        obs,
        (
            "observation.language.special_tokens_mask",
            "language.special_tokens_mask",
            "special_tokens_mask",
        ),
    )
    if special_mask is not None:
        payload["special_tokens_mask"] = special_mask
    return payload


def _lookup_nested_value(payload: Mapping[str, Any], candidates: tuple[str, ...]) -> Any | None:
    flat: dict[str, Any] = {}
    _flatten_mapping(payload, prefix="", out=flat)
    normalized = {key.lower(): value for key, value in flat.items()}
    for candidate in candidates:
        key = candidate.lower()
        if key in normalized:
            return normalized[key]
    for candidate in candidates:
        suffix = candidate.lower()
        for key, value in normalized.items():
            if key.endswith(suffix):
                return value
    return None


def _flatten_mapping(payload: Mapping[str, Any], *, prefix: str, out: dict[str, Any]) -> None:
    for key, value in payload.items():
        text = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            _flatten_mapping(value, prefix=text, out=out)
        else:
            out[text] = value


def _first_info(info: Any) -> Any:
    if isinstance(info, Mapping):
        out: dict[str, Any] = {}
        for key, value in info.items():
            if isinstance(value, Mapping):
                out[str(key)] = _first_info(value)
                continue
            out[str(key)] = _first_batch_value(value)
        return out
    if isinstance(info, (list, tuple)) and info:
        return info[0]
    return info


def _first_batch_value(value: Any) -> Any:
    if isinstance(value, (str, bytes)) or value is None:
        return value
    try:
        array = np.asarray(value)
    except Exception:
        return value
    if not hasattr(array, "ndim"):
        return array
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    if not hasattr(array, "ndim"):
        return array
    if array.ndim == 0:
        return array.item()
    return array


def _jsonable_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return {"value": _jsonable(value)}


def _json_dumps(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _info_bool(info: Mapping[str, Any], key: str) -> bool | None:
    if key in info:
        value = info[key]
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "success"}
        if value is None:
            return None
        return bool(value)
    for value in info.values():
        if isinstance(value, Mapping):
            found = _info_bool(value, key)
            if found is not None:
                return found
    return None


def _to_numpy(value: Any, *, dtype: np.dtype | str | None = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "to"):
        value = value.to("cpu")
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    return array.astype(np.dtype("float32") if dtype is None else np.dtype(dtype), copy=False)


if __name__ == "__main__":
    main()
