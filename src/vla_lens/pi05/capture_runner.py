"""PI0.5 capture runner helpers."""

from __future__ import annotations

import argparse
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from vla_lens.capture import (
    validate_lerobot_v3_dataset,
)
from vla_lens.dataset import build_dataset_index
from vla_lens.pi05.capture_predict import (
    _predict_action_chunk,
)
from vla_lens.pi05.capture_schema import (
    ATTENTION_RESOLUTIONS,
    HIDDEN_RESOLUTIONS,
    PROFILE_CHOICES,
    PROFILE_EXPERT_LAYERS,
    PROFILE_VLM_LAYERS,
    STORAGE_DTYPES,
    CapturePlan,
    EpisodeBuffer,
    PI05CaptureRuntime,
    canonical_profile,
)
from vla_lens.pi05.capture_tables import (
    _attach_token_metadata,
)
from vla_lens.pi05.capture_utils import (
    _extract_frames,
    _first_info,
    _jsonable_mapping,
    _task_prompt,
)
from vla_lens.pi05.capture_writer import (
    _episode_success,
    _write_episode,
)
from vla_lens.pi05.context_capture import (
    capture_camera_snapshot,
    capture_scene_snapshot,
)
from vla_lens.pi05.scene_mutation import apply_scene_mutation, scene_mutation_from_json
from vla_lens.traces import TraceDataset


def main(argv: list[str] | None = None) -> None:
    """Run PI0.5 capture from CLI arguments in a capture-specific environment."""
    args = parse_args(argv)
    if args.delete_existing and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    _run_capture(args)
    build_dataset_index(args.output_root, overwrite=True)
    dataset = TraceDataset.open(args.output_root)
    validation = validate_lerobot_v3_dataset(args.output_root)
    if not validation.valid:
        raise SystemExit(validation.to_dict())
    print(f"wrote {len(dataset.bundles)} LeRobot episode(s) to {args.output_root}")

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse package-native PI0.5 capture flags.

    Normal users should reach this through `scripts/pi05_capture.sh --backend
    ...`, not through plain `uv run`, so the heavy LeRobot/LIBERO/Torch stack
    stays isolated from the repo development environment.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="lerobot/pi05_libero_finetuned")
    parser.add_argument("--benchmark", default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument(
        "--seed-list",
        help=(
            "Comma-separated explicit episode seeds. When provided, this overrides "
            "--episodes/--start-seed iteration while preserving the same output naming."
        ),
    )
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
        help="Numeric dtype used for captured model internals in VLA Lens overlay arrays.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/pi05_golden"),
        help="Output LeRobot v3 dataset root.",
    )
    parser.add_argument(
        "--dataset-id",
        help="Dataset identifier stored in every trace manifest metadata.",
    )
    parser.add_argument(
        "--capture-design",
        choices=("single_trace", "paired_counterfactual"),
        default="single_trace",
        help="Higher-level capture design. This does not change tensor families.",
    )
    parser.add_argument(
        "--trace-variant",
        help="Stable variant suffix for paired traces, for example clean or corrupt.",
    )
    parser.add_argument("--counterfactual-group-id")
    parser.add_argument("--counterfactual-role")
    parser.add_argument("--counterfactual-type")
    parser.add_argument("--pair-index", type=int)
    parser.add_argument("--paired-trace-id")
    parser.add_argument(
        "--changed-fields",
        help="Comma-separated or JSON list of fields changed within a counterfactual group.",
    )
    parser.add_argument(
        "--matched-fields",
        help="Comma-separated or JSON list of fields intentionally matched within a group.",
    )
    parser.add_argument("--target-object-id")
    parser.add_argument("--counterfactual-target-object-id")
    parser.add_argument("--obs-size", type=int, default=256)
    parser.add_argument(
        "--layout-id",
        type=int,
        help="Explicit LIBERO init-state index used for capture and later replay.",
    )
    parser.add_argument(
        "--scene-mutation-json",
        help="Inline JSON or JSON path for a replayable scene mutation.",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--delete-existing", action="store_true")
    return parser.parse_args(argv)

def _run_capture(args: argparse.Namespace) -> None:
    plan = _resolve_capture_plan(args)
    print(f"capture plan: {plan.to_metadata()}", flush=True)
    runtime = load_pi05_capture_runtime(args)
    run_pi05_capture_task(args, runtime=runtime, plan=plan)

def load_pi05_capture_runtime(args: argparse.Namespace) -> PI05CaptureRuntime:
    """Load PI0.5 policy, preprocessing, and LIBERO factories.

    This function intentionally imports Torch, LeRobot, and LIBERO lazily so
    normal tests and dashboard work can import `vla_lens.pi05.capture` without
    pulling capture-only dependencies into `.venv`.
    """
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
    from lerobot.envs.utils import add_envs_task, preprocess_observation
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05 import PI05Policy
    from libero.libero.benchmark import get_benchmark

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
    return PI05CaptureRuntime(
        torch=torch,
        policy_cfg=policy_cfg,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        make_env=make_env,
        make_env_config=make_env_config,
        make_env_pre_post_processors=make_env_pre_post_processors,
        add_envs_task=add_envs_task,
        preprocess_observation=preprocess_observation,
        get_benchmark=get_benchmark,
    )

def run_pi05_capture_task(
    args: argparse.Namespace,
    *,
    runtime: PI05CaptureRuntime,
    plan: CapturePlan | None = None,
) -> None:
    """Roll out LIBERO episodes, capture PI0.5 internals, and write overlays."""
    plan = plan or _resolve_capture_plan(args)
    benchmark = runtime.get_benchmark(args.benchmark)(task_order_index=0)
    task = benchmark.get_task(args.task_id)
    task_name = str(task.name)

    env_cfg = runtime.make_env_config(
        "libero",
        task=args.benchmark,
        task_ids=[args.task_id],
        observation_height=args.obs_size,
        observation_width=args.obs_size,
        camera_name="agentview_image,robot0_eye_in_hand_image",
        control_mode="relative",
    )
    envs = runtime.make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[args.benchmark][args.task_id]
    env_preprocessor, env_postprocessor = runtime.make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=runtime.policy_cfg,
    )
    policy = runtime.policy

    try:
        for seed in _episode_seeds(args):
            trace_id = _trace_id_for_seed(args, seed)
            buffer = EpisodeBuffer(
                trace_id=trace_id,
                task_id=args.task_id,
                task_name=task_name,
                prompt=_task_prompt(task),
                seed=seed,
            )
            policy.reset()
            base_env = env.envs[0] if getattr(env, "envs", None) else None
            if args.layout_id is not None and base_env is not None:
                base_env.episode_index = int(args.layout_id)
                base_env.init_state_id = int(args.layout_id)
            observation, _ = env.reset(seed=[seed])
            scene_mutation = scene_mutation_from_json(args.scene_mutation_json)
            if scene_mutation is not None:
                observation, buffer.scene_mutation_report = apply_scene_mutation(
                    env, scene_mutation
                )
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
                    obs = runtime.preprocess_observation(observation)
                    obs = runtime.add_envs_task(env, obs)
                    obs = env_preprocessor(obs)
                    obs = runtime.preprocessor(obs)
                    call = _predict_action_chunk(policy, obs, len(buffer.calls), step, plan)
                    _attach_token_metadata(call, obs, buffer)
                    buffer.calls.append(call)
                    actions = runtime.postprocessor(
                        runtime.torch.as_tensor(call.final_action_chunk)
                    )
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

def namespace_for_capture_args(**overrides: Any) -> argparse.Namespace:
    """Build a test namespace using CLI defaults plus explicit overrides."""
    defaults = vars(parse_args([]))
    defaults.update(overrides)
    return SimpleNamespace(**defaults)

def _episode_seeds(args: argparse.Namespace) -> list[int]:
    seed_list = str(args.seed_list or "").strip()
    if not seed_list:
        return [int(args.start_seed) + index for index in range(int(args.episodes))]
    seeds = [int(item.strip()) for item in seed_list.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seed-list did not contain any seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seed-list contains duplicate seeds: {seed_list}")
    return seeds

def _trace_id_for_seed(args: argparse.Namespace, seed: int) -> str:
    base = f"pi05_{args.capture_profile}_{args.benchmark}_task{args.task_id}_seed{seed}"
    suffix = _trace_variant_suffix(
        str(args.trace_variant or "") or str(args.counterfactual_role or "")
    )
    return f"{base}_{suffix}" if suffix else base

def _trace_variant_suffix(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")

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
            "audit_sampled",
            "audit_windowed",
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
                "audit_sampled",
                "audit_windowed",
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
