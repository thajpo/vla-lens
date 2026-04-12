#!/usr/bin/env python3
"""List and run selected LIBERO tasks with MiniVLA (prismatic family)."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import imageio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OPENVLA_MINI_ROOT = ROOT / "third_party" / "openvla-mini"
VQ_BET_ROOT = ROOT / "third_party" / "vq_bet_official"
if str(OPENVLA_MINI_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENVLA_MINI_ROOT))
if str(VQ_BET_ROOT) not in sys.path:
    sys.path.insert(0, str(VQ_BET_ROOT))

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ImportError as exc:
    raise SystemExit(
        "LIBERO is not installed. Run:\n"
        "  uv pip install third_party/LIBERO/"
    ) from exc

try:
    import torch
    from prismatic.models.load import load_vla
except ImportError as exc:
    raise SystemExit(
        "Prismatic VLA not available. Check third_party/openvla-mini is present."
    ) from exc


DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
ACTION_DIM = 7
DEFAULT_CHECKPOINT = "Stanford-ILIAD/minivla-vq-libero90-prismatic"
MAX_STEPS_BY_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def get_libero_image(obs: dict, resize_size: int, key: str = "agentview_image") -> np.ndarray:
    """
    Extracts, decodes, resizes, and processes an image from LIBERO obs.
    """
    img = obs[key]
    img = np.flipud(img)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((resize_size, resize_size), Image.Resampling.LANCZOS)
    return np.asarray(pil_img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Environment utilities
# ---------------------------------------------------------------------------

def get_libero_env(task, resolution: int = 224):
    task_description = task.language
    bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(0)
    return env, task_description


def get_libero_dummy_action() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to axis-angle representation."""
    denom = np.linalg.norm(quat[:3])
    if denom < 1e-6:
        return np.zeros(3)
    ax = quat[:3] / denom
    angle = 2.0 * math.asin(min(1.0, denom))
    return ax * angle


# ---------------------------------------------------------------------------
# Action utilities
# ---------------------------------------------------------------------------

def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """Map gripper from [0,1] → [-1,+1] and optionally binarize."""
    action = action.copy()
    action[..., -1] = 2.0 * action[..., -1] - 1.0
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    action = action.copy()
    action[..., -1] *= -1.0
    return action


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

def load_model(checkpoint: str, hf_token: str | None = None):
    from huggingface_hub import snapshot_download

    print(f"Loading VLA from checkpoint: {checkpoint}")

    # load_vla expects a local .pt path; use local dir or download from HF
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir() and (checkpoint_path / "config.json").exists():
        snapshot_dir = checkpoint_path
    else:
        snapshot_dir = Path(
            snapshot_download(
                repo_id=checkpoint,
                allow_patterns=["config.json", "dataset_statistics.json", "checkpoints/*.pt"],
                token=hf_token,
            )
        )
    ckpt_paths = sorted((snapshot_dir / "checkpoints").glob("step-*.pt"))
    if not ckpt_paths:
        raise RuntimeError(f"No checkpoint .pt found under {snapshot_dir / 'checkpoints'}")
    ckpt_path = ckpt_paths[-1]
    print(f"Using checkpoint: {ckpt_path}")

    vla = load_vla(str(ckpt_path), hf_token=hf_token, load_for_training=False, image_sequence_len=1)
    half_dtype = vla.llm_backbone.half_precision_dtype
    vla.vision_backbone.to(dtype=half_dtype)
    vla.llm_backbone.to(dtype=half_dtype)
    vla.to(dtype=half_dtype)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    vla.to(device)
    if hasattr(vla, 'action_tokenizer') and hasattr(vla.action_tokenizer, 'vq_vae'):
        # VqVae is a custom class, not an nn.Module, so we move its components
        vla.action_tokenizer.vq_vae.encoder = vla.action_tokenizer.vq_vae.encoder.to(device)
        vla.action_tokenizer.vq_vae.decoder = vla.action_tokenizer.vq_vae.decoder.to(device)
        vla.action_tokenizer.vq_vae.vq_layer = vla.action_tokenizer.vq_vae.vq_layer.to(device)
        vla.action_tokenizer.vq_vae.device = device
        vla.action_tokenizer.device = device
    print(f"Model loaded on {device}")
    return vla


def get_action(
    model,
    image_history: list[np.ndarray],
    task_label: str,
    unnorm_key: str,
    center_crop: bool = True,
    step: int = -1,
) -> np.ndarray:
    import hashlib
    raw = image_history[-1]
    raw_hash = hashlib.md5(raw.tobytes()).hexdigest()[:8]
    print(f"[DBG] get_action step={step}  raw_img shape={raw.shape} dtype={raw.dtype}  "
          f"mean={raw.mean():.2f}  std={raw.std():.2f}  hash={raw_hash}")

    images = [Image.fromarray(img).convert("RGB") for img in image_history]
    if center_crop:
        images = [_apply_center_crop(img) for img in images]
    processed = images[0] if len(images) == 1 else images
    action = model.predict_action(processed, task_label, unnorm_key=unnorm_key)
    assert action.shape == (ACTION_DIM,), f"Unexpected action shape {action.shape}"
    print(f"[DBG] final action: {action}")
    return action


def _apply_center_crop(image: Image.Image, crop_scale: float = 0.9) -> Image.Image:
    w, h = image.size
    t_h = int(math.sqrt(crop_scale) * h)
    t_w = int(math.sqrt(crop_scale) * w)
    top = (h - t_h) // 2
    left = (w - t_w) // 2
    cropped = image.crop((left, top, left + t_w, top + t_h))
    return cropped.resize((w, h), Image.Resampling.BILINEAR)


# ---------------------------------------------------------------------------
# Video saving
# ---------------------------------------------------------------------------

def save_rollout_video(
    images: list[np.ndarray],
    path: Path,
    fps: int = 30,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps)
    for img in images:
        writer.append_data(img)
    writer.close()
    print(f"Saved video to {path}")
    return str(path)


# ---------------------------------------------------------------------------
# Suite / task selection
# ---------------------------------------------------------------------------

def get_suite(task_suite_name: str):
    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        raise SystemExit(f"Unknown suite {task_suite_name!r}. Available: {sorted(benchmark_dict)}")
    return benchmark_dict[task_suite_name]()


def list_tasks(suite) -> list[dict]:
    return [
        {
            "task_id": i,
            "name": getattr(suite.get_task(i), "name", ""),
            "language": getattr(suite.get_task(i), "language", ""),
        }
        for i in range(suite.n_tasks)
    ]


def select_tasks(args: argparse.Namespace, suite) -> list[tuple[int, object]]:
    selected = []
    for i in range(suite.n_tasks):
        task = suite.get_task(i)
        if args.task_id is not None and i != args.task_id:
            continue
        if args.task_name_substring and args.task_name_substring.lower() not in getattr(task, "name", "").lower():
            continue
        if args.task_language_substring and args.task_language_substring.lower() not in getattr(task, "language", "").lower():
            continue
        selected.append((i, task))
    return selected


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout_task(
    args: argparse.Namespace,
    model,
    task_id: int,
    task,
    initial_states: list,
    unnorm_key: str,
) -> dict:
    env, task_description = get_libero_env(task, resolution=224)
    max_steps = args.max_steps or MAX_STEPS_BY_SUITE.get(args.task_suite_name, 400)

    log_dir = ROOT / args.local_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{DATE_TIME}--suite={args.task_suite_name}--task={task_id}.log"

    successes = 0
    n_trials = min(args.num_trials_per_task, len(initial_states))

    with log_path.open("w") as log_file:
        log_file.write(f"task_id={task_id}\nlanguage={task_description}\ncheckpoint={args.pretrained_checkpoint}\n")
        log_file.flush()

        for ep_idx in range(n_trials):
            env.reset()
            obs = env.set_init_state(initial_states[ep_idx])

            replay_images: list[np.ndarray] = []
            done = False
            reward = 0.0

            for t in range(max_steps + args.num_steps_wait):
                if t < args.num_steps_wait:
                    obs, reward, done, _ = env.step(get_libero_dummy_action())
                    continue

                img = get_libero_image(obs, resize_size=224)
                replay_images.append(img)

                history = replay_images[-args.obs_history:]
                if len(history) < args.obs_history:
                    history = [replay_images[-1]] * (args.obs_history - len(history)) + history

                action = get_action(model, history, task_description, unnorm_key, center_crop=args.center_crop, step=t)
                action = normalize_gripper_action(action, binarize=True)
                action = invert_gripper_action(action)
                if t < 15 or t % 50 == 0:
                    print(f"[DBG] env_action step={t}: {np.round(action, 4).tolist()}")

                obs, reward, done, _ = env.step(action.tolist())

                if t % 25 == 0 or done:
                    msg = f"ep={ep_idx} step={t} reward={reward:.4f} done={done}"
                    print(msg)
                    log_file.write(msg + "\n")
                    log_file.flush()

                if done:
                    successes += 1
                    break

            summary = {
                "suite": args.task_suite_name,
                "task_id": task_id,
                "task_language": task_description,
                "episode_idx": ep_idx,
                "success": bool(done),
                "final_reward": float(reward),
                "checkpoint": args.pretrained_checkpoint,
            }
            print(json.dumps(summary))

            if args.save_videos and replay_images:
                video_dir = ROOT / "artifacts" / "videos" / "libero"
                video_path = video_dir / f"{DATE_TIME}--task={task_id}--ep={ep_idx}--success={done}.mp4"
                summary["video_path"] = save_rollout_video(replay_images, video_path)

            summary_path = ROOT / args.summary_jsonl
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with summary_path.open("a") as f:
                f.write(json.dumps(summary) + "\n")

    env.close()
    return {"task_id": task_id, "episodes": n_trials, "successes": successes, "log_path": str(log_path)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained-checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--hf-token", default=None, help="HF token string or path to .hf_token file")
    p.add_argument("--task-suite-name", default="libero_90")
    p.add_argument("--task-id", type=int)
    p.add_argument("--task-name-substring")
    p.add_argument("--task-language-substring")
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--obs-history", type=int, default=1)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--num-trials-per-task", type=int, default=1)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--local-log-dir", default="artifacts/logs/libero")
    p.add_argument("--summary-jsonl", default="artifacts/logs/libero_runs.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed_everywhere(args.seed)

    suite = get_suite(args.task_suite_name)

    if args.list_tasks:
        for record in list_tasks(suite):
            print(json.dumps(record))
        return

    selected = select_tasks(args, suite)
    if not selected:
        raise SystemExit("No tasks matched. Use --list-tasks to inspect the suite.")

    hf_token = None
    if args.hf_token:
        p = Path(args.hf_token)
        hf_token = p.read_text().strip() if p.exists() else args.hf_token

    unnorm_key = args.task_suite_name

    print(f"Selected {len(selected)} task(s) from {args.task_suite_name}")
    for task_id, task in selected:
        print(json.dumps({"task_id": task_id, "language": getattr(task, "language", "")}))

    model = load_model(args.pretrained_checkpoint, hf_token=hf_token)

    for task_id, task in selected:
        initial_states = suite.get_task_init_states(task_id)
        result = rollout_task(args, model, task_id, task, initial_states, unnorm_key)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
