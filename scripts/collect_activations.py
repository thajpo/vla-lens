#!/usr/bin/env python3
"""
Collect hidden-state activations and step-level metadata from LIBERO rollouts.

This script runs LIBERO tasks in activation-capture mode. For each rollout step
it saves:
  - Hidden-state tensors from specified LLM layers at specified token positions
  - Step-level metadata: ee_pos, mug positions, contacted_object, VQ codes, decoded action
  - Episode-level metadata: task, seed, success

Output is three JSONL files (written incrementally) + tensor directory per run.
At run end, JSONL files are converted to parquet for querying:
  artifacts/activations/{run_id}/episodes.jsonl   -> episodes.parquet
  artifacts/activations/{run_id}/steps.jsonl      -> steps.parquet
  artifacts/activations/{run_id}/activations.jsonl -> activations.parquet
  artifacts/activations/{run_id}/tensors/ep{N}_l{layer_idx}_{pos}.pt

Crash-safe: each episode is written atomically to JSONL after completion.
Resume: on restart, completed (task_id, episode_idx) pairs are skipped.

Run the pilot (20 eps per task) before scaling to 200:
  python scripts/collect_activations.py \\
      --task-ids 71 72 --num-trials-per-task 20 \\
      --layers 14 16 18 --token-positions color_word final \\
      --run-id pilot_20ep
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OPENVLA_MINI_ROOT = ROOT / "third_party" / "openvla-mini"
VQ_BET_ROOT = ROOT / "third_party" / "vq_bet_official"
for p in [str(OPENVLA_MINI_ROOT), str(VQ_BET_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
except ImportError as exc:
    raise SystemExit("LIBERO not installed. Run: uv pip install third_party/LIBERO/") from exc

try:
    import torch
    from prismatic.models.load import load_vla
except ImportError as exc:
    raise SystemExit("Prismatic VLA not available.") from exc

from openvla_steering.interp.hooks import (
    HookManager,
    discover_modules,
    llm_layer_names,
    resolve_token_position_in_full_seq,
)

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEFAULT_CHECKPOINT = "Stanford-ILIAD/minivla-vq-libero90-prismatic"


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def get_libero_image(obs: dict, resize_size: int = 224) -> np.ndarray:
    img = obs["agentview_image"]
    img = np.flipud(img)
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((resize_size, resize_size), Image.Resampling.LANCZOS)
    return np.asarray(pil_img, dtype=np.uint8)


def apply_center_crop(image: Image.Image, crop_scale: float = 0.9) -> Image.Image:
    import math
    w, h = image.size
    t_h = int(math.sqrt(crop_scale) * h)
    t_w = int(math.sqrt(crop_scale) * w)
    top = (h - t_h) // 2
    left = (w - t_w) // 2
    cropped = image.crop((left, top, left + t_w, top + t_h))
    return cropped.resize((w, h), Image.Resampling.BILINEAR)


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
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
# Environment metadata helpers
# ---------------------------------------------------------------------------

def get_ee_pos(env) -> list[float] | None:
    """Return end-effector position from LIBERO env, or None if unavailable."""
    try:
        sim = env.sim
        for site_name in ["gripper0_grip_site", "robot0_eef_site", "eef_site"]:
            try:
                site_id = sim.model.site_name2id(site_name)
                pos = sim.data.site_xpos[site_id]
                return pos.tolist()
            except Exception:
                continue
        return None
    except Exception:
        return None


def get_body_pos(env, body_name: str) -> list[float] | None:
    """Return position of a named MuJoCo body, or None if not found."""
    try:
        sim = env.sim
        body_id = sim.model.body_name2id(body_name)
        pos = sim.data.body_xpos[body_id]
        return pos.tolist()
    except Exception:
        return None


def get_contacted_object(env, target_body_names: list[str]) -> str | None:
    """
    Check if the gripper is in contact with any of the named bodies.
    Returns the first matching body name, or None.
    """
    try:
        sim = env.sim
        gripper_geom_ids = set()
        for i in range(sim.model.ngeom):
            geom_body_id = sim.model.geom_bodyid[i]
            body_name = sim.model.body_id2name(geom_body_id)
            if "gripper" in body_name.lower():
                gripper_geom_ids.add(i)

        target_body_ids = {}
        for name in target_body_names:
            try:
                bid = sim.model.body_name2id(name)
                target_body_ids[bid] = name
            except Exception:
                pass

        for contact in sim.data.contact[:sim.data.ncon]:
            geom1, geom2 = contact.geom1, contact.geom2
            for gid in [geom1, geom2]:
                other_gid = geom2 if gid == geom1 else geom1
                if gid in gripper_geom_ids:
                    other_body_id = sim.model.geom_bodyid[other_gid]
                    if other_body_id in target_body_ids:
                        return target_body_ids[other_body_id]
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint: str, hf_token: str | None = None):
    from huggingface_hub import snapshot_download

    print(f"Loading VLA from {checkpoint}")
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

    if hasattr(vla, "action_tokenizer") and hasattr(vla.action_tokenizer, "vq_vae"):
        vla.action_tokenizer.vq_vae.encoder = vla.action_tokenizer.vq_vae.encoder.to(device)
        vla.action_tokenizer.vq_vae.decoder = vla.action_tokenizer.vq_vae.decoder.to(device)
        vla.action_tokenizer.vq_vae.vq_layer = vla.action_tokenizer.vq_vae.vq_layer.to(device)
        vla.action_tokenizer.vq_vae.device = device
        vla.action_tokenizer.device = device

    print(f"Model loaded on {device}")
    return vla, device


# ---------------------------------------------------------------------------
# VQ code capture
# ---------------------------------------------------------------------------

@contextmanager
def capture_vq_output(action_tokenizer):
    """
    Intercept decode_token_ids_to_actions to capture VQ codes and pre-norm action.

    VQ codes are derived from action token IDs:
        codes = tokenizer_len - 1 - action_token_ids  (clipped to [0, n_bins-1])

    Yields a dict that is populated with 'vq_codes' and 'decoded_action_vq'
    after the wrapped predict_action call returns.
    """
    captured: dict = {}
    original = action_tokenizer.decode_token_ids_to_actions

    def patched(action_token_ids):
        codes = action_tokenizer.tokenizer_len - 1 - action_token_ids
        codes = np.clip(codes, 0, action_tokenizer.n_bins - 1)
        captured["vq_codes"] = codes.tolist()
        result = original(action_token_ids)
        captured["decoded_action_vq"] = result.tolist()  # raw VQ decoder output, pre-normalization
        return result

    action_tokenizer.decode_token_ids_to_actions = patched
    try:
        yield captured
    finally:
        action_tokenizer.decode_token_ids_to_actions = original


# ---------------------------------------------------------------------------
# Per-step activation + inference
# ---------------------------------------------------------------------------

def capture_input_ids_hook(model) -> tuple[dict, object]:
    """
    Register a one-shot hook on the LLM embedding layer to capture the actual
    text input_ids during the prefill forward pass.
    """
    captured: dict = {}

    def hook(module, input, output):
        if "input_ids" in captured:
            return
        if len(input) > 0:
            ids = input[0]
            if hasattr(ids, "shape") and len(ids.shape) == 2 and ids.shape[1] > 1:
                captured["input_ids"] = ids[0].cpu().tolist()

    handle = model.llm_backbone.llm.model.embed_tokens.register_forward_hook(hook)
    return captured, handle


def step_with_capture(
    model,
    image_np: np.ndarray,
    instruction: str,
    unnorm_key: str,
    hook_manager: HookManager,
    layer_names: list[str],
    token_position_specs: list[str],
    tokenizer,
    center_crop: bool = False,
) -> tuple[np.ndarray, dict[str, dict[str, torch.Tensor]], list | None, list | None, dict[str, int]]:
    """
    Run model inference for one step while capturing activations and VQ codes.

    Returns:
        action:           (7,) numpy array (post-processed, ready for env.step)
        activations:      {layer_name: {pos_spec: tensor (hidden_dim,)}}
        vq_codes:         list[int] len 7, or None if capture failed
        decoded_action_vq: list[float] len 7 (raw VQ decoder output, pre-normalization), or None
        color_word_positions: {pos_spec: int} resolved token indices in full LLM sequence
    """
    pil_img = Image.fromarray(image_np).convert("RGB")
    if center_crop:
        pil_img = apply_center_crop(pil_img)

    captured_ids, ids_handle = capture_input_ids_hook(model)

    vq_codes = None
    decoded_action_vq = None

    with hook_manager:
        with torch.no_grad():
            if hasattr(model, "action_tokenizer"):
                with capture_vq_output(model.action_tokenizer) as vq_captured:
                    action_raw = model.predict_action(pil_img, instruction, unnorm_key=unnorm_key)
                vq_codes = vq_captured.get("vq_codes")
                decoded_action_vq = vq_captured.get("decoded_action_vq")
            else:
                action_raw = model.predict_action(pil_img, instruction, unnorm_key=unnorm_key)

        ids_handle.remove()
        full_input_ids = captured_ids.get("input_ids", [])

        # Resolve token positions against the full LLM sequence (vision + text)
        pos_indices: dict[str, int] = {}
        for spec in token_position_specs:
            try:
                idx = resolve_token_position_in_full_seq(
                    spec, full_input_ids, instruction, tokenizer
                )
            except ValueError as e:
                print(f"[WARN] Position resolution failed for spec={spec!r}: {e}")
                idx = -1
            pos_indices[spec] = idx

        activations: dict[str, dict[str, torch.Tensor]] = {}
        for layer in layer_names:
            full_tensor = hook_manager.get(layer)  # (full_seq_len, hidden_dim)
            activations[layer] = {}
            for spec, idx in pos_indices.items():
                if idx == -1 or (0 <= idx < full_tensor.shape[0]):
                    activations[layer][spec] = full_tensor[idx].clone()
                else:
                    print(f"[WARN] Position idx={idx} out of range for tensor shape {full_tensor.shape}, falling back to -1")
                    activations[layer][spec] = full_tensor[-1].clone()

    action = normalize_gripper_action(action_raw, binarize=True)
    action = invert_gripper_action(action)
    return action, activations, vq_codes, decoded_action_vq, pos_indices


# ---------------------------------------------------------------------------
# JSONL I/O helpers
# ---------------------------------------------------------------------------

def append_jsonl(path: Path, records: list[dict]) -> None:
    """Append records to a JSONL file, one JSON object per line."""
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_completed_episodes(out_dir: Path) -> set[tuple[int, int]]:
    """Return set of (task_id, episode_idx) already written to episodes.jsonl."""
    path = out_dir / "episodes.jsonl"
    if not path.exists():
        return set()
    done: set[tuple[int, int]] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add((r["task_id"], r["episode_idx"]))
            except Exception:
                pass
    return done



def jsonl_to_parquet(jsonl_path: Path, parquet_path: Path) -> None:
    """Convert a JSONL file to parquet. No-op if the JSONL doesn't exist or is empty."""
    import pandas as pd
    if not jsonl_path.exists():
        return
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    if records:
        pd.DataFrame(records).to_parquet(parquet_path, index=False)
        print(f"  {jsonl_path.name} -> {parquet_path.name} ({len(records)} rows)")


# ---------------------------------------------------------------------------
# Main rollout loop
# ---------------------------------------------------------------------------

def collect_task(
    args: argparse.Namespace,
    model,
    tokenizer,
    task_id: int,
    task,
    initial_states: list,
    layer_names: list[str],
    out_dir: Path,
    run_id: str,
    episode_offset: int = 0,
) -> int:
    """
    Roll out one task for num_trials episodes, capturing activations.

    Writes episodes, steps, and activation records to JSONL incrementally.
    Skips episodes already present in episodes.jsonl (resume support).
    Per-episode try/except ensures crashes don't lose prior work.

    Returns:
        Number of episodes successfully completed (including previously completed ones
        that were skipped).
    """
    task_description = task.language
    bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=224,
        camera_widths=224,
    )
    env.seed(0)

    max_steps = args.max_steps or 400
    num_steps_wait = args.num_steps_wait

    TARGET_BODIES = {
        71: ("red_coffee_mug_1", "porcelain_mug_1"),
        72: ("porcelain_mug_1", "red_coffee_mug_1"),
    }
    target_body, other_body = TARGET_BODIES.get(task_id, (None, None))

    hook_manager = HookManager(model, layer_names)
    tensor_dir = out_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    completed = load_completed_episodes(out_dir)
    n_trials = min(args.num_trials_per_task, len(initial_states))

    for ep_idx in range(n_trials):
        if (task_id, ep_idx) in completed:
            print(f"  [skip] task={task_id} ep={ep_idx} already done")
            continue

        global_ep_id = episode_offset + ep_idx
        print(f"\n[task={task_id} ep={ep_idx}] {task_description}")

        try:
            env.reset()
            obs = env.set_init_state(initial_states[ep_idx])

            done = False
            final_reward = 0.0
            step_count = 0
            ep_activations: dict[str, dict[str, list[torch.Tensor]]] = {
                layer: {spec: [] for spec in args.token_positions}
                for layer in layer_names
            }
            step_records_for_ep: list[dict] = []
            # Track color_word_pos from first step for episode-level validation
            first_color_word_pos: int | None = None

            for t in range(max_steps + num_steps_wait):
                if t < num_steps_wait:
                    obs, final_reward, done, _ = env.step([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
                    continue

                image_np = get_libero_image(obs)

                action, activations, vq_codes, decoded_action_vq, pos_indices = step_with_capture(
                    model, image_np, task_description,
                    unnorm_key=args.task_suite_name,
                    hook_manager=hook_manager,
                    layer_names=layer_names,
                    token_position_specs=args.token_positions,
                    tokenizer=tokenizer,
                    center_crop=args.center_crop,
                )

                for layer in layer_names:
                    for spec in args.token_positions:
                        ep_activations[layer][spec].append(activations[layer][spec])

                color_word_pos = pos_indices.get("color_word", None)
                if first_color_word_pos is None and color_word_pos is not None and color_word_pos != -1:
                    first_color_word_pos = color_word_pos

                ee_pos = get_ee_pos(env)
                target_pos = get_body_pos(env, target_body) if target_body else None
                other_pos = get_body_pos(env, other_body) if other_body else None
                contacted = get_contacted_object(
                    env, [target_body, other_body] if target_body else []
                )

                obs, step_reward, done, _ = env.step(action.tolist())
                final_reward = step_reward
                step_count += 1

                step_rec = {
                    "run_id": run_id,
                    "task_id": task_id,
                    "episode_id": global_ep_id,
                    "step": step_count - 1,  # 0-indexed
                    "reward": float(step_reward),
                    "ee_pos_x": ee_pos[0] if ee_pos else None,
                    "ee_pos_y": ee_pos[1] if ee_pos else None,
                    "ee_pos_z": ee_pos[2] if ee_pos else None,
                    "target_mug_pos_x": target_pos[0] if target_pos else None,
                    "target_mug_pos_y": target_pos[1] if target_pos else None,
                    "target_mug_pos_z": target_pos[2] if target_pos else None,
                    "other_mug_pos_x": other_pos[0] if other_pos else None,
                    "other_mug_pos_y": other_pos[1] if other_pos else None,
                    "other_mug_pos_z": other_pos[2] if other_pos else None,
                    "contacted_object": contacted,
                    "gripper_state": float(action[-1]),
                    "vq_codes": vq_codes,
                    "decoded_action_vq": decoded_action_vq,
                    "decoded_action_env": action.tolist(),
                    "color_word_pos": color_word_pos if color_word_pos != -1 else None,
                }
                step_records_for_ep.append(step_rec)

                if done:
                    break

            # Save activation tensors to disk
            layer_idx_map = {name: i for i, name in enumerate(layer_names)}
            activation_records_for_ep: list[dict] = []
            for layer in layer_names:
                l_idx = layer_idx_map[layer]
                for spec in args.token_positions:
                    frames = ep_activations[layer][spec]
                    if not frames:
                        continue
                    tensor = torch.stack(frames, dim=0).to(torch.float16)
                    fname = f"ep{global_ep_id:04d}_l{l_idx:02d}_{spec}.pt"
                    fpath = tensor_dir / fname
                    torch.save(tensor, fpath)
                    activation_records_for_ep.append({
                        "run_id": run_id,
                        "task_id": task_id,
                        "episode_id": global_ep_id,
                        "layer": layer,
                        "layer_idx": l_idx,
                        "token_position": spec,
                        "n_steps": len(frames),
                        "hidden_dim": tensor.shape[1],
                        "tensor_path": str(fpath.relative_to(ROOT)),
                    })

            episode_rec = {
                "run_id": run_id,
                "task_id": task_id,
                "task_language": task_description,
                "episode_id": global_ep_id,
                "episode_idx": ep_idx,
                "seed": args.seed,
                "success": bool(done),
                "n_steps": step_count,
                "checkpoint": args.pretrained_checkpoint,
                "color_word_pos": first_color_word_pos,
            }

            # Write atomically after all data is ready (tensors already on disk)
            append_jsonl(out_dir / "episodes.jsonl", [episode_rec])
            append_jsonl(out_dir / "steps.jsonl", step_records_for_ep)
            append_jsonl(out_dir / "activations.jsonl", activation_records_for_ep)

            print(f"  ep={ep_idx} steps={step_count} success={done} reward={final_reward:.3f} color_word_pos={first_color_word_pos}")

        except Exception as e:
            import traceback
            print(f"[ERROR] task={task_id} ep={ep_idx}: {e}")
            traceback.print_exc()
            try:
                env.reset()
            except Exception:
                pass
            continue  # do not write JSONL; this episode will be retried on next run

    env.close()
    n_completed_after = sum(
        1 for (tid, _) in load_completed_episodes(out_dir) if tid == task_id
    )
    return n_completed_after


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrained-checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--hf-token", default=None)
    p.add_argument("--task-suite-name", default="libero_90")
    p.add_argument("--task-ids", type=int, nargs="+", required=True,
                   help="Task IDs to collect (e.g. 71 72)")
    p.add_argument("--layers", type=int, nargs="+",
                   help="LLM layer indices to capture (e.g. 12 14 16 18 20 22). "
                        "Default: all 24 layers.")
    p.add_argument("--token-positions", nargs="+", default=["color_word", "final"],
                   choices=["color_word", "final", "eos"],
                   help="Token positions to capture activations at.")
    p.add_argument("--num-trials-per-task", type=int, default=20)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--run-id", required=True, help="Unique identifier for this collection run")
    p.add_argument("--output-dir", default="artifacts/activations")
    p.add_argument("--discover-modules", action="store_true",
                   help="Print all model module names and exit (for debugging).")
    p.add_argument("--discover-filter", default="",
                   help="Substring filter for --discover-modules output.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed_everywhere(args.seed)

    hf_token = None
    if args.hf_token:
        p = Path(args.hf_token)
        hf_token = p.read_text().strip() if p.exists() else args.hf_token

    model, device = load_model(args.pretrained_checkpoint, hf_token=hf_token)
    tokenizer = model.llm_backbone.tokenizer

    if args.discover_modules:
        print("\n=== Model module names ===")
        discover_modules(model, args.discover_filter)
        return

    if args.layers is not None:
        n_layers_total = len(model.llm_backbone.llm.model.layers)
        for l in args.layers:
            if l >= n_layers_total:
                raise SystemExit(
                    f"Layer index {l} out of range (model has {n_layers_total} layers)."
                )
        layer_names = [f"llm_backbone.llm.model.layers.{i}" for i in args.layers]
    else:
        n_layers_total = len(model.llm_backbone.llm.model.layers)
        layer_names = llm_layer_names(n_layers_total)

    print(f"Capturing {len(layer_names)} layer(s) at positions: {args.token_positions}")

    try:
        dummy = HookManager(model, layer_names)
    except KeyError as e:
        raise SystemExit(
            f"Module name validation failed:\n{e}\n"
            f"Run with --discover-modules to list available names."
        )

    out_dir = ROOT / args.output_dir / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Save run config (overwrite on resume — config should be identical)
    config = {
        "run_id": args.run_id,
        "checkpoint": args.pretrained_checkpoint,
        "task_suite_name": args.task_suite_name,
        "task_ids": args.task_ids,
        "layer_indices": args.layers,
        "layer_names": layer_names,
        "token_positions": args.token_positions,
        "num_trials_per_task": args.num_trials_per_task,
        "seed": args.seed,
        "date_time": DATE_TIME,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[args.task_suite_name]()

    total_eps = 0
    for task_idx, task_id in enumerate(args.task_ids):
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        print(f"\n=== task {task_id}: {task.language} ===")

        # episode_offset: count only episodes from tasks that precede this one in the
        # task list. Using total count would inflate the offset on resume because
        # partially-completed later tasks are already counted.
        prior_task_ids = set(args.task_ids[:task_idx])
        completed_all = load_completed_episodes(out_dir)
        episode_offset = sum(1 for (tid, _) in completed_all if tid in prior_task_ids)

        n_done = collect_task(
            args, model, tokenizer,
            task_id, task, initial_states,
            layer_names, out_dir, args.run_id,
            episode_offset=episode_offset,
        )
        total_eps += n_done

    # Convert JSONL -> parquet at the end
    print("\nFinalizing: converting JSONL -> parquet...")
    jsonl_to_parquet(out_dir / "episodes.jsonl", out_dir / "episodes.parquet")
    jsonl_to_parquet(out_dir / "steps.jsonl", out_dir / "steps.parquet")
    jsonl_to_parquet(out_dir / "activations.jsonl", out_dir / "activations.parquet")

    print(f"\nDone. Total episodes: {total_eps}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
