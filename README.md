# OpenVLA Steering Experiment

This repository is a stepwise scaffold for a robotics interpretability project.

Research goal:

- determine whether a VLA internally represents which object it is going to pick before motor commitment
- build a probe-ready rollout pipeline for hidden-state decoding
- eventually test causal steering of that decoded target signal

Current phase:

- `uv`-managed Python project
- minimal `robosuite` simulation setup
- experiment-facing `Stack` wrapper with deterministic seeded resets
- baseline rollout logging for scripted two-object picks
- backend abstraction with `scripted_pick`, `openvla`, and `minivla`
- working OpenVLA action path into robosuite
- MiniVLA loader integration in progress via upstream Prismatic code

Immediate goal:

- finish validating a lightweight baseline VLA rollout path
- add lean per-step rollout logging needed for probe dataset construction
- avoid bloated instrumentation before the first probe-ready dataset exists

## Quick Start

```bash
uv sync
MUJOCO_GL=egl uv run python scripts/visualize_scene.py
```

If rendering fails because the local machine needs a different MuJoCo / OpenGL backend, set the backend explicitly before running:

```bash
MUJOCO_GL=egl uv run python scripts/visualize_scene.py
```

or:

```bash
MUJOCO_GL=glfw uv run python scripts/visualize_scene.py
```

The default config now uses `robosuite`'s `Stack` environment as a simple proxy for the target task:

- two colored cubes
- one scripted pick target
- saved offscreen debug video by default
- deterministic seeded resets for matched-scene checks
- structured rollout summaries written to Parquet

You can switch the scripted target with:

```bash
MUJOCO_GL=egl uv run python scripts/visualize_scene.py model.scripted_pick.target_object=cubeB
```

Backend selection now lives under `model.backend`.
Current supported values:

- `scripted_pick`
- `openvla`
- `minivla`

## Current Workflow

1. Visualize a scripted rollout

```bash
MUJOCO_GL=egl uv run python scripts/visualize_scene.py
```

Or explicitly:

```bash
MUJOCO_GL=egl uv run python scripts/visualize_scene.py model.backend=scripted_pick
```

2. Verify that the same seed reproduces the same scene

```bash
MUJOCO_GL=egl uv run python scripts/check_matched_scene.py
```

3. Run and log baseline scripted rollouts

```bash
MUJOCO_GL=egl uv run python scripts/run_stack_rollouts.py run.num_rollouts=4 run.save_video=false
```

## Backend Status

### OpenVLA

The `openvla` backend is now runnable in this environment.

What is in place:

- backend selection via `model.backend=openvla`
- camera-observation extraction from `{camera_name}_image`
- direct `(prompt, image) -> processor -> predict_action(...)` path
- explicit `unnorm_key` support, with `bridge_orig` currently configured
- clipping and 7D action validation before robosuite stepping

Current status:

- smoke tests produced valid 7D actions in the robosuite loop
- this is enough to treat OpenVLA as the current baseline VLA path

### MiniVLA

The `minivla` backend is wired, but it does not use the Hugging Face AutoClasses path.

Important note:

- Stanford MiniVLA checkpoints such as `Stanford-ILIAD/minivla-vq-bridge-prismatic` are not currently deployable through vanilla `AutoProcessor` / `AutoModelForVision2Seq`
- the repo now routes MiniVLA through the upstream `openvla-mini` Prismatic loader instead

Current status:

- backend selection via `model.backend=minivla`
- vendored upstream `openvla-mini` code under `third_party/openvla-mini`
- local dependency path updated to support the Prismatic loader
- MiniVLA smoke validation is in progress

## ROCm Setup

Use the ROCm PyTorch wheel path from the official PyTorch install docs. For Linux ROCm 6.4, the command is:

```bash
uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/rocm6.4
```

Then install the minimal OpenVLA inference stack referenced by the OpenVLA README:

```bash
uv pip install transformers timm tokenizers sentencepiece pillow accelerate
```

If you want to mirror the OpenVLA README more closely, its minimal path is based on Hugging Face `AutoProcessor` plus `AutoModelForVision2Seq` with `trust_remote_code=True`.

## VLA Smoke Paths

OpenVLA:

```bash
MUJOCO_GL=egl uv run python scripts/visualize_scene.py \
  model.backend=openvla \
  env.use_camera_obs=true \
  env.has_offscreen_renderer=true
```

MiniVLA:

```bash
MUJOCO_GL=egl PRISMATIC_DATA_ROOT=/tmp uv run python scripts/visualize_scene.py \
  model.backend=minivla \
  env.use_camera_obs=true \
  env.has_offscreen_renderer=true
```

For MiniVLA, the loader path is heavier because it uses the upstream Prismatic checkpoint format rather than a Hugging Face-exported inference checkpoint.

Rollout records are written to:

```text
artifacts/logs/stack_rollouts.parquet
```

Saved videos, when enabled, are written under:

```text
artifacts/videos/
```

## Repository Shape

```text
src/openvla_steering/
  model/
  env/
  interp/
  utils/
configs/
scripts/
artifacts/
```

## Key Files

- [scripts/visualize_scene.py](/home/j/Projects/OpenVLA_Patching_Experiment/scripts/visualize_scene.py): scripted debug rollout with optional video
- [scripts/check_matched_scene.py](/home/j/Projects/OpenVLA_Patching_Experiment/scripts/check_matched_scene.py): confirms deterministic same-seed resets
- [scripts/run_stack_rollouts.py](/home/j/Projects/OpenVLA_Patching_Experiment/scripts/run_stack_rollouts.py): baseline rollout runner and Parquet logger
- [src/openvla_steering/env/stack_task.py](/home/j/Projects/OpenVLA_Patching_Experiment/src/openvla_steering/env/stack_task.py): experiment-facing wrapper around robosuite `Stack`

The repo stays intentionally small until the first end-to-end simulation, logging, and matched-scene path is stable.

## Next Work

The next milestone is not "more model plumbing". It is lean probe-oriented data collection:

- save episode metadata plus per-step actions and kinematics
- keep activation capture optional and selective
- build the first probe dataset split by episode, not timestep
- only then add intervention hooks
