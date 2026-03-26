# OpenVLA Steering Experiment

This repository is a stepwise scaffold for a robotics interpretability project.

Current phase:

- `uv`-managed Python project
- minimal `robosuite` simulation setup
- experiment-facing `Stack` wrapper with deterministic seeded resets
- baseline rollout logging for scripted two-object picks

Immediate goal:

- stand up a deterministic arm + objects simulation loop
- make the scene measurable before adding model code or interpretation hooks

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
MUJOCO_GL=egl uv run python scripts/visualize_scene.py policy.target_object=cubeB
```

## Current Workflow

1. Visualize a scripted rollout

```bash
MUJOCO_GL=egl uv run python scripts/visualize_scene.py
```

2. Verify that the same seed reproduces the same scene

```bash
MUJOCO_GL=egl uv run python scripts/check_matched_scene.py
```

3. Run and log baseline scripted rollouts

```bash
MUJOCO_GL=egl uv run python scripts/run_stack_rollouts.py run.num_rollouts=4 run.save_video=false
```

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
