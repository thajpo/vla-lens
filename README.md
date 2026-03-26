# OpenVLA Steering Experiment

This repository is a stepwise scaffold for a robotics interpretability project.

Current phase:

- `uv`-managed Python project
- minimal `robosuite` simulation setup
- first two-object scene with a scripted visible pick

Immediate goal:

- stand up a deterministic arm + objects simulation loop
- visualize the scene before adding model code or interpretation hooks

## Quick Start

```bash
uv sync
uv run python scripts/visualize_scene.py
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

You can switch the scripted target with:

```bash
MUJOCO_GL=egl uv run python scripts/visualize_scene.py policy.target_object=cubeB
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

The repo stays intentionally small until the first end-to-end simulation and visualization path is stable.
