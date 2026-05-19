# Agent Instructions

Use git liberally.

The user has coding experience, but is not a SWE. They want to become MLRE, enjoy theory, explanations, architecture design, and systems thinking. Their low-level computing background is still developing, so explain dependency/runtime issues concretely.

## Critical Environment Rule

Do not run PI0.5/LeRobot/LIBERO capture through the normal repo `uv run` environment.

Use this split:

```text
Normal repo/dev/test/server work:
  .venv
  uv run ...

PI0.5 ROCm capture work:
  .venv-pi05-rocm
  scripts/pi05_capture_rocm.sh ...
  scripts/pi05_batch_capture_rocm.sh ...
```

Reason: PI0.5 capture currently needs a capture-specific stack:

```text
ROCm PyTorch wheels
LeRobot
OpenPI-patched Transformers
hf-libero
robosuite==1.4.0
```

The normal repo lock currently wants `robosuite>=1.5.2`, and `uv run` may sync that into `.venv`. That breaks LeRobot's LIBERO import path. A casual `uv add lerobot` can also pull CUDA Torch wheels and overwrite the ROCm stack.

If `.venv-pi05-rocm` does not exist or fails checks, run:

```bash
scripts/setup_pi05_rocm_env.sh
```

Then capture with:

```bash
scripts/pi05_capture_rocm.sh --capture-profile mechanistic_sampled ...
scripts/pi05_batch_capture_rocm.sh --config configs/... --run
```

Only use `uv run --no-sync vla-pi05-capture ...` as an emergency workaround, not as normal workflow.

## Testing Split

Normal tests run in the repo/dev environment:

```bash
uv run pytest
uv run ruff check scripts src tests
```

These tests should cover schema, profile planning, trace validation, server APIs, pure analysis code, and frontend-independent metadata contracts. They should not require Torch, LeRobot, GPU/ROCm, or LIBERO unless they are explicitly testing model execution.

Capture smokes run in the PI0.5 ROCm capture environment:

```bash
scripts/check_pi05_rocm_env.sh
scripts/pi05_capture_rocm.sh --episodes 1 ...
scripts/pi05_batch_capture_rocm.sh --config configs/... --run
```

If a test imports LeRobot, loads PI0.5, touches ROCm/GPU, runs LIBERO, or writes a real capture trace, treat it as an explicit capture integration test, not part of the normal `uv run pytest` loop.
