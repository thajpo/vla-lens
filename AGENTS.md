# Agent Instructions

Use git liberally.

The user has coding experience, but is not a SWE. They want to become MLRE, enjoy theory, explanations, architecture design, and systems thinking. Their low-level computing background is still developing, so explain dependency/runtime issues concretely.

## UI Design Rule

For dashboard/workbench UI work, read `docs/research_ui_principles.md` before
making layout, copy, color, or component-organization changes. Treat it as the
repo-local taste contract: prefer screen-level coherence, human-readable
research/ML language, fewer redundant panels, and semantic color over reactive
one-widget fixes.

## Probe Training Rule

When asked to train a probe, read `docs/probe_hypothesis_guidance.md` and run
probe preflight before spending compute. Do not silently assume global mean
pooling. Show the relevant representation choices: learned layer mixing,
tokenwise analysis, object-conditioned decoding, and set decoding, along with
whether each is ready, data-ready but missing a specialized runner, or blocked
by missing capture data.

Use `RESEARCH.md` as the canonical question and findings log. Add or update the
relevant entry whenever an experiment is planned, completed, corrected, or
superseded. Campaign wrappers and implementation retries are history, not new
research questions.

## Critical Environment Rule

Do not run PI0.5/LeRobot/LIBERO capture through the normal repo `uv run` environment.

Use this split:

```text
Normal repo/dev/test/server work:
  .venv
  uv run ...

PI0.5 hardware capture work:
  .venv-pi05-rocm / .venv-pi05-cuda / .venv-pi05-mps
  scripts/pi05_capture.sh --backend rocm|cuda|mps ...
  scripts/pi05_batch_capture.sh --backend rocm|cuda|mps ...
```

Reason: PI0.5 capture currently needs a capture-specific stack:

```text
Hardware-specific PyTorch wheels
LeRobot
OpenPI-patched Transformers
hf-libero
robosuite==1.4.0
```

The normal repo lock intentionally does not own this stack. A casual `uv add
lerobot` can pull CUDA Torch wheels or resolver choices that overwrite the ROCm
stack, and mixing simulator dependencies into `.venv` makes the dashboard/test
environment harder to keep portable.

If the capture virtualenv does not exist or fails checks, run the matching setup:

```bash
scripts/setup_pi05_rocm_env.sh
scripts/setup_pi05_cuda_env.sh
scripts/setup_pi05_mps_env.sh
```

Then capture with:

```bash
scripts/pi05_capture_rocm.sh --capture-profile mechanistic_sampled ...
scripts/pi05_batch_capture_rocm.sh --config configs/... --run
```

or the generic wrappers:

```bash
scripts/pi05_capture.sh --backend cuda --capture-profile mechanistic_sampled ...
scripts/pi05_batch_capture.sh --backend mps --config configs/... --run
```

For Linux CUDA/ROCm containerized capture, use:

```bash
scripts/docker_pi05_cuda.sh --config configs/... --run
scripts/docker_pi05_rocm.sh --config configs/... --run
```

These Docker paths package capture runtimes; they do not replace normal local
development or the native virtualenv capture wrappers.

Only use `uv run --no-sync vla-pi05-capture ...` as an emergency workaround, not as normal workflow.

## Testing Split

Normal tests run in the repo/dev environment:

```bash
uv run pytest
uv run ruff check scripts src tests
```

These tests should cover schema, profile planning, LeRobot v3 + overlay
validation, server APIs, pure analysis code, and frontend-independent metadata
contracts. They should not require Torch, LeRobot, GPU/ROCm, or LIBERO unless
they are explicitly testing model execution.

Capture smokes run in the PI0.5 hardware capture environment:

```bash
scripts/check_pi05_env.sh --backend rocm
scripts/pi05_capture.sh --backend rocm --episodes 1 ...
scripts/pi05_batch_capture.sh --backend rocm --config configs/... --run
```

If a test imports LeRobot, loads PI0.5, touches ROCm/GPU, runs LIBERO, or writes a real capture trace, treat it as an explicit capture integration test, not part of the normal `uv run pytest` loop.
