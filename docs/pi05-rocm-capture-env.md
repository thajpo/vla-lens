# PI0.5 ROCm Capture Environment

Status: operational environment contract for local PI0.5/LeRobot/LIBERO capture.

For the cross-hardware run-path overview, see
[hardware-run-paths.md](hardware-run-paths.md). This file keeps the ROCm-specific
details and known-good package versions.

## The Problem

There are two dependency worlds in this repo:

```text
1. Normal VLA Lens development
   - server
   - frontend API work
   - LeRobot v3 + overlay schema tests
   - pure Python analysis/probe code
   - pyproject.toml + uv.lock

2. PI0.5 ROCm capture on a Linux AMD GPU machine
   - ROCm PyTorch wheels
   - LeRobot PI0.5 policy code
   - OpenPI-patched Transformers
   - hf-libero
   - robosuite==1.4.0
```

Trying to force both into one `.venv` is brittle.

The normal project lock intentionally excludes simulator/capture packages. It
keeps the dashboard and dataset tooling on ordinary Python data dependencies
such as:

```text
pyarrow>=19.0,<25.0
pandas
imageio
zarr
```

LeRobot/LIBERO capture currently needs:

```text
robosuite==1.4.0
OpenPI transformers_replace patch
ROCm torch wheels
```

LeRobot package metadata may declare Torch constraints that do not match the
working ROCm capture stack. Install LeRobot without dependency resolution in
the dedicated capture venv; do not let it replace Torch.

```text
known-good ROCm capture stack, May 25, 2026:
  torch==2.12.0+rocm7.2
  torchvision==0.27.0+rocm7.2
  torchaudio==2.11.0+rocm7.2
  lerobot==0.4.4
  numpy>=2.0,<2.3
  pyarrow>=21.0,<25.0
  datasets==4.8.5
  opencv-python-headless==4.12.0.88
  rerun-sdk==0.26.2
  hf-libero==0.1.3
  robosuite==1.4.0
```

Do not infer the current environment from this document alone. Verify it with:

```bash
scripts/check_pi05_rocm_env.sh
```

So a casual command like this is dangerous:

```bash
uv add lerobot
uv run vla-pi05-capture ...
```

It can either rewrite the lock/venv around incompatible capture dependencies or pull non-ROCm Torch wheels.

## Rule

Use separate environments.

```text
Normal repo/dev/test/server:
  .venv
  uv run ...

PI0.5 ROCm capture:
  .venv-pi05-rocm
  scripts/pi05_capture_rocm.sh ...
  scripts/pi05_batch_capture_rocm.sh ...
```

Do not install LeRobot into the normal `.venv`.

Do not run PI0.5 capture with plain `uv run vla-pi05-capture`.

## Testing Split

Normal tests run in the repo/dev environment:

```bash
uv run pytest
uv run ruff check scripts src tests
cd frontend && npm run build
```

These tests should cover:

```text
schema contracts
capture profile planning
model-site declarations
trace validation
server API payloads
frontend type/build checks
pure analysis/probe code
saved-trace behavior
```

They should not require Torch, LeRobot, GPU/ROCm, or LIBERO unless a specific test is intentionally marked as model-execution/capture integration.

Capture integration checks run in the ROCm capture environment:

```bash
scripts/check_pi05_rocm_env.sh
scripts/pi05_capture_rocm.sh --episodes 1 ...
scripts/pi05_batch_capture_rocm.sh --config configs/... --run
```

Use this path for tests or smokes that:

```text
import LeRobot PI0.5
load PI0.5 weights
touch ROCm/GPU
run LIBERO
write a real captured LeRobot root plus `vla_lens/` overlay
benchmark capture runtime or storage
```

This split is intentional. A normal profile/schema test that accidentally needs Torch is usually coupled too tightly and should be refactored.

## Setup

Create or refresh the capture environment:

```bash
scripts/setup_pi05_rocm_env.sh
```

The script creates `.venv-pi05-rocm`, installs VLA Lens editable without using the normal lock as the source of truth for capture, installs ROCm Torch wheels, installs LeRobot without dependency resolution, installs the required non-torch runtime packages, applies the OpenPI Transformers replacement patch, and verifies the critical imports.

The capture environment intentionally follows LeRobot 0.4.4's newer data stack
for packages such as NumPy, PyArrow, datasets, OpenCV, and Rerun. The normal
dashboard/test environment can stay on the versions pinned in `uv.lock`; the
project metadata is broad enough for both.

`scripts/check_pi05_rocm_env.sh` asserts the explicit capture package matrix
instead of using `uv pip check`. That is intentional: the working accelerator
stack uses LeRobot without dependency resolution because LeRobot's package
metadata currently excludes the newer ROCm/CUDA torchvision wheels used by the
known-good runtime.

The ROCm Torch wheel is large, so the first setup can take a while and may download several GiB. The wrapper scripts run `scripts/check_pi05_rocm_env.sh` before capture, so a missing or half-built `.venv-pi05-rocm` should fail loudly instead of starting a broken capture.

Expected checks:

```text
torch contains +rocm
transformers.models.siglip.check passes
lerobot.policies.pi05.modeling_pi05 imports
libero.libero.envs.OffScreenRenderEnv imports
robosuite is 1.4.0
the expected NumPy/PyArrow/datasets/OpenCV/Rerun package range is present
```

## Capture Commands

Check the environment explicitly:

```bash
scripts/check_pi05_rocm_env.sh
```

Single capture:

```bash
scripts/pi05_capture_rocm.sh \
  --episodes 1 \
  --start-seed 1002 \
  --benchmark libero_object \
  --task-id 0 \
  --capture-profile mechanistic_sampled \
  --vlatrace-out-root "/path/to/vla-lens/pi05-smoke"
```

Batch capture:

```bash
scripts/pi05_batch_capture_rocm.sh \
  --config configs/pi05_light_5_test.yaml \
  --run
```

## Emergency Workaround

If a venv is already correctly configured and you need to use uv only as a command runner, this can avoid sync:

```bash
uv run --no-sync vla-pi05-capture ...
```

This is not the recommended workflow. The recommended workflow is the dedicated `.venv-pi05-rocm` wrapper.

## Known Tradeoff

This is not a permanent packaging solution. It is the safest near-term operational contract.

Future cleanup should consider one of:

```text
dedicated capture lock file
uv dependency group with ROCm torch sources and explicit conflicts
container image for capture
small environment bootstrap tool with checksums for the OpenPI patch
```
