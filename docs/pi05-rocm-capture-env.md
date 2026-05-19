# PI0.5 ROCm Capture Environment

Status: operational environment contract for local PI0.5/LeRobot/LIBERO capture.

## The Problem

There are two dependency worlds in this repo:

```text
1. Normal VLA Lens development
   - server
   - frontend API work
   - trace schema tests
   - pure Python analysis/probe code
   - pyproject.toml + uv.lock

2. PI0.5 capture on this workstation
   - ROCm PyTorch wheels
   - LeRobot PI0.5 policy code
   - OpenPI-patched Transformers
   - hf-libero
   - robosuite==1.4.0
```

Trying to force both into one `.venv` is brittle.

The normal project lock currently includes:

```text
robosuite>=1.5.2,<2.0
pyarrow>=19.0,<20.0
```

LeRobot/LIBERO capture currently needs:

```text
robosuite==1.4.0
OpenPI transformers_replace patch
ROCm torch wheels
```

LeRobot metadata also declares:

```text
torch<2.11
torchvision<0.26
```

but this workstation intentionally uses:

```text
torch==2.11.0+rocm7.2
torchvision==0.26.0+rocm7.2
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
write a real captured .vlatrace
benchmark capture runtime or storage
```

This split is intentional. A normal profile/schema test that accidentally needs Torch is usually coupled too tightly and should be refactored.

## Setup

Create or refresh the capture environment:

```bash
scripts/setup_pi05_rocm_env.sh
```

The script creates `.venv-pi05-rocm`, installs VLA Lens editable without using the normal lock as the source of truth for capture, installs ROCm Torch wheels, installs LeRobot without dependency resolution, installs the required non-torch runtime packages, applies the OpenPI Transformers replacement patch, and verifies the critical imports.

The ROCm Torch wheel is large, so the first setup can take a while and may download several GiB. The wrapper scripts run `scripts/check_pi05_rocm_env.sh` before capture, so a missing or half-built `.venv-pi05-rocm` should fail loudly instead of starting a broken capture.

Expected checks:

```text
torch contains +rocm
transformers.models.siglip.check passes
lerobot.policies.pi05.modeling_pi05 imports
libero.libero.envs.OffScreenRenderEnv imports
robosuite is 1.4.0
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
  --vlatrace-out-root "/media/j/New Volume/vla-lens/pi05-smoke"
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
