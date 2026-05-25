# Hardware Run Paths

Status: active operational guidance.

Last updated: May 25, 2026.

## Mental Model

VLA Lens has two runtime lanes:

```text
Normal repo/dev/test/dashboard lane:
  .venv
  uv run ...

PI0.5/LeRobot/LIBERO capture lane:
  .venv-pi05-rocm
  .venv-pi05-cuda
  .venv-pi05-mps
  scripts/pi05_* wrappers
```

Keep these separate. Torch wheels are hardware-bound: ROCm, CUDA, and Apple
MPS are not interchangeable installs. Mixing them in one virtualenv makes a
working capture machine hard to reproduce.

## Portable Reviewer Path

Run the portable repo checks:

```bash
scripts/check_vla_lens.sh
```

Start a synthetic demo dataset, backend, and React workbench:

```bash
scripts/run_vla_lens_demo.sh
```

Open:

```text
http://127.0.0.1:5173/
```

This path does not need Torch, LeRobot, LIBERO, a simulator, or a GPU.

Run the dashboard through Docker:

```bash
scripts/docker_dashboard.sh
scripts/docker_dashboard.sh runs/pi05-light-5-test
```

Open:

```text
http://127.0.0.1:8080/
```

See [docker.md](docker.md) for the dashboard container and planned capture
container split.

Run the same built dashboard locally without Docker:

```bash
scripts/view_vla_lens.sh runs/pi05-light-5-test
```

## PI0.5 Capture Setup

Use one setup command per hardware family:

```bash
# AMD ROCm on Linux
scripts/setup_pi05_rocm_env.sh

# NVIDIA CUDA on Linux
scripts/setup_pi05_cuda_env.sh

# Apple Silicon MPS on macOS
scripts/setup_pi05_mps_env.sh
```

The generic form is also available:

```bash
scripts/setup_pi05_env.sh --backend rocm
scripts/setup_pi05_env.sh --backend cuda
scripts/setup_pi05_env.sh --backend mps
scripts/setup_pi05_env.sh --backend auto
```

Each setup command installs the package shell, PI0.5/LIBERO runtime, OpenPI
Transformers replacement patch, `robosuite==1.4.0`, and the backend-specific
Torch stack.

## Capture Commands

Run a batch through the matching wrapper:

```bash
# AMD ROCm
scripts/pi05_batch_capture_rocm.sh \
  --config configs/pi05_light_5_test.yaml \
  --output-root /path/to/pi05-light-5-test \
  --run

# NVIDIA CUDA
scripts/pi05_batch_capture_cuda.sh \
  --config configs/pi05_light_5_test.yaml \
  --output-root /path/to/pi05-light-5-test \
  --run

# Apple Silicon MPS
scripts/pi05_batch_capture_mps.sh \
  --config configs/pi05_light_5_test.yaml \
  --output-root /path/to/pi05-light-5-test \
  --run
```

The generic form is:

```bash
scripts/pi05_batch_capture.sh --backend cuda --config configs/pi05_light_5_test.yaml --run
scripts/pi05_capture.sh --backend cuda --episodes 1 ...
```

Linux CUDA/ROCm Docker capture is also available:

```bash
scripts/docker_pi05_cuda.sh --config configs/pi05_light_5_test.yaml --run
scripts/docker_pi05_rocm.sh --config configs/pi05_light_5_test.yaml --run
```

For high-volume cloud or workstation capture, point output at the disk or
mounted storage volume you actually want to keep:

```bash
scripts/docker_pi05_rocm.sh \
  --config configs/pi05_light_5_test.yaml \
  --output-root /mnt/nvme/vla-lens/pi05-light-5-test \
  --run
```

Absolute Docker output paths are interpreted as host paths. The wrapper mounts
that directory into the container and rewrites the internal command so the
captured LeRobot dataset roots land where the user asked.

The wrappers force batch-generated capture commands to use the selected
virtualenv, device, and dtype. Older YAML files can still describe the dataset
while the wrapper owns the machine-specific runtime choice.

## Backend Defaults

| Backend | Virtualenv | Torch install default | Capture device | Policy dtype |
| --- | --- | --- | --- | --- |
| `rocm` | `.venv-pi05-rocm` | `https://download.pytorch.org/whl/rocm7.2` | `cuda` | `bfloat16` |
| `cuda` | `.venv-pi05-cuda` | `https://download.pytorch.org/whl/cu128` | `cuda` | `bfloat16` |
| `mps` | `.venv-pi05-mps` | PyPI Torch wheels | `mps` | `float32` |
| `cpu` | `.venv-pi05-cpu` | PyPI Torch wheels | `cpu` | `float32` |

ROCm uses the `cuda` device string because PyTorch exposes ROCm through the
CUDA-compatible API surface.

Override defaults when needed:

```bash
PI05_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu126 scripts/setup_pi05_cuda_env.sh
PI05_ROCM_INDEX_URL=https://download.pytorch.org/whl/rocm7.2 scripts/setup_pi05_rocm_env.sh
VLA_LENS_CAPTURE_DTYPE=float32 scripts/pi05_batch_capture_cuda.sh --config ... --run
```

## Checks

Check one capture runtime:

```bash
scripts/check_pi05_env.sh --backend rocm
scripts/check_pi05_env.sh --backend cuda
scripts/check_pi05_env.sh --backend mps
```

If you need to check imports without requiring visible GPU hardware:

```bash
PI05_STRICT_DEVICE_CHECK=0 scripts/check_pi05_env.sh --backend cuda
```

Use non-strict mode for diagnostics only, not for claiming a capture machine is
ready.

## CI Boundary

GitHub Actions validates the portable lane:

```text
lint: Python Ruff plus frontend ESLint
test: Python pytest
frontend-build: TypeScript and Vite production build
```

CI does not claim ROCm, CUDA, or MPS capture readiness because hosted GitHub
runners do not provide the same simulator/GPU stack. Hardware capture readiness
is proven by the matching setup/check/capture wrapper on the target machine.

## Apple Silicon Caveat

The MPS path is intentionally explicit, but PI0.5/LIBERO capture still needs
local validation on macOS. Torch/MPS can be installed on Apple Silicon, but the
full stack also includes LeRobot, MuJoCo/robosuite, LIBERO, and offscreen
rendering. Treat `scripts/setup_pi05_mps_env.sh` as the first target to test and
repair on that machine, not as a proven parity claim yet.
