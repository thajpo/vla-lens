# Hardware Run Paths

Status: active operational guidance.

Last updated: July 14, 2026.

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

See [docker.md](docker.md) for the dashboard container and capture
container split.

Run the same built dashboard locally without Docker:

```bash
scripts/view_vla_lens.sh runs/pi05-light-5-test
```

The dashboard input can be a single LeRobot dataset root or a top-level capture
output containing many nested LeRobot roots.

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

## Replay-Gated Intervention Smoke

Interventions use the same capture-specific runtime split. The first supported
path is intentionally narrow: reconstruct one saved policy-call observation,
inject its captured initial flow noise, measure repeated no-op drift, and only
then allow a synthetic action-head hook smoke. The synthetic direction checks
hook wiring; it is saved with `claim_eligible: false` and is not scientific
evidence.

Validation status: the CLI, wrapper, replay gate, hook restoration, and evidence
recording pass the normal runtime-free test suite. A real ROCm/CUDA/MPS replay
using a newly captured exact-noise trace has not yet been completed. Treat this
section as an experimental validation procedure, not a known-good hardware
workflow.

Every PI0.5 profile now stores exact float32 `flow_initial_noise` as replay
provenance. The current hook smoke additionally needs a capture that declares
`pi05.action_head.input`; `mechanistic_sampled` is the normal profile for that
purpose. With the current PI0.5 internal shape of `50 x 32`, exact noise adds
6,400 bytes (6.25 KiB) per policy call before Zarr metadata and compression.

Use a request shaped like this, replacing the trace and call with a real
mechanistic PI0.5 capture that declares `pi05.action_head.input`:

```json
{
  "runtime_adapter": "pi05",
  "target": {
    "kind": "manual",
    "model_family": "pi05",
    "model_site": "pi05.action_head.input",
    "token_space": "pi05.action_suffix"
  },
  "baseline": {
    "context": {"trace_id": "TRACE_ID", "policy_call_index": 0}
  },
  "intervention": {
    "request": {
      "operator": {
        "operator": "add_direction",
        "strength": 0.01,
        "parameters": {
          "mode": "synthetic_hook_smoke",
          "dimension": 0,
          "control_seed": 0
        }
      },
      "schedule": {
        "policy_calls": [0],
        "generation_steps": "all",
        "tokens": "action"
      },
      "outcome": {"kind": "action", "basis": ["raw"]},
      "controls": [{"kind": "random_direction"}]
    }
  }
}
```

Inspect metadata without loading the model, then measure replay without an
intervention:

```bash
scripts/pi05_intervene.sh --backend rocm /path/to/dataset \
  --request /path/to/request.json --dry-run

scripts/pi05_intervene.sh --backend rocm /path/to/dataset \
  --request /path/to/request.json --noop-repeats 3
```

Both commands persist a JSON report under
`DATASET_ROOT/vla_lens/intervention_reports/` unless `--output` is supplied.
After reading the measured drift, explicitly set tolerances and opt into the
non-claiming hook smoke:

```bash
scripts/pi05_intervene.sh --backend rocm /path/to/dataset \
  --request /path/to/request.json \
  --noop-repeats 3 \
  --run-intervention \
  --max-noop-l2 0.001 \
  --max-noop-max-abs 0.0001
```

The command exits with status 3 and does not invoke the hook when preflight,
exact-noise availability, or either replay tolerance fails. Choose tolerances
from the replay-only report; the example numbers above are illustrative, not
validated PI0.5 thresholds. Do not select wider tolerances merely to make the
gate pass; investigate replay drift first.

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
| `rocm` | `.venv-pi05-rocm` | `torch==2.12.0+rocm7.2`, `torchvision==0.27.0+rocm7.2`, `torchaudio==2.11.0+rocm7.2` from the ROCm 7.2 index | `cuda` | `bfloat16` |
| `cuda` | `.venv-pi05-cuda` | `torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`, `torchaudio==2.11.0+cu128` from the CUDA 12.8 index | `cuda` | `bfloat16` |
| `mps` | `.venv-pi05-mps` | PyPI Torch wheels | `mps` | `float32` |
| `cpu` | `.venv-pi05-cpu` | PyPI Torch wheels | `cpu` | `float32` |

ROCm uses the `cuda` device string because PyTorch exposes ROCm through the
CUDA-compatible API surface.

Override defaults when needed:

```bash
PI05_CUDA_TORCH_VERSION=2.11.0+cu128 scripts/setup_pi05_cuda_env.sh
PI05_ROCM_TORCH_VERSION=2.12.0+rocm7.2 scripts/setup_pi05_rocm_env.sh
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
