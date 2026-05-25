# Docker Runtime Plan

Status: active dashboard and Linux capture container guidance.

Last updated: May 25, 2026.

## Position

VLA Lens is not Docker-only. The normal development path remains local:

```bash
uv run pytest
uv run ruff check scripts src tests
cd frontend && npm run dev
```

Docker is the reproducible packaging layer:

```text
dashboard container:
  portable visualizer, backend API, built React app

capture containers:
  Linux CUDA/ROCm PI0.5 inference and dataset writing

native venv:
  Apple Silicon MPS capture
```

The durable boundary is a LeRobot v3 dataset root plus the `vla_lens/` overlay.
Capture runtimes write robot data and interpretability internals; the dashboard
runtime reads the dataset root.

## Dashboard Container

Build and run the demo dashboard:

```bash
scripts/docker_dashboard.sh
```

Open:

```text
http://127.0.0.1:8080/
```

Serve an existing LeRobot v3 dataset root, trace dataset, or one `.vlatrace`
bundle:

```bash
scripts/docker_dashboard.sh runs/pi05-light-5-test
scripts/docker_dashboard.sh /path/to/some-dataset
scripts/docker_dashboard.sh /path/to/episode.vlatrace
```

With no argument, the script mounts local `./runs` and creates/serves
`runs/vla_lens_demo` if needed. With an explicit root, the script mounts that
path directly and fails if it is neither a LeRobot v3 dataset root nor a legacy
trace bundle/root.

Run the same single-origin dashboard without Docker:

```bash
scripts/view_vla_lens.sh
scripts/view_vla_lens.sh runs/pi05-light-5-test
```

Equivalent direct commands:

```bash
docker build -f docker/dashboard.Dockerfile -t vla-lens-dashboard:local .

docker run --rm \
  -p 8080:8080 \
  -v "$PWD/runs:/data/vla-lens/runs" \
  vla-lens-dashboard:local
```

Serve a different mounted trace root:

```bash
docker run --rm \
  -p 8080:8080 \
  -v "/path/to/traces:/data/vla-lens/runs" \
  -e VLA_LENS_TRACE_ROOT=/data/vla-lens/runs \
  -e VLA_LENS_BOOTSTRAP_DEMO=0 \
  vla-lens-dashboard:local
```

## How The Dashboard Image Works

The image has two build paths:

```text
frontend-builder:
  npm ci
  npm run build

python-builder:
  uv sync --frozen --no-dev
  build the normal VLA Lens environment

runtime:
  copy .venv, src, scripts, and built frontend assets
  run scripts/docker_dashboard_entrypoint.sh
```

The runtime starts:

```text
scripts/serve_vla_lens_app.py
```

That process starts the existing Python dashboard backend on an internal port,
serves built React assets, and proxies `/api/*` to the backend. The user gets
one browser origin:

```text
http://127.0.0.1:8080/
```

## CI Sync

The `docker-dashboard` GitHub Actions job builds:

```text
docker/dashboard.Dockerfile
```

on PRs and pushes. It does not publish an image yet; it only keeps the Docker
packaging honest against the current git tree.

## Current Cost

The dashboard image is now scoped to the normal app stack. Simulator/capture
packages such as `robosuite`, LIBERO, LeRobot policy runtimes, and accelerator
Torch wheels belong in capture environments, not the visualizer image.

## Capture Containers

The Linux capture images are explicit accelerator runtimes:

```text
docker/capture.cuda.Dockerfile
docker/capture.rocm.Dockerfile
```

They:

```text
1. install the matching Torch backend;
2. install LeRobot, LIBERO, OpenPI Transformers patch, and robosuite==1.4.0;
3. mount the same dataset volume as the dashboard;
4. write LeRobot v3 robot data plus `vla_lens/` overlay into that volume.
```

Run CUDA batch capture:

```bash
scripts/docker_pi05_cuda.sh \
  --config configs/pi05_light_5_test.yaml \
  --run
```

Run ROCm batch capture:

```bash
scripts/docker_pi05_rocm.sh \
  --config configs/pi05_light_5_test.yaml \
  --run
```

Write a high-volume run to a host or cloud-mounted volume:

```bash
scripts/docker_pi05_rocm.sh \
  --config configs/pi05_light_5_test.yaml \
  --output-root /mnt/nvme/vla-lens/pi05-light-5-test \
  --run
```

For absolute `--output-root` and `--vlatrace-out-root` values, the wrapper
creates the host directory, mounts it at `/capture-output`, and rewrites the
container command. This keeps the user-facing command honest: the path you pass
is the path that receives the LeRobot dataset roots on the host. The
`--vlatrace-out-root` flag name is retained only so existing scripts do not
break; the artifact written there is now a LeRobot root.

Run a single capture:

```bash
scripts/docker_pi05_cuda.sh capture --episodes 1 --capture-profile mechanistic_sampled ...
scripts/docker_pi05_rocm.sh capture --episodes 1 --capture-profile mechanistic_sampled ...
```

Check the packaged capture environment:

```bash
scripts/docker_pi05_cuda.sh check
scripts/docker_pi05_rocm.sh check
```

The wrapper scripts mount:

```text
$VLA_LENS_RUNS_DIR or ./runs -> /app/runs
$VLA_LENS_HF_CACHE_DIR or ~/.cache/huggingface -> /root/.cache/huggingface
$VLA_LENS_LIBERO_CACHE_DIR or ~/.cache/libero -> /root/.cache/libero
```

Because the configs use `output_root: runs/...`, captured datasets are written
back to the host checkout by default. For cloud jobs, prefer a POSIX write
target during capture: local NVMe, a mounted block volume, NFS/EFS/FSx, or a
FUSE-mounted object-store path that is reliable for many small file writes.
Syncing completed dataset roots to S3/GCS/Azure Blob after capture is usually
safer than writing directly to object storage mid-rollout.

Equivalent compose services exist behind profiles:

```bash
docker compose --profile cuda run --rm capture-cuda --config configs/pi05_light_5_test.yaml --run
docker compose --profile rocm run --rm capture-rocm --config configs/pi05_light_5_test.yaml --run
```

CUDA still requires host NVIDIA driver support plus `nvidia-container-toolkit`.
ROCm still requires host AMDGPU/ROCm support, `/dev/kfd`, `/dev/dri`, and
correct render/video group access.

On this workstation, `scripts/docker_pi05_rocm.sh --no-build check` has been
validated with strict device checking against an RX 7900 XTX. A one-command
real capture smoke also wrote a valid dataset root to an absolute host
output root:

```bash
PI05_STRICT_DEVICE_CHECK=1 scripts/docker_pi05_rocm.sh \
  --config configs/pi05_light_5_test.yaml \
  --output-root /tmp/vla-lens-rocm-smoke \
  --limit-commands 1 \
  --run
```

The resulting trace opened in the dashboard from the same host path:

```bash
scripts/view_vla_lens.sh /tmp/vla-lens-rocm-smoke
```

Apple Silicon MPS should remain a native virtualenv path unless a concrete
container strategy proves better. Docker Desktop runs Linux containers in a VM,
while MPS is a macOS backend.

The native non-Docker capture path remains supported:

```bash
scripts/setup_pi05_cuda_env.sh
scripts/pi05_batch_capture_cuda.sh --config configs/pi05_light_5_test.yaml --run

scripts/setup_pi05_rocm_env.sh
scripts/pi05_batch_capture_rocm.sh --config configs/pi05_light_5_test.yaml --run
```

The Docker containers package specific runtime slices; they do not replace
local development or native capture workflows.
