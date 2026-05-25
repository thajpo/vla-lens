# Cloud Capture

Status: active operating model.

Last updated: May 25, 2026.

## Goal

Make real VLA interpretability capture boring:

```text
choose accelerator
choose config
choose output location
run one command
open the same traces in the dashboard
```

The durable artifact is the `.vlatrace` directory bundle. Capture jobs write
bundles; dashboard jobs read bundles.

## Storage Model

During capture, write to a POSIX filesystem path:

```text
local NVMe
attached cloud block volume
NFS/EFS/FSx
reliable FUSE-mounted object-store path
```

This is the right default because capture writes Parquet tables, array shards,
metadata, and status files as one logical bundle. Object stores are excellent
for storing completed bundles, but less pleasant as the active write target for
many small coordinated files.

After capture, sync completed output roots wherever the user wants:

```bash
aws s3 sync /mnt/nvme/vla-lens/pi05-light-5-test s3://my-bucket/vla-lens/pi05-light-5-test
gsutil -m rsync -r /mnt/nvme/vla-lens/pi05-light-5-test gs://my-bucket/vla-lens/pi05-light-5-test
rclone sync /mnt/nvme/vla-lens/pi05-light-5-test remote:vla-lens/pi05-light-5-test
```

## ROCm Cloud Or Workstation

Use an explicit host output root:

```bash
scripts/docker_pi05_rocm.sh \
  --config configs/pi05_light_5_test.yaml \
  --output-root /mnt/nvme/vla-lens/pi05-light-5-test \
  --run
```

The wrapper creates `/mnt/nvme/vla-lens/pi05-light-5-test`, mounts it into the
container, and rewrites the internal command so the trace bundles land on the
host path.

Then view the same dataset:

```bash
scripts/docker_dashboard.sh /mnt/nvme/vla-lens/pi05-light-5-test
```

## Shared Runs Directory

For repeated jobs, set one host runs directory and keep configs portable:

```bash
export VLA_LENS_RUNS_DIR=/mnt/nvme/vla-lens-runs

scripts/docker_pi05_rocm.sh \
  --config configs/pi05_light_5_test.yaml \
  --run

scripts/docker_dashboard.sh "$VLA_LENS_RUNS_DIR/pi05-light-5-test"
```

This works because the repo configs use relative `output_root: runs/...`, and
the Docker wrapper mounts `$VLA_LENS_RUNS_DIR` at `/app/runs`.

## Secrets And Caches

Forward Hugging Face auth with the normal environment variable:

```bash
HF_TOKEN=... scripts/docker_pi05_rocm.sh --config configs/pi05_light_5_test.yaml --run
```

The wrapper mounts:

```text
$VLA_LENS_HF_CACHE_DIR or ~/.cache/huggingface
  -> /root/.cache/huggingface

$VLA_LENS_LIBERO_CACHE_DIR or ~/.cache/libero
  -> /root/.cache/libero
```

On cloud workers, put this cache on a persistent volume when possible. That
keeps model downloads from dominating every job.

## Validated ROCm Smoke

On May 25, 2026, the ROCm Docker path was validated on an RX 7900 XTX:

```bash
PI05_STRICT_DEVICE_CHECK=1 scripts/docker_pi05_rocm.sh --no-build check
```

The container reported ROCm Torch `2.12.0+rocm7.2`, one visible GPU, and a
successful CUDA-device tensor allocation through PyTorch's ROCm-backed CUDA API.

A one-command capture smoke also completed:

```bash
PI05_STRICT_DEVICE_CHECK=1 scripts/docker_pi05_rocm.sh \
  --config configs/pi05_light_5_test.yaml \
  --output-root /tmp/vla-lens-rocm-smoke \
  --limit-commands 1 \
  --run
```

It wrote one valid `.vlatrace` bundle with 520 timesteps and 11 policy calls.
The task outcome was `failure`, but the capture pipeline itself completed:
model load, LIBERO rollout, activation capture, trace validation, and dashboard
readback all worked.

## Current Boundary

The CUDA and ROCm Docker paths package Linux capture. Apple Silicon MPS remains
a native virtualenv path because Docker Desktop does not expose macOS MPS into
Linux containers.

Publishing prebuilt capture images is the next usability step. Local image
builds are useful for validation, but the ROCm/CUDA stacks are large enough that
research users should usually pull a tested image rather than build one.
