# Remote GPU Capture To Local Analysis

Status: active workflow guide.

Last updated: May 26, 2026.

## Goal

This is the workflow for a researcher who does not own a strong GPU but wants
to do VLA interpretability work with a rented GPU host such as a cloud VM,
Vast.ai-style marketplace instance, lab workstation, or temporary cluster node:

```text
rent GPU briefly
capture model traces there
copy the completed dataset to storage you control
analyze locally without the capture stack
optionally host or archive the completed activations online
```

The key design point is that the expensive machine is only needed for policy
rollout and activation capture. After capture, the durable artifact is just a
directory: a LeRobot v3 dataset root plus the `vla_lens/` interpretability
overlay.

```text
rented GPU host
  PI0.5 + LIBERO + Torch + LeRobot
  writes dataset-root/

local laptop or desktop
  normal VLA Lens environment
  reads dataset-root/
  runs dashboard, selectors, probes, reports, and saved artifacts
```

The local analysis path should not require PI0.5, LIBERO, LeRobot, Torch, or a
GPU unless the analysis is explicitly doing new model execution.

## Current Support

| Workflow | Supported | Notes |
| --- | --- | --- |
| Capture on rented NVIDIA CUDA host, copy home, analyze locally | Yes | Use the CUDA Docker capture wrapper. |
| Capture on rented AMD ROCm host, copy home, analyze locally | Yes | Use the ROCm Docker capture wrapper where the host has ROCm device support. |
| Capture on cloud block/NVMe storage, then sync to S3/GCS/rclone remote | Yes | Sync completed roots after capture. |
| Serve dashboard from a VM or local machine that has the dataset mounted | Yes | Use the dashboard container or local dashboard script. |
| Open `s3://`, `gs://`, HTTP, or Hugging Face paths directly in `TraceDataset.open` | No | Download, sync, or mount the dataset as a local filesystem path first. |
| Public multi-user hosted activation service with auth and lazy remote tensor reads | No | This needs a remote storage/API layer. The current dashboard assumes local files. |

## What Gets Moved

Move the top-level capture output directory, not individual activation files.
The dataset contains coordinated metadata, tables, video shards, arrays, and
artifacts:

```text
pi05-diverse-100/
  episode_plan.csv
  probe_splits.csv
  capture_status.jsonl
  traces/
    pi05-diverse-100/
      mechanistic_sampled/
        libero_object/
          task_00/
            meta/
            data/
            videos/
            vla_lens/
```

Inside each LeRobot root:

```text
meta/             robot dataset metadata
data/             low-dimensional robot/action rows
videos/           MP4 camera streams
vla_lens/         model internals, policy-call alignment, probes, artifacts
```

The `vla_lens/` overlay is where activation tensors, attention tensors, token
metadata, action-generation arrays, fingerprints, and saved analysis artifacts
live.

## Choose A Capture Budget

Start with the cheapest profile that can answer the question.

| Profile | Use when | Local-analysis value |
| --- | --- | --- |
| `rollout` | You only need behavior, actions, success/failure, and videos. | Good for task coverage and failure review. |
| `features` | You want representation probes over hidden states. | Cheap first pass for "is X represented?" |
| `mechanistic_sampled` | You want the normal VLA Lens inspector dataset. | Best default for attention, sampled layers, probes, and action-head analysis. |
| `mechanistic_all` | You need all-layer semantic curves for a smaller number of traces. | Better localization, higher storage. |
| `audit_sampled`, `audit_windowed`, `audit_full` | You have a concrete circuit/transcoder/debugging question. | Expensive; do not collect broadly by default. |

For a low-compute researcher, `features` and `mechanistic_sampled` are the
normal starting points. Audit profiles can produce GiB-scale episodes, so treat
them like targeted experiments, not dataset defaults.

## Remote Capture

On a rented NVIDIA machine, use CUDA. On an AMD Linux machine with ROCm exposed,
use ROCm. The examples below use CUDA because that is the common rented-GPU
case.

Use a persistent disk path for output and caches. Many rented GPU images have
fast ephemeral storage that disappears when the instance is destroyed. Put
`VLA_LENS_RUNS_DIR`, Hugging Face cache, and LIBERO cache on storage that will
survive the capture job.

```bash
git clone <repo-url> vla-lens
cd vla-lens

export HF_TOKEN=...
export VLA_LENS_RUNS_DIR=/mnt/persistent/vla-lens-runs
export VLA_LENS_HF_CACHE_DIR=/mnt/persistent/hf-cache
export VLA_LENS_LIBERO_CACHE_DIR=/mnt/persistent/libero-cache
```

Check the packaged capture runtime and visible GPU:

```bash
PI05_STRICT_DEVICE_CHECK=1 scripts/docker_pi05_cuda.sh check
```

Plan a small run first. Omitting `--run` writes and prints the plan without
executing capture:

```bash
scripts/docker_pi05_cuda.sh \
  --config configs/pi05_light_5_test.yaml \
  --output-root /mnt/persistent/vla-lens-runs/pi05-light-5-test
```

Then run capture:

```bash
scripts/docker_pi05_cuda.sh \
  --config configs/pi05_light_5_test.yaml \
  --output-root /mnt/persistent/vla-lens-runs/pi05-light-5-test \
  --run
```

For a larger first probe dataset:

```bash
scripts/docker_pi05_cuda.sh \
  --config configs/pi05_diverse_100.yaml \
  --output-root /mnt/persistent/vla-lens-runs/pi05-diverse-100 \
  --run
```

ROCm uses the same shape:

```bash
PI05_STRICT_DEVICE_CHECK=1 scripts/docker_pi05_rocm.sh check

scripts/docker_pi05_rocm.sh \
  --config configs/pi05_diverse_100.yaml \
  --output-root /mnt/persistent/vla-lens-runs/pi05-diverse-100 \
  --run
```

Do not run PI0.5 capture through plain `uv run vla-pi05-capture` or
`uv run vla-pi05-batch-capture` in the normal repo environment. Capture has a
hardware-specific Torch/LeRobot/LIBERO stack; normal dashboard and analysis work
uses the normal repo environment.

## Verify Before Deleting The GPU Instance

Before stopping the rented machine, verify that the dataset opens. The safest
quick check is to serve the dashboard from the capture host:

```bash
scripts/docker_dashboard.sh /mnt/persistent/vla-lens-runs/pi05-diverse-100
```

If the dashboard is running on the remote host, open it through an SSH tunnel
from your local machine:

```bash
ssh -L 8080:127.0.0.1:8080 user@gpu-host
```

Then open:

```text
http://127.0.0.1:8080/
```

At minimum, check that the output contains LeRobot roots and overlays:

```bash
find /mnt/persistent/vla-lens-runs/pi05-diverse-100 -path '*/meta/info.json' | head
find /mnt/persistent/vla-lens-runs/pi05-diverse-100 -path '*/vla_lens/overlay.json' | head
```

Batch capture writes:

```text
episode_plan.csv        intended episodes
probe_splits.csv        train/test/probe split metadata
capture_status.jsonl    completed, failed, or skipped command status
```

If a batch is interrupted, re-running the same command normally skips traces
that already exist and are valid. Use `--force` only when you intentionally want
to recapture existing traces.

## Copy The Dataset Home

For a direct machine-to-machine copy:

```bash
rsync -aP --info=progress2 \
  user@gpu-host:/mnt/persistent/vla-lens-runs/pi05-diverse-100 \
  /media/$USER/vla-lens/
```

For object storage archive, sync only after capture has finished:

```bash
aws s3 sync \
  /mnt/persistent/vla-lens-runs/pi05-diverse-100 \
  s3://my-bucket/vla-lens/pi05-diverse-100
```

Later, restore it locally:

```bash
aws s3 sync \
  s3://my-bucket/vla-lens/pi05-diverse-100 \
  /media/$USER/vla-lens/pi05-diverse-100
```

Object stores are good for completed roots. They are not the preferred active
write target during capture because capture writes many coordinated Parquet,
MP4, Zarr, JSON, and status files. A normal POSIX filesystem path is safer
during rollout.

## Local Analysis

Once the dataset is on a local drive, use the normal repo environment:

```bash
scripts/view_vla_lens.sh /media/$USER/vla-lens/pi05-diverse-100
```

Open:

```text
http://127.0.0.1:8080/
```

Run dataset diagnostics:

```bash
uv run python scripts/save_vla_lens_dataset_report.py \
  /media/$USER/vla-lens/pi05-diverse-100
```

Train a probe from a saved YAML spec:

```bash
uv run python scripts/train_vla_lens_probe.py \
  /media/$USER/vla-lens/pi05-diverse-100 \
  --spec configs/probes/pi05_broad_1000_target_lifted_expert_action_hidden.yaml
```

Use the library directly from notebooks or scripts:

```python
import os

from vla_lens import ActivationQuery, TraceDataset

dataset = TraceDataset.open(
    os.path.expandvars("/media/$USER/vla-lens/pi05-diverse-100")
)

features, rows = dataset.select_model_sites(
    ActivationQuery(
        module="pi05.expert.layers.*",
        layers=[17],
        tensor_type="hidden_tokens",
        token_kind="action",
        reduce_tokens="mean",
    )
).to_matrix(cache=True)

print(features.shape)
print(rows.head())
```

The feature cache is written under the dataset root as `.vla_cache/`. If the
dataset is on a slow external disk, the first run may be slower; repeated runs
reuse cached matrices.

## Hosting Activations Online

There are three different meanings of "host the activations online."

### 1. Archive Online

This means storing completed dataset roots in S3, GCS, Azure Blob, rclone
remotes, or a shared file server. This is supported today.

```bash
aws s3 sync /local/pi05-diverse-100 s3://my-bucket/vla-lens/pi05-diverse-100
rclone sync /local/pi05-diverse-100 remote:vla-lens/pi05-diverse-100
```

Researchers download or mount the root before using VLA Lens.

### 2. Host The Dashboard Near The Data

This means running the dashboard on a machine that has the dataset mounted:

```bash
scripts/docker_dashboard.sh /mnt/datasets/pi05-diverse-100
```

For private work, prefer SSH tunnels or a VPN:

```bash
ssh -L 8080:127.0.0.1:8080 user@dashboard-host
```

For broader sharing, put the dashboard behind real authentication and TLS using
your infrastructure's reverse proxy. Treat VLA Lens as a trusted research
dashboard, not a hardened public SaaS service.

### 3. True Remote Activation Service

This would let the dashboard or Python API stream only the needed tensor chunks
from `s3://`, `gs://`, HTTP, or Hugging Face without first syncing the dataset
to a local filesystem.

That is not implemented yet. The current loaders expect local filesystem paths.
The workbench schema has storage references that can describe future remote
data, but the actual readers still open local Parquet, MP4, and Zarr paths.

A future implementation would need:

- remote-aware dataset opening
- auth-aware object-store readers
- lazy Zarr/Parquet chunk reads
- local cache management
- dashboard API controls for multi-user access
- dataset manifests that distinguish public metadata from private data

Until that exists, use one of these supported patterns:

```text
object store archive -> local sync -> VLA Lens
mounted shared filesystem -> VLA Lens
remote VM with mounted dataset -> dashboard over SSH/VPN/reverse proxy
```

## Common Failure Modes

- Capturing to ephemeral rented-GPU disk and losing the root when the instance
  stops. Use a persistent volume or sync before shutdown.
- Writing directly to object storage during capture. Prefer a POSIX filesystem
  during rollout, then sync the completed root.
- Running capture through the normal `uv run` environment. Use
  `scripts/docker_pi05_cuda.sh`, `scripts/docker_pi05_rocm.sh`, or the native
  `scripts/pi05_batch_capture.sh --backend ...` wrappers.
- Collecting audit profiles before the research question needs them. Start with
  `features` or `mechanistic_sampled`.
- Copying only `vla_lens/` or only `data/`. Keep the whole dataset root so robot
  rows, videos, activation arrays, metadata, fingerprints, and artifacts stay
  aligned.
