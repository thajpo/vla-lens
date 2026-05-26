# Current VLA Lens State

Status: active operational summary.

Last updated: May 26, 2026.

## Direction

VLA Lens is an episode-grounded causal interpretability workbench for
vision-language-action models.

Dataset-layer cutoff: LeRobotDataset v3 is the canonical robot-data layer.
VLA Lens adds a `vla_lens/` interpretability overlay for model internals,
policy-call alignment, token metadata, probes, artifacts, fingerprints, and
dashboard state. Standalone `.vlatrace` episode bundles are old internal storage,
not a compatibility layer to preserve in the new dataset contract. See
[dataset-format.md](dataset-format.md).

The current implementation focus is PI0.5 on LIBERO. Older CogACT planning docs
are archived and are not the active project direction.

The product loop is:

```text
observe episode behavior
localize model signals
hypothesize mechanisms
intervene on activations/features
measure action and rollout changes
visualize the result in episode context
```

## Environment Contract

Normal repo work:

```bash
scripts/check_vla_lens.sh
uv run pytest
uv run ruff check scripts src tests
cd frontend && npm run build
```

Portable demo:

```bash
scripts/run_vla_lens_demo.sh
```

Dashboard container:

```bash
scripts/docker_dashboard.sh
scripts/docker_dashboard.sh runs/pi05-light-5-test
scripts/view_vla_lens.sh runs/pi05-light-5-test
```

Dashboard paths can now point at one LeRobot root or a top-level batch output;
the dataset opener discovers nested LeRobot roots and serves them together.
Captured overlay trace IDs remain unchanged. Plain nested LeRobot roots without
overlays get path-prefixed episode IDs to avoid `episode_000000` collisions.

PI0.5 hardware capture:

```bash
scripts/setup_pi05_rocm_env.sh
scripts/setup_pi05_cuda_env.sh
scripts/setup_pi05_mps_env.sh
scripts/check_pi05_env.sh --backend rocm
scripts/pi05_batch_capture.sh --backend rocm ...
```

PI0.5 Linux capture containers:

```bash
scripts/docker_pi05_cuda.sh --config configs/pi05_light_5_test.yaml --run
scripts/docker_pi05_rocm.sh --config configs/pi05_light_5_test.yaml --run
```

Validated ROCm Docker smoke on May 25, 2026:

```bash
PI05_STRICT_DEVICE_CHECK=1 scripts/docker_pi05_rocm.sh --no-build check
PI05_STRICT_DEVICE_CHECK=1 scripts/docker_pi05_rocm.sh \
  --config configs/pi05_light_5_test.yaml \
  --output-root /tmp/vla-lens-rocm-smoke \
  --limit-commands 1 \
  --run
```

Result after the LeRobot writer switch: one valid LeRobot v3 dataset root plus
`vla_lens/` overlay, 520 timesteps, 11 policy calls, dashboard API readback OK.
Task outcome was `failure`, which is acceptable for this runtime smoke because
the test target was capture plumbing, not policy success.

The same root loaded with `lerobot.datasets.lerobot_dataset.LeRobotDataset` in
the capture image for metadata and tabular rows (`len=520`, one episode, action
and `observation.state` tensors readable). Full `dataset[0]` video decoding in
that image currently needs a working LeRobot video backend (`torchcodec` or a
Torchvision build with `VideoReader`).

Captured root shape:

```text
meta/info.json
meta/stats.json
meta/tasks.parquet
meta/episodes/chunk-000/file-000.parquet
data/chunk-000/file-000.parquet
videos/observation.images.main/chunk-000/file-000.mp4
videos/observation.images.wrist/chunk-000/file-000.mp4
vla_lens/overlay.json
vla_lens/tables/episode_refs.parquet
vla_lens/episodes/episode_000000/...
```

Do not run PI0.5 capture through plain `uv run vla-pi05-capture` or `uv run
vla-pi05-batch-capture`.

Use [hardware-run-paths.md](hardware-run-paths.md) for the current ROCm, CUDA,
and Apple Silicon setup/capture surface. Use [docker.md](docker.md) for the
dashboard container and Linux CUDA/ROCm capture-container paths.

Known-good ROCm capture environment from the last real smoke:

```text
torch:                  2.12.0+rocm7.2
torchvision:            0.27.0+rocm7.2
torchaudio:             2.11.0+rocm7.2
lerobot:                0.4.4
numpy:                  >=2.0,<2.3
pyarrow:                >=21.0,<25.0
datasets:               4.8.5
opencv-python-headless: 4.12.0.88
rerun-sdk:              0.26.2
transformers:           4.53.2 with OpenPI replacement patch
peft:                   0.19.1
hf-libero:              0.1.3
robosuite:              1.4.0
```

Always verify the target machine state with:

```bash
scripts/check_pi05_env.sh --backend rocm
```

## Current Capture Profiles

Use [pi05-capture-profiles.md](pi05-capture-profiles.md) for full detail.

```text
rollout              behavior only
features             representation probes
mechanistic_sampled  default inspector and probe dataset
mechanistic_all      all-layer semantic inspector
internals_sampled    sampled operation internals
audit_sampled        sampled circuit-boundary audit
audit_windowed       adjacent-layer whole-episode circuit/transcoder capture
audit_full           exhaustive raw/debug capture
custom               explicit one-off profile
```

Measured audit costs from May 20, 2026:

```text
audit_sampled / libero_object / 143 steps / 3 policy calls:
  2,233.6 MiB

audit_sampled / libero_spatial / 76 steps / 2 policy calls:
  1,510.5 MiB

audit_sampled / libero_goal / 123 steps / 3 policy calls:
  2,229.2 MiB

audit_windowed / libero_object / 144 steps / 3 policy calls:
  4,465.4 MiB
```

Interpretation:

```text
mechanistic_sampled is the normal profile.
audit_sampled and audit_windowed work, but they are GiB-scale audit profiles.
audit_windowed is about 2x comparable audit_sampled for the object smoke.
Do not collect audit_windowed broadly without a concrete circuit/transcoder question.
```

## Current Validated Facts

- Schema is `0.3.0`.
- `audit_sampled` uses layers `[0, 4, 8, 12, 17]`.
- `audit_windowed` uses layers `[0, 1, 4, 5, 8, 9, 12, 13, 16, 17]`.
- VLM prefix K/V is stored as exact per-layer key/value sites.
- The non-materialized runtime collection is `pi05.vlm.past_key_values`.
- `/api/activation-sites` returns PI0.5 architecture edges.
- K/V conditioning edges pair equal-index VLM and Expert layers.
- Expert attention query space is `pi05.action_suffix`.
- Expert attention key space is `pi05.expert_context`.

## Current Commands

Serve an existing dataset:

```bash
uv run python scripts/serve_vla_lens_dashboard.py /path/to/lerobot-root --port 8765
```

Run a single PI0.5 capture:

```bash
scripts/pi05_capture_rocm.sh \
  --model-id lerobot/pi05_libero_finetuned \
  --episodes 1 \
  --start-seed 1002 \
  --benchmark libero_object \
  --task-id 0 \
  --capture-profile mechanistic_sampled \
  --vlatrace-out-root "/path/to/vla-lens/pi05-smoke" \
  --dataset-id pi05-smoke
```

Run a batch from a config:

```bash
scripts/pi05_batch_capture_rocm.sh \
  --config configs/pi05_light_5_test.yaml \
  --run
```

Run the same config on NVIDIA CUDA:

```bash
scripts/pi05_batch_capture_cuda.sh \
  --config configs/pi05_light_5_test.yaml \
  --run
```

Run the same config on Apple Silicon MPS:

```bash
scripts/pi05_batch_capture_mps.sh \
  --config configs/pi05_light_5_test.yaml \
  --run
```

Run a batch from an explicit episode plan:

```bash
scripts/pi05_batch_capture_rocm.sh \
  --episode-plan path/to/episode_plan.csv \
  --output-root "/path/to/vla-lens/some-run" \
  --run
```

Run research guardrails without capture:

```bash
uv run python scripts/lint_research_guardrails.py --root .
uv run python scripts/lint_research_guardrails.py \
  --root . \
  --episode-plan path/to/episode_plan.csv \
  --audit-contract path/to/audit_contract.yaml
uv run python scripts/validate_vla_lens_dataset_trust.py /path/to/dataset-root
```

The config linter is read-only. It parses `configs/*.yaml` and
`configs/probes/*.yaml`, warns about machine-local runtime fields that wrapper
scripts override, blocks broad audit-profile accidents, and enforces
`requires_episode_plan: true` for broad-1000 capture configs.

The dataset trust gate is also read-only. It opens an existing local root and
checks schema/overlay validity, `probe_splits.csv`, activation coverage, outcome
balance, and saved artifact freshness before treating the root as probe-grade.

## Current Next Actions

Do not start by collecting more audit data.

The next useful research implementation should pick one concrete question, for
example:

```text
Can an Expert MLP skip transcoder at layer 8 explain a gripper/action change?
Can an audit_windowed pair show that Expert L8 writes a feature consumed by L9?
Can object-grounded attention routing predict action direction without causal overclaim?
```

Then choose the cheapest profile that supports that question.

## Documentation Hygiene

When work validates or invalidates a claim:

```text
1. update this file if it changes current operating truth;
2. update the current planning registry if it changes sequencing;
3. mark old experiment notes as historical if their commands are stale;
4. prefer deleting obsolete instructions over preserving contradictory plans.
```
