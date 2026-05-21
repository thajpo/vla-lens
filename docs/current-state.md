# Current VLA Lens State

Status: active operational summary.

Last updated: May 20, 2026.

## Direction

VLA Lens is an episode-grounded causal interpretability workbench for
vision-language-action models.

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
uv run pytest
uv run ruff check scripts src tests
cd frontend && npm run build
```

PI0.5 ROCm capture:

```bash
scripts/setup_pi05_rocm_env.sh
scripts/check_pi05_rocm_env.sh
scripts/pi05_capture_rocm.sh ...
scripts/pi05_batch_capture_rocm.sh ...
```

Do not run PI0.5 capture through plain `uv run vla-pi05-capture` or `uv run
vla-pi05-batch-capture`.

Known-good capture environment from the last real smoke:

```text
torch:       2.12.0+rocm7.2
torchvision: 0.27.0+rocm7.2
lerobot:     0.4.4
transformers: 4.53.2 with OpenPI replacement patch
peft:        0.19.1
hf-libero:   0.1.3
robosuite:   1.4.0
```

Always verify the current machine state with:

```bash
scripts/check_pi05_rocm_env.sh
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
- PI0.5 profile aliases still work:
  - `representation -> features`
  - `mechanistic_light -> mechanistic_sampled`
  - `mechanistic_heavy -> mechanistic_all`
  - `full -> audit_full`
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
uv run python scripts/serve_vla_lens_dashboard.py /path/to/vlatrace/root --port 8765
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
  --vlatrace-out-root "/media/j/New Volume/vla-lens/pi05-smoke" \
  --dataset-id pi05-smoke
```

Run a batch from a config:

```bash
scripts/pi05_batch_capture_rocm.sh \
  --config configs/pi05_light_5_test.yaml \
  --run
```

Run a batch from an explicit episode plan:

```bash
scripts/pi05_batch_capture_rocm.sh \
  --episode-plan path/to/episode_plan.csv \
  --output-root "/media/j/New Volume/vla-lens/some-run" \
  --run
```

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
2. update the living roadmap if it changes sequencing;
3. archive or mark old experiment docs as historical if their commands are stale;
4. prefer deleting obsolete instructions over preserving contradictory plans.
```
