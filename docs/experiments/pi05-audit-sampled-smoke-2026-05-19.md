# PI0.5 `audit_sampled` Smoke Benchmark

Date: May 19, 2026

Status: superseded by the May 20, 2026 wrapper benchmark below. The original
one-trace smoke remains useful as the first proof that `audit_sampled` could
materialize and load.

## May 20, 2026 Wrapper Benchmark Update

The capture environment was rebuilt with the dedicated ROCm wrapper path:

```bash
scripts/setup_pi05_rocm_env.sh
bash scripts/check_pi05_rocm_env.sh
```

Environment check result:

```text
PI0.5 ROCm capture environment OK
torch:       2.12.0+rocm7.2
torchvision: 0.27.0+rocm7.2
lerobot:     0.4.4
transformers: 4.53.2
peft:        0.19.1
hf-libero:   0.1.3
robosuite:   1.4.0
```

The ROCm Torch install is large. The `torch` wheel alone was 5.8 GiB and the
setup took about 23.5 minutes for that package on this machine. Future agents
should not interrupt this step unless it has clearly failed.

### 3-Trace `audit_sampled` Benchmark

Commands used `scripts/pi05_capture_rocm.sh`, not plain `uv run`:

```bash
/usr/bin/time -v scripts/pi05_capture_rocm.sh \
  --model-id lerobot/pi05_libero_finetuned \
  --episodes 1 \
  --start-seed 1002 \
  --benchmark libero_object \
  --task-id 0 \
  --capture-profile audit_sampled \
  --vlatrace-out-root "/media/j/New Volume/vla-lens/pi05-audit-sampled-benchmark" \
  --dataset-id pi05-audit-sampled-benchmark

/usr/bin/time -v scripts/pi05_capture_rocm.sh \
  --model-id lerobot/pi05_libero_finetuned \
  --episodes 1 \
  --start-seed 1002 \
  --benchmark libero_spatial \
  --task-id 0 \
  --capture-profile audit_sampled \
  --vlatrace-out-root "/media/j/New Volume/vla-lens/pi05-audit-sampled-benchmark" \
  --dataset-id pi05-audit-sampled-benchmark

/usr/bin/time -v scripts/pi05_capture_rocm.sh \
  --model-id lerobot/pi05_libero_finetuned \
  --episodes 1 \
  --start-seed 1002 \
  --benchmark libero_goal \
  --task-id 0 \
  --capture-profile audit_sampled \
  --vlatrace-out-root "/media/j/New Volume/vla-lens/pi05-audit-sampled-benchmark" \
  --dataset-id pi05-audit-sampled-benchmark
```

Trace results:

```text
pi05_audit_sampled_libero_object_task0_seed1002
  steps: 143
  policy calls: 3
  success: true
  wall clock: 1:19.24
  max CPU RSS: 16,518,340 KB
  size: 2,233.6 MiB
  model sites: 244
  runtime collection members: 10
  architecture edges: 5

pi05_audit_sampled_libero_spatial_task0_seed1002
  steps: 76
  policy calls: 2
  success: true
  wall clock: 1:07.91
  max CPU RSS: 16,518,160 KB
  size: 1,510.5 MiB
  model sites: 244
  runtime collection members: 10
  architecture edges: 5

pi05_audit_sampled_libero_goal_task0_seed1002
  steps: 123
  policy calls: 3
  success: true
  wall clock: 1:13.44
  max CPU RSS: 16,519,772 KB
  size: 2,229.2 MiB
  model sites: 244
  runtime collection members: 10
  architecture edges: 5
```

The sampled benchmark dataset validates cleanly:

```text
validate_trace_dataset(...).valid == true
warnings == []
```

Per-trace model storage by family:

```text
object, 3 calls:
  attention:      704.7 MiB
  cache:            9.5 MiB
  mlp:          1,193.5 MiB
  normalization:   95.8 MiB
  representation:  61.4 MiB
  residual:       161.3 MiB
  action_head:      2.3 MiB

spatial, 2 calls:
  attention:      481.0 MiB
  cache:            6.5 MiB
  mlp:            804.2 MiB
  normalization:   64.9 MiB
  representation:  41.6 MiB
  residual:       108.6 MiB
  action_head:      1.5 MiB

goal, 3 calls:
  attention:      703.6 MiB
  cache:            9.6 MiB
  mlp:          1,192.9 MiB
  normalization:   95.3 MiB
  representation:  61.3 MiB
  residual:       161.0 MiB
  action_head:      2.3 MiB
```

### Whole-Episode `audit_windowed` Smoke

Command:

```bash
/usr/bin/time -v scripts/pi05_capture_rocm.sh \
  --model-id lerobot/pi05_libero_finetuned \
  --episodes 1 \
  --start-seed 1002 \
  --benchmark libero_object \
  --task-id 0 \
  --capture-profile audit_windowed \
  --vlatrace-out-root "/media/j/New Volume/vla-lens/pi05-audit-windowed-smoke" \
  --dataset-id pi05-audit-windowed-smoke \
  --delete-existing
```

Result:

```text
trace:
  pi05_audit_windowed_libero_object_task0_seed1002

steps:
  144

policy calls:
  3

success:
  true

wall clock:
  2:11.44

max CPU RSS:
  21,068,688 KB

size:
  4,465.4 MiB

model sites:
  484

runtime collection members:
  20

architecture edges:
  10

edge layers:
  [0, 1, 4, 5, 8, 9, 12, 13, 16, 17]
```

The `audit_windowed` trace validates cleanly:

```text
validate_trace_dataset(...).valid == true
warnings == []
```

Model storage by family:

```text
attention:
  1,403.3 MiB

cache:
  19.6 MiB

mlp:
  2,400.7 MiB

normalization:
  194.1 MiB

representation:
  115.7 MiB

residual:
  324.6 MiB

action_head:
  2.3 MiB
```

Projection:

```text
audit_windowed is roughly 2x audit_sampled for a comparable 3-policy-call
object episode: 4,465.4 MiB vs 2,233.6 MiB.

100 audit_windowed episodes at this object-trace size:
  about 436 GiB

1,000 audit_windowed episodes at this object-trace size:
  about 4.26 TiB
```

Interpretation:

```text
audit_windowed is feasible for targeted whole-episode transcoder/circuit
captures, but it should not be the default dataset-scale profile.

MLP internals are the dominant cost in both audit profiles. If storage becomes
too high, trim specific MLP roles deliberately instead of silently changing
audit_windowed from whole-episode capture into selected-policy-call capture.
```

## Command

```bash
/usr/bin/time -v .venv/bin/vla-pi05-capture \
  --episodes 1 \
  --start-seed 1002 \
  --benchmark libero_object \
  --task-id 0 \
  --capture-profile audit_sampled \
  --vlatrace-out-root "/media/j/New Volume/vla-lens/pi05-audit-sampled-smoke" \
  --delete-existing
```

Historical note: this used a direct venv entrypoint instead of `uv run`,
because `uv run` can resync dependencies in a way that breaks the local
LIBERO/LeRobot capture environment. Do not repeat this exact command now; use
`scripts/pi05_capture_rocm.sh` so the dedicated `.venv-pi05-rocm` stack is
checked before capture.

## Environment

```text
torch:        2.11.0+rocm7.2
torchvision:  0.26.0+rocm7.2
lerobot:      0.4.4
transformers: 4.53.2 with OpenPI transformers_replace patch copied into site-packages
peft:         0.19.1
hf-libero:    0.1.3
robosuite:    1.4.0
```

LeRobot was installed into the venv without allowing it to replace the ROCm torch stack.

## Trace

```text
root:
  /media/j/New Volume/vla-lens/pi05-audit-sampled-smoke

trace:
  /media/j/New Volume/vla-lens/pi05-audit-sampled-smoke/pi05_audit_sampled_libero_object_task0_seed1002.vlatrace

episode:
  pi05_audit_sampled_libero_object_task0_seed1002

benchmark:
  libero_object

task:
  0

seed:
  1002

steps:
  138

policy calls:
  3

success:
  true
```

## Runtime

```text
wall clock:
  1:21.41

user time:
  167.77s

system time:
  17.36s

max CPU RSS:
  16,717,972 KB

file outputs:
  5,575,144 blocks
```

This does not yet include baseline timing, mechanistic-sampled timing, trace-write-only timing, or peak GPU memory.

## Validation

The generated trace loaded through `TraceDataset.open(...)` and passed dataset validation:

```text
validate_trace_dataset(dataset).valid:
  true

model sites:
  244

runtime collections:
  1

architecture nodes:
  14

architecture edges:
  5
```

Architecture edges were the expected same-index per-layer K/V conditioning edges:

```text
pi05.vlm.layers.0.kv_to_expert.layers.0
pi05.vlm.layers.4.kv_to_expert.layers.4
pi05.vlm.layers.8.kv_to_expert.layers.8
pi05.vlm.layers.12.kv_to_expert.layers.12
pi05.vlm.layers.17.kv_to_expert.layers.17
```

## Storage

Top-level trace storage:

```text
du size:
  2.4G

file bytes:
  2.18 GiB

arrays:
  2.18 GiB

model arrays:
  2.18 GiB

media:
  4.44 MiB

tables:
  0.22 MiB

action/context/episode arrays outside model:
  <1 MiB combined
```

Model bytes by family:

```text
mlp:
  1.17 GiB

attention:
  706.05 MiB

residual:
  161.39 MiB

normalization:
  95.78 MiB

representation:
  61.39 MiB

cache:
  9.54 MiB

action_head:
  2.26 MiB
```

Model bytes by stack:

```text
VLM:
  1.68 GiB

Expert:
  512.43 MiB

Action head:
  2.26 MiB
```

Model-site counts by capture family:

```text
attention:
  90

normalization:
  50

mlp:
  50

residual:
  30

representation:
  12

cache:
  10

action_head:
  2
```

Largest storage roles:

```text
mlp_intermediate:
  396.23 MB across 10 sites

mlp_up:
  375.97 MB across 10 sites

mlp_gate:
  373.69 MB across 10 sites

attention_probs:
  216.99 MB across 20 sites

pre_mask_scores:
  202.22 MB across 10 sites

post_mask_logits:
  120.34 MB across 10 sites
```

Largest individual sites were VLM MLP internals:

```text
pi05.vlm.layers.17.mlp.intermediate  74.33 MB  [3, 968, 16384]
pi05.vlm.layers.17.mlp.up            72.28 MB  [3, 968, 16384]
pi05.vlm.layers.17.mlp.gate          71.76 MB  [3, 968, 16384]
pi05.vlm.layers.12.mlp.gate          70.34 MB  [3, 968, 16384]
pi05.vlm.layers.0.mlp.intermediate   70.17 MB  [3, 968, 16384]
```

## Immediate Interpretation

`audit_sampled` is materially heavier than the old `mechanistic_sampled` run. In this smoke trace, the dominant cost is not K/V cache and not media; it is selected VLM MLP gate/up/intermediate activations, with attention tensors second.

This means `audit_sampled` should remain an audit/debug profile until 3-5 more traces establish the size distribution and runtime slowdown. The next benchmark should vary episode length and policy-call count, then decide whether to:

```text
keep audit_sampled v0 as-is for small audit subsets,
trim selected roles,
make VLM MLP internals optional,
or move circuit-boundary internals to event-windowed capture.
```

## Follow-up Status: `audit_windowed`

Date: May 19, 2026

`audit_windowed` has been added as the whole-episode adjacent-layer capture profile for transcoder/circuit work. It reuses the `audit_sampled` raw role set and changes layer coverage to:

```text
VLM:    [0,1], [4,5], [8,9], [12,13], [16,17]
Expert: [0,1], [4,5], [8,9], [12,13], [16,17]
```

The profile is unit validated but not real-capture validated yet.

Attempted capture environment check:

```bash
bash scripts/check_pi05_rocm_env.sh
scripts/pi05_capture_rocm.sh --model-id lerobot/pi05_libero_finetuned --episodes 1 --start-seed 1002 --benchmark libero_object --task-id 0 --capture-profile audit_windowed --vlatrace-out-root "/media/j/New Volume/vla-lens/pi05-audit-windowed-smoke" --delete-existing
```

Result:

```text
blocked: /home/j/Projects/vla-lens/.venv-pi05-rocm/bin/python is missing
blocked: /home/j/Projects/vla-lens/.venv-pi05-rocm/bin/vla-pi05-capture is missing
```

Because the dedicated ROCm capture environment is missing, the requested 3-trace `audit_sampled` benchmark and one-trace `audit_windowed` smoke could not be run in this pass.

Naive projection from the one real `audit_sampled` trace:

```text
audit_sampled layers:
  5 VLM + 5 Expert sampled layers

audit_windowed layers:
  10 VLM + 10 Expert windowed layers

audit_sampled observed size:
  2.18 GiB file bytes for 3 policy calls

rough audit_windowed projection for a similar episode:
  about 4 GiB, because nearly all bytes are model arrays and most model arrays are per-layer
```

This projection is not a substitute for a real smoke. The next validation step is:

```text
1. Rebuild .venv-pi05-rocm with scripts/setup_pi05_rocm_env.sh.
2. Run 3 audit_sampled traces through scripts/pi05_capture_rocm.sh.
3. Run 1 audit_windowed trace through scripts/pi05_capture_rocm.sh.
4. Update this note with measured storage/runtime and architecture-edge counts.
```
