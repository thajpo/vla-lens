# PI0.5 `audit_sampled` Smoke Benchmark

Date: May 19, 2026

Status: one-trace smoke benchmark. This is enough to validate that `audit_sampled` can materialize and load, but not enough to make scale-up decisions.

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

Important: this used the direct venv entrypoint, not `uv run`, because `uv run` currently resyncs dependencies in a way that breaks the local LIBERO/LeRobot capture environment.

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
