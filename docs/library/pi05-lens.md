# PI0.5 Lens Library

This is the first interpretability-library layer for the repo.  It is additive:
it wraps the existing PI0.5 capture schema and does not replace the current
experiment scripts.

## Goal

The library turns repeated experiment moves into named primitives:

- capture-store indexing
- activation selection
- probe dataset construction
- metadata baseline comparison
- declarative intervention specs
- experiment manifests

The first backend is intentionally PI0.5-only.  PI0.5's important interface is
the PaliGemma prefix `past_key_values` cache passed into the Gemma expert
denoising loop, not a single conditioning vector.

## Basic Usage

```python
from vla_lens.pi05 import PI05CaptureStore
from vla_lens.pi05.datasets import build_call_feature_table

store = PI05CaptureStore("/media/j/New Volume/vla-lens-artifacts/pi05_target_binding_captures_clean")

table = build_call_feature_table(
    store,
    selectors=[
        "vlm.layer.8.mean",
        "vlm.final.mean",
        "expert.layer.17.final_step.mean",
        "flow.step.4.flat",
        "action.final.flat",
    ],
    canonical=True,
    max_calls_per_rollout=1,
)

print(table.rows.head())
print(table.features["expert.layer.17.final_step.mean"].shape)
```

## Selector Names

Supported selector forms:

```text
vlm.layer.{layer}.mean
vlm.layer.{layer}.flat
vlm.final.mean
vlm.prefix.mean
vlm.handoff.kv.layer.{layer}.flat
expert.layer.{layer}.final_step.mean
expert.layer.{layer}.flow_step.{step}.mean
flow.step.{step}.flat
action.final.flat
```

Use `.mean` when you want one vector per call.  Use `.flat` for action chunks,
flow states, or KV-cache attribution where the shape itself is meaningful.

## Probe Suite

```python
from vla_lens.probes import run_probe_suite

rows = table.rows.copy()
rows["split"] = rows["layout_episode_index"].map(lambda x: "train" if int(x) < 20 else "test")

results = run_probe_suite(
    rows=rows,
    features=table.features,
    targets=["first_moved_object", "first_lifted_object", "success"],
    metadata_baseline_columns=["task_name", "layout_episode_index"],
)

print(results.sort_values("score", ascending=False).head())
```

The baseline comparison is deliberately built in because PI0.5 analyses have
repeatedly found that task, layout, and object-position shortcuts can dominate
naive activation probes.

## Intervention Specs

```python
from vla_lens.pi05 import InterventionSpec

spec = InterventionSpec.kv_rescue(
    recipient_rollout_id="bad_rollout",
    donor_rollout_id="good_rollout",
    layer=8,
    call_index=0,
)

print(spec.to_record())
```

These specs are not execution engines yet.  They are stable, auditable records
that existing causal-trace scripts can consume.

## Intended Next Steps

1. Convert one probe script to use `PI05CaptureStore` and selectors.
2. Convert one Scene 4 causal-trace script to emit `InterventionSpec` records.
3. Add visualizations over the resulting standard tables.
4. Add a second backend only after the PI0.5 interface feels stable.
