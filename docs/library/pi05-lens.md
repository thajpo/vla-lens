# PI0.5 Lens Library

Status: active library notes.

Last updated: May 27, 2026.

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
from vla_lens import ActivationQuery, TraceDataset

dataset = TraceDataset.open("runs/pi05-light-5-test")

features, rows = dataset.select_model_sites(
    ActivationQuery(
        module="pi05.expert.layers.*",
        layers=[17],
        reduce_tokens="mean",
    )
).to_matrix(
    cache=True,
)

print(rows.head())
print(features.shape)
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

probe_rows = rows.copy()
probe_rows["target"] = probe_rows["trace_id"].map({"trace-a": "success", "trace-b": "failure"})
probe_rows["split"] = probe_rows["trace_id"].map(lambda trace_id: "test" if trace_id.endswith("b") else "train")

results = run_probe_suite(
    rows=probe_rows,
    features={"expert.layer.17.mean": features},
    targets=["target"],
    metadata_baseline_columns=["task_id", "prompt"],
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

1. Keep PI0.5 selector aliases thin wrappers over `TraceDataset` and `ActivationQuery`.
2. Convert causal-trace scripts to emit `InterventionSpec` records.
3. Keep probe outputs as dataset artifacts under the `vla_lens/` overlay.
4. Add a second backend only after the PI0.5 interface feels stable.
