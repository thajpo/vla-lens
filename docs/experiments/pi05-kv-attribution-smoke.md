# Experiment 3: KV-Cache Causal Attribution

## Method

For each paired same-task same-layout success/failure rollout from Scene 1,
we run offline causal tracing on the VLM-expert handoff (the 18-layer prefix KV cache).

**Clean/corrupt framing:**
- Clean = success rollout's KV cache
- Corrupt = failure rollout's KV cache
- Observation context (prefix_pad_masks, noise_x0) always from the failure rollout

**Rescue patch:** Replace failure KV at layer *i* with success KV at layer *i*,
keeping failure KV at all other layers. If this moves the action toward the
success baseline, layer *i* carries causally important handoff content.

**Null control (wrong-layer swap):** Replace failure KV at position *i* with
success KV from layer *(i+1) mod 18* -- architecturally mismatched content.
If rescue patches outperform null controls, the expert is specifically
sensitive to which layer's content is at each cache position.

**Recovery score:** 1 - ||patched - clean|| / ||corrupt - clean||

## Pairs analyzed

Total: 1 pairs
- cream_cheese: 1 unique layouts

## Layer attribution (all pairs)

| layer | rescue_mean | rescue_std | null_mean | null_std | rescue-null | n_pairs |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000 | nan | -0.7036 | nan | +0.7036 | 1 |
| 14 | 0.0000 | nan | -0.4240 | nan | +0.4240 | 1 |
| 16 | 0.0000 | nan | -0.2616 | nan | +0.2616 | 1 |
| 12 | 0.0000 | nan | -0.0477 | nan | +0.0477 | 1 |
| 9 | 0.0000 | nan | -0.0325 | nan | +0.0325 | 1 |
| 13 | 0.0000 | nan | -0.0233 | nan | +0.0233 | 1 |
| 11 | 0.0000 | nan | -0.0049 | nan | +0.0049 | 1 |
| 6 | 0.0000 | nan | -0.0046 | nan | +0.0046 | 1 |
| 8 | 0.0000 | nan | -0.0045 | nan | +0.0045 | 1 |
| 3 | 0.0000 | nan | -0.0039 | nan | +0.0039 | 1 |
| 4 | 0.0000 | nan | 0.0027 | nan | -0.0027 | 1 |
| 2 | 0.0000 | nan | 0.0031 | nan | -0.0031 | 1 |
| 10 | 0.0000 | nan | 0.0034 | nan | -0.0034 | 1 |
| 5 | 0.0000 | nan | 0.0044 | nan | -0.0044 | 1 |
| 1 | 0.0000 | nan | 0.0051 | nan | -0.0051 | 1 |
| 7 | 0.0000 | nan | 0.0081 | nan | -0.0081 | 1 |
| 15 | 0.0000 | nan | 0.0149 | nan | -0.0149 | 1 |
| 17 | 0.0000 | nan | 0.0156 | nan | -0.0156 | 1 |

### Top 5 layers by rescue-minus-null

- Layer 0: rescue=0.0000+/-nan, null=-0.7036+/-nan, delta=+0.7036
- Layer 14: rescue=0.0000+/-nan, null=-0.4240+/-nan, delta=+0.4240
- Layer 16: rescue=0.0000+/-nan, null=-0.2616+/-nan, delta=+0.2616
- Layer 12: rescue=0.0000+/-nan, null=-0.0477+/-nan, delta=+0.0477
- Layer 9: rescue=0.0000+/-nan, null=-0.0325+/-nan, delta=+0.0325

## Per-object breakdown: cream_cheese

| layer | rescue_mean | rescue_std | null_mean | null_std | rescue-null | n_pairs |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000 | nan | -0.7036 | nan | +0.7036 | 1 |
| 14 | 0.0000 | nan | -0.4240 | nan | +0.4240 | 1 |
| 16 | 0.0000 | nan | -0.2616 | nan | +0.2616 | 1 |
| 12 | 0.0000 | nan | -0.0477 | nan | +0.0477 | 1 |
| 9 | 0.0000 | nan | -0.0325 | nan | +0.0325 | 1 |
| 13 | 0.0000 | nan | -0.0233 | nan | +0.0233 | 1 |
| 11 | 0.0000 | nan | -0.0049 | nan | +0.0049 | 1 |
| 6 | 0.0000 | nan | -0.0046 | nan | +0.0046 | 1 |
| 8 | 0.0000 | nan | -0.0045 | nan | +0.0045 | 1 |
| 3 | 0.0000 | nan | -0.0039 | nan | +0.0039 | 1 |
| 4 | 0.0000 | nan | 0.0027 | nan | -0.0027 | 1 |
| 2 | 0.0000 | nan | 0.0031 | nan | -0.0031 | 1 |
| 10 | 0.0000 | nan | 0.0034 | nan | -0.0034 | 1 |
| 5 | 0.0000 | nan | 0.0044 | nan | -0.0044 | 1 |
| 1 | 0.0000 | nan | 0.0051 | nan | -0.0051 | 1 |
| 7 | 0.0000 | nan | 0.0081 | nan | -0.0081 | 1 |
| 15 | 0.0000 | nan | 0.0149 | nan | -0.0149 | 1 |
| 17 | 0.0000 | nan | 0.0156 | nan | -0.0156 | 1 |

## Interpretation

This smoke test is useful mostly as a negative/diagnostic result.

Compact baseline metrics:

| condition | l2_to_success_action | l2_to_failure_action | recovery |
|---|---:|---:|---:|
| clean_baseline: success KV + failure noise/context | 5.8751 | 0.0000 | 0.0000 |
| corrupt_baseline: failure KV + failure noise/context | 5.8751 | 0.0000 | 0.0000 |

In this offline setup, replacing the failure KV cache with the success KV cache for `call_00` did **not** move the generated action away from the failure trajectory at all. Every same-layer rescue patch also had recovery `0.0000`.

So for this pair, the saved first-call prefix KV cache alone is not sufficient to reproduce the live handoff-swap effects seen elsewhere. Possible interpretations:

- the decisive difference is not in single-layer KV content for `call_00`
- the relevant effect may require full live rollout feedback across multiple policy calls
- the failure/success distinction may depend on observation/state/noise context, not just donor KV
- the offline clean/corrupt construction may be too constrained because it uses failure noise and failure prefix masks

The positive rescue-minus-null numbers in the table should not be read as rescue. They occur because some wrong-layer null controls make the output even farther from the success action, while correct-layer rescue still leaves the output exactly at the failure action.

Next step: do not scale this attribution setup until its clean baseline can move the trajectory. First test whole-KV full-call swaps across all matched calls or score the existing live handoff-swap runs with the calibrated object-relative metrics.
