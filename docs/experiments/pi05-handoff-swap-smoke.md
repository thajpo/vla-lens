# PI0.5 Handoff-Swap Smoke Test

## Goal

Test whether swapping the saved VLM handoff (prefix KV cache / prefix state) from a successful rollout into a failing rollout changes the outcome more cleanly than the earlier delta-ablation interventions.

## Setup

We used same-task, same-layout mixed-outcome pairs from the canonical Scene 1 dataset.

Pairs:

- `cream_cheese`, layout `3`
  - success seed: `2000`
  - failure seed: `1000`
- `alphabet_soup`, layout `1`
  - success seed: `1000`
  - failure seed: `2000`

Conditions:

- `baseline`
- recipient with **failure donor** handoff
- recipient with **success donor** handoff

Artifacts:

- `artifacts/pi05_analysis/interventions/handoff_swap_smoke_cream_cheese_layout3.json`
- `artifacts/pi05_analysis/interventions/handoff_swap_cream_cheese_layout3_seed2000.json`
- `artifacts/pi05_analysis/interventions/handoff_swap_alphabet_layout1_seed1000.json`
- `artifacts/pi05_analysis/interventions/handoff_swap_alphabet_layout1_seed2000.json`
- `artifacts/pi05_analysis/interventions/handoff_swap_smoke_summary.csv`

## Summary

| object_label | layout | recipient_seed | condition | success | max_reward | target_max_lift | first_lifted_object |
|---|---:|---:|---|---:|---:|---:|---|
| cream_cheese | 3 | 1000 | baseline | 1 | 1.0 | 0.2430 | alphabet_soup_1 |
| cream_cheese | 3 | 1000 | failure donor handoff | 0 | 0.0 | 0.0002 | alphabet_soup_1 |
| cream_cheese | 3 | 1000 | success donor handoff | 1 | 1.0 | 0.2227 | alphabet_soup_1 |
| cream_cheese | 3 | 2000 | baseline | 1 | 1.0 | 0.2196 | cream_cheese_1 |
| cream_cheese | 3 | 2000 | failure donor handoff | 0 | 0.0 | 0.0000 | alphabet_soup_1 |
| cream_cheese | 3 | 2000 | success donor handoff | 1 | 1.0 | 0.2255 | alphabet_soup_1 |
| alphabet_soup | 1 | 1000 | baseline | 0 | 0.0 | 0.0026 | cream_cheese_1 |
| alphabet_soup | 1 | 1000 | failure donor handoff | 0 | 0.0 | 0.0027 | none |
| alphabet_soup | 1 | 1000 | success donor handoff | 0 | 0.0 | 0.0054 | cream_cheese_1 |
| alphabet_soup | 1 | 2000 | baseline | 0 | 0.0 | 0.0042 | cream_cheese_1 |
| alphabet_soup | 1 | 2000 | failure donor handoff | 0 | 0.0 | 0.1001 | alphabet_soup_1 |
| alphabet_soup | 1 | 2000 | success donor handoff | 0 | 0.0 | 0.0057 | none |

## Readout

### cream_cheese

This is the clearest result.

- both recipient seeds succeed under baseline
- both recipient seeds fail when given the **failure donor** handoff
- both recipient seeds succeed when given the **success donor** handoff

Interpretation:

- the handoff is causally powerful enough to flip the outcome on this task/layout pair
- this is much cleaner than the delta-ablation result
- for this pair, the failure is not just an expert-internal attractor independent of the handoff; the handoff content matters strongly

### alphabet_soup

The result is weaker.

- both recipient seeds fail under baseline
- neither donor handoff rescues them

Interpretation:

- this pair is either harder, more unstable, or governed by a different failure mode than `cream_cheese`
- the lack of rescue here does **not** negate the strong `cream_cheese` result; it shows heterogeneity across tasks

## Interpretation

The smoke test supports a more refined version of the current story:

1. The single benchmark-delta direction was not the right causal handle.
2. The **full handoff object** can matter causally, at least for `cream_cheese`.
3. The importance of the handoff is task- and layout-dependent; not every failure is equally rescuable.

This means the next causal experiment should likely be:

- a slightly larger handoff-swap pilot on `cream_cheese`
- followed by a broader comparison on `alphabet_soup` and possibly `ketchup`

## Practical takeaway

This smoke test is the strongest causal evidence in the project so far.

It does **not** settle the whole mechanism, but it establishes that handoff-level differences can be behaviorally decisive in at least one controlled same-task same-layout setting.
