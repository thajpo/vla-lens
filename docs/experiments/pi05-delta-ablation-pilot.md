# PI0.5 Delta Ablation Pilot

## Goal

Test whether the shared expert benchmark-delta direction identified from the canonical captures is causally responsible for Scene 1 failures.

## Intervention

At each denoising step, after the expert produces `suffix_out` and before `action_out_proj`, subtract the projection of `suffix_out` onto the shared `expert_final` benchmark-delta direction:

`x' = x - alpha * <x, d> d`

with `alpha = 1.0`.

## Conditions

Pilot conditions per task/layout:

1. `baseline`
2. `delta_ablate`
3. `random_ablate`
4. `orthogonal_ablate`

## Tasks and layouts

- Scene 1 quartet
- layouts `0-4`
- seed `1000`

Artifact:

- `artifacts/pi05_analysis/interventions/delta_ablation_pilot.json`

## Summary

| object_label | condition | episodes | success_rate | mean_target_max_lift | mean_min_target_distance |
|---|---|---:|---:|---:|---:|
| alphabet_soup | baseline | 5 | 0.6 | 0.1637 | 0.0231 |
| alphabet_soup | delta_ablate | 5 | 0.6 | 0.1263 | 0.0224 |
| alphabet_soup | random_ablate | 5 | 0.6 | 0.1210 | 0.0233 |
| alphabet_soup | orthogonal_ablate | 5 | 0.6 | 0.1465 | 0.0228 |
| cream_cheese | baseline | 5 | 0.8 | 0.2216 | 0.0133 |
| cream_cheese | delta_ablate | 5 | 0.6 | 0.1254 | 0.0528 |
| cream_cheese | random_ablate | 5 | 1.0 | 0.2227 | 0.0127 |
| cream_cheese | orthogonal_ablate | 5 | 0.6 | 0.2155 | 0.0134 |
| ketchup | baseline | 5 | 0.0 | 0.0000 | 0.1850 |
| ketchup | delta_ablate | 5 | 0.0 | 0.0000 | 0.1718 |
| ketchup | random_ablate | 5 | 0.0 | 0.0000 | 0.1753 |
| ketchup | orthogonal_ablate | 5 | 0.0 | 0.0000 | 0.1682 |
| tomato_sauce | baseline | 5 | 0.0 | 0.0004 | 0.1115 |
| tomato_sauce | delta_ablate | 5 | 0.0 | 0.0000 | 0.1230 |
| tomato_sauce | random_ablate | 5 | 0.0 | 0.0004 | 0.1122 |
| tomato_sauce | orthogonal_ablate | 5 | 0.0 | 0.0000 | 0.1131 |

## Readout

### What changed

- `cream_cheese` worsened under `delta_ablate` (`0.8 -> 0.6` success), but also worsened under `orthogonal_ablate` (`0.8 -> 0.6`)
- `alphabet_soup` was unchanged in success rate across all conditions (`0.6`)
- `ketchup` and `tomato_sauce` remained `0.0` under all conditions

### What did not happen

- delta ablation did **not** rescue the failing Scene 1 tasks
- delta ablation did **not** produce a cleaner effect than the orthogonal control on the mixed-outcome tasks

## Interpretation

This pilot does **not** support the strong causal claim:

> the shared benchmark-delta direction by itself is the failure mechanism.

At best, the pilot suggests:

- the benchmark-delta direction may correlate with failure
- but ablating that direction alone, at this intervention point and strength, is not sufficient to rescue failure

Possible interpretations:

1. the delta is correlated but not causal
2. the delta is causal but this intervention point is too late / too local
3. the causal mechanism depends on a broader subspace, not a single shared direction
4. the intervention needs to target the handoff or earlier expert computation, not only `suffix_out`

## Practical takeaway

The observational findings remain strong, but the first delta-ablation pilot is a **negative / ambiguous** causal result.

The next causal experiment should probably not be "more of the same" with the same intervention point. More promising follow-ups are:

1. handoff swaps between success and failure rollouts
2. broader subspace ablations based on benchmark-classifier directions or success/failure directions
3. earlier expert interventions (e.g. at flow step 0 or at the handoff itself)

## Follow-up: truly-random controls

To check whether the helpful random effect was specific to the orthogonal complement of the delta direction, we added **truly random** directions that were **not** orthogonalized away from delta.

Artifact:

- `artifacts/pi05_analysis/interventions/diag_ablation_truerandom.json`

Aggregated result across four truly-random seeds:

| object_label | condition_family | episodes | success_rate |
|---|---|---:|---:|
| alphabet_soup | baseline | 5 | 0.60 |
| alphabet_soup | truerandom | 20 | 0.80 |
| cream_cheese | baseline | 5 | 0.80 |
| cream_cheese | truerandom | 20 | 1.00 |

Interpretation:

- the random-helps effect is **not** specific to the orthogonal complement of delta
- truly random directions also improve performance, especially for `cream_cheese`
- this strengthens the interpretation that the policy is trapped in a fragile / overcommitted trajectory regime and that perturbations can help it escape

Updated takeaway:

- delta ablation is not the right causal handle
- random perturbation appears more like a regularization or trajectory-escape mechanism than a targeted causal correction
- the best next causal experiment is now a **handoff swap**, not more single-direction ablation
