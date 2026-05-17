# PI0.5 Handoff Rescue (Scaled Same-Task Run)

## Goal

Scale the same-task rescue design to a small set of mixed-layout pairs for:

- `alphabet_soup`
- `cream_cheese`

and check whether the success-donor handoff can rescue failing recipients more consistently.

## Important caveat

Pairs were selected from the canonical capture as mixed-outcome layouts. However, on rerun, some recipients did **not** reproduce their original failure/success status.

So this scaled run needs to be interpreted in two layers:

1. **true rescue cases**: recipient fails at baseline on rerun, then success-donor swap rescues it
2. **donor effect cases**: recipient succeeds at baseline, and donor swap changes that behavior

The first is the stronger causal claim. The second is still informative but is not a rescue result.

## Artifact

- `artifacts/pi05_analysis/interventions/handoff_rescue_same_task_5each.json`
- `artifacts/pi05_analysis/interventions/handoff_rescue_same_task_5each_summary.csv`
- `artifacts/pi05_analysis/interventions/handoff_rescue_same_task_5each_pairs.csv`

## Summary

| recipient_object | condition | episodes | success_rate | mean_target_max_lift | mean_min_target_distance |
|---|---|---:|---:|---:|---:|
| alphabet_soup | baseline | 5 | 0.60 | 0.1363 | 0.0328 |
| alphabet_soup | current_self_path | 5 | 0.60 | 0.1363 | 0.0328 |
| alphabet_soup | success_donor_swap | 5 | 0.80 | 0.1755 | 0.0227 |
| cream_cheese | baseline | 5 | 1.00 | 0.2238 | 0.0179 |
| cream_cheese | current_self_path | 5 | 1.00 | 0.2238 | 0.0179 |
| cream_cheese | success_donor_swap | 5 | 0.60 | 0.1810 | 0.0170 |

## Pair-level interpretation

### alphabet_soup

Recipient layouts tested:

- `1`
- `8`
- `10`
- `17`
- `19`

On rerun:

- baseline failed on layouts `1` and `17`
- baseline succeeded on `8`, `10`, and `19`

Success-donor swap outcome:

- layout `1`: **not rescued**
- layout `17`: **rescued**

So among the true failing recipients:

- `1 / 2` were rescued

### cream_cheese

Recipient layouts tested:

- `3`
- `4`
- `6`
- `18`
- `22`

On rerun:

- baseline succeeded on **all 5** layouts

So there were **no true rescue opportunities** in this scaled run.

What the donor swap still showed:

- layouts `18` and `22` were degraded from success to failure by the success donor handoff

Interpretation:

- handoff content is clearly behaviorally active
- but this particular run does not support a rescue claim for `cream_cheese`, because the selected recipients did not fail at rerun baseline

## Main takeaways

1. `current_self_path` again matched baseline exactly, which continues to validate the swap machinery.
2. There is at least one clean same-task rescue case (`alphabet_soup`, layout `17`).
3. Rescue is not universal (`alphabet_soup`, layout `1` did not rescue).
4. Donor handoff content can also **degrade** otherwise successful recipients (`cream_cheese`, layouts `18`, `22`).

## Interpretation

The scaled run still supports the broader causal claim:

- handoff content is behaviorally important
- the effect is not trivial or one-directional
- rescue/degradation is highly task- and layout-dependent

But it also sharpens an operational concern:

- rerunning a stored canonical pair does not always reproduce the original success/failure outcome

So future rescue experiments should explicitly verify the recipient's baseline outcome *in the rerun* before counting it as a rescue case.

## Best current summary

The strongest defensible statement is:

> The VLM handoff can be causally decisive, but its rescue power is heterogeneous across tasks and layouts, and the same donor handoff can rescue some recipients while degrading others.
