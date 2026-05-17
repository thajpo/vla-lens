# PI0.5 Handoff Rescue Pilot

## Goal

Test the **rescue direction** of handoff swaps:

- when a recipient rollout fails at baseline, can a success donor handoff rescue it?

This is stronger than the earlier failure-induction smoke test because it asks whether handoff content is sufficient to overcome a failing setup, not merely whether different handoffs can induce different behavior.

## Design

### Same-task rescue

Use all available same-layout mixed-outcome pairs in Scene 1 for:

- `alphabet_soup`
- `cream_cheese`

For each pair:

- recipient = one failing-seed rollout from the canonical dataset
- donor = one successful rollout of the **same task and same layout**

Conditions:

1. `baseline`
2. `current_self_path` (same computation path as swapping, but using the recipient's own live handoff)
3. `success_donor_swap`

### Cross-task rescue

Use `cream_cheese` success donors on matched layouts to try to rescue:

- `ketchup` failures
- `tomato_sauce` failures

Same three conditions.

Artifacts:

- `artifacts/pi05_analysis/interventions/handoff_rescue_pilot.json`
- `artifacts/pi05_analysis/interventions/handoff_rescue_pilot_summary.csv`
- `artifacts/pi05_analysis/interventions/handoff_rescue_pilot_pairs.csv`

## Summary

| experiment | recipient_object | condition | episodes | success_rate | mean_target_max_lift | mean_min_target_distance |
|---|---|---:|---:|---:|---:|---:|
| same_task_rescue | alphabet_soup | baseline | 8 | 0.625 | 0.1407 | 0.0303 |
| same_task_rescue | alphabet_soup | current_self_path | 8 | 0.625 | 0.1407 | 0.0303 |
| same_task_rescue | alphabet_soup | success_donor_swap | 8 | 0.750 | 0.1635 | 0.0256 |
| same_task_rescue | cream_cheese | baseline | 7 | 0.857 | 0.2082 | 0.0196 |
| same_task_rescue | cream_cheese | current_self_path | 7 | 0.857 | 0.2082 | 0.0196 |
| same_task_rescue | cream_cheese | success_donor_swap | 7 | 0.714 | 0.1915 | 0.0181 |
| cross_task_rescue | ketchup | baseline | 5 | 0.000 | 0.0000 | 0.1562 |
| cross_task_rescue | ketchup | current_self_path | 5 | 0.000 | 0.0000 | 0.1562 |
| cross_task_rescue | ketchup | success_donor_swap | 5 | 0.000 | 0.0000 | 0.1805 |
| cross_task_rescue | tomato_sauce | baseline | 5 | 0.000 | 0.0010 | 0.1090 |
| cross_task_rescue | tomato_sauce | current_self_path | 5 | 0.000 | 0.0010 | 0.1090 |
| cross_task_rescue | tomato_sauce | success_donor_swap | 5 | 0.000 | 0.0000 | 0.1155 |

## Sanity check: current-self path

For every tested pair:

- `current_self_path` exactly matched `baseline`

This is important. It means the swap machinery itself is not changing behavior when it reuses the recipient's own current handoff.

## Pair-level readout

### Same-task rescue: alphabet_soup

Baseline failures occurred on layouts:

- `1`
- `17`
- `26`

Rescue outcome with success donor handoff:

- layout `1`: **not rescued**
- layout `17`: **rescued**
- layout `26`: **not rescued**

So:

- `1 / 3` failing alphabet_soup recipients were rescued

### Same-task rescue: cream_cheese

Baseline failure occurred on:

- layout `27`

Rescue outcome with success donor handoff:

- layout `27`: **rescued**

So:

- `1 / 1` failing cream_cheese recipient was rescued

### Cross-task rescue: cream_cheese -> ketchup / tomato_sauce

No recipient was rescued.

So:

- the handoff does **not** trivially override the recipient task to produce a successful cross-target behavior

## Interpretation

This is a stronger causal result than the earlier failure-induction smoke test.

### What it shows

1. Handoff content is behaviorally decisive in at least some same-task same-layout failures.
2. The rescue effect is **not universal**.
3. Cross-task handoff substitution does not rescue all-failure tasks like `ketchup` and `tomato_sauce`.

### What it suggests

- `cream_cheese` appears to be highly handoff-sensitive.
- `alphabet_soup` is mixed: some failing layouts are rescuable, others are not.
- `ketchup` / `tomato_sauce` failures are not easily repaired by simply injecting a successful `cream_cheese` handoff.

So the failure story is now more differentiated:

- some failures are handoff-sensitive and rescuable
- some are not, implying stronger downstream or task-specific constraints

## Practical takeaway

The handoff swap is a stronger causal handle than the earlier single-direction delta ablation.

It does not support a universal "handoff explains everything" story, but it does show that handoff content can be sufficient to rescue at least a subset of failures in a controlled same-task same-layout setting.

That is the strongest causal evidence in the project so far.
