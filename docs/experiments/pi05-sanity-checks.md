# PI0.5 Sanity Checks

## Goal

Run the two cheap checks needed before over-interpreting the current benchmark landscape.

## Check 1: Are Scene 1 ketchup / tomato tasks actually solvable according to LIBERO data?

### Method

Downloaded the official demonstration files from the LIBERO dataset repo:

- `libero_90/LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket_demo.hdf5`
- `libero_90/LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket_demo.hdf5`

Inspected several trajectories per file and checked:

- reward sums
- final `done`

### Result

For both tasks, sampled demonstrations showed:

- `sum_reward = 1`
- `final_done = 1`

### Interpretation

This is enough to rule out the strongest environment-confound interpretation.

Conclusion:

- `ketchup` and `tomato_sauce` are **task-valid** in Scene 1
- the model's `0 / 20` canonical failure on these tasks is not explained by the benchmark being unsolved by design

## Check 2: Does successful Scene 1 cream_cheese-swap really mean cream_cheese reaches the basket?

### Motivation

The most surprising perturbation result was:

- `cream_cheese` in Scene 1 under target swap: `8 / 10` success

Before treating that as a robust object-grounding phenomenon, we wanted to verify that the `cream_cheese` object specifically was the one satisfying the goal.

### Method attempted

Reran the 10 swapped `cream_cheese` episodes by stepping the underlying `LiberoEnv` directly to avoid vector-env auto-reset, and checked the basket contain-region predicate at the moment reward first became `1.0`.

### Result

This check produced an inconsistency:

- rollout reward signaled success in `9 / 10`
- but the direct contain-region check did **not** report any of the tracked movable objects inside the basket at that success step

### Interpretation

This means the manual perturbation replay path still has an unresolved semantic mismatch.

Most likely possibilities:

1. the direct low-level contain-region API is not being queried in the same way the task success code uses it after our manual state edits
2. our manual perturbation editing path changes internal assumptions enough that reward-based success is not yet fully trustworthy as a literal object-in-basket event under swaps

### Consequence

The `cream_cheese` swap robustness result should currently be treated as:

- **behaviorally real in the existing eval path**
- but **not yet fully verified as a literal target-in-basket success under the hand-edited swap path**

So this result is usable as a perturbation finding, but it should be marked **provisional** until the success semantics under manual swaps are cleaned up.

## Bottom line

### Clean result

- Scene 1 `ketchup` and `tomato_sauce` failures are model-specific, not benchmark-invalidity.

### Provisional result

- Scene 1 `cream_cheese` swap robustness is promising but still needs a cleaner success-semantic verification path.
