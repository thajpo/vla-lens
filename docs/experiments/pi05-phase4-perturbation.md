# Phase 4: Perturbation Stress Test

## Goal

Test how canonical routing and success respond when the target object is swapped with a distractor.

This is a targeted anti-memorization perturbation:

- if the model follows object identity, routing should adapt
- if the model relies on canonical positions or fixed motor programs, success should collapse

## Artifacts

- `artifacts/pi05_libero/object_subset_swap10.json`
- `artifacts/pi05_libero/scene1_swap10.json`

## Perturbation

For each task, the target object joint pose is swapped with a chosen distractor partner before rollout.

### LIBERO_OBJECT partners

- alphabet_soup ↔ cream_cheese
- cream_cheese ↔ alphabet_soup
- butter ↔ ketchup
- milk ↔ cream_cheese

### Scene 1 partners

- alphabet_soup ↔ cream_cheese
- cream_cheese ↔ alphabet_soup
- ketchup ↔ tomato_sauce
- tomato_sauce ↔ ketchup

## Results

### LIBERO_OBJECT target-swap sweep

| Target | Success | First lifted object |
|---|---:|---|
| alphabet_soup | 0 / 10 | `cream_cheese_1` in 8, `None` in 2 |
| cream_cheese | 0 / 10 | `alphabet_soup_1` in 1, `cream_cheese_1` in 2, `None` in 7 |
| butter | 0 / 10 | `None` in 10 |
| milk | 0 / 10 | `None` in 10 |

Interpretation:

- the easy benchmark collapses completely under targeted swaps
- this strongly suggests that canonical success depends heavily on the original position/object arrangement

### Scene 1 target-swap sweep

| Target | Success | First moved object | First lifted object |
|---|---:|---|---|
| alphabet_soup | 0 / 10 | `cream_cheese_1` in 6, `None` in 4 | `cream_cheese_1` in 5, `None` in 5 |
| cream_cheese | 8 / 10 | `cream_cheese_1` in 8, `None` in 2 | `cream_cheese_1` in 8, `None` in 2 |
| ketchup | 0 / 10 | `tomato_sauce_1` in 9, `ketchup_1` in 1 | `ketchup_1` in 10 |
| tomato_sauce | 0 / 10 | `tomato_sauce_1` in 8, `ketchup_1` in 2 | `ketchup_1` in 10 |

Interpretation:

- `cream_cheese` remains surprisingly robust even after the swap (`8 / 10`)
- `alphabet_soup` collapses completely under the same manipulation
- `ketchup` and `tomato_sauce` remain failures, but their early movement / lift patterns still differ

## Main takeaways

1. `LIBERO_OBJECT` canonical success is fragile under object-location swaps.
2. `Scene 1` reveals object-specific robustness structure rather than generic perturbation collapse.
3. `cream_cheese` is the most robust task in the current Scene 1 quartet.

## Caution

Hard swaps can occasionally introduce simulation artifacts or object intersections.
Because of that, the most trustworthy Phase 4 readouts are:

- success rate
- first moved object
- first lifted object

and not raw max-lift magnitudes alone.
