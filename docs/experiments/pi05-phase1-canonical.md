# Phase 1: Canonical Reliability Baseline

## Goal

Establish how reliably `lerobot/pi05_libero_finetuned` completes tasks under canonical benchmark layouts.

## Benchmarks

### LIBERO_OBJECT routing baseline

20 episodes each on:

1. `pick_up_the_alphabet_soup_and_place_it_in_the_basket`
2. `pick_up_the_cream_cheese_and_place_it_in_the_basket`
3. `pick_up_the_butter_and_place_it_in_the_basket`
4. `pick_up_the_milk_and_place_it_in_the_basket`

Artifact:

- `artifacts/pi05_libero/object_subset_20ep.json`

### LIBERO_90 Scene 1 mechanistic workbench

20 canonical init-state layouts each on:

1. `LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket`
2. `LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket`
3. `LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket`
4. `LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket`

Artifact:

- `artifacts/pi05_libero/scene1_layout20_canonical.json`

## Results

### LIBERO_OBJECT

| Target | Success | 95% Wilson CI |
|---|---:|---:|
| alphabet_soup | 19 / 20 | [0.764, 0.991] |
| cream_cheese | 20 / 20 | [0.839, 1.000] |
| butter | 20 / 20 | [0.839, 1.000] |
| milk | 18 / 20 | [0.699, 0.972] |

Pooled:

- **77 / 80 = 96.25%** success

### Scene 1

| Target | Success | 95% Wilson CI |
|---|---:|---:|
| alphabet_soup | 14 / 20 | [0.481, 0.855] |
| cream_cheese | 18 / 20 | [0.699, 0.972] |
| ketchup | 0 / 20 | [0.000, 0.161] |
| tomato_sauce | 0 / 20 | [0.000, 0.161] |

Pooled:

- **32 / 80 = 40.0%** success

## Takeaways

1. `LIBERO_OBJECT` is a strong canonical benchmark for this checkpoint.
2. `Scene 1` has a sharp object-dependent split rather than generic difficulty.
3. The split is now statistically hard to dismiss:
   - `ketchup` and `tomato_sauce` are both at `0 / 20`
   - `cream_cheese` remains strong at `18 / 20`

## Decision impact

Phase 1 supports the dual-benchmark framing:

- `LIBERO_OBJECT` as the positive-control routing benchmark
- `Scene 1` as the structured partial-success benchmark
