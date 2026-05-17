# PI0.5 Benchmark Decision

## Question

Which benchmark is the better primary substrate for immediate pretrained-model experiments on `lerobot/pi05_libero_finetuned`?

- `LIBERO_OBJECT`: simpler object-pick tasks with distractors present
- `LIBERO_90` `LIVING_ROOM_SCENE1`: same-scene quartet with object identity changes under a fixed scene layout

The benchmark should be chosen based on **routing behavior**, not just raw success.

## Evaluation Setup

### `LIBERO_OBJECT` object subset

Tasks evaluated for 20 episodes each:

1. `pick_up_the_alphabet_soup_and_place_it_in_the_basket`
2. `pick_up_the_cream_cheese_and_place_it_in_the_basket`
3. `pick_up_the_butter_and_place_it_in_the_basket`
4. `pick_up_the_milk_and_place_it_in_the_basket`

Metrics logged per episode:

- task success
- first moved object
- first lifted object
- max displacement object
- max lift object

Artifact:

- `artifacts/pi05_libero/object_subset_20ep.json`

### `LIBERO_90` `LIVING_ROOM_SCENE1` canonical quartet

Tasks evaluated on the same 5 canonical layout indices:

1. `LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket`
2. `LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket`
3. `LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket`
4. `LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket`

Metrics logged per rollout:

- task success
- first moved object
- first lifted object
- max lift object

Artifact:

- `artifacts/pi05_libero/scene1_layout5_canonical_fixed.json`

## Corrected Success Metric

The first Scene 1 analysis mistakenly read success from the underlying env state after the vector env had already auto-reset on termination. The corrected success metric is:

- `success = max_reward >= 1.0`

This matches LeRobot's own `eval_policy` behavior.

## Results

### `LIBERO_OBJECT` summary

| Target | Success | First moved object | First lifted object |
|---|---:|---|---|
| alphabet_soup | 19 / 20 | `alphabet_soup_1` in 19, `None` in 1 | `alphabet_soup_1` in 19, `None` in 1 |
| cream_cheese | 20 / 20 | `cream_cheese_1` in 20 | `cream_cheese_1` in 20 |
| butter | 20 / 20 | `butter_1` in 20 | `butter_1` in 20 |
| milk | 18 / 20 | `milk_1` in 19, `None` in 1 | `milk_1` in 18, `None` in 2 |

Key point:

- On the `LIBERO_OBJECT` subset, the model almost always moves and lifts the instructed object first.
- Success is high and routing appears instruction-consistent.

### `LIBERO_90` Scene 1 summary

| Target | Success | First moved object | First lifted object |
|---|---:|---|---|
| alphabet_soup | 5 / 5 | `ketchup_1` in 5 | `alphabet_soup_1` in 5 |
| cream_cheese | 5 / 5 | `ketchup_1` in 5 | `alphabet_soup_1` in 5 |
| ketchup | 0 / 5 | `ketchup_1` in 5 | `alphabet_soup_1` in 2, `None` in 3 |
| tomato_sauce | 0 / 5 | `ketchup_1` in 5 | `alphabet_soup_1` in 3, `None` in 2 |

Per-layout pattern:

- `alphabet_soup`: succeeds on all 5 layouts and lifts `alphabet_soup_1` first
- `cream_cheese`: also succeeds on all 5 layouts, but still lifts `alphabet_soup_1` first on every layout
- `ketchup`: fails on all 5 layouts; first movement is still `ketchup_1` but first lift is absent or `alphabet_soup_1`
- `tomato_sauce`: fails on all 5 layouts; first movement is still `ketchup_1` and first lift is absent or `alphabet_soup_1`

## Interpretation

### What `LIBERO_OBJECT` tells us

`LIBERO_OBJECT` is a **working benchmark** for this checkpoint.

- It gives high canonical success.
- It gives clean instruction-conditioned routing signal.
- It is therefore a good primary benchmark for immediate pretrained-model experiments.

### What `LIBERO_90` Scene 1 tells us

`LIVING_ROOM_SCENE1` is **not useless**, but it is not a clean primary success benchmark.

It shows a more interesting failure structure:

- the model can succeed on `alphabet_soup` and `cream_cheese`
- but early routing is partially collapsed
- `cream_cheese` succeeds despite consistently lifting `alphabet_soup_1` first
- `ketchup` and `tomato_sauce` fail entirely on these canonical layouts

This means Scene 1 is best treated as a **secondary controlled failure benchmark**.

## Decision

### Primary benchmark

Use `LIBERO_OBJECT` for the next pretrained-model phases.

Reason:

- success is real and high
- first-object routing follows the instruction for the tested subset
- we can study canonical reliability, object-choice behavior, and perturbation sensitivity without fighting immediate benchmark collapse

### Secondary benchmark

Keep `LIBERO_90` `LIVING_ROOM_SCENE1` as a failure-case benchmark.

Reason:

- it exposes partial language collapse / object bias under a more controlled same-scene setup
- it is valuable later for failure localization and probing
- but it is not the right primary benchmark for Phases 1-4 on the pretrained model

## Immediate implication for scope

The pretrained-model experiment program should proceed as:

1. Phases 1-4 on `LIBERO_OBJECT`
2. Reserve `LIVING_ROOM_SCENE1` for secondary analyses on failure, misrouting, and robustness collapse
