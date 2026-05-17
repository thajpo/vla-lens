# PI0.5 on LIBERO_90 Scene 1

## Role

`LIBERO_90` `LIVING_ROOM_SCENE1` is the **mechanistic workbench / structured partial-failure benchmark**.

It is not the strongest benchmark for overall success, but it is the most scientifically interesting current benchmark because it contains:

- the same scene
- the same distractors
- the same basket target
- object-specific success / failure asymmetry
- partially collapsed early routing

## Tasks evaluated

Canonical layout-controlled evaluation on the first 20 init-state layouts:

1. `LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket`
2. `LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket`
3. `LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket`
4. `LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket`

Artifact:

- `artifacts/pi05_libero/scene1_layout20_canonical.json`

## Corrected results

| Target | Success | First moved object | First lifted object |
|---|---:|---|---|
| alphabet_soup | 14 / 20 | `ketchup_1` in 20 | `alphabet_soup_1` in 15, `None` in 4, `cream_cheese_1` in 1 |
| cream_cheese | 18 / 20 | `ketchup_1` in 20 | `alphabet_soup_1` in 16, `cream_cheese_1` in 4 |
| ketchup | 0 / 20 | `ketchup_1` in 20 | `alphabet_soup_1` in 8, `cream_cheese_1` in 1, `None` in 11 |
| tomato_sauce | 0 / 20 | `ketchup_1` in 20 | `alphabet_soup_1` in 12, `None` in 8 |

## Interpretation

This benchmark shows a structured asymmetry:

- `alphabet_soup` succeeds on most layouts but not all (`14 / 20`)
- `cream_cheese` succeeds on most layouts and remains the strongest task (`18 / 20`)
- `ketchup` fails on all tested canonical layouts (`0 / 20`)
- `tomato_sauce` fails on all tested canonical layouts (`0 / 20`)

At the same time, early routing is partially collapsed:

- first move is always `ketchup_1`
- first lift is often `alphabet_soup_1`
- `cream_cheese` can still succeed despite lifting `alphabet_soup_1` first in most episodes
- `ketchup` is special: its first movement is target-consistent, but first lift and task completion still fail

So the Scene 1 signal is not “the model just fails.”
It is:

- **language routing is only partially controlling behavior**
- **object-specific success emerges on top of a biased early motor pattern**

## Failure modes to investigate

This benchmark is the right place to investigate:

1. **early routing collapse**
   - why does first movement default to `ketchup_1`?

2. **late recovery vs. non-recovery**
   - why can `cream_cheese` succeed despite wrong first lift?
   - why do `ketchup` and `tomato_sauce` fail completely?

3. **object-specific asymmetry**
   - what makes soup/cheese tractable while ketchup/sauce fail?

## Justification

Scene 1 should be retained because it provides exactly the sort of structured contrast that mechanistic analysis needs:

- success and failure in one benchmark
- same-scene object contrasts
- a measurable gap between early and late behavior

It is not the best benchmark for establishing that the model works.
It is the best current benchmark for studying **how and where the model breaks**.

## Phase 4 note

Under target-distractor swaps, the Scene 1 quartet remains highly asymmetric.

Perturbation artifact:

- `artifacts/pi05_libero/scene1_swap10.json`

Most strikingly:

- `cream_cheese` still succeeds in `8 / 10` swap episodes
- `alphabet_soup`, `ketchup`, and `tomato_sauce` all collapse to `0 / 10`

This suggests Scene 1 is not just a hard benchmark. It exposes object-specific robustness and failure structure under controlled perturbation.
