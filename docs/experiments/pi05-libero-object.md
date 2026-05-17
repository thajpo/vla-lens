# PI0.5 on LIBERO_OBJECT

## Role

`LIBERO_OBJECT` is the **routing baseline / positive control** for `lerobot/pi05_libero_finetuned`.

Why this benchmark matters:

- the checkpoint is clearly capable here
- object-level routing follows the instruction on the tested subset
- this gives a benchmark where internal probes can be validated against strong behavior

## Tasks evaluated

20 episodes each:

1. `pick_up_the_alphabet_soup_and_place_it_in_the_basket`
2. `pick_up_the_cream_cheese_and_place_it_in_the_basket`
3. `pick_up_the_butter_and_place_it_in_the_basket`
4. `pick_up_the_milk_and_place_it_in_the_basket`

Artifact:

- `artifacts/pi05_libero/object_subset_20ep.json`

## Corrected results

| Target | Success | First moved object | First lifted object |
|---|---:|---|---|
| alphabet_soup | 19 / 20 | `alphabet_soup_1` in 19, `None` in 1 | `alphabet_soup_1` in 19, `None` in 1 |
| cream_cheese | 20 / 20 | `cream_cheese_1` in 20 | `cream_cheese_1` in 20 |
| butter | 20 / 20 | `butter_1` in 20 | `butter_1` in 20 |
| milk | 18 / 20 | `milk_1` in 19, `None` in 1 | `milk_1` in 18, `None` in 2 |

## Interpretation

This benchmark provides both:

1. **high canonical success**
2. **clean language-conditioned routing**

That makes it the right place to run:

- Phase 1 canonical reliability
- Phase 2 object-choice reliability
- Phase 4 perturbation as a positive-control robustness test

Overall canonical success on the tested subset is **77 / 80 = 96.25%**.

Target-first routing on the tested subset is:

- first moved object matches target in **78 / 80** episodes
- first lifted object matches target in **77 / 80** episodes

## Failure modes worth tracking

Even here, the small number of failures are useful:

- `alphabet_soup`: 1 miss out of 20
- `milk`: 2 misses out of 20

These are the right places to inspect when comparing successful vs. near-success behavior inside an otherwise strong benchmark.

## Justification

`LIBERO_OBJECT` should be treated as the benchmark where we establish that:

- the model can follow object-targeting language
- the logging/probing pipeline recovers behaviorally meaningful contrasts

It is not the most tightly controlled object-contrast benchmark, but it is the strongest **working** benchmark currently available for this pretrained model.

## Phase 4 note

Under target-distractor swaps, this benchmark collapses completely on the tested subset.

Perturbation artifact:

- `artifacts/pi05_libero/object_subset_swap10.json`

This makes `LIBERO_OBJECT` useful as:

- a strong positive control at canonical positions
- a strong memorization-sensitive benchmark under targeted layout edits
