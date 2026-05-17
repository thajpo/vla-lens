# Phase 2: Object-Choice Reliability

## Goal

Measure whether the policy's early behavior follows the instructed object, not just whether the task eventually succeeds.

Metrics:

- first moved object
- first lifted object
- target-first move rate
- target-first lift rate

## Artifacts

- `artifacts/pi05_libero/object_subset_20ep.json`
- `artifacts/pi05_libero/scene1_layout20_canonical.json`

## Results

### LIBERO_OBJECT

| Target | Target-first move | Target-first lift |
|---|---:|---:|
| alphabet_soup | 19 / 20 | 19 / 20 |
| cream_cheese | 20 / 20 | 20 / 20 |
| butter | 20 / 20 | 20 / 20 |
| milk | 19 / 20 | 18 / 20 |

Pooled:

- target-first move: **78 / 80**
- target-first lift: **77 / 80**

Interpretation:

- routing is clean and instruction-consistent on the tested subset
- the positive-control benchmark is doing what we need it to do

### Scene 1

| Target | Target-first move | Target-first lift |
|---|---:|---:|
| alphabet_soup | 0 / 20 | 15 / 20 |
| cream_cheese | 0 / 20 | 4 / 20 |
| ketchup | 20 / 20 | 0 / 20 |
| tomato_sauce | 0 / 20 | 0 / 20 |

First-move pattern:

- `ketchup_1` is the first moved object in **all 80 Scene 1 canonical rollouts**

First-lift pattern:

- `alphabet_soup_1` dominates first-lift behavior for `alphabet_soup`, `cream_cheese`, and `tomato_sauce`
- `ketchup` often does not reach a clean first-lift event at all

## Interpretation

This phase shows the two benchmarks are qualitatively different:

### LIBERO_OBJECT

- early routing follows language
- task success and routing agree

### Scene 1

- early routing is partially collapsed
- success can still occur despite wrong early routing (`cream_cheese`)
- some tasks fail even when the first movement is target-consistent (`ketchup`)

## Consequence for later phases

Phase 2 justifies using:

- `LIBERO_OBJECT` to establish what correct routing looks like
- `Scene 1` to study disagreement between early behavior and final outcome
