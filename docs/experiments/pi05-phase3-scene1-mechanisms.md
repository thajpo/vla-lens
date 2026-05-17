# Phase 3: Scene 1 Mechanism Analysis

## Goal

Use the same-scene `LIVING_ROOM_SCENE1` quartet to localize where the policy succeeds, where it fails, and how those failures are structured over time.

## Artifact

- `artifacts/pi05_libero/scene1_layout20_canonical.json`

## Core empirical pattern

### Success asymmetry

| Target | Success |
|---|---:|
| alphabet_soup | 14 / 20 |
| cream_cheese | 18 / 20 |
| ketchup | 0 / 20 |
| tomato_sauce | 0 / 20 |

### Early routing asymmetry

- first moved object is always `ketchup_1`
- first lifted object is usually `alphabet_soup_1`

This means Scene 1 is **not** just a hard benchmark.
It has a specific motor bias that interacts with object identity.

## Failure modes under investigation

### 1. Early movement collapse

Even when the instruction is not about ketchup, the first movement goes to `ketchup_1`.

Interpretation:

- the initial policy dynamics are dominated by a scene-specific prior or motor program
- early behavior is not cleanly language-routed in this scene

### 2. Delayed recovery for cream cheese

`cream_cheese` succeeds on `18 / 20` episodes, but:

- first move is still always `ketchup_1`
- first lift is often `alphabet_soup_1`

Interpretation:

- the model can recover from a collapsed early bias and still complete the task
- this makes `cream_cheese` especially valuable for temporal analyses of when target-specific control takes over

### 3. Total non-recovery for ketchup and tomato sauce

`ketchup` and `tomato_sauce` both fail on all tested layouts.

Yet the structure differs:

- `ketchup`: first move is target-consistent, but first lift and completion fail
- `tomato_sauce`: neither early movement nor lifting is target-consistent

Interpretation:

- `ketchup` may fail later in the control pipeline
- `tomato_sauce` appears to fail from the start

## Why Scene 1 matters

Scene 1 is the benchmark where we can compare, within one scene family:

- success with clean target-first lift (`alphabet_soup`, partially)
- success despite wrong early routing (`cream_cheese`)
- failure after target-consistent first movement (`ketchup`)
- failure without target-consistent routing (`tomato_sauce`)

This is the strongest current benchmark for mechanistic analysis of:

- early vs. late target commitment
- object-specific recovery vs. collapse
- where routing and execution diverge
