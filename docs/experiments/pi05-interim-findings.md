# PI0.5 Interim Findings

> Status: historical. This document predates the full 1420-episode target-binding dataset and the strict metadata-prior gate. Use `docs/experiments/pi05-consolidated-findings.md` as the current synthesis. Claims here about target identity, success/failure, benchmark deltas, or handoff should be treated as background unless they are restated there.

## Current scope

These findings use only the **canonical 480-episode capture set**.

Benchmarks:

- `LIBERO_OBJECT` (positive-control routing benchmark)
- `LIBERO_90 Scene 1` (mechanistic workbench)

Analyses completed:

1. Experiment 1: language routing
2. Experiment 2 pooled identity probes
3. Experiment 2 expert attention analysis
4. Experiment 2 richer geometry / relation probes
5. Experiment 2 Scene 1 success vs failure comparison
6. Experiment 2 cross-benchmark delta analysis
7. Experiment 2 benchmark-classifier / top-feature analysis

## 1. Behavioral baseline

### LIBERO_OBJECT

- success is near-perfect on the tested four-task subset
- first moved / first lifted object is almost always the instructed target

Interpretation:

- this is a valid positive-control benchmark where language routing works behaviorally

### Scene 1

- `alphabet_soup`: mixed but mostly successful
- `cream_cheese`: mixed but mostly successful
- `ketchup`: canonical failure
- `tomato_sauce`: canonical failure
- first moved object is consistently biased toward `ketchup_1`
- first lifted object is often `alphabet_soup_1`

Interpretation:

- Scene 1 is a structured partial-failure benchmark, not generic noise

## 2. Identity is encoded almost everywhere

Pooled identity probes show:

- VLM pooled representations decode target identity extremely well
- handoff pooled representations decode target identity extremely well
- expert pooled representations decode target identity extremely well
- expert flow-step pooled representations decode target identity from the first denoising step

Interpretation:

- the model knows which object is the target
- target identity survives the VLM->expert transition
- flow matching does not appear to be doing late semantic commitment; target identity is already present at step 0

## 3. Expert-side transfer is asymmetric

Cross-benchmark transfer on the overlapping classes (`alphabet_soup`, `cream_cheese`) shows:

- `Scene 1 -> LIBERO_OBJECT`: essentially perfect transfer
- `LIBERO_OBJECT -> Scene 1`: degraded transfer, especially at the expert level

Interpretation:

- target identity is represented in a partly shared, partly benchmark-specific way
- the strongest benchmark-specificity appears in the expert

This suggests the expert is adding scene-specific structure beyond the target label itself.

## 4. Attention analysis does not show a dramatic success/failure split

The expert final-layer attention analysis shows:

- most attention mass stays on suffix/action tokens in both benchmarks
- prefix attention to image and text is relatively small in aggregate
- `cream_cheese` success vs failure differences exist but are modest in the coarse aggregate view

Interpretation:

- the expert is not obviously flipping from "good prefix usage" to "bad prefix usage" in a way that is visible from coarse image/text/suffix aggregates alone
- if attention is the failure locus, it is likely more localized than this coarse summary can reveal

## 5. Richer probes show that geometry is also strongly preserved

This is the most important new result beyond the identity probes.

We probed for:

- absolute target position
- target-to-gripper relative vector
- target-to-basket relative vector

across VLM, handoff, and expert pooled representations.

### Main finding

These variables are all highly predictable from the captured representations.

Representative results:

- `Scene 1`, pooled VLM / handoff / expert all achieve very high `R^2` on target position and target-relative geometry, often around `0.84 - 0.95`
- `LIBERO_OBJECT` also shows strong geometry decoding, often around `0.90+`

Interpretation:

- the handoff does not preserve only the target label
- it also preserves substantial control-relevant spatial information

This weakens the hypothesis that Scene 1 failure is simply caused by the VLM/handoff delivering only a class label but not geometry.

## 6. Same-target success/failure comparisons suggest failure is not just missing target information

In `Scene 1`, mixed-outcome tasks (`alphabet_soup`, `cream_cheese`) show:

- successful and failed runs differ behaviorally
- but early movement bias remains similar
- failures are associated with lower target lift and worse target approach
- aggregated attention differences between success and failure are present but not dramatic

Interpretation:

- successful and failed runs are not cleanly separated by a complete absence of target information
- the failure seems more consistent with how the expert uses or specializes available information into action

## 7. Benchmark-specific structure is perfectly separable

Using only the overlapping classes (`alphabet_soup`, `cream_cheese`), benchmark classifiers reach essentially perfect accuracy on:

- VLM pooled features
- handoff pooled features
- expert pooled features
- expert flow-step-0 pooled features

Interpretation:

- benchmark identity is strongly encoded even when object identity is held fixed
- this confirms that the cross-benchmark transfer asymmetry is not noise
- the representation contains explicit benchmark-specific structure at every stage we tested

## 8. The benchmark delta becomes more coherent in the expert

The cross-benchmark delta analysis compared mean representations for the overlapping classes between `LIBERO_OBJECT` and `Scene 1`.

Main observations:

- VLM / handoff deltas for `alphabet_soup` and `cream_cheese` are not strongly aligned; their cosine is weak or even negative in some pooled views
- expert deltas are much larger in norm than VLM / handoff deltas
- expert delta directions are more aligned across the two overlapping classes (`cos ~ 0.44 - 0.48`)

Interpretation:

- benchmark-specificity is not just a shallow VLM phenomenon
- a more coherent shared benchmark-direction emerges inside the expert
- this supports the idea that the expert is adding scene-specific structure on top of the incoming target representation

## 9. The benchmark delta may be related to success vs failure

Projecting Scene 1 rollouts onto the shared expert delta direction shows a suggestive pattern:

- failed `alphabet_soup` and `cream_cheese` rollouts tend to have larger positive projection than successful ones

This is not yet a causal result, but it suggests the benchmark-specific direction is not merely decorative context; it may be tied to whether the expert succeeds or fails on the task.

Statistical validation:

- `alphabet_soup`: Mann-Whitney U, failures > successes, `p ≈ 0.0164`
- `cream_cheese`: Mann-Whitney U, failures > successes, `p ≈ 0.0056`

So the success/failure projection gap is not just a visual impression in the current sample.

## 10. Benchmark-feature overlap is stronger in the expert than upstream

Comparing the top benchmark-classifier dimensions with the top delta dimensions:

- VLM / handoff overlap is small
  - typically `1-2` shared dimensions in the top-20 sets
- expert overlap is larger
  - `expert_final`: `3` shared dims for `alphabet_soup`, `5` for `cream_cheese`

Interpretation:

- the benchmark classifier and the benchmark delta are describing a more coherent common structure inside the expert than in the VLM/handoff
- this strengthens the case that the expert is where benchmark-specific specialization becomes organized and behaviorally relevant

## Provisional overall interpretation

The current data supports the following picture:

1. The VLM encodes target identity strongly.
2. The handoff preserves target identity strongly.
3. The handoff also preserves substantial target geometry and relational information.
4. The expert receives this information and still shows benchmark-specific behavioral failures.
5. The expert also appears to add a strong benchmark-specific representational signature beyond what is visible at the VLM/handoff level.

So the most plausible current hypothesis is:

- the failure is not primarily that the VLM/handoff omits the target or its geometry
- the failure is more likely in the expert's scene-specific control computation
- the cross-benchmark delta gives a candidate direction in representation space that may help localize this computation more precisely

This should still be treated as a **working hypothesis**, not a final conclusion, because a causal intervention has not yet been run.

## Best next steps

The most valuable next analyses are now:

1. use the benchmark delta and benchmark-classifier directions as explicit analysis targets
2. finer-grained expert attention analysis
   - more localized token-region analysis rather than only image/text/suffix aggregates
3. causal handoff swap / activation patching
   - to distinguish insufficient handoff content from failure to use sufficient content
4. perturbation-based memorization localization
   - once we want to move beyond the canonical dataset

## Bottom line

The strongest current result is:

> PI0.5 preserves target identity and substantial target geometry through the VLM, handoff, and expert, yet still fails behaviorally in a structured way on Scene 1.

and

> the expert contains a strong benchmark-specific direction that is not fully explained by target identity alone and may be tied to failure.

That means the project has moved past the question of whether the model "knows the target" and into the more interesting question of why it sometimes fails to act correctly on information it appears to have.
