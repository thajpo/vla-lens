# PI0.5 Experiment-by-Experiment Writeup Draft

> Status: historical draft. This document summarizes the earlier canonical 480-episode workbench. It predates the full target-binding dataset and strict metadata-prior gate. The current defensible claim is narrower: hidden states predict `first_moved_object` and `first_lifted_object` beyond metadata priors; broad success/failure and target-identity probe claims are not headline evidence. See `docs/experiments/pi05-consolidated-findings.md`.

## Framing

The motivating question was not just whether PI0.5 can represent the instructed object, but where its structured failures come from.

Two concrete questions drove the work:

1. Does the model internally represent the instructed target cleanly?
2. If it does, why does behavior still fail in some scene and target combinations?

The current evidence supports a narrower claim than "the model has intent." It supports this claim:

> PI0.5 preserves target identity and substantial target-relative geometry through the VLM, handoff, and expert, yet still shows structured behavioral failures in `LIBERO_90 Scene 1`.

That pushes the project away from the question "does the model know the target?" and toward the question "why does the policy sometimes act badly on information it appears to have?"

## Model and benchmarks

These analyses use PI0.5 in LIBERO, with two benchmarks playing different roles.

- `LIBERO_OBJECT`: positive-control benchmark with clean language-conditioned behavior on the tested four-task subset.
- `LIBERO_90 Scene 1`: mechanistic workbench with structured partial failure in a shared physical scene.

From the canonical capture set used in the current analyses:

- `LIBERO_OBJECT`
  - `alphabet_soup`: `0.915`
  - `butter`: `1.0`
  - `cream_cheese`: `1.0`
  - `milk`: `0.95`
- `Scene 1`
  - `alphabet_soup`: `0.80`
  - `cream_cheese`: `0.678`
  - `ketchup`: `0.0`
  - `tomato_sauce`: `0.0`

The important point is not just that Scene 1 is harder. It is structured.

- In `Scene 1`, the first moved object is always `ketchup_1`.
- The first lifted object is usually `alphabet_soup_1`.
- `cream_cheese` often succeeds despite wrong early routing.
- `ketchup` and `tomato_sauce` do not recover.

So the failure mode is not generic noise. The policy appears to have scene-specific motor biases that interact with target identity.

## Experiment 1: Target identity probes

### Question

Does the model represent which object is being requested, and where in the stack is that information available?

### What was done

Linear classification probes were trained across pooled representations from:

- VLM text states
- VLM image states
- the VLM-to-expert handoff
- expert pooled states
- expert flow-step-0 states

### Result

Target identity is linearly decodable almost everywhere.

- VLM pooled representations decode target identity extremely well.
- The handoff preserves that information.
- The expert preserves that information.
- Expert flow-step-0 pooled representations still decode target identity strongly.

In the earlier summary artifacts, text-side target decoding is already very strong in early VLM layers, and target identity remains highly decodable even in failed Scene 1 rollouts.

### Interpretation

This is the strongest evidence that the model is not failing because it entirely loses the requested object label.

It is also consistent with a simple architectural picture:

- semantic target information is available early
- the expert inherits it rather than reconstructing it from scratch late in denoising

### What this does not show

This experiment does not establish "intent" in a strong mechanistic sense.

- A probe shows linearly accessible information, not necessarily used information.
- Because text tokens encode the instruction directly, high decoding from early text states could partly reflect prompt persistence rather than grounded visual selection.
- This does not yet show the model is visually singling out the target rather than merely carrying the instruction label forward.

### Main gap

The missing control is selectivity.

The most important follow-up is to test whether the target is privileged relative to distractors:

- probe non-target object identities
- probe non-target object positions
- compare target decoding to an `identity + scene prior` baseline

If all objects are equally decodable, then the result is about scene representation, not targeting.

## Experiment 2: Geometry and relation probes

### Question

If identity survives the handoff, does control-relevant geometry survive too?

### What was done

Regression probes were trained to predict:

- absolute target position
- target-to-gripper vector
- target-to-basket vector

across VLM, handoff, expert, and expert flow-step-0 pooled states.

### Result

These variables are strongly decodable across the stack.

Representative `R^2` values from the richer-probe artifact are typically in the `0.84-0.95` range, with examples such as:

- `Scene 1` VLM-text pooled `target_pos x`: `0.9518`
- `Scene 1` handoff pooled `target_pos x`: `0.9461`
- `Scene 1` expert-final pooled `target_to_gripper x`: `0.9275`
- `LIBERO_OBJECT` expert-final pooled `target_to_gripper x`: `0.9496`
- `LIBERO_OBJECT` expert-flow0 pooled `target_pos z`: `0.9346`

### Interpretation

This weakens a simple bottleneck story where the VLM or handoff preserves only the target label and leaves the expert to infer geometry later.

At minimum, the representation contains substantial information about:

- where the target is
- where it is relative to the gripper
- where it needs to go relative to the basket

### What this does not show

This still does not prove that the policy is using the probed geometry in a target-specific way.

There is an important confound here: in a benchmark with repeated scene layouts, object identity can partially predict object position. A probe may exploit stable scene regularities rather than online grounded localization.

### Main gap

The key missing baseline is an `object identity + benchmark` predictor.

If a simple baseline predicts geometry almost as well as the probe, then the current result is weaker than it sounds. The next clean control is:

1. predict geometry from object label and benchmark only
2. compare that against representation-based probes
3. repeat for target and non-target objects

## Experiment 3: Cross-benchmark transfer asymmetry

### Question

If both benchmarks encode target identity, are they doing so in the same representational format?

### Result

No, not fully.

Across the overlapping classes `alphabet_soup` and `cream_cheese`:

- `Scene 1 -> LIBERO_OBJECT` transfer is essentially perfect
- `LIBERO_OBJECT -> Scene 1` transfer degrades, especially in the expert

The interim summary describes this as the expert adding scene-specific structure beyond the target label itself.

### Interpretation

This is one of the more interesting representational results in the project.

The model is not just encoding "cream cheese" in a benchmark-invariant way. By the time the representation reaches the expert, the same nominal target is embedded differently depending on the scene family.

### What this does not show

The asymmetry does not identify what the extra Scene 1 structure actually is.

It could reflect any mixture of:

- clutter
- camera framing
- distractor configuration
- reachability differences
- contact difficulty
- benchmark-specific motor priors

### Main gap

This experiment motivates later delta analysis, but by itself it is still descriptive.

## Experiment 4: Benchmark classifiers and delta directions

### Question

Can the benchmark-specific component be isolated more explicitly, and does it become more coherent inside the expert?

### What was done

Two related analyses were run on the overlapping classes:

1. benchmark classification with object identity held fixed
2. cross-benchmark mean-difference directions for each class

### Result

Benchmark identity is strongly encoded throughout the stack, and the benchmark-delta becomes much more coherent in the expert.

From the delta artifact:

- VLM-all shared-delta cosine: `-0.4656`
- VLM-text shared-delta cosine: `0.1985`
- expert-flow0 shared-delta cosine: `0.4372`
- expert-final shared-delta cosine: `0.4842`

The expert deltas are also much larger in norm than the upstream deltas.

### Interpretation

This suggests a benchmark-specific direction is not merely inherited from the VLM in a fixed form. It becomes more organized inside the expert.

That is compatible with the idea that the expert is not just consuming target information. It is adding benchmark-specific control structure on top of it.

### What this does not show

This does not imply the delta direction is the failure mechanism.

It only shows:

- there is a coherent benchmark-specific shift
- it is strongest in the expert

That shift could still be adaptive, necessary, or epiphenomenal.

## Experiment 5: Success/failure overlap with the benchmark delta

### Question

Does the benchmark-specific expert direction also track failure within Scene 1 itself?

### Result

Yes, suggestively.

For the mixed-outcome tasks:

- `alphabet_soup` expert-final projection onto the shared delta
  - success mean: `77.544`
  - failure mean: `88.6368`
  - `p_fail_greater = 0.016402`
- `cream_cheese` expert-final projection onto the shared delta
  - success mean: `69.7418`
  - failure mean: `79.7919`
  - `p_fail_greater = 0.005583`

The explicit success/failure directions are also positively aligned with the benchmark-delta direction:

- `alphabet_soup` expert-final cosine: `0.257895`
- `cream_cheese` expert-final cosine: `0.332121`

### Interpretation

This is the cleanest observational link between benchmark-specific representation and behavioral failure.

The same direction that distinguishes Scene 1 from `LIBERO_OBJECT` is expressed more strongly in failing Scene 1 rollouts than in successful ones.

### What this does not show

This is still not causal.

The direction may be:

- a cause of failure
- a consequence of entering a failing control regime
- a correlated marker of task difficulty

At this stage, "candidate mechanistic handle" is justified. "Mechanism found" is not.

## Experiment 6: Delta ablation

### Question

If the benchmark-delta direction is failure-causing, does removing it rescue Scene 1 failures?

### What was done

At each denoising step, the projection of the expert final state onto the shared expert delta direction was subtracted before the action output projection.

Controls included:

- random directions
- orthogonal directions
- stronger ablation strengths
- all-layer ablations

### Result

The simple causal story failed.

Pilot:

- `alpha = 1.0` did not rescue the all-failure tasks
- effects on mixed-outcome tasks were weak and not cleaner than controls

Diagnostics:

- aggressive ablation (`alpha = 5`, `10`) collapsed both mixed-outcome tasks to `0.0`
- projection magnitude grows across denoising and is largest for the more failure-prone tasks
- random perturbations replicated above baseline across multiple seeds

### Interpretation

This forces a revision of the earlier story.

The delta direction is not a clean single-direction failure switch that can simply be removed. Large removals break the policy instead of rescuing it.

The most interesting updated result is that random perturbation often helps.

That points toward a different picture:

- the policy may be entering a brittle, failure-prone trajectory regime
- small perturbations can sometimes knock it into a better trajectory

### What this does not show

Even here, the wording needs care.

Off-manifold ablation is a blunt intervention. Saying the direction is strictly "load-bearing" is directionally reasonable, but not fully identified. What is cleanly supported is narrower:

- large removals of this direction are destructive
- targeted subtraction does not rescue
- random perturbation can sometimes improve outcomes

## Experiment 7: Handoff swap

### Question

If single-direction ablation is the wrong handle, does swapping the full VLM handoff between success and failure rollouts produce clearer causal effects?

### Result

Yes, but heterogeneously.

The smoke test showed a strong same-task same-layout causal effect for `cream_cheese` layout `3`:

- success-donor handoff preserved success
- failure-donor handoff induced failure

The rescue pilots sharpened the picture:

- `current_self_path` exactly matched baseline, validating the swap machinery
- `alphabet_soup` had at least one clean same-task rescue case and at least one non-rescuable failing layout
- `cream_cheese` showed strong handoff sensitivity, but scaled reruns exposed reproducibility issues in which some originally failing recipients no longer failed at rerun baseline
- cross-task rescue from `cream_cheese` to `ketchup` or `tomato_sauce` did not rescue the all-failure tasks

### Interpretation

This is the strongest causal evidence in the project so far.

It supports three claims:

1. Handoff content can be behaviorally decisive.
2. That effect is task- and layout-dependent.
3. The failure story is not universal; some failures are handoff-sensitive and some are not.

### What this does not show

The handoff is not yet shown to be sufficient in a broad, reproducible sense.

The biggest caveat is rerun instability. Rescue claims should be counted only when the recipient truly fails in the rerun baseline and then succeeds under swap.

## Overall interpretation

The most defensible current picture is this:

1. PI0.5 clearly carries target identity through the whole stack.
2. It also carries substantial target-relative geometry.
3. Scene 1 failures therefore are not well explained by total absence of target information.
4. A benchmark-specific representational shift becomes more coherent inside the expert and correlates with failure.
5. Simple removal of that direction does not rescue behavior.
6. Full handoff swaps provide stronger causal evidence that upstream conditioning can matter, but the effect is heterogeneous and not universally rescuing.

So the writeup should not conclude "the failure is representation-level" or "the failure is trajectory-level" in a binary way.

The stronger version is:

> The current evidence rules out a very simple missing-target-information story. It points toward a failure mode in how scene-conditioned information is organized and used by downstream control, with both handoff content and expert dynamics contributing.

## Main unresolved questions

These are the clearest next questions if the goal is to turn this into a sharper paper argument.

1. Is the target privileged, or are all scene objects equally decodable?
2. Do geometry probes beat an `identity + benchmark` baseline by a meaningful margin?
3. What distinguishes the all-failure objects `ketchup` and `tomato_sauce` from the mixed-outcome shared objects?
4. What properties of random perturbation matter: magnitude, layer, timing, or subspace?
5. Can the handoff-swap results be scaled using rerun-verified failing recipients only?

## Writing guidance

If this becomes the main narrative draft, keep the wording conservative.

Preferred language:

- "linearly decodable" over "the model knows"
- "consistent with" over "shows"
- "candidate mechanism" over "mechanism"
- "causally influential in some settings" over "causal explanation of failure"

The evidence is good enough for a strong story, but only if the claims stay at the level the experiments actually support.
