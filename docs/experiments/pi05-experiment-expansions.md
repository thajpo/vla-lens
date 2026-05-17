# PI0.5 Experiment Expansions

## Purpose

This file translates the current PI0.5 results into concrete next experiments, organized by the existing `E1-E8` groups in `pi05-experiment-matrix.md`.

It is written as an execution handoff for future agents. Each group answers four questions:

1. what the group is trying to establish
2. what is already established
3. what the next clean expansion is
4. what would count as a meaningful result

## Global rules

- Treat the canonical 480-episode capture set as the source of truth for observational work unless a rerun is explicitly required.
- For rescue claims, count a case only if the recipient fails in the rerun baseline and succeeds under the intervention.
- Prefer layout-based comparisons over call-level comparisons when training probes or summarizing effects.
- Keep claims at the level of the evidence:
  - "decodable" is not the same as "used"
  - correlation is not causation
  - off-manifold ablation is not a clean necessity proof

## E1. Language Routing

### Intent

Define the behavioral phenomena that later internal analyses are supposed to explain.

### Already established

- `LIBERO_OBJECT` is a valid positive-control routing benchmark.
- `Scene 1` has structured early-routing collapse rather than generic difficulty.
- `ketchup_1` is the first moved object in all canonical `Scene 1` rollouts.
- `cream_cheese` can recover from wrong early routing, while `ketchup` and `tomato_sauce` do not.

Primary references:

- `pi05-phase1-canonical.md`
- `pi05-phase2-routing.md`
- `pi05-experiment1-routing.md`

### Next expansion

#### E1A. Per-layout recovery analysis

Question:
When early routing is wrong, which layouts recover and which do not?

Compute:

- per-layout first moved object
- per-layout first lifted object
- final success
- earliest timestep where trajectory becomes target-consistent, if it does

Why this is next:

- it turns the current benchmark-level result into a layout-level map of failure modes

Done when:

- there is a table for each `Scene 1` target showing `layout -> first_move, first_lift, success, recovery_step`

#### E1B. First-chunk divergence summary

Question:
Are same-layout trajectories already different in the first action chunk, or do they only diverge later?

Compute:

- action-chunk distance between targets within the same layout
- early end-effector displacement vectors
- same-layout clustering plots or summary statistics

Why this is next:

- it directly informs the "rigid trajectory" hypothesis

Done when:

- the writeup can say whether the early collapse is caused by nearly identical first chunks or by different chunks passing through the same bad region

## E2. Target Identity Encoding

### Intent

Determine whether target information is represented, and whether that representation is target-selective rather than just prompt-carrying.

### Already established

- pooled target-identity probes are strong across VLM, handoff, expert, and expert flow step 0
- cross-benchmark transfer is asymmetric on overlapping classes
- geometry and relation probes are also strong

Primary references:

- `pi05-interim-findings.md`
- `pi05-experiment2-richer-probes.md`
- `pi05-experiment2-delta.md`

### Next expansion

#### E2A. Selectivity controls

Question:
Is the target privileged, or are all objects in the scene equally decodable?

Compute:

- non-target object identity probes
- non-target object geometry probes
- target-vs-non-target decoding gap by representation family

Why this is next:

- current probe results support representation, but not yet targeting

Done when:

- there is a direct comparison table showing target vs non-target decodability for the same scenes and capture points

#### E2B. Baseline comparison against scene priors

Question:
Do geometry probes outperform a trivial predictor based on object label and benchmark/layout priors?

Compute:

- baseline predictor using object identity plus benchmark
- stronger baseline using object identity plus layout
- probe-minus-baseline performance gap

Why this is next:

- it addresses the main confound in the current geometry result

Done when:

- the geometry writeup can say whether representation-based probes add real information beyond stable scene regularities

#### E2C. Token localization only after controls

Question:
Where does target-selective information live: text tokens, image tokens, or both?

Compute:

- stride-pooled or per-token probes
- target-selective heatmaps by token region

Why this is later inside E2:

- localization is most useful after the selectivity question is settled

Done when:

- there is a token-region summary that can guide later causal patching

## E3. Temporal Commitment / Handoff

### Intent

Determine whether the handoff already contains enough task-relevant information and whether swapping it changes behavior.

### Already established

- target identity is decodable at flow step 0
- benchmark-specific projections grow through denoising
- same-task handoff swaps can induce failure or rescue in some cases

Primary references:

- `pi05-current-directions.md`
- `pi05-handoff-swap-smoke.md`
- `pi05-handoff-rescue-pilot.md`
- `pi05-handoff-rescue-scaled.md`

### Next expansion

#### E3A. Rerun-verified rescue set

Question:
How often does a same-task success-donor handoff rescue a recipient that actually fails in the rerun baseline?

Compute:

- baseline rerun outcome for each selected recipient
- `current_self_path` outcome
- donor-swap outcome
- rescue rate on true failing recipients only

Why this is next:

- it removes the biggest ambiguity in the current handoff-swap story

Done when:

- rescue rates are reported only on verified failing recipients, with task- and layout-level breakdowns

#### E3B. Handoff decomposition

Question:
Which part of the handoff is causally active?

Intervention candidates:

- prefix KV cache only
- prefix hidden states only
- partial-layer swaps
- partial-token-region swaps if token localization becomes available

Why this is next:

- full handoff swap is informative but too coarse to explain mechanism

Done when:

- at least one narrower swap reproduces a substantial fraction of the full-swap effect, or the full swap is shown to be irreducibly distributed

#### E3C. Temporal effect of rescue

Question:
Does rescue change flow-step-0 organization, late denoising organization, or both?

Compute:

- flow-step projection profiles before and after swap
- first-chunk divergence before and after swap
- success/failure direction projections under swap

Done when:

- the project can say whether rescue acts by changing early commitment, later refinement, or the whole trajectory

## E4. Success vs Failure Comparison

### Intent

Characterize what differs internally between success and failure without yet intervening.

### Already established

- mixed-outcome tasks in `Scene 1` show success/failure projection gaps along the benchmark-delta direction
- those directions are positively aligned with explicit success/failure directions

Primary references:

- `pi05-experiment2-success-failure.md`
- `pi05-experiment2-success-failure-direction.md`

### Next expansion

#### E4A. Mixed-outcome vs all-failure comparison

Question:
What separates `alphabet_soup` and `cream_cheese` from `ketchup` and `tomato_sauce`?

Compute:

- benchmark-delta projection summaries for all four tasks
- geometry-probe residuals by task
- first-move-conditioned representation comparisons

Why this is next:

- the all-failure tasks are currently underused despite being the sharpest failures in the benchmark

Done when:

- the writeup can state whether the all-failure tasks look like stronger versions of the mixed-outcome failures or qualitatively different failure modes

#### E4B. Layout-conditioned success/failure analysis

Question:
Do the observed success/failure differences survive after controlling for layout?

Compute:

- within-layout success/failure comparisons where available
- effect sizes with uncertainty intervals

Done when:

- at least one success/failure effect is shown to persist after tighter conditioning than task-level pooling

## E5. First-Move Pattern Analysis

### Intent

Localize the earliest behavioral point where successful and failed policies diverge.

### Already established

- `Scene 1` shows complete first-move collapse toward `ketchup_1`
- some tasks recover from that collapse and others do not

Primary references:

- `pi05-phase2-routing.md`
- `pi05-phase3-scene1-mechanisms.md`

### Next expansion

#### E5A. First action chunk clustering

Question:
Is the bad early behavior coming from nearly identical first chunks?

Compute:

- pairwise distances between first action chunks
- cluster membership by task and layout
- whether success cases cluster with failure cases or form their own subgroup

Done when:

- there is a clear answer to whether the collapse is already present in the first chunk itself

#### E5B. Recovery window analysis

Question:
For recovering tasks such as `cream_cheese`, when does the trajectory stop looking like the failed ones?

Compute:

- earliest timestep where end-effector-to-target distance begins improving relative to failed tasks
- earliest timestep where target-consistent lift becomes likely

Done when:

- the project can point to a specific recovery window instead of saying recovery happens "later"

## E6. Behavioral Memorization / Perturbation

### Intent

Test whether success depends on canonical object arrangement rather than robust task grounding.

### Already established

- hard target swaps collapse `LIBERO_OBJECT`
- `Scene 1` shows object-specific robustness rather than uniform collapse
- `cream_cheese` is unusually robust under the current swap design

Primary references:

- `pi05-phase4-perturbation.md`

### Next expansion

#### E6A. Target-only displacement sweep

Question:
How much target movement can each task tolerate before behavior breaks?

Compute:

- perturbation magnitude curve per task
- success, first move, and first lift under displacement
- collision/intersection validation flags

Why this is next:

- it is cleaner than hard swaps and gives a graded robustness curve

Done when:

- each task has a perturbation-sensitivity profile instead of a single swap result

#### E6B. Same-layout matched perturbation comparisons

Question:
Does `cream_cheese` remain robust under matched perturbations that collapse `alphabet_soup`?

Done when:

- there is a paired comparison showing whether robustness differences persist under equal perturbation severity

## E7. Memorization Localization

### Intent

Use perturbation data to identify where representational failure appears when behavior becomes brittle.

### Already established

- prerequisites are mostly ready: canonical probes exist and perturbation infrastructure exists at least for swaps

### Next expansion

#### E7A. Apply canonical probes to perturbed rollouts

Question:
Under perturbation, what remains stable: target identity, target geometry, both, or neither?

Compute:

- canonical-trained identity probe accuracy on perturbed data
- canonical-trained geometry probe performance on perturbed data
- delta between canonical and perturbed probe outputs

Done when:

- the project can say whether perturbation primarily breaks representation, downstream control, or only final task completion

#### E7B. Perturbation-selectivity analysis

Question:
Does perturbation make distractors more competitive in representation space?

Compute:

- target vs non-target decoding under perturbation
- target-vs-distractor projection gaps before and after perturbation

Done when:

- there is a clear statement about whether perturbation breaks target selectivity or merely action execution

## E8. Causal Steering / Intervention

### Intent

Test interventions that can change outcomes and help localize the causally active computation.

### Already established

- single-direction delta ablation did not rescue failure
- larger delta ablations were destructive
- random perturbation improved mixed-outcome tasks across multiple seeds
- handoff swap is stronger than delta ablation as a causal handle

Primary references:

- `pi05-delta-ablation-pilot.md`
- `pi05-experiment2-diag-ablation.md`
- `pi05-handoff-swap-smoke.md`

### Next expansion

#### E8A. Random-helps characterization

Question:
Why does random perturbation help?

Sweep:

- perturbation magnitude
- intervention layer
- flow-step timing
- subspace restrictions

Measure:

- success rate
- first move / first lift changes
- projection changes onto known success/failure and benchmark-delta directions

Done when:

- random-helps is either localized to a specific regime or shown to be a broad regularization effect

#### E8B. Narrower-than-full handoff interventions

Question:
Can a smaller intervention reproduce the handoff-swap effect?

Candidates:

- KV-only swap
- hidden-state-only swap
- partial prefix swaps
- top-feature subspace swaps once localization is stronger

Done when:

- the current strongest causal effect is narrowed to a more specific representational object than "the whole handoff"

#### E8C. Deprioritized path

Do not prioritize classic vector steering right now.

Reason:

- the single benchmark-delta direction was not a good causal handle
- future steering should be informed by narrower causal localization first

## Recommended near-term order

If a future agent wants the best next sequence, do this:

1. `E3A` rerun-verified same-task handoff rescue set
2. `E2A` target vs non-target selectivity controls
3. `E5A` first action chunk clustering
4. `E6A` target-only displacement sweep
5. `E7A` apply canonical probes to perturbation captures
6. `E8A` random-helps characterization

That ordering keeps the project focused on the current central uncertainty:

> PI0.5 seems to have substantial target information, but its downstream scene-conditioned control is brittle and heterogeneous.
