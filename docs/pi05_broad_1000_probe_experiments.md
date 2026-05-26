# PI0.5 Broad 1000 Probe Experiments

This document is the review surface for the PI0.5 broad 1000 probe campaign.
The probes are not new capture. They train on the existing mech-light activation
features plus post-processed interaction labels.

## Dataset Contract

- Dataset root: `/mnt/new-volume/vla-lens/pi05-broad-1000-mech-light-lerobot-v3`
- Dataset shorthand: mech-light
- Actual capture profile: `mechanistic_sampled`
- Split column: `split`
- Train split: `train`
- Selection split: `val_heldout_task`
- Final report split: `test_heldout_task`
- Primary model: linear probe
- Secondary capacity check: one-hidden-layer MLP
- Primary classification metric: balanced accuracy
- Saved diagnostics: predictions, per-split metrics, per-group metrics, null metrics, metadata baselines, linear weights when available
- Runtime contract: normal repo work, saved-trace analysis, probe training, UI
  work, and tests use `.venv` / `uv run`. PI0.5 execution work uses
  `.venv-pi05-*` plus wrapper scripts. "Execution work" means capture,
  model forward/replay, LIBERO/LeRobot env execution, hardware model loading,
  or writing real `.vlatrace` files.
- Capture preflight contract: do not infer the current PI0.5 execution
  environment from docs alone. Run `scripts/check_pi05_env.sh --backend ...` before
  capture, forward replay, or other PI0.5 execution work.
- Provenance contract: durable analyses should save trace IDs, source episode
  fingerprints, feature/target/row-index fingerprints, split definitions,
  selector/spec, method, metrics, and display payloads in VLA-lens artifacts.
- Dataset immutability contract: probe artifacts should record dataset
  manifest/version, trace inventory hash or equivalent fingerprint, label
  generation script/config version, post-processing config, source schema
  versions, row counts after filters, class balance, and excluded-row counts.
- Trust-gate contract: before training or interpreting broad-1000 probes, run
  `uv run python scripts/validate_vla_lens_dataset_trust.py "/mnt/new-volume/vla-lens/pi05-broad-1000-mech-light-lerobot-v3"`.
  The gate is local and read-only; it checks schema/overlay validity, split
  sidecars, activation coverage, outcome balance, and artifact freshness.
- Latest local gate: passed on 2026-05-26 with 1000 episodes, 34000 activation
  site rows, 1.0 activation coverage, train/val/test split counts of
  600/200/200 episodes, and 7 checked artifacts.

## Artifact Contract

Every replicated experiment should become a VLA-lens artifact, not a loose
notebook or one-off result table.

Required fields:
- source trace references and episode IDs
- activation query / selector spec with module, layer, tensor type, token kind,
  axes, shape, timestep alignment, model-call alignment, generation step, and
  reduction
- label selector and row filters
- split definition and split fingerprints
- model/loss/regularization config
- null metrics and metadata baselines
- per-split metrics, per-group metrics, confusion matrices or regression
  residual summaries as appropriate
- predictions with row IDs so the UI can show prediction traces and failure
  decompositions
- feature/target/cache fingerprints so cached feature tables are reproducible
- status: `planned`, `completed`, `superseded`, `blocked`, or `archive_only`
- claim level: `integration_smoke`, `decodable`, `candidate_mechanism`, or
  `causal_intervention`

Operational meaning:
- `decodable` requires held-out evaluation, null comparison, metadata baseline,
  and enough class support after filtering.
- `candidate_mechanism` additionally requires localization consistency and a
  concrete intervention/replay plan.
- `causal_intervention` requires replay reproduction, tensor/site preflight,
  controls, and rerun-verified behavior.

The code-level claim gate lives in `vla_lens.research_guardrails`. It classifies
probe artifacts into `integration_smoke`, `decodable`, `candidate_mechanism`,
or `causal_intervention` from required evidence fields, and rejects overclaims
when an artifact declares a stronger level than its saved evidence supports.

## Guardrail Commands

Run these before changing broad-1000 configs, episode plans, or claim language:

```bash
uv run python scripts/lint_research_guardrails.py --root .
uv run python scripts/lint_research_guardrails.py \
  --root . \
  --episode-plan "/path/to/episode_plan.csv"
uv run python scripts/validate_vla_lens_dataset_trust.py \
  "/mnt/new-volume/vla-lens/pi05-broad-1000-mech-light-lerobot-v3"
```

For a future audit/circuit capture, start from
`configs/pi05_audit_circuit_capture_contract.template.yaml` and lint it:

```bash
uv run python scripts/lint_research_guardrails.py \
  --root . \
  --audit-contract path/to/audit_contract.yaml
```

This is a planning/check contract only. It is not a capture command and must not
be treated as approval to collect broad audit data.

UI meaning:
- Probe artifacts should support layer x call/time heatmaps, metrics tables,
  prediction traces, confusion/failure decomposition views, and source-episode
  drilldown.
- Attribution/intervention artifacts should support camera frames,
  token-to-pixel or bin maps, scalar patch scores, and scrub controls over
  timestep, layer, and head/site.
- Action-generation artifacts should preserve generated trajectories, final
  action chunks, executed actions, and receding-horizon comparisons when those
  views exist.

## Completed VLA-Lens Artifacts

These are the probe artifacts already present on the broad-1000 dataset.

### Target Moved - Expert Action Hidden

- Artifact: `probe_suite-pi0.5-broad-1000-target-moved---expert-action-hidden-ca5380446b`
- Spec: `configs/probes/pi05_broad_1000_target_moved_expert_action_hidden.yaml`
- Target: `target_moved`
- Feature: `pi05.expert.layers.*` hidden action tokens
- Result: promising decodability candidate.
- Best score: `0.749`
- Metadata baseline: `0.660`
- Delta: `+0.089`
- Null p-value: `0.048`
- Best site: `layer=4.0, policy_call_index=6`
- Interpretation: this clears the metadata baseline on the selection split and
  is worth inspecting in the probe-suite UI before deciding whether to run a
  causal follow-up or a neighboring target-contact/lift probe.

### Target Contacted - Expert Action Hidden

- Artifact: `probe_suite-pi0.5-broad-1000-target-contacted---expert-action-hidden-b8c82b1877`
- Spec: `configs/probes/pi05_broad_1000_target_contacted_expert_action_hidden.yaml`
- Target: `target_contacted`
- Feature: `pi05.expert.layers.*` hidden action tokens
- Result: decodable object-interaction signal, not causal evidence.
- Best score: `0.669`
- Metadata baseline: `0.578`
- Delta: `+0.091`
- Null p-value: `0.048`
- Best site: `layer=0.0, policy_call_index=6`
- Selection split: `val_heldout_task`
- Final held-out split aggregate balanced accuracy: `0.804`
- Source episodes: `1000`
- Training rows: `27880`
- Prediction rows: `22270`
- Target distribution: `False=18845`, `True=9035`
- Interpretation: this clears the strongest metadata baseline on the selection
  split and survives final held-out-task aggregation. Per-group performance is
  uneven, so treat this as a candidate for UI inspection and localization work,
  not as proof that the site causally controls target contact.

### First Moved Is Target - Action Head Output

- Artifact: `probe_suite-pi0.5-broad-1000-first-moved-is-target---action-head-output-6051e97b3f`
- Former spec: unfiltered action-head-output first-moved target probe
  (deleted as superseded cleanup)
- Target: `first_moved_is_target`
- Feature: `pi05.action_head.output`
- Result: weak/superseded.
- Best score: `0.688`
- Metadata baseline: `0.930`
- Delta: `-0.243`
- Null p-value: `0.905`
- Interpretation: do not replicate this exact form first. The activation probe
  underperformed metadata baselines, and the unfiltered first-event label is now
  superseded by filtered first-event target-relative probes.

### Outcome Robust - Action Head Input

- Artifact: `probe_suite-pi0.5-broad-1000-outcome-robust---action-head-input-7f7ec8ae54`
- Spec: `configs/probes/pi05_broad_1000_outcome_action_head_input_robust.yaml`
- Target: `outcome`
- Feature: `pi05.action_head.input`
- Result: useful integration/UI artifact, not strong mechanistic evidence.
- Best score: `0.663`
- Metadata baseline: `0.668`
- Delta: `-0.0047`
- Selected split: `val_heldout_task`
- Selected model: linear
- Interpretation: keep as a schema/UI smoke artifact. Outcome remains highly
  confounded by task/benchmark difficulty, so this is not a priority scientific
  replication unless the UI needs a known probe artifact.

## Active Next Run

The first integration probe and two object-interaction probes have been run.
Do not rerun the outcome action-head-input, target-moved expert-action-hidden,
or target-contacted expert-action-hidden probes unless the UI/artifact loader
needs a regression check.

Recommended next work after UI review:
- Inspect the target-contacted artifact in the dashboard/workbench, especially
  `layer=0.0, policy_call_index=6`, prediction traces, confusion slices, and
  low-scoring groups.
- If the goal is physical interaction localization, run neighboring-window
  analysis around the target-contacted best site before proposing any replay or
  intervention.
- If the goal is a new broad probe, choose `target_lifted` only after checking
  class support, or choose filtered `first_moved_is_target` for target-relative
  binding.

Current interpretation: target contact is decodable above metadata and null
baselines, but the evidence is still observational. The next claim upgrade
requires localization consistency plus a concrete replay/intervention plan.

## Feature Contracts

### Expert Action Hidden

Used for target movement/contact/lift and first-event probes.

```yaml
module: pi05.expert.layers.*
tensor_type: hidden_tokens
token_kind: action
layers: [0, 4, 8, 12, 17]
timesteps: all
policy_calls: [0, 1, 2, 3, 4, 5, 6]
generation_step: final
reduction: mean
dtype: float32
sweep: [layer, policy_call_index]
```

Interpretation: action-token hidden representation from selected expert layers,
mean-pooled over action tokens, at the final generation step, evaluated across
early policy calls.

Run decisions:
- Accepted default for broad-1000 first pass: final generation step, early
  policy calls `0-6`, and mean pooling over action tokens.
- Future sweep: generation-step curves and horizon-token-specific probes if the
  first pass shows a real activation-over-metadata signal.
- Decision needed before causal follow-up: exact call/phase window for any
  intervention candidate.

### Action Head Input

Used for the first single-probe run because it is cheaper than the full expert
layer x policy-call sweep.

```yaml
name: pi05.action_head.input
module: pi05.action_head
tensor_type: action_head
token_kind: action
timesteps: all
policy_calls: [0, 1, 2, 3, 4, 5, 6]
generation_step: final
reduction: mean
dtype: float32
sweep: policy_call_index
```

Interpretation: representation entering the action head, mean-pooled over action
tokens, tested across early policy calls.

Run decisions:
- Accepted role: cheap integration and downstream-control feature.
- Caveat: it is close to action production and outcome labels are
  task-confounded, so positive results require metadata-baseline inspection
  before interpretation.

### VLM Prefix Hidden

Used for target-parse-status probing.

```yaml
module: pi05.vlm.layers.*
tensor_type: hidden_tokens
token_kind: prefix
layers: [0, 4, 8, 12, 17]
timesteps: all
policy_calls: [0]
generation_step: final
reduction: mean
dtype: float32
sweep: layer
```

Interpretation: prefix-side representation, mainly to test whether prompt/target
parse state is visible upstream.

Run decisions:
- Accepted default: use `policy_calls: [0]` for prefix-side static parse
  inspection on this capture.
- Future sweep: compare VLM prefix, expert prefix, text-token, image-token, and
  action-token positions if target-parse status becomes a primary question.

## Experiment Registry

### 1. Target Moved - Expert Action Hidden

- Spec: `configs/probes/pi05_broad_1000_target_moved_expert_action_hidden.yaml`
- Status: completed / inspect before causal follow-up
- Artifact: `probe_suite-pi0.5-broad-1000-target-moved---expert-action-hidden-ca5380446b`
- Target: `target_moved`
- Feature: expert action hidden
- Sweep: layer x policy call
- Baselines: majority, benchmark, task id, scene family, task verb, primary target object
- Gate: require adequate class support in each split and activation score above
  best metadata baseline by a meaningful margin before calling it decodable.
- Purpose: test whether the representation predicts whether any target object moved.
- Suspicious result if: metadata baseline matches or beats the activation probe, or signal exists only on train/val and disappears on heldout task.

### 2. Target Lifted - Expert Action Hidden

- Spec: `configs/probes/pi05_broad_1000_target_lifted_expert_action_hidden.yaml`
- Status: planned / run after target-moved if class support is adequate
- Target: `target_lifted`
- Feature: expert action hidden
- Sweep: layer x policy call
- Baselines: majority, benchmark, task id, scene family, task verb, primary target object
- Gate: require enough positive lifted examples per split; otherwise demote to
  casebook/control rather than headline probe.
- Purpose: test whether representation predicts successful vertical target interaction.
- Suspicious result if: class imbalance is severe, or the probe mostly learns benchmark/task identity.

### 3. Target Contacted - Expert Action Hidden

- Spec: `configs/probes/pi05_broad_1000_target_contacted_expert_action_hidden.yaml`
- Status: completed / inspect before causal follow-up
- Artifact: `probe_suite-pi0.5-broad-1000-target-contacted---expert-action-hidden-b8c82b1877`
- Target: `target_contacted`
- Feature: expert action hidden
- Sweep: layer x policy call
- Baselines: majority, benchmark, task id, scene family, task verb, primary target object
- Gate: require contact-label quality audit because contact can be noisy and
  object-family dependent.
- Purpose: test whether representation predicts contact with the target object.
- Result: clears strongest metadata baseline on the selection split by `+0.091`
  balanced accuracy and has final held-out-task aggregate balanced accuracy
  `0.804`.
- Suspicious result if: contact labels are noisy, dominated by one object family,
  or the site does not survive localization/neighboring-window checks.

### 4. First Moved Is Target, Filtered - Expert Action Hidden

- Spec: `configs/probes/pi05_broad_1000_first_moved_is_target_filtered_expert_action_hidden.yaml`
- Status: planned / target-binding fallback next probe
- Target: `first_moved_is_target`
- Row filters: require `first_moved_object` and exclude `ambiguous_first_moved`
- Feature: expert action hidden
- Sweep: layer x policy call
- Baselines: majority, benchmark, task id, scene family, task verb, primary target object
- Gate: save filtered row count, dropped-row reasons, and class balance before
  training; abort or demote if filtering leaves a narrow task/object slice.
- Purpose: test whether representation predicts whether the first moved object is the target.
- Suspicious result if: most examples are filtered away, or object identity baseline dominates.

### 5. First Lifted Is Target, Filtered - Expert Action Hidden

- Spec: `configs/probes/pi05_broad_1000_first_lifted_is_target_filtered_expert_action_hidden.yaml`
- Status: planned / likely sparse
- Target: `first_lifted_is_target`
- Row filters: require `first_lifted_object` and exclude `ambiguous_first_lifted`
- Feature: expert action hidden
- Sweep: layer x policy call
- Baselines: majority, benchmark, task id, scene family, task verb, primary target object
- Gate: require enough lifted rows and both classes in held-out splits.
- Purpose: test whether representation predicts whether the first lifted object is the target.
- Suspicious result if: lifted examples are sparse or concentrated in a small task family.

### 6. Outcome - Action Head Input

- Spec: `configs/probes/pi05_broad_1000_outcome_action_head_input_robust.yaml`
- Status: completed integration smoke
- Artifact: `probe_suite-pi0.5-broad-1000-outcome-robust---action-head-input-7f7ec8ae54`
- Target: `outcome`
- Feature: action head input
- Sweep: policy call
- Baselines: majority, benchmark, task id, scene family, task verb, primary target object
- Gate: already failed to beat metadata baseline meaningfully; keep as UI/schema
  artifact, not scientific positive evidence.
- Purpose: cheap first end-to-end probe artifact for UI inspection and split/metric validation.
- Suspicious result if: task/benchmark metadata baseline explains the score, because outcome is especially confounded by task difficulty.

### 7. Target Parse Status - VLM Prefix Hidden

- Spec: `configs/probes/pi05_broad_1000_target_parse_status_vlm_prefix_hidden.yaml`
- Status: planned control
- Target: `target_parse_status`
- Feature: VLM prefix hidden
- Sweep: layer
- Baselines: majority, benchmark, task id, scene family, task verb
- Gate: require enough parse failures; otherwise report prevalence and demote
  to dataset-quality metadata.
- Purpose: test whether target parsing quality/status is visible in prefix representations.
- Suspicious result if: parse failures are too rare to support a meaningful classifier.

## Completed First Probe

```bash
uv run python scripts/train_vla_lens_probe.py \
  "/mnt/new-volume/vla-lens/pi05-broad-1000-mech-light-lerobot-v3" \
  --spec configs/probes/pi05_broad_1000_outcome_action_head_input_robust.yaml
```

Why this was trained first:
- It is the smallest campaign probe.
- It exercises the new artifact schema, val/test split handling, model comparison, metrics, baselines, and UI artifact loading.
- It is not the most scientifically interesting probe, but it is the best first integration check.

Result: completed as
`probe_suite-pi0.5-broad-1000-outcome-robust---action-head-input-7f7ec8ae54`.
It should now serve as a regression/UI artifact. The target-moved and
target-contacted probes have both been run; use `Active Next Run` for the
current follow-up recommendation.

## Superseded YAML Specs

These specs are useful history but should not be treated as the current
replication plan.

### Unfiltered First-Moved Target, Action-Head Output (deleted)

- Status: superseded by filtered first-event target probes.
- Reason: already ran as a VLA-lens artifact and underperformed metadata
  baselines.
- Preserve idea: action-head output can be used as a downstream control, but
  first-event labels should be filtered for present/non-ambiguous event rows.

### Raw First-Moved Object, Expert Action Hidden (deleted)

- Status: lower priority / likely superseded.
- Reason: raw object multiclass labels are harder to interpret than
  target-relative labels such as `first_moved_is_target`.
- Preserve idea: raw object decoding can be a control for object identity
  availability, not the primary target-binding claim.

### Non-Robust Outcome, Action-Head Input (deleted)

- Status: superseded by the robust action-head input outcome spec.
- Reason: the robust spec adds validation/test evaluation, model metadata,
  richer baselines, final generation-step selection, and artifact schema v3.

### Outcome, Expert Action Hidden (deleted)

- Status: low priority.
- Reason: outcome labels are task-confounded. This may still be useful as a
  control, but interaction labels are more directly tied to target binding.

### Task Identity, VLM Prefix Hidden (deleted)

- Status: control, not primary experiment.
- Reason: task identity decoding is expected and can expose leakage/confounds.
- Preserve idea: useful as a sanity/control artifact for instruction/task
  information in prefix representations.

## Legacy Ideas To Preserve

The pre-VLA-lens artifacts and notes contain useful design ideas. Most should
not be rerun exactly. They should be translated into VLA-lens artifacts only
when they answer a current question or serve as a gate/control for probe
interpretation.

### Required Gates Before Strong Claims

- **Metadata leakage audit.** Preserve unsafe-field checks and metadata-only
  priors before interpreting activation probes. In older controls,
  `target_guess == object_label` in all rollouts, and fields like `task_id`,
  `object_label`, `target_guess`, `layout_id`, and `task_id+layout_id` can
  inflate results.
- **Held-out split discipline.** Layout splits are weak smoke tests because
  they can preserve task/benchmark/scene-template structure. Prefer held-out
  task, benchmark, scene-family, or stricter held-out-layout gates depending on
  the question.
- **Success is not clean target binding.** Several old audits showed successful
  or target-lifted rollouts can still have messy hidden/flow chains. Donor
  selection must use chain-cleanliness criteria, not outcome alone.
- **Replay/preflight gates for causal tracing.** Tensor preflight and replay
  checks are prerequisite artifacts, not causal evidence. They should be stored
  as readiness/provenance artifacts attached to later causal traces.
- **Coverage before experiment design.** Dataset browsing/coverage should be a
  first-class validation step before training: success/failure balance, task
  coverage, activation coverage, episode length, call density, object coverage,
  and filtered-label support.
- **Episode-safe rows.** Samples within an episode are correlated. Probe splits
  must be episode/task/layout safe for the claim; row-level IID splits are not
  acceptable for scientific interpretation.

### VLA-Lens Workflow And Selector Contracts

- Preserve the architecture boundary: capture/import normalizes raw model/env
  outputs into episode-aligned trace bundles; probe suites operate later through
  capture-store queries / feature views; artifacts are registered back into the
  dataset index.
- The durable unit is the trace bundle plus artifact, not a loose experiment
  output. Core indexes should separate episode, timestep, activation, and
  artifact metadata.
- Prefer named primitives over bespoke scripts:
  capture-store indexing, activation selection, probe dataset construction,
  metadata baseline comparison, intervention specs, manifests, and artifact
  registration.
- Preserve selector semantics:
  - `.mean` selectors produce one vector per row/call and are appropriate for
    first-pass probes.
  - `.flat` selectors preserve action chunks, flow states, or KV/attention
    shapes where axis structure matters.
  - selector/cache keys must include source trace identity, activation query,
    reduction, labels, filters, and split fingerprints.
- Preserve `InterventionSpec` / `kv_rescue`-style records as auditable causal
  trace inputs. They are not equivalent to successful execution until replay,
  shape/site checks, controls, and rerun behavior are attached.
- The important PI0.5 interface is the PaliGemma prefix `past_key_values`
  entering the Gemma expert denoising loop, not a single conditioning vector.
  Do not draw or claim fake all-to-all VLM-to-Expert attention paths.

### Capture Profile Ladder

- `rollout`: behavior-only questions.
- `features`: cheapest sufficient profile for broad decodability probes.
- `mechanistic_sampled`: current broad-1000 profile; use for normal VLA-lens
  inspection, landmark layers, attention routing, K/V cache summaries, and
  action-head I/O.
- `mechanistic_all`: all-layer localization or checking whether sampled
  landmark layers missed a transition.
- `audit_sampled`, `audit_windowed`, `audit_full`: narrow causal/circuit
  follow-ups only. Do not scale these across broad 1000 without a specific
  circuit question and storage/runtime budget.
- Causal ordering for broad follow-up work: probe grid -> attention/routing maps ->
  counterfactual pairs -> no-intervention replay -> patching/steering -> sparse
  dictionaries or transcoder-style work.

### Behavior And Failure Taxonomy

- Preserve the full object-binding chain as the central label ontology:
  requested object -> model/internal selected object -> action-suggested object
  -> first moved object -> first lifted object.
- Preserve the broader semantic-to-motor chain as a lens for artifact design:
  object available -> object used -> object dominant -> correct motion ->
  successful manipulation. Probe rows should make clear which link they label.
- Prefer clean wrong-object moved/lifted cases over generic success/failure
  probes when asking target-binding questions.
- Compute Scene 1 success from `max_reward >= 1.0`; older env-state-after-reset
  success reads are superseded.
- Preserve aggregate scene-family and failure-type baselines from the
  target-binding-control corpus: `wrong_object_lifted`, `wrong_object_moved`,
  `approach_failure`, target-distance failures, and mixed-outcome task groups.
- Preserve failure-case selection logic for task-level mixed-outcome groups.
  Exact layout cells often had only one rollout, so task-level reruns are more
  actionable than layout-level many-seed claims.
- High-value candidate family from legacy notes: `living_room_scene_4`, task
  `61`, target `chocolate_pudding_1`, because it had mixed outcomes and clear
  wrong-object failure modes.
- Preserve Scene 1 as a structured partial-success benchmark:
  `cream_cheese` can recover despite wrong early routing, `ketchup` can move
  target-first but fail later, and `tomato_sauce` fails from early routing
  onward in older notes.
- Preserve target-swap / same-layout contrast conditions as first-class dataset
  metadata. These test whether behavior follows language, visual location, or
  learned object priors.
- Preserve the target-binding capture-plan idea as planning provenance: broad
  diversity matters more than many seeds per task, with held-back episodes and
  family/tier metadata saved explicitly. Do not resurrect the older
  many-seeds-per-task plan as the default unless the question is variance or
  layout stability.

### Donor / Recipient / Patch Readiness

- Preserve donor eligibility tiers:
  - strict chain-correct donor
  - success but no strict donor
  - target-lifted but chain-messy
- Preserve recipient tiers:
  - strict chain-wrong recipient
  - ambiguous failure
- Preserve donor/recipient matchmaker outputs as pair-manifest artifacts:
  candidate count, strict-valid count, criteria, controls, and source rows.
- Preserve patch-manifest readiness as a non-causal artifact with tensor shapes,
  layer/call/phase grid, controls, and pass/fail status.

### Object Binding And Flow Probes

- Preserve the refined flow-binding idea, not the original broad version:
  compare target action direction against semantic distractors separately from
  receptacle/destination/support objects.
- Preserve EEF displacement calibration because action XYZ is policy
  action-space; cosine direction is safer than metric magnitude.
- Preserve candidate-object class definitions, fallback provenance, ambiguity
  rows, and grouped summaries.
- Do not treat broad "best non-target" margins as final evidence; receptacles
  and supports can mask object-binding behavior.
- Preserve behavioral routing baselines as canonical context: confusion tables
  for first moved/first lifted object, success, steps, VLM calls, and expert
  calls by benchmark/task/object.
- Fine-grained attention binning is useful as a descriptive artifact tied to
  routing, especially when machine-readable token/bin definitions are saved.
  Coarse attention summaries are superseded by finer bins for localization.
- Preserve layer/phase object probes as non-causal localization maps:
  first moved/lifted object by saved hidden state/action chunk, layer, call, and
  feature family. Do not treat call/layer dips as mechanisms without replay or
  intervention.
- Preserve flow outcome probe curves only with structured priors. A flow probe
  should pass only if it beats constant, object, and object+phase priors, not
  merely the constant baseline.
- Preserve flow-probe evolution over denoising steps as a probe-history
  artifact: call index, flow step, target, metric, baseline, sample count,
  class balance, and phase bin. Legacy notes suggest target identity and
  geometry become more decodable late in denoising, especially call `00`, but
  success probes are often imbalanced or prior-dominated.

### Object Presence / VLM Controls

- Preserve object-presence probes as controls, not target-binding evidence.
- Useful contract:
  - multi-label linear head over object vocabulary
  - BCE-with-logits
  - per-object positive weights
  - AUROC/AP only when train/test positives and negatives are sufficient
  - per-object and fold-level validity counts
- Useful feature ideas:
  - early/intermediate VLM layers
  - image/text/all token pooling
  - local image-window pooling such as windows64
  - instruction-token embeddings as a language-leakage control
- Legacy result pattern: benchmark and scene-family holdouts were modest
  overall, around `0.52-0.62` mean AUROC depending on feature/split. Layout
  splits that looked near-perfect should be considered weak/superseded for
  interpretation.
- Preserve target-identity probe history separately from object-presence
  controls. Earlier pooled VLM/handoff/expert/flow probes found near-perfect
  target identity decoding on narrower canonical captures, but these were a
  breadth-first first pass and are now controls unless paired with stronger
  target-vs-distractor and split gates.
- Preserve stride/window pooling as a localization stepping stone before
  per-token sweeps.

### Geometry Controls

- Preserve geometry controls as guardrails against layout shortcuts.
- Important warning: layout plus task can effectively key individual rollouts
  in-sample.
- Benchmark confounds remain a central interpretation risk. A high probe score
  can mean the model knows the benchmark/task/layout prior, not that it uses
  grounded target information.
- Preserve same-layout contrasts, pose-reuse drift, target-distractor
  separation, shortcut-prior RMSE/top-1 metrics, and links to source
  rollout/object rows.
- Preserve richer geometry/relation probes as controls:
  - `target_pos`
  - `target_to_gripper`
  - `target_to_basket`
- Preserve target-vs-distractor geometry probes with degeneracy flags. Any
  zero-variance label dimensions or impossible `R2=1, MAE=0` rows must be
  flagged before ranking results.
- Do not claim grounded target use unless a result survives at least one strong
  guardrail:
  - metadata/object/layout prior baseline
  - target-vs-distractor selectivity control
  - perturbation or displaced-object generalization
  - recipient-fixed causal intervention
  - held-out task/benchmark/scene-family split appropriate to the claim

### Layer / Phase / Object Probes

- Preserve strict held-out-layout layer/phase/call probes as a design idea.
- Attach split/gate artifacts so results are not mistaken for random-split or
  in-sample probes.
- Preserve the strict metadata-gate contract:
  - same held-out `layout_id` folds for metadata and activation probes
  - categorical probes must beat best metadata baseline by a fixed margin
  - continuous probes must beat metadata/constant baselines by an MAE reduction
    threshold
  - pass/fail rows should be saved, not only best scores
- Legacy strict activation probes reported that object-interaction identity
  targets (`first_moved_object`, `first_lifted_object`) passed more reliably
  than `success`, `failure_type`, `min_target_distance`, or `target_max_lift`.
  Preserve those negative results so outcome probes are not overinterpreted.
- Preserve the old strict-gate thresholds as historical baselines to beat or
  consciously revise:
  - success: `>= 0.9261`
  - failure type: `>= 0.8704`
  - first moved object: `>= 0.8373`
  - first lifted object: `>= 0.7535`
  - target lift height error: `<= 0.0413`
  - closest target distance error: `<= 0.0149`
- Legacy summary: hidden-state probes passed first-moved and first-lifted object
  gates more convincingly than success/failure-type/lift-height/distance gates.
  This supports prioritizing object-chain labels over broad outcome labels.
- Expert hidden families were stronger than compact action/flow pooled features
  in strict held-out-layout object-interaction probes. This supports the current
  robust campaign's focus on expert hidden features.
- VLM call-00 mean-pooled features also carried some object identity signal, but
  not clean success/failure or continuous-control-quality signal.
- Current broad-1000 robust campaign partially inherits this idea by sweeping
  expert layer and policy call, but it does not yet reproduce the full strict
  layer/phase/control setup.

### Capture Schema Lessons From High10 / Pre-VLA-Lens Artifacts

- Preserve inventory-level metadata from deleted/pre-VLA payloads, not old raw
  paths: benchmark, task, instruction, seeds, layout, success, step count, call
  counts, image counts, schema/version, and deletion/provenance notes.
- Validate empty/anomalous imports. Older inventories included tiny malformed
  rollout dirs and suspicious image/call count mismatches.
- Preserve timing/profile metadata as derived runtime artifacts: failed episodes
  often run to timeout while successful episodes terminate early, so steps and
  success must be logged together.
- Preserve the high10 schema idea of one rollout directory containing behavior,
  state trajectory, actions, VLM call tensors, and expert call tensors, but map
  it into `.vlatrace` bundles and VLA-lens artifacts.
- Preserve the non-duplicated cache-reference model: VLM prefix/past-key-values
  should be stored once and expert calls should reference the matching VLM call.
- Preserve denoising-step expert internals as possible future probe axes:
  suffix embeddings, AdaRMS conditioning, residual inputs, hidden states, and
  selected attention maps.
- Preserve explicit model/action dimension metadata so probe artifacts know
  which channel/horizon/action axes they used.

### Causal Trace And Attribution Patching

- First reproduce saved actions with a no-training forward replay before
  trusting intervention results.
- Fix wrong-time / wrong-phase controls before using them as controls; they must
  actually shift time or phase, not accidentally reproduce the same state.
- Preserve scene4 interface-level causal trace as a candidate causal artifact,
  not a generic probe:
  - scene: `living_room_scene_4`
  - task: `61`
  - target: `chocolate_pudding_1`
  - wrong objects: `akita_black_bowl_1/2`
  - key layers from legacy summaries: `8`, `12`, `14`, `16`
  - strongest local signal reported around layer `12` or `14` depending on
    trace variant and metric
- Preserve exact rescue-vs-control comparison. Do not report raw rescue deltas
  without best-control rows.
- Preserve cross-object transfer as mechanistic evidence with caveats, not as a
  broad token-transfer result. Legacy notes suggested mechanism type transfers
  better than exact token IDs, and more raw transfer sweeps are lower value than
  feature-ID work.
- Preserve cumulative token patching and role tests because they distinguish
  "success injection" from "bad feature removal."
- Preserve controls:
  - call-shifted
  - layer-shuffled
  - self/random controls where available
- Preserve replay gate and tensor preflight gate as separate prerequisite
  artifacts.
- Preserve rerun-verified intervention records. Legacy handoff/rescue evidence
  was heterogeneous and sensitive to rerun instability; rescue claims require
  recipient baseline failure in rerun plus success or margin improvement under
  intervention.
- Preserve full handoff swap/rescue artifacts as more promising than
  single-direction delta ablations:
  - donor/recipient task, layout, seed
  - canonical outcome
  - rerun baseline outcome
  - current-self sanity path
  - swap outcome
  - true-rescue vs degradation labels
- Legacy handoff results were heterogeneous: some same-task same-layout failures
  rescued, some did not, and some successful recipients degraded under donor
  handoffs. This argues for artifact-level case tables, not single headline
  claims.
- Preserve phase-trajectory intervention readouts. Same-layout handoff smoke
  suggested early approach/close metrics can look similar while lift diverges,
  so causal artifacts should report approach, close/grasp timing, lift, and
  recovery phases rather than one action-vector metric.
- Preserve attribution patching as two levels:
  - coarse K/V prefix groups
  - binned/spatial token localization
- Higher-priority localization idea from legacy notes: layer `14` value stream,
  vision bins around `09-11` and `15` of `24`, and tokens such as `331`, `327`,
  `323`, `347`, `330`. These need manifest-backed token metadata and a clear
  disclaimer that visual overlays are grounding aids, not proof by themselves.
- Scene 4 remains the preferred first model-change/intervention family. Scene 3
  task 59 remains useful autopsy material unless new clean good examples are
  collected.

### Negative Controls / Do Not Accidentally Reuse

- Scene 3 task 59 is a donor-pathology case in the old notes. It had successes,
  but strict hidden-flow target agreement was absent in the audited cards. Do
  not use those successes as clean rescue donors unless a new donor audit
  changes the eligibility status.
- Expert-hidden scene4 trace was a negative/control result in one legacy run:
  no tested layer rescued every pair. Preserve this as a negative artifact if
  replicated, not as a positive mechanistic claim.
- Coarse attention localization and reduced mean prefix features were weak in
  the old sweep. Preserve as low-priority controls, not primary evidence.
- Broad success/failure summaries and broad action summaries are too vague for
  current target-binding claims. Prefer object-chain, phase, and intervention
  artifacts.
- Single shared delta-direction ablation at `suffix_out -> action_out_proj` was
  negative/ambiguous. Do not rerun "more of the same" as a priority; preserve it
  as negative intervention history and prefer handoff swaps or more specific
  causal traces.
- Strong benchmark-delta ablation collapsed performance in older notes, while
  random perturbations sometimes helped. Preserve the updated conclusion:
  benchmark-delta directions are observational/load-bearing structure, not a
  clean rescue handle by themselves.
- Single-pair offline KV attribution smoke produced no action movement in one
  old run. Do not scale that exact setup until clean/corrupt baselines actually
  move the measured trajectory; preserve it as a negative-control artifact.

### Benchmark / Directional Probe History

- Preserve benchmark classifiers on overlapping object classes as explicit
  benchmark-leakage/domain-separability artifacts.
- Useful fields: held-fixed object classes, split sizes, model family, accuracy,
  balanced accuracy, regularization, top dimensions, and top weights.
- Legacy result pattern: benchmark separability was extremely high across VLM,
  handoff, expert-final, and expert-flow features. This strengthens the warning
  that benchmark/domain structure can dominate probe results.
- Preserve cross-benchmark delta analyses as observational artifacts:
  delta norms, shared cosine, top dims, projections by success/failure, and
  caveats that shared directions are not causal failure mechanisms by default.
- Preserve success/failure direction-overlap probes with counts, direction
  orientation, cosine overlap, projection means, test statistic, and p-value.
- Preserve paired same-task/same-layout success-vs-failure divergence artifacts:
  rollout IDs, match criteria, matched call counts, representation cosine by
  layer, flow-step hidden cosine/MSE, attention JS/cosine, final action
  divergence, and object-event summaries.

### Archive / CogACT-Era Ideas Worth Keeping Abstractly

- Do not replicate CogACT-specific hook names, DDIM assumptions, Prismatic/Llama
  details, or the exact `10` denoising-step / `16` action-chunk axes as if they
  applied directly to PI0.5.
- Preserve the abstract experiment families:
  - matched-scene probes
  - layer/stage sweeps for when intent or selected-object information appears
  - accuracy-over-generation-step curves
  - early-vs-late temporal transfer probes
  - episode-level confidence curves for success/failure trajectories
  - additive steering/intervention using probe directions
  - baseline-vs-intervention rollouts
  - safety-monitor probes that detect intended target/action before execution
- Preserve probe controls from the old intent-probe plan:
  - shuffled labels
  - language-only or blank-image baselines
  - cross-seed / cross-episode generalization
  - failure decomposition tables separating probe-correct/task-success,
    probe-correct/task-failure, probe-wrong/task-success, and
    probe-wrong/task-failure
- Preserve the methodological warning that samples within one episode are
  correlated. Splits must be episode/task/layout-safe, not row/step IID.
- Treat archived docs as idea provenance only. Active PI0.5 execution must use
  current capture/env docs and ROCm wrapper contracts.

### Literature / General Probe Design Ideas

- Preserve linear probes as the first-pass localization method. MLP probes are
  capacity checks, not the main interpretability claim, because they can weaken
  locality and make metadata leakage harder to reason about.
- Preserve middle-layer hypotheses: world/semantic/task information often peaks
  before final layers, while final/action-suffix positions can better reflect
  action commitment.
- Preserve token-position comparisons when the question calls for them:
  language/color/instruction tokens, vision tokens, prefix/EOS/final tokens, and
  action suffix tokens can answer different questions.
- Preserve cross-modal comparisons. Visual pathways may dominate many action
  decisions, while instruction-disambiguated tasks are useful exceptions.
- Preserve failure decomposition tables: `probe_correct x task_success`,
  `probe_correct x contacted_object`, and `probe_correct x first_moved/lifted`
  help separate perception/selection failures from motor execution failures.
- Preserve calibrated-monitoring ideas as future safety/control work:
  calibration split, conformal scores, prediction-set size, critical-window or
  sliding-window pooling, and pre-grasp alerts. Do not average uncertainty over
  whole trajectories when the claim is phase-local.
- Avoid Gaussian/noise corruption as a primary causal corruption. Prefer valid
  semantic corruptions: prompt target swaps, paired same-scene/same-layout
  counterfactuals, object-pose swaps, or recipient-fixed interventions.
- SAE/transcoder/sparse-dictionary work belongs after localization identifies a
  concrete layer/site/question; it is not a default broad-1000 replication step.

### Current-State Circuit Ideas

- Preserve these as future circuit questions, not default broad-capture work:
  - Expert MLP / skip-transcoder hypothesis around layer `8`
  - `audit_windowed` pair showing Expert L8 writes a feature consumed by L9
  - object-grounded attention routing that predicts action direction without
    being treated as causal by itself
- These require `audit_sampled` or `audit_windowed` only when tied to a concrete
  circuit question. They should not be rolled into the broad robust probe
  campaign by default.

### Capture-Cost And Audit Profile Lessons

- Preserve `audit_sampled` and `audit_windowed` smoke history as capture-cost
  planning artifacts, not as probe experiments.
- Useful fields for future capture-cost artifacts: steps, policy calls, success,
  wall clock, RSS, trace size, model sites, runtime members, architecture edges,
  and model storage grouped by family.
- `audit_windowed` is materially more expensive than `audit_sampled`; old notes
  estimated roughly 2x trace size for some object smokes. Do not request broad
  audit-windowed capture without a concrete circuit question.

### Dataset Diversity Lessons

- Preserve the diverse-capture motivation: probes trained only on
  `LIBERO_OBJECT` / Scene 1 can learn benchmark, task, scene, or layout priors.
- Preserve task-tier labels:
  - `clean_single_object`
  - `secondary_object`
  - `exclude_for_object_probes`
- First object-position/object-binding probe datasets should use clean
  single-object tasks. Secondary or multi-object tasks should be analyzed
  separately.
- Preserve benchmark/task/layout/seed/success/object-state metadata and object
  list extraction. Older hard-coded assumptions like every task having
  `basket_1` caused failures and should not reappear.
- Preserve activation storage/cost accounting by component; older diverse pilot
  notes found VLM activations dominated storage.
- Preserve the many-seed control capture plan as a proposed control artifact,
  not completed evidence: repeated seeds, task IDs, layouts, contrast
  structure, and current `episode_plan.csv` schema.

### Positive-Control Benchmarks

- Preserve `LIBERO_OBJECT` as a routing positive control for the pretrained
  PI0.5 model: high canonical performance and clean first-moved/first-lifted
  target routing in older notes.
- Preserve Scene 1 / `LIBERO_90` as a partial-success / structured-failure
  benchmark, not as generic failure. Difficulty is object-dependent.
- Do not generalize canonical `LIBERO_OBJECT` success to perturbation/layout
  robustness. Target-distractor swaps collapsed in legacy notes and should be
  linked as separate perturbation artifacts.
- Preserve hard target-pose swaps as anti-memorization tests, with swap partner
  metadata and warnings about possible simulation intersections. Use success,
  first moved, and first lifted as more reliable summary fields than raw lift
  magnitude alone.
- Preserve sanity-check status for benchmark validity:
  ketchup/tomato_sauce Scene 1 failures were model-specific under old demo
  checks, while cream-cheese swap semantics remained provisional pending cleaner
  contain-region/task-success verification.

## Claim-Language Rules

- Prefer "decodable", "consistent with", "candidate mechanism", and
  "observational signal" unless a replay/intervention artifact supports a
  causal claim.
- Avoid "intent", "meaning mistake", "mechanism identified", "causal control",
  and "success rescue" unless the relevant gates are represented:
  clean donor/recipient definitions, replay reproduction, current-self sanity,
  controls, and rerun-verified behavior.
- Casebooks and probe maps select hypotheses; they do not establish mechanisms
  by themselves.

## Replication Priority

1. **Current robust probes:** target moved/contacted/lifted and filtered
   first-event target-relative probes over expert action hidden features.
2. **Required gates:** metadata leakage audit, split/gate summaries, interaction
   label quality, and failure taxonomy.
3. **Controls:** object-presence VLM probes, task identity probes, geometry
   shortcut controls.
4. **Causal candidates:** scene4 interface trace / attribution patching only
   after replay and tensor preflight artifacts are represented.
5. **Archive-only unless needed:** raw object multiclass probes, old broad
   best-non-target flow margins, coarse attention localization, and unfiltered
   action-head first-event probe.
