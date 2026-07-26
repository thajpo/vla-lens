# PI0.5 Broad 1000 Probe Experiments

Status: historical campaign record. `RESEARCH.md` is the current question and
findings log.

Last updated: July 22, 2026.

This document preserves the detailed review history for the PI0.5 broad 1000
probe campaign. It is not the active experiment registry or current method
guidance. The probes are not new capture; they train on the existing mech-light
activation features plus post-processed interaction labels.

## Dataset Contract

- Dataset root: `/mnt/new-volume/vla-lens/pi05-broad-1000-mech-light-lerobot-v3`
- Dataset shorthand: mech-light
- Actual capture profile: `mechanistic_sampled`
- Split column: `split`
- Train split: `train`
- Selection split: `val_heldout_task`
- Final report split: `test_heldout_task`
- Historical primary model: linear probe. Current generic probe work trains the
  standard linear and small-MLP battery together; see
  [probe hypothesis guidance](probe_hypothesis_guidance.md).
- Primary classification metric: balanced accuracy
- Saved diagnostics: predictions, per-split metrics, per-group metrics, null metrics, metadata baselines, linear weights when available
- Runtime contract: normal repo work, saved-trace analysis, probe training, UI
  work, and tests use `.venv` / `uv run`. PI0.5 execution work uses
  `.venv-pi05-*` plus wrapper scripts. "Execution work" means capture,
  model forward/replay, LIBERO/LeRobot env execution, hardware model loading,
  or writing real LeRobot capture roots.
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
- Latest local gate: passed on 2026-07-18 with 1000 episodes, 34000 activation
  site rows, 1.0 activation coverage, train/val/test split counts of
  600/200/200 episodes, and 42 checked artifacts at the start of this round.

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
Legacy entries that predate the mandatory `policy_call_index` metadata baseline
are preserved for context, but they are superseded when the same question was
rerun in the June 17 stronger-baseline round below.

## June 18, 2026 Geometry Probe Plan

Purpose: answer a new question rather than tune the earlier binary event
probes: is 3D object location represented in PI0.5 expert action hidden states,
and where? The key methodological shift is from event classification to scalar
geometry regression. Primary probes use mean-pooled action hidden tokens,
layer-wise ridge regression, held-out-task selection, and metadata regression
baselines.

Shared decision tree:

1. Use expert action hidden states first because the question is whether the
   action-producing representation carries scene geometry. VLM/image-token
   geometry remains a separate follow-up if expert features fail or look
   metadata-like.
2. Use scalar x/y/z regressions rather than a vector probe because the current
   artifact contract and UI support scalar targets cleanly. This also exposes
   axis-specific shortcuts, such as z being mostly phase/contact.
3. Use mean token pooling and linear/ridge probes as claim-bearing defaults
   because the broad-1000 dataset has modest independent task/episode count
   relative to 1024D hidden states. Concatenation, learned pooling, and MLP
   probes are upper-bound diagnostics, not first claims.
4. Sweep layer only. Policy calls are included as rows and as a metadata
   baseline, not as an extra sweep axis, to avoid tiny layer x call regression
   cells.
5. Use validation held-out tasks for layer selection and test held-out tasks
   only for final reporting.

Option A - active manipulated object absolute position:

- Question: can the representation decode the current world x/y/z position of
  the object currently being manipulated?
- Decision tree:
  - If `active_manipulated_object` exists for a row, select that object's
    `scene_object_pos` component.
  - If no active object exists, drop the row; forcing a default object would
    turn the probe into a scene-layout shortcut.
  - Baseline against task, prompt, phase, active object identity, primary target,
    candidate objects, and policy-call index.
- Specs:
  `pi05_broad_1000_active_object_position_{x,y,z}_expert_action_hidden.yaml`

Option B - all scene-object absolute position:

- Question: can the same action hidden state decode each scene object's current
  position when each object is evaluated as a candidate row?
- Decision tree:
  - Expand each activation row by object-role rows from the object-flow artifact.
  - Keep ordinary scene objects; fixtures are excluded for this first pass
    because fixtures and movable objects have different priors and support.
  - Decode `scene_object_pos` for `probe_object_name`.
  - Baseline against object identity/base name, object role, task, prompt,
    phase, and policy-call index.
- Specs:
  `pi05_broad_1000_all_object_position_{x,y,z}_expert_action_hidden.yaml`

Option C - active manipulated object gripper-relative position:

- Question: can the representation decode active-object position relative to
  the end effector, a more control-relevant geometry than absolute world pose?
- Decision tree:
  - Reuse Option A's active-object row contract.
  - Target is selected object position minus `eef_pos`, component-wise.
  - Drop rows without an active object or resolvable object position.
  - Use the same metadata baselines as Option A; if metadata solves this, the
    result is not clean evidence of geometric state in activations.
- Specs:
  `pi05_broad_1000_active_object_relative_position_{x,y,z}_expert_action_hidden.yaml`

Option D - target-vs-distractor geometry quality:

- Question: is position decoding better for task-relevant/manipulated objects
  than for distractors?
- Decision tree:
  - Do not train duplicate D-specific readouts if Option B already trains the
    correct object-local regression over both manipulated and distractor rows.
  - Use Option B predictions and inspect residuals by
    `probe_object_role_manipulated`, `probe_object_role_distractor`, and
    prompt-mentioned status.
  - If target/manipulated residuals are lower than distractor residuals on the
    validation-selected readout and held-out test, that suggests task relevance
    changes geometric availability. If all roles are similar or metadata
    baselines dominate, treat it as scene-layout/object-identity decoding rather
    than target-aware representation.

Planned campaign:

- `configs/probes/pi05_broad_1000_geometry_probe_campaign.yaml`
- Preflight all nine specs before training.
- Train only if rows, feature dimension, and target coverage are acceptable.
- Report MAE, R2, strongest metadata baseline, and held-out-test delta for each
  coordinate. For Option D, additionally report residuals by object role from
  the all-object prediction tables.

Final campaign:

- Campaign artifact:
  `probe_campaign-pi0.5-broad-1000-geometry-probe-campaign-684e2b6168`
- All nine specs completed.
- Preflight summary: all nine specs had 1024D mean-pooled features and zero
  automatic warnings.
  - Active-object rows after filters: 23,555 total, with 12,490 train,
    5,180 validation, and 5,885 final-test rows before layer grouping.
  - All-object rows after expansion: 103,990 total, with 59,850 train,
    27,470 validation, and 16,670 final-test rows before layer grouping.
- Training caveat: all-object ridge fits emitted repeated ill-conditioned-matrix
  warnings. Results are usable as first-pass ridge readouts, but PCA or stronger
  ridge regularization should be treated as the next numerical robustness check
  before building on small deltas.

Selected readouts:

| Option | Target | Selected layer | Val MAE | Val baseline MAE | Val delta | Test MAE | Test baseline MAE | Test delta | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | active object world x | 0 | 0.086 | 0.098 | +0.012 | 0.097 | 0.086 | -0.011 | Does not beat held-out metadata baseline. |
| A | active object world y | 0 | 0.116 | 0.110 | -0.006 | 0.103 | 0.131 | +0.028 | Test-positive but not validation-positive; not claim-bearing. |
| A | active object world z | 8 | 0.112 | 0.160 | +0.049 | 0.111 | 0.084 | -0.027 | Validation-positive, final-test baseline wins. |
| B | all-object world x | 12 | 0.098 | 0.073 | -0.025 | 0.097 | 0.104 | +0.007 | Weak and inconsistent; metadata/object prior dominates validation. |
| B | all-object world y | 0 | 0.149 | 0.120 | -0.029 | 0.136 | 0.136 | -0.000 | Negative. |
| B | all-object world z | 8 | 0.108 | 0.233 | +0.124 | 0.089 | 0.084 | -0.004 | Strong validation signal, but final-test metadata baseline slightly wins. |
| C | active object gripper-relative x | 0 | 0.088 | 0.091 | +0.003 | 0.079 | 0.068 | -0.011 | Tiny validation edge, not held-out robust. |
| C | active object gripper-relative y | 0 | 0.119 | 0.121 | +0.003 | 0.108 | 0.068 | -0.040 | Tiny validation edge, final-test baseline much better. |
| C | active object gripper-relative z | 0 | 0.051 | 0.054 | +0.004 | 0.063 | 0.055 | -0.007 | Tiny validation edge, not held-out robust. |

Option-specific conclusions:

- **A: active-object absolute position.** Decision tree was: use the active
  manipulated object when present, drop rows without one, decode x/y/z
  separately, and compare against task/object/phase/timing metadata. Result:
  no coordinate gives validation-positive and final-test-positive evidence
  over the strongest metadata baseline. Active-object geometry is not cleanly
  established by this first-pass mean-pooled action-hidden probe.
- **B: all-object absolute position.** Decision tree was: expand rows by
  scene-object role rows, exclude fixtures, decode each object candidate's
  x/y/z, and baseline against object identity/base name/role plus task and
  timing. Result: y is negative, x is weak/inconsistent, and z has a strong
  validation signal but loses narrowly to metadata on final test. This looks
  more like scene/object/phase priors plus a possible z-state signal than a
  robust object-local position representation.
- **C: active-object gripper-relative position.** Decision tree was: subtract
  `eef_pos` from the selected active object's position to test control-relevant
  geometry. Result: all validation deltas are tiny and all final-test deltas
  are negative. This first pass does not support a useful gripper-relative
  geometry probe from mean-pooled expert action hidden states.
- **D: target/manipulated-vs-distractor quality.** Decision tree was: do not
  train duplicate D readouts; use B predictions and inspect residuals by object
  role. Result: role effects are inconsistent by axis. On final test, x error
  is lower for manipulated/prompt-mentioned objects, y error is lower for
  distractors, and z is nearly role-neutral. There is no stable evidence that
  target/manipulated objects have better position decoding than distractors.

D final-test residuals from selected B readouts:

| Axis | Group | Count | Mean abs error | Median abs error |
| --- | --- | ---: | ---: | ---: |
| x | manipulated=false | 1,530 | 0.101 | 0.095 |
| x | manipulated=true | 1,804 | 0.093 | 0.071 |
| x | distractor=false | 2,024 | 0.093 | 0.071 |
| x | distractor=true | 1,310 | 0.103 | 0.100 |
| y | manipulated=false | 1,530 | 0.111 | 0.102 |
| y | manipulated=true | 1,804 | 0.157 | 0.153 |
| y | distractor=false | 2,024 | 0.156 | 0.150 |
| y | distractor=true | 1,310 | 0.106 | 0.098 |
| z | manipulated=false | 1,530 | 0.087 | 0.072 |
| z | manipulated=true | 1,804 | 0.090 | 0.072 |
| z | distractor=false | 2,024 | 0.092 | 0.074 |
| z | distractor=true | 1,310 | 0.083 | 0.070 |

Overall readout:

- Treat this as a mostly negative/diagnostic geometry round. The result does
  not justify a causal or intervention claim.
- The most interesting follow-up is not more layer fishing. If geometry remains
  important, improve the method first: campaign-level feature caching,
  precomputed geometry target tables, stronger ridge/PCA controls, and
  possibly a probe-specific capture profile with more independent tasks or
  environments.
- If choosing one follow-up target, object-local z is the only coordinate with
  a strong validation gain, but its final-test baseline loss and numerical
  conditioning warning mean it should be rerun with PCA/regularization and a
  locked confirmation plan before interpretation.

## July 18, 2026 Vector Geometry Robustness Round

Purpose: replace the scalar, ill-conditioned first pass with a vector-aware
test of whether PI0.5 globally pooled states linearly expose the instructed
target object's pose, and especially whether they expose pose updates beyond
temporal persistence.

Method:

- Use exactly one uniquely referable object per episode: the primary instructed
  target. The earlier all-object row expansion duplicated one global activation
  across several incompatible object labels and is not valid for this question.
- Fit joint multi-output ridge readouts after train-only PCA. Select PCA
  dimension from 64/128/256 and ridge alpha from 0.1/1/10/100 on validation,
  then lock the readout for final test.
- Measure position with episode-weighted Euclidean error in meters and rotation
  with episode-weighted SO(3) geodesic error in degrees.
- Test world, initial-relative, previous-call-relative, and end-effector-relative
  position. Test quaternion, 6D rotation, rotation-vector, Euler sine/cosine,
  initial-relative 6D, previous-call-relative 6D, and end-effector-relative 6D
  orientation targets.
- Compare every readout with train-mean/metadata controls and the matching
  physical baseline. For previous-call-relative targets, zero translation and
  identity rotation are the persistence baseline.
- Run both held-out-task splits and a deterministic within-task episode split.
  Episodes remain intact in every split.

Completed artifacts:

- Expert hidden, held-out task:
  `geometry_probe_study-pi0.5-broad-1000-object-geometry-study-b40227ee15`
- Expert hidden, within-task episode:
  `geometry_probe_study-pi0.5-broad-1000-object-geometry-within-task-study-7667296721`
- Action-head input, held-out task:
  `geometry_probe_study-pi0.5-broad-1000-action-head-object-geometry-study-93398f6bdf`
- Image tokens and VLM prefix endpoints, held-out task:
  `geometry_probe_study-pi0.5-broad-1000-vlm-object-geometry-study-e156f65fdf`

The previous-call-relative targets are the cleanest summary because they ask
whether a state predicts the update missed by simply carrying the last pose
forward:

| Features / split | Position val | Baseline val | Position test | Baseline test | Rotation val | Baseline val | Rotation test | Baseline test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Expert / held-out task | 0.0688 m | 0.0541 m | 0.0826 m | 0.0542 m | 5.11 deg | 3.74 deg | 11.23 deg | 9.25 deg |
| Expert / within-task episode | 0.0581 m | 0.0485 m | 0.0571 m | 0.0464 m | 6.41 deg | 5.29 deg | 6.44 deg | 5.11 deg |
| Action head / held-out task | 0.0727 m | 0.0541 m | 0.0778 m | 0.0542 m | 5.22 deg | 3.74 deg | 11.99 deg | 9.25 deg |
| Image prefix / held-out task | 0.0727 m | 0.0541 m | 0.0788 m | 0.0542 m | 6.14 deg | 3.74 deg | 11.59 deg | 9.25 deg |
| VLM endpoint / held-out task | 0.0741 m | 0.0541 m | 0.0742 m | 0.0542 m | 5.18 deg | 3.74 deg | 10.59 deg | 9.25 deg |

Interpretation:

- No selected globally mean-pooled linear readout beats its physical baseline
  on validation or final test. The easier within-task split does not rescue the
  result, and the conclusion is consistent across expert, action-head, image,
  and VLM endpoint features.
- This does not show that PI0.5 lacks object geometry. It shows that primary
  target pose and pose updates are not linearly accessible from these global
  reductions beyond strong temporal persistence controls.
- Do not run an MLP capacity sweep or intervention from these readouts. The
  predeclared stopping rule was to try nonlinear capacity only after a linear
  readout approached or beat its controls.
- The next scientifically distinct measurement is object-conditioned or
  spatial/token-local decoding. It needs an object query, pixel/token region,
  or contrastive object-local feature; more global layer fishing would repeat
  the same failed measurement.

Runtime and storage observations:

- The first cold expert feature materialization took about 404 seconds. Cached
  follow-up feature materialization took 8-23 seconds for expert/action-head
  families; the two VLM families took 194 seconds total from reduced cached
  data rather than rereading raw token tensors.
- The study caches aligned target tables and reduced feature matrices under
  `.vla_cache`; those are derived and evictable. Durable artifacts retain specs,
  fingerprints, selected hyperparameters, metrics, and row-level predictions,
  but do not duplicate feature tensors.
- Managed cache budgets, pinning, pruning, and capture-time reusable feature
  packs are tracked in GitHub issue #21.

## July 18, 2026 Motion-Aware Geometry Follow-Up

Purpose: resolve the incomplete part of the vector geometry round. The earlier
average mixed many stationary calls with a smaller number of large movements
and compared them mainly with a no-change guess. This follow-up asks whether
activations add information beyond ordinary task context, the robot hand's
movement, and the actions that were actually executed.

- Artifact:
  `pi0.5-broad-1000-object-motion-follow-up-study-geometry_motion_study-8fd6fa322e`
- Fixed movement ranges: position over 1 cm and over 10 cm; rotation over 1
  degree and over 15 degrees. These final held-out tasks had already been viewed
  during exploration, so this is saved as exploratory evidence rather than a
  new confirmation result.
- The first policy call is excluded because it has no previous interval.
- Comparisons include no movement, average train-set movement, task/scene/object/
  phase/call information, robot hand plus executed-action information, and the
  combination of task context and robot movement.
- Saved tables include all candidates, selected models, row-level predictions
  for every comparison, movement amount, direction and magnitude errors,
  task-level paired uncertainty, best/worst source examples, all scene-object
  movements, and matched target-versus-other-object scenes. A report can be
  regenerated from these tables without making a separate report the source of
  truth.

Large position movement results:

| Information used | Validation error | Final-test error | Final-test direction error |
| --- | ---: | ---: | ---: |
| Expert activation, selected layer 12 | 0.169 m | 0.185 m | 40.3 deg |
| Image-token global average | 0.200 m | 0.201 m | 53.7 deg |
| VLM endpoint, selected layer 17 | 0.190 m | 0.207 m | 46.7 deg |
| Robot movement and executed actions | 0.078 m | 0.076 m | 20.1 deg |
| Task context plus robot movement | 0.082 m | 0.068 m | 15.3 deg |
| No movement | 0.325 m | 0.271 m | not defined |

The activation probes do beat the no-movement and average-movement guesses on
large translations. They do not beat robot movement. The task-level final-test
gap between expert activations and the robot comparison is 10.1 cm against the
activation probe, with a 95% interval from 7.2 to 13.4 cm.

Movement detection tells the same story. For translations over 10 cm, expert
features reach 0.798 balanced accuracy and globally averaged image tokens reach
0.821 on final test. Task context reaches 0.791, while task context plus robot
movement reaches 0.837. Activation features do not show a stable advantage over
the ordinary comparisons across validation and final test.

Large rotation results are also negative against stronger comparisons. Expert
features have 36.8-degree final-test error, compared with 31.8 degrees from
robot movement and 39.8 degrees from guessing no rotation. The expert advantage
over no rotation is small and uncertain across tasks; it loses to robot
movement.

Matched-scene evidence:

- There are 636 later policy steps across 411 episodes and 81 tasks where the
  primary target moves more than 10 cm.
- In 95.8% of those scenes, every other object remains within 1 cm.
- Ordinary distractors almost never move after the initial interval: 0.05% of
  distractor rows exceed 1 cm.
- Other task-manipulated objects do move, which is expected in multi-object
  instructions and is retained explicitly in the saved object table.

Revised interpretation:

- The earlier large-movement improvement was real relative to the weak
  no-change guess, but it is not evidence that the model visually tracks the
  moving object specially.
- The current global activations largely reveal that the robot is in a movement
  event. The robot's measured movement predicts the object's direction and
  distance much more accurately.
- Token-level localization is not run in this round. The declared stopping rule
  required activations to add information beyond task and robot movement first.
  Object-local visual analysis remains appropriate for the controlled
  whole-scene experiment in GitHub issue #22, where object position can vary
  independently of the robot action.

## July 18, 2026 Joint Whole-Scene Object Study

Purpose: test the more direct whole-scene question from GitHub issue #22. Given
one saved PI0.5 activation at one policy call, can a linear decoder return every
object identity and every current XYZ position at once?

- Final artifact:
  `pi0.5-broad-1000-joint-object-identity-and-location-study-scene_map_probe_study-d2e23e2740`
- The output has one fixed slot per exact object instance, including separate
  slots such as `akita_black_bowl_1` and `akita_black_bowl_2`.
- The 39-slot vocabulary includes all objects in the 1,000 episodes. Thirty-five
  occur during training. Two `yellow_book` instances occur only in final-test
  tasks and therefore remain explicit unseen-identity failures.
- One decoder predicts the full scene roster and current visibility. One XYZ
  head per object uses only episodes where that object exists, avoiding fake
  zero-coordinate labels for missing objects.
- Model choices use validation tasks. Final-test tasks remain separate.
- Comparisons include instruction text plus ordinary scene information, the
  episode's initial object positions, and the previous policy call's positions.
- Globally averaged image tokens, VLM endpoints, expert layers, and action-head
  input are all tested. No token or pixel region is selected in this round.

Object identity results over identities seen during training:

| Information used | Scene overlap | Average precision | Entire roster exactly right |
| --- | ---: | ---: | ---: |
| Expert layer 12 activation | 0.408 | 0.703 | 10.7% |
| Instruction and scene information | 0.414 | 0.934 | 37.0% |
| Expert activation plus instruction/scene | 0.537 | 0.894 | 34.6% |
| Training frequency | 0.083 | 0.127 | 0.0% |

The combined model retrieves more true objects at its validation-chosen cutoff,
which raises scene overlap by 0.123 on average. The episode-level 95% range is
0.092 to 0.155. This is not clean evidence that the activation contains extra
visual identity information: the instruction/scene comparison ranks identities
better and gets more complete rosters exactly right. The combined model mainly
changes the precision-versus-recall tradeoff.

The visibility target does not fix the problem. Across final-test policy calls,
99.7% of present object rows are marked visible in at least one captured camera.
It is therefore almost the same label as the fixed scene roster, not an
independent test of recognizing what has entered or left view. A stronger
identity experiment needs object sets to vary within the same instruction and
scene family.

XYZ location results:

| Information used | All present objects | Objects moved over 10 cm |
| --- | ---: | ---: |
| Action-head activation | 0.223 m | 0.216 m |
| Instruction and scene information | 0.214 m | 0.249 m |
| Action-head activation plus instruction/scene | 0.208 m | 0.211 m |
| Episode's initial position | 0.038 m | 0.330 m |
| Previous policy-call position | 0.016 m | 0.183 m |

Adding action-head activations improves on instruction/scene information by
0.6 cm over all objects, with an episode-level 95% range of 0.3 to 0.9 cm. On
objects moved more than 10 cm, the improvement is 3.8 cm, with a range of 2.9
to 4.8 cm. That is real information about the current state, but it is not an
accurate whole-scene map: carrying forward the previous position is still 19.1
cm better overall and 2.8 cm better on the large-movement subset.

The combined decoder's coordinate errors are 7.6 cm on x, 13.1 cm on y, and
8.7 cm on z. Its gains are uneven across objects. It improves notably for the
porcelain mug, white/yellow mug, moka pot, and black book, while worsening for
cream cheese, orange juice, chocolate pudding, and several fixed scene objects.
This looks more like selective task/movement information than a stable geometric
record of every object.

Interpretation:

- This experiment does not support a globally averaged, linearly accessible
  full scene graph in PI0.5.
- It does support a narrower finding: later action-path activations add some
  information about which familiar scene objects matter and where substantially
  moved objects are now.
- The identity part is limited by task confounding. In these episodes, the
  instruction and scene family already determine most of the object roster.
- The location part is limited by global averaging. A fixed object slot cannot
  ask the representation where one particular object is, and averaging all
  tokens can erase spatial structure even if individual tokens contain it.
- The next experiment should use an explicit object query or a spatial/token
  decoder, and it should vary object presence and placement within otherwise
  matched scenes. That is a new measurement, not another global layer sweep.

Runtime and storage:

- The final artifact is 26 MB and contains 128,596 compact scene-level
  prediction rows, full comparison metrics, vocabulary/support counts, and
  source examples. The large source activation tensors remain in the capture
  and are not copied.
- The corrected full rerun took about eight minutes. Roughly half was repeated
  feature reduction and probe fitting. The runner now caches the train-fitted
  reduced feature projections under `.vla_cache`, so later reruns can skip PCA.
  These files are derived and evictable rather than permanent experiment data.
  Target tables already load in under a second after the first run.

## July 19, 2026 Token-Preserving And Layer-Mixture Study

Purpose: test whether the negative whole-scene result above was caused by
averaging the action tokens or by choosing one expert layer at a time.

- Primary artifact:
  `token_scene_probe_study-pi0.5-broad-1000-token-preserving-scene-object-study-channel-64-fa4f5edb8d`
- Lower-capacity check:
  `token_scene_probe_study-pi0.5-broad-1000-token-preserving-scene-object-study-6b02da1589`
- The study uses 6,184 policy-call scenes: 3,711 rows from 600 training
  episodes, 1,137 rows from 200 validation episodes, and 1,336 rows from 200
  final-test episodes. The final-test set has 20 held-out tasks.
- Every method sees the same rows, 39-object vocabulary, labels, and split.
- The source is the 50-token action suffix at the final generation step from
  expert layers 0, 4, 8, 12, and 17.
- `pooled` averages the 50 raw token vectors, then applies a training-only PCA.
- `tokenwise` applies a training-only 64-dimensional channel PCA inside each
  token, keeps all 50 token positions separate, then applies a second
  training-only PCA. Both paths give the final decoder 64 or 128 features, so
  the tokenwise decoder does not receive a larger final feature vector merely
  because it retains token positions.
- `single_layer` chooses one layer on validation. `learned_layer_mix` learns
  non-negative layer weights summing to one on validation and fits the shared
  linear decoder on training scenes.

The 64-channel projection retains 77.7% of sampled channel variance. The final
128-dimensional tokenwise projection retains 69.6% of the resulting flattened
token variance; the pooled projection retains 93.6% of pooled variance. A
16-channel run retained only 51.5% at the first step and was therefore kept as
a lower-capacity check rather than the primary result.

Final-test results:

| Representation | Layer treatment | Scene identity overlap | Identity average precision | XYZ error | XYZ error, moved over 10 cm |
| --- | --- | ---: | ---: | ---: | ---: |
| Pooled | Best single layer | 0.392 | 0.703 | 0.215 m | 0.207 m |
| Pooled | Learned layer mix | 0.392 | 0.699 | 0.218 m | 0.212 m |
| Tokenwise | Best single layer | 0.202 | 0.458 | 0.227 m | 0.224 m |
| Tokenwise | Learned layer mix | 0.163 | 0.457 | 0.227 m | 0.218 m |

Paired uncertainty uses equal-weight final-test episodes and separately
equal-weight final-test tasks. Positive values mean the candidate is better.

- The tokenwise single layer loses 0.190 scene-overlap points to the pooled
  single layer across episodes. Its 95% bootstrap interval is -0.211 to -0.170;
  across tasks it is -0.242 to -0.151.
- The tokenwise single layer adds 1.16 cm of XYZ error. Expressed as error
  reduction, the episode-level interval is -1.45 to -0.89 cm and the task-level
  interval is -1.33 to -0.37 cm.
- Mixing pooled layers changes scene overlap by less than 0.001, with an
  episode-level interval of -0.006 to 0.006. It makes XYZ error
  0.32 cm worse, with an episode-level interval of 0.19 to 0.44 cm worse.
- Mixing tokenwise layers does not reliably change XYZ error relative to the
  best tokenwise layer: the interval for error reduction is -0.21 to 0.27 cm.
  It lowers identity overlap by 0.038, with an interval of -0.048 to -0.028.

The best pooled identity layer is 12 and the best pooled XYZ layer is 8. The
learned mixtures use several layers, but that does not improve held-out
performance. For pooled XYZ, the largest mixture weight is 0.55 on layer 4;
for pooled identity, the largest weight is 0.47 on layer 0. This is a useful
warning: non-zero learned layer weights do not by themselves show that combining
depths produces a better representation.

The tokenwise coefficient summaries concentrate on later action-horizon tokens,
especially positions 43-49 for identity and roughly 32-46 for XYZ. These are
correlated linear coefficient norms after PCA, not causal importance. Because
the tokenwise models perform worse, this pattern should be used only to form a
future intervention or ablation question, not as evidence that those tokens
store the scene map.

Interpretation:

- This study does not support the idea that global token averaging was hiding
  a more accurate linearly decodable whole-scene map in the expert action
  suffix. Keeping token positions separate makes both tasks worse under the
  matched 64/128-feature comparison.
- The negative result is stable across held-out episodes and tasks, and it
  persists when the per-token channel width increases from 16 to 64. Only 8 of
  39 objects have lower XYZ error in the best tokenwise model.
- This does not show that no token-local object representation exists anywhere
  in PI0.5. The experiment tests expert action tokens, a linear decoder, and a
  variance-based compression. It does not yet test visual patch tokens, an
  explicit object query, or a nonlinear set decoder.
- The simple state comparisons remain stronger. The previous policy-call
  position has 1.6 cm error over all objects and 18.3 cm over objects moved more
  than 10 cm, compared with 21.5 cm and 20.7 cm for the best pooled expert
  decoder. Instruction and scene information also has much higher identity
  average precision (0.934) than the pooled expert representation (0.703).
- The next justified step is an explicit object-conditioned decoder on matched
  scenes, followed by visual-token localization. A higher-capacity set decoder
  should remain an exploratory upper bound because it can learn scene templates
  without exposing a clean representation.

Runtime and storage:

- The corrected full run took 14.5 minutes after the stricter token-topology
  check invalidated the earlier compact cache: 9.4 minutes to rebuild the
  reduced features and 5.1 minutes to fit and save the probe grid. Later runs
  can reuse the corrected reduced-feature cache, but still refit the probes.
- The final artifact is 14 MB. It contains source rows/sites, the exact token
  table, all projection transforms, selected decoder parameters, layer weights,
  per-scene predictions, per-object results, token coefficients, examples, and
  paired uncertainty.
- The reusable 64-channel cache is 34 MB. Raw captured activations are never
  copied. The 50-by-64 intermediate token tensor exists only in memory and can
  be reconstructed from the capture and saved projection.

One preliminary full run exposed a token-metadata bug: dynamic action-token rows
were repeated once per policy call, which selected the same 50 tensor positions
multiple times. That artifact was rejected, the loader now deduplicates exact
token indices, and a regression test covers this case. The accepted artifacts
contain exactly token positions 0 through 49.

## July 21, 2026 Matched-Scene Visual Localization Study

Purpose: test a narrower question than whole-scene decoding. If one object is
moved between two otherwise similar initial scenes, do the visual patch tokens
that cover the object's old and new locations change more than the other visual
patch tokens?

- Accepted artifact:
  `matched_scene_localization_study-pi0.5-broad-1000-matched-initial-scene-visual-localization-study-random-ranking-v2-e71204f77e`
- This is a matched-scene comparison, not a trained probe. It directly compares
  the saved token vectors for the two scenes.
- Scenes are matched by benchmark environment, task, prompt, and split. A pair
  is kept when exactly one object moves more than 1 cm, every other object moves
  at most 1 cm, and the initial end-effector position changes by at most 1 cm.
- The target image region is the union of the moved object's old and new
  simulator-derived bounding boxes in the main camera.
- The score for a visual patch is the size of the change in its token vector,
  divided by the average size of that vector in the two scenes.
- Sources are the image features entering the language model and visual prefix
  layers 0, 4, 8, 12, and 17. Raw pixel change is a positive control. A visible
  stationary object's image region is a negative control.
- The final set contains 34 pairs from 22 scene groups: 15 train pairs from 11
  scene groups, 10 validation pairs from 3 scene groups, and 9 final-test pairs
  from 8 scene groups. Layer 17 was selected using validation only.

Final-test results:

| Source | Target-region average precision | Expected random-ranking AP | Improvement over random | 95% scene interval | ROC AUC | Top-region recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw pixels | 0.483 | 0.105 | +0.379 | +0.235 to +0.529 | 0.854 | 0.443 |
| Image features entering the language model | 0.103 | 0.105 | -0.002 | -0.036 to +0.035 | 0.419 | 0.061 |
| Visual prefix layer 17, selected on validation | 0.102 | 0.105 | -0.003 | -0.022 to +0.017 | 0.453 | 0.061 |

Each scene group receives equal weight in the averages and confidence
intervals, so scenes that happen to produce more matched pairs do not dominate
the result. The confidence interval describes the range supported by resampling
the available scene groups; an interval crossing zero does not establish an
improvement over random patch ranking.

Interpretation:

- The positive control is strong. Raw image change reliably points to the moved
  object, so the scene matching, camera alignment, patch grid, and object boxes
  are usable for this question.
- The selected model layer is 0.003 average-precision points below the exact
  random-ranking expectation on held-out tasks, and its interval crosses zero.
  Its ROC AUC is below 0.5,
  and it ranks the moved-object region slightly worse than the stationary-object
  control region. This study therefore does not show reliable spatial
  localization from simple token-change magnitude.
- The conclusion is deliberately narrow. It does not show that PI0.5 lacks
  object identity or position information. The captured image features have
  already passed through a vision encoder that can mix information across
  patches, and the VLM layers can mix it again. Object information may therefore
  be distributed across tokens, available only after an object-specific query,
  or encoded as a relation rather than as a locally large change.
- Validation is small and dominated by one moved object (`basket_1`), so layer
  selection is uncertain. The held-out result covers eight scene groups and
  seven moved-object identities, but it is still an exploratory sample rather
  than a definitive negative result.

Runtime and storage:

- The accepted run took about 30 seconds. It reused the saved captures and did
  not load PI0.5 or run the simulator.
- The artifact saves the exact matched trace pairs, source tensor descriptions,
  camera patch boxes, all per-patch scores, per-pair metrics, scene-weighted
  summaries, split rules, and source-trace fingerprints. It references the
  original activation tensors instead of copying them.
- Two earlier artifacts are retained for audit history and marked superseded.
  One mixed pair-weighted headline averages with scene-weighted intervals. The
  next used target-patch prevalence as if it were expected average precision
  under a random ranking. Regression tests now cover equal scene weighting and
  the exact finite-ranking expectation.

The next useful experiment is an object-conditioned visual readout: provide an
object identity or object query and test whether a low-capacity decoder can
select that object's patch region or recover its XYZ position. That tests
distributed object information without assuming the patch that changes most
must be the patch containing the object.

## June 17, 2026 Stronger-Baseline Round

Purpose: rerun the most plausible interaction/outcome probes after applying
`docs/probe_hypothesis_guidance.md`. The key change was adding
`policy_call_index` as a metadata baseline and limiting the first pass to
linear probes. Preflight was clean for the six runnable interaction/outcome
specs after this change; the VLM-prefix target-parse spec was not runnable on
this dataset because the selector matched no rows.

- Campaign artifact:
  `probe_campaign-pi0.5-broad-1000-probe-research-round-1-e6fa9055b3`
- Command shape:
  `uv run python scripts/run_vla_lens_probe_batch.py ... --run --fail-fast`
- Trained specs:
  target moved, target lifted, target contacted, and outcome.
- Held-out task split remains the final report split. Sites were selected on
  `val_heldout_task`.

| Probe | Artifact | Val score | Val baseline | Delta | Selected site | Test score | Test baseline | Test delta | Main baseline |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |
| `target_moved` | `probe_suite-pi0.5-broad-1000-target-moved---expert-action-hidden-39fb62eadb` | 0.563 | 0.582 | -0.019 | `layer=12.0, policy_call_index=4` | 0.548 | 0.580 | -0.032 | benchmark |
| `target_lifted` | `probe_suite-pi0.5-broad-1000-target-lifted---expert-action-hidden-6f9a08c7cd` | 0.677 | 0.793 | -0.116 | `layer=17.0, policy_call_index=4` | 0.543 | 0.653 | -0.110 | primary target object |
| `target_contacted` | `probe_suite-pi0.5-broad-1000-target-contacted---expert-action-hidden-1302462495` | 0.764 | 0.767 | -0.003 | `layer=17.0, policy_call_index=5` | 0.386 | 0.642 | -0.256 | primary target object |
| `outcome` | `probe_suite-pi0.5-broad-1000-outcome-robust---action-head-input-51bcf9926f` | 0.677 | 0.700 | -0.023 | `policy_call_index 6` | 0.889 | 0.981 | -0.093 | benchmark |

Interpretation:

- This is mostly a negative result. None of the selected activation probes beat
  the strongest metadata baseline on validation once `policy_call_index` was
  included.
- The old target-moved and target-contacted positive deltas were not robust to
  this baseline revision. Treat those older artifacts as useful UI/regression
  examples, not as current evidence for a decodable mechanism.
- The target-contacted probe came closest on validation, but its selected site
  collapsed on the final held-out split. At the selected site
  `layer=17, policy_call_index=5`, test rows had 29 actual positives but 82
  predicted positives, driving balanced accuracy down to `0.386`. Do not
  promote it to intervention work without a new locked confirmation design.
- Some test-best sites show positive deltas, especially target-contacted
  `layer=17, policy_call_index=6`, but those are ad hoc test observations and
  should be treated only as hypothesis generators.
- Outcome has high raw test score at the selected policy call, but benchmark
  metadata performs even better. This is likely task/benchmark difficulty, not
  a clean failure mechanism.

Recommended next probe work:

- Do not rerun broad episode-level target moved/lifted/contacted as-is. The
  labels are too well explained by object/task/timing metadata.
- Prefer narrower, temporally local questions such as "will target contact
  occur within the next 1-2 policy calls?" or target-vs-distractor choices
  before contact. These should reduce the episode-level metadata prior.
- If revisiting outcome, stratify by benchmark/task or train within a task
  family so benchmark identity cannot dominate the baseline.
- Keep first-moved and first-lifted filtered specs as possible next candidates,
  but treat them as object-choice probes requiring the same policy-call and
  target-object baselines.

## June 17, 2026 Temporal Target-Event Round

Purpose: test the recommended local-horizon variant after broad episode-level
target interaction probes failed stronger metadata baselines. The new row labels
ask whether any instructed target object will contact or first move within the
next two policy calls. Rows are restricted to pre-event policy calls, and
`policy_call_index` remains a metadata baseline.

Implementation notes:

- Added derived row labels:
  `target_contact_within_2_policy_calls`,
  `target_motion_within_2_policy_calls`, plus future-event filters.
- Initial layer x policy-call specs were trained but rejected as scientifically
  weak: the validation-selected cells had only 7-8 rows, and some test cells
  had a single target class. Preflight now warns on low-support sweep groups.
- Final specs pool policy calls and sweep only layer. This keeps the timing
  baseline visible while avoiding tiny per-call selection cells.
- Diagnostic prediction retention was fixed so saved predictions match the
  validation-selected readout and include both validation and final test splits.

Final pooled campaign:

- Campaign artifact:
  `probe_campaign-pi0.5-broad-1000-temporal-target-event-probe-campaign-9e7cdd598f`
- Claim split: held-out task. Layer selected on `val_heldout_task`; test split
  is reported only after that selection.

| Probe | Artifact | Selected layer | Val BA | Val baseline | Val delta | Test BA | Test rows | Main readout |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| target contact within 2 calls | `probe_suite-pi0.5-broad-1000-target-contact-within-2-calls---expert-action-hidden-0f1aaf7868` | layer 8 | 0.752 | 0.631 | +0.121 | 0.477 | 168 | Fails final held-out task despite strong validation. |
| target motion within 2 calls | `probe_suite-pi0.5-broad-1000-target-motion-within-2-calls---expert-action-hidden-7105cc57cf` | layer 12 | 0.647 | 0.644 | +0.003 | 0.564 | 349 | Essentially metadata-level on validation. |

Interpretation:

- Do not promote either temporal target-event probe to intervention yet.
- Contact-within-two-calls is the more interesting negative/mixed result: it
  beats metadata on validation but reverses on held-out tasks. The selected
  model overpredicts positive contact on test (`131` predicted positives vs
  `91` actual positives), giving balanced accuracy below chance-like baseline.
- Motion-within-two-calls is not compelling. Its validation delta over metadata
  is only `+0.003`; the test split is also heavily positive-skewed
  (`309` positives, `40` negatives), so raw accuracy is not meaningful.
- The useful scientific lesson is about experimental design: local-horizon
  labels are directionally better than broad episode labels, but sweeping by
  policy call after pre-event filtering can create tiny validation cells. Use
  pooled policy calls first, then only inspect per-call behavior as a secondary
  analysis with explicit support thresholds.

Recommended next probe work:

- Try a target-vs-distractor formulation rather than a binary "event soon"
  label. For example: among visible candidate objects before contact, is the
  eventual target object distinguishable from distractors in representation
  space?
- If retaining local horizon labels, consider task-family-stratified runs or
  locked confirmation splits, because the pooled contact signal did not
  generalize across held-out tasks.
- Add any future per-policy-call sweep only with a minimum support gate per
  selected cell; do not select a layer/call from cells below the support
  threshold.

## June 17, 2026 Object-Choice Round

Purpose: follow the temporal probe failure with a cleaner object-choice
question. Instead of predicting the exact next object class under held-out
tasks, ask whether the next manipulated object before contact is the primary
instructed target rather than a distractor, receptacle, or support object.

Preflight decisions:

- The legacy multiclass `next_manipulated_object` artifact looked strong on
  validation, but it is not a clean held-out-task claim: some validation/test
  object labels are absent from training, so exact object-class prediction is
  partly an unseen-label problem.
- A first binary version using `target_objects` was also rejected in preflight:
  train and validation were all positive because `target_objects` is broad
  enough to include non-primary mentioned objects.
- The runnable target is therefore
  `next_manipulated_is_primary_target`, derived by comparing
  `next_manipulated_object` with `primary_target_object` after base-name
  normalization.

Final object-choice campaign:

- Campaign artifact:
  `probe_campaign-pi0.5-broad-1000-object-choice-probe-campaign-cbc2463339`
- Probe artifact:
  `probe_suite-pi0.5-broad-1000-next-manipulated-is-primary-target-pre-contact---expert-action-hidden-500da80fb0`
- Question: before contact, is the next manipulated object the primary target?
- Cohort: rows with a non-empty next manipulated object and `is_pre_contact`.
- Feature: expert action hidden states, policy calls pooled, layer sweep only.
- Baselines: benchmark, task ID, prompt, task phase, primary target object,
  candidate object set, visible candidate set/count, and policy-call index.

| Probe | Selected layer | Val BA | Val baseline | Val delta | Test BA | Test rows | Main readout |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| next manipulated is primary target | layer 8 | 0.732 | 0.932 | -0.200 | 0.507 | 136 | Strong metadata dominates; activation probe overpredicts target on test. |

Interpretation:

- This is a negative result for target-vs-distractor object choice in the
  current feature contract.
- The validation activation score is above null but far below the combined
  metadata baseline. The strongest metadata signal is benchmark/task context,
  not activation-specific evidence.
- Final held-out-task behavior is poor: the selected readout predicts `True`
  for `133/136` test rows, while the actual split is nearly balanced
  (`67` true, `69` false). Balanced accuracy is only `0.507`.
- Do not use this probe for intervention. It currently says that primary-target
  routing before contact is heavily explainable by dataset/task priors and that
  the selected activation readout does not generalize.

Recommended next probe work:

- If pursuing object choice, move away from global action-token pooling. The
  next useful variant would need a more object-local feature or explicit
  candidate-object contrast, not another pooled binary readout.
- Treat object-presence, target identity, and candidate-set metadata as controls
  rather than evidence of grounded target selection.
- A task-family-stratified object-choice run may still be useful, but it would
  answer an operational within-family question rather than a broad held-out-task
  mechanistic claim.

## June 17, 2026 Physical-State Round

Purpose: pivot away from tuning target-choice probes and ask a genuinely new
question: do expert action hidden states encode the current physical interaction
state of the robot/object system? This targets contact, motion, and lift state,
not target identity.

Rejected preflight:

- Active receptacle/destination decoding was considered first. It is a useful
  scientific question, but the current held-out-task split makes it a poor
  multiclass probe: several destination/receptacle labels appear in validation
  or test but not train, and minority classes are thin per selected layer.
- This should be revisited only with a different label design, such as
  destination type/category or a task-family-specific split.

Final physical-state campaign:

- Campaign artifact:
  `probe_campaign-pi0.5-broad-1000-physical-state-probe-campaign-3be4322597`
- Feature: expert action hidden states, policy calls pooled, layer sweep only.
- Baselines: benchmark, task ID, prompt, task phase, primary target object,
  candidate object set, visible candidate count, and policy-call index.
- Claim split: held-out task. Layer selected on `val_heldout_task`; test split
  reported after selection.

| Probe | Artifact | Selected layer | Val BA | Val baseline | Val delta | Test BA | Test baseline | Test delta | Main readout |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| current contact present | `probe_suite-pi0.5-broad-1000-current-contact-present---expert-action-hidden-4174b1ff7a` | layer 17 | 0.898 | 0.846 | +0.051 | 0.805 | 0.896 | -0.091 | Interesting validation signal, but does not beat final held-out-task metadata. |
| current moved present | `probe_suite-pi0.5-broad-1000-current-moved-present---expert-action-hidden-2cb35b3bfb` | layer 4 | 0.894 | 1.000 | -0.106 | 0.784 | 1.000 | -0.216 | `task_phase` perfectly predicts this label. |
| current lifted present | `probe_suite-pi0.5-broad-1000-current-lifted-present---expert-action-hidden-a06b2dfd5b` | layer 12 | 0.902 | 1.000 | -0.098 | 0.801 | 1.000 | -0.199 | `task_phase` perfectly predicts this label. |

Interpretation:

- Do not promote these probes to intervention work.
- Contact-present is the only one with a validation activation-over-metadata
  margin, but that margin reverses on final held-out tasks. The selected contact
  readout overpredicts contact on test (`223` predicted positives vs `86`
  actual positives), yielding good raw accuracy only because non-contact rows
  dominate.
- Moved-present and lifted-present are useful controls: action hidden states do
  encode behavior phase strongly, but the behavior-derived `task_phase` baseline
  is perfect. These results are not evidence for a mechanistic representation
  beyond the existing phase label.
- Weak cohorts for contact include `study_scene3` and `study_scene4`, suggesting
  the contact readout is not stable across scene families.

Recommended next probe work:

- For physical state, move from broad binary state to a cleaner residual
  question: can activations predict contact/lift within a task phase after
  stratifying or conditioning on phase?
- For destination/receptacle, avoid exact object-class decoding under
  held-out-task splits. Prefer destination category/type, relation class, or
  within-task-family confirmation.
- Do not spend more compute on pooled action-token binary target-choice probes
  unless the feature contract changes.

## June 17, 2026 Residual Contact-Within-Phase Round

Purpose: test the residual physical-state question suggested by the previous
round. Instead of asking whether contact is decodable while `task_phase` is
available as a metadata baseline, condition on a fixed phase and ask whether
contact state remains decodable inside that phase.

Preflight decisions:

- `approach`, `contact`, and `idle_or_post` were rejected before training
  because contact state is constant inside those phases on this dataset.
- `move_or_transport` and `lift_or_transport` both had contact/no-contact
  variation in train, validation, and test, with adequate support.
- `task_phase` was removed from baselines inside each fixed-phase cohort because
  it is constant after filtering. Task/object/timing/candidate baselines remain.

Final residual campaign:

- Campaign artifact:
  `probe_campaign-pi0.5-broad-1000-residual-contact-by-phase-probe-campaign-065a64c3e2`
- Feature: expert action hidden states, policy calls pooled, layer sweep only.
- Claim split: held-out task. Layer selected on `val_heldout_task`; test split
  reported after selection.

| Probe | Artifact | Selected layer | Val BA | Val baseline | Val delta | Test BA | Test baseline | Test delta | Main readout |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| contact within move/transport | `probe_suite-pi0.5-broad-1000-current-contact-within-move-phase---expert-action-hidden-11195046b1` | layer 8 | 0.920 | 0.709 | +0.211 | 0.589 | 0.874 | -0.285 | Strong validation signal collapses on held-out tasks. |
| contact within lift/transport | `probe_suite-pi0.5-broad-1000-current-contact-within-lift-phase---expert-action-hidden-6852dd80b5` | layer 4 | 0.647 | 0.662 | -0.015 | 0.840 | 0.760 | +0.080 | Validation-selected readout does not clear metadata; positive test is post-selection only. |

Interpretation:

- Do not promote residual contact probes to intervention.
- Move-phase contact is the most tempting number so far, but it does not survive
  held-out-task evaluation. On test, it predicts too many contacts (`72`
  predicted positives vs `20` actual positives), dropping balanced accuracy to
  `0.589` while the benchmark metadata baseline reaches `0.874`.
- Lift-phase contact is not selected by the protocol: validation delta is
  negative against `primary_target_object`. Its positive test delta is an
  after-the-fact observation, not evidence for a generalizable site.
- The failure is useful: current action-token pooled features appear to encode
  phase/contact-like information in-sample or within validation tasks, but the
  signal is not stable across held-out task families.

Recommended next probe work:

- Stop spending broad held-out-task compute on pooled action-token binary probes
  for now. They repeatedly fail metadata or final-task generalization.
- A genuinely different next question should change the feature contract or the
  unit of prediction, for example object-local visual/VLM features,
  candidate-wise contrasts, or continuous geometry residuals.
- If contact remains interesting, design a confirmation run inside a single
  task family with a locked split and treat it as an operational within-family
  probe, not a broad mechanistic claim.

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
- Status: legacy/superseded by the June 17 stronger-baseline rerun.
- Interpretation: this clears the metadata baseline on the selection split and
  is worth inspecting in the probe-suite UI before deciding whether to run a
  causal follow-up or a neighboring target-contact/lift probe. After adding
  `policy_call_index` as a baseline, the rerun did not clear metadata baselines.

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
- Status: legacy/superseded by the June 17 stronger-baseline rerun.
- Selection split: `val_heldout_task`
- Final held-out split aggregate balanced accuracy: `0.804`
- Source episodes: `1000`
- Training rows: `27880`
- Prediction rows: `22270`
- Target distribution: `False=18845`, `True=9035`
- Interpretation: this clears the strongest metadata baseline on the selection
  split and survives final held-out-task aggregation. Per-group performance is
  uneven, so treat this as a candidate for UI inspection and localization work,
  not as proof that the site causally controls target contact. After adding
  `policy_call_index` as a baseline, the rerun did not clear metadata baselines.

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

## Post-Campaign Next Work

The stronger-baseline interaction/outcome round and the pooled temporal
target-event round have both been run. Do not rerun the broad episode-level
outcome, target-moved, target-lifted, or target-contacted probes as-is unless
the UI/artifact loader needs a regression check.

Recommended next work after UI review:
- Prefer a feature-contract change over another pooled action-token binary
  readout. Pooled target-choice, physical-state, and residual contact probes all
  fail metadata or final held-out-task generalization.
- Run filtered first-moved/first-lifted probes only if the preflight support
  remains clean and the metadata baselines include `policy_call_index`.
- Do not sweep policy call as a selection axis after pre-event filtering unless
  every candidate cell clears the support threshold.

Current interpretation: broad episode-level target interaction and outcome
labels are not good enough for the next intervention candidate. The local
contact horizon is more interesting but still failed final held-out-task
generalization. The pooled object-choice probe also failed. Physical-state and
residual contact probes show strong validation decodability in places, but not
stable held-out-task evidence beyond metadata. The next cleanest scientific
question should use a more object-local representation, candidate-wise contrast,
or continuous geometry residual rather than another pooled action-token binary
readout.

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
- Status: completed / negative against stronger metadata baselines
- Artifact: `probe_suite-pi0.5-broad-1000-target-lifted---expert-action-hidden-6f9a08c7cd`
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
- Status: blocked / selector matched no rows in the current dataset
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
target-contacted probes have both been run; use `Post-Campaign Next Work` for the
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
  it into LeRobot v3 roots, VLA Lens overlays, and artifacts.
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

1. **Research validation:** run a locked PCA/regularization confirmation for
   object-local z only if that weak geometry lead still justifies the effort.
2. **Workflow validation:** complete a human browser pass on a current dataset
   and preserve exact links from aggregate evidence to source moments.
3. **Support-gated probes:** retain filtered first-event probes only when every
   candidate cell clears the support threshold and metadata controls remain
   competitive.
4. **Causal work:** promote a candidate only through a claim-eligible
   intervention with replay, no-op, matched controls, and recorded outcomes.
5. **Archive-only unless needed:** target-parse on the current dataset, raw object multiclass probes, old broad
   best-non-target flow margins, coarse attention localization, and unfiltered
   action-head first-event probe.
