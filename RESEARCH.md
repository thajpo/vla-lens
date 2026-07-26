# VLA Lens Research Log

Status: active source of truth for research questions and findings.

Last updated: July 26, 2026.

This file answers three questions:

1. What scientific questions have we asked?
2. What exactly did each experiment test?
3. What should we revisit next?

Detailed tables remain in saved VLA Lens artifacts. Long implementation notes
remain in `docs/pi05_broad_1000_probe_experiments.md`. This file is the clean
human and agent handoff surface.

## Experiment Vocabulary

- **Probe:** a trained classifier or regressor that predicts a target from
  frozen VLA activations.
- **Diagnostic:** a measurement of activations with no trained readout.
- **Baseline:** a competing predictor that does not use the activations being
  tested, such as task metadata, previous physical state, or pixels.
- **Intervention:** a controlled change to activations followed by a model or
  environment rerun.

Probe success means the target is decodable under the saved data and readout
contract. It does not show that the VLA uses the decoded information. Causal
claims require interventions.

## Research Framing: Scene, Instruction, Robot State, And Action

VLA Lens should study semantics and action together. Semantic questions remain
important: we still want to know whether the model distinguishes objects,
positions, poses, appearance, and task roles. But PI0.5 was trained to produce
actions, so we should not assume that it stores those facts as stable,
human-readable object records.

The main framing is:

> What reusable information lets the model combine the scene, instruction, and
> robot state into appropriate behavior, and when does that process fail?

Test three separate properties rather than collapsing them into one claim:

1. **Readable:** can a trained readout recover the information?
2. **Reusable:** does the relationship hold across layouts, objects,
   instructions, robot starts, tasks, and harmless visual changes?
3. **Used:** does changing the candidate path selectively change the relevant
   action or closed-loop behavior?

A signal can pass one test and fail another. Known-region identity can be
readable without being a general identity code. A broad scene patch can affect
the action without identifying the scene property or mechanism that mattered.

Use crossed controlled conditions whenever possible. Independently change the
target pose, named target, robot start, appearance, camera, and irrelevant
objects. Measure the result after action postprocessing and through the actual
controller in physical units; raw distance over a normalized 50-by-7 action
chunk is diagnostic, not a behavioral endpoint. Repeat matched flow noise, but
count tasks and scene clusters—not noise samples, tokens, steps, or action
elements—as independent evidence.

Then follow a repeatable physical action difference through visual tokens, the
exact visual key/value memory consumed by each expert layer, expert states,
predicted action updates, and the final action. At each denoising time, compare
scene conditions while holding the current noisy action guess fixed. Otherwise
later differences mix scene conditioning with action trajectories that have
already diverged. Preserve token, head, action-position, and time structure
rather than averaging it away by default.

This action-centered view and direct semantic probes are complementary. A
semantic probe can show that information is available. A consistent link
between a scene change, an internal change, and an action correction can show
what that information means for the policy. Internal interventions should come
after this measurement identifies a small, repeatable model location worth
testing.

The current evidence is mostly a set of useful constraints, not a discovered
object model. We have a positive known-region identity result, broad visual
dependence of an open-loop action, and late whole-action-state replacement that
can force donor-like output. We have not shown that the altered action is
correct, localized object binding, isolated scene conditioning from the
evolving action guess, or found a small semantic feature set. Future entries
must keep those distinctions explicit.

## Required Record For New Questions

Before running an experiment, record:

- question and hypothesis;
- experiment kind;
- prediction target and how it is constructed;
- activation source, layer, token handling, and feature transform;
- readout models;
- train, validation, and test units;
- baselines and positive/negative controls;
- important confounds;
- metrics and uncertainty method;
- allowed conclusion and stopping rule.
- strongest competing explanations and what would distinguish them;
- discovery versus prospectively held-out confirmation status;
- independent sample unit and total search/selection space;

For action-centered scene experiments, also record:

- the scene factor changed and everything deliberately held fixed;
- the expected action consequence, including relevant action dimensions and
  horizon positions;
- the internal stages and axes used to follow that action difference;
- whether the result concerns semantic availability, action relevance, or
  both, and whether it generalizes beyond the discovery cohort;
- the physical or closed-loop behavior endpoint, or an explicit statement that
  the run measures only model output.

After the run, add artifact IDs, the result, limitations, and the next decision.

## Default Cheap Probe Battery

When asked whether activations encode a target, agents should automatically run
the inexpensive, data-supported comparison:

- linear or ridge probe;
- standard small MLP;
- every declared layer;
- pooled and token-preserving inputs when supported;
- feature compression or feature-level readouts when appropriate;
- metadata and physical-state baselines;
- a probe retrained on shuffled training labels;
- validation-only selection and episode/task-grouped uncertainty.

The user should be asked only when a choice changes the scientific target or
cohort, requires new capture, launches an intervention, or has material cost.

## Shared Dataset

- Root: `/mnt/new-volume/vla-lens/pi05-broad-1000-mech-light-lerobot-v3`
- Episodes: 1,000
- Split: 600 train, 200 validation held-out task, 200 test held-out task
- Capture: PI0.5 `mechanistic_sampled`
- Important limitation: tasks, scenes, object rosters, and object positions are
  correlated. High probe accuracy can come from these priors.
- Confirmation limitation: the existing final-test tasks have been inspected
  repeatedly during exploration. New positive findings should eventually use a
  locked fresh confirmation split or capture.

## Research Question Index

| ID | Question | Kind | Status | Revisit? |
| --- | --- | --- | --- | --- |
| RQ-001 | Do global action representations encode broad target events or success? | Probe | Mostly negative | No, not unchanged |
| RQ-002 | Do they predict target contact or movement in the next two calls? | Probe | Mixed validation, negative held-out | Later with object-local inputs |
| RQ-003 | Do they identify which object will be manipulated? | Probe | Negative/confounded | Yes, object-conditioned |
| RQ-004 | Do they encode current contact, movement, or lift state? | Probe | Metadata explains more | No, not unchanged |
| RQ-005 | Is contact decodable after holding task phase fixed? | Probe | Not held-out robust | Later within a cleaner family |
| RQ-006 | Is object XYZ linearly decodable from pooled action features? | Probe | Mostly negative; superseded | Revisit the question, not the method |
| RQ-007 | Are target position and rotation available beyond physical persistence? | Probe | Negative for pooled linear readouts | Yes, MLP and object/token inputs |
| RQ-008 | Do activations track object motion beyond robot motion and actions? | Probe/diagnostic | No | Not with global inputs |
| RQ-009 | Can one global activation reconstruct the whole scene object map? | Probe | Limited state signal, inaccurate map | Yes, object-conditioned |
| RQ-010 | Do action-token positions or layer mixtures reveal a better scene map? | Probe | No | Yes with visual tokens and MLP |
| RQ-011 | Do visual-token changes spatially localize a moved object? | Diagnostic | No | Move to trained object queries |
| RQ-012 | Do main-camera visual tokens encode the initial object roster and each object's XYZ? | Probe | Identity yes; XYZ mostly scene prior | Yes, localize identity and improve geometry target |
| RQ-013 | Does the identity probe's evidence spatially cover the named object? | Probe diagnostic | Modestly for some objects; not movement-sensitive | Yes, inspect examples and intervene carefully |
| RQ-014 | Do visual tokens explicitly encode every visible object's image location? | Probe | Run; local probe negative, scene decoder confounded | Revisit with matched scenes or object-centric decoder |
| RQ-015 | Does a known object region identify the object occupying it? | Probe | Positive exploratory result | Confirm on fresh locked data |
| RQ-016 | Can an explicit object query locate that object? | Probe | Negative for episode-specific location | Revisit with matched scenes or object-centric features |
| RQ-017 | Are pose changes nonlinearly decodable from pooled representations? | Probe | Negative on validation | No larger global pooled probe sweep |
| RQ-018 | Does the layer-8 object-identity direction causally affect PI0.5's action? | Intervention | Completed; negative semantic-specificity result | Calibrate region and dose controls |
| RQ-019 | Does exchanging target and distractor poses naturally change PI0.5's action? | Counterfactual | Positive pilot | Yes, across tasks and object pairs |
| RQ-020 | Where can donor activations transfer that natural action change? | Activation patching | Broad visual-prefix result; narrow object regions negative | Yes, follow the signal into the action stream |
| RQ-021 | What object property does a successful patch carry? | Counterfactual intervention | Planned inside RQ-024 | After controlled behavior result |
| RQ-022 | Which residual, key/value, or action-bridge path carries the effect? | Path intervention | Late whole-state replacement; scene/action path confounded | After fixed-action-state comparison |
| RQ-023 | Does a specific action-level transfer change closed-loop behavior? | Rollout intervention | Planned inside RQ-024 | Only after locked causal confirmation |
| RQ-024 | Does PI0.5 form a reusable scene-instruction-robot-state control relationship? | Crossed behavior/mechanism campaign | Planned | Active campaign |

## RQ-001: Broad Target Events And Outcome

**Question.** Can pooled PI0.5 action-path activations predict whether the
target moved, lifted, or was contacted, or whether the episode succeeded?

**Hypothesis.** Action-producing representations contain target-interaction and
success state beyond task, object, and timing priors.

**Method.** Linear logistic probes. Target-event probes used mean-pooled final
generation-step expert action tokens across layers 0, 4, 8, 12, and 17. Outcome
used mean-pooled action-head input. Validation selected layer/call; test used
held-out tasks.

**Baselines and controls.** Majority class, benchmark, task, scene family, task
verb, primary target identity, and policy-call index; fixed-prediction label
permutation in the legacy artifact.

**Confounds.** Events accumulate over time. Benchmark, task, object identity,
and policy-call index strongly predict the labels. Rows within an episode are
correlated.

**Result.** None of the four validation-selected activation probes beat the
strongest metadata baseline. Target-contact performance also collapsed on
held-out tasks. Outcome accuracy was high, but benchmark metadata was higher.

**Decision.** Do not rerun these global pooled binary probes unchanged. Keep
them as negative findings and UI/artifact examples.

**Accepted artifacts.**

- `probe_suite-pi0.5-broad-1000-target-moved---expert-action-hidden-39fb62eadb`
- `probe_suite-pi0.5-broad-1000-target-lifted---expert-action-hidden-6f9a08c7cd`
- `probe_suite-pi0.5-broad-1000-target-contacted---expert-action-hidden-1302462495`
- `probe_suite-pi0.5-broad-1000-outcome-robust---action-head-input-51bcf9926f`

## RQ-002: Near-Future Target Events

**Question.** Before the event occurs, can activations predict target contact or
movement within the next two policy calls?

**Hypothesis.** A short future window removes much of the accumulated-event and
timing shortcut in RQ-001.

**Method.** Linear logistic probes over pooled expert action tokens. Policy
calls were pooled after early layer-by-call attempts produced cells with too
few examples. Validation selected the layer.

**Baselines and controls.** Task/object/scene/timing metadata, class-support
preflight, and held-out tasks.

**Confounds.** Positive rates still vary by task and phase. Several early sweep
cells contained only 1-8 evaluation rows.

**Result.** Contact beat metadata on validation by 0.121 balanced-accuracy
points but fell to 0.477 on test. Motion was only 0.003 above metadata on
validation.

**Decision.** Do not treat as an intervention candidate. Revisit only after an
object-local or candidate-wise representation is available.

**Accepted artifacts.**

- `probe_suite-pi0.5-broad-1000-target-contact-within-2-calls---expert-action-hidden-0f1aaf7868`
- `probe_suite-pi0.5-broad-1000-target-motion-within-2-calls---expert-action-hidden-7105cc57cf`

Three earlier campaign pairs are retained as rejected implementation history;
they are not separate scientific results.

## RQ-003: Object Choice Before Contact

**Question.** Can action representations identify the next manipulated object,
or at least whether it is the primary instructed target?

**Hypothesis.** Target routing should be visible before physical contact.

**Method.** A legacy multiclass linear probe predicted exact object identity.
The accepted binary linear probe predicted whether the next manipulated object
was the primary target, using pooled expert action tokens and a layer sweep.

**Baselines and controls.** Benchmark, task, prompt, phase, primary target,
candidate and visible object sets, and policy-call index.

**Confounds.** Some exact object labels were absent from training. Task and
scene context nearly determine the target in many episodes.

**Result.** The binary probe scored 0.732 on validation but the metadata
baseline scored 0.932. Test predictions collapsed to nearly all-positive.

**Decision.** Revisit with one row/query per candidate object and visual tokens;
do not add capacity to the same global binary input.

**Artifacts.**

- Legacy: `probe_suite-pi0.5-broad-1000-next-manipulated-pre-contact---expert-action-hidden-919db3e817`
- Accepted binary: `probe_suite-pi0.5-broad-1000-next-manipulated-is-primary-target-pre-contact---expert-action-hidden-500da80fb0`

## RQ-004: Current Physical State

**Question.** Do expert action representations encode current contact,
movement, or lift state?

**Hypothesis.** Physical interaction state should be present in the action
pathway even if target identity is not.

**Method.** Three linear logistic probes over pooled expert action tokens with
validation-selected layers.

**Baselines and controls.** Task, prompt, object, phase, candidate set, and
policy-call metadata.

**Confounds.** The derived task phase is constructed from the same behavior and
perfectly predicts moved/lifted state in this dataset.

**Result.** Raw activation scores were high. Metadata was better on held-out
tasks; phase perfectly predicted moved and lifted state.

**Decision.** Keep as evidence that these labels are too easy to explain. Do
not rerun unchanged with an MLP.

**Artifacts.**

- `probe_suite-pi0.5-broad-1000-current-contact-present---expert-action-hidden-4174b1ff7a`
- `probe_suite-pi0.5-broad-1000-current-moved-present---expert-action-hidden-2cb35b3bfb`
- `probe_suite-pi0.5-broad-1000-current-lifted-present---expert-action-hidden-a06b2dfd5b`

## RQ-005: Contact Within A Fixed Phase

**Question.** Is contact state decodable after holding task phase fixed?

**Hypothesis.** Conditioning on phase removes the strongest shortcut in RQ-004.

**Method.** Linear logistic probes inside move/transport and lift/transport
cohorts. Other phases had no label variation and were rejected before training.

**Baselines and controls.** Task, object, timing, and candidate-set metadata;
held-out tasks.

**Confounds.** Phase-conditioned cohorts remain task- and object-dependent.

**Result.** Move-phase contact was strong on validation and poor on test.
Lift-phase contact did not beat metadata on validation.

**Decision.** Revisit only as a locked within-task-family operational question,
not as broad mechanistic evidence.

**Artifacts.**

- `probe_suite-pi0.5-broad-1000-current-contact-within-move-phase---expert-action-hidden-11195046b1`
- `probe_suite-pi0.5-broad-1000-current-contact-within-lift-phase---expert-action-hidden-6852dd80b5`

## RQ-006: Scalar Object XYZ

**Question.** Can pooled expert action activations linearly decode active-object
world XYZ, every scene object's XYZ, or active-object gripper-relative XYZ?

**Hypothesis.** Object geometry is linearly accessible in the action pathway,
especially for manipulated objects.

**Method.** Nine ridge probes: three axes for each of active world position,
all-object world position, and active gripper-relative position. Layers were
selected on validation.

**Baselines and controls.** Task, prompt, phase, object identity/role, and
policy-call metadata.

**Confounds.** Global activation rows were duplicated across object candidates
in the all-object experiment. Object and scene priors predict much of absolute
position. Several fits were ill-conditioned.

**Result.** No axis produced a stable validation-and-test advantage over
metadata. Object-local z was the strongest validation hint but lost on test.

**Decision.** Superseded by RQ-007. Do not rerun the nine scalar probes; revisit
geometry with vector targets, MLP capacity, and object/token conditioning.

**Campaign artifact.**
`probe_campaign-pi0.5-broad-1000-geometry-probe-campaign-684e2b6168`

## RQ-007: Vector Position And Rotation Beyond Persistence

**Question.** Do representations expose target position and rotation, including
updates not captured by carrying the previous physical state forward?

**Hypothesis.** If PI0.5 tracks object pose, activations should improve on
initial-state and previous-state baselines.

**Method.** Train-only PCA followed by multi-output ridge. Targets included
world, initial-relative, previous-call-relative, and end-effector-relative
position plus quaternion, 6D, rotation-vector, Euler, and relative rotation
forms. Features covered expert hidden states, action-head input, image tokens,
and VLM endpoints. Held-out-task and within-task episode splits were tested.

**Baselines and controls.** Train mean, metadata, initial pose, previous pose,
zero update, and identity rotation.

**Confounds.** Global mean pooling may erase object-local information. The
instruction fixes one target but task/scene still constrain its pose.

**Result.** No selected global linear readout beat its matching physical
baseline on validation and test.

**Decision.** Revisit now with the cheap probe battery: linear plus MLP, visual
token-preserving inputs, object query, all layers, and the same physical
baselines.

**Accepted artifacts.**

- `geometry_probe_study-pi0.5-broad-1000-object-geometry-study-b40227ee15`
- `geometry_probe_study-pi0.5-broad-1000-object-geometry-within-task-study-7667296721`
- `geometry_probe_study-pi0.5-broad-1000-action-head-object-geometry-study-93398f6bdf`
- `geometry_probe_study-pi0.5-broad-1000-vlm-object-geometry-study-e156f65fdf`

Four earlier artifacts from this family were corrected reruns and should be
marked superseded in the dataset index.

## RQ-008: Object Motion Beyond Robot Motion

**Question.** Do activations predict object movement, direction, distance, or
rotation beyond task context, robot-hand movement, and executed actions?

**Hypothesis.** A tracked object representation should add information beyond
the robot's own movement.

**Method.** Ridge movement readouts and movement classifiers on large-motion
subsets, plus matched target-versus-other-object scene analysis.

**Baselines and controls.** No movement, average movement, task context, robot
movement, executed actions, and task plus robot movement.

**Confounds.** Object movement is mechanically coupled to the robot. Most
distractors remain fixed.

**Result.** Activations detect movement events and beat no-change guesses, but
robot/action controls predict direction and distance substantially better.

**Decision.** Do not extend the same global readout. Object-local visual probes
are the distinct next measurement.

**Artifact.**
`pi0.5-broad-1000-object-motion-follow-up-study-geometry_motion_study-8fd6fa322e`

## RQ-009: Whole-Scene Object Map

**Question.** Can one activation jointly recover every scene object's identity
and XYZ position?

**Hypothesis.** PI0.5 contains a structured scene representation rather than
only target/action state.

**Method.** Fixed object slots with multi-label identity and one XYZ ridge head
per object. Inputs were globally averaged expert, action-head, image-token, and
VLM endpoint features.

**Baselines and controls.** Training frequency, instruction plus scene context,
initial positions, previous positions, and activation-plus-context models.

**Confounds.** Instructions and scene families nearly determine object rosters.
Fixed slots bake object identity into the decoder. Almost every present object
is visible. Global averaging discards spatial token structure.

**Result.** Activations add some information about familiar relevant objects
and moved-object state, but they do not recover an accurate full scene map.
Previous position is substantially better for XYZ.

**Decision.** Revisit now using one object query at a time, visual tokens, and
linear/MLP comparisons. Do not repeat the fixed-slot global decoder.

**Artifact.**
`pi0.5-broad-1000-joint-object-identity-and-location-study-scene_map_probe_study-d2e23e2740`

## RQ-010: Action Tokens And Layer Mixtures

**Question.** Was RQ-009 limited by token averaging or choosing one layer?

**Hypothesis.** Keeping action-token positions separate or learning a layer
mixture will reveal a stronger scene representation.

**Method.** Train-only channel and readout PCA, followed by ridge decoders.
Compared pooled versus tokenwise expert action tokens and single layers versus
non-negative learned layer mixtures. Two channel capacities were run.

**Baselines and controls.** Matched rows, output width, labels, and splits;
validation-only layer/mixture selection; paired episode/task uncertainty.

**Confounds.** PCA preserves variance, not necessarily object information.
Action suffix tokens are horizon positions, not visual patches. Linear readouts
may miss nonlinear distributed information.

**Result.** Tokenwise inputs made identity and XYZ worse. Layer mixtures did not
improve held-out performance.

**Decision.** Do not rerun the same action-token ridge grid. Revisit with visual
tokens and MLP/object-conditioned readouts.

**Primary artifact.**
`token_scene_probe_study-pi0.5-broad-1000-token-preserving-scene-object-study-channel-64-fa4f5edb8d`

The 16-channel artifact is a lower-capacity check:
`token_scene_probe_study-pi0.5-broad-1000-token-preserving-scene-object-study-6b02da1589`.

## RQ-011: Matched-Scene Visual Localization

**Question.** When exactly one object moves between matched initial scenes, do
visual patch tokens change most in that object's old and new image regions?

**Hypothesis.** A spatially local object representation should produce larger
token changes at the moved object's patches.

**Kind.** Diagnostic; no probe was trained.

**Method.** Rank main-camera patches by relative activation-vector change for
image features and VLM layers 0, 4, 8, 12, and 17. The target region is the
union of simulator-derived old and new object boxes.

**Baselines and controls.** Raw pixel change is the positive control; a visible
stationary object is the negative control. Scenes restrict other-object and
end-effector movement to at most 1 cm.

**Confounds.** Vision encoder and VLM attention may distribute information
across patches. The matched cohort has only 22 scene groups, and validation is
dominated by `basket_1`.

**Result.** Raw pixels localize the changed object. Activation change magnitude
does not. Exact random-ranking expected average precision is 0.105 on test,
versus 0.102 for layer 17. The scene-grouped interval for the difference is
-0.022 to +0.017.

**Decision.** The corrected artifact is explicitly a diagnostic. Move to
trained object-conditioned visual probes.

**Accepted artifact.**
`matched_scene_localization_study-pi0.5-broad-1000-matched-initial-scene-visual-localization-study-random-ranking-v2-e71204f77e`

The earlier pair-weighted and prevalence-baseline artifacts are superseded.

## RQ-012: Initial Visual Object Identity And XYZ

**Question.** Do main-camera visual tokens encode which objects are present and
the XYZ position of each named object in the initial scene?

**Hypothesis.** Preserving visual patch positions and adding small nonlinear
readouts will reveal a structured scene representation that global action
features missed.

**Method.** One initial-scene row per episode. Main-camera image patches from
VLM layers 0, 4, 8, 12, and 17 were compared as pooled and token-preserving
readouts. Training-only PCA produced 32- and 64-dimensional inputs. The battery
fit ridge and one-hidden-layer MLP probes. Identity was predicted as the whole
object roster; XYZ used a separate head for each named object. Layers,
capacity, regularization, and learned linear layer mixtures were selected only
on validation tasks. Test contains 200 episodes from 20 held-out tasks.

**Baselines and controls.** Object frequency, each object's training-set mean
position, prompt plus scene metadata, and 20 probes retrained after shuffling
training scenes. Paired uncertainty resampled both episodes and tasks. Initial
and previous position were intentionally excluded: at the initial frame they
are the answer, not fair baselines.

**Confounds.** Object rosters and locations remain strongly tied to task and
scene. Four of 39 object identities did not have enough training support, which
produced 64 unseen positive labels in test. Nearly all evaluated objects were
visible. Named XYZ heads assume object identity instead of solving visual
correspondence. All selected MLPs reached the 300-step limit, so their results
are provisional rather than evidence that nonlinear probes cannot help.

**Result.** Initial object identity is meaningfully decodable. The
validation-selected token-preserving linear probe reached 0.579 scene Jaccard
on held-out tasks, versus 0.424 from prompt and scene metadata and about 0.079
for shuffled-label probes. The activation gain over metadata remained positive
when resampling either episodes or tasks. Keeping patch positions separate
only slightly improved the linear probe over pooling, while it substantially
helped the MLP.

XYZ is much weaker. The best tested linear representation reached 0.196 m mean
3D error, compared with 0.204 m for prompt and scene metadata, 0.248 m for the
per-object training mean, and about 0.272 m after shuffling training scenes.
The roughly 8 mm advantage over metadata was positive across episodes but not
stable across held-out tasks. Y was the largest average coordinate error. MLP
XYZ probes were almost entirely unconverged and worse (0.249-0.281 m).

**Decision.** Accept initial scene identity as a positive decodability finding,
not yet a causal or object-local representation claim. Treat initial XYZ as
mostly explained by scene/task priors. Next, inspect identity successes and
failures over episodes and patches, then test object localization or
camera-frame geometry before choosing an intervention.

**Accepted artifact.**
`token_scene_probe_study-pi0.5-broad-1000-initial-visual-object-query-identity-and-location-study-ed33f5bd37`

**Uncertainty correction.** The saved task bootstrap grouped bare numeric
`task_id` values, which accidentally combined different LIBERO benchmarks at
IDs 8 and 9. Recomputing with benchmark plus task name produced 26 separate
tasks and did not change the identity conclusion: the mean scene-Jaccard gain
over prompt and scene metadata was 0.148, with a 95% interval of 0.068 to
0.234. Future token-scene studies use the corrected task key.

## RQ-013: Does The Identity Probe Look At The Object?

**Question.** When the RQ-012 probe says that an object is present, does the
episode-specific evidence come from image patches covering that object, or
from the rest of the scene?

**Hypothesis.** If the probe reads a spatial object representation, positive
patch contributions for a visible object should rank that object's image
region above random ranking and above the probe's fixed patch preferences.

**Method.** Replayed the validation-selected token-preserving linear
identity probe from RQ-012 without retraining it. Each held-out object score
was decomposed through the saved PCA transforms into signed contributions from
each image patch. Positive, signed, and absolute episode-specific contributions
were compared with simulator object boxes. Matched scenes in which one
object's initial position changed tested whether its contribution map followed
the moved image region.

**Baselines and controls.** Exact random-ranking average precision; the
probe's fixed coefficient magnitude at each patch; other visible object
regions in the same frame; raw-pixel change in matched scenes; exact replay of
the saved probe score; and episode- and task-grouped uncertainty.

**Confounds.** A linear contribution is a faithful decomposition of this
probe, not a causal explanation of the VLA. Large object boxes are easier to
rank. Correlated patches can share or cancel signed evidence. Simulator boxes
may include background, occlusion, or only a small visible surface. The probe
was trained on whole-scene identity, so scene context may be a valid and useful
source of its prediction even if the signal does not localize.

**Decision rule.** Call the signal object-local only if episode-specific
positive contributions beat both random ranking and fixed coefficient
preferences on held-out tasks, with task-grouped uncertainty above zero. Use
the matched-scene result as a stricter supporting check, not as the primary
claim because few pairs are available.

**Result.** The primary cohort contains 577 visible objects with trained
decoder heads that the probe predicted as present, across 168 held-out
episodes. Positive patch
contributions reached 0.194 mean average precision, compared with 0.102 for
random ranking and 0.175 for the probe's fixed patch preferences. The
episode-specific lift over the fixed map was 0.034 when averaging 22 benchmark
tasks (95% interval 0.010 to 0.059). It was less stable across 16 distinct
instructions: 0.018 with an interval of -0.007 to 0.043.

The secondary all-visible cohort contained 848 objects across all 200 held-out
episodes. Its mean average precision was 0.180, versus 0.097 random and 0.150
fixed. Objects the probe missed still had some localized positive evidence,
showing that patch localization alone does not determine whether the complete
probe score crosses its threshold. Signed evidence was larger inside the
named object than inside another visible object's region. The saved probe
scores replayed to a maximum absolute error of `4.4e-7`.

The aggregate hides strong variation. Among probe-positive objects,
white/yellow mugs and black books localized clearly. Many other objects did
not beat the fixed spatial pattern; wine bottles, wine racks, porcelain mugs,
and the flat stove were notable failures. In nine held-out matched pairs,
change in the probe contribution map did not follow the moved object beyond
random (mean lift 0.003, interval -0.026 to 0.039), while raw-pixel change
localized strongly (0.379, 0.233 to 0.526).

**Decision.** Accept modest object-local evidence for a subset of identities,
not a general object-tracking representation. The result is stable across the
held-out benchmark tasks represented in the positive cohort, but not yet
across distinct instruction groups. The probe combines episode-specific visual
evidence with stable scene and patch-location cues. Use the episode maps to
inspect which object categories support a proposed intervention. Do not treat
the identity direction as a position-sensitive object handle without a
stronger localization probe.

**Accepted artifact.**
`identity_localization_study-pi0.5-broad-1000-held-out-object-identity-patch-localization-study-b3d4c2db57`

## RQ-014: Explicit Image-Plane Object Location

**Question.** Can PI0.5 visual-token representations recover the identity and
image-plane location of every visible scene object on held-out tasks?

**Hypothesis.** A probe trained directly for spatial localization will recover
object boxes more consistently than the RQ-013 attribution of a whole-scene
identity probe. If location is carried locally, a shared patch head should
identify which patches overlap each named object. If it is only recoverable
from the scene as a whole, a token-preserving coordinate decoder may work
while the local patch head fails.

**Method.** Used the same initial main-camera rows, held-out-task split,
training-only channel projection, and 39-name object vocabulary as RQ-012. A
shared local Ridge head predicted, for every named object, whether each of 256
image patches overlapped its simulator box. Separate named-object Ridge heads
predicted normalized box center and size from pooled and token-preserving scene
readouts. Validation selected layer, input width, and regularization. The test
set contained 784 visible supported-object instances from 200 episodes and 27
represented object identities.

A fixed 32-unit MLP was also tried at each linear probe's selected input as a
bounded capacity check, not a second architecture search. The patch MLP used at
most 100,000 training tokens and 80 iterations. Per-object box MLPs used at
most 200 iterations.

**Baselines and controls.** Per-object training-mean image location, fixed
per-object patch maps, prompt plus scene metadata, shuffled scene-to-box
assignments, wrong-object regions, and unsupported identities. Report
episode-, benchmark-task-, instruction-, and object-level results. Select all
layers and hyperparameters on validation only.

**Confounds.** Object identity, image location, task, and scene remain
correlated. Large boxes are easier to localize. A named-object head assumes
correspondence rather than discovering an unordered object set. Simulator
boxes include some background and can be affected by visibility and occlusion.
Good image-plane decoding would establish accessible visual geometry, not a
causal object representation.

**Decision rule.** Require held-out improvement over both fixed spatial maps
and prompt/scene or training-mean location baselines. A general location claim
also requires positive results across object identities rather than an
aggregate driven by a few mugs, books, or large scene fixtures.

**Result.** The local patch probe failed its controls. Its held-out average
precision was 0.299, below the fixed per-object spatial map at 0.458 and the
mean of shuffled-scene probes at 0.309. The episode-grouped AP difference from
the fixed map was -0.161, 95% CI [-0.177, -0.143]. It did put its single highest
patch slightly nearer the box center: 65.6 pixels versus 68.7 pixels for the
fixed map at the raw-instance level, despite ranking the full object region
substantially worse.
This does not support episode-specific object location in the tested local
linear token features.

The whole-scene linear box decoder contained some location signal. It reached
44.5-pixel center error and 0.392 IoU, versus 53.3 pixels and 0.284 IoU for each
object's training-mean box. It also beat shuffled scene-to-box training, whose
ten-run means were 55.3 pixels and 0.270 IoU. The improvement over the mean-box
baseline remained positive when benchmark tasks, instructions, or object
identities were weighted equally.

However, the activation decoder did not beat the stronger prompt-and-scene
baseline, which reached 43.1 pixels and 0.404 IoU. Episode-weighted activation
minus context center-error improvement was -1.24 pixels, 95% CI
[-1.79, -0.70]. Task- and instruction-weighted intervals also did not establish
an activation advantage. The current evidence therefore fits a scene/task
prior at least as well as an explicit visual object map.

The bounded MLP check did not help. The patch MLP hit its 80-iteration cap and
dropped to 0.243 AP. Only 3 of 35 box heads converged within 200 iterations, and
their combined result was much worse. This rules out a cheap improvement from
that fixed MLP setup; it does not rule out nonlinear decoding in general.

**Artifact.**
`image_location_probe_study-pi0.5-broad-1000-explicit-image-plane-object-location-study-v2-b0f3f863ca`.
It saves the exact rows, sites, patches, boxes, visibility masks, validation
choices, linear and MLP weights, baseline predictions, shuffled controls,
grouped confidence intervals, examples, and test predictions. Replaying every
saved head reproduced scores within 2.4e-6.

**Workflow result.** A 71 MB rebuildable compact-token cache and 653 KB image-box
cache reduced a repeat from about 12 minutes to 55 seconds for the linear study,
or about 77-81 seconds with the bounded MLP checks. Raw captured activations
remain the source of truth.

**Status.** Run under GitHub issue #30. The direct local probe is negative. The
scene decoder is positive against simple and shuffled baselines but not against
task/scene context, so a general explicit object-location claim is not
supported. A matched-scene position decoder or an object-centric set/query
decoder is the most informative revisit.

## RQ-015: Known-Region Object Identity

**Question.** When an object's image region is supplied, do the visual tokens
inside that region identify which object occupies it?

**Hypothesis.** Object identity will be more cleanly accessible from the
object's own region than from the entire image or another region in the same
scene, even though scene and box geometry remain predictive.

**Method.** Used every initial, visible object instance with adequate training
support. Averaged the existing 16-channel compact visual-token vectors
only over patches intersecting its simulator box at layers 0, 4, 8, 12, and
17. Fit multinomial linear and fixed 64-unit MLP identity readouts. Select
layer, model, and regularization on held-out validation tasks only.

**Baselines and controls.** Whole-image token mean; task, scene, prompt, and box
geometry; another visible object's region; background patches; and probes
trained after shuffling training identities. Report macro balanced accuracy,
macro average precision, per-object recall, and paired task-, instruction-,
and object-grouped intervals.

**Allowed conclusion.** A positive result establishes object-local identity
decodability under a known-region contract. It does not show that the model can
find the object without the box or that it uses the identity causally. The
existing final test has already been inspected by earlier work, so any positive
result remains exploratory until fresh confirmation.

**Result.** The known object region contains a strong identity signal. The
validation-selected layer-4 MLP reached 0.466 balanced accuracy and 0.526
average precision on held-out test tasks. The same metrics were 0.193 and
0.134 for the whole-image token mean, 0.226 and 0.226 for a different object's
region, 0.180 and 0.133 for background patches, and 0.290 and 0.362 for task,
scene, prompt, and box metadata. Ten shuffled-label probes averaged only 0.029
balanced accuracy and 0.061 average precision.

The selected MLP hit its 300-iteration limit, so a separate robustness run fit
the same question with a converged linear probe. The layer-8 linear probe
reached 0.410 balanced accuracy and 0.448 average precision, versus 0.286
balanced accuracy for the strongest task/scene/box control. Its correct-rate
advantage over a wrong object region remained positive when whole objects were
the resampling unit: +0.201, 95% CI [0.054, 0.355]. This confirms that the core
finding does not depend on an unfinished nonlinear optimizer.

Performance still varied sharply by object identity: some objects were decoded
almost perfectly and many had zero recall. The result therefore says that
identity is accessible when the correct region is supplied, not that PI0.5 has
a uniformly clean object code or can locate the region itself.

**Decision.** Accept as a positive exploratory result and use its saved
episode/object predictions for inspection. Confirm on fresh locked data before
making a broad claim. For interventions, choose identities that decode reliably
and keep poorly decoded identities as negative controls.

**Accepted artifacts.**

- Primary battery: `object_roi_identity_study-pi0.5-broad-1000-known-region-visible-object-identity-study-c8c63c09c5`
- Converged linear check: `object_roi_identity_study-pi0.5-broad-1000-known-region-object-identity-converged-linear-check-4f24179aef`

Both artifacts save the fitted readouts, exact evaluated instances,
predictions, confidence intervals, and enough numeric parameters to replay the
probe outside scikit-learn.

## RQ-016: Explicit Object-Query Localization

**Question.** Given a named object query, can one shared decoder identify the
image patches occupied by that object?

**Hypothesis.** Conditioning a local visual-token readout on object identity
will recover episode-specific location that the independent named-object heads
in RQ-014 missed.

**Method.** Constructed `(episode, object query, image patch)` examples
from initial visible objects. Kept every positive and deterministically sampled
one near-box, one wrong-object, and one background negative per positive, with
at most 400,000 training examples. Concatenate each patch's 16 compact
activation channels, normalized patch coordinates, and a query one-hot vector.
Fit linear and fixed 64-unit MLP readouts at layers 0, 4, 8, 12, and 17, with
validation-only selection. Apply the frozen decoder to all supported queries
on test episodes and to accepted matched-scene pairs.

**Baselines and controls.** Fixed per-object spatial maps; query and coordinates
without activations; prompt and scene context; wrong-object queries;
within-task shuffled episode activations; and a fixed patch-position
permutation. Report patch average precision, peak-center error, predicted-box
IoU, scene Jaccard, grouped intervals, and whether matched-scene predicted
displacement improves over zero displacement.

**Allowed conclusion.** A positive result requires improvement over the fixed
spatial and context controls plus displacement evidence in matched scenes. It
would establish an accessible object-conditioned visual location signal, not
an unordered scene graph or causal mechanism. As with RQ-015, fresh locked data
is required before treating a positive test result as confirmation.

**Result.** The validation gate passed, and the selected layer-4 MLP reached
0.515 patch average precision on held-out test tasks, compared with 0.458 for
the fixed per-object spatial map. Query identity mattered: supplying the wrong
object query reduced average precision to 0.184.

However, shuffling activation maps between episodes of the same task did not
hurt. It slightly improved average precision to 0.524. In the matched-scene
test, the activation query was also 4.52 pixels worse than predicting no
movement at all. Thus the decoder can combine an object name with familiar
task/scene layout, but the tested activation map does not tell it where that
object is in the current episode.

**Decision.** Treat this as a negative result for episode-specific localization
or object tracking. Do not intervene on this readout as though it were a moving
object handle. Its most useful inspection view is a side-by-side comparison of
the real activation query, within-task shuffled activation, fixed spatial map,
and true object box so researchers can see the shortcut directly.

**Artifact.**
`object_query_localization_study-pi0.5-broad-1000-explicit-visual-object-query-localization-study-2c530d8218`.
It saves the selected model, test predictions, query and shuffle controls,
matched-scene results, and grouped confidence intervals.

## RQ-017: Nonlinear Pose Capacity Beyond Physical Baselines

**Question.** Can a small nonlinear readout recover object position or rotation
information that the pooled ridge studies in RQ-007 missed?

**Hypothesis.** Pose may be present in the same global representations but not
linearly accessible. A fixed one-hidden-layer MLP should improve on the matching
physical and metadata baselines on held-out tasks if that is true.

**Method.** Compared the existing multi-output ridge readout with one
fixed `MLPRegressor`: 64 hidden units, `alpha=1e-4`, at most 300 iterations, and
random seed 0. Fit the feature scaler and one maximum PCA on training rows only;
reuse PCA prefixes at 64, 128, and 256 dimensions. Sweep expert layers, action
head input, image-prefix hidden states, and VLM endpoints. Restrict this capacity
check to world XYZ, XYZ change since the previous policy call, world rotation in
6D form, and relative rotation since the previous call in 6D form.

**Splits.** Held-out-task validation and test are primary. A separately named
within-task episode split is secondary. Layer, site, PCA width, model family,
and ridge strength are selected using validation only.

**Baselines and controls.** Train mean; task, scene, object, phase, and timing
metadata; previous or initial pose for absolute targets; zero update for
position change; and identity rotation for relative rotation.

**Confounds.** The MLP can exploit task and scene priors, global mean pooling
can erase object-local structure, and repeated policy-call rows are correlated.
An MLP-only result is a nonlinear capacity result, not evidence that the model
uses a clean pose representation.

**Metrics and uncertainty.** Episode-weighted Euclidean position error and
SO(3) geodesic rotation error. Report paired probe-minus-baseline intervals by
resampling whole benchmark-task groups for the primary split and whole episodes
for the within-task secondary split.

**Stopping rule.** Choose the best MLP only on validation. If it does not beat
its strongest matching physical or metadata baseline there, do not report its
test result and do not promote it. A positive held-out result remains
exploratory until a fresh locked confirmation split or capture.

**Result.** None of the 16 validation-selected MLP readouts beat its matching
physical control on held-out tasks, even though all 16 optimizers converged.
The closest was previous-call-relative rotation from expert layer 0: 10.05
degrees of validation error versus 3.74 degrees from predicting no rotation
change, or 2.69 times worse. The selected MLP was also worse than the selected
ridge readout for all four targets. In accordance with the stopping rule, no
MLP test-set result was computed or reported.

The failure was broad: the sweep covered 540 candidate combinations and 32
final selections across expert action hidden states, action-head input,
image-prefix hidden states, and VLM endpoints. It tested absolute XYZ, XYZ
change, absolute 6D rotation, and relative 6D rotation with 64-, 128-, and
256-dimensional training-only PCA inputs. This makes “a little more nonlinear
capacity fixes the global pooled pose probe” a poor next bet.

**Decision.** Accept the validation result as negative and stop expanding the
same global pooled architecture. It does not rule out object-local or
token-to-object geometry, because pooling may discard the correspondence we
need. A future geometry study should change the representation contract, not
just enlarge the readout.

**Artifact.**
`geometry_probe_study-pi0.5-broad-1000-nonlinear-pose-capacity-study-59aa1ec2e9`.
It contains 540 candidates, 32 validation selections, fitted parameters,
predictions for allowed linear controls, grouped confidence intervals, and
explicit records showing that all nonlinear test results were withheld.

**Workflow result.** Shared managed caches allowed the four feature families to
run safely in parallel with other studies and remain reusable. Specialized
review-only preflight now recognizes this geometry-study spec and reports its
targets, splits, controls, representation families, and model/PCA sweep without
training or materializing features.

## RQ-018: Object-Local Identity Direction Intervention

**Question.** Does changing the RQ-015 layer-8 linear identity direction only
inside a known object's image tokens change PI0.5's action in a direction- and
region-specific way?

**Hypothesis.** If the decoded object identity is used by the policy, removing
the desk-caddy feature direction from the caddy ROI should change the replayed
action more than a raw-norm-matched random direction, a poorly decoded object
identity direction, or the same caddy direction applied to another object's
ROI.

**Target and method.** Start with RQ-015 held-out instance 2648: `desk_caddy_1`
in policy call 0 of
`pi05_mechanistic_sampled_libero_90_task73_seed2000`. Use the converged linear
object-ROI probe at `pi05.vlm.layers.8.prefix.hidden_tokens`. Define identity as
the target class weight minus the mean class weight in the probe's standardized
16-dimensional feature space. Map that delta back to the raw 2,048-dimensional
hidden space with the exact minimum-norm inverse of the saved scaler and PCA.
Apply it only to the 45 main-camera prefix tokens saved for that instance.

The first operation is `project_out_direction` at strength 1. This removes the
feature-dependent coordinate from the current ROI mean but leaves the
classifier intercept. It is not a general “erase identity” operator, and the
full pre/post probe margin must be reported. Later signed add-direction checks
may use smaller calibrated strengths; one standardized unit is a large decoded
effect and is not a safe default for addition.

**Baselines and controls.** Require exact stored-noise replay and repeated no-op
actions within explicit L2 and max-absolute tolerances. Run an orthogonal random
direction matched to the realized raw perturbation norm; the poorly decoded
`red_coffee_mug_1` class-mean direction, also raw-norm matched; and the caddy
direction on the saved wrong-object ROI. Preserve the stored original, every
no-op, main intervention, and control action chunk.

**Confounds.** RQ-015 is exploratory and uses a known simulator box. One policy
call cannot establish a behavioral mechanism. PCA inversion chooses the
minimum-norm raw delta but does not prove that it is on the model's natural
activation manifold. A large action change can reflect generic sensitivity,
which is why both direction and ROI controls are required.

**Metrics.** Save the probe margin before and after the runtime-dtype hook, raw
perturbation L2/RMS, raw and named action deltas from no-op, main-versus-control
action differences, exact artifact/table/array hashes, resolved checkpoint,
layer, tensor shape, token indices, and replay environment.

**Allowed conclusion.** Passing replay and control gates makes the run eligible
for a local, action-level causal comparison. It does not automatically support
the identity-mechanism hypothesis. A positive scientific result additionally
requires the main action effect to exceed the matched random, wrong-identity,
and wrong-ROI effects. Behavioral or general identity claims require repeated
recipients and rollouts.

**Stopping rule.** Do not run the hook if artifact reconstruction, checkpoint,
policy call, layer, prefix layout, ROI token mapping, initial-noise exactness,
or no-op replay fails. Stop after the first full controlled action-level run and
inspect effect sizes before choosing strengths or a cohort. Do not call a single
eligible run behavioral evidence.

**Result.** Completed on ROCm with exact saved generation noise. Three no-op
reruns were bit-for-bit identical to one another. Each differed slightly from
the older stored action chunk (L2 `0.01554`, maximum element `0.002537`), so the
controlled run used measured replay gates of `0.02` L2 and `0.003` maximum
absolute error.

The main project-out changed the caddy direction coordinate from `1.1781` to
`0.0008` and its target-versus-class-mean probe margin from `32.58` to `-4.12`.
The resulting action change from the no-op was L2 `0.1172`. This establishes
that the hook changed both the intended decoded quantity and the model's action
output.

It did not pass the semantic-specificity comparison. With the same raw hidden
perturbation L2 (`14.882`), the orthogonal random control changed the action by
`0.1855` and the red-mug direction changed it by `0.1671`, both more than the
caddy direction. The wrong-ROI control changed the action by only `0.02037`,
but its natural project-out perturbation was also only L2 `3.973`; that run
cannot distinguish region specificity from perturbation size.

**Conclusion.** Layer-8 caddy tokens are locally action-sensitive, and the
saved probe can now be reconstructed and intervened on exactly. This recipient
does not show that the decoded caddy identity direction is more causally
important than generic or wrong-identity directions. Treat this as a negative
semantic result, not as evidence that identity is unused everywhere.

**Decision.** Before a broader recipient cohort, add a raw-norm-matched
wrong-ROI comparison and a small signed dose curve for the main, random, and
wrong-identity directions. This will separate region, direction, and nonlinear
large-perturbation effects. Do not run rollouts or claim behavioral identity
effects yet.

**Artifacts.** Request:
`configs/interventions/rq018_caddy_identity_project_out.json`. Replay report:
`vla_lens/intervention_reports/rq018_caddy_replay.json`. Controlled report:
`vla_lens/intervention_reports/rq018_caddy_controlled.json`. Saved artifact:
`intervention_run-rq-018-caddy-identity-direction-at-layer-8-d2f6e0d415`.

## RQ-019: Natural Pose-Exchange Effect

**Question.** When the instructed target and a visible distractor exchange
poses while the instruction, robot, camera, checkpoint, and all other scene
state remain fixed, does PI0.5 produce a meaningfully different action chunk?

**Hypothesis.** If PI0.5 conditions its action on object identity and location,
the same instruction should produce a reliable recipient-to-donor action change
when the target and distractor exchange positions.

**Method.** Five matched LIBERO-90 task-73 scenes used the same checkpoint,
instruction, simulator seed, camera calibration, and robot state. The recipient
kept the original scene. The donor exchanged the full MuJoCo poses of the black
book and white/yellow mug without stepping physics. Layouts 0-4 used seeds
3100-3104. The scientific action comparison regenerated both scenes with the
same saved noise; separately saved capture noise is retained only as a
descriptive number.

**Baselines and controls.** Exact repeated no-op generation, a scene pair with
no changed variables, and per-pair compatibility checks for model, checkpoint,
prompt, camera, robot state, action shape, and token layout.

**Confounds.** A pose exchange changes pixels, occlusion, and potentially
collision geometry. A natural action difference does not by itself identify
which object property or internal pathway caused it. Simulator scene mutation
must not silently alter robot or camera state.

**Metrics.** Saved-noise replay L2 and maximum element error; recipient-to-donor
action L2; per-horizon and per-action-dimension deltas; pair validity and hook
eligibility. Confidence intervals group by scene pair, not by action element.

**Result.** All five pairs passed. Robot and camera fields matched exactly, the
two object poses exchanged exactly, and every other object stayed fixed. Two
replays per recipient reproduced the saved action exactly (`L2 = 0`, maximum
error `= 0`). Under shared noise, the pose exchange changed the 50-by-7 action
chunk by L2 distances `10.43`, `9.79`, `6.13`, `7.52`, and `10.86` (mean
`8.95`). This is a reproducible nonzero direction for a causal patching test;
its size has no physical interpretation under this normalized metric.

**Conclusion.** PI0.5's generated open-loop action chunk changes reproducibly
under this scene change. The capture used `max_steps: 1`, so it does not show a
rollout, successful behavior, or an appropriate response. The L2 value mixes
normalized action dimensions and has no direct physical interpretation. The
experiment also does not separate object identity from position, appearance,
occlusion, or the fact that two objects changed at once. It establishes a
model-output counterfactual and allows RQ-020 to proceed.

**Artifacts.** Capture recipe: `configs/capture/rq019_pose_exchange_pilot.json`.
Dataset root: `/mnt/new-volume/vla-lens/rq019-pose-exchange-pilot`. Reconstructable
pair manifest:
`vla_lens/counterfactual_pairs/rq019_pose_exchange_pairs.json` under that root.

## RQ-020: Donor Activation Patch Localization

**Question.** At which layer and token region can donor activations move the
recipient action toward the donor action produced under the same noise?

**Hypothesis.** If an internal site carries object-conditioned information used
by the action policy, patching that donor value into the recipient should move
the action along the natural recipient-to-donor action change, not merely make
the action different.

**Method.** For each valid RQ-019 pair, donor activations were cached in memory
and patched into the recipient at VLM prefix layers 0, 4, 8, 12, and 16. The
first pass compared 12 book tokens, 12 mug tokens, their 24-token union, and 24
background tokens. A confirmation pass added recipient-self, donor-self, alpha
zero, shuffled-donor, and wrong-region controls. Because the narrow regions
failed, a predeclared broad pass compared all 256 main-camera tokens, all 256
wrist-camera tokens, all 512 image tokens, the 149 active language tokens, and
the full 968-token prefix. Recipient and donor always used the same noise.

**Baselines and controls.** Recipient self-patch, donor self-patch, alpha zero,
shuffled tokens, same-identity donor where available, unrelated donor,
stationary or wrong-region tokens with matched token count, and at least eight
norm-matched random patches. Every runtime hook must fire exactly once and be
removed after its trial.

**Confounds.** Residual patch size can create generic sensitivity. Token regions
may have different natural activation norms. A late residual hook may be a
structural no-op because later VLM layers already produced their key/value
cache. Cross-position key patching is invalid without rotary-position handling.

**Metrics.** Let recipient, donor, and patched actions be r, d, and p. Save
natural change d-r and patch change p-r. Report their cosine agreement, the
fraction of natural change transferred, remaining distance from p to d,
recovered donor distance, and movement perpendicular to d-r. Preserve the full
50 by 7 action arrays and named axes; do not reduce the result to one number.

**Narrow-region result.** Across 100 localization patches, the best narrow
result was the layer-0 mug region: it transferred only `0.88%` of the natural
action change (pair bootstrap 95% interval `0.67%-1.09%`). It was positive in
all five layouts, but the confirmation study showed that shuffled donor tokens
transferred `1.43%` on average. None of the four confirmed layer/region choices
beat its strongest control. The small positive signal is real but not specific
to the selected object tokens.

**Broad-region result.** The broad pass completed 125/125 patches with no
failures. At layer 0, patching both cameras transferred `99.36%` of the natural
change; patching the full prefix transferred `100.06%` (95% interval
`99.87%-100.31%`) and passed in all five layouts. The full-prefix transfer was
`99.97%` at layer 4, `94.24%` at layer 8, `51.37%` at layer 12, and `0.27%` at
layer 16. Image-only transfer followed the same pattern: `99.36%`, `93.83%`,
`85.55%`, `47.43%`, and `0.19%`. Language tokens never transferred more than
`2.85%`. At layer 0, the main camera transferred `29.35%` and the wrist camera
`33.55%`; their effects are not additive, which is expected in a nonlinear
network.

**Controls.** Recipient-self patches were exact no-ops and donor-self patches
reproduced the donor action exactly. These controls show that the hook and
action comparison work. Shuffled and wrong-region controls rule out the narrow
object-box interpretation. Every runtime hook fired exactly once and was
removed after its trial.

**Conclusion.** This establishes a broad causal dependency: replacing nearly
all early visual context can reproduce nearly all of the donor action change.
It does not identify which changed scene property matters. A 12-token box
around either object did not pass specificity controls, but that does not prove
that the model lacks an object-local representation. Distributed information,
globally contextual visual tokens, box/token mismatch, occlusion, and the
two-object swap remain alternatives. The loss of late residual-output transfer
also does not locate a single VLM-to-expert handoff because each layer has
already written its same-layer prefix key/value cache.

**Limits.** One task, one object pair, five initial layouts, and open-loop action
chunks are not evidence for a general scene graph. The exchange changes two
identities-at-positions and their pixels at once. RQ-021 must use one-factor
counterfactuals, and closed-loop behavior remains untested.

**Artifacts.** Under
`/mnt/new-volume/vla-lens/rq019-pose-exchange-pilot/vla_lens/patch_studies/`:
`rq020-pose-exchange-localization`, `rq020-pose-exchange-confirmation`, and
`rq020-pose-exchange-broad-localization`. Each contains the exact plan, full
action chunks, pair/trial tables, decisions, failures, hashes, bootstrap
analysis, and confidence intervals. Donor hidden states are intentionally
temporary because they can be rebuilt from the saved traces and plans.

**Runtime validation, not the RQ-020 answer.** The donor source-patch machinery
was validated on existing matched pair `506357b9a6fb733db638`: task-73 seeds
2000 and 2004, where the distractor `white_yellow_mug_1` moved. This is not the
planned target/distractor pose exchange, so it is an engineering smoke and one
exploratory negative example, not the scientific pilot.

The ROCm run passed the replay gate, every hook fired once, recipient-self patch
was an exact no-op, and donor-self patch reproduced the donor action under the
same recipient noise. The natural recipient-to-donor action distance was
`0.47157`. Patching the 20 union-of-old/new mug tokens at layer 8 changed the
action by `0.06624`, but its direction agreement was only `0.2979`, transfer
fraction `0.04184`, and donor recovery `0.03251`. It failed all three declared
transfer gates. After bfloat16 conversion, shuffled-donor, random, and
wrong-region controls were raw-norm matched to approximately the same hidden
perturbation. Their action changes were `0.04780`, `0.04299`, and `0.05125`,
respectively, versus `0.06624` for the intended patch.

**Runtime validation decision.** The hook, shared-noise comparison, in-memory
donor cache, action saving, and control machinery work. This example does not
show meaningful donor-directed transfer, and it should not influence layer
selection for the pose-exchange pilot. Continue to RQ-019 pair construction,
then run the predeclared layer/token sweep.

**Runtime validation artifact.**
`intervention_run-rq-020-source-patch-machinery-smoke-on-an-existing-moved-object-pair-f65174098b`.
The saved artifact contains recipient, donor, patched, and five control action
chunks. Request: `configs/interventions/rq020_existing_pair_layer8_smoke.json`.
Replay report:
`vla_lens/intervention_reports/rq020_existing_pair_layer8_smoke.json`.

**Study-runner implementation.** A reconstructable study job now expands pair,
layer, and token-region axes into deterministic trial IDs. Live execution uses
one PI0.5 model load, captures all requested donor layers in one call per pair,
checks recipient replay once per pair, and checkpoints after every trial. A
retry skips completed work. Permanent outputs are full action chunks, pair and
trial tables, decisions, failures, hashes, and the exact request. Donor hidden
states remain in memory and are rebuilt from the saved traces and plan instead
of consuming permanent disk. Runtime validation uses
`configs/interventions/rq020_existing_pair_runner_smoke.json`; it remains an
engineering smoke and cannot answer RQ-020.

The ROCm runner smoke completed both declared trials with one model load, one
two-layer donor cache fill, and one three-repeat recipient replay check. A
second identical command loaded no model and skipped both completed trials.
The permanent study is 523 KB and contains 16 full 50-by-7 action chunks plus
the pair, trial, decision, failure, and sweep records; it contains no hidden
states. Layers 0 and 8 were both `nonspecific`. Layer 0 had direction agreement
`0.2045` and transfer fraction `0.04554`; layer 8 had direction agreement
`0.2979` and transfer fraction `0.04184`. These negative smoke values only
confirm that the study store and aggregation preserve the measurements. Study:
`vla_lens/patch_studies/rq020-existing-pair-runner-smoke`; report:
`vla_lens/intervention_reports/rq020_existing_pair_runner_smoke.json`.

**Workflow result.** The reusable runner now plans deterministic pair/layer/token
grids, checks the capture and replay contract before loading the model, loads
PI0.5 once, caches donor states only in memory, saves reconstructable action
artifacts after every trial, resumes incomplete studies, and produces
pair-bootstrap summaries automatically. The Intervention page reads those
summaries and shows the layer-by-scope matrix beside the matched scene frames.

## RQ-022: Action-Expert Path Localization

**Question.** After the broad visual-prefix patch stops transferring the
pose-exchange action change, where and when does the corresponding difference
appear in PI0.5's action expert?

**Hypothesis.** Replacing donor action-expert hidden states at matching layer,
denoising step, and action-horizon position will increasingly move the
recipient toward the donor in late expert layers. The effect should beat
same-sized shuffled and random perturbations.

**Method.** Reused the five validated RQ-019 book/mug pose-exchange pairs and
their shared initial noise. The localization pass patched all 50 action
positions at expert layers 0, 4, 8, 12, 16, and 17 on every one of PI0.5's ten
denoising steps. All six donor layers were collected in one donor generation
per pair and kept only in memory. A confirmation pass at layers 16 and 17 added
recipient-self, donor-self, zero-strength, shuffled-position, and
norm-matched-random controls. Every expert hook fired the expected ten times,
once per denoising step, and was removed after the action call.

**Layer result.** Early donor expert states were not compatible replacements
for the recipient state: mean transfer was `-93.48%` at layer 0, `-93.25%` at
layer 4, `-86.01%` at layer 8, `-34.03%` at layer 12, `99.45%` at layer 16,
and `99.9978%` at layer 17. The five-pair bootstrap interval was
`98.30%-100.61%` at layer 16 and `99.9975%-99.9981%` at layer 17. All five
layouts were positive at both late layers. A negative early-layer transplant
does not show that the layer lacks scene information; replacing the whole
state can put later computation off the recipient's trajectory.

**Controls.** Recipient-self and zero-strength patches were exact no-ops.
Donor-self reproduced the donor action exactly. At layer 16, shuffled action
positions transferred `30.04%` and norm-matched random values transferred
`2.77%`, versus `99.45%` for the aligned donor patch. At layer 17 the same
controls transferred `37.87%` and `2.03%`, versus `99.9978%`. The aligned
patch beat the strongest negative control by 69.41 percentage points at layer
16 and 62.12 points at layer 17.

**Action-horizon narrowing.** At layers 16 and 17, patching only the first ten
future action positions transferred `5.87%` and `6.69%`; the middle ten
transferred `33.16%` and `34.50%`; and the last ten transferred `19.59%` and
`21.64%`. This mostly describes where the natural donor and recipient action
chunks differ. At the final layer, replacing one horizon position nearly
replaces that output position directly, so the larger middle value is not
evidence for a better middle-horizon scene representation.

**Denoising-step narrowing.** Patching all action positions only during early
steps 0-2 transferred `4.13%` at layer 16 and `4.11%` at layer 17. Middle steps
3-6 transferred `13.66%` and `13.58%`; late steps 7-9 transferred `29.58%`
and `29.84%`. These isolated blocks do not add to the all-step result. After
the first denoising update, donor and recipient have different current action
guesses, so later hidden differences mix scene conditioning with an already
diverged action trajectory. The supported result is only that late partial
trajectory replacement changes the final output more. A fixed-action-state
comparison is required before saying that the scene-conditioned correction is
stronger late.

**Conclusion.** This is a whole-action-state replacement result. By expert
layer 16, replacing all 50 donor action states across every denoising step is
enough to reproduce nearly all of the donor's open-loop output under shared
noise. Because this replaces most of the action trajectory close to the output,
it is a useful plumbing and interchangeability check rather than semantic
localization. It does not identify an object-specific representation or isolate
the effect of scene context from the current action guess.

**Architectural correction.** The visual-prefix layer-output hook used in
RQ-020 runs after that same layer has already written its key/value tensors to
the prefix cache. Therefore VLM layer 16 and expert layer 16 are not matched
sides of one causal boundary. Their complementary transfer curves are real,
but they do not yet prove that the handoff occurs specifically between VLM
layers 12 and 16. The clean next path experiment is to patch the donor prefix
key/value cache consumed by selected expert layers, preserving key and value,
head, token, and rotary-position alignment.

**Artifacts.** Under
`/mnt/new-volume/vla-lens/rq019-pose-exchange-pilot/vla_lens/patch_studies/`:
`rq022-expert-action-localization`, `rq022-expert-action-confirmation`,
`rq022-expert-action-horizon`, and `rq022-expert-denoise-{early,middle,late}`.
Together these saved studies use about 22 MB. They contain exact plans, full
50-by-7 action arrays, hashes, hook counts, controls, pair tables, bootstrap
intervals, and reconstruction metadata, but no permanent hidden tensors.

**Workflow result.** Named PI0.5 patch sites now distinguish visual-prefix and
action-expert streams. The runner can capture many expert layers in one donor
generation, align repeated hooks by denoising step, select action positions or
denoising ranges, checkpoint every trial, and expose the saved stream and step
scope in the existing Intervention page. See
`docs/pi05-action-stream-patching.md` for commands and artifact behavior.

## RQ-021, RQ-023, And RQ-024: Controlled Scene-To-Behavior Campaign

RQ-024 is the active campaign-level question: does PI0.5 combine scene,
instruction, and robot state into appropriate, reusable behavior rather than
familiar-layout or generic-visual shortcuts? RQ-021 supplies the one-factor
scene comparisons. RQ-023 supplies the eventual closed-loop causal test. These
remain separate evidence claims and advance only through explicit gates.

The auditable plan is
`configs/campaigns/rq024_controlled_scene_to_behavior.yaml`; the reusable agent
protocol is `docs/autonomous-research-campaigns.md`, and the human control
surface is [GitHub issue #37](https://github.com/thajpo/vla-lens/issues/37).
The program is now an explicit 16-study graph rather than the earlier P0/P1
campaign sketch.

**Behavior branches.** FOUNDATION deterministically orders 24 task-object
families, preassigns 12 discovery and 12 confirmation candidates before
outcomes, runs three baseline rollouts per family, and selects the first six
eligible families in each already-fixed order. Geometry and named-target
binding then run as independent discovery and confirmation branches. A failure
in one branch does not close or select the other.

**Semantics and mechanisms.** Identity readout first uses discovery families
only. It tests PI0.5 decodability against metadata, wrong-region, and the full
readout-selection null; generic vision is reported separately as a specificity
comparison. Geometry and binding each have their own fixed-action-state
internal, prefix-key/value causal, causal-confirmation, and rollout chain. Their
metrics never merge. Internal selection is restricted to the discovery pool
and to sites the next child can actually patch.

**Confirmation access.** Confirmation families expose only baseline fields
during FOUNDATION and frozen behavior outputs during behavior confirmation.
No confirmation-family activation, readout, key/value, fixed-action-state, or
patch output may be opened until every activated phase-two child is committed
and independently audited. Confirmation is prospective for that unopened
endpoint, not described as globally untouched.

**Measurement correction.** Internal and narrow causal studies now use an
explicit dimensionless target in PI0.5's normalized 50-by-7 velocity space.
They do not project normalized velocity directly onto meters. Physical geometry
gain and named-target distance return only in their separately confirmed
closed-loop branches.

## Not Yet Run

- Feature-level sparse object-location probes beyond the fixed MLP battery
- Set decoder for unordered scene objects
- Filtered first-moved/first-lifted target probes
- Target-parse VLM probe (the existing selector matched no rows)
- Raw-norm-matched wrong-ROI control for RQ-018
- Small signed direction-dose comparison for the RQ-018 recipient
- PI0.5 prefix key/value-cache patching at matched expert layers
- One-factor object counterfactuals separating identity, pose, and appearance
- Controller-level physical action and trajectory comparison
- Fixed-current-action-state scene comparison across denoising time

## Current Priority

1. Prepare the exact FOUNDATION child from
   `configs/campaigns/rq024_foundation.child.template.yaml`: implement and hash
   the LIBERO-90 family parser, candidate/rejection table, exact 72-trial table,
   separate environment/policy/flow-noise seeds, checkpoint receipt, and
   machine-readable capture-environment receipt.
2. Do not execute FOUNDATION until `scripts/validate_research_child.py` verifies
   its committed files and a separate lock receipt contains passing schema,
   design, runner, and budget audits. The current template is intentionally
   blocked rather than pretending those missing inputs exist.
3. Add simulator contact telemetry as its own measurement. Existing
   end-effector-distance logic is a proxy and must not be called contact or
   collision. FOUNDATION eligibility itself still uses simulator success.
4. After FOUNDATION passes, run geometry discovery, binding discovery, and
   reliability design independently. Do not prioritize issue #36's broad prefix
   patch sweep before a behavior branch is confirmed.
5. Keep RQ-015 as exploratory readable identity, RQ-018 as a negative
   specificity result, RQ-019 as an open-loop counterfactual, and RQ-020/RQ-022
   as broad plumbing constraints until fresh controlled evidence changes them.
