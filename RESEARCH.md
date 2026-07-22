# VLA Lens Research Log

Status: active source of truth for research questions and findings.

Last updated: July 22, 2026.

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

## Not Yet Run

- Feature-level sparse or nonlinear object-location probes
- Set decoder for unordered scene objects
- Object localization or camera-frame geometry probes
- MLP target pose and rotation probes
- Filtered first-moved/first-lifted target probes
- Target-parse VLM probe (the existing selector matched no rows)
- Representation interventions based on these probe results

## Current Priority

1. Inspect RQ-012 identity successes and failures across episodes and patches.
2. Test whether the identity signal localizes to the correct image region.
3. Reformulate geometry around object localization or camera-frame coordinates
   before adding more probe capacity.
4. Only then select an identity or localization intervention candidate.
