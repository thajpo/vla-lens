# PI0.5 Semantic-To-Motor Binding Program

## Thesis

The project is not broad VLA interpretability. The sharper question is semantic-to-motor binding in generative action heads:

> When target information is represented inside a VLA, does it causally constrain the generated robot trajectory? If not, where does that binding break?

The current results suggest target identity and geometry can be decodable while manipulation still fails. The next experiments should therefore distinguish information availability from causal use.

## Core Chain

Analyze failures along this chain:

```text
available -> used -> dominant -> correct motion -> successful manipulation
```

Current status:

| Link | Current Evidence | Status |
|---|---|---|
| available | target identity, geometry, and flow-state information are decodable | mostly yes |
| used | unknown whether target information causally redirects action | open |
| dominant | unknown whether target signal beats scene/distractor/motor priors | open |
| correct motion | object-relative flow metrics show distractor-aligned and phase-dependent behavior | mixed |
| successful manipulation | some recipient-fixed handoff changes lift without changing first-call state | mixed |

## Main Confound

Probes can learn benchmark priors:

```text
object label / scene / layout -> typical target position / success / trajectory
```

This can look like grounded target understanding even when the online scene is not causally controlling action. Strong evidence must therefore come from recipient-fixed or counterfactual tests where scene priors are controlled.

## Priority Experiments

### 1. Recipient-Fixed Handoff Binding Tests

Hold recipient scene/layout fixed and change handoff conditioning.

Readout:

- target-vs-distractor flow alignment
- target-relative distance improvement
- gripper close timing and distance-to-target at close
- future target lift delta
- commitment step inside denoising
- final success only as secondary

Interpretation:

| Result | Meaning |
|---|---|
| target patch redirects motion | target representation is causally bound to trajectory |
| target patch redirects motion but lift still fails | binding works; failure is downstream control/contact/recovery |
| target patch does not redirect motion | decodable target information is weakly or non-causally coupled |
| target patch only works with distractor/scene-prior ablation | target signal is used but overridden |
| patch destroys behavior | intervention is likely OOD or globally load-bearing |

### 2. Object-Relative Flow Commitment

For each policy call and flow step, measure when the denoising trajectory commits to target, distractor, or bad motor mode.

Useful metrics:

- earliest target-commit flow step
- earliest distractor-commit flow step
- stable nearest-object trajectory label
- action-chunk similarity to final chunk
- target-relative endpoint and prefix displacement

### 3. Competing-Signal Override Tests

Run paired interventions:

- target boost
- target ablation
- distractor/scene-prior ablation
- target boost plus competing-signal ablation
- norm-matched random controls

The goal is to distinguish absent binding from binding that loses a competition against stronger priors.

### 4. Trajectory Mode Diversity

Sample multiple action chunks from the same observation and cluster in object-relative trajectory space.

Questions:

- do successful modes exist but have low probability?
- does random activation noise move probability mass between modes?
- does best-of-N improve target-directed motion or final success?
- does closed-loop replanning destroy otherwise good chunks?

## Current Smoke Result

`docs/experiments/pi05-semantic-to-motor-binding-handoff-smoke.md` applies the recipient-fixed binding readout to the captured `cream_cheese` handoff smoke.

Key result:

- first policy call is identical across baseline, success-donor, and failure-donor conditions
- failure-donor does not show a clean early wrong-target redirection relative to baseline
- the observed flip is more consistent with later lift/recovery or phase-specific binding than missing target representation

This is only a smoke test. It should be repeated across more recipient-fixed handoff captures before making a strong claim.

## Near-Term Execution

1. Expand recipient-fixed handoff captures for layouts where donor handoff flips lift.
2. Run `analyze_pi05_object_relative_flow.py` and `analyze_pi05_binding_interventions.py` on each capture family.
3. Add phase labels and close/lift variables to the binding delta report.
4. Add token/layer-restricted handoff patches only after full-handoff deltas are stable.
5. Use object/scene-prior probes only as controls, not as headline evidence.

## Paper Frame

Working title:

> Decodable Is Not Used: Semantic-To-Motor Binding Failures in Flow-Matching Vision-Language-Action Models

Contribution:

> Robot-space causal interpretability for generative VLA action heads, evaluated by whether target representations bind to object-relative trajectory formation.
