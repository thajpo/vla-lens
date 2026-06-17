# Probe Hypothesis Guidance

Status: active guidance.

Last updated: June 17, 2026.

This document is the repo-local protocol for agents and humans proposing new
probe work. Existing probe artifacts remain the post-training source of truth.
The missing step this guide covers is the review surface before compute is
spent: what question is being asked, what labels are used, which model sites are
searched, and what boring explanations must be checked.

## Core Rule

Probe training should produce an auditable artifact, not just a score. Before
training, the proposed probe must make these choices visible:

- Research question in normal language.
- Target label or regression value, including how it is constructed.
- Row cohort and filters.
- Model representation sites.
- Token/feature aggregation.
- Split policy and which split is used for selection.
- Metadata baselines.
- Sweep axes such as layer, policy call, feature site, and model type.
- Main metrics and known failure modes.

The goal is not to stop broad search. It is to prevent broad search from
quietly becoming a test-set fishing expedition.

## What Can Be Automated

These checks should be handled by scripts whenever possible:

| Check | Why |
| --- | --- |
| Resolve selected feature rows | Confirms the selector actually maps to data. |
| Resolve target labels | Catches missing labels before training. |
| Apply row filters | Shows whether the intended cohort survives filtering. |
| Summarize splits | Prevents training/eval on empty or accidental splits. |
| Summarize class support or regression variance | Prevents one score from hiding under-supported claims. |
| List metadata baselines | Makes boring explanations visible. |
| Flag target-as-baseline leakage | Avoids probes that compare against their own label. |
| Count planned readouts | Makes "train all layers" an explicit sweep. |
| Warn when selection equals test | Prevents using final evaluation as model/site selection. |

These checks do not decide whether a scientific question matters. They only make
the proposed experiment reviewable.

## What Needs Human Judgment

Agents should not claim to solve these automatically:

- Whether the question is important.
- Whether the label is scientifically clean or just convenient.
- Whether a strong metadata baseline makes the probe boring.
- Whether a decodable signal is intervention-worthy.
- Whether the result supports a causal mechanism.

The default interpretation of a probe is "decodable under this data contract."
Causal language requires intervention or replay evidence.

## Preflight Command

Use the preflight script before training a new probe spec:

```bash
uv run python scripts/preflight_vla_lens_probe.py \
  /path/to/dataset \
  --spec configs/probes/my_probe.yaml \
  --format markdown
```

For machine-readable output:

```bash
uv run python scripts/preflight_vla_lens_probe.py \
  /path/to/dataset \
  --spec configs/probes/my_probe.yaml \
  --format json \
  --output runs/preflight/my_probe.json
```

The preflight path reuses the normal YAML probe spec. It inspects selected rows,
target labels, filters, splits, baselines, and sweep size, but it does not fit a
probe.

## Probe Spec Additions

Normal training fields still drive execution. Add these optional review fields
when an agent proposes new work:

```yaml
name: PI0.5 broad 1000 target contact next - expert hidden
question: Will the robot contact the instructed target object in the next action window?
hypothesis_family: object choice before contact
intended_claim: target contact intent is decodable before contact
```

The current trainer ignores these fields for fitting, but preflight and review
surfaces use them.

## Default Split Policy

For broad PI0.5 probe work, the default claim-bearing split is:

```yaml
split:
  kind: heldout_task
  column: split
  train_value: train
  selection_value: val_heldout_task
  test_value: test_heldout_task
  eval_values: [val_heldout_task, test_heldout_task]
```

Use validation for layer/site/model selection. Treat the test split as the final
report surface, not as a search surface.

## Sweeps Are Allowed

It is often correct to train all plausible layers or policy calls because we do
not know where a signal might live. The guardrail is not "avoid sweeping." The
guardrail is:

- Declare the sweep axes before training.
- Save all readouts, not only the best one.
- Select on validation.
- Report the test split only after selection.
- Include null and metadata baselines.
- Record how many readouts were searched.

Large sweeps should be treated as discovery unless followed by a locked
confirmation run.

## Required Baseline Thinking

Every probe should include majority/train-mean baselines and relevant metadata
baselines. Common metadata candidates:

| Probe family | Required baseline candidates |
| --- | --- |
| Object choice | task ID, prompt, target object, receptacle, policy-call index |
| Receptacle/destination | task ID, prompt, target object, receptacle, scene family |
| Outcome/failure | task ID, prompt, object identities, scene family, policy-call index |
| Motor execution | task ID, policy-call index, phase-like labels, target object |
| Language parsing | prompt, task ID, benchmark, target object parse status |
| Pose/orientation | task ID, object ID/slot, scene family, policy-call index, previous pose when available |

If a metadata baseline matches or exceeds the activation probe, the result may
still be operationally useful, but it is weak mechanistic evidence.

## Concrete Example: Euler Orientation Decoding

Question:

> Is each object's orientation decodable from model representations?

A reviewable first probe plan should say:

```yaml
name: PI0.5 broad 1000 object orientation decoding
question: Is current object orientation decodable from model representations?
hypothesis_family: scene-state decoding
intended_claim: object orientation information is linearly available at selected model sites
target:
  name: object_orientation_euler
  kind: regression
  source: table
  table: scene_state
  column: object_orientation_euler
  missing_policy: drop
features:
  module: pi05.expert.layers.*
  tensor_type: hidden_tokens
  token_kind: action
  layers: [0, 4, 8, 12, 17]
  timesteps: all
  policy_calls: [0, 1, 2, 3, 4, 5, 6]
  generation_step: final
  reduction: mean
  dtype: float32
split:
  kind: heldout_task
  column: split
  train_value: train
  selection_value: val_heldout_task
  test_value: test_heldout_task
  eval_values: [val_heldout_task, test_heldout_task]
baseline:
  - majority_class
  - task_id
  - prompt
  - policy_call_index
  - scene_family
sweep: [layer, policy_call_index]
probe:
  models: [linear]
```

This example may need target adapter work before it runs, because object pose is
usually multi-object and multi-coordinate. The preflight step should make that
failure explicit rather than allowing an agent to silently redefine the target.

Specific risks for orientation probes:

- Euler angles wrap around, so raw absolute error can be misleading.
- Quaternion and Euler targets have different geometry.
- Object slots can leak scene layout.
- Task and scene may strongly predict object pose.
- Current pose is less mechanistically interesting than future pose or pose
  error unless the research question is pure state representation.

## Ready-To-Review Agent Output

When asked to "train some probes," an agent should first produce a short review
packet:

1. Proposed question.
2. Exact target construction.
3. Feature sites and sweep axes.
4. Row filters.
5. Split policy.
6. Baselines.
7. Metrics.
8. Automatic preflight warnings.
9. Why this is worth training.
10. What result would make it worth inspecting or intervening on.

After approval, the agent can train the existing probe specs and rely on normal
probe artifacts for ranking and UI inspection.
