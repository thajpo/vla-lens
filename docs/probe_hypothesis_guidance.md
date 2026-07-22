# Probe Hypothesis Guidance

Status: active guidance.

Last updated: July 21, 2026.

This document is the repo-local protocol for agents and humans proposing new
probe work. Existing probe artifacts remain the post-training source of truth.
The missing step this guide covers is the review surface before compute is
spent: what question is being asked, what labels are used, which model sites are
searched, and what boring explanations must be checked.

## Core Rule

Probe training should produce a reusable artifact, not just a score. Before
training, the experiment card must put the choices that change the research
claim first:

- Research question in normal language.
- Target label or regression value, including how it is constructed.
- Row cohort and filters.
- Model representation sites.
- Token/feature aggregation.
- Feature dimension after aggregation and effective training-row count.
- Split policy and which split is used for selection.
- Metadata baselines.
- Sweep axes such as layer, policy call, feature site, and model type.
- Main metrics and known failure modes.

The card then shows method choices such as linear versus MLP and the planned
sweep. File paths, cache keys, library versions, and similar internal details
are saved in the artifact, but they should not dominate the review. This keeps
"train a probe on X" understandable without hiding anything needed to check or
repeat the work.

The goal is not to stop broad search. It is to prevent broad search from
quietly becoming a test-set fishing expedition.

## Default Cheap Battery

When the user asks whether activations encode a target, do not ask them to pick
between inexpensive readouts. Preflight the request, then train both the linear
probe and the standard small MLP unless the spec explicitly narrows the models.
Sweep every declared layer on validation and save the complete comparison. The
validation-selected readout is replayable; test remains a reporting split.

Tokenwise, feature-level, layer-mixture, and object-conditioned variants should
also run automatically when the capture supports them and a runner exists. If a
runner is missing, say that plainly instead of silently falling back to global
mean pooling. Ask the user only when the choice changes the target or cohort,
requires new capture, launches an intervention, or has material compute/storage
cost.

The MLP is a capacity check, not automatically the stronger scientific claim.
Interpret the comparison: linear success means easy access; MLP-only success
means a nonlinear readout found signal and needs stronger confound controls.

## What Can Be Automated

These checks should be handled by scripts whenever possible:

| Check | Why |
| --- | --- |
| Resolve selected feature rows | Confirms the selector actually maps to data. |
| Resolve target labels | Catches missing labels before training. |
| Apply row filters | Shows whether the intended cohort survives filtering. |
| Summarize splits | Prevents training/eval on empty or accidental splits. |
| Summarize class support or regression variance | Prevents one score from hiding under-supported claims. |
| Report feature dimension after pooling | Makes sample-complexity risk visible before training. |
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

## Sample Complexity And Pooling

Probe aggregation is a statistical choice, not just a convenience. PI0.5 hidden
states are high-dimensional, while current broad datasets are usually modest:
roughly 1000 episodes and a few thousand policy-call rows. A 1024-dimensional
pooled feature can already be large relative to the number of independent
held-out-task examples. Concatenating tokens, layers, calls, or sites can turn a
probe into a powerful search procedure unless the dataset is much larger or the
claim is explicitly exploratory.

Before training, record:

- independent unit count: episodes, tasks, and policy-call rows after filters
- feature dimension after pooling
- rows per target class, or regression variance per split
- number of readouts searched by layer, call, site, model, and pooling variant
- whether pooling is the claim-bearing default or an exploratory upper bound

Use this rule of thumb for PI0.5-style first-pass probes:

| Effective training rows | Feature shape | Recommended probe posture |
| ---: | --- | --- |
| `< 500` | any high-dimensional activation | Do not make a claim-bearing probe unless the target is extremely simple and baselines are strong. Prefer more data, narrower labels, or descriptive analysis. |
| `500-2k` | one pooled vector, about 1024D | Keep the linear readout as the claim-bearing result; run the small MLP automatically as a fragile capacity check with trained shuffled-label controls. |
| `2k-10k` | one pooled vector, about 1024D | Train linear and small MLP probes. Treat MLP-only gains as exploratory and inspect metadata and shuffled-label controls. |
| `> 10k` | pooled or moderately reduced features | Train linear and small MLP probes; PCA, random projection, and limited pooling ablations are reasonable. Select on validation and report the number searched. |
| `> 50k` | richer tensor summaries | Add learned pooling, token concatenation, or cross-layer variants when supported; they remain exploratory unless confirmed by simpler localization evidence. |

These thresholds are not laws. They are guardrails for avoiding the common
failure mode where the probe has enough capacity to exploit task identity,
object identity, policy-call timing, or split quirks.

Default claim-bearing pooling:

- choose a semantically justified feature site before training
- mean-pool the relevant token set
- train one layer/site/call readout at a time
- use linear/logistic regression for classification and ridge regression for
  scalar targets
- standardize features using train split statistics only
- choose layer/site/call on validation, then report test
- include metadata baselines that could explain the label without activations

Pooling and dimensionality variants should be labeled by purpose:

| Variant | Use when | Claim strength |
| --- | --- | --- |
| Mean token pooling | Default first pass; signal should be broadly available in a token group. | Strongest simple decodability claim if it beats baselines. |
| Final/special token only | Architecture gives that token a clear summary role. | Clean if justified; arbitrary otherwise. |
| Max pooling | Signal is expected to be sparse over tokens. | Riskier; can latch onto outliers. |
| PCA fitted on train | Rows are too few for raw high-dimensional features or regression is unstable. | Useful stability check; report component count and variance. |
| Random projection | Need a dimension-sensitivity control. | Diagnostic, not primary evidence. |
| Sparse linear probe | Need localization over channels/features. | Requires nested validation or locked regularization. |
| Concatenate tokens/layers/calls | Asking whether information exists somewhere in a large tensor. | Exploratory upper bound only unless replicated with simpler pooling. |
| Learned attention pooling or MLP | Testing whether a richer extractor can find signal. | Exploratory; must be paired with controls/selectivity-style checks. |

If a richer aggregation wins but the simple mean-pooled probe fails, the honest
interpretation is usually "the signal may be present somewhere in the tensor,"
not "the representation cleanly exposes this property."

## Data Expansion And Capture Planning

When a probe needs richer pooling or more target granularity than the current
dataset can support, prefer more independent environments or tasks over adding
more nearby rows from the same episodes. More policy-call rows help, but they
are correlated within episode and task. For held-out-task claims, the effective
sample size is closer to the number and diversity of tasks/episodes than to the
raw activation-row count.

If broad LIBERO coverage is saturated, new probe questions may require:

- adding environment/task support beyond the current benchmark mix
- a probe-specific mechanistic capture profile that records only the sites
  needed for many more episodes
- object-local or candidate-wise row construction so each episode yields more
  scientifically meaningful examples
- locked confirmation captures after exploratory sweeps find a promising
  question/pooling combination

Mechanistic-light captures are appropriate for broad first-pass probing when
they preserve the relevant feature sites. Probe-specific profiles should be used
when the question is known and the bottleneck is episode count, not model-site
coverage.

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
probe. Markdown shows the short experiment card by default. Add `--details` for
the full split, label-support, and baseline tables, or use JSON when another
tool needs every field.

## After Training

New generic probe artifacts save the fitted model, the exact chosen rows, the
complete training recipe, and the details of any uncertainty calculation. They
do not copy the large activation matrix into every artifact. The feature cache
is only a speed-up and may be deleted; replay can rebuild the features from the
source capture.

Use the saved artifact directly:

```bash
uv run python scripts/use_vla_lens_probe.py \
  /path/to/dataset ARTIFACT_ID explain

uv run python scripts/use_vla_lens_probe.py \
  /path/to/dataset ARTIFACT_ID replay

uv run python scripts/use_vla_lens_probe.py \
  /path/to/dataset ARTIFACT_ID use \
  --features compatible_features.npy \
  --output predictions.npy
```

`explain` shows the experiment card. `replay` rebuilds the selected features
and checks them against the saved predictions without fitting. `use` applies
the fitted probe to another compatible feature matrix. Older artifacts remain
readable, but the command says plainly when they do not contain enough data to
replay or reuse.

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
  models: [linear, mlp]
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
8. Effective rows, feature dimension, and pooling/capacity posture.
9. Automatic preflight warnings.
10. Why this is worth training.
11. What result would make it worth inspecting or intervening on.

After approval, the agent can train the existing probe specs and rely on normal
probe artifacts for ranking and UI inspection.
