# PI0.5 Capture Profiles for Interpretability

This document explains what each PI0.5 capture profile is for in interpretability terms.

The short version:

```text
rollout              behavior only
features             representation probes
mechanistic_sampled  default inspector and probe dataset
mechanistic_all      all-layer semantic inspector
internals_sampled    sampled operation internals
audit_sampled        sampled circuit-boundary audit
audit_windowed       adjacent-layer whole-episode circuit/transcoder capture
audit_full           exhaustive raw/debug capture
custom               explicit one-off profile
```

Do not pick a profile by asking "how much can I capture?" Pick it by asking
"what interpretability claim do I want to test?"

## Profile Ladder

| Profile | Best for | Cost | Main limitation |
| --- | --- | --- | --- |
| `rollout` | Behavior datasets, action metrics, success/failure review | Lowest | No model internals |
| `features` | Representation probes and feature availability | Low | No attention, cache, action-head I/O, or internals |
| `mechanistic_sampled` | Normal VLA Lens inspection, probes, attention routing, denoising localization | Medium | Sampled layers only, no circuit boundaries |
| `mechanistic_all` | All-layer semantic curves and serious single-trace inspection | High | Still no Q/K/V, MLP internals, residual boundaries |
| `internals_sampled` | Operation-level inspection at landmark layers | High | Missing enough boundaries for strong circuit claims |
| `audit_sampled` | Sampled-layer circuit audits, patching prep, same-layer transcoders | Very high | Sampled layers only |
| `audit_windowed` | Adjacent-layer whole-episode circuit/transcoder work | Very high | Expensive, not dataset-scale by default |
| `audit_full` | Raw debugging and exhaustive audit | Highest | Too large/noisy for normal workflows |
| `custom` | Explicit experimental budgets | Variable | Easy to create incomparable traces |

## `rollout`

Use `rollout` when the question is:

```text
What happened in the episode?
Did the policy succeed?
What did the robot do?
What objects/cameras/actions/states were present?
```

It captures episode and environment data such as frames, prompts, actions,
robot state, object/camera context, policy calls, and metadata. It does not
capture model activation arrays.

Good uses:

- behavior dataset construction
- video review
- action statistics
- success/failure slicing
- task/seed coverage checks
- phase/failure labeling
- baselines before expensive capture

Do not use it for:

- hidden-state probes
- attention routing
- feature browsing
- activation patching
- causal model-internal claims

Interpretability claim level:

```text
Behavioral evidence only. No model-internal evidence.
```

## `features`

Use `features` when the question is:

```text
Where is information linearly available?
Do VLM or Expert hidden tokens encode object/task/action variables?
Which sampled layer is worth inspecting later?
```

It captures representation tokens at sampled landmark layers:

```text
VLM / Expert layers:
  [0, 4, 8, 12, 17]

Families:
  representations = tokens
  attention       = none
  cache           = none
  action_head     = none
  internals       = none
```

Good uses:

- cheap representation probes
- layer-wise feature availability curves
- target object, task ID, gripper state, action-direction probe pilots
- finding promising layers before collecting attention or internals
- early SAE pilot data if all you need is residual/hidden states

Do not use it for:

- attention-to-object claims
- K/V conditioning analysis
- action-head attribution
- denoising attention analysis
- circuit-level claims

Interpretability claim level:

```text
This variable is decodable or represented at this sampled layer/token group.
It does not show that the model used the variable.
```

## `mechanistic_sampled`

Use `mechanistic_sampled` as the default model-inspector profile.

Question:

```text
How do sampled VLM/Expert layers represent, attend, cache, and produce actions?
```

It captures landmark layers:

```text
VLM / Expert layers:
  [0, 4, 8, 12, 17]

Families:
  representations = tokens
  attention       = full_probs
  cache           = VLM layer-wise K/V
  action_head     = input/output
  internals       = none
```

Good uses:

- normal VLA Lens episode inspection
- object-grounded attention routing
- action-suffix to image/language/action-token attention views
- denoising-step localization
- probe grids over sampled layers
- K/V handoff inspection via `pi05.vlm.past_key_values`
- action-head input/output analysis
- scalable-ish dataset capture compared with audit profiles

Do not use it for:

- Q/K/V circuit analysis
- MLP feature write/read claims
- residual stream patching at component boundaries
- attention-logit reconstruction
- strong causal claims

Interpretability claim level:

```text
Good for finding where a phenomenon appears and how attention is routed.
Not enough to prove the component-level mechanism.
```

## `mechanistic_all`

Use `mechanistic_all` when the question is:

```text
How does a semantic signal move across all VLM/Expert layers?
```

It captures the same semantic families as `mechanistic_sampled`, but all layers:

```text
VLM / Expert layers:
  [0..17]

Families:
  representations = tokens
  attention       = full_probs
  cache           = VLM layer-wise K/V
  action_head     = input/output
  internals       = none
```

Good uses:

- all-layer probe curves
- all-layer attention routing summaries
- serious single-trace model inspection
- deciding which layers deserve `audit_sampled`
- checking whether sampled landmark layers missed an important transition

Do not use it for:

- component-level circuit claims
- MLP/transcoder work
- attention-logit debugging
- residual-boundary patching

Interpretability claim level:

```text
Stronger localization across layers than mechanistic_sampled.
Still semantic-routing evidence, not component-level circuit evidence.
```

## `internals_sampled`

Use `internals_sampled` when the question is:

```text
What are sampled attention/MLP operations doing internally?
```

It captures sampled layers and selected operation internals. It is narrower
than `audit_sampled`.

Typical included families:

```text
sampled layers:
  [0, 4, 8, 12, 17]

selected internals:
  Q / K / V
  attention probabilities
  attention output projection
  MLP gate / up / intermediate / down
  selected Expert AdaRMS gates
  VLM K/V cache
```

Good uses:

- inspecting selected Q/K/V and MLP tensors
- debugging renderer/API support for internals
- identifying whether an attention or MLP sub-block looks promising
- early component-level hypothesis generation

Do not use it for:

- complete circuit-boundary patching
- verifying attention probabilities from logits/masks
- strong residual-stream write/read claims
- full replay or exhaustive raw debugging

Interpretability claim level:

```text
Operation-level evidence at sampled layers.
Useful for hypotheses, but not complete enough for strong circuit claims.
```

## `audit_sampled`

Use `audit_sampled` when the question is:

```text
At sampled layers, what did each component read, compute, and write?
```

It captures the same sampled layers as `mechanistic_sampled`, but adds
circuit-boundary internals.

```text
VLM / Expert layers:
  [0, 4, 8, 12, 17]

Families:
  representations = tokens
  attention       = full_probs plus audit attention internals
  cache           = VLM layer-wise K/V
  action_head     = input/output
  internals       = sampled_audit
  state_setup     = none
```

Good uses:

- same-layer skip-transcoder pilots
- component write/read analysis at landmark layers
- activation patching prep
- steering-site selection
- attention-logit and attention-probability debugging
- residual/pre/post attention and MLP boundary inspection
- checking whether an `internals_sampled` hypothesis is real enough to pursue

Good first sites:

```text
Expert residual_pre_mlp -> Expert mlp_output
Expert residual_pre_attention -> attention output / o_proj
Action-head input -> action-head output
VLM MLP input/output at layers 8 or 12
```

Measured cost from May 20, 2026:

```text
libero_object task0 seed1002:
  2,233.6 MiB
  143 steps
  3 policy calls

libero_spatial task0 seed1002:
  1,510.5 MiB
  76 steps
  2 policy calls

libero_goal task0 seed1002:
  2,229.2 MiB
  123 steps
  3 policy calls
```

Do not use it for:

- broad dataset capture without a reason
- all-layer circuit completeness
- adjacent-layer "what did L+1 consume?" questions
- replacing `mechanistic_sampled` as the normal profile

Interpretability claim level:

```text
Circuit-boundary evidence at sampled layers.
Good enough to start causal experiments, but sampled layers can miss mechanisms.
```

## `audit_windowed`

Use `audit_windowed` when the question is:

```text
What did layer L write, and what did adjacent layer L+1 consume?
```

It is the whole-episode adjacent-window profile for transcoder and circuit work.

```text
VLM / Expert layers:
  [0, 1, 4, 5, 8, 9, 12, 13, 16, 17]

Windows:
  [0,1], [4,5], [8,9], [12,13], [16,17]

Families:
  same raw role filter as audit_sampled
  internals = windowed_audit
  state_setup = none
```

Good uses:

- adjacent-layer transcoder pilots
- asking whether layer `L` writes a feature used by layer `L+1`
- checking whether a feature is preserved, transformed, routed, or erased
- cross-layer residual stream analysis
- collecting whole-episode examples for a specific circuit hypothesis

Measured cost from May 20, 2026:

```text
libero_object task0 seed1002:
  4,465.4 MiB
  144 steps
  3 policy calls
  484 model sites
  20 VLM K/V runtime-collection members
  10 per-layer K/V architecture edges
```

Do not use it for:

- default dataset-scale capture
- vague "more internals" collection
- selected-policy-call capture, unless current-state guidance explicitly changes
- all-layer completeness

Interpretability claim level:

```text
Adjacent-layer circuit evidence for selected layer windows.
This is the first profile that directly supports "write then consume" questions.
```

## `audit_full`

Use `audit_full` when the question is:

```text
Did we miss something in the capture schema?
Do we need exhaustive raw debugging?
Do we need all-layer raw internals for a selected failure or audit case?
```

It is the all-layer raw/debug profile.

It may include:

```text
all layers
residual boundaries
norm outputs
Q/K/V
attention scores/logits/probs
attention outputs
MLP internals
AdaRMS scale/shift/gate
masks
position IDs
RoPE
K/V cache
action head
```

Good uses:

- schema audit
- debugging missing tensor roles
- validating whether `audit_sampled` omits something important
- selected high-value failure cases
- short event windows if/when event-windowed capture exists

Do not use it for:

- normal inspection
- broad episode datasets
- first-pass probe work
- UI-first analysis

Interpretability claim level:

```text
Maximum raw evidence.
Also maximum noise and cost. Use only when the narrower audit profiles are insufficient.
```

## `custom`

Use `custom` when the question is:

```text
Can I test a very specific storage/analysis hypothesis?
```

Good uses:

- storage ablations
- one-off model-site combinations
- testing whether a cheaper profile can replace an expensive one
- analysis-specific capture contracts

Do not use it for:

- shared datasets without clear documentation
- benchmark comparisons unless the exact plan is recorded
- default examples

Interpretability claim level:

```text
Depends entirely on the explicit capture plan.
Record the dimensions and do not compare casually against canonical profiles.
```

## Choosing a Profile by Question

| Question | Start with |
| --- | --- |
| Did the robot succeed, fail, or behave strangely? | `rollout` |
| Is target object / gripper / phase decodable? | `features` |
| What is the default useful trace for VLA Lens? | `mechanistic_sampled` |
| Did sampled layers miss the important layer? | `mechanistic_all` |
| What are Q/K/V or MLP tensors doing at sampled layers? | `internals_sampled` |
| Can I patch or steer a sampled component boundary? | `audit_sampled` |
| Did layer L write something layer L+1 consumed? | `audit_windowed` |
| Is the schema missing raw internals? | `audit_full` |
| Can I make a cheaper profile for one analysis? | `custom` |

## What Each Profile Can Support

| Method | Minimum useful profile | Better profile |
| --- | --- | --- |
| behavior review | `rollout` | `mechanistic_sampled` |
| linear probes over hidden states | `features` | `mechanistic_sampled` or `mechanistic_all` |
| object-grounded attention routing | `mechanistic_sampled` | `mechanistic_all` |
| denoising-step localization | `mechanistic_sampled` | `mechanistic_all` |
| action-head input/output analysis | `mechanistic_sampled` | `audit_sampled` |
| SAE feature browsing | `features` | `mechanistic_sampled` |
| same-layer skip transcoders | `audit_sampled` | `audit_full` |
| adjacent-layer transcoders | `audit_windowed` | `audit_full` |
| activation patching | `audit_sampled` | `audit_windowed` or `audit_full` |
| steering from probe directions | `mechanistic_sampled` | `audit_sampled` |
| attention-logit debugging | `audit_sampled` | `audit_full` |
| schema/raw capture audit | `audit_full` | `audit_full` |

## Claims Discipline

Use this as the default standard:

```text
rollout:
  behavior happened

features:
  information is represented or decodable

mechanistic_sampled / mechanistic_all:
  information and attention routing are visible

internals_sampled:
  selected operations expose plausible mechanisms

audit_sampled:
  sampled component boundaries support causal tests

audit_windowed:
  adjacent layer windows support write/consume tests

audit_full:
  raw evidence is available, but interpretation still needs tests
```

Attention is routing evidence, not causal importance by itself.

Probes show availability, not use.

SAEs show sparse reconstruction features, not mechanisms by themselves.

Transcoders become interesting only when paired with component boundaries,
interventions, and action/rollout measurements.

The strongest VLA Lens loop is:

```text
observe -> localize -> hypothesize -> intervene -> measure -> visualize
```

Choose the cheapest profile that supports the next step in that loop.
