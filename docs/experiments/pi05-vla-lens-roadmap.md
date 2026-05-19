# PI0.5 VLA Lens Roadmap: Capture, Architecture, and Causal Interpretability

Status: living planning contract from the May 18, 2026 capture-profile and inspector design discussion.

Last implementation update: Phase 1 code contract is implemented, unit/type validated, and one real `audit_sampled` PI0.5/LIBERO smoke trace has been captured and validated. The next scale-up gate is a 3-5 trace benchmark across varied episode lengths.

Current planning update: adjacent-layer capture for transcoders is tracked as `audit_windowed`, a future capture tier. It is not implemented yet and must stay gated on the `audit_sampled` benchmark.

Audience: future agents, GPT Pro review, and implementation sessions before changing code.

Purpose: preserve the high-resolution product/research direction so future implementation does not drift into already-solved work, vague "more tensors" capture, or UI diagrams that misrepresent PI0.5.

This is a working document, not a museum.

As VLA Lens evolves, update this file aggressively:

```text
If an assumption is validated:
  move it to Repo Ground Truth, link the test/benchmark/doc that proves it, and delete speculative wording.

If an assumption is invalidated:
  remove or rewrite it, record the reason, and point to the failed test/benchmark/result.

If a phase is completed:
  collapse the phase into a short completed note, preserve only durable decisions, and move new follow-up work into the next phase.

If implementation reveals a better design:
  update this roadmap before expanding scope.

If the document starts preserving stale possibilities:
  delete them.
```

The rule is: work, test, validate, then prune. This file should become sharper as the repo becomes more real.

## One-Sentence Direction

VLA Lens is not missing "full capture." It already has major ontology and full-capture machinery. The missing product/research tier is `audit_sampled`: circuit-useful internals at sampled layers, plus explicit architecture-edge metadata so the UI can truthfully show how VLM prefix K/V conditions expert/action layers.

## Hard Sprint Boundary

This roadmap is broad strategy, but each implementation session should stay narrow.

The original Phase 1 coding sprint was only allowed to do:

```text
1. Add canonical `audit_sampled`.
2. Populate `/api/activation-sites` architecture.nodes and architecture.edges.
3. Benchmark 3-5 `audit_sampled` traces.
4. Add tests proving `internals_sampled` and `audit_sampled` are distinct.
5. Make the existing PI0.5 layer diagram consume architecture edges if UI work is included.
```

Everything else in this document is sequencing context, not permission to implement it during the next sprint.

That boundary is now completed. The current narrow follow-up is metadata-only
paired counterfactual capture:

```text
1. Preserve old single-trace plans.
2. Allow paired rows with trace variants.
3. Store pair metadata in manifests.
4. Expose pair groups through the server.
5. Add a minimal UI navigation affordance.
```

Explicitly still out of scope for this follow-up:

```text
Probe Grid v0
object-grounded attention dashboards
policy-call replay
activation patching
steering
SAEs
transcoders
crosscoders
CLT / attribution graph work
new UI shell or broad visual redesign
```

Reason: paired metadata is the prerequisite for future clean/corrupt patching,
but it should not pretend that replay or intervention tooling exists yet.

## Living Document Protocol

Every implementation phase must end with a validation pass and a roadmap update.

Minimum end-of-phase checklist:

```text
1. Run the tests named by that phase.
2. Run at least one real smoke path, not only unit tests, when the phase touches capture/server/frontend behavior.
3. Write or update a benchmark/result artifact when the phase changes storage, runtime, model outputs, or analysis claims.
4. Update this roadmap:
   - move validated items into Repo Ground Truth,
   - delete deferred ideas that are no longer useful,
   - rewrite failed assumptions,
   - shrink the next phase to the smallest falsifiable target.
5. Leave a short "Validated by" note with command names, artifact paths, or trace IDs.
```

Do not accumulate vague future work. If a future item is not actionable, testable, or tied to a current product/research goal, remove it.

Recommended validation note format:

```text
Validated by:
  tests:
    - uv run pytest ...
  smoke traces:
    - path/to/trace.vlatrace
  benchmark artifact:
    - docs/experiments/...
  result:
    - short factual outcome
  roadmap update:
    - what moved to ground truth
    - what was deleted or deferred
```

## Current Validation Log

### 2026-05-18/19 Phase 1 Implementation Pass

Validated by:

```text
tests:
  - uv run pytest -q
    result: 101 passed

  - uv run pytest tests/pi05_capture_success_test.py tests/pi05_token_metadata_test.py tests/pi05_full_capture_test.py tests/vla_lens_trace_mvp_test.py -q
    result: 89 passed

  - uv run ruff check scripts/run_capture_profile_smoke.py src/vla_lens/capture/adapters.py src/vla_lens/capture/records.py src/vla_lens/pi05/capture.py src/vla_lens/pi05/full_capture.py src/vla_lens/server.py src/vla_lens/validation.py tests/pi05_capture_success_test.py tests/pi05_full_capture_test.py tests/vla_lens_trace_mvp_test.py
    result: all checks passed

frontend:
  - cd frontend && npm run build
    result: TypeScript and Vite build passed

capture environment:
  - uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
    result: preserved ROCm torch stack instead of pulling CUDA wheels

  - uv pip install lerobot --no-deps
    result: installed lerobot 0.4.4 into the venv without replacing ROCm torch

  - installed LeRobot's non-torch runtime dependencies manually
    result: PI0.5 import path resolved after adding the OpenPI transformers replacement patch and peft

environment hardening:
  - AGENTS.md
    result: added explicit split-env rule for future agents

  - docs/pi05-rocm-capture-env.md
    result: documented why plain uv run is unsafe for PI0.5 ROCm capture

  - scripts/setup_pi05_rocm_env.sh
    result: added reproducible setup script for .venv-pi05-rocm

  - scripts/pi05_capture_rocm.sh and scripts/pi05_batch_capture_rocm.sh
    result: added wrapper entrypoints that avoid accidental uv sync for capture

smoke trace:
  - /usr/bin/time -v .venv/bin/vla-pi05-capture --episodes 1 --start-seed 1002 --benchmark libero_object --task-id 0 --capture-profile audit_sampled --vlatrace-out-root "/media/j/New Volume/vla-lens/pi05-audit-sampled-smoke" --delete-existing
    result: success
    trace: /media/j/New Volume/vla-lens/pi05-audit-sampled-smoke/pi05_audit_sampled_libero_object_task0_seed1002.vlatrace
    episode: pi05_audit_sampled_libero_object_task0_seed1002
    steps: 138
    policy calls: 3
    success: true
    wall clock: 1:21.41
    max CPU RSS: 16,717,972 KB

smoke validation:
  - PYTHONPATH=src .venv/bin/python validation script
    result: validate_trace_dataset(dataset).valid == true
    model_sites: 244
    runtime_collections: 1
    architecture.nodes: 14
    architecture.edges: 5

benchmark artifact:
  - docs/experiments/pi05-audit-sampled-smoke-2026-05-19.md
    result: single-trace storage/runtime note written
```

Validated facts:

```text
audit_sampled is now a canonical profile in code.
audit_sampled uses sampled VLM/Expert layers [0, 4, 8, 12, 17].
audit_sampled is distinct from internals_sampled.
internals_sampled keeps selected-op semantics.
audit_full remains the all-layer raw/debug profile.
audit_sampled persists circuit-boundary internals but excludes stored state-setup tensors by default.
Q/K/V/logit/probability declarations carry PI0.5 attention coordinate metadata.
/api/activation-sites returns architecture.nodes and architecture.edges for PI0.5 traces.
per_layer_kv_conditioning edges pair equal-index VLM and Expert layers.
non-PI0.5 activation payloads keep empty architecture metadata.
the existing PI0.5 diagram consumes architecture.edges for same-index K/V connectors.
one real audit_sampled trace can be captured with the direct venv entrypoint.
the one-trace audit_sampled smoke is much larger than mechanistic_sampled: 2.18 GiB file bytes / 2.4G du size.
the smoke trace's model arrays dominate storage: 2.18 GiB of 2.18 GiB total file bytes.
MLP internals dominate the smoke trace's model storage, followed by attention tensors.
```

Still not validated:

```text
3-5 trace audit_sampled benchmark across varied episode lengths
audit_sampled runtime slowdown vs no capture and mechanistic_sampled
audit_sampled peak GPU memory overhead
audit_sampled trace write time split from rollout time
attention_probs numeric reconstruction on a real captured trace
```

Next action before any scale-up:

```text
run scripts/setup_pi05_rocm_env.sh to create .venv-pi05-rocm
use scripts/pi05_capture_rocm.sh and scripts/pi05_batch_capture_rocm.sh for capture, not plain uv run
run 3-5 additional real audit_sampled traces
extend the storage/runtime benchmark artifact
decide whether audit_sampled needs role trimming or event-windowed capture before any larger run
estimate whole-episode audit_windowed cost from audit_sampled family-size data
then prune this roadmap again
```

### 2026-05-19 Metadata-Only Paired Capture Pass

Validated by:

```text
tests:
  - uv run pytest tests/pi05_batch_capture_test.py tests/vla_lens_trace_mvp_test.py::test_dataset_payload_groups_counterfactual_pairs -q
    result: 5 passed

  - uv run pytest tests/pi05_capture_success_test.py::test_counterfactual_capture_args_make_variant_trace_id_and_metadata -q
    result: 1 passed

  - uv run ruff check src/vla_lens/pi05/batch_capture.py src/vla_lens/pi05/capture.py src/vla_lens/server.py tests/pi05_batch_capture_test.py tests/vla_lens_trace_mvp_test.py
    result: all checks passed

  - uv run ruff check scripts src tests
    result: all checks passed

  - uv run pytest tests/pi05_batch_capture_test.py tests/vla_lens_trace_mvp_test.py -q
    result: 71 passed

  - uv run pytest -q
    result: 104 passed

frontend:
  - cd frontend && npm run build
    result: TypeScript and Vite build passed
```

Validated facts:

```text
paired counterfactual capture is implemented as capture_design metadata, not a capture profile.
old six-column episode plans still parse.
paired episode plans can contain two rows with the same benchmark/task/seed/profile.
counterfactual_role or trace_variant gives each side a distinct trace_id suffix.
the batch runner auto-links exactly two members in a counterfactual_group_id with paired_trace_id.
capture CLI accepts and stores pair metadata in manifest.metadata and capture_request.
/api/dataset includes counterfactual_pairs.
/api/counterfactual-pairs exposes the same grouping with count.
the episode navigation bar shows a compact pair switcher when the active trace belongs to a group.
```

Still not validated:

```text
real PI0.5 prompt-target-swap pair capture
true prompt mutation support inside the LIBERO/PI0.5 capture path
side-by-side synchronized video/tensor comparison
policy-call replay for paired traces
activation patching across paired traces
```

Next action before causal work:

```text
capture one real prompt-target-swap pair or build the prompt mutation hook needed to do so.
verify both traces share matched scene fields and differ only in changed_fields.
do not start activation patching until no-intervention replay is validated.
```

### 2026-05-19 Adjacent-Layer Capture Planning Note

Validated by:

```text
doc update:
  - audit_windowed is tracked as the adjacent-layer capture tier for transcoders.
  - no code profile was added.
  - no README profile list was changed, because audit_windowed is not implemented.
```

Captured decision:

```text
audit_sampled supports first same-layer skip-transcoder pilots.
audit_windowed is needed for stronger "what did layer L write and what did L+1 consume" questions.
audit_windowed should be whole-episode static adjacent-window capture, not selected-policy-call capture.
audit_windowed must not be implemented until audit_sampled has a 3-5 trace storage/runtime benchmark.
```

Still not validated:

```text
real audit_windowed trace size
whether adjacent windows are feasible for whole episodes
whether rotating windows are useful for dataset-scale coverage
```

Next action before implementation:

```text
finish audit_sampled benchmark.
estimate audit_windowed cost from captured tensor family sizes.
if whole-episode audit_windowed cost is too high, trim tensor roles rather than silently changing it to selected policy calls.
```

## Core Product Identity

VLA Lens should be an episode-grounded causal interpretability workbench for VLAs.

It should not become:

- a raw tensor dump
- a generic Neuronpedia clone for robot activations
- a collection of disconnected probe scripts
- an attention heatmap viewer that implies causality
- a feature-dictionary showcase with no intervention loop

The main loop should be:

```text
observe
  episode video, prompt, robot state, object geometry, action chunks

localize
  probes over site x layer x token group x denoising step x component

hypothesize
  attention routing, feature browser, transcoder features, object-grounded views

intervene
  activation patching, feature ablation, steering/injection

measure
  action-head delta, EEF delta, gripper delta, closed-loop rollout change

visualize
  replay the episode with overlays showing what changed
```

This loop is the product. Capture profiles, schema metadata, probes, SAEs, transcoders, and UI diagrams are support machinery for that loop.

## Repo Ground Truth: Already Implemented

Do not re-plan or re-implement these as if they do not exist.

### Canonical Profiles

The repo already has canonical PI0.5 capture profiles in `src/vla_lens/pi05/capture.py`:

```text
rollout
features
mechanistic_sampled
mechanistic_all
internals_sampled
audit_sampled
audit_full
custom
```

Backward aliases already exist:

```text
representation      -> features
mechanistic_light   -> mechanistic_sampled
mechanistic_heavy   -> mechanistic_all
full                -> audit_full
```

Current layer presets:

```text
LANDMARK_5_LAYERS = [0, 4, 8, 12, 17]
ALL_PI05_LAYERS   = [0, 1, 2, ..., 17]
```

Current `PROFILE_LAYERS` behavior:

```text
rollout                []
features               [0, 4, 8, 12, 17]
mechanistic_sampled    [0, 4, 8, 12, 17]
mechanistic_all        [0..17]
internals_sampled      [0, 4, 8, 12, 17]
audit_sampled          [0, 4, 8, 12, 17]
audit_full             [0..17]
custom                 [0, 4, 8, 12, 17]
```

Important correction from the conversation: earlier planning suggested expert sampled layers like `[0, 2, 5, 8, 11, 14, 17]`. The implemented direction now uses the same sampled layers for VLM and Expert. This is the cleaner choice for representing the one-to-one per-layer K/V relationship between the VLM prefix pass and action expert.

### Schema 0.3.0 and Site Metadata

The trace schema is already at `0.3.0` in `src/vla_lens/traces.py`.

`ModelSiteSpec` and model-site tables already support:

```text
capture_family
view_kind
capture_role
default_view
derived_from
derivation
```

These fields exist to separate exact raw capture from UI-facing semantic views.

### Runtime Collection for Past Key Values

The repo already avoids creating a fake standalone VLM cache tensor.

Current exact stored sites look like:

```text
pi05.vlm.layers.0.kv_cache.key
pi05.vlm.layers.0.kv_cache.value
...
pi05.vlm.layers.17.kv_cache.key
pi05.vlm.layers.17.kv_cache.value
```

The non-materialized runtime collection is:

```text
id:              pi05.vlm.past_key_values
label:           Layer-wise prefix K/V
kind:            runtime_collection
materialized:    false
aggregation:     none
members:         exact per-layer key/value sites
```

This is implemented in capture metadata and reconstructed by `/api/activation-sites`.

### `audit_sampled` v0 Code Contract

The code-level `audit_sampled` contract is now implemented and one real smoke trace has been captured. It is not yet benchmarked broadly enough for scale-up decisions.

Ground-truth behavior:

```text
profile name:
  audit_sampled

layer coverage:
  VLM:    [0, 4, 8, 12, 17]
  Expert: [0, 4, 8, 12, 17]

profile dimensions:
  families.internals = sampled_audit
  families.state_setup = none

required persisted families:
  normal representation/attention/cache/action-head bridge sites
  selected residual boundaries
  attention norm output
  q/k/v
  pre_mask_scores
  post_mask_logits
  attention_probs
  attn_output_pre_o_proj
  o_proj
  residual_post_attention
  residual_pre_mlp
  mlp_norm_output
  MLP gate/up/intermediate/down/output
  residual_post_mlp
  expert AdaRMS scale/shift/gate
  VLM per-layer K/V
  expert input embeddings
  action_head input/output

excluded from persisted audit_sampled v0:
  input attention masks
  causal masks
  position IDs
  RoPE cos/sin/metadata
  Expert per-layer K/V cache
```

Attention coordinate metadata is now attached to full-capture declarations:

```text
q:
  post-RoPE
  after attention norm and linear projection
  before head scaling

k:
  post-RoPE
  pre-repeat_kv
  before head scaling

v:
  not RoPE-rotated
  pre-repeat_kv

pre_mask_scores:
  q @ repeat_kv(k).T * scaling

post_mask_logits:
  pre_mask_scores + additive_attention_mask

attention_probs:
  softmax(post_mask_logits, dim=-1, dtype=float32), cast to query dtype before dropout

attn_output_pre_o_proj:
  attention_probs_after_dropout @ repeat_kv(v)
```

Important caveat:

```text
The code captures enough named tensors for this contract, and a real smoke trace validates materialization and schema loading.
It still has not verified numeric attention_probs reconstruction against q/k/logits/mask state.
```

### PI0.5 Capture Environment With ROCm Torch

PI0.5 capture should use a dedicated ROCm capture environment, not the normal repo `.venv`.

Ground-truth environment state from the May 19, 2026 smoke trace:

```text
torch:        2.11.0+rocm7.2
torchvision:  0.26.0+rocm7.2
lerobot:      0.4.4
transformers: 4.53.2 with OpenPI transformers_replace files copied into site-packages
peft:         0.19.1
hf-libero:    0.1.3
robosuite:    1.4.0
```

Important operational detail:

```text
Use after setup:
  scripts/pi05_capture_rocm.sh ...
  scripts/pi05_batch_capture_rocm.sh ...

Do not use for capture:
  uv run vla-pi05-capture ...
  uv run vla-pi05-batch-capture ...
```

Reason:

```text
uv run syncs from pyproject/uv.lock and can restore robosuite 1.5.2.
LIBERO capture currently needs robosuite 1.4.0 because LeRobot's LIBERO path imports robosuite.environments.manipulation.single_arm_env.SingleArmEnv.
LeRobot's package metadata wants torch<2.11 and torchvision<0.26, but this workstation intentionally uses ROCm torch 2.11.0+rocm7.2.
```

The near-term contract is documented in `AGENTS.md` and `docs/pi05-rocm-capture-env.md`.

Testing split:

```text
Normal repo tests:
  uv run pytest
  uv run ruff check scripts src tests
  cd frontend && npm run build

Capture integration smokes:
  scripts/check_pi05_rocm_env.sh
  scripts/pi05_capture_rocm.sh ...
  scripts/pi05_batch_capture_rocm.sh ...
```

Normal tests should cover profile planning, declarations, schema, server payloads, trace validation, frontend type/build behavior, and pure analysis code without requiring Torch/LeRobot/GPU. Real PI0.5 model execution belongs in the explicit capture smoke path.

This is acceptable for local capture smoke work, but it is not a permanent dependency strategy. Future work should formalize a capture extra, dedicated lock, or container that preserves ROCm torch and the OpenPI transformers replacement patch.

### Activation-Site Architecture Metadata

`/api/activation-sites` now exposes PI0.5 architecture metadata when PI0.5 sites exist.

Ground-truth behavior:

```text
architecture.nodes:
  PI0.5 input node
  captured VLM layer nodes
  expert input / x_t node
  captured Expert layer nodes
  action head/output nodes

architecture.edges:
  kind = per_layer_kv_conditioning
  source = pi05.vlm.layers.{i}
  target = pi05.expert.layers.{i}
  layer = i
  source_sites = exact key/value cache site names
  source_token_space = pi05.prefix
  query_token_space = pi05.action_suffix
  key_token_space = pi05.expert_context
  runtime_collection = pi05.vlm.past_key_values
  materialized = false
```

For sampled traces, expected edges are:

```text
VLM L0  -> Expert L0
VLM L4  -> Expert L4
VLM L8  -> Expert L8
VLM L12 -> Expert L12
VLM L17 -> Expert L17
```

Old or non-PI0.5 traces keep empty architecture metadata.

### Expert Context Token Space

The repo already models expert attention's key side as a composite context:

```text
pi05.expert_context = pi05.prefix + pi05.action_suffix
```

Expert attention binding:

```text
query_token_space_id = pi05.action_suffix
key_token_space_id   = pi05.expert_context
```

This matters because expert attention has shape like:

```text
policy_call x generation_step x head x query_token x key_token
6 x 10 x 8 x 50 x 1018
```

The 50 query tokens are action/suffix tokens. The 1018 key tokens are approximately:

```text
968 prefix tokens + 50 action tokens
```

Do not label the expert key axis as pure action suffix.

### K/V Axis Naming

K/V cache axes already use:

```text
policy_call
kv_head
cached_token
head_channel
```

This avoids confusing K/V heads with attention heads.

### Full Capture Machinery

The repo already has an `audit_full` profile and declarative true-full capture machinery in `src/vla_lens/pi05/full_capture.py`.

Current declaration scale from the May 18, 2026 inspection:

```text
internals_sampled selected raw declarations: 120
audit_full raw declarations:                879
```

`audit_full` includes broad raw/debug coverage:

```text
action_head:      2 declarations
attention:        288 declarations
cache:            72 declarations
embedding:        2 declarations
mask:             4 declarations
mlp:              180 declarations
normalization:    180 declarations
position:         7 declarations
residual:         144 declarations
```

Therefore the missing thing is not "make full capture exist." It exists.

## Current Gap: `internals_sampled` Is Not `audit_sampled`

Current `internals_sampled` is useful, but it means selected operation-level internals, not a sampled audit profile with circuit boundaries.

Current selected internal roles:

```text
q
k
v
attention_probs
o_proj
mlp_gate
mlp_up
mlp_intermediate
mlp_down
adarms_gate
kv_cache_key
kv_cache_value
```

This excludes several circuit-critical boundary and score sites:

```text
residual_pre_attention
attention_norm_output
pre_mask_scores
post_mask_logits
attn_output_pre_o_proj
residual_post_attention
residual_pre_mlp
mlp_norm_output
mlp_output
residual_post_mlp
adarms_scale
adarms_shift
attention_mask
causal_mask
position_ids
rope.cos
rope.sin
rope.metadata
```

Many of those exist in `audit_full`; they are just filtered out of `internals_sampled`.

The next profile should be a named middle tier:

```text
audit_sampled
```

This is the missing profile between:

```text
internals_sampled = selected ops at sampled layers
audit_full        = all raw/debug internals at all layers
```

## Proposed Capture Profile Stack

Keep the existing names. Add only where the distinction is real.

```text
rollout
features
mechanistic_sampled
mechanistic_all
internals_sampled
audit_sampled
audit_windowed
audit_full
custom
```

### `rollout`

Purpose: cheap episode/world capture.

Use for:

```text
dataset-scale statistics
rollout inspection
behavioral metadata
world-state and action summaries
```

Capture:

```text
prompt
benchmark / suite / task / seed / success
frames
robot state
EEF pose
gripper state
object poses and geometry
camera intrinsics/extrinsics
2D boxes / visibility where available
executed actions
action chunks
generation actions / velocities where available
policy-call ranges
token metadata where available
```

No heavy model arrays.

Do not add a separate `census` profile unless it has materially different semantics. `rollout` already fills that role.

### `features`

Purpose: representation-only feature capture.

Sampled layers:

```text
VLM:    [0, 4, 8, 12, 17]
Expert: [0, 4, 8, 12, 17]
```

Capture:

```text
pi05.vlm.prefix.image_hidden_tokens
pi05.vlm.layers.{layer}.prefix.hidden_tokens
pi05.expert.layers.{layer}.by_step.hidden_tokens
```

Do not store `hidden_mean` when `hidden_tokens` exists.

### `mechanistic_sampled`

Purpose: scalable model-inspector and probe profile.

This is the current "mech-light" path.

Sampled layers:

```text
VLM:    [0, 4, 8, 12, 17]
Expert: [0, 4, 8, 12, 17]
```

Capture:

```text
pi05.vlm.prefix.image_hidden_tokens

For selected VLM layers:
  pi05.vlm.layers.{layer}.prefix.hidden_tokens
  pi05.vlm.layers.{layer}.prefix.attention
  pi05.vlm.layers.{layer}.kv_cache.key
  pi05.vlm.layers.{layer}.kv_cache.value

For selected Expert layers:
  pi05.expert.layers.{layer}.by_step.hidden_tokens
  pi05.expert.layers.{layer}.by_step.attention

Global/action:
  pi05.expert.by_step.input_embeddings
  pi05.action_head.input
  pi05.action_head.output
```

Use for:

```text
probes
episode visualization
attention routing dashboards
token/denoising-step localization
SAE pilots
action-space attribution candidates
```

Do not treat it as sufficient for strong circuit-level claims.

### `mechanistic_all`

Purpose: all-layer semantic inspector profile.

Capture the same semantic families as `mechanistic_sampled`, but all layers:

```text
VLM:    [0..17]
Expert: [0..17]
```

This is the best single-trace profile for all-layer semantic inspection, but not full circuit audit.

### `internals_sampled`

Purpose: selected operation-level internals at sampled layers.

Keep this profile. Do not rename it.

It should continue to mean:

```text
mechanistic_sampled
+ selected op-level internals
```

Current selected roles:

```text
q
k
v
attention_probs
o_proj
mlp_gate
mlp_up
mlp_intermediate
mlp_down
adarms_gate
kv_cache_key
kv_cache_value
```

This profile is useful for targeted debugging and first-pass component inspection, but it intentionally omits many circuit-boundary states.

### `audit_sampled`

Purpose: circuit-useful sampled profile.

This is the main proposed addition.

Canonical profile name:

```text
audit_sampled
```

Implementation target:

```text
audit_sampled_v0 semantics under the canonical profile name `audit_sampled`
```

Do not create separate public profile names like `full_light` or `audit_sampled_v0` unless a later compatibility need appears. `audit_sampled_v0` is the contract version, not the user-facing profile name.

Definition:

```text
same sampled layers as mechanistic_sampled
include read/write boundaries around attention and MLP blocks
include attention scores/logits and post-mask logits
include AdaRMS scale/shift/gate for expert
include VLM per-layer K/V and action-head I/O
avoid all-layer explosion
```

Sampled layers:

```text
VLM:    [0, 4, 8, 12, 17]
Expert: [0, 4, 8, 12, 17]
```

#### Coordinate-System Requirements

This is the part most likely to drift. `audit_sampled` is not acceptable if it records tensors named `q`, `k`, `v`, or `logits` without specifying what coordinate system they live in.

Every `audit_sampled` attention-family declaration must document:

```text
q:
  post attention-norm input projection output
  specify pre-RoPE or post-RoPE
  specify pre-head-scaling or post-head-scaling
  specify whether grouped-query / grouped-key-value expansion has happened

k:
  post attention-norm input projection output
  specify pre-RoPE or post-RoPE
  specify pre-head-scaling or post-head-scaling
  specify whether grouped-query / grouped-key-value expansion has happened

v:
  post attention-norm value projection output
  specify whether grouped-value expansion has happened
  no RoPE should be implied unless the actual backend applies it, which would be unusual

pre_mask_scores:
  q @ k.T / sqrt(d_head)
  computed from the q/k coordinate system actually used by attention
  before causal / prefix / padding mask
  before softmax

post_mask_logits:
  pre_mask_scores + additive attention mask
  before softmax

attention_probs:
  softmax(post_mask_logits)
  after masking
  before dropout, if dropout exists

attn_output_pre_o_proj:
  attention_probs @ v
  after head-wise value mixing
  before output projection

o_proj:
  attention output projection result
  after output projection
  before residual addition unless backend naming proves otherwise
```

Preferred `audit_sampled_v0` attention contract:

```text
store post-RoPE q/k if RoPE is used
store v in the coordinate system used by attention
store both pre_mask_scores and post_mask_logits for the first benchmark
store attention_probs
store attn_output_pre_o_proj
store o_proj
```

If storage/runtime is too high after benchmarking:

```text
keep post_mask_logits and attention_probs
document how pre_mask_scores can be reconstructed
only drop pre_mask_scores if the captured mask/RoPE metadata is enough to verify the relation
```

Do not store pre-RoPE q/k in `audit_sampled` unless enough RoPE metadata is also captured to reconstruct post-RoPE q/k and verify attention.

#### `audit_sampled_v0` Required Roles

```text
residual_pre_attention
attention_norm_output
q
k
v
pre_mask_scores
post_mask_logits
attention_probs
attn_output_pre_o_proj
o_proj
residual_post_attention

residual_pre_mlp
mlp_norm_output
mlp_gate
mlp_up
mlp_intermediate
mlp_down
mlp_output
residual_post_mlp

expert adarms_gate
expert adarms_scale
expert adarms_shift

VLM kv_cache.key
VLM kv_cache.value

expert.by_step.input_embeddings
action_head.input
action_head.output
```

#### `audit_sampled_v1` Optional Roles

```text
attention_mask
causal_mask
position_ids
rope.cos
rope.sin
rope.metadata
extra normalization diagnostics
```

Default rule for optional setup tensors:

```text
Do not add optional setup tensors to audit_sampled unless they are needed to:
  1. verify attention_probs,
  2. replay one policy call,
  3. explain a failed audit_sampled smoke test.
```

These tensors remain important for `audit_full`, exact replay debugging, and schema audits. They are not automatically part of the sampled profile.

Acceptance criteria:

```text
1. `audit_sampled` must not replace or rename `internals_sampled`.
2. It must use sampled layers first: [0, 4, 8, 12, 17].
3. It must include residual/norm/component output boundaries.
4. It must include attention scores/logits before/around softmax.
5. It must preserve exact tensor provenance.
6. It must not create a fake standalone VLM cache tensor.
7. It must report profile dimensions distinctly from `internals_sampled`.
8. Tests must assert exact role inclusion and exclusion.
9. Size must be benchmarked before scaling.
10. Q/K/V and attention-logit metadata must define pre/post RoPE, mask, scale, and expansion semantics.
11. A smoke test must recompute or verify attention_probs from captured q/k/logit/mask state for at least one VLM layer and one Expert layer.
```

### `audit_windowed`

Purpose: circuit-useful adjacent layer windows.

Do not implement before `audit_sampled` is benchmarked unless there is a specific need.

This profile exists for transcoder and downstream-use questions, not for ordinary inspection.
It should reuse the `audit_sampled` circuit-boundary tensor families and change only layer
coverage/windowing, unless the benchmark proves that role trimming is required.

Capture scope decision:

```text
whole episode
static adjacent windows
not selected-policy-call windows
```

Proposed windows:

```text
VLM:    [0, 1], [4, 5], [8, 9], [12, 13], [16, 17]
Expert: [0, 1], [4, 5], [8, 9], [12, 13], [16, 17]
```

Why windows matter:

```text
An isolated layer lets us inspect what exists at layer L.
A layer window lets us inspect what layer L writes and what layer L+1 consumes.
```

This is more circuit-useful than isolated layers, but more expensive and more complex to present.

Transcoder relevance:

```text
audit_sampled can support same-layer pilots:
  expert L8 residual_pre_mlp -> expert L8 mlp_output
  expert L8 residual_pre_attention -> expert L8 attn_output/o_proj
  action_head input -> action_head output

audit_windowed supports adjacent-layer questions:
  did L8 write a feature that L9 consumed?
  was the feature preserved, transformed, routed, or erased?
  does the write direction line up with the next layer's read/use?
```

Acceptance criteria before implementation is crossed off:

```text
1. audit_sampled 3-5 trace benchmark exists.
2. expected audit_windowed size/runtime is estimated from real family-size data.
3. profile name and layer windows are tested distinctly from audit_sampled.
4. internals_sampled and audit_full remain unchanged.
5. per-layer VLM K/V -> Expert attention architecture edges still pair only equal layer indices.
6. audit_windowed captures whole episodes.
7. at least one real audit_windowed smoke trace validates successfully.
8. roadmap is updated with measured size/runtime and stale assumptions removed.
```

Potential future variant:

```text
Always capture anchor layers: [0, 8, 17]
Rotate one adjacent window per episode group:
  group A: [1, 2]
  group B: [3, 4]
  group C: [5, 6]
  ...
```

This gives dataset-wide coverage without making every episode circuit-complete.

### `audit_full`

Purpose: exhaustive raw/debug profile.

Keep this as all-layer full raw capture.

Use for:

```text
schema audit
raw debugging
selected failure episodes
short high-value policy-call windows
validating audit_sampled omissions
```

Do not present it as the recommended normal profile.

Do not run it for every full episode unless the storage and runtime cost have been measured and accepted.

## Storage Ground Truth and Benchmarking Protocol

Conversation-time snapshot from the external-drive `mechanistic_sampled` run:

```text
completed traces: 67
trace total:      17.28 GiB
average size:     264.1 MiB / episode
min size:         103.3 MiB
max size:         579.6 MiB
```

Earlier estimates around 328 MB/episode came from a smaller early sample and should not be treated as canonical.

Single-trace `audit_sampled` smoke from May 19, 2026:

```text
trace:
  /media/j/New Volume/vla-lens/pi05-audit-sampled-smoke/pi05_audit_sampled_libero_object_task0_seed1002.vlatrace

task:
  libero_object task 0 seed 1002

episode:
  steps: 138
  policy calls: 3
  success: true

storage:
  du size:          2.4G
  file bytes:       2.18 GiB
  model arrays:     2.18 GiB
  media:            4.44 MiB
  tables:           0.22 MiB
  action/context/episode arrays outside model: <1 MiB combined

runtime:
  wall clock:        1:21.41
  max CPU RSS:       16,717,972 KB

model-site count:
  total:             244
  attention:         90
  normalization:     50
  mlp:               50
  residual:          30
  representation:    12
  cache:             10
  action_head:       2

model bytes by family:
  mlp:               1.17 GiB
  attention:         706.05 MiB
  residual:          161.39 MiB
  normalization:     95.78 MiB
  representation:    61.39 MiB
  cache:             9.54 MiB
  action_head:       2.26 MiB

model bytes by stack:
  VLM:               1.68 GiB
  Expert:            512.43 MiB
  Action head:       2.26 MiB
```

Interpretation:

```text
audit_sampled v0 is not a modest "mech-light plus a few internals" profile.
The selected VLM MLP gate/up/intermediate tensors are currently the dominant storage cost.
The first smoke trace is roughly 8x the earlier mechanistic_sampled average from the 67-trace external-drive snapshot.
Do not scale audit_sampled until 3-5 traces confirm whether this episode is representative and whether selected roles should be trimmed or windowed.
```

Important storage implications:

```text
Full attention is already a major fraction of mechanistic_sampled storage.
Attention logits/scores are attention-sized.
Residual and norm boundaries are hidden-state-sized.
Q/K/V and MLP internals can easily double or triple sampled profile cost.
Episode length and number of policy calls strongly affect size.
```

Do not trust guessed `audit_sampled` sizes. Benchmark before scale-up.

Benchmark protocol for `audit_sampled`:

```text
Run 3-5 traces across varied episode lengths.

Measure:
  total trace size
  size per policy call
  size by tensor family
  VLM vs Expert split
  attention probs vs attention scores/logits
  residual/norm boundary contribution
  MLP contribution
  cache contribution
  media/table overhead
  compression ratio by family if available
  wall-clock rollout time without capture
  wall-clock rollout time with mechanistic_sampled
  wall-clock rollout time with audit_sampled
  slowdown multiplier vs no capture
  slowdown multiplier vs mechanistic_sampled
  peak GPU memory
  peak CPU RAM
  trace write time

Report:
  mean
  median
  min
  max
  per-policy-call slope
  rough projection to 100 / 1,000 episodes
```

Size is not the only failure mode. If `audit_sampled` requires disabling fused attention, adding slow hooks, or moving large tensors through CPU synchronously, runtime may be the limiting factor even if disk size is acceptable.

Decision gates:

```text
If audit_sampled <= roughly 2x mechanistic_sampled:
  suitable for selected medium-scale subsets.

If audit_sampled is 3x-5x mechanistic_sampled:
  use for targeted audit subsets only.

If audit_sampled approaches audit_full size:
  narrow included roles or capture only short policy-call windows.
```

## PI0.5 Architecture Semantics for UI

This is the other main gap.

The UI should not show a fake `VLM Cache` block. It should show per-layer K/V conditioning.

Correct relationship:

```text
VLM layer i emits prefix K/V.
Expert/action layer i receives that layer's prefix K/V as past_key_values.
Expert action-token queries attend over:
  pi05.expert_context = pi05.prefix + pi05.action_suffix
```

The action denoiser can be colloquially described as using all layer K/V caches, but architecturally this is a per-layer list consumed by corresponding expert layers, not one monolithic cache tensor and not one all-to-all attention band.

Bad diagrams:

```text
VLM stack -> giant VLM Cache block -> Expert stack

VLM L0..L17 ========> Expert L0..L17
```

The first implies a materialized standalone cache. The second can imply all-to-all cross-layer attention.

Better diagrams:

```text
VLM L0  -K/V-> Expert L0
VLM L4  -K/V-> Expert L4
VLM L8  -K/V-> Expert L8
VLM L12 -K/V-> Expert L12
VLM L17 -K/V-> Expert L17
```

or a compact aligned bus:

```text
VLM layers:     L0     L4     L8     L12    L17
                |      |      |      |      |
                K/V    K/V    K/V    K/V    K/V
                |      |      |      |      |
Expert layers:  L0     L4     L8     L12    L17
```

For dense/all-layer profiles, use repeated faint connectors or a labeled per-layer K/V bus with tick marks.

Recommended architecture metadata:

```json
{
  "architecture": {
    "nodes": [
      {
        "id": "pi05.vlm.layers.0",
        "label": "VLM L0",
        "kind": "vlm_layer",
        "stage": "prefix",
        "layer": 0
      },
      {
        "id": "pi05.expert.layers.0",
        "label": "Expert L0",
        "kind": "expert_layer",
        "stage": "action_denoiser",
        "layer": 0
      }
    ],
    "edges": [
      {
        "id": "pi05.vlm.layers.0.kv_to_expert.layers.0",
        "kind": "per_layer_kv_conditioning",
        "source": "pi05.vlm.layers.0",
        "target": "pi05.expert.layers.0",
        "source_sites": [
          "pi05.vlm.layers.0.kv_cache.key",
          "pi05.vlm.layers.0.kv_cache.value"
        ],
        "target_site_family": "pi05.expert.layers.0.by_step.attention",
        "source_token_space": "pi05.prefix",
        "query_token_space": "pi05.action_suffix",
        "key_token_space": "pi05.expert_context",
        "runtime_collection": "pi05.vlm.past_key_values",
        "materialized": false
      }
    ]
  }
}
```

Current API gap:

```text
/api/activation-sites returns runtime_collections but currently returns architecture: {}
```

Hard ordering rule:

```text
Do not modify the PI0.5 visual diagram until /api/activation-sites returns architecture.nodes and architecture.edges.
```

Reason:

```text
Hard-coding another diagram from name filters repeats the original ontology bug.
The UI should consume architecture metadata, not infer model structure from tensor names.
```

Implementation target:

```text
Populate architecture.nodes and architecture.edges from available model sites.
Old traces should continue to return empty metadata gracefully.
```

Architecture edge acceptance tests:

```text
Given captured sampled layers [0, 4, 8, 12, 17],
architecture.edges contains exactly five per_layer_kv_conditioning edges:
  pi05.vlm.layers.0  -> pi05.expert.layers.0
  pi05.vlm.layers.4  -> pi05.expert.layers.4
  pi05.vlm.layers.8  -> pi05.expert.layers.8
  pi05.vlm.layers.12 -> pi05.expert.layers.12
  pi05.vlm.layers.17 -> pi05.expert.layers.17

For all-layer traces:
  edge count == count(intersection(captured_vlm_layers, captured_expert_layers))
  every edge pairs the same numeric layer index on VLM and Expert

For partial traces:
  edges only exist for layer indices where both VLM K/V and matching Expert attention layer metadata are present or inferable.
```

UI acceptance criteria:

```text
1. No standalone fake VLM cache block.
2. No visual line that implies all-to-all cross-layer attention.
3. Show layer-aligned K/V conditioning.
4. Expert attention ruler must distinguish prefix/image/text keys from action-suffix keys.
5. Active overlay label must say what is shown on the frame.
6. Raw tensor names remain available but are not the main navigation.
```

## Current Dataset Context

Conversation-time external-drive dataset:

```text
root:
  /media/j/New Volume/vla-lens/pi05-diverse-100-mech-light

profile:
  mechanistic_sampled

planned traces:
  100

task spread:
  33 tasks across LIBERO suites

seeds:
  mostly 1000, 2000, 3000 per task
  one extra seed 4000
```

This dataset is useful for:

```text
schema validation
frontend inspection
probe feasibility
attention routing summaries
storage estimates
first SAE/feature browser pilots
finding candidate causal sites
```

This dataset is not enough for:

```text
strong claims across many tasks and probe families
well-powered success/failure probes if failures are rare
clean causal patching without counterfactual pairs
object-identity claims if object labels are confounded with task ID
row-level statistical claims treating policy calls/tokens as independent
```

The independent unit is the episode, not the activation row, policy call, token, or denoising step.

## Probe Grid: First Practical Tool

Build this before sparse dictionary work.

The current `mechanistic_sampled` data already supports a useful probe grid.

Feature axes:

```text
site
layer
stack: VLM / Expert / Action head
token group
policy call
generation step
action horizon
reduction strategy
```

Candidate targets:

```text
benchmark / suite
task ID
seed / layout
target object
distractor object
object pose
EEF-to-target vector
EEF-to-receptacle vector
gripper state
task phase
next action
action chunk direction
success/failure, only if balanced
```

Probe Grid v0 should be intentionally smaller than the full target list.

Do not build a generic probe universe first. Build the smallest grid that can prove the machinery works and expose leakage.

Probe Grid v0 targets:

```text
classification:
  benchmark / suite
  task_id
  seed / layout leakage
  gripper open / closed

regression:
  next action chunk, flattened or per-action-dim
  EEF delta / action direction if already available in the trace
```

Probe Grid v0 localization axes:

```text
site
layer
stack: VLM / Expert / action_head
generation_step for Expert/action sites
token reduction: mean over token group only
```

Explicitly defer until labels are reliable:

```text
target object
distractor object
object pose
EEF-to-target vector
EEF-to-receptacle vector
phase
success/failure
held-out object splits
```

Reason:

```text
The roadmap says entity labels, token-to-world mappings, phase labels, and failure labels are high-value additions.
They are not guaranteed ground truth yet.
Probe Grid v0 should not depend on labels that may be noisy or confounded.
```

Split discipline:

```text
Prefer episode-level splits at minimum.
Prefer held-out seed/layout for leakage checks.
Use held-out task for task-general claims.
Use held-out benchmark/suite for stronger transfer claims.
Use held-out object only when labels are balanced across tasks/layouts.
Never split rows randomly if rows share an episode.
Do not treat token, policy-call, or denoising-step rows as independent samples.
```

Control probes:

```text
task ID baseline
benchmark/suite baseline
seed/layout leakage probe
shuffled labels within task
shuffled labels within episode where meaningful
metadata-only baseline
object-label confound checks
phase confound checks
```

Output views:

```text
concept x layer heatmap
concept x token group heatmap
concept x denoising step heatmap
layer x denoising step curves
probe-vs-metadata baseline comparison
confidence intervals clustered by episode
```

Interpretation discipline:

```text
A probe localizes available information.
A probe does not prove the policy uses that information.
Probe results should nominate sites for patching, steering, or ablation.
```

## Object-Grounded Attention Dashboard

This is the most immediately VLA Lens-native visualization.

Do not present raw attention as explanation. Present it as attention routing.

For each selected attention site:

```text
stack
layer
head
policy call
generation step
query token group
key token segment
object box / patch overlap
```

Compute attention mass from action-suffix queries to:

```text
target object image tokens
distractor object image tokens
robot / gripper / arm tokens
language tokens
background / nuisance tokens
prior action-suffix tokens
```

Views:

```text
attention mass over episode time
attention mass over denoising step
head ranking by target-object mass
target vs distractor mass
robot vs object vs background mass
phase-aligned attention shifts
```

Required labels:

```text
attention routing
attention mass
not causal importance unless validated by intervention
```

Future extension:

```text
interventional visual masking / ISS-style object-region perturbations
```

## Counterfactual Pair Capture

This is more important than adding a large number of random traces.

Implementation status:

```text
metadata-only paired capture contract is now an allowed near-term increment.
This does not mean activation patching is implemented.
It means traces can be captured, named, loaded, and grouped as clean/corrupt units.
```

Design rule:

```text
paired counterfactual capture is not a tensor profile.
It is a capture_design layer above the tensor profile.

Valid examples:
  capture_design = paired_counterfactual
  capture_profile = mechanistic_sampled

  capture_design = paired_counterfactual
  capture_profile = audit_sampled

Invalid framing:
  capture_profile = paired
  capture_profile = clean_corrupt
```

Why:

```text
The pair relationship is an experimental design/provenance fact.
The profile still controls which tensors are stored.
```

Required metadata:

```text
counterfactual_group_id
counterfactual_role: clean / corrupt / intervention / control
counterfactual_type
trace_variant
paired_trace_id
pair_index
matched_fields
changed_fields
```

Useful pair types:

```text
same scene, prompt target swapped
same prompt, target object pose moved
same scene, distractor added/removed
same task, camera/viewpoint perturbed
same task, language paraphrased
successful run vs induced failure
target visible vs target occluded
wrong-object behavior vs correct-object behavior
```

First counterfactual family:

```text
same initial scene
same seed
same task template
same initial object poses
same camera configuration
prompt target noun swapped between two visible objects
no object pose changes
no camera changes
no distractor insertion/removal
```

Why this first:

```text
It isolates language-to-object binding better than pose perturbation or induced failure.
It lines up with the architecture question:
  VLM prefix K/V -> Expert action attention -> action output
```

Required metadata for the first family:

```text
counterfactual_group_id
counterfactual_role = clean | corrupt
counterfactual_type = prompt_target_swap
changed_fields = ["prompt.target_object"]
matched_fields = [
  "benchmark",
  "task_id",
  "seed",
  "initial_object_poses",
  "camera_config"
]
target_object_id_clean
target_object_id_corrupt
```

Hard validity rule:

```text
No activation patching experiment is valid unless both traces share a counterfactual_group_id and changed_fields is non-empty and specific.
```

MVP acceptance for metadata-only paired capture:

```text
batch episode_plan.csv can contain two rows with the same benchmark/task/seed/profile.
trace_variant or counterfactual_role makes the trace IDs distinct.
the capture CLI stores counterfactual metadata in manifest.metadata.
/api/dataset exposes counterfactual_pairs.
the UI can navigate between paired members without guessing from trace names.
old traces and old episode plans still load.
```

Why:

```text
Activation patching needs clean/corrupt pairs.
Without explicit pairing, restored behavior is ambiguous.
It may reflect object identity, layout, prompt wording, phase, or memorized trajectory.
```

## Activation Patching Workbench

Requires deterministic replay or at least exact policy-call replay.

Do not build patching before replay verification exists.

Minimum gate:

```text
Given trace_id + policy_call_id + profile,
the replay API must reproduce the captured action_head.output under documented tolerances.
Only then can patching or steering be enabled for that policy call.
```

Start coarse, then refine.

Initial patch sites for `audit_sampled`:

```text
residual_pre_attention
attn_output_pre_o_proj
o_proj
residual_post_attention
residual_pre_mlp
mlp_output
residual_post_mlp
action_head.input
```

Patch granularities:

```text
whole site
layer
token group
action token
generation step
head
MLP output
attention output
```

Metrics:

```text
action MSE to clean
gripper open/close agreement
EEF delta direction agreement
horizon endpoint error
target-object movement
closed-loop success if rollout replay is available
```

First experiments:

```text
patch clean target-object state into corrupted prompt run
patch successful pre-grasp state into failed pre-grasp run
patch target-visible run into target-occluded run
patch action_head.input at selected denoising steps
patch VLM K/V value cache for target object token regions
```

Do not start with every head/component. Use the probe grid and attention dashboard to nominate candidate sites.

## Steering and Injection

Do this before waiting for perfect SAE/transcoder tooling.

Use linear probe directions first.

Targets:

```text
close gripper
open gripper
EEF delta x/y/z
action chunk direction
move toward / away from target only if EEF-to-target labels exist
```

Defer from steering v0:

```text
target object identity
pre-grasp phase
place phase
slow down / speed up
```

Reason:

```text
Gripper and action dimensions have immediate action-head metrics.
Target identity and phase steering are easier to confound with task ID, prompt token identity, object position, or phase labels.
```

Intervention:

```text
activation += alpha * direction
```

Sweep:

```text
alpha in negative and positive directions
site
layer
token group
generation step
policy call
```

Measure:

```text
action-head delta
EEF delta
gripper dimension delta
trajectory endpoint delta
behavioral degradation outside intended slice
closed-loop rollout outcome when available
```

Steering is a fast causal sanity check for whether a localized direction actually touches behavior.

## Sparse Dictionary Methods: Role and Ordering

Do not center the roadmap on SAEs.

Correct division of labor:

```text
SAEs             feature discovery and browsing
Matryoshka SAEs coarse-to-fine behavior feature browsing
Skip transcoders component write/read explanations
Crosscoders      model/condition diffing
CLTs             later-stage attribution graph replacement models
Patching/steer   causal validation
```

Hard acceptance gate:

```text
No SAE/transcoder training PR should be accepted until:
  1. Probe Grid v0 exists,
  2. audit_sampled benchmark exists,
  3. policy-call replay verification exists,
  4. at least one activation patching experiment runs end-to-end.
```

Reason:

```text
Sparse dictionaries are easy to make look impressive.
Without replay and patching, they remain feature browsing rather than causal evidence.
```

### Vanilla SAEs

Useful for:

```text
top-activating episode browser
feature labeling
phase/object/action feature discovery
visual clustering
candidate steering directions
```

First sites:

```text
late VLM residual / hidden tokens
expert action-token hidden states
action_head.input
```

Limitations:

```text
do not say what computed the feature
do not say what used the feature
do not prove causal necessity
can learn convenient reconstruction bases rather than mechanisms
can split or absorb features depending on dictionary size
```

### Matryoshka SAEs

More UI-relevant than flat vanilla SAEs because VLA Lens wants coarse-to-fine behavior browsing.

Example hierarchy:

```text
coarse: grasp phase
  finer: pre-grasp approach
  finer: gripper closure
  finer: object lift
  finer: failed grasp recovery
```

Example hierarchy:

```text
coarse: target object grounding
  finer: target visible in main camera
  finer: target visible in wrist camera
  finer: distractor competition
  finer: occluded target / weak grounding
```

First sites:

```text
expert layer 8 or 12 action-token stream
action_head.input
late VLM layer hidden stream
```

### Skip Transcoders

Likely the best first dictionary method after probes and patching.

A transcoder approximates:

```text
component input -> sparse features -> component output
```

Useful VLA targets:

```text
expert residual_pre_mlp -> expert mlp_output
expert residual_pre_attention -> attention output
action_head.input -> action_head.output
```

Why skip transcoders:

```text
Robotics has smooth continuous control geometry.
The skip path can carry boring linear-ish transforms.
Sparse features can focus on semantically interesting nonlinear writes.
```

Candidate features:

```text
gripper close write
target approach write
place-phase write
distractor suppression write
wrist-camera correction write
failure-recovery write
```

### Cross-Layer Transcoders

Attractive long-term direction, not first implementation.

Potential VLA graph:

```text
language token "mug"
-> VLM target-object feature
-> expert target-selected feature
-> EEF-to-target vector feature
-> pre-grasp phase feature
-> gripper-close / descend action feature
-> action-head dimensions
```

Requirements before serious CLT work:

```text
audit_sampled or audit_windowed
deterministic replay
counterfactual pairs
exact patching validation
clear action/rollout metrics
faithfulness checks
```

### Crosscoders

Use for diffing, not first-pass circuits.

Good comparisons:

```text
pi0 vs pi0.5
base vs fine-tuned
successful vs failed episodes
clean vs corrupted prompt
clean vs perturbed camera/viewpoint
LIBERO suite A vs suite B
before vs after steering
```

Feature categories to expose:

```text
shared
condition-specific
model-specific
shared but decoder direction changed
shared but activation frequency changed
```

Important caution:

```text
An exclusive feature is not automatically a meaningful mechanism.
In robotics, "shared but used differently" may be more important than "exclusive."
```

### Sparse Mixtures of Linear Transforms

Watch this, but do not implement first.

Possible relevance:

```text
continuous control may be better explained by sparse transforms than sparse vector features
example: rotate EEF-relative geometry into action-space coordinates
```

Defer until simpler probes, patching, steering, and skip transcoders reveal what is missing.

## Deterministic Replay and Hookability

Static trace capture is not enough for causal claims.

Needed for patching/steering:

```text
model hash
tokenizer/version
prompt template
camera ordering
image preprocessing
action normalization
random seeds
input tensors sufficient to replay policy call
policy-call-level replay entrypoint
activation replacement hooks
activation addition hooks
action-head comparison API
optional closed-loop rollout replay
```

Minimal Policy-Call Replay Contract:

```text
Given trace_id + policy_call_id + profile, the replay API must:
  1. reconstruct the exact model inputs for that policy call,
  2. run a no-intervention forward pass,
  3. compare action_head.output to the captured trace,
  4. report max_abs_error, mean_abs_error, and allclose under configured tolerances,
  5. only allow patching/steering if replay verification passes.
```

Suggested tolerances:

```text
float32-ish deterministic replay:
  max_abs_error <= 1e-5

mixed precision / nondeterministic kernels:
  max_abs_error <= 1e-3
  plus cosine or action-direction agreement

If neither threshold is met:
  patching results are invalid unless the tolerance gap is explained and accepted.
```

Design principle:

```text
Trace files preserve what happened.
Replay infrastructure tests what would happen if an activation changed.
```

Do not conflate the two.

## Entity, Token, and World Metadata

High-value schema additions before a serious causal dataset:

```text
target object ID
target object aliases from prompt
receptacle / goal object ID
distractor object IDs
object family / color / material / shape when available
task-critical vs support vs nuisance role
object visibility per camera
object bbox and, eventually, segmentation mask
```

Token-to-world mappings:

```text
image token -> camera
image token -> patch coordinates
image token -> pixel region
image token -> object bbox/mask overlap
language token -> decoded text span
language token -> semantic role when parseable
action token -> action horizon / chunk position
action dimension -> semantic action dimension
generation step -> denoising step / flow step
```

Why:

```text
These mappings turn tensors into episode-grounded explanations.
They enable object-grounded attention, object probes, and interpretable patching sites.
```

## Failure and Phase Labels

Even heuristic labels are useful.

Candidate phase labels:

```text
search
approach
pre-grasp
grasp contact
lift
transport
place
retract
recovery
```

Candidate failure labels:

```text
wrong object
missed grasp
dropped object
wrong receptacle
collision
timeout
ignored language
visual confusion
stuck / no progress
bad gripper timing
```

Use cases:

```text
phase probes
failure prediction
failure-localized audit captures
patching clean/corrupt pairs
attention routing slices
SAE feature labels
```

## UI Roadmap

Next sprint UI scope, if UI is touched at all:

```text
consume architecture.edges if present
render per-layer K/V connectors from metadata
add or preserve expert attention ruler labels
add or preserve active model-overlay label
```

Out of scope for the next sprint:

```text
new layout system
new feature browser
new dashboard shell
new visual design pass
hard-coded PI0.5 diagram from tensor-name filters
```

Do not begin visual diagram work before architecture metadata exists.

### Inspector Principles

The UI should expose semantic views over exact tensor sites.

Do not expose capture taxonomy directly as the product ontology.

Rules:

```text
0 options: hide the control
1 option: render directly
2-4 meaningful options: chips / segmented control
5+ options: grouped menu or search
raw tensor list: raw details drawer, not primary UI
```

Primary views:

```text
Features
Attention
Cache
Action
Internals
Raw tensors
```

`Raw tensors` should always be available for provenance, but should not dominate normal exploration.

### Pipeline Diagram

Replace obstructive K/V block with aligned per-layer K/V conditioning.

Preferred normal view:

```text
VLM layers above
Expert layers below
small K/V connectors aligned by layer
no giant cache rectangle
no all-to-all solid band
```

For sampled profile:

```text
show paired sampled layers:
  L0, L4, L8, L12, L17

show dots or faded gaps between sampled layers.
```

For all-layer profile:

```text
show compact dense row or collapsible stack
allow zoom/fullsize if needed
```

Avoid:

```text
large explanatory text inside the diagram
decorative blocks that imply materialized tensors
lines that imply all-to-all layer attention
```

### Active Overlay Label

The camera/frame overlay should always say what is being rendered:

```text
Model overlay: VLM L8 features c674
Model overlay: Expert L12 attention head 3 action token 12
Model overlay: VLM L4 K/V value norm
```

This prevents users from guessing what the visual overlay means.

### Expert Attention Ruler

Expert attention should expose:

```text
query: action token
keys: prefix + action suffix
```

Visual key ruler:

```text
| image patches | prompt/text | action suffix |
```

If text/image segmentation is not precise yet, label conservatively:

```text
prefix/image+text
action tokens
```

### Capture Coverage Badges

Each architecture node should show coverage:

```text
Features
Attention
Cache
Internals
Raw
Missing
```

Distinguish:

```text
the model has this layer
the trace captured this layer
the UI has a semantic renderer for this capture
```

## Implementation Phases

This replaces any broad "immediate implementation order." Treat Phase 1 as the next sprint boundary. Later phases are sequencing context.

### Phase 0: Contract Hardening

Answer or encode these before writing capture code:

```text
audit_sampled Q/K/V coordinate definitions
pre_mask_scores and post_mask_logits semantics
attention_probs verification strategy
minimal policy-call replay tolerance
first counterfactual family
benchmark metrics and artifact path
```

Phase 0 output can be a short doc patch or implementation notes in the PR description, but the definitions must exist before hooks are added.

Phase 0 validation:

```text
review the proposed definitions against current full_capture declarations
confirm every required role maps to an existing hook point or a named new hook point
confirm attention_probs verification is mathematically specified
update this roadmap with any definitions that become ground truth
delete any role that cannot be captured or verified in v0
```

### Phase 1: Capture/Schema Sprint

Only allowed scope:

```text
add canonical audit_sampled
keep internals_sampled unchanged
keep audit_full unchanged
add architecture.nodes and architecture.edges to activation-site payloads
add per_layer_kv_conditioning edges
preserve pi05.vlm.past_key_values runtime_collection behavior
add server/frontend types needed to carry metadata
optionally update the existing PI0.5 diagram to consume architecture.edges
run 3-5 audit_sampled smoke traces
write benchmark artifact
```

Explicitly disallowed in Phase 1:

```text
Probe Grid v0
object-grounded attention dashboard
counterfactual capture
policy-call replay implementation
activation patching
steering
SAEs
transcoders
crosscoders
CLT / attribution graph work
new UI shell
```

Phase 1 success condition:

```text
audit_sampled is a tested, benchmarked, distinct profile
architecture metadata truthfully represents per-layer VLM K/V -> Expert attention conditioning
```

Phase 1 validation:

```text
unit tests:
  profile canonicalization
  audit_sampled inclusion/exclusion
  internals_sampled unchanged
  audit_full unchanged
  architecture edge pairing
  expert attention token spaces
  no fake standalone cache tensor

smoke tests:
  generate 3-5 audit_sampled traces
  load them through TraceDataset
  load them through /api/activation-sites
  open the existing UI if diagram metadata is consumed

benchmark:
  write a benchmark artifact with size, runtime, memory, and trace-write cost

roadmap update:
  move implemented audit_sampled and architecture-edge facts to Repo Ground Truth
  delete or rewrite any wrong role/size/runtime assumptions
  reduce Phase 2 scope if benchmark results show a constraint
```

### Phase 2: Current-Data Analysis

Use the existing `mechanistic_sampled` dataset first.

Allowed:

```text
Probe Grid v0
attention routing summaries
basic object/token segment aggregation
no causal claims
```

Disallowed:

```text
strong object/phase/success claims without labels and splits
row-level independence claims
dictionary training as a substitute for causal tests
```

Phase 2 validation:

```text
unit/integration tests:
  probe dataset construction
  episode-level split behavior
  metadata baseline behavior
  attention segment aggregation

analysis smoke:
  run Probe Grid v0 on a small subset and one full current dataset slice
  produce one attention-routing summary artifact
  verify no row-level split leakage

roadmap update:
  move validated Probe Grid v0 capabilities into Repo Ground Truth
  delete targets blocked by missing labels
  promote only the next label/schema gap that analysis actually needs
```

### Phase 3: Causal Infrastructure

Implement:

```text
policy-call replay verification
counterfactual pair capture, starting with prompt target swap
coarse activation patching
probe-direction steering for motor/action variables
```

Phase 3 success condition:

```text
one clean/corrupt pair can be replayed
one coarse patch can be applied
the action-head delta is measured against a verified no-intervention replay
```

Phase 3 validation:

```text
replay tests:
  no-intervention replay reproduces action_head.output within tolerance
  failed replay blocks patching/steering

counterfactual tests:
  prompt-target-swap pairs share counterfactual_group_id
  changed_fields and matched_fields are specific and non-empty

patching smoke:
  one clean/corrupt pair
  one coarse patch
  action-head delta reported against verified replay

roadmap update:
  record the achieved replay tolerance
  move the first valid counterfactual family into Repo Ground Truth
  delete patching metrics that proved unhelpful
```

### Phase 4: Dictionary Methods

Only after replay and patching are real:

```text
Matryoshka SAE pilot
skip-transcoder pilot
feature browser linked to episode/video/action context
```

Still defer:

```text
crosscoders
CLT attribution graphs
large SAE suites across many layers
```

Phase 4 validation:

```text
training tests:
  artifact metadata records trace_id/model_hash/site_id/layer/token_space/source hash
  dictionary artifact loads independently of .vlatrace

interpretability smoke:
  one Matryoshka SAE or skip-transcoder feature page links back to episodes
  at least one feature/direction is tested with patching or steering

roadmap update:
  keep only dictionary methods that pass the causal/episode-grounded loop
  delete methods that only produce attractive dashboards without intervention value
```

### Phase 5: Diffing and Attribution Graphs

Later, after the causal stack matures:

```text
pi0 vs pi0.5 crosscoders
base vs fine-tuned crosscoders
clean vs corrupted condition diffing
CLT-style replacement models
attribution graph visualization and perturbation validation
```

Phase 5 validation:

```text
faithfulness tests:
  replacement model error is measured
  perturbation validation is run on top graph claims
  graph claims link to action or rollout metrics

roadmap update:
  convert mature graph/diffing methods into normal workflows
  delete speculative graph methods that fail faithfulness or intervention tests
```

## Tests and Acceptance Checks for Future Coding

### Capture Profile Tests

Add/update tests that assert:

```text
canonical_profile("mechanistic_light") == "mechanistic_sampled"
canonical_profile("full") == "audit_full"
`audit_sampled` is canonical
`audit_sampled` uses sampled layers
`internals_sampled` keeps selected-op semantics
`audit_sampled` includes residual/norm/logit/boundary roles
`audit_sampled` excludes optional state setup unless intentionally included
`audit_sampled` declares Q/K/V coordinate-system metadata
`audit_sampled` supports attention_probs verification in a smoke trace
`audit_full` still includes all raw/debug roles
```

### Model-Site Tests

Assert:

```text
hidden_mean is not emitted when hidden_tokens exists
attention_key_mass is not stored when full attention exists
K/V cache axes use kv_head
expert attention query_token_space_id is pi05.action_suffix
expert attention key_token_space_id is pi05.expert_context
runtime collection pi05.vlm.past_key_values is non-materialized
runtime collection members are exact per-layer key/value sites
```

### Architecture API Tests

Assert:

```text
/api/activation-sites returns sites
/api/activation-sites returns runtime_collections
/api/activation-sites returns architecture.nodes
/api/activation-sites returns architecture.edges
sampled traces expose exactly five per_layer_kv_conditioning edges for [0, 4, 8, 12, 17]
all-layer traces expose one per_layer_kv_conditioning edge per shared VLM/Expert layer index
edge source and target layer indices are equal for per-layer K/V conditioning
old/minimal traces still load with empty architecture metadata
```

### Frontend Tests / Snapshots

Assert:

```text
single-option controls are hidden/rendered directly
Raw details does not duplicate primary view
Cache replaces Saved State
pipeline diagram does not render fake cache block
per-layer K/V connectors align VLM and Expert layers
pipeline diagram consumes architecture.edges instead of hard-coded name filters
expert attention ruler appears for pi05.expert_context
active model overlay label is visible
```

### Storage Benchmark Artifact

For any new profile:

```text
write a benchmark note before recommending scale-up
include trace count and task/seed mix
include mean/min/max size
include family/site breakdown
include size per policy call
include wall-clock slowdown
include peak GPU memory
include peak CPU RAM
include trace write time
include rough projection
```

## Decisions and Remaining Open Questions

Several earlier "open questions" are now answered for the next sprint.

### Answered for Phase 1

Should `audit_sampled` include masks/RoPE/position IDs by default?

```text
No, not as standalone raw setup tensors by default.
But audit_sampled must capture enough Q/K/logit/mask semantics to verify attention_probs.
If the current backend requires masks or RoPE tensors for that verification, include the minimal required subset.
```

Should `audit_sampled` include `pre_mask_scores`, `post_mask_logits`, or both?

```text
Prefer both for the first benchmark.
If storage/runtime is too high, keep post_mask_logits and document how pre_mask_scores can be reconstructed.
```

Should `audit_sampled` include `attn_output_pre_o_proj` for both VLM and Expert?

```text
Yes.
Without pre-O-proj attention output, head-level output attribution is crippled.
```

Should `audit_windowed` be static windows or rotating coverage?

```text
Static first if implemented at all:
  [0,1], [4,5], [8,9], [12,13], [16,17]

Rotating coverage is a later dataset-design feature, not a first implementation target.
```

Should `audit_windowed` be whole-episode or selected-policy-call capture?

```text
Whole episode.
Do not silently narrow audit_windowed to selected policy calls.
If storage/runtime is too high, make an explicit role-trimming decision or defer implementation.
```

What is the first clean/corrupt counterfactual family?

```text
Prompt target swap with same scene, seed, object poses, and cameras.
```

Should sparse dictionary artifacts live inside `.vlatrace`?

```text
No for now.
Store as separate workbench artifacts linked by:
  trace_id
  model_hash
  site_id
  layer
  token_space
  training dataset manifest
  activation source hash
```

### Still Open

```text
How much replay input should be stored in .vlatrace versus referenced externally?

What object labels can be derived reliably from LIBERO task metadata versus prompt parsing?

What action metric best predicts rollout-relevant restoration beyond action_head.output MSE?

What tolerance is realistic for deterministic replay on the actual PI0.5 backend and hardware?

What is the minimal object/token mapping needed for target-vs-distractor attention summaries?
```

## Non-Goals for the Next Sprint

Do not do these before the profile/architecture foundation is stable:

```text
train large SAE suites across every layer
build CLT attribution graphs
claim attention is causal importance
claim probe success proves model use
scale audit captures without size benchmarks
build a new full UI shell from scratch
invent aggregate tensors that were not captured
make raw tensor names the primary navigation
```

## Common Failure Modes

Future agents should explicitly check against this list before and after implementation.

```text
1. Re-implementing the existing profile ontology.
2. Renaming `internals_sampled` instead of adding `audit_sampled`.
3. Letting `audit_sampled` silently become `audit_full` with fewer layers.
4. Creating a materialized fake VLM cache tensor.
5. Drawing a cache block or all-to-all VLM -> Expert bus.
6. Capturing Q/K/V without defining pre/post RoPE and mask semantics.
7. Capturing attention logits in a way that silently disables fast attention and makes capture impractically slow.
8. Scaling `audit_sampled` before size/runtime benchmarks.
9. Starting SAE/transcoder work before replay and patching.
10. Treating attention mass as causal importance.
11. Treating activation rows as independent probe samples.
12. Building a new UI shell instead of consuming architecture metadata.
13. Adding aggregate tensors without exact provenance.
14. Implementing patching before no-intervention replay reproduces action_head.output.
15. Claiming object/phase/success probes before labels and splits are reliable.
```

## External Research Anchors to Verify Before Citing

These were discussed as directional anchors. Verify details before formal citation.

```text
Transformer Circuits, Circuit Tracing / attribution graphs:
https://transformer-circuits.pub/2025/attribution-graphs/methods.html

Transformer Circuits, Progress on Attention:
https://transformer-circuits.pub/2025/attention-update/index.html

Transformer Circuits, Crosscoder diffing update:
https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html

Transformer Circuits, Sparse mixtures of linear transforms:
https://transformer-circuits.pub/2025/bulk-update/index.html

Google DeepMind, Gemma Scope 2:
https://deepmind.google/blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/
```

## Recommended Phase 1 Agent Prompt

Use this when starting the next implementation session.

```text
Read `docs/experiments/pi05-vla-lens-roadmap.md`.

Your task is only Phase 1.

Implement `audit_sampled` as a new canonical PI0.5 capture profile.

Do not rename or change the meaning of `internals_sampled`.
Do not change `audit_full` except where tests need to share declarations.
Do not implement probes, SAEs, transcoders, patching, steering, counterfactual capture, or a new UI shell.

`audit_sampled` must:
  - use sampled layers [0, 4, 8, 12, 17] for both VLM and Expert,
  - include circuit-boundary roles around attention and MLP blocks,
  - include Q/K/V with coordinate-system metadata,
  - include pre_mask_scores and post_mask_logits for the first benchmark unless impossible,
  - include attention_probs,
  - include attn_output_pre_o_proj and o_proj,
  - include residual and norm boundaries,
  - include MLP gate/up/intermediate/down/output,
  - include Expert AdaRMS gate/scale/shift,
  - include VLM per-layer K/V,
  - include Expert input embeddings and action_head input/output,
  - avoid fake materialized cache tensors.

Before writing capture hooks, define exact semantics for:
  - q,
  - k,
  - v,
  - pre_mask_scores,
  - post_mask_logits,
  - attention_probs,
  - attn_output_pre_o_proj,
  - o_proj,
  - residual_pre_attention,
  - residual_post_attention,
  - residual_pre_mlp,
  - residual_post_mlp,
  - mlp_output.

The Q/K/logit definitions must state:
  - pre/post RoPE,
  - pre/post head scaling,
  - pre/post grouped-query/key-value expansion,
  - mask semantics,
  - whether attention_probs can be recomputed or verified.

Add architecture metadata to `/api/activation-sites`:
  - architecture.nodes
  - architecture.edges
  - one `per_layer_kv_conditioning` edge for each captured VLM/Expert layer pair
  - sampled traces must expose exactly:
      VLM L0  -> Expert L0
      VLM L4  -> Expert L4
      VLM L8  -> Expert L8
      VLM L12 -> Expert L12
      VLM L17 -> Expert L17
  - preserve existing `pi05.vlm.past_key_values` runtime_collection behavior
  - old traces must load with empty or partial architecture metadata

If UI is touched:
  - consume architecture.edges,
  - render per-layer K/V connectors,
  - do not hard-code a new PI0.5 graph from tensor-name filters,
  - do not create a new UI shell.

Required tests:
  - profile canonicalization,
  - `audit_sampled` role inclusion/exclusion,
  - `internals_sampled` unchanged,
  - `audit_full` unchanged,
  - Q/K/logit coordinate metadata exists,
  - attention_probs verification smoke test for one VLM and one Expert layer,
  - architecture edge count and exact layer pairing,
  - expert attention token spaces remain query=pi05.action_suffix and key=pi05.expert_context,
  - old traces load with empty/partial architecture metadata,
  - no fake standalone VLM cache tensor appears.

After implementation, run 3-5 smoke traces and produce a benchmark note with:
  - total size,
  - size by tensor family/site,
  - size per policy call,
  - wall-clock slowdown,
  - peak GPU memory,
  - peak CPU RAM,
  - trace write time,
  - projection to 100 and 1,000 episodes.

Stop after that.
```

## Final Strategic Correction

The correct framing for future work:

```text
VLA Lens already has the major profile ontology, schema fields, runtime K/V collection, expert-context token space, and audit_full machinery.

The next missing research/product layer is audit_sampled:
  circuit-useful internals at sampled layers
  benchmarked storage cost
  exact provenance
  no fake tensors

The next missing UI/schema layer is architecture-edge metadata:
  VLM layer i prefix K/V -> Expert layer i attention
  query token space = action suffix
  key token space = expert context
  visualized as per-layer conditioning, not a cache block or all-to-all bus

The next missing research workflow is causal:
  probe grid -> attention routing -> counterfactual pairs -> patching -> steering -> sparse dictionaries
```

This roadmap should be read before coding changes in this area.
