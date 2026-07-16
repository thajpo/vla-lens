# VLA Lens Intervention Evidence Layer — Implementation Plan

Status: remaining-work roadmap.

Last updated: 2026-07-15.
Primary goal: finish claim-eligible intervention execution and comparison without changing the implemented evidence spine.

---

## 0. North Star

VLA Lens should let a researcher do this cleanly:

```text
open episode/cohort
→ inspect discovery artifact
→ convert artifact to target
→ choose intervention operator
→ run or inspect counterfactual
→ compare original/no-op/intervened action or rollout
→ save lossless evidence
→ later scale to controls, sweeps, and cohorts
```

The central durable object is:

```text
InterventionRun = executed or inspected intervention spec
                + trials/outcomes
                + runtime/preflight status
                + provenance
                + claim/evidence metadata
```

Saved, inspectable, lossless records now precede live PI0.5 execution. Remaining
runtime work stays behind preflight and capability boundaries.

---

## 1. Hard Invariants

These should hold across all phases.

### 1.1 Saved evidence is inspectable without model runtime

A saved `InterventionRun` must be readable in the normal dashboard environment even when PI0.5, Torch, LeRobot, LIBERO, GPU dependencies, or model checkpoints are unavailable.

### 1.2 Runtime is capability-gated

The presence of an episode, policy call, probe, or target does **not** imply the intervention can run. Runtime execution requires preflight.

### 1.3 No causal overclaim

A saved-only or inspected-only record is not causal evidence. The schema must distinguish:

```text
inspected_only
ok
partial
failed
```

and separately distinguish evidence labels:

```text
observation
predictive
causal_local
action_level
rollout_level
causal_cohort
behavioral
specific
```

### 1.4 Typed contracts first, heavy runtime later

Core contracts must not import PI0.5, Torch, LeRobot, LIBERO, or GPU-specific runtime dependencies.

### 1.5 Lossless does not mean inline everything

Large arrays should be referenced through storage refs or artifact array refs. The saved record must contain exact specs, IDs, hashes/fingerprints, metrics, and array references, not necessarily inline tensor payloads.

### 1.6 Workbench record is canonical; LensArtifact is index/display

Do not create two independent sources of truth. The typed intervention payload should be saved through the workbench intervention record. `LensArtifact(type="intervention_run")` should index and display it.

### 1.7 No old shell compatibility

V0 should not preserve old intervention shell aliases. New typed records should emit:

```text
intervention_type = "intervention_record"
```

The shell type is just storage classification, not evidence strength. Causal interpretation must come from typed fields such as `status`, `trials`, `outcomes`, controls, claim labels, and provenance.

---

## 2. Architecture Shape

Target architecture:

```text
src/vla_lens/interventions/
  __init__.py
  specs.py          # context, target, operator, schedule, outcome, control specs
  results.py        # trial/run/sweep/study/result dataclasses
  serialization.py  # shell mapping and compatibility helpers, if needed
  artifacts.py      # typed run -> LensArtifact index/display shell
  preflight.py      # runtime-agnostic capability checks
  action_basis.py   # action basis metadata/resolution, no heavy deps
  runtime.py        # lightweight protocol/interface only

src/vla_lens/pi05/
  intervention_preflight.py  # PI0.5-specific resolver/checks
  intervention_runtime.py    # PI0.5-specific live execution, heavy env only
  interventions.py           # existing low-level PI0.5 specs remain adapter-local

src/vla_lens/workbench/
  schema.py          # existing InterventionRunSpec shell remains compatible
  api.py/server code # existing saved-route support extended carefully

frontend/src/types/
  interventions.ts

frontend/src/api/
  interventions.ts

frontend/src/components/interventions/
  InterventionCard.tsx
  EvidenceLibrary.tsx
  InterventionRunDetail.tsx
  InterventionLab.tsx
```

Do not make the PI0.5-specific intervention schema the global ontology. It should become one runtime adapter implementation of the generic target/operator/schedule/result contracts.

---

## 3. Canonical Object Spine

Everything should plug into this spine:

```text
DiscoveryArtifact
  → TargetSpec
  → InterventionRequest
  → RuntimePreflightResult
  → InterventionTrial(s)
  → InterventionRun
  → LensArtifact / InterventionCard
```

### 3.1 DiscoveryArtifact

Existing or future artifact that suggests a candidate signal.

Examples:

```text
probe
mean_difference_direction
action_generation_artifact
attention_or_attribution_map
activation_cluster
manual_selection
sae_feature
transcoder_feature
crosscoder_feature
```

Discovery artifacts are hypothesis-generating only.

### 3.2 TargetSpec

The normalized description of what internal object is intended to be manipulated.

Required v0 fields:

```text
schema_version
kind
model_site
token_space or token_space_id, if known
token_selector
reduction
representation
source_artifact_id, when artifact-derived
metadata
```

`source_artifact_id` is optional for manual selections and required when the target was created from a probe/SAE/transcoder/etc.

### 3.3 InterventionRequest

Ephemeral request to inspect or execute an intervention. Not necessarily persisted as a separate object.

Contains:

```text
context
recipient
target
operator
schedule
outcome
controls
requested_by / UI source, optional
```

The eventual persisted `InterventionRun` should contain the exact request-like payload that produced it.

### 3.4 RuntimePreflightResult

Capability check result.

Must include:

```text
status: ok | partial | failed | inspected_only
checks: list[PreflightCheck]
warnings
errors
runtime_resolution, if any
missing_capabilities
```

### 3.5 InterventionTrial

One actual or inspected trial.

Examples:

```text
noop_rerun trial
main intervention trial
random direction control trial
wrong layer control trial
failed attempted trial
inspected-only placeholder trial
```

### 3.6 InterventionRun

Canonical saved evidence object.

Required identity/provenance fields:

```text
run_id
schema_version
dataset_id or dataset_root_id
dataset_fingerprint
trace_id
episode_id, if available
policy_call_index
source_artifact_id, when artifact-derived
created_utc
status
```

Payload fields:

```text
title
claim
claim_strengths
context
target
intervention
outcome_request
preflight
trials
outcome_results
controls
runtime
provenance
outputs
```

---

## 4. Phase Plan

The implemented foundation now includes typed contracts, workbench persistence,
artifact indexing, saved-record APIs and UI, runtime-free preflight, action-basis
resolution, sweep/study shells, and a replay-gated non-claiming PI0.5 hook
smoke. The remaining sequence is:

```text
artifact-derived probe direction + project-out/specificity control
→ live Intervention Lab execution and action comparison
→ sweep/cohort execution runner and UI
```

The durable contracts below remain constraints, not unfinished milestones.

---

## 4.1 Remaining Verification Boundary

The remaining runtime must stay in the dedicated PI0.5 environment. Normal
schema, API, aggregation, and frontend checks remain in the repo environment.
A claim-eligible run must add evidence beyond the existing deterministic,
non-claiming synthetic hook smoke.

# Remaining Milestone 1 — Claim-Eligible PI0.5 Probe Runtime

## Current Boundary

The CLI-first runner already proves exact replay, saves an `InterventionRun`
and artifact, and can execute a non-claiming synthetic one-hot `add_direction`
hook with a matched random control. It is intentionally labeled
`synthetic_hook_smoke` and `claim_eligible = false`.

## Remaining Work

- Resolve an artifact-derived probe direction into the requested PI0.5 hook.
- Support both `add_direction` and `project_out_direction`.
- Add at least one specificity control such as a wrong layer, time, or token.
- Preserve replay gating, runtime resolution, stored/no-op/intervened/control
  action refs, action-basis deltas, diagnostics, and provenance.
- Keep execution in the dedicated PI0.5 environment; the normal suite must stay
  free of PI0.5, Torch, LeRobot, LIBERO, and GPU requirements.

## Acceptance

- A saved probe direction and reconstructable policy call produce no-op,
  intervention, and matched-control trials.
- Runtime records the requested and resolved target, model/checkpoint, call,
  schedule mapping, tensor shape/dtype/device summary, and environment.
- Project-out and the chosen specificity control are tested.
- Claim eligibility is granted only when the experiment contract and controls
  support it; failures remain saved as partial/failed runs with diagnostics.

---
# Remaining Milestone 2 — Live Intervention Lab Comparison

## Current Boundary

Probe and Episode Lens entry points seed the Intervention Lab, backend-normalized
targets are preferred over explicit local fallbacks, preflight explains
availability, inspected-only records can be saved, and saved results appear in
the evidence surfaces. The Run action does not yet invoke a live executor.

## Remaining Work

- Connect the Lab to an explicit live execution boundary without importing the
  capture stack into the normal dashboard process.
- Render stored-original, no-op, intervened, and control action chunks with a
  delta chart by horizon and action dimension or resolved basis.
- Keep status, warnings, provenance, and unavailable controls visible.

## Acceptance

- An executable request can run through the Lab and save the returned result.
- The comparison view distinguishes every trial role and exposes action deltas.
- Runtime-unavailable requests still explain why and preserve inspected-only
  evidence without implying a causal result.

---
# Remaining Milestone 3 — Sweep And Cohort Execution

## Current Boundary

`InterventionSweep`, `InterventionStudy`, promotion, aggregation, artifact
indexing, and claim gating exist. No runner materializes many live runs across a
sweep or cohort, and there is no corresponding UI.

## Remaining Work

- Execute a single-run spec across strength, layer, time, generation step,
  target/source artifact, donor/recipient pair, and control seed axes.
- Run over explicit cohorts such as top moments, matched success/failure pairs,
  held-out episodes, or task/object subsets.
- Persist each `InterventionRun` plus one sweep/study summary with coverage,
  failures, effect aggregates, monotonicity, specificity, and split comparison.
- Add UI that never upgrades action/cohort/behavior/specific evidence labels
  beyond the recorded controls and outcomes.

---
# Future Artifact-Family Runtimes

## Goal

The shared family registry and target/operator/outcome/control contracts exist.
Future work can add SAE, transcoder, crosscoder, attention/path, and rollout
runtimes without changing the core spine.

## Rule

Every new artifact type must answer:

```text
How does this produce a TargetSpec?
What operators are legal for that target?
What outcomes make sense?
What controls are required before causal claims?
```

## Examples

```text
SAEFeature
  → TargetSpec(kind="feature")
  → operators: feature_boost, feature_clamp, feature_ablate
  → outcomes: action, rollout, token

TranscoderFeature
  → TargetSpec(kind="feature" or "path")
  → operators: feature_clamp, path_patch
  → outcomes: downstream activation, action, rollout

AttentionEdge
  → TargetSpec(kind="edge")
  → operators: attention_patch, head_ablate
  → outcomes: attention pattern, action, rollout

ActivationCluster
  → TargetSpec(kind="contrast_direction")
  → operators: add_direction, project_out_direction
  → outcomes: action, rollout
```

---

## 5. Remaining Delivery Slices

1. Claim-eligible artifact-derived PI0.5 direction runtime with project-out and
   a specificity control.
2. Live Intervention Lab execution plus original/no-op/intervened/control
   comparison.
3. Sweep/cohort execution runner and evidence-labeled UI.

Keep these slices separate so runtime evidence, UI behavior, and study-scale
aggregation remain independently reviewable.

## 6. Remaining Verification

- Run normal schema/API/aggregation tests and frontend tests in the repo
  environment.
- Run replay and intervention smokes only through the dedicated PI0.5 wrapper.
- Add focused coverage for artifact-derived direction resolution, project-out,
  specificity controls, live Lab result handling, action-delta comparison, and
  sweep/cohort materialization.
- Retain frontend fixtures for inspected-only, action-level, partial-basis, and
  failed-runtime records.

---
## 7. Minimal Concrete v0 Scenario

This is the first end-to-end story to make real.

```text
Artifact:
  gripper-close probe direction

Context:
  selected episode and selected policy call

Target:
  expert action-token hidden state, mean-reduced, layer/site from probe

Operator:
  add_direction
  project_out_direction

Schedule:
  one policy call
  all action tokens
  all generation steps, or adapter default if generation-step scheduling unavailable

Outcome:
  raw action chunk delta
  gripper delta if basis metadata exists
  eef_delta_xyz if basis metadata exists

Controls:
  noop rerun
  random direction
  wrong layer if easy

Output:
  saved InterventionRun
  LensArtifact(type="intervention_run")
  Intervention Card
```

The UI button should say:

```text
Intervene with this signal
```

not:

```text
Create TargetSpec
```

---

## 8. Migration / Compatibility Notes

1. Typed v0 records require an explicit `intervention_type`; the typed
   conversion path uses `intervention_record` and does not restore old shell
   aliases.

2. Do not mutate existing probe or action-generation artifacts to fit intervention needs. Instead, create adapter functions that turn them into `TargetSpec`s.

3. Do not add required fields to existing `PolicyCallRecord` until preflight proves they cannot be derived from existing dataset tables/manifest metadata/model adapter metadata.

4. Do not inline action arrays in workbench records. Store refs and compact metrics.

5. Keep frontend tolerant of partial/missing runtime fields because many records will be inspectable but not executable.

6. Runtime failures should produce failed/partial records when possible, not just throw.

---

## 9. Main Risks

### Risk: schema split between backend and frontend

Mitigation:

```text
fixture JSON files checked by both Python tests and frontend tests
schema_version on payloads
stable typed examples in docs
```

### Risk: action basis lies silently

Mitigation:

```text
require units/sign/source_dimensions/normalization
partial result if metadata missing
never infer gripper/xyz mapping without provenance
```

### Risk: generic intervention schema becomes too abstract to implement

Mitigation:

```text
v0 only probe direction + add/project-out + action outcome
all other target/operator kinds are typed but may be unsupported by preflight
```

### Risk: runtime contaminates normal dashboard env

Mitigation:

```text
CLI-first PI0.5 runtime or lazy adapter registration
normal preflight returns runtime unavailable
normal tests assert no heavy imports
```

### Risk: causal overclaim

Mitigation:

```text
separate status from claim labels
show no-op/control availability
saved-only = inspected_only
single action delta = action_level/causal_local at most
cohort/rollout/specific labels require evidence
```

### Risk: arrays bloat JSON records

Mitigation:

```text
array refs in outputs/artifact arrays
compact previews only
```

---

## 10. Final Sequencing Summary

The implemented evidence spine is now the stable base. Remaining work is:

```text
1. Run an artifact-derived, controlled, claim-eligible PI0.5 intervention.
2. Expose live execution and action comparison in the Intervention Lab.
3. Materialize and inspect controlled sweeps and cohort studies.
```

Future model-internal methods should reuse the same target, trial, outcome,
control, provenance, and claim-evidence contracts.
