# VLA Lens Intervention Evidence Layer — Implementation Plan

Status: implementation scaffold  
Date: 2026-06-06  
Primary goal: implement the Intervention Evidence Layer incrementally, without forcing later schema or runtime rewrites.

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

The first implementation should prioritize saved, inspectable, lossless records before live PI0.5 execution. Runtime should land behind preflight and capability boundaries.

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
  InterventionLab.tsx       # later, after saved evidence and preflight work
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

The efficient sequence is:

```text
spec/docs
→ schema contracts
→ saved workbench compatibility
→ artifact indexing
→ read APIs
→ saved evidence UI
→ preflight
→ action basis resolution
→ PI0.5 runtime v0
→ Intervention Lab UI
→ sweeps/cohorts
```

Do **not** start by building the live runtime and UI first. That creates schema churn and makes every later result harder to preserve losslessly.

---

## 4.1 Codex Goal Mode Usage

Use one Goal Mode run per phase. Do not ask Codex to implement multiple phases
unless the earlier phase has already passed its tests and the git diff is clean
enough to review.

When the user says only `/goal implement Phase X`, treat this file as the source
of truth. Find the matching `# Phase X` heading, implement that phase only, and
use the universal contract plus phase verification matrix below.

Recommended command shape:

```text
/goal implement Phase X from vla_lens_intervention_implementation_plan.md.
Stay inside that phase only. Add or update the tests named for the phase as you
implement. Run the targeted tests first, then the relevant normal verification
commands. Do not import PI0.5/Torch/LeRobot/LIBERO into the normal repo env.
```

Universal implementation contract for every phase:

```text
1. Start with git status and inspect the files named in the phase.
2. Implement only the named phase.
3. Add focused tests before or alongside behavior changes.
4. Keep typed evidence runtime-free until the PI0.5 runtime phase.
5. Use `intervention_record` for v0 workbench intervention records.
6. Do not emit or accept old intervention shell aliases in the typed v0 path.
7. Do not use `intervention_type` as a causal-evidence flag.
8. Store arrays by ref, not inline JSON payloads.
9. Run phase tests and report any skipped or blocked verification.
10. End with git status and a concise diff/test summary.
```

Resolved implementation decisions:

```text
Canonical typed payload placement:
  Split across target/baseline/intervention/readouts/provenance using
  deterministic conversion helpers. Do not also store a second full copy unless
  it is explicitly marked derived.

Workbench intervention_type:
  intervention_record for all v0 typed records. Whether a record is causal is
  interpreted from status, trials, outcomes, controls, claim labels, and
  provenance.

Validation style:
  dataclasses plus explicit from_dict validation, consistent with current
  workbench style.

Runtime boundary:
  CLI-first for PI0.5 runtime. Add an opt-in server route only after the CLI
  path writes saved records cleanly.

First action basis beyond raw:
  gripper, only if action normalization/schema metadata gives an auditable
  dimension mapping.

First target adapter:
  probe artifact -> TargetSpec(kind="probe_direction").
```

Phase verification matrix:

```text
Phase 0 docs:
  rg -n "intervention[_-]delta" vla_lens_intervention_implementation_plan.md vla_lens_intervention_evidence_spec_v2.md docs
  # Expected: no matches. For this check, rg exit code 1 means success.

Phase 1 schema:
  uv run pytest tests/test_intervention_specs.py tests/test_intervention_results.py
  uv run ruff check src/vla_lens/interventions tests/test_intervention_specs.py tests/test_intervention_results.py

Phase 2 workbench compatibility:
  uv run pytest tests/test_intervention_workbench_compat.py tests/vla_lens_trace_selection_test.py
  uv run ruff check src/vla_lens/interventions src/vla_lens/workbench tests/test_intervention_workbench_compat.py

Phase 3 artifact indexing:
  uv run pytest tests/test_intervention_artifact_indexing.py tests/vla_lens_trace_artifacts_test.py
  uv run ruff check src/vla_lens/interventions tests/test_intervention_artifact_indexing.py

Phase 4 saved evidence API:
  uv run pytest tests/fastapi_server_test.py tests/server_api_test.py
  uv run ruff check src/vla_lens/server src/vla_lens/workbench tests/fastapi_server_test.py

Phase 5 frontend saved evidence view:
  cd frontend && npm run test && npm run build && npm run lint

Phase 6 runtime-free preflight:
  uv run pytest tests/test_intervention_preflight.py tests/fastapi_server_test.py
  uv run ruff check src/vla_lens/interventions src/vla_lens/server tests/test_intervention_preflight.py

Phase 7 action basis:
  uv run pytest tests/test_action_basis.py tests/test_intervention_preflight.py
  uv run ruff check src/vla_lens/interventions tests/test_action_basis.py

Phase 8 PI0.5 runtime:
  uv run pytest tests/test_pi05_intervention_runtime_contract.py
  scripts/check_pi05_env.sh --backend rocm
  scripts/pi05_capture.sh --backend rocm [explicit runtime smoke args]

Phase 9 Intervention Lab UI:
  cd frontend && npm run test && npm run build && npm run lint
  uv run pytest tests/fastapi_server_test.py

Phase 10 sweeps/cohorts:
  uv run pytest tests/test_intervention_sweeps.py tests/test_intervention_artifact_indexing.py
  uv run ruff check src/vla_lens/interventions tests/test_intervention_sweeps.py
```

If a named test file does not exist yet, create it during that phase. If a
phase legitimately cannot run one command, record the concrete reason in the
final Goal Mode summary.

---

# Phase 0 — Normalize The Spec

## Goal

Make the written spec internally consistent before writing code.

## Changes

1. Add/update docs:

```text
docs/intervention-evidence-layer.md
docs/glossary.md
```

2. Replace intervention shell type:

```text
removed/unsupported: old intervention shell aliases
v0 emitted type: intervention_record
```

3. Split acceptance criteria explicitly:

```text
Evidence Layer v0:
  saved/listable/inspectable records

Runtime v0:
  live PI0.5 direction intervention
```

4. Add `ActionBasis` provenance language:

```text
action_schema_ref
basis_resolution
units
sign_convention
source_dimensions
normalization
```

5. Require persisted `InterventionRun` identity fields:

```text
dataset_id or dataset_root_id
dataset_fingerprint
trace_id
policy_call_index
source_artifact_id when artifact-derived
```

6. State that saved-only records are not causal evidence.

## Acceptance

- Docs distinguish saved evidence from live runtime.
- Docs define `donor`, `recipient`, `site`, `target`, `runtime hook`, `operator`, `schedule`, `outcome`, `control`, `trial`, `run`, `sweep`, `study`, and `mechanism card`.
- No doc implies that an inspected-only record is causal.

---

# Phase 1 — Schema Shell

## Goal

Add typed intervention contracts with JSON roundtrip tests, without touching runtime or frontend.

## Files

```text
src/vla_lens/interventions/__init__.py
src/vla_lens/interventions/specs.py
src/vla_lens/interventions/results.py
src/vla_lens/interventions/serialization.py   # optional if helpers grow

tests/test_intervention_specs.py
tests/test_intervention_results.py
```

## Types

In `specs.py`:

```text
ContextSpec
TraceRef
PolicyCallRef
RecipientSpec
DonorSpec
TargetSpec
InterventionScheduleSpec
InterventionOperatorSpec
ControlSpec
OutcomeSpec
ActionBasisRequest
RuntimePreflightResult
RuntimeResolution
PreflightCheck
```

In `results.py`:

```text
InterventionTrial
ActionOutcomeResult
RolloutOutcomeResult
TokenOutcomeResult
ControlResult
InterventionRun
InterventionSweep       # stub or typed shell only
InterventionStudy       # stub or typed shell only
```

## Implementation rules

- Use dataclasses consistent with existing workbench style.
- Prefer `frozen=True, slots=True` unless mutation is needed.
- Include `schema_version` on top-level saved payloads.
- Include `to_dict` / `from_dict` methods.
- Use `Mapping[str, Any]` escape hatches for future fields, but enforce identity fields for persisted runs.
- No Torch/PI0.5/LeRobot imports.
- Large outputs are refs, not inline arrays.
- `InterventionRun.status` is separate from claim/evidence labels.

## Minimal enum/string literals

```text
RunStatus:
  inspected_only | ok | partial | failed

TrialKind:
  noop_rerun | intervention | random_direction_control |
  wrong_layer_control | wrong_time_control | wrong_token_control |
  source_patch_control | manual

TargetKind:
  probe_direction | contrast_direction | activation_slice | feature |
  subspace | head | edge | manual

OperatorKind:
  add_direction | project_out_direction | replace | ablate |
  clamp | source_patch | mean_replace

OutcomeKind:
  action | rollout | token | probe | metric
```

## Acceptance

- Unit tests construct a probe-direction intervention run and roundtrip through JSON.
- Missing required persisted identity fields fail validation or raise clean errors.
- Tests prove no import of heavy PI0.5 runtime dependencies.
- Legacy flexibility exists through `metadata` fields, but canonical fields are present.

---

# Phase 2 — Workbench Compatibility

## Goal

Save typed `InterventionRun` payloads through the current workbench storage route/shell.

## Current compatibility target

Existing shell:

```text
vla_lens.workbench.schema.InterventionRunSpec
  run_id
  intervention_type
  target
  baseline
  intervention
  readouts
  outputs
  provenance
```

## Mapping strategy

Do not replace the shell immediately. Add conversion helpers:

```text
InterventionRun.to_workbench_spec() -> InterventionRunSpec
InterventionRun.from_workbench_spec(spec) -> InterventionRun
```

Recommended mapping:

```text
InterventionRunSpec.run_id
  = run.run_id

InterventionRunSpec.intervention_type
  = "intervention_record" for typed v0 records

Interpretation of whether the record is causal comes from status, trials,
outcomes, controls, claim labels, and provenance; not from intervention_type.

InterventionRunSpec.target
  = run.target.to_dict()

InterventionRunSpec.baseline
  = {
      "context": run.context,
      "recipient": run.recipient,
      "donor": run.donor_if_any,
      "stored_original_refs": ...,
      "noop_refs": ...
    }

InterventionRunSpec.intervention
  = {
      "operator": run.operator,
      "schedule": run.schedule,
      "controls_requested": run.controls_requested,
      "request": run.request_like_payload
    }

InterventionRunSpec.readouts
  = {
      "status": run.status,
      "claim": run.claim,
      "claim_strengths": run.claim_strengths,
      "preflight": run.preflight,
      "trials": run.trials,
      "outcome_request": run.outcome_request,
      "outcome_results": run.outcome_results,
      "control_results": run.control_results,
      "runtime": run.runtime
    }

InterventionRunSpec.outputs
  = tuple of array/artifact/media refs

InterventionRunSpec.provenance
  = {
      "schema_kind": "vla_lens.intervention_run",
      "schema_version": run.schema_version,
      "dataset_id": ...,
      "dataset_root_id": ...,
      "dataset_fingerprint": ...,
      "trace_id": ...,
      "episode_id": ...,
      "policy_call_index": ...,
      "source_artifact_id": ...,
      "code_version": ...,
      "created_utc": ...
    }
```

## Rules

- Do not emit or accept old intervention shell aliases in v0 typed records.
- `intervention_type` must not be used as a causal-evidence flag.
- `causal_evidence` or equivalent display/claim interpretation is false unless actual trials/outcomes/controls justify it.
- The shell mapping must be lossless for all typed fields.

## Tests

```text
tests/test_intervention_workbench_compat.py
```

Test cases:

- typed run -> shell -> typed run roundtrip
- saved-only run is not causal evidence
- executed-looking records are interpreted from typed trials/outcomes, not `intervention_type`
- artifact-derived target without `source_artifact_id` fails validation
- manual target without `source_artifact_id` is allowed

## Acceptance

- Existing `/api/intervention-runs` route can save and list typed nested payloads.
- No frontend/runtime code is required yet.
- No heavy runtime imports.

---

# Phase 3 — LensArtifact Indexing

## Goal

Make intervention runs discoverable through the artifact system without making `LensArtifact` canonical.

## Files

```text
src/vla_lens/interventions/artifacts.py

tests/test_intervention_artifact_indexing.py
```

## Helper

```text
intervention_run_to_lens_artifact(run: InterventionRun) -> LensArtifact
```

## Mapping

```text
LensArtifact.artifact_type
  = "intervention_run"

LensArtifact.name
  = run.title or generated concise title

LensArtifact.selector
  = context + target summary

LensArtifact.method
  = operator + schedule + request hash

LensArtifact.metrics
  = compact outcome/control metrics only

LensArtifact.arrays
  = refs to action chunks/deltas/videos/tables

LensArtifact.display
  = Intervention Card summary

LensArtifact.tags
  = evidence labels, status, operator kind, outcome kind

LensArtifact.source_trace_ids
  = recipient trace + donor traces if any
```

## Rules

- The artifact points back to the workbench run ID.
- The workbench `InterventionRun` remains canonical.
- Array-heavy data is by ref.

## Acceptance

- A saved typed run creates a valid `LensArtifact(type="intervention_run")`.
- Artifact browser can list it like probes/action-generation artifacts.
- The artifact display has enough data to render a card without opening large arrays.

---

# Phase 4 — Backend Saved Evidence Read APIs

## Goal

Backend can save, list, and open saved intervention evidence.

## Routes

Use/extend existing saved-state route:

```text
GET  /api/intervention-runs
POST /api/intervention-runs
```

Add if missing:

```text
GET /api/intervention-runs/{run_id}
```

Optional helper:

```text
GET /api/artifacts/{artifact_id}
```

should be able to link back to the intervention run.

## Do not add live execution yet

No route in this phase should run a model, import PI0.5, load Torch, or execute an intervention.

## Acceptance

- POST saves a typed run.
- GET lists saved runs.
- GET by ID opens one run with context, target, intervention, trials/outcomes, status, claim, and provenance.
- Existing saved-route behavior remains stable for typed `intervention_record` records.
- Saved records survive process restart and reload from disk.

---

# Phase 5 — Frontend Saved Evidence View

## Goal

Users can inspect saved runs before live intervention exists.

## Files

```text
frontend/src/types/interventions.ts
frontend/src/api/interventions.ts
frontend/src/components/interventions/InterventionCard.tsx
frontend/src/components/interventions/InterventionRunDetail.tsx
frontend/src/components/interventions/EvidenceLibrary.tsx
```

## UI surfaces

### Evidence Library

List saved intervention runs.

Columns/cards:

```text
title
status
claim/evidence labels
episode/trace
policy call
target summary
operator summary
outcome kind
created time
```

### Intervention Card

Compact summary:

```text
Claim
Context
Target
Intervention
Outcome
Controls
Status
Provenance
```

### Detail view

Shows full typed payload with collapsible sections.

## Status badges

```text
inspected_only
ok
partial
failed
```

## Evidence labels

```text
observation
predictive
action_level
causal_local
rollout_level
causal_cohort
behavioral
specific
controls_missing
controls_partial
```

## Acceptance

- Fixture saved run renders without model runtime.
- `inspected_only` records are visibly not causal evidence.
- A saved run can be opened from artifact browser or evidence library.
- Frontend types match backend fixtures.

---

# Phase 6 — Runtime Preflight Without Heavy Runtime

## Goal

Explain whether a selected intervention can run from the opened dataset root.

## Files

```text
src/vla_lens/interventions/preflight.py
src/vla_lens/server/interventions.py      # or existing server module

tests/test_intervention_preflight.py
```

## Route

```text
POST /api/interventions/preflight
```

## Checks

```text
policy_call_exists
stored_action_exists
stored_action_chunk_exists
source_artifact_exists, if artifact-derived
target_site_declared_in_model_site_index
token_space_declared, if required
action_decoder_metadata_available
action_basis_metadata_available
runtime_adapter_declared
model_runtime_available
runtime_environment_safe
```

## Result behavior

- If all saved evidence exists but model runtime unavailable:

```text
status = inspected_only or partial
message = "Can inspect saved context, but cannot rerun this policy call from this environment."
```

- If action basis metadata is missing:

```text
status = partial
raw basis available, requested named basis unavailable
```

- If source artifact missing:

```text
status = failed
clear error with missing ID
```

## Critical rule

Normal dashboard `.venv` path must not import PI0.5/Torch/LeRobot/LIBERO runtime. PI0.5-specific checks can be registered lazily or reported as unavailable unless running in the dedicated runtime environment.

## Acceptance

- Preflight returns actionable checks/errors/warnings.
- Preflight is deterministic and testable on synthetic demo data.
- Preflight does not import heavy runtime dependencies.
- UI can call preflight to decide whether to show “Run now” or only “Inspect/save record.”

---

# Phase 7 — Action Basis Provenance And Adapters

## Goal

Make action deltas auditable and not misleading.

## Files

```text
src/vla_lens/interventions/action_basis.py
src/vla_lens/action_basis.py              # only if broader package-level module preferred

tests/test_action_basis.py
```

## Types

```text
ActionBasisRequest
ActionBasisResolution
ActionBasisResult
ActionSchemaRef
ActionNormalizationSpec
```

## Required provenance fields

```text
action_schema_ref
basis_name
basis_resolution
units
sign_convention
source_dimensions
normalization
coordinate_frame
warnings
```

## v0 basis support

Always attempt:

```text
raw
```

Support only when metadata exists:

```text
gripper
eef_delta_xyz
rotation
speed
```

Future:

```text
object_relative
task_relative
pca
learned_action_feature
```

## Failure behavior

Missing basis metadata should produce a partial result, not a crash.

Example:

```text
requested: [raw, gripper, eef_delta_xyz]
available: [raw]
missing: [gripper, eef_delta_xyz]
status: partial
```

## Metrics

For each resolved basis:

```text
raw_delta
noop_delta
intervened_minus_noop
intervened_minus_stored_original
normalized_delta, if normalization available
side_effect_score, if intended dimensions declared
```

## Acceptance

- Raw action basis works on synthetic/demo saved action chunks.
- Missing gripper/EEF basis metadata yields partial status.
- Sign conventions and units are explicit in output.
- No silent assumptions about action dimension meaning.

---

# Phase 8 — PI0.5 Runtime v0

## Goal

First live direction intervention path.

This should come only after saved records, artifact indexing, read APIs, UI cards, preflight, and action basis refs are stable.

## Runtime boundary options

Choose one of these implementation surfaces.

### Option A: CLI-first runtime

Preferred initial safety path.

```text
scripts/run_pi05_intervention.py
# or package command:
vla-pi05-intervention-run
```

Runs in PI0.5 dedicated env and writes a saved `InterventionRun` record/artifact. The normal dashboard can then inspect it.

### Option B: opt-in live server route

Only when dashboard is intentionally running in a runtime-capable PI0.5 environment.

```text
POST /api/interventions/run
```

Must first call or internally perform preflight. In normal `.venv`, it returns unavailable without importing heavy deps.

### Option C: runtime broker/process

Later option if live UI must stay in normal dashboard while execution runs elsewhere.

## Files

```text
src/vla_lens/pi05/intervention_preflight.py
src/vla_lens/pi05/intervention_runtime.py
src/vla_lens/pi05/interventions.py        # keep existing low-level specs compatible
src/vla_lens/interventions/runtime.py     # generic protocol only

tests/test_pi05_intervention_runtime_contract.py  # no heavy import unless marked
```

## Operators v0

```text
noop_rerun
add_direction
project_out_direction
random_direction_control
wrong_layer_control, if cheap
```

## Required runtime output

A real `InterventionRun` with:

```text
preflight result
runtime resolution
noop trial, if available
main intervention trial
control trials, if requested and available
stored original action ref
noop action ref
intervened action ref
action deltas in resolved bases
status
errors/warnings
provenance
```

## RuntimeResolution must record

```text
requested target
resolved runtime hook
model adapter
model checkpoint / model id
call index
generation step mapping, if applicable
token selector mapping
actual tensor shape/dtype/device summary
runtime environment
```

## Acceptance

- Given a saved probe direction and reconstructable PI0.5 policy call, runtime can run no-op and add-direction.
- Produces a saved `InterventionRun` that opens in existing Evidence view.
- Random direction control can run with matched norm/shape.
- If runtime fails, failure is saved as a failed/partial run with useful diagnostics.
- Normal test suite still runs without PI0.5/Torch/GPU.

---

# Phase 9 — Intervention Lab UI

## Goal

Expose the researcher workflow.

## Entry points

```text
Probe artifact page:
  "Intervene with this signal"

Episode/policy call view:
  "Send to Intervention"

Artifact browser:
  "Open in Intervention Lab"
```

## UI flow

Use simple researcher-facing labels:

```text
1. Where?
   episode, trace, policy call, frame/timestep

2. What signal?
   probe direction, manual site, future SAE feature, etc.

3. How change?
   add direction, project out, no-op/control

4. When?
   policy call, tokens, generation steps, action horizon

5. What measure?
   raw action, gripper, xyz, rotation, rollout, token

6. Controls?
   no-op, random direction, wrong layer/time/token

7. Compare
   original vs no-op vs intervened

8. Save evidence
   InterventionRun + LensArtifact
```

## Initial UI constraints

- v0 can only support one policy call.
- Probe and Episode Lens entry points should seed the lab with the
  backend-normalized `TargetSpec` from
  `/api/discovery-artifacts/{artifact_id}/target` when available, passing trace,
  policy call, model site, and token space.
- If the target fetch fails, the UI may use an inspectable local fallback target
  only when it marks `target.metadata.target_source` as `local_fallback`.
- v0 can show unavailable controls as disabled with reasons from preflight.
- v0 should allow saving an inspected-only record even if runtime unavailable.
- v0 should not require full rollout video generation.

## Comparison view

Minimum:

```text
stored original action chunk
noop regenerated action chunk, if available
intervened action chunk, if available
delta chart by horizon and dimension/basis
status/warnings
control summary
```

Later:

```text
side-by-side video/replay
closed-loop rollout comparison
contact/success metrics
```

## Acceptance

- User can start from a probe artifact and construct a valid intervention request.
- Preflight runs and determines whether live execution is possible.
- If not possible, UI still explains why and can save an inspected-only record.
- If possible, UI can run and save a result.
- Saved result appears in Evidence Library and Artifact Browser.

---

# Phase 10 — Sweeps And Cohorts

## Goal

Scale local evidence into research-grade evidence.

## Types

```text
InterventionSweep
InterventionStudy
CohortInterventionRequest
SweepAxis
AggregateOutcomeResult
```

## Sweep axes

```text
strength
layer
policy_call/time
generation_step
token selector
target/source artifact
donor/recipient pair
random seed/control seed
```

## Cohort support

Run over:

```text
top activating moments
matched success/failure pairs
heldout episodes
specific task/object subsets
gripper-close failures
object-contact phase windows
```

## Aggregates

```text
mean effect
median effect
std/confidence interval, later
monotonicity
specificity
side-effect score
coverage
failure counts
heldout/train split comparison
```

## Acceptance

- A single-run spec can be promoted to a sweep without changing its meaning.
- Sweeps produce many `InterventionRun`s plus one `InterventionSweep` summary.
- Studies can reference sweeps and controls over cohorts.
- UI labels evidence as cohort/action/behavior/specific only when appropriate.

---

# Phase 11 — Future Artifact Families

## Goal

Add SAEs, transcoders, crosscoders, attention/path patching, and rollout outcomes without changing the core spine.

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

## 5. Suggested PR Breakdown

### PR 1 — Docs and glossary

- Add/update intervention evidence spec docs.
- Add glossary.
- Define non-goals and evidence labels.

### PR 2 — Core dataclasses

- Add `src/vla_lens/interventions/specs.py` and `results.py`.
- Add roundtrip tests.

### PR 3 — Workbench shell compatibility

- Add typed-run ↔ `InterventionRunSpec` conversion.
- Remove old intervention shell aliases from the v0 typed-record path.
- Add tests.

### PR 4 — Artifact indexing

- Add `intervention_run_to_lens_artifact`.
- Add artifact browser fixture.

### PR 5 — Saved evidence API

- Ensure list/save/open saved intervention runs.
- Add GET by ID if missing.
- Add API tests.

### PR 6 — Frontend saved evidence view

- Add TS types/fixtures.
- Add Evidence Library and Intervention Card.

### PR 7 — Preflight

- Add runtime-agnostic preflight.
- Add `/api/interventions/preflight`.
- No heavy imports.

### PR 8 — Action basis

- Add action-basis provenance/resolution.
- Raw basis first; named bases partial if metadata absent.

### PR 9 — PI0.5 runtime v0

- CLI-first or opt-in runtime route.
- No-op, add-direction, project-out, random control.
- Output saved `InterventionRun`.

### PR 10 — Intervention Lab

- UI workflow from probe/episode/artifact to preflight/run/save.

### PR 11 — Sweeps/cohorts

- Promote run specs to sweep/study.
- Aggregate outcomes and controls.

---

## 6. Tests And Fixtures

## Unit tests

```text
test_target_spec_roundtrip
test_intervention_operator_roundtrip
test_schedule_roundtrip
test_outcome_spec_roundtrip
test_intervention_run_required_identity_fields
test_intervention_run_json_roundtrip
test_artifact_derived_target_requires_source_artifact
test_manual_target_allows_no_source_artifact
test_intervention_type_is_not_causal_claim
```

## Workbench compatibility tests

```text
test_typed_run_to_workbench_spec_roundtrip
test_saved_only_record_not_causal
test_workbench_spec_rejects_unknown_intervention_type
test_workbench_spec_outputs_refs_not_inline_arrays
```

## Artifact tests

```text
test_intervention_run_to_lens_artifact
test_artifact_display_summary_has_context_target_outcome
test_artifact_source_trace_ids_include_recipient_and_donor
```

## API tests

```text
test_post_intervention_run
test_list_intervention_runs
test_get_intervention_run_by_id
test_preflight_missing_artifact
test_preflight_runtime_unavailable_no_heavy_import
```

## Action basis tests

```text
test_raw_action_basis_always_available_with_action_chunk
test_named_basis_missing_returns_partial
test_basis_resolution_records_units_sign_dimensions_normalization
```

## Runtime tests

Mark heavy tests separately.

```text
@pytest.mark.pi05_runtime
@pytest.mark.gpu_optional
```

Examples:

```text
test_pi05_noop_rerun_contract
test_pi05_add_direction_outputs_action_outcome
test_pi05_random_direction_control_matched_norm
test_pi05_runtime_failure_saves_failed_run
```

## Frontend fixtures

Maintain JSON fixtures for:

```text
inspected_only saved record
ok action-level intervention run
partial action-basis result
failed runtime result
```

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

1. Remove old intervention shell compatibility paths for v0 typed records. `InterventionRunSpec.from_dict` should require an explicit `intervention_type`, and the typed conversion path should use `intervention_record`.

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

## 10. Decision Points For Implementation Agent

Before coding, choose these explicitly.

1. Where exactly does canonical typed payload live inside `InterventionRunSpec`?

Recommended:

```text
split across target/baseline/intervention/readouts/provenance using deterministic conversion helpers
```

Alternative:

```text
store full typed payload under readouts["typed_payload"]
```

Avoid keeping both unless one is explicitly marked derived.

2. Is PI0.5 runtime v0 CLI-first or route-first?

Recommended:

```text
CLI-first, then optional route if dashboard runs in PI0.5 runtime env
```

3. Should dataclasses or Pydantic own validation?

Recommended v0:

```text
dataclasses + explicit from_dict validation, consistent with current workbench style
```

Potential later:

```text
Pydantic boundary models if OpenAPI/generated clients become a priority
```

4. What is the first action basis beyond raw?

Recommended:

```text
gripper, only if action normalization/schema metadata gives an auditable dimension mapping
```

5. What is the first probe-derived target adapter?

Recommended:

```text
probe artifact → TargetSpec(kind="probe_direction")
```

---

## 11. Final Sequencing Summary

The plan is:

```text
1. Define the evidence ontology in docs.
2. Add typed, runtime-free contracts.
3. Persist typed runs through existing workbench records.
4. Index/display runs as LensArtifacts.
5. Add backend read APIs.
6. Add frontend cards/evidence library.
7. Add runtime preflight.
8. Add auditable action basis resolution.
9. Add PI0.5 live direction intervention.
10. Add Intervention Lab UI.
11. Add sweeps/cohort studies.
```

This gets the durable evidence model into the codebase first, then lets every future runtime, SAE feature, transcoder feature, action-basis adapter, or cohort study land into the same stable evidence format.
