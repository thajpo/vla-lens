# Documentation Review And Plan — July 15, 2026

Status: dated review record. Use `current-state.md` and the linked remaining-work
documents for current execution truth.

## Review Scope

This review covered the 34 files under `docs/`, the two root intervention
design documents, and their relevant code, tests, scripts, and Git history.
Operational, architecture, and research-history documents were audited
independently and then reconciled against the current `master` implementation.

The cleanup used one conservative rule:

> Delete a planning or checklist block only when current code and focused tests
> prove the whole block is implemented.

Implemented operational instructions, durable contracts, fixed-revision audit
evidence, negative research results, and partially completed plans were kept.

## What Was Removed

| Area | Completed planning material removed | Implementation evidence |
| --- | --- | --- |
| Repository consolidation | Phase 1–6 execution instructions and obsolete pending-branch cleanup | PRs #7–#10 and #12 merged; `master` is clean; completed branches/worktrees were removed. |
| Probe evidence v1 | Phase 0–7 implementation plan, completed fixture/selector checklist, resolved audit questions | Probe evidence types, adapters, selectors, Dataset/Episode integration, pins, guardrails, and focused tests exist. |
| Model/dataset/simulator generalization | Completed contract, fake-adapter, compliance-test, and capability-manifest phases | Fake adapters, adapter compliance tests, dataset capability summary, and capability-gated frontend paths exist. |
| Intervention evidence foundation | Completed schema, persistence, artifact, saved API/UI, preflight, and action-basis phases; obsolete PR/test/decision checklists | Typed intervention contracts, workbench conversion, artifacts, APIs, Evidence UI, preflight, action-basis, and focused tests exist. |
| Operational backlog | The completed replay-slice sentence and completed capture-container cleanup bullet | Replay-gated PI0.5 wrapper/tests and CUDA/ROCm capture containers exist. |
| Static system audit | Completed replay-seed recommendation and implemented intervention-trial test proposal | Exact replay seed/input coverage and stored/no-op/intervention/control runtime-contract tests exist. |

The consolidation execution record, architecture invariants, and rollback rules
were retained because they are history and safety guidance rather than
unfinished implementation work.

## What Was Corrected

- Quickstart now distinguishes the Vite demo at port 5173 from the Docker
  dashboard at port 8080 and describes the active PI0.5 path as model/simulator
  capture rather than real-robot capture.
- Docker documentation is labeled as an active guide, not a plan.
- CUDA remote capture is labeled implemented but not hardware-smoke validated;
  ROCm remains the recorded hardware validation.
- The architecture workflow uses the implemented `TargetSpec`, preflight,
  PI0.5 runner, `InterventionRun`, and `LensArtifact` objects instead of an
  unimplemented offline `Patcher` API.
- The probe campaign registry records target-lifted as a completed negative
  result, target-parse as blocked on missing rows, and geometry work as a
  completed mostly negative/diagnostic campaign.
- The documentation index now includes the probe-evidence contract, inventory,
  feedback log, and the detailed intervention documents.

## Intentionally Preserved

- Capture, replay, Docker, cloud, and remote-GPU runbooks: implemented commands
  remain necessary instructions.
- Architecture and evidence contracts: implementation makes these more useful,
  not obsolete.
- System-review documents 00–07 and 09: these are fixed-revision evidence;
  only exact recommendations proven complete were removed.
- Probe experiment history and null results: negative evidence constrains future
  research and must not be treated as disposable completed work.
- Partial work: frontend browsing coverage for a non-PI0.5 dataset, a real
  second model/environment, remote object-store reads, live Intervention Lab
  execution, and cohort runners remain explicit.

## Remaining Product And Architecture Plan

### 1. Build The Shared Research Data Spine

1. Add a dataset-level policy-call index with stable identity and uniqueness,
   label-join, and site-coverage tests.
2. Add an exact, reusable `ExampleManifest` whose rows still resolve after
   indexes are rebuilt.
3. Add a conservative `ExperimentRecipe` and public example-building API around
   the current probe workflow.

These three pieces reduce repeated scans and make probe, intervention, and
future method results reconstructable from the same population definition.

### 2. Unify Research Navigation And Evidence

1. Unify dataset, lens/run, episode, policy-call, timestep, site, and contributor
   selection across routes and saved workspaces.
2. Add exact drilldowns from aggregate cells and readouts to source examples.
3. Standardize method-independent evidence lineage, status, controls, and claim
   labels while keeping method-specific metrics.

### 3. Finish The First Claim-Eligible Intervention

1. Resolve a saved probe direction into the replay-gated PI0.5 runtime.
2. Support add-direction, project-out, a matched random control, and at least one
   specificity control.
3. Connect the Intervention Lab to an explicit execution boundary and render
   stored-original/no-op/intervened/control action comparisons.
4. Only then add sweep/cohort execution over the already implemented study and
   aggregation contracts.

The current synthetic action-head hook is a valuable engineering control, but
it is intentionally non-claiming and is not the scientific milestone above.

### 4. Tighten Reproducibility And Validation

1. Persist checkpoint/config/runtime hashes, action execution mapping, and
   requested-to-resolved runtime/site records.
2. Add remaining normal-lane boundary tests for blocked heavy imports,
   incomplete-capture warnings, and artifact/example resolution.
3. Complete a human browser pass on a current probe dataset.
4. Revisit object-local z only with locked PCA/regularization controls; keep
   filtered first-event probes only when their support gates pass.

### 5. Defer New Method Families

SAE, transcoder, crosscoder, attribution, broad cohort comparison, and
generalization surfaces should wait until the shared example/evidence spine and
first controlled intervention are reliable. A real non-PI0.5 model and a real
non-LIBERO source remain the next agnosticism proofs after that foundation.

## Suggested Delivery Order

| Order | Review unit | Why it is separable |
| --- | --- | --- |
| 1 | Policy-call index | Storage/index contract with focused invariants; no product semantics required. |
| 2 | Example manifest | Reconstructability contract that can reuse the index without changing UI. |
| 3 | Experiment recipe | Research API built on the first two contracts. |
| 4 | Unified selection and drilldowns | Frontend/backend navigation vertical slice. |
| 5 | Claim-eligible PI0.5 intervention | Scientific runtime slice with explicit controls and hardware evidence. |
| 6 | Live Lab comparison | Product surface over a proven runtime result. |
| 7 | Sweep/cohort execution | Scale only after one controlled run is trustworthy. |

Each unit should remain its own issue/PR unless implementation proves two units
are inseparable. Hardware evidence belongs only to the intervention-runtime
unit; normal documentation, schema, API, and UI checks stay in the normal repo
environment.
