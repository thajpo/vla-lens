# Interventions

Status: implemented evidence foundation, live probe-direction intervention,
matched counterfactual capture, and resumable donor patch studies. A confirmed
narrow semantic or behavioral intervention remains open.

## What This Part Of VLA Lens Does

An intervention asks whether changing a candidate internal signal changes the
model's output.

The durable flow is:

```text
discovery artifact
-> exact target
-> runtime preflight
-> original, no-op, intervention, and control trials
-> measured action or rollout outcomes
-> saved intervention run
```

Discovery is not causal evidence. A probe or interesting activation only
suggests what to test.

## What Works Now

VLA Lens can:

- turn probe and Episode Lens selections into typed intervention targets;
- preserve model site, layer, token space, source artifact, and selection
  provenance;
- explain through preflight whether a selected policy call is reconstructable;
- save and reopen intervention records without PI0.5 or GPU dependencies;
- index saved runs as normal VLA Lens artifacts;
- resolve raw and named action bases without guessing missing semantics;
- aggregate saved runs into sweep and study records;
- replay a PI0.5 policy call through the dedicated capture environment;
- require repeated no-op replay to match the stored action within explicit
  tolerances before applying a hook;
- save a synthetic action-head intervention and matched random control.
- resolve the accepted RQ-015 linear object-ROI probe back into raw PI0.5 VLM
  hidden space with exact saved scaler/PCA replay;
- add or project out that direction at explicit prefix-token indices, with
  matched-random, wrong-identity, and wrong-ROI controls.
- load a compatible recipient/donor trace pair once, capture donor VLM layers
  into an in-memory cache, and patch explicit donor prefix tokens into the
  recipient under the recipient's exact saved noise;
- run recipient-self, donor-self, shuffled-donor, norm-matched random, and
  wrong-region source-patch controls;
- save the donor action alongside recipient, patched, and control action chunks
  and report whether the patch moved the action toward the donor.
- expand declared pair, layer, and token-region axes into stable trial IDs;
- execute a study with one model load, one multi-layer donor cache fill per
  pair, and one replay gate per recipient;
- resume without repeating completed trials, while retaining failures for
  explicit retry;
- keep full action chunks, compact tables, decisions, hashes, and the exact
  plan permanently while keeping large donor hidden states only in memory.
- address PI0.5 action-expert hidden states by layer, denoising step, and action
  position, capturing several requested expert layers in one donor generation;
- build full or sliced expert-action studies without hand-editing JSON. See
  [pi05-action-stream-patching.md](pi05-action-stream-patching.md).

The synthetic one-hot action-head hook remains an engineering smoke and records
`claim_eligible = false`. The artifact-derived layer-8 hook can become eligible
for a local action-level comparison only after exact replay and all required
controls pass. Eligibility says the method ran correctly; it is not a positive
scientific verdict.

## The Main Objects

`TargetSpec` says exactly what internal object should change. Artifact-derived
targets retain their source artifact.

`InterventionRequest` combines the selected context, target, operator,
schedule, requested outcome, and controls.

`RuntimePreflightResult` records whether execution is possible and why. A
viewable dataset is not automatically replayable.

`InterventionTrial` represents one original, no-op, intervention, control, or
failed attempt.

`InterventionRun` is the canonical saved evidence. It contains the executed
request, trials, outcomes, runtime resolution, provenance, warnings, and claim
labels.

`CounterfactualPairManifest`, `PatchTrialManifest`, and `PatchStudyArtifact`
record the larger recipient/donor experiment. They preserve the scene recipe,
shared-noise identity, token mapping and hashes, named action axes, decisions,
and enough lineage to rebuild any disposable activation cache.

`LensArtifact(type="intervention_run")` is the compact browser/index view of
that run. It is not a second source of truth.

## Evidence Rules

- An inspected-only record is not causal evidence.
- A stored-original versus intervention difference is weak if no-op replay
  drifts.
- A single changed action chunk is action-level evidence, not behavioral rescue.
- A random direction with a similar effect weakens the candidate direction.
- A wrong layer, time, or token with a similar effect weakens specificity.
- Merely completing a control does not earn a `specific` label. The intended
  patch must beat the measured control by the declared margin.
- Rollout and cohort claims require rollout and cohort evidence.
- Missing controls or runtime identity must remain visible in the saved record.

Large arrays stay in dataset storage and are referenced from the run. Saved JSON
records keep exact identities, hashes, metrics, and array references rather than
embedding tensors.

## Runtime Boundary

Normal development, saved-record APIs, preflight, aggregation, and dashboard
viewing run in the normal repo environment.

PI0.5 replay and intervention execution run only through the dedicated capture
environment:

```bash
scripts/pi05_intervene.sh --backend rocm ...
scripts/pi05_patch_study.sh --backend rocm DATASET_ROOT \
  --study configs/interventions/STUDY.json

# Execute only after inspection succeeds:
scripts/pi05_patch_study.sh --backend rocm DATASET_ROOT \
  --study configs/interventions/STUDY.json --run-study \
  --max-noop-l2 0.02 --max-noop-max-abs 0.003
```

Do not run live PI0.5 intervention work through the normal `uv run`
environment. See [hardware-run-paths.md](hardware-run-paths.md) for the current
command and replay gate.

## Remaining Work

The active work is tracked in GitHub rather than another planning document:

- [#20: sweep and cohort execution](https://github.com/thajpo/vla-lens/issues/20)
- [#36: patch PI0.5 prefix key/value cache](https://github.com/thajpo/vla-lens/issues/36)

The active scientific order is governed by the
[controlled scene-to-behavior campaign](autonomous-research-campaigns.md).
Exact prefix-cache patching should follow a factor-specific physical behavior
result rather than extend the old two-object pilot by default.

There is no active temporary implementation spec. When one of these issues is
selected, its issue body becomes the plan by default.

## Code Map

- `src/vla_lens/interventions/`: runtime-free specs, results, preflight,
  artifacts, action bases, families, and sweep/study logic.
- `src/vla_lens/pi05/`: PI0.5 replay, runtime resolution, execution, and saved
  trial assembly.
- `src/vla_lens/server/`: saved intervention and preflight APIs.
- `frontend/src/components/interventions/`: Intervention Lab and saved evidence
  views.
- `tests/test_intervention_*.py` and `tests/test_pi05_intervention_*.py`:
  contract coverage that does not require live hardware unless explicitly
  marked.
- `configs/interventions/rq018_caddy_identity_project_out.json`: first prepared
  artifact-derived request; execute only through `scripts/pi05_intervene.sh`.
- `configs/interventions/rq020_source_patch_template.json`: replace every
  placeholder and token region from a validated counterfactual pair before
  executing through the same dedicated wrapper.
- `configs/interventions/rq020_existing_pair_layer8_smoke.json`: a real
  moved-distractor pair for runtime validation only. It is not the cleaner
  target/distractor pose-exchange experiment planned by RQ-019.
- `configs/interventions/rq020_existing_pair_runner_smoke.json`: a bounded
  two-layer job for validating the resumable study runner. It is also not the
  RQ-020 pose-exchange pilot.
