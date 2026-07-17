# Interventions

Status: implemented evidence foundation; live scientific execution remains in
progress.

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

The current live hook is an engineering smoke. It uses a synthetic one-hot
direction and records `claim_eligible = false`. It proves replay and hook
plumbing, not a scientific mechanism.

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

`LensArtifact(type="intervention_run")` is the compact browser/index view of
that run. It is not a second source of truth.

## Evidence Rules

- An inspected-only record is not causal evidence.
- A stored-original versus intervention difference is weak if no-op replay
  drifts.
- A single changed action chunk is action-level evidence, not behavioral rescue.
- A random direction with a similar effect weakens the candidate direction.
- A wrong layer, time, or token with a similar effect weakens specificity.
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
```

Do not run live PI0.5 intervention work through the normal `uv run`
environment. See [hardware-run-paths.md](hardware-run-paths.md) for the current
command and replay gate.

## Remaining Work

The active work is tracked in GitHub rather than another planning document:

- [#18: first claim-eligible PI0.5 probe-direction intervention](https://github.com/thajpo/vla-lens/issues/18)
- [#19: live Intervention Lab execution and action comparison](https://github.com/thajpo/vla-lens/issues/19)
- [#20: sweep and cohort execution](https://github.com/thajpo/vla-lens/issues/20)

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
