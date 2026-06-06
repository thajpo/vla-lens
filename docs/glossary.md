# Glossary

Status: active glossary.

Last updated: June 6, 2026.

This glossary defines project vocabulary used by VLA Lens evidence workflows.
The core rule is simple: records are data. Whether a record supports a causal
claim must be interpreted from the saved status, trials, outcomes, controls,
claim labels, and provenance.

## Core Terms

`InterventionRun`:
  The canonical saved evidence record. It contains the exact request-like
  payload, status, trials, outcomes, controls, runtime or preflight status, and
  provenance.

`intervention_record`:
  The v0 workbench `intervention_type` for typed intervention records. This is
  a storage category only. It does not mean the record is causal.

`LensArtifact(type="intervention_run")`:
  The artifact-browser/index/display shell for an `InterventionRun`. It should
  point back to the canonical workbench record and store compact display data
  plus array refs, not a second independent source of truth.

`Evidence Layer v0`:
  Saved, listable, inspectable intervention records that can be opened without
  PI0.5, Torch, LeRobot, LIBERO, GPU dependencies, or model checkpoints.

`Runtime v0`:
  Live PI0.5 direction intervention execution. This is later than saved
  evidence and must be gated by preflight plus the dedicated PI0.5 runtime
  environment.

## Context Terms

`Episode`:
  One robot trajectory or rollout in a dataset.

`Trace`:
  A recorded sequence of model/environment events. A trace can represent an
  original, no-op, control, or intervened execution.

`PolicyCall`:
  One model invocation aligned to episode time. V0 interventions use a selected
  policy call as the main address.

`Recipient`:
  The trace, episode, policy call, or model execution that receives the
  intervention.

`Donor`:
  The source trace, policy call, activation, or tensor used for replacement or
  patching. Direction steering usually has no donor.

## Target Terms

`Site` or `ModelSite`:
  A named internal tensor location from overlay/model metadata, such as a
  hidden-state, attention, KV, or action-head tensor location.

`Target` or `TargetSpec`:
  The specific object inside a site that should be manipulated: a probe
  direction, contrast direction, activation slice, feature, subspace, head,
  edge, or manual selection.

`Runtime hook`:
  The adapter-specific live model boundary used to implement the manipulation.
  The saved run should record both the requested target and the resolved hook
  when runtime execution happens.

`DiscoveryArtifact`:
  A candidate generator such as a probe, contrast direction, attention map,
  action-generation artifact, activation cluster, manual selection, SAE
  feature, transcoder feature, or crosscoder feature. Discovery artifacts are
  not causal evidence by themselves.

## Intervention Terms

`Operator`:
  The operation applied to a target, such as `add_direction`,
  `project_out_direction`, `replace`, `ablate`, `clamp`, `source_patch`, or
  `mean_replace`.

`Schedule`:
  When and where an operator is active: policy call, timestep, generation step,
  token selector, or action horizon.

`Outcome`:
  What changed after the intervention. V0 prioritizes action outcomes:
  original/no-op/intervened action chunks and their deltas.

`Control`:
  A comparison intended to reduce false causal interpretation, such as no-op
  rerun, random direction, wrong layer, wrong time, wrong token, or shuffled
  donor.

`Trial`:
  One inspected or executed attempt: stored original, no-op, main intervention,
  control, failed attempt, or skipped attempt.

`Run`:
  One saved `InterventionRun` over a concrete context, target, operator,
  schedule, outcome request, and set of trials.

`Sweep`:
  Many related runs over one or more axes, such as strength, layer, policy
  call, generation step, token selector, target, donor, or random seed.

`Study`:
  A higher-level collection of runs and sweeps, usually across a cohort, with
  aggregate outcomes and controls.

`MechanismCard`:
  A human-readable claim artifact assembled from multiple evidence records. It
  should not replace the underlying runs, sweeps, controls, and provenance.

## Action Basis Terms

`ActionBasis`:
  The coordinate system used to interpret action vectors. Examples include raw,
  gripper, end-effector xyz delta, rotation, speed, object-relative,
  task-relative, PCA, or learned action feature bases.

`ActionBasis provenance`:
  The saved explanation for how an action basis was resolved. It must include
  `action_schema_ref`, `basis_resolution`, `units`, `sign_convention`,
  `source_dimensions`, and `normalization` when available.

`Raw action basis`:
  The native model/environment action vector. This is the only basis that v0
  can assume when no additional metadata exists.

## Evidence Labels

`inspected_only`:
  The request or context can be inspected, but no live intervention trial was
  executed from this environment.

`ok`:
  The requested saved or runtime operation completed.

`partial`:
  Some evidence exists, but requested runtime support, basis metadata, controls,
  or outputs were unavailable.

`failed`:
  The requested operation could not be completed. The failed record should keep
  useful diagnostics when possible.

`observation`:
  Interesting example, activation, attention, or cluster evidence.

`predictive`:
  Heldout probe or association evidence.

`causal_local`:
  An intervention changed one local output such as one policy-call action
  chunk. This does not imply rollout-level behavior changed.

`action_level`:
  Evidence is about generated actions, not full closed-loop behavior.

`rollout_level`:
  Evidence includes closed-loop rollout or trajectory outcomes.

`causal_cohort`:
  The effect holds across a cohort, not only one local example.

`behavioral`:
  Evidence concerns task behavior, environment outcome, or robot trajectory.

`specific`:
  Controls suggest the effect is not generic model damage.
