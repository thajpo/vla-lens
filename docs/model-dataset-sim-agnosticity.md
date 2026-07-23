# Model, Dataset, And Simulator Agnosticism

Status: active architecture rule.

## Goal

VLA Lens should support multiple VLA models and robot environments without
turning PI0.5 or LIBERO names into the core data model.

This does not mean every model must expose the same internals. It means generic
code depends on stable dataset contracts and declared capabilities, while
model-specific execution stays behind adapters.

## Stable Boundary

LeRobotDataset v3 owns robot data:

- episodes and frames;
- observations, state, and actions;
- timestamps, tasks, and camera media;
- dataset metadata and statistics.

The VLA Lens overlay owns interpretability data:

- policy-call alignment;
- model descriptions and model sites;
- tokens, activations, attention, and generation traces;
- probes, interventions, evidence, and provenance;
- links back to the canonical robot data.

The dashboard reads this combined dataset. It should not need the original model
runtime merely to inspect saved evidence.

## Generic Core

Generic code should understand concepts such as:

- dataset, episode, frame, and policy call;
- model site and tensor geometry;
- array reference and artifact;
- capability and unavailable reason;
- selection, evidence, target, trial, outcome, and provenance.

It should not assume a PI0.5 module path, LIBERO task type, fixed token layout,
or a particular action representation.

## Adapter Responsibilities

A model adapter owns model loading, observation preprocessing, action
postprocessing, runtime hooks, and model-specific site resolution.

An environment or importer owns reset/step behavior, task metadata, observations,
actions, timestamps, and outcome labels.

A capture profile declares which optional internals are saved. Missing
capabilities are valid and must produce clear unavailable states rather than
broken panels.

## Capability-Driven UI

The dataset API reports available capabilities derived from the saved robot
data and overlay. The frontend uses those capabilities to decide whether to
query or show model sites, policy calls, probes, token spaces, attention,
image-token maps, and action-generation views.

Generic episode browsing must still work when no model internals exist. Rich
PI0.5 pipeline views remain useful optional specializations.

## Current Proof

The repo includes tiny fake dataset, environment, and model adapters. Normal
tests write their output through the generic LeRobot v3 plus overlay path, open
it through `TraceDataset`, summarize non-PI0.5 model sites, and verify capability
gating without importing PI0.5, Torch, LeRobot runtime code, LIBERO, or GPU
dependencies.

This proves the storage and server boundary. It does not yet prove a complete
frontend workflow over a real second model or environment.

## Remaining Expansion

Future integrations should add:

- frontend end-to-end coverage over the synthetic non-PI0.5 dataset;
- one real second model adapter, likely OpenVLA;
- one real non-LIBERO environment or importer.

These are deferred directions, not active implementation specs. Create a GitHub
issue when one is selected for work.

## Rules For New Code

- Put robot-data facts in the LeRobot-compatible layer.
- Put cross-model interpretability contracts in the generic VLA Lens layer.
- Put model or environment execution behind an adapter.
- Isolate heavyweight runtime dependencies from the normal development and
  dashboard environment.
- Declare optional capabilities instead of pretending every model supports
  every visualization.
- Do not create a parallel dataset format for each model family.
