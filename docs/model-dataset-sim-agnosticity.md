# Model, Dataset, And Simulator Agnosticism

Status: active, living

This document defines the goal for making VLA Lens agnostic to model family,
robot dataset source, and robot simulator while preserving rich model-specific
interpretability views.

## Goal

VLA Lens should be usable with more than one VLA model, more than one robotics
dataset source, and more than one robot environment. PI0.5 + LIBERO/robosuite is
the first concrete implementation, not the permanent shape of the whole stack.

The target architecture is:

```text
any supported robot data source
  -> LeRobot v3 robot layer
  -> optional VLA Lens interpretability overlay
  -> dashboard, probes, reports, and artifacts
```

For live capture, the target architecture is:

```text
EnvironmentAdapter + ModelCaptureAdapter
  -> CaptureRunner
  -> LeRobot v3 robot data + vla_lens/ overlay
```

The LeRobot v3 layer owns robot episode semantics: episodes, frames, timestamps,
tasks, observations, actions, and camera media. The VLA Lens overlay owns
interpretability semantics: policy calls, model sites, activations, token
spaces, attention maps, action-generation traces, probes, and derived artifacts.

## Why This Matters

A researcher should not need to use PI0.5 or LIBERO to benefit from VLA Lens.
They should be able to point the dashboard at a LeRobot v3 dataset and inspect
episodes immediately. If they also capture model internals, the same dataset
root should gain richer interpretability views through the `vla_lens/` overlay.

This also makes the repo stronger as a software artifact:

- The storage contract is explicit instead of hidden inside one capture script.
- New models can be added as adapters instead of forks of the whole stack.
- New simulators can be added as environment adapters instead of new dataset
  formats.
- The frontend can render capabilities declared by a dataset rather than
  guessing from PI0.5-specific names.

## What Agnostic Means

"Agnostic" does not mean every model or simulator works automatically. It means
the core repo has stable boundaries where new support can be added without
rewriting storage, validation, analysis, or the dashboard.

The intended boundaries are:

```text
DatasetEpisodeAdapter
  converts existing robot logs/datasets into episode records

EnvironmentAdapter
  describes or controls live robot/simulator environments

ModelCaptureAdapter
  loads a model, declares hookable sites, captures internals, and emits overlay data

CaptureRunner
  orchestrates env reset/step, model calls, alignment, and writing

Dashboard capability manifest
  tells the frontend which views are valid for the current dataset
```

PI0.5-specific concepts should remain available, but they should be optional
capabilities rather than assumptions in the core path.

## Capability Model

The frontend should not need to understand every VLA architecture. The backend
should expose a dataset/model capability manifest describing what is present.

Examples:

```text
robot_episodes
cameras
policy_calls
model_sites
token_spaces
image_token_maps
attention_maps
action_chunks
action_generation
architecture_graph
probe_artifacts
intervention_artifacts
```

The dashboard should show generic panels for generic capabilities and specialized
panels only when the dataset declares the required capability.

For example:

```text
PI0.5:
  action_generation: yes
  vlm_expert_pipeline: yes
  prefix_to_expert_attention: yes

OpenVLA:
  transformer_layers: yes
  image_tokens: likely
  language_tokens: likely
  action_generation: model-dependent

LeRobot-only dataset:
  robot_episodes: yes
  cameras: maybe
  model_sites: no
  probes: no
```

## Frontend Direction

The frontend is the riskiest part of this transition because the richest current
views assume PI0.5 concepts such as VLM prefix tokens, expert layers, action
denoising, generation steps, and PI0.5 token spaces.

The desired frontend model is:

```text
Episode browser:
  generic LeRobot view

Activation site browser:
  generic model_sites table and tensor slices

Token-space browser:
  generic token streams, image patches, language tokens, action tokens

Attention/attribution panels:
  shown only when token-space and attention/score metadata exist

Action trajectory panels:
  shown for action chunks and generation traces when present

Architecture graph:
  generated from adapter metadata, not hardcoded PI0.5 names

Model-specific panels:
  optional specializations registered by capability
```

The dashboard should degrade gracefully:

```text
LeRobot only:
  episodes, frames, actions, state, metadata

LeRobot + generic activations:
  plus model sites, tensor slices, probes

LeRobot + rich model overlay:
  plus tokens, attention, action-generation, architecture, interventions
```

## Phases

### Phase 1: Contract And Documentation

Define the agnostic target clearly.

Acceptance criteria:

- This document exists and is linked from the docs index.
- The existing adapter protocols are treated as the direction of travel.
- PI0.5 is documented as the first implementation, not the core assumption.
- New work can point to this document when deciding whether code belongs in
  core, PI0.5, frontend generic UI, or a future adapter.

### Phase 2: Adapter Compliance Tests

Prove the core can run without PI0.5.

Acceptance criteria:

- Add tiny fake dataset, environment, and model adapters.
- Add tests that write a minimal LeRobot v3 + `vla_lens/` overlay through the
  generic path.
- Add tests that the dashboard/server can summarize the fake model's sites
  without `pi05.*` names.
- Keep PI0.5 capture tests separate from normal `uv run pytest`.

### Phase 3: Capability Manifest

Make available views explicit.

Acceptance criteria:

- The backend exposes a capability manifest for each opened dataset.
- Capabilities are derived from LeRobot metadata, overlay tables, arrays, and
  model-site metadata.
- Frontend panels key off capabilities instead of PI0.5 name prefixes wherever
  possible.

### Phase 4: Frontend Generalization

Separate generic views from PI0.5-specific views.

Acceptance criteria:

- Generic LeRobot episode browsing works without model internals.
- Generic model-site browsing works for a non-PI0.5 fake adapter.
- PI0.5 pipeline views still work, but are clearly optional specializations.
- Empty/missing capabilities produce clear empty states rather than broken
  panels.

### Phase 5: First Non-PI0.5 Model

Add one real non-PI0.5 model adapter, likely OpenVLA.

Acceptance criteria:

- OpenVLA can emit policy calls, model metadata, model sites, and activations
  into the existing overlay format.
- The dashboard can inspect the resulting dataset without PI0.5-specific code
  paths being required.
- OpenVLA-specific capture dependencies remain isolated from the normal
  dashboard/dev environment.

### Phase 6: First Non-LIBERO Environment

Add one real non-LIBERO environment source or importer.

Acceptance criteria:

- A second simulator or dataset family can produce LeRobot v3-compatible robot
  data.
- The capture/import path preserves actions, state, camera frames, timestamps,
  task metadata, and success/outcome labels when available.
- The same dashboard path opens the output.

## Non-Goals

- Do not make every model share the same interpretability views.
- Do not flatten PI0.5-specific research affordances into a lowest common
  denominator UI.
- Do not add heavyweight model or simulator dependencies to the normal
  dashboard/test environment.
- Do not create parallel dataset formats for each model family.
- Do not preserve old standalone `.vlatrace` behavior as a primary future path.

## Design Rule

Core code should depend on declared contracts and capabilities. Model-specific
code should live behind adapters or optional capability-specific helpers.

When adding a new feature, ask:

```text
Is this true for all LeRobot datasets?
  Put it in the generic dataset/dashboard layer.

Is this true for all model overlays?
  Put it in the generic overlay/model-site layer.

Is this true for one model family?
  Put it in that model adapter or optional specialization.

Is this true for one simulator?
  Put it in that environment adapter or importer.
```

That boundary is what lets VLA Lens become broadly useful without losing the
depth that makes model-specific interpretability valuable.
