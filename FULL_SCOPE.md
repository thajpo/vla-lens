# OpenVLA Steering Project: Full Research Scope

## Purpose

Build a minimal but extensible research codebase for testing whether a pretrained autoregressive vision-language-action policy can be feature-steered so that, in a scene with multiple candidate objects, the policy changes which object it selects.

The first motivating example is a simple semantic selection task such as choosing a red cube versus a blue cube in simulation.

This project is not about training a new robot policy from scratch. It is about:

- capturing internal activations from an existing policy
- identifying simple semantic or quasi-semantic directions in representation space
- intervening on hidden states during inference
- measuring whether object choice changes under matched conditions

## Research Position

The project should treat autoregressive VLA policies as the first testbed, not because diffusion or flow-matching policies are inherently uninterpretable, but because autoregressive action-token prediction gives a cleaner first-pass causal story for semantic steering.

The longer-term research arc is:

1. Show the phenomenon in a clean autoregressive setting.
2. Establish reproducible intervention and evaluation tooling.
3. Test whether similar steering effects survive in more complex and more current policy architectures.

## System Constraints

This repository is intended for a single-user Linux workstation with an AMD GPU.

Assume:

- Radeon 7900 XTX
- 24 GB VRAM
- ROCm-compatible PyTorch stack

Use pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2

The implementation must avoid CUDA-only assumptions where possible. The model backend should be swappable so the rest of the experiment harness remains usable if a particular VLA stack proves difficult on ROCm.

## Core Technical Goal

The core experiment is a matched-scene intervention:

1. Create a deterministic scene with multiple candidate objects.
2. Run the policy normally and log which object it chooses.
3. Re-run the exact same scene with a hidden-state intervention at a chosen module.
4. Measure whether object selection changes and whether the intervention causes side effects such as failure or unstable behavior.

## Full-Scope Stack

Keep the stack conservative and easy to debug.

Required libraries and tools:

- Python
- PyTorch
- Hugging Face Transformers where needed by the selected model backend
- Hydra for configuration
- scikit-learn for linear probes
- pandas for tabular analysis
- pyarrow / Parquet for structured logs
- matplotlib for offline plotting

Optional later additions:

- video logging for rollout comparison
- richer analysis notebooks
- alternate model backends
- more advanced interpretability tooling

Explicitly out of scope for phase one:

- SAE training
- TransformerLens-style deep framework integration
- distributed training
- database-backed experiment tracking
- NVIDIA-specific assumptions

## Repository Shape

The repository should remain intentionally small:

```text
src/
  model/
  env/
  interp/
  utils/
configs/
scripts/
artifacts/
notebooks/   # optional
```

Do not split this further unless the code clearly earns it.

## Major Components

### 1. Model Layer

The model layer should provide a thin wrapper around a pretrained autoregressive VLA backend.

Responsibilities:

- load a checkpoint
- run inference from observation + instruction
- expose hookable internal module names
- support a normal forward pass
- support an intervention forward pass with additive steering

Design rule:

- model-specific complexity should live here and not leak into the rest of the codebase

The initial target can be OpenVLA, but the rest of the repository should not depend tightly on OpenVLA-specific internals.

### 2. Environment Layer

The environment layer should provide a very small, controlled object-selection task in simulation.

Responsibilities:

- reset
- step
- render
- expose deterministic seeding
- expose fixed-scene debug mode
- expose scene metadata

Scene metadata should include, where available:

- object IDs
- colors
- positions
- candidate target set
- selected or grasped object

The environment should prioritize clean target-choice isolation over long-horizon task richness.

### 3. Interpretation Layer

This is the core layer of the project.

It should include:

- a hook manager for activation capture
- additive steering at a named module
- activation caching
- probe fitting utilities
- steering-vector derivation
- matched-scene intervention evaluation

The first steering mode is additive only:

- intercept activation tensor at module `M`
- add `alpha * v`
- continue the forward pass

The code must validate tensor shapes and fail loudly on mismatches.

### 4. Utilities Layer

This layer should keep the project operationally clean.

Responsibilities:

- reproducible seeding
- artifact path management
- log record definitions
- config serialization
- common dataclasses or typed schemas

## Experiment Flows

The full research harness should support these flows.

### Baseline Rollouts

Run many seeded scenes with no intervention and log:

- scene metadata
- instruction
- chosen object
- task success or failure

### Activation Collection

Capture activations from one or more named modules across repeated rollouts and save them with labels needed for downstream probing.

### Probe Training

Fit simple linear probes offline using cached activations.

Initial label tasks may include:

- target color
- target identity
- chosen object identity
- position-related labels if the environment makes those easy to recover

### Steering-Vector Derivation

Support at least two simple steering direction sources:

- difference of means between two conditions
- linear probe coefficients

### Matched-Scene Steering

Re-run the same scene twice:

- baseline
- intervention

Sweep over:

- hook site
- steering vector
- steering strength `alpha`

### Analysis

Read logs and generate simple plots and summaries such as:

- baseline object-choice distribution across seeds
- probe accuracy by module or layer
- steering strength versus probability of selecting object A
- intervention flip rate
- intervention side-effect rate

## Logging Requirements

Logging should be explicit and reconstruction-friendly.

### Rollout Log Record

Each rollout record should include at minimum:

- run ID
- experiment ID
- timestamp
- git commit hash if available
- model name or checkpoint
- environment name
- seed
- instruction
- object metadata
- baseline or intervention flag
- hook site
- steering alpha
- steering vector ID
- chosen object
- success flag
- paths to saved artifacts such as videos or tensors

Preferred storage:

- Parquet for tabular metadata

### Activation Cache Record

Each cached activation entry should include:

- rollout ID
- module name
- tensor shape
- dtype
- label metadata
- file path to the saved tensor, or the tensor if storage remains small

Preferred storage:

- tensor files plus a small Parquet index

## Reproducibility Requirements

The code should be strict about reproducibility.

Required behaviors:

- seed Python, NumPy, and PyTorch
- record all seeds in logs
- save resolved Hydra configs with every run
- make artifact paths deterministic
- fail loudly if a requested module does not exist
- validate steering-vector compatibility before runtime intervention

## Debuggability Requirements

The code should also be strict about debuggability.

Required behaviors:

- hook manager dry-run mode for shape inspection only
- no-op intervention mode with `alpha = 0`
- fixed-scene environment mode
- clear logging for module names, tensor shapes, and artifact paths

## Testing Expectations

Keep testing lightweight but meaningful.

Expected early tests:

- hook registration and activation capture
- steering-vector shape validation
- rollout log serialization
- one tiny smoke test if practical

## Documentation Expectations

The repository should include:

- a top-level README that explains the experiment flow
- a short developer note describing how to add a new model backend
- a short developer note describing how to add a new hook site or probe label

## Full Research Arc

The full research task, beyond the initial MVP, is to move from infrastructure validation to actual scientific claims:

1. Reliable matched-scene intervention harness.
2. Evidence that relevant labels are linearly decodable from at least one internal site.
3. Evidence that additive steering changes object choice in some controlled settings.
4. Measurement of collateral side effects and robustness across seeds.
5. Comparison across multiple hook sites and steering-direction sources.
6. Eventual comparison against more complex policy architectures if the initial approach succeeds.

## Success Criteria For Full Scope

The full-scope project is successful if it can support a credible causal experiment of the following form:

- same scene
- same instruction
- same model
- same seed
- different hidden-state intervention
- different object choice
- clear logging of whether success and stability were preserved

That is the standard the repo should be designed to support.
