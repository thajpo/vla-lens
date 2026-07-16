# VLA Lens Intervention Evidence Layer — Implementation Spec v0.2

Status: implemented evidence contract with remaining runtime milestones.

Last updated: 2026-07-15.
Primary goal: make the VLA Lens workflow executable, auditable, and UI-addressable.

This spec consolidates the current intervention/evidence design direction into an implementation-oriented contract. It is intentionally narrower than a full interpretability taxonomy. The first useful layer should let a researcher take an existing episode and an existing discovery artifact, convert the artifact into an intervention target, run or record a counterfactual, measure action/rollout change, and save a lossless evidence record.

---

## 0. Executive Decisions

1. **The core product object is `InterventionRun`.**
   - It is the exact executed spec plus outputs, metrics, runtime resolution, controls, and provenance.
   - It should be inspectable, rerunnable when runtime support exists, aggregatable into sweeps/studies, and displayable as an Intervention Card.

2. **Do not make `InterventionPlan` a required persisted object in v0.**
   - A live API may accept an `InterventionRequest`.
   - The saved `InterventionRun` must include the full request-like payload that was executed.
   - If a user configures but does not run an intervention, that is UI state or a draft, not a core evidence artifact.

3. **Discovery artifacts are candidate generators, not causal evidence.**
   - Probe, SAE, transcoder, attention map, cluster, contrast direction, or manual selection all produce candidates.
   - They become useful for this layer only when converted into a `TargetSpec`.

4. **Interventions are model-agnostic at the spec layer and adapter-specific at runtime.**
   - The core spec describes target, operator, schedule, outcome, controls, and provenance.
   - A model adapter resolves that spec into actual hooks.
   - PI0.5 is the first concrete implementation, not the permanent ontology.

5. **The privileged first outcome is `ActionOutcome`.**
   - VLA Lens should support VLM-style token/image outcomes later or when actions are unavailable.
   - The differentiator is: internal intervention → action delta → optionally rollout delta.

6. **Runtime support is capability-gated.**
   - A dataset can be viewable even when live rerun is unavailable.
   - A selected policy call can be inspectable even when it is not reconstructable.
   - Preflight must explain why a requested intervention can or cannot run.

7. **V0 is `Probe Direction Intervention`.**
   - Input: probe artifact, selected episode, selected policy call, selected model site, strength/sweep, action basis.
   - Operators: `add_direction`, `project_out_direction`.
   - Outcomes: stored original, no-op rerun when available, intervened action chunk, deltas.
   - Controls: no-op, random direction, wrong layer if cheap.

---

## 1. Product Loop

The intended loop is:

```text
Open episode or cohort
  → inspect discovery artifact
  → convert candidate to TargetSpec
  → choose operator and schedule
  → run fast policy-call counterfactual if supported
  → compare stored original, no-op, and intervened outputs
  → save InterventionRun
  → optionally scale to sweep, controls, or cohort
```

Researcher-facing UI language should be simple:

```text
Intervene with this signal
Send to Intervention
Turn this up/down
Run counterfactual
Save evidence
```

Backend language should stay precise:

```text
ContextSpec
TargetSpec
InterventionSpec
ScheduleSpec
OutcomeSpec
ControlSpec
RuntimePreflightResult
InterventionTrial
InterventionRun
```

The UI should not force researchers to think in every spec object name, but the saved artifact must be lossless enough that later code can inspect, rerun, aggregate, or reject overclaims without guessing what happened.

---

## 2. Conceptual Layers

```text
Context Layer
  Where did behavior happen?

Discovery Layer
  What internal signal might matter?

Target Layer
  What exact internal object can be manipulated?

Intervention Layer
  What operation is applied?

Outcome Layer
  What changed?

Control Layer
  Was the effect specific?

Evidence Layer
  What claim can be responsibly made?
```

Lifecycle:

```text
Episode / PolicyCall / ModelSite
        ↓
DiscoveryArtifact: probe, contrast direction, SAE feature, cluster, attention map
        ↓
TargetSpec
        ↓
InterventionRequest + RuntimePreflight
        ↓
InterventionTrial(s)
        ↓
InterventionRun
        ↓
LensArtifact / workbench intervention record
        ↓
InterventionSweep / InterventionStudy / MechanismCard
```

---

## 3. Glossary

### 3.1 Context Terms

| Term | Definition | Implementation note |
|---|---|---|
| `DatasetRoot` | Concrete opened dataset root containing LeRobot data plus optional `vla_lens/` overlay. | Dashboard opens one root or nested batch output. |
| `Episode` | One recorded robot trajectory or rollout. | Data-oriented term. |
| `Rollout` | One policy/environment execution. | Use when emphasizing execution. Often synonymous with episode. |
| `Trace` | A recorded sequence of model/environment events. | Useful when there are clean/corrupt/intervened variants. |
| `Frame` | One visual observation timestep. | Camera frame or multi-camera observation. |
| `Timestep` | Temporal index in an episode. | May differ from policy call index. |
| `PolicyCall` | One model invocation aligned to episode time, typically producing one action or action chunk. | Primary address for v0 interventions. |
| `Observation` | Model input at a policy call: images, instruction, robot state, preprocessing/tokenization metadata. | Must be reconstructable for live rerun. |
| `Instruction` / `Task` | Language goal or task metadata. | Used for filtering, context, and cards. |
| `Action` | Robot control output at one timestep. | Could be raw vector, tokenized action, delta pose, joint command, gripper command. |
| `ActionChunk` | Sequence of future actions generated by one policy call. | First-class for PI0.5-like models. |
| `GeneratedAction` | Action produced by the model. | Compare original/no-op/intervened generated chunks. |
| `ExecutedAction` | Action actually sent to environment/robot. | May differ due to clipping/wrappers. |
| `Outcome` | Any measured result after intervention. | Action, rollout, token/logit, scalar metric. |
| `Cohort` | Filtered set of episodes, timesteps, policy calls, units, or examples. | Used for scaling from local evidence to cohort evidence. |

### 3.2 Model-Internal Terms

| Term | Definition | Implementation note |
|---|---|---|
| `ModelSite` | Named internal tensor location in the overlay/model metadata. | Example: `pi05.expert.layers.12.hidden_tokens`. |
| `ModulePath` | Adapter-specific live model path. | Runtime detail; may differ from overlay site name. |
| `Layer` | Layer index in a module family. | Often used in sweeps and wrong-layer controls. |
| `TensorType` | Kind of tensor captured or hooked. | Hidden state, attention score, KV cache, MLP activation, flow state. |
| `TokenSpace` | Semantic token axis. | `language`, `image`, `action`, `expert`, or model-specific. |
| `TokenSelector` | Rule selecting tokens within a token space. | all action tokens, image region tokens, token index, horizon slice. |
| `GenerationStep` | Index in iterative action generation. | Flow/diffusion/denoising/action refinement step. |
| `FlowStep` | PI0.5-style generation step. | Adapter-specific specialization of generation step. |
| `ActivationTensor` | Actual tensor value at a model site during a policy call. | Stored or live. |
| `Direction` | Vector in activation space. | Probe coefficient, contrast direction, SAE decoder vector. |
| `Subspace` | Multi-vector basis. | PCA basis, probe subspace, SAE feature group. |
| `Feature` | Scalar latent or interpretable unit. | SAE feature, transcoder feature, neuron, probe score. |
| `Head` | Attention head. | Possible future target. |
| `Edge` | Relationship/path between internal components. | Token-to-token attention edge, graph/circuit edge. |

Key distinction:

```text
ModelSite = where a tensor lives.
TargetSpec = the selected object inside that site that will be manipulated.
RuntimeHook = the adapter-specific live boundary used to implement the manipulation.
```

### 3.3 Discovery Terms

| Term | Definition | Role |
|---|---|---|
| `DiscoveryArtifact` | Anything that suggests a candidate internal signal. | Candidate generator. |
| `Probe` | Supervised readout from activations to behavior/label/outcome. | Produces direction, score, coefficients, performance. |
| `ProbeDirection` | Direction from a linear probe. | Excellent v0 target. |
| `MeanDifferenceDirection` | Difference between average activations for two groups. | Simple contrast target: close vs open, success vs failure. |
| `ActionGenerationArtifact` | Summary of how action chunks form over generation steps. | Localizes action formation. |
| `AttentionOrAttributionMap` | Localization clue over tokens, patches, heads, or sites. | Not causal by itself. |
| `ActivationCluster` | Group of similar activations or robot moments. | Cohort/candidate discovery. |
| `ManualSelection` | Researcher-chosen site/token/layer/call/feature. | Debug target. |
| `SAEFeature` | Sparse latent from an SAE. | Future candidate target. |
| `TranscoderFeature` | Feature/pathway in a learned transformation decomposition. | Future mechanism/path target. |
| `CrosscoderFeature` | Shared/differential feature across models/sites/modalities. | Future comparative target. |
| `CandidateSignal` | Generic name for anything discovery produces. | Must become `TargetSpec` to intervene. |

Rule:

```text
Discovery artifacts create hypotheses.
Interventions test hypotheses.
Evidence cards communicate what happened.
```

### 3.4 Target Terms

| Term | Definition | Notes |
|---|---|---|
| `TargetSpec` | Normalized description of the thing to manipulate. | Bridge from discovery artifacts to runtime. |
| `TargetKind` | Kind of target. | `probe_direction`, `contrast_direction`, `activation_slice`, `feature`, `subspace`, `head`, `edge`, `manual`. |
| `SourceArtifact` | Artifact that produced the target. | Probe artifact, SAE artifact, attention artifact, etc. |
| `Recipient` | Trace/policy call/model execution receiving the intervention. | Always exists for an executed intervention. |
| `Donor` | Source trace/policy call/activation copied from during patching/replacement. | Only exists for source patching or replacement. Not used for direction steering. |
| `CleanTrace` | Good/reference trace in clean/corrupt analyses. | Can be donor or baseline depending on experiment. |
| `CorruptTrace` | Bad/perturbed/failed contrast trace. | Can be recipient. |
| `PatchSource` | General term for donor activation. | Avoid using donor for non-copy interventions. |
| `PatchDestination` | General term for recipient activation location. | Usually target in recipient. |
| `Selector` | Rule for selecting axes or slices. | Token, layer, horizon, generation step, object region. |
| `Reduction` | Rule collapsing tensor axes to a vector/score. | mean, none, first token, flat, horizon slice. |

Donor/recipient examples:

```text
Activation patching:
  donor = success episode / clean trace
  recipient = failure episode / corrupt trace

Direction steering:
  source_artifact = gripper-close probe
  recipient = selected policy call
  donor = none

SAE feature boosting:
  source_artifact = SAE feature artifact
  recipient = selected policy call
  donor = none
```

### 3.5 Intervention Terms

| Term | Definition | Typical question |
|---|---|---|
| `InterventionSpec` | Operation, target, schedule, parameters, and controls. | What are we doing? |
| `Operator` | Type of operation. | add, project out, replace, ablate, clamp, patch. |
| `AddDirection` | Add `alpha * direction` to activation. | Can this increase/induce behavior? |
| `ProjectOutDirection` | Remove projection onto direction. | Is this direction necessary? |
| `ScaleDirection` | Scale component along direction. | How does behavior vary with strength? |
| `Ablation` | Remove/zero a signal. | Is signal necessary? Risk: off-manifold damage. |
| `MeanReplacement` | Replace activation with baseline/mean. | What if site carries neutral/default information? |
| `SourcePatch` | Copy donor activation into recipient. | Does internal state transfer behavior? |
| `KVReplacement` | Replace attention KV cache/context. | Does prefix/context representation matter? |
| `HiddenReplacement` | Replace hidden state at site. | Does representation at this layer matter? |
| `FlowStateReplacement` | Replace state during action generation/flow. | Does generation-step state matter? |
| `FeatureClamp` | Set feature activation to chosen value. | Does sparse feature control behavior? |
| `FeatureBoost` | Increase feature activation. | Can feature induce behavior? |
| `AttentionPatch` | Modify attention weights/patterns. | Does token relation matter? |
| `PathPatch` | Modify pathway between sites. | Does computational path matter? |
| `ActionOverride` | Directly replace output action. | Useful baseline, not model-internal causal evidence. |

V0 operators:

```text
add_direction
project_out_direction
noop_rerun
random_direction_control
wrong_layer_control, if cheap
```

### 3.6 Schedule Terms

| Term | Definition |
|---|---|
| `InterventionSchedule` | When and where the intervention is active. |
| `CallSchedule` | Policy call index or range. |
| `TimestepSchedule` | Environment timestep range. |
| `GenerationStepSchedule` | Action-generation/flow/denoising steps. |
| `TokenSchedule` | Token selector active during intervention. |
| `ActionHorizonSchedule` | Future action positions affected or measured. |
| `ConditionalSchedule` | Apply only when condition holds, such as probe score above threshold. |
| `StrengthSweep` | Repeat across strengths. |
| `LayerSweep` | Repeat across layers. |
| `TimeSweep` | Repeat across calls/timesteps. |

Why schedules matter:

```text
Robotics is temporal. Turning up a signal everywhere may be meaningless or destructive.
Often the question is: does this signal matter at this policy call, this phase, this generation step, and this token space?
```

V0 may only apply to one selected policy call, but schemas should leave room for generation-step and horizon schedules.

### 3.7 Outcome Terms

| Term | Definition |
|---|---|
| `OutcomeSpec` | What to measure after intervention. |
| `ActionOutcome` | Difference between original/no-op/intervened action chunks. |
| `RolloutOutcome` | Difference in environment behavior: success, contact, collision, final distance, trajectory, video. |
| `TokenOutcome` | Text/logit/image-token result for VLM-style tests. |
| `ProbeOutcome` | Before/after probe score. |
| `MetricOutcome` | Arbitrary scalar metric. |
| `CounterfactualOutcome` | Result under intervention compared to original/no-op. |
| `ActionDelta` | `intervened_action - baseline_action`. Prefer no-op baseline when available. |
| `RolloutDelta` | Difference in rollout metrics. |
| `NoopDelta` | Difference between stored original and regenerated no-op output. |
| `SideEffect` | Change in unrelated action dimensions or unrelated behavior. |

First-class outcome:

```text
ActionOutcome = stored original action + no-op regenerated action + intervened action + action deltas + metrics
```

### 3.8 Action Basis Terms

| Term | Definition |
|---|---|
| `ActionBasis` | Coordinate system used to interpret actions. |
| `RawActionBasis` | Native model/environment action vector. |
| `EEFDeltaBasis` | End-effector Δx, Δy, Δz. |
| `RotationBasis` | Wrist/orientation delta. |
| `GripperBasis` | Open/close scalar or binary. |
| `SpeedBasis` | Magnitude of translational motion. |
| `DirectionBasis` | Movement direction: left/right/up/down/forward/back. |
| `ObjectRelativeBasis` | Motion relative to an object: toward handle, away from mug, above bowl. |
| `TaskRelativeBasis` | Motion relative to task semantics: toward target, lift, place, retract. |
| `PCABasis` | Learned principal components over actions. |
| `LearnedActionFeatureBasis` | Learned action abstraction from data. |

Rule:

```text
Do not only show action[6] changed by +0.22.
Show gripper close increased, translation stayed mostly stable, wrist rotation changed slightly.
```

### 3.9 Effect-Size Terms

| Term | Definition |
|---|---|
| `RawDelta` | Direct difference in action values. |
| `NormalizedDelta` | Delta divided by dataset/action standard deviation. |
| `ZScoredEffect` | Effect in standard deviations. |
| `RelativeEffect` | Effect relative to original magnitude. |
| `NoopBaseline` | Rerun without intervention. |
| `EffectSize` | Normalized summary of intervention impact. |
| `SideEffectScore` | Degree of unintended action-dimension change. |
| `SpecificityScore` | Concentration of effect on intended dimensions vs side effects/controls. |
| `Monotonicity` | Whether strength sweep produces consistent directional change. |
| `Robustness` | Whether effect holds over seeds/episodes/cohorts. |

Always prefer this comparison when runtime rerun is available:

```text
stored original
noop regenerated
intervened regenerated
```

### 3.10 Control Terms

| Term | Definition |
|---|---|
| `ControlSpec` | Planned control condition for an intervention. |
| `NoopRerun` | Rerun with no intervention. |
| `RandomDirectionControl` | Matched-norm random direction. |
| `WrongLayerControl` | Same intervention at another layer. |
| `WrongTimeControl` | Same intervention at irrelevant policy call/timestep. |
| `WrongTokenControl` | Same intervention on irrelevant token space/index. |
| `ShuffledDonorControl` | Mismatched donor examples. |
| `MatchedCohortControl` | Matched task/scene/action baseline. |
| `PlaceboTargetControl` | Intervene on target expected not to affect behavior. |
| `StrengthSweepControl` | Check smoothness/monotonicity over strengths. |
| `HeldoutSplitControl` | Check beyond discovery/train examples. |

Controls are what separate a cool demo from a claim. They should not block v0, but the schema must support them from the beginning.

### 3.11 Evidence Terms

| Term | Definition |
|---|---|
| `InterventionRequest` | Ephemeral live-run request. Not necessarily persisted unless executed. |
| `InterventionTrial` | One execution attempt: noop/control/intervened, one strength/seed/etc. |
| `InterventionRun` | Saved exact executed spec, trial outputs, metrics, and provenance. |
| `InterventionSweep` | Related runs over strengths, layers, calls, tokens, or seeds. |
| `InterventionStudy` | Sweep plus controls across a cohort. |
| `InterventionCard` | Human-readable summary of a run or small sweep. |
| `MechanismCard` | Higher-level claim assembled from multiple evidence artifacts. |
| `EvidenceBundle` | Exportable package of specs, arrays, metrics, plots, media, and provenance. |
| `ClaimStrength` | UI label describing responsible interpretation. |
| `ArtifactProvenance` | Code/data/model/spec/runtime sources. |

Claim-strength labels:

| Label | Meaning |
|---|---|
| `observation` | Interesting activation/example/attention/cluster pattern. |
| `predictive` | Heldout probe or association evidence. |
| `causal_local` | Intervention changes one policy-call output. |
| `causal_cohort` | Effect holds over a cohort. |
| `behavioral` | Rollout-level behavior changes. |
| `specific` | Controls reduce chance of generic model damage. |
| `mechanistic` | Multiple artifacts support localized causal pathway. |

V0 does not need automated claim grading. It must store enough data that later guardrails can classify or reject overclaims.

---

## 4. Alignment With Current Codebase

This section is implementation guidance, not a rewrite mandate.

### 4.1 Current architecture already supports the boundary

The current repo already separates LeRobot robot data from VLA Lens overlay semantics. Keep that direction:

```text
LeRobot layer owns:
  episodes, frames, timestamps, tasks, observations, actions, camera media

VLA Lens overlay owns:
  policy calls, model sites, activations, token spaces, attention maps,
  action-generation traces, probes, derived artifacts
```

This intervention layer belongs in the VLA Lens overlay/workbench/evidence side, not in the base LeRobot robot-data contract.

### 4.2 Current `LensArtifact` is a provenance shell

Current `LensArtifact` is intentionally small and generic. It has fields like:

```text
artifact_id
artifact_type
name
group_id
scope
selector
method
metrics
arrays
display
tags
created_utc
source_trace_ids
path
```

Do not replace it with a giant intervention class. Instead:

```text
InterventionRun = typed semantic payload
LensArtifact = artifact-browser/display/provenance shell
```

Recommended mapping:

```text
LensArtifact(type="intervention_run")
  selector = context + target summary
  method = operator + schedule + request hash
  metrics = compact outcome/control metrics
  arrays = refs to stored action chunks/deltas
  display = InterventionCard summary fields
  source_trace_ids = recipient/donor trace ids
```

### 4.3 Current workbench already has `InterventionRunSpec`

Current workbench `InterventionRunSpec` is a saved readout record, not a live execution request. It currently has approximately this shape:

```python
InterventionRunSpec(
    run_id: str,
    intervention_type: str,
    target: Mapping[str, Any],
    baseline: Mapping[str, Any],
    intervention: Mapping[str, Any],
    readouts: Mapping[str, Any],
    outputs: tuple[str, ...],
    provenance: Mapping[str, Any],
)
```

Use this as the compatibility shell for v0. Do not break existing saved records.

Recommended compatibility mapping:

```text
run_id:
  stable run id

intervention_type:
  use "intervention_record" for v0 typed intervention records
  do not encode causal strength in this shell field

target:
  TargetSpec as dict

baseline:
  ContextSpec + stored original refs + optional noop baseline refs

intervention:
  InterventionSpec + ScheduleSpec + request_payload/executed_spec

readouts:
  Outcome results + Trial records + metrics + card summary

outputs:
  array ids / LensArray ids / artifact ids / media ids

provenance:
  RuntimePreflightResult, RuntimeResolution, model/data/code fingerprints,
  source artifact ids, evidence_level, warnings/errors
```

Typed dataclasses now wrap these mappings and serialize through the existing
workbench shell.

### 4.4 Current PI0.5 intervention specs are useful but too PI0.5-shaped

Current PI0.5 specs include operations like:

```text
kv_layer_replace
kv_wrong_layer_control
expert_hidden_replace
flow_state_replace
direction_add
direction_project_out
```

They also already serialize intended site/donor/recipient/control information. Keep them as adapter-specific implementation specs, but introduce a model-agnostic layer above them:

```text
src/vla_lens/interventions/specs.py       # generic specs
src/vla_lens/pi05/intervention_runtime.py # PI0.5 resolver/runtime
src/vla_lens/pi05/interventions.py        # PI0.5 low-level compatibility or adapter ops
```

### 4.5 Current dashboard route semantics should remain clear

Existing `/api/intervention-runs` is a saved workbench-state route. Do not silently turn it into a live model execution route.

Implemented and remaining route split:

```text
GET  /api/intervention-runs
POST /api/intervention-runs
  Persist or list saved intervention readout records.

POST /api/interventions/preflight
  Implemented: check whether a request can run without loading heavy runtime
  dependencies in the normal environment.

POST /api/interventions/run
  Remaining: live execution through an explicit runtime-capable boundary.

POST /api/interventions/save
  Optional explicit save route if live execution returns an unsaved draft.
```

The saved-record and preflight routes must remain runtime-free. A live-run route
must stay behind explicit capability and environment boundaries.

---

## 5. Core Contracts

The contracts below are Python-dataclass style pseudocode. Names can change, but the semantics should remain stable.

### 5.1 `ContextSpec`

```python
@dataclass(frozen=True, slots=True)
class ContextSpec:
    dataset_id: str | None = None
    dataset_root_id: str | None = None
    trace_id: str | None = None
    episode_id: str | None = None
    policy_call_index: int | None = None
    timestep: int | None = None
    frame_index: int | None = None
    instruction: str | None = None
    task: str | None = None
    preview_media: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

Required for v0 execution:

```text
trace_id
policy_call_index
```

Required for useful display:

```text
instruction or task
frame_index or preview_media
stored original action ref
```

### 5.2 `TargetSpec`

```python
@dataclass(frozen=True, slots=True)
class TargetSpec:
    kind: str  # probe_direction | contrast_direction | activation_slice | feature | subspace | head | edge | manual
    source_artifact_id: str | None = None
    source_artifact_type: str | None = None
    model_id: str | None = None
    model_family: str | None = None
    model_site: str | None = None
    site_id: str | None = None
    module_path: str | None = None
    layer: int | None = None
    tensor_type: str | None = None
    token_space: str | None = None
    token_selector: Mapping[str, Any] = field(default_factory=dict)
    generation_step_selector: Mapping[str, Any] = field(default_factory=dict)
    reduction: str | None = None
    representation: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

Representation examples:

```json
{"kind": "vector", "array_ref": "artifact://probe/coef", "normalization": "unit_norm"}
{"kind": "feature_index", "feature_index": 9182, "feature_family": "sae"}
{"kind": "subspace", "basis_ref": "artifact://probe/subspace", "rank": 4}
{"kind": "activation_slice", "donor": {"trace_id": "success_017", "policy_call_index": 5}}
```

### 5.3 `InterventionScheduleSpec`

```python
@dataclass(frozen=True, slots=True)
class InterventionScheduleSpec:
    policy_calls: tuple[int, ...] | str = "selected"  # selected | all | explicit tuple
    timesteps: Mapping[str, Any] = field(default_factory=dict)
    generation_steps: Mapping[str, Any] | str = "all"
    action_horizon: Mapping[str, Any] | str = "full_chunk"
    tokens: Mapping[str, Any] | str = "target_tokens"
    condition: Mapping[str, Any] = field(default_factory=dict)
```

Examples:

```json
{"policy_calls": [7], "generation_steps": "all", "tokens": "action"}
{"policy_calls": [7], "generation_steps": {"start": 3, "stop": 8}, "tokens": {"token_space": "action", "indices": "all"}}
```

### 5.4 `InterventionOperatorSpec`

```python
@dataclass(frozen=True, slots=True)
class InterventionOperatorSpec:
    operator: str  # add_direction | project_out_direction | mean_replace | source_patch | ablate | clamp
    strength: float | None = None
    strengths: tuple[float, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)
```

Rules:

```text
Use `strength` for a single trial.
Use `strengths` for a sweep request.
Saved `InterventionRun` should store actual trials separately, not hide them behind strengths only.
```

### 5.5 `ControlSpec`

```python
@dataclass(frozen=True, slots=True)
class ControlSpec:
    kind: str  # noop_rerun | random_direction | wrong_layer | wrong_time | wrong_token | shuffled_donor
    parameters: Mapping[str, Any] = field(default_factory=dict)
    expected_effect: str | None = None
```

V0 controls:

```text
noop_rerun: always run when runtime is available
random_direction: matched norm/dim to target direction
wrong_layer: if target resolver can cheaply map layer ±k
```

### 5.6 `OutcomeSpec`

```python
@dataclass(frozen=True, slots=True)
class OutcomeSpec:
    kind: str  # action | rollout | token | probe | metric
    basis: tuple[str, ...] = ("raw",)
    horizon: str | Mapping[str, Any] = "full_chunk"
    metrics: tuple[str, ...] = ("raw_delta", "normalized_delta")
    compare_to: str = "noop_if_available_else_stored_original"
    parameters: Mapping[str, Any] = field(default_factory=dict)
```

V0 default:

```json
{
  "kind": "action",
  "basis": ["raw", "gripper", "eef_delta_xyz", "rotation"],
  "horizon": "full_chunk",
  "compare_to": "noop_if_available_else_stored_original"
}
```

### 5.7 `InterventionRequest`

Live-run request. Not necessarily persisted unless executed.

```python
@dataclass(frozen=True, slots=True)
class InterventionRequest:
    schema_version: str
    context: ContextSpec
    target: TargetSpec
    operator: InterventionOperatorSpec
    schedule: InterventionScheduleSpec
    outcome: OutcomeSpec
    controls: tuple[ControlSpec, ...] = ()
    ui: Mapping[str, Any] = field(default_factory=dict)
```

### 5.8 `RuntimePreflightResult`

```python
@dataclass(frozen=True, slots=True)
class RuntimePreflightResult:
    ok: bool
    capability_status: Mapping[str, bool]
    target_resolution: Mapping[str, Any] = field(default_factory=dict)
    action_basis_status: Mapping[str, bool] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    runtime_environment: Mapping[str, Any] = field(default_factory=dict)
```

Required checks:

```text
reconstructable_policy_call
model_runtime_available
target_resolvable
action_decoder_available
action_basis_available
runtime_environment_safe
```

Preflight should not mutate state. It should be safe to call from the UI repeatedly.

### 5.9 `RuntimeResolution`

Records how the generic target was mapped into a concrete runtime hook.

```python
@dataclass(frozen=True, slots=True)
class RuntimeResolution:
    adapter: str
    model_family: str
    requested_target: Mapping[str, Any]
    resolved_hook: Mapping[str, Any]
    resolved_tensor_shape: tuple[int, ...] = ()
    resolved_dtype: str | None = None
    resolved_device: str | None = None
    warnings: tuple[str, ...] = ()
```

Important: the saved run should record both requested target and actual resolved hook. This catches target/hook mismatch later.

### 5.10 `InterventionTrial`

One actual execution attempt.

```python
@dataclass(frozen=True, slots=True)
class InterventionTrial:
    trial_id: str
    trial_kind: str  # stored_original | noop | intervention | control
    control_kind: str | None = None
    strength: float | None = None
    seed: int | None = None
    target_override: Mapping[str, Any] = field(default_factory=dict)
    operator_override: Mapping[str, Any] = field(default_factory=dict)
    schedule_override: Mapping[str, Any] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    status: str = "ok"  # ok | failed | skipped
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
```

Trial kinds:

```text
stored_original: not executed; action already saved in dataset
noop: regenerated with no intervention
intervention: target operator applied
control: random/wrong-layer/wrong-time/etc.
```

### 5.11 `ActionOutcomeResult`

```python
@dataclass(frozen=True, slots=True)
class ActionOutcomeResult:
    basis: str
    horizon: Mapping[str, Any] | str
    baseline_trial_id: str
    intervention_trial_id: str
    action_ref_baseline: str | None = None
    action_ref_intervened: str | None = None
    delta_ref: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    summaries: Mapping[str, Any] = field(default_factory=dict)
```

Metric examples:

```json
{
  "gripper_mean_delta": 0.31,
  "gripper_max_delta": 0.44,
  "eef_delta_norm": 0.06,
  "rotation_delta_norm": 0.02,
  "normalized_effect_size": 1.8,
  "noop_drift_norm": 0.03,
  "side_effect_score": 0.12,
  "specificity_score": 0.79
}
```

### 5.12 `InterventionRun`

Canonical saved evidence payload.

```python
@dataclass(frozen=True, slots=True)
class InterventionRun:
    schema_version: str
    run_id: str
    title: str
    status: str  # ok | partial | failed | inspected_only
    created_utc: str

    context: ContextSpec
    target: TargetSpec
    request: Mapping[str, Any]
    preflight: RuntimePreflightResult
    runtime_resolution: RuntimeResolution | None

    trials: tuple[InterventionTrial, ...]
    outcomes: tuple[Mapping[str, Any], ...]
    controls: tuple[Mapping[str, Any], ...]

    outputs: tuple[str, ...]
    display: Mapping[str, Any]
    claim: Mapping[str, Any]
    provenance: Mapping[str, Any]
```

Recommended `claim` fields:

```json
{
  "claim_text": "Adding the gripper-close probe direction increases generated gripper closure.",
  "claim_strength": ["causal_local"],
  "evidence_kind": "action_counterfactual",
  "limitations": ["single policy call", "no rollout outcome", "controls incomplete"]
}
```

### 5.13 Compatibility Wrapper: current `InterventionRunSpec`

Until the current workbench schema is upgraded, save an `InterventionRun` inside the existing shell like this:

```python
InterventionRunSpec(
    run_id=run.run_id,
    intervention_type="intervention_record",
    target=run.target.to_dict(),
    baseline={
        "context": run.context.to_dict(),
        "stored_original_trial": "trial_original",
        "noop_trial": "trial_noop",
    },
    intervention={
        "request": run.request,
        "operator": run.request["operator"],
        "schedule": run.request["schedule"],
    },
    readouts={
        "schema_version": run.schema_version,
        "trials": [trial.to_dict() for trial in run.trials],
        "outcomes": list(run.outcomes),
        "controls": list(run.controls),
        "display": run.display,
        "claim": run.claim,
        "status": run.status,
    },
    outputs=run.outputs,
    provenance={
        **run.provenance,
        "preflight": run.preflight.to_dict(),
        "runtime_resolution": run.runtime_resolution.to_dict() if run.runtime_resolution else None,
        "evidence_kind": run.claim.get("evidence_kind"),
        "evidence_level": run.claim.get("claim_strength"),
    },
)
```

`intervention_type` is a storage/category field only. Whether the record supports a causal claim must be inferred from `status`, explicit trials, outcomes, controls, claim labels, and provenance.

---

## 6. Persistence Model

There are two related persistence surfaces:

```text
1. Workbench intervention record
   Machine-readable canonical record used by dashboard state and APIs.

2. LensArtifact(type="intervention_run")
   Artifact-browser/display/provenance shell with array refs and card metadata.
```

Recommended v0 rule:

```text
The workbench InterventionRun payload is canonical.
The LensArtifact indexes/displays it and points to arrays/media.
Do not store conflicting summaries in both places.
```

Array storage should avoid bloating JSON:

```text
arrays/action_original.npy
arrays/action_noop.npy
arrays/action_intervened_strength_+2.npy
arrays/action_delta_strength_+2.npy
arrays/action_basis_gripper_delta.npy
```

`outputs` should include stable refs:

```json
[
  "lens-array://intervention/run_123/action_original",
  "lens-array://intervention/run_123/action_intervened_strength_2",
  "artifact://intervention_run/run_123"
]
```

---

## 7. Runtime Services

### 7.1 `PreflightService`

Responsibility:

```text
Given InterventionRequest, return RuntimePreflightResult.
Do not load heavyweight runtime unless necessary or explicitly allowed.
Do not mutate dataset state.
```

Checks:

```text
selected dataset root exists
policy call exists
generated/stored action exists
model runtime can be loaded in current environment
target can be resolved to hook
action decoder exists
action basis adapters exist
controls can be constructed
requested schedule is supported
```

### 7.2 `TargetResolver`

Responsibility:

```text
Map TargetSpec to adapter-specific runtime hook.
```

Inputs:

```text
TargetSpec
model adapter metadata
model-site table
source artifact payload/arrays
policy call context
```

Outputs:

```text
RuntimeResolution
callable hook spec / adapter-private hook object
```

### 7.3 `InterventionRuntime`

Responsibility:

```text
Execute no-op, intervention, and controls for one request.
```

Steps:

```text
load model/checkpoint
reconstruct policy-call input
resolve target
register hook
run model forward
extract/generated action chunk
decode action
write arrays/results
return InterventionRun
```

### 7.4 `OutcomeAdapter`

Responsibility:

```text
Convert model-specific outputs into generic outcome objects.
```

Examples:

```text
PI0.5 action chunk → ActionOutcome
OpenVLA action tokens → ActionOutcome
VLM logits → TokenOutcome
```

### 7.5 `ActionBasisAdapter`

Responsibility:

```text
Convert raw actions into interpretable bases.
```

Required v0 bases:

```text
raw: always if action chunk exists
gripper: only if action metadata maps gripper dimension
eef_delta_xyz: only if action metadata maps translation dimensions
rotation: only if action metadata maps rotation dimensions
```

Missing basis should produce a partial result, not a failed run.

### 7.6 `ControlFactory`

Responsibility:

```text
Construct control trials from the target/operator/request.
```

V0:

```text
noop_rerun
random_direction matched to target vector shape/norm
wrong_layer if layer-resolvable
```

### 7.7 `InterventionArtifactWriter`

Responsibility:

```text
Write arrays, workbench InterventionRun, optional LensArtifact, and display metadata.
```

It should be possible to call this with an inspected-only or failed run so failures are auditable.

---

## 8. API Design

### 8.1 Saved Record API

Current-compatible route:

```http
GET  /api/intervention-runs
POST /api/intervention-runs
```

Semantics:

```text
Persist/list saved intervention readout records.
Does not imply live model execution.
```

### 8.2 Preflight API

```http
POST /api/interventions/preflight
```

Request:

```json
{
  "schema_version": "0.2.0",
  "context": {"trace_id": "episode_000042", "policy_call_index": 7},
  "target": {"kind": "probe_direction", "source_artifact_id": "probe-gripper-close-v1"},
  "operator": {"operator": "add_direction", "strengths": [-3, -1, 0, 1, 3]},
  "schedule": {"policy_calls": [7], "generation_steps": "all", "tokens": "action"},
  "outcome": {"kind": "action", "basis": ["raw", "gripper", "eef_delta_xyz"]},
  "controls": [{"kind": "noop_rerun"}, {"kind": "random_direction"}]
}
```

Response:

```json
{
  "ok": true,
  "capability_status": {
    "reconstructable_policy_call": true,
    "model_runtime_available": true,
    "target_resolvable": true,
    "action_decoder_available": true,
    "action_basis_available": true,
    "runtime_environment_safe": true
  },
  "target_resolution": {
    "adapter": "pi05",
    "resolved_hook_name": "expert.layers.12.hidden_tokens",
    "resolved_shape": [1, 16, 2048]
  },
  "warnings": [],
  "errors": []
}
```

### 8.3 Live Run API

```http
POST /api/interventions/run
```

Semantics:

```text
Capability-gated live execution.
May live in a PI0.5 runtime server/process rather than the normal dashboard process.
Should return an InterventionRun payload and optionally save it.
```

Response statuses:

```text
ok: all requested trials ran
partial: main trial ran but some controls/bases failed
failed: no useful trial ran
inspected_only: request saved/inspected but not executable from this root
```

### 8.4 Example Full Request

```json
{
  "schema_version": "0.2.0",
  "context": {
    "dataset_id": "pi05-light-5-test",
    "trace_id": "episode_000042",
    "policy_call_index": 7,
    "frame_index": 40,
    "instruction": "put the mug in the bowl"
  },
  "target": {
    "kind": "probe_direction",
    "source_artifact_id": "probe-gripper-close-v1",
    "source_artifact_type": "probe",
    "model_site": "pi05.expert.layers.12.hidden_tokens",
    "layer": 12,
    "tensor_type": "hidden_tokens",
    "token_space": "action",
    "token_selector": {"kind": "all"},
    "reduction": "mean",
    "representation": {
      "kind": "vector",
      "array_ref": "artifact://probe-gripper-close-v1/coef_layer_12",
      "normalization": "unit_norm"
    }
  },
  "operator": {
    "operator": "add_direction",
    "strengths": [-3, -1, 0, 1, 3],
    "parameters": {"normalize_direction": true}
  },
  "schedule": {
    "policy_calls": [7],
    "generation_steps": "all",
    "action_horizon": "full_chunk",
    "tokens": "target_tokens"
  },
  "outcome": {
    "kind": "action",
    "basis": ["raw", "gripper", "eef_delta_xyz", "rotation"],
    "horizon": "full_chunk",
    "metrics": ["raw_delta", "normalized_delta", "side_effect_score"],
    "compare_to": "noop_if_available_else_stored_original"
  },
  "controls": [
    {"kind": "noop_rerun"},
    {"kind": "random_direction", "parameters": {"count": 1, "matched_norm": true}},
    {"kind": "wrong_layer", "parameters": {"layer_offset": -4}}
  ],
  "ui": {
    "source_surface": "probe_artifact_page",
    "action_label": "Intervene with this signal"
  }
}
```

---

## 9. UI Surfaces

The dictionary should not appear all at once in the UI. Use simple researcher-facing workflows.

### 9.1 Episode Microscope

Purpose:

```text
Understand a specific behavior moment.
```

Shows:

```text
video/frame
instruction/task
timeline
policy calls
generated/executed action chunk
model-site summary
available artifacts
```

Primary actions:

```text
Send moment to Intervention
Open related artifacts
Find similar moments
Open model site
```

### 9.2 Artifact Browser

Purpose:

```text
Inspect discovery artifacts and convert them into targets.
```

Artifact pages:

```text
probe page
contrast direction page
activation cluster page
action-generation page
attention/attribution page
future SAE/transcoder feature page
```

Primary actions:

```text
View top moments
Filter by action/outcome
Intervene with this signal
Run control sweep
```

### 9.3 Intervention Lab

Purpose:

```text
Run or inspect counterfactuals.
```

Seed contract:

```text
Probe and Episode Lens entry points should request a backend-normalized
TargetSpec from /api/discovery-artifacts/{artifact_id}/target using artifact
id, trace id, policy call, model site, and token space when known.

If that request is unavailable, the UI may continue with an inspectable local
fallback target, but the fallback must mark metadata.target_source as
local_fallback.
```

UI wizard labels:

```text
1. Where?       episode / policy call / frame
2. What signal? target from probe/artifact/manual site
3. How change?  add, remove/project out, patch, replace
4. When?        selected call, generation steps, tokens
5. What measure? action basis, horizon, rollout/token if available
6. Controls?    no-op, random, wrong layer/time/token
7. Compare      original vs no-op vs intervened
```

Advanced drawer can expose:

```text
TargetSpec
InterventionSpec
ScheduleSpec
OutcomeSpec
ControlSpec
RuntimeResolution
```

### 9.4 Intervention Card

Purpose:

```text
Compact saved evidence object.
```

Card fields:

```text
Claim
Context
Target
Intervention
Outcome
Controls
Claim strength
Limitations
Open full run
Scale to cohort
Export
```

Example:

```text
Claim:
  Adding the gripper-close probe direction increases generated gripper closure.

Context:
  Episode 42, policy call 7, robot approaching drawer handle.

Target:
  expert.layers.12, action tokens, probe direction.

Intervention:
  Add direction, strength +2.0, all generation steps.

Outcome:
  Gripper close +0.31 normalized units.
  Translation mostly unchanged.
  No-op drift +0.02.

Controls:
  Random direction +0.01.
  Wrong layer +0.04.

Claim strength:
  causal_local, action-level, controls partial.
```

### 9.5 Evidence Library

Purpose:

```text
Review saved runs, sweeps, studies, and cards.
```

Filters:

```text
artifact type
claim strength
model/dataset/task
source artifact
operator
outcome kind/action basis
status
has controls
has rollout outcome
```

---

## 10. Minimal Product Slice: Probe Direction Intervention v0

### 10.1 Required Inputs

```text
probe artifact
selected episode / trace_id
selected policy call
selected model site / inferred probe target site
strength or strength sweep
action basis request
```

### 10.2 Operation

```text
AddDirection
ProjectOutDirection
```

### 10.3 Outcomes

```text
stored original action chunk
noop regenerated action chunk, if available
intervened action chunk
delta vs no-op, or delta vs stored original when no-op unavailable
raw action delta
gripper / xyz / rotation deltas when metadata exists
```

### 10.4 Controls

```text
noop rerun
random direction
wrong layer, if adapter can resolve it cheaply
```

### 10.5 Output

```text
Workbench InterventionRun record
LensArtifact(type="intervention_run") index/display shell
arrays for action chunks and deltas
Intervention Card display metadata
```

### 10.6 Acceptance Criteria

1. A saved probe direction, rather than a synthetic one-hot direction, can be
   resolved and applied to a reconstructable PI0.5 policy call.
2. Both add-direction and project-out are supported with a matched random
   control and at least one specificity control.
3. The Intervention Lab can invoke the live execution boundary and compare
   stored-original, no-op, intervened, and control action chunks.
4. Claim eligibility remains false unless the saved experiment records the
   controls and outcomes required by its claim contract.

---

## 11. Remaining Implementation Order

### 11.1 Claim-Eligible PI0.5 Probe-Direction Runtime

The current CLI-first runner proves deterministic replay and hook plumbing with
a non-claiming synthetic one-hot direction. Extend it to resolve a saved probe
direction, support add and project-out, and record matched random plus
specificity controls.

### 11.2 Live Intervention Lab Comparison

The Lab already supports target seeding, preflight, inspected-only saves, and
saved evidence display. Connect it to the live execution boundary and add
stored-original/no-op/intervened/control action comparison charts.

### 11.3 Sweep And Cohort Execution

Sweep/study types, promotion, aggregation, indexing, and claim gating exist.
Add the runner and UI that materialize controlled runs over explicit axes and
cohorts.

---
## 12. Failure Modes To Make Visible

Interventions will often fail or be messy. Save those failures.

| Observation | Possible interpretation |
|---|---|
| Probe predicts action but steering does nothing. | Direction may be readout-correlated, not causally used; target/hook mismatch; wrong schedule. |
| Intervention changes every action dimension. | Nonspecific effect or off-manifold damage. |
| Effect appears only at one layer/time. | Possible localized mechanism or fragile artifact. |
| Effect appears on train but not heldout. | Discovery artifact may not generalize. |
| Action chunk changes but rollout does not. | Action effect may be too small, badly timed, or compensated by environment. |
| No-op drift is large. | Rerun nondeterminism makes causal attribution weak. |
| Wrong-layer control is similar to main effect. | Signal may be nonspecific or operator too blunt. |
| Random direction control has similar effect. | Target direction may not be special; intervention magnitude too large. |

UI should make these visible through warnings and limitations, not hide them.

---

## 13. Non-Goals For V0

Do not make v0 depend on:

```text
new PI0.5 capture profiles
full closed-loop rollout
real-robot execution
SAE/transcoder support
automatic mechanism discovery
separate queued-job planning product
automated claim grading
high-powered statistics before one run works
changing PolicyCallRecord unless a concrete missing field is proven
rewriting current LensArtifact or workbench stores
```

Existing captures should remain valid. Runtime support should be capability-gated and checked by preflight, not assumed from policy-call rows.

---

## 14. Future Extensions

### 14.1 SAEs

SAE feature pages should become discovery pages whose primary causal affordance is:

```text
Intervene with this feature
```

Mapping:

```text
SAEFeature → TargetSpec(kind="feature", representation={feature_index, decoder_vector})
```

Intervention operators:

```text
feature_boost
feature_clamp
feature_ablate
add decoder direction
```

### 14.2 Transcoders and Crosscoders

Use them later for pathway/mechanism hypotheses:

```text
TranscoderFeature → pathway TargetSpec
CrosscoderFeature → shared/differential TargetSpec
```

Likely operators:

```text
feature clamp
path patch
source patch
project out pathway direction
```

Do not prioritize before the intervention spine works.

### 14.3 Closed-loop rollout outcomes

Add when model/env runtime can reset/replay reliably:

```text
InterventionSchedule over policy-call ranges
closed-loop rollout execution
RolloutOutcome metrics
side-by-side video
success/failure/contact/collision/final distance
```

### 14.4 Mechanism Cards

A `MechanismCard` should aggregate:

```text
discovery artifacts
local intervention runs
controls
cohort sweeps
rollout outcomes
limitations
paper-level figure exports
```

This is not v0.

---

## 15. Implementation Spine

Preserve this implemented spine:

```text
DiscoveryArtifact
  → TargetSpec
  → InterventionRequest
  → RuntimePreflightResult
  → InterventionTrial(s)
  → InterventionRun
  → InterventionCard / LensArtifact
```

Remaining runtimes should preserve this minimum researcher loop:

```text
Open probe
  → click Intervene with this signal
  → choose episode policy call
  → add/project-out direction
  → run no-op + intervention
  → compare action chunk
  → save InterventionRun
```

The next useful result is not an SAE page or more dashboard surface. It is a
saved, lossless, inspectable, controlled counterfactual that records what
changed when the runtime applied an artifact-derived candidate signal.
