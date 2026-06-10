# Probe Evidence Contract - Phased Implementation Plan

Status: draft for GPT Pro audit before implementation.

Source intent: preserve the architecture discussion and GPT Pro audit in an implementation-ready plan.

Repo UI contract: this plan follows `docs/research_ui_principles.md`. The dataset page is treated as a lens-conditioned episode browser, and the episode page is treated as the synchronized microscope.

## Executive Summary

The immediate implementation target is not a universal Lens Evidence IR.

The immediate target is a v1 `ProbeEvidenceBundle` and panel-gating system that lets the dataset page act as a lens-conditioned episode browser and lets the episode page act as a probe-aware microscope.

The first shipped loop is:

```text
select dataset + probe
  -> inspect top, bottom, and uncertain moments
  -> open an episode with lens context preserved
  -> see probe prediction, score, source site, and weighted contributors if available
  -> pin the selected evidence state
```

Generic lens concepts are allowed only as extension seams:

```text
LensGeometry
LensCapability
PanelSpec
ResearchSelectionState
```

SAE, transcoder, crosscoder, attribution, and intervention implementations are out of v1.

The main enemy is bucket rot:

```text
One rich episode capture
  -> many UI-specific buckets
  -> UI changes
  -> stale buckets remain
  -> tests do not catch semantic drift
```

The desired architecture is:

```text
Raw episode capture
  -> existing trace/workbench/model-site metadata
  -> canonical probe evidence bundle
  -> shared research selection state
  -> capability-gated panels
  -> disposable panel view models
```

The architecture to avoid is:

```text
Raw episode capture
  -> dataset table object
  -> episode card object
  -> probe stats object
  -> model viewer object
  -> timeline object
  -> evidence tab object
  -> workbench object
  -> random local transforms inside components
```

Raw episode capture should not be the public research-facing UI API. It can be huge, rich, and ugly. Research-facing UI should mostly consume a smaller vocabulary of canonical probe evidence objects.

## Core Principle

The evidence contract exists to reduce representational freedom, not increase it.

Avoid this:

```text
LensEvidence = giant flexible blob that can represent everything
```

Prefer this:

```text
ProbeEvidenceBundle =
  small number of typed evidence primitives
  + explicit geometry/capability metadata
  + references to large arrays/tensors
  + strict panel contracts
  + explicit unavailable reasons
```

Every new UI surface should either consume existing evidence primitives or justify a new primitive.

## Product Frame

The project goal is not only to display episodes. The goal is to let a researcher apply a probe and quickly understand whether it says something real about model behavior.

The UI should not ask the system to "show the most useful information for each type of probe." That prompt is too unconstrained.

Instead, implement:

```text
Show all panels whose required capabilities are satisfied by this probe's evidence geometry.
For unavailable panels, show a precise reason.
For all rendered panels, consume only canonical evidence primitives.
```

## Required Interaction Contracts for V1

These contracts come before data contracts. A data type is only useful if it supports a research interaction.

### Interaction 1: Select Probe

Research question:

```text
What does this probe detect across this dataset?
```

State set:

```text
dataset_id
lens_id
lens_run_id
```

Evidence shown:

```text
provenance
score summary
top moments
bottom moments
uncertain moments
unavailable reasons
```

Required behavior:

- Dataset page becomes lens-over-dataset-first.
- Probe summary uses researcher-native fields such as `Prediction`, `Input`, `Output`, `Objective`, `Split`, `Model site`, and `Policy call`.
- The UI must distinguish "probe artifact exists" from "probe has aligned results for this dataset."

### Interaction 2: Select Ranked Moment

Research question:

```text
Why did this moment matter under this probe?
```

State set:

```text
episode_id
timestep or policy_call
lens_id
lens_run_id
ranking
```

Evidence shown:

```text
frame or video
action trace
score
prediction
source site
contribution if available
unavailable reasons if not available
```

Required behavior:

- Clicking a ranked moment updates `ResearchSelectionState`.
- Opening the episode preserves dataset, lens, lens run, episode, and timestep context.
- The episode microscope synchronizes video, timeline, action trace, model inspector, and probe readout around the selected moment.

### Interaction 3: Select Model Site or Source

Research question:

```text
Where did this probe read from?
```

State set:

```text
model_locus
policy_call or timestep
lens_id
lens_run_id
```

Evidence shown:

```text
source metadata
compatible model inspector view
probe source highlights
available contribution grouping
```

Required behavior:

- Probe source sites are highlighted or defaulted in the model pipeline/map.
- The UI should not claim head, token, or channel specificity if the probe was trained on pooled representations.

### Interaction 4: Select Contributor

Research question:

```text
What pushed this probe score?
```

State set:

```text
feature_id or contributor_id
model_locus if available
episode_id
timestep or policy_call
lens_id
lens_run_id
```

Evidence shown:

```text
weighted contribution
sign
rank
raw activation comparison if available
model locus if available
human label or semantic hypothesis only if claim level supports it
```

Required behavior:

- Contribution views must expose `claim_level`.
- Raw activation dimensions are numeric contributors, not semantic features.
- SAE features or human-labeled features can support stronger feature-level copy.

### Interaction 5: Pin Evidence

Research question:

```text
What observation should survive navigation?
```

State set:

```text
saved evidence ref
```

Evidence saved:

```text
lens artifact id
lens run id
dataset id
episode id
timestep or policy call
model site
selected contributor if any
panel/evidence primitive kind
score/prediction if available
claim level if available
short optional note
```

Required behavior:

- Pins preserve research state, not page layout state.
- V1 pinning is not a full lab notebook, markdown editor, tagging system, or artifact gallery.
- The saved payload should be serializable into a future artifact/lab-notebook record.

## Durable Contracts and Disposable Layer

Durable contracts:

```text
CaptureAvailabilityView
LensArtifact
LensRun
LensGeometry
ProbeEvidenceBundle
ResearchSelectionState
PanelSpec
```

Disposable layer:

```text
Panel view models
```

Panel view models can change every week. The durable contracts should not.

## Contract 1: CaptureAvailabilityView

Question answered:

```text
What data exists?
```

This should be an adapter or normalized read model over existing trace, workbench, model-site, and dataset metadata. Do not create a new source of truth if existing manifests already answer the question.

```ts
type CaptureAvailabilityView = {
  episode_id: string;
  dataset_id: string;

  time_axis: {
    kind: "timestep" | "frame" | "token" | "window" | "policy_call";
    length: number;
    fps?: number;
  };

  modalities: {
    video?: boolean;
    actions?: boolean;
    observations?: boolean;
    rewards?: boolean;
    model_internals?: boolean;
  };

  model_sources: ModelSourceSpec[];

  metadata_schema?: Record<string, string>;
};
```

Purpose:

- Prevent UI code from checking arbitrary raw episode fields directly.
- Keep raw capture rich while making availability explicit.
- Avoid duplicating existing manifest/schema concepts.

## Contract 2: LensArtifact

Question answered:

```text
What is this probe or lens artifact?
```

For probe-centered v1 work, this is provenance and identity.

```ts
type LensArtifact = {
  lens_id: string;
  lens_version: string;
  lens_type: "probe" | "sae" | "transcoder" | "crosscoder" | "attribution" | "manual_metric";

  name: string;
  target?: string;

  source_model: {
    model_id: string;
    model_version?: string;
  };

  source: ModelSourceSpec;

  training?: {
    dataset_id?: string;
    split?: string;
    objective?: string;
    metrics?: Record<string, number>;
  };

  created_at?: string;
};
```

Required product behavior:

- The UI should show what trained the probe, what it predicts, and what objective/metrics exist.
- The UI should show what model layer, module, token scope, or source the probe used.
- Provenance should be concise by default and detailed only in artifact/debug contexts.

## Contract 3: LensRun

Question answered:

```text
Has this lens artifact been applied to this dataset, episode set, or capture profile?
```

A lens artifact is not the same thing as its aligned result on a dataset.

```ts
type LensRun = {
  lens_run_id: string;
  lens_id: string;
  lens_version: string;

  dataset_id: string;
  episode_ids?: string[];
  capture_profile_id?: string;

  computed_at: string;
  result_version: string;
  status: "complete" | "partial" | "failed";

  evidence_bundle_id?: string;
};
```

Purpose:

- Distinguish "the probe exists" from "the probe has valid aligned scores for this dataset."
- Allow the same probe to be applied to different datasets, splits, capture profiles, and result versions.
- Give selection state and pins a stable run-level reference.

## Contract 4: LensGeometry

Question answered:

```text
What shape of evidence can this probe honestly produce?
```

This is the main guardrail against combinatorial UI explosion.

```ts
type LensGeometry = {
  temporal_scope:
    | "episode"
    | "timestep"
    | "window"
    | "event"
    | "token"
    | "frame"
    | "policy_call";

  output_kind:
    | "scalar"
    | "class_label"
    | "class_distribution"
    | "vector"
    | "ranked_features"
    | "attribution_map"
    | "heatmap";

  input_basis:
    | "layer_activation"
    | "pooled_layer_activation"
    | "sae_feature"
    | "attention_head_output"
    | "token_state"
    | "image_patch"
    | "action_state"
    | "custom";

  locus_kind:
    | "none"
    | "model_locus"
    | "visual_locus"
    | "action_locus"
    | "token_locus"
    | "mixed_locus";

  capabilities: LensCapability[];
};

type LensCapability =
  | "score_series"
  | "ranked_moments"
  | "thresholding"
  | "prediction"
  | "uncertainty"
  | "contribution_breakdown"
  | "model_locus_view"
  | "visual_heatmap"
  | "cohort_summary"
  | "failure_cases"
  | "comparison";
```

UI rule:

```ts
if (geometry.capabilities.includes("score_series")) {
  // show score timeline
}

if (geometry.capabilities.includes("contribution_breakdown")) {
  // show contribution panel
}

if (geometry.locus_kind === "model_locus") {
  // show model locus panel
}
```

Avoid:

```ts
if (lens.type === "probe") {
  // show probe stuff
}
```

Reason:

- Lens type alone is not enough to know what the UI can honestly show.
- A probe trained on pooled layer activations should not imply token/head specificity.
- A probe trained on SAE features can support stronger feature-level claims.
- A probe trained on attention-head outputs can support grouped head-level contribution views.
- Geometry and capabilities gate panels more honestly than broad lens-type checks.

## Contract 5: ProbeEvidenceBundle

Question answered:

```text
What evidence do we have for this probe run across this dataset, episode, or selected moment?
```

V1 should use a probe-specific bundle. Do not add SAE, transcoder, crosscoder, attribution, or intervention branches until there is one real consumer.

```ts
type ProbeEvidenceBundle = {
  bundle_id: string;
  family: "probe";

  artifact: LensArtifact;
  run: LensRun;
  geometry: LensGeometry;
  capabilities: LensCapability[];

  primitives: EvidencePrimitive[];
  unavailable: UnavailableReason[];
};

type LensEvidenceBundle = ProbeEvidenceBundle; // v1 only
```

Evidence primitives:

```ts
type EvidencePrimitive =
  | LensProvenanceEvidence
  | ScoreSeriesEvidence
  | RankedMomentsEvidence
  | PredictionEvidence
  | ContributionEvidence
  | ModelLocusEvidence
  | CohortSummaryEvidence
  | FailureCaseEvidence;
```

Evidence primitives should be typed, boring, and small. Large arrays should be referenced, not eagerly stuffed into giant JSON responses.

### ScoreSeriesEvidence

```ts
type ScoreSeriesEvidence = {
  kind: "score_series";
  lens_id: string;
  lens_run_id: string;
  episode_id: string;
  time_axis: "timestep" | "frame" | "token" | "window" | "policy_call";
  values_ref: ArrayRef;
  summary: {
    min: number;
    max: number;
    mean: number;
    p50?: number;
    p95?: number;
  };
  threshold?: number;
};
```

Use cases:

- Probe score timeline.
- Episode-level score summary.
- Threshold crossings.
- High, low, and uncertain score regions.

### RankedMomentsEvidence

```ts
type RankedMomentsEvidence = {
  kind: "ranked_moments";
  lens_id: string;
  lens_run_id: string;
  ranking:
    | "top"
    | "bottom"
    | "uncertain"
    | "false_positive"
    | "false_negative"
    | "largest_delta";
  moments: {
    episode_id: string;
    timestep?: number;
    policy_call?: number;
    frame_idx?: number;
    score?: number;
    prediction?: string;
    label?: string;
    confidence?: number;
    thumbnail_ref?: string;
  }[];
};
```

Use cases:

- Top activations.
- Bottom activations.
- Ambiguous or uncertain cases.
- False positives and false negatives when labels or proxy targets exist.
- Largest deltas when comparing two states, cohorts, runs, or lenses.

### PredictionEvidence

```ts
type PredictionEvidence = {
  kind: "prediction";
  lens_id: string;
  lens_run_id: string;
  episode_id: string;
  timestep?: number;
  policy_call?: number;
  prediction: string | number | boolean;
  label?: string | number | boolean;
  confidence?: number;
  correct?: boolean;
  split?: "train" | "validation" | "test" | "missing";
};
```

Use cases:

- Probe readout at selected timestep or policy call.
- Split-aware trust language.
- Correct, wrong, and high-confidence wrong displays when labels exist.

### ContributionEvidence

```ts
type EvidenceClaimLevel =
  | "numeric_only"
  | "grouped_model_locus"
  | "human_labeled_feature"
  | "semantic_hypothesis";

type ContributionEvidence = {
  kind: "contribution";
  lens_id: string;
  lens_run_id: string;
  episode_id: string;
  timestep?: number;
  policy_call?: number;

  basis:
    | "raw_activation_dimension"
    | "sae_feature"
    | "attention_head_output"
    | "token_state"
    | "action_dimension"
    | "custom";

  claim_level: EvidenceClaimLevel;

  items: {
    key: string;
    value: number;
    rank: number;
    sign?: "positive" | "negative";
    model_locus?: ModelLocusRef;
    label?: string;
    description?: string;
  }[];
};
```

Claim level is required.

Reason:

- It prevents the UI from implying that raw activation dimensions are semantic features.
- It lets the UI distinguish numeric contribution, grouped model locus, human-labeled feature, and semantic hypothesis.
- It encodes probe caveats in the contract instead of leaving them to component authors.

### ModelLocusEvidence

```ts
type ModelLocusEvidence = {
  kind: "model_locus";
  lens_id: string;
  lens_run_id: string;
  episode_id?: string;
  timestep?: number;
  policy_call?: number;
  locus: ModelLocusRef;
  source_label?: string;
};
```

Use cases:

- Model source/provenance panel.
- Right inspector source highlighting.
- Default model pipeline/map selection.

### CohortSummaryEvidence

```ts
type EvidenceCohortRef = {
  cohort_id: string;
  source: "ranking" | "filter" | "manual" | "saved";
  selection: ResearchSelectionState;
  count: number;
};

type CohortSummaryEvidence = {
  kind: "cohort_summary";
  lens_id: string;
  lens_run_id: string;
  cohort: EvidenceCohortRef;
  summary: Record<string, number>;
};
```

V1 can create transient cohorts from top, bottom, and uncertain shelves. Saved cohort management is v2.

### FailureCaseEvidence

```ts
type FailureCaseEvidence = {
  kind: "failure_case";
  lens_id: string;
  lens_run_id: string;
  ranking: "false_positive" | "false_negative" | "high_confidence_wrong";
  moments: RankedMomentsEvidence["moments"];
};
```

Failure case evidence is available only if labels, human annotations, success/failure events, or proxy targets exist.

## Unavailable Reasons

Do not silently omit panels. Omitted panels become invisible failure modes.

```ts
type UnavailableReason = {
  capability: LensCapability;
  panel_id?: string;
  reason:
    | "missing_scores"
    | "missing_labels"
    | "missing_contribution_basis"
    | "pooled_representation"
    | "missing_model_locus"
    | "unsupported_probe_type"
    | "not_computed";
  message: string;
};
```

Examples:

- Contribution breakdown unavailable because this probe exposes scores but not decomposed inputs.
- False positive browser unavailable because no labels or proxy targets exist.
- Model locus unavailable because this lens uses pooled layer activations.

## Contract 6: ResearchSelectionState

Question answered:

```text
What is the researcher currently pointing their brain at?
```

```ts
type ResearchSelectionState = {
  dataset_id?: string;
  lens_id?: string;
  lens_run_id?: string;

  episode_id?: string;
  timestep?: number;
  policy_call?: number;
  time_window?: {
    start: number;
    end: number;
  };

  ranking?: "top" | "bottom" | "uncertain" | "false_positive" | "false_negative";

  cohort_id?: string;

  model_locus?: {
    layer?: number;
    module?: string;
    stream?: string;
    head_index?: number;
    token_index?: number;
    channel_index?: number;
  };

  feature_id?: string;
};
```

Click behavior:

```text
Click top-ranked moment
  -> set episode_id
  -> set timestep or policy_call
  -> set lens_id
  -> set lens_run_id
  -> set ranking = "top"
```

Panel behavior:

- Episode list highlights the selected row.
- Video jumps to the selected frame.
- Timeline draws a vertical cursor.
- Details panel shows score, prediction, and contribution.
- Model panel shows source or locus if available.
- Pins panel can save this exact state.

A click should not directly mutate five panels. A click updates `ResearchSelectionState`, and panels derive their own view.

## Contract 7: PanelSpec

Question answered:

```text
Which panels are allowed to render for this probe evidence bundle?
```

```ts
type EvidencePrimitiveKind = EvidencePrimitive["kind"];

type PanelSpec = {
  panel_id: string;
  consumes: EvidencePrimitiveKind[];
  requires_capabilities: LensCapability[];
  requires_geometry?: Partial<LensGeometry>;
  unavailable_copy: string;
};
```

Example:

```ts
const ContributionPanelSpec: PanelSpec = {
  panel_id: "contribution_panel",
  consumes: ["contribution"],
  requires_capabilities: ["contribution_breakdown"],
  unavailable_copy: "Contribution breakdown is unavailable for this probe run.",
};
```

```ts
const ModelLocusPanelSpec: PanelSpec = {
  panel_id: "model_locus_panel",
  consumes: ["model_locus"],
  requires_capabilities: ["model_locus_view"],
  requires_geometry: { locus_kind: "model_locus" },
  unavailable_copy: "Model locus is unavailable for this probe run.",
};
```

Then pages call:

```ts
selectAvailablePanels(bundle.geometry, bundle.primitives, panelRegistry)
```

The page does not decide by vibes.

## Disposable Layer: Panel View Models

Panel view models are not durable contracts.

Example:

```ts
type TopMomentsPanelViewModel = {
  rows: {
    episode_id: string;
    timestep?: number;
    policy_call?: number;
    title: string;
    subtitle: string;
    score?: number;
    thumbnail_url?: string;
  }[];
};
```

Rule:

- Panel view models can change every week.
- Durable contracts should not change every week.
- Durable upstream contracts are `LensArtifact`, `LensRun`, `LensGeometry`, `ProbeEvidenceBundle`, `ResearchSelectionState`, and `PanelSpec`.

## Probe-Specific V1 Contract

The current research workflow is probe-centered. The first implementation should not attempt universal support for SAEs, transcoders, crosscoders, attribution, and interventions.

Start with:

```ts
type ProbeEvidenceContract = {
  lens_id: string;
  lens_version: string;
  lens_run_id: string;

  target_name: string;

  probe_type:
    | "linear"
    | "logistic"
    | "multiclass_linear"
    | "mlp"
    | "other";

  source: {
    model_id: string;
    layer?: number;
    module?: string;
    stream?: "residual" | "attention" | "mlp" | "head_output" | "embedding" | "custom";
    head_index?: number;
    token_scope?: "single_token" | "all_tokens" | "pooled" | "unknown";
  };

  geometry: LensGeometry;

  outputs: {
    score_series?: ScoreSeriesEvidence;
    predictions?: PredictionEvidence;
    top_moments?: RankedMomentsEvidence;
    bottom_moments?: RankedMomentsEvidence;
    uncertain_moments?: RankedMomentsEvidence;
    contribution?: ContributionEvidence;
  };

  unavailable?: UnavailableReason[];
};
```

The crucial field is not `probe_type`.

The crucial fields are:

```text
source
geometry.input_basis
geometry.temporal_scope
geometry.locus_kind
geometry.capabilities
```

These tell the UI what it is allowed to show.

## Probe Interpretation Caveats

For a linear probe:

```text
score = w dot h + b
```

A rough per-dimension contribution can be computed as:

```text
contribution_i = w_i * h_i
```

But raw activation dimensions are not automatically meaningful features.

The UI should distinguish:

```text
Model source:
  Where the input representation came from.

Probe contribution:
  Which input dimensions, features, or grouped sources pushed the score.

Semantic interpretation:
  What we think those contributors mean.
```

Examples:

- Probe trained on raw layer vector: can show top contributing dimensions, but should not call them semantic features.
- Probe trained on SAE features: can show top SAE features and may show feature descriptions if available.
- Probe trained on attention-head outputs: can group contributions by head.
- Probe trained on pooled representation: should not claim token/head specificity without extra attribution or a finer captured basis.

This distinction must be encoded in `ContributionEvidence.basis` and `ContributionEvidence.claim_level`.

## EpisodeLensAdapter

`ProbeEvidenceBundle` is the durable evidence contract. `EpisodeLensAdapter` is the concrete v1 bridge into the episode microscope.

```ts
type EpisodeLensAdapter = {
  family: "probe";
  defaultSelection(bundle: ProbeEvidenceBundle): Partial<ResearchSelectionState>;
  pipelineAnnotations(bundle: ProbeEvidenceBundle): PipelineLensAnnotation[];
  channelRanking(bundle: ProbeEvidenceBundle, selection: ResearchSelectionState): LensFeatureContribution[];
  timelineRows(bundle: ProbeEvidenceBundle): LensTemporalRow[];
  interventionSeed(bundle: ProbeEvidenceBundle, selection: ResearchSelectionState): InterventionLabSeed | null;
};
```

Probe adapter responsibilities:

- Highlight trained source layers/sites.
- Default the model inspector to the probe source site when useful.
- Compute or expose top probe contributors when weights and activations exist.
- Preserve raw activations as a secondary toggle beside lens-weighted contributors.
- Expose "Send to intervention" as a bridge if enough state exists, while intervention remains a separate workspace.

## Probe-Oriented Episode Microscope

When no probe is selected, the episode viewer answers:

```text
What happened in this episode?
```

When a probe is selected, the episode viewer should answer:

```text
How does this probe interpret this episode?
```

Required architecture decision:

```text
Active probe evidence is integrated into the existing right-side model inspector.
Do not create two equal tabs called "Model internals" and "Lens view".
Do not build a detached probe summary panel below the episode as the primary experience.
Do not create a separate episode lens page for v1.
```

Required behavior:

- Episode open state carries `activeLensArtifactId` and `lensRunId`, not only `probeId`.
- Probe source sites are highlighted/defaulted in the model pipeline/map.
- The right inspector shows compact probe readout: `Prediction`, `Actual`, `Confidence`, `Correct`, `Input`, `Policy call`, and `Model site`.
- Top contributor panel supports `Lens-weighted contributors` and `Raw activations`.
- "Send to intervention" is available as a bridge when possible, but intervention remains a separate workspace.
- The old below-episode probe panel is removed or demoted once the right inspector owns the job.

Primary panels when a probe is selected:

- Video or observation.
- Action trace.
- Probe score timeline.
- Selected timestep detail.
- Top moments within this episode.
- Model source and provenance.
- Contribution breakdown if available.
- Related moments across dataset if available.

At any selected timestep, the UI should answer:

- What did the robot see?
- What did it do?
- What did the probe score or predict?
- Was this high, low, uncertain, or threshold-crossing?
- Where did the probe input come from?
- What pushed the score up or down?
- Can those contributors be mapped to layer, head, token, SAE feature, or action dimension?
- What can we honestly claim?
- What can we not claim?

This is the concrete target behind "make the episode viewer probe-oriented."

## Dataset Tab as Lens Workbench

The dataset tab can become the Lens Workbench. A separate page is not required just because the mental model has a different name.

Shift the mental model from:

```text
Dataset -> episodes -> lens stats
```

to:

```text
Lens over dataset -> ranked evidence -> episode/moment drilldown
```

Sorting episodes is still useful. It becomes one view over a deeper object:

```text
ranked evidence under a selected lens
```

V1 should support two density modes if practical.

Compact mode:

- Rows of episodes or moments.
- Score badges.
- Prediction and split when available.
- Small timeline sparkline.
- Top timestep or policy call marker.
- Quick provenance badges.

Expanded mode:

- Selected episode row opens into a video strip.
- Selected episode row opens into a probe timeline.
- Selected episode row shows top contributing dimensions or features if available.
- Selected episode row shows model source or locus summary.
- Selected episode row has an "open in episode viewer" action.

This preserves the ability to scroll across many episodes without forcing the full episode viewer into every row.

## Evidence-Oriented API Shape

Avoid APIs named after pages.

Avoid:

```http
GET /dataset-tab-data
GET /episode-card-probe-stats
GET /model-viewer-row-data
```

Prefer evidence-oriented APIs or shared selector outputs:

```http
GET /api/lenses
GET /api/lenses/:lens_id
GET /api/datasets/:dataset_id/episodes

GET /api/lens-runs
GET /api/lens-runs/:lens_run_id

GET /api/lens-runs/:lens_run_id/manifest
GET /api/lens-runs/:lens_run_id/evidence
GET /api/lens-runs/:lens_run_id/moments?ranking=top&limit=50
GET /api/lens-runs/:lens_run_id/slice?episode_id=...&timestep=...
```

Important response shape:

```ts
type ProbeEvidenceResponse = ProbeEvidenceBundle;
```

Open design decision:

```text
ProbeEvidenceBundle may be built server-side, client-side, or in a shared typed selector layer.
Do not decide this until existing repo sources are inventoried.
```

## UI Invariants

These are implementation constraints from `docs/research_ui_principles.md` and the audit.

- Every visible region has one job.
- Do not add a panel because data exists.
- Do not repeat the same statistic in cards, bars, tables, and prose.
- Use researcher-native labels such as `Prediction`, `Input`, `Output`, `Objective`, `Split`, `Model site`, and `Policy call`.
- If a field explains what the current probe does, show it.
- If a field explains storage, provenance, dtype, paths, fingerprints, or backend audit details, hide it unless debugging.
- Missing capability copy must be explicit and quiet.
- Technical density is acceptable. Chrome density is not.
- Color should encode state or severity, not decoration.

## Mechanical Bloat Prevention Rules

### Rule 1: No new durable UI bucket without mapping to an evidence primitive

Before adding a new panel or interaction, ask:

```text
Is this a new primitive?
Or is this just a new rendering of ScoreSeries, RankedMoment, Contribution, ModelLocus, or Prediction?
```

Most things are new renderings, not new primitives.

### Rule 2: Panels declare requirements

Every research-facing panel gets a `PanelSpec`.

Then the page calls:

```ts
selectAvailablePanels(bundle.geometry, bundle.primitives, panelRegistry)
```

The page does not decide by vibes.

### Rule 3: Restrict raw capture imports

Research-facing UI components should not import raw capture schemas directly.

Allowed:

```text
UI -> evidence selectors -> ProbeEvidenceBundle
UI -> panel view models -> EvidencePrimitive
```

Forbidden in ordinary app UI:

```text
UI -> raw capture internals
UI -> arbitrary tensor metadata
UI -> ad hoc episode JSON traversal
```

Exception:

- Debug and dev panels can inspect raw capture.
- Debug panels should be visibly debug panels.

### Rule 4: Do not durably cache page-specific derived data

Durably cache:

- Raw captures.
- Lens artifacts.
- Lens runs.
- Large arrays and tensors.
- Evidence summaries.
- Ranked moment indexes, if expensive.

Do not durably cache:

- Episode card props.
- Dataset row props.
- Expanded row props.
- Timeline component props.
- Panel layout objects.

Those should be derived from canonical evidence.

## Research-Use Feedback Loop

During each implementation phase, use the tool on one real probe workflow and record:

- Question attempted.
- Click path.
- Missing evidence.
- Confusing panel.
- Wished-for comparison.
- Whether the contract changed.

Implementation is not complete until at least one real probe workflow has been exercised.

This keeps the architecture grounded in real research pain instead of hypothetical users.

## V1 Supported Scope

Supported:

- Scalar per timestep probe.
- Scalar per episode probe.
- Optional predictions.
- Optional thresholds.
- Top moments.
- Bottom moments.
- Uncertain moments.
- Source and provenance.
- Optional contribution breakdown.
- Capability-gated panels.
- Precise unavailable reasons.
- Shared research selection state.
- Right-inspector integration in the episode microscope.
- Minimal pinning as selected state plus evidence refs plus optional short note.
- Transient cohorts from top, bottom, and uncertain shelves.

Not supported yet:

- Causal interventions as a full workspace.
- Full generalization analysis.
- Clustering claims.
- Automatic semantic feature naming.
- Cross-lens causal claims.
- Universal SAE UI.
- Universal transcoder UI.
- Universal crosscoder UI.
- Full attribution UI.
- Saved cohort management.
- Full notes/lab-notebook system.
- Markdown editor.
- Artifact gallery.

Preserved for v2:

- Lens-vs-lens comparison.
- Layer-vs-layer probe comparison.
- Trained-vs-random/control probe comparison.
- Success-vs-failure cohort comparison.
- SAE feature workbench.
- Transcoder and crosscoder evidence.
- Attribution maps.
- Visual heatmaps.
- Intervention and counterfactual evidence.
- Generalization views across task, object, scene, and split metadata.
- Nearest-neighbor or clustering browsers.

## Phased Implementation Plan

### Phase 0: Inventory Existing Sources

Goal:

```text
Identify current repo objects before adding new contracts.
```

Tasks:

- Identify existing TraceManifest, workbench, model-site, index, and dataset metadata fields that already cover `CaptureAvailabilityView`.
- Identify current probe artifact fields that map to `LensArtifact`.
- Identify current probe result/run fields that map to `LensRun`.
- Identify current episode open state and selected probe state that map to `ResearchSelectionState`.
- Identify components that directly consume raw episode/probe payloads.
- Identify existing right-inspector/model-internals components that should become probe-aware.
- Identify existing below-episode probe panels that should be removed or demoted later.
- Do not refactor UI yet.

Exit criteria:

- Existing sources are mapped to the planned contracts.
- Duplicative schema risks are known.
- First real probe fixture is chosen as the canonical golden example.

### Phase 1: Define V1 Probe Evidence Types and Fixtures

Goal:

```text
Create the narrow waist without changing visible behavior.
```

Tasks:

- Add typed contracts for `LensGeometry`, `LensCapability`, `LensRun`, `ProbeEvidenceBundle`, `EvidencePrimitive`, `UnavailableReason`, `ResearchSelectionState`, and `PanelSpec`.
- Add probe-specific contract types.
- Add `EvidenceClaimLevel` and require it on contribution evidence.
- Add golden fixtures for scalar timestep, pooled layer, raw layer contribution, SAE feature contribution, and attention-head grouped probes.
- Keep large arrays referenced through refs rather than embedded in page payloads where possible.
- Do not add SAE, transcoder, crosscoder, attribution, or intervention implementations.

Exit criteria:

- Types exist.
- Fixtures exist.
- No research-facing UI depends on the new contract yet.
- The contract can represent current probe artifacts without pretending to support unavailable analyses.

### Phase 2: Build Backend or Shared Evidence Adapter

Goal:

```text
Translate existing probe artifact and trace/workbench metadata into ProbeEvidenceBundle.
```

Tasks:

- Build adapter from existing probe artifact plus trace/workbench/model-site metadata to `ProbeEvidenceBundle`.
- Validate bundle shape.
- Compute or expose `ScoreSeriesEvidence`.
- Compute top, bottom, and uncertain `RankedMomentsEvidence`.
- Expose `PredictionEvidence` where predictions/labels exist.
- For linear probes, compute weighted contributors when activation and weights exist.
- Emit explicit `UnavailableReason` entries for unsupported or missing capabilities.

Exit criteria:

- Existing probe data can produce a `ProbeEvidenceBundle`.
- Missing data produces explicit unavailable reasons.
- No fake panels are required to explain missing analyses.

### Phase 3: Build Selectors, Panel Specs, and Adapter Seams

Goal:

```text
Stop pages from deciding panel availability by vibes.
```

Tasks:

- Implement `selectAvailablePanels(...)`.
- Implement `selectTopMoments(...)`.
- Implement `selectCurrentMomentEvidence(...)`.
- Implement `selectContributionRows(...)`.
- Implement `selectContributionClaimLevel(...)`.
- Implement `selectUnavailableReasons(...)`.
- Define `PanelSpec` registry for provenance, score series, ranked moments, prediction, contribution, model locus, and unavailable panels.
- Define v1 `EpisodeLensAdapter` for probe bundles.

Exit criteria:

- Panels declare requirements.
- The workbench can tell the user why a panel is hidden or unavailable.
- Scalar timestep probes expose only panels they can honestly support.
- Episode microscope has a concrete adapter seam for active probe evidence.

### Phase 4: Dataset Page / Data Lens Integration

Goal:

```text
Make the dataset page lens-over-dataset-first.
```

Tasks:

- Preserve dataset selection.
- Preserve lens/probe selection.
- Resolve selected probe to a `LensRun` for the current dataset.
- Render probe summary from `ProbeEvidenceBundle`.
- Render ranked evidence under the selected probe.
- Add compact rows with score badges, prediction/split where available, timeline sparkline, top timestep/policy call marker, and quick provenance badges.
- Add expanded row mode if practical: video strip, probe timeline, contribution summary if available, model source/locus summary, and "open in episode viewer" action.
- Ensure row clicks update `ResearchSelectionState`.

Exit criteria:

- User can pick a dataset and probe.
- User can see top, bottom, and uncertain moments.
- User can click a moment and preserve lens context.
- The dataset tab behaves as a lens-conditioned episode browser.

### Phase 5: Episode Microscope / Right Inspector Integration

Goal:

```text
When a probe is selected, the episode microscope answers how the probe interprets the episode.
```

Tasks:

- Make episode open state carry `activeLensArtifactId` and `lensRunId`.
- Make episode viewer consume active `ResearchSelectionState`.
- Make the existing right-side model inspector lens-aware through `EpisodeLensAdapter`.
- Highlight/default probe source sites in the model pipeline/map.
- Show compact probe readout: `Prediction`, `Actual`, `Confidence`, `Correct`, `Input`, `Policy call`, and `Model site`.
- Add or adapt probe score timeline.
- Add selected timestep or policy call detail.
- Add top moments within the episode.
- Add contribution breakdown if available.
- Support `Lens-weighted contributors` and `Raw activations` views.
- Show precise unavailable reasons when contribution, model locus, labels, or source geometry are missing.
- Remove or demote detached below-episode probe panels once the right inspector owns the job.

Exit criteria:

- Opening from the workbench preserves dataset, lens, lens run, episode, and timestep/policy call context.
- The selected timestep updates all relevant panels.
- The page does not imply semantic feature claims unless `claim_level` supports them.
- The active probe constrains and annotates the existing model inspector instead of competing with it.

### Phase 6: Pins / Evidence Refs

Goal:

```text
Let the researcher preserve moments that support or challenge a probe claim.
```

Tasks:

- Define a pin payload based on `ResearchSelectionState`.
- Let the user pin selected moments.
- Include lens artifact id, lens run id, dataset id, episode id, timestep/policy call, model site, selected contributor if any, evidence primitive kind, score/prediction if available, claim level if available, and optional short note.
- Keep pins evidence-oriented, not page-layout-oriented.
- Do not build full notes, tagging, markdown editing, notebook pages, export, or artifact gallery in v1.

Exit criteria:

- A pinned item can reopen the same research state.
- Pins can represent top, bottom, uncertain, false positive, false negative, or manually interesting evidence when those concepts are available.
- Pin payload can evolve into an artifact/lab-notebook record later.

### Phase 7: Contract Tests and Regression Guardrails

Goal:

```text
Catch semantic drift before UI regressions become invisible.
```

Tasks:

- Test `selectAvailablePanels(...)` against all golden fixtures.
- Test `selectTopMoments(...)`.
- Test `selectCurrentMomentEvidence(...)`.
- Test `selectContributionRows(...)`.
- Test `selectContributionClaimLevel(...)`.
- Test `selectUnavailableReasons(...)`.
- Add capability-gated panel tests.
- Add an import-boundary rule or review checklist preventing research-facing UI from consuming raw capture directly except debug panels.
- Add a regression checklist for interaction contracts: select probe, select moment, select model site, select contributor, pin evidence.

Exit criteria:

- Scalar timestep probes expose score series, top moments, low moments, and provenance.
- Pooled layer probes do not expose token/head attribution without supporting evidence.
- Raw layer vector contribution panels use numeric-only or appropriate claim level.
- SAE feature probes can show feature-level contribution when feature metadata exists.
- Missing labels disable failure panels with a precise reason.
- Research-facing UI import boundaries are enforced by test, lint, or review policy.

### Phase 8: Deferred Extensions

Goal:

```text
Use the proven probe contract as the extension seam for later lens types.
```

Deferred candidates:

- SAE feature workbench.
- Transcoder and crosscoder evidence.
- Attribution maps.
- Visual heatmaps.
- Intervention and counterfactual evidence.
- Cohort comparison.
- Lens comparison.
- Failure cases from labels or proxy targets.
- Generalization views across task, object, scene, and split metadata.
- Nearest-neighbor or clustering browsers.

Rule:

```text
Each extension must add or reuse typed evidence primitives.
Each extension must declare geometry and capabilities.
Each extension must provide unavailable reasons instead of fake panels.
```

## Testing Strategy

More tests alone will not solve bucket rot. The tests need to check product promises.

Weak test:

```text
component renders without crashing
```

Useful contract test:

```text
Given scalar-per-timestep probe evidence:
  Workbench exposes:
    top moments
    low moments
    score series
    provenance

  Workbench does not expose:
    model head attribution
    visual heatmap
```

Create small golden fixtures:

- `probe_scalar_timestep`
- `probe_pooled_layer_no_contributions`
- `probe_raw_layer_vector_with_contributions`
- `probe_sae_feature_with_contributions`
- `probe_attention_head_grouped`
- `episode_success_small`
- `episode_failure_small`

Prioritize selector tests:

- `selectAvailablePanels(...)`
- `selectTopMoments(...)`
- `selectCurrentMomentEvidence(...)`
- `selectContributionRows(...)`
- `selectContributionClaimLevel(...)`
- `selectUnavailableReasons(...)`

Selector tests are higher leverage than React rendering tests because the primary failure mode is semantic transformation drift, not DOM failure.

## Open Audit Questions

Questions for GPT Pro and user audit:

- Does `CaptureAvailabilityView` duplicate an existing manifest/schema concept in the repo?
- Do existing probe artifacts already encode enough source geometry to fill `LensGeometry`?
- Are `LensCapability` values too broad, too narrow, or named incorrectly for current code?
- Should `PredictionEvidence`, `LensProvenanceEvidence`, `ModelLocusEvidence`, `CohortSummaryEvidence`, and `FailureCaseEvidence` be specified now or deferred until first use?
- Should `ProbeEvidenceBundle` be built server-side, client-side, or as a shared typed selector layer?
- What is the first real probe fixture that should become the canonical golden example?
- Does the current dataset tab already have the right host structure for Phase 4, or should the workbench be separated after all?
- Which existing below-episode probe panels should be deleted or demoted once the right inspector owns probe evidence?
- What existing UI buckets should be deleted once the evidence primitives exist?
- Is `policy_call` the right first-class time coordinate for current PI0.5/probe workflows, or should it remain an alias over timestep/frame?

## Non-Negotiable Implementation Constraints

- Do not build a broad universal lens IR before probe workflows are coherent.
- Do not let `ProbeEvidenceBundle` become a junk drawer.
- Do not let research-facing UI panels consume raw capture internals by default.
- Do not cache page-specific component props as durable data.
- Do not imply semantic meaning for raw activation dimensions.
- Do not expose failure-case panels unless labels, human annotations, success/failure events, or proxy targets exist.
- Do not add a panel because data exists.
- Do not duplicate the same statistic across cards, bars, tables, and prose.
- Do show precise unavailable reasons.
- Do make each click update shared research selection state.
- Do make every visible region answer one research job.
- Do integrate active probe evidence into the episode microscope/right inspector instead of creating a detached competing lens page for v1.
