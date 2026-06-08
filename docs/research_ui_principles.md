# Research UI Principles

Status: active design principles.

Last updated: June 7, 2026.

VLA-lens UI should help a researcher move from an observed episode behavior to a testable mechanistic hypothesis.

This document is also agent guidance. When changing the workbench UI, use these
principles as the local taste contract. Prefer a coherent screen-level pass over
reactive one-widget fixes.

## Product Frame

VLA-lens is a synchronized investigation workspace. The episode, timestep,
object, model site, and discovery artifact are shared state. Probes, SAEs,
attribution, patching, crosscoders, and interventions are lenses over that
state.

The dataset page is not a generic dashboard. It is a lens-conditioned episode
browser:

- Pick a lens.
- See what the lens claims or measures.
- Rank/filter episodes through that lens.
- Open an episode with the same lens context preserved.

The episode page is not separate from model internals. It is the microscope:

- Keep episode/video/timeline context visible.
- Show model/artifact readouts as synchronized inspectors.
- Avoid layouts that force the researcher to choose between seeing the episode
  and seeing the model state that explains it.

## Default Surface

- Keep controls, selections, comparisons, and drilldowns visible.
- Hide passive metadata behind disclosures unless it changes a decision.
- Prefer split-aware trust language: train, validation, test, or split missing.
- Treat train-episode probe evidence as a sanity check, not as generalization evidence.
- Design probe views for hundreds or thousands of probes: search, rank, filter, and summarize before listing raw rows.

Default surface rule:

- If a field explains what the current lens does, show it.
- If a field explains how the lens behaves over the dataset, show it in the
  split/result maps.
- If a field only explains provenance, storage, schema, dtype, file paths, or
  audit strings, hide it unless the user is debugging the artifact itself.
- If a field repeats another nearby panel, remove it or move it to the panel
  that owns that job.

## Research Loop

The main loop is:

1. Find an interesting behavior or cohort.
2. Select the artifact that claims to explain it.
3. Check whether that claim holds on validation/test episodes.
4. Jump from cohort evidence to the episode microscope.
5. Inspect the mapped model site, layer, call, feature, token, or attention patch.
6. Turn the observation into a patching, steering, or counterfactual test.

## Page Jobs

Every visible region should have one job.

Dataset lens selector:

- Answers: "What lens am I using to rank the dataset?"
- Shows: none, probe X, SAE feature Y, attribution run Z, etc.
- Avoid: duplicating probe summary cards or showing raw family registry details.

Selected lens/probe summary:

- Answers: "What is this lens doing?"
- Shows concise ML-spec fields such as `Prediction`, `Input`, `Output`, and
  `Objective`.
- Avoid: `Trust`, `Heldout wrong`, `Visible scored`, and other statistics that
  are already better represented by split/result maps.

Split/result maps:

- Answer: "How does this lens behave over the dataset?"
- Show correct, wrong, high-confidence wrong, scored, unscored, and split counts.
- Use color semantically: correct/scored, wrong, high-confidence wrong,
  selected, unavailable.
- Avoid prose summaries that restate the same numbers in another panel.

Episode table:

- Answers: "Which episodes should I inspect next?"
- Shows sortable/rankable episode rows with the active lens result.
- Avoid generic database columns unless they support filtering, sorting, or
  episode selection.

Episode microscope:

- Answers: "What happened in this episode, and what does the active lens say at
  this timestep/model site?"
- Keep video/timeline/model/artifact state synchronized.
- Avoid detached model-internals pages that lose episode navigation context.

## Probe Evidence

Probe UI should answer:

- What question is this probe answering?
- Is this episode train, validation, test, or split missing for the probe?
- Where in the model is the probe reading from?
- Which episodes are most worth inspecting because the probe is wrong, confidently wrong, heldout, or ambiguous?
- Does the probe support a research claim, or is it only a debugging sanity check?

Avoid spending default space on raw tensor shapes, dtype, exact float storage details, or static bookkeeping unless they directly explain what a researcher can do next.

## Human-Readable Probe Specs

Probe summaries should read like a short ML spec, not like artifact JSON.

Good default fields:

- `Prediction`: Target contacted
- `Input`: Expert hidden states
- `Output`: False / True
- `Objective`: Logistic regression

Good details:

- action tokens
- layers 0, 4, 8, 12, 17
- final step
- metric: balanced accuracy

Bad default fields:

- feature matrix fingerprint
- row index fingerprint
- dtype float16
- exact cache keys
- long audit/provenance strings
- "selected lens trust" when the split map already shows the evidence

If the researcher must understand the concept to use the page, do not hide it
behind a disclosure. If the researcher only needs it for debugging or
reproducibility, hide it in details or move it to an artifact/debug view.

## Visual Taste Rules

Prefer technical density, not chrome density.

Do:

- Use fewer bordered cards.
- Use typography, spacing, and alignment to group information.
- Keep labels short and concrete.
- Make selected state obvious.
- Align temporal traces to the episode timeline.
- Use tables when researchers need sorting, filtering, scanning, and comparison.
- Use hover/focus affordances lightly; default surfaces should usually be quiet.

Avoid:

- Bordering every small fact.
- Adding a panel because data exists.
- Making hidden information loud just to make the disclosure discoverable.
- Using "details" as a dumping ground for fields that actually belong in the
  default concept model.
- Repeating the same statistic in cards, bars, and prose.
- Decorative color. Color should mean state or severity.

Severity color guidance:

- Correct/scored: calm green or blue-green.
- Wrong: orange.
- High-confidence wrong: red.
- Selected/active: restrained gold.
- Missing/unavailable: gray.

## Copy Rules

Use researcher-native, short labels. Prefer:

- Prediction
- Input
- Output
- Objective
- Split
- Correct
- Wrong
- High-conf wrong
- Model site
- Policy call

Avoid vague or UI-internal labels:

- Selected lens trust
- Visible scored
- Coverage
- Current diagnostics
- Lens registry unavailable
- Artifact metadata
- Details

If a label needs a paragraph to explain why it exists, the UI concept is likely
wrong or the field belongs somewhere else.

## Progressive Disclosure

Use disclosure only when the hidden content is secondary.

Good disclosure targets:

- raw rows
- provenance
- exact storage paths
- fingerprints
- long audit strings
- backend debug payloads
- advanced filters that are not part of the common research loop

Bad disclosure targets:

- what the probe predicts
- what the input features are
- what the output space is
- what objective trained the probe
- which episode/timestep/model site is selected

When a disclosure is used, its collapsed label should say what will open. Do not
make it visually loud by default; use a quiet default and clear hover/focus
state.

## Counterfactual Direction

Counterfactual artifacts should record the source episode, intervention site, intervention time/call, method, replacement or steering source, resulting trajectory, and comparison metrics. The UI should show original and intervened episodes side by side with synchronized timeline, changed probe readouts, and changed object/action behavior.

## Agent Checklist For UI Changes

Before editing UI code, answer:

1. What job does this screen region perform?
2. Is this information concept, behavior, navigation, or debug/provenance?
3. Is the same information already shown more clearly nearby?
4. Does this label use human ML/research language instead of backend language?
5. Does color encode state or severity?
6. Does the default view show what the researcher needs to decide the next
   action?
7. If the change affects layout, did you inspect the running UI or screenshot
   instead of relying only on code?

When in doubt, remove redundancy before adding another panel.
