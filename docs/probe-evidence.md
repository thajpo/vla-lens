# Probe Evidence

Status: implemented observational evidence contract.

## What This Part Of VLA Lens Does

A probe is a lens over a dataset. It asks whether a recorded model
representation predicts a defined target, then helps the researcher inspect the
examples supporting or challenging that result.

The normal research loop is:

```text
select a dataset and probe
-> inspect high, low, and uncertain examples
-> open an episode with probe context preserved
-> inspect prediction, score, source site, and contributors when supported
-> pin an exact evidence state
```

## What Works Now

VLA Lens can:

- train probe suites from typed selectors and targets;
- compare activation probes with metadata-only baselines;
- save predictions, metrics, source geometry, and provenance;
- adapt saved probe artifacts into one evidence bundle;
- rank high, low, uncertain, and failure examples;
- preserve dataset, run, episode, policy-call, timestep, site, and contributor
  selection;
- capability-gate panels and explain why evidence is unavailable;
- show probe evidence in the dataset browser and Episode Lens inspector;
- save evidence pins that reopen the same research state.

## What The Contract Protects

`ProbeEvidenceBundle` is the narrow boundary between saved probe data and the
research UI. Pages should not invent their own durable probe payloads.

`LensGeometry` describes what the probe score or contribution is indexed by,
such as episode, policy call, timestep, layer, token, head, or feature.

`ResearchSelectionState` carries the active dataset, lens run, episode, moment,
model site, and contributor across views.

`PanelSpec` declares the evidence and capability a panel requires. Missing
evidence produces an explicit unavailable reason instead of an empty or
misleading visualization.

Large arrays stay behind references. UI-specific rows and chart models are
derived and disposable.

## Interpretation Rules

- Decodability is not causality.
- A probe that loses to metadata baselines is diagnostic, not mechanistic.
- Raw activation dimensions do not have semantic names merely because a linear
  probe assigns them weight.
- Contribution claims must match the representation geometry.
- Failure-case views require real labels, annotations, events, or an explicit
  proxy target.
- Every aggregate result should link back to exact source examples.

## Current Research Result

The broad pooled PI0.5 probes did not reveal a reliable held-out object-position
or pose code beyond stronger physical and metadata baselines. A later
known-region study found exploratory object-identity information when the
correct simulator region was supplied, but its probe direction did not beat
semantic-specificity controls in an intervention. These results constrain the
methods tested; they do not show that PI0.5 lacks object binding generally.

The full scientific record, including null results, remains in
[pi05_broad_1000_probe_experiments.md](pi05_broad_1000_probe_experiments.md).
Those results are history, not an implementation plan.

## Remaining Work

The current work is the
[controlled scene-to-behavior campaign](autonomous-research-campaigns.md).
Probe methods remain useful for its readable and reusable-information tests,
but a positive probe does not skip the behavior, control, and confirmation
gates.

## Code Map

- `src/vla_lens/probe_evidence.py`: evidence types, selectors, and panel
  availability.
- `src/vla_lens/probe_evidence_adapter.py`: saved artifact to evidence-bundle
  conversion.
- `src/vla_lens/probes/`: training, baselines, diagnostics, and persistence.
- `frontend/src/types/probeEvidence.ts`: frontend contract and derived view
  helpers.
- `frontend/src/pages/workbench/` and `frontend/src/pages/episodes/`: dataset and
  episode integrations.
- `tests/probe_evidence_*_test.py`: contract, selector, and adapter coverage.
