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

The broad PI0.5 probe campaign found that stronger metadata baselines explain
the pooled binary probe candidates as well as or better than the activations.
The geometry campaign was mostly negative or diagnostic. Object-local `z` may
justify one controlled methodological confirmation, but it is not currently an
intervention target.

The full scientific record, including null results, remains in
[pi05_broad_1000_probe_experiments.md](pi05_broad_1000_probe_experiments.md).
Those results are history, not an implementation plan.

## Remaining Work

The shared research-data work is tracked in GitHub:

- [#14: dataset-level policy-call index](https://github.com/thajpo/vla-lens/issues/14)
- [#15: exact example manifests](https://github.com/thajpo/vla-lens/issues/15)
- [#16: reusable experiment recipes](https://github.com/thajpo/vla-lens/issues/16)
- [#17: unified selection and exact source-example drilldowns](https://github.com/thajpo/vla-lens/issues/17)

New SAE, transcoder, attribution, clustering, or broad comparison surfaces are
deferred until this shared evidence spine is reliable.

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
