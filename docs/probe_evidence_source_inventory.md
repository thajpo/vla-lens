# Probe Evidence Source Inventory

This is the implemented-source inventory for `docs/probe_evidence_contract_phased.md`.
It records which existing repo objects feed the v1 probe evidence contract and
which older UI buckets remain as bridges.

## Canonical source mapping

| Planned contract | Existing source | Current adapter or consumer |
| --- | --- | --- |
| `CaptureAvailabilityView` | `TraceManifest`, `TraceDataset`, `ModelSiteSpec`, and dataset index tables | `src/vla_lens/traces/`, `src/vla_lens/dataset/index.py`, server indexed APIs |
| `LensArtifact` | Existing durable artifact records | `src/vla_lens/artifacts.py`, artifact index rows consumed by `src/vla_lens/probe_evidence_adapter.py` |
| `LensRun` | Indexed probe result application to a dataset | `probe_evidence_bundle_from_index(...)` derives `indexed:{lens_id}:{dataset_id}` run ids |
| `LensGeometry` | Probe artifact method/metrics plus indexed prediction/model-site rows | `_lens_geometry(...)` in `src/vla_lens/probe_evidence_adapter.py` |
| `ProbeEvidenceBundle` | Artifact index, probe predictions, probe episode index, model-site index, episode index | `src/vla_lens/probe_evidence_adapter.py` |
| `ResearchSelectionState` | Dataset browser moment context, episode route context, pin payload | `frontend/src/pages/workbench/datasetBrowserModel.ts`, `frontend/src/pages/workbench/episodeRouteModel.ts`, `frontend/src/pages/evidencePinsModel.ts` |
| `PanelSpec` | Probe evidence selector registry | `src/vla_lens/probe_evidence.py`, `frontend/src/types/probeEvidence.ts` |
| `EpisodeLensAdapter` | Probe evidence bundle selector seam into episode microscope | `frontend/src/types/probeEvidence.ts`, `frontend/src/pages/episodes/episodeLensModel.ts` |

## Existing metadata that already covers availability

Do not add a new durable capture metadata source unless these are insufficient:

- `TraceManifest`: episode/camera/action/model-site storage metadata.
- `TraceDataset`: common query surface over LeRobot root plus `vla_lens/` overlay.
- Dataset indexes: fast dashboard tables for episodes, artifacts, probes, predictions, and model sites.
- Capability manifest / capability gating: frontend query availability and backend API availability.

`CaptureAvailabilityView` remains a planned read model over these sources, not a
new source of truth.

## Probe artifact fields that map to `LensArtifact`

Existing `LensArtifact` / artifact index rows provide:

- artifact id and type.
- display name.
- created timestamp.
- `method` payload for probe schema, source/input/target/split/evaluation.
- `metrics` payload for target, source model/layer/module, split metrics, and score summaries.
- array refs for predictions, metrics, and optional contribution artifacts.

The v1 `ProbeLensArtifact` intentionally narrows this to provenance needed by
research-facing UI.

## Probe result fields that map to `LensRun`

The indexed adapter derives a run-level reference from:

- selected `probe_id`.
- selected `dataset_id`.
- artifact version / created timestamp.
- available prediction rows and episode rows.
- result version `probe_evidence.indexed.v1`.

This distinguishes "probe artifact exists" from "probe has aligned evidence for
this dataset."

## Current selection-state sources

Selection state is now centered on `ResearchSelectionState`:

- Dataset browser ranked rows build episode open contexts from selected
  `ProbeEvidenceBundle` moments.
- Episode route hashes preserve dataset, lens, run, episode, timestep/policy
  call, ranking, model site, and contributor.
- Pins persist evidence references using selection state plus evidence metadata.

This replaces direct mutation of independent page widgets as the durable
research interaction model.

## Raw-schema bridge inventory

Research-facing UI still has legacy bridges that import dataset/raw capture
types. These are explicitly allowlisted and tested in
`tests/research_ui_import_boundary_test.py`.

Bridge categories:

- Episode microscope data loading and playback.
- Activation-site/model-pipeline rendering.
- Legacy demoted probe readout.
- Intervention episode/artifact picker.
- Debug panels.

New raw-schema imports in research-facing UI should fail the import-boundary
test unless they are intentionally added to the bridge allowlist with a reason.

## Duplicative schema risks

- `EpisodeLensView` is still a legacy probe readout shape. It is allowed only as
  a bridge into `ProbeEvidenceBundle` selectors.
- `DatasetEpisode`, `ProbeDatasetIndex`, and `ProbeEpisodeIndex` remain useful
  page/API payloads but should not become new durable evidence contracts.
- Panel view models in dataset and episode pages are disposable. They should be
  derived from evidence primitives instead of persisted or cached as durable
  objects.
- Raw activation dimensions are numeric contributors unless a stronger
  `claim_level` is present.

## Canonical golden probe workflow

The first canonical workflow is:

```text
raw_layer_contribution fixture
  -> probe-grasp-intent / run-probe-grasp-intent
  -> top ranked moment episode-1, policy_call 3
  -> model locus action_head.layers.8.resid
  -> contributor dim_42
  -> claim_level numeric_only
```

The corresponding workflow exercise is recorded in
`docs/probe_evidence_workflow_feedback.md`.
