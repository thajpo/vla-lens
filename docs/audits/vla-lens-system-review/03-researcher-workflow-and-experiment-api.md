# 03 - Researcher Workflow And Experiment API

## Inspection Metadata

- Inspected commit: `882eeb83be8c8a69a80fc8f6ec8829c311ca4630`
- Pre-write git status: clean. `git status --short` and `git status --porcelain=v1` printed no rows before this report was created.
- Post-write git status: `?? docs/audits/` from `git status --short`; this is the audit report tree containing this owned file.
- Expanded final status: `git status --short --untracked-files=all` also showed sibling audit files `02-data-storage-and-indexing.md`, `04-capture-and-model-execution.md`, `06-evidence-interventions-and-method-extensions.md`, and `09-questions-for-owner.md`. Worker 03 did not edit those files.
- Scope: static inspection only. I did not run captures, simulators, model downloads, tests, `uv run`, or PI0.5/LeRobot/LIBERO runtime commands.
- Owned file: `docs/audits/vla-lens-system-review/03-researcher-workflow-and-experiment-api.md`.

Commands used:

- `pwd`
- `nl -ba AGENTS.md`
- `nl -ba /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt`
- `sed -n '521,760p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt`
- `sed -n '761,927p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt`
- `rg --files`
- `git rev-parse HEAD`
- `git status --short`
- `git status --porcelain=v1`
- `git status --short --untracked-files=all`
- `wc -l docs/audits/vla-lens-system-review/03-researcher-workflow-and-experiment-api.md`
- `sed -n '1,80p' docs/audits/vla-lens-system-review/03-researcher-workflow-and-experiment-api.md`
- `rg -n "Current Probe Workflow|What A Training Example|Target And Feature|Minimal Serializable|Researcher-Facing|Friction Report|Recommended Next" docs/audits/vla-lens-system-review/03-researcher-workflow-and-experiment-api.md`
- Static search commands with `rg -n` over `src/vla_lens`, `scripts`, `configs/probes`, `docs`, and `frontend/src`.
- Line-numbered reads with `nl -ba ... | sed -n ...` for the source/config files cited below.

## Executive Finding

The current probe workflow is usable if the researcher already knows the hidden path: write or clone a YAML spec, run probe preflight, train, optionally run the PI0.5 diagnostics script, refresh/build dashboard indexes if needed, then inspect one of several probe-specific UI surfaces. The core backend has strong reusable pieces: `TraceDataset`, `ActivationQuery`, materialized feature rows, `LensArtifact`, dashboard indexes, workbench selection/cohort records, and runtime-free intervention contracts. The missing layer is a small, serializable "experiment recipe" and "example manifest" abstraction that joins these pieces into one auditable research object.

The most important architectural distinction is:

- General today: dataset indexing, model-site addressing, activation selection, artifact storage, workbench selection/cohort records, and intervention target/result contracts.
- Probe-specific today: target resolution assumes scalar supervised labels, row expansion is object-role only, training is sklearn probe only, diagnostics and study UI are probe-specific, and intervention seeding from discovery artifacts mostly works for `probe_suite` artifacts.
- Missing today: a first-class experiment API that says "this is the population, these are the examples, these are the signals/targets, this is the method runner, these are the saved outputs" across probes, SAEs, transcoders, contrast directions, and steering-discovery workflows.

## Current Probe Workflow As Usability Test

| Researcher step | Existing command/API | File and symbol | Data structure today | Manual glue / assumptions / failure modes |
|---|---|---|---|---|
| 1. Choose dataset | CLI root argument, or `TraceDataset.open(root)` | `scripts/train_vla_lens_probe.py:21-25`, `src/vla_lens/traces/dataset.py:40-61` | LeRobot root or nested batch root; `TraceDataset` exposes episode/model/artifact indexes. | Researcher must already know the dataset root and whether indexes are current. No single "experiment workspace" handle. |
| 2. Find eligible episodes and policy calls | Preflight or feature selector materialization | `scripts/preflight_vla_lens_probe.py:43-52`, `src/vla_lens/probes/preflight.py:37-83`, `src/vla_lens/selectors.py:190-218` | `episode_index`, `model_site_index`, `policy_calls` rows, feature rows. | Preflight reuses materialization and can write `.vla_cache`; it is not just an index-only support estimate. Eligibility is discovered by running the selector, not by a researcher-facing catalog query. |
| 3. Derive/select target | YAML `target:` or CLI `--target`; private target resolver | `configs/probes/pi05_broad_1000_target_lifted_expert_action_hidden.yaml:6-8`, `src/vla_lens/probes/workflow_targets.py:14-24`, `src/vla_lens/probes/workflow_targets.py:31-61` | Target spec with `name/kind/source/selector/alignment/transform/missing_policy`; result is a scalar column in `rows`. | CLI only handles simple target kind/column. Rich sources such as `scene_state`, `array`, `robot_state`, alignment offsets, and missing policy require YAML and source-code knowledge. |
| 4. Select model signal | YAML `features:` or `ActivationQuery` | `src/vla_lens/selectors.py:21-42`, `configs/probes/pi05_broad_1000_target_lifted_expert_action_hidden.yaml:9-18` | `ActivationQuery` fields: episodes, name, module, layers, tensor_type, token_kind, timesteps, policy_calls, generation_step, reduce_tokens, dtype. | This piece is genuinely general for activation-like arrays, but discovery of legal modules/layers/tokens is indirect through indexes/UI. |
| 5. Select site/layer/token/time | Same feature spec | `src/vla_lens/selectors.py:190-218`, `src/vla_lens/selectors.py:221-248`, `src/vla_lens/selectors.py:282-305`, `src/vla_lens/selectors.py:364-373` | Filtered `model_site_index`, selected timestep or policy-call sample, selected token kind, selected generation step. | Token selection is by token kind and then optional pooling; there is not yet an explicit token-index/example axis in the high-level spec. |
| 6. Transform/pool | `features.reduction`, target transform | `src/vla_lens/selectors.py:260-279`, `src/vla_lens/probes/workflow_targets.py:78-107` | `mean`, `flat`, or `none` over token axis; target transform supports identity or numeric threshold. | Feature transforms are minimal. SAEs/transcoders need method-specific normalization/windowing, not just token pooling. |
| 7. Construct input rows | `dataset.select_model_sites(selector).materialize(cache=True)` | `src/vla_lens/selectors.py:71-98`, `src/vla_lens/selectors.py:112-188` | `FeatureMatrix(X, rows, selector, cache_key)`. Rows include `trace_id`, `timestep`, `policy_call_index`, `model_site_id`, `layer`, `tensor_type`, `token_kind`, `generation_step`, `feature_dim`. | This is a good narrow waist for vector features. The row unit is "selected activation row", not a general experiment example with full manifest. |
| 8. Attach metadata, targets, row expansion | Training workflow internals | `src/vla_lens/probes/workflow_training.py:83-105`, `src/vla_lens/probes/workflow_prepare.py:21-37`, `src/vla_lens/probes/workflow_prepare.py:730-811` | Feature rows are merged with episode metadata, split sidecars, object-flow labels, policy-call labels, target-role columns, and optional object-role expansion. | Metadata attachment is private, PI0.5-heavy, and latest-artifact based. Object expansion duplicates `X` rows for candidate objects, but only supports `object_roles`/`scene_objects`. |
| 9. Define grouped splits | YAML `split:` plus `_ensure_split` | `configs/probes/pi05_broad_1000_target_lifted_expert_action_hidden.yaml:19-26`, `src/vla_lens/probes/workflow_prepare.py:574-596`, `src/vla_lens/probes/workflow_prepare.py:814-851` | Split column values, train/selection/test/eval values, heldout split kind. | If the split column exists with any non-null value, `_ensure_split` returns rows as-is. Partial split coverage can sneak through until training/eval support warnings. |
| 10. Train probe | CLI or Python API | `scripts/train_vla_lens_probe.py:76-139`, `src/vla_lens/probes/workflow_training.py:63-81`, `src/vla_lens/probes/workflow_training.py:113-127`, `src/vla_lens/probes/suite.py:66-128` | `run_probe_suite` fits classification/regression probes over feature groups and eval splits. | Runner is supervised sklearn only. `_classification_result` skips readouts with empty eval or fewer than two train classes (`src/vla_lens/probes/suite.py:144-147`), which is correct but can surprise if preflight was skipped. |
| 11. Save result | `LensArtifact` plus parquet/zarr outputs | `src/vla_lens/probes/workflow_training.py:153-173`, `src/vla_lens/probes/workflow_training.py:175-251`, `src/vla_lens/probes/workflow_training.py:278-351`, `src/vla_lens/artifacts.py:33-50` | Artifact type `probe_suite`, method metadata, metrics, predictions, scored predictions, per-split/group/null metrics, weights/bias/normalizer arrays. | Strong provenance, but no full example manifest table is saved. `_probe_examples` saves counts and a fingerprint, not all selected rows (`src/vla_lens/probes/workflow_artifacts.py:154-182`). |
| 12. Expose backend | FastAPI routes, dataset indexes, score cache | `src/vla_lens/server/fastapi_app.py:392-450`, `src/vla_lens/dataset/index.py:173-246`, `src/vla_lens/dataset/index.py:387-468`, `src/vla_lens/probes/score_cache.py:61-146` | `/api/probe-index`, `/api/probe-studies`, `/api/probes/{id}/evidence`, `/api/probes/{id}/evidence-bundle`, indexed prediction and episode tables. | Training rebuilds indexes, but new compatible episodes require mutable score-cache refresh. Index rows also omit `example_id` even though prediction records contain it. |
| 13. Inspect UI | Dataset browser probe mode and ProbeSuite preset | `frontend/src/pages/workbench/DatasetBrowser.tsx:120-160`, `frontend/src/pages/workbench/DatasetBrowser.tsx:311-329`, `frontend/src/pages/workbench/DatasetBrowser.tsx:482-530`, `frontend/src/components/workflows/ProbeSuitePreset.tsx:79-155` | Probe lens selector, readout selector, episode ranking, workbench model, probe studies. | UI has multiple overlapping surfaces: indexed probe evidence, probe studies, and discovery artifact lens views. Non-probe discovery families mostly report "not supported yet" for episode ranking/readouts. |
| 14. Link correct/incorrect examples | Prediction parquet, probe index, diagnostics error browser | `src/vla_lens/probes/suite.py:595-648`, `src/vla_lens/dataset/index.py:408-443`, `src/vla_lens/server/probe_studies.py:140-220`, `frontend/src/components/workflows/ProbeSuitePreset.tsx:553-599` | Prediction records have `example_id`; index rows expose trace/policy call/layer/site/confidence/correct; diagnostics error browser links trace and policy call. | The dashboard index drops `example_id`; diagnostics error rows are separate files and are saved only by the diagnostics script, not by core training. |
| 15. Compare another run | Batch campaign artifact and UI run/readout selectors | `scripts/run_vla_lens_probe_batch.py:57-120`, `scripts/run_vla_lens_probe_batch.py:163-213`, `frontend/src/components/workflows/ProbeSuitePreset.tsx:155-291` | `probe_campaign` artifact summarizes spec runs; UI can switch studies/readouts and sort by metrics. | There is no normalized experiment-compare API across methods or across recipes. Campaigns summarize probe artifacts but do not encode a reusable comparison plan. |
| 16. Use as source for intervention | Discovery-artifact target endpoint, intervention lab/preflight | `src/vla_lens/interventions/families.py:83-185`, `src/vla_lens/interventions/families.py:211-273`, `src/vla_lens/server/discovery_artifacts.py:171-208`, `frontend/src/components/interventions/interventionLabModel.ts:7-59`, `src/vla_lens/interventions/preflight.py:1-6` | `TargetSpec` candidates, operator/outcome/control families, inspected intervention records. | Good runtime-free contract, but live execution is intentionally outside normal env. Probe episode lens can seed interventions (`src/vla_lens/server/episode_lens_probe.py:628-679`); `ProbeEvidenceBundle` itself returns no intervention seed in its adapter path (`src/vla_lens/probe_evidence.py:1089-1095`). |

## What A Training Example Means Today

The current probe artifact declares the example unit as `selected_activation_row` and defines row construction as "one row per selected trace/model_site/sample after selector reduction" (`src/vla_lens/probes/workflow_artifacts.py:154-167`). Concretely:

- If the selected tensor has a `policy_call` axis, one base example is one selected policy-call slice of one model site. `_resolve_samples` emits one `(policy_call, index)` item per selected call (`src/vla_lens/selectors.py:293-304`).
- If the tensor has a `timestep` axis, one base example is one selected timestep slice, with best-effort mapping back to `policy_call_index` (`src/vla_lens/selectors.py:290-292`, `src/vla_lens/selectors.py:351-361`).
- If neither sample axis exists, one base example is one selected model-site array (`src/vla_lens/selectors.py:305`).
- `token_kind` selects token indices from token metadata, and `reduce_tokens` then either averages over the token axis, flattens, or requires an already-vector value (`src/vla_lens/selectors.py:244-248`, `src/vla_lens/selectors.py:260-279`, `src/vla_lens/selectors.py:364-373`).
- Object-local geometry probes can expand one activation row into several candidate-object examples by joining the latest object-role table and duplicating `X` rows (`src/vla_lens/probes/workflow_prepare.py:730-811`). Example identity then includes object columns such as `probe_object_name` in prediction metadata (`src/vla_lens/probes/suite.py:627-647`).
- The model sees a 2D feature matrix `X[row, feature_dim]`. The researcher sees rows, prediction records, and metrics, but the exact selected row manifest is not saved as a first-class artifact table. `_probe_examples` stores counts, split counts, an example-id definition, and a row fingerprint (`src/vla_lens/probes/workflow_artifacts.py:160-182`).

This is acceptable for linear probes, but it is not yet expressive enough for SAE/transcoder/steering workflows where an "example" may be an activation stream item, paired source/target activations, a contrast pair, a token span, or an intervention candidate across a cohort.

## Target And Feature Extraction

Feature extraction is the strongest reusable piece. `ActivationQuery` is a compact serializable address for activations (`src/vla_lens/selectors.py:21-42`). `FeatureView.materialize` caches rows and `X.zarr` by hashing the selector, episode trace ids/lengths, and model-site storage signatures (`src/vla_lens/selectors.py:71-107`). `_compute` records enough model-locus metadata for dashboard linkage (`src/vla_lens/selectors.py:112-188`).

Target extraction is useful but more probe-shaped. `_resolve_probe_target` handles existing row columns and then delegates to `_resolve_target_value` for saved evaluation metrics, bundle tables, arrays, action arrays, robot state, and scene state (`src/vla_lens/probes/workflow_targets.py:31-61`, `src/vla_lens/probes/workflow_targets.py:110-168`). Array targets are forced to scalar labels or scalar regression values (`src/vla_lens/probes/workflow_targets.py:245-315`). Alignment is based on selected row timestep or `target_timestep` plus an offset (`src/vla_lens/probes/workflow_targets.py:171-179`). This should be kept for scalar supervised labels, but a general experiment API should wrap it as one target resolver among several, not make it the global target interface.

The current YAML examples show what researchers actually have to write:

- Simple episode-level label: `target_lifted` with hidden action features, heldout task split, metadata baselines, and sweep over `layer, policy_call_index` (`configs/probes/pi05_broad_1000_target_lifted_expert_action_hidden.yaml:1-37`).
- Object-local regression: `scene_state` target, `row_expand.kind: object_roles`, candidate object column, and object-role baselines (`configs/probes/pi05_broad_1000_all_object_position_x_expert_action_hidden.yaml:6-47`).

These examples are good research configs, but they are probe specs, not general experiment recipes.

## General Pipeline Versus Probe-Specific Pipeline

| Pipeline part | Status | Evidence |
|---|---|---|
| Dataset and trace access | General | `TraceDataset.open`, `episode_index`, `model_site_index`, `artifact_index`, and `select_model_sites` are method-neutral (`src/vla_lens/traces/dataset.py:40-131`). |
| Activation/signal selection | Mostly general | `ActivationQuery` and `FeatureMatrix` can feed probes, SAEs, transcoders, contrast directions, and steering discovery if those methods accept vector rows (`src/vla_lens/selectors.py:21-53`). Needs streaming/chunking for large unsupervised methods. |
| Example population | Missing general abstraction | The probe workflow builds rows internally, and saved artifacts keep only counts/fingerprints (`src/vla_lens/probes/workflow_training.py:83-105`, `src/vla_lens/probes/workflow_artifacts.py:154-182`). |
| Metadata attachment | Useful but PI0.5/probe-specific | `_attach_episode_metadata` merges sidecars, interaction metrics, object-flow, policy-call labels, and temporal target columns (`src/vla_lens/probes/workflow_prepare.py:21-37`). |
| Target resolution | Semi-general, supervised-scalar | Supports many saved data sources, but returns one scalar target column and errors on vector targets (`src/vla_lens/probes/workflow_targets.py:245-315`). |
| Splits and filters | Semi-general | Row filters and grouped split kinds are reusable (`src/vla_lens/probes/workflow_prepare.py:599-674`, `src/vla_lens/probes/workflow_prepare.py:814-851`), but not exposed as an `ExampleSet` API. |
| Method runner | Probe-specific | `_run_sweep` and `run_probe_suite` assume `rows`, `X`, scalar target, sklearn models, and row-level evaluation (`src/vla_lens/probes/workflow_training.py:426-492`, `src/vla_lens/probes/suite.py:66-128`). |
| Artifact shell | General | `LensArtifact` explicitly covers probe/attribution/intervention/visualization-style durable records (`src/vla_lens/artifacts.py:1-6`, `src/vla_lens/artifacts.py:33-50`). |
| Dashboard indexes | Mostly general, probe-specialized tables | Episode/model/artifact indexes are general, but probe prediction/episode indexes are hard-coded for `probe_suite` (`src/vla_lens/dataset/index.py:173-246`, `src/vla_lens/dataset/index.py:387-468`). |
| Probe study diagnostics | Probe-specific | Rich study UI reads `diagnostics/summary.json` and probe diagnostic parquet names (`src/vla_lens/server/probe_studies.py:664-692`). |
| Workbench selection/cohorts | General | `SelectionState`, `CohortSpec`, `AnalysisRunSpec`, and `InterventionRunSpec` are method-neutral records (`src/vla_lens/workbench/schema.py:275-342`, `src/vla_lens/workbench/schema.py:424-487`). |
| Intervention targets | General contract, partial integration | Artifact-family registry covers probes, contrast directions, SAE/transcoder/crosscoder features, and attention edges (`src/vla_lens/interventions/families.py:83-185`), but dashboard episode/readout support is still probe-only (`src/vla_lens/server/discovery_artifacts.py:56-67`, `src/vla_lens/server/discovery_artifacts.py:94-103`, `src/vla_lens/server/discovery_artifacts.py:234-247`). |

## Minimal Serializable Experiment Recipe

The smallest useful addition is not a new storage system. It is a thin recipe layer over existing selectors, target resolvers, artifacts, and indexes:

```yaml
schema: vla_lens.experiment_recipe.v0
name: short human name
question: what claim this run can support
dataset:
  root: runs/pi05-broad-1000
  fingerprint: optional pinned dashboard fingerprint
population:
  episode_filter: {}
  row_filter: []
  row_expand: null
signals:
  - id: main
    source: activation
    selector: {}        # ActivationQuery-compatible
    transform:
      reduction: mean
targets:
  - id: y
    resolver: {}        # current probe target spec, or another resolver
split:
  kind: heldout_task
  column: split
  train_value: train
  selection_value: val_heldout_task
  test_value: test_heldout_task
  eval_values: [val_heldout_task, test_heldout_task]
method:
  family: linear_probe  # or sae, transcoder, contrast_direction, steering_discovery
  params: {}
outputs:
  artifact_type: method-selected
  save_example_manifest: true
  publish_dashboard: true
```

Required persisted objects:

- `ExperimentRecipe`: normalized JSON/YAML record.
- `ExampleManifest`: exact selected rows after metadata, expansion, target resolution, filters, and split. It should include `example_id`, `trace_id`, sample address, signal ids, target ids, split, and source fingerprints.
- `ExperimentRun`: method-specific output mapped into `LensArtifact` plus workbench `AnalysisRunSpec`.

Example - linear probe:

```yaml
schema: vla_lens.experiment_recipe.v0
name: target lifted linear probe
question: Is target lift state decodable from expert action hidden states?
population:
  row_filter: []
signals:
  - id: expert_action_hidden
    source: activation
    selector:
      module: pi05.expert.layers.*
      tensor_type: hidden_tokens
      token_kind: action
      layers: [0, 4, 8, 12, 17]
      policy_calls: [0, 1, 2, 3, 4, 5, 6]
      generation_step: final
      reduce_tokens: mean
      dtype: float32
targets:
  - id: target_lifted
    resolver: {kind: target_lifted, missing_policy: drop}
split:
  kind: heldout_task
  column: split
  train_value: train
  selection_value: val_heldout_task
  test_value: test_heldout_task
  eval_values: [val_heldout_task, test_heldout_task]
method:
  family: linear_probe
  params:
    models: [linear]
    sweep: [layer, policy_call_index]
    baselines: [majority_class, benchmark, task_id, scene_family, task_verb, primary_target_object, policy_call_index]
outputs:
  artifact_type: probe_suite
  diagnostics: standard
```

Example - SAE:

```yaml
schema: vla_lens.experiment_recipe.v0
name: expert action hidden SAE
question: What sparse features explain expert action hidden-state variation?
signals:
  - id: expert_action_hidden
    source: activation
    selector:
      module: pi05.expert.layers.*
      tensor_type: hidden_tokens
      token_kind: action
      layers: [12]
      policy_calls: all
      generation_step: final
      reduce_tokens: mean
method:
  family: sae
  params:
    dictionary_size: 16384
    l1: 0.001
    normalize: per_feature_standardize
outputs:
  artifact_type: sae_feature
  arrays: [encoder, decoder, feature_stats]
  save_example_manifest: true
```

Example - transcoder:

```yaml
schema: vla_lens.experiment_recipe.v0
name: expert layer 8 to action-head transcoder
question: Which learned features mediate from expert hidden states to action-head input?
signals:
  - id: source_hidden
    source: activation
    selector:
      module: pi05.expert.layers.*
      tensor_type: hidden_tokens
      token_kind: action
      layers: [8]
      generation_step: final
      reduce_tokens: mean
  - id: target_action_head
    source: activation
    selector:
      name: pi05.action_head.input
      policy_calls: all
      reduce_tokens: flat
method:
  family: transcoder
  params:
    source: source_hidden
    target: target_action_head
    feature_count: 8192
    sparsity: 0.001
outputs:
  artifact_type: transcoder_feature
  arrays: [encoder, decoder, feature_activations]
```

Example - steering discovery:

```yaml
schema: vla_lens.experiment_recipe.v0
name: success minus failure target-lift direction
question: Is there a contrast direction that separates successful and failed target-lift attempts?
population:
  cohorts:
    positive: {filter: [{column: target_lifted, op: ==, value: true}]}
    negative: {filter: [{column: target_lifted, op: ==, value: false}]}
signals:
  - id: expert_action_hidden
    source: activation
    selector:
      module: pi05.expert.layers.*
      tensor_type: hidden_tokens
      token_kind: action
      layers: [12]
      policy_calls: [0, 1, 2]
      generation_step: final
      reduce_tokens: mean
method:
  family: contrast_direction
  params:
    positive: positive
    negative: negative
    estimator: mean_difference
    controls: [random_direction, wrong_layer, wrong_time]
outputs:
  artifact_type: contrast_direction
  intervention_target: true
```

## Researcher-Facing API Proposal

Python API:

```python
from vla_lens.experiments import ExperimentRecipe
from vla_lens.traces import TraceDataset

dataset = TraceDataset.open("runs/pi05-broad-1000")
recipe = ExperimentRecipe.from_yaml("configs/experiments/target_lifted_probe.yaml")

packet = recipe.preflight(dataset)
print(packet.to_markdown())

run = recipe.run(dataset)
print(run.artifact.artifact_id)
print(run.example_manifest.path)
```

Lower-level API for notebook work:

```python
from vla_lens.experiments import build_example_set, run_method
from vla_lens.selectors import ActivationQuery

examples = build_example_set(
    dataset,
    signals={
        "x": ActivationQuery(
            module="pi05.expert.layers.*",
            layers=[12],
            tensor_type="hidden_tokens",
            token_kind="action",
            policy_calls=[0, 1, 2],
            generation_step="final",
            reduce_tokens="mean",
        ),
    },
    targets={"y": {"kind": "target_lifted", "missing_policy": "drop"}},
    split={"kind": "heldout_task", "column": "split"},
)

run = run_method("linear_probe", examples, models=["linear"], sweep=["layer", "policy_call_index"])
run.save(dataset, publish_dashboard=True)
```

CLI surface:

```bash
uv run python scripts/vla_lens_experiment.py preflight \
  runs/pi05-broad-1000 \
  --recipe configs/experiments/target_lifted_probe.yaml \
  --format markdown

uv run python scripts/vla_lens_experiment.py run \
  runs/pi05-broad-1000 \
  --recipe configs/experiments/target_lifted_probe.yaml \
  --publish-dashboard \
  --diagnostics standard

uv run python scripts/vla_lens_experiment.py export-examples \
  runs/pi05-broad-1000 \
  --recipe configs/experiments/target_lifted_probe.yaml \
  --output runs/pi05-broad-1000/vla_lens/tables/example_manifests/target_lifted_probe.parquet
```

This API should call existing internals first:

- `ActivationQuery` for activation addressability.
- `_attach_episode_metadata`, `_resolve_probe_target`, `_apply_row_expansion`, `_apply_row_filters`, `_apply_missing_policy`, and `_ensure_split` behind a public `build_example_set`.
- `run_probe_suite` behind a `linear_probe` method runner.
- `LensArtifact` and `AnalysisRunSpec` for outputs.
- `build_dataset_index` and optional `refresh_all_probe_score_caches` for dashboard publication.

## Friction Report - 10 Highest-Friction Steps

1. No exact example manifest.
   - Location: `_probe_examples` records counts and fingerprint only (`src/vla_lens/probes/workflow_artifacts.py:154-182`).
   - Reduction: save `ExampleManifest` parquet for every run and include its path in `artifact.method.outputs`.

2. Feature/target/support discovery requires code knowledge.
   - Location: `ActivationQuery` fields are public (`src/vla_lens/selectors.py:21-42`), but target source dispatch is private switch logic (`src/vla_lens/probes/workflow_targets.py:110-168`).
   - Reduction: add `vla-lens experiment inspect-dataset ROOT` and `/api/experiment/capabilities` returning legal model sites, target resolvers, row columns, token spaces, axes, and supported reductions.

3. CLI supports only a subset of YAML spec power.
   - Location: `scripts/train_vla_lens_probe.py:19-73`.
   - Reduction: deprecate flag-only rich runs in favor of `--recipe`, or add explicit flags for `target.source`, `target.selector`, `row_filter`, `row_expand`, `eval_values`, and `selection_value`.

4. Preflight is not purely a dry run.
   - Location: `probe_preflight_report` materializes selector output with `cache=True` (`src/vla_lens/probes/preflight.py:37-83`).
   - Reduction: add `support-estimate` mode from indexes only, and label the current mode `materialized-preflight`.

5. Split semantics are implicit.
   - Location: `_ensure_split` returns existing split column if any non-null values exist (`src/vla_lens/probes/workflow_prepare.py:574-584`).
   - Reduction: validate full split coverage, report null/unknown split counts, and require `allow_partial_split: true` to proceed.

6. Object expansion is powerful but hard-coded.
   - Location: `_apply_row_expansion` supports only `object_roles` and `scene_objects` (`src/vla_lens/probes/workflow_prepare.py:730-744`).
   - Reduction: expose row expansion as registry functions: `object_candidates`, `token_candidates`, `time_windows`, `contrast_pairs`.

7. Diagnostics are a second manual workflow.
   - Location: core training writes predictions/metrics (`src/vla_lens/probes/workflow_training.py:325-339`), but study UI diagnostics are written by `scripts/report_pi05_probe_diagnostics.py:157-292` and `scripts/report_pi05_probe_diagnostics.py:1199-1227`.
   - Reduction: make `diagnostics: none|standard|claim_bearing` part of the recipe and run it from the same command after training.

8. Dashboard visibility depends on index/cache state.
   - Location: build index writes probe prediction and episode tables (`src/vla_lens/dataset/index.py:216-225`); score cache refresh is separate (`src/vla_lens/probes/score_cache.py:61-146`).
   - Reduction: `run(..., publish_dashboard=True)` should refresh scores if requested, rebuild indexes, and print exact UI-ready artifact ids.

9. Multiple backend/UI probe surfaces duplicate mental models.
   - Location: dataset browser uses probe index, probe studies, discovery artifact episodes, and evidence bundle queries (`frontend/src/pages/workbench/DatasetBrowser.tsx:137-160`, `frontend/src/pages/workbench/DatasetBrowser.tsx:223-329`); ProbeSuite preset independently fetches studies (`frontend/src/components/workflows/ProbeSuitePreset.tsx:79-155`).
   - Reduction: add one experiment-readout endpoint keyed by `analysis_run_id` and `readout_id`, then let UI panels compose that payload.

10. Intervention integration names targets but does not complete the experiment loop.
    - Location: artifact family registry covers future methods (`src/vla_lens/interventions/families.py:83-185`), target conversion exists (`src/vla_lens/server/discovery_artifacts.py:171-208`), and runtime preflight is metadata-only by design (`src/vla_lens/interventions/preflight.py:1-6`).
    - Reduction: recipe outputs should be able to declare `intervention_target: true`, producing a `TargetSpec` candidate, required controls, and a cohort intervention request shell without implying live execution in the normal repo env.

## Recommended Next API Surface

Add a `vla_lens.experiments` package with these public symbols:

- `ExperimentRecipe`: normalized YAML/JSON spec with schema versioning.
- `ExperimentPreflight`: support counts, missing targets, split coverage, selected model sites, estimated feature shape, warnings.
- `ExampleSet`: in-memory object with `signals`, `targets`, `rows`, `splits`, `fingerprints`, and `save_manifest`.
- `MethodRunner`: registry keyed by `linear_probe`, `sae`, `transcoder`, `contrast_direction`, `steering_discovery`.
- `ExperimentRun`: saved `LensArtifact`, `AnalysisRunSpec`, output refs, example manifest ref, dashboard refs.

The implementation can start conservatively:

1. Move the current private probe preparation path into a public `build_example_set` wrapper without changing behavior.
2. Make `train_probe_artifact_from_spec` call `ExperimentRecipe(...).run(method="linear_probe")` internally.
3. Preserve existing probe YAML compatibility by mapping it into the new recipe shape.
4. Save example manifests for probes first.
5. Add method stubs for SAE/transcoder/contrast direction that preflight and save inspected-only artifacts before training code exists.

This gives researchers one mental model: choose a population, select signals, resolve targets or cohorts, build examples, run a method, publish artifacts, inspect or seed interventions. It also gives engineers a narrow integration seam for new interpretability methods without weakening the existing probe path.
