# 06 - Evidence, Interventions, and Method Extensions

Inspected commit: `882eeb83be8c8a69a80fc8f6ec8829c311ca4630`

Git status at inspection time: `## master...origin/master`

Scope: static inspection only. I did not run captures, simulators, model downloads,
frontend builds, server launches, or tests.

## Executive Assessment

The evidence plane is partially coherent but split across three layers:

1. `LensArtifact` is the durable storage envelope for analysis products, with
   generic fields for selector, method, metrics, arrays, display, tags, source
   traces, and path (`src/vla_lens/artifacts.py:33-50`). It is useful as a
   persistence and indexing primitive, but it does not by itself distinguish
   outputs, observations, evidence, controls, claims, and limitations.
2. Probe evidence is the strongest typed evidence model. Probe training writes a
   rich `probe_suite` artifact plus prediction tables and metrics
   (`src/vla_lens/probes/workflow_training.py:278-351`), then the dashboard
   adapts indexed probe outputs into `ProbeEvidenceBundle` primitives
   (`src/vla_lens/probe_evidence.py:496-524`,
   `src/vla_lens/probe_evidence_adapter.py:60-119`). This is implemented but
   probe-specific.
3. Intervention evidence has good typed contracts and storage shells, including
   preflight, runtime resolution, trials, outcomes, controls, and claims
   (`src/vla_lens/interventions/specs.py:598-750`,
   `src/vla_lens/interventions/results.py:390-482`). The live causal path is
   not generally wired through the dashboard: the UI can preflight and save an
   inspected record, but the Run button is disabled when live runtime is absent
   (`frontend/src/components/interventions/InterventionLab.tsx:106-136`).

The smallest scientifically useful intervention vertical slice is therefore
action-level, not rollout-level: take one saved linear `probe_suite` direction,
one held-out PI0.5 policy call, run original/no-op/intervened/random-control
action chunks in the dedicated PI0.5 runtime, save an `InterventionRun`, and
render the action delta comparison. That can support a local causal observation
about a policy-call/action representation. It should not be marketed as a broad
behavioral claim.

## Artifact Model

### Persistent Schemas And Artifact-Related Types

| Type or schema | Implemented? | Purpose and inputs | Outputs and provenance | Metrics, plots, raw data, status, frontend support |
| --- | --- | --- | --- | --- |
| `LensArtifact` | Implemented generic envelope. | Durable record for probes, attributions, interventions, visualizations, or other analysis products. Fields include `artifact_id`, `artifact_type`, `name`, `group_id`, `scope`, `selector`, `method`, `metrics`, `arrays`, `display`, `tags`, `created_utc`, `source_trace_ids`, and `path` (`src/vla_lens/artifacts.py:3-6`, `src/vla_lens/artifacts.py:33-50`). | Created through `LensArtifact.create` (`src/vla_lens/artifacts.py:52-81`), serialized through `to_record`/`from_dict` (`src/vla_lens/artifacts.py:89-124`). Dataset artifacts are written under an artifact directory with `artifact.json` and optional array stores (`src/vla_lens/traces/dataset.py:133-186`). Arrays are Zarr-backed (`src/vla_lens/traces/io.py:106-127`). | Generic metrics/display exist, but no generic status/failure field. Dashboard artifact index columns are generic (`src/vla_lens/dataset/index.py:79-97`), and the server lists artifact summaries from `dashboard_artifact_index.parquet` (`src/vla_lens/server/indexed.py:176-187`). |
| Dataset and bundle artifact indexes | Implemented. | Make artifacts discoverable across dataset-level and trace-bundle scope. | Dataset index merges dataset and bundle artifact rows and annotates `trace_id`, `episode_id`, and `artifact_scope` (`src/vla_lens/dataset/index.py:355-376`). Dashboard table paths include artifact and probe-prediction indexes (`src/vla_lens/dataset/index.py:24-31`). | If the dashboard index is deleted, raw `artifact.json` files may still exist, but artifact lists, probe discovery, and probe evidence routes lose their indexed view until rebuilt (`src/vla_lens/server/indexed.py:176-191`, `src/vla_lens/server/indexed_probes.py:25-75`). |
| `probe_suite` `LensArtifact` | Implemented. | Supervised probe workflow over selected activations. The selector is materialized with `dataset.select_model_sites(selector).materialize(cache=True)` (`src/vla_lens/probes/workflow_training.py:83-91`). Method payload records workflow, schema, research framing, lineage, source, input, target, examples, split, normalization, evaluation, outputs, null controls, and notes (`src/vla_lens/probes/workflow_training.py:175-251`). | Writes arrays such as model weights/normalizers, `predictions.parquet`, `scored_predictions.parquet`, split/group/null metrics, `metrics.json`, a workbench `AnalysisRunSpec`, and rebuilt dashboard indexes (`src/vla_lens/probes/workflow_training.py:325-351`). Lineage captures git commit and library versions (`src/vla_lens/probes/workflow_artifacts.py:22-32`); input/source/example/split payloads capture trace IDs, row identity, feature shape, target resolution, row fingerprints, and leakage risk (`src/vla_lens/probes/workflow_artifacts.py:35-215`). | Strongest implemented artifact type. Still correlational: the notes explicitly warn that outcome probes are not mechanism by themselves (`src/vla_lens/probes/workflow_artifacts.py:510-531`). Failure before save produces no artifact; partial status is inferred later by the evidence adapter. Frontend support exists through probe indexes, episode panels, and evidence bundles. |
| Probe prediction indexes | Implemented, probe-specific. | Make `probe_suite` predictions queryable by episode/policy call/timestep/model site. | `_probe_prediction_index_table` only indexes artifacts where `artifact_type == "probe_suite"` and reads saved prediction tables from artifact outputs (`src/vla_lens/dataset/index.py:387-405`, `src/vla_lens/dataset/index.py:524-546`). Rows include policy-call, timestep, generation-step, model-site, split, label, prediction, probability, score, and correctness fields (`src/vla_lens/dataset/index.py:408-443`). | This is not method-independent. New methods will not automatically get evidence pages unless they provide their own index or a common evidence index is added. |
| `ProbeEvidenceBundle` | Implemented derived API payload, not persisted as a `LensArtifact`. | Typed probe evidence contract with `LensRun`, `LensGeometry`, provenance, score series, ranked moments, predictions, model-locus evidence, cohort summaries, failure cases, and unavailable reasons (`src/vla_lens/probe_evidence.py:219-524`). | Built from dashboard indexes and probe artifacts by `indexed_probe_evidence_bundle_payload` and `_build_probe_evidence_bundle` (`src/vla_lens/probe_evidence_adapter.py:42-119`). It is a view over saved artifact/index data, not the source of truth. | It declares unavailable evidence rather than faking it. For example, contribution evidence is unavailable when the adapter lacks aligned activation values or probe weights (`src/vla_lens/probe_evidence_adapter.py:467-534`). Frontend consumes it through `/api/probes/{probe_id}/evidence-bundle` (`src/vla_lens/server/fastapi_app.py:440-449`). |
| `AnalysisRunSpec` | Implemented workbench record. | Workbench-level record for analysis workflow execution. Fields are `run_id`, `workflow`, `inputs`, `outputs`, and `provenance` (`src/vla_lens/workbench/schema.py:425-447`). | Saved/listed through workbench API helpers (`src/vla_lens/workbench/api.py:112-125`). Probe and PI0.5 derived artifacts also save analysis-run records. | Useful for provenance but too generic to act as an evidence schema. No first-class metrics/status/claims. |
| PI0.5 derived label artifacts: `pi05_object_flow`, `pi05_interaction_metrics`, `pi05_policy_call_labels` | Implemented derived artifacts. | Turn object-flow and interaction tables into episode/policy-call labels. `pi05_policy_call_labels` reads object-flow artifact outputs, builds one label row per policy call, and records source artifact identity (`src/vla_lens/pi05/policy_call_labels.py:30-65`, `src/vla_lens/pi05/policy_call_labels.py:125-180`). Object-flow and interaction-metric artifacts write multiple parquet outputs and workbench records (`src/vla_lens/pi05/object_flow.py:47-184`, `src/vla_lens/pi05/interaction_metrics.py:39-160`). | They persist parquet outputs and update artifact JSON/indexes; policy-call labels explicitly write `policy_call_labels.parquet` and update `method.outputs` (`src/vla_lens/pi05/policy_call_labels.py:82-121`). | These are observations/labels used downstream by probes, not causal evidence. They have metrics/display summaries but no generic status/failure field. |
| `action_generation` | Implemented derived artifact. | Summarizes how generated actions, chunks, and executed actions relate across policy calls (`src/vla_lens/action_generation.py:25-121`). | Saves arrays/tables for commitment, executed-vs-predicted, delta-to-final, step delta, final-vs-executed, plus an `AnalysisRunSpec` (`src/vla_lens/action_generation.py:104-113`, `src/vla_lens/action_generation.py:124-214`). | Supports action-formation observations. It is not an intervention or causal control result. |
| `InterventionRunSpec` | Implemented workbench shell. | Stores saved intervention records in the workbench layer. Fields include `run_id`, `intervention_type`, `target`, `baseline`, `intervention`, `readouts`, `outputs`, and `provenance` (`src/vla_lens/workbench/schema.py:453-486`). | Save/list/detail routes exist (`src/vla_lens/workbench/api.py:127-161`, `src/vla_lens/server/workbench_payloads.py:93-126`, `src/vla_lens/server/fastapi_app.py:183-197`). `save_intervention_run` also writes an `AnalysisRunSpec` with a conservative `causal_evidence` flag (`src/vla_lens/workbench/api.py:127-178`). | Better status/claim support than generic artifacts, but stored separately from `LensArtifact` unless explicitly converted. Frontend Evidence page lists/detail views and raw JSON (`frontend/src/pages/EvidencePage.tsx:17-94`, `frontend/src/components/interventions/InterventionRunDetail.tsx:9-58`). |
| `InterventionRun` and `intervention_run` `LensArtifact` | Contracts implemented; live execution partial. | Typed canonical intervention payload with context, target, request, preflight, runtime resolution, trials, outcomes, controls, outputs, display, claim, and provenance (`src/vla_lens/interventions/results.py:390-440`). | Converts to/from `InterventionRunSpec` (`src/vla_lens/interventions/results.py:442-563`) and to a `LensArtifact` of type `intervention_run` (`src/vla_lens/interventions/artifacts.py:17-75`). PI0.5 runtime code can build original/no-op/intervened/control action evidence when an executor is injected (`src/vla_lens/pi05/intervention_runtime.py:57-159`). | This is the right causal-evidence shell. Dashboard live run remains unavailable unless a safe model runtime is present; inspected records can have empty trials/outcomes/controls. |
| `intervention_sweep` and `InterventionStudy` | Implemented aggregation shell. | Aggregates compatible intervention runs into sweeps/studies (`src/vla_lens/interventions/sweeps.py:21-84`, `src/vla_lens/interventions/sweeps.py:87-133`). | Produces aggregate outcomes and claim labels such as `action_level`, `behavioral`, `specific`, and `causal_cohort` based on run composition (`src/vla_lens/interventions/sweeps.py:300-322`). | Good later-stage comparison model, but it depends on real saved intervention runs. |
| Future discovery families: `sae_feature`, `transcoder_feature`, `crosscoder_feature`, `attention_map` | Specified/tested contracts, not full training pipelines. | Artifact-family registry maps discovery artifact types to target kinds, legal operators, outcomes, and controls (`src/vla_lens/interventions/families.py:18-59`, `src/vla_lens/interventions/families.py:121-145`). Tests confirm `sae_feature` maps to a feature target and `transcoder_feature` maps to a path target (`tests/test_intervention_artifact_families.py:22-49`, `tests/test_intervention_artifact_families.py:79-107`). | `target_from_discovery_artifact` can normalize an artifact into a `TargetSpec` (`src/vla_lens/interventions/families.py:211-273`). | This is intervention-handoff scaffolding, not SAE/transcoder training, dictionary storage, feature activation indexing, or runtime patching. |

### Artifact Questions

1. **What does "artifact" mean here?** A durable, indexed analysis product
   envelope. It is not automatically "evidence" or "causal evidence."
2. **Is there one canonical artifact model?** There is one canonical storage
   envelope, `LensArtifact`, but not one canonical typed scientific payload.
   Probe evidence and intervention evidence have separate typed contracts.
3. **Where do typed payloads live?** Mostly inside `method`, `metrics`,
   `arrays`, and `display` dictionaries for `LensArtifact`, with stronger
   typed shells in `ProbeEvidenceBundle` and `InterventionRun`.
4. **Are dataset/model entities linked?** Partially. `source_trace_ids`,
   selector fields, probe example rows, `model_site_id`, `trace_id`,
   `policy_call_index`, and intervention `ContextSpec` connect artifacts to
   episodes and policy calls. The link is strong for probes and interventions,
   weaker for generic artifacts.
5. **Can an artifact cite another artifact?** Yes in specific places:
   intervention targets require `source_artifact_id` for artifact-derived
   targets (`src/vla_lens/interventions/specs.py:288-362`), and derived PI0.5
   label artifacts record source artifact IDs. There is no universal upstream
   artifact graph field in `LensArtifact`.
6. **Can artifacts be compared programmatically?** Probe comparisons are
   probe-index-specific; intervention sweeps can aggregate intervention runs.
   There is no generic `compare(any two artifacts)` contract.
7. **Is there an exact cohort/example manifest?** Probes come closest:
   `_probe_examples` records row construction, filters, counts, row identity
   definition, and row fingerprint (`src/vla_lens/probes/workflow_artifacts.py:154-182`).
   That should be lifted into a method-independent example manifest.
8. **How are failed/partial artifacts represented?** Generic `LensArtifact`
   has no status. Probe adapter creates partial `LensRun` status from missing
   predictions/episodes (`src/vla_lens/probe_evidence_adapter.py:183-213`).
   `InterventionRun` has first-class status values (`src/vla_lens/interventions/results.py:390-440`).
9. **What breaks if dashboard indexes are deleted?** Raw artifact files may
   remain, but artifact lists, probe discovery, probe evidence pages, and
   discovery artifact payloads rely on dashboard index tables
   (`src/vla_lens/server/indexed.py:176-191`,
   `src/vla_lens/server/indexed_probes.py:25-75`,
   `src/vla_lens/server/discovery_artifacts.py:263-284`).
10. **Are plots raw data or views?** In the inspected code, artifacts mainly
    store arrays, parquet tables, metrics, and display summaries. Dashboard
    panels render views from those data. A plot should be treated as a view
    unless a concrete plot artifact file is explicitly stored and indexed.

## Evidence Versus Claims

### Current Distinctions

The code already contains several useful boundaries, but they are not expressed
as one shared evidence ontology.

| Concept | Current representation | Assessment |
| --- | --- | --- |
| Method output | `LensArtifact` outputs, probe predictions, metrics, arrays, PI0.5 derived tables. | Implemented. This is a product of a method run, not automatically evidence for a scientific claim. |
| Observation | Probe primitives such as score series, ranked moments, predictions, model locus, failure cases; intervention outcomes/trials. | Implemented for probes and interventions, absent as a generic artifact concept. |
| Evidence | `ProbeEvidenceBundle` combines run geometry, provenance, primitives, cohort refs, and unavailable reasons (`src/vla_lens/probe_evidence.py:496-524`). `InterventionRun` combines preflight, runtime resolution, trials, outcomes, controls, and claim labels. | Strongest direction. Needs method-independent observation IDs and reusable limitations/controls. |
| Control | Probe null metrics include label-shuffle controls (`src/vla_lens/probes/workflow_artifacts.py:455-478`). Intervention contracts include no-op, random, wrong-layer, wrong-time, wrong-token, placebo, heldout, matched, and family-specific controls (`src/vla_lens/interventions/families.py:83-145`). | Specified well. Real execution/control comparison is only partially implemented. |
| Claim | Probe method payloads contain research framing and notes; contribution primitives carry `claim_level`; intervention runs carry `claim` dictionaries. | Not yet a first-class citable object. Current claims are labels/notes attached to outputs. |
| Hypothesis | Mostly implicit in method config, workflow names, and UI framing. | Needs a small explicit field if the system wants to track "this run was designed to test X." |

### Minimal Evidence Model

The smallest useful shared model should be narrower than the current generic
artifact envelope and broader than probe-only evidence:

```text
MethodRun
  -> MethodOutput
  -> Observation[]
  -> ControlObservation[]
  -> Claim[]
```

For VLA Lens specifically:

- `ProbeRun -> DecodabilityObservation`: says a target was predictable from a
  selected representation under a split, null control, and cohort definition.
  This is correlational evidence.
- `LocalizationSweep -> LocalizationObservation`: says a signal was localized
  to a site/time/token/path under a measurement method and comparison baseline.
- `InterventionRun -> CausalObservation`: says an intervention changed a local
  readout, action chunk, token distribution, or rollout metric relative to
  original/no-op/control trials.
- `ControlRun -> AlternativeExplanationConstraint`: says a no-op, random,
  wrong-layer, wrong-time, wrong-token, or matched control did or did not
  reproduce the effect.
- `Claim -> citations`: cites observation IDs, control IDs, cohorts, limits,
  and unsupported extrapolations. A claim should never cite only a dashboard
  chart or an artifact ID without the observation/control payload.

This model avoids using broad labels as evidence. The scientifically useful
claim text should name the method, representation, site, time/policy-call
scope, outcome basis, and controls.

## Intervention Readiness

### End-To-End Trace

| Step | Current state | Gaps and risks |
| --- | --- | --- |
| Analysis result -> candidate feature/direction/site | `probe_suite` artifacts store selector/method/arrays and probe indexes. Episode probe panel can seed intervention from a probe and selected policy call (`frontend/src/pages/episodes/EpisodeProbePanel.tsx:46-83`, `frontend/src/pages/episodes/EpisodeProbePanel.tsx:170-177`; `frontend/src/pages/episodes/interventionSeed.ts:6-50`). `target_from_discovery_artifact` can convert discovery artifacts into target specs. | Probe target extraction is partly heuristic. `ProbeEvidenceBundle` contribution evidence is often unavailable because aligned activation values and weights are not in the adapter view (`src/vla_lens/probe_evidence_adapter.py:467-534`). |
| Candidate -> `TargetSpec` | `TargetSpec` carries artifact source, model family/id, model site, site ID, module path, layer, tensor type, token space, token selector, generation selector, reduction, representation, and metadata (`src/vla_lens/interventions/specs.py:288-362`). | There is no single ontology proving that a probe `model_site_id`, stored activation name, and runtime hook name are the same intervention site. Preflight checks declaration, not semantic equivalence. |
| Target/request -> runtime preflight | Generic preflight is static and intentionally has no heavy runtime imports (`src/vla_lens/interventions/preflight.py:1-6`). It checks policy-call identity, stored actions/chunks, source artifact, target site, token space, action decoder/basis metadata, runtime adapter/model runtime, and environment safety (`src/vla_lens/interventions/preflight.py:28-152`). | Preflight can return `inspected_only` when the model runtime is unavailable (`src/vla_lens/interventions/preflight.py:155-166`). That is correct, but the UI needs to keep this visibly separate from executed intervention evidence. |
| Runtime resolution | `RuntimeResolution` records adapter, model family/id, checkpoint, requested target, resolved hook, generation/token mappings, shape, dtype, device, env, and warnings (`src/vla_lens/interventions/specs.py:677-750`). | The schema is ready, but populated runtime resolution depends on actual PI0.5 executor integration. |
| Trial execution | `ActionInterventionExecutor` is a protocol with `run_noop`, `run_intervention`, and `run_control` (`src/vla_lens/interventions/runtime.py:42-58`). `run_pi05_intervention` builds original/no-op/intervened/control action trials when a live executor is injected (`src/vla_lens/pi05/intervention_runtime.py:57-159`). | No general dashboard live-run route exists in the inspected frontend. This is not a normal `uv run` path; it belongs in the dedicated PI0.5 runtime environment. |
| Saved result | `InterventionRun` converts to `InterventionRunSpec`; `intervention_run_to_lens_artifact` converts it to a `LensArtifact` (`src/vla_lens/interventions/results.py:442-482`, `src/vla_lens/interventions/artifacts.py:17-75`). | Inspected records saved from the UI may have empty `trials`, `outcomes`, and `controls` (`frontend/src/components/interventions/interventionLabModel.ts:61-106`). Those should be displayed as preflight/planning records, not evidence. |
| Comparison UI | Evidence page lists intervention runs and shows detail/raw JSON (`frontend/src/pages/EvidencePage.tsx:17-94`, `frontend/src/components/interventions/InterventionRunDetail.tsx:9-58`). | There is no dedicated original/no-op/intervened/control action-delta panel or rollout comparison view. This is the main missing piece for making action-level intervention evidence usable by a researcher. |

### Action Versus Rollout Evidence

The implemented PI0.5 intervention path is action-level first. It records stored
original actions, no-op runtime actions, intervened runtime actions,
intervened-minus-no-op deltas, and optional control arrays
(`src/vla_lens/pi05/intervention_runtime.py:57-159`). Action basis resolution
supports raw/gripper/eef/rotation/speed style metrics
(`src/vla_lens/interventions/action_basis.py:24-27`,
`src/vla_lens/interventions/action_basis.py:257-320`). Rollout outcome schemas
exist (`src/vla_lens/interventions/results.py:167-191`), but I did not find an
implemented live rollout execution/comparison path equivalent to the action
path.

### Smallest Scientifically Useful Intervention Vertical Slice

The minimum vertical slice should be deliberately narrow:

1. Start from one saved, held-out `probe_suite` artifact with a linear direction
   and saved prediction rows.
2. Select one held-out episode and one policy call with an available stored
   action chunk.
3. Build a `TargetSpec` with `kind="probe_direction"`,
   `source_artifact_id`, `source_artifact_type="probe_suite"`, model site,
   token space, generation selector, and explicit representation array ref.
4. Preflight through the existing static preflight API and require:
   `policy_call_exists`, `stored_action_chunk_exists`, `source_artifact_exists`,
   `target_site_declared`, `token_space_declared`, `action_basis_metadata_available`,
   `runtime_adapter_declared`, and `runtime_environment_safe`.
5. In the dedicated PI0.5 capture/runtime environment, execute exactly:
   original/stored baseline, no-op live pass, one intervened pass, and one
   random-direction or wrong-time control.
6. Save one `InterventionRun` plus `intervention_run` artifact containing
   aligned arrays for `stored_original`, `noop`, `intervened`,
   `intervened_minus_noop`, and control output refs.
7. Add one comparison panel showing action chunk deltas and control deltas by
   action basis. The claim level is `causal_local` + `action_level` only.

This slice is small because it avoids simulator rollouts, long-horizon
behavioral interpretation, learned SAE dictionaries, and broad causal claims.
It is still scientifically useful because it closes the loop from a saved
analysis result to a controlled action-level perturbation with provenance.

## SAEs And Transcoders Fit

1. **Can they reuse current feature extraction?** Partially. `ActivationQuery`
   already describes activation slices by episode filters, name/module/layer,
   tensor type, token kind, timesteps, policy calls, generation step, token
   reduction, and dtype (`src/vla_lens/selectors.py:21-42`). `FeatureView`
   records row identity fields including trace, episode, timestep, policy call,
   activation name, model site, token space, layer, tensor type, token kind,
   generation step, reduction, and feature dimension (`src/vla_lens/selectors.py:112-188`).
   That row identity is reusable.
2. **Can they stream or batch high-dimensional data now?** Not really.
   `FeatureView.materialize` returns an in-memory `np.ndarray` and cached Zarr
   array, and `_compute` stacks all vectors (`src/vla_lens/selectors.py:71-97`,
   `src/vla_lens/selectors.py:112-188`). That is acceptable for probes but not
   enough for large SAE/transcoder training.
3. **Is there a common training-run abstraction?** No. Probe training has a rich
   method payload, but it is supervised-probe-specific. `AnalysisRunSpec` is too
   generic, and `LensArtifact` is too loose.
4. **Can dictionaries/transcoders be versioned and stored?** The generic array
   storage can store encoder/decoder/dictionary arrays, but there is no
   concrete SAE/transcoder training artifact schema with dictionary version,
   sparsity hyperparameters, optimizer state, reconstruction metrics, feature
   activation index, or top-example manifest.
5. **Can learned features link back to episodes and policy calls?** The selector
   row identity can support this. The missing piece is a method-independent
   feature activation table: feature ID, activation value, row ID, trace ID,
   policy call, token/generation site, source signal ref, and split/cohort.
6. **Is inspector exposure present?** Partial. The workbench catalog has a
   `unit_explorer` workflow and includes `sae_feature` as a unit kind
   (`src/vla_lens/workbench/catalog.py:812-819`). Frontend type unions include
   `sae_feature` and `transcoder_feature`
   (`frontend/src/types/dataset.ts:322-329`). That is UI vocabulary, not full
   data plumbing.
7. **Can a learned feature become an intervention target?** Contractually yes.
   `sae_feature` maps to target kind `feature`, with legal operators such as
   `feature_boost`, `feature_clamp`, `feature_ablate`, and
   `add_decoder_direction`; `transcoder_feature` maps to target kind `path`
   with path/source patch operators (`src/vla_lens/interventions/families.py:121-145`).
   Tests assert these mappings (`tests/test_intervention_artifact_families.py:22-49`,
   `tests/test_intervention_artifact_families.py:79-107`). Runtime execution is
   not implemented by those tests.
8. **What probe-specific blockers exist?** Probe training assumes target labels,
   split metrics, sklearn-style supervised models, prediction/scored-prediction
   tables, and probe-specific dashboard indexes (`src/vla_lens/probes/workflow_training.py:175-351`,
   `src/vla_lens/dataset/index.py:387-405`). The evidence adapter expects
   probe IDs and probe predictions. SAEs/transcoders need self-supervised loss,
   reconstruction metrics, feature sparsity, and top-activation evidence
   instead.
9. **What should be common versus method-specific?** Common: signal selection,
   row/example manifests, provenance, array refs, run status, cohort links,
   observation/control records, limitations, frontend capability declarations,
   and optional intervention target handoff. Method-specific: training loop,
   losses, metrics, learned object schema, interpretation level, and legal
   intervention operators.
10. **What current SAE evidence is real?** Current SAE evidence is fixture and
    contract level. Probe evidence types include `sae_feature` as an input or
    contribution basis (`src/vla_lens/probe_evidence.py:49-58`,
    `src/vla_lens/probe_evidence.py:344-386`), and tests assert a fixture with
    `human_labeled_feature` claim level
    (`tests/probe_evidence_contract_test.py:82-100`). That proves the UI/evidence
    vocabulary can express such evidence; it does not prove an SAE trainer or
    dictionary artifact exists.

## Minimal Analysis-Method Extension Protocol

Do not force every method into the probe metric schema. Instead, define a small
common protocol that every method must emit, then let each method own its typed
payload.

### Required Common Payload

Every analysis method should emit:

1. `method_id`, `method_family`, `schema_version`, `code_version`,
   `random_seed`, dependency versions, and runtime environment.
2. `input_requirements`: accepted observation units (`episode`, `timestep`,
   `policy_call`, `token`, `generation_step`), required activation axes,
   accepted tensor types, target/label requirements if any, and whether the
   method supports streaming or requires materialization.
3. `signal_selection`: a serializable selector compatible with
   `ActivationQuery`, plus the selected model-site rows and a source fingerprint.
4. `example_manifest`: stable row/example IDs with trace ID, episode ID,
   timestep, policy call, model site, token/generation fields, split/cohort,
   and row fingerprint. This generalizes the current probe example metadata.
5. `execution_record`: start/end timestamps, status (`complete`, `partial`,
   `failed`, `inspected_only`), failure reason, warnings, and output refs.
6. `typed_payload`: method-owned object. Examples: probe coefficients and
   predictions, SAE dictionary and feature activations, transcoder path weights,
   attribution maps, localization scores.
7. `metric_blocks`: method-owned metrics grouped by purpose, not a universal
   flat metric schema. Each metric block must declare unit, split/cohort,
   baseline/control, and whether higher/lower is better.
8. `observations`: typed observations with IDs, source outputs, examples,
   cohort, statistics, controls used, and limitations.
9. `frontend_capabilities`: panels/readouts the frontend can safely show, plus
   unavailable reasons when data are absent.
10. `intervention_handoff`: optional target candidate list with target kind,
    model site, representation array refs, legal operators, legal outcomes, and
    required controls.

### Method-Specific Payload Examples

- Probe: target definition, split, model coefficients, prediction table,
  calibration/null metrics, decodability observations, and optional direction
  target candidate.
- SAE: source activation selection, dictionary config, encoder/decoder refs,
  feature activation index, reconstruction/sparsity/dead-feature metrics,
  top-example observations, feature labels if available, and feature target
  candidates.
- Transcoder: source and destination activation selectors, path/dictionary
  refs, reconstruction/pathway metrics, top source-destination examples,
  path observations, and path-patch target candidates.
- Localization/attribution: scoring method, baseline, heatmap/path refs,
  localization observations, false-positive controls, and target candidates
  only when runtime patching is supported.

### Protocol Boundary

The common protocol should answer "what data was selected, how it was run, what
examples support the observation, what controls limit alternative explanations,
and what intervention target can be handed off." It should not try to make SAE
losses, probe AUROC, transcoder reconstruction, and intervention action deltas
look like the same metric.

## Priority Recommendations

1. Add a small method-independent `ObservationRecord` / `ControlObservation` /
   `ClaimRecord` layer, then adapt probe and intervention outputs into it.
2. Lift probe row/example metadata into a reusable `ExampleManifest` schema.
3. Add status/failure fields to generic persisted analysis artifacts or require
   every typed method payload to include execution status.
4. Build the action-level intervention comparison panel before adding broader
   rollout claims.
5. Treat SAE/transcoder support as a new method family: reuse selectors and
   artifact storage, but add streaming training, dictionary artifacts, feature
   activation indexes, top-example evidence, and runtime target resolution
   before claiming support beyond contracts.

## Commands Used

Read-only/static commands run:

- `pwd`
- `sed -n '1,240p' AGENTS.md`
- `wc -l /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt`
- `sed -n '1,260p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt`
- `sed -n '261,620p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt`
- `sed -n '621,1040p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt`
- `sed -n '927,1075p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt`
- `sed -n '1076,1490p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt`
- `git rev-parse HEAD`
- `git status --short --branch`
- `find docs/audits/vla-lens-system-review -maxdepth 1 -type f -print`
- `rg --files`
- `rg -n "06-evidence|intervention|SAE|transcoder|method|04-|05-|06-" /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt`
- `rg -n "LensArtifact|artifact_type|ProbeEvidenceBundle|InterventionRun|AnalysisRunSpec|sae_feature|transcoder|preflight|evidence" src tests frontend scripts docs`
- Static line-number reads with `nl -ba ... | sed -n ...` over:
  `src/vla_lens/artifacts.py`, `src/vla_lens/traces/dataset.py`,
  `src/vla_lens/traces/bundle.py`, `src/vla_lens/traces/io.py`,
  `src/vla_lens/dataset/index.py`, `src/vla_lens/server/artifacts.py`,
  `src/vla_lens/server/fastapi_app.py`, `src/vla_lens/server/indexed.py`,
  `src/vla_lens/server/indexed_probes.py`,
  `src/vla_lens/server/discovery_artifacts.py`,
  `src/vla_lens/workbench/schema.py`, `src/vla_lens/workbench/api.py`,
  `src/vla_lens/workbench/catalog.py`,
  `src/vla_lens/server/workbench_payloads.py`,
  `src/vla_lens/probes/workflow_training.py`,
  `src/vla_lens/probes/workflow_artifacts.py`,
  `src/vla_lens/probes/workflow_spec.py`,
  `src/vla_lens/probes/workflow_types.py`,
  `src/vla_lens/probe_evidence.py`,
  `src/vla_lens/probe_evidence_adapter.py`,
  `src/vla_lens/selectors.py`,
  `src/vla_lens/action_generation.py`,
  `src/vla_lens/pi05/interaction_metrics.py`,
  `src/vla_lens/pi05/object_flow.py`,
  `src/vla_lens/pi05/policy_call_labels.py`,
  `src/vla_lens/interventions/specs.py`,
  `src/vla_lens/interventions/results.py`,
  `src/vla_lens/interventions/preflight.py`,
  `src/vla_lens/interventions/runtime.py`,
  `src/vla_lens/interventions/artifacts.py`,
  `src/vla_lens/interventions/families.py`,
  `src/vla_lens/interventions/sweeps.py`,
  `src/vla_lens/interventions/action_basis.py`,
  `src/vla_lens/pi05/intervention_preflight.py`,
  `src/vla_lens/pi05/intervention_runtime.py`,
  `frontend/src/types/interventions.ts`,
  `frontend/src/types/dataset.ts`,
  `frontend/src/api/interventions.ts`,
  `frontend/src/components/interventions/InterventionCard.tsx`,
  `frontend/src/components/interventions/InterventionRunDetail.tsx`,
  `frontend/src/components/interventions/InterventionLab.tsx`,
  `frontend/src/components/interventions/interventionLabModel.ts`,
  `frontend/src/components/interventions/interventionDisplay.ts`,
  `frontend/src/components/interventions/EvidenceLibrary.tsx`,
  `frontend/src/pages/EvidencePage.tsx`,
  `frontend/src/pages/episodes/EpisodeProbePanel.tsx`,
  `frontend/src/pages/episodes/interventionSeed.ts`,
  `tests/test_intervention_artifact_families.py`, and
  `tests/probe_evidence_contract_test.py`.

Not run:

- `uv run pytest`
- `uv run ruff check scripts src tests`
- Frontend build/dev server
- PI0.5/LeRobot/LIBERO captures, simulators, model downloads, or hardware
  runtime commands
- Destructive git or filesystem commands
