# 01 - Domain Object Model Audit

Repo root for all paths: `/home/j/Projects/vla-lens`.

## Inspection Metadata

- Inspected commit: `882eeb83be8c8a69a80fc8f6ec8829c311ca4630`
- Git status before writing this audit: clean. `git status --short` produced no rows.
- Git status after writing this audit: `git status --short` printed `?? docs/audits/` because other audit files were concurrently present under the same untracked directory. Target-specific status is `?? docs/audits/vla-lens-system-review/01-domain-object-model.md`.
- Method: static inspection and lightweight shell reads only. No captures, simulators, model downloads, heavy runtime imports, destructive commands, or tests were run.
- Required instructions read first: `AGENTS.md` and `/home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt`.

Legend used below:

- **Observed**: directly present in code or persisted schemas.
- **Inferred**: intent inferred from code organization, naming, or comments.
- **Recommendation**: proposed vocabulary or model contract.

## Summary

Observed: the strongest durable spine is:

`dataset root -> trace_id episode -> timestep frame -> policy_call_index call -> action/model-site/artifact rows`.

The storage layer mostly treats `trace_id` as the stable episode key and `policy_call_index` as a trace-local policy invocation key. The intervention layer has the cleanest explicit identity type, `PolicyCallRef(trace_id, policy_call_index)` in `src/vla_lens/interventions/specs.py:124`. The capture/storage layer writes that same pair across `policy_calls`, `timesteps`, token metadata, action chunks, generation steps, and model-site axes.

Main gap: several backend/frontend paths rename or flatten the same identity:

- Stored call key: `policy_call_index`.
- Metrics API payload: list `index` plus stored `model_call_index` in `src/vla_lens/server/metrics.py:36`.
- Activation endpoints: query parameter `call_index` in `src/vla_lens/server/common.py:304` and `frontend/src/api/dataset.ts:371`.
- Evidence model: `policy_call` in `frontend/src/types/probeEvidence.ts:208`.
- Episode hash: `call` in `frontend/src/pages/workbench/episodeRouteModel.ts:38`.
- Intervention draft: `policyCallIndex` in `frontend/src/types/interventions.ts:55`.

Recommendation: make `PolicyCallRef = {trace_id, policy_call_index}` the canonical vocabulary and keep `call_index`, `model_call_index`, `call`, and camelCase names as explicit UI/API aliases only.

## Domain Object Inventory

| Concept | Canonical code type or schema | File | Persistent representation | Stable identifier | Producer | Consumers | Lifecycle | Versioning | Frontend representation | Problems or ambiguities |
|---|---|---|---|---|---|---|---|---|---|---|
| Dataset | `TraceDataset` | `src/vla_lens/traces/dataset.py:15` | Directory containing trace bundles or LeRobot root plus `vla_lens` overlay | Dataset root path. Workbench derives `dataset_id` from root name in `src/vla_lens/workbench/catalog.py:49` | Capture writer, overlay writer, artifact workflows | Server, workbench, probe workflow, intervention preflight | Opened lazily from root; indexes/artifacts can be rebuilt | Dataset index schema `0.2.0` in `src/vla_lens/dataset/index.py:24` | `DatasetPayload.root` in `frontend/src/types/dataset.ts:17`; `WorkbenchManifest.dataset_id` in `frontend/src/types/workbench.ts:169` | `dataset_id` is not a persisted global UUID; it can be root-name derived. |
| Dataset root | LeRobot root or native trace root | `src/vla_lens/traces/dataset.py:40`; `src/vla_lens/dataset/reader.py:24` | Native bundle root or LeRobot v3 root with `meta/info.json`, `data`, and `vla_lens` overlay | Filesystem path | Capture setup/writer | `TraceDataset.open`, dataset index builder, dashboard | Stable while files remain; can contain nested roots | LeRobot validation in `src/vla_lens/capture/lerobot_v3.py:151` | Dataset root shown in API payload | Path identity is machine-local; no canonical cross-machine root id except optional metadata fields. |
| Episode / trace | `TraceManifest` | `src/vla_lens/traces/types.py:13` | `manifest.json`; LeRobot episode metadata plus overlay manifest | `trace_id`; `episode_id` defaults to `trace_id` in `TraceManifest.from_dict` at `src/vla_lens/traces/types.py:38` | `TraceBundle.create`; PI0.5 writer; LeRobot writer | Dataset index, server episode routes, workbench selection | One row per rollout episode | Trace schema `0.3.0` in `src/vla_lens/traces/layout.py:3`; manifest has `schema_version` | `DatasetEpisode.trace_id` and `episode_id` in `frontend/src/types/dataset.ts:1` | `episode_index` is LeRobot ordinal and not the durable identity; `episode_id` often aliases `trace_id`. |
| Frame | `ImageFrameSpec`; LeRobot row fields | `src/vla_lens/workbench/schema.py:147`; `src/vla_lens/capture/lerobot_v3.py:36` | JPEG sequence, video, or LeRobot image column; frame API reads by `trace_id`, `camera`, `timestep` | `FrameRef = trace_id, camera, timestep` | Capture writer; LeRobot reader | Dashboard frame/video routes, workbench episode viewer | Per timestep per camera | No separate frame schema version; derives from trace/LeRobot schema | `frameUrl(traceId, camera, timestep)` in `frontend/src/api/dataset.ts:584` | `frame`, `frame_idx`, `frame_index`, and `timestep` are aliases in workbench axes, but storage mostly uses `timestep`. |
| Task | Manifest fields and LeRobot task index | `src/vla_lens/traces/types.py:16`; `src/vla_lens/dataset/writer.py:441` | Manifest `task_id`, prompt, LeRobot `task_index`, meta tasks | `task_id`; LeRobot also stores `task_index` | Capture runner / dataset writer | Episode index, filters, object-flow, probes | Episode metadata | No independent task schema | `DatasetEpisode.task_id` in `frontend/src/types/dataset.ts:5` | `task_id`, prompt, task name, benchmark, and suite overlap; canonical task object is not first-class. |
| Policy call | `PolicyCallRecord`; `PolicyCallRef` | `src/vla_lens/capture/records.py:19`; `src/vla_lens/interventions/specs.py:124` | `tables/policy_calls.parquet` with `policy_call_index`, `observation_timestep`, `env_timestep_start`, metadata | Recommended canonical key: `trace_id`, `policy_call_index` | PI0.5 capture writer creates records in `src/vla_lens/pi05/capture_writer.py:176`; token metadata also emits policy call rows | Timesteps, token metadata, model arrays, action chunks, probes, intervention preflight, dashboard | One row per model invocation/action chunk | Stored under trace schema; no separate call schema version | `PolicyCall` has positional `index` and stored `model_call_index` in `frontend/src/types/dataset.ts:760`; evidence uses `policy_call` | Main identity split: stored `policy_call_index` vs API `index`/`call_index`/`model_call_index`. |
| Action chunk | Episode array `action_chunks` | `src/vla_lens/pi05/capture_arrays.py:35`; `src/vla_lens/traces/bundle.py:359` | Zarr array axes `policy_call`, `horizon`, `action_dim`; overlay canonicalization points raw action ref to LeRobot `action` in `src/vla_lens/dataset/overlay.py:164` | `trace_id`, `policy_call_index`, optional `action_horizon_index`, `action_dim` | PI0.5 capture writer | Action-generation artifact, metrics, interventions, workbench action panel | One planned final chunk per policy call | Array axes recorded in `array_index` | Workbench axes `policy_call`, `action_horizon`, `action_dim` | Horizon is sometimes called `horizon` in storage and `action_horizon` in workbench. |
| Generated action trajectory | `generation_actions`, `generation_velocities` arrays | `src/vla_lens/pi05/capture_arrays.py:35`; `src/vla_lens/action_generation.py:124` | Zarr axes `policy_call`, `generation_step`, `horizon`, `action_dim` | `trace_id`, `policy_call_index`, `generation_step`, horizon, dim | PI0.5 capture hooks in `_predict_action_chunk` | Action-generation artifact, generation commitment route, workbench | Per denoising/generation step inside a policy call | Array axes in `array_index`; artifact arrays derive dims in `src/vla_lens/workbench/utils.py:300` | `MatrixSeriesResponse` for commitment; action heatmap uses workbench axes | Some summaries call rows `call_index` positionally in `src/vla_lens/action_generation.py:269`. |
| Executed action | LeRobot `action` or native `executed_actions` | `src/vla_lens/dataset/bundle.py:364`; `src/vla_lens/traces/io.py:157` | LeRobot parquet action column or Zarr `executed_actions` axes `timestep`, `action_dim` | `trace_id`, `timestep`, action dim | Environment rollout in `src/vla_lens/pi05/capture_runner.py:298`; LeRobot writer | Metrics, action-generation comparisons, intervention preflight | One executed control action per env timestep | LeRobot v3 plus trace schema | Action norm series in frontend | Storage distinguishes generated chunks from executed actions; some APIs expose generic `actions()` and can hide basis differences. |
| Observation | Robot/image/state/context tables | `src/vla_lens/dataset/writer.py:390`; `src/vla_lens/traces/bundle.py:50` | LeRobot robot fields; frames; overlay context tables `robot_state`, `scene_state`, `camera_state`, prompt/image preprocessing | `trace_id`, `timestep`, plus camera or field | Capture runner buffers observations in `src/vla_lens/pi05/capture_runner.py:280` | Frame routes, context tables, probes, object-flow | Per timestep, with context snapshots around policy calls | Trace table paths in `src/vla_lens/traces/bundle.py:41` | Episode detail arrays and frame URLs | "Observation" is not one schema object; it is a family of state/image/context tables. |
| Model site | `ModelSiteSpec` storage and workbench | `src/vla_lens/traces/types.py:58`; `src/vla_lens/workbench/schema.py:189` | `tables/model_sites.parquet`; model arrays in Zarr | `site_id` or `name`; sample identity adds axes | PI0.5 full-site declarations in `src/vla_lens/pi05/full_capture.py:26`; capture arrays | Selectors, activation APIs, workbench catalog, intervention targets | Declared per trace and array | Storage has rich fields; dashboard index schema at `src/vla_lens/dataset/index.py:62` stores fewer fields | `ActivationSite` and workbench `ModelSiteSpec` in `frontend/src/types/dataset.ts:898` and `frontend/src/types/workbench.ts:75` | `name`, `site_id`, `model_site`, `module`, and `model_path` are all accepted in different places. |
| Model layer | Field on model site | `src/vla_lens/traces/types.py:64`; `src/vla_lens/workbench/catalog.py:190` | Numeric `layer` column in model-site tables/indexes | Not standalone; part of model-site query | Capture declarations | Feature selectors, probe specs, UI filters | Attribute of site | None independently | `layer?: number` in frontend activation/model locus types | Layer alone is not a model signal identity; it needs site/family/token space. |
| Token group / token space | `PI05TokenMetadata`; stream/token-space/token rows | `src/vla_lens/pi05/token_metadata.py:25`; `src/vla_lens/pi05/token_metadata.py:282` | `streams.parquet`, `token_spaces.parquet`, `tokens.parquet` | `trace_id`, `token_space_id`, `token_index`, often policy call | PI0.5 token metadata builder | Token views, model-site token-space refs, intervention preflight | Built per capture and attached to calls | `TOKEN_SPACE_COLUMNS` and `TOKEN_COLUMNS` in `src/vla_lens/pi05/token_metadata.py:37` | Frontend has token-space refs on activation sites | Selector token filtering uses token kind globally; per-call token layout differences are not strongly represented. |
| Activation signal | `ActivationQuery`, `FeatureMatrix`, model-site Zarr | `src/vla_lens/selectors.py:21`; `src/vla_lens/selectors.py:45` | Model-site arrays with axes; cached feature matrices in `.vla_cache/features` | `ModelSignalRef = trace_id, model_site_id/name, axes selectors` | Capture hooks and `TraceBundle.create` model arrays | Probe training, activation endpoints, workbench | Stored per trace; materialized matrices cached | Cache key includes selector and source signatures in `src/vla_lens/selectors.py:99` | `ActivationSliceResponse` in `frontend/src/types/dataset.ts:980` | UI `feature` often means channel/unit index; not the same as feature spec. |
| Attention signal | Model-site tensor with tensor_type `attention` | `src/vla_lens/pi05/capture_arrays.py:477`; `src/vla_lens/server/dataset_summary.py:60` | Zarr axes include `policy_call`, optional `generation_step`, `head`, token axes | `trace_id`, attention site, policy call, generation step, head, query/key token | PI0.5 capture hooks | Attention map/prompt routes, workbench overlays | Per captured attention site | Model-site metadata fields `query_token_space_id`, `key_token_space_id` | `AttentionMapResponse`; API uses `call_index`, `head`, `query_token` | Key-mass/full attention variants share nearby naming; need explicit `tensor_type`, token spaces, and summary fields. |
| Capture profile | `CapturePlan` and profile constants | `src/vla_lens/pi05/capture_schema.py:14`; `src/vla_lens/pi05/capture_schema.py:93` | Manifest metadata and capture reports | `capture_profile`, requested/actual profile | Capture plan resolver and writer | Capture report, dataset index profile, dashboard capability flags | Per capture run / episode | Capture plan metadata has `axis_strategy = policy_call` in `src/vla_lens/pi05/capture_schema.py:178` | Not directly typed in frontend | Profiles are metadata strings, not first-class persisted objects. |
| Capture run / trace record | `TraceRecord`, `EpisodeRecord`, `ModelTraceRecord` | `src/vla_lens/capture/records.py:71`; `src/vla_lens/capture/records.py:128`; `src/vla_lens/capture/records.py:153` | Merged trace bundle manifest, arrays, tables | `trace_id` | Capture runner/writer | TraceBundle, dataset writer | Produced once per episode; may be exported as LeRobot plus overlay | Trace schema `0.3.0`; capture reports have profile fields | Not direct frontend type | "Run" also means analysis run and lens run elsewhere. |
| Overlay | LeRobot VLA Lens overlay | `src/vla_lens/dataset/overlay.py:57`; `src/vla_lens/dataset/overlay.py:114` | `vla_lens/overlay.json`, `vla_lens/tables/episode_refs.parquet`, overlay episode bundles | `episode_index`, `trace_id`, `overlay_path` | LeRobot dataset writer | `TraceDataset.open`, LeRobot bundle reader, dataset index | Created/updated alongside LeRobot episode data | Overlay JSON has `schema_version` in `src/vla_lens/dataset/overlay.py:138` | Hidden under normal dataset APIs | Overlay deliberately excludes robot action/observation/frame arrays in `src/vla_lens/dataset/overlay.py:149`; this boundary should stay explicit. |
| Dataset index | Dashboard materialized views | `src/vla_lens/dataset/index.py:24` | `vla_lens/tables/*.parquet` plus `index_manifest.json` | Dataset fingerprint plus indexed `trace_id` rows | `build_dataset_index` | Server startup validation, listing, evidence adapter | Rebuildable, append-capable | `INDEX_SCHEMA_VERSION = 0.2.0` | `DatasetPayload.index` fields | No materialized policy-call index table, although workbench can query `policy_calls` virtually. |
| Episode annotation | User/server annotation record | `frontend/src/types/dataset.ts:749`; `src/vla_lens/server/dataset.py:9` | Workbench/server JSON sidecar via annotation helpers | `trace_id` | User UI | Episode pages | Mutable note/star metadata | No schema version observed in type | `EpisodeAnnotation` | Only episode-level; no canonical policy-call or model-site annotation type. |
| Event | Object-flow interaction events | `src/vla_lens/pi05/object_flow.py:421` | Artifact table `interaction_events.parquet` | `trace_id`, object name, `event_type`, `onset_timestep` | Object-flow postprocess | Episode interaction UI, probe metadata merge | Derived deterministic labels | Object-flow artifact schema version `1` at `src/vla_lens/pi05/object_flow.py:34` | `EpisodeInteractionLabel`/object metrics | Events are inferred, not ground truth; no stable `event_id`. |
| Object role | Object-flow role rows | `src/vla_lens/pi05/object_flow.py:38`; `src/vla_lens/pi05/object_flow.py:295` | Artifact table `object_roles.parquet` | `trace_id`, `object_index` or `object_name` | Object-flow postprocess | Workflow prepare, interactions UI, target labels | Derived per episode/object | Object-flow schema `1` | Episode interactions response | Role fields are automatic research labels, not manual annotations. |
| Task-flow state | Object-flow timestep labels | `src/vla_lens/pi05/object_flow.py:557` | `timestep_labels.parquet` with `task_phase`, active/next objects, `policy_call_index` | `trace_id`, `timestep` | Object-flow postprocess | Probe target/metadata preparation | Per timestep | Object-flow schema `1` | Not a distinct typed frontend object | `policy_call_index` mapped by span logic in `src/vla_lens/pi05/object_flow.py:760`; exact span semantics should be shared with timesteps table. |
| Feature spec | Probe/workflow feature selector | `src/vla_lens/probes/workflow_spec.py:13`; `src/vla_lens/selectors.py:21` | Probe spec JSON/method payload with selector | Selector hash plus model-site fields | User/API workflow, default specs | Probe training/materialization | Input contract to feature matrix | Probe artifact schema version `3` in `src/vla_lens/probes/workflow_types.py:12` | `ProbeSuiteLensPayload.training_spec` in `frontend/src/types/dataset.ts:540` | "feature" can mean selector, channel/unit, or best probe label. |
| Feature matrix | `FeatureMatrix` | `src/vla_lens/selectors.py:45` | Cached `rows.parquet` and `X.zarr` | Cache key from selector and records | `FeatureView.materialize` | Probe training and score cache | Rebuildable cache | Cache key versioning is implicit in selector/source hash | Not directly exposed except through artifact/probe rows | Exact matrix row population is cached but not always persisted as a durable example manifest. |
| Target | Probe target spec and intervention target spec | `src/vla_lens/probes/workflow_targets.py:14`; `src/vla_lens/interventions/specs.py:288` | Probe method target; intervention run target JSON | Target name/source/column or target site/artifact id | Probe workflow and intervention UI | Training, preflight, evidence | Per analysis/intervention | Probe artifact schema and intervention `SCHEMA_VERSION = 0.1.0` at `src/vla_lens/interventions/specs.py:20` | Intervention draft `target` and probe target display | "Target" means ML label target, object target, and intervention manipulation target. |
| Cohort | `CohortSpec` | `src/vla_lens/workbench/schema.py:313` | Workbench JSON in `vla_lens/workbench/cohorts` | `cohort_id` | Selection resolver / user save | Workbench compare, graph, frontend cohort APIs | Mutable saved subset | No cohort schema version field | `CohortSpec` in `frontend/src/types/workbench.ts:151` | Members currently store `trace_id` and `example_id`; policy-call membership is indirect. |
| Split | Probe split metadata | `src/vla_lens/probes/workflow_artifacts.py:185`; `src/vla_lens/dataset/index.py:105` | Probe artifact method and prediction tables; sidecar metadata | Split column/value per example/episode | Probe workflow prepare/training | Evaluation, probe index, frontend filters | Per run; may be artifact-defined | Probe artifact schema version `3` | `split`, `eval_split`, `split_category` fields | Multiple split fields can disagree; canonical split unit should be explicit. |
| Example | Probe/example rows and workbench examples | `src/vla_lens/probes/workflow_artifacts.py:154`; `src/vla_lens/workbench/selection.py:278` | Probe predictions/scored predictions; workbench derived examples; linked examples in displays | Probe artifact defines example id as selected activation row fields | Probe training, workbench resolver | Evidence, episode lens, cohorts | Per feature-matrix row or representative display row | Probe artifact examples section | `EpisodeProbePrediction`, evidence `RankedMoment`, workbench examples | No global `ExampleRef`; representative episode index intentionally collapses multiple rows. |
| Probe spec | Normalized probe spec | `src/vla_lens/probes/workflow_types.py:26`; `src/vla_lens/probes/workflow_spec.py:13` | JSON/dict spec passed to training and stored in artifact method | No global id before saved artifact | User/API/default workflow | Probe training | Input to probe run | Probe artifact schema version `3` | Training spec fields in lens payload | Method-specific and not a general analysis spec. |
| Probe run / suite | `train_probe_artifact`; `SavedProbeSuite` | `src/vla_lens/probes/workflow_training.py:63`; `src/vla_lens/probes/workflow_types.py:19` | Dataset `LensArtifact` type `probe_suite`; arrays and prediction tables | `artifact_id`; possibly `probe_id` in index equals artifact id | Probe training workflow | Evidence adapter, dashboard index, episode lens UI | Frozen artifact plus mutable score cache | `PROBE_ARTIFACT_SCHEMA_VERSION = 3` | `ProbeDatasetIndex`, `ProbeEvidenceBundle`, `EpisodeLensView` | Score cache can add refreshed predictions outside the frozen artifact population. |
| Generic artifact | `LensArtifact` | `src/vla_lens/artifacts.py:33` | `artifact.json`, artifact arrays, `artifact_index.parquet` | `artifact_id`, `artifact_type`, `scope`; path | Dataset/bundle artifact save APIs and workflows | Artifact browser, index, workbench lens arrays, evidence adapter | Dataset or bundle scoped | No envelope schema version field; typed methods may embed one | `ArtifactRecord` in `frontend/src/types/dataset.ts:702` | Artifact id uniqueness is assumed across dataset index; scope/path also needed for refs. |
| Lens artifact / lens run | Evidence view over artifact | `src/vla_lens/probe_evidence.py:195`; `src/vla_lens/probe_evidence.py:219` | Built from indexed artifact/probe predictions, not necessarily a separate raw artifact | `lens_id`, `lens_version`, `lens_run_id` | Evidence adapter | Probe evidence pages, pins | Read model over durable artifact/index | Evidence bundle family/probe version fields | `ProbeLensArtifact`, `LensRun` in `frontend/src/types/probeEvidence.ts:60` | Lens ids can equal artifact ids; lens run ids like `indexed:<lens>:<dataset>` are adapter-generated. |
| Workbench state | `SelectionState`, `SavedWorkspace`, catalogs | `src/vla_lens/workbench/schema.py:275`; `src/vla_lens/workbench/schema.py:488` | JSON workbench state; manifest payload | `selection_id`, `workspace_id`, axis values | Workbench UI/backend | Selection resolver, cohorts, panels | Mutable UI state | Workbench manifest schema `0.1.0` in `src/vla_lens/workbench/api.py:70` | `SelectionState` in `frontend/src/types/workbench.ts:107` | Parallel evidence selection type exists; no shared frontend adapter. |
| Intervention request | Runtime-free request shell | `src/vla_lens/interventions/specs.py:220`; `src/vla_lens/interventions/specs.py:364`; `src/vla_lens/server/interventions.py:11` | POST payload; optional saved run JSON | Request id only for cohort requests; single requests depend on context | Frontend intervention lab | Preflight and save-run route | Metadata-only preflight unless live runtime added later | Intervention schema `0.1.0` | `InterventionLabDraft` in `frontend/src/types/interventions.ts:55` | Draft uses camelCase and workbench shell uses untyped records; typed request is backend-only. |
| Intervention target | `TargetSpec` | `src/vla_lens/interventions/specs.py:288` | Intervention run target JSON | Source artifact id plus target site fields | Frontend/request payload | Preflight, saved intervention run | Per request/run | Intervention schema `0.1.0` | `InterventionRunRecord.target` | Accepts several site aliases; artifact-derived target requires `source_artifact_id`. |
| Intervention trial | `InterventionTrial` | `src/vla_lens/interventions/results.py:47` | Saved intervention run `trials` list | `trial_id` inside `run_id` | Intervention runtime or inspected-only result builder | Intervention evidence/readouts | Per attempted condition | Intervention schema `0.1.0` | `InterventionRunRecord.readouts.trials` untyped | Trial ids are local to run; controls and outcomes are flexible mappings. |
| Evidence | Probe evidence primitives and intervention readouts | `src/vla_lens/probe_evidence.py:273`; `src/vla_lens/interventions/results.py:390` | Evidence bundle built from index; intervention run JSON | Evidence bundle id or run id plus primitive key | Evidence adapter, intervention save | Evidence UI, pins, workbench | Derived view over artifacts and indexes | Probe evidence bundle validates family/capabilities; intervention schema version | `ProbeEvidenceBundle`, `CurrentMomentEvidence` in `frontend/src/types/probeEvidence.ts:304` | Probe evidence is typed; intervention evidence is mostly saved readout mappings. |
| Claim | Evidence claim level and intervention claim field | `src/vla_lens/probe_evidence.py:85`; `src/vla_lens/interventions/results.py:409` | Probe contribution claim level; intervention run `claim` mapping | No global claim id | Evidence adapter/intervention runner | UI display and causal-evidence heuristic | Attached to evidence/run | No independent claim schema | `EvidenceClaimLevel` in `frontend/src/types/probeEvidence.ts:85` | Claim semantics differ: probe claim level is epistemic; intervention claim is free-form and used for causal flags. |

## Identity And Join Analysis

### 1. Canonical Policy Call Identity

Observed:

- Capture records use `PolicyCallRecord.call_index` and serialize it as `policy_call_index` in `src/vla_lens/capture/records.py:19`.
- PI0.5 capture creates `CaptureCall(call_index, env_timestep, final_action_chunk, denoising_actions, ...)` in `src/vla_lens/pi05/capture_schema.py:325`.
- The rollout creates a new `CaptureCall` only when the action iterator is exhausted and a new chunk is predicted in `src/vla_lens/pi05/capture_runner.py:280`.
- The capture writer records `policy_call_index`, `env_timestep_start`, `env_timestep_end`, model/action metadata, and action horizon/dim in `src/vla_lens/pi05/capture_writer.py:176`.
- Intervention code already defines `PolicyCallRef(trace_id, policy_call_index)` in `src/vla_lens/interventions/specs.py:124`.

Inferred:

- A policy call is one model invocation that observes an env timestep and emits one planned action chunk. Its trace-local row id is `policy_call_index`.

Recommendation:

- Canonical type:

```text
PolicyCallRef = {
  trace_id: string,
  policy_call_index: int
}
```

- Optional display id only:

```text
policy_call_id = "{trace_id}#policy_call:{policy_call_index}"
```

- Do not make positional API list index a durable identity.

### 2. Addressing Across Storage, Backend, And Frontend

Observed storage:

- `timesteps` maps each timestep to `policy_call_index` and `horizon_index` in `src/vla_lens/pi05/capture_tables.py:37`.
- Generation steps store `policy_call_index` and `generation_step` in `src/vla_lens/pi05/capture_tables.py:64`.
- Model/action arrays use the axis name `policy_call` in `src/vla_lens/pi05/capture_arrays.py:35`.
- Workbench axis aliases normalize `call`, `call_index`, `model_call`, and `model_call_index` to `policy_call` in `src/vla_lens/workbench/schema.py:14`.

Observed backend/frontend split:

- Metrics payload emits `index` as list position and `model_call_index` as stored `policy_call_index` in `src/vla_lens/server/metrics.py:36`.
- Activation routes require query `call_index` in `src/vla_lens/server/common.py:304`.
- Frontend activation calls send `call_index` in `frontend/src/api/dataset.ts:371`.
- Discovery episode lens view sends `policy_call_index` in `frontend/src/api/dataset.ts:89`.
- Evidence selection stores `policy_call` in `frontend/src/types/probeEvidence.ts:208`.
- Episode hash serializes the call as `call` in `frontend/src/pages/workbench/episodeRouteModel.ts:38`.
- Intervention draft uses `policyCallIndex` in `frontend/src/types/interventions.ts:55`.

Recommendation:

- Backend routes should accept aliases but return `policy_call_index` wherever a stored call id is meant.
- Keep frontend camelCase at component boundaries if needed, but normalize immediately into `PolicyCallRef` or workbench axis `policy_call`.

### 3. Artifact References And Joins

Observed:

- `LensArtifact` envelope stores `artifact_id`, `artifact_type`, `scope`, `selector`, `method`, `metrics`, `arrays`, `display`, `source_trace_ids`, and path in `src/vla_lens/artifacts.py:33`.
- Dataset artifacts save under dataset artifact root and update `artifact_index` in `src/vla_lens/traces/dataset.py:133`.
- Bundle artifacts save under bundle scope in `src/vla_lens/traces/bundle.py:365`.
- Dataset index merges dataset and bundle artifacts with `artifact_scope`, `trace_id`, and `episode_id` in `src/vla_lens/dataset/index.py:355`.
- Workbench artifact arrays get ids `artifact.<artifact_id>.<array_name>` in `src/vla_lens/workbench/catalog.py:886`.

Inferred:

- `artifact_id` is intended to be globally unique inside the opened dataset. Scope is still required to resolve relative paths safely.

Recommendation:

```text
ArtifactRef = {
  artifact_id: string,
  artifact_type: string,
  scope: "dataset" | "bundle",
  trace_id?: string
}
```

Use `ArtifactArrayRef = ArtifactRef + array_name` for arrays, not path strings alone.

### 4. Axis Conflation Risks

Observed aliases:

- Workbench normalizes `frame`, `frame_idx`, `step`, `time`, and `t` to `timestep` in `src/vla_lens/workbench/schema.py:14`.
- It normalizes `call`, `call_index`, `model_call`, and `model_call_index` to `policy_call` in the same alias map.
- It normalizes `feature`, `channel`, and `neuron` to `unit`.
- It normalizes `horizon` to `action_horizon` and `dim` to `action_dim`.
- Workbench array dims are also remapped by `_axis_names_for_array` in `src/vla_lens/workbench/utils.py:217`.

Recommendation:

- Store physical axes as written by capture (`timestep`, `policy_call`, `generation_step`, `horizon`, `action_dim`) and expose semantic workbench axes (`timestep`, `policy_call`, `generation_step`, `action_horizon`, `action_dim`) with an explicit alias table.
- Do not allow API response fields to silently mix stored index and list position.

### 5. Generated Versus Executed Actions

Observed:

- `action_chunks` stores the final planned action chunk per policy call with axes `policy_call`, `horizon`, `action_dim` in `src/vla_lens/pi05/capture_arrays.py:35`.
- `generation_actions` stores intermediate generated actions with axes `policy_call`, `generation_step`, `horizon`, `action_dim` in `src/vla_lens/pi05/capture_arrays.py:35`.
- `executed_actions` stores environment actions per timestep with axes `timestep`, `action_dim` in `src/vla_lens/pi05/capture_arrays.py:35`.
- LeRobot-backed bundles expose executed actions through LeRobot `action`, while overlay arrays contain model/capture data in `src/vla_lens/dataset/bundle.py:364`.
- Overlay action normalization rewrites `unnormalized_action_array_ref` to LeRobot `action` in `src/vla_lens/dataset/overlay.py:164`.

Recommendation:

- Use these names consistently:
  - `ExecutedActionRef = trace_id, timestep, action_dim`.
  - `ActionChunkRef = trace_id, policy_call_index, action_horizon, action_dim`.
  - `GeneratedActionRef = trace_id, policy_call_index, generation_step, action_horizon, action_dim`.
- Avoid generic "action" unless referring specifically to the LeRobot field or include `basis` and `source`.

### 6. Model Signal Dimensions

Observed:

- Storage `ModelSiteSpec` includes site identity, module, layer, tensor type, token kind, generation step, token spaces, parent/summary/capture metadata in `src/vla_lens/traces/types.py:58`.
- `TraceBundle.create` writes model-site rows with `site_id`, `name`, relative path, axes, dtype, module, layer, family, role, segment, materialization, exactness, token spaces, and capture fields in `src/vla_lens/traces/bundle.py:175`.
- Dashboard dataset index model-site columns are narrower and stop at family/role/segment in `src/vla_lens/dataset/index.py:62`.
- Workbench model-site catalog can group richer fields if they exist in `dataset.model_site_index` in `src/vla_lens/workbench/catalog.py:412`.

Recommendation:

```text
ModelSiteRef = {
  model_site_id: string,
  module?: string,
  layer?: int,
  tensor_type?: string,
  token_kind?: string,
  token_space_id?: string,
  family?: string,
  role?: string,
  segment?: string
}

ModelSignalSampleRef = {
  trace_id: string,
  model_site_id: string,
  policy_call_index?: int,
  timestep?: int,
  generation_step?: int,
  token_space_id?: string,
  token_index?: int,
  head_index?: int,
  unit?: int
}
```

Keep "site" identity separate from "sample" identity.

### 7. Selection Representations

Observed:

- Workbench canonical selection is `SelectionState(selection_id, axis_values, unit_refs, cohort_refs, source_panel_id, intent)` in `src/vla_lens/workbench/schema.py:275`.
- Workbench resolver returns matching episodes, lens arrays, model sites, examples, panels, provenance, and valid refs in `src/vla_lens/workbench/selection.py:82`.
- Frontend workbench type mirrors `SelectionState` in `frontend/src/types/workbench.ts:107`.
- Probe evidence has a separate `ResearchSelectionState(dataset_id, lens_id, lens_run_id, episode_id, timestep, policy_call, ranking, model_locus, feature_id)` in `frontend/src/types/probeEvidence.ts:208`.
- Episode inspector adds `EpisodeInspectorSelection(trace_id, timestep, policy_call_index, model_site_id, layer, feature, mode)` in `frontend/src/types/dataset.ts:331`.
- Intervention draft adds `InterventionLabDraft(traceId, policyCallIndex, modelSite, artifactId, tokenSpace, ...)` in `frontend/src/types/interventions.ts:55`.

Inferred:

- Workbench selection is the intended general contract, while evidence/episode/intervention pages grew task-specific typed selections.

Recommendation:

- Define frontend adapters:
  - `ResearchSelectionState -> SelectionState.axis_values`.
  - `EpisodeInspectorSelection -> SelectionState.axis_values`.
  - `InterventionLabDraft -> PolicyCallRef + ModelSiteRef + ArtifactRef`.
- Persist only canonical workbench/evidence refs; keep URL and component names as view-layer aliases.

## Duplicate Or Overloaded Concepts

| Duplicate concept | Observed variants | Risk | Proposed canonical vocabulary |
|---|---|---|---|
| Episode identity | `trace_id`, `episode_id`, `episode_index`, LeRobot `episode_index` | Joining by ordinal or alias can break across merged datasets | `TraceRef = {trace_id, episode_id?, episode_index?}`; only `trace_id` is stable. |
| Policy call identity | `policy_call_index`, `index`, `model_call_index`, `call_index`, `call`, `policyCallIndex`, `policy_call` | List position can be mistaken for stored id | `PolicyCallRef = {trace_id, policy_call_index}`. |
| Time/frame identity | `timestep`, `env_timestep`, `observation_timestep`, `frame`, `frame_idx`, `timestamp` | Confuses frame display, model observation time, and executed action time | `TimestepRef = {trace_id, timestep}` plus optional timestamp/frame aliases. |
| Model signal identity | `name`, `site_id`, `model_site`, `module`, `model_path`, `feature` | Layer/feature labels can lose actual captured site | `ModelSiteRef` plus `ModelSignalSampleRef`; do not use layer alone. |
| Feature/unit/channel | `feature`, `unit`, `channel`, `neuron`, `dim_*`, probe `feature` labels | Probe feature and activation channel collide | Use `unit` for activation channel; use `FeatureSpec` for selector; use `feature_id` only for interpreted derived features. |
| Action identity | `action`, `executed_actions`, `action_chunks`, `generation_actions`, `final_action_chunk` | Planned, generated, normalized, and executed actions can be compared without basis | Use `ExecutedAction`, `ActionChunk`, `GeneratedActionTrajectory` and record basis. |
| Artifact/lens/run identity | `artifact_id`, `probe_id`, `lens_id`, `lens_run_id`, `analysis_run`, `run_id` | UI may treat an indexed evidence run as the durable artifact | Use `ArtifactRef`; `LensRunRef` for view/application; `AnalysisRunRef` for workflow provenance. |
| Target | Probe target, target object, intervention target, target site | Ambiguous in UI and payloads | Qualify as `ProbeTarget`, `ObjectRoleTarget`, `InterventionTarget`, or `ModelSiteTarget`. |
| Selection | `SelectionState`, `ResearchSelectionState`, `EpisodeInspectorSelection`, URL hash params, intervention draft | Same user focus cannot move across panels without lossy adapters | Canonical `SelectionState`; typed views are adapters. |
| Event/annotation | Episode notes, object-flow events, evidence pins, pipeline annotations | Manual notes and derived events can appear equally authoritative | Separate `ManualAnnotation`, `DerivedEvent`, `EvidencePin`, `PipelineAnnotation`. |

## Proposed Canonical Vocabulary

Recommended durable refs:

```text
DatasetRef = {
  dataset_root: string,
  dataset_id?: string,
  dataset_fingerprint?: string
}

TraceRef = {
  trace_id: string,
  episode_id?: string,
  episode_index?: int
}

FrameRef = {
  trace_id: string,
  camera: string,
  timestep: int
}

PolicyCallRef = {
  trace_id: string,
  policy_call_index: int
}

ActionChunkRef = {
  trace_id: string,
  policy_call_index: int,
  action_horizon?: int,
  action_dim?: int,
  basis?: string
}

GeneratedActionRef = {
  trace_id: string,
  policy_call_index: int,
  generation_step: int,
  action_horizon?: int,
  action_dim?: int,
  basis?: string
}

ExecutedActionRef = {
  trace_id: string,
  timestep: int,
  action_dim?: int,
  basis?: string
}

ModelSiteRef = {
  model_site_id: string,
  module?: string,
  layer?: int,
  tensor_type?: string,
  token_kind?: string,
  token_space_id?: string
}

ModelSignalSampleRef = {
  trace_id: string,
  model_site_id: string,
  policy_call_index?: int,
  timestep?: int,
  generation_step?: int,
  token_index?: int,
  head_index?: int,
  unit?: int
}

ArtifactRef = {
  artifact_id: string,
  artifact_type: string,
  scope: "dataset" | "bundle",
  trace_id?: string
}

ExampleRef = {
  dataset_id?: string,
  trace_id: string,
  timestep?: int,
  policy_call_index?: int,
  model_site_id?: string,
  generation_step?: int,
  token_space_id?: string,
  token_index?: int,
  unit?: int,
  target_name?: string
}
```

Recommended axis names:

- `episode`: categorical trace/episode selection. Values should be trace ids unless explicitly labeled as LeRobot ordinals.
- `timestep`: environment timestep/frame index.
- `policy_call`: model invocation/action chunk index. Value equals stored `policy_call_index`.
- `generation_step`: denoising or internal generation step inside one policy call.
- `action_horizon`: planned action offset inside a chunk.
- `action_dim`: action vector component.
- `token`: token index inside a token space.
- `unit`: activation channel, neuron, feature dimension, or head-local component depending on site type.
- `analysis_run`: workflow run/artifact-producing run id.

## Recommendations

1. Add a virtual or materialized `policy_call_index` table to the dashboard index.

   Workbench can already union `policy_calls` with `trace_id`, `episode_id`, and `bundle_path` through DuckDB in `src/vla_lens/workbench/tables.py:131`, so this can be added without scanning dense arrays. The table should expose `trace_id`, `policy_call_index`, `observation_timestep`, `env_timestep_start`, `env_timestep_end`, action horizon/dim, model id/family, and token-space ids when available.

2. Normalize API response naming.

   Keep accepting `call_index` as a query alias, but emit `policy_call_index` in response objects. If the API also returns a list position, name it `position` or `ordinal`, not `index`.

3. Promote `PolicyCallRef` to shared Python/TypeScript schema.

   Backend already has `PolicyCallRef` in intervention specs. Mirror it in frontend types and use it in evidence pins, episode hashes, intervention drafts, and activation fetch helpers.

4. Preserve model-site richness in browse indexes.

   The raw model-site schema has token-space and materialization/exactness fields, but dashboard index columns are narrower. Add `materialization`, `exactness`, `token_space_id`, `query_token_space_id`, `key_token_space_id`, `parent_site_id`, and `summary_type` to `MODEL_SITE_COLUMNS` or ensure consumers do not rely on the dashboard index for these fields.

5. Persist exact example manifests for probe/lens runs.

   Probe artifacts describe example id construction in `src/vla_lens/probes/workflow_artifacts.py:154`, and prediction tables store many rows, but a generic `examples.parquet` or `example_manifest.parquet` would give downstream workbench/evidence/cohort code a stable population independent of representative summaries.

6. Split "target" into typed subterms in UI and docs.

   Use `ProbeTarget`, `ObjectTarget`, `InterventionTarget`, and `ModelSiteTarget`. This avoids conflating labels, task objects, and internal manipulation loci.

7. Keep LeRobot robot data and VLA Lens overlay boundaries explicit.

   Overlay code intentionally excludes robot action/observation/frame arrays from VLA Lens overlay arrays in `src/vla_lens/dataset/overlay.py:149`. That is the correct ownership split: robot ground-truth data remains LeRobot; model/capture/analysis data lives in `vla_lens`.

## Commands Used

Static/context commands:

```bash
pwd
sed -n '1,240p' AGENTS.md
sed -n '1,260p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt
sed -n '261,620p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt
sed -n '621,980p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt
sed -n '981,1320p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt
sed -n '1321,1520p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt
wc -l /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt
git rev-parse HEAD
git status --short
git status --short -- docs/audits/vla-lens-system-review/01-domain-object-model.md
git status --porcelain=v1 -uall -- docs/audits/vla-lens-system-review/01-domain-object-model.md
rg --files
find docs/audits -maxdepth 3 -type f -print
git ls-files docs/audits/vla-lens-system-review/01-domain-object-model.md
ls -ld docs docs/audits docs/audits/vla-lens-system-review
rg -n "PolicyCall|policy_call|call_index|model_site_id|artifact_id|SelectionState|CohortSpec|LensArray" frontend/src/types frontend/src/pages frontend/src/components frontend/src/api
```

Representative file-inspection commands used repeatedly:

```bash
nl -ba <file> | sed -n '<line-range>p'
```

Files inspected with that pattern:

```text
src/vla_lens/traces/types.py
src/vla_lens/traces/layout.py
src/vla_lens/traces/bundle.py
src/vla_lens/traces/io.py
src/vla_lens/traces/dataset.py
src/vla_lens/capture/records.py
src/vla_lens/capture/lerobot_v3.py
src/vla_lens/dataset/reader.py
src/vla_lens/dataset/bundle.py
src/vla_lens/dataset/overlay.py
src/vla_lens/dataset/writer.py
src/vla_lens/dataset/index.py
src/vla_lens/pi05/capture_schema.py
src/vla_lens/pi05/capture_runner.py
src/vla_lens/pi05/capture_predict.py
src/vla_lens/pi05/capture_writer.py
src/vla_lens/pi05/capture_tables.py
src/vla_lens/pi05/capture_arrays.py
src/vla_lens/pi05/token_metadata.py
src/vla_lens/pi05/full_capture.py
src/vla_lens/pi05/object_flow.py
src/vla_lens/selectors.py
src/vla_lens/artifacts.py
src/vla_lens/probe_evidence.py
src/vla_lens/probe_evidence_adapter.py
src/vla_lens/probes/workflow_types.py
src/vla_lens/probes/workflow_spec.py
src/vla_lens/probes/workflow_training.py
src/vla_lens/probes/workflow_artifacts.py
src/vla_lens/probes/workflow_prepare.py
src/vla_lens/probes/workflow_targets.py
src/vla_lens/probes/score_cache.py
src/vla_lens/action_generation.py
src/vla_lens/target_object.py
src/vla_lens/workbench/schema.py
src/vla_lens/workbench/catalog.py
src/vla_lens/workbench/selection.py
src/vla_lens/workbench/tables.py
src/vla_lens/workbench/api.py
src/vla_lens/workbench/validation.py
src/vla_lens/workbench/utils.py
src/vla_lens/server/common.py
src/vla_lens/server/state.py
src/vla_lens/server/interventions.py
src/vla_lens/server/metrics.py
src/vla_lens/server/fastapi_app.py
src/vla_lens/server/artifacts.py
src/vla_lens/server/dataset.py
src/vla_lens/server/dataset_summary.py
src/vla_lens/server/workbench_payloads.py
src/vla_lens/interventions/specs.py
src/vla_lens/interventions/preflight.py
src/vla_lens/interventions/results.py
frontend/src/types/dataset.ts
frontend/src/types/workbench.ts
frontend/src/types/probeEvidence.ts
frontend/src/types/interventions.ts
frontend/src/api/dataset.ts
frontend/src/api/discoveryArtifactParams.ts
frontend/src/api/selections.ts
frontend/src/api/cohorts.ts
frontend/src/api/lensArrays.ts
frontend/src/store/workbenchStore.ts
frontend/src/pages/workbench/episodeRouteModel.ts
frontend/src/pages/episodes/useEpisodeHashSync.ts
frontend/src/pages/episodes/useEpisodeInspectorModel.ts
frontend/src/pages/episodes/episodeLensModel.ts
```

One attempted file read returned "No such file or directory" and made no changes:

```bash
nl -ba src/vla_lens/workbench/io.py | sed -n '1,360p'
```

Commands intentionally not run:

- No `uv run pytest`, `ruff`, or frontend build, because this audit is static and code behavior was not modified.
- No PI0.5, LeRobot, LIBERO, simulator, Torch, capture, replay, Docker, or model download commands.
- No destructive git commands.
