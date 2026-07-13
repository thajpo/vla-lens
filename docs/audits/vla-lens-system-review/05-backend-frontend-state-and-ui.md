# 05 Backend/Frontend State And UI Audit

## Inspected Revision

- Commit: `882eeb83be8c8a69a80fc8f6ec8829c311ca4630`
- Git status at start: clean short status; branch status `## master...origin/master`
- Git status after writing this audit: untracked audit files under `docs/audits/vla-lens-system-review/`, including this Worker 05 file and peer-worker files `01`, `02`, `03`, `04`, `06`, and `09`. I did not edit those peer-worker files.
- Scope: static inspection only. No captures, simulators, model downloads, server starts, destructive git commands, or test runs.
- Ownership: this audit writes only `docs/audits/vla-lens-system-review/05-backend-frontend-state-and-ui.md`.

## Main Answer

Scientific selection meaning only partially survives UI navigation.

It survives best on the dataset-to-episode path when the selected probe has a `ProbeEvidenceBundle`: `DatasetBrowser.openDatasetEpisode` first asks `probeEvidenceContextForEpisode(...)`, which builds an `EpisodeOpenContext` carrying `researchSelection`, `lensRunId`, `policyCall`, `rankingMode`, `siteName`, and `feature` (`frontend/src/pages/workbench/DatasetBrowser.tsx:422`, `frontend/src/pages/workbench/datasetBrowserModel.ts:737`). `WorkbenchPage.handleOpenDatasetEpisode` then serializes a subset into `#episode/<trace>?probe_id=...&lens_run_id=...&dataset_id=...&call=...&timestep=...&rank=...&feature=...&site=...` (`frontend/src/pages/WorkbenchPage.tsx:104`, `frontend/src/pages/workbench/episodeRouteModel.ts:85`). The episode page rehydrates that into `initialResearchSelection` and a selected probe lens (`frontend/src/pages/WorkbenchPage.tsx:57`, `frontend/src/pages/episodes/useProbeEvidenceLensContext.ts:37`).

It weakens in the more common probe-study/readout path. `probe_study_episodes_payload` filters exact probe rows, then collapses them to one representative row per `trace_id` (`src/vla_lens/server/probe_studies.py:197`, `src/vla_lens/server/probe_studies.py:450`). The frontend fallback `episodeOpenContextForProbe` carries only `probeId`, `policyCall`, and `fromCohort` (`frontend/src/pages/workbench/datasetBrowserModel.ts:722`). It does not carry the selected target, readout id, class, split, exact diagnostic row, or model locus unless the evidence bundle path is available.

It does not survive for dataset filter state, class/confusion cells, most probe diagnostics, comparison navigation, or intervention seeds. Those are local component state or aggregate rows rather than a shareable typed selection.

## Backend And Frontend Map

| User view or workflow | Frontend route and components | Frontend state | Backend endpoint | Backend builder or service | Persistent objects consumed | Notes |
|---|---|---|---|---|---|---|
| App shell and top-level navigation | `App` renders only `WorkbenchPage` (`frontend/src/App.tsx:1`); `AppShell` exposes Dataset, Probes, Evidence (`frontend/src/components/layout/AppShell.tsx:4`) | Hash page plus local React state in `WorkbenchPage`; global Zustand only stores `activeRunId` (`frontend/src/store/workbenchStore.ts:8`) | None directly | None | None | There is no React Router. Hash routing is hand-written in `initialPage` (`frontend/src/pages/WorkbenchPage.tsx:159`). |
| Dataset summary and episode browser | `#dataset`, `DatasetBrowser` (`frontend/src/pages/WorkbenchPage.tsx:54`) | Local query/filter/sort/page/lens/readout state (`frontend/src/pages/workbench/DatasetBrowser.tsx:169`) | `/api/dataset`, `/api/artifacts`, `/api/discovery-artifact-families`, `/api/probe-index`, `/api/probe-studies`, `/api/episodes` (`src/vla_lens/server/fastapi_app.py:81`, `:241`, `:257`, `:402`, `:409`, `:316`) | `indexed_dataset_payload`, `indexed_episodes_payload`, `indexed_probe_index_payload`, `probe_studies_payload` | `EPISODE_INDEX`, `ARTIFACT_INDEX`, `PROBE_EPISODE_INDEX`, probe diagnostics parquet | Dataset filters are not in the URL and are reset by tab navigation. |
| Discovery artifact episode browsing | Same `DatasetBrowser`; selected lens controls switch request path | `selectedLensId`, `selectedProbeReadoutId`, probe split/prediction/cohort filters (`frontend/src/pages/workbench/DatasetBrowser.tsx:175`) | `/api/discovery-artifacts/{artifact_id}/episodes`; `/api/probe-studies/{artifact_id}/episodes` (`src/vla_lens/server/fastapi_app.py:265`, `:417`) | `discovery_artifact_episodes_payload`; `probe_study_episodes_payload` | Artifact registry and probe diagnostic tables | Family registry exposes many artifact kinds, but ranking/readout/lens-view paths currently support `probe_suite` only (`src/vla_lens/server/discovery_artifacts.py:56`, `:94`, `:234`). |
| Open episode from dataset/probe cohort | `DatasetBrowser.openDatasetEpisode` into `#episode/...` (`frontend/src/pages/workbench/DatasetBrowser.tsx:422`) | `EpisodeOpenContext`; sometimes `ResearchSelectionState` | Same as above, then episode endpoints | `probeEvidenceContextForEpisode`, `episodeOpenContextForProbeMoment`, fallback `episodeOpenContextForProbe` | Probe evidence bundle or probe episode index | Strong path carries ranked moment semantics; fallback carries only probe id and policy call. |
| Episode microscope | `#episode/<trace>`, `EpisodesPage`, `EpisodeStageView`, `EpisodeNavigationBar` (`frontend/src/pages/WorkbenchPage.tsx:57`, `frontend/src/pages/episodes/EpisodeStageView.tsx:22`) | Many local states: timestep, playback, overlays, site, mode, attention head/query, feature, patch, prompt/expert token, generation step, probe artifact (`frontend/src/pages/EpisodesPage.tsx:70`) | `/api/episodes/{trace_id}`, `/api/episodes/{trace_id}/neighbors`, `/api/episode-annotations` (`src/vla_lens/server/fastapi_app.py:323`, `:333`, `:214`) | `_episode_payload`, `indexed_episode_neighbors_payload`, annotation persistence | Trace bundle, episode index, local annotation JSON | Previous/next navigation resets timestep and local token/patch state (`frontend/src/pages/EpisodesPage.tsx:113`). |
| Frame and timestep selection | `FramePlaybackControls`, `CameraGrid`, `MetricPlotPanel` (`frontend/src/pages/episodes/FramePlaybackControls.tsx:7`, `frontend/src/pages/episodes/CameraTimeline.tsx:19`, `frontend/src/pages/episodes/MetricPlots.tsx:6`) | Local `timestep`, `isPlayingFrames`, `fps`; hash includes timestep only via probe evidence selection, not arbitrary timeline clicks (`frontend/src/pages/episodes/useEpisodeHashSync.ts:51`) | `/api/frame`, `/api/episode-video`, `/api/episode-metrics`, `/api/object-camera-overlay` (`src/vla_lens/server/fastapi_app.py:340`, `:349`, `:378`, `:473`) | frame/video helpers, `_episode_metrics_payload`, `_object_camera_overlay_payload` | Frames, video, metrics arrays, object overlay data | Metric plot clicks call `onTimestepChange` (`frontend/src/pages/episodes/MetricPlots.tsx:291`), but this does not always become durable selection. |
| Policy-call selection | Timeline call markers and probe jump buttons (`frontend/src/pages/episodes/FramePlaybackControls.tsx:171`, `frontend/src/pages/EpisodesPage.tsx:356`) | Active call is reconstructed from `timestep` by segment bounds (`frontend/src/pages/episodes/useEpisodeInspectorModel.ts:145`) and sometimes serialized as `call` (`frontend/src/pages/episodes/useEpisodeHashSync.ts:48`) | `/api/policy-calls`; downstream activation APIs accept `call_index` (`src/vla_lens/server/fastapi_app.py:357`, `:459`) | `_policy_calls_payload`, `_policy_calls` | `bundle.policy_calls` | Backend returns UI `index` and persisted `model_call_index` (`src/vla_lens/server/metrics.py:54`). Frontend consistently uses `index`, which is stable only if policy calls remain contiguous and sorted the same way. |
| Model-site and activation inspection | `EpisodeInspectorColumn`, `ActivationSitePanel`, `PipelineMap`, `TopChannelPanel`, `LensCompactReadout` (`frontend/src/pages/episodes/EpisodeInspectorColumn.tsx:12`, `frontend/src/pages/episodes/InspectorPanels.tsx:56`) | Local `selectedSiteName`, `inspectionMode`, `feature`, `generationStep`, clip and top-k controls; URL has site/mode/feature only (`frontend/src/pages/episodes/useEpisodeHashSync.ts:54`) | `/api/activation-sites`, `/api/activation-slice`, `/api/image-token-map`, `/api/prompt-feature-map` (`src/vla_lens/server/fastapi_app.py:452`, `:459`, `:466`, `:501`) | `_activation_sites_payload`, `_activation_slice_payload`, `_image_token_map_payload`, `_prompt_feature_map_payload` | Model site index and activation arrays | Site payload is rich (`src/vla_lens/server/activation.py:38`), but selected generation step and many token axes remain local-only. |
| Attention, prompt token, image patch, expert token | `AttentionAxisControls`, `PromptAttentionStrip`, `ActivationGridOverlay`, `ExpertTokenFlow` (`frontend/src/pages/episodes/InspectorPanels.tsx:358`, `:542`, `frontend/src/pages/episodes/CameraTimeline.tsx:394`) | Local head, query token/action slot, selected patch, selected prompt token, selected expert token, generation step | `/api/attention-map`, `/api/patch-features`, `/api/prompt-attention`, `/api/expert-token-activations`, `/api/expert-token-details` (`src/vla_lens/server/fastapi_app.py:480`, `:487`, `:494`, `:508`, `:515`) | attention/activation APIs | Attention arrays, token maps, patch feature values | These are good local inspector controls, but not shareable. |
| Probe result in episode context | `EpisodeProbePanel`, `ArtifactReadoutPanel`, `EpisodeProbeTimeline`, `LensCompactReadout` (`frontend/src/pages/episodes/EpisodeProbePanel.tsx:139`, `:252`, `:487`; `frontend/src/pages/episodes/LensInspectorPanels.tsx:141`) | Selected probe artifact, selected probe site/policy call inferred from rows | `/api/episode-probes`, `/api/discovery-artifacts/{id}/readout`, `/api/discovery-artifacts/{id}/episode-lens-view`, `/api/probes/{id}/evidence-bundle` (`src/vla_lens/server/fastapi_app.py:392`, `:277`, `:289`, `:440`) | `indexed_episode_probes_payload`, `discovery_artifact_readout_payload`, `_probe_suite_episode_lens_view`, `indexed_probe_evidence_bundle_payload` | Probe episode rows, artifact metadata, evidence bundle | This is the best mounted path for raw episode evidence. It still blends legacy readout, episode lens view, and typed evidence bundle contracts. |
| Probe-study page | `#probes`, `ProbeSuitePreset` (`frontend/src/components/workflows/ProbeSuitePreset.tsx:79`) | Local target/split/layer/sort/readout state (`frontend/src/components/workflows/ProbeSuitePreset.tsx:91`) plus global `activeRunId` | `/api/probe-studies` (`src/vla_lens/server/fastapi_app.py:409`) | `probe_studies_payload`, `_study_payload` | Probe diagnostics parquet: layer split, readout battery, null controls, lead time, per-class, confusion, errors (`src/vla_lens/server/probe_studies.py:680`) | Strong summary of training and diagnostics. Only error examples link to episodes (`frontend/src/components/workflows/ProbeSuitePreset.tsx:553`). |
| Probe diagnostics in dataset page | `ProbeDatasetAnalysisPanel`, `ProbeSummaryVisual`, class/confusion/failure cards (`frontend/src/pages/workbench/DatasetBrowser.tsx:1730`, `:2164`, `:2581`) | Local selected readout plus dataset browser filters | `/api/probe-studies/{id}/episodes`, `/api/probe-studies` | `probe_study_episodes_payload`, `_study_payload` | Same diagnostics parquet | Class/confusion panels are aggregate displays. Confusion cells are `<span>`s, not selectable drilldowns (`frontend/src/pages/workbench/DatasetBrowser.tsx:2581`). |
| Observational artifact comparison | `ObservationalComparisonPanel` inside `EpisodeProbePanel` (`frontend/src/pages/episodes/EpisodeProbePanel.tsx:393`) | Local comparison candidate click passes only `trace_id` plus reduced context (`frontend/src/pages/EpisodesPage.tsx:281`) | `/api/observational-comparisons` (`src/vla_lens/server/fastapi_app.py:104`) | `_observational_comparisons_payload` | Episode index and probe rows | Explicitly observational, not causal (`frontend/src/pages/episodes/EpisodeProbePanel.tsx:422`). It cannot preserve the candidate's exact best probe row. |
| Intervention lab and evidence records | `#evidence`, `EvidencePage`, `InterventionLab`, `InterventionRunDetail` (`frontend/src/pages/EvidencePage.tsx:17`) | `interventionSeed` is in-memory only; selected saved run can be URL `#evidence/<run>` (`frontend/src/pages/WorkbenchPage.tsx:150`) | `/api/discovery-artifacts/{id}/target`, `/api/interventions/preflight`, `/api/intervention-runs` (`src/vla_lens/server/fastapi_app.py:304`, `:588`, `:581`) | `discovery_artifact_target_payload`, `_intervention_preflight_payload`, `save_intervention_run` | Workbench intervention run JSON | Current save path records `inspected_only`; live Run button reports unavailable (`frontend/src/components/interventions/interventionLabModel.ts:70`, `frontend/src/components/interventions/InterventionLab.tsx:115`). |
| Generic workbench state, cohorts, saved workspaces | APIs and types exist; generic panels exist but are not mounted by `App` | Backend `SelectionState`; TS `SelectionState`; not the active probe `ResearchSelectionState` | `/api/workbench`, `/api/selections/resolve`, `/api/cohorts/from-selection`, `/api/workspaces`, `/api/lens-arrays/{id}/slice` (`src/vla_lens/server/fastapi_app.py:114`, `:543`, `:557`, `:598`, `:626`) | `workbench_manifest`, `resolve_selection`, `cohort_from_selection`, `save_workspace`, `slice_lens_array` | Workbench JSON, lens arrays, model sites, tables | `HeatmapPanel` emits `SelectionState` (`frontend/src/components/panels/HeatmapPanel.tsx:25`), but no mounted UI calls `resolveSelection`, `saveWorkspace`, or `saveCohortFromSelection` (`rg` result). |

## Selection Semantics

The system currently has at least four selection representations:

- Backend generic workbench `SelectionState`: axis-native `axis_values`, `unit_refs`, `cohort_refs`, `source_panel_id`, `intent` (`src/vla_lens/workbench/schema.py:275`). Axis aliases normalize `frame`, `call`, `feature`, `run`, etc. (`src/vla_lens/workbench/schema.py:14`).
- Frontend generic workbench `SelectionState`: hand-written TypeScript mirror (`frontend/src/types/workbench.ts:107`).
- Frontend probe `ResearchSelectionState`: dataset/lens/run/episode/time/ranking/cohort/model_locus/feature (`frontend/src/types/probeEvidence.ts:208`).
- Episode lens `EpisodeInspectorSelection`: trace/timestep/policy_call/model_site/layer/feature/mode (`frontend/src/types/dataset.ts:331`).

They overlap, but they are not one generated contract. This is the root cause of most semantic loss.

| Selection meaning | URL/global/local/reconstructed | Backend contract | Survival judgment |
|---|---|---|---|
| Dataset id/fingerprint | Dataset filter is local; `dataset_id` enters episode hash only through research selection (`frontend/src/pages/workbench/episodeRouteModel.ts:93`) | `/api/dataset` exposes fingerprint in index payload; `workbench_manifest` exposes `dataset_id` (`src/vla_lens/workbench/api.py:76`) | Partial. Dataset identity survives evidence-bundle deep links, not dataset browsing. |
| Episode/trace | URL path `#episode/<trace>` (`frontend/src/pages/workbench/episodeRouteModel.ts:121`) | `trace_id` in episode index and trace bundle endpoints | Strong. |
| Timestep/frame | Local `timestep`; URL `timestep` only from research selection (`frontend/src/pages/episodes/useEpisodeHashSync.ts:51`) | Generic axis alias maps `frame` to `timestep` (`src/vla_lens/workbench/schema.py:15`) and episode lens reads `timestep` from query | Partial. If `call` is also present, route application jumps to call start/env timestep (`frontend/src/pages/episodes/useEpisodeRouteContext.ts:101`), which can override exact frame intent. |
| Policy call | URL `call`; local active call reconstructed from timestep; APIs use `call_index` | Backend returns `index` and `model_call_index` (`src/vla_lens/server/metrics.py:54`) | Partial. The UI uses `index`, while indexed probe rows name the axis `policy_call_index`; mismatch risk if persisted call ids are not contiguous. |
| Task/outcome/profile/benchmark | Dataset browser local filters (`frontend/src/pages/workbench/DatasetBrowser.tsx:169`) | `/api/episodes` filters them in SQL (`src/vla_lens/server/indexed.py:283`) | Does not survive navigation except as current episode metadata. |
| Object/target object | Generic backend selection axis `object`; probe target/readout local | `resolve_selection` filters object/task/outcome (`src/vla_lens/workbench/selection.py:128`) | Mostly not surfaced in active UI. |
| Artifact/lens/probe | URL `probe_id`; selected lens local on dataset; evidence bundle has `lens_id`/`lens_run_id` | Artifact family endpoints and evidence bundle endpoints | Good for episode links, weak for dataset page links. |
| Lens run | URL `lens_run_id` when evidence bundle path exists | `ProbeEvidenceBundle.run.lens_run_id`; frontend selection stores it (`frontend/src/types/probeEvidence.ts:208`) | Good for ranked evidence path; absent from fallback probe context. |
| Ranking/cohort | URL `rank` and `from=cohort`; dataset cohort presets local | Evidence bundle ranked moments; probe SQL filters cohort presets (`src/vla_lens/server/indexed.py:301`) | Partial. Ranking survives in episode link; cohort filter semantics do not. |
| Model site/layer | URL `site`; layer is implicit in site or readout; local selected site | Activation site payload, model site catalog, probe lens source scope | Partial. Site survives; selected readout layer may not unless encoded by site or evidence bundle. |
| Feature/unit/channel | URL numeric `feature` or `contributor`; generic backend calls this `unit` | Activation slice accepts `feature`; generic alias maps feature/channel/neuron to `unit` (`src/vla_lens/workbench/schema.py:25`) | Partial. Numeric feature survives; semantic contributor id can be parsed but not always preserved in mounted controls. |
| Attention head/query token/action slot | Local only (`frontend/src/pages/episodes/InspectorPanels.tsx:358`) | Attention endpoints accept `head` and `query_token` (`frontend/src/api/dataset.ts:436`) | Does not survive share/deep link. |
| Image patch | Local `SelectedPatch`; generic backend has `image_patch` axis | Patch feature endpoint accepts camera/row/col (`frontend/src/api/dataset.ts:517`) | Does not survive share/deep link. |
| Generation step/action position | Local `generationStep`; generic backend has `generation_step`/`action_horizon`/`action_dim` axes | Activation slice and attention APIs accept `generation_step` (`frontend/src/api/dataset.ts:371`, `:436`) | Does not survive share/deep link. |
| Probe target/readout/split/class | Local selected readout and filters; error browser links encode only episode hash | Probe studies expose target/layer/split/readouts and diagnostic tables (`src/vla_lens/server/probe_studies.py:229`) | Weak. Summary survives in mounted page state, not in a cross-page selection. |
| Confusion cell | Aggregate UI only; no click selection (`frontend/src/pages/workbench/DatasetBrowser.tsx:2581`) | Confusion parquet loaded and exposed as records (`src/vla_lens/server/probe_studies.py:686`) | Does not survive; cannot open exact moment from a confusion cell. |
| Comparison run/candidate | Local candidate click passes trace id and reduced current context (`frontend/src/pages/EpisodesPage.tsx:281`) | Observational comparison payload includes source/candidate contract (`frontend/src/types/dataset.ts:645`) | Weak. Candidate exact probe row is not carried into the episode route. |
| Intervention target | In-memory seed to Evidence page; saved run id is URL-able after save | Target payload and preflight/save endpoints | Partial. Pre-save target is not deep-linkable; saved record is an inspected/preflight record, not a live run. |

## URL, Global, Local, And Reconstructed State

URL state:

- `#dataset`, `#probes`, `#evidence`, `#evidence/<run>`, `#episode/<trace>`.
- Episode query supports `probe_id`, `lens_run_id`, `dataset_id`, `call`, `timestep`, `rank`, `feature`, `contributor`, `ranking`, `mode`, `site`, and `from=cohort` (`frontend/src/pages/workbench/episodeRouteModel.ts:33`, `frontend/src/pages/episodes/useEpisodeHashSync.ts:23`).

Global state:

- Only `activeRunId` for the Probes page (`frontend/src/store/workbenchStore.ts:8`).

Local state:

- Dataset browser filters, selected lens, selected readout, probe split/prediction/cohort filters, and pagination (`frontend/src/pages/workbench/DatasetBrowser.tsx:169`).
- Episode microscope controls: playback, overlays, selected site, mode, attention head/query, feature, clip/top-k, patch, prompt/expert token, generation step, plot tab, selected probe (`frontend/src/pages/EpisodesPage.tsx:70`).
- Intervention lab draft before save (`frontend/src/components/interventions/InterventionLab.tsx:47`).

Reconstructed state:

- Active policy call is reconstructed from current timestep and call segment (`frontend/src/pages/episodes/useEpisodeInspectorModel.ts:145`).
- Probe evidence selection is reconstructed by mixing default bundle selection, initial route selection, active trace, current timestep, and current policy call (`frontend/src/pages/episodes/episodeLensModel.ts:108`).
- Episode lens default selection can auto-apply a recommended site/call/timestep unless route fields suppress it (`frontend/src/pages/episodes/useEpisodeLensView.ts:122`).

Practical consequence: a shared episode link is useful, but it is not a lossless research state. A shared dataset/probe diagnostics view is not currently possible without manual reproduction.

## Backend/Frontend Contract Gaps

1. Selection model split. The backend generic `SelectionState` can resolve episodes, arrays, model sites, panels, provenance, and valid refs (`src/vla_lens/workbench/selection.py:82`). The mounted probe UI uses `ResearchSelectionState` and `EpisodeInspectorSelection` instead. TypeScript mirrors are hand-written (`frontend/src/types/workbench.ts:107`, `frontend/src/types/probeEvidence.ts:208`), so there is no single generated source of truth.

2. Policy-call id ambiguity. Backend `PolicyCall` uses `index` as loop position and `model_call_index` as persisted call id (`src/vla_lens/server/metrics.py:54`). Frontend deep links, activation queries, and jump logic use `call.index` (`frontend/src/pages/episodes/useEpisodeInspectorModel.ts:166`, `frontend/src/pages/EpisodesPage.tsx:356`). Probe rows call their field `policy_call_index` (`src/vla_lens/server/indexed_probes.py:217`). If those ever diverge, selection meaning will drift.

3. Representative episode rows collapse exact rows. `probe_study_episodes_payload` filters exact target/layer/split rows, then drops duplicate traces (`src/vla_lens/server/probe_studies.py:197`, `:463`). This makes browsing efficient but turns "this policy-call row" into "best representative episode."

4. Aggregate diagnostics lack drilldown contracts. Class performance, train gaps, confusion matrix, and high-confidence failures are rendered as aggregate cards (`frontend/src/pages/workbench/DatasetBrowser.tsx:2280`). Confusion matrix cells are non-interactive spans (`frontend/src/pages/workbench/DatasetBrowser.tsx:2602`). The Probes page error browser links episodes (`frontend/src/components/workflows/ProbeSuitePreset.tsx:577`), but not full target/layer/split/readout state.

5. Probe evidence bundle lacks contribution breakdown. The adapter intentionally reports contribution unavailable when aligned activations and probe weights are absent from the indexed path (`src/vla_lens/probe_evidence_adapter.py:486`). Episode UI therefore falls back to raw activations for some "probe contribution" questions.

6. Evidence selection matching can over-match. `timeMatchesSelection` returns true if either side lacks timestep or policy call (`frontend/src/types/probeEvidence.ts:819`). That is useful for coarse evidence, but it can display a primitive as relevant to a precise selected call/frame even when the primitive does not carry that axis.

7. Non-probe discovery families are registered but not usable in the same UI paths. The family registry includes contrast directions, activation clusters, SAE/transcoder/crosscoder features, attention maps/edges (`src/vla_lens/interventions/families.py:83`), but discovery ranking, readout, and episode LensView reject non-`probe_suite` artifacts (`src/vla_lens/server/discovery_artifacts.py:56`, `:94`, `:234`).

8. Saved workspace/cohort APIs are not mounted. `/api/workbench`, `/api/selections/resolve`, `/api/cohorts/from-selection`, `/api/workspaces`, and generic panels exist (`src/vla_lens/server/fastapi_app.py:114`, `:543`, `:557`, `:598`; `frontend/src/components/panels/HeatmapPanel.tsx:152`), but no active app path calls `resolveSelection`, `saveWorkspace`, or `saveCohortFromSelection`.

9. Intervention seed is not shareable pre-save. `handleSendToIntervention` moves to `#evidence` and stores the seed in memory (`frontend/src/pages/WorkbenchPage.tsx:150`). Refreshing or sharing loses it. Saved intervention records are preflight/inspected records (`frontend/src/components/interventions/interventionLabModel.ts:61`).

## Direct Workflow Answers

- Deep link to exact scientific selection: partial. Exact enough for episode/probe/site/feature/call in common cases; not exact for dataset filters, target/readout/class/split, attention head/query, patch, generation step, comparison candidate, or intervention seed.
- Share confusion matrix cell and open exact moment: no. Confusion cells are aggregates without a selectable row id or route contract.
- Move from artifact summary to raw episode evidence: possible but uneven. Ranked evidence rows preserve more context; probe-study rows and aggregate diagnostic cards do not.
- Carry the same selection across Dataset, Episode, Probes, Evidence: no. The mounted app has page-local state and multiple selection types.
- Is one typed model shared by backend and frontend: no. There are backend dataclasses and hand-written TS types, but no generated shared schema used as the active UI contract.

## Artifact Rendering Inventory

- Probe suite summaries: rendered by `ProbeLensWorkbench`, `ProbeSummaryVisual`, `ProbeDatasetAnalysisPanel`, and `ProbeSuitePreset` from `/api/probe-index`, `/api/probe-studies`, `/api/probe-studies/{id}/episodes`, and `/api/probes/{id}/evidence-bundle`.
- Episode probe readouts: rendered by `EpisodeProbePanel`, `ArtifactReadoutPanel`, `EpisodeProbeTimeline`, and `LensCompactReadout` from `/api/episode-probes`, discovery readout, episode LensView, and evidence bundle APIs.
- Activation and attention: rendered in browser from array-backed API responses, not pre-rendered images. `ActivationGridOverlay` draws heatmaps on canvas (`frontend/src/pages/episodes/CameraTimeline.tsx:394`); model sites and slices come from activation endpoints (`src/vla_lens/server/activation.py:38`, `:273`).
- Offline probe diagnostics: parquet diagnostics are loaded server-side and serialized as tables/summaries (`src/vla_lens/server/probe_studies.py:680`). Browser renders them as tables, bars, scatter plots, class cards, and confusion cells.
- Generic workbench arrays: `/api/lens-arrays/{id}/slice` returns bounded JSON previews, explicitly not full tensor transport (`src/vla_lens/workbench/api.py:294`).
- Media: frames and videos are served as `/api/frame` and `/api/episode-video`, then composed in `CameraGrid` and MP4 links (`frontend/src/pages/episodes/CameraTimeline.tsx:133`, `frontend/src/pages/episodes/FramePlaybackControls.tsx:105`).

## Probe Result Usability

| Question a researcher asks | Current answerability | Evidence |
|---|---|---|
| What question does this probe answer? | Easy | Probe pages show Prediction/Input/Output/Objective and question labels (`frontend/src/components/workflows/ProbeSuitePreset.tsx:175`, `frontend/src/pages/workbench/DatasetBrowser.tsx:855`). |
| What split, score, and training summary am I looking at? | Easy in Probes page, possible in Dataset page | Readout inspector shows split, layer, metrics, train gap, rows, labels (`frontend/src/components/workflows/ProbeSuitePreset.tsx:381`). |
| Which episodes, policy calls, model sites, layers, and features support the claim? | Possible but awkward | Episode panel shows site/call/readout rows (`frontend/src/pages/episodes/EpisodeProbePanel.tsx:139`), but aggregate pages collapse rows and do not preserve all axes. |
| Can I inspect one raw row from a summary? | Possible for ranked moments and Probes-page error examples; unclear or not possible for class/confusion summaries | Error browser links episodes (`frontend/src/components/workflows/ProbeSuitePreset.tsx:577`); Dataset confusion matrix is not clickable (`frontend/src/pages/workbench/DatasetBrowser.tsx:2602`). |
| Where does performance concentrate by class/task/outcome? | Possible but awkward | Dataset analysis renders class/task/outcome cards (`frontend/src/pages/workbench/DatasetBrowser.tsx:2280`), but no exact class/task drilldown state. |
| Does it beat controls or look like leakage/overfitting? | Possible | Null controls and train-to-heldout gaps are displayed (`frontend/src/components/workflows/ProbeSuitePreset.tsx:488`, `frontend/src/pages/workbench/DatasetBrowser.tsx:2281`). |
| Which failures should I inspect first? | Easy to find, awkward to open exactly | High-confidence wrong filters and failure cards exist (`frontend/src/pages/workbench/DatasetBrowser.tsx:1786`, `:2703`), but failure cards do not open exact rows in Dataset view. |
| How does the probe change across time/policy calls? | Possible | Episode timeline and temporal evidence maps exist (`frontend/src/pages/episodes/EpisodeProbePanel.tsx:487`, `frontend/src/pages/workbench/DatasetBrowser.tsx:2293`). |
| Can I compare two probe configurations or layers side by side? | Mostly not possible | Readout table can sort/select one readout (`frontend/src/components/workflows/ProbeSuitePreset.tsx:282`), but there is no side-by-side comparison state. Observational episode comparison is not probe-config comparison. |
| Can I turn a result into a cohort or intervention? | Partial | Intervention seed path exists from episode lens/probe (`frontend/src/pages/episodes/useEpisodeLensView.ts:148`); cohort-from-selection backend exists but is not mounted. |

## Proposed Information Architecture

Use one serializable `InvestigationSelection` as the cross-page state contract. It should be a small superset of the current backend `SelectionState`, frontend `ResearchSelectionState`, and `EpisodeInspectorSelection`:

```text
dataset_id, dataset_fingerprint
artifact_id, artifact_type, lens_id, lens_run_id, readout_id
trace_id, episode_id, timestep, policy_call_index, model_call_index
task_id, outcome, profile, benchmark
target, split, class_actual, class_predicted, cohort_preset, cohort_id
model_site_id, layer, token_kind, token_space_id
feature_id, unit, attention_head, query_token, action_slot, image_patch, generation_step
ranking, comparison_source_id, comparison_candidate_id
intervention_target, intervention_operator, evidence_pin_id
```

Recommended page layout:

1. Scope: dataset, artifact/lens, target/readout, split/cohort, current trace/call/site. This should be visible and shareable on every page.
2. Summary: question, prediction/input/output/objective, score, split, verdict, confidence, row counts.
3. Localization: model site, layer, token kind, feature/unit, image patch, prompt/expert token, attention head/query.
4. Temporal: timestep, policy call, generation step, action horizon, timeline marks, lead-time bins.
5. Class/task/errors: class mix, confusion cells, task/outcome slices, high-confidence failures. Every cell should emit a selection with target/split/class/readout and open a filtered evidence row list.
6. Provenance and controls: training recipe, null controls, train/eval gap, source artifact, dataset fingerprint.
7. Comparisons: side-by-side selected readouts or episodes, with source and candidate selections both serialized.
8. Follow-up actions: save evidence pin, save cohort, preflight intervention, save inspected record. These should all consume and emit the same selection envelope.

Backend payload recommendations:

- Keep: `workbench_manifest`, `resolve_selection`, `ProbeEvidenceBundle`, `EpisodeLensView`, and `ProbeStudy` as the main building blocks.
- Generalize: `ProbeStudyReadout` and diagnostic records should include stable row ids plus `trace_id`, `policy_call_index`, `model_site_id`, `layer`, `target`, `split`, `actual`, `predicted`, and `readout_id` everywhere a card/cell can be clicked.
- Consolidate duplicates: probe index, probe studies, discovery artifact readout, and evidence bundle all describe probe identity/readout. Prefer one discovery-artifact lens contract that every family implements.
- Add missing API: a drilldown endpoint for diagnostic cells, for example `POST /api/artifact-diagnostics/query`, accepting the unified selection and returning exact evidence rows plus episode hash targets.
- Add generated contract: generate TypeScript types from Python/OpenAPI or a shared schema file. Hand-maintained mirrors are already diverging in names and selection shape.
- Add route/save support: serialize the unified selection into hash/query and saved workspaces. Dataset filters, readout selection, diagnostic cells, comparisons, and intervention drafts should round-trip.

## Commands Used

Safe static inspection commands were used only. Representative command log:

```bash
pwd && git rev-parse HEAD && git status --short
git status --short --branch
git status --short --untracked-files=all
git diff -- docs/audits/vla-lens-system-review/05-backend-frontend-state-and-ui.md
sed -n '1,240p' AGENTS.md
sed -n '1,260p' docs/research_ui_principles.md
sed -n '1,260p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt
sed -n '261,620p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt
sed -n '621,980p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt
sed -n '981,1320p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt
rg --files -g '!*node_modules*'
rg -n "@(app\.|router\.|.*\.get|.*\.post|.*\.put|.*\.delete)|APIRouter|FastAPI|add_api_route|include_router" src/vla_lens/server src/vla_lens/workbench scripts/serve_vla_lens_app.py
rg -n "fetch\(|apiGet|apiPost|/api/|build.*Url|URLSearchParams|location\.hash|window\.location|selected|artifact|lens|probe|episode" frontend/src -g '*.ts' -g '*.tsx'
rg -n "<HeatmapPanel|<ConfusionMatrixPanel|<ExamplesPanel|resolveSelection\(|saveWorkspace\(|saveCohortFromSelection\(" frontend/src -g '*.tsx' -g '*.ts'
ls -la docs/audits docs/audits/vla-lens-system-review
sed -n '1,120p' docs/audits/vla-lens-system-review/05-backend-frontend-state-and-ui.md
wc -l docs/audits/vla-lens-system-review/05-backend-frontend-state-and-ui.md
LC_ALL=C grep -n '[^ -~]' docs/audits/vla-lens-system-review/05-backend-frontend-state-and-ui.md || true
```

Line-numbered reads used `nl -ba <file> | sed -n '<range>p'` over these files:

```text
src/vla_lens/server/fastapi_app.py
src/vla_lens/server/indexed.py
src/vla_lens/server/indexed_probes.py
src/vla_lens/server/probe_studies.py
src/vla_lens/server/discovery_artifacts.py
src/vla_lens/server/episode_lens_probe.py
src/vla_lens/server/metrics.py
src/vla_lens/server/activation.py
src/vla_lens/workbench/schema.py
src/vla_lens/workbench/selection.py
src/vla_lens/workbench/api.py
src/vla_lens/probe_evidence.py
src/vla_lens/probe_evidence_adapter.py
src/vla_lens/interventions/families.py
frontend/src/App.tsx
frontend/src/pages/WorkbenchPage.tsx
frontend/src/pages/workbench/DatasetBrowser.tsx
frontend/src/pages/workbench/datasetBrowserModel.ts
frontend/src/pages/workbench/episodeRouteModel.ts
frontend/src/pages/EpisodesPage.tsx
frontend/src/pages/EvidencePage.tsx
frontend/src/pages/evidencePinsModel.ts
frontend/src/pages/episodes/useEpisodeHashSync.ts
frontend/src/pages/episodes/useEpisodeRouteContext.ts
frontend/src/pages/episodes/useEpisodeInspectorModel.ts
frontend/src/pages/episodes/useEpisodeLensView.ts
frontend/src/pages/episodes/useProbeEvidenceLensContext.ts
frontend/src/pages/episodes/episodeLensModel.ts
frontend/src/pages/episodes/EpisodeStageView.tsx
frontend/src/pages/episodes/EpisodeProbePanel.tsx
frontend/src/pages/episodes/LensInspectorPanels.tsx
frontend/src/pages/episodes/InspectorPanels.tsx
frontend/src/pages/episodes/CameraTimeline.tsx
frontend/src/pages/episodes/FramePlaybackControls.tsx
frontend/src/pages/episodes/MetricPlots.tsx
frontend/src/components/layout/AppShell.tsx
frontend/src/components/workflows/ProbeSuitePreset.tsx
frontend/src/components/interventions/InterventionLab.tsx
frontend/src/components/interventions/interventionLabModel.ts
frontend/src/components/panels/HeatmapPanel.tsx
frontend/src/components/panels/ConfusionMatrixPanel.tsx
frontend/src/components/panels/ExamplesPanel.tsx
frontend/src/types/dataset.ts
frontend/src/types/probeEvidence.ts
frontend/src/types/workbench.ts
frontend/src/api/dataset.ts
frontend/src/store/workbenchStore.ts
```

One attempted static read used the wrong path and failed before the corrected read:

```bash
nl -ba src/vla_lens/server/probe_evidence_adapter.py | sed -n '42,120p'
```

Not run: `uv run pytest`, `uv run ruff`, dev server, capture scripts, simulators, model downloads, PI0.5/LeRobot/LIBERO commands, or any destructive git command.
