# 09 - Questions for Owner

Inspected commit: `882eeb83be8c8a69a80fc8f6ec8829c311ca4630`

Creation status: the worktree was clean before `docs/audits/vla-lens-system-review/` was generated. The audit was later preserved on the dedicated `codex/system-review-audit` branch.

Commands used: `git rev-parse HEAD`; `git status --short`; `rg` over `src`, `tests`, `frontend/src`, `docs`, and `scripts`; targeted `nl -ba ... | sed -n ...` reads of storage, workbench, probe, server, frontend, capture, and test files.

## Intended Scientific Scope

| Question | Why It Changes Architecture | Repository Alternatives Visible Today | Default Assumption |
|---|---|---|---|
| Should VLA Lens optimize first for observational representation analysis, or for causal intervention studies? | The current workbench can store `AnalysisRunSpec` and `InterventionRunSpec`, but intervention records are saved readouts, not live execution requests (`src/vla_lens/workbench/schema.py:424`, `src/vla_lens/workbench/schema.py:452`). A causal-first product needs stronger trial/control/result contracts before broad UI work. | Probe/artifact-first workflow in `src/vla_lens/probes/workflow_training.py:63`; intervention preflight/save routes in `src/vla_lens/server/fastapi_app.py:581`; current summary in `docs/interventions.md`. | Treat interventions as the next vertical slice after making observational experiment recipes and source examples first-class. |
| Is a "claim" meant to be a durable scientific object in the product, or just a human interpretation in reports/docs? | If claims are durable, artifacts need citation/lineage/control fields beyond current `LensArtifact` maps (`src/vla_lens/artifacts.py:33`). If not, evidence records can stay method outputs plus provenance. | `LensArtifact` stores selector/method/metrics/display, while workbench runs store inputs/outputs/provenance (`src/vla_lens/workbench/schema.py:424`). No explicit `Claim` type is implemented. | Do not add a claim system yet; add evidence relationships and controls to artifacts/runs first. |
| Which evidence threshold should distinguish "debugging sanity check" from "research evidence"? | UI copy already warns that train-episode probe evidence is a sanity check (`docs/research_ui_principles.md:37`). Architecture changes if evidence needs enforced split/control gates. | Probe artifacts store split summaries and baseline metrics (`src/vla_lens/probes/workflow_training.py:184`, `src/vla_lens/probes/workflow_training.py:252`) but no generic evidence-rating contract. | Encode split/control availability as facts, not as a global badge, until a research policy is explicit. |

## Intended Users

| Question | Why It Changes Architecture | Repository Alternatives Visible Today | Default Assumption |
|---|---|---|---|
| Is the primary user a single local researcher, a small lab, or external users of a packaged tool? | Multi-user/collaboration changes artifact identity, locking, provenance, and path portability. Current storage is local dataset-root JSON/Parquet/Zarr. | Local FastAPI server and React workbench (`src/vla_lens/server/fastapi_app.py:51`, `frontend/src/App.tsx:1`); dataset-local workbench stores in JSON (`src/vla_lens/workbench/api.py:99`). | Optimize for a single local researcher plus shareable dataset roots, not concurrent multi-user service semantics. |
| Should non-SWE ML researchers write YAML/CLI recipes, Python notebooks, or mostly use the UI? | This determines whether experiment recipe APIs should be Python-first, CLI-first, or UI-generated. | Probe YAML support exists (`src/vla_lens/probes/workflow_spec.py:40`; `scripts/train_vla_lens_probe.py:230` in README example), while workbench selection/cohort APIs exist but are not the training entrypoint. | Support Python plus YAML recipes first; make UI generate or edit those recipes later. |
| How much low-level tensor vocabulary should be visible by default? | The UI principles prefer researcher-native labels and hiding dtype/storage details (`docs/research_ui_principles.md:116`, `docs/research_ui_principles.md:207`). If users are model engineers, more tensor detail should be surfaced. | Workbench axes expose model/time/unit concepts (`src/vla_lens/workbench/catalog.py:150`); frontend copy deliberately renames probe fields (`frontend/src/copy/researchCopy.ts:1`). | Keep default views conceptual, with explicit debug/provenance drilldowns. |

## Public Versus Internal APIs

| Question | Why It Changes Architecture | Repository Alternatives Visible Today | Default Assumption |
|---|---|---|---|
| Which APIs are intended to be stable public contracts: `TraceDataset`, `ActivationQuery`, workbench APIs, or CLI scripts? | Stable APIs need versioning, migration tests, and documentation. Internal helpers can evolve faster. | Public-ish imports are exported from `src/vla_lens/__init__.py`; `ActivationQuery` is the probe feature selector (`src/vla_lens/selectors.py:21`); many workflows use underscore helpers. | Stabilize storage readers/writers, `TraceDataset`, `ActivationQuery`, and future experiment recipes; treat server payload helpers as internal until schemas are generated. |
| Should `LensArtifact` remain the one artifact envelope for every method? | If yes, it needs typed payload conventions and lineage fields. If not, method-specific artifact schemas need a common registry. | `LensArtifact` is intentionally small and generic (`src/vla_lens/artifacts.py:33`); probes pack rich schema into `method` and `display` (`src/vla_lens/probes/workflow_training.py:175`). | Keep `LensArtifact`, but add typed `method_contract`/`example_manifest` conventions rather than replacing it. |

## Expected Dataset Scale

| Question | Why It Changes Architecture | Repository Alternatives Visible Today | Default Assumption |
|---|---|---|---|
| What is the near-term scale target: 100, 1,000, 10,000, or more episodes? | This determines whether eager feature materialization and Parquet indexes are enough, or whether streaming/query planning is required for SAEs/transcoders. | Feature matrices materialize to `X.zarr` plus `rows.parquet` (`src/vla_lens/selectors.py:71`); dataset indexes are rebuildable materialized views (`src/vla_lens/dataset/index.py:1`). Configs include 100, 500, and 1,000 episode capture plans. | Design for 1,000 episodes locally; add streaming before large SAE/transcoder runs. |
| Are datasets expected to move between machines or external drives frequently? | Current records use relative overlay paths in many places, but some indexes include bundle paths. Portability tests and path policy depend on this. | Artifact output paths are tested to stay dataset-root relative (`tests/lerobot_dataset_storage_test.py:369`); dashboard indexes include `path` fields (`src/vla_lens/dataset/index.py:300`). | Assume dataset roots move; persisted science records should avoid absolute paths. |

## Runtime Deployment

| Question | Why It Changes Architecture | Repository Alternatives Visible Today | Default Assumption |
|---|---|---|---|
| Should capture run only locally/on owned hardware, or also as a remote job service? | Remote capture needs job specs, resumability, log shipping, and artifact registration. | Current split is local wrapper scripts and Docker capture wrappers; README warns not to use normal `uv run` for PI0.5 capture (`README.md:152`). | Keep capture as explicit local/Docker jobs; make outputs inspectable without runtime. |
| Should a future intervention runtime execute inside the dashboard process or only through capture-runtime wrappers? | Running in the dashboard would violate the current dependency split. External execution needs request/result files and preflight. | Dashboard can save intervention readouts (`src/vla_lens/workbench/api.py:136`) and preflight requests (`src/vla_lens/server/fastapi_app.py:588`), but capture runtime lives in PI0.5 wrappers. | Keep live intervention execution outside the dashboard process. |

## Frontend Workflow

| Question | Why It Changes Architecture | Repository Alternatives Visible Today | Default Assumption |
|---|---|---|---|
| Is the central UI object an episode, an artifact/lens, or a saved research workspace? | URL/state design and page boundaries depend on this. | `WorkbenchPage` routes between dataset, episode, probes, and evidence pages (`frontend/src/pages/WorkbenchPage.tsx:19`); workbench schema already supports saved workspaces (`src/vla_lens/workbench/schema.py:488`). | Make the central object a serializable selection/workspace, with episode and artifact as axes inside it. |
| Should every analysis plot support click-through to an exact policy call? | This requires all plot payloads to preserve source row IDs, not just aggregate metrics. | Probe predictions preserve `trace_id`, `policy_call_index`, `timestep`, `model_site_id` in the dashboard index (`src/vla_lens/dataset/index.py:98`), but not every artifact type has this contract. | Yes for research-facing evidence plots; no for pure summary/debug plots. |

## Artifact Permanence

| Question | Why It Changes Architecture | Repository Alternatives Visible Today | Default Assumption |
|---|---|---|---|
| Are artifacts immutable once saved? | Mutable artifacts make comparison and provenance harder; immutable artifacts require replacement/versioning UX. | `save_artifact` drops duplicate artifact IDs keeping last (`src/vla_lens/traces/dataset.py:177`; `src/vla_lens/traces/bundle.py:402`). | Treat scientific artifacts as immutable by convention; use new IDs for reruns, with explicit supersedes lineage later. |
| Should caches be allowed to disappear without invalidating artifacts? | If yes, artifacts must store exact population/recipe/outputs independent of `.vla_cache`. | Feature matrices cache under `.vla_cache/features` (`src/vla_lens/selectors.py:109`); probe artifacts store feature matrix fingerprints and row fingerprints (`src/vla_lens/probes/workflow_artifacts.py:82`). | Caches are disposable; artifacts must remain interpretable without them. |

## Collaboration / Multi-User Expectations

| Question | Why It Changes Architecture | Repository Alternatives Visible Today | Default Assumption |
|---|---|---|---|
| Do saved workspaces, annotations, and evidence pins need authorship/review state? | Multi-user review needs author IDs, timestamps, conflicts, and status transitions. | Evidence pins persist local research selections (`tests/evidence_pins_test.py:9`); `SavedWorkspace` has no author field (`src/vla_lens/workbench/schema.py:488`). | Defer multi-user fields; add optional `created_by` later if collaboration becomes real. |
| Should failed/partial runs be shared as first-class records? | This affects run schemas and UI filters. | `InterventionRunSpec` readouts can encode statuses, and causal evidence treats `ok`/`partial` specially (`src/vla_lens/workbench/api.py:64`). Probe failures are mostly exceptions before artifact save. | Yes for interventions and long-running analyses; add partial/failure run records before remote execution. |

## Model and Dataset Expansion

| Question | Why It Changes Architecture | Repository Alternatives Visible Today | Default Assumption |
|---|---|---|---|
| What is the second model family likely to be? | Site ontology and adapter boundaries should be validated against the next model, not an abstract plugin ideal. | Generic capture records exist (`src/vla_lens/capture/records.py:97`) but PI0.5-specific site names dominate capture and tests. | Choose one second VLA model/dataset as a validation target before designing a broad plugin framework. |
| Are non-LIBERO datasets in scope soon? | The environment/task/object semantics layer changes if real robot or other sim datasets are near-term. | LeRobot v3 is canonical storage; PI0.5/LIBERO context extraction is specific (`src/vla_lens/pi05/context_capture.py`). | Keep LeRobot as storage contract; isolate LIBERO object-flow labels as optional annotations. |

## Definitions of Evidence

| Question | Why It Changes Architecture | Repository Alternatives Visible Today | Default Assumption |
|---|---|---|---|
| What counts as a useful control for intervention evidence: no-op, random direction, matched feature, or behavioral counterfactual? | Controls determine artifact schemas, run grouping, and comparison UI. | Intervention family/spec/result modules exist, but current workbench record is a generic saved readout (`src/vla_lens/workbench/schema.py:452`). | Require at least original, no-op, and one intervention trial for a useful first slice. |
| Should negative results be highlighted or merely retained? | Highlighting negative results needs status/filter language and claim limitations. | Artifacts can store metrics/display, but no `Observation`/`Claim` model exists. | Retain negative/failed results first; add claim synthesis later. |
