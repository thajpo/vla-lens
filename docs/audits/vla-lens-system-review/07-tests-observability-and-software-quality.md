# 07 - Tests, Observability, and Software Quality

Inspected commit: `882eeb83be8c8a69a80fc8f6ec8829c311ca4630`

Creation status: the worktree was clean before the audit output directory was generated. The audit was later preserved on the dedicated `codex/system-review-audit` branch.

Inspection mode: static inspection only. No captures, simulators, model downloads, servers, broad tests, or destructive commands were run.

Commands used: `git rev-parse HEAD`; `git status --short`; `rg -n "def test_|policy_call|SelectionState|LensArtifact|build_dataset_index|TraceDataset|capture|runtime|workbench" tests`; targeted `nl -ba ... | sed -n ...` reads of `tests`, `src/vla_lens`, `frontend/src`, `docs`, `scripts`, and `pyproject.toml`.

## Executive Finding

The repo has unusually strong normal-lane coverage for a research prototype. It tests LeRobot/overlay storage, dashboard indexes, workbench contracts, probe artifacts, evidence pins, intervention specs/results, server APIs, frontend model functions, and the PI0.5 import boundary. The important gap is not raw quantity of tests. The gap is that the central scientific invariants are still protected indirectly by probe and UI tests rather than by explicit contracts.

The first tests to strengthen should make these invariants first-class:

- `(trace_id, policy_call_index)` is the stable model-decision identity.
- feature rows and prediction rows resolve back to exact policy calls/model sites.
- experiment populations are saved as reusable manifests, not only fingerprints.
- selections round-trip across backend, frontend route state, and evidence pins.
- artifacts remain interpretable after dashboard indexes are deleted/rebuilt.
- normal imports and inspection never require PI0.5/Torch/LeRobot/LIBERO.

## Test Map

| Category | Existing coverage | Representative files | What is not tested enough |
|---|---|---|---|
| Schema/unit | Capture profile resolution, token metadata, intervention specs, probe evidence contracts | `tests/pi05_capture_success_test.py`, `tests/pi05_token_metadata_test.py`, `tests/test_intervention_specs.py`, `tests/probe_evidence_contract_test.py` | A shared experiment recipe schema and generic example manifest do not exist yet, so they cannot be tested. |
| Data contract | LeRobot v3 writer/reader, overlay boundary, validator, dataset trust | `tests/lerobot_dataset_storage_test.py`, `tests/lerobot_v3_contract_test.py`, `tests/table_io_test.py` | Cross-dataset policy-call relation and derived-label versioning are not explicit contracts. |
| Indexing | Deterministic rebuild, stale schema/fingerprint failure, artifact-index behavior, probe episode representative row | `tests/dataset_index_test.py`, `tests/dashboard_probe_index_test.py` | No dataset-level policy-call index; no query test for policy-call + label + model-site availability. |
| Storage | Artifact paths under overlay, root-relative outputs, feature cache invalidation, nested LeRobot roots | `tests/lerobot_dataset_storage_test.py`, `tests/probe_score_cache_test.py` | Artifact interpretation after deleting/rebuilding dashboard indexes is only partially covered. |
| Feature extraction | Activation selector materializes/caches, generation-step slicing, LeRobot overlay model-site materialization | `tests/vla_lens_trace_workbench_test.py:196`, `tests/vla_lens_trace_workbench_test.py:216`, `tests/lerobot_dataset_storage_test.py:80` | Feature row identity is not tested as a stable reusable example manifest. |
| Analysis/probes | Probe preflight, YAML spec training, research framing, split sidecar, score cache refresh, diagnostics | `tests/probe_preflight_test.py`, `tests/vla_lens_trace_workbench_test.py:298`, `tests/probe_score_cache_test.py`, `tests/pi05_probe_diagnostics_test.py` | Probe artifacts retain row fingerprints but not full selected-row manifests; tests therefore cannot prove another method can reuse the same population. |
| Backend | FastAPI server, dataset API, frame path, workbench payloads, evidence pins | `tests/fastapi_server_test.py`, `tests/server_api_test.py`, `tests/server_frame_path_test.py`, `tests/evidence_pins_test.py` | API schemas are not generated into frontend types; route-level selection state has limited round-trip tests. |
| Frontend | Model-level JS tests for dataset browser, episode route, evidence pins, intervention lab, capability gating | `frontend/src/pages/workbench/datasetBrowserModel.test.mjs`, `frontend/src/pages/workbench/episodeRouteModel.test.mjs`, `frontend/src/components/interventions/interventionLabModel.test.mjs` | No browser/screenshot verification for dense workbench state; no test that clicking aggregate plots opens exact source examples. |
| End-to-end normal lane | Synthetic/demo paths and server/app checks exist | `tests/serve_vla_lens_app_test.py`, `tests/fastapi_server_test.py`, `scripts/run_vla_lens_demo.sh` | A single fixture should exercise capture-free storage -> index -> probe -> evidence -> selection -> API resolution. |
| Runtime/model | Static PI0.5 profile/site/capture writer semantics, wrappers, success logic | `tests/pi05_capture_success_test.py`, `tests/pi05_batch_capture_test.py`, `tests/docker_pi05_wrapper_test.py` | Real capture smokes are intentionally outside normal tests; runtime version/fingerprint capture should be asserted without loading the runtime. |
| LeRobot integration | LeRobot root writing/opening/validation and overlay reference handling | `tests/lerobot_dataset_storage_test.py`, `tests/lerobot_v3_contract_test.py` | Arbitrary external LeRobot shard layouts are not covered; current tests mostly validate the project writer contract. |
| Migrations/compatibility | Legacy output paths, alias normalization, old artifact index paths | `tests/lerobot_dataset_storage_test.py:329`, `tests/probe_score_cache_test.py:27`, `tests/vla_lens_trace_workbench_test.py:24` | No formal migration matrix for schema versions across overlay, index, artifact, and workbench state. |
| Performance | Source file size checks and bounded lens-array slice payloads | `tests/source_file_size_check_test.py`, `tests/vla_lens_trace_workbench_test.py:5` | No performance/I/O tests for feature extraction over hundreds or thousands of episodes. |

## Scientific Invariants

| Invariant | Current protection | Gap |
|---|---|---|
| One stable policy-call identity | `PolicyCallRecord.to_row` writes `policy_call_index`; capture tests assert policy-call axes in arrays (`src/vla_lens/capture/records.py:27`, `tests/pi05_capture_success_test.py:267`) | No dataset-level `call_id` or uniqueness test across `(trace_id, policy_call_index)`. |
| Frame/policy-call/action alignment | Timesteps table maps each timestep to policy call and horizon (`src/vla_lens/pi05/capture_tables.py:37`); selectors map policy calls to timesteps (`src/vla_lens/selectors.py:335`) | No explicit invariant test for generated action chunk -> executed timesteps -> UI active call across all payloads. |
| Generated vs executed action distinction | Capture stores `action_chunks`, `generation_actions`, and `executed_actions`; writer rewrites normalization refs to canonical `action` (`src/vla_lens/pi05/capture_writer.py:78`, `src/vla_lens/dataset/overlay.py:164`) | Need a test that UI/API never labels generated action arrays as executed actions. |
| Feature rows resolve to source calls | Feature rows include `trace_id`, `timestep`, `policy_call_index`, `model_site_id`, and `generation_step` (`src/vla_lens/selectors.py:163`) | No saved manifest lets tests reopen an artifact and resolve every source row without recomputing features. |
| Experiment manifests reproducible | Probe artifacts store selector, input fingerprints, target, split, and row fingerprint (`src/vla_lens/probes/workflow_artifacts.py:82`) | Fingerprints are not the same as an exact selected-population manifest. |
| Grouped splits preserved | Probe split metadata records group key and trace lists (`src/vla_lens/probes/workflow_artifacts.py:185`) | Need generic split contract independent of probes. |
| Preprocessing/transforms recorded | Probe input records pooling and feature transform; capture stores preprocessing/action normalization tables (`src/vla_lens/probes/workflow_artifacts.py:82`, `src/vla_lens/pi05/capture_tables.py:305`) | Transform contracts are spread across capture/probe tables rather than one experiment recipe. |
| Artifact provenance resolves after reindexing | Dataset index rebuild tests exist (`tests/dataset_index_test.py:22`); artifact paths root-relative tests exist (`tests/lerobot_dataset_storage_test.py:291`) | Need direct test: delete dashboard indexes, rebuild, load artifact, resolve examples and arrays. |
| UI example links open exact source moment | Episode route parser/builder tested; evidence pins preserve selection (`frontend/src/pages/workbench/episodeRouteModel.test.mjs`, `tests/evidence_pins_test.py:9`) | Need aggregate plot/cell -> exact episode/call/site route tests. |

## Boundary Tests

| Boundary | Current evidence | Gap / recommendation |
|---|---|---|
| Normal imports avoid PI0.5/Torch/LIBERO | Runtime imports are lazy in `load_pi05_capture_runtime` (`src/vla_lens/pi05/capture_runner.py:175`); `tests/research_ui_import_boundary_test.py` exists | Add a test that imports every normal server/workbench/probe module under an import hook that fails on `torch`, `lerobot`, `libero`, and `robosuite`. |
| Runtime outputs inspectable without runtime | `TraceDataset.open` and dashboard APIs read existing roots without capture runtime (`src/vla_lens/traces/dataset.py:40`) | Add fixture with PI0.5-looking model sites/capture report and assert server/workbench payloads load with no runtime deps. |
| Compatibility paths explicit | Legacy probe paths and aliases are tested (`tests/lerobot_dataset_storage_test.py:329`, `tests/probe_score_cache_test.py:27`) | Create a compatibility test list tied to schema versions rather than scattered regression tests. |
| Indexes rebuildable | `test_dataset_index_overwrite_rebuild_is_deterministic` exists (`tests/dataset_index_test.py:22`) | Extend to policy-call index and artifact example manifest once added. |
| Failed/partial records inspectable | Intervention statuses and causal evidence checks exist (`src/vla_lens/workbench/api.py:64`) | Probe failures mostly raise before a durable run record exists; long analyses should save failed run shells. |
| Frontend/backend types match | Hand-written TS and Python types exist (`frontend/src/types/workbench.ts`, `src/vla_lens/workbench/schema.py`) | No generated schema or cross-language snapshot validates drift. |
| Unsupported capabilities fail explicitly | Capability gating tests exist in frontend; intervention preflight exists (`frontend/src/pages/capabilityGating.test.mjs`, `src/vla_lens/interventions/preflight.py`) | Add backend tests that unsupported model/site/intervention capabilities return typed `unsupported` payloads, not empty success. |

## Observability

Current observability is practical but uneven:

- Capture commands print the resolved capture plan and per-episode success summaries (`src/vla_lens/pi05/capture_runner.py:169`, `src/vla_lens/pi05/capture_runner.py:308`).
- Capture reports record declared/captured/missing model sites and context availability (`src/vla_lens/pi05/capture_writer.py:211`).
- Dataset index validation gives explicit rebuild errors for schema/fingerprint/count problems (`src/vla_lens/dataset/index.py:249`).
- Probe artifacts persist metrics, predictions, per-split/group/null metrics, weights, feature fingerprints, and split summaries (`src/vla_lens/probes/workflow_training.py:153`, `src/vla_lens/probes/workflow_training.py:252`).
- Workbench validation and diagnostics endpoints exist (`src/vla_lens/server/fastapi_app.py:122`, `src/vla_lens/server/fastapi_app.py:206`).

Weak spots:

- Feature extraction has little structured progress or I/O timing; a large selector can appear idle while reading zarr arrays (`src/vla_lens/selectors.py:112`).
- Empty feature selections raise a useful exception in training, but UI/API paths can still present empty panels without an actionable explanation (`src/vla_lens/probes/workflow_training.py:85`).
- Workbench selection resolver returns bounded arrays/sites/examples but does not include warnings for filtered-away axes or ambiguous axes (`src/vla_lens/workbench/selection.py:82`).
- Probe-study representative rows collapse many policy-call rows into one trace-level row, which can make aggregate UI look more exact than it is (`src/vla_lens/server/probe_studies.py:197`).
- Intervention save path can record `inspected_only` readouts, but the UI should make the difference between preflight/readout/live causal result impossible to miss (`frontend/src/components/interventions/interventionLabModel.ts:70`).

## Highest-Value Tests

| Test name | Specific invariant | Target files/symbols | Fixture needed | Failure it would catch |
|---|---|---|---|---|
| `test_policy_call_index_has_one_global_row_per_call` | `(trace_id, policy_call_index)` is unique and materialized as `call_id` | future `build_policy_call_index`; `src/vla_lens/dataset/index.py:173`; `TraceBundle.POLICY_CALLS` | Synthetic LeRobot root with two episodes and multi-call overlays | Duplicate/missing policy calls and UI queries relying on per-bundle scans. |
| `test_policy_call_index_joins_labels_without_owning_arrays` | Index joins scalar labels/site availability but does not copy tensors | future policy-call index builder; `src/vla_lens/pi05/policy_call_labels.py`; `model_site_index` | Fixture with policy-call label artifact and model arrays | Wide table/tensor bloat or missing semantic labels. |
| `test_probe_saves_example_manifest_with_exact_rows` | A probe run saves all selected examples after expansion/filter/split | `src/vla_lens/probes/workflow_training.py:63`; `_probe_examples` | Existing synthetic probe fixture | Future methods cannot reuse the population; plot points cannot resolve to rows. |
| `test_feature_manifest_rows_resolve_after_reindex` | Artifact examples resolve after deleting dashboard indexes and rebuilding | `TraceDataset.load_artifact`; `build_dataset_index`; feature manifest | Synthetic probe artifact with predictions and manifest | Artifact interpretation depending on stale/local dashboard index. |
| `test_selection_round_trip_backend_frontend_route_axes` | A scientific selection preserves episode, call, timestep, site, unit, run, split through backend dict and route model | `SelectionState`; `episodeRouteModel.ts`; `/api/selections/resolve` | JS route test plus Python resolver fixture | Lossy deep links and axis-name drift. |
| `test_confusion_cell_links_to_exact_prediction_examples` | Aggregate class/confusion cells expose source example IDs | `src/vla_lens/server/probe_studies.py`; `DatasetBrowser` model | Probe diagnostics fixture with confusion/error rows | UI cannot drill from a result to source moment. |
| `test_normal_lane_imports_block_capture_deps` | Normal package/server/workbench imports do not import `torch`, `lerobot`, `libero`, `robosuite` | `src/vla_lens/server/*`; `src/vla_lens/workbench/*`; `src/vla_lens/probes/*` | Import hook/monkeypatch that raises on blocked modules | Accidental runtime dependency crossing into `.venv`. |
| `test_runtime_capture_report_exposes_missing_sites_as_warnings` | Incomplete capture remains inspectable and clearly incomplete | `_capture_report`; server dataset diagnostics | Trace bundle with missing_model_sites | Silent empty activation panels from partial capture. |
| `test_experiment_recipe_reconstructs_feature_target_split` | A recipe can rebuild the same selector/target/split metadata without retraining | future `ExperimentRecipe`; existing probe spec wrapper | Probe YAML fixture and synthetic dataset | Recipe drift, hidden defaults, unreproducible runs. |

## Proposed Normal-Lane End-to-End Fixture

Create `tests/fixtures/minimal_policy_call_workbench.py` or extend `tests/_support/vla_lens_trace_mvp.py` with:

1. Two synthetic LeRobot episodes, each with two policy calls.
2. `policy_calls.parquet`, `timesteps.parquet`, `generation_steps.parquet`, token metadata, and two model sites with `policy_call` axes.
3. One derived policy-call label table/artifact with a pre-contact/contact label.
4. One probe recipe over expert hidden states and a scalar target.
5. Saved probe artifact with predictions, manifest, and weights.
6. Rebuilt dashboard index.
7. Workbench selection resolving a failed heldout prediction to exact episode/call/site.

This fixture should use only normal dependencies from `pyproject.toml` (`numpy`, `pandas`, `zarr`, `pyarrow`, `fastapi`, `sklearn`) and no Torch/LeRobot/LIBERO/GPU. It would exercise the core research loop without violating the environment split.
