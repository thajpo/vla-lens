# 02 - Data Storage and Indexing Audit

Worker: 02

Inspected commit: `882eeb83be8c8a69a80fc8f6ec8829c311ca4630`

Git status before writing this audit: clean. `git status --short` returned no rows during the inspection snapshot.

Scope: static inspection only. I did not run captures, simulators, model downloads, servers, tests, or destructive commands.

## Executive Answer

VLA Lens has a clear two-layer on-disk contract: LeRobot v3 owns the robot episode data and media, while `vla_lens/` owns interpretability overlays, tensors, artifacts, indexes, caches, and workbench state. The boundary is mostly enforced in code: the writer puts robot fields in LeRobot parquet/video files and excludes canonical robot arrays from overlay storage, while readers merge robot rows with overlay metadata when an overlay is present.

There is per-episode policy-call storage. Each overlay bundle has `tables/policy_calls.parquet`, PI0.5 capture writes one row per `CaptureCall`, and a derived PI0.5 label artifact can produce one label row per policy call. However, there is not a first-class dataset-level `PolicyCallIndex` materialized by the dashboard index builder. The dashboard index stores episode rows, model-site rows, artifact rows, probe prediction rows, and probe episode summaries, but not one indexed row per policy call.

There is also not a generic `ExampleManifest`. Probe artifacts contain a probe-specific equivalent in their `method.input`, `method.target`, `method.examples`, `method.split`, predictions, and scored predictions. That is useful provenance, but it is not a reusable manifest for arbitrary policy-call examples across probes, interventions, and workbench cohorts.

Smallest compatible addition: add a rebuildable `vla_lens/tables/policy_call_index.parquet` materialized view with one row per `(trace_id, policy_call_index)`, built from bundle `policy_calls`, episode metadata, model-site availability, and optional latest policy-call-label artifact columns. Do not copy high-dimensional tensors into it. Register it in the workbench/server table catalogs. This would remove most manual glue for queries like "all pre-contact policy calls from successful episodes with model site X captured."

## On-Disk Contract

The public dataset shape is documented as a LeRobot v3 root plus a VLA Lens overlay: `meta/`, `data/`, `videos/`, and `vla_lens/` with `overlay.json`, `tables`, `arrays`, and `artifacts` (`docs/dataset-format.md:7`, `docs/dataset-format.md:80`). `TraceBundle` also describes the local overlay-bundle layout: manifests, parquet tables, zarr arrays, JPEG frame media, artifact indexes, and capture JSON (`src/vla_lens/traces/bundle.py:41`, `src/vla_lens/traces/bundle.py:77`).

| Storage class | Path and format | Source of truth | Rebuild/delete behavior |
| --- | --- | --- | --- |
| LeRobot metadata | `meta/info.json`, `meta/tasks.*`, `meta/episodes/**/*.parquet`, `meta/stats.json` | Source of truth for robot dataset metadata | Not a derived cache. Do not delete unless regenerating the dataset. |
| LeRobot robot rows | `data/chunk-*/episode_*.parquet` | Source of truth for `index`, `episode_index`, `frame_index`, `timestamp`, `task_index`, `action`, reward/done flags, and `observation.state` (`src/vla_lens/dataset/writer.py:390`) | Not a derived cache. Writer assumes one episode-oriented file layout via path helpers (`src/vla_lens/dataset/common.py:37`). |
| LeRobot videos | `videos/chunk-*/<camera>/episode_*.mp4` | Source of truth for canonical image observations (`src/vla_lens/dataset/writer.py:409`) | Not a derived cache. Reader can load single frames or full videos (`src/vla_lens/dataset/bundle.py:332`). |
| Overlay root | `vla_lens/overlay.json`, `vla_lens/tables/episode_refs.parquet` | Source of truth for episode-to-overlay mapping (`src/vla_lens/dataset/overlay.py:114`) | Required for canonical overlay discovery. Fallback canonical paths exist, but the format document says standalone bundles are not the public compatibility layer (`docs/dataset-format.md:62`). |
| Overlay episode bundle | `vla_lens/episodes/episode_XXXXXX/manifest.json`, `tables/*.parquet`, `arrays/*`, `artifacts/*` | Source of truth for model internals, policy-call alignment, token metadata, model-site tensors, artifacts, and fingerprints (`docs/dataset-format.md:30`, `docs/dataset-format.md:58`) | Durable data. Do not delete unless intentionally removing the overlay. |
| Dataset artifacts | `vla_lens/artifacts/<artifact_id>/artifact.json` plus outputs, indexed in `vla_lens/tables/artifact_index.parquet` | Durable derived research outputs. `LensArtifact` stores provenance plus references rather than owning every payload (`src/vla_lens/artifacts.py:33`, `src/vla_lens/traces/dataset.py:133`) | Not merely cache. Rebuild only by rerunning the artifact workflow. |
| Dashboard indexes | `vla_lens/tables/index_manifest.json`, `episode_index.parquet`, `model_site_index.parquet`, `dashboard_artifact_index.parquet`, `probe_predictions.parquet`, `probe_episode_index.parquet` (`src/vla_lens/dataset/index.py:24`) | Rebuildable materialized views. Code states raw traces remain source of truth (`src/vla_lens/dataset/index.py:1`) | Safe to delete and rebuild, but append mode can preserve old rows unless `overwrite=True` (`src/vla_lens/dataset/index.py:181`). |
| Feature cache | `.vla_cache/features/<cache_key>/X.zarr`, `rows.parquet` | Ephemeral acceleration for selectors (`src/vla_lens/selectors.py:71`, `src/vla_lens/selectors.py:109`) | Safe to delete. Cache key does not include full array content fingerprints, only selector, episode rows, and model-site records (`src/vla_lens/selectors.py:99`). |
| Workbench state | `vla_lens/workbench/cohorts`, `analysis_runs`, `intervention_runs`, `workspaces` JSON | Mutable user/research state (`src/vla_lens/workbench/utils.py:164`, `src/vla_lens/workbench/api.py:99`) | Not a rebuildable index. It records user choices and run provenance. |
| Intervention artifacts | Dataset artifacts of type `intervention_run` or `intervention_sweep` (`src/vla_lens/interventions/artifacts.py:13`) | Durable evidence summaries derived from typed intervention runs | Shell artifacts reference output refs and provenance; run JSON can also be saved in workbench state (`src/vla_lens/workbench/api.py:136`). |

Versioning is split. The overlay layout has `SCHEMA_VERSION = "0.3.0"` and zarr/zstd/JPEG storage constants (`src/vla_lens/traces/layout.py:3`). Dashboard indexes have `INDEX_SCHEMA_VERSION = "0.2.0"` (`src/vla_lens/dataset/index.py:24`). PI0.5 policy-call labels have their own artifact schema version (`src/vla_lens/pi05/policy_call_labels.py:20`). This is reasonable, but the version boundaries should be documented as independent contracts.

Path portability is mixed. Array and artifact records generally store relative paths, which is good for external drives and moved datasets (`src/vla_lens/traces/io.py:20`, `src/vla_lens/artifacts.py:89`). Dashboard episode rows store `str(bundle.path)` and the index fingerprint includes `str(root)`, so the dashboard index is intentionally local and non-portable across mount points (`src/vla_lens/dataset/index.py:312`, `src/vla_lens/dataset/index.py:580`).

## LeRobot and Overlay Boundary

The intended boundary is explicit: LeRobot owns observations, actions, episode/frame/timestamp/task fields, and camera media; VLA Lens owns model internals, policy-call alignment, token metadata, model-site metadata, artifacts, fingerprints, and dashboard/research state (`docs/dataset-format.md:30`). Joins are through `episode_index`, `frame_index`, `timestamp`, and `task_index`.

The writer mostly enforces this. `write_lerobot_trace_record` writes robot data and metadata, then writes the overlay bundle and overlay root (`src/vla_lens/dataset/writer.py:63`). `_write_robot_data` writes canonical LeRobot columns including action/state/rewards/done (`src/vla_lens/dataset/writer.py:390`). `_overlay_episode_arrays` excludes `executed_actions`, `frames.*`, `action`, `observation.state`, and `observation.images.*`, preventing robot fields from being duplicated as overlay arrays (`src/vla_lens/dataset/overlay.py:149`). Action normalization is rewritten to reference canonical LeRobot `action` (`src/vla_lens/dataset/overlay.py:164`).

The reader merges the two layers. `open_lerobot_dataset` validates the LeRobot root and attaches an optional overlay bundle per episode (`src/vla_lens/dataset/reader.py:31`). `LeRobotEpisodeBundle` presents a `TraceBundle`-like interface, pulling robot arrays and video frames from LeRobot and model/tokens/artifacts from overlay (`src/vla_lens/dataset/bundle.py:44`, `src/vla_lens/dataset/bundle.py:233`). `TraceDataset.open` accepts a LeRobot root or a directory of nested LeRobot roots, not arbitrary legacy bundle layouts (`src/vla_lens/traces/dataset.py:32`, `src/vla_lens/traces/io.py:78`).

Compatibility risk: the writer assumes VLA Lens's LeRobot path scheme, not arbitrary LeRobot sharding. The validator checks paths and overlay references (`src/vla_lens/capture/lerobot_v3.py:151`, `src/vla_lens/capture/lerobot_v3.py:277`), but the writer path helpers use one episode per chunk/file convention (`src/vla_lens/dataset/common.py:37`). This is fine as the project-owned writer contract, but should not be confused with full LeRobot ecosystem compatibility.

## Index Contents and Build Behavior

The current dashboard index contains:

- `episode_index.parquet`: one row per trace/episode, with task/prompt/outcome/length, dataset metadata, camera names, model-site count, artifact count, array names, `policy_call_count`, and token-space count (`src/vla_lens/dataset/index.py:33`, `src/vla_lens/dataset/index.py:286`).
- `model_site_index.parquet`: one row per trace/model-site with site id, name, module, layer, tensor type, token kind, axes, shape, dtype, relative path, family/role/segment (`src/vla_lens/dataset/index.py:62`, `src/vla_lens/dataset/index.py:327`).
- `dashboard_artifact_index.parquet`: artifact-browser rows derived from dataset and bundle artifacts (`src/vla_lens/dataset/index.py:79`, `src/vla_lens/dataset/index.py:216`).
- `probe_predictions.parquet`: prediction-level rows, including `policy_call_index`, timestep, generation step, model-site id, token-space id, split, confidence, and correctness (`src/vla_lens/dataset/index.py:98`).
- `probe_episode_index.parquet`: a representative aggregate per probe/trace, not all policy calls (`src/vla_lens/dataset/index.py:127`, `src/vla_lens/dataset/index.py:446`).

`build_dataset_index` opens the dataset, optionally appends only new trace IDs, writes the five parquet tables, and writes `index_manifest.json` (`src/vla_lens/dataset/index.py:173`). It does not build a policy-call index. `validate_dataset_index` checks schema version, duplicate trace IDs, a source fingerprint, and declared episode count (`src/vla_lens/dataset/index.py:249`).

The index is rebuildable, but not fully content-addressed. `_dataset_fingerprint` hashes the absolute root path, episode IDs/lengths, and source file size/mtime signatures for metadata and overlay manifests (`src/vla_lens/dataset/index.py:580`). That is enough to detect many local stale-index cases. It is not a portable or semantic dataset hash.

Workbench table querying can expose bundle `policy_calls` by unioning each bundle's `tables/policy_calls.parquet` and adding `trace_id`, `episode_id`, and `bundle_path` (`src/vla_lens/workbench/tables.py:59`). The schema even aliases `"policy_call_index"` to the `policy_calls` table (`src/vla_lens/workbench/schema.py:221`). That is a query convenience, not the same thing as a persisted dataset-level `PolicyCallIndex`.

Current answer to the seven indexing questions:

1. **What is indexed?** Episodes, model sites, artifacts, probe predictions, and probe episode summaries.
2. **What is not indexed?** Policy calls as a dataset-level relation, timestep labels as first-class index rows, token rows, arbitrary context rows, array chunks, and raw high-dimensional tensors.
3. **How is it built?** By static scan through `TraceDataset.open` and bundle tables in `build_dataset_index`; probe predictions are read from probe-suite artifacts.
4. **Can it be deleted and rebuilt deterministically?** Yes as a local dashboard view, modulo timestamps, absolute root path, append mode, and non-index workflows that must be rerun to recreate artifacts.
5. **Does it contain research labels?** Only indirectly through probe predictions and artifact metadata. PI0.5 policy-call labels remain artifact outputs unless manually joined or consumed by probe prep.
6. **Does it support cross-table research queries directly?** Not generally. `query_table` targets one table; server episode queries can join probe episode summaries, not arbitrary policy-call labels and model-site availability (`src/vla_lens/server/indexed.py:247`).
7. **Is there a stable global policy-call key?** The practical key is `(trace_id, policy_call_index)`. The PI0.5 label artifact also emits `policy_call_id`, but it is just the local integer, not globally unique (`src/vla_lens/pi05/policy_call_labels.py:164`).

## High-Dimensional Array Storage and Reading

High-dimensional arrays are stored as zarr v2 arrays with Blosc zstd compression and default chunking. `_write_zarr_array` uses `zarr.open_array(..., mode="w")`, compressor `Blosc(cname="zstd", clevel=3, shuffle=BITSHUFFLE)`, and `_default_chunks` that keeps axis 0 at up to 16, spatial axes at up to 64, and later axes at up to 128 (`src/vla_lens/traces/io.py:106`, `src/vla_lens/traces/io.py:168`). Frames are a special case: frame arrays are written as JPEG sequences in overlay bundles (`src/vla_lens/traces/bundle.py:147`, `src/vla_lens/traces/io.py:130`). Canonical LeRobot images are mp4 videos, not overlay zarr.

Model arrays are stored under `arrays/model/<slug>.zarr`, and `model_sites.parquet` records site metadata, axes, shape, dtype, storage path, module/layer/tensor type, token kind, materialization, exactness, and token references (`src/vla_lens/traces/bundle.py:175`). Episode/action/context arrays use `arrays/episode`, `arrays/action`, and `arrays/context` based on name prefixes (`src/vla_lens/traces/io.py:157`).

PI0.5 array axes are research-friendly. Capture arrays include `action_chunks` with `[policy_call, horizon, action_dim]`, generation actions/velocities with `[policy_call, generation_step, horizon, action_dim]`, image-prefix/VLM/expert hidden states, KV caches, attention, action-head inputs, and action-head outputs with axes that preserve policy-call and generation-step structure (`src/vla_lens/pi05/capture_arrays.py:35`, `src/vla_lens/pi05/capture_arrays.py:69`). The capture plan documents `axis_strategy="policy_call"` and storage dtype choices such as `float16`/`float32` (`src/vla_lens/pi05/capture_schema.py:61`, `src/vla_lens/pi05/capture_schema.py:93`).

Read behavior is partly lazy. `_read_zarr_array` returns a zarr array in read mode (`src/vla_lens/traces/io.py:124`), and server endpoints slice selected policy-call, generation-step, head, query, or channel axes before converting to numpy (`src/vla_lens/server/common.py:242`, `src/vla_lens/server/activation.py:273`, `src/vla_lens/server/attention_maps.py:86`). `slice_lens_array` also slices before returning small previews (`src/vla_lens/workbench/api.py:294`).

There are still copy-heavy paths:

- Cached feature matrices are loaded with `np.asarray(zarr.open_array(...))`, ignoring the `mmap` flag and copying the full cached `X.zarr` into memory on cache hits (`src/vla_lens/selectors.py:71`).
- Unit-example ranking can load a whole selected array with `np.asarray(_load_lens_array(...), dtype=float32)` before selecting examples (`src/vla_lens/workbench/selection.py:546`).
- Trace fingerprinting hashes full zarr array contents via `array[:]`, which is deterministic but expensive for large tensors (`src/vla_lens/traces/fingerprints.py:206`).

The feature cache is useful but not a source-of-truth index. `FeatureView.cache_key` includes selector, episode IDs/lengths, and model-site metadata, but not raw array content fingerprints (`src/vla_lens/selectors.py:99`). If tensor contents change without model-site metadata changing, the cache may stay valid-looking.

## Policy-Call Research Relation

There is a row-grain policy-call table inside each overlay bundle. PI0.5 capture creates one `PolicyCallRecord` per `buffer.calls` item, with `call_index`, observation timestep, end timestep, prompt/model ids, action horizon, action dimension, and call metadata (`src/vla_lens/pi05/capture_writer.py:176`). The generic `PolicyCallRecord.to_row` emits `policy_call_index`, `observation_timestep`, and `env_timestep_start`; `_policy_call_frame` drops duplicate `policy_call_index` values and sorts them (`src/vla_lens/capture/records.py:19`, `src/vla_lens/capture/records.py:295`).

Timesteps are aligned back to calls. `_timesteps_table` fills each timestep with the active `policy_call_index` and `horizon_index` until the next policy call starts (`src/vla_lens/pi05/capture_tables.py:37`). Generation steps are also keyed by `policy_call_index` (`src/vla_lens/pi05/capture_tables.py:64`).

PI0.5 has a derived one-row-per-policy-call label artifact. `save_pi05_policy_call_labels_artifact` reads object-flow outputs, builds labels, saves `policy_call_labels.parquet`, updates the dataset artifact index, saves a workbench analysis run, and optionally rebuilds the dashboard index (`src/vla_lens/pi05/policy_call_labels.py:30`). `build_policy_call_labels` explicitly returns one object-flow label row per policy call and records task, prompt, observation timestep, start/end timestep, task phase, next/active/current object fields, onset times, pre-contact/pre-motion/pre-lift booleans, and visible/candidate objects (`src/vla_lens/pi05/policy_call_labels.py:125`).

Probe training consumes this relation opportunistically. `_merge_policy_call_labels` loads the latest `pi05_policy_call_labels` artifact and joins it into selected feature rows on `trace_id` and `policy_call_index` (`src/vla_lens/probes/workflow_prepare.py:410`). This is a research workflow join, not a globally indexed relation.

Feature/probe rows are not the same as policy-call rows. `FeatureView` emits one row per selected trace/model-site/sample after selector reduction, carrying `policy_call_index`, model site, layer, tensor type, generation step, token kind, and feature dimension (`src/vla_lens/selectors.py:112`). If multiple model sites are selected, there are multiple feature rows for the same policy call. Probe artifact provenance records this as "one row per selected trace/model_site/sample after selector reduction" and defines example IDs with trace, policy call, generation step, token-space id, token index, model-site id, and target (`src/vla_lens/probes/workflow_artifacts.py:154`).

For the target query, "all pre-contact policy calls from successful episodes in tasks containing a movable object, where model site X was captured," today's user/research code needs to manually combine:

1. `episode_index.parquet` or `dataset.episode_index` for successful episodes and task metadata.
2. Latest `pi05_policy_call_labels` artifact output for `is_pre_contact`, object fields, and `policy_call_index`.
3. `model_site_index.parquet` for traces where model site X exists.
4. Optionally object-flow role artifacts if "movable object" means role or movement evidence rather than label presence.
5. Per-bundle arrays/model-site zarr references for actual activation reads.

This is workable, but high-friction. The code has all pieces, but the relational spine is not first-class.

## PolicyCallIndex and ExampleManifest Equivalents

`PolicyCallIndex` equivalent: **partial, not canonical**.

- Present: per-bundle `tables/policy_calls.parquet`; workbench can union this as the `policy_calls` table; PI0.5 label artifacts can emit one row per call.
- Missing: a persisted dataset-level policy-call table in `build_dataset_index`; a globally stable `policy_call_id`; direct server/indexed support for filtering calls by episode labels, policy-call labels, and model-site coverage.

`ExampleManifest` equivalent: **probe-specific, not generic**.

- Present: probe artifacts store `method.source`, `method.input`, `method.target`, `method.examples`, `method.split`, output paths, predictions, scored predictions, and fingerprints (`src/vla_lens/probes/workflow_training.py:174`, `src/vla_lens/probes/workflow_artifacts.py:82`).
- Missing: a reusable manifest object for arbitrary example sets that can be shared across probes, interventions, cohorts, and dashboard panels without retracing selector logic.

## Concrete Improvements

### P0 - Add `policy_call_index.parquet`

Add a rebuildable materialized view:

`vla_lens/tables/policy_call_index.parquet`

Minimum columns:

- Identity: `call_id` as a stable string such as `<trace_id>:<policy_call_index>`, `trace_id`, `episode_id`, `episode_index`, `policy_call_index`.
- Time: `observation_timestep`, `env_timestep_start`, `env_timestep_end`, optional `horizon_length`.
- Episode metadata: `task_id`, `prompt`, `outcome`, `dataset_id`, `benchmark`, `profile`, `seed`, `split`.
- Capture metadata: `model_id`, `model_family`, `model_call_kind`, `action_generator_kind`, `action_horizon`, `action_dim`.
- Labels when available: `task_phase`, `next_manipulated_object`, active/current object fields, `is_pre_contact`, `is_pre_motion`, `is_pre_lift`, visible/candidate object fields, and `label_artifact_id`.
- Coverage summary: `model_site_count`, `model_site_ids` or a compact JSON list, and optionally `token_space_count`.
- References only: no tensor payloads; array/model-site paths remain in `array_index` and `model_site_index`.

Implementation sketch:

1. Add `POLICY_CALL_INDEX` and `POLICY_CALL_COLUMNS` in `src/vla_lens/dataset/index.py`.
2. Add `_policy_call_rows(dataset, bundle)` that reads `bundle.policy_calls`, joins scalar episode metadata from `_episode_index_row`, and adds coverage summaries from `bundle.model_sites`.
3. Optionally join latest `pi05_policy_call_labels` artifact if present, with label columns namespaced or collision-safe.
4. Write the table in `build_dataset_index`; validate it in the manifest like the other tables.
5. Register it in `src/vla_lens/workbench/schema.py`, `src/vla_lens/workbench/tables.py`, and server indexed payloads.
6. Add tests that build two traces with multiple calls and assert one row per `(trace_id, policy_call_index)`, stable `call_id`, label join behavior, and no tensor reads.

Why this is compatible: it treats raw traces and artifacts as source truth, matches the existing rebuildable-index design, avoids changing capture output, and does not duplicate high-dimensional arrays.

### P1 - Add a Generic Example Manifest

Introduce a dataset artifact type such as `example_manifest` whose primary output is a parquet table of selected examples. It should snapshot selected rows after a selector/query, with stable IDs, policy-call references, model-site references, target columns, split columns, cohort IDs, and source artifact IDs. Probe artifacts can then point at an example manifest instead of embedding only probe-local example semantics.

This should come after `policy_call_index.parquet`; otherwise each manifest still has to rebuild the same manual policy-call joins.

### P1 - Make Cache Validity More Content-Aware

Feature cache keys should include trace/model-site fingerprints or array content fingerprints already computed in capture reports where available. At minimum, include `manifest.metadata.fingerprints` or model-site array file signatures. This reduces stale feature matrices when array contents change without index metadata changing.

### P1 - Avoid Full-Matrix Loads Where the API Claims Lazy Access

Honor the `mmap` argument or return zarr-backed access for cached feature matrices instead of always `np.asarray(...)` on cache hit. Review unit-example selection and fingerprint paths for bounded slicing. These are not correctness blockers, but they will matter for large PI0.5 activation captures.

### P2 - Clarify Local vs Portable Index Semantics

Document that dashboard indexes are local rebuildable views because their fingerprints include absolute root paths and mtimes. If portable index artifacts are desired later, add a separate content-derived dataset fingerprint that excludes the mount path and uses semantic source hashes.

## Commands Used

All commands were static/read-only inspection commands except the final write of this audit file.

- `pwd && git rev-parse HEAD && git status --short && sed -n '1,220p' AGENTS.md`
- `sed -n '1,260p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt`
- `sed -n '261,620p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt`
- `sed -n '621,1040p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt`
- `find . -maxdepth 3 -type f | sed 's#^./##' | sort | head -n 260`
- `rg --files src scripts tests docs frontend/src | sort`
- Targeted `rg -n` searches for storage/index/policy-call/artifact symbols across `src`, `scripts`, `tests`, `docs`, `configs`, and `frontend/src`.
- Targeted `nl -ba <file> | sed -n '<range>p'` reads for the cited files and symbols, including `docs/dataset-format.md`, `src/vla_lens/traces/*`, `src/vla_lens/dataset/*`, `src/vla_lens/capture/*`, `src/vla_lens/pi05/*`, `src/vla_lens/probes/*`, `src/vla_lens/workbench/*`, `src/vla_lens/server/*`, `src/vla_lens/interventions/*`, and `src/vla_lens/selectors.py`.
- `git status --short && ls -la docs/audits/vla-lens-system-review 2>/dev/null || true`

Not run: `uv run pytest`, `uv run ruff`, app servers, capture scripts, simulators, LeRobot/PI0.5 runtime checks, model downloads, Docker, or destructive git/filesystem commands.
