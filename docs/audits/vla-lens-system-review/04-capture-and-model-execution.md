# 04 - Capture and Model Execution Audit

## Inspection Metadata

- Worker: 04
- Scope: model execution plane, PI0.5 capture runtime, temporal alignment, model-site ontology, capture profiles, reconstructability, and environment boundary.
- Inspected commit: `882eeb83be8c8a69a80fc8f6ec8829c311ca4630`
- Git status before writing this audit: clean (`git status --short` returned no output)
- Git status after writing this audit (`git status --short --untracked-files=all`): `?? docs/audits/vla-lens-system-review/02-data-storage-and-indexing.md`; `?? docs/audits/vla-lens-system-review/03-researcher-workflow-and-experiment-api.md`; `?? docs/audits/vla-lens-system-review/04-capture-and-model-execution.md`; `?? docs/audits/vla-lens-system-review/06-evidence-interventions-and-method-extensions.md`; `?? docs/audits/vla-lens-system-review/09-questions-for-owner.md`
- Safety posture: static inspection and read-only commands only. I did not run captures, simulators, model downloads, model imports, GPU checks, `uv run`, or destructive commands.

## Executive Findings

The PI0.5 capture path is a real execution plane, not only a schema sketch. A policy call can be traced from LIBERO observation capture through preprocessing, `policy.predict_action_chunk`, tensor hooks, action postprocessing, env stepping, and LeRobot v3 plus overlay persistence. The strongest implementation spine is:

- `src/vla_lens/pi05/capture_runner.py:224` `run_pi05_capture_task`
- `src/vla_lens/pi05/capture_predict.py:29` `_predict_action_chunk`
- `src/vla_lens/pi05/capture_writer.py:63` `_write_episode`
- `src/vla_lens/dataset/writer.py:63` `write_lerobot_trace_record`
- `src/vla_lens/dataset/overlay.py:57` `_write_overlay_bundle`

The main risks are not that capture is absent. The risks are that important identities are still inferred instead of persisted: exact runtime versions, exact checkpoint fingerprint, explicit generated-action-to-executed-action mapping, per-call observation preprocessing payloads, and site catalog/runtime hook resolution are incomplete. This limits exact replay, intervention reproducibility, and temporal debugging.

The environment boundary is documented and mostly enforced by wrappers, but not enforced at the package entrypoint. `pyproject.toml:24` exposes `vla-pi05-capture`, so a normal `uv run vla-pi05-capture ...` remains technically possible even though `AGENTS.md` and the docs forbid it for normal use.

## Runtime Entry Points

### Observed Implementation Facts

| Surface | Evidence | Assessment |
| --- | --- | --- |
| CLI entrypoints | `pyproject.toml:24` exposes `vla-pi05-capture = "vla_lens.pi05.capture:main"` and `pyproject.toml:25` exposes `vla-pi05-batch-capture = "vla_lens.pi05.batch_capture:main"`. | The PI0.5 path is discoverable from the normal package, which is convenient but weakens the environment split. |
| Capture facade | `src/vla_lens/pi05/capture.py:1` describes a re-export facade for the PI0.5 capture package. It re-exports `main`, `parse_args`, `_run_capture`, `load_pi05_capture_runtime`, and `run_pi05_capture_task` at `src/vla_lens/pi05/capture.py:22`. | Good compatibility shim. Heavy runtime imports remain lazy in the runner, not at facade import time. |
| CLI parse and run | `src/vla_lens/pi05/capture_runner.py:54` `main` parses args, optionally deletes output, runs `_run_capture`, builds the dataset index, opens `TraceDataset`, and validates LeRobot v3 output. | This is the live single-run entrypoint. |
| Capture arguments | `src/vla_lens/pi05/capture_runner.py:69` `parse_args` defines model id, benchmark/task, episodes/seeds, capture profile, hidden/attention resolutions, dtype, output root, dataset id, design metadata, obs size, max steps, device, and dtype. | The command surface exposes most capture design knobs. |
| Runtime loading | `src/vla_lens/pi05/capture_runner.py:175` `load_pi05_capture_runtime` imports Torch, LeRobot, LIBERO, and model components lazily. | This is the correct boundary: import heavy runtime only inside capture runtime loading. |
| Env/model loop | `src/vla_lens/pi05/capture_runner.py:224` `run_pi05_capture_task` creates the env, resets it, buffers observations, invokes policy calls when the current chunk is exhausted, steps the env, and writes episodes. | This is the core execution loop. |
| Model invocation and hooks | `src/vla_lens/pi05/capture_predict.py:29` `_predict_action_chunk` monkey-patches PI0.5 internals, calls `policy.predict_action_chunk(obs)` at `src/vla_lens/pi05/capture_predict.py:345`, captures tensors, and restores originals at `src/vla_lens/pi05/capture_predict.py:348`. | The implementation is invasive but bounded to one policy call via `try/finally`. |
| Full internals recorder | `src/vla_lens/pi05/capture_hooks.py:19` `_PI05FullSiteRecorder` stores declared full-capture tensors per generation step and validates required sites at `src/vla_lens/pi05/capture_hooks.py:48`. | Good explicit completeness mechanism for audit profiles. |
| Generic adapter intent | `src/vla_lens/capture/adapters.py:176` defines `DatasetEpisodeAdapter`, `EnvironmentAdapter`, and `ModelCaptureAdapter` protocols. `src/vla_lens/capture/fake_adapters.py:206` has a fake model adapter that emits policy calls, action chunks, generation actions, and model sites. | The generic contract exists and is tested, but PI0.5 live capture currently bypasses it with a specialized runner. |
| Batch orchestration | `src/vla_lens/pi05/batch_capture.py:148` `main` builds plans, supports dry-run by default, and only executes subprocess commands with `--run`. `_capture_commands` builds `python -m vla_lens.pi05.capture` at `src/vla_lens/pi05/batch_capture.py:358`. | Useful batch plane, but the subprocess Python can fall back to config or `sys.executable` if wrapper env is absent. |
| Shell wrappers | `scripts/pi05_capture.sh:49` selects `.venv-pi05-{rocm,cuda,mps}`, runs `scripts/check_pi05_env.sh` at `scripts/pi05_capture.sh:102`, and execs the capture venv binary at `scripts/pi05_capture.sh:111`. | This is the intended safety boundary for single captures. |
| Batch wrappers | `scripts/pi05_batch_capture.sh:91` exports `VLA_LENS_CAPTURE_PYTHON`, `PYTHONPATH`, device, and dtype before execing `vla-pi05-batch-capture`. | Good wrapper-level handoff into batch capture. |
| Environment checker | `scripts/check_pi05_env.sh:107` checks Torch/backend, `scripts/check_pi05_env.sh:133` checks OpenPI-patched Transformers, `scripts/check_pi05_env.sh:141` checks LeRobot PI0.5, `scripts/check_pi05_env.sh:146` checks LIBERO, and `scripts/check_pi05_env.sh:151` checks `robosuite==1.4.0`. | Strong preflight for capture venvs; currently not embedded into per-trace provenance. |

### One Policy Call Trace

This is the static path for one policy call from observation to persisted overlay.

1. Env reset and per-step observation buffering happen in `src/vla_lens/pi05/capture_runner.py:264` and `src/vla_lens/pi05/capture_runner.py:270`. The runner records observations, context summaries, rendered frames, and wrist frames before deciding whether a new policy call is needed.
2. The policy-call boundary is `if action_iter is None` at `src/vla_lens/pi05/capture_runner.py:280`. This means one model call starts when there are no remaining actions from the previous chunk.
3. The raw environment observation is converted through the PI0.5 preprocessing chain at `src/vla_lens/pi05/capture_runner.py:283` through `src/vla_lens/pi05/capture_runner.py:286`: runtime observation preprocessing, task injection, environment preprocessor, then PI0.5 policy preprocessor.
4. The model call is `_predict_action_chunk(policy, obs, len(buffer.calls), step, plan)` at `src/vla_lens/pi05/capture_runner.py:287`.
5. `_predict_action_chunk` installs wrappers for image embeddings, prefix/suffix embeddings, VLM and expert model forwards, attention, denoising, and action head forward at `src/vla_lens/pi05/capture_predict.py:87`, `src/vla_lens/pi05/capture_predict.py:92`, `src/vla_lens/pi05/capture_predict.py:124`, `src/vla_lens/pi05/capture_predict.py:187`, `src/vla_lens/pi05/capture_predict.py:217`, `src/vla_lens/pi05/capture_predict.py:314`, and `src/vla_lens/pi05/capture_predict.py:166`.
6. The actual policy invocation is `policy.predict_action_chunk(obs)` at `src/vla_lens/pi05/capture_predict.py:345`.
7. The original methods and hooks are restored in `finally` at `src/vla_lens/pi05/capture_predict.py:348` through `src/vla_lens/pi05/capture_predict.py:362`.
8. Captured tensors are converted into arrays at `src/vla_lens/pi05/capture_predict.py:364` through `src/vla_lens/pi05/capture_predict.py:397`.
9. The resulting `CaptureCall` is built at `src/vla_lens/pi05/capture_predict.py:404`. It includes `call_index`, `env_timestep`, `final_action_chunk`, denoising actions, hidden states, attention summaries/full arrays, KV cache, generation embeddings, action head outputs, and metadata.
10. Token and policy-call metadata are attached at `src/vla_lens/pi05/capture_runner.py:288` via `_attach_token_metadata`, implemented at `src/vla_lens/pi05/capture_tables.py:79`.
11. The final chunk is postprocessed into env-space actions at `src/vla_lens/pi05/capture_runner.py:290` through `src/vla_lens/pi05/capture_runner.py:294`.
12. The runner executes the next action with `env.step(action)` at `src/vla_lens/pi05/capture_runner.py:299` and appends `executed_actions`, rewards, terminated/truncated flags, and info at `src/vla_lens/pi05/capture_runner.py:300` through `src/vla_lens/pi05/capture_runner.py:304`.
13. Episode persistence starts at `src/vla_lens/pi05/capture_runner.py:309` via `_write_episode`.
14. `_write_episode` builds arrays and tables at `src/vla_lens/pi05/capture_writer.py:78` through `src/vla_lens/pi05/capture_writer.py:89`, records capture metadata at `src/vla_lens/pi05/capture_writer.py:90`, constructs `PolicyCallRecord` rows at `src/vla_lens/pi05/capture_writer.py:176`, merges episode/model records at `src/vla_lens/pi05/capture_writer.py:198`, and writes LeRobot plus overlay output at `src/vla_lens/pi05/capture_writer.py:199`.
15. The LeRobot writer stores robot frames/actions/timestamps in `src/vla_lens/dataset/writer.py:390` `_write_robot_data`. The overlay writer stores PI0.5 model arrays, policy calls, token tables, and capture tables under `vla_lens/episodes/...` in `src/vla_lens/dataset/overlay.py:57` `_write_overlay_bundle`.

### Inferred Architectural Intent

The system is trying to keep the robot dataset canonical and portable while treating VLA-specific tensors as an overlay. That intent is visible in `src/vla_lens/dataset/overlay.py:149`, which filters robot-owned arrays out of the overlay, and in `docs/model-dataset-sim-agnosticity.md:34`, which describes LeRobot as the canonical robot data layer with VLA Lens metadata in sidecars.

The generic adapter protocols show a longer-term model/dataset/sim agnostic direction, but the production PI0.5 capture path is still specialized around PI0.5, LIBERO, and specific model internals.

## Temporal Alignment

### Observed Implementation Facts

| Question | Evidence | Assessment |
| --- | --- | --- |
| What is a policy-call boundary? | `src/vla_lens/pi05/capture_runner.py:280` starts a call when `action_iter is None`. `src/vla_lens/pi05/capture_runner.py:295` converts the postprocessed chunk into an iterator. | A call corresponds to one generated action chunk, not every env step. |
| How is the observation timestep assigned? | `CaptureCall.env_timestep` is set from `step` at `src/vla_lens/pi05/capture_predict.py:406`. `PolicyCallRecord.to_row` writes `observation_timestep` and `env_timestep_start` from `env_timestep` at `src/vla_lens/capture/records.py:27`. | Good single integer alignment key. |
| How are executed timesteps linked to policy calls? | `_timesteps_table` walks calls and fills `policy_call_for_timestep` and `horizon_index` at `src/vla_lens/pi05/capture_tables.py:37`. | The mapping is inferred into the timesteps table. |
| How is the end of a policy-call segment computed? | `_call_end_timestep` returns the next call start minus one, or episode length minus one, at `src/vla_lens/pi05/capture_writer.py:332`. | Reasonable for sequential chunk execution. |
| Where are generated chunks stored? | `_episode_arrays` writes `action_chunks` with axes `["policy_call", "horizon", "action_dim"]` at `src/vla_lens/pi05/capture_arrays.py:44`, and `generation_actions` with axes `["policy_call", "generation_step", "horizon", "action_dim"]` at `src/vla_lens/pi05/capture_arrays.py:51`. | Good array-level distinction between final chunk and denoising trajectory. |
| Where are generation steps represented as rows? | `_generation_steps_table` emits one row per policy call and generation step at `src/vla_lens/pi05/capture_tables.py:64`. | It captures denoising step identity, but not which horizons were eventually executed. |
| How are robot timestamps produced? | `_write_robot_data` writes `timestamp = frame_index / fps` at `src/vla_lens/dataset/writer.py:402`. | Robot time is frame-index-derived, not runtime wall-clock-derived. |
| How does UI/backend select policy-call axes? | `_policy_call_axis_selection` uses call `index` for `policy_call` axes and `env_timestep` for timestep axes at `src/vla_lens/server/common.py:269`. | This can conflate UI row index with stable `policy_call_index` if they diverge. |
| How are action-generation artifacts summarized? | `_bundle_summary` compares `generation[index, -1, horizon]` to `actions[timestep + horizon]` at `src/vla_lens/action_generation.py:159`. | Generated-to-executed mapping is derived, not persisted as a first-class relation. |

### Gaps

1. There is no explicit persisted table saying "policy call N, horizon H, planned action A, executed at timestep T, status executed/truncated/dropped/repeated." The system can infer this from `timesteps`, `policy_calls`, `action_chunks`, and `executed_actions`, but inference becomes fragile around early termination, truncation, env errors, or future asynchronous execution.
2. `generation_steps` has no horizon or execution linkage. This makes it hard to inspect whether intermediate denoising steps for a specific horizon correspond to an action that actually reached the env.
3. Dropped, repeated, and truncated chunk tails are implicit. `terminated` and `truncated` are stored in `timesteps` and evaluation tables, but not tied back to unexecuted horizon positions.
4. Clock semantics are single-clock and frame-index based. The code records env step and LeRobot timestamp, but not separate observation acquisition time, policy-call start/end wall time, model generation latency, or env step latency.
5. `_attach_token_metadata` uses `buffer.frames[0]` for image preprocessing metadata at `src/vla_lens/pi05/capture_tables.py:93`, which appears to describe the first episode frame rather than the per-call observation frame. This is acceptable for static shape metadata but misleading if preprocessing metadata becomes image-specific.

### Exact Improvements

1. Add an `action_execution_map` overlay table in `src/vla_lens/pi05/capture_tables.py` with columns: `trace_id`, `episode_id`, `policy_call_index`, `env_timestep_start`, `env_timestep_end`, `horizon_index`, `planned_env_timestep`, `executed_env_timestep`, `status`, `action_chunk_array_ref`, `executed_action_array_ref`, and optional error/truncation reason.
2. Populate the table in `_write_episode` from `buffer.calls`, `buffer.executed_actions`, `buffer.terminated`, and `buffer.truncated`. Persist it through `TraceBundle.create` and `_write_overlay_bundle`.
3. Add a static unit test around `_timesteps_table`, `_call_end_timestep`, `action_chunks`, and `executed_actions` for normal chunks, early termination, truncation, and a final partial chunk.
4. In `src/vla_lens/server/common.py:269`, prefer stable `model_call_index`/`policy_call_index` for `policy_call` axis selection when present, and keep row index as a UI-only fallback.
5. Change `_attach_token_metadata` to use the frame at `call.env_timestep` when computing image preprocessing metadata, or label the current values as processor-shape metadata rather than observation-specific metadata.

## Model-Site Ontology

### Observed Implementation Facts

| Area | Evidence | Assessment |
| --- | --- | --- |
| Generic site schema | `src/vla_lens/traces/types.py:58` `ModelSiteSpec` includes `name`, `array`, `axes`, `module`, `layer`, `tensor_type`, `token_kind`, metadata, family, role, segment, materialization, exactness, token spaces, capture family, view kind, and derivation. | Good shared representation for storage and UI. |
| PI0.5 full declarations | `src/vla_lens/pi05/full_capture.py:26` `SiteCaptureDeclaration` stores site name, axes, module, tensor type, family, role, segment, layer, token spaces, and required/full metadata. | Strong declaration layer for exact internal sites. |
| Semantic plus runtime path | `_transformer_layer_declarations` declares names such as VLM/expert layer tensors and runtime modules at `src/vla_lens/pi05/full_capture.py:411`. | The site name is semantic-ish but PI0.5-specific; `module` is runtime-path-oriented. |
| Attention coordinate metadata | `_attention_coordinate_metadata` records coordinate system version `pi05_attention_v1`, backend, mask semantics, and formula at `src/vla_lens/pi05/full_capture.py:839`. | This is a good precedent for ontology versioning. |
| Materialized site rows | `TraceBundle.create` writes model site rows with `site_id`, `name`, `module`, `layer`, `tensor_type`, `token_kind`, axes, shape, dtype, model path, family, role, segment, materialization, exactness, token spaces, capture family, view kind, and derivation at `src/vla_lens/traces/bundle.py:175`. | Good storage-level ontology. |
| Indexed site rows | `src/vla_lens/dataset/index.py:62` `MODEL_SITE_COLUMNS` include name/module/layer/tensor type/token kind/axes/shape/dtype/path/family/role/segment. | Index supports discovery, but omits some richer fields such as exactness, materialization, capture role, token spaces, and derivation. |
| Intervention target contract | `src/vla_lens/interventions/specs.py:288` `TargetSpec` can request target by `model_site`, `site_id`, `module_path`, layer, tensor type, token space, token selector, generation step selector, reduction, and value basis. | The intervention interface is designed around model-site identities. |
| Intervention preflight | `_target_site_check` resolves targets against `dataset.model_site_index` at `src/vla_lens/interventions/preflight.py:314`. | Good runtime-free validation, but it depends on index richness. |
| Runtime resolution record | `_runtime_resolution` stores requested target and resolved hook metadata at `src/vla_lens/pi05/intervention_runtime.py:237`. | Good design, but the actual PI0.5 hook executor is injected and not implemented here. |

### Gaps

1. There is no explicit site catalog version for PI0.5 beyond attention coordinate metadata. A PI0.5 architecture or hook-name change could preserve array shapes while changing semantics.
2. Requested versus resolved target identity is complete for intervention runtime records, but capture itself mostly stores declared/captured/missing site sets in `capture_report` at `src/vla_lens/pi05/capture_writer.py:211`; it does not store a per-site requested target to resolved runtime hook map.
3. The generic adapter contract does not yet drive PI0.5 live capture. That means model-site ontology is partly generic in storage but partly PI0.5-specific in execution.
4. The model-site index loses fields that would help downstream tools distinguish raw tensors from reductions and exact sites from approximate/derived views.
5. `src/vla_lens/pi05/selectors.py:22` imports Torch at module import time. This is a model-specific runtime dependency in a selector module and should stay out of normal-lane imports.

### Exact Improvements

1. Add `site_catalog_version` and `runtime_hook_catalog_version` to `CapturePlan.to_metadata` in `src/vla_lens/pi05/capture_schema.py:131` and to each PI0.5 `ModelSiteSpec.metadata`.
2. Add a per-site capture resolution table with `requested_site`, `declared_site`, `resolved_module_path`, `hook_kind`, `materialized_array`, `exactness`, `missing_reason`, and `runtime_shape`.
3. Extend `MODEL_SITE_COLUMNS` in `src/vla_lens/dataset/index.py:62` to include `exactness`, `materialization`, `capture_role`, `view_kind`, `token_space_ids`, `derived_from`, and `derivation`.
4. Move the top-level Torch import in `src/vla_lens/pi05/selectors.py:22` into the functions/classes that require it, or mark this module as capture-runtime-only and add a normal import boundary test.
5. If PI0.5 is intended to be one implementation of `ModelCaptureAdapter`, add an adapter wrapper that exposes PI0.5 capabilities through `src/vla_lens/capture/adapters.py:193` rather than only through `run_pi05_capture_task`.

## Capture Profiles

### Observed Implementation Facts

| Profile/Mechanism | Evidence | Assessment |
| --- | --- | --- |
| Canonical profiles | `src/vla_lens/pi05/capture_schema.py:14` defines profile constants including rollout, features, mechanistic sampled/all, internals sampled, audit sampled/windowed/full, and aliases. | Clear finite profile ladder. |
| Layer selection | `src/vla_lens/pi05/capture_schema.py:32` maps profiles to layer selections. `src/vla_lens/pi05/capture_runner.py:345` resolves requested layers and custom overrides. | Good profile-to-plan translation. |
| Hidden/attention resolutions | `CapturePlan` stores hidden and attention resolution at `src/vla_lens/pi05/capture_schema.py:93`; `_resolve_capture_plan` fills these at `src/vla_lens/pi05/capture_runner.py:345`. | Good explicit storage for shape/cost decisions. |
| Serializable plan | `CapturePlan.to_metadata` serializes profile, requested profile, layers, dtype, dimensions, bridge sites, audit sites, and runtime collections at `src/vla_lens/pi05/capture_schema.py:131`. | Good manifest-level visibility. |
| Profile dimensions | `profile_dimensions` documents expected tensors, layers, reductions, and profile semantics at `src/vla_lens/pi05/capture_schema.py:205`. | Good UI/doc bridge. |
| Docs | `docs/pi05-capture-profiles.md:26` explains the profile ladder and cost/claim discipline. Audit profiles are documented at `docs/pi05-capture-profiles.md:267` and `docs/pi05-capture-profiles.md:343`. | Strong user-facing profile documentation. |
| Capture report | `_capture_report` records profile, complete flag, declared/captured/missing sites, and required site status at `src/vla_lens/pi05/capture_writer.py:211`. | Good validation artifact. |
| Tests | `tests/pi05_capture_success_test.py:106` tests profile aliases; `tests/pi05_capture_success_test.py:126` tests audit dimensions; `tests/pi05_full_capture_test.py:135` tests completeness behavior. | Good schema/declaration coverage. |

### Gaps

1. Profiles are serializable, but not separately hashable as profile contracts. Trace fingerprints include capture plan/report via `src/vla_lens/traces/fingerprints.py:37`, but there is no standalone `capture_profile_hash`.
2. Sampling frequency is mostly profile/layer based. There is no generic per-site frequency/window policy beyond `audit_windowed` and layer selection.
3. Token subset policy is implicit in hidden/attention resolution and token-space metadata. There is no reusable token subset selector contract that says exactly which image/language/action tokens were retained for sampled views.
4. Storage policy is global by dtype and zarr compression. Per-site dtype/compression/chunk policy is not part of the profile contract.
5. Custom profiles are CLI-configurable for layer/resolution/dtype, but not a fully declarative external profile file with stable hash and validation report.

### Exact Improvements

1. Add `capture_profile_hash` to `CapturePlan.to_metadata`, computed from canonical JSON of profile, layer list, hidden/attention resolution, dtype, full-site declarations, token subset policy, and storage policy.
2. Represent sampled token/reduction decisions as explicit profile fields: `token_selection_policy`, `attention_reduction_policy`, and `generation_step_policy`.
3. Store per-site `requested_resolution`, `materialized_resolution`, and `storage_dtype` in model-site metadata.
4. Add a `profiles/*.yaml` or equivalent declarative profile source for custom profiles, then validate it with the same logic as `CapturePlan`.

## Reconstructability

### What Is Persisted

| Needed for reconstruction | Evidence | Current classification |
| --- | --- | --- |
| Robot actions, rewards, done/truncated, timestamps | `_write_robot_data` writes action, reward, done, truncated, observation state, and frame-index timestamps at `src/vla_lens/dataset/writer.py:390`. | Persisted for action replay and dataset inspection. |
| Frames/videos | `write_lerobot_trace_record` writes robot data and videos; overlay keeps non-robot arrays separate at `src/vla_lens/dataset/writer.py:63` and `src/vla_lens/dataset/overlay.py:149`. | Persisted/referenced through LeRobot v3 layout. |
| Task/prompt/seed | `_write_episode` stores prompt, task name, task id, and seed in manifest/context/environment metadata at `src/vla_lens/pi05/capture_writer.py:90` and `src/vla_lens/pi05/capture_writer.py:130`. | Mostly persisted. |
| Policy-call records | `PolicyCallRecord` stores `policy_call_index`, `observation_timestep`, `env_timestep_start`, and metadata at `src/vla_lens/capture/records.py:19`. | Persisted. |
| Final and intermediate actions | `action_chunks`, `generation_actions`, and `generation_velocities` are written at `src/vla_lens/pi05/capture_arrays.py:44` through `src/vla_lens/pi05/capture_arrays.py:62`. | Persisted as arrays. |
| Model activations | `_model_arrays` writes hidden, attention, KV, expert input, and action-head arrays at `src/vla_lens/pi05/capture_arrays.py:69`. Full internals come from `_full_model_site_specs` at `src/vla_lens/pi05/capture_arrays.py:364`. | Persisted according to profile. |
| Context snapshots | `src/vla_lens/pi05/context_capture.py:29` aggregates robot, env, scene, and camera context. Tests cover time-aligned scene/camera snapshots in `tests/pi05_context_capture_test.py:256` and `tests/pi05_context_capture_test.py:329`. | Persisted when accessible; missing fields are represented with status rows. |
| Prompt/token/image metadata | `_attach_token_metadata` and token table builders live at `src/vla_lens/pi05/capture_tables.py:79` and `src/vla_lens/pi05/capture_tables.py:153`. | Persisted, but image preprocessing metadata is not clearly per-call. |
| Action normalization | `_action_normalization_table` writes normalization rows at `src/vla_lens/pi05/capture_tables.py:331`, and `_canonical_action_normalization` maps normalized arrays to LeRobot action at `src/vla_lens/dataset/overlay.py:164`. | Persisted as metadata, but not enough to reconstruct processor internals alone. |
| Fingerprints | `_compute_trace_fingerprints` hashes trajectory, context, and schema payloads at `src/vla_lens/traces/fingerprints.py:37`. | Good dataset integrity signal. |

### What Is Missing or Too Weak

1. `ModelDescriptor.checkpoint_sha` exists at `src/vla_lens/capture/records.py:97`, but PI0.5 `_write_episode` constructs `ModelDescriptor` without setting it at `src/vla_lens/pi05/capture_writer.py:163`.
2. Runtime package versions are checked by `scripts/check_pi05_env.sh:190`, but they are not embedded in each trace. A trace should record Python version, executable path, platform, Torch version/build/backend, LeRobot version, Transformers/OpenPI patch identity, LIBERO version/path, robosuite version, and VLA Lens commit.
3. The exact model/preprocessor/postprocessor config and normalization statistics are not fingerprinted as first-class artifacts. `preprocess_id` and `postprocess_id` are stored as `lerobot.default` at `src/vla_lens/pi05/capture_writer.py:189`, which is too coarse.
4. RNG and denoising state are not fully persisted. The episode seed is stored, but policy/model RNG states, sampler state, deterministic flags, and any model-side random sources are not recorded.
5. The exact observation tensor payload after all preprocessors is not persisted as a named artifact. Raw-ish observations/context are captured, but not necessarily the final model input dictionary after `runtime.preprocessor`.
6. `src/vla_lens/pi05/replay.py:93` `replay_config_from_bundle` looks for `env_seed` or `policy_seed`, while `_write_episode` stores `seed` in metadata at `src/vla_lens/pi05/capture_writer.py:99` and environment metadata at `src/vla_lens/pi05/capture_writer.py:136`. Replay can therefore default to seed `0` for PI0.5 traces even when the trace recorded a seed.

### Reconstruction Classification

| Reconstruction target | Classification | Reason |
| --- | --- | --- |
| Inspect saved trace without PI0.5 runtime | Strong | Overlay and LeRobot data are portable; `TraceBundle` and `TraceDataset` read persisted tables/arrays. |
| Replay executed action sequence in simulator | Partial | Executed actions and task metadata exist, and `src/vla_lens/pi05/replay.py:35` can drive LIBERO replay, but seed lookup and runtime version provenance are incomplete. |
| Recompute exact original policy action chunk | Weak | Model id, prompt, seed, and some preprocessing metadata exist, but checkpoint sha, full runtime versions, processor config fingerprints, RNG state, and final model input payload are missing. |
| Reconstruct exact hidden/attention tensors from source | Weak to impossible | Captured tensors are persisted for the selected profile, but exact re-execution depends on missing checkpoint/runtime/RNG/proprocessor details. |
| Validate trace integrity after capture | Moderate | Trace fingerprints cover trajectory, context, schema, and arrays in `src/vla_lens/traces/fingerprints.py:37`, but do not prove replayability. |

### Exact Improvements

1. Populate `ModelDescriptor.checkpoint_sha` and add `model_revision`, `model_config_hash`, `preprocessor_config_hash`, `postprocessor_config_hash`, and `normalization_stats_hash`.
2. Add a `runtime_versions` table or manifest block during `_write_episode` that embeds the successful `check_pi05_env` facts and the VLA Lens git commit.
3. Persist the final preprocessed model input payload schema and optionally arrays for small non-image fields. For large tensors, persist hashes, shapes, dtypes, and references.
4. Record policy-call wall-clock timings: observation captured, preprocessing start/end, model call start/end, postprocess end, env step start/end.
5. Fix `replay_config_from_bundle` to read `metadata["seed"]` and nested environment seed, or write `env_seed` explicitly from `_write_episode`.

## Environment Boundary

### Observed Implementation Facts

| Boundary item | Evidence | Assessment |
| --- | --- | --- |
| Project rule | `AGENTS.md` says normal repo work uses `.venv` and `uv run`, while PI0.5 capture uses `.venv-pi05-rocm`, `.venv-pi05-cuda`, `.venv-pi05-mps`, or Docker wrappers. | Correct operational split. |
| Dedicated docs | `docs/pi05-rocm-capture-env.md:11` describes the two-world split and `docs/pi05-rocm-capture-env.md:86` states the practical rule. `docs/hardware-run-paths.md:27` defines the portable path without Torch/LeRobot/LIBERO/GPU. | Good documentation. |
| Capture wrappers | `scripts/pi05_capture.sh:49` chooses capture venv by backend and refuses missing entrypoint at `scripts/pi05_capture.sh:84`. | Good wrapper enforcement. |
| Batch wrapper env | `scripts/pi05_batch_capture.sh:91` exports capture Python and runtime env vars. | Good batch boundary when wrapper is used. |
| Capture runner lazy imports | `load_pi05_capture_runtime` imports heavy packages at `src/vla_lens/pi05/capture_runner.py:175`, not at normal module import. | Good implementation boundary. |
| Fake adapter tests | `tests/adapter_compliance_test.py:15` verifies fake adapter output without PI0.5 dependencies. | Good normal-lane compliance coverage. |
| Normal-lane risk | `src/vla_lens/pi05/selectors.py:22` imports Torch at top level. | This is a concrete import-boundary leak. |
| Entry point risk | `pyproject.toml:24` exposes capture CLI in the normal package. `src/vla_lens/pi05/capture_runner.py:72` documents that users should use wrappers, but it does not enforce the rule. | A user can still run the wrong lane. |
| Batch runtime risk | `_capture_commands` can use config `python_executable` or `sys.executable` at `src/vla_lens/pi05/batch_capture.py:322` if wrapper env is absent. | `--run` should fail closed without wrapper/capture env. |

### Exact Improvements

1. Add a fail-fast guard in `src/vla_lens/pi05/capture_runner.py:54` `main`: require an environment marker set by `scripts/pi05_capture.sh`, `scripts/pi05_batch_capture.sh`, or Docker, unless an explicit emergency flag is passed. This turns the documented critical rule into an enforceable runtime rule.
2. Have wrappers export something like `VLA_LENS_PI05_CAPTURE_ENV=rocm|cuda|mps|docker` and have the Python entrypoints record that value in trace metadata.
3. In `src/vla_lens/pi05/batch_capture.py:148`, when `--run` is set and `VLA_LENS_CAPTURE_PYTHON` is absent, fail with a message telling the user to run `scripts/pi05_batch_capture.sh --backend ...`.
4. Move `src/vla_lens/pi05/selectors.py:22` Torch import behind function boundaries, or add a normal-env import test that proves no dashboard/server/index path imports it.
5. Add a static test for normal CLI/server imports that asserts `torch`, `lerobot`, `libero`, and `robosuite` are not loaded for normal trace/dataset/dashboard operations.
6. Embed the output facts from `scripts/check_pi05_env.sh` into each trace so future debugging does not require reconstructing which capture venv was used.

## Highest Leverage Fix List

1. Persist an explicit generated-action-to-executed-action map. This closes the largest temporal-alignment ambiguity and gives the UI a stable source of truth for chunks, horizons, truncation, and failures.
2. Store per-trace runtime provenance and checkpoint/config hashes. This is the biggest reconstructability improvement.
3. Enforce the capture environment boundary at Python entrypoints, not only shell wrappers and docs.
4. Add PI0.5 site catalog versioning plus per-site requested-to-resolved hook records.
5. Fix replay seed resolution so PI0.5 traces replay with the recorded seed.
6. Make `pi05.selectors` lazy-import Torch or isolate it as runtime-only.
7. Add normal-lane import boundary tests and temporal alignment tests for truncated/partial chunks.

## Commands Used

All commands were static/read-only except creating this single audit file with `apply_patch`. No capture, simulator, model-download, GPU, or destructive commands were run.

```bash
sed -n '1,220p' AGENTS.md
wc -l /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt
sed -n '1,260p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt
sed -n '261,620p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt
sed -n '621,980p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt
sed -n '981,1320p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt
sed -n '1321,1485p' /home/j/.codex/attachments/844bb549-31eb-4ad9-861b-4d3a16a10472/pasted-text.txt
git rev-parse HEAD
git status --short
git status --short --untracked-files=all
git status --short -- docs/audits/vla-lens-system-review/04-capture-and-model-execution.md
rg --files
rg -n "policy_call|env_timestep|horizon_index|generation_step|action_chunks|executed_actions|timestamp|clock|truncated|dropped|repeated|seed|fingerprint|checkpoint" src tests docs
rg -n "import torch|from lerobot|import lerobot|libero|robosuite|transformers" src tests scripts docs
rg -n "checkpoint_sha|checkpoint|package|version|versions|torch|lerobot|preprocess|postprocess|seed|rng|determin|fingerprint" src/vla_lens/pi05 src/vla_lens/capture src/vla_lens/traces scripts docs/current-state.md docs/pi05-rocm-capture-env.md
nl -ba pyproject.toml | sed -n '1,80p'
nl -ba src/vla_lens/pi05/capture.py | sed -n '1,180p'
nl -ba src/vla_lens/pi05/capture_runner.py | sed -n '1,430p'
nl -ba src/vla_lens/pi05/capture_schema.py | sed -n '1,430p'
nl -ba src/vla_lens/pi05/capture_predict.py | sed -n '1,520p'
nl -ba src/vla_lens/pi05/capture_hooks.py | sed -n '1,470p'
nl -ba src/vla_lens/pi05/full_capture.py | sed -n '1,930p'
nl -ba src/vla_lens/pi05/capture_arrays.py | sed -n '1,540p'
nl -ba src/vla_lens/pi05/capture_tables.py | sed -n '1,430p'
nl -ba src/vla_lens/pi05/capture_writer.py | sed -n '1,370p'
nl -ba src/vla_lens/capture/adapters.py | sed -n '1,230p'
nl -ba src/vla_lens/capture/fake_adapters.py | sed -n '1,330p'
nl -ba src/vla_lens/capture/records.py | sed -n '1,330p'
nl -ba src/vla_lens/traces/types.py | sed -n '1,130p'
nl -ba src/vla_lens/dataset/writer.py | sed -n '1,520p'
nl -ba src/vla_lens/dataset/overlay.py | sed -n '1,210p'
nl -ba src/vla_lens/traces/bundle.py | sed -n '1,390p'
nl -ba src/vla_lens/traces/io.py | sed -n '1,220p'
nl -ba src/vla_lens/traces/layout.py | sed -n '1,80p'
nl -ba src/vla_lens/traces/fingerprints.py | sed -n '1,250p'
nl -ba src/vla_lens/dataset/index.py | sed -n '1,380p'
nl -ba scripts/pi05_capture.sh | sed -n '1,140p'
nl -ba scripts/pi05_batch_capture.sh | sed -n '1,130p'
nl -ba scripts/check_pi05_env.sh | sed -n '1,230p'
nl -ba scripts/setup_pi05_env.sh | sed -n '1,240p'
nl -ba scripts/docker_pi05.sh | sed -n '1,240p'
nl -ba configs/pi05_light_5_test.yaml | sed -n '1,80p'
nl -ba configs/pi05_broad_1000_mech_light.yaml | sed -n '1,90p'
nl -ba src/vla_lens/pi05/batch_capture.py | sed -n '1,730p'
nl -ba docs/pi05-rocm-capture-env.md | sed -n '1,230p'
nl -ba docs/hardware-run-paths.md | sed -n '1,230p'
nl -ba docs/pi05-capture-profiles.md | sed -n '1,580p'
nl -ba docs/current-state.md | sed -n '1,260p'
nl -ba docs/model-dataset-sim-agnosticity.md | sed -n '1,270p'
nl -ba tests/pi05_capture_success_test.py | sed -n '1,530p'
nl -ba tests/pi05_full_capture_test.py | sed -n '1,220p'
nl -ba tests/pi05_batch_capture_test.py | sed -n '1,280p'
nl -ba tests/docker_pi05_wrapper_test.py | sed -n '1,90p'
nl -ba tests/adapter_compliance_test.py | sed -n '1,100p'
nl -ba tests/lerobot_v3_contract_test.py | sed -n '1,120p'
nl -ba tests/pi05_context_capture_test.py | sed -n '1,460p'
nl -ba tests/vla_lens_trace_workbench_test.py | sed -n '200,310p'
nl -ba tests/research_ui_import_boundary_test.py | sed -n '1,130p'
nl -ba src/vla_lens/server/metrics.py | sed -n '1,130p'
nl -ba src/vla_lens/server/common.py | sed -n '230,290p'
nl -ba src/vla_lens/action_generation.py | sed -n '1,320p'
nl -ba src/vla_lens/selectors.py | sed -n '1,390p'
nl -ba src/vla_lens/pi05/selectors.py | sed -n '1,260p'
nl -ba src/vla_lens/pi05/replay.py | sed -n '1,180p'
nl -ba src/vla_lens/pi05/context_capture.py | sed -n '1,120p'
nl -ba src/vla_lens/pi05/context_capture_common.py | sed -n '1,270p'
nl -ba src/vla_lens/interventions/specs.py | sed -n '1,390p'
nl -ba src/vla_lens/interventions/preflight.py | sed -n '1,620p'
nl -ba src/vla_lens/pi05/intervention_preflight.py | sed -n '1,90p'
nl -ba src/vla_lens/pi05/intervention_runtime.py | sed -n '1,290p'
nl -ba src/vla_lens/interventions/runtime.py | sed -n '1,90p'
pwd
git ls-files docs/audits/vla-lens-system-review/04-capture-and-model-execution.md
ls -la docs/audits/vla-lens-system-review
wc -l docs/audits/vla-lens-system-review/04-capture-and-model-execution.md
rg -n "<status-placeholder-or-marker-patterns>" docs/audits/vla-lens-system-review/04-capture-and-model-execution.md
rg -n "<status-placeholder>|Inspected commit|Git status" docs/audits/vla-lens-system-review/04-capture-and-model-execution.md
sed -n '1,80p' docs/audits/vla-lens-system-review/04-capture-and-model-execution.md
tail -n 80 docs/audits/vla-lens-system-review/04-capture-and-model-execution.md
```
