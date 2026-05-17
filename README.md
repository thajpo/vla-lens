# VLA-lens

Episode-aligned interpretability packaging for Vision-Language-Action robot
policies.

VLA-lens stores robot rollouts, camera frames, actions, model internals, and saved
analysis outputs in a trace format. It then provides selectors, probes, action-head
analysis, reusable artifacts, and a local dashboard for inspecting what a policy saw,
represented, and planned over time.

The main idea:

```text
HF model + environment + capture profile
→ captured episodes
→ VLA-lens interpretability package
→ linked visual workbench and reproducible analyses
```

## What You Can Do

- Run PI0.5 in LIBERO and receive a reusable `.vlatrace` interpretability package.
- Import compatible capture directories into the same trace contract when needed.
- Browse datasets and open individual robot episodes.
- Inspect frames, prompts, actions, model data, feature activations, image-token overlays,
  expert/action-token activations, and generation/action-flow traces.
- Choose PI0.5 capture profiles for rollout, features, mechanistic_sampled, mechanistic_all, internals_sampled, audit_full, or custom interpretability budgets.
- Run a dataset analyzer that recommends defensible next analyses with concrete evidence.
- Generate compressed episode videos from recorded policy decisions.
- Train probe suites from YAML specs and save them as `LensArtifact`s.
- Save VLA-specific `ActionGeneration` artifacts that summarize how action chunks form over
  generation steps.

## Package Split

`src/vla_lens` is the framework:

- `traces.py`: trace bundles, dataset indexes, array loading, artifact persistence.
- `selectors.py`: axis-aware activation selection and feature matrix caching.
- `artifacts.py`: saved analysis records with provenance and display metadata.
- `analyzer.py`: dataset-aware analysis recommendations.
- `probes/`: probe training workflows and baselines.
- `action_generation.py`: action-head generation summaries.
- `server.py` and `live_dashboard.py`: local dashboard and APIs.
- `importers/pi05_legacy.py`: one-way PI0.5 capture-to-`.vlatrace` converter.
- `pi05/`: PI0.5-specific selectors, replay, and intervention specs.

## Quick Start

Serve an existing trace dataset:

```bash
uv run python scripts/serve_vla_lens_dashboard.py runs/pi05_high10_vlatraces --port 8765
```

Open:

```text
http://127.0.0.1:8765/
```

Capture PI0.5 episodes directly into VLA-lens traces:

```bash
uv run vla-pi05-batch-capture --config configs/pi05_light_5_test.yaml --run
```

The `.vlatrace` bundle is the canonical VLA-lens episode record used by the
backend and webapp.

The batch runner is the normal run surface. It writes an `episode_plan.csv`
containing one row per intended episode:

```csv
dataset_id,benchmark,task_id,seed,split,capture_profile
pi05-light-5-test,libero_object,0,1300,train,mechanistic_sampled
```

Each captured episode stores that value as `manifest.metadata.dataset_id`. The
CSV controls dataset/task/seed/profile variation; the capture code stays one
package-native command.

For schema/UI smoke testing, run a tiny matrix over all capture profiles:

```bash
uv run python scripts/run_capture_profile_smoke.py \
  --model-id lerobot/pi05_libero_finetuned \
  --episodes 2 \
  --delete-existing \
  --capture-command 'uv run vla-pi05-capture --model-id {model_id} --episodes {episodes} --start-seed {start_seed} --capture-profile {profile} --dataset-id {dataset_id} --vlatrace-out-root {traces_root}'
```

The smoke script owns the profile roots and trace validation. The runner
command is a template because the concrete PI0.5 capture
entrypoint is model/environment-specific.

Run dataset diagnostics:

```bash
uv run python scripts/save_vla_lens_dataset_report.py runs/pi05_high10_vlatraces
```

Train a probe from a YAML spec:

```bash
uv run python scripts/train_vla_lens_probe.py runs/pi05_high10_vlatraces --spec - <<'YAML'
name: Outcome probe over expert action features
target:
  kind: outcome
features:
  module: pi05.expert.layers.*
  tensor_type: hidden_tokens
  token_kind: action
  layers: null
  timesteps: all
  generation_step: null
  reduction: mean
split:
  kind: heldout_benchmark
baseline:
  - majority_class
  - benchmark
  - target_object
sweep: layer
YAML
```

Save an action-generation artifact:

```bash
uv run python scripts/save_vla_lens_action_generation.py runs/pi05_high10_vlatraces
```

Refresh the dashboard and open the Artifacts page to inspect saved results.

PI0.5 capture profiles are named `rollout`, `features`,
`mechanistic_sampled`, `mechanistic_all`, `internals_sampled`, `audit_full`,
and `custom`. In short: use `vla-pi05-batch-capture` for dataset-scale work.
`mechanistic_sampled` is the cheap default; `mechanistic_all` is the best
serious single-trace inspector profile; `audit_full` adds raw forward internals
and is intentionally expensive. Legacy aliases still work for one compatibility
cycle: `representation`, `mechanistic_light`, `mechanistic_heavy`, and `full`.
Sampled PI0.5 model profiles capture the same VLM and expert layer indices
(`0, 4, 8, 12, 17`) so inspected prefix K/V pairs line up as
`VLM L_i -> Expert L_i`.

Use `dataset_id` for capture-run provenance. It is stored in
`manifest.metadata.dataset_id` and in generated `episode_plan.csv` /
`probe_splits.csv` files so datasets can stay flat-ish while still being easy to
filter and audit.

Every newly written `.vlatrace` also stores fast provenance fingerprints in
`tables/fingerprints.json`, `manifest.metadata.fingerprints`, and
`capture_report.fingerprints`:

- `trajectory_fingerprint`: executed/actions/generation trajectory plus timestep-policy mapping.
- `context_fingerprint`: object/robot/camera/evaluation/preprocessing context.
- `trace_schema_fingerprint`: token/model-site/table semantics and capture request/plan/report.
- `trace_fingerprint`: the combined trace identity for provenance checks.

Probe artifacts record source episode fingerprints plus the concrete feature
matrix, target, and row-index fingerprints, so a later probe result can tell
whether behavior, context, trace semantics, or the extracted training data changed.

## Trace And Artifact Workflow

```mermaid
flowchart LR
    Capture["Policy rollout / legacy capture"] --> Import["Importer / recorder"]
    Import --> Trace[".vlatrace bundle"]
    Trace --> Dataset["TraceDataset"]
    Dataset --> Selector["ActivationQuery"]
    Selector --> Probe["ProbeSuite"]
    Dataset --> Action["ActionGeneration"]
    Probe --> Artifact["LensArtifact"]
    Action --> Artifact
    Artifact --> Dashboard["Dashboard"]
```

## Dataset Analyzer Philosophy

The analyzer should guide researchers with evidence, not jargon.

Good guidance looks like:

```text
Outcome probe is possible.
Evidence: 14 success and 6 failure episodes are available.
Risk: 0/20 tasks have both success and failure, so task identity can be confused with behavior.
Suggested artifact: ProbeSuite over expert action-token features with benchmark/object baselines.
```

Bad guidance looks like:

```text
Warning: seed split invalid.
```

The goal is to help a researcher choose analyses that the current dataset can actually support.

## Current Direction

The dashboard remains episode-first for inspection:

- Home: dataset analyzer and artifact overview.
- Episodes: browse and inspect robot episodes.
- Artifacts: saved probes, videos, reports, action-generation summaries, and future analysis outputs.

Artifacts are becoming the durable research objects. Every serious analysis should save:

- source episodes
- selector/spec
- method
- metrics
- display data
- linked dashboard state when possible

## Development

Run checks:

```bash
uv run ruff check src tests scripts
uv run pytest
```

Run the focused trace MVP tests:

```bash
uv run pytest tests/vla_lens_trace_mvp_test.py
```
