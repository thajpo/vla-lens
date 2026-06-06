# Dashboard API Contract

Status: active API contract.

Last updated: June 6, 2026.

## Position

The dashboard API is a local, file-backed interface over one opened VLA Lens
dataset root. The root can be:

```text
LeRobot v3 root
LeRobot v3 root plus vla_lens/ overlay
top-level batch output containing nested LeRobot roots
```

Normal dashboard/API work runs in the repo environment with `uv run`. It must
not import PI0.5 hardware capture dependencies, run LIBERO, or load accelerator
Torch stacks.

The FastAPI app exposes:

```text
/api/docs
/api/redoc
/api/openapi.json
```

Those generated docs are useful for route discovery. The typed payload contract
still lives primarily in Python dataclasses under `vla_lens.workbench` and
frontend TypeScript types under `frontend/src/types`.

## Serving

Serve a dataset root directly:

```bash
uv run python scripts/serve_vla_lens_dashboard.py /path/to/dataset-root --port 8765
```

Serve the built single-origin app:

```bash
scripts/view_vla_lens.sh /path/to/dataset-root
```

Run the synthetic dev demo:

```bash
scripts/run_vla_lens_demo.sh
```

## Response Shape

Successful JSON routes return a JSON object. Error routes return:

```json
{"error": "Not Found", "message": "Unknown route: /api/example"}
```

GET routes may use short private caches for paint speed. POST routes and
diagnostic/workbench-state routes use no-store responses and clear the server's
dataset payload cache after state-changing writes.

Media routes return JPEG or MP4 bytes. Pass `v=<fingerprint>` when the frontend
has a dataset/media version and wants immutable media caching.

## Route Groups

| Group | Routes | Purpose |
| --- | --- | --- |
| Service | `/`, `/api/health` | Backend reachability and opened dataset counts. |
| Dataset | `/api/dataset`, `/api/episodes/{trace_id}` | Dataset overview and per-episode detail. |
| Media | `/api/frame`, `/api/episode-video` | Camera frames and cached episode videos. |
| Workbench manifest | `/api/workbench`, `/api/workbench/validate`, `/api/spatial-overlays`, `/api/lens-arrays`, `/api/lens-arrays/{array_id}` | Axes, arrays, panels, workflows, and contract validation. |
| Saved workbench state | `/api/cohorts`, `/api/analysis-runs`, `/api/workspaces`, `/api/intervention-runs`, `/api/intervention-runs/{run_id}` | Persisted cohorts, runs, workspaces, and intervention records. |
| Intervention checks | `/api/interventions/preflight` | Runtime-free readiness checks for intervention requests and saved records. |
| Selection and views | `/api/selections/resolve`, `/api/projection`, `/api/graph`, `/api/tables/query`, `/api/lens-arrays/{array_id}/slice` | Linked-selection resolution and bounded data previews. |
| Episode evidence | `/api/policy-calls`, `/api/action-norm`, `/api/generation-commitment`, `/api/episode-metrics`, `/api/episode-interactions`, `/api/episode-probes` | Time-aligned behavior, action-generation, interaction, and probe evidence. |
| Model internals | `/api/activation-sites`, `/api/activation-slice`, `/api/image-token-map`, `/api/object-camera-overlay`, `/api/attention-map`, `/api/patch-features`, `/api/prompt-attention`, `/api/prompt-feature-map`, `/api/expert-token-activations`, `/api/expert-token-details` | Activation, attention, token, camera, and object overlays. |
| Artifacts | `/api/artifacts`, `/api/artifacts/{artifact_id}`, `/api/artifacts/create/*` | Saved artifacts and built-in artifact creation helpers. |
| Diagnostics | `/api/dataset-diagnostics`, `/api/dataset-diagnostics/run`, `/api/probe-index`, `/api/counterfactual-pairs`, `/api/observational-comparisons` | Trust, probe, pairing, and observational comparison summaries. |

## Query Conventions

Episode-scoped routes generally require `trace_id`. The value comes from
`/api/dataset` or `/api/episodes/{trace_id}`.

Model-site routes usually use:

```text
trace_id          episode trace ID
name              model site name from /api/activation-sites
call              policy-call index, defaulting to the first available call
feature           channel/feature index
generation_step   denoising/action-generation step when the tensor has that axis
```

Attention routes additionally accept:

```text
kind          expert or vlm
head          optional attention-head index
query_token   optional query-token index
```

Media and camera-overlay routes use:

```text
camera     camera name, or all where supported
timestep   environment frame index
source     auto, trace, sparse, or replay for /api/frame
```

## Mutation Conventions

POST routes accept JSON objects. The current server validates the object shape in
the called workbench/dataset helper rather than through Pydantic models at the
FastAPI boundary.

The important body families are:

```text
SelectionState       /api/selections/resolve
CohortSpec           /api/cohorts
AnalysisRunSpec      /api/analysis-runs
InterventionRunSpec  /api/intervention-runs and /api/intervention-runs/{run_id}
Intervention preflight request /api/interventions/preflight
SavedWorkspace       /api/workspaces
table/slice requests /api/tables/query and /api/lens-arrays/{array_id}/slice
```

`/api/interventions/preflight` checks only saved metadata and artifact records:
policy-call rows, stored action arrays, source artifacts, model-site and token
space declarations, action decoder/basis metadata, runtime adapter declarations,
and whether the live model runtime is available. In the normal dashboard
environment it reports model runtime as unavailable rather than importing PI0.5,
Torch, LeRobot, LIBERO, or simulator dependencies.

Keep payload examples in tests or typed frontend fixtures when possible. The
OpenAPI schema describes route intent and common parameters; it is not yet a
complete generated client contract.

## Future Work

- Complete the `/api/frame` single-frame migration after the current direct-file
  and `bundle.frame(...)` tests have stabilized: route all normal frame reads
  through the single-frame API, then remove the legacy full-array `frames()`
  fallback once current dataset writers no longer need it.
