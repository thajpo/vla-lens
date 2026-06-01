# Workbench Frontend Contract

Status: active frontend architecture contract.

Last updated: May 27, 2026.

## Position

The frontend is a Vite/React workbench for local dataset inspection. It should
render whatever the backend says the dataset can support, instead of assuming
PI0.5 internals are always present.

The frontend does not run capture. It talks to the normal dashboard API and can
inspect plain LeRobot roots, LeRobot roots with `vla_lens/` overlays, and
nested batch outputs.

## Main Modules

| Area | Files | Responsibility |
| --- | --- | --- |
| App shell | `frontend/src/App.tsx`, `components/layout/*` | Page selection, navigation, layout frame. |
| Dataset/workbench pages | `pages/WorkbenchPage.tsx`, `pages/workbench/*` | Dataset overview, artifacts, cohorts, saved workbench state. |
| Episode microscope | `pages/EpisodesPage.tsx`, `pages/episodes/*` | Episode navigation, camera timeline, model-site inspector, probe panels, pipeline map. |
| Capability gating | `pages/capabilityGating.ts` | Converts backend capabilities into visible/enabled UI surfaces. |
| API clients | `frontend/src/api/*` | Thin fetch wrappers for dashboard API routes. |
| Shared contracts | `frontend/src/types/dataset.ts`, `frontend/src/types/workbench.ts` | TypeScript mirrors of backend payload shapes. |
| State | `frontend/src/store/workbenchStore.ts` | Linked selection and workbench-local state. |
| Styles | `frontend/src/styles/*.css` | Split CSS bands for shell, probes, controls, inspector, metrics, and pipeline map. |

## Data Flow

```text
FastAPI dashboard
  -> frontend/src/api/*
  -> frontend/src/types/*
  -> page model helpers
  -> React panels
  -> linked selection / saved state POST routes
```

`/api/dataset` is the cheap starting payload. `/api/workbench` is the richer
manifest for axes, arrays, panel recipes, workflow presets, cohorts, runs, and
saved workspaces.

Episode pages load detail payloads lazily after an episode is selected. Media
URLs should include the dataset/media version when one is available so browser
cache behavior stays predictable.

## Capability Gating

The backend reports dataset capabilities through `/api/dataset` and richer
workbench contracts through `/api/workbench`. The frontend should use those
signals to decide which controls and panels are visible or enabled.

Expected behavior:

- Plain robot data shows episode, frame, action, and metadata views.
- Missing model internals produce clear empty states, not broken controls.
- PI0.5-specific panels appear only when the dataset declares compatible model
  sites, token spaces, attention arrays, or architecture graph data.
- Plain LeRobot roots without capability manifests stay usable through
  conservative defaults covered by `capabilityGating.test.mjs`.

## Development

Run frontend checks from the repo root through the normal environment:

```bash
scripts/check_vla_lens.sh
```

Or run focused frontend commands:

```bash
npm ci --prefix frontend
npm run lint --prefix frontend
npm run test --prefix frontend
npm run build --prefix frontend
```

Run the full local demo:

```bash
scripts/run_vla_lens_demo.sh
```

If only the React dev server is running, start a backend on the expected dev
port:

```bash
uv run python scripts/build_vla_lens_demo.py --out runs/vla_lens_demo --overwrite
uv run python scripts/serve_vla_lens_dashboard.py runs/vla_lens_demo --port 8765
```

Do not start PI0.5 capture or install capture-only dependencies for frontend
work.
