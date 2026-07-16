# VLA Lens Quickstart

Use this as the first operational document after checkout.
Then review [`vla-lens-architecture-workflows.md`](vla-lens-architecture-workflows.md) and [`current-state.md`](current-state.md) before code changes.

## 1) Choose the workflow

- **Normal workflow (default):** non-hardware development, checks, dashboard/API work, and analysis logic.
  - Environment: `.venv`
  - Commands: `scripts/check_vla_lens.sh`, `uv run pytest`.

- **PI0.5 capture workflow (hardware + CUDA/ROCm/LIBERO):** model and simulator capture on a hardware-specific runtime.
  - Environment: `.venv-pi05-rocm` / `.venv-pi05-cuda` / `.venv-pi05-mps`
  - Commands: `scripts/check_pi05_env.sh`, `scripts/pi05_capture.sh`, `scripts/pi05_batch_capture.sh`.

This split is required for reproducible environments.

## 2) Sanity check without hardware

From repo root:

```bash
scripts/check_vla_lens.sh
```

Then generate and open a synthetic demo dataset:

```bash
scripts/run_vla_lens_demo.sh
```

Open the local dashboard at:

- `http://127.0.0.1:5173/` (local dev)

The separate Docker dashboard path serves at `http://127.0.0.1:8080/`; see
[`docker.md`](docker.md).

This validates package bootstrap, API contracts, and workbench rendering.

## 3) Canonical execution flow

For most changes, use this sequence:

1. **Command/Script** starts the flow (capture CLI, dataset report, backend startup, UI action).
2. **`vla_lens` package** dispatches by module domain: `capture/`, `pi05/`, `traces.py`, `server/`, `frontend`.
3. **Artifacts** are persisted as:
   - LeRobot v3 robot data (`data/`, `videos/`, `meta/`)
   - VLA Lens overlay artifacts under `vla_lens/`.
4. **Server/API** reads artifact metadata and serves typed endpoints.
5. **Workbench UI** renders episode selection, token metadata, activations, probes, and evidence.

Keep this chain in mind while inspecting behavior.

## 4) File map

- `README.md`: project purpose and operational entrypoint references.
- `docs/current-state.md`: active state and known-good commands.
- `docs/vla-lens-architecture-workflows.md`: architecture and workflow contracts.
- `docs/dashboard-api.md`: endpoint conventions and payload contracts.
- `src/vla_lens/`:
  - `capture/`: capture contracts and adapters.
  - `pi05/`: PI0.5-specific capture/replay/intervention code.
  - `traces.py`: dataset indexing and artifact access.
  - `selectors.py`: activation-feature selectors and cache behavior.
  - `artifacts.py`: artifact persistence and provenance.
  - `server/`: backend API routes.
- `scripts/`: command surfaces for both normal and PI0.5 workflows.
- `tests/`: expected behavior and regression coverage.

## 5) PI0.5 capture path

Set up capture environment once:

```bash
scripts/setup_pi05_rocm_env.sh
scripts/setup_pi05_cuda_env.sh
scripts/setup_pi05_mps_env.sh
```

Validate environment before capture:

```bash
scripts/check_pi05_env.sh --backend rocm
```

Run a single capture:

```bash
scripts/pi05_capture.sh --backend rocm --capture-profile mechanistic_sampled ...
```

Run batch capture:

```bash
scripts/pi05_batch_capture.sh --backend rocm --config configs/pi05_light_5_test.yaml --run
```

Run in Linux containers when required:

```bash
scripts/docker_pi05_rocm.sh --config configs/pi05_light_5_test.yaml --run
```

## 6) Failure triage

- Start from the smallest signal:
  - command output
  - argument parsing
  - validation output from `scripts/check_pi05_env.sh` or `scripts/check_vla_lens.sh`
  - test failure trace

- API/UI mismatch: `tests/server_api_test.py`, `tests/fastapi_server_test.py`.
- Capture metadata/schema mismatch: `tests/lerobot_v3_contract_test.py`, `tests/pi05_*_test.py`, `tests/refactor_contract_test.py`.
- Data integrity issues: `tests/serve_vla_lens_app_test.py`, `tests/vla_lens_trace_*`.

## 7) Recommended first maintenance cycle

Pick one subsystem and one focused cycle:

- Subsystem: `scripts`, `capture`, `server`, or `frontend`.
- Add or improve one contract test.
- Make one targeted naming/structure cleanup.
- Re-run `scripts/check_vla_lens.sh` for the affected area.
