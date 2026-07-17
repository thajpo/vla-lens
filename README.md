# VLA Lens

VLA Lens packages robot episodes and model internals into one inspectable
dataset for vision-language-action research.

LeRobotDataset v3 stores the robot data. A `vla_lens/` overlay stores policy
calls, model sites, activations, tokens, probes, intervention evidence, and
dashboard state.

```text
model + environment + capture profile
-> LeRobot v3 robot data
-> VLA Lens interpretability overlay
-> local workbench and reproducible analysis artifacts
```

## Try It Without Hardware

Run the normal checks:

```bash
scripts/check_vla_lens.sh
```

Create a synthetic dataset and start the backend and frontend:

```bash
scripts/run_vla_lens_demo.sh
```

Open `http://127.0.0.1:5173/`.

The demo exercises dataset loading, APIs, and the React workbench without
PI0.5, LeRobot runtime packages, LIBERO, Torch, or a GPU.

To use Docker instead:

```bash
scripts/docker_dashboard.sh
```

Open `http://127.0.0.1:8080/`.

## Open An Existing Dataset

```bash
scripts/view_vla_lens.sh /path/to/lerobot-root
```

The path can be one LeRobot v3 dataset or a directory containing nested dataset
roots from batch capture.

## What Works

- PI0.5 capture in LIBERO through dedicated ROCm, CUDA, and MPS wrappers.
- Native and Dockerized Linux capture paths.
- LeRobot v3 robot data plus a portable interpretability overlay.
- Dataset and episode browsing with frames, actions, tokens, model sites, and
  saved internals when available.
- Probe training with metadata baselines, diagnostics, artifacts, and exact
  source navigation.
- Typed intervention targets, preflight, saved runs, controls, action-basis
  metrics, and sweep/study records.
- Deterministic PI0.5 policy-call replay gated by repeated no-op agreement.

The current live intervention is a non-claiming synthetic hook smoke. The first
artifact-derived scientific intervention is tracked in
[issue #18](https://github.com/thajpo/vla-lens/issues/18).

## Critical Environment Split

Normal development, tests, server, dashboard, and saved-artifact analysis use
the repo environment:

```bash
uv run pytest
uv run ruff check scripts src tests
cd frontend && npm test && npm run build
```

PI0.5 capture and replay use a hardware-specific environment:

```bash
scripts/setup_pi05_rocm_env.sh
scripts/setup_pi05_cuda_env.sh
scripts/setup_pi05_mps_env.sh

scripts/pi05_capture.sh --backend rocm ...
scripts/pi05_batch_capture.sh --backend cuda ...
scripts/pi05_intervene.sh --backend rocm ...
```

Do not run PI0.5, LeRobot, or LIBERO execution through the normal repo `uv run`
environment. Their hardware-specific Torch and simulator dependencies are
intentionally isolated.

## Linux Capture Containers

```bash
scripts/docker_pi05_cuda.sh --config configs/pi05_light_5_test.yaml --run
scripts/docker_pi05_rocm.sh --config configs/pi05_light_5_test.yaml --run
```

For a large run, put `--output-root` on mounted NVMe or block storage rather
than the container filesystem.

## Main Code Areas

- `src/vla_lens/traces/`: dataset and trace access.
- `src/vla_lens/dataset/`: LeRobot and overlay indexing.
- `src/vla_lens/capture/`: generic capture contracts and adapters.
- `src/vla_lens/pi05/`: PI0.5 capture, replay, and runtime integration.
- `src/vla_lens/probes/`: probe training and diagnostics.
- `src/vla_lens/interventions/`: runtime-free intervention evidence contracts.
- `src/vla_lens/server/`: local dashboard APIs.
- `frontend/`: research workbench.

## Documentation

You do not need to read every Markdown file.

- [Quickstart](docs/quickstart.md): first successful local run.
- [Current state](docs/current-state.md): what works, what does not, and the
  active issues.
- [Documentation index](docs/README.md): task-specific runbooks, system
  explanations, research records, and archive.
- [Dataset format](docs/dataset-format.md): the durable storage boundary.
- [Probe evidence](docs/probe-evidence.md): how probe results map back to source
  examples.
- [Interventions](docs/interventions.md): evidence, replay gates, controls, and
  current limitations.

Unfinished feature work belongs in GitHub issues. Temporary planning documents
are deleted before merge after their useful decisions and validation are moved
into the PR description.

## License

MIT. See [LICENSE](LICENSE).
