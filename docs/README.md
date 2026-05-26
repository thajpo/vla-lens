# VLA Lens Documentation Index

This is the current documentation entrypoint. Prefer these files over older
experiment notes when deciding what to run or implement.

## Read First

- [current-state.md](current-state.md): current repo direction, known-good
  commands, environment split, measured capture costs, and next action items.
- [dataset-format.md](dataset-format.md): canonical dataset contract:
  LeRobot v3 robot data plus the `vla_lens/` interpretability overlay.
- [hardware-run-paths.md](hardware-run-paths.md): one-command portable demo and
  backend-specific PI0.5 setup/capture paths for ROCm, CUDA, and Apple Silicon.
- [docker.md](docker.md): dashboard container usage and the split between
  dashboard and Linux CUDA/ROCm capture containers.
- [cloud-capture.md](cloud-capture.md): high-volume capture storage model,
  output-root commands, cache/secrets handling, and dashboard handoff.
- [remote-gpu-local-analysis.md](remote-gpu-local-analysis.md): rented-GPU
  capture, local hard-drive analysis, and current options for online activation
  hosting or archival.
- [model-dataset-sim-agnosticity.md](model-dataset-sim-agnosticity.md):
  target architecture for supporting multiple VLA models, robot datasets, and
  simulators through adapters and dataset capabilities.
- [pi05-capture-profiles.md](pi05-capture-profiles.md): what each PI0.5 capture
  profile is for in interpretability terms.
- [pi05-rocm-capture-env.md](pi05-rocm-capture-env.md): how the dedicated ROCm
  PI0.5/LeRobot/LIBERO environment works and why normal `uv run` capture is
  unsafe.
- [pi05_broad_1000_probe_experiments.md](pi05_broad_1000_probe_experiments.md):
  consolidated probe experiment registry, useful legacy ideas, null results,
  and replication priorities.
- [vla-lens-architecture-workflows.md](vla-lens-architecture-workflows.md):
  architecture and workflow contracts.
- [research_ui_principles.md](research_ui_principles.md): design principles for
  research-facing UI and causal-evidence displays.
- [library/pi05-lens.md](library/pi05-lens.md): PI0.5 library notes and
  reusable analysis primitives.

## Status Labels

Use these labels when adding or editing docs:

```text
active:
  operational guidance or current roadmap truth.

living:
  updated as work validates or invalidates assumptions.

historical:
  useful result or planning context, but not current run guidance.

superseded:
  preserved only because it explains how we got here. Do not follow commands
  without checking current-state.md first.

archive:
  old project direction kept for memory, not active VLA Lens work.
```

## Important Rule

PI0.5 capture must use the hardware wrapper scripts:

```bash
scripts/pi05_capture.sh --backend rocm ...
scripts/pi05_batch_capture.sh --backend cuda ...
scripts/pi05_batch_capture_mps.sh ...
```

Do not run PI0.5/LeRobot/LIBERO capture through plain `uv run
vla-pi05-capture` or `uv run vla-pi05-batch-capture` in the normal repo
environment.

## Historical Experiment Notes

Old per-experiment markdown files and CogACT-era planning docs were removed
after their useful probe-design ideas were consolidated into
[pi05_broad_1000_probe_experiments.md](pi05_broad_1000_probe_experiments.md).
