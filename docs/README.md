# VLA Lens Documentation Index

This is the current documentation entrypoint. Prefer these files over older
experiment notes when deciding what to run or implement.

## Read First

- [current-state.md](current-state.md): current repo direction, known-good
  commands, environment split, measured capture costs, and next action items.
- [pi05-capture-profiles.md](pi05-capture-profiles.md): what each PI0.5 capture
  profile is for in interpretability terms.
- [pi05-rocm-capture-env.md](pi05-rocm-capture-env.md): how the dedicated ROCm
  PI0.5/LeRobot/LIBERO environment works and why normal `uv run` capture is
  unsafe.
- [experiments/pi05-vla-lens-roadmap.md](experiments/pi05-vla-lens-roadmap.md):
  living research/product roadmap. The top sections are current; deeper sections
  may preserve historical implementation prompts for context.

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

PI0.5 capture must use the ROCm wrapper scripts:

```bash
scripts/pi05_capture_rocm.sh ...
scripts/pi05_batch_capture_rocm.sh ...
```

Do not run PI0.5/LeRobot/LIBERO capture through plain `uv run
vla-pi05-capture` or `uv run vla-pi05-batch-capture` on this workstation.

## Historical Experiment Notes

Most files under [experiments/](experiments/) are snapshots from specific
analysis moments. They can contain old paths, old hypotheses, or old commands.
Treat them as evidence/history unless they are explicitly referenced from
[current-state.md](current-state.md) or the top of the living roadmap.

Old CogACT-era planning files have been moved to
[archive/legacy-cogact/](archive/legacy-cogact/).
