# Current State

Last updated: July 16, 2026.

## What VLA Lens Is

VLA Lens is an episode-grounded interpretability workbench for
vision-language-action models.

LeRobotDataset v3 stores robot episodes, observations, actions, tasks, and
video. The `vla_lens/` overlay stores policy-call alignment, model internals,
tokens, probes, artifacts, fingerprints, and saved workbench evidence.

The current concrete runtime is PI0.5 in LIBERO. The storage and dashboard core
are designed to support other models and environments through adapters and
capabilities.

## What Works

- Create valid LeRobot v3 plus VLA Lens datasets through native or Dockerized
  PI0.5 capture environments.
- Open one dataset or a directory containing nested datasets.
- Browse episodes, frames, actions, tokens, model sites, activations, attention,
  and action-generation traces when captured.
- Train probe suites with metadata baselines and save exact artifacts.
- Inspect high, low, uncertain, and failure examples in dataset and episode
  context.
- Preserve probe selection and evidence pins across navigation.
- Seed typed intervention targets from probe and Episode Lens evidence.
- Preflight, save, reopen, and aggregate intervention records without loading
  PI0.5 or GPU dependencies.
- Replay a selected PI0.5 policy call with the stored observation and initial
  flow noise.
- Require repeated no-op action agreement before running the current synthetic
  hook smoke.

## What Does Not Work Yet

The live intervention is not yet a scientific result. It applies a synthetic
one-hot direction at `pi05.action_head.input`, records a matched random control,
and deliberately sets `claim_eligible = false`.

The dashboard cannot yet launch that runtime or render a complete
stored-original/no-op/intervened/control action comparison.

Dataset-wide policy-call identity, exact reusable example manifests, and a
method-independent experiment recipe are not yet first-class contracts.

The generic adapter architecture has synthetic compliance coverage, but no real
second model or second environment integration yet.

## Environment Split

Normal development, tests, server, dashboard, and saved-artifact analysis use
the normal repo environment:

```bash
scripts/check_vla_lens.sh
uv run pytest
uv run ruff check scripts src tests
```

PI0.5 capture and replay use a dedicated hardware environment:

```bash
scripts/setup_pi05_rocm_env.sh
scripts/check_pi05_env.sh --backend rocm
scripts/pi05_batch_capture.sh --backend rocm ...
scripts/pi05_intervene.sh --backend rocm ...
```

Do not execute PI0.5, LeRobot, or LIBERO through the normal `uv run`
environment. See [hardware run paths](hardware-run-paths.md) for commands and
[PI0.5 capture profiles](pi05-capture-profiles.md) for capture cost and purpose.

## Hardware Evidence

ROCm capture is validated on a real machine. The May 2026 Docker smoke wrote a
valid 520-step LeRobot v3 dataset with 11 policy calls and a readable VLA Lens
overlay.

The July 2026 replay smoke reproduced a selected stored action exactly across
three no-op replays, passed a zero-tolerance gate, and saved the synthetic
intervention plus random-direction control.

CUDA container plumbing exists, but this repository does not yet record a CUDA
hardware smoke. MPS capture and replay also remain unverified.

## Research Conclusion

The broad pooled binary PI0.5 probes did not beat stronger metadata-only
baselines reliably enough to justify intervention. The geometry campaign was
mostly negative or diagnostic. Object-local `z` may justify one controlled
methodological confirmation, but it is not currently an intervention target.

See [Probe evidence](probe-evidence.md) for interpretation and the
[broad-1000 research record](pi05_broad_1000_probe_experiments.md) for the full
results.

## Active Work

The active backlog is in GitHub:

- [#14: dataset-level policy-call index](https://github.com/thajpo/vla-lens/issues/14)
- [#15: exact example manifests](https://github.com/thajpo/vla-lens/issues/15)
- [#16: reusable experiment recipes](https://github.com/thajpo/vla-lens/issues/16)
- [#17: unified selection and source-example drilldowns](https://github.com/thajpo/vla-lens/issues/17)
- [#18: first claim-eligible PI0.5 intervention](https://github.com/thajpo/vla-lens/issues/18)
- [#19: live Intervention Lab comparison](https://github.com/thajpo/vla-lens/issues/19)
- [#20: sweep and cohort execution](https://github.com/thajpo/vla-lens/issues/20)

There are currently zero temporary implementation specs. When an issue is
selected, its body becomes the plan unless cross-cutting architecture or a
migration genuinely needs a short-lived repo-local document.
