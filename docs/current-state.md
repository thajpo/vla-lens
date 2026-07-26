# Current State

Last updated: July 26, 2026.

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
- Require repeated no-op action agreement before a claim-eligible hook.
- Reconstruct an accepted object-region probe direction in raw PI0.5 hidden
  space and compare intended, random, wrong-identity, and wrong-region changes.
- Capture matched pose-exchange scenes and rerun donor and recipient actions
  under the same saved flow noise.
- Run resumable visual-prefix and action-expert patch studies with in-memory
  donor caches, permanent action arrays, controls, hashes, and recomputable
  summaries.
- Audit a cross-method autonomous research plan and print a stable content
  fingerprint plus a compact human or JSON summary.

## What Does Not Work Yet

The repository now contains real probe-direction and donor-patching results,
but no narrow intervention has shown a confirmed semantic or behavioral
mechanism. Broad visual replacement and near-output whole-action replacement
are useful dependency and plumbing results, not object localization.

The dashboard cannot launch the live PI0.5 runtime. It can inspect saved patch
studies, but the generic campaign-level plan, audit, claim, and generated result
envelope is not yet wired into research-run records or the UI.

One-factor scene mutation capture, controller-level physical action comparison,
fixed-current-action-state denoising analysis, and exact prefix key/value-cache
patching still need specialized runners.

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

Broad pooled probes did not reveal a reliable object-position or pose code.
Known simulator regions contain exploratory object-identity information, but
the corresponding probe direction failed semantic-specificity controls. A
two-object pose exchange caused a large normalized open-loop action difference;
replacing nearly all early visual context reproduced that difference, and
replacing whole late expert action states forced donor-like output. These do not
show correct behavior, object binding, or a small semantic mechanism.

The next campaign therefore starts with untouched rollout competence and
physical scene-to-behavior measurements. It then compares scene contexts at the
same denoising time and the exact same current action guess before attempting a
narrow intervention. See the [research log](../RESEARCH.md) and
[autonomous campaign protocol](autonomous-research-campaigns.md).

## Immediate Priorities

The full active backlog is in GitHub. The immediate priorities are:

- [#37: controlled scene-to-behavior research program](https://github.com/thajpo/vla-lens/issues/37)
- [#21: tiered storage and managed feature caches](https://github.com/thajpo/vla-lens/issues/21)

The exact next action in #37 is to prepare and independently audit the locked
FOUNDATION child: deterministic task-object-family parsing, candidate and
rejection tables, the 72-rollout seed-separated manifest, checkpoint and
environment receipts, and runner config. It is not ready for hardware execution
yet. [#20: broad intervention sweeps](https://github.com/thajpo/vla-lens/issues/20)
and [#36: prefix key/value patching](https://github.com/thajpo/vla-lens/issues/36)
are conditional on a factor-specific behavior branch surviving prospective
confirmation; they are not immediate work.

There are currently zero temporary implementation specs. When an issue is
selected, its body becomes the plan unless cross-cutting architecture or a
migration genuinely needs a short-lived repo-local document.
