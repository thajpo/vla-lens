# PI0.5 Action-Stream Patching

This workflow follows a scene-driven action change from PI0.5's visual prefix
into its action expert. It reuses matched counterfactual scenes and the
resumable patch-study runner.

## What Is Patched

PI0.5 builds the visual/language prefix once, then runs an 18-layer action
expert at every denoising step. The action expert has 50 positions, one for each
action-horizon slot.

VLA Lens names the supported live sites explicitly:

```text
pi05.vlm.layers.{layer}.prefix.hidden_tokens
  axes: token x channel

pi05.expert.layers.{layer}.by_step.hidden_tokens
  axes: denoising step x action position x channel
```

Expert patches align donor and recipient values by both denoising step and
action position. A full expert-layer patch therefore replaces 50 positions at
each model denoising step; it does not flatten or average either axis.

The visual-prefix and expert hooks have different valid call counts. A visual
prefix layer runs once per action generation. An expert layer runs once per
denoising step, currently ten times. The runtime records both the expected and
observed counts and rejects a trial if they disagree.

## Fast Path

The planner and analysis are lightweight and run in the normal repo
environment. Only live PI0.5 execution uses the capture environment.

```bash
DATASET_ROOT=/mnt/new-volume/vla-lens/rq019-pose-exchange-pilot
PAIR_JSON=$DATASET_ROOT/vla_lens/counterfactual_pairs/rq019_pose_exchange_pairs.json
JOB_JSON=$DATASET_ROOT/vla_lens/study_jobs/rq022_expert_action_localization.json

# Rebuild pair metadata from existing traces. With complete traces this skips
# capture, loads no model, and adds the reusable action-position scopes.
PI05_VENV=/home/j/Projects/vla-lens/.venv-pi05-rocm \
  scripts/pi05_pose_exchange_capture.sh --backend rocm \
  "$DATASET_ROOT" --job configs/capture/rq019_pose_exchange_pilot.json \
  --run-capture

# Build a six-layer, full-action-suffix study. The default expert layers are
# 0, 4, 8, 12, 16, and 17; the default scope is all 50 action positions.
uv run vla-pi05-pose-exchange-study \
  --pairs "$PAIR_JSON" --output "$JOB_JSON" \
  --study-id rq022-expert-action-localization \
  --phase localization --stream expert_action

# Inspect every trial and preflight without loading PI0.5.
PI05_VENV=/home/j/Projects/vla-lens/.venv-pi05-rocm \
  scripts/pi05_patch_study.sh --backend rocm "$DATASET_ROOT" \
  --study "$JOB_JSON"

# Execute after inspection. Hidden states stay in memory and are captured for
# all requested layers in one donor generation per matched scene.
PI05_VENV=/home/j/Projects/vla-lens/.venv-pi05-rocm \
  scripts/pi05_patch_study.sh --backend rocm "$DATASET_ROOT" \
  --study "$JOB_JSON" --run-study \
  --max-noop-l2 0.000001 --max-noop-max-abs 0.000001
```

The runner checkpoints every completed trial. Repeating the same command skips
finished work. Add `--retry-failed` only after fixing the recorded failure.

Rebuild the analysis from saved actions without loading PI0.5:

```bash
uv run vla-patch-study-analyze \
  "$DATASET_ROOT/vla_lens/patch_studies/rq022-expert-action-localization"
```

The Intervention page discovers the resulting `analysis.json` automatically.
It shows visual-prefix and action-expert studies in the same explorer and names
any selected denoising-step range.

## Ready-Made Scopes

The pair manifest provides these action-position scopes:

- `action_all`: all 50 horizon positions;
- `action_first_10`: positions 0-9;
- `action_middle_10`: positions 20-29;
- `action_last_10`: positions 40-49.

Use `--token-regions action_first_10,action_middle_10,action_last_10` to compare
where in the predicted action chunk a transferred state matters.

Use `--generation-steps all`, a comma-separated list such as `0,1,2`, or a
half-open range such as `0:5` to control which denoising iterations receive the
patch. Action positions and denoising steps are different axes.

## Confirmation Controls

`--phase confirmation --stream expert_action` automatically runs:

- recipient self-patch;
- donor self-patch;
- zero-strength donor patch;
- shuffled donor action positions;
- random values matched to the donor-patch norm.

The layer sweep itself supplies the wrong-layer comparison. A result is called
specific only when the intended patch moves toward the shared-noise donor
action and beats the strongest measured negative control.

## Saved Versus Temporary Data

Saved permanently:

- the exact study job and runtime-site names;
- recipient, donor, patched, and control action chunks;
- token and denoising-step selectors;
- hook counts, hashes, failures, and decisions;
- pair-level bootstrap summaries and 95% intervals.

Kept only in memory:

- donor and recipient hidden states.

This keeps studies small while preserving everything needed to reconstruct a
hidden-state cache from the source traces and exact plan.

## How To Read The RQ-022 Result

Across the five pose-exchange scenes, full action-state patches transferred
`-93.48%`, `-93.25%`, `-86.01%`, `-34.03%`, `99.45%`, and `99.9978%` at expert
layers 0, 4, 8, 12, 16, and 17. The aligned layer-16 patch beat shuffled action
positions (`30.04%`) and norm-matched random values (`2.77%`). Self and
zero-strength patches were exact no-ops.

This localizes the donor-directed action state late in the expert. It does not
localize an object representation: all 50 high-dimensional action positions
were replaced. Negative early-layer transfer also does not mean those layers
lack scene information; a whole-state transplant can put the recipient on an
incompatible computation path.

Patching denoising steps 0-2, 3-6, or 7-9 alone transferred about `4.1%`,
`13.6%`, or `29.7%`. These values are not additive. The practical reading is
that the donor trajectory must be followed throughout refinement and that its
scene-conditioned correction is strongest late.

## VLM-To-Expert Timing Caveat

The existing visual-prefix hook modifies a decoder layer's output after that
same layer has already written its key/value tensors into the prefix cache.
Consequently, VLM layer 16 and expert layer 16 are not the two sides of one
matched causal boundary. Comparing their transfer values is useful, but it
does not identify the exact handoff layer.

The next bridge experiment should patch the prefix key/value cache consumed by
selected expert layers while keeping key/value, head, token, and rotary-position
alignment explicit. Do not patch keys across different token positions without
handling their positional rotation.
