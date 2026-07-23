# Probe Run Artifacts

Status: active guidance.

Last updated: July 20, 2026.

A trained probe should answer two different questions:

1. What experiment did we run?
2. Can we use and check the fitted probe without training it again?

Probe artifact schema 4 answers both. The small experiment card is for the
researcher. The complete saved contract is for replay, later analysis, and
tools built on top of the probe.

## What The Researcher Sees

The experiment card groups information by importance:

- Choices that change the claim: target, model input, token handling, held-out
  split, baselines, and evaluation units.
- Method choices: probe model and sweep axes.
- Execution details: row count, feature size, and cache behavior.

This is the information an agent should summarize when asked to train a probe.
The card avoids making file layout and library settings look like research
decisions, while the artifact still saves those details.

## What Is Saved

Each new generic probe artifact contains:

- the complete normalized training recipe;
- the selector and fingerprints for every source trace;
- a lightweight table of all selected model-site rows;
- the exact rows used by the chosen readout, in prediction order;
- the fitted standardizer and model arrays at their original numeric precision,
  with content fingerprints checked before reuse;
- row-level predictions, metrics, baselines, and null-test settings;
- claim-grouped confidence intervals for each evaluation split;
- a trained, group-aware shuffled-label control for the validation-selected readout;
- `candidate_results.parquet`, comparing every trained layer/model candidate,
  its validation rank, baseline, and held-out score;
- identifiers for external label artifacts used by the dataset, when present.

The artifact does not contain another copy of the activation matrix. Those
tensors are usually the large part. VLA Lens keeps them in the original capture
and may also keep a removable feature cache for speed.

This gives us the useful storage tradeoff:

- repeated analysis is fast while the cache exists;
- deleting the cache saves disk space;
- the artifact remains understandable after cache deletion;
- replay can rebuild features from the capture and verify source, feature, and
  fitted-model fingerprints.

Deleting the original capture is different: replay needs that source data.

## Commands

Inspect the research setup:

```bash
uv run python scripts/use_vla_lens_probe.py \
  /path/to/dataset ARTIFACT_ID explain
```

Rebuild the features and compare predictions:

```bash
uv run python scripts/use_vla_lens_probe.py \
  /path/to/dataset ARTIFACT_ID replay
```

Apply the fitted probe to a compatible two-dimensional NumPy feature matrix:

```bash
uv run python scripts/use_vla_lens_probe.py \
  /path/to/dataset ARTIFACT_ID use \
  --features features.npy \
  --output predictions.npy
```

Use `explain --format json` to list the best retained candidate from each model
family, then pass `--readout READOUT_ID` to apply one of those alternatives.
Only the validation-selected readout supports exact replay from the capture;
alternative retained readouts support direct use with compatible features.

Replay checks classification labels exactly. For regression, it records and
uses explicit absolute and relative tolerances based on numeric precision,
feature dimension, and prediction scale. The combined allowance is capped at
0.01% so low-precision or very wide inputs cannot turn replay into a weak check.
Replay reports both tolerances and the largest observed difference.

## Older Artifacts

Artifacts written before schema 4 do not contain the replay contract. VLA Lens
still loads and explains the metadata they have. It does not claim that they
can be replayed or reused when the fitted state or exact rows are missing.
Probe-run contract v1 artifacts from early development builds are handled the
same way because they predate fitted-array integrity checks. New replayable
artifacts use probe-run contract v2.

## Default Readouts

Unless a spec explicitly narrows the choice, the generic trainer fits a linear
probe and the standard small MLP at every declared sweep position. Validation
selects the replayable readout across model and sweep choices. The artifact
keeps the complete metric comparison. Exact capture replay rows are retained
for the selected readout. Fitted state is also retained for the best validation
candidate from each model family, so a researcher can apply and compare the
linear and MLP alternatives without retraining them.

The final test split is never a valid selection split. If a dataset has only
train and test labels, the trainer creates a deterministic validation split from
whole training groups. The group follows the claim: whole tasks for a
held-out-task split, whole objects for held-out-object work, and whole episodes
for episode generalization. An explicitly test-selected spec fails before
training.

Confidence intervals use that same group, and saved predictions include the
winning baseline's row-level predictions. This permits paired intervals for the
probe's improvement over the baseline instead of comparing unrelated headline
scores.

The shuffled-label comparison records its p-value resolution explicitly. With
20 refits, for example, the smallest possible value is `1/21 = 0.0476`; this is
an exploratory control, not precise evidence for a significance threshold. The
paired effect and its group-level interval are the main result.

## Representation Contract

Probe specs name the representation constructed before fitting. The generic
trainer supports average-token vectors, flattened token-position vectors, and
already-vector inputs. Preflight also reports richer data-supported options,
including shared token readouts, learned layer mixtures, object-conditioned
probes, and set decoders. Richer requests fail clearly until their specialized
runner is wired into the common study contract; they are never silently
replaced by mean pooling.
