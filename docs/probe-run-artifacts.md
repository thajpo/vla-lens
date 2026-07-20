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
- the fitted standardizer and model arrays at their original numeric precision;
- row-level predictions, metrics, baselines, and null-test settings;
- a clear statement when confidence intervals were not calculated;
- identifiers for external label artifacts used by the dataset, when present.

The artifact does not contain another copy of the activation matrix. Those
tensors are usually the large part. VLA Lens keeps them in the original capture
and may also keep a removable feature cache for speed.

This gives us the useful storage tradeoff:

- repeated analysis is fast while the cache exists;
- deleting the cache saves disk space;
- the artifact remains understandable after cache deletion;
- replay can rebuild features from the capture and verify their fingerprints.

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

Replay checks classification labels exactly. For regression, it records and
uses a small absolute tolerance based on the fitted model's numeric precision,
then reports the largest observed difference. This accounts for machine-level
rounding differences between numerical libraries without accepting a changed
prediction.

## Older Artifacts

Artifacts written before schema 4 do not contain the replay contract. VLA Lens
still loads and explains the metadata they have. It does not claim that they
can be replayed or reused when the fitted state or exact rows are missing.
