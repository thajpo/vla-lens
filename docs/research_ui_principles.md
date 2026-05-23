# Research UI Principles

VLA-lens UI should help a researcher move from an observed episode behavior to a testable mechanistic hypothesis.

## Default Surface

- Keep controls, selections, comparisons, and drilldowns visible.
- Hide passive metadata behind disclosures unless it changes a decision.
- Prefer split-aware trust language: train, validation, test, or split missing.
- Treat train-episode probe evidence as a sanity check, not as generalization evidence.
- Design probe views for hundreds or thousands of probes: search, rank, filter, and summarize before listing raw rows.

## Research Loop

The main loop is:

1. Find an interesting behavior or cohort.
2. Select the artifact that claims to explain it.
3. Check whether that claim holds on validation/test episodes.
4. Jump from cohort evidence to the episode microscope.
5. Inspect the mapped model site, layer, call, feature, token, or attention patch.
6. Turn the observation into a patching, steering, or counterfactual test.

## Probe Evidence

Probe UI should answer:

- What question is this probe answering?
- Is this episode train, validation, test, or split missing for the probe?
- Where in the model is the probe reading from?
- Which episodes are most worth inspecting because the probe is wrong, confidently wrong, heldout, or ambiguous?
- Does the probe support a research claim, or is it only a debugging sanity check?

Avoid spending default space on raw tensor shapes, dtype, exact float storage details, or static bookkeeping unless they directly explain what a researcher can do next.

## Counterfactual Direction

Counterfactual artifacts should record the source episode, intervention site, intervention time/call, method, replacement or steering source, resulting trajectory, and comparison metrics. The UI should show original and intervened episodes side by side with synchronized timeline, changed probe readouts, and changed object/action behavior.
