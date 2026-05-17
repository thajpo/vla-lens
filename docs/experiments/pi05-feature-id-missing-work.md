# PI0.5 Feature ID Missing Work

## Current Working Label

The task61 site is best described as a sparse success-feature injection site in the VLM-to-action handoff, not as a plain visual object patch.

## Atlas Status

I generated a task61 hot-token atlas at:

- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_hot_token_atlas_task61/`

The atlas shows that the top task61 tokens are spatially in `observation.images.image2` rows `4-5` and usually fall on/near bowl-like distractors or scene context, not directly on the chocolate-pudding package. This supports treating the site as a contextual handoff feature rather than target pixels.

Cross-object transfer is logged at:

- `docs/experiments/pi05-cross-object-transfer.md`

The transfer result is weak/asymmetric. Task61 tokens weakly help task60, task60 tokens do not meaningfully help task61, and task60 native tokens show a different role-test pattern where neutralizing bad tokens helps almost as much as donor patching.

## What We Have Not Identified

- Whether the feature encodes object identity, geometry, camera confidence, action direction, or trajectory phase.
- Whether any exact hot tokens strongly transfer across object classes; the first task60/task61 transfer check was weak/asymmetric.
- Whether the feature improves live rollouts, not only offline action chunks.
- Whether masking target or distractor pixels moves the activation in the same direction as donor patching.
- Whether the feature is produced by VLM attention routing, MLP computation, or earlier camera fusion.
- What object lies under each token in an automated way; current atlas grounding is visual/manual, not segmentation-verified.

## Confounds Still Open

- Donor and recipient runs can differ in scene geometry, camera visibility, robot pose, and timestep.
- The hot layer-14 token location is a spatial address after many nonlinear layers, not direct evidence about the pixels under that patch.
- Current flow trace used three donor pairings for one bad recipient, so it does not establish recipient-diverse trajectory generality.
- Current metric is target-vs-wrong action margin, not physical task success.
- Existing captures do not include enough per-denoising-step expert hidden states for deeper expert-side tracing.

## Best Next Tests

- Cross-object transfer: patch task61 tokens into task60 and task60 tokens into task61.
- Task60 token scan and role test on layer-12 `vision_bin_04_of_24` key/value features.
- Masking response: mask target, wrong object, both bowls, and matched background controls; measure both hot activation movement and action movement.
- Cumulative flow trace across distinct bad recipients, not only one recipient with multiple donors.
- Live rollout patching or intervention replay to test whether the offline correction survives closed-loop execution.
