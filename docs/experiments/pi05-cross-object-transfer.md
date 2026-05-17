# PI0.5 Cross-Object Transfer

## Question

Do sparse causal handoff tokens discovered for one object task transfer to another object task?

This tests whether the hot sites are reusable target-binding features or task/object-specific features.

## Setup

Task61 native feature:

- Task: `61`
- Target: `chocolate_pudding_1`
- Wrong objects: `akita_black_bowl_1`, `akita_black_bowl_2`
- Site: layer `14`, value cache
- Tokens: `331,327,323,347,330,339,328,336`

Task60 native feature:

- Task: `60`
- Target: `akita_black_bowl_1`
- Wrong object: `akita_black_bowl_2`
- Site: layer `12`, value cache
- Tokens from exact scan: `146,145,130,129,147,144,131,128`

## Task61 Tokens Transferred To Task60

Output:

- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_transfer_task61_tokens_to_task60/scene4_attribution_patch_token_role_test.csv`

| intervention | rows | recipients | donors | mean delta | positive rate | patched mean margin |
|---|---:|---:|---:|---:|---:|---:|
| `success_to_failure` | 12 | 4 | 3 | 0.0776 | 1.00 | -1.3198 |
| `failure_to_success` | 12 | 4 | 3 | 0.0028 | 0.33 | 1.5377 |
| `failure_zero` | 12 | 4 | 3 | -0.0060 | 0.50 | -1.4034 |
| `failure_bin_mean` | 12 | 4 | 3 | -0.0073 | 0.25 | -1.4047 |
| `failure_layer_mean` | 12 | 4 | 3 | -0.0196 | 0.25 | -1.4169 |

Interpretation: task61 tokens weakly but consistently help task60 under donor patching. The effect is much smaller than task61 native rescue and does not come close to flipping the very negative task60 baseline margin.

## Task60 Native Token Scan

Output:

- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_task60_token_scan_value_bin04/scene4_attribution_patch_exact_tokens_summary.csv`

Top layer-12 value tokens inside `vision_bin_04_of_24`:

| token | camera | row | col | exact delta | positive rate |
|---:|---|---:|---:|---:|---:|
| 146 | `image` | 9 | 2 | 0.0357 | 1.00 |
| 145 | `image` | 9 | 1 | 0.0269 | 1.00 |
| 130 | `image` | 8 | 2 | 0.0210 | 1.00 |
| 129 | `image` | 8 | 1 | 0.0112 | 1.00 |
| 147 | `image` | 9 | 3 | 0.0102 | 1.00 |
| 144 | `image` | 9 | 0 | 0.0054 | 1.00 |
| 131 | `image` | 8 | 3 | 0.0053 | 1.00 |
| 128 | `image` | 8 | 0 | 0.0011 | 1.00 |

Task60 native atlas:

- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_hot_token_atlas_task60/`

The task60 tokens are in first camera `observation.images.image`, rows `8-9`, columns `0-4`. In the atlas, these positions sit around the left/lower visual region near the target-side workspace and robot/table context, not in the same camera or region as task61 tokens.

## Task60 Native Tokens On Task60

Output:

- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_task60_native_value_tokens_role/scene4_attribution_patch_token_role_test.csv`

| intervention | rows | recipients | donors | mean delta | positive rate | patched mean margin |
|---|---:|---:|---:|---:|---:|---:|
| `success_to_failure` | 12 | 4 | 3 | 0.0696 | 1.00 | -1.3278 |
| `failure_layer_mean` | 12 | 4 | 3 | 0.0677 | 1.00 | -1.3297 |
| `failure_bin_mean` | 12 | 4 | 3 | 0.0596 | 1.00 | -1.3378 |
| `failure_zero` | 12 | 4 | 3 | 0.0594 | 1.00 | -1.3380 |
| `failure_to_success` | 12 | 4 | 3 | -0.0062 | 0.00 | 1.5287 |

Interpretation: task60 differs from task61. Neutralizing the bad task60 tokens helps almost as much as donor patching, which is more compatible with a removable bad/attractor feature or local interference feature. Task61 did not show this pattern; there, neutral ablations were tiny and inconsistent.

## Task60 Tokens Transferred To Task61

Output:

- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_transfer_task60_tokens_to_task61/scene4_attribution_patch_token_role_test.csv`

| intervention | rows | recipients | donors | mean delta | positive rate | patched mean margin |
|---|---:|---:|---:|---:|---:|---:|
| `success_to_failure` | 6 | 2 | 3 | 0.0023 | 1.00 | -0.2718 |
| `failure_bin_mean` | 6 | 2 | 3 | 0.0019 | 1.00 | -0.2723 |
| `failure_to_success` | 6 | 2 | 3 | 0.0003 | 0.50 | 0.6016 |
| `failure_zero` | 6 | 2 | 3 | 0.0003 | 0.50 | -0.2739 |
| `failure_layer_mean` | 6 | 2 | 3 | 0.0002 | 0.50 | -0.2740 |

Interpretation: task60 native tokens do not transfer meaningfully to task61.

## Summary Finding

The mechanism type transfers better than exact token identities.

- Task61 tokens have a weak positive effect on task60, but not enough to rescue the task60 margin.
- Task60 native tokens do not meaningfully help task61.
- Task60 native tokens behave differently from task61 native tokens: neutralizing them helps, suggesting task60 may involve a removable local interference signal, while task61 looks more like a missing success-feature injection site.

## Confounds

- Task60 baseline margins are much more negative than task61, so equal-sized token patches may be insufficient even if they are directionally useful.
- Task60 and task61 use different targets and different wrong-object relations; task60 is bowl-vs-bowl, while task61 is pudding-vs-bowl.
- Donor/recipient calls differ in timestep and robot pose; transfer may be affected by trajectory phase, not only target identity.
- Token addresses differ by camera and spatial region, so transfer tests confound object class with camera/viewpoint and geometry.
- Current tests are offline action-chunk interventions, not closed-loop rollout success.
- The atlas is visual/manual grounding, not segmentation-verified object-under-token labeling.

## Next Decision

The most useful next test is feature-ID, not more transfer sweeps: determine whether task60's removable feature and task61's success-injection feature correlate with action direction, target/wrong geometry, camera visibility, or phase.
