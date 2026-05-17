# PI0.5 Scene 4 Causal Trace

## Question

Can replacing saved VLM-to-action interface state from a good run shift a bad run's offline action chunk toward the requested object?

This is an automated causal tracing test over saved PI0.5 prefix KV-cache layers. It does not run live robot rollouts.

## Setup

- Scene family: `living_room_scene_4`
- Task id: `61`
- Target object: `chocolate_pudding_1`
- Wrong objects: `akita_black_bowl_1`, `akita_black_bowl_2`
- Pairs: 6 donor-recipient pairs
- Layers: `0,4,8,12,16,17`
- Trace rows: 192
- Patch rows: 180

Outputs:

- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_causal_trace/scene4_replay_check.csv`
- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_causal_trace/scene4_interface_causal_trace.csv`
- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_causal_trace/scene4_interface_causal_trace_gate_assessment.csv`
- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_causal_trace/scene4_interface_causal_trace_pair_layer_gate.csv`

## Replay Gate

Replay passed exactly for all unique calls.

| rows checked | rows passed | max flow diff | max action diff |
|---:|---:|---:|---:|
| 5 | 5 | 0.0 | 0.0 |

This means the saved KV cache, saved noise, and offline denoising loop reproduce the captured action chunks for the selected calls.

## Main Result

Mean target-vs-wrong-object action margin improved most when patching layers 8, 12, and 16.

| layer | rescue delta | best control delta | rescue advantage | pass |
|---:|---:|---:|---:|---|
| 0 | 0.0046 | 0.0046 | -0.0001 | False |
| 4 | 0.0078 | 0.0099 | -0.0020 | False |
| 8 | 0.0226 | 0.0072 | 0.0154 | True |
| 12 | 0.1243 | 0.0000 | 0.1243 | True |
| 16 | 0.0184 | 0.0046 | 0.0138 | True |
| 17 | 0.0589 | 0.0617 | -0.0028 | False |

Layers 8 and 12 passed the control gate for every donor-recipient pair. Layer 16 passed overall, but not for the second recipient family.

## Recipient-Level Consistency

| recipient | wrong object | robust layers |
|---|---|---|
| `fef9933d` | `akita_black_bowl_1` | 4, 8, 12, 16 |
| `6dca6d01` | `akita_black_bowl_2` | 8, 12 |

The strongest shared signal is layer 12. Layer 8 is smaller but consistent across both wrong-object variants.

## Interpretation

This supports a modest causal claim:

> In Scene 4 task 61, replacing specific saved VLM-to-action interface KV layers from good `chocolate_pudding_1` runs shifts wrong-object action chunks toward the requested object more than controls.

This does not prove the full mechanism. It identifies a causal interface-level site worth narrowing with expert-hidden-state tracing.

## Next Step

Run the same good/bad pairs with narrower expert-hidden-state patching at layers 8 and 12 first. Keep layer 16 as a secondary candidate and treat layer 17 cautiously because call-shifted controls were competitive there.

## Follow-Up: Expert Hidden Patch

I ran a focused expert-hidden patch at the final denoising step for layers `8,12,16`.

Outputs:

- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_expert_hidden_trace/scene4_expert_hidden_trace.csv`
- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_expert_hidden_trace/scene4_expert_hidden_trace_gate_assessment.csv`
- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_expert_hidden_trace/scene4_expert_hidden_trace_pair_layer_gate.csv`

Replay again passed exactly. Hidden tensors were present for all 6 pairs, shape `1x50x1024`.

| layer | rescue delta | best control delta | rescue advantage | pass |
|---:|---:|---:|---:|---|
| 8 | 0.0127 | 0.0000 | 0.0127 | True |
| 12 | -0.0493 | 0.0000 | -0.0493 | False |
| 16 | -0.0160 | 0.0000 | -0.0160 | False |

No expert-hidden layer passed for every pair. This means the clean layer-12 interface signal did not simply transfer to replacing the expert layer-12 hidden state at the final denoising step.

Interpretation: the causal site is more likely in the VLM prefix KV cache consumed by the expert than in a final-step expert hidden-state value with the same numeric layer index. It is also possible that expert-hidden patching needs per-denoising-step donor states. The capture code now supports `expert_selected_hidden_by_step`, but existing episodes need to be regenerated before this can be tested broadly.

## Follow-Up: Local Interface Sweep

I then ran a finer VLM-interface sweep around the layer-12 region: layers `6,7,8,9,10,11,12,13,14,15`.

Outputs:

- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_causal_trace_local/scene4_interface_causal_trace.csv`
- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_causal_trace_local/scene4_interface_causal_trace_gate_assessment.csv`
- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_causal_trace_local/scene4_interface_causal_trace_pair_layer_gate.csv`

| layer | rescue delta | best control delta | rescue advantage | pass |
|---:|---:|---:|---:|---|
| 6 | 0.0109 | 0.0181 | -0.0072 | False |
| 7 | 0.0048 | 0.0000 | 0.0048 | True |
| 8 | 0.0226 | 0.0072 | 0.0154 | True |
| 9 | 0.0165 | 0.0000 | 0.0165 | True |
| 10 | 0.0396 | 0.0071 | 0.0325 | True |
| 11 | 0.0023 | 0.0080 | -0.0057 | False |
| 12 | 0.1243 | 0.0000 | 0.1243 | True |
| 13 | 0.0986 | 0.0897 | 0.0089 | True |
| 14 | 0.7928 | 0.3553 | 0.4375 | True |
| 15 | 0.0213 | 0.0134 | 0.0079 | True |

Layers passing for every pair: `7,8,10,12,14`.

The local sweep changes the ranking. Layer 12 remains a strong robust site, but layer 14 is much larger. Layer 14 also has larger controls, especially call-shifted and layer-shuffled controls, so it is strong but less clean than layer 12. Layer 8 remains a smaller, clean, consistent site.

Updated interpretation: the strongest interface-level causal band is in the middle VLM KV layers, especially layers `12` and `14`, with layer `8` as a smaller robust early-mid site. The next best check is to replicate layer `12` and `14` on another task family before claiming generality.

## Follow-Up: Attribution Patching

I added a gradient attribution-patching screen over layers `8,12,14` to identify which KV-cache subcomponents explain the layer-level causal effects.

Outputs:

- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_attribution_patching/scene4_attribution_patch_scores.csv`
- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_attribution_patching/scene4_attribution_patch_summary.csv`
- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_attribution_patching/scene4_attribution_patch_exact_topk.csv`

The attribution estimate used:

```text
grad_bad(target-vs-wrong action margin) dot (donor KV - recipient KV)
```

Top attribution-patching features:

| layer | K/V | component | mean signed attribution | positive rate |
|---:|---|---|---:|---:|
| 14 | value | vision_prefix | 0.3424 | 1.00 |
| 14 | value | mid_prefix | 0.2144 | 1.00 |
| 12 | value | vision_prefix | 0.1747 | 0.83 |
| 14 | value | early_prefix | 0.1280 | 1.00 |
| 12 | key | early_prefix | 0.0983 | 1.00 |
| 14 | key | vision_prefix | 0.0969 | 1.00 |

I then exact-patched the top attribution-ranked token groups. Exact patching agreed with the attribution ranking for the strongest features:

| rank | layer | K/V | component | exact delta | positive rate |
|---:|---:|---|---|---:|---:|
| 1 | 14 | value | vision_prefix | 0.7052 | 1.00 |
| 2 | 14 | value | mid_prefix | 0.4658 | 1.00 |
| 7 | 14 | key | vision_prefix | 0.2742 | 1.00 |
| 4 | 14 | value | early_prefix | 0.1372 | 1.00 |
| 3 | 12 | value | vision_prefix | 0.0179 | 0.83 |

This sharpens the result. The strongest useful feature is not just “layer 14”; it is mostly the layer-14 value cache over vision-prefix tokens. Layer 12 value-cache vision-prefix remains positive but is much smaller under exact token-group patching.

## Follow-Up: Visual Bin Localization

I split the `768` visual-prefix tokens into `24` bins and repeated the attribution-patching screen for layers `12,14`.

Outputs:

- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_attribution_patching_bins/scene4_attribution_patch_scores.csv`
- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_attribution_patching_bins/scene4_attribution_patch_summary.csv`
- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_attribution_patching_bins/scene4_attribution_patch_exact_topk.csv`
- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_attribution_patching_bins/scene4_visual_bin_map.csv`

The visual prefix is `3 * 256` tokens: `observation.images.image`, `observation.images.image2`, and the configured `observation.images.empty_camera_0`. Each image is a `16x16` patch grid, because PaliGemma/SigLIP uses `224x224` images with patch size `14`.

Top binned attribution features from the initial six-pair task-61 screen:

| layer | K/V | component | mean signed attribution | positive rate | map |
|---:|---|---|---:|---:|---|
| 14 | value | `vision_bin_10_of_24` | 0.2012 | 1.00 | `image2`, rows 4-5, cols 0-15 |
| 14 | value | `vision_bin_09_of_24` | 0.1135 | 1.00 | `image2`, rows 2-3, cols 0-15 |
| 12 | value | `vision_bin_04_of_24` | 0.1109 | 0.50 | `image`, rows 8-9, cols 0-15 |
| 12 | key | `vision_bin_04_of_24` | 0.1083 | 0.50 | `image`, rows 8-9, cols 0-15 |
| 14 | key | `vision_bin_10_of_24` | 0.0819 | 0.67 | `image2`, rows 4-5, cols 0-15 |
| 14 | value | `vision_bin_15_of_24` | 0.0698 | 1.00 | `image2`, rows 14-15, cols 0-15 |

I then exact-patched the top candidates across all `27` available task-61 `chocolate_pudding_1` donor-recipient rows, not just the original six-pair subset.

| layer | K/V | component | rows | exact delta | positive rate | map |
|---:|---|---|---:|---:|---:|---|
| 14 | value | `vision_prefix` | 27 | 0.7432 | 1.00 | all visual tokens |
| 14 | value | `mid_prefix` | 27 | 0.3420 | 0.89 | mixed visual/text middle third |
| 14 | value | `early_prefix` | 27 | 0.2891 | 1.00 | early prefix, mostly first camera and start of second |
| 14 | value | `vision_bin_10_of_24` | 27 | 0.2427 | 0.96 | `image2`, rows 4-5, cols 0-15 |
| 12 | value | `vision_prefix` | 27 | 0.0245 | 0.85 | all visual tokens |
| 12 | value | `early_prefix` | 27 | 0.0112 | 0.70 | early prefix |

This makes the localization much sharper: a 32-token strip in the second camera at layer-14 value cache recovers about one third of the full layer-14 visual-prefix exact effect in the expanded task-61 set (`0.2427 / 0.7432`). It is positive in `26/27` exact patches.

I then exact-scanned individual tokens inside task61's hot strip, using the original six-pair subset:

```text
layer: 14
K/V: value
component: vision_bin_10_of_24
tokens: 320-351
map: observation.images.image2, patch rows 4-5, cols 0-15
```

Top individual-token exact patches:

| token | image | patch row | patch col | exact delta | positive rate |
|---:|---|---:|---:|---:|---:|
| 331 | `image2` | 4 | 11 | 0.0740 | 1.00 |
| 327 | `image2` | 4 | 7 | 0.0645 | 1.00 |
| 323 | `image2` | 4 | 3 | 0.0491 | 1.00 |
| 347 | `image2` | 5 | 11 | 0.0261 | 1.00 |
| 330 | `image2` | 4 | 10 | 0.0204 | 0.83 |
| 339 | `image2` | 5 | 3 | 0.0120 | 1.00 |
| 328 | `image2` | 4 | 8 | 0.0067 | 0.67 |
| 336 | `image2` | 5 | 0 | 0.0049 | 0.50 |

Token-scan interpretation: the bin-level effect is itself sparse. The strongest tokens are on patch row `4`, not evenly distributed across rows `4-5`; row `4` accounts for about `0.217` summed mean exact delta across its 16 tokens, while row `5` accounts for about `0.036`. This suggests the causal feature may correspond to a localized horizontal band in the second camera view rather than the whole 32-token bin.

For visual inspection, I wrote an approximate overlay of the top eight hot tokens on the representative bad call's second-camera frame:

- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_attribution_patching_bins/task61_hot_tokens_call03_image2_overlay.png`

The overlay scales the `16x16` token grid over the saved `256x256` camera image. It should be treated as a spatial guide, not a pixel-perfect reconstruction of the model's internal `224x224` preprocessed image.

### Hot-Token Role Test

I then tested what role the hot activation plays by applying three kinds of intervention to the top task61 tokens:

- `success_to_failure`: patch donor/success tokens into the bad recipient.
- `failure_to_success`: patch bad recipient tokens into the donor/success run.
- neutral ablations: replace bad recipient tokens with zero, layer mean, or bin mean.

For the top three tokens `331,327,323`:

| intervention | mean delta | positive rate | interpretation |
|---|---:|---:|---|
| `success_to_failure` | 0.2717 | 1.00 | strong rescue |
| `failure_zero` | 0.0121 | 0.50 | tiny, inconsistent |
| `failure_layer_mean` | 0.0060 | 0.50 | tiny, inconsistent |
| `failure_bin_mean` | -0.0006 | 0.50 | no useful effect |
| `failure_to_success` | 0.0017 | 0.83 | no damage to success |

For the top eight tokens `331,327,323,347,330,339,328,336`:

| intervention | mean delta | positive rate | interpretation |
|---|---:|---:|---|
| `success_to_failure` | 0.3879 | 1.00 | nearly recovers the full hot-bin effect |
| `failure_zero` | 0.0367 | 0.50 | small, inconsistent |
| `failure_layer_mean` | 0.0239 | 0.50 | small, inconsistent |
| `failure_bin_mean` | 0.0027 | 0.50 | no useful effect |
| `failure_to_success` | 0.0021 | 0.83 | no damage to success |

Role-test interpretation: these tokens behave more like a donor/success feature that can be added to rescue the bad run than like a toxic bad-run feature that must be removed. If the bad run's hot tokens were directly pulling the action toward the wrong object, zeroing or mean-replacing them should have helped more reliably. Instead, neutral ablations are tiny and inconsistent, while donor replacement is large and consistent. Also, patching bad tokens into success does not noticeably damage the good run, so success appears robust to these few bad-token values or uses redundant evidence elsewhere.

### Cumulative Token and Flow Tests

I then checked whether the top tokens compose smoothly and whether the patch changes the denoising trajectory, not only the final score.

Outputs:

- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_attribution_patching_cumulative/scene4_attribution_patch_cumulative_tokens.csv`
- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_attribution_patching_flow/scene4_attribution_patch_flow_trace.csv`

Cumulative donor-token rescue across the original six task61 pairs:

| top-k tokens | mean delta | positive rate | patched mean margin |
|---:|---:|---:|---:|
| 1 | 0.0740 | 1.00 | -0.2001 |
| 2 | 0.1853 | 1.00 | -0.0889 |
| 3 | 0.2717 | 1.00 | -0.0025 |
| 4 | 0.3256 | 1.00 | 0.0514 |
| 8 | 0.3879 | 1.00 | 0.1137 |

This is a smooth cumulative rescue curve: each additional high-ranked token adds useful signal, and the top eight tokens move the mean first-action margin from negative to positive.

Flow trace was run on three donor pairings for the same bad recipient call, so it tests donor consistency for one failure case rather than recipient diversity. The top-eight patch changed the denoising trajectory from early/mid steps onward:

| denoise step | baseline margin | patched margin | patched - baseline |
|---:|---:|---:|---:|
| 0 | 0.1005 | 0.1005 | 0.0000 |
| 1 | 0.0493 | 0.0747 | 0.0255 |
| 2 | -0.1167 | -0.0838 | 0.0329 |
| 3 | -0.3851 | -0.2461 | 0.1389 |
| 4 | -0.3225 | -0.1926 | 0.1300 |
| 5 | -0.2810 | -0.1740 | 0.1070 |
| 6 | -0.2572 | -0.1607 | 0.0964 |
| 7 | -0.2390 | -0.1489 | 0.0901 |
| 8 | -0.2223 | -0.1366 | 0.0857 |
| 9 | -0.2020 | -0.1208 | 0.0813 |
| 10 | -0.1821 | -0.1076 | 0.0745 |

Flow interpretation: the patch does not merely alter the final readout after denoising. It changes the action sample trajectory during denoising, with the largest margin separation appearing around steps `3-5` and persisting through the final action chunk. The tested three rows share the same recipient, so this should be replicated across distinct bad recipients before claiming trajectory-level generality.

### Hot-Token Atlas

I built an overlay atlas for the top eight task61 tokens across all `27` available donor-recipient pair rows.

Outputs:

- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_hot_token_atlas_task61/hot_token_atlas_contact_sheet.png`
- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_hot_token_atlas_task61/hot_token_atlas_manifest.csv`
- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_hot_token_atlas_task61/pair_XX_hot_token_atlas.png`

Atlas observations:

- All eight hot tokens are in `observation.images.image2`, patch rows `4-5`.
- Across recipients, those spatial addresses generally land on or near the upper band containing bowl-like distractors, table/crate edges, or nearby scene context.
- They do not generally land on the chocolate-pudding package. In many recipient frames, the package is more clearly visible in `observation.images.image`, not `image2`.
- Donor/good frames show the same token addresses in a similar upper-band region of `image2`; these locations are also usually not directly on the chocolate-pudding package.

Atlas interpretation: this weakens a naive “these are target-object pixels” story. The stronger interpretation remains that these are spatially indexed layer-14 handoff sites where success-run contextual information can be injected. The current atlas cannot identify the encoded variable by itself; it only grounds the token addresses and shows that the feature is not trivially target-pixel-aligned.

## Follow-Up: Task 60 Replication

I replicated the attribution-patching screen on a second Scene 4 family:

- Task id: `60`
- Target object: `akita_black_bowl_1`
- Wrong object: `akita_black_bowl_2`
- Pairs: `12`
- Output dir: `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/scene4_attribution_patching_task60/`

Top task-60 attribution features shifted somewhat toward layer 12, but vision-prefix KV cache remained the main interface object:

| layer | K/V | component | mean signed attribution | positive rate |
|---:|---|---|---:|---:|
| 12 | value | `vision_prefix` | 0.1964 | 1.00 |
| 12 | value | `early_prefix` | 0.1755 | 1.00 |
| 12 | key | `early_prefix` | 0.1679 | 1.00 |
| 14 | value | `vision_prefix` | 0.1655 | 1.00 |
| 12 | key | `vision_bin_04_of_24` | 0.1330 | 1.00 |
| 12 | value | `vision_bin_04_of_24` | 0.1162 | 1.00 |

Exact top-k validation for task 60:

| layer | K/V | component | rows | exact delta | positive rate | map |
|---:|---|---|---:|---:|---:|---|
| 14 | value | `vision_prefix` | 12 | 0.1975 | 1.00 | all visual tokens |
| 12 | key | `early_prefix` | 12 | 0.1030 | 1.00 | early prefix |
| 12 | value | `vision_prefix` | 12 | 0.1019 | 1.00 | all visual tokens |
| 12 | key | `vision_prefix` | 12 | 0.0816 | 1.00 | all visual tokens |
| 12 | value | `early_prefix` | 12 | 0.0810 | 1.00 | early prefix |
| 12 | value | `vision_bin_04_of_24` | 12 | 0.0651 | 1.00 | `image`, rows 8-9, cols 0-15 |
| 12 | key | `vision_bin_04_of_24` | 12 | 0.0459 | 1.00 | `image`, rows 8-9, cols 0-15 |
| 14 | key | `vision_prefix` | 12 | 0.0323 | 0.58 | all visual tokens |

Replication interpretation: the exact causal site is not identical across target families. Task 61 is dominated by layer-14 value-cache visual features in the second camera; task 60 has a stronger layer-12 contribution and a meaningful first-camera bin. The shared pattern is that target binding is carried by visual-prefix KV cache features at the VLM-to-expert handoff, especially value cache, rather than by final expert hidden-state replacement.
