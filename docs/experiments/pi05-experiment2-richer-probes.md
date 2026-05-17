# Experiment 2B: Richer Geometry / Relation Probes

These probes test whether the canonical captures preserve not just target identity, but also control-relevant spatial information.

## Best results by target kind

| benchmark     | family              | target_kind       | target_dim   |   train_samples |   val_samples |   test_samples |   best_alpha |     r2 |    mae |   baseline_mae |
|:--------------|:--------------------|:------------------|:-------------|----------------:|--------------:|---------------:|-------------:|-------:|-------:|---------------:|
| libero_90     | vlm_text_pooled     | target_pos        | x            |             954 |           245 |            240 |          100 | 0.9518 | 0.0217 |         0.1246 |
| libero_90     | handoff_all_pooled  | target_pos        | x            |             954 |           245 |            240 |          100 | 0.9461 | 0.0238 |         0.1246 |
| libero_90     | vlm_image_pooled    | target_pos        | x            |             954 |           245 |            240 |          100 | 0.9439 | 0.0241 |         0.1246 |
| libero_90     | expert_final_pooled | target_pos        | x            |             954 |           245 |            240 |           10 | 0.9272 | 0.0273 |         0.1246 |
| libero_90     | expert_flow0_pooled | target_pos        | x            |             954 |           245 |            240 |           10 | 0.9262 | 0.0279 |         0.1246 |
| libero_90     | vlm_text_pooled     | target_to_basket  | x            |             954 |           245 |            240 |          100 | 0.9515 | 0.0221 |         0.1246 |
| libero_90     | handoff_all_pooled  | target_to_basket  | x            |             954 |           245 |            240 |           10 | 0.9297 | 0.0263 |         0.1246 |
| libero_90     | vlm_image_pooled    | target_to_basket  | x            |             954 |           245 |            240 |           10 | 0.9255 | 0.0273 |         0.1246 |
| libero_90     | expert_flow0_pooled | target_to_basket  | x            |             954 |           245 |            240 |           10 | 0.9244 | 0.0285 |         0.1246 |
| libero_90     | expert_final_pooled | target_to_basket  | x            |             954 |           245 |            240 |           10 | 0.9238 | 0.0281 |         0.1246 |
| libero_90     | vlm_text_pooled     | target_to_gripper | x            |             954 |           245 |            240 |          100 | 0.9435 | 0.025  |         0.125  |
| libero_90     | handoff_all_pooled  | target_to_gripper | x            |             954 |           245 |            240 |          100 | 0.9385 | 0.0264 |         0.125  |
| libero_90     | vlm_image_pooled    | target_to_gripper | x            |             954 |           245 |            240 |          100 | 0.9368 | 0.0268 |         0.125  |
| libero_90     | expert_final_pooled | target_to_gripper | x            |             954 |           245 |            240 |           10 | 0.9275 | 0.0295 |         0.125  |
| libero_90     | vlm_text_pooled     | target_to_gripper | y            |             954 |           245 |            240 |          100 | 0.9266 | 0.0345 |         0.129  |
| libero_object | expert_final_pooled | target_pos        | z            |             510 |           124 |            127 |          100 | 0.9369 | 0.0153 |         0.1045 |
| libero_object | vlm_image_pooled    | target_pos        | z            |             510 |           124 |            127 |          100 | 0.9357 | 0.0161 |         0.1045 |
| libero_object | handoff_all_pooled  | target_pos        | z            |             510 |           124 |            127 |          100 | 0.9353 | 0.016  |         0.1045 |
| libero_object | expert_flow0_pooled | target_pos        | z            |             510 |           124 |            127 |            1 | 0.9346 | 0.0155 |         0.1045 |
| libero_object | expert_flow0_pooled | target_pos        | x            |             510 |           124 |            127 |           10 | 0.922  | 0.013  |         0.0637 |
| libero_object | expert_final_pooled | target_to_basket  | z            |             510 |           124 |            127 |          100 | 0.9369 | 0.0153 |         0.1045 |
| libero_object | vlm_image_pooled    | target_to_basket  | z            |             510 |           124 |            127 |          100 | 0.9357 | 0.0161 |         0.1045 |
| libero_object | handoff_all_pooled  | target_to_basket  | z            |             510 |           124 |            127 |          100 | 0.9353 | 0.016  |         0.1045 |
| libero_object | expert_flow0_pooled | target_to_basket  | z            |             510 |           124 |            127 |            1 | 0.9346 | 0.0155 |         0.1045 |
| libero_object | vlm_image_pooled    | target_to_basket  | x            |             510 |           124 |            127 |          100 | 0.9215 | 0.0149 |         0.0652 |
| libero_object | expert_final_pooled | target_to_gripper | x            |             510 |           124 |            127 |          100 | 0.9496 | 0.0093 |         0.0367 |
| libero_object | expert_flow0_pooled | target_to_gripper | x            |             510 |           124 |            127 |          100 | 0.9492 | 0.0096 |         0.0367 |
| libero_object | handoff_all_pooled  | target_to_gripper | x            |             510 |           124 |            127 |          100 | 0.9291 | 0.0115 |         0.0367 |
| libero_object | vlm_image_pooled    | target_to_gripper | x            |             510 |           124 |            127 |          100 | 0.9256 | 0.0117 |         0.0367 |
| libero_object | vlm_text_pooled     | target_to_gripper | x            |             510 |           124 |            127 |          100 | 0.9248 | 0.0128 |         0.0367 |

## Interpretation guide

- Strong `target_pos` probes suggest the representation preserves where the target is in absolute scene coordinates.
- Strong `target_to_gripper` probes suggest the representation preserves control-relevant relative geometry for reaching/grasping.
- Strong `target_to_basket` probes suggest the representation preserves relational goal information beyond object identity.
- Comparing VLM/handoff/expert families helps distinguish insufficient handoff content from downstream expert usage failures.
