# PI0.5 Dataset Shape For Blog

This is a target-binding dataset: the instruction names one object, the scene contains plausible distractors, and we record whether the policy acts on the named target.

## At a glance

- Probe/control rollouts: 1420
- LIBERO tasks: 51
- Scene families: 8
- Held-out layout ids: 40
- Object table rows: 9680
- Saved policy calls: 6689
- Smaller rollout-report sanity subset: 308 records (220 canonical, 88 target-swap)
- Older activation manifest records: 482

## Figures

### Better fit for the blog intro

![Dataset design](../../artifacts/pi05_analysis/blog_dataset/context_dataset_design.png)

![Object vocabulary heatmap](../../artifacts/pi05_analysis/blog_dataset/context_object_vocab_heatmap.png)

![Object positions](../../artifacts/pi05_analysis/blog_dataset/context_object_positions.png)

![Layout split units](../../artifacts/pi05_analysis/blog_dataset/context_layout_units.png)

![VLM probe gate](../../artifacts/pi05_analysis/blog_dataset/context_vlm_probe_gate.png)

![Layer phase first moved](../../artifacts/pi05_analysis/blog_dataset/context_layer_phase_first_moved_object.png)

![Layer phase first lifted](../../artifacts/pi05_analysis/blog_dataset/context_layer_phase_first_lifted_object.png)

![Phase probe summary](../../artifacts/pi05_analysis/blog_dataset/context_phase_probe_summary.png)

![Geometry probe R2](../../artifacts/pi05_analysis/blog_dataset/context_geometry_probe_r2.png)

### Earlier rollout sanity plots

![Episodes by family and condition](../../artifacts/pi05_analysis/blog_dataset/episodes_by_family_condition.png)

![Success by target object](../../artifacts/pi05_analysis/blog_dataset/success_by_target_object.png)

![Steps histogram](../../artifacts/pi05_analysis/blog_dataset/steps_histogram.png)

![Target binding behavior](../../artifacts/pi05_analysis/blog_dataset/target_binding_behavior.png)

## Blog interpretation

- The main axis is not task count alone; it is contrastive target choice.
- `LIBERO-Object` gives cleaner grocery-object basket tasks.
- `LIBERO-90 Scene 1` gives the same broad action template in a distractor-rich living-room scene.
- `target_swap` runs deliberately move the target location against a partner object; these are useful for asking whether the model follows language, visual location, or a learned object prior.
- The first-moved and first-lifted labels are cheap behavioral probes for target binding before looking at activations.
- For the linear-probe section, prefer the strict-gate and layer/phase heatmaps above. They speak directly to where information is decodable from activations.
- The position scatter and object-vocabulary heatmap explain what the labels mean before introducing the probe equation.

## Suggested blog wording

The dataset contains `1420` PI0.5 rollouts from `51` LIBERO tasks across `8` scene families.
Each rollout stores the task instruction, target object, simulator object table, and outcome labels; across all rollouts there are `6689` saved policy calls.
The object table gives `9680` object rows over `29` object names, including initial/final XYZ positions, target flags, maximum lift, and closest gripper distance.
Train/test splits are held out by layout id (`40` layouts), so the probe cannot simply memorize near-duplicate frames from the same scene arrangement.

## Summary tables

### Episodes by report

| source_file                         | benchmark_family   | condition   |   episodes |   success_rate |   median_steps |
|:------------------------------------|:-------------------|:------------|-----------:|---------------:|---------------:|
| object_subset_20ep.json             | LIBERO-Object      | canonical   |         80 |          0.962 |          136   |
| object_subset_smoke.json            | LIBERO-Object      | canonical   |          4 |          1     |          133.5 |
| object_subset_swap10.json           | LIBERO-Object      | target_swap |         40 |          0     |          280   |
| object_subset_swap_smoke.json       | LIBERO-Object      | target_swap |          4 |          0     |          280   |
| pilot_scene1_canonical.json         | LIBERO-90 Scene 1  | canonical   |         12 |          0     |          400   |
| scene1_layout20_canonical.json      | LIBERO-90 Scene 1  | canonical   |         80 |          0.4   |          400   |
| scene1_layout5_canonical.json       | LIBERO-90 Scene 1  | canonical   |         20 |          0     |          331   |
| scene1_layout5_canonical_fixed.json | LIBERO-90 Scene 1  | canonical   |         20 |          0.5   |          328   |
| scene1_swap10.json                  | LIBERO-90 Scene 1  | target_swap |         40 |          0.2   |          400   |
| smoke_scene1.json                   | LIBERO-90 Scene 1  | canonical   |          4 |          0     |          321.5 |
| smoke_scene1.json                   | LIBERO-90 Scene 1  | target_swap |          4 |          0     |          400   |

### Episodes by object

| benchmark_family   | condition   | readable_object   |   episodes |   success_rate |   target_selected_first_rate |   target_lifted_first_rate |   median_steps |
|:-------------------|:------------|:------------------|-----------:|---------------:|-----------------------------:|---------------------------:|---------------:|
| LIBERO-90 Scene 1  | canonical   | alphabet soup     |         34 |          0.559 |                        0     |                      0.824 |          123   |
| LIBERO-90 Scene 1  | canonical   | cream cheese      |         34 |          0.676 |                        0     |                      0.118 |          250.5 |
| LIBERO-90 Scene 1  | canonical   | ketchup           |         34 |          0     |                        1     |                      0     |          400   |
| LIBERO-90 Scene 1  | canonical   | tomato sauce      |         34 |          0     |                        0     |                      0     |          400   |
| LIBERO-90 Scene 1  | target_swap | alphabet soup     |         11 |          0     |                        0     |                      0     |          400   |
| LIBERO-90 Scene 1  | target_swap | cream cheese      |         11 |          0.727 |                        0.818 |                      0.818 |          146   |
| LIBERO-90 Scene 1  | target_swap | ketchup           |         11 |          0     |                        0.091 |                      1     |          400   |
| LIBERO-90 Scene 1  | target_swap | tomato sauce      |         11 |          0     |                        0.818 |                      0     |          400   |
| LIBERO-Object      | canonical   | alphabet soup     |         21 |          0.952 |                        0.952 |                      0.952 |          142   |
| LIBERO-Object      | canonical   | butter            |         21 |          1     |                        1     |                      1     |          146   |
| LIBERO-Object      | canonical   | cream cheese      |         21 |          1     |                        1     |                      1     |          125   |
| LIBERO-Object      | canonical   | milk              |         21 |          0.905 |                        0.952 |                      0.905 |          128   |
| LIBERO-Object      | target_swap | alphabet soup     |         11 |          0     |                        0     |                      0     |          280   |
| LIBERO-Object      | target_swap | butter            |         11 |          0     |                        0     |                      0     |          280   |
| LIBERO-Object      | target_swap | cream cheese      |         11 |          0     |                        0.364 |                      0.182 |          280   |
| LIBERO-Object      | target_swap | milk              |         11 |          0     |                        0     |                      0     |          280   |
