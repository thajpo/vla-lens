# Experiment 2C: Scene 1 Success vs Failure

Mixed-outcome Scene 1 targets found: `alphabet_soup, cream_cheese`.

## Aggregated comparison

| object_label   | success   |   episodes |   mean_steps |   mean_target_max_lift |   mean_min_target_distance |   mean_first_close_step |   mean_action_dx |   mean_action_dy |   mean_action_dz |   mean_action_gripper |   mean_attn_image_mass |   mean_attn_text_mass |   mean_attn_suffix_mass |
|:---------------|:----------|-----------:|-------------:|-----------------------:|---------------------------:|------------------------:|-----------------:|-----------------:|-----------------:|----------------------:|-----------------------:|----------------------:|------------------------:|
| alphabet_soup  | False     |         12 |      320     |                 0.0053 |                     0.0371 |                 60.6667 |           0.2091 |          -0.4057 |           0.2055 |               -0.0223 |                 0.0712 |                0.0271 |                  0.9017 |
| alphabet_soup  | True      |         48 |      145.854 |                 0.2039 |                     0.0154 |                 59.4375 |           0.1039 |          -0.5271 |           0.1975 |               -0.0037 |                 0.0595 |                0.0316 |                  0.9089 |
| cream_cheese   | False     |         19 |      320     |                 0.1558 |                     0.0197 |                 75.7368 |           0.1863 |          -0.3195 |           0.2077 |               -0.0342 |                 0.072  |                0.0299 |                  0.898  |
| cream_cheese   | True      |         40 |      248.85  |                 0.2149 |                     0.0156 |                108.781  |           0.0483 |          -0.425  |           0.1874 |               -0.0677 |                 0.068  |                0.03   |                  0.902  |

## First moved / first lifted breakdown

### alphabet_soup first moved object by success

| success   |   ketchup_1 |
|:----------|------------:|
| False     |          12 |
| True      |          48 |

### alphabet_soup first lifted object by success

| success   |   alphabet_soup_1 |   cream_cheese_1 |   nan |
|:----------|------------------:|-----------------:|------:|
| False     |                 0 |                4 |     8 |
| True      |                48 |                0 |     0 |

### cream_cheese first moved object by success

| success   |   ketchup_1 |
|:----------|------------:|
| False     |          19 |
| True      |          40 |

### cream_cheese first lifted object by success

| success   |   alphabet_soup_1 |   cream_cheese_1 |   nan |
|:----------|------------------:|-----------------:|------:|
| False     |                13 |                4 |     2 |
| True      |                39 |                1 |     0 |

## Interpretation guide

- Differences in first action / gripper command between success and failure suggest the expert diverges very early.
- Differences in attention mass between success and failure suggest the expert is consulting different parts of the prefix.
- Differences in minimum target distance and close timing localize whether failure is approach-related or grasp-related.
