# Experiment 1: Language Routing

## Dataset

| benchmark     |   tasks |   episodes |
|:--------------|--------:|-----------:|
| libero_90     |       4 |        239 |
| libero_object |       4 |        239 |

The canonical main-capture split is organized by layout. The split files for later probes were generated here as a convenience and written to `artifacts/pi05_analysis/routing/layout_splits.json`.

## Per-task summary

| benchmark     | object_label   |   episodes |   success_rate |   mean_steps |   median_steps |   mean_vlm_calls |   mean_expert_calls |
|:--------------|:---------------|-----------:|---------------:|-------------:|---------------:|-----------------:|--------------------:|
| libero_90     | alphabet_soup  |         60 |          0.8   |        180.7 |          149   |             4.18 |                4.18 |
| libero_90     | cream_cheese   |         59 |          0.678 |        271.8 |          255   |             5.9  |                5.9  |
| libero_90     | ketchup        |         60 |          0     |        320   |          320   |             7    |                7    |
| libero_90     | tomato_sauce   |         60 |          0     |        320   |          320   |             7    |                7    |
| libero_object | alphabet_soup  |         59 |          0.915 |        155.7 |          144   |             3.32 |                3.32 |
| libero_object | butter         |         60 |          1     |        147.5 |          146   |             3.27 |                3.27 |
| libero_object | cream_cheese   |         60 |          1     |        123.5 |          123   |             3    |                3    |
| libero_object | milk           |         60 |          0.95  |        136.9 |          128.5 |             3.15 |                3.15 |

## Target-first routing rates

| benchmark     | object_label   |   episodes |   first_move_target_rate |   first_lift_target_rate |   success_rate |
|:--------------|:---------------|-----------:|-------------------------:|-------------------------:|---------------:|
| libero_90     | alphabet_soup  |         60 |                    0     |                    0.8   |          0.8   |
| libero_90     | cream_cheese   |         59 |                    0     |                    0.085 |          0.678 |
| libero_90     | ketchup        |         60 |                    1     |                    0     |          0     |
| libero_90     | tomato_sauce   |         60 |                    0     |                    0     |          0     |
| libero_object | alphabet_soup  |         59 |                    0.983 |                    0.949 |          0.915 |
| libero_object | butter         |         60 |                    1     |                    1     |          1     |
| libero_object | cream_cheese   |         60 |                    1     |                    1     |          1     |
| libero_object | milk           |         60 |                    1     |                    0.95  |          0.95  |

## libero_90 first moved object confusion

| object_label   |   ketchup_1 |
|:---------------|------------:|
| alphabet_soup  |          60 |
| cream_cheese   |          59 |
| ketchup        |          60 |
| tomato_sauce   |          60 |

## libero_90 first lifted object confusion

| object_label   |   alphabet_soup_1 |   cream_cheese_1 |   tomato_sauce_1 |   nan |
|:---------------|------------------:|-----------------:|-----------------:|------:|
| alphabet_soup  |                48 |                4 |                0 |     8 |
| cream_cheese   |                52 |                5 |                0 |     2 |
| ketchup        |                27 |                1 |                1 |    31 |
| tomato_sauce   |                42 |                0 |                0 |    18 |

## libero_object first moved object confusion

| object_label   |   alphabet_soup_1 |   butter_1 |   cream_cheese_1 |   milk_1 |   nan |
|:---------------|------------------:|-----------:|-----------------:|---------:|------:|
| alphabet_soup  |                58 |          0 |                0 |        0 |     1 |
| butter         |                 0 |         60 |                0 |        0 |     0 |
| cream_cheese   |                 0 |          0 |               60 |        0 |     0 |
| milk           |                 0 |          0 |                0 |       60 |     0 |

## libero_object first lifted object confusion

| object_label   |   alphabet_soup_1 |   butter_1 |   cream_cheese_1 |   milk_1 |   nan |
|:---------------|------------------:|-----------:|-----------------:|---------:|------:|
| alphabet_soup  |                56 |          0 |                0 |        0 |     3 |
| butter         |                 0 |         60 |                0 |        0 |     0 |
| cream_cheese   |                 0 |          0 |               60 |        0 |     0 |
| milk           |                 0 |          0 |                0 |       57 |     3 |

## Interpretation

- `LIBERO_OBJECT` should show strong diagonal structure in both first-move and first-lift confusion tables if the routing baseline is intact.
- `Scene 1` should reveal the already-observed asymmetry: strong cream_cheese success, zero ketchup / tomato_sauce success, and partially collapsed early routing.
- These behavioral outputs define the benchmark roles used by later mechanistic analyses.
