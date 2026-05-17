# Experiment 2D: Cross-Benchmark Delta Analysis

This analysis isolates what changes between `LIBERO_OBJECT` and `Scene 1` for the overlapping classes `alphabet_soup` and `cream_cheese`.

## Delta summary

| family       | object_label   |   delta_norm |   shared_delta_cosine | top_dims                                                                                                         |
|:-------------|:---------------|-------------:|----------------------:|:-----------------------------------------------------------------------------------------------------------------|
| vlm_all      | alphabet_soup  |       3.763  |               -0.4656 | [50, 1959, 1703, 1120, 256, 46, 1713, 1722, 1967, 940, 1048, 1634, 1732, 157, 2019, 1699, 1097, 387, 222, 1985]  |
| vlm_all      | cream_cheese   |       2.6527 |               -0.4656 | [50, 1959, 1722, 1577, 2044, 387, 46, 1819, 2007, 1645, 617, 1314, 1151, 653, 1929, 1907, 1790, 1566, 1156, 44]  |
| vlm_image    | alphabet_soup  |       3.8542 |               -0.3762 | [1959, 50, 1703, 256, 46, 1120, 1713, 1967, 1722, 1634, 940, 157, 1048, 80, 1732, 418, 726, 387, 1097, 1985]     |
| vlm_image    | cream_cheese   |       3.2615 |               -0.3762 | [50, 1959, 1722, 387, 1819, 46, 2044, 1790, 1634, 2015, 2007, 1277, 610, 706, 1151, 453, 1645, 1929, 921, 1067]  |
| vlm_text     | alphabet_soup  |       4.2352 |                0.1985 | [50, 1959, 2015, 1577, 1699, 1722, 674, 981, 1120, 2019, 1713, 1826, 1790, 1891, 1703, 880, 1116, 233, 367, 304] |
| vlm_text     | cream_cheese   |       4.0084 |                0.1985 | [1959, 50, 1577, 617, 1713, 885, 2015, 1511, 1499, 653, 981, 706, 1905, 813, 474, 1967, 307, 1294, 1968, 1813]   |
| handoff_all  | alphabet_soup  |       3.763  |               -0.4656 | [50, 1959, 1703, 1120, 256, 46, 1713, 1722, 1967, 940, 1048, 1634, 1732, 157, 2019, 1699, 1097, 387, 222, 1985]  |
| handoff_all  | cream_cheese   |       2.6527 |               -0.4656 | [50, 1959, 1722, 1577, 2044, 387, 46, 1819, 2007, 1645, 617, 1314, 1151, 653, 1929, 1907, 1790, 1566, 1156, 44]  |
| expert_final | alphabet_soup  |      67.7135 |                0.4842 | [885, 183, 460, 777, 294, 305, 364, 859, 84, 743, 517, 483, 612, 35, 13, 485, 333, 707, 759, 543]                |
| expert_final | cream_cheese   |      52.0359 |                0.4842 | [954, 859, 5, 460, 617, 483, 310, 750, 129, 539, 917, 281, 295, 66, 8, 928, 291, 485, 538, 735]                  |
| expert_flow0 | alphabet_soup  |       1.061  |                0.4372 | [35, 983, 1007, 743, 183, 248, 841, 854, 579, 659, 928, 1016, 1004, 310, 116, 431, 805, 293, 777, 341]           |
| expert_flow0 | cream_cheese   |       0.719  |                0.4372 | [83, 248, 300, 954, 928, 983, 421, 310, 298, 663, 5, 483, 819, 772, 800, 617, 3, 684, 995, 865]                  |

## Projection onto shared benchmark-delta direction

| family       | benchmark     | object_label   | success   |   projection_mean |   projection_std |   count |
|:-------------|:--------------|:---------------|:----------|------------------:|-----------------:|--------:|
| expert_final | libero_90     | alphabet_soup  | False     |           88.6368 |          29.5061 |      84 |
| expert_final | libero_90     | alphabet_soup  | True      |           77.544  |          35.3872 |     167 |
| expert_final | libero_90     | cream_cheese   | False     |           79.7919 |          29.5217 |     133 |
| expert_final | libero_90     | cream_cheese   | True      |           69.7418 |          32.0378 |     215 |
| expert_final | libero_object | alphabet_soup  | False     |           51.5379 |          28.8951 |      30 |
| expert_final | libero_object | alphabet_soup  | True      |           14.8332 |          38.6625 |     166 |
| expert_final | libero_object | cream_cheese   | True      |           30.9227 |          21.8983 |     180 |
| expert_flow0 | libero_90     | alphabet_soup  | False     |            0.7244 |           0.7583 |      84 |
| expert_flow0 | libero_90     | alphabet_soup  | True      |            0.5507 |           1.052  |     167 |
| expert_flow0 | libero_90     | cream_cheese   | False     |            0.5512 |           0.9238 |     133 |
| expert_flow0 | libero_90     | cream_cheese   | True      |            0.2973 |           0.9745 |     215 |
| expert_flow0 | libero_object | alphabet_soup  | False     |            0.1959 |           0.7798 |      30 |
| expert_flow0 | libero_object | alphabet_soup  | True      |           -0.4503 |           1.1599 |     166 |
| expert_flow0 | libero_object | cream_cheese   | True      |           -0.1652 |           0.8154 |     180 |
| handoff_all  | libero_90     | alphabet_soup  | False     |            0.6739 |           1.5822 |      84 |
| handoff_all  | libero_90     | alphabet_soup  | True      |            0.5844 |           2.2668 |     167 |
| handoff_all  | libero_90     | cream_cheese   | False     |            0.4683 |           2.1454 |     133 |
| handoff_all  | libero_90     | cream_cheese   | True      |           -0.3211 |           1.8203 |     215 |
| handoff_all  | libero_object | alphabet_soup  | False     |           -0.0015 |           1.3244 |      30 |
| handoff_all  | libero_object | alphabet_soup  | True      |           -2.5301 |           1.8925 |     166 |
| handoff_all  | libero_object | cream_cheese   | True      |           -0.7119 |           1.3464 |     180 |
| vlm_all      | libero_90     | alphabet_soup  | False     |            0.6739 |           1.5822 |      84 |
| vlm_all      | libero_90     | alphabet_soup  | True      |            0.5844 |           2.2668 |     167 |
| vlm_all      | libero_90     | cream_cheese   | False     |            0.4683 |           2.1454 |     133 |
| vlm_all      | libero_90     | cream_cheese   | True      |           -0.3211 |           1.8203 |     215 |
| vlm_all      | libero_object | alphabet_soup  | False     |           -0.0015 |           1.3244 |      30 |
| vlm_all      | libero_object | alphabet_soup  | True      |           -2.5301 |           1.8925 |     166 |
| vlm_all      | libero_object | cream_cheese   | True      |           -0.7119 |           1.3464 |     180 |
| vlm_image    | libero_90     | alphabet_soup  | False     |           -4.0289 |           0.9319 |      84 |
| vlm_image    | libero_90     | alphabet_soup  | True      |           -3.9988 |           1.2991 |     167 |
| vlm_image    | libero_90     | cream_cheese   | False     |           -3.8415 |           1.2967 |     133 |
| vlm_image    | libero_90     | cream_cheese   | True      |           -4.2156 |           1.3072 |     215 |
| vlm_image    | libero_object | alphabet_soup  | False     |           -5.1855 |           0.777  |      30 |
| vlm_image    | libero_object | alphabet_soup  | True      |           -6.782  |           1.2054 |     166 |
| vlm_image    | libero_object | cream_cheese   | True      |           -5.5481 |           0.9874 |     180 |
| vlm_text     | libero_90     | alphabet_soup  | False     |           42.3108 |           1.8144 |      84 |
| vlm_text     | libero_90     | alphabet_soup  | True      |           41.9276 |           2.2043 |     167 |
| vlm_text     | libero_90     | cream_cheese   | False     |           40.864  |           2.4322 |     133 |
| vlm_text     | libero_90     | cream_cheese   | True      |           40.6309 |           2.7013 |     215 |
| vlm_text     | libero_object | alphabet_soup  | False     |           41.276  |           1.7958 |      30 |
| vlm_text     | libero_object | alphabet_soup  | True      |           38.2555 |           2.8569 |     166 |
| vlm_text     | libero_object | cream_cheese   | True      |           37.6749 |           2.3555 |     180 |

## Interpretation guide

- A high cosine between the soup and cheese deltas suggests a shared benchmark-specific direction, not pure class-specific noise.
- Projection differences between successful and failed Scene 1 rollouts test whether the benchmark delta is related to the failure mechanism.
- Large expert-family delta norms relative to VLM/handoff families suggest benchmark-specificity grows inside the expert.
