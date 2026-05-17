# Experiment 2E: Benchmark Classifier and Top Features

This analysis trains benchmark classifiers on the overlapping classes (`alphabet_soup`, `cream_cheese`) to identify what dimensions distinguish `Scene 1` from `LIBERO_OBJECT`.

## Benchmark-classifier summary

| family       |   train_samples |   val_samples |   test_samples |   best_C |   accuracy |   balanced_accuracy | labels                         |
|:-------------|----------------:|--------------:|---------------:|---------:|-----------:|--------------------:|:-------------------------------|
| vlm_all      |             645 |           169 |            161 |     0.1  |     1      |              1      | ['libero_object', 'libero_90'] |
| vlm_image    |             645 |           169 |            161 |     1    |     0.9938 |              0.9918 | ['libero_object', 'libero_90'] |
| vlm_text     |             645 |           169 |            161 |     0.01 |     1      |              1      | ['libero_object', 'libero_90'] |
| handoff_all  |             645 |           169 |            161 |     0.1  |     1      |              1      | ['libero_object', 'libero_90'] |
| expert_final |             645 |           169 |            161 |     0.01 |     1      |              1      | ['libero_object', 'libero_90'] |
| expert_flow0 |             645 |           169 |            161 |     0.01 |     1      |              1      | ['libero_object', 'libero_90'] |

## Top dimensions by family

### vlm_all

- top dims: `[1389, 1578, 366, 1394, 1151, 476, 46, 122, 1580, 1756, 1453, 1089, 1414, 281, 1500, 1640, 1309, 1914, 1737, 158]`
- top |weights|: `[0.083102, 0.08294, 0.08183, 0.078262, 0.075672, 0.075197, 0.070207, 0.069176, 0.068694, 0.065305, 0.064896, 0.064357, 0.063576, 0.062461, 0.05994, 0.059722, 0.059647, 0.059234, 0.059057, 0.058801]`

### vlm_image

- top dims: `[1453, 1089, 1151, 1389, 1756, 1737, 1578, 1394, 476, 1580, 1747, 1269, 1742, 1644, 1942, 1134, 170, 887, 1650, 281]`
- top |weights|: `[0.106735, 0.106319, 0.104899, 0.104779, 0.102542, 0.10201, 0.10032, 0.09805, 0.097816, 0.096471, 0.092457, 0.092314, 0.091846, 0.089069, 0.088088, 0.087622, 0.087609, 0.086304, 0.086197, 0.086022]`

### vlm_text

- top dims: `[963, 1578, 1389, 366, 839, 195, 883, 1571, 826, 743, 1557, 1527, 860, 1762, 2021, 935, 1068, 1151, 344, 1072]`
- top |weights|: `[0.057772, 0.052666, 0.051604, 0.047203, 0.046814, 0.046365, 0.046276, 0.044513, 0.043702, 0.042801, 0.039902, 0.039655, 0.039647, 0.03928, 0.037191, 0.037094, 0.03689, 0.03677, 0.036711, 0.036668]`

### handoff_all

- top dims: `[1389, 1578, 366, 1394, 1151, 476, 46, 122, 1580, 1756, 1453, 1089, 1414, 281, 1500, 1640, 1309, 1914, 1737, 158]`
- top |weights|: `[0.083102, 0.08294, 0.08183, 0.078262, 0.075672, 0.075197, 0.070207, 0.069176, 0.068694, 0.065305, 0.064896, 0.064357, 0.063576, 0.062461, 0.05994, 0.059722, 0.059647, 0.059234, 0.059057, 0.058801]`

### expert_final

- top dims: `[8, 621, 832, 333, 686, 31, 483, 121, 435, 783, 454, 673, 485, 948, 735, 815, 71, 887, 947, 5]`
- top |weights|: `[0.09328, 0.08415, 0.082126, 0.075477, 0.075295, 0.075211, 0.075115, 0.074018, 0.07239, 0.071419, 0.070803, 0.070402, 0.070388, 0.070188, 0.069551, 0.069487, 0.069453, 0.067812, 0.067154, 0.066981]`

### expert_flow0

- top dims: `[621, 735, 832, 288, 796, 887, 71, 483, 454, 718, 464, 915, 849, 122, 664, 783, 549, 435, 686, 578]`
- top |weights|: `[0.100722, 0.091936, 0.08748, 0.082212, 0.08105, 0.079504, 0.077291, 0.076377, 0.075357, 0.07484, 0.073943, 0.073289, 0.072511, 0.071428, 0.071142, 0.070777, 0.07008, 0.068683, 0.067914, 0.067263]`

## Interpretation guide

- High benchmark-classifier accuracy means the representation contains benchmark-specific structure even when object identity is held fixed.
- Stronger benchmark separability in expert families than VLM/handoff families supports the idea that benchmark-specificity grows inside the expert.
- The top dimensions are candidate feature directions for later overlap and intervention analyses.
