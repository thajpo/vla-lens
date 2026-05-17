# Experiment 2G: Success/Failure Direction Overlap

This analysis compares the explicit success-vs-failure direction within Scene 1 mixed-outcome tasks to the previously identified benchmark-delta direction.

| family       | object_label   |   success_count |   failure_count |   delta_overlap_cosine |   success_projection_mean |   failure_projection_mean |   mann_whitney_u |   p_fail_greater |
|:-------------|:---------------|----------------:|----------------:|-----------------------:|--------------------------:|--------------------------:|-----------------:|-----------------:|
| expert_final | alphabet_soup  |             167 |              84 |               0.257895 |                 77.544    |                 88.6368   |             8173 |         0.016402 |
| expert_final | cream_cheese   |             215 |             133 |               0.332121 |                 69.7418   |                 79.7919   |            16612 |         0.005583 |
| expert_flow0 | alphabet_soup  |             167 |              84 |               0.285104 |                  0.550722 |                  0.724381 |             7183 |         0.378109 |
| expert_flow0 | cream_cheese   |             215 |             133 |               0.56882  |                  0.297342 |                  0.551223 |            16436 |         0.009528 |

## Interpretation guide

- Positive cosine means the success/failure direction and benchmark-delta direction are aligned.
- The Mann-Whitney test checks whether failures project more strongly onto the benchmark-delta direction than successes.
