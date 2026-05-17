# PI0.5 Flow Outcome Probes

## Method

This predicts future manipulation variables from each saved flow state `flow_x_t`, using a future window of `20` environment steps.

Baselines are included to expose benchmark/scene bias: constant train prior, object-label prior, and object-label plus phase prior.

## Best Scores By Phase And Target

| phase             |   flow_step | target                 | probe_type     |   score |   constant_score |   object_prior_score |   object_phase_prior_score |      mae |   constant_mae |   object_prior_mae |   object_phase_prior_mae |   train_samples |   test_samples |
|:------------------|------------:|:-----------------------|:---------------|--------:|-----------------:|---------------------:|---------------------------:|---------:|---------------:|-------------------:|-------------------------:|----------------:|---------------:|
| all               |          10 | closes_soon            | classification |  0.7125 |           0.7875 |               0.7875 |                     0.7875 | nan      |       nan      |           nan      |                 nan      |             954 |            240 |
| all               |           8 | final_success          | classification |  0.7125 |           0.7    |               0.8833 |                     0.8625 | nan      |       nan      |           nan      |                 nan      |             954 |            240 |
| all               |           7 | future_lift_delta      | regression     | -0.2867 |          -0.0002 |               0.0871 |                     0.2277 |   0.024  |         0.0132 |             0.0117 |                   0.0109 |             954 |            240 |
| all               |           8 | future_min_target_dist | regression     | -0.3104 |          -0.0006 |               0.224  |                     0.3761 |   0.1334 |         0.1098 |             0.1056 |                   0.0851 |             954 |            240 |
| call_00           |           9 | closes_soon            | classification |  0.8    |           0.75   |               0.75   |                     0.75   | nan      |       nan      |           nan      |                 nan      |             159 |             40 |
| call_00           |           9 | final_success          | classification |  0.725  |           0.6    |               0.9    |                     0.9    | nan      |       nan      |           nan      |                 nan      |             159 |             40 |
| call_00           |           0 | future_lift_delta      | regression     |  1      |           1      |               1      |                     1      |   0      |         0      |             0      |                   0      |             159 |             40 |
| call_00           |          10 | future_min_target_dist | regression     |  0.4495 |          -0.0223 |               0.6948 |                     0.6948 |   0.0194 |         0.0293 |             0.0157 |                   0.0157 |             159 |             40 |
| close_to_lift     |           0 | closes_soon            | classification |  0.8235 |           0.8235 |               0.8235 |                     0.8235 | nan      |       nan      |           nan      |                 nan      |              86 |             34 |
| close_to_lift     |           0 | final_success          | classification |  0.5294 |           0.6471 |               0.5    |                     0.5    | nan      |       nan      |           nan      |                 nan      |              86 |             34 |
| close_to_lift     |           9 | future_lift_delta      | regression     |  0.0181 |          -0.4379 |              -0.5413 |                    -0.5413 |   0.0177 |         0.0242 |             0.0215 |                   0.0215 |              86 |             34 |
| close_to_lift     |           9 | future_min_target_dist | regression     |  0.3638 |          -0      |               0.0369 |                     0.0369 |   0.0674 |         0.0905 |             0.0797 |                   0.0797 |              86 |             34 |
| no_close          |          10 | future_lift_delta      | regression     |  0.4159 |          -0.1152 |               0.111  |                     0.111  |   0.0424 |         0.0587 |             0.058  |                   0.058  |              65 |             16 |
| no_close          |          10 | future_min_target_dist | regression     | -0.1742 |          -0.9006 |               0.3667 |                     0.3667 |   0.1185 |         0.1507 |             0.0658 |                   0.0658 |              65 |             16 |
| post_lift_or_late |          10 | closes_soon            | classification |  0.7583 |           0.7833 |               0.7833 |                     0.7833 | nan      |       nan      |           nan      |                 nan      |             481 |            120 |
| post_lift_or_late |           2 | final_success          | classification |  0.9167 |           0.925  |               0.925  |                     0.925  | nan      |       nan      |           nan      |                 nan      |             481 |            120 |
| post_lift_or_late |           5 | future_lift_delta      | regression     |  0.0203 |          -0      |               0.2793 |                     0.2793 |   0.0132 |         0.0082 |             0.0052 |                   0.0052 |             481 |            120 |
| post_lift_or_late |           7 | future_min_target_dist | regression     |  0.1212 |          -0.01   |               0.2064 |                     0.2064 |   0.1125 |         0.1212 |             0.113  |                   0.113  |             481 |            120 |
| pre_close         |           0 | closes_soon            | classification |  0.6667 |           0.7    |               0.7    |                     0.7    | nan      |       nan      |           nan      |                 nan      |             163 |             30 |
| pre_close         |           8 | final_success          | classification |  0.8333 |           0.7    |               0.9    |                     0.9    | nan      |       nan      |           nan      |                 nan      |             163 |             30 |
| pre_close         |          10 | future_lift_delta      | regression     |  0.2294 |          -0.026  |              -0.0073 |                    -0.0073 |   0.0117 |         0.0122 |             0.0107 |                   0.0107 |             163 |             30 |
| pre_close         |           8 | future_min_target_dist | regression     |  0.3596 |          -0.0008 |               0.3925 |                     0.3925 |   0.0968 |         0.1226 |             0.0822 |                   0.0822 |             163 |             30 |

## Interpretation Rule

A flow probe is only interesting when it beats object/object+phase priors, not merely the constant baseline. Otherwise the result may be benchmark scene bias or repeated task prior rather than online flow-state information.
