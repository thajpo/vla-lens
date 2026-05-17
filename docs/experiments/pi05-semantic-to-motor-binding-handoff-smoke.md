# PI0.5 Semantic-To-Motor Binding Handoff Smoke

## Question

Holding recipient scene/layout fixed, does changing handoff conditioning change target-relative trajectory formation?

This is the first narrow semantic-to-motor binding readout: not whether target information is decodable, but whether changed conditioning moves the generated action in robot-space coordinates.

## Condition Summary

| intervention_condition   |   calls | success   |   mean_final_margin |   mean_final_target_alignment |   mean_eef_delta_5_target_alignment |   mean_eef_delta_5_target_improvement |
|:-------------------------|--------:|:----------|--------------------:|------------------------------:|------------------------------------:|--------------------------------------:|
| baseline                 |       6 | True      |             -0.5022 |                       -0.0507 |                             -0.018  |                               -0.0031 |
| failure_donor_swap       |       7 | False     |             -0.5099 |                        0.0827 |                              0.217  |                                0.0047 |
| success_donor_swap       |       6 | True      |             -0.4896 |                        0.0295 |                              0.0804 |                               -0.0032 |

## Deltas Versus Baseline

| intervention_condition   |   matched_calls |   mean_delta_final_margin |   mean_delta_target_alignment |   mean_delta_eef_delta_5_target_alignment |   mean_delta_eef_delta_5_target_improvement |   max_abs_delta_final_margin |
|:-------------------------|----------------:|--------------------------:|------------------------------:|------------------------------------------:|--------------------------------------------:|-----------------------------:|
| failure_donor_swap       |               6 |                   -0.1565 |                        0.0033 |                                    0.1305 |                                      0.0034 |                       1.2662 |
| success_donor_swap       |               6 |                    0.0126 |                        0.0802 |                                    0.0984 |                                     -0.0001 |                       0.2689 |

## First Divergence By Condition

| intervention_condition   |   first_divergent_call |   delta_final_margin |   delta_eef_delta_5_target_improvement |
|:-------------------------|-----------------------:|---------------------:|---------------------------------------:|
| failure_donor_swap       |                      1 |               0.0204 |                                 0      |
| success_donor_swap       |                      1 |              -0.0344 |                                 0.0009 |

## Current Interpretation

The first policy call is identical across conditions, which is a useful control: the recipient scene and early action state are fixed.

The failure-donor condition does not show a clean early wrong-target redirection relative to baseline. Instead, the major behavioral difference remains downstream: baseline and success-donor lift the target, while failure-donor does not.

This supports the current hypothesis that the handoff-sensitive failure is not simply missing semantic target information. It is more likely a later semantic-to-motor binding, gripper/lift, recovery, or trajectory-mode issue.

## Next Test

Run the same report on more recipient-fixed handoff captures, especially layouts where donor handoff flips target lift. Then add token/layer-restricted patches and score the same object-relative deltas.

## Output

- `artifacts/pi05_analysis/binding_interventions/handoff_smoke_call_deltas.csv`
