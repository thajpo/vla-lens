# PI0.5 Phase-Aware Trajectory Analysis

## Method

This analysis uses saved `state_trajectory.npz` and `actions.npz` to summarize physical trajectory phases, rather than probing hidden states or interpreting a single flow vector.

Phase proxies:

- approach: early and pre-close target-distance reduction
- close/grasp timing: first gripper-close step and target distance at close
- lift: target max lift and first target-lift step
- recovery: success despite non-target first move/lift or bad early alignment

## Task Summary

| benchmark   | object_label   | intervention_condition   |   episodes |   success_rate |   target_first_moved_rate |   target_first_lifted_rate |   mean_initial_target_distance |   mean_min_target_distance |   mean_final_target_distance |   mean_early_distance_delta |   mean_preclose_distance_delta |   mean_postclose_distance_delta |   mean_first_close_step |   mean_target_distance_at_close |   mean_target_max_lift |   mean_final_target_lift |   close_before_near_target_rate |   mean_early_alignment |   mean_preclose_alignment |   mean_postclose_alignment |
|:------------|:---------------|:-------------------------|-----------:|---------------:|--------------------------:|---------------------------:|-------------------------------:|---------------------------:|-----------------------------:|----------------------------:|-------------------------------:|--------------------------------:|------------------------:|--------------------------------:|-----------------------:|-------------------------:|--------------------------------:|-----------------------:|--------------------------:|---------------------------:|
| libero_90   | cream_cheese   | baseline                 |          1 |              1 |                         0 |                          0 |                         0.2563 |                     0.0174 |                       0.2493 |                     -0.0352 |                         0.0006 |                         -0.0533 |                       6 |                          0.2558 |                 0.243  |                        0 |                               1 |                -0.1103 |                    0.2193 |                    -0.0539 |
| libero_90   | cream_cheese   | failure_donor_swap       |          1 |              0 |                         0 |                          0 |                         0.2563 |                     0.0271 |                       0.036  |                     -0.0352 |                         0.0006 |                         -0.0533 |                       6 |                          0.2558 |                 0.0002 |                        0 |                               1 |                -0.1103 |                    0.2193 |                    -0.0539 |
| libero_90   | cream_cheese   | success_donor_swap       |          1 |              1 |                         0 |                          0 |                         0.2563 |                     0.014  |                       0.2493 |                     -0.0352 |                         0.0006 |                         -0.0533 |                       6 |                          0.2558 |                 0.2227 |                        0 |                               1 |                -0.1103 |                    0.2193 |                    -0.0539 |

## Success/Failure Summary

| benchmark   | object_label   | intervention_condition   | success   |   episodes |   success_rate |   target_first_moved_rate |   target_first_lifted_rate |   mean_initial_target_distance |   mean_min_target_distance |   mean_final_target_distance |   mean_early_distance_delta |   mean_preclose_distance_delta |   mean_postclose_distance_delta |   mean_first_close_step |   mean_target_distance_at_close |   mean_target_max_lift |   mean_final_target_lift |   close_before_near_target_rate |   mean_early_alignment |   mean_preclose_alignment |   mean_postclose_alignment |
|:------------|:---------------|:-------------------------|:----------|-----------:|---------------:|--------------------------:|---------------------------:|-------------------------------:|---------------------------:|-----------------------------:|----------------------------:|-------------------------------:|--------------------------------:|------------------------:|--------------------------------:|-----------------------:|-------------------------:|--------------------------------:|-----------------------:|--------------------------:|---------------------------:|
| libero_90   | cream_cheese   | baseline                 | True      |          1 |              1 |                         0 |                          0 |                         0.2563 |                     0.0174 |                       0.2493 |                     -0.0352 |                         0.0006 |                         -0.0533 |                       6 |                          0.2558 |                 0.243  |                        0 |                               1 |                -0.1103 |                    0.2193 |                    -0.0539 |
| libero_90   | cream_cheese   | failure_donor_swap       | False     |          1 |              0 |                         0 |                          0 |                         0.2563 |                     0.0271 |                       0.036  |                     -0.0352 |                         0.0006 |                         -0.0533 |                       6 |                          0.2558 |                 0.0002 |                        0 |                               1 |                -0.1103 |                    0.2193 |                    -0.0539 |
| libero_90   | cream_cheese   | success_donor_swap       | True      |          1 |              1 |                         0 |                          0 |                         0.2563 |                     0.014  |                       0.2493 |                     -0.0352 |                         0.0006 |                         -0.0533 |                       6 |                          0.2558 |                 0.2227 |                        0 |                               1 |                -0.1103 |                    0.2193 |                    -0.0539 |

## Same-Layout Success/Failure Pairs

| object_label   |   layout_episode_index | success_rollout_id               | failure_rollout_id               |   success_min_target_distance |   failure_min_target_distance |   delta_min_target_distance_success_minus_failure |   success_target_max_lift |   failure_target_max_lift |   delta_target_max_lift_success_minus_failure |   success_first_close_step |   failure_first_close_step |   success_distance_at_close |   failure_distance_at_close | success_close_before_near_target   | failure_close_before_near_target   |
|:---------------|-----------------------:|:---------------------------------|:---------------------------------|------------------------------:|------------------------------:|--------------------------------------------------:|--------------------------:|--------------------------:|----------------------------------------------:|---------------------------:|---------------------------:|----------------------------:|----------------------------:|:-----------------------------------|:-----------------------------------|
| cream_cheese   |                      3 | da89968352c04481a96697b57e82cd66 | 8df7de3b63fe4bda850b4dcdfbddc760 |                        0.0174 |                        0.0271 |                                           -0.0097 |                     0.243 |                    0.0002 |                                        0.2428 |                          6 |                          6 |                      0.2558 |                      0.2558 | True                               | True                               |

## Main Readout

This captured intervention run is the clearest phase-level result so far.

Conditions:

- `baseline`: success, target max lift `0.2430`
- `success_donor_swap`: success, target max lift `0.2227`
- `failure_donor_swap`: failure, target max lift `0.0002`

The early phase metrics are nearly identical across all three conditions:

- initial target distance: `0.2563`
- early distance delta: `-0.0352`
- pre-close distance delta: `0.0006`
- first close step: `6`
- target distance at close: `0.2558`
- early/preclose/postclose alignments: identical at the rounded precision shown

The decisive split is not initial approach. The split is target lift / manipulation outcome:

- baseline and success-donor eventually lift the target substantially
- failure-donor does not lift the target at all

Interpretation:

> The donor handoff can causally flip the outcome without changing the coarse early approach/close phase. For this `cream_cheese` layout 3 case, the handoff-sensitive computation appears to affect later manipulation/lift/recovery rather than first-object approach.

This supports the broader phase-aware reframe: causal intervention readouts need to inspect lift and recovery dynamics, not just target-vs-distractor initial motion.

## Output Files

- `artifacts/pi05_analysis/phase_trajectory/phase_trajectory_rollouts.csv`
- `artifacts/pi05_analysis/phase_trajectory/phase_trajectory_success_summary.csv`
- `artifacts/pi05_analysis/phase_trajectory/phase_trajectory_task_summary.csv`
- `artifacts/pi05_analysis/phase_trajectory/phase_trajectory_pairwise_success_failure.csv`
