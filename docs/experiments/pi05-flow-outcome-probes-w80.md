# PI0.5 Flow Outcome Probes

## Method

This predicts future manipulation variables from each saved flow state `flow_x_t`, using a future window of `80` environment steps.

Baselines are included to expose benchmark/scene bias: constant train prior, object-label prior, and object-label plus phase prior.

## Best Scores By Phase And Target

| phase             |   flow_step | target                 | probe_type     |   score |   constant_score |   object_prior_score |   object_phase_prior_score |      mae |   constant_mae |   object_prior_mae |   object_phase_prior_mae |   train_samples |   test_samples |
|:------------------|------------:|:-----------------------|:---------------|--------:|-----------------:|---------------------:|---------------------------:|---------:|---------------:|-------------------:|-------------------------:|----------------:|---------------:|
| all               |           9 | closes_soon            | classification |  0.6167 |           0.4958 |               0.5375 |                     0.5958 | nan      |       nan      |           nan      |                 nan      |             954 |            240 |
| all               |           8 | final_success          | classification |  0.7125 |           0.7    |               0.8833 |                     0.8625 | nan      |       nan      |           nan      |                 nan      |             954 |            240 |
| all               |           8 | future_lift_delta      | regression     | -0.6429 |          -0.0002 |               0.2071 |                     0.2808 |   0.0596 |         0.0398 |             0.0311 |                   0.0284 |             954 |            240 |
| all               |           8 | future_min_target_dist | regression     | -0.3036 |          -0.0004 |               0.3027 |                     0.4888 |   0.1358 |         0.1211 |             0.1    |                   0.0786 |             954 |            240 |
| call_00           |          10 | closes_soon            | classification |  0.6    |           0.375  |               0.475  |                     0.475  | nan      |       nan      |           nan      |                 nan      |             159 |             40 |
| call_00           |           9 | final_success          | classification |  0.725  |           0.6    |               0.9    |                     0.9    | nan      |       nan      |           nan      |                 nan      |             159 |             40 |
| call_00           |          10 | future_lift_delta      | regression     |  0.1973 |          -0.0036 |               0.4357 |                     0.4357 |   0.0409 |         0.0412 |             0.0246 |                   0.0246 |             159 |             40 |
| call_00           |          10 | future_min_target_dist | regression     |  0.1901 |          -0.0027 |               0.9329 |                     0.9329 |   0.0667 |         0.0802 |             0.0167 |                   0.0167 |             159 |             40 |
| close_to_lift     |           0 | closes_soon            | classification |  0.4412 |           0.4118 |               0.6176 |                     0.6176 | nan      |       nan      |           nan      |                 nan      |              86 |             34 |
| close_to_lift     |           0 | final_success          | classification |  0.5294 |           0.6471 |               0.5    |                     0.5    | nan      |       nan      |           nan      |                 nan      |              86 |             34 |
| close_to_lift     |           9 | future_lift_delta      | regression     |  0.1841 |          -0.0684 |               0.1246 |                     0.1246 |   0.0759 |         0.0948 |             0.0756 |                   0.0756 |              86 |             34 |
| close_to_lift     |           8 | future_min_target_dist | regression     |  0.0961 |          -0.0043 |               0.0496 |                     0.0496 |   0.062  |         0.0639 |             0.0621 |                   0.0621 |              86 |             34 |
| no_close          |           9 | future_lift_delta      | regression     |  0.3921 |          -0.0947 |               0.0283 |                     0.0283 |   0.0634 |         0.093  |             0.0951 |                   0.0951 |              65 |             16 |
| no_close          |          10 | future_min_target_dist | regression     | -0.5046 |          -2.119  |               0.1392 |                     0.1392 |   0.0915 |         0.1402 |             0.049  |                   0.049  |              65 |             16 |
| post_lift_or_late |           7 | closes_soon            | classification |  0.6917 |           0.55   |               0.5333 |                     0.5333 | nan      |       nan      |           nan      |                 nan      |             481 |            120 |
| post_lift_or_late |           2 | final_success          | classification |  0.9167 |           0.925  |               0.925  |                     0.925  | nan      |       nan      |           nan      |                 nan      |             481 |            120 |
| post_lift_or_late |           5 | future_lift_delta      | regression     |  0.0142 |          -0      |               0.2789 |                     0.2789 |   0.0134 |         0.0084 |             0.0053 |                   0.0053 |             481 |            120 |
| post_lift_or_late |           7 | future_min_target_dist | regression     |  0.0343 |          -0.0156 |               0.1867 |                     0.1867 |   0.1138 |         0.1148 |             0.1096 |                   0.1096 |             481 |            120 |
| pre_close         |          10 | closes_soon            | classification |  0.7667 |           0.7333 |               0.7667 |                     0.7667 | nan      |       nan      |           nan      |                 nan      |             163 |             30 |
| pre_close         |           8 | final_success          | classification |  0.8333 |           0.7    |               0.9    |                     0.9    | nan      |       nan      |           nan      |                 nan      |             163 |             30 |
| pre_close         |          10 | future_lift_delta      | regression     | -0.2336 |          -0.1868 |              -0.5536 |                    -0.5536 |   0.0404 |         0.0448 |             0.0368 |                   0.0368 |             163 |             30 |
| pre_close         |           9 | future_min_target_dist | regression     |  0.2572 |          -0.0081 |               0.4807 |                     0.4807 |   0.0992 |         0.1189 |             0.0716 |                   0.0716 |             163 |             30 |

## Interpretation Rule

A flow probe is only interesting when it beats object/object+phase priors, not merely the constant baseline. Otherwise the result may be benchmark scene bias or repeated task prior rather than online flow-state information.

## Current Readout

The object-prior controls are doing real work here. Several apparently useful flow readouts are weaker than object or object+phase priors, which means they are not safe evidence for online grounded use.

Main findings:

- `final_success` is mostly explained by object priors. For `call_00`, the best flow probe reaches `0.725`, but the object prior is `0.9`.
- `future_min_target_dist` is also heavily scene/object biased early. For `call_00`, the best flow probe reaches `R2 = 0.1901`, while the object prior reaches `R2 = 0.9329`.
- `closes_soon` is more promising in `call_00` and `post_lift_or_late`: flow beats object priors there, suggesting some phase/action signal beyond object identity.
- `future_lift_delta` is mixed. It beats object priors in `close_to_lift` and `no_close`, but not in `call_00` or pooled `all` calls.
- Pooled `all` results are mostly poor and should not be interpreted causally because mixed phases have different action semantics.

Interpretation:

> Benchmark scene/object priors explain a large fraction of outcome predictability. The useful flow-level signal is narrower: gripper/close timing and some phase-specific future-lift readouts, especially outside the early pooled setting.

This pushes the next experiment toward matched handoff/intervention comparisons rather than more generic success probes. The clean test is whether a donor handoff changes the phase-specific flow readout for close/lift while holding recipient scene/layout fixed.
