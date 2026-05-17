# Experiment 2A: Expert Attention Analysis

We summarize expert final-layer attention by flow step, collapsed into three coarse regions:

- image-prefix tokens
- text-prefix tokens
- suffix/action tokens

## By benchmark and flow step

| benchmark     |   flow_step |   image_prefix_attention |   text_prefix_attention |   suffix_attention |   attention_entropy |
|:--------------|------------:|-------------------------:|------------------------:|-------------------:|--------------------:|
| libero_90     |           0 |                   0.0614 |                  0.0513 |             0.8873 |              4.2392 |
| libero_90     |           1 |                   0.0622 |                  0.0507 |             0.8871 |              4.2504 |
| libero_90     |           2 |                   0.0603 |                  0.0521 |             0.8876 |              4.2432 |
| libero_90     |           3 |                   0.06   |                  0.053  |             0.8871 |              4.2443 |
| libero_90     |           4 |                   0.0624 |                  0.0546 |             0.883  |              4.258  |
| libero_90     |           5 |                   0.0607 |                  0.0556 |             0.8838 |              4.2516 |
| libero_90     |           6 |                   0.0615 |                  0.0574 |             0.8811 |              4.2583 |
| libero_90     |           7 |                   0.059  |                  0.0577 |             0.8833 |              4.2498 |
| libero_90     |           8 |                   0.056  |                  0.0562 |             0.8878 |              4.2415 |
| libero_90     |           9 |                   0.0439 |                  0.0472 |             0.9088 |              4.1964 |
| libero_object |           0 |                   0.055  |                  0.0593 |             0.8857 |              4.2086 |
| libero_object |           1 |                   0.0571 |                  0.0587 |             0.8842 |              4.2229 |
| libero_object |           2 |                   0.0559 |                  0.0601 |             0.8839 |              4.2165 |
| libero_object |           3 |                   0.0564 |                  0.0612 |             0.8824 |              4.2195 |
| libero_object |           4 |                   0.0595 |                  0.0626 |             0.8779 |              4.2341 |
| libero_object |           5 |                   0.0581 |                  0.0637 |             0.8782 |              4.2268 |
| libero_object |           6 |                   0.0593 |                  0.0657 |             0.875  |              4.2322 |
| libero_object |           7 |                   0.057  |                  0.0666 |             0.8764 |              4.2214 |
| libero_object |           8 |                   0.054  |                  0.0653 |             0.8807 |              4.2104 |
| libero_object |           9 |                   0.0417 |                  0.0547 |             0.9035 |              4.1654 |

## Scene 1 focus: cream_cheese vs ketchup

| object_label   | success   |   image_prefix_attention |   text_prefix_attention |   suffix_attention |   attention_entropy |
|:---------------|:----------|-------------------------:|------------------------:|-------------------:|--------------------:|
| cream_cheese   | False     |                   0.0573 |                  0.0508 |             0.8919 |              4.2398 |
| cream_cheese   | True      |                   0.0608 |                  0.0525 |             0.8867 |              4.247  |
| ketchup        | False     |                   0.0617 |                  0.0525 |             0.8858 |              4.264  |

## Scene 1 by task, success, and flow step (sample)

| object_label   | success   |   flow_step |   image_prefix_attention |   text_prefix_attention |   suffix_attention |   attention_entropy |
|:---------------|:----------|------------:|-------------------------:|------------------------:|-------------------:|--------------------:|
| alphabet_soup  | False     |           0 |                   0.0636 |                  0.0442 |             0.8922 |              4.2692 |
| alphabet_soup  | False     |           5 |                   0.0616 |                  0.0522 |             0.8862 |              4.2798 |
| alphabet_soup  | False     |           9 |                   0.0436 |                  0.0469 |             0.9095 |              4.2112 |
| alphabet_soup  | True      |           0 |                   0.0602 |                  0.0479 |             0.8919 |              4.2447 |
| alphabet_soup  | True      |           5 |                   0.0621 |                  0.0541 |             0.8838 |              4.2632 |
| alphabet_soup  | True      |           9 |                   0.0473 |                  0.0499 |             0.9027 |              4.2106 |
| cream_cheese   | False     |           0 |                   0.0589 |                  0.0477 |             0.8935 |              4.2327 |
| cream_cheese   | False     |           5 |                   0.0591 |                  0.0524 |             0.8885 |              4.2474 |
| cream_cheese   | False     |           9 |                   0.0436 |                  0.0468 |             0.9096 |              4.1972 |
| cream_cheese   | True      |           0 |                   0.0599 |                  0.0486 |             0.8915 |              4.2352 |
| cream_cheese   | True      |           5 |                   0.0628 |                  0.0539 |             0.8833 |              4.2548 |
| cream_cheese   | True      |           9 |                   0.0494 |                  0.0497 |             0.9009 |              4.2111 |
| ketchup        | False     |           0 |                   0.0679 |                  0.0504 |             0.8817 |              4.2718 |
| ketchup        | False     |           5 |                   0.0635 |                  0.0545 |             0.8819 |              4.2728 |
| ketchup        | False     |           9 |                   0.0437 |                  0.0461 |             0.9102 |              4.2025 |
| tomato_sauce   | False     |           0 |                   0.0565 |                  0.0577 |             0.8859 |              4.2026 |
| tomato_sauce   | False     |           5 |                   0.0565 |                  0.0597 |             0.8838 |              4.2199 |
| tomato_sauce   | False     |           9 |                   0.0403 |                  0.0461 |             0.9136 |              4.1739 |

## Interpretation guide

- Higher image-prefix attention in successful runs would support the idea that the expert is consulting scene information more effectively.
- Higher text-prefix attention with weak behavior would suggest the model keeps the target label available but may fail to bind it to the scene.
- Higher suffix attention can indicate stronger reliance on autoregressive / action-context refinement within the expert.
