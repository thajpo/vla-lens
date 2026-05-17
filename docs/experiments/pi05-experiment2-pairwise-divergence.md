# Experiment 2H: Pairwise Success/Failure Divergence

This forward-only analysis compares paired same-task same-layout success/failure rollouts in Scene 1 to localize where their representations diverge.

## Paired layouts

| object_label   |   layout | success_rollout_id               | success_path                                             | failure_rollout_id               | failure_path                                             |   matched_calls |
|:---------------|---------:|:---------------------------------|:---------------------------------------------------------|:---------------------------------|:---------------------------------------------------------|----------------:|
| alphabet_soup  |        1 | 0fa7a54f180a4f07a369058c4bfec129 | artifacts/pi05_captures/0fa7a54f180a4f07a369058c4bfec129 | 1a3574d1fcc64c139a388f2d581860cb | artifacts/pi05_captures/1a3574d1fcc64c139a388f2d581860cb |               4 |
| alphabet_soup  |        8 | affb5529acb34af4a1a27e13abe64d14 | artifacts/pi05_captures/affb5529acb34af4a1a27e13abe64d14 | 23195fcdc7324c8f9403abddaac11f95 | artifacts/pi05_captures/23195fcdc7324c8f9403abddaac11f95 |               3 |
| alphabet_soup  |       10 | 338651b0a079462d96847a768be9a73e | artifacts/pi05_captures/338651b0a079462d96847a768be9a73e | 0bb52e7418424e9cb75d3caeb97a43fc | artifacts/pi05_captures/0bb52e7418424e9cb75d3caeb97a43fc |               5 |
| alphabet_soup  |       17 | 802726d85c2a4153ac8cfeae7b84897c | artifacts/pi05_captures/802726d85c2a4153ac8cfeae7b84897c | f3c8335a45e046acbd46c7ea4bf1e6d1 | artifacts/pi05_captures/f3c8335a45e046acbd46c7ea4bf1e6d1 |               4 |
| alphabet_soup  |       19 | ed3f986261994ee99eb11259587fa65a | artifacts/pi05_captures/ed3f986261994ee99eb11259587fa65a | ee581c8fa8c646b18223a3764056f86a | artifacts/pi05_captures/ee581c8fa8c646b18223a3764056f86a |               3 |
| alphabet_soup  |       23 | 62af635c2a684cb99e4b43fb5d30b45a | artifacts/pi05_captures/62af635c2a684cb99e4b43fb5d30b45a | de140580e86449b6a505f2f0ef431401 | artifacts/pi05_captures/de140580e86449b6a505f2f0ef431401 |               3 |
| alphabet_soup  |       26 | 6185a39275df4de5b3fc3a46b87456a1 | artifacts/pi05_captures/6185a39275df4de5b3fc3a46b87456a1 | 4c696700991b4c8ba040939b20b00f22 | artifacts/pi05_captures/4c696700991b4c8ba040939b20b00f22 |               3 |
| alphabet_soup  |       27 | 453d25354d6b4cf4b4ce63beb0f6d4ac | artifacts/pi05_captures/453d25354d6b4cf4b4ce63beb0f6d4ac | 30ae217af5e541ba99635f5c73c4dfa1 | artifacts/pi05_captures/30ae217af5e541ba99635f5c73c4dfa1 |               3 |
| cream_cheese   |        3 | 22c14d6566754553a00e31fc4cc6bfdc | artifacts/pi05_captures/22c14d6566754553a00e31fc4cc6bfdc | cf104831f7294de2bc529e55f8cf7c12 | artifacts/pi05_captures/cf104831f7294de2bc529e55f8cf7c12 |               6 |
| cream_cheese   |        4 | eee796ed0f5c49e297592659ad4c576d | artifacts/pi05_captures/eee796ed0f5c49e297592659ad4c576d | 940001f901e244e7bb2ef5bcd2ed0b73 | artifacts/pi05_captures/940001f901e244e7bb2ef5bcd2ed0b73 |               5 |
| cream_cheese   |        6 | f8e31ce6303349b2be2c9cd75413fc4f | artifacts/pi05_captures/f8e31ce6303349b2be2c9cd75413fc4f | d065810f635148358923e2f556a79f52 | artifacts/pi05_captures/d065810f635148358923e2f556a79f52 |               5 |
| cream_cheese   |       18 | 4d8d87cc399b43739bd434b98bf9e9ea | artifacts/pi05_captures/4d8d87cc399b43739bd434b98bf9e9ea | 97b375761f3141e4b5c57c7d7e568100 | artifacts/pi05_captures/97b375761f3141e4b5c57c7d7e568100 |               5 |
| cream_cheese   |       22 | e1fb9ebd6f0748378c34d3cf32a6e171 | artifacts/pi05_captures/e1fb9ebd6f0748378c34d3cf32a6e171 | b51c7e6e79034531adc611b6d1800193 | artifacts/pi05_captures/b51c7e6e79034531adc611b6d1800193 |               6 |
| cream_cheese   |       26 | 1ab962805dc24cb79d4dfcc23ae87121 | artifacts/pi05_captures/1ab962805dc24cb79d4dfcc23ae87121 | 70ff65bf555a42a8991da37181e119e7 | artifacts/pi05_captures/70ff65bf555a42a8991da37181e119e7 |               5 |
| cream_cheese   |       27 | bf68a8ace3e346708448cdb0c6e81f29 | artifacts/pi05_captures/bf68a8ace3e346708448cdb0c6e81f29 | 82f183cdacfd491aa5cb88f7a7a46e9e | artifacts/pi05_captures/82f183cdacfd491aa5cb88f7a7a46e9e |               6 |

## Representation similarity at final flow step

| object_label   | representation             |   cosine_similarity |
|:---------------|:---------------------------|--------------------:|
| alphabet_soup  | expert_layer_4_final_flow  |              0.95   |
| alphabet_soup  | expert_layer_0_final_flow  |              0.9516 |
| alphabet_soup  | expert_layer_16_final_flow |              0.9638 |
| alphabet_soup  | expert_layer_17_final_flow |              0.9654 |
| alphabet_soup  | expert_layer_8_final_flow  |              0.9802 |
| alphabet_soup  | vlm_handoff_final          |              0.983  |
| alphabet_soup  | expert_layer_12_final_flow |              0.9848 |
| alphabet_soup  | vlm_final                  |              0.9963 |
| cream_cheese   | expert_layer_4_final_flow  |              0.9327 |
| cream_cheese   | expert_layer_0_final_flow  |              0.9365 |
| cream_cheese   | expert_layer_16_final_flow |              0.9579 |
| cream_cheese   | expert_layer_17_final_flow |              0.9583 |
| cream_cheese   | expert_layer_8_final_flow  |              0.9728 |
| cream_cheese   | expert_layer_12_final_flow |              0.9802 |
| cream_cheese   | vlm_handoff_final          |              0.9849 |
| cream_cheese   | vlm_final                  |              0.996  |

## Final-layer flow-step divergence

| object_label   |   flow_step |   hidden_cosine |   hidden_mse |
|:---------------|------------:|----------------:|-------------:|
| alphabet_soup  |           0 |          0.562  |       0.2009 |
| alphabet_soup  |           1 |          0.5612 |       0.2074 |
| alphabet_soup  |           2 |          0.5611 |       0.2142 |
| alphabet_soup  |           3 |          0.5694 |       0.225  |
| alphabet_soup  |           4 |          0.5845 |       0.2389 |
| alphabet_soup  |           5 |          0.6049 |       0.2581 |
| alphabet_soup  |           6 |          0.618  |       0.285  |
| alphabet_soup  |           7 |          0.6344 |       0.3236 |
| alphabet_soup  |           8 |          0.6516 |       0.3892 |
| alphabet_soup  |           9 |          0.674  |       0.4864 |
| cream_cheese   |           0 |          0.4907 |       0.2038 |
| cream_cheese   |           1 |          0.4951 |       0.2103 |
| cream_cheese   |           2 |          0.4997 |       0.2174 |
| cream_cheese   |           3 |          0.512  |       0.2286 |
| cream_cheese   |           4 |          0.5265 |       0.2432 |
| cream_cheese   |           5 |          0.543  |       0.2635 |
| cream_cheese   |           6 |          0.5495 |       0.292  |
| cream_cheese   |           7 |          0.5637 |       0.3335 |
| cream_cheese   |           8 |          0.5745 |       0.4049 |
| cream_cheese   |           9 |          0.5915 |       0.5114 |

## Final-layer attention divergence

| object_label   |   flow_step |   attention_js |   attention_cosine |
|:---------------|------------:|---------------:|-------------------:|
| alphabet_soup  |           0 |         0.0171 |             0.9762 |
| alphabet_soup  |           1 |         0.0169 |             0.9764 |
| alphabet_soup  |           2 |         0.0166 |             0.9766 |
| alphabet_soup  |           3 |         0.0164 |             0.9768 |
| alphabet_soup  |           4 |         0.0166 |             0.9764 |
| alphabet_soup  |           5 |         0.0162 |             0.9767 |
| alphabet_soup  |           6 |         0.016  |             0.9771 |
| alphabet_soup  |           7 |         0.0154 |             0.9778 |
| alphabet_soup  |           8 |         0.0148 |             0.978  |
| alphabet_soup  |           9 |         0.0127 |             0.9803 |
| cream_cheese   |           0 |         0.0177 |             0.9717 |
| cream_cheese   |           1 |         0.0176 |             0.9728 |
| cream_cheese   |           2 |         0.0173 |             0.9728 |
| cream_cheese   |           3 |         0.0173 |             0.973  |
| cream_cheese   |           4 |         0.0175 |             0.9732 |
| cream_cheese   |           5 |         0.0172 |             0.9737 |
| cream_cheese   |           6 |         0.0171 |             0.9742 |
| cream_cheese   |           7 |         0.0168 |             0.9748 |
| cream_cheese   |           8 |         0.0163 |             0.9756 |
| cream_cheese   |           9 |         0.0146 |             0.9774 |

## Final action divergence

| object_label   |   final_action_mse |   final_action_cosine |   gripper_flow_mse |
|:---------------|-------------------:|----------------------:|-------------------:|
| alphabet_soup  |             0.139  |                0.5572 |             0.9414 |
| cream_cheese   |             0.1594 |                0.4822 |             1.0563 |
