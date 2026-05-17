# PI0.5 Flow Probe Evolution

## Method

This retrains lightweight probes directly on the saved flow-matching output `flow_x_t` at each denoising step. It does not reuse earlier probe models, because those scripts saved CSV summaries rather than fitted estimators.

The purpose is to ask what information is present in the action trajectory itself as denoising evolves.

## Classification Probes

|   flow_step | probe_type     | target          | dim   |   train_samples |   val_samples |   test_samples |   best_param |   score |   baseline_score |   mae |   baseline_mae |
|------------:|:---------------|:----------------|:------|----------------:|--------------:|---------------:|-------------:|--------:|-----------------:|------:|---------------:|
|           0 | classification | target_identity | label |             159 |            40 |             40 |         0.01 |   0.2   |             0.25 |   nan |            nan |
|           0 | classification | success         | label |             159 |            40 |             40 |         0.1  |   0.45  |             0.6  |   nan |            nan |
|           1 | classification | target_identity | label |             159 |            40 |             40 |         0.01 |   0.175 |             0.25 |   nan |            nan |
|           1 | classification | success         | label |             159 |            40 |             40 |         0.1  |   0.45  |             0.6  |   nan |            nan |
|           2 | classification | target_identity | label |             159 |            40 |             40 |         1    |   0.2   |             0.25 |   nan |            nan |
|           2 | classification | success         | label |             159 |            40 |             40 |         0.01 |   0.475 |             0.6  |   nan |            nan |
|           3 | classification | target_identity | label |             159 |            40 |             40 |         0.1  |   0.2   |             0.25 |   nan |            nan |
|           3 | classification | success         | label |             159 |            40 |             40 |         0.01 |   0.475 |             0.6  |   nan |            nan |
|           4 | classification | target_identity | label |             159 |            40 |             40 |         0.1  |   0.25  |             0.25 |   nan |            nan |
|           4 | classification | success         | label |             159 |            40 |             40 |         0.01 |   0.475 |             0.6  |   nan |            nan |
|           5 | classification | target_identity | label |             159 |            40 |             40 |         0.01 |   0.325 |             0.25 |   nan |            nan |
|           5 | classification | success         | label |             159 |            40 |             40 |         0.1  |   0.45  |             0.6  |   nan |            nan |
|           6 | classification | target_identity | label |             159 |            40 |             40 |         1    |   0.45  |             0.25 |   nan |            nan |
|           6 | classification | success         | label |             159 |            40 |             40 |         0.01 |   0.6   |             0.6  |   nan |            nan |
|           7 | classification | target_identity | label |             159 |            40 |             40 |         0.01 |   0.6   |             0.25 |   nan |            nan |
|           7 | classification | success         | label |             159 |            40 |             40 |         0.01 |   0.625 |             0.6  |   nan |            nan |
|           8 | classification | target_identity | label |             159 |            40 |             40 |         0.01 |   0.7   |             0.25 |   nan |            nan |
|           8 | classification | success         | label |             159 |            40 |             40 |         0.01 |   0.65  |             0.6  |   nan |            nan |
|           9 | classification | target_identity | label |             159 |            40 |             40 |        10    |   0.725 |             0.25 |   nan |            nan |
|           9 | classification | success         | label |             159 |            40 |             40 |         0.01 |   0.725 |             0.6  |   nan |            nan |
|          10 | classification | target_identity | label |             159 |            40 |             40 |         0.01 |   0.85  |             0.25 |   nan |            nan |
|          10 | classification | success         | label |             159 |            40 |             40 |         1    |   0.725 |             0.6  |   nan |            nan |

## Selected Regression Probes

|   flow_step | probe_type   | target            | dim    |   train_samples |   val_samples |   test_samples |   best_param |   score |   baseline_score |    mae |   baseline_mae |
|------------:|:-------------|:------------------|:-------|----------------:|--------------:|---------------:|-------------:|--------:|-----------------:|-------:|---------------:|
|           0 | regression   | target_pos        | x      |             159 |            40 |             40 |         0.01 | -0.2111 |          -0.0006 | 0.1277 |         0.1222 |
|           0 | regression   | target_pos        | y      |             159 |            40 |             40 |       100    | -0.2847 |          -0.0021 | 0.0916 |         0.0769 |
|           0 | regression   | target_pos        | z      |             159 |            40 |             40 |         0.01 |  0.0376 |          -0      | 0.1774 |         0.1921 |
|           0 | regression   | target_to_gripper | x      |             159 |            40 |             40 |         0.01 | -0.2244 |          -0.0006 | 0.1268 |         0.1207 |
|           0 | regression   | target_to_gripper | y      |             159 |            40 |             40 |       100    | -0.3073 |          -0.0016 | 0.0896 |         0.0743 |
|           0 | regression   | target_to_gripper | z      |             159 |            40 |             40 |         0.01 |  0.0488 |          -0      | 0.1767 |         0.194  |
|           0 | regression   | target_max_lift   | scalar |             159 |            40 |             40 |       100    | -0.2618 |          -0.0004 | 0.1083 |         0.101  |
|           1 | regression   | target_pos        | x      |             159 |            40 |             40 |         0.01 | -0.2224 |          -0.0006 | 0.1281 |         0.1222 |
|           1 | regression   | target_pos        | y      |             159 |            40 |             40 |       100    | -0.28   |          -0.0021 | 0.0914 |         0.0769 |
|           1 | regression   | target_pos        | z      |             159 |            40 |             40 |         0.01 |  0.0496 |          -0      | 0.1768 |         0.1921 |
|           1 | regression   | target_to_gripper | x      |             159 |            40 |             40 |         0.01 | -0.2363 |          -0.0006 | 0.1272 |         0.1207 |
|           1 | regression   | target_to_gripper | y      |             159 |            40 |             40 |       100    | -0.3027 |          -0.0016 | 0.0894 |         0.0743 |
|           1 | regression   | target_to_gripper | z      |             159 |            40 |             40 |         0.01 |  0.061  |          -0      | 0.1761 |         0.194  |
|           1 | regression   | target_max_lift   | scalar |             159 |            40 |             40 |       100    | -0.257  |          -0.0004 | 0.1081 |         0.101  |
|           2 | regression   | target_pos        | x      |             159 |            40 |             40 |         0.01 | -0.2236 |          -0.0006 | 0.1279 |         0.1222 |
|           2 | regression   | target_pos        | y      |             159 |            40 |             40 |       100    | -0.2681 |          -0.0021 | 0.0908 |         0.0769 |
|           2 | regression   | target_pos        | z      |             159 |            40 |             40 |         0.01 |  0.0796 |          -0      | 0.1743 |         0.1921 |
|           2 | regression   | target_to_gripper | x      |             159 |            40 |             40 |         0.01 | -0.2376 |          -0.0006 | 0.1269 |         0.1207 |
|           2 | regression   | target_to_gripper | y      |             159 |            40 |             40 |       100    | -0.2906 |          -0.0016 | 0.0889 |         0.0743 |
|           2 | regression   | target_to_gripper | z      |             159 |            40 |             40 |         0.01 |  0.0909 |          -0      | 0.1737 |         0.194  |
|           2 | regression   | target_max_lift   | scalar |             159 |            40 |             40 |         0.01 | -0.2757 |          -0.0004 | 0.108  |         0.101  |
|           3 | regression   | target_pos        | x      |             159 |            40 |             40 |         0.01 | -0.2061 |          -0.0006 | 0.1267 |         0.1222 |
|           3 | regression   | target_pos        | y      |             159 |            40 |             40 |       100    | -0.245  |          -0.0021 | 0.0898 |         0.0769 |
|           3 | regression   | target_pos        | z      |             159 |            40 |             40 |         0.01 |  0.1365 |          -0      | 0.1696 |         0.1921 |
|           3 | regression   | target_to_gripper | x      |             159 |            40 |             40 |         0.01 | -0.219  |          -0.0006 | 0.1255 |         0.1207 |
|           3 | regression   | target_to_gripper | y      |             159 |            40 |             40 |       100    | -0.2672 |          -0.0016 | 0.088  |         0.0743 |
|           3 | regression   | target_to_gripper | z      |             159 |            40 |             40 |         0.01 |  0.1469 |          -0      | 0.1689 |         0.194  |
|           3 | regression   | target_max_lift   | scalar |             159 |            40 |             40 |         0.01 | -0.2541 |          -0.0004 | 0.1071 |         0.101  |
|           4 | regression   | target_pos        | x      |             159 |            40 |             40 |         0.01 | -0.1592 |          -0.0006 | 0.1237 |         0.1222 |
|           4 | regression   | target_pos        | y      |             159 |            40 |             40 |       100    | -0.2071 |          -0.0021 | 0.0881 |         0.0769 |
|           4 | regression   | target_pos        | z      |             159 |            40 |             40 |         0.01 |  0.2243 |          -0      | 0.1609 |         0.1921 |
|           4 | regression   | target_to_gripper | x      |             159 |            40 |             40 |         0.01 | -0.1695 |          -0.0006 | 0.1223 |         0.1207 |
|           4 | regression   | target_to_gripper | y      |             159 |            40 |             40 |       100    | -0.2288 |          -0.0016 | 0.0864 |         0.0743 |
|           4 | regression   | target_to_gripper | z      |             159 |            40 |             40 |         0.01 |  0.2331 |          -0      | 0.1604 |         0.194  |
|           4 | regression   | target_max_lift   | scalar |             159 |            40 |             40 |         0.01 | -0.2214 |          -0.0004 | 0.1056 |         0.101  |
|           5 | regression   | target_pos        | x      |             159 |            40 |             40 |         0.01 | -0.0768 |          -0.0006 | 0.1181 |         0.1222 |
|           5 | regression   | target_pos        | y      |             159 |            40 |             40 |       100    | -0.1502 |          -0.0021 | 0.0856 |         0.0769 |
|           5 | regression   | target_pos        | z      |             159 |            40 |             40 |         0.01 |  0.3351 |          -0      | 0.1476 |         0.1921 |
|           5 | regression   | target_to_gripper | x      |             159 |            40 |             40 |         0.01 | -0.0827 |          -0.0006 | 0.1166 |         0.1207 |
|           5 | regression   | target_to_gripper | y      |             159 |            40 |             40 |       100    | -0.1712 |          -0.0016 | 0.084  |         0.0743 |
|           5 | regression   | target_to_gripper | z      |             159 |            40 |             40 |         0.01 |  0.3418 |          -0      | 0.1476 |         0.194  |
|           5 | regression   | target_max_lift   | scalar |             159 |            40 |             40 |         0.01 | -0.1765 |          -0.0004 | 0.1037 |         0.101  |
|           6 | regression   | target_pos        | x      |             159 |            40 |             40 |         0.01 |  0.0299 |          -0.0006 | 0.1104 |         0.1222 |
|           6 | regression   | target_pos        | y      |             159 |            40 |             40 |       100    | -0.0734 |          -0.0021 | 0.0822 |         0.0769 |
|           6 | regression   | target_pos        | z      |             159 |            40 |             40 |         0.01 |  0.4449 |          -0      | 0.1326 |         0.1921 |
|           6 | regression   | target_to_gripper | x      |             159 |            40 |             40 |         0.01 |  0.0298 |          -0.0006 | 0.1087 |         0.1207 |
|           6 | regression   | target_to_gripper | y      |             159 |            40 |             40 |       100    | -0.093  |          -0.0016 | 0.0807 |         0.0743 |
|           6 | regression   | target_to_gripper | z      |             159 |            40 |             40 |         0.01 |  0.4495 |          -0      | 0.1344 |         0.194  |
|           6 | regression   | target_max_lift   | scalar |             159 |            40 |             40 |        10    | -0.1168 |          -0.0004 | 0.1011 |         0.101  |
|           7 | regression   | target_pos        | x      |             159 |            40 |             40 |         0.01 |  0.1362 |          -0.0006 | 0.1022 |         0.1222 |
|           7 | regression   | target_pos        | y      |             159 |            40 |             40 |       100    |  0.0266 |          -0.0021 | 0.078  |         0.0769 |
|           7 | regression   | target_pos        | z      |             159 |            40 |             40 |         0.01 |  0.5369 |          -0      | 0.1202 |         0.1921 |
|           7 | regression   | target_to_gripper | x      |             159 |            40 |             40 |         0.01 |  0.1424 |          -0.0006 | 0.1003 |         0.1207 |
|           7 | regression   | target_to_gripper | y      |             159 |            40 |             40 |       100    |  0.0106 |          -0.0016 | 0.0762 |         0.0743 |
|           7 | regression   | target_to_gripper | z      |             159 |            40 |             40 |         0.01 |  0.5394 |          -0      | 0.1225 |         0.194  |
|           7 | regression   | target_max_lift   | scalar |             159 |            40 |             40 |       100    | -0.0254 |          -0.0004 | 0.0969 |         0.101  |
|           8 | regression   | target_pos        | x      |             159 |            40 |             40 |         0.01 |  0.2203 |          -0.0006 | 0.0973 |         0.1222 |
|           8 | regression   | target_pos        | y      |             159 |            40 |             40 |         0.01 |  0.129  |          -0.0021 | 0.0736 |         0.0769 |
|           8 | regression   | target_pos        | z      |             159 |            40 |             40 |         0.01 |  0.6137 |          -0      | 0.1111 |         0.1921 |
|           8 | regression   | target_to_gripper | x      |             159 |            40 |             40 |         0.01 |  0.232  |          -0.0006 | 0.0947 |         0.1207 |
|           8 | regression   | target_to_gripper | y      |             159 |            40 |             40 |         0.01 |  0.118  |          -0.0016 | 0.0712 |         0.0743 |
|           8 | regression   | target_to_gripper | z      |             159 |            40 |             40 |         0.01 |  0.6142 |          -0      | 0.112  |         0.194  |
|           8 | regression   | target_max_lift   | scalar |             159 |            40 |             40 |       100    |  0.0736 |          -0.0004 | 0.0916 |         0.101  |
|           9 | regression   | target_pos        | x      |             159 |            40 |             40 |         0.01 |  0.293  |          -0.0006 | 0.0935 |         0.1222 |
|           9 | regression   | target_pos        | y      |             159 |            40 |             40 |         0.01 |  0.2565 |          -0.0021 | 0.0684 |         0.0769 |
|           9 | regression   | target_pos        | z      |             159 |            40 |             40 |         0.01 |  0.6853 |          -0      | 0.1017 |         0.1921 |
|           9 | regression   | target_to_gripper | x      |             159 |            40 |             40 |         0.01 |  0.3075 |          -0.0006 | 0.0909 |         0.1207 |
|           9 | regression   | target_to_gripper | y      |             159 |            40 |             40 |         0.01 |  0.2542 |          -0.0016 | 0.0666 |         0.0743 |
|           9 | regression   | target_to_gripper | z      |             159 |            40 |             40 |         0.01 |  0.6839 |          -0      | 0.103  |         0.194  |
|           9 | regression   | target_max_lift   | scalar |             159 |            40 |             40 |       100    |  0.187  |          -0.0004 | 0.0845 |         0.101  |
|          10 | regression   | target_pos        | x      |             159 |            40 |             40 |         0.01 |  0.4275 |          -0.0006 | 0.0811 |         0.1222 |
|          10 | regression   | target_pos        | y      |             159 |            40 |             40 |         0.01 |  0.3854 |          -0.0021 | 0.0608 |         0.0769 |
|          10 | regression   | target_pos        | z      |             159 |            40 |             40 |         0.01 |  0.7321 |          -0      | 0.0948 |         0.1921 |
|          10 | regression   | target_to_gripper | x      |             159 |            40 |             40 |         0.01 |  0.4435 |          -0.0006 | 0.0787 |         0.1207 |
|          10 | regression   | target_to_gripper | y      |             159 |            40 |             40 |         0.01 |  0.3888 |          -0.0016 | 0.0586 |         0.0743 |
|          10 | regression   | target_to_gripper | z      |             159 |            40 |             40 |         0.01 |  0.7295 |          -0      | 0.0959 |         0.194  |
|          10 | regression   | target_max_lift   | scalar |             159 |            40 |             40 |         0.01 |  0.2523 |          -0.0004 | 0.0791 |         0.101  |

## Best Steps By Target

|   flow_step | probe_type     | target            | dim    |   train_samples |   val_samples |   test_samples |   best_param |   score |   baseline_score |      mae |   baseline_mae |
|------------:|:---------------|:------------------|:-------|----------------:|--------------:|---------------:|-------------:|--------:|-----------------:|---------:|---------------:|
|           9 | classification | success           | label  |             159 |            40 |             40 |         0.01 |  0.725  |           0.6    | nan      |       nan      |
|          10 | classification | success           | label  |             159 |            40 |             40 |         1    |  0.725  |           0.6    | nan      |       nan      |
|           8 | classification | success           | label  |             159 |            40 |             40 |         0.01 |  0.65   |           0.6    | nan      |       nan      |
|          10 | classification | target_identity   | label  |             159 |            40 |             40 |         0.01 |  0.85   |           0.25   | nan      |       nan      |
|           9 | classification | target_identity   | label  |             159 |            40 |             40 |        10    |  0.725  |           0.25   | nan      |       nan      |
|           8 | classification | target_identity   | label  |             159 |            40 |             40 |         0.01 |  0.7    |           0.25   | nan      |       nan      |
|          10 | regression     | target_max_lift   | scalar |             159 |            40 |             40 |         0.01 |  0.2523 |          -0.0004 |   0.0791 |         0.101  |
|           9 | regression     | target_max_lift   | scalar |             159 |            40 |             40 |       100    |  0.187  |          -0.0004 |   0.0845 |         0.101  |
|           8 | regression     | target_max_lift   | scalar |             159 |            40 |             40 |       100    |  0.0736 |          -0.0004 |   0.0916 |         0.101  |
|          10 | regression     | target_pos        | x      |             159 |            40 |             40 |         0.01 |  0.4275 |          -0.0006 |   0.0811 |         0.1222 |
|           9 | regression     | target_pos        | x      |             159 |            40 |             40 |         0.01 |  0.293  |          -0.0006 |   0.0935 |         0.1222 |
|           8 | regression     | target_pos        | x      |             159 |            40 |             40 |         0.01 |  0.2203 |          -0.0006 |   0.0973 |         0.1222 |
|          10 | regression     | target_pos        | y      |             159 |            40 |             40 |         0.01 |  0.3854 |          -0.0021 |   0.0608 |         0.0769 |
|           9 | regression     | target_pos        | y      |             159 |            40 |             40 |         0.01 |  0.2565 |          -0.0021 |   0.0684 |         0.0769 |
|           8 | regression     | target_pos        | y      |             159 |            40 |             40 |         0.01 |  0.129  |          -0.0021 |   0.0736 |         0.0769 |
|          10 | regression     | target_pos        | z      |             159 |            40 |             40 |         0.01 |  0.7321 |          -0      |   0.0948 |         0.1921 |
|           9 | regression     | target_pos        | z      |             159 |            40 |             40 |         0.01 |  0.6853 |          -0      |   0.1017 |         0.1921 |
|           8 | regression     | target_pos        | z      |             159 |            40 |             40 |         0.01 |  0.6137 |          -0      |   0.1111 |         0.1921 |
|          10 | regression     | target_to_basket  | x      |             159 |            40 |             40 |         0.01 |  0.4172 |          -0.0009 |   0.0815 |         0.1219 |
|           9 | regression     | target_to_basket  | x      |             159 |            40 |             40 |         0.01 |  0.2796 |          -0.0009 |   0.094  |         0.1219 |
|           8 | regression     | target_to_basket  | x      |             159 |            40 |             40 |         0.01 |  0.2101 |          -0.0009 |   0.0975 |         0.1219 |
|          10 | regression     | target_to_basket  | y      |             159 |            40 |             40 |         0.01 |  0.379  |          -0.0032 |   0.0607 |         0.0773 |
|           9 | regression     | target_to_basket  | y      |             159 |            40 |             40 |         0.01 |  0.2551 |          -0.0032 |   0.0677 |         0.0773 |
|           8 | regression     | target_to_basket  | y      |             159 |            40 |             40 |         0.01 |  0.1303 |          -0.0032 |   0.0727 |         0.0773 |
|          10 | regression     | target_to_basket  | z      |             159 |            40 |             40 |         0.01 |  0.7321 |          -0      |   0.0948 |         0.1921 |
|           9 | regression     | target_to_basket  | z      |             159 |            40 |             40 |         0.01 |  0.6853 |          -0      |   0.1017 |         0.1921 |
|           8 | regression     | target_to_basket  | z      |             159 |            40 |             40 |         0.01 |  0.6137 |          -0      |   0.1111 |         0.1921 |
|          10 | regression     | target_to_gripper | x      |             159 |            40 |             40 |         0.01 |  0.4435 |          -0.0006 |   0.0787 |         0.1207 |
|           9 | regression     | target_to_gripper | x      |             159 |            40 |             40 |         0.01 |  0.3075 |          -0.0006 |   0.0909 |         0.1207 |
|           8 | regression     | target_to_gripper | x      |             159 |            40 |             40 |         0.01 |  0.232  |          -0.0006 |   0.0947 |         0.1207 |
|          10 | regression     | target_to_gripper | y      |             159 |            40 |             40 |         0.01 |  0.3888 |          -0.0016 |   0.0586 |         0.0743 |
|           9 | regression     | target_to_gripper | y      |             159 |            40 |             40 |         0.01 |  0.2542 |          -0.0016 |   0.0666 |         0.0743 |
|           8 | regression     | target_to_gripper | y      |             159 |            40 |             40 |         0.01 |  0.118  |          -0.0016 |   0.0712 |         0.0743 |
|          10 | regression     | target_to_gripper | z      |             159 |            40 |             40 |         0.01 |  0.7295 |          -0      |   0.0959 |         0.194  |
|           9 | regression     | target_to_gripper | z      |             159 |            40 |             40 |         0.01 |  0.6839 |          -0      |   0.103  |         0.194  |
|           8 | regression     | target_to_gripper | z      |             159 |            40 |             40 |         0.01 |  0.6142 |          -0      |   0.112  |         0.194  |

## Interpretation Guide

- High target-identity accuracy means the generated action trajectory itself is task-specific.
- Geometry regression from `flow_x_t` is not semantic proof; it may reflect memorized scene/action priors.
- Increasing scores across flow steps would suggest denoising sharpens task/geometry/action information.
- Success/lift probes indicate whether the action trajectory contains outcome information before execution finishes.

## Current Readout

This first run uses only `call_00` for each rollout, so it describes early policy-call denoising rather than later lift/recovery calls.

Main findings:

- Target identity is not decodable from early noisy action samples, but becomes strongly decodable as denoising finishes: accuracy rises from `0.20` at flow step 0 to `0.85` at flow step 10, versus a `0.25` majority baseline.
- Target geometry also becomes more decodable across denoising. For example, `target_to_gripper z` rises from `R2 = 0.0488` at flow step 0 to `R2 = 0.7295` at flow step 10.
- Target max lift is weakly but increasingly predictable from the denoised action trajectory: `R2 = -0.2618` at flow step 0 and `R2 = 0.2523` at flow step 10.
- Success classification only becomes modestly above baseline late: `0.725` at steps 9-10 versus `0.60` baseline.

Interpretation:

> The flow-matching output itself becomes progressively more task- and geometry-specific during denoising. The action trajectory starts close to uninformative noise and ends with substantial target/task information.

This is useful because it connects semantic/scene information to the generated action object, not just hidden-state probes. But it does not yet prove causal use, and benchmark/layout priors remain a confound.

Next steps:

- run the same probe evolution for later policy calls, especially calls around close/lift
- compare successful vs failed lift-phase calls
- apply this to captured handoff swaps to see whether failure donor changes late-flow lift predictors
- add object/layout-prior baselines before claiming grounded geometry
