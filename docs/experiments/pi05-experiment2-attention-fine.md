# Experiment 2F: Fine-Grained Expert Attention

This analysis refines the earlier coarse attention summary by binning the prefix and suffix tokens into smaller windows and comparing attention mass at that resolution.

## Largest benchmark attention deltas (Scene 1 minus LIBERO_OBJECT)

|   flow_step | bin_name   |   libero_90 |   libero_object |   delta_scene1_minus_object |
|------------:|:-----------|------------:|----------------:|----------------------------:|
|           4 | suffix_00  |     0.20198 |         0.19487 |                     0.00712 |
|           0 | suffix_00  |     0.20569 |         0.19881 |                     0.00688 |
|           3 | suffix_00  |     0.20358 |         0.19682 |                     0.00676 |
|           1 | suffix_00  |     0.20378 |         0.19703 |                     0.00675 |
|           5 | suffix_00  |     0.20109 |         0.19437 |                     0.00672 |
|           2 | suffix_00  |     0.20426 |         0.19785 |                     0.0064  |
|           6 | suffix_00  |     0.1983  |         0.19215 |                     0.00615 |
|           0 | image_06   |     0.01529 |         0.01035 |                     0.00493 |
|           7 | suffix_00  |     0.19636 |         0.1916  |                     0.00477 |
|           1 | image_06   |     0.01527 |         0.0107  |                     0.00458 |
|           9 | suffix_04  |     0.16473 |         0.16023 |                     0.0045  |
|           2 | image_06   |     0.01496 |         0.01051 |                     0.00445 |
|           3 | image_06   |     0.01482 |         0.0106  |                     0.00422 |
|           4 | image_06   |     0.01512 |         0.01109 |                     0.00403 |
|           5 | image_06   |     0.01474 |         0.01093 |                     0.00381 |
|           6 | image_06   |     0.01478 |         0.0112  |                     0.00358 |
|           8 | suffix_00  |     0.1943  |         0.19078 |                     0.00352 |
|           8 | suffix_04  |     0.15677 |         0.15335 |                     0.00343 |
|           7 | image_06   |     0.01417 |         0.01095 |                     0.00322 |
|           8 | image_06   |     0.01344 |         0.01056 |                     0.00288 |

## Largest success-vs-failure deltas for mixed-outcome Scene 1 tasks

|   flow_step | bin_name   |   False |    True |   delta_success_minus_failure | object_label   |
|------------:|:-----------|--------:|--------:|------------------------------:|:---------------|
|           6 | suffix_03  | 0.1644  | 0.17046 |                       0.00606 | alphabet_soup  |
|           0 | suffix_02  | 0.17979 | 0.18572 |                       0.00593 | alphabet_soup  |
|           5 | suffix_03  | 0.16434 | 0.17021 |                       0.00587 | alphabet_soup  |
|           4 | suffix_03  | 0.1638  | 0.16944 |                       0.00564 | alphabet_soup  |
|           7 | suffix_03  | 0.16593 | 0.17154 |                       0.0056  | alphabet_soup  |
|           1 | suffix_02  | 0.18045 | 0.18598 |                       0.00553 | alphabet_soup  |
|           0 | text_00    | 0.03751 | 0.04259 |                       0.00508 | alphabet_soup  |
|           9 | text_00    | 0.04008 | 0.04497 |                       0.0049  | alphabet_soup  |
|           8 | suffix_03  | 0.16803 | 0.17292 |                       0.00489 | alphabet_soup  |
|           2 | text_00    | 0.03893 | 0.0437  |                       0.00477 | alphabet_soup  |
|           8 | text_00    | 0.04702 | 0.05175 |                       0.00474 | alphabet_soup  |
|           3 | text_00    | 0.04029 | 0.04502 |                       0.00473 | alphabet_soup  |
|           3 | suffix_03  | 0.16538 | 0.17003 |                       0.00464 | alphabet_soup  |
|           1 | text_00    | 0.03748 | 0.04208 |                       0.0046  | alphabet_soup  |
|           7 | text_00    | 0.04716 | 0.05154 |                       0.00438 | alphabet_soup  |
|           9 | suffix_03  | 0.17483 | 0.1792  |                       0.00437 | alphabet_soup  |
|           4 | text_00    | 0.04206 | 0.0463  |                       0.00424 | alphabet_soup  |
|           6 | text_00    | 0.04567 | 0.04989 |                       0.00422 | alphabet_soup  |
|           9 | suffix_02  | 0.18028 | 0.18446 |                       0.00418 | alphabet_soup  |
|           2 | suffix_02  | 0.18047 | 0.18465 |                       0.00418 | alphabet_soup  |

## Largest cream_cheese-success vs ketchup-failure deltas

|   flow_step | bin_name   |   cream_cheese |   ketchup |   delta_cheese_success_minus_ketchup_failure |
|------------:|:-----------|---------------:|----------:|---------------------------------------------:|
|           0 | suffix_01  |        0.19336 |   0.18403 |                                      0.00933 |
|           1 | suffix_01  |        0.1928  |   0.18433 |                                      0.00847 |
|           2 | suffix_01  |        0.19238 |   0.18406 |                                      0.00832 |
|           3 | suffix_01  |        0.19167 |   0.18397 |                                      0.00771 |
|           4 | suffix_01  |        0.19054 |   0.1838  |                                      0.00673 |
|           5 | suffix_01  |        0.18986 |   0.18423 |                                      0.00563 |
|           6 | suffix_01  |        0.18834 |   0.18429 |                                      0.00405 |
|           0 | suffix_02  |        0.17823 |   0.17508 |                                      0.00315 |
|           7 | suffix_01  |        0.18796 |   0.18484 |                                      0.00311 |
|           9 | text_00    |        0.04439 |   0.0413  |                                      0.00309 |
|           8 | text_00    |        0.05164 |   0.04875 |                                      0.0029  |
|           5 | suffix_03  |        0.16856 |   0.16644 |                                      0.00212 |
|           1 | suffix_02  |        0.17863 |   0.17653 |                                      0.0021  |
|           8 | image_07   |        0.01583 |   0.01375 |                                      0.00207 |
|           9 | image_07   |        0.01288 |   0.0109  |                                      0.00198 |
|           8 | suffix_01  |        0.18771 |   0.18581 |                                      0.0019  |
|           4 | suffix_03  |        0.16804 |   0.16614 |                                      0.0019  |
|           7 | image_07   |        0.01616 |   0.01435 |                                      0.00181 |
|           6 | suffix_03  |        0.16853 |   0.16678 |                                      0.00175 |
|           8 | image_00   |        0.00545 |   0.00381 |                                      0.00164 |

## Interpretation guide

- Image-bin deltas localize where visual attention differs between benchmarks or outcomes.
- Text-bin deltas show whether specific regions of the instruction token block are consulted differently.
- Suffix-bin deltas indicate changes in reliance on action-history / time-conditioned tokens inside the expert.
