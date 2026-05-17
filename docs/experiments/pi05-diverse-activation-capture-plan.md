# PI0.5 Diverse Activation Capture Plan

## Goal

Plan `1500` new full activation episodes across LIBERO benchmarks, excluding the separate delta/intervention set.

The main aim is to test whether object/position probes still work when task, scene, and benchmark priors are more diverse.

## Task Tiers

- `clean_single_object`: best for object identity and object-position probes. These tasks name one manipulated object.
- `secondary_object`: useful but less clean. These include multi-object, stacking, or multi-step tasks.
- `exclude_for_object_probes`: drawer/stove/microwave state tasks or unclear object targets. These can be useful later, but not for the first object-position probe dataset.

## Inventory

| benchmark      |   clean_single_object |   exclude_for_object_probes |   secondary_object |
|:---------------|----------------------:|----------------------------:|-------------------:|
| libero_10      |                     3 |                           0 |                  7 |
| libero_90      |                    70 |                          12 |                  8 |
| libero_goal    |                     7 |                           2 |                  1 |
| libero_object  |                    10 |                           0 |                  0 |
| libero_spatial |                    10 |                           0 |                  0 |

## Episode Allocation By Benchmark

| benchmark      |   tasks |   episodes |
|:---------------|--------:|-----------:|
| libero_10      |       3 |         45 |
| libero_90      |      70 |       1050 |
| libero_goal    |       7 |        105 |
| libero_object  |      10 |        150 |
| libero_spatial |      10 |        150 |

## Episode Allocation By Tier

| benchmark      | tier                |   tasks |   episodes |
|:---------------|:--------------------|--------:|-----------:|
| libero_10      | clean_single_object |       3 |         45 |
| libero_90      | clean_single_object |      70 |       1050 |
| libero_goal    | clean_single_object |       7 |        105 |
| libero_object  | clean_single_object |      10 |        150 |
| libero_spatial | clean_single_object |      10 |        150 |

## Clean Single-Object Tasks

| benchmark      |   task_id | target_guess                                       | task_name                                                                                 |
|:---------------|----------:|:---------------------------------------------------|:------------------------------------------------------------------------------------------|
| libero_spatial |         0 | black_bowl_between_the_plate_and_the_ramekin       | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate        |
| libero_spatial |         1 | black_bowl_next_to_the_ramekin                     | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate                      |
| libero_spatial |         2 | black_bowl_from_table_center                       | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate                        |
| libero_spatial |         3 | black_bowl_on_the_cookie_box                       | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate                        |
| libero_spatial |         4 | black_bowl_in_the_top_drawer_of_the_wooden_cabinet | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate  |
| libero_spatial |         5 | black_bowl_on_the_ramekin                          | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate                           |
| libero_spatial |         6 | black_bowl_next_to_the_cookie_box                  | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate                   |
| libero_spatial |         7 | black_bowl_on_the_stove                            | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate                             |
| libero_spatial |         8 | black_bowl_next_to_the_plate                       | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate                        |
| libero_spatial |         9 | black_bowl_on_the_wooden_cabinet                   | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate                    |
| libero_object  |         0 | alphabet_soup                                      | pick_up_the_alphabet_soup_and_place_it_in_the_basket                                      |
| libero_object  |         1 | cream_cheese                                       | pick_up_the_cream_cheese_and_place_it_in_the_basket                                       |
| libero_object  |         2 | salad_dressing                                     | pick_up_the_salad_dressing_and_place_it_in_the_basket                                     |
| libero_object  |         3 | bbq_sauce                                          | pick_up_the_bbq_sauce_and_place_it_in_the_basket                                          |
| libero_object  |         4 | ketchup                                            | pick_up_the_ketchup_and_place_it_in_the_basket                                            |
| libero_object  |         5 | tomato_sauce                                       | pick_up_the_tomato_sauce_and_place_it_in_the_basket                                       |
| libero_object  |         6 | butter                                             | pick_up_the_butter_and_place_it_in_the_basket                                             |
| libero_object  |         7 | milk                                               | pick_up_the_milk_and_place_it_in_the_basket                                               |
| libero_object  |         8 | chocolate_pudding                                  | pick_up_the_chocolate_pudding_and_place_it_in_the_basket                                  |
| libero_object  |         9 | orange_juice                                       | pick_up_the_orange_juice_and_place_it_in_the_basket                                       |
| libero_goal    |         1 | bowl                                               | put_the_bowl_on_the_stove                                                                 |
| libero_goal    |         2 | wine_bottle                                        | put_the_wine_bottle_on_top_of_the_cabinet                                                 |
| libero_goal    |         4 | bowl                                               | put_the_bowl_on_top_of_the_cabinet                                                        |
| libero_goal    |         5 | plate                                              | push_the_plate_to_the_front_of_the_stove                                                  |
| libero_goal    |         6 | cream_cheese                                       | put_the_cream_cheese_in_the_bowl                                                          |
| libero_goal    |         8 | bowl                                               | put_the_bowl_on_the_plate                                                                 |
| libero_goal    |         9 | wine_bottle                                        | put_the_wine_bottle_on_the_rack                                                           |
| libero_10      |         3 | black_bowl                                         | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it        |
| libero_10      |         5 | book                                               | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy           |
| libero_10      |         9 | yellow_and_white_mug                               | KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it                 |
| libero_90      |         2 | black_bowl                                         | KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet                       |
| libero_90      |         3 | butter_at_the_back                                 | KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it  |
| libero_90      |         4 | butter_at_the_front                                | KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it |
| libero_90      |         5 | chocolate_pudding                                  | KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it   |
| libero_90      |         9 | black_bowl                                         | KITCHEN_SCENE1_put_the_black_bowl_on_the_plate                                            |
| libero_90      |        10 | black_bowl                                         | KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet                                   |
| libero_90      |        12 | black_bowl_at_the_back                             | KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate                                |
| libero_90      |        13 | black_bowl_at_the_front                            | KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate                               |
| libero_90      |        14 | middle_black_bowl                                  | KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate                                     |
| libero_90      |        15 | middle_black_bowl                                  | KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet                            |
| libero_90      |        18 | frying_pan                                         | KITCHEN_SCENE3_put_the_frying_pan_on_the_stove                                            |
| libero_90      |        19 | moka_pot                                           | KITCHEN_SCENE3_put_the_moka_pot_on_the_stove                                              |
| libero_90      |        24 | black_bowl                                         | KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet                     |
| libero_90      |        25 | black_bowl                                         | KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet                                   |
| libero_90      |        26 | wine_bottle                                        | KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet                    |
| libero_90      |        27 | wine_bottle                                        | KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack                                       |
| libero_90      |        29 | black_bowl                                         | KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet                        |
| libero_90      |        30 | black_bowl                                         | KITCHEN_SCENE5_put_the_black_bowl_on_the_plate                                            |
| libero_90      |        31 | black_bowl                                         | KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet                                   |
| libero_90      |        32 | ketchup                                            | KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet                           |
| libero_90      |        34 | yellow_and_white_mug                               | KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug                 |
| libero_90      |        36 | white_bowl                                         | KITCHEN_SCENE7_put_the_white_bowl_on_the_plate                                            |
| libero_90      |        37 | white_bowl                                         | KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate                               |
| libero_90      |        38 | right_moka_pot                                     | KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove                                        |
| libero_90      |        40 | frying_pan                                         | KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf                                    |
| libero_90      |        41 | frying_pan                                         | KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet                                   |
| libero_90      |        42 | frying_pan                                         | KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf                                 |
| libero_90      |        43 | white_bowl                                         | KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet                                   |
| libero_90      |        46 | alphabet_soup                                      | LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket                     |
| libero_90      |        47 | cream_cheese_box                                   | LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket                  |
| libero_90      |        48 | ketchup                                            | LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket                           |
| libero_90      |        49 | tomato_sauce                                       | LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket                      |
| libero_90      |        50 | alphabet_soup                                      | LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket                     |
| libero_90      |        51 | butter                                             | LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket                            |
| libero_90      |        52 | milk                                               | LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket                              |
| libero_90      |        53 | orange_juice                                       | LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket                      |
| libero_90      |        54 | tomato_sauce                                       | LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket                      |
| libero_90      |        55 | alphabet_soup                                      | LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray                       |
| libero_90      |        56 | butter                                             | LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray                              |
| libero_90      |        57 | cream_cheese                                       | LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray                        |
| libero_90      |        58 | ketchup                                            | LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray                             |
| libero_90      |        59 | tomato_sauce                                       | LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray                        |
| libero_90      |        60 | black_bowl_on_the_left                             | LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray              |
| libero_90      |        61 | chocolate_pudding                                  | LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray                   |
| libero_90      |        62 | salad_dressing                                     | LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray                      |
| libero_90      |        65 | red_mug                                            | LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate                                      |
| libero_90      |        66 | red_mug                                            | LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate                                     |
| libero_90      |        67 | white_mug                                          | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate                                    |
| libero_90      |        68 | yellow_and_white_mug                               | LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate                        |
| libero_90      |        69 | chocolate_pudding                                  | LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate                     |
| libero_90      |        70 | chocolate_pudding                                  | LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate                    |
| libero_90      |        71 | red_mug                                            | LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate                                           |
| libero_90      |        72 | white_mug                                          | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate                                         |
| libero_90      |        73 | book                                               | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy          |
| libero_90      |        74 | book                                               | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy           |
| libero_90      |        75 | book                                               | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy          |
| libero_90      |        76 | yellow_and_white_mug                               | STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy      |
| libero_90      |        77 | book                                               | STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy           |
| libero_90      |        78 | book                                               | STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy          |
| libero_90      |        79 | book                                               | STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy           |
| libero_90      |        80 | book                                               | STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy          |
| libero_90      |        81 | book                                               | STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy          |
| libero_90      |        82 | book                                               | STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy           |
| libero_90      |        83 | book                                               | STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy          |
| libero_90      |        84 | red_mug                                            | STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy                   |
| libero_90      |        85 | white_mug                                          | STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy                 |
| libero_90      |        86 | book_in_the_middle                                 | STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf             |
| libero_90      |        87 | book_on_the_left                                   | STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf                |
| libero_90      |        88 | book_on_the_right                                  | STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf              |
| libero_90      |        89 | book_on_the_right                                  | STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf           |

## Secondary Object Tasks

| benchmark   |   task_id | target_guess   | reason                             | task_name                                                                                                  |
|:------------|----------:|:---------------|:-----------------------------------|:-----------------------------------------------------------------------------------------------------------|
| libero_goal |         3 |                | state change plus object task      | open_the_top_drawer_and_put_the_bowl_inside                                                                |
| libero_10   |         0 |                | multi-object task                  | LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket                           |
| libero_10   |         1 |                | multi-object task                  | LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket                              |
| libero_10   |         2 | moka_pot       | appliance plus object task         | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it                                                |
| libero_10   |         4 | white_mug      | two named object placements        | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate |
| libero_10   |         6 | white_mug      | two named object placements        | LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate  |
| libero_10   |         7 |                | multi-object task                  | LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket                       |
| libero_10   |         8 |                | multi-object task                  | KITCHEN_SCENE8_put_both_moka_pots_on_the_stove                                                             |
| libero_90   |         1 | black_bowl     | state change plus object task      | KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it                    |
| libero_90   |         8 | bowl           | state change plus object task      | KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it                                   |
| libero_90   |        16 |                | stacking has multiple object roles | KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle                           |
| libero_90   |        17 |                | stacking has multiple object roles | KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl                                          |
| libero_90   |        21 | frying_pan     | appliance plus object task         | KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it                                              |
| libero_90   |        45 | frying_pan     | appliance plus object task         | KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it                                              |
| libero_90   |        63 |                | stacking has multiple object roles | LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray                        |
| libero_90   |        64 |                | stacking has multiple object roles | LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray                        |

## Excluded From First Object-Probe Dataset

| benchmark   |   task_id | reason                                             | task_name                                                                     |
|:------------|----------:|:---------------------------------------------------|:------------------------------------------------------------------------------|
| libero_goal |         0 | drawer/microwave state target, not object position | open_the_middle_drawer_of_the_cabinet                                         |
| libero_goal |         7 | appliance state target, not object position        | turn_on_the_stove                                                             |
| libero_90   |         0 | drawer/microwave state target, not object position | KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet                           |
| libero_90   |         6 | drawer/microwave state target, not object position | KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet                          |
| libero_90   |         7 | drawer/microwave state target, not object position | KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet                             |
| libero_90   |        11 | drawer/microwave state target, not object position | KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet                             |
| libero_90   |        20 | appliance state target, not object position        | KITCHEN_SCENE3_turn_on_the_stove                                              |
| libero_90   |        22 | drawer/microwave state target, not object position | KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet                         |
| libero_90   |        23 | drawer/microwave state target, not object position | KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer |
| libero_90   |        28 | drawer/microwave state target, not object position | KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet                            |
| libero_90   |        33 | drawer/microwave state target, not object position | KITCHEN_SCENE6_close_the_microwave                                            |
| libero_90   |        35 | drawer/microwave state target, not object position | KITCHEN_SCENE7_open_the_microwave                                             |
| libero_90   |        39 | appliance state target, not object position        | KITCHEN_SCENE8_turn_off_the_stove                                             |
| libero_90   |        44 | appliance state target, not object position        | KITCHEN_SCENE9_turn_on_the_stove                                              |

## Recommended Rule

Start with 20 episodes per clean single-object task. That usually means 10 layouts x 2 seeds, which is enough to support held-out-layout checks better than 5 episodes/task.

Use remaining budget on secondary object tasks, but analyze them separately because their target labels are more complex.
