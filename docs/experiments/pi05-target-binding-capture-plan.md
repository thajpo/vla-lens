# PI0.5 Target-Binding Capture Plan

## Goal

Capture `1500` new episodes, with `1420` assigned now and `80` held back for failed-task oversampling or sanity checks.

This replaces the broad `100 tasks x 15 episodes` plan. The new plan focuses on tasks where the instruction selects one target among plausible distractors.

## Allocation By Group

| binding_type          | scene_family              |   tasks |   episodes |
|:----------------------|:--------------------------|--------:|-----------:|
| destination_reference | mug_plate                 |       4 |         40 |
| destination_reference | study_caddy               |      10 |        100 |
| object_identity       | libero_object_basket      |      10 |        350 |
| object_identity       | living_room_scene_1       |       4 |        160 |
| object_identity       | living_room_scene_2       |       5 |        200 |
| object_identity       | living_room_scene_3       |       5 |        200 |
| object_identity       | living_room_scene_4       |       3 |        120 |
| spatial_reference     | libero_spatial_black_bowl |      10 |        250 |

## Allocation By Benchmark

| benchmark      |   tasks |   episodes |
|:---------------|--------:|-----------:|
| libero_90      |      31 |        820 |
| libero_object  |      10 |        350 |
| libero_spatial |      10 |        250 |

## Full Allocation

| benchmark      |   task_id | binding_type          | scene_family              | target_guess                                       |   episodes | task_name                                                                                |
|:---------------|----------:|:----------------------|:--------------------------|:---------------------------------------------------|-----------:|:-----------------------------------------------------------------------------------------|
| libero_90      |        46 | object_identity       | living_room_scene_1       | alphabet_soup                                      |         40 | LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket                    |
| libero_90      |        47 | object_identity       | living_room_scene_1       | cream_cheese_box                                   |         40 | LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket                 |
| libero_90      |        48 | object_identity       | living_room_scene_1       | ketchup                                            |         40 | LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket                          |
| libero_90      |        49 | object_identity       | living_room_scene_1       | tomato_sauce                                       |         40 | LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket                     |
| libero_90      |        50 | object_identity       | living_room_scene_2       | alphabet_soup                                      |         40 | LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket                    |
| libero_90      |        51 | object_identity       | living_room_scene_2       | butter                                             |         40 | LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket                           |
| libero_90      |        52 | object_identity       | living_room_scene_2       | milk                                               |         40 | LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket                             |
| libero_90      |        53 | object_identity       | living_room_scene_2       | orange_juice                                       |         40 | LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket                     |
| libero_90      |        54 | object_identity       | living_room_scene_2       | tomato_sauce                                       |         40 | LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket                     |
| libero_90      |        55 | object_identity       | living_room_scene_3       | alphabet_soup                                      |         40 | LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray                      |
| libero_90      |        56 | object_identity       | living_room_scene_3       | butter                                             |         40 | LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray                             |
| libero_90      |        57 | object_identity       | living_room_scene_3       | cream_cheese                                       |         40 | LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray                       |
| libero_90      |        58 | object_identity       | living_room_scene_3       | ketchup                                            |         40 | LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray                            |
| libero_90      |        59 | object_identity       | living_room_scene_3       | tomato_sauce                                       |         40 | LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray                       |
| libero_90      |        60 | object_identity       | living_room_scene_4       | black_bowl_on_the_left                             |         40 | LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray             |
| libero_90      |        61 | object_identity       | living_room_scene_4       | chocolate_pudding                                  |         40 | LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray                  |
| libero_90      |        62 | object_identity       | living_room_scene_4       | salad_dressing                                     |         40 | LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray                     |
| libero_object  |         0 | object_identity       | libero_object_basket      | alphabet_soup                                      |         35 | pick_up_the_alphabet_soup_and_place_it_in_the_basket                                     |
| libero_object  |         1 | object_identity       | libero_object_basket      | cream_cheese                                       |         35 | pick_up_the_cream_cheese_and_place_it_in_the_basket                                      |
| libero_object  |         2 | object_identity       | libero_object_basket      | salad_dressing                                     |         35 | pick_up_the_salad_dressing_and_place_it_in_the_basket                                    |
| libero_object  |         3 | object_identity       | libero_object_basket      | bbq_sauce                                          |         35 | pick_up_the_bbq_sauce_and_place_it_in_the_basket                                         |
| libero_object  |         4 | object_identity       | libero_object_basket      | ketchup                                            |         35 | pick_up_the_ketchup_and_place_it_in_the_basket                                           |
| libero_object  |         5 | object_identity       | libero_object_basket      | tomato_sauce                                       |         35 | pick_up_the_tomato_sauce_and_place_it_in_the_basket                                      |
| libero_object  |         6 | object_identity       | libero_object_basket      | butter                                             |         35 | pick_up_the_butter_and_place_it_in_the_basket                                            |
| libero_object  |         7 | object_identity       | libero_object_basket      | milk                                               |         35 | pick_up_the_milk_and_place_it_in_the_basket                                              |
| libero_object  |         8 | object_identity       | libero_object_basket      | chocolate_pudding                                  |         35 | pick_up_the_chocolate_pudding_and_place_it_in_the_basket                                 |
| libero_object  |         9 | object_identity       | libero_object_basket      | orange_juice                                       |         35 | pick_up_the_orange_juice_and_place_it_in_the_basket                                      |
| libero_spatial |         0 | spatial_reference     | libero_spatial_black_bowl | black_bowl_between_the_plate_and_the_ramekin       |         25 | pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate       |
| libero_spatial |         1 | spatial_reference     | libero_spatial_black_bowl | black_bowl_next_to_the_ramekin                     |         25 | pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate                     |
| libero_spatial |         2 | spatial_reference     | libero_spatial_black_bowl | black_bowl_from_table_center                       |         25 | pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate                       |
| libero_spatial |         3 | spatial_reference     | libero_spatial_black_bowl | black_bowl_on_the_cookie_box                       |         25 | pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate                       |
| libero_spatial |         4 | spatial_reference     | libero_spatial_black_bowl | black_bowl_in_the_top_drawer_of_the_wooden_cabinet |         25 | pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate |
| libero_spatial |         5 | spatial_reference     | libero_spatial_black_bowl | black_bowl_on_the_ramekin                          |         25 | pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate                          |
| libero_spatial |         6 | spatial_reference     | libero_spatial_black_bowl | black_bowl_next_to_the_cookie_box                  |         25 | pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate                  |
| libero_spatial |         7 | spatial_reference     | libero_spatial_black_bowl | black_bowl_on_the_stove                            |         25 | pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate                            |
| libero_spatial |         8 | spatial_reference     | libero_spatial_black_bowl | black_bowl_next_to_the_plate                       |         25 | pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate                       |
| libero_spatial |         9 | spatial_reference     | libero_spatial_black_bowl | black_bowl_on_the_wooden_cabinet                   |         25 | pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate                   |
| libero_90      |        65 | destination_reference | mug_plate                 | red_mug                                            |         10 | LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate                                     |
| libero_90      |        66 | destination_reference | mug_plate                 | red_mug                                            |         10 | LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate                                    |
| libero_90      |        67 | destination_reference | mug_plate                 | white_mug                                          |         10 | LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate                                   |
| libero_90      |        68 | destination_reference | mug_plate                 | yellow_and_white_mug                               |         10 | LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate                       |
| libero_90      |        73 | destination_reference | study_caddy               | book                                               |         10 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy         |
| libero_90      |        74 | destination_reference | study_caddy               | book                                               |         10 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy          |
| libero_90      |        75 | destination_reference | study_caddy               | book                                               |         10 | STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy         |
| libero_90      |        77 | destination_reference | study_caddy               | book                                               |         10 | STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy          |
| libero_90      |        78 | destination_reference | study_caddy               | book                                               |         10 | STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy         |
| libero_90      |        79 | destination_reference | study_caddy               | book                                               |         10 | STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy          |
| libero_90      |        80 | destination_reference | study_caddy               | book                                               |         10 | STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy         |
| libero_90      |        81 | destination_reference | study_caddy               | book                                               |         10 | STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy         |
| libero_90      |        82 | destination_reference | study_caddy               | book                                               |         10 | STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy          |
| libero_90      |        83 | destination_reference | study_caddy               | book                                               |         10 | STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy         |

## Run Command

Status: historical capture plan. If revived, generate a current
`episode_plan.csv` with one row per episode and verify the disk budget before
running.

```bash
scripts/pi05_batch_capture_rocm.sh \
  --episode-plan artifacts/pi05_analysis/target_binding_capture_plan/episode_plan.csv \
  --output-root "/media/j/New Volume/vla-lens-artifacts/pi05_target_binding_captures" \
  --run
```

## Why This Is Better

The central question is not broad task diversity. It is whether target language binds to the correct object when distractors are present.

This plan gives many repeated examples in contrastive scenes, especially LIBERO-90 living-room grocery scenes and LIBERO-Object basket tasks.
