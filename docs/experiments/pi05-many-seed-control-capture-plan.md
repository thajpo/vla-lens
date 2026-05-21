# PI0.5 Many-Seed Control Capture Plan

## Goal

Capture `400` high-control episodes with repeated seeds for the same task/layout contrasts.

This is not a broad data run. It is designed to test whether our target-binding readouts survive the strict shortcut controls that the mostly one-seed dataset cannot fully provide.

## Design

- Seeds per task/layout: `1000, 1001, 1002, 1003, 1004`
- Layouts per task: `10` by default
- Contrast structure: same scene family and layout index, different target instruction
- Primary use: held-out-layout probes, object/layout/phase priors, first-moved/first-lifted prediction, and selecting causal intervention cases

## Allocation By Group

| binding_type    | scene_family        |   tasks |   layouts |   seeds |   episodes |
|:----------------|:--------------------|--------:|----------:|--------:|-----------:|
| object_identity | living_room_scene_1 |       2 |        10 |       5 |        100 |
| object_identity | living_room_scene_2 |       2 |        10 |       5 |        100 |
| object_identity | living_room_scene_3 |       2 |        10 |       5 |        100 |
| object_identity | living_room_scene_4 |       2 |        10 |       5 |        100 |

## Allocation By Task

| benchmark   |   task_id | scene_family        | target_guess           | task_name                                                                    |   layouts |   seeds |   episodes |
|:------------|----------:|:--------------------|:-----------------------|:-----------------------------------------------------------------------------|----------:|--------:|-----------:|
| libero_90   |        46 | living_room_scene_1 | alphabet_soup          | LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket        |        10 |       5 |         50 |
| libero_90   |        47 | living_room_scene_1 | cream_cheese_box       | LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket     |        10 |       5 |         50 |
| libero_90   |        51 | living_room_scene_2 | butter                 | LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket               |        10 |       5 |         50 |
| libero_90   |        54 | living_room_scene_2 | tomato_sauce           | LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket         |        10 |       5 |         50 |
| libero_90   |        57 | living_room_scene_3 | cream_cheese           | LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray           |        10 |       5 |         50 |
| libero_90   |        59 | living_room_scene_3 | tomato_sauce           | LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray           |        10 |       5 |         50 |
| libero_90   |        60 | living_room_scene_4 | black_bowl_on_the_left | LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray |        10 |       5 |         50 |
| libero_90   |        61 | living_room_scene_4 | chocolate_pudding      | LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray      |        10 |       5 |         50 |

## Run Command

Status: historical capture plan. Do not run this until the current analysis says
this exact control set is worth spending the disk/time budget. If revived,
convert the allocation file to the current `episode_plan.csv` schema first.

```bash
scripts/pi05_batch_capture_rocm.sh \
  --episode-plan artifacts/pi05_analysis/many_seed_control_capture_plan/episode_plan.csv \
  --output-root "/media/j/New Volume/vla-lens-artifacts/pi05_many_seed_control_captures" \
  --run
```

## Why This Is Stricter

The old weak claim is: a probe can decode target/task information.

The strict claim this dataset enables is: for the same scene family and layout index, changing the instruction target changes the action/flow representation toward the selected object, and this survives object/task/layout/phase baselines across repeated seeds.
