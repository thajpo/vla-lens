# PI0.5 Object-Chain Casebook

This is the plain-language map of the current PI0.5 project.

It is not a cause-and-effect results report. It says which failures are interesting, which comparisons are usable, and what still has to happen before we can claim that changing the model changes the robot's choice of object.

## Short Version

PI0.5 often has information about the requested object, but the robot can still act as if a different object won.

The useful question is no longer only:

> Does the model know the target?

The better question is:

> When the robot acts, did the requested object stay in control of the action?

## How To Read The Failures

For each rollout, we track a simple chain:

```text
requested object
object suggested by hidden states
object suggested by action direction
first object moved
first object lifted
```

A clean wrong-object failure looks like this:

```text
requested object: tomato_sauce_1
hidden states point to: ketchup_1
first moved object: ketchup_1
```

That kind of case is much easier to study than a vague “the robot failed” case.

## What We Know

| Status | Plain-language claim |
|---|---|
| solid | Wrong-object failures are often structured, not random. |
| solid | The robot often keeps choosing the same wrong object across related tasks. |
| solid | Task, layout, and object positions explain a lot, so we must control for them. |
| solid | Scene 3/task 59 is a clear failure story: `tomato_sauce_1` often turns into `ketchup_1`. |
| solid | Scene 4 has the cleanest good/bad examples for the first serious model-change test. |
| not proven | We have not shown that changing a saved internal state changes which object the robot acts on. |
| not proven | We have not found a circuit or exact mechanism. |
| not proven | We have not shown that repeated wrong objects are meaning mistakes rather than position or grasping effects. |

## Current Numbers

- offline audits completed: `12`
- usable good/bad task families for a first model-change test: `4`
- usable good/bad rollout pairs: `39`
- checked rows with matching saved-array shapes: `400`
- clean good examples across the dataset: `333`
- clean wrong-object examples across the dataset: `185`

## Best Places For A First Model-Change Test

These are the current best task families where we have both clean bad examples and clean good examples.

| scene | task | requested object | wrong object | bad examples | good examples |
|---|---:|---|---|---:|---:|
| `living_room_scene_4` | 61 | `chocolate_pudding_1` | `akita_black_bowl_1` | 3 | 21 |
| `living_room_scene_4` | 61 | `chocolate_pudding_1` | `akita_black_bowl_2` | 5 | 21 |
| `living_room_scene_4` | 61 | `chocolate_pudding_1` | `akita_black_bowl_2` | 1 | 21 |
| `living_room_scene_4` | 60 | `akita_black_bowl_1` | `akita_black_bowl_2` | 4 | 13 |

Plain meaning:

> Scene 4 is where we should first test whether changing a saved internal state can move the robot's action away from the wrong object and toward the requested object.

## Good Failure Stories That Are Not Ready For Rescue Tests

These cases have clean wrong-object failures but no clean same-task good examples in the current captures.

| scene | task | requested object | wrong object | clean wrong examples | why blocked |
|---|---:|---|---|---:|---|
| `living_room_scene_1` | 49 | `tomato_sauce_1` | `ketchup_1` | 28 | no clean good examples |
| `living_room_scene_3` | 59 | `tomato_sauce_1` | `ketchup_1` | 21 | no clean good examples |
| `living_room_scene_4` | 62 | `new_salad_dressing_1` | `akita_black_bowl_1` | 20 | no clean good examples |
| `living_room_scene_1` | 49 | `tomato_sauce_1` | `ketchup_1` | 12 | no clean good examples |
| `living_room_scene_3` | 57 | `cream_cheese_1` | `ketchup_1` | 9 | no clean good examples |
| `living_room_scene_3` | 59 | `tomato_sauce_1` | `ketchup_1` | 9 | no clean good examples |

Plain meaning:

> These are good for understanding failures. They are not yet good for showing rescue.

## Scene 3 Task 59

This is the memorable case:

```text
requested object: tomato_sauce_1
common wrong object: ketchup_1
often lifted object: alphabet_soup_1
```

There are `10` successful same-task rollouts, but none are clean good examples.

The successes look like this:

| layout | first moved | first lifted | how it succeeded |
|---:|---|---|---|
| 4 | `ketchup_1` | `alphabet_soup_1` | action sometimes pointed to target, but hidden states did not |
| 9 | `ketchup_1` | `alphabet_soup_1` | action sometimes pointed to target, but hidden states did not |
| 10 | `ketchup_1` | `tomato_sauce_1` | wrong first contact, target lifted later |
| 12 | `ketchup_1` | `alphabet_soup_1` | unclear recovery path |
| 13 | `ketchup_1` | `alphabet_soup_1` | action sometimes pointed to target, but hidden states did not |
| 14 | `ketchup_1` | `alphabet_soup_1` | action sometimes pointed to target, but hidden states did not |
| 24 | `ketchup_1` | `alphabet_soup_1` | unclear recovery path |
| 35 | `ketchup_1` | `tomato_sauce_1` | wrong first contact, target lifted later |
| 37 | `ketchup_1` | `alphabet_soup_1` | unclear recovery path |
| 38 | `ketchup_1` | `alphabet_soup_1` | unclear recovery path |

Plain meaning:

> Scene 3/task 59 is a great “what went wrong?” story. It is not yet a clean “can we rescue it?” test.

## Where The Wrong Object Shows Up

In wrong-object failures, we asked where the wrong object first appears.

| where it shows up | count | plain meaning |
|---|---:|---|
| physical behavior only | 178 | hidden/action summaries did not explain it cleanly |
| hidden states first | 100 | the wrong object appears internally before action fully follows |
| hidden states and action together | 84 | both point wrong at about the same time |
| action first | 1 | action points wrong before hidden summaries do |

Plain meaning:

> In many cases, the wrong object appears inside the model before or at the same time as the action points there. That makes a saved-state replacement test worth trying, but it does not prove cause and effect yet.

## Same-Layout Lock-In

Sometimes the object positions stay fixed, the instruction changes, and the robot still keeps choosing the same wrong object.

| scene | layout | targets tested | hidden switched? | action switched? | moved object switched? | repeated wrong object |
|---|---:|---:|---:|---:|---:|---|
| `living_room_scene_1` | 6 | 4 | 0.00 | 0.00 | 0.00 | `ketchup_1` |
| `living_room_scene_1` | 7 | 4 | 0.00 | 0.00 | 0.00 | `ketchup_1` |
| `living_room_scene_1` | 20 | 4 | 0.00 | 0.00 | 0.00 | `ketchup_1` |
| `living_room_scene_1` | 18 | 4 | 0.00 | 0.00 | 0.00 | `ketchup_1` |
| `living_room_scene_1` | 25 | 4 | 0.00 | 0.00 | 0.00 | `ketchup_1` |

Plain meaning:

> In some layouts, changing the instruction does not change what the robot seems to focus on.

## Objects The Robot Gets Pulled Toward

Some objects are repeatedly selected or moved when they are not the requested object.

| object | often moved/lifted | often suggested by hidden states | often suggested by action | after position controls |
|---|---:|---:|---:|---|
| `porcelain_mug_1` | 0.57 | 0.58 | 0.03 | mostly explained by position/ease |
| `akita_black_bowl_1` | 0.54 | 0.51 | 0.06 | mostly explained by position/ease |
| `milk_1` | 0.46 | 0.48 | 0.11 | small leftover effect |
| `ketchup_1` | 0.43 | 0.43 | 0.54 | high raw score, but position/ease explain a lot |
| `alphabet_soup_1` | 0.35 | 0.00 | 0.06 | mostly explained by position/ease |

Plain meaning:

> “Attractor object” is a useful description, but it is not yet a mechanism. Many apparent attractors may simply be close, easy, or well positioned.

## Success Is Usually Messy

Successful rollouts are often not clean examples of the robot following the requested object from start to finish.

- success with action pointing wrong: `673`
- success after wrong first contact: `305`
- clean success by strict rules: `3`

Plain meaning:

> We should not treat “successful rollout” as automatically useful for comparison. It may have succeeded for messy reasons.

## Object-Scoring Lesson

We tried predicting object choices using different information.

| thing predicted | global saved state only | object position only | global state plus position |
|---|---:|---:|---:|
| requested object | 0.58 | 0.95 | 0.95 |
| hidden-selected object | 0.58 | 0.93 | 0.95 |
| action-selected object | 0.58 | 0.87 | 0.89 |
| first moved object | 0.57 | 0.93 | 0.95 |
| first lifted object | 0.57 | 0.99 | 1.00 |

Plain meaning:

> Object position is very powerful. Broad saved-state summaries are too crude. Future object-scoring tests should look at each object separately.

## What Each Path Means Now

| Path | Current role | Recommendation |
|---|---|---|
| Casebook | Helps us understand the terrain. | Use it to choose cases and explain findings without claiming cause and effect. |
| Model-change test | Needed for the first real “this state matters” claim. | Start with Scene 4 after controls and replay checks pass. |
| Visualization | Useful only if it helps a research question. | Build views that show object choice over time or before/after a model-change test. |
| New data | Needed for blocked stories like Scene 3/task 59. | Collect only if we need clean good examples for a specific blocked case. |

## Before Any Cause-And-Effect Claim

Do not interpret a model-change test unless all of this is true:

| Gate | Requirement |
|---|---|
| bad example | the robot clearly follows the wrong object |
| good example | there is a clean comparison rollout where the robot follows the requested object |
| controls | self, random, wrong-object, wrong-time, and wrong-layer comparisons are included |
| timing | the good and bad examples are at comparable parts of the task |
| replay | loading saved arrays reproduces the saved action before any model change |
| metric | first measure whether action shifts toward the requested object, not full task success |
| claim | only claim object-control influence if the good comparison beats all controls |

## Immediate Next Gate

Before running the first serious model-change test:

- fix the current wrong-time control so it really uses a different time step
- reproduce saved actions with a fresh no-training model forward pass
- then test one Scene 4 family with the smallest useful set of comparisons

## Source Tables

- `artifacts/pi05_analysis/target_binding_controls/causal_design_overnight/`
- `artifacts/pi05_analysis/target_binding_controls/overnight_20_sweep/case_focus_report/`
