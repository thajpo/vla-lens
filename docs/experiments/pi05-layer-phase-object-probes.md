# PI0.5 Layer And Time Check

This file answers a simple question:

> At which saved model layers and time points can we predict the object the robot will move or lift?

This is a map, not a cause-and-effect result.

## What Was Tested

The script tested saved hidden states across action chunks `0-6`.

It predicted two things:

- first object moved
- first object lifted

It used the same held-out-layout shortcut checks as the stricter summary.

## Script

```bash
.venv-lerobot/bin/python scripts/analyze_pi05_layer_phase_object_probes.py --call-indices all
```

Outputs:

```text
artifacts/pi05_analysis/target_binding_controls/layer_phase_object_probes/
```

## Main Result

Information about the object moved or lifted is spread across many saved hidden states.

It is not limited to the first action chunk.

## First Object Moved

| action chunk | best accuracy | median accuracy | saved-state families passing |
|---:|---:|---:|---:|
| 0 | 0.9535 | 0.9504 | 24/24 |
| 1 | 0.9655 | 0.9570 | 24/24 |
| 2 | 0.9526 | 0.9410 | 24/24 |
| 3 | 0.9318 | 0.8893 | 24/24 |
| 4 | 0.9038 | 0.8558 | 23/24 |
| 5 | 0.9018 | 0.8261 | 11/24 |
| 6 | 0.9290 | 0.8612 | 23/24 |

Plain meaning:

> The first moved object is usually easy to predict from hidden states, but there is a noticeable dip around call 5.

## First Object Lifted

| action chunk | best accuracy | median accuracy | saved-state families passing |
|---:|---:|---:|---:|
| 0 | 0.8303 | 0.8155 | 23/24 |
| 1 | 0.8725 | 0.8588 | 24/24 |
| 2 | 0.9075 | 0.8974 | 24/24 |
| 3 | 0.8506 | 0.8244 | 24/24 |
| 4 | 0.8654 | 0.8314 | 24/24 |
| 5 | 0.8636 | 0.8186 | 24/24 |
| 6 | 0.8894 | 0.8455 | 24/24 |

Plain meaning:

> The first lifted object is also predictable from hidden states, strongest around call 2.

## The Call-5 Dip

At call 5, many action-expert hidden-state families stop passing for first moved object, while the best vision/language hidden-state features still pass.

This could mean:

- the model is in a different part of the task
- the action side is changing from approach to lift/recovery
- our saved summaries are too crude at that point
- the object-choice signal is no longer the same thing as the physical outcome

It does not prove a mechanism.

## How To Use This

Use this result to choose where to look.

Do not use it to claim cause and effect.

Good uses:

- choose calls for closer inspection
- compare good and bad Scene 4 examples
- decide where a model-change test might be worth trying

Bad uses:

- claiming call 5 is the mechanism
- claiming the model intends to move an object
- claiming hidden states cause behavior without a model-change test

## Next Step

For the first serious model-change test, use this as background only.

The required next gate is still:

```text
reproduce saved actions from saved arrays
then test one small Scene 4 good/bad pair
```
