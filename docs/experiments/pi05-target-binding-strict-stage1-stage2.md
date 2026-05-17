# PI0.5 Strict Check Summary

This file explains the strict checks in plain language.

## Question

Do saved model states tell us which object the robot will move or lift, beyond easy task and layout guesses?

## Why The Check Is Strict

Task and layout are strong shortcuts.

For example, if a task usually fails in one layout, a simple task/layout guess can look smart without reading the model at all.

So any saved-state result must beat the best task/layout shortcut on held-out layouts.

## Thresholds

| thing predicted | best shortcut | saved-state result must be |
|---|---|---:|
| success | task ID | accuracy at least `0.9261` |
| failure type | task ID | accuracy at least `0.8704` |
| first object moved | task ID | accuracy at least `0.8373` |
| first object lifted | target guess | accuracy at least `0.7535` |
| target lift height | task ID | error at most `0.0413` |
| closest target distance | task ID | error at most `0.0149` |

## What Passed

Saved hidden states passed for the first object moved and the first object lifted.

| saved state | first moved object | first lifted object |
|---|---:|---:|
| vision/language hidden state | `0.9535` pass | `0.8162` pass |
| middle/prefix hidden state | `0.9528` pass | `0.8282` pass |
| action-expert hidden state | `0.9592` pass | `0.8387` pass |
| action-expert hidden states pooled over steps | `0.9528` pass | `0.8408` pass |

Plain meaning:

> Saved hidden states contain information about which object the robot will interact with.

## What Failed

No tested saved-state feature passed for:

- success
- failure type
- target lift height
- closest target distance

Broad action summaries also failed:

- pooled final action chunk
- pooled final action-generation state

Plain meaning:

> The hidden states are useful for object interaction. Broad success and broad action summaries are not yet good enough.

## Action-Direction Result

We also compared the generated action direction against directions to scene objects.

| failure type | rollouts | median target-vs-distractor margin | target wins rate |
|---|---:|---:|---:|
| approach failure | 99 | 0.2498 | 0.7174 |
| close but far from target | 70 | -0.2139 | 0.3476 |
| wrong object lifted | 285 | -0.2154 | 0.2915 |
| wrong object moved | 78 | -0.2752 | 0.2463 |

Plain meaning:

> Wrong-object moved/lifted failures are the cleanest place to study object-choice mistakes. Approach failures often still point toward the target, so they may be about reach or timing.

## What This Supports

Safe claim:

> Hidden states contain useful information about which object the robot will move or lift, beyond task/layout shortcuts.

Not safe yet:

> The hidden state causes the robot to choose that object.

That cause-and-effect claim needs a model-change test.

## Next Direction

- Focus on wrong-object moved/lifted cases.
- Use object-specific action directions instead of broad action summaries.
- Use Scene 4 for the first clean model-change test.
- Treat Scene 3/task 59 as a failure story until clean good examples are captured.

## Output References

- `artifacts/pi05_analysis/target_binding_controls/strict_activation_probes_vlm/`
- `artifacts/pi05_analysis/target_binding_controls/strict_activation_probes_expert_flow/`
- `artifacts/pi05_analysis/target_binding_controls/flow_binding_refined/`
- `artifacts/pi05_analysis/target_binding_controls/stage2_failure_flow_synthesis/`
