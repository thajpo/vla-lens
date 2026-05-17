# PI0.5 Open Questions

This is the plain-language question list.

## Main Question

When the instruction names an object, does the robot keep acting toward that object?

## Current Questions

| Priority | Question | Why It Matters | Current Answer | Next Step |
|---:|---|---|---|---|
| 1 | Can changing a saved internal state change which object the action points toward? | This is the first real cause-and-effect test. | Not tested yet. | Start with one Scene 4 family after replay checks. |
| 2 | Which saved states are useful for predicting the object moved or lifted? | Tells us where to look first. | Hidden states are useful; broad action summaries are weak. | Use this only to guide model-change tests. |
| 3 | Does the requested object lose control before action, during action, or only at physical contact? | Determines where to intervene. | Many cases are hidden-first or hidden-plus-action; many are behavior-only. | Keep using the object chain to classify cases. |
| 4 | Why does Scene 3/task 59 collapse from `tomato_sauce_1` to `ketchup_1`? | It is the clearest story. | It has no clean good examples, so rescue is blocked. | Study as a failure case or collect targeted clean examples. |
| 5 | Are repeated wrong objects real model habits or just easy positions? | Prevents overclaiming. | Geometry explains a lot. | Break down position, graspability, and task habit. |
| 6 | Why are many successful rollouts internally messy? | A success may be a bad comparison example. | Most successes are not clean under strict rules. | Use strict good-example labels before model-change tests. |
| 7 | Can object-by-object scoring improve over broad saved-state summaries? | The robot chooses among objects, not among labels in isolation. | Object position is strong; global saved-state summaries are weak. | Score each object separately. |

## Safe Answers So Far

- Some failures repeat in a clear wrong-object pattern.
- Hidden states contain useful information about which object gets moved or lifted.
- Broad success/failure is too vague to be the main target.
- Position and layout are powerful shortcuts.
- Scene 4 is the best current place for a first model-change test.
- Scene 3/task 59 is a failure-story case, not a rescue-test case yet.

## Answers We Do Not Have Yet

- We do not know if changing a hidden state changes object choice.
- We do not know the exact mechanism.
- We do not know whether `ketchup_1`-style failures are meaning errors, position effects, grasp-related, or learned habits.
- We do not know whether full task success can be rescued.

## Current Working Bet

The model often has information about the requested object, but that information does not always win when the action is generated.

The next step is to test whether changing one saved internal state can make the action point toward the requested object instead of the wrong one.

See:

```text
docs/experiments/pi05-object-chain-casebook.md
```
