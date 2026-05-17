# PI0.5 Ranked Next Work

This table ranks what is worth doing next. The ranking is based on whether the work can make a future cause-and-effect claim clearer.

## Ranking

| Rank | Work | Why It Matters | Current Status | Next Step |
|---:|---|---|---|---|
| 1 | Reproduce saved actions before model changes | If we cannot replay the saved action, changing internals will be hard to trust. | Saved-array shapes match for many Scene 4 pairs. | Run a fresh no-training forward pass and compare to saved actions. |
| 2 | First small Scene 4 model-change test | Scene 4 has clean good and bad examples. | `39` usable good/bad pairs found. | Test one family before sweeping layers or scenes. |
| 3 | Fix the wrong-time control | The current file labels this control but does not yet truly shift time. | Known control bug. | Use a good example from one step earlier/later, or a mismatched task phase. |
| 4 | Keep the object-chain casebook current | This prevents experiment-name fog. | Plain-language casebook exists. | Regenerate/update after each major result. |
| 5 | Scene 3/task 59 failure autopsy | It is the clearest memorable failure story. | No clean good examples exist yet. | Study as pathology, not rescue. |
| 6 | Targeted new captures for blocked cases | Some favorite stories need clean good examples. | Scene 3/task 59 is blocked. | Only collect new data if a specific blocked claim needs it. |
| 7 | Object-by-object scoring | Broad saved-state summaries are too crude. | Object position dominates the current scoring pilot. | Build tests that score each object separately. |
| 8 | Same-layout lock-in analysis | Shows when changing the instruction does not change the robot's focus. | Strong `ketchup_1` lock-in found in Scene 1. | Use as case-selection evidence. |
| 9 | Repeated-wrong-object breakdown | Repeated wrong objects may be due to position, graspability, or learned habits. | Raw repeated-wrong-object patterns found; position explains much. | Separate these explanations before claiming meaning errors. |
| 10 | Older broad prediction tests | They showed information exists, but not whether it controls action. | Useful background only. | Do not expand unless needed as a control. |

## What Not To Prioritize

- More broad success/failure prediction tests.
- More attention summaries without a specific object-choice question.
- More data collection without a clear missing-good-example target.
- Sweeping many model-change tests before the smallest Scene 4 test works.

## Current Decision

The next serious experiment should be:

```text
Scene 4
one clean bad example
one clean good example
one model-forward replay check
one small internal-state replacement test
strict controls
```

The first success metric should be simple:

> Does the generated action point more toward the requested object and less toward the wrong object?

Full task success should come later.
