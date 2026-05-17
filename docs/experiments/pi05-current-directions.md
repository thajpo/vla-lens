# PI0.5 Current Directions

This is the short version. For the full plain-language map, read:

```text
docs/experiments/pi05-object-chain-casebook.md
```

For the older technical summary, read:

```text
docs/experiments/pi05-consolidated-findings.md
```

## Current Question

When the instruction names an object, does the robot keep acting toward that object, or does another object take over?

The useful chain is:

```text
requested object
object suggested inside the model
object suggested by the action
first object moved
first object lifted
```

## What Changed Recently

We stopped treating “success or failure” as the main thing to explain.

That was too vague.

A better target is a clean wrong-object case, such as:

```text
requested: tomato_sauce_1
robot focuses on: ketchup_1
robot moves: ketchup_1
```

## What We Know

- The model often contains useful information about objects.
- Task, layout, and object position explain a lot, so simple prediction tests can be misleading.
- Some wrong-object failures are repeatable and structured.
- Scene 3/task 59 is a clear failure story but has no clean good examples yet.
- Scene 4 currently has the cleanest good/bad examples for the first model-change test.
- Scene 4 causal tracing found robust VLM KV-cache interface effects around layers 8, 12, and 14.
- Attribution patching sharpened the strongest Scene 4/task 61 site to layer-14 value-cache visual-prefix tokens, especially `vision_bin_10_of_24` in the second camera.
- Exact token scanning inside that hot bin found a sparse row-4 signal in `observation.images.image2`, led by tokens `331`, `327`, and `323`.
- Role tests suggest those hot tokens are donor/success features that can rescue the bad run, not toxic bad-run features whose removal reliably fixes the failure.
- Cumulative token tests show the top eight task61 tokens compose smoothly and move the offline action margin positive; flow tracing shows the patch changes the denoising trajectory from early/mid steps onward.
- A Scene 4/task 60 replication preserved the visual-prefix KV-cache pattern, but shifted more of the signal toward layer 12.
- Cross-object transfer is weak/asymmetric: task61 hot tokens weakly help task60, but task60 native tokens do not meaningfully help task61. Task60 native tokens also behave more like removable bad/local interference features, unlike task61's success-injection pattern.

## What We Do Not Know Yet

- We have shown a small offline interface-level causal effect, but not a full mechanism.
- We have not found a circuit or exact mechanism.
- We have not shown whether repeated wrong objects are meaning mistakes, position effects, grasping effects, or learned habits.
- Existing captures do not contain selected expert hidden states at every denoising step. New captures now support this, but old episodes need regeneration for deeper hidden-state tracing.
- We still have not identified what the hot feature encodes. The current missing-work log is `docs/experiments/pi05-feature-id-missing-work.md`.
- Cross-object transfer results are logged in `docs/experiments/pi05-cross-object-transfer.md`.

## Best Next Step

The next serious test should check whether the success feature is redundant or compositional:

```text
bad example: robot acts toward the wrong object
good example: robot acts toward the requested object
test: patch top tokens cumulatively, then repeat the role test for task 60's top bin
```

The current concrete candidates are task-61 layer-14 value tokens `331,327,323` and task-60 layer-12 key/value `vision_bin_04_of_24`.

For deeper expert-side causal tracing, regenerate the relevant episode set with the new `expert_selected_hidden_by_step` capture field.

## What Not To Do Next

- Do not run more broad “can we predict X?” tests unless they support a specific control.
- Do not claim success examples are clean unless the whole object chain is clean.
- Do not use Scene 3/task 59 for rescue claims until clean good examples exist.
- Do not claim how the model works from a casebook or prediction test alone.
