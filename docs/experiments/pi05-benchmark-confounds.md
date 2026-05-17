# PI0.5 Benchmark Confounds

## Current Position

Benchmark confounds are not solved. They are now explicitly treated as a threat to interpretation.

The main concern is that PI0.5 may encode the whole scene and exploit repeated LIBERO layouts. If so, high identity/geometry probe scores can reflect task/layout regularities rather than online grounded target use.

## Confounds Identified

| Confound | Why It Matters | Current Handling | Remaining Gap |
|---|---|---|---|
| Repeated layouts | Object identity can predict object position | Layout-based train/val/test splits | Need stronger held-out-layout and perturbation tests |
| Whole-scene encoding | Target and distractor info may both be decodable | Non-target geometry probes | Need concise target-vs-distractor selectivity under perturbation |
| Prompt persistence | Target identity may be in text tokens without visual grounding | Image/text/handoff variants | Need object-local visual/token interventions |
| Benchmark priors | `LIBERO_OBJECT` and `Scene 1` differ in scene/motor priors | Cross-benchmark transfer and deltas | Deltas are supplemental, not causal |
| Action priors | The model may output memorized motor programs | Hard object-swap perturbations | Need graded target displacement and full captures |
| Outcome instability | Stored success/failure labels may not reproduce on rerun | Rerun-verified rescue rule | Need more rerun-verified captured interventions |

## What Has Been Done

- Probe splits are layout-based rather than random call-level splits.
- Non-target geometry probes were run to test whether distractors are also decodable.
- Hard target-swap perturbations were run and showed fragility/object-specific robustness.
- Benchmark deltas were demoted to supplemental analysis after ablation failed.
- The project shifted from hidden-state semantic probes toward action/phase readouts.
- Flow outcome probes now include constant, object-label, and object-label-plus-phase priors.

## What Has Not Been Solved

- We do not yet have an object-label + layout-prior baseline for every geometry probe.
- We do not yet have graded perturbation captures with full hidden/flow trajectories.
- We do not yet know whether probe-decoded geometry tracks actual displaced object positions under perturbation.
- We do not yet have sufficient data diversity to claim benchmark-independent generality.

## Practical Rule

Do not claim that a probe result means grounded target use unless it survives at least one of:

- object/layout-prior baseline
- target-vs-distractor selectivity control
- perturbation generalization
- causal intervention that moves the action/lift phase in the predicted direction

## Current Best Interpretation

The probe results show information availability. The action/phase and intervention results are the stronger evidence for use.

The flow outcome prior controls sharpen this: many outcome probes are weaker than object priors, especially `final_success` and early `future_min_target_dist`. Strong claims should therefore focus on variables where flow beats object/phase priors, or on recipient-fixed interventions where scene priors are held constant.
