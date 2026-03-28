# TODO

## Project Goal

Determine whether a VLA internally represents which object it is going to pick before overt motor commitment, and later test whether that internal target signal can be steered.

The intended setup is:
- two candidate objects in scene
- randomized object positions
- episode-level label = eventually selected / grasped object
- probes trained on hidden activations across layers and rollout timesteps

## What We Have

- deterministic robosuite stack environment wrapper
- episode rollouts and summary logging
- backend abstraction for `scripted_pick`, `openvla`, and `minivla`
- working OpenVLA action path into robosuite
- OpenVLA smoke run that produced a valid 7D action under `bridge_orig`
- MiniVLA backend now using the upstream Prismatic loader path instead of unsupported HF AutoClasses

## What We Do Not Have Yet

- probe-ready dataset
- per-step rollout logging saved in a lean analysis-friendly format
- hidden-state capture
- intervention hooks
- proof that MiniVLA completes a rollout in this environment

## Immediate Next Steps

1. Finish MiniVLA smoke validation.
2. Add lean per-episode logging:
   - episode metadata
   - per-step action
   - per-step end-effector position
   - per-step object positions
   - eventual selected object label
3. Keep activation capture optional and narrow:
   - selected layers only
   - early rollout steps only
   - one pooled vector per chosen layer / step
4. Materialize a probe dataset split by episode, not timestep.
5. Train the first activation-only probe before adding interventions.

## Anti-Bloat Rules

- do not dump all activations by default
- do not save raw images for every step unless needed
- do not add generic monitoring infrastructure
- prefer opt-in capture flags over always-on logging
