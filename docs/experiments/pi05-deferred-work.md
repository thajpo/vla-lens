# PI0.5 Deferred Work

## Not immediate

These items are intentionally deferred until the benchmark roles are fully locked and the canonical analyses are written up.

## Deferred items

### 1. Scene 1 expert / demonstration sanity check

Goal:

- verify whether LIBERO expert demonstrations succeed on `ketchup` and `tomato_sauce` in `LIVING_ROOM_SCENE1`

Status:

- completed at a basic level in `docs/experiments/pi05-sanity-checks.md`

Follow-up still useful:

- replay the demonstrations end-to-end in the current env wrapper if we later need a stronger task-validity appendix

### 2. Full 20-episode Scene 1 expansion

Current status:

- 5 episodes per object show a clean split

Need:

- 20 episodes per object for tighter confidence intervals

Status:

- completed for canonical evaluation in `artifacts/pi05_libero/scene1_layout20_canonical.json`

Remaining follow-up:

- if needed, expand the Scene 1 swap sweep beyond 10 episodes per task

### 3. Phase 4 perturbation infrastructure

Planned perturbations:

- target-only displacement
- target-distractor swap

Status:

- completed for target-swap perturbations on the current benchmark subsets

Remaining follow-up:

- add target-only displacement perturbations
- verify success semantics under manual swap edits more rigorously

### 4. Internal probe capture for PI0.5

Likely hook points:

- `policy.model.paligemma_with_expert.paligemma`
- `policy.model.paligemma_with_expert.gemma_expert`
- `policy.model.action_in_proj`
- `policy.model.action_out_proj`

Reason deferred:

- benchmark choice and behavioral baselines should be stable first

### 5. Resource management note

Operational lesson:

- do **not** run two heavy `pi05` evaluation sweeps in parallel on this machine

Observed behavior:

- parallel runs caused instability and likely pushed the desktop into memory pressure

Policy going forward:

- run `pi05` sweeps sequentially
- kill lingering processes before starting the next run
- keep swap enabled during large-model evaluation
