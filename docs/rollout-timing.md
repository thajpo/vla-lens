# Rollout Timing Reference

> **Note**: The measurements below are from MiniVLA (Qwen2.5-0.5B) on Apple Silicon MPS.
> The current project target is CogACT-Small (Prismatic 7B) on ROCm (Radeon 7900 XTX).
> These numbers are stale for planning purposes. Update this file after the first CogACT
> timing runs. The qualitative patterns (failed episodes run to timeout, successful
> episodes terminate early) will likely hold.

---

# Rollout Timing on Apple Silicon (MPS) — MiniVLA Reference (Archived)

Empirical measurements from completed runs on MiniVLA (minivla-vq-libero90-prismatic),
measured 2026-04-12, Apple Silicon MPS backend.

---

## Per-Task Step Counts (post-tokenizer-fix runs)

From 5-episode runs after the tokenizer_len fix (151921, not 151936):

| Task | Description | Episodes | Steps (each) | Mean Steps | Successes |
|------|-------------|----------|-------------|-----------|-----------|
| 0 | close top drawer of cabinet | 5 | 78, 70, 75, 74, 73 | 74 | 5/5 |
| 5 | put chocolate pudding in drawer and close | 5 | 176, 400, 400, 400, 400 | 355 | 1/5 |
| 15 | put middle black bowl on top of cabinet | 5 | 141, 138, 210, 130, 134 | 151 | 5/5 |
| 30 | put black bowl on plate | 5 | 159, 169, 126, 141, 156 | 150 | 5/5 |
| 50 | pick up alphabet soup and put in basket | 5 | 400, 400, 164, 400, 229 | 319 | 2/5 |
| 70 | put chocolate pudding to right of plate | 3 | 400, 400, 400 | 400 | 0/3 |

**Key pattern**: Successful episodes terminate early (70–210 steps). Failed episodes always run to `max_steps=400` (the timeout).

---

## Wall-Clock Timing

Measured via filename timestamps (start) and file mtime (end). Model loading is
excluded — these are inference-only figures for a warm model.

| Task | Mean Steps | s/step (MPS) | s/episode | min/episode |
|------|-----------|-------------|----------|-------------|
| 0 | 74 | 5.4* | 401 | 6.7 |
| 5 | 355 | 5.1* | 1796 | 29.9 |
| 15 | 151 | 1.4 | 218 | 3.6 |
| 30 | 150 | 1.1 | 162 | 2.7 |
| 50 | 319 | 1.0 | 309 | 5.2 |
| 70 | 400 | 1.2 | 465 | 7.7 |

*Tasks 0 and 5 include model loading time (~5–10 min) in the wall estimate.
True inference-only s/step based on tasks 15–70: **~1.1–1.4 s/step**.

**Reliable figure**: ~1.2 s/step for warm-model inference on MPS.

---

## Overhead from Activation Capture

Activation capture (`collect_activations.py`) adds per-step overhead:
- Hook registration/deregistration: ~1ms (negligible — hooks are registered once
  per episode, not per step, after the `collect_task` refactor)
- Tensor detach + CPU copy: ~5–20ms per layer per step (depends on hidden_dim=896)
- At 6 layers × 2 positions: 12 tensors × ~10ms = ~120ms/step overhead

**Estimated capture overhead**: +10–15% per step → ~1.35 s/step with 6 layers.

---

## Time Budget for Full Data Collection

### Tasks 71/72 Expected Step Counts

Tasks 71 and 72 (red mug / white mug on plate, LIVING_ROOM_SCENE6) have not been
run with the VQ model yet. Estimates based on task complexity:
- Similar to task 15/30 (single-object pick-and-place): success rate ~60–80%
- Expected mean steps: ~150–250 per episode (mix of early successes + full-timeout failures)

**Conservative estimate**: 250 mean steps, 1.35 s/step with capture.

### Projection Table

| Scenario | Episodes | Mean Steps | s/step | Total Hours |
|----------|----------|-----------|--------|-------------|
| Pilot: 20 eps/task × 2 tasks | 40 | 250 | 1.35 | ~3.8h |
| Full: 200 eps/task × 2 tasks | 400 | 250 | 1.35 | ~37.5h |
| Full (worst case: all fail) | 400 | 400 | 1.35 | ~60h |
| Full (best case: all succeed ~150 steps) | 400 | 150 | 1.35 | ~22.5h |

### Practical Implications

1. **Run the pilot overnight** (20 eps/task): confirms success rate and actual step distribution before committing to the full run.

2. **Expect 2–4 days of compute** for 200 episodes/task on MPS. Consider running in sessions:
   - 100 eps/task (2 tasks) → ~18h → stop → analyze → continue if pilot results look good.

3. **Success rate on 71/72 matters a lot**: if the VQ model achieves >70% success on both tasks, mean steps drop and full collection finishes in ~22h. If success is low (<30%), budget 50–60h.

4. **Model loading time**: ~3–5 minutes per process startup. Prefer fewer long runs over many short runs.

---

## Notes on `max_steps` for Tasks 71/72

The current `MAX_STEPS_BY_SUITE` in `run_libero_task.py` sets `libero_90 → 400`.
Tasks 71/72 are pick-and-place with a single object — similar to task 30 (150 steps
for successes). If model achieves high success rate, `max_steps=400` wastes ~250 steps
per success. Consider lowering to 300 after the pilot to reduce collection time.
The experiments do not depend on running to 400 steps — early-termination episodes
are valid data points.
