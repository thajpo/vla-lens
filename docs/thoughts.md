# Open Questions and Thinking-in-Progress

A place for half-formed ideas and questions worth returning to. Not polished —
these are threads to pull on.

---

## Probe Training and the Absence of Temporal Causality

**The question**: LLMs process a static sequence, not a temporal stream. How
does this affect how we should train and interpret probes on activations
collected across rollout steps?

### The core tension

Each rollout step yields one activation vector. Naively treating all steps from
all episodes as i.i.d. samples gives a large dataset, but those samples are not
independent — consecutive steps share nearly identical observations (the scene
changes slowly), and all steps within an episode share the same instruction
(hence the same color encoding). A probe trained this way has inflated apparent
accuracy because it is partly memorizing episode identity, not learning a
generalizable color representation.

### What the model actually does

The LLM has no recurrent state across steps. Each forward pass is completely
independent — the model re-reads the full instruction from scratch every step.
This means the color word token position should encode color identity equally
well at step 1 and step 150. The instruction encoding is stationary within an
episode, which is actually a *strength* for probe training: the target label
(color) is stable and the relevant signal does not drift.

The absence of temporal causality cuts both ways:
- It means we cannot interpret probe accuracy as evidence of "planning" or
  "anticipation" — the model is not accumulating information over time.
- But it also means we don't need to worry about the model "forgetting" the
  color word as the episode progresses (it re-reads it every step).

### Practical implications for Exp 2 (color probe)

1. **One sample per episode, not per step.** Either take a single step (e.g.,
   step 10, past the wait period) or mean-pool activations across steps within
   an episode. Then treat episodes as the i.i.d. unit. This is what
   `plot_probe_sweep.py` currently does (episode mean). It is correct.

2. **CV splits on episodes, not steps.** If you split on steps, test episodes
   bleed into train via correlated steps — a data leak. `StratifiedKFold` over
   episodes is the right unit.

3. **Step 10 vs episode mean.** Step 10 is cleaner (no autocorrelation
   artifacts from averaging correlated frames). Episode mean is more stable
   (less noise). Worth comparing both — if they give similar probe accuracy,
   either is defensible. If step 10 is much weaker, it suggests early steps are
   noisy before the arm has oriented.

### Open question: does probe accuracy vary across steps?

If the color probe is strong at step 1 and equally strong at step 150, that is
the expected result and suggests the representation is truly encoding the
instruction, not the visual state.

But if probe accuracy *degrades* late in failed episodes, that is more
interesting: it could mean the model's internal representation of "which mug to
pick up" becomes less crisp as the arm drifts away from the correct object.
This would be relevant to Exp 7 (conformal monitor) — a declining probe
confidence could be an early signal of failure.

Worth testing: train probe on step-10 activations only, then evaluate it on
step-10, step-50, step-100, step-150 activations from held-out episodes. Does
accuracy hold?

### Open question: what does the probe actually prove?

A linear probe decoding color from a hidden state shows the information is
*linearly accessible* at that layer and position. It does not show:
- That the model *uses* this representation when generating the action.
- That the representation is causal (Exp 5, activation patching, tests this).
- That the representation is exclusive to the color word token position
  (the color word identity could be redundantly encoded at many positions).

The probe is evidence that the information is there. Patching is evidence that
it matters. Both together make a strong claim.
