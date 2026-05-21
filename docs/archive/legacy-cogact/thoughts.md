# Open Questions and Thinking-in-Progress

A place for half-formed ideas and questions worth returning to. Not polished —
these are threads to pull on.

**Current model target**: CogACT-Small (Prismatic 7B VLM + DiT-S). The notes
below were originally written with MiniVLA in mind but the core reasoning
applies to CogACT. Layer count = 32 (Llama 2), hidden_dim = 4096. Conditioning
vector shape = to be determined from model inspection.

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
(hence the same target-object encoding). A probe trained this way has inflated
apparent accuracy because it is partly memorizing episode identity, not learning
a generalizable direction.

### What the model actually does

The LLM has no recurrent state across steps. Each forward pass is completely
independent — the model re-reads the full instruction from scratch every step.
This means the instruction token position should encode target identity equally
well at step 1 and step 150. The instruction encoding is stationary within an
episode, which is actually a *strength* for probe training: the target label
is stable and the relevant signal does not drift.

The absence of temporal causality cuts both ways:
- It means we cannot interpret probe accuracy as evidence of "planning" or
  "anticipation" — the model is not accumulating information over time.
- But it also means we don't need to worry about the model "forgetting" the
  target object as the episode progresses (it re-reads it every step).

### Practical implications for Exp 1 (conditioning vector probe)

For CogACT, the probe input is the conditioning vector from a single forward
pass, not a step-level activation. Since CogACT predicts a 16-step action
chunk per forward pass (not one step), the "per forward pass" granularity is
already coarser. But the same principle applies:

1. **One sample per episode, not per forward pass.** Take the conditioning
   vector at a fixed forward pass (e.g., the 3rd call, after the arm has
   oriented). Or mean-pool conditioning vectors across forward passes within
   an episode. Treat episodes as the i.i.d. unit.

2. **CV splits on episodes, not forward passes.** Same logic: forward passes
   within an episode are correlated (the image changes slowly). `StratifiedKFold`
   over episodes is the right unit.

3. **First forward pass vs. episode mean.** The first forward pass uses only
   the initial scene image — most relevant for testing whether the conditioning
   vector encodes intent before any motor commitment. Episode mean is more
   stable but harder to interpret causally.

### Open question: does probe accuracy vary across forward passes?

For CogACT, the analog question is: does the conditioning vector probe trained
on forward pass #1 transfer to forward passes #5, #10, #20? If it degrades late
in failed episodes, the model's intent representation is becoming less crisp as
the arm drifts from the correct cube. That is directly relevant to Exp 7
(conformal monitor) — declining probe confidence = early failure signal.

Worth testing: train probe on forward-pass-1 conditioning vectors, evaluate on
forward-pass-5, forward-pass-10, forward-pass-20 conditioning vectors from
held-out episodes. Does accuracy hold?

For DDIM: the accuracy-over-τ curve (Exp 2) should be stable across episodes.
If the curve shape changes between successful and failed episodes, that is a
new finding — the diffusion process commits differently depending on eventual
success.

### Open question: what does the probe actually prove?

A linear probe decoding target identity from the conditioning vector shows the
information is *linearly accessible* at that interface. It does not show:
- That the model *uses* this representation when generating the action.
- That the representation is causal (Exp 5, conditioning vector patching, tests this).
- That the representation is not redundantly encoded elsewhere in the DiT or
  in the denoising process (Exp 2 characterizes whether the DiT independently
  encodes it).

The probe is evidence that the information is there. Patching is evidence that
it matters. Both together make a strong claim.

For CogACT specifically: if conditioning vector probe accuracy is near-perfect
(Exp 1) AND replacing the conditioning vector changes object choice (Exp 5),
that is the complete argument. The transformer latent is both necessary and
sufficient for intent.

---

## The Diffusion Case: Does the DiT "Know" Instantly or Deliberate?

**Relevant to Experiment 2.**

The conditioning vector is provided to the DiT at τ=0 as a fixed input — it
does not change across denoising steps. So the DiT has the full conditioning
information available from the very first step. The question is whether it
*uses* that information immediately or whether it takes several steps to "fold
in" the conditioning signal.

Two ways to interpret a late accuracy spike in the DDIM trajectory:

1. The DiT uses the conditioning vector only weakly at first (the denoising
   starts near-uniform over all possible targets) and the conditioning signal
   takes time to dominate the trajectory. This is about *how* the DiT processes
   its conditioning, not about *what* it knows.

2. The conditioning vector itself is ambiguous (low Exp 1 accuracy), and the
   DiT is resolving the ambiguity through its own computation.

These are distinguishable: if Exp 1 accuracy is high (> 85%) but DDIM accuracy
spikes late (interpretation 1), the issue is the DiT's conditioning dynamics,
not the information content. If Exp 1 accuracy is low (interpretation 2), the
hypothesis fails for this model.

This distinction matters for where to place the safety monitor. If the
conditioning vector is high-accuracy, the monitor sits there regardless of the
DDIM curve shape.
