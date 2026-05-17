# CogACT Intent Probe Experiments: VLM→DiT Interface

## Thesis

VLA models are black boxes at deployment time. When a robot arm reaches toward an object, there is currently no way to know — from the model's internals — what it intends to interact with until the action has already been executed. This project asks: **can we decode target-object identity from a VLA's internal representations before the action is committed, and can that decoded signal serve as a real-time safety monitor?**

The primary experimental vehicle is CogACT-Small on a robosuite two-cube selection task. CogACT's architecture gives a unique opportunity: the VLM (transformer backbone) produces a conditioning vector that is handed off to a separate DiT (diffusion action head). This interface is the key hypothesis test. If the conditioning vector already encodes "which cube" at near-perfect accuracy, the transformer did the thinking and the DiT is a trajectory renderer — interp on the transformer is sufficient for monitoring intent.

Every experiment serves one of three roles:

- **Descriptive**: What does the model represent internally? (Experiments 1, 2, 4, 6)
- **Causal**: Is that representation actually used to select actions? (Experiment 5)
- **Applied**: Can we extract it in real time and use it for safety? (Experiment 3, 7)

The safety contribution is the full pipeline: mechanistic understanding → real-time extraction → calibrated monitoring. The scientific contribution is the foundational result about where intent lives in componentized VLAs — transformer or action head?

---

## Model and Task

**Model**: CogACT-Small (Prismatic 7B VLM + DiT-S, MIT license, ~15 GB bf16)

**Task**: Robosuite `Stack` environment — two colored cubes (cubeA, cubeB) in a scene with deterministic seeded resets. Instruction specifies one cube. Label = which cube was selected.

**Why this task**: Minimal two-way selection with a clear ground-truth label. CogACT was pretrained on dexterous manipulation tasks; fine-tuning on robosuite may be needed for reliable rollouts. Start with scripted baseline to validate the scene, then bring in CogACT.

**Why CogACT over OpenVLA for this experiment**: OpenVLA has no architectural interface to probe — the plan and the action emerge from the same autoregressive token stream. CogACT's VLM→DiT handoff is a natural monitoring checkpoint.

---

## Dependency Graph

```
Exp 1 (VLM→DiT conditioning vector probe)
    │
    ├──► Exp 2 (DDIM denoising trajectory probe)
    │
    └──► Exp 5 (causal patching on conditioning vector)
              │
              └──► Exp 7 (conformal safety monitor)

Exp 4 (layer sweep: where in VLM does intent crystallize?)
    │
    └──► (informs monitor placement for Exp 7)

Exp 6 (OpenVLA comparison — autoregressive baseline)
```

**Recommended execution order**: 1 → 2 → 4 → 5 → 7 → 6

**Minimum viable path**: 1 → 2 → 5 → 7

---

## Experiment 1: VLM→DiT Conditioning Vector Probe (Start Here)

### Hypothesis

The conditioning vector the VLM passes to the DiT already encodes target object identity with near-perfect linear separability. The transformer did the thinking; the DiT just renders.

### Setup

- Run N ≥ 100 seeded episodes, split between cubeA and cubeB targets (50 each).
- At each forward pass, hook the conditioning vector at the VLM→DiT interface. Save per-episode.
- Label: target_object ∈ {cubeA, cubeB}.
- Train logistic regression probe: X = conditioning vector (flattened), y = target_object.
- 5-fold stratified CV, split by episode (not by step — all steps from one episode go to the same fold).

### Independent Variables

- Probe regularization C (sweep: 0.01, 0.1, 1.0)
- Training set size (for learning curve)

### Dependent Variables

- CV accuracy
- AUROC
- Probe weight vector (for Experiment 5 steering direction)

### Controls

- **Scrambled label baseline**: refit probe with shuffled labels → should give ~50%. Any real probe must exceed this by a statistically significant margin (permutation test).
- **Language-only baseline**: run VLM forward pass with instruction tokens but blank/random image. If probe accuracy on a blind-image run equals full-run accuracy, the model is solving from instruction text alone without visual grounding.
- **Cross-episode generalization**: train on episodes from seed 0–50, test on seeds 51–100. If accuracy drops, the probe is memorizing episode-level statistics rather than learning a generalizable direction.

### Interpretation

| Accuracy | Interpretation |
|----------|---------------|
| > 85% | Transformer latent is a complete plan. Hypothesis confirmed. Proceed to Experiment 2 to characterize DiT behavior. |
| ~70% | DiT contributes to decision. Do not declare confirmed until Experiment 2 completes. |
| ~50% | Data pipeline problem, or CogACT's conditioning vector in this setting doesn't encode target-relevant information. Investigate before scaling. |

### Papers

- Alain & Bengio (2016) "Understanding Intermediate Representations with Linear Classifiers"
- Burns et al. (2023) "Discovering Latent Knowledge"
- Lu et al. (2025) "Probing a VLA for Symbolic States" — same methodology, OpenVLA, 7B, >90% accuracy across layers

---

## Experiment 2: DDIM Denoising Trajectory Probe

### Hypothesis

Probe accuracy on DDIM intermediate states increases over denoising steps τ. The shape of the accuracy-over-τ curve determines whether the transformer or the DiT made the decision.

### Setup

- Same rollout data as Experiment 1.
- At each forward pass, capture intermediate trajectory estimates at each of 10 DDIM denoising steps. Each intermediate is shape (16, 7) = 112 floats.
- Train one probe per τ, OR train one probe pooled across all τ and evaluate per-τ.
- Label: same target_object.

### Independent Variables

- Denoising step τ ∈ {0, 1, ..., 9}
- Probe type: per-τ separate vs. pooled-then-evaluated

### Dependent Variables

- Accuracy at each τ
- Shape of the accuracy curve

### Curve Shapes and Interpretations

| Pattern | Interpretation |
|---------|---------------|
| High accuracy at τ=0-1, flat thereafter | DiT locks in target immediately. Transformer did the thinking. |
| Accuracy climbs steadily from τ=0 to τ=9 | DiT genuinely deliberates. Transformer latent was ambiguous. |
| Staged: "which object" spikes early, "approach direction" spikes late | DiT has a natural goal-first / motion-second decomposition. Novel finding. |
| Accuracy at τ=0 matches Exp 1 conditioning vector | The DiT initial state already contains the transformer's full plan. |

### Secondary Analysis

If accuracy is high at τ=1-2: run a reduced denoising experiment. Execute rollouts with DDIM truncated to 2 steps instead of 10. Does behavior change significantly (worse success rate, different object choice)? If not, most denoising steps are refinement, not decision-making. Implications for inference speed and safety monitoring latency.

### Papers

- Denoising trajectory probing is novel for VLAs. No direct prior work.
- Analogous to: probing Stable Diffusion intermediate states for compositional commitment (conceptual precedent only).

---

## Experiment 5: Causal Patching on Conditioning Vector

### Hypothesis

The conditioning vector probe direction is causally necessary: patching the conditioning vector from a "cubeA episode" into a "cubeB episode" causes the model to select cubeA instead.

### Setup

Adapted from Meng et al. (2022) ROME, using instruction-swap corruption rather than Gaussian noise.

**Corruption strategy**: Keep the image fixed; swap the instruction text (cubeA → cubeB). This produces a coherent but wrong forward pass — the model attempts to select cubeB. "Recovery" means patching the conditioning vector from the clean (cubeA) run causes the model to switch back to cubeA.

**Offline evaluation protocol**: Run on saved observations from clean rollouts. Per episode:
1. Clean forward pass: image + "pick up cubeA" → save conditioning vector, final action chunk.
2. Corrupt forward pass: same image + "pick up cubeB" → save conditioning vector, final action chunk.
3. Patched forward pass: corrupt instruction + DiT receives clean conditioning vector → save action chunk.
4. Compute recovery score.

**Recovery metric**: `1 - ||action_patched - action_clean||₂ / ||action_corrupt - action_clean||₂`, clipped to [0, 1]. Score of 1.0 = full recovery to clean behavior.

### Independent Variables

- Patch source: full conditioning vector vs. projected onto probe direction only
- Number of episodes (50 minimum)

### Dependent Variables

- Recovery score
- Whether the patched rollout selects cubeA (behavioral flip)

### Controls

- Null patch: replace conditioning vector with zero vector. Should give near-zero recovery.
- Probe direction only: project conditioning vector onto probe weight direction, patch only that component. If this recovers as well as the full vector, the probe direction is the causally relevant subspace.

### Papers

- Meng et al. (2022) ROME
- Palit et al. / NOTICE (2024) — illusory patching artifacts from noise corruption (why instruction-swap is better)

---

## Experiment 4: VLM Layer Sweep

### Hypothesis

Within the VLM backbone, there is a specific layer (or narrow range of layers) where target-object identity crystallizes in the residual stream. Identifying this layer determines the optimal placement for an inline safety monitor.

### Setup

- Use the same rollout episodes as Experiment 1.
- Register forward hooks at each of the VLM's transformer layers (32 layers for Llama 2 7B).
- Hook position: last instruction token position.
- Train one probe per layer, same target-object label, same CV protocol.
- Result: accuracy heatmap over (layer × token position).

### Token Positions to Sweep

- Last instruction token (the token at the end of the instruction, before vision tokens)
- Object-word token (the specific token encoding "cubeA" or "cubeB" in the instruction)
- Last token of the full sequence (final position before action generation)

### Expected Finding

Accuracy should rise from near-chance at early layers to high accuracy at mid-to-upper layers. The layer where accuracy first exceeds 80% is the minimum-depth monitor placement. If this is shallower than the conditioning vector layer, it suggests intent is encoded earlier in the forward pass than the handoff point.

### Notes

This is secondary to Experiments 1 and 2. Do not run until the conditioning vector probe result is clear.

### Papers

- Alain & Bengio (2016) — slope of probe accuracy across layers as diagnostic
- Belrose et al. (2023) — logit lens, tracking when the model "decides" layer by layer
- Molinari et al. (2025) "Emergent World Representations in OpenVLA" — world-model info concentrated in middle layers

---

## Experiment 7: Conformal Intent Monitor

### Hypothesis

A conformal predictor built on the conditioning vector probe produces calibrated prediction sets. When the prediction set is large (probe is uncertain), task failure rate is elevated. The monitor provides formal coverage guarantees without assuming probe calibration.

### Setup

Split conformal prediction (Angelopoulos & Bates 2021):
1. Reserve a calibration split of n_cal ≥ 50 episodes from Experiment 1 rollouts.
2. For each calibration episode, compute the softmax score of the correct target label under the trained probe.
3. Set threshold q at the (1-α) quantile of (1 - softmax_correct) to guarantee coverage 1-α.
4. At inference, output prediction set = {labels with softmax score > 1-q}.

**Safety signal**: If the prediction set contains both cubeA and cubeB (or is empty), flag as ambiguous. Log ambiguity rate vs. episode success rate.

**Temporal aggregation**: Use max-based sliding window pooling over the first 20 steps, not naive averaging. Naive averaging over a long trajectory poorly discriminates success from failure (see "Averaging Trap" / Shifting Uncertainty to Critical Moments, 2026).

### Dependent Variables

- Empirical coverage (should equal 1-α within ±0.02)
- Ambiguity rate per episode
- Precision/recall of ambiguity flag as predictor of task failure

### Papers

- Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction"
- SAFE (Toyota Research Institute, NeurIPS 2025) — conformal monitoring on OpenVLA
- FIPER (NeurIPS 2025) — action-chunk entropy for VLA uncertainty
- "Averaging Trap" / Shifting Uncertainty to Critical Moments (2026) — max-pooling for failure prediction

---

## Experiment 6: OpenVLA Comparison (Autoregressive Baseline)

### Hypothesis

The same target-object probe trained on OpenVLA hidden states (last instruction token, mid layers) achieves similar accuracy to the CogACT conditioning vector probe. But because OpenVLA has no separate action head, there is no clean monitoring checkpoint — the probe must be placed inside the residual stream.

### Setup

- Run same episodes with OpenVLA backend.
- Hook the residual stream at each layer, last instruction token position.
- Train probes per layer, same label.
- Compare:
  - Peak accuracy (CogACT conditioning vector vs. OpenVLA best layer)
  - Which layers achieve > 80% accuracy (is it more or fewer in OpenVLA?)
  - Whether the probe weight direction is geometrically similar across models

### Notes

This comparison is valuable primarily to contextualize the CogACT results. If OpenVLA achieves similar accuracy with no clean monitoring checkpoint, the argument for CogACT-style componentized architectures as more monitorable is weakened. If CogACT's conditioning vector probe significantly outperforms OpenVLA's best layer probe, it supports the claim that explicit architecture boundaries are better for safety.

OpenVLA backend is already working. This experiment is low cost.

---

## Failure Decomposition (Cross-Cutting)

For any completed rollout dataset, compute:

| | probe_correct=True | probe_correct=False |
|---|---|---|
| task_success=1 | A (ideal) | B (lucky) |
| task_success=0 | C (execution failure) | D (intent failure) |

- **C** is the most informative cell: correct intent but execution failed. These are motor planning errors, not perception errors. Training more data helps D; improving action head quality helps C.
- **D**: intent was wrong. The model misidentified the target. Relevant to adversarial injection scenarios — a perturbed conditioning vector would push episodes into D.

This decomposition separates "model understood but fumbled" from "model misunderstood the instruction," which has different implications for training and safety.

---

## Data Collection Requirements

| Experiment | Episodes Needed | Estimate |
|------------|-----------------|----------|
| Exp 1 (conditioning vector probe) | 100 (50 per target) | Pilot: 40 eps first |
| Exp 2 (DDIM trajectory) | Same 100 rollouts | 0 extra (reuse) |
| Exp 5 (causal patching) | 50 clean + patched | ~0.5× data cost |
| Exp 4 (layer sweep) | Same 100 rollouts | 0 extra (reuse with more hooks) |
| Exp 7 (conformal) | 100 cal + 100 test | Split from Exp 1 |
| Exp 6 (OpenVLA comparison) | Same 100 episodes | Re-run with different backend |

**Minimum to publish**: 100 episodes per target (200 total). Exp 1 + Exp 2 + failure decomposition runs on this.

**Pilot protocol**: Run 20 episodes first (10 per target). Check: (a) do conditioning vectors have visible separation in PCA? (b) is pilot probe accuracy > 60%? If yes to both, proceed to full collection. If no to either, investigate before scaling.
