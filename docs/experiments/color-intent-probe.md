# Color Intent Probe Experiments: MiniVLA on LIBERO-90

## Overview

**Model**: MiniVLA (Qwen2.5-0.5B + DINOv2 + SigLIP + VQ-VAE action chunking)
**Tasks**: LIBERO-90 tasks 71 and 72 — both set in LIVING_ROOM_SCENE6, both containing `porcelain_mug_1` (white) and `red_coffee_mug_1` (red), same scene layout, differing only in which mug is the instruction target.
**Core question**: Does MiniVLA construct an internal intent representation (specifically, which colored object it is targeting) that is (a) linearly decodable, (b) causally necessary for correct action selection, and (c) exploitable as a safety monitor?

---

## Dependency Graph

```
Exp 2 (color probe)
    │
    ├──► Exp 1 (intent probe from action space)
    │
    └──► Exp 5 (causal tracing)  ──► Exp 7 (conformal monitor)
    
Exp 3 (VQ codebook structure)
    │
    └──► Exp 6 (sparse autoencoders)
    
Exp 4 (CKA)  ──► (cross-cutting, informs Exp 1 & 5)
```

**Recommended execution order**: 2 → 3 → 1 → 5 → 4 → 6 → 7

**Minimum viable path (safety narrative)**: 2 → 1 → 5 → 7

---

## Experiment 2: Color Probe (Start Here)

### Hypothesis
MiniVLA hidden states at at least one layer encode the target color (red vs. white mug) as a direction that is linearly separable from the color of other objects in the scene.

### Setup
- Run tasks 71 (red mug) and 72 (white mug) with `N ≥ 100` seeded episodes each.
- At each inference step, capture hidden states from all 24 LLM transformer layers at:
  - The final token position (last generated token)
  - The token corresponding to the color word in the instruction ("red" / "white")
  - The EOS / padding token
- Save activations as (episode_id, step, layer, position, hidden_dim=896) tensors.

### Independent Variables
- Layer index (0–23)
- Token position type: {instruction_color_word, final_token, EOS}
- Training set size (for learning curve)

### Dependent Variables
- Linear probe accuracy (logistic regression, L2 regularized, 5-fold stratified CV)
- AUROC
- Probe weight vector (for subsequent steering experiments)

### Controls
- **Scrambled label baseline**: refit probe with shuffled task labels → should give ~50% accuracy. Any real probe must beat this by margin.
- **Spatial confound control**: Include episodes with left/right position of the two mugs varied (if LIBERO initial states provide this variation). Probe must generalize across spatial positions to be credited as semantic.
- **Non-target object color baseline**: train a probe to predict the *other* mug's color (the one not being grasped). If this probe also achieves high accuracy from the same layer/position, the feature is scene-level not intent-level.

### Analysis
Train one probe per (layer, token_position) pair. Report accuracy as a heatmap over (layer × position). Identify the peak layer and position. Fit a sigmoid learning curve over training set size to estimate data efficiency.

### Expected Finding
Peak accuracy in layers 14–20 (upper-middle of the 24-layer stack), likely highest at the instruction color-word token position. If intent is represented, expect >85% accuracy at peak layer.

### Papers
- Alain & Bengio (2016) "Understanding Intermediate Representations with Linear Classifiers"
- Burns et al. (2023) "Discovering Latent Knowledge" (contrast consistent probing)
- Ghandeharioun et al. (2024) "Patchscopes" (token position selection)

---

## Experiment 1: Intent Probe from Action Space

### Hypothesis
The VQ codebook distribution over an episode is predictably different between red-mug and white-mug trials in a way that a linear classifier can detect from early steps — before the policy has committed to grasping either object.

### Why This Matters
If an intent signal appears in the action token space before the grasp occurs, it constitutes evidence that the model computes a categorical intent variable, not just reactive sensorimotor mapping. This is the scientific contribution: bridging representation (Exp 2) to behavioral commitment.

### Setup
- From the same rollout data as Exp 2, extract the VQ code sequence at each step.
- VQ codes are 7 integers in [0, 127] per step.
- Derive a step-level feature vector: histogram of VQ codes across the 7 groups (7 × 128 = 896-dim one-hot, or just the 7-dim code vector).
- Train a linear classifier on step-level code vectors to predict task identity (red vs. white).
- Train one classifier per step index to get a temporal accuracy curve.

### Independent Variables
- Step index in the rollout
- Representation: raw 7-dim codes vs. per-group histograms vs. decoded continuous action

### Dependent Variables
- Linear probe accuracy at each step
- Step at which accuracy first exceeds 70% (commitment point)

### Controls
- Randomized episode ordering (prevent temporal leakage)
- Train/test split by episode seed (not by step)
- Compare accuracy of action-space probe vs. hidden-state probe (Exp 2) at the same time step

### Analysis
Plot accuracy over time for (action-space probe, best hidden-state probe). The gap between where the hidden-state probe first succeeds and where the action-space probe first succeeds is the lag between internal intent formation and behavioral commitment.

### Expected Finding
Hidden-state probe succeeds ~5–15 steps earlier than action-space probe, consistent with intent being computed upstream of the action-decoding stage.

### Papers
- Mnih et al. style commitment timing in decision-making
- Srivastava et al. (2022) "Behavior Cloning from Observation"

---

## Experiment 5: Activation Patching (Causal Tracing)

### Hypothesis
The intent signal identified by Exp 2 is causally necessary: corrupting it causes the policy to select the wrong mug, and restoring it recovers correct behavior.

### Setup
Adapted from Meng et al. (2022) ROME / Hernandez et al. (2023) RAVEL causal tracing protocol.

Three forward passes per step:
1. **Clean**: correct instruction (e.g., "pick up red mug"), record all residual stream states `h_l^{clean}` for l=0..23.
2. **Corrupted**: wrong instruction ("pick up white mug"), record `h_l^{corrupt}` and final VQ codes.
3. **Patched**: run corrupted forward pass, but restore `h_l^{clean}` at one layer `l` and one token position. Measure VQ code recovery.

**Recovery metric**: Euclidean distance in decoded-action space between patched output and clean output, normalized by clean-corrupted distance. A score of 1.0 = full recovery; 0.0 = no recovery.

Run for all l ∈ {0..23} × positions ∈ {color_word_token, final_token} → 48 patch sites.

### Independent Variables
- Patch layer l
- Patch token position
- Patch site: residual stream, attention output, MLP output (if feasible)

### Dependent Variables
- Recovery score per patch site
- Whether patched VQ codes match clean VQ codes exactly (strict) or approximately (soft)

### Controls
- Patch a control layer that Exp 2 shows has low probe accuracy → should show near-zero recovery
- Vary the corruption: wrong color word vs. completely different instruction → check if recovery requires semantic similarity
- Null patch: patch same layer with zero vector → should give same result as corrupted baseline

### Analysis
Report recovery score as a heatmap over (layer × position). Identify the minimum set of layers sufficient for >90% recovery. This is the "intent localization."

### Expected Finding
A compact set of 2–4 layers in the upper half of the network dominates recovery. The color-word token position at those layers is the causally sufficient site.

### Papers
- Meng et al. (2022) "Locating and Editing Factual Associations in GPT" (ROME)
- Hernandez et al. (2023) "RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations"
- Geiger et al. (2023) "Finding Alignments Between Interpretability Techniques and Behavioral Phenomena"

---

## Experiment 7: Conformal Intent Monitor

### Hypothesis
A conformal predictor built on the intent probe (Exp 2) produces calibrated prediction sets whose non-conformity scores can serve as a real-time safety signal: when the policy's internal intent representation is ambiguous, the prediction set is large, and task failure rate is elevated.

### Setup
Use split conformal prediction (Angelopoulos & Bates 2021):
1. Reserve a calibration split of `n_cal ≥ 50` episodes from Exp 2 rollouts.
2. For each calibration episode and step, compute the softmax score of the correct intent label under the trained probe.
3. Set threshold q at the (1-α) quantile of (1 - softmax_correct) across calibration episodes and steps to guarantee marginal coverage `1-α` (e.g., α=0.1).
4. At deployment, output prediction set = {labels with softmax score > 1-q}.

**Safety signal**: At any step, if the prediction set contains both "red" and "white" (or is empty), flag as ambiguous. Log ambiguity rate vs. episode success rate.

### Independent Variables
- Coverage level α ∈ {0.05, 0.10, 0.20}
- Probe layer and position (use best from Exp 2)
- Aggregation: step-level vs. episode-level (first ambiguous step triggers flag)

### Dependent Variables
- Empirical coverage (should equal 1-α within ±0.02 on test set)
- Ambiguity rate per episode
- Precision/recall of ambiguity flag as a predictor of task failure

### Controls
- Compare conformal monitor to: (a) raw softmax threshold, (b) probe accuracy threshold
- FIPER baseline: use fixed prediction interval rather than conformal

### Analysis
Failure decomposition table (see also failure_decomp analysis in Exp 2):
```
                | Monitor: confident | Monitor: ambiguous
Success         |        TP          |        FP
Failure         |        FN          |        TN
```
Measure precision, recall, F1. Plot AUC of (ambiguity score) as predictor of failure.

### Expected Finding
Conformal monitor achieves exact marginal coverage by construction. Ambiguity rate has positive correlation with failure rate (expected Spearman r ≥ 0.4). Monitor is more reliable than raw softmax threshold.

### Papers
- Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"
- SAFE paper (conformal for robot safety monitoring, if available)
- FIPER: Fixed Prediction Intervals for Robot Evaluation

---

## Experiment 3: VQ Codebook Structure

### Hypothesis
The 128-class VQ codebook used for action chunking is not uniformly utilized across tasks 71 and 72. Certain codes are task-specific, and the codebook structure reflects semantic task clusters at the action level.

### Setup
- From rollout data: for each step, record the 7 VQ codes (one per group).
- Aggregate per-task histograms over all steps and episodes: `count[task, group, code]`.
- Compute per-group Jensen-Shannon divergence between task 71 and task 72 distributions.
- Use UMAP or PCA to embed the decoded actions (7D VQ code → 7D action via VQ decoder) and color by task.

### Independent Variables
- VQ group index (0–6)
- Task (71 vs. 72)
- Rollout phase (pre-grasp, grasp, post-grasp if segmentable)

### Dependent Variables
- Per-group JSD between tasks
- Separation of decoded action trajectories in 2D embedding
- Number of "exclusive" codes used only in one task

### Controls
- Compare to two tasks that do not share the same scene (e.g., task 0 vs. task 5) to check if same-scene similarity is meaningful

### Expected Finding
At least 2 of 7 VQ groups show JSD > 0.2 between tasks, reflecting that the reaching/grasping trajectory differs systematically between red and white mug targets.

### Papers
- van den Oord et al. (2017) "Neural Discrete Representation Learning" (VQ-VAE)
- Lee et al. (2024) VQ-BeT paper

---

## Experiment 4: CKA Representational Similarity

### Hypothesis
The visual subspace (derived from DINOv2 + SigLIP patch tokens) and the language subspace (derived from LLM residual stream at instruction tokens) show increasing alignment in upper layers, and this alignment predicts task success.

### Setup
Use centered kernel alignment (Kornblith et al. 2019):
```
CKA(X, Y) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))
```
where K = X X^T, L = Y Y^T (linear kernel).

- Collect hidden states from vision pathway (patch tokens after vision encoder) and language pathway (instruction tokens) at each LLM layer.
- Compute CKA between vision and language activations across all episodes.
- Track CKA per layer, per step, per episode outcome (success/failure).

### Independent Variables
- Layer index
- Step index (temporal)
- Modality pair: (vision, language), (vision, action output), (language, action output)

### Dependent Variables
- CKA score per (layer, step)
- Difference in CKA between success and failure episodes

### Controls
- Compute CKA with random (shuffled episode) pairings as null baseline
- Compare CKA within-task (71-71 or 72-72) vs. cross-task (71-72) to assess discriminability

### Expected Finding
CKA increases across layers (modalities align as information propagates). Cross-task CKA in the upper layers is lower than within-task CKA, suggesting semantic differentiation. CKA between vision and action tokens correlates positively with success.

### Papers
- Kornblith et al. (2019) "Similarity of Neural Network Representations Revisited"
- Nguyen et al. (2021) "Do Wide and Deep Networks Learn the Same Things?"

---

## Experiment 6: Sparse Autoencoders

### Hypothesis
Hidden states in the MiniVLA LLM backbone contain sparse, monosemantic features — identifiable via SAE training — that activate selectively for "red mug target" or "white mug target" scenes.

### Setup
Train a TopK sparse autoencoder (SAE) with:
- Input: residual stream activations at the identified peak layer (from Exp 2), hidden_dim=896
- Latent dimension: 4096 (4× expansion)
- TopK=32 (active features per forward pass)
- Training data: all episodes from Exp 2 rollouts, flattened to (n_steps × n_episodes, 896)

Training:
```
min_{W_enc, W_dec, b}  ||x - W_dec TopK(W_enc x + b)||^2
```
Evaluate each latent dimension's selectivity:
- Compute activation frequency per class (red vs. white)
- Flag features with class frequency ratio > 3.0 as "selective"

### Independent Variables
- SAE latent dimension count
- TopK value
- Layer (compare peak layer to mid-layer)

### Dependent Variables
- Reconstruction loss
- Number of selective features per class
- Correlation between selective feature activation and task success

### Controls
- Compare SAE features to linear probe direction: do SAE-selective features project onto the probe weight vector?
- Ablation: zero out selective features → does success rate drop?

### Expected Finding
Similar to Dr. VLA findings on π0.5: SAEs will find features, but they will be largely correlated with demonstrations rather than purely semantic. Some features may be color-selective, but pure monosemanticity is not expected.

### Papers
- Cunningham et al. (2023) "Sparse Autoencoders Find Highly Interpretable Features in Language Models"
- Dr. VLA (2024) "Interpreting Robot Foundation Models via Sparse Autoencoders"
- Templeton et al. (2024) "Scaling Monosemanticity"

---

## Failure Decomposition (Cross-Cutting Analysis)

This analysis runs on data from Exp 2 and applies to any probe experiment. It is the primary scientific contribution of this project.

### Setup
For each episode, compute:
- `probe_correct`: does the color probe (at peak layer, final step before grasp) correctly predict which mug was targeted?
- `task_success`: did the episode succeed?

Fill the 2×2 table:
```
                | probe_correct=True | probe_correct=False
task_success=1  |       A (ideal)    |        B (lucky)
task_success=0  |    C (exec fail)   |     D (intent fail)
```

**Cell interpretations**:
- **A**: Probe correct + success. Expected majority.
- **B**: Probe wrong + success. Model succeeded despite probe prediction being wrong — probe/layer choice may be off, or model recovered.
- **C**: Probe correct + failure. Intent was correctly represented but execution failed — motor planning error, grasping failure, not perception.
- **D**: Probe wrong + failure. Intent representation was wrong — likely the source of failure.

**Key test**: Is cell C / (C + D) significantly greater than 0.5? If yes, most failures are execution failures given correct intent, not perception failures.

### Statistical Analysis
Fisher's exact test for association between probe_correct and task_success. Report odds ratio and 95% CI. With N=200 episodes per task (400 total), this has ~80% power to detect OR > 2.5.

### Why This Matters
This decomposition separates "the model did not understand the instruction" from "the model understood but failed at execution." These have different implications for safety, training data quality, and human oversight.

---

## Data Collection Requirements

| Experiment | Episodes Needed | Hours (estimated MPS) |
|------------|-----------------|----------------------|
| Exp 2 (color probe) | 200 per task × 2 = 400 | ~8h |
| Exp 3 (VQ codebook) | same 400 rollouts | 0 (reuse) |
| Exp 1 (intent probe, action) | same 400 rollouts | 0 (reuse) |
| Exp 5 (causal tracing) | 50 clean+corrupt pairs | ~1h |
| Exp 4 (CKA) | same 400 rollouts | 0 (reuse) |
| Exp 6 (SAE training) | same 400 rollouts | ~2h CPU training |
| Exp 7 (conformal) | 100 cal + 100 test | ~4h (or split from above) |

**Minimum to publish**: 200 episodes per task (400 total). All Exp 2 + failure decomposition runs on this.
