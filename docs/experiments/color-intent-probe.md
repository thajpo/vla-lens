# Color Intent Probe Experiments: MiniVLA on LIBERO-90

## Thesis

VLA models are black boxes at deployment time. When a robot arm reaches toward an object, there is currently no way to know — from the model's internals — what it intends to interact with, how it plans to interact with it, or whether its internal intent matches the operator's instruction, until the action has already been executed. This project asks: **can we build a lightweight, real-time intent monitor that decodes target-object identity and interaction plan from a VLA's hidden activations before the action is committed?**

The red/white mug experiment (LIBERO tasks 71 and 72) is a controlled case study for this broader question. Two visually distinct objects sit in the same scene; the instruction specifies one. This is the minimal setting in which intent monitoring is both necessary (the scene is ambiguous) and verifiable (we know ground truth from the simulator). Every experiment in this document serves one of three roles:

- **Descriptive**: What does the model represent internally? (Experiments 2, 3, 4, 6)
- **Causal**: Is that representation actually used to select actions? (Experiment 5)
- **Applied**: Can we extract it in real time with calibrated confidence? (Experiment 7)

The safety contribution is the full pipeline: mechanistic understanding → real-time extraction → calibrated monitoring. The scientific contribution is the failure decomposition: separating "the model had the wrong intent" from "the model had the right intent but failed at execution," which has different implications for training, data collection, and human oversight.

Future work extends beyond color/object identity to interaction type (grasp type, approach vector, contact geometry), which requires task pairs that share the same object but differ in the required manipulation. This is scoped but deferred — the infrastructure built here supports it directly.

---

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

#### Additional metadata to capture per step

Beyond activations, the rollout harness must log the following at every step for downstream analyses (failure decomposition, behavioral confusion matrix, Experiment 1 temporal analysis):

| Field | Source | Type | Purpose |
|-------|--------|------|---------|
| `ee_pos` | `env.get_ee_pos()` or MuJoCo `sim.data.site_xpos` | float[3] | End-effector position for reach-direction analysis |
| `contacted_object` | MuJoCo contact pair detection | str or None | Which object (if any) the gripper is contacting this step |
| `target_mug_pos` | MuJoCo body xpos for target mug | float[3] | Ground-truth target location |
| `other_mug_pos` | MuJoCo body xpos for non-target mug | float[3] | Distractor location |
| `gripper_state` | Action dim 7 (gripper open/close) | float | Whether gripper is commanding open or close |
| `vq_codes` | VQ-VAE hard assignment output | int[7] | The 7 VQ codebook indices selected this step |
| `vq_logits` | Pre-quantization continuous vectors | float[7 × 128] | Soft assignments before hard VQ selection (needed for Exp 3, Exp 7) |
| `decoded_action` | VQ-VAE decoder output | float[7] | The continuous action chunk decoded from VQ codes |

The `contacted_object` field is critical for the failure decomposition. Without it, cell C ("probe correct + task failure") is ambiguous — you can't tell whether the arm reached for the right mug and fumbled the grasp, or whether the arm went somewhere unrelated. LIBERO's MuJoCo environment exposes contact pairs via `sim.data.contact`; filter for contacts involving the gripper body and either mug body.

### Independent Variables
- Layer index (0–23)
- Token position type: {instruction_color_word, final_token, EOS}
- Training set size (for learning curve)

### Dependent Variables
- Linear probe accuracy (logistic regression, L2 regularized, 5-fold stratified CV)
- AUROC
- Probe weight vector (for subsequent steering experiments)

### Controls

- **Scrambled label baseline**: Refit probe with shuffled task labels → should give ~50% accuracy. Any real probe must exceed this by a statistically significant margin (permutation test, not just point estimate).

- **Spatial confound control**: The red and white mugs occupy different positions in LIVING_ROOM_SCENE6. If the model has simply memorized "instruction says red → move left" without any visual grounding, the probe would still succeed but the representation wouldn't generalize. To test this: check whether LIBERO's initial state randomization varies mug positions across episodes. If it does, verify that probe accuracy holds across spatial configurations. If it doesn't (mug positions are fixed), acknowledge this as a limitation: **the probe may be detecting a learned spatial prior rather than a genuine color-object binding.** This is the single most important confound in the experiment.

- **Non-target object color baseline**: Train a probe to predict the *other* mug's color (the one not being grasped). In this specific setup (red and white are always both present), this is trivially anti-correlated with the target probe — if target is red, non-target is white. This control is only informative if extended to scenes with more than two objects, where the non-target probe should be at chance.

- **Language-only baseline**: Run the model's forward pass with the instruction tokens but *no image tokens* (or a blank/random image). Probe the same layers. If probe accuracy is similar to the full-model probe, the model is solving the task from instruction alone without visual grounding. This isolates the "did the model just read the word red" confound. Expected: language-only probe accuracy should be near 100% at the instruction token position (the word "red" is literally in the input) but lower at the final token position (where visual context should matter). If both are equally high, the model may not be visually grounding at all.

- **Cross-temporal generalization**: Train probe on step 0 activations, test on step T activations. If the intent representation is stable, accuracy should remain high across the trajectory. If it degrades, the model's intent representation is time-varying and the probe must be recalibrated per phase.

### Analysis
Train one probe per (layer, token_position) pair. Report accuracy as a heatmap over (layer × position). Identify the peak layer and position. Fit a sigmoid learning curve over training set size to estimate data efficiency.

### Expected Finding
Peak accuracy in layers 14–20 (upper-middle of the 24-layer stack), likely highest at the instruction color-word token position. If intent is represented, expect >85% accuracy at peak layer.

### Papers
- Alain & Bengio (2016) "Understanding Intermediate Representations with Linear Classifiers"
- Burns et al. (2023) "Discovering Latent Knowledge" (contrast consistent probing)
- Ghandeharioun et al. (2024) "Patchscopes" (token position selection)
- Lu et al. (2025) "Probing a VLA for Symbolic States"

---

## Experiment 1: Intent Probe from Action Space

### Hypothesis
The VQ codebook distribution over an episode is predictably different between red-mug and white-mug trials in a way that a linear classifier can detect from early steps — before the policy has committed to grasping either object.

### Setup
- From the same rollout data as Exp 2, extract the VQ code sequence at each step.
- VQ codes are 7 integers in [0, 127] per step.
- Derive a step-level feature vector: histogram of VQ codes across the 7 groups (7 × 128 = 896-dim one-hot, or just the 7-dim code vector).
- Train a linear classifier on step-level code vectors to predict task identity (red vs. white).
- Train one classifier per step index to get a temporal accuracy curve.

#### Relationship to Experiment 2

Experiment 1 and Experiment 2 ask the same question ("can we predict which mug the model targets?") but from different representations:

- Experiment 2 probes the **hidden state** (the model's internal computation, pre-action).
- Experiment 1 probes the **action output** (the model's behavioral commitment, post-action-decoding).

The key scientific result is the **lag** between these two. If hidden-state probe accuracy rises at step 3 but action-space probe accuracy rises at step 8, there are 5 steps where the model "knows" internally which mug it's targeting but hasn't yet committed to a distinguishable motor plan. That gap is the window in which a safety monitor could intervene.

If the two probes rise simultaneously, the model has no pre-commitment phase — intent and action are computed together. This would be a negative result for the safety-monitor narrative but still informative.

If the action-space probe rises *before* the hidden-state probe (unlikely but possible), this would suggest the model's action tokenizer compresses task-relevant information that the backbone doesn't explicitly represent, which would be surprising and worth investigating.

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

Adapted from Meng et al. (2022) ROME, with the targeted corruption approach recommended by NOTICE (2024) to avoid illusory patching artifacts from Gaussian noise.

**Corruption strategy**: We use instruction-swap corruption (replace "red" with "white" in the instruction text, keeping all other tokens identical) rather than Gaussian noise on image embeddings. This is superior to noise corruption for two reasons: (1) it produces a semantically valid but *wrong* forward pass rather than a degenerate one, making recovery scores interpretable; (2) it avoids the illusory patching artifacts that Palit et al. and the NOTICE authors identified, where noise corruption creates artificial "causal" signals at early layers that are actually just denoising artifacts.

**Important semantic note**: Because the corrupted instruction ("pick up white mug") is itself a valid LIBERO task, the corrupted forward pass produces a *coherent but wrong* action — the model will attempt to pick up the white mug. "Recovery" therefore means: patching at layer L causes the model to switch from the white-mug trajectory back to the red-mug trajectory. This is behavioral *flipping*, not noise recovery, which makes the causal claim stronger.

**Offline evaluation protocol**: Causal tracing runs on *saved observations* from clean rollouts, not live environments. The procedure is:

1. Run a clean rollout of task 71 (red mug), saving observations `{obs_0, obs_1, ..., obs_T}` at every step.
2. For each saved observation `obs_t`:
   a. **Clean pass**: Forward pass with `obs_t` + "pick up red mug" → record all `h_l^{clean}`, record output VQ codes and decoded action.
   b. **Corrupt pass**: Forward pass with `obs_t` + "pick up white mug" → record all `h_l^{corrupt}`, record output VQ codes and decoded action.
   c. **Patched passes** (one per patch site): Forward pass with `obs_t` + "pick up white mug", but at layer `l` and token position `p`, replace `h_l^{corrupt}` with `h_l^{clean}` → record output VQ codes and decoded action.
3. Compute recovery score for each patch site.

This is explicitly *not* a live rollout with patched actions executed in the environment. The environment state is frozen per step. This isolates the patching effect from cascading environmental changes.

**Why offline**: If you patch at step 5 and execute the patched action, step 6's observation changes, making step 6's patching results confounded by step 5's intervention. Offline evaluation keeps each step independent.

**Recovery metric**: `1 - ||a_patched - a_clean||₂ / ||a_corrupt - a_clean||₂`, clipped to [0, 1]. A score of 1.0 means the patched action exactly matches the clean action (full recovery). A score of 0.0 means the patch had no effect.

Run for all l ∈ {0..23} × positions ∈ {color_word_token, final_token} → 48 patch sites per step.

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

#### Additional corruption variants (secondary analysis)

Beyond the primary color-word swap, test these corruptions to characterize the breadth of the causal pathway:

- **Synonym corruption**: Replace "red" with "crimson" or "scarlet." If recovery patterns change, the model is sensitive to exact token identity, not just semantic content. This probes tokenizer-level versus concept-level representation.

- **Unrelated instruction corruption**: Replace the full instruction with a different LIBERO-90 task instruction (e.g., "close the top drawer"). If patching still recovers the red-mug action, the causal site stores intent independent of instruction format.

- **Visual corruption** (secondary mode): Instead of swapping the instruction, keep the instruction fixed but swap the image — feed an observation from task 72 with the instruction from task 71. Patch visual token positions. This tests whether the causal pathway runs through the visual tokens versus the language tokens. Requires `collect_activations.py` to save raw observations alongside hidden states.

### Analysis
Report recovery score as a heatmap over (layer × position). Identify the minimum set of layers sufficient for >90% recovery. This is the "intent localization."

### Expected Finding
A compact set of 2–4 layers in the upper half of the network dominates recovery. The color-word token position at those layers is the causally sufficient site.

### Papers
- Meng et al. (2022) "Locating and Editing Factual Associations in GPT" (ROME)
- Hernandez et al. (2023) "RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations"
- Geiger et al. (2023) "Finding Alignments Between Interpretability Techniques and Behavioral Phenomena"
- Palit et al. / NOTICE (2024) — illusory patching artifacts from noise corruption

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

**Critical note on step aggregation**: Do not average softmax uncertainty across all steps. Naive averaging poorly discriminates success from failure because successful trajectories contain high-entropy segments (Shifting Uncertainty to Critical Moments, 2026). Use max-based sliding window pooling: `episode_uncertainty = max(window_max(step_uncertainty, window=5))` over the first 30 steps. Focus on the pre-grasp phase.

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
Failure decomposition table:
```
                | Monitor: confident | Monitor: ambiguous
Success         |        TP          |        FP
Failure         |        FN          |        TN
```
Measure precision, recall, F1. Plot AUC of (ambiguity score) as predictor of failure.

### Expected Finding
Conformal monitor achieves exact marginal coverage by construction. Ambiguity rate has positive correlation with failure rate (expected Spearman r ≥ 0.4). Monitor is more reliable than raw softmax threshold. Max-pooled uncertainty outperforms average uncertainty for failure prediction.

### Papers
- Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification"
- SAFE (Toyota Research Institute, NeurIPS 2025) — binary success/failure conformal monitoring
- FIPER (NeurIPS 2025) — action-chunk entropy for uncertainty
- "Averaging Trap" / Shifting Uncertainty to Critical Moments (2026) — max-pooling for VLA uncertainty

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
CKA increases across layers. Cross-task CKA in upper layers is lower than within-task CKA. If Grant et al.'s visual-dominance finding holds here, CKA between visual tokens and action tokens should be high while language-action CKA remains lower. Finding the opposite (high language-action CKA) would suggest instruction-disambiguated tasks are a special regime where language drives action selection.

### Papers
- Kornblith et al. (2019) "Similarity of Neural Network Representations Revisited"
- Nguyen et al. (2021) "Do Wide and Deep Networks Learn the Same Things?"
- Grant et al. (ICLR 2026 Workshop) — visual vs. language pathway dominance in VLAs

---

## Experiment 6: Sparse Autoencoders

### Hypothesis
Hidden states in the MiniVLA LLM backbone contain sparse, monosemantic features — identifiable via SAE training — that activate selectively for "red mug target" or "white mug target" scenes.

### Setup
Train a TopK sparse autoencoder (SAE) with:
- Input: residual stream activations at the identified peak layer (from Exp 2), hidden_dim=896
- Latent dimension: 4096 (4× expansion) and 7168 (8× expansion, to test Dr. VLA's scaling recommendation)
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
- SAE latent dimension count (4096 vs. 7168)
- TopK value
- Layer (compare peak layer to mid-layer)

### Dependent Variables
- Reconstruction loss
- Number of selective features per class
- Fraction of features that are "memorized" (highly correlated with specific episode seeds) vs. "generalizable"
- Correlation between selective feature activation and task success

### Controls
- Compare SAE features to linear probe direction: do SAE-selective features project onto the probe weight vector?
- Ablation: zero out selective features → does success rate drop?
- Compare fraction-memorized vs. fraction-generalizable to Dr. VLA's published numbers on 7B models

### Expected Finding
Similar to Dr. VLA findings: majority of features correspond to memorized demonstration patterns. Some features show color selectivity, but pure monosemanticity is limited at 0.5B scale. The fraction-memorized metric should be lower than Dr. VLA's 7B results (less capacity = less memorization room). The 8× expansion ratio may be necessary to resolve superposed color features that the 4× model conflates.

### Papers
- Cunningham et al. (2023) "Sparse Autoencoders Find Highly Interpretable Features in Language Models"
- Dr. VLA / Swann et al. (March 2026) — SAEs on π0.5 and OpenVLA
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

### Behavioral Ground Truth for Cell Assignment

The 2×2 table requires a definition of `probe_correct` and `task_success`. But it also benefits from a finer-grained behavioral categorization that uses the `contacted_object` and `ee_pos` metadata:

| Behavior | Definition | Cell assignment |
|----------|-----------|-----------------|
| Correct grasp | Contacted correct mug, task succeeded | A |
| Lucky recovery | Probe predicted wrong mug, but task succeeded (model corrected mid-trajectory or probe layer was wrong) | B |
| Motor failure | Contacted correct mug but task failed (dropped it, placed it wrong, timeout) | C |
| Binding failure | Contacted wrong mug, or contacted nothing, task failed | D |
| Reach-direction error | End-effector moved toward wrong mug (computed from `ee_pos` delta vs. `target_mug_pos` delta) even if no contact occurred | D (subcategory) |

The `contacted_object` field enables this decomposition. Without it, cells C and D are distinguished only by the probe prediction, which is circular — you'd be using the probe to validate the probe.

**Reach-direction metric**: At each step, compute `cos(ee_velocity, direction_to_target_mug)`. Average over the first 10 steps of the episode (pre-grasp phase). If this cosine is negative (moving away from target), classify as a reach-direction error regardless of eventual outcome.

### Statistical Analysis
Fisher's exact test for association between probe_correct and task_success. Report odds ratio and 95% CI. With N=200 episodes per task (400 total), this has ~80% power to detect OR > 2.5.

### Why This Matters
This decomposition separates "the model did not understand the instruction" from "the model understood but failed at execution." These have different implications for safety, training data quality, and human oversight.

---

## Future Scope: Interaction Type Probing

The red/white mug experiment probes **what** the model intends to interact with. The natural extension is **how** it intends to interact — grasp type, approach angle, force profile. This requires task pairs that share the same target object but differ in the required manipulation:

- LIBERO tasks involving the same object with different verbs (e.g., "pick up X" vs. "push X" vs. "place X on Y")
- Tasks requiring top-down vs. lateral grasps of the same object

The infrastructure built for the color probe (activation capture, probe training, causal tracing) transfers directly. The only change is the label: instead of color identity, the probe predicts interaction type. This is deferred because LIBERO-90 tasks 71/72 use the same grasp type for both mugs, making interaction-type probing uninformative on this specific task pair.

The activation capture harness and metadata schema are designed to support this extension without modification — `contacted_object`, `ee_pos`, and `gripper_state` provide the labels needed for grasp-type classification.

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
