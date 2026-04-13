# Literature Review: Mechanistic Interpretability for Robot Policies

Thematic synthesis of the research landscape relevant to this project.

---

## 1. Linear Probing and Representation Geometry

**Core finding**: Hidden states in large neural networks are geometrically structured. Linear probes — logistic regression fits on frozen hidden states — reliably recover high-level semantic variables well before the output layer.

**Key ideas**:
- Alain & Bengio (2016) showed that intermediate layers in deep networks are increasingly accurate predictors of downstream labels. The slope of probe accuracy across layers is itself diagnostic — steep slope = information is assembled across layers, plateau = stable representation.
- Burns et al. (2023) introduced *contrast-consistent search* (CCS): find probe directions that are maximally self-consistent (if a probe gives 90% for P, it should give 10% for ¬P). This addresses the concern that probes recover the model's "beliefs" rather than arbitrary statistics.
- Belrose et al. (2023) showed that the logit lens (projecting intermediate residuals into vocab space) can track when a model "decides" its output, layer by layer. Equivalent protocol applies to robot policies: project intermediate states into action token space and check when action probabilities peak.
- **Lu et al. (2025)** "Probing a VLA for Symbolic States": Trained linear probes on all 33 hidden layers of OpenVLA's Llama-2 backbone to predict symbolic object and action states, achieving >90% accuracy across most layers. Integrated results into a real-time cognitive architecture. Directly validates our probing methodology — if it works at 7B, it should work at 0.5B, though accuracy may be lower.
- **Molinari et al. (2025)** "Emergent World Representations in OpenVLA": Discovered that internal activations predict state transitions better than raw embeddings, with world-model information concentrated in middle layers. Linear probes outperformed MLPs — consistent with the linear representation hypothesis. Predicts that our probe accuracy should peak in middle-to-upper layers (roughly layers 10–18 of 24 in Qwen2.5-0.5B).

**Applicability here**: Color probe training follows standard Alain & Bengio protocol. CCS is a sanity check: probe for "red mug is target" and "white mug is target" should be anti-correlated by construction. If not, the probe is recovering something else (e.g., scene statistics).

**Key concern**: Probes can be accurate without being causal. A probe on a redundant, read-only copy of task information would succeed but intervention on it would have no effect. Must pair with Exp 5 (causal tracing) to establish causal relevance.

---

## 1.5 VLA-Specific Mechanistic Interpretability

**Core finding**: A small but rapidly growing body of work (late 2025–early 2026) has begun applying mechanistic interpretability directly to VLA models, revealing that these models predominantly memorize training demonstrations, rely on visual pathways more than language, and retain steerable semantic directions from pretraining.

**Key papers**:

- **Häon et al. (CoRL 2025)**, "Mechanistic Interpretability for Steering VLAs": The first mechanistic interpretability framework for VLAs. Projected FFN value vectors in π0 and OpenVLA onto the token embedding basis to identify sparse semantic directions ("speed," "up," "slow") causally linked to action selection. Key finding: less than 25% of FFN neurons are rewired during VLA fine-tuning; the rest retain pretrained semantics. Activation steering using these directions achieves zero-shot behavioral control on both LIBERO and a physical UR5, without retraining. This is the closest prior work to our project — they demonstrate steering, we aim for monitoring.

- **Dr. VLA / Swann et al. (March 2026)**, "Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models": Applied SAEs to π0.5 and OpenVLA residual streams. The majority of SAE features correspond to memorized sequences from specific training demonstrations rather than generalizable primitives. A minority encode interpretable motion primitives (pre-grasp alignment, approach vectors). Steering these features in closed-loop rollouts causally modifies behavior. Directly informs our Experiment 6: expect mostly memorized features at 0.5B scale too, but possibly a higher generalizable fraction due to lower capacity.

- **Grant et al. (ICLR 2026 Workshop)**, "Not All Features Are Created Equal": Scaled mechanistic analysis across six VLA architectures (π0.5, OpenVLA-OFT, X-VLA, SmolVLA, GR00T N1.5, ACT) on 394,000+ rollout episodes. Three architecture-independent findings: (1) **the visual pathway dominates action generation**, (2) language sensitivity depends on task structure not model design, (3) expert pathways encode motor programs while VLM pathways encode goal semantics. If our color probes on language tokens show high accuracy, this *contradicts* their visual-dominance finding for this specific task type (instruction-disambiguated), which would be an interesting result.

- **Lu et al. (2025)**, "Probing a VLA for Symbolic States": (Also cited in §1.) Trained linear probes on all 33 hidden layers of OpenVLA's Llama-2 backbone, achieving >90% accuracy for symbolic object/action states across most layers.

- **Molinari et al. (2025)**, "Emergent World Representations in OpenVLA": (Also cited in §1.) World-model information concentrated in middle layers; linear probes outperform MLPs.

**Gap this project fills**: Every one of these studies targets 7B+ parameter models (OpenVLA, π0.5). No mechanistic work exists on sub-1B VLAs. No work examines how VQ-VAE action tokenizers interact with semantic representations. No work frames interpretability as a safety monitoring tool with calibrated uncertainty. Our project fills three gaps simultaneously — smaller scale (0.5B), VQ-VAE action tokenizers, and safety-oriented framing with conformal prediction.

---

## 2. Causal Tracing and Factual Localization

**Core finding**: The causal contribution of internal computations to model outputs can be measured by the "activation patching" protocol: corrupt input, restore clean activations at a specific site, measure output recovery.

**Key ideas**:
- Meng et al. (2022) ROME: applied causal tracing to GPT-J and found that factual associations are primarily stored in specific MLP layers. This is the canonical reference for the activation patching protocol.
- Hernandez et al. (2023) RAVEL: extended causal tracing to disentangled feature identification. Key result: the site that controls "which city" differs from the site that controls "which country," even for related facts. Implies that intent features may be localized differently from scene features.
- Geiger et al. (2023) DAS (Distributed Alignment Search): more general form — search over *linear* subspaces, not just activation vectors, for causal relevance. Strictly more powerful than activation patching.

**Critical methodological note**: Palit et al. (2023) and the NOTICE authors (2024) demonstrated that Gaussian noise corruption — the standard Meng et al. protocol — can produce illusory patching effects. When input is corrupted with noise, early-layer patching appears causally important because it partially restores the signal-to-noise ratio, not because the layer stores task-relevant information. NOTICE's solution is Semantic Image Pairs: pairs of images that differ only in the target attribute, producing valid (not noisy) corrupted forward passes. Our Experiment 5 uses instruction-swap corruption (replacing "red" with "white"), which achieves the same effect: the corrupted forward pass processes a valid, semantically coherent input, so recovery at any layer reflects genuine causal contribution rather than denoising.

**Applicability here**: The activation patching protocol in Exp 5 closely follows ROME. The main adaptation is that "output" is now VQ codes (action tokens) rather than next-word logits. Recovery metric is action-space rather than logit-space.

---

## 3. Token Position Analysis

**Core finding**: In autoregressive transformers, different token positions carry systematically different information. The residual stream at the subject token of a factual query is the primary site for factual retrieval (Meng et al.). Last-token positions are used for prediction.

**Key ideas**:
- Ghandeharioun et al. (2024) Patchscopes: a unified framework for "reading out" intermediate representations at different positions by transferring them to a separate model. Relevant for our logit-lens style analysis.
- The "attention sink" phenomenon (Xiao et al., 2023): the first token accumulates disproportionate attention in many LLM architectures. In MiniVLA, image patch tokens may play a similar structural role. The color-word instruction token is likely the semantically loaded position.
- For VLMs specifically, the patch tokens from the vision backbone serve as a prefix to the LLM, and the LLM must do cross-modal integration at every layer. The transition from visual to language-dominated representations typically peaks at an intermediate layer.

**Applicability here**: Run probes at multiple positions (color word, EOS, final token). Expect the color-word token to dominate for intent, but final token position may dominate for action-commitment signals.

---

## 4. Sparse Autoencoders for Feature Discovery

**Core finding**: MLP activations in transformers are polysemantic (each neuron responds to many unrelated concepts). Sparse autoencoders (SAEs) trained on these activations decompose them into sparse, more monosemantic features.

**Key ideas**:
- Cunningham et al. (2023): trained SAEs on GPT-2 MLP layers and found features selective for specific semantic content. Key design choice: use TopK sparsity rather than L1 to control the number of active features per forward pass.
- Templeton et al. / Anthropic (2024) "Scaling Monosemanticity": scaled SAEs to Claude 3 Sonnet. Found millions of interpretable features, including multimodal features and abstract concepts. Features form a "feature geometry" — similar concepts have nearby directions in the SAE latent space.
- **Dr. VLA / Swann et al. (March 2026)**: Applied SAEs to π0.5 and OpenVLA. Key findings: (a) most high-frequency features correspond to memorized demonstration-level patterns rather than semantic task concepts, (b) a small number of features show selectivity for object identities, but coverage is sparse. Most directly relevant prior work.

**Scaling considerations for 0.5B models**: All published VLA SAE work uses 7B+ parameter models. At 0.5B, several properties likely change:

- **Less capacity for memorization**: The 0.5B backbone has ~14× fewer parameters than OpenVLA's 7B. With LIBERO-90's 90 tasks, there's less room to memorize individual demonstrations, so a higher fraction of SAE features may be generalizable primitives rather than demonstration-specific patterns.
- **More severe superposition**: Smaller models must pack more concepts per dimension, so SAEs may require larger expansion factors (8× or 16× rather than 4×) to resolve superposed features.
- **Fewer interpretable motion primitives**: With less capacity, the model may use simpler, less decomposable strategies.

These are testable hypotheses for Experiment 6. Comparing our SAE statistics (fraction interpretable, fraction memorized, reconstruction loss vs. expansion factor) against Dr. VLA's published numbers on 7B models would be the first scaling law for VLA interpretability.

**Applicability here**: Given Dr. VLA's findings, expectation is that SAEs will find some color-selective features but with low coverage. The productive comparison is: do SAE-selective features align with the linear probe direction from Exp 2? If yes, the probe finds a compressed version of what the SAE decomposes.

---

## 5. Representational Similarity (CKA)

**Core finding**: CKA (Centered Kernel Alignment) is a robust metric for comparing neural representations across architectures or conditions, invariant to invertible linear transformations and orthogonal rotations.

**Key ideas**:
- Kornblith et al. (2019): introduced CKA for comparing layers within and across neural networks. Linear CKA (using dot-product kernel: K = X X^T) is computationally efficient and empirically reliable.
- Nguyen et al. (2021): applied CKA to compare wide and deep network representations, finding that deeper layers are more similar across architectures than shallow layers (convergent evolution of representation).
- For VLMs: visual and language representations are most aligned in the upper layers, where cross-attention (or implicit cross-modal integration) has operated for more layers.

**Specific predictions from VLA literature**: Grant et al.'s finding that visual pathways dominate action generation predicts that CKA between vision tokens and action tokens should be high in upper layers, while CKA between language tokens and action tokens should remain low. If we observe the opposite (high language-action CKA), it would suggest that instruction-disambiguated tasks are a special regime where language matters more than the architecture-averaged findings suggest. This would be an interesting positive result given the "visual dominance" consensus.

**Applicability here**: CKA between vision patch tokens and instruction tokens at each LLM layer gives a direct readout of when cross-modal integration has occurred. If visual and linguistic representations are most aligned at the layer where Exp 2 finds the highest probe accuracy, this is convergent evidence.

**Key concern**: CKA is descriptive, not causal. High CKA means the representations are similar in structure, not that one caused the other. Pair with causal tracing (Exp 5) for causal claims.

---

## 6. Conformal Prediction for Robot Safety

**Core finding**: Split conformal prediction provides finite-sample, distribution-free coverage guarantees for prediction sets. A well-calibrated conformal predictor covering at level 1-α will contain the true label in at least (1-α) fraction of test episodes, regardless of the underlying distribution.

**Key ideas**:
- Angelopoulos & Bates (2021) "A Gentle Introduction": clearest reference for the split conformal protocol. Non-conformity score can be any function of input and label (e.g., 1 - softmax_correct); the threshold is the (1-α) quantile of calibration scores.
- **SAFE (Toyota Research Institute, NeurIPS 2025)**: Trains lightweight classifiers on OpenVLA's final-layer features to predict scalar failure likelihood. Key finding: VLA features are linearly separable for success vs. failure across diverse tasks. Uses functional conformal prediction for calibrated thresholds. Generalizes to unseen tasks. Our Experiment 7 extends this from binary success/failure to multi-class intent monitoring.
- **FIPER (NeurIPS 2025)**: Monitors two signals from generative policy internals: OOD observations via random network distillation in the policy's embedding space, and action uncertainty via a novel action-chunk entropy score. Complementary to SAFE — FIPER detects when the model is uncertain, SAFE detects when it's likely to fail.
- Selective conformal prediction: abstain when prediction set is too large. Directly applicable to our safety monitor design.

**Applicability here**: The intent probe output (softmax score for red vs. white) serves as the non-conformity score. Conformal calibration guarantees coverage without assuming that the probe's probabilities are well-calibrated (they're just a black-box score function). The prediction set size at each step is the interpretable safety signal.

---

## 6.5 Safety Monitoring and Failure Prediction for VLAs

**Core finding**: Recent work has demonstrated that VLA internal features are linearly separable for success vs. failure, and that conformal prediction provides calibrated safety guarantees, but all existing monitors are coarse-grained (binary success/failure) and none probe for fine-grained intent.

**Key papers**:

- **SAFE (Toyota Research Institute, NeurIPS 2025)**: Trains lightweight classifiers on OpenVLA's final-layer features to predict scalar failure likelihood. Uses functional conformal prediction. Key finding: VLA features are linearly separable for success vs. failure across diverse tasks. Generalizes to unseen tasks. Our Experiment 7 extends this from binary success/failure to multi-class intent monitoring.

- **FIPER (NeurIPS 2025)**: Monitors two signals: OOD observations via random network distillation, and action uncertainty via action-chunk entropy. The action-chunk entropy component is directly relevant — it operates at the VQ-code level, which is exactly our Experiment 7's uncertainty signal source.

- **Sentinel (CoRL 2024)**: Combines statistical temporal action consistency (STAC) with VLM-based video QA for failure mode detection. The STAC component — measuring whether the model's actions are temporally consistent — could inform our temporal intent analysis (Experiment 1): if action-space intent probe accuracy drops mid-trajectory, STAC-style consistency checking would also degrade.

- **"Averaging Trap" / "Shifting Uncertainty to Critical Moments" (2026)**: Identifies a critical challenge for VLA uncertainty quantification — naive token-level entropy poorly discriminates success from failure because successful trajectories contain high-entropy segments while failures can have low average entropy. Max-based sliding window pooling significantly improves discrimination. **Directly relevant to Experiment 7**: do not average uncertainty across all steps — focus on the pre-grasp phase using max-pooling.

**Gap this project fills**: All existing monitors classify binary success/failure or scalar risk. None probe for fine-grained intent (which object, what grasp type, what approach vector). Our contribution extends the monitoring target from "will this succeed?" to "what does the model intend to do?" — enabling intervention before failure rather than detection during failure.

---

## 7. Imitation Learning and VLA Context

**Core finding**: VLA policies trained on demonstration data learn task-specific representations, but these representations are entangled: the same model learns perception, planning, and motor skills simultaneously without explicit decomposition.

**Key ideas**:
- **OpenVLA** (Kim et al., 2024): 7B parameter VLA using LLaMA-2 backbone. Showed that larger LLMs improve generalization across tasks. Key result: action prediction accuracy depends critically on the tokenizer design (bins, normalization).
- **MiniVLA** (Stanford ILIAD, 2024): 1B parameter VLA using Qwen2.5-0.5B backbone with VQ action chunking. Key design choice: VQ-VAE discretizes action chunks rather than individual action values, reducing the number of tokens per step while using a 128-class codebook across 7 groups.
- **π0.5** (Black et al., 2024): Diffusion policy based VLA. Not directly comparable (not autoregressive), but Dr. VLA results apply to its transformer backbone.
- **BehaviorTransformer**: showed that VQ action chunking significantly reduces action sequence multimodality compared to per-value discretization.

**Key design note for this project**: The VQ-VAE discretizes *action chunks* (8-step windows, 7 groups of codes each from 128 classes). This means the semantic granularity of the action token space is coarser than per-value tokenization. Probes on action tokens have less fine-grained signal. Continuous decoded actions are richer for Exp 1.

---

## 8. Failure Analysis and Behavioral Evaluation

**Core finding**: Standard success rate evaluation conflates multiple failure modes (perceptual misidentification, motor planning failure, execution failure). Separating these requires additional diagnostic instrumentation.

**Key ideas**:
- **Behavioral cloning failure modes**: In IL, failures can arise from compounding errors, distributional shift, or incorrect goal understanding. The correct response to each is different (more data, data augmentation, or instruction tuning).
- **Confusion matrix analysis for robot tasks**: Similar to ML confusion matrices, but over behavioral outcomes. The 2×2 (probe correct × success) table used in this project is a specific instance of this general diagnostic approach.
- Scene randomization as a generalization test: varying spatial positions of objects controls for the model having learned a position-specific policy rather than a semantically-driven one. LIBERO initial states provide some positional variation.
- **Contact-based ground truth**: The `contacted_object` field from MuJoCo enables cell assignment in the failure decomposition table independent of the probe prediction, avoiding circularity.

**Applicability here**: The failure decomposition table (Exp 2 cross-cutting analysis) is the primary diagnostic contribution. Publishing this decomposition with causal tracing results (Exp 5) constitutes a complete argument for causal, intent-level safety monitoring.

---

## Summary Table

| Theme | Key Papers | Applicability |
|-------|-----------|---------------|
| Linear probing | Alain & Bengio 2016, Burns 2023, Lu 2025, Molinari 2025 | Exp 2, direct |
| VLA-specific interp | Häon CoRL 2025, Dr. VLA 2026, Grant ICLR 2026 | Background + gap framing |
| Causal tracing | Meng 2022 (ROME), Hernandez 2023 (RAVEL), NOTICE 2024 | Exp 5, direct |
| Token positions | Ghandeharioun 2024 (Patchscopes) | Exp 2 position selection |
| Sparse autoencoders | Cunningham 2023, Dr. VLA 2026 | Exp 6 |
| CKA | Kornblith 2019, Grant ICLR 2026 | Exp 4 |
| Conformal prediction | Angelopoulos 2021, SAFE 2025, FIPER 2025 | Exp 7 |
| Safety monitoring | SAFE 2025, FIPER 2025, Sentinel 2024, Averaging Trap 2026 | Exp 7, direct |
| VLA architecture | OpenVLA 2024, MiniVLA 2024 | Background |
| Failure analysis | Behavioral cloning lit | Cross-cutting failure decomp |
