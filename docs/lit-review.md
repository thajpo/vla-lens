# Literature Review: Mechanistic Interpretability for Robot Policies

Thematic synthesis of the research landscape relevant to this project.

---

## 1. Linear Probing and Representation Geometry

**Core finding**: Hidden states in large neural networks are geometrically structured. Linear probes — logistic regression fits on frozen hidden states — reliably recover high-level semantic variables well before the output layer.

**Key ideas**:
- Alain & Bengio (2016) showed that intermediate layers in deep networks are increasingly accurate predictors of downstream labels. The slope of probe accuracy across layers is itself diagnostic — steep slope = information is assembled across layers, plateau = stable representation.
- Burns et al. (2023) introduced *contrast-consistent search* (CCS): find probe directions that are maximally self-consistent (if a probe gives 90% for P, it should give 10% for ¬P). This addresses the concern that probes recover the model's "beliefs" rather than arbitrary statistics.
- Belrose et al. (2023) showed that the logit lens (projecting intermediate residuals into vocab space) can track when a model "decides" its output, layer by layer. Equivalent protocol applies to robot policies: project intermediate states into action token space and check when action probabilities peak.

**Applicability here**: Color probe training follows standard Alain & Bengio protocol. CCS is a sanity check: probe for "red mug is target" and "white mug is target" should be anti-correlated by construction. If not, the probe is recovering something else (e.g., scene statistics).

**Key concern**: Probes can be accurate without being causal. A probe on a redundant, read-only copy of task information would succeed but intervention on it would have no effect. Must pair with Exp 5 (causal tracing) to establish causal relevance.

---

## 2. Causal Tracing and Factual Localization

**Core finding**: The causal contribution of internal computations to model outputs can be measured by the "activation patching" protocol: corrupt input, restore clean activations at a specific site, measure output recovery.

**Key ideas**:
- Meng et al. (2022) ROME: applied causal tracing to GPT-J and found that factual associations are primarily stored in specific MLP layers. This is the canonical reference for the activation patching protocol.
- Hernandez et al. (2023) RAVEL: extended causal tracing to disentangled feature identification. Key result: the site that controls "which city" differs from the site that controls "which country," even for related facts. Implies that intent features may be localized differently from scene features.
- Geiger et al. (2023) DAS (Distributed Alignment Search): more general form — search over *linear* subspaces, not just activation vectors, for causal relevance. Strictly more powerful than activation patching.

**Applicability here**: The activation patching protocol in Exp 5 closely follows ROME. The main adaptation is that "output" is now VQ codes (action tokens) rather than next-word logits. Recovery metric is action-space rather than logit-space.

**Key concern**: Corruption by wrong instruction is a strong perturbation — it changes both the task-relevant signal and potentially the syntactic/positional structure of the prompt. Use targeted corruptions where possible (swap only the color adjective, not the full instruction).

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
- Cunningham et al. (2023): trained SAEs on GPT-2 MLP layers and found features selective for specific semantic content (e.g., "names of scientists", "Python import statements"). Key design choice: use TopK sparsity rather than L1 to control the number of active features per forward pass.
- Templeton et al. / Anthropic (2024) "Scaling Monosemanticity": scaled SAEs to Claude 3 Sonnet. Found millions of interpretable features, including multimodal features and abstract concepts. Key finding: features form a "feature geometry" — similar concepts have nearby directions in the SAE latent space.
- **Dr. VLA** (2024): Applied SAEs to π0.5 (action diffusion transformer) and OpenVLA. Key findings: (a) most high-frequency features correspond to memorized demonstration-level patterns rather than semantic task concepts, (b) a small number of features do show selectivity for object identities, but coverage is sparse. This is the most directly relevant prior work.

**Applicability here**: Given Dr. VLA's findings, expectation is that SAEs will find some color-selective features but with low coverage. The more productive comparison is: do SAE-selective features align with the linear probe direction from Exp 2? If yes, the probe finds a compressed version of what the SAE decomposes.

**Key concern**: SAE training requires substantial computation. The 896-dim hidden state with 4096 latents needs ~400 episodes × 400 steps × 24 layers = ~3.8M samples for one layer. Select one or two peak layers rather than all 24.

---

## 5. Representational Similarity (CKA)

**Core finding**: CKA (Centered Kernel Alignment) is a robust metric for comparing neural representations across architectures or conditions, invariant to invertible linear transformations and orthogonal rotations.

**Key ideas**:
- Kornblith et al. (2019): introduced CKA for comparing layers within and across neural networks. Linear CKA (using dot-product kernel: K = X X^T) is computationally efficient and empirically reliable.
- Nguyen et al. (2021): applied CKA to compare wide and deep network representations, finding that deeper layers are more similar across architectures than shallow layers (convergent evolution of representation).
- For VLMs: visual and language representations are most aligned in the upper layers, where cross-attention (or implicit cross-modal integration via attention over mixed token sequences) has operated for more layers.

**Applicability here**: CKA between vision patch tokens and instruction tokens at each LLM layer gives a direct readout of when cross-modal integration has occurred. If visual and linguistic representations are most aligned at the layer where Exp 2 finds the highest probe accuracy, this is convergent evidence.

**Key concern**: CKA is descriptive, not causal. High CKA means the representations are similar in structure, not that one caused the other. Pair with causal tracing (Exp 5) for causal claims.

---

## 6. Conformal Prediction for Robot Safety

**Core finding**: Split conformal prediction provides finite-sample, distribution-free coverage guarantees for prediction sets. A well-calibrated conformal predictor covering at level 1-α will contain the true label in at least (1-α) fraction of test episodes, regardless of the underlying distribution.

**Key ideas**:
- Angelopoulos & Bates (2021) "A Gentle Introduction": clearest reference for the split conformal protocol. Non-conformity score can be any function of input and label (e.g., 1 - softmax_correct); the threshold is the (1-α) quantile of calibration scores.
- **SAFE** (Luo et al., 2024, if available): applies conformal prediction to robot safety monitoring — generates conformal sets over predicted failure modes to guarantee that the true failure mode is always included.
- **FIPER**: Fixed prediction intervals for robot performance evaluation — related framework for calibrated interval estimation of robot success rates rather than failure classification.
- Selective conformal prediction: abstain when prediction set is too large (i.e., when uncertainty is too high to commit). Directly applicable to our safety monitor design.

**Applicability here**: The intent probe output (softmax score for red vs. white) serves as the non-conformity score. Conformal calibration guarantees coverage without assuming that the probe's probabilities are well-calibrated (they're just a black-box score function). The prediction set size at each step is the interpretable safety signal.

---

## 7. Imitation Learning and VLA Context

**Core finding**: VLA policies trained on demonstration data learn task-specific representations, but these representations are entangled: the same model learns perception, planning, and motor skills simultaneously without explicit decomposition.

**Key ideas**:
- **OpenVLA** (Kim et al., 2024): 7B parameter VLA using LLaMA-2 backbone. Showed that larger LLMs improve generalization across tasks. Key result: action prediction accuracy depends critically on the tokenizer design (bins, normalization).
- **MiniVLA** (Stanford ILIAD, 2024): 1B parameter VLA using Qwen2.5-0.5B backbone with VQ action chunking. Key design choice: VQ-VAE discretizes action chunks rather than individual action values, reducing the number of tokens per step from 7 to 7 (groups, but each group from a 128-class codebook rather than 256-class per-value).
- **π0.5** (Black et al., 2024): Diffusion policy based VLA. Not directly comparable (not autoregressive), but Dr. VLA results apply.
- **BehaviorTransformer**: showed that VQ action chunking significantly reduces action sequence multimodality compared to per-value discretization.

**Key design note for this project**: The VQ-VAE discretizes *action chunks* (8-step windows, 7 groups of codes each from 128 classes). This means the semantic granularity of the action token space is coarser than per-value tokenization. Probes on action tokens have less fine-grained signal. Continuous decoded actions are richer for Exp 1.

---

## 8. Failure Analysis and Behavioral Evaluation

**Core finding**: Standard success rate evaluation conflates multiple failure modes (perceptual misidentification, motor planning failure, execution failure). Separating these requires additional diagnostic instrumentation.

**Key ideas**:
- **Behavioral cloning failure modes**: In IL, failures can arise from compounding errors, distributional shift, or incorrect goal understanding. The correct response to each is different (more data, data augmentation, or instruction tuning).
- **Confusion matrix analysis for robot tasks**: Similar to ML confusion matrices, but over behavioral outcomes. The 2×2 (probe correct × success) table used in this project is a specific instance of this general diagnostic approach.
- Scene randomization as a generalization test: varying spatial positions of objects controls for the model having learned a position-specific policy rather than a semantically-driven one. LIBERO initial states provide some positional variation.

**Applicability here**: The failure decomposition table (Exp 2 cross-cutting analysis) is the primary diagnostic contribution. Publishing this decomposition with causal tracing results (Exp 5) constitutes a complete argument for causal, intent-level safety monitoring.

---

## Summary Table

| Theme | Key Papers | Applicability |
|-------|-----------|---------------|
| Linear probing | Alain & Bengio 2016, Burns 2023 | Exp 2, direct |
| Causal tracing | Meng 2022 (ROME), Hernandez 2023 (RAVEL) | Exp 5, direct |
| Token positions | Ghandeharioun 2024 (Patchscopes) | Exp 2 position selection |
| Sparse autoencoders | Cunningham 2023, Dr. VLA 2024 | Exp 6 |
| CKA | Kornblith 2019, Nguyen 2021 | Exp 4 |
| Conformal prediction | Angelopoulos 2021, SAFE | Exp 7 |
| VLA architecture | OpenVLA 2024, MiniVLA 2024 | Background |
| Failure analysis | Behavioral cloning lit | Cross-cutting failure decomp |
