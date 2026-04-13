# Code Organization for Mechanistic Interpretability Experiments

This document defines the infrastructure needed to run the experiments in `docs/experiments/color-intent-probe.md` and specifies how to organize it within the existing project layout.

---

## Design Principles

1. **No premature abstraction.** Build what the experiments actually require. A single script that captures activations and writes a parquet file is better than a framework that might someday support arbitrary models.
2. **Reuse rollout data.** All activation data should be captured in a single rollout pass and saved. Experiments 1, 3, 4, 6 all run from the same cached data. Do not re-roll episodes per experiment.
3. **The three modes of the rollout harness.** Every rollout is one of: (a) baseline (no hooks, just success logging), (b) activation capture (forward hooks, saves tensors), (c) intervention (replaces or adds to activations). These are exclusive; add a `--mode` flag.
4. **Fail loudly.** Missing module names, shape mismatches, and missing calibration files are errors, not warnings.

---

## Repository Layout (Target State)

```
scripts/
    run_libero_task.py          # Existing: baseline rollouts. Add --mode flag.
    collect_activations.py      # New: activation capture mode
    patch_and_rollout.py        # New: activation patching (Exp 5)
    analysis/
        plot_probe_sweep.py         # Exp 2: Heatmap of probe accuracy (layer × position)
        failure_decomp.py           # Exp 2: 2×2 table with behavioral ground truth
        vq_code_histogram.py        # Exp 3: Per-group code distributions, JSD
        vq_semantic_consistency.py  # Exp 3: Within/between cluster variance ratios
        plot_recovery_heatmap.py    # Exp 5: Patching recovery score heatmap
        intent_lag_plot.py          # Exp 1 + 2: Temporal accuracy curves (hidden-state vs. action-space)
        conformal_calibration.py    # Exp 7: Conformal threshold computation and coverage analysis
        reach_direction_analysis.py # Cross-cutting: Cosine similarity of ee_velocity vs. target direction

src/openvla_steering/
    interp/
        __init__.py             # Existing (empty)
        hooks.py                # New: forward hook registration and activation cache
        probes.py               # New: linear probe fit/eval, logistic regression wrapper
        patching.py             # New: activation patching protocol
        sae.py                  # New (later): sparse autoencoder training and eval
        cka.py                  # New (later): CKA computation
    utils/
        __init__.py             # Existing
        io.py                   # Existing: parquet write helpers
        seeding.py              # Existing: seed helper

artifacts/
    activations/                # Saved activation tensors (per rollout, per layer)
        {run_id}/
            episodes.parquet    # Episode-level summary
            steps.parquet       # Step-level metadata + VQ codes
            activations.parquet # Activation tensor pointers
            tensors/
                ep{N}_l{L}_{pos}.pt  # Actual tensors: (n_steps, hidden_dim)
    logs/
        libero/                 # Existing: per-task rollout logs
        libero_runs.jsonl       # Existing: per-episode summary records
        patch_{run_id}.parquet  # Patching experiment output
    probes/                     # Saved probe models (.pkl) and eval results (.parquet)
    figures/                    # Analysis output
        {experiment}/
    videos/                     # Existing: rollout videos

docs/
    experiments/
        color-intent-probe.md   # Experiment definitions (this project)
    lit-review.md               # Literature synthesis
    code-organization.md        # This file
```

---

## Module Specifications

### `src/openvla_steering/interp/hooks.py`

**Purpose**: Register and manage PyTorch forward hooks on named modules. Cache activations during a forward pass. Support dry-run mode (shapes only).

**Public API**:

```python
class HookManager:
    def __init__(self, model: nn.Module, module_names: list[str]):
        ...
    def __enter__(self) -> "HookManager": ...   # registers hooks
    def __exit__(self, *_): ...                 # deregisters hooks, clears cache
    def get(self, module_name: str) -> torch.Tensor: ...  # (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
    def dry_run(self, sample_input: dict) -> dict[str, tuple]: ...  # returns shapes
```

**Implementation notes**:
- Use `model.named_modules()` to resolve names. Raise `KeyError` if a requested name is not found — do not silently skip.
- Hook captures `output` of the module (the residual stream after the layer, not the attention output separately).
- For MiniVLA: LLM layers are at `llm_backbone.llm.model.layers.{i}`. Vision layers are at `vision_backbone.featurizer.{...}`.
- Store captured tensors in a `dict[str, list[torch.Tensor]]` (one per forward pass if batching) and concatenate on exit.
- Do not hold references to tensors between episodes — clear cache between episodes.

**Module names to support for Experiments 2 & 5**:
```python
LLM_LAYERS = [f"llm_backbone.llm.model.layers.{i}" for i in range(24)]
```

### VQ-VAE module names (required for Experiments 1, 3, 7)

In addition to LLM layer hooks, the `HookManager` must support hooks on the VQ-VAE action tokenizer. The exact module paths depend on MiniVLA's implementation — inspect with `model.named_modules()` and search for the quantization layer. Expected structure:

```python
# These are approximate — verify against actual model architecture
VQ_MODULES = {
    "pre_quantize": "policy.action_tokenizer.encoder_mlp",
    "codebook":     "policy.action_tokenizer.vq_layer",    # or .fsq, .quantize
    "post_quantize": "policy.action_tokenizer.decoder_mlp",
}
```

**What to capture at each**:

| Hook target | Tensor to save | Shape | Used by |
|-------------|---------------|-------|---------|
| `pre_quantize` output | Continuous action embedding before hard assignment | `(n_groups, embed_dim)` | Exp 3 (semantic consistency), Exp 7 (aleatoric uncertainty) |
| `codebook` | Selected indices AND soft distances to all codes | indices: `(7,)` int; distances: `(7, 128)` float | Exp 1 (action-space probe), Exp 3 (JSD analysis), Exp 7 (codebook entropy) |
| `post_quantize` output | Decoded continuous action chunk | `(action_dim,)` i.e. `(7,)` | Exp 1, Exp 5 (recovery metric) |

**Codebook distances**: Standard VQ-VAE implementations compute distances internally and only output the hard-assigned index. Either (a) subclass the quantizer to expose distances, or (b) register a hook that recomputes distances from the pre-quantize vector and the codebook embedding matrix. Option (b) is cleaner — it doesn't modify the model.

```python
def compute_vq_distances(pre_quant_vector, codebook_weights):
    """Compute L2 distances from pre-quantization vector to all codebook entries.
    
    Args:
        pre_quant_vector: (n_groups, embed_dim)
        codebook_weights: (n_groups, n_codes, embed_dim) or (n_codes, embed_dim)
    Returns:
        distances: (n_groups, n_codes) — lower = closer match
    """
    return torch.cdist(pre_quant_vector.unsqueeze(1), codebook_weights).squeeze(1)
```

### Token position resolution: edge cases

The `resolve_token_position(tokenizer, instruction, position_spec)` function must handle:

1. **Multiple occurrences**: If the instruction contains "red" more than once, use the **first** occurrence. Log a warning if duplicates are found.

2. **Subword tokenization**: Qwen2.5's tokenizer may split color words into subword tokens. If so, use the **last** subword token of the color word — in transformer architectures, the final subword position accumulates the full word representation via causal attention.

3. **"final" position**: Return index of the last non-padding token before action token generation.

4. **Validation**: After resolving, assert that the token at the resolved position decodes back to the expected string (or a subword of it). Raise an error with the full tokenized sequence if not.

```python
def resolve_token_position(tokenizer, instruction: str, spec: str) -> int:
    """
    Resolve a position spec to a token index.
    
    Args:
        spec: One of "color_word", "final", "eos"
    Returns:
        Integer index into the tokenized sequence.
    Raises:
        ValueError if the color word is not found or ambiguous.
    """
    tokens = tokenizer.encode(instruction)
    
    if spec == "final":
        return len(tokens) - 1
    
    if spec == "eos":
        eos_id = tokenizer.eos_token_id
        if eos_id in tokens:
            return tokens.index(eos_id)
        return len(tokens) - 1
    
    if spec == "color_word":
        for color in ["red", "white"]:
            if color in instruction.lower():
                char_start = instruction.lower().index(color)
                char_end = char_start + len(color)
                encoding = tokenizer(instruction, return_offsets_mapping=True)
                offsets = encoding["offset_mapping"]
                color_token_indices = [
                    i for i, (s, e) in enumerate(offsets)
                    if s < char_end and e > char_start and s != e
                ]
                if not color_token_indices:
                    raise ValueError(f"Color word '{color}' not found in tokens")
                return color_token_indices[-1]  # last subword token
        
        raise ValueError(f"No color word found in instruction: {instruction}")
```

---

### `src/openvla_steering/interp/probes.py`

**Purpose**: Fit and evaluate linear probes (logistic regression) on cached activations.

**Public API**:

```python
def fit_probe(
    X: np.ndarray,        # (n_samples, hidden_dim)
    y: np.ndarray,        # (n_samples,) int labels
    C: float = 1.0,       # L2 regularization
    cv: int = 5,          # stratified k-fold
) -> dict:
    """Returns: {'model': LogisticRegression, 'cv_accuracy': float, 'cv_auc': float}"""

def eval_probe(
    probe,                # fitted LogisticRegression
    X: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Returns: {'accuracy': float, 'auc': float, 'confusion_matrix': np.ndarray}"""

def probe_sweep(
    activations: dict[str, np.ndarray],  # module_name -> (n_samples, hidden_dim)
    labels: np.ndarray,
) -> pd.DataFrame:
    """Fit one probe per key, return DataFrame with columns [module, accuracy, auc]."""
```

**Implementation notes**:
- Use `sklearn.linear_model.LogisticRegression(C=C, max_iter=1000, solver='lbfgs')`.
- Use `sklearn.model_selection.StratifiedKFold` for CV.
- Save probe weights (`.coef_`) alongside eval results — needed for steering vector derivation.
- The probe output from `probe_sweep` is the primary artifact for Exp 2. Write it to `artifacts/probes/{run_id}_sweep.parquet`.

---

### `src/openvla_steering/interp/patching.py`

**Purpose**: Implement the activation patching protocol for Exp 5.

**Public API**:

```python
def run_clean_and_corrupt(
    model,
    clean_obs: dict,
    clean_instruction: str,
    corrupt_instruction: str,
    hook_manager: HookManager,
    layer_names: list[str],
) -> tuple[dict, dict]:
    """Run clean and corrupt forward passes. Return (clean_cache, corrupt_cache)."""

def run_patched(
    model,
    corrupt_obs: dict,
    corrupt_instruction: str,
    clean_cache: dict,
    patch_layer: str,
    patch_position: int | slice,
) -> np.ndarray:
    """Run forward pass with corrupt input, patching clean activations at patch_layer/position.
    Returns decoded action (7,) numpy array."""

def recovery_score(
    clean_action: np.ndarray,
    corrupt_action: np.ndarray,
    patched_action: np.ndarray,
) -> float:
    """Euclidean recovery score in [0, 1]. 1.0 = full recovery to clean."""
```

**Implementation notes**:
- Patching is implemented by registering a forward *pre-hook* that overwrites the module's output with the cached clean activation before the residual add.
- The `patch_position` parameter selects token positions to patch — use `resolve_token_position` from `hooks.py`.
- Recovery score: `1 - ||patched - clean|| / ||corrupt - clean||`, clipped to [0, 1].
- Run for all patch_layer × patch_position combinations in a single model-load — do not reload the model per patch site.

---

### `scripts/collect_activations.py`

**Purpose**: Run LIBERO rollouts in activation-capture mode. Reuses inference logic from `run_libero_task.py` but wraps each forward pass with a `HookManager`.

**CLI**:
```
python scripts/collect_activations.py \
    --task-suite-name libero_90 \
    --task-ids 71 72 \
    --num-trials-per-task 200 \
    --layers 12 14 16 18 20 22 \
    --token-positions color_word final \
    --save-observations \
    --run-id my_probe_run \
    --output-dir artifacts/activations
```

The `--save-observations` flag saves raw image observations as compressed numpy arrays alongside activations. Required for Exp 5 offline patching. Storage cost: ~200KB per step × 20 steps × 50 episodes ≈ 200MB per task pair.

**Output schema** — three separate parquet files per run:

`artifacts/activations/{run_id}/episodes.parquet`:
```
run_id, episode_id, task_id, task_language, seed, success, n_steps
```

`artifacts/activations/{run_id}/steps.parquet`:
```
run_id, episode_id, step,
ee_pos_x, ee_pos_y, ee_pos_z,
target_mug_pos_x, target_mug_pos_y, target_mug_pos_z,
other_mug_pos_x, other_mug_pos_y, other_mug_pos_z,
contacted_object,
gripper_state,
vq_codes,          # list[int], length 7
decoded_action,    # list[float], length 7
```

`artifacts/activations/{run_id}/activations.parquet`:
```
run_id, episode_id, step, layer, token_position, tensor_path
```

Each `tensor_path` points to `tensors/ep{N}_l{L}_{pos}.pt` containing a matrix of shape `(n_steps, hidden_dim)` in fp16. Storage cost: ~900KB per (episode × layer × position) for 400 steps × 896 dim × 2 bytes. At 400 episodes × 6 layers × 2 positions: ~4.3GB.

**Keeping metadata separate from activations** allows all of Exp 1, failure decomposition, and Exp 3 to run without loading activation tensors — they only need `steps.parquet`.

---

### `scripts/patch_and_rollout.py`

**Purpose**: Run the activation patching protocol (Exp 5) on **saved observations from clean rollouts**. This script does NOT run a live environment. It loads observations from `collect_activations.py` output and performs clean/corrupt/patched forward passes offline.

**Prerequisites**: A completed activation-capture run with `--save-observations`. Reads from `artifacts/activations/{source_run_id}/`.

**CLI**:
```
python scripts/patch_and_rollout.py \
    --source-run-id my_probe_run \
    --clean-task-id 71 \
    --corrupt-task-id 72 \
    --num-episodes 50 \
    --steps-per-episode 20 \
    --patch-layers 14 16 18 20 \
    --patch-positions color_word final \
    --run-id patch_exp5
```

**Protocol per (episode, step)**:
1. Load `obs_t` from the saved clean rollout.
2. Clean forward pass: `model(obs_t, "pick up the red mug on the plate")` → save activations and action.
3. Corrupt forward pass: `model(obs_t, "pick up the white mug on the plate")` → save activations and action.
4. For each (patch_layer, patch_position):
   a. Corrupt forward pass with a pre-hook that replaces `h[patch_layer][patch_position]` with the clean activation.
   b. Record the patched action.
5. Compute recovery scores.

**Output** — `artifacts/logs/patch_{run_id}.parquet`:
```
episode_id, step, patch_layer, patch_position,
clean_vq_codes, corrupt_vq_codes, patched_vq_codes,
clean_action, corrupt_action, patched_action,
recovery_score,
vq_code_match_clean,    # bool: patched codes exactly match clean codes
vq_code_match_corrupt   # bool: patched codes exactly match corrupt codes
```

The `vq_code_match_clean` and `vq_code_match_corrupt` boolean columns give a stricter measure than the continuous recovery score — essential for Exp 5's claim that patching causes behavioral *flipping*.

---

## Rollout Harness: Three Modes Summary

Extend `run_libero_task.py` with a `--mode` flag:

| Mode | Behavior | When to use |
|------|----------|-------------|
| `baseline` | No hooks. Log success/failure and basic metadata. | Initial characterization runs. |
| `capture` | Forward hooks on named layers. Save activations to disk. | Probe training data collection. |
| `patch` | Use `patching.py` protocol. | Exp 5 causal tracing. |

The `capture` mode lives in `collect_activations.py` (separate script) to keep `run_libero_task.py` clean. All three share the same `load_model`, `get_libero_env`, `get_action` infrastructure. Factor these into `src/openvla_steering/utils/rollout.py` only when `collect_activations.py` actually needs to import them. Not before.

---

## Analysis Scripts (Offline, Not Real-Time)

Scripts read from `artifacts/` and write to `artifacts/figures/{experiment}/`. CLI convention: `--run-id`, `--output-dir`, optional `--format` (png/pdf/svg).

```
scripts/analysis/
    plot_probe_sweep.py           # Exp 2: Heatmap of probe accuracy (layer × position)
    failure_decomp.py             # Exp 2: 2×2 table with behavioral ground truth
    vq_code_histogram.py          # Exp 3: Per-group code distributions, JSD
    vq_semantic_consistency.py    # Exp 3: Within/between cluster variance ratios
    plot_recovery_heatmap.py      # Exp 5: Patching recovery score heatmap
    intent_lag_plot.py            # Exp 1 + 2: Temporal accuracy curves
    conformal_calibration.py      # Exp 7: Conformal threshold and coverage analysis
    reach_direction_analysis.py   # Cross-cutting: ee_velocity cosine analysis
```

Do not use notebooks for data collection or training — only for exploratory analysis. If a notebook produces a result you want to keep, convert it to a script.

---

## Storage Budget

| Artifact | Size estimate | Retention |
|----------|--------------|-----------|
| Activation tensors (fp16) | ~4.3GB per run (400 eps × 6 layers × 2 positions) | Keep 2 most recent runs |
| Saved observations (compressed) | ~200MB per task pair × 2 | Keep only while Exp 5 is running |
| Rollout videos | ~50MB per episode | Keep only failure episodes |
| Probe models (.pkl) | < 1MB each | Keep all |
| Parquet logs | < 100MB total | Keep all |
| VQ code sequences | < 10MB | Keep all (derived from steps.parquet) |

Total expected peak: ~10GB. Clean observation saves and old activation runs when probe training is done.

---

## First Three Implementation Tasks

1. **`hooks.py`**: `HookManager` with `__enter__`/`__exit__`, dry-run, and named module resolution. Test it on a single MiniVLA forward pass — print shapes for all 24 LLM layers. No data collection yet. Also verify VQ module names by inspecting `model.named_modules()`.

2. **`collect_activations.py`**: Collect 20 episodes × 2 tasks using layers {14, 16, 18} and token position "final". Verify parquet output is correct. Spot-check that `vq_codes` in steps.parquet match what the script printed during rollout.

3. **`probes.py`** + **`scripts/analysis/plot_probe_sweep.py`**: Fit probes on the 20-episode pilot, generate the heatmap. If peak accuracy is > 70%, proceed to full 200-episode collection. If < 60%, something is wrong with the data pipeline — investigate before scaling.

Everything after task 3 follows from probe results.
