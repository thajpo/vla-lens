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
            meta.parquet        # Episode metadata + tensor file paths
            ep{N}_l{L}.pt       # Activation tensor: (n_steps, hidden_dim)
    logs/
        libero/                 # Existing: per-task rollout logs
        libero_runs.jsonl       # Existing: per-episode summary records
    probes/                     # Saved probe models (.pkl) and eval results (.parquet)
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
Start with these. Extend to attention output vs. MLP output only if Exp 5 results are ambiguous.

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
- Save probe weights (`.coef_`) alongside eval results — needed for steering vector (Exp 5 and future steering experiments).
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
- The `patch_position` parameter selects token positions to patch. For the color-word position, this is the index of the color adjective in the tokenized instruction — compute with `tokenizer(instruction, return_offsets_mapping=True)` to find it.
- Recovery score: `1 - ||patched - clean|| / ||corrupt - clean||`, clipped to [0, 1].
- Run for all patch_layer × patch_position combinations. This is the inner loop — keep it fast by not reloading the model.

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
    --run-id my_probe_run \
    --output-dir artifacts/activations
```

**Output schema**:

Per run, write one parquet file `artifacts/activations/{run_id}/meta.parquet` with columns:
```
run_id, episode_id, task_id, task_language, episode_idx, seed,
step, success, layer, token_position, tensor_path
```

Each `tensor_path` points to a `.pt` file containing a tensor of shape `(hidden_dim,)` = the activation at that (episode, step, layer, token_position).

**Implementation notes**:
- Batch tensor saves: write one `.pt` per (episode, layer, token_position) containing a matrix `(n_steps, hidden_dim)`. The parquet index stores the path and step range.
- Token position resolution: add a `resolve_token_position(tokenizer, instruction, position_spec)` helper that returns an integer index. Spec "color_word" = index of "red" or "white" in the tokenized instruction. Spec "final" = -1. Spec "eos" = index of EOS token.
- If storage is a concern: fp16 tensors (~900KB per episode per layer for 400 steps × 896 dim × 2 bytes). 400 episodes × 6 layers = ~2.1GB. Acceptable.

---

### `scripts/patch_and_rollout.py`

**Purpose**: Run the activation patching protocol (Exp 5). Loads a pair of tasks (clean + corrupt instruction), runs the three forward passes, records recovery scores.

**CLI**:
```
python scripts/patch_and_rollout.py \
    --clean-task-id 71 \
    --corrupt-task-id 72 \
    --num-pairs 50 \
    --patch-layers 14 16 18 20 \
    --patch-positions color_word final \
    --run-id patch_exp5
```

**Output**: `artifacts/logs/patch_{run_id}.parquet` with columns:
```
episode_id, step, patch_layer, patch_position,
clean_vq_codes, corrupt_vq_codes, patched_vq_codes,
clean_action, corrupt_action, patched_action,
recovery_score
```

---

## Rollout Harness: Three Modes Summary

Extend `run_libero_task.py` with a `--mode` flag:

| Mode | Behavior | When to use |
|------|----------|-------------|
| `baseline` | No hooks. Log success/failure and basic metadata. | Initial characterization runs. |
| `capture` | Forward hooks on named layers. Save activations to disk. | Probe training data collection. |
| `patch` | Use `patching.py` protocol. | Exp 5 causal tracing. |

The `capture` mode should be in `collect_activations.py` (separate script) to keep `run_libero_task.py` clean. But all three share the same `load_model`, `get_libero_env`, `get_action` infrastructure — factor these into a shared module `src/openvla_steering/utils/rollout.py` only when `collect_activations.py` actually needs to import them. Not before.

---

## Analysis Scripts (Offline, Not Real-Time)

These are small scripts that read parquet files and produce figures. Keep them as standalone scripts in `scripts/analysis/`, not as library code.

```
scripts/analysis/
    plot_probe_sweep.py         # Heatmap of probe accuracy over (layer × token_position)
    failure_decomp.py           # 2x2 table: probe_correct × task_success
    vq_code_histogram.py        # Per-group code distribution, JSD between tasks
    plot_recovery_heatmap.py    # Patching recovery score heatmap (layer × position)
```

Each analysis script has a simple interface: `--run-id`, `--output-dir`, and prints a table or saves a figure.

---

## Notebook Usage

Do not use notebooks for data collection or training — only for exploratory analysis after a run is complete. If a notebook produces a result you want to keep, convert it to a script.

---

## Storage Budget

| Artifact | Size estimate | Retention |
|----------|--------------|-----------|
| Activation tensors (fp16) | ~2GB per run (400 eps × 6 layers) | Keep 2 most recent runs |
| Rollout videos | ~50MB per episode | Keep only failure episodes |
| Probe models (.pkl) | < 1MB each | Keep all |
| Parquet logs | < 100MB total | Keep all |
| VQ code sequences | < 10MB | Keep all (derived from tensors) |

Total expected peak: ~5GB. Clean old activation runs when probe training is done.

---

## First Three Implementation Tasks

1. **`hooks.py`**: `HookManager` with `__enter__`/`__exit__`, dry-run, and named module resolution. Test it on a single MiniVLA forward pass — print shapes. No data collection yet.

2. **`collect_activations.py`**: Collect 20 episodes × 2 tasks using layers {14, 16, 18} and token position "final". Verify parquet output is correct. Scale to 200 × 2 only after verification.

3. **`probes.py`** + **`scripts/analysis/plot_probe_sweep.py`**: Fit probes on the 20-episode pilot, generate the heatmap. If peak accuracy is > 70%, proceed to full 200-episode collection. If < 60%, something is wrong with the data pipeline.

Everything after task 3 follows from probe results.
