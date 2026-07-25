# Nonlinear pose capacity study

RQ-017 extends the vector geometry runner with an opt-in small MLP while
leaving existing ridge-only specs unchanged.

The primary request is:

```bash
uv run python scripts/run_vla_lens_geometry_study.py \
  /mnt/new-volume/vla-lens/pi05-broad-1000-mech-light-lerobot-v3 \
  --spec configs/probes/pi05_broad_1000_nonlinear_pose_capacity_study.yaml
```

Without `--run`, this only prints the request. Add `--run` when the planned
broad study is ready. The separately named within-task spec is a secondary
generalization check, not a replacement for held-out-task evaluation.

## Selection and reporting contract

- Ridge remains the default when `probe.models` is omitted.
- `models: [ridge, mlp]` adds the fixed 64-unit `MLPRegressor` capacity check.
- Scaler and PCA are fitted on training rows only. One maximum PCA fit is reused
  at 64, 128, and 256 dimensions.
- Validation chooses the feature site, layer, PCA width, model, and ridge
  strength. Test error never participates in selection or promotion.
- The best MLP for a target must beat that row cohort's strongest physical or
  metadata baseline on validation before any MLP test metric or test prediction
  is emitted.
- Positive error deltas mean the probe is better than the baseline.

## Saved evidence

The geometry artifact saves all validation candidates, family-level
selections, row predictions with matching baseline predictions, convergence
metadata, and paired grouped confidence intervals. `fitted_readouts.json` and
`fitted_arrays.npz` contain the train-fitted scaler, PCA, ridge or MLP weights,
target blocks, and hyperparameters. `predict_geometry_readout` reconstructs a
selected readout without fitting it again.

The probe preflight recognizes this as a specialized multi-feature geometry
study before applying generic defaults. It reports the requested targets,
object cohort, representation families and their availability, split, controls,
and model/PCA sweep without fitting or materializing features.
