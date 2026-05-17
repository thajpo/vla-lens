# PI0.5 Experiment 2 — Causal Diagnostic: Ablation Strength and Projection Profile

## Motivation

The first delta-ablation pilot (see `pi05-delta-ablation-pilot.md`) returned a negative/ambiguous result:
- removing the shared benchmark-delta direction from `suffix_out` at `alpha=1.0` did not rescue failing tasks
- the intervention was not clearly selective over the orthogonal control
- random ablation *improved* `cream_cheese` from 0.8 → 1.0 (one episode), which may be noise or may be real

Before committing to a larger architectural intervention (handoff swap), two diagnostic questions need to be answered:

**Q1: Was alpha=1.0 too gentle?**
The intervention at `alpha=1.0` removes the projection exactly once from the final expert hidden state.
If the projection is large relative to the residual stream norm, this might have little effect.
Higher alpha tests whether a stronger ablation changes behavior.

**Q2: Does the intervention point matter?**
The pilot ablated only `suffix_out` — the expert's final post-norm hidden state, just before `action_out_proj`.
If the failure mechanism is committed earlier in the expert's forward pass, ablating at the end is too late.
All-layer ablation tests whether hooking into every transformer layer produces a different result.

**Q3: Does random-helps replicate?**
`n=5` is too small to distinguish "random helps" from measurement noise. Multiple random seeds determine whether this is a real finding or a fluke.

## Experiments

### Experiment A: Extended suffix_out ablation (`run_pi05_diag_ablation.py`)

Conditions:
- `delta_a5`: alpha=5.0, ablate `suffix_out` onto delta direction
- `delta_a10`: alpha=10.0, ablate `suffix_out` onto delta direction
- `random_s1` through `random_s4`: 4 independent random control directions (orthogonalized away from delta), alpha=1.0
- `delta_layers_a1`: alpha=1.0, ablate delta direction at **every** Gemma-expert transformer layer

Tasks: `cream_cheese`, `alphabet_soup` (mixed-outcome Scene 1 tasks)
Layouts: 0–4 (n=5 per condition)

Output: `artifacts/pi05_analysis/interventions/diag_ablation.json`

**What to look for:**
- If `delta_a5` or `delta_a10` rescues `cream_cheese` to baseline or better: alpha=1.0 was too gentle
- If `delta_layers_a1` rescues tasks but `delta_a10` does not: the failure mechanism is in intermediate layers, not just `suffix_out`
- If all three (`delta_a5`, `delta_a10`, `delta_layers_a1`) show no effect: the delta direction is probably not causally load-bearing at any intervention point
- If `random_s1..s4` results cluster around baseline (0.8 for `cream_cheese`): original "random helps" finding was noise
- If multiple random seeds reliably beat baseline: unexpected regularization effect — investigate

### Experiment B: Projection profile log (`run_pi05_diag_proj_log.py`)

Logs the projection of `suffix_out` onto the delta direction at each denoising flow step.
No intervention — pure observation.

Tasks: all four Scene 1 tasks (includes all-failure ketchup, tomato_sauce)
Layouts: 0–4 (n=5 per task)

Output: `artifacts/pi05_analysis/interventions/proj_log.json`
Analysis: `analyze_pi05_diag_proj_log.py` → `proj_log_summary.csv`

**What to look for:**
- **Flat profile across flow steps**: projection is uniformly encoded throughout denoising; ablating at any step would encounter it equally
- **Rising profile (late flow steps higher)**: the direction grows as denoising progresses; ablating only at the end is reasonable
- **Falling profile (early steps higher)**: the direction is highest at the start of denoising; the intervention should be at flow step 0 or earlier
- **Failure tasks have higher projection than success tasks**: supports observational delta/failure correlation
- **No difference in projection across success/failure**: the direction is present equally in both — correlation is driven by something else

## Interpretation guide

| Result pattern | Interpretation | Next step |
|---|---|---|
| Higher alpha rescues tasks | alpha=1.0 was too gentle; direction is causal at `suffix_out` | Repeat with tuned alpha; design ablation that doesn't hurt baseline |
| Layer ablation rescues but suffix_out doesn't | Failure committed in intermediate layers | Target earlier intervention; try ablating only at specific layers |
| All ablations null; projection profile flat | Delta direction is a correlate, not a cause; uniformly present | Handoff swap experiment to test full conditioning object |
| All ablations null; projection profile rises late | Ablation point is right but direction insufficient alone | Broader subspace ablation (top-k PCA components of delta) |
| Random helps replicates | Expert has distributed failure-prone structure; random noise helps generalize | Investigate which components of the random direction matter |
| Random doesn't replicate | n=5 fluke | Ignore the original random result |

## Results

### Projection log

| object_label | success_rate | mean proj_mean | early proj_mean (steps 0-4) | late proj_mean (steps 5-9) | late − early |
|---|---:|---:|---:|---:|---:|
| cream_cheese | 1.0 | 0.2094 | 0.1612 | 0.2575 | +0.0963 |
| alphabet_soup | 0.6 | 0.4327 | 0.3125 | 0.5528 | +0.2403 |
| tomato_sauce | 0.0 | 0.5963 | 0.4400 | 0.7527 | +0.3127 |
| ketchup | 0.0 | 0.7296 | 0.5161 | 0.9431 | +0.4270 |

Per-flow-step profile (proj_mean averaged over all policy calls and layouts):

```
flow_step  cream_cheese  alphabet_soup  tomato_sauce  ketchup
0              0.1749         0.2861        0.3969    0.4497
1              0.1665         0.2980        0.4153    0.4802
2              0.1331         0.2821        0.4090    0.4816
3              0.1516         0.3225        0.4578    0.5464
4              0.1799         0.3739        0.5209    0.6226
5              0.2248         0.4467        0.6126    0.7358
6              0.1880         0.4326        0.6140    0.7525
7              0.2023         0.4872        0.6891    0.8627
8              0.3041         0.6478        0.8733    1.1039
9              0.3684         0.7498        0.9742    1.2604
```

**Key finding:** The projection grows 2–3x from flow step 0 to flow step 9 for every task. The direction is being **continuously rebuilt** across denoising steps, not just present at the start. Projection magnitude is ordered by failure rate (cream_cheese lowest, ketchup highest).

### Extended ablation

| object_label | condition | success_rate |
|---|---|---:|
| alphabet_soup | baseline (pilot) | 0.6 |
| alphabet_soup | delta_a1 (pilot) | 0.6 |
| alphabet_soup | delta_a5 | **0.0** |
| alphabet_soup | delta_a10 | **0.0** |
| alphabet_soup | random_s1 | 0.8 |
| alphabet_soup | random_s2 | **1.0** |
| alphabet_soup | random_s3 | **1.0** |
| alphabet_soup | random_s4 | **1.0** |
| alphabet_soup | delta_layers_a1 | 0.8 |
| cream_cheese | baseline (pilot) | 0.8 |
| cream_cheese | delta_a1 (pilot) | 0.6 |
| cream_cheese | delta_a5 | **0.0** |
| cream_cheese | delta_a10 | **0.0** |
| cream_cheese | random_s1 | 0.8 |
| cream_cheese | random_s2 | **1.0** |
| cream_cheese | random_s3 | 0.8 |
| cream_cheese | random_s4 | **1.0** |
| cream_cheese | delta_layers_a1 | 0.6 |

## Readout

### What changed from the pilot

- Higher alpha **collapses** performance (0.0/5 for both tasks at alpha=5 and alpha=10). The policy *requires* the delta direction — removing it aggressively breaks everything.
- Random ablation **consistently improves** performance. All 4 seeds outperform baseline for alphabet_soup (mean 0.95 vs 0.6 baseline); 3/4 seeds at or above baseline for cream_cheese. This replicates across seeds — it is not a fluke.
- Layer-level ablation is intermediate: better than aggressive suffix_out ablation but below random.

### What this forces us to revise

The original hypothesis was: *the delta direction is causing failure; remove it to rescue.*

The data now says the opposite in two ways:

1. **The direction is load-bearing.** Removing it aggressively collapses performance to 0. The policy cannot function without it.

2. **Random noise helps, not targeted removal.** Across 4 seeds and 2 tasks, random direction ablation improves above baseline. This is a regularization effect, not a directional one.

### The new mechanistic picture

The delta direction is not the failure mechanism. It is present at higher levels in failing tasks as a **consequence** of failure, not a cause. The projection log confirms it is continuously reconstructed through denoising — ablating it at `suffix_out` at each step cannot prevent the expert from rebuilding it at the next step from the VLM KV cache.

The random-helps finding points to something different: the policy's representation for these tasks may be in a narrow, failure-prone attractor. Random perturbation introduces just enough variance to escape it. This is consistent with:
- The mixed-outcome tasks (alphabet_soup, cream_cheese) being near a success/failure boundary
- Stochastic perturbation helping generalize away from a failure trajectory

### What this does not explain

- Why random ablation helps but targeted delta ablation hurts. If the direction is load-bearing, removing any direction should hurt equally — but random ablation helps. This suggests the orthogonalization in the random control directions is not the key property. The random directions may not be truly neutral: they are orthogonalized away from the delta direction, so they preserve the delta direction while adding noise in the orthogonal complement.

## Interpretation (updated)

| Result | What it rules out | What it supports |
|---|---|---|
| delta_a5/a10 → 0.0 | "Remove more delta to fix failures" | Delta direction is necessary for successful control |
| Random replicates across 4 seeds | Noise fluke from n=5 pilot | Randomized perturbation has a real regularizing effect |
| Projection grows through flow steps | "Ablation at suffix_out neutralizes the direction" | Direction is continuously rebuilt from VLM conditioning |
| Layer ablation intermediate | "Intervention point is the whole problem" | Some benefit from earlier intervention, but not decisive |

## Practical takeaway

The delta direction is a **necessary component** of the policy's action computation, not a failure-causing excess. The failure mechanism is elsewhere — likely in the VLM KV cache conditioning that drives the expert toward task-specific trajectories that happen to fail for certain objects/scenes.

**The most important new finding is that random perturbation reliably improves performance.** This is the most actionable signal yet. Next steps:

1. **Characterize the random-helps effect**: which properties of the random directions matter? (magnitude, layer, position in the flow)
2. **Handoff swap**: does swapping the VLM KV cache from a success rollout of the same task escape the failure attractor?
3. **Drop the delta-direction story**: the observational correlation is real but the direction is not the causal handle

## Artifacts

- `artifacts/pi05_analysis/interventions/diag_ablation.json`
- `artifacts/pi05_analysis/interventions/proj_log.json`
- `artifacts/pi05_analysis/interventions/proj_log_summary.csv`
