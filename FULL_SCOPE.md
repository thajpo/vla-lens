# VLA Lens: Full Research Scope

## Purpose

Build a minimal but extensible research codebase for mechanistic interpretability of Vision-Language-Action (VLA) models, with a safety-oriented framing. The primary target is CogACT-Small, a componentized VLA with an explicit architectural boundary between its transformer backbone (VLM) and its diffusion action head (DiT).

The core scientific question is: does the transformer backbone produce a conditioning vector that already encodes complete task intent before the diffusion head runs, or does the diffusion process deliberate? The answer determines where to place a safety monitor and how much information a probe-based defense can recover.

This is not a training project. All experiments use pretrained checkpoints and probe or perturb frozen models.

## Research Position

### Primary model: CogACT-Small

CogACT-Small uses the Prismatic 7B VLM (DINOv2 + SigLIP visual encoders, Llama 2 language backbone) plus a DiT-S (~100M param) diffusion action head. The VLM and DiT are separate modules with a clean conditioning interface: the VLM produces a context vector that the DiT uses as conditioning alongside a noisy action chunk. The DiT then runs 10 DDIM denoising steps to produce a 16-step action chunk.

The architectural separation creates two hookable surfaces:
- The **conditioning vector**: the tensor the VLM hands off to the DiT. This is the "plan" if the hypothesis holds.
- The **DDIM intermediate states**: the partially denoised trajectory estimate at each of 10 denoising steps. This is where deliberation would appear if the hypothesis fails.

### Comparison model: OpenVLA

OpenVLA (7B, autoregressive, actions as discrete tokens) is retained as a comparison baseline. It is already running in the environment. The comparison is structurally interesting: OpenVLA has no separate action head — the "plan" and the "action" emerge from the same autoregressive process, so there is no clean interface to probe. Any intent signal in OpenVLA must be found inside the residual stream.

### Why the autoregressive case is not primary

OpenVLA is stateless per-step: observe image + instruction → predict one action → execute → observe again. There is no temporal process of commitment across steps.

CogACT gives a genuine temporal axis within a single forward pass: the 10 DDIM denoising steps. This is the sequential decision process worth studying.

## Safety Framing

If an adversary injects a "harmful intent" vector into the VLM's activation space — a direction that causes the conditioning vector to encode "approach human aggressively" — the conditioning vector probe becomes a classifier that can fire before the DiT action head runs.

The statelessness of current VLAs is a security property here, not a limitation. There is no slow accumulation of hidden state across steps. Every forward pass is independent. Every forward pass can be monitored. The attacker cannot do a slow boil.

The research agenda thus becomes:
1. Can we train a probe on the conditioning vector that reliably classifies safe vs. dangerous intent?
2. What is the dimensionality of the dangerous intent subspace? How many probes are needed?
3. Can an attacker craft a perturbation that causes harm but is orthogonal to the probe's detection direction?

## System Constraints

- Hardware: 24 GB VRAM limit.
- CogACT-Small in bf16 uses ~15 GB for model weights, leaving ~9 GB for activations, KV cache, and hook tensors. Hook callbacks must avoid accumulating gradients/tensors excessively.

## Core Technical Goal

The core experiment is a matched-scene probe study:

1. Create a scene with two candidate objects.
2. Run CogACT-Small, capturing the conditioning vector and DDIM intermediate states.
3. Train probes with label = selected object.
4. Compare accuracy on the conditioning vector (before DiT runs) vs. accuracy at each denoising step τ=0..9.

The accuracy comparison is the foundational result. It directly tests whether the transformer "knows" before the DiT starts.

A secondary experiment adds additive intervention: insert steering vector into the conditioning vector and measure whether object choice changes. This validates that the probe target is causally relevant.

## Experiment Flows

### Experiment 1: VLM→DiT Interface Probe

Capture: conditioning vector from VLM to DiT.
Label: target object.
Probe: Train a linear probe on the conditioning vector.

Interpretation:
- High accuracy: transformer latent is a complete plan. Hypothesis confirmed. Monitor sits here.
- Partial accuracy: DiT contributes. Do not declare confirmed until Experiment 2 completes.

### Experiment 2: DDIM Denoising Trajectory Probe

Capture: intermediate trajectory estimates at each τ = 0..9.
Probe: one probe per τ, evaluated per-τ.
Result: accuracy-over-τ curve.

Curve shapes:
- Early spike: DiT locks in target immediately. Transformer did the thinking.
- Late spike: DiT genuinely deliberates. Monitoring must happen inside denoising.
- Staged spike (target-object early, approach-direction late): DiT has a natural decomposition. Novel finding.

### Experiment 3: Safety Monitor and Adversarial Injection

Phase A: Map safety-relevant geometry. Train probes on conditioning vector with labels beyond target object.
Phase B: Attack simulation. Add harmful-intent steering vector to conditioning vector. Measure minimum strength for behavioral change.
Phase C: Monitor evaluation. Does the probe detect the perturbation before harm threshold?
Phase D: Adversarial robustness. Craft a perturbation orthogonal to the probe direction that still causes harm.

### Layer Sweep (Secondary)

Find where in the VLM intent crystallizes to identify optimal monitoring layers before the conditioning handoff.

### Baseline Rollouts / Matched-Scene Intervention

Run seeded baseline scenes followed by the same scene with targeted interventions to verify causal effects.
