# TODO

## Project Goal

Determine where intent lives in CogACT — does the transformer backbone encode complete task intent in the conditioning vector it passes to the DiT action head, or does the diffusion process itself deliberate? Then test whether linear probes on that conditioning vector can serve as a runtime safety monitor.

## What We Have

- deterministic robosuite Stack environment wrapper
- episode rollouts and summary logging
- backend abstraction for OpenVLA (retained as comparison baseline)

## What We Do Not Have Yet

- CogACT-Small backend
- hook at the VLM→DiT conditioning vector interface
- hook at DDIM intermediate denoising states
- probe dataset with label = target object (cubeA vs cubeB)
- any results from Experiments 1, 2, or 3

## Immediate Next Steps

1. **Drop MiniVLA.** Remove the `minivla` backend. It adds complexity and does not contribute to the research question.

2. **Add CogACT-Small backend**. Load the model and wire the output to the environment. Ensure smoke validation passes.

3. **Wire the two hook sites**:
   - **Conditioning vector**: the tensor the VLM passes to the DiT at inference time. This is the primary probe target.
   - **DDIM intermediates**: the partially denoised trajectory estimate at each of the 10 denoising steps.

4. **Run baseline rollouts with captures**:
   - Run multi-episode rollouts to capture intermediate states and conditioning vectors, varying target objectives.

5. **Train first probe** (Experiment 1):
   - Train a probe on the conditioning vector (X) predicting target object label (y).
   - Interpret accuracy to determine if the transformer latent serves as a complete plan or if the DiT contributes to the decision.

6. **Probe the DDIM trajectory** (Experiment 2):
   - Plot accuracy over denoising steps to characterize when diffusion commits to a target.

## Later (After Experiments 1 and 2)

- Layer sweep on the VLM backbone: which layer + token position does intent crystallize?
- Experiment 3: adversarial injection and safety monitor.
- Causal patching: does zeroing/replacing the conditioning vector change object choice?
- OpenVLA comparison.

## Anti-Bloat Rules

- Keep data capture minimized to required hooks.
- Build the first probe result before adding intervention hooks.
- No fine-tuning or LoRA during data collection.
