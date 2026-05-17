# PI0.5 Capture Regeneration Note

## Why Regenerate

Existing PI0.5 target-binding captures are enough for interface-level KV-cache causal tracing, but they are incomplete for deeper expert-side mechanism work.

The key missing field was:

```text
expert_selected_hidden_by_step
```

The capture code now writes this field for future episodes. Existing episodes do not have it, so any investigation that asks where inside the denoising process an expert hidden state matters will require regenerating the relevant episodes.

## New Fields

For each expert policy call, future captures now store these per-denoising-step fields:

```text
expert_selected_hidden_by_step[denoise_step][layer]
expert_selected_residual_input_by_step[denoise_step][layer]
expert_selected_attention_by_step[denoise_step][layer]
suffix_embs_by_step[denoise_step]
adarms_cond_by_step[denoise_step]
```

Expected shape for each tensor:

```text
1 x 50 x 1024 bf16
```

Current selected expert layers:

```text
0, 4, 8, 12, 16, 17
```

This preserves the existing field:

```text
expert_selected_hidden_final_step
```

for compatibility with old analysis scripts.

Each VLM and expert call also now stores `capture_metadata`, including schema version, selected layers, final layer indices, chunk size, action dimension, inference steps, and model class.

## Plain-Language Glossary

We are not capturing literally every activation in the model. We capture a focused set of activations around the VLM-to-action pathway. Full activation capture would include every hidden state, residual stream, attention map, MLP intermediate, normalization value, and projection inside both the VLM and action expert. That would be much larger and is not necessary yet.

`suffix_embs_by_step`:

The expert does not directly process the action chunk as raw numbers. At each denoising step, it embeds the current noisy action `x_t` plus the denoising timestep into token-like vectors. Those vectors are the suffix embeddings. They are the action-side input tokens that get appended after the VLM prefix cache.

`adarms_cond_by_step`:

AdaRMS conditioning is a small conditioning signal used by the expert's adaptive normalization. In plain terms, it tells parts of the expert how to scale/shift computation for the current denoising timestep and action state. It is not the action itself, but it can gate how the expert processes that action state.

`expert_selected_residual_input_by_step`:

Transformer layers use a residual stream: a running hidden vector that flows through the stack. The residual input is the hidden state entering a layer before that layer modifies it. Capturing both residual input and layer output helps answer whether a layer creates a useful signal or merely receives one from earlier layers.

`expert_selected_hidden_by_step`:

This is the selected layer output after the layer has processed its residual input. This is the value we patch when asking whether a specific expert layer at a specific denoising step changes the action.

`expert_selected_attention_by_step`:

Attention weights show which tokens attend to which other tokens inside selected expert layers. In this setting, they can show whether action tokens are looking back at VLM prefix tokens or mostly at other action tokens. They are not proof of causality by themselves, but they are useful diagnostics once a causal layer is identified.

## Storage Cost

Measured from current captures:

```text
one selected expert hidden tensor = 102,400 bytes
6 layers x 9 additional denoising steps = ~5.5 MB per policy call
```

Across sampled episodes, the estimated full-episode storage increase is:

```text
mean:   ~6.97%
median: ~6.97%
```

This is acceptable for focused future captures.

## Regeneration Policy

Do not immediately regenerate every historical episode unless storage/time is cheap. Instead:

1. Regenerate the Scene 4 task 61 good/bad family first.
2. Regenerate the next candidate family only after Scene 4 per-step hidden tracing produces a useful result.
3. If the per-step hidden traces are informative, regenerate the broader target-binding dataset.

The old dataset remains valid for observational analyses and VLM KV-cache interface tracing.

## Other Capture Points Worth Considering

### Highest Priority

- `expert_selected_hidden_by_step`: implemented; needed for per-denoising-step hidden patching.
- `expert_selected_residual_input_by_step`: implemented; distinguishes incoming vs layer-created signal.
- `suffix_embs_by_step`: implemented; captures the action-side tokens entering the expert.
- `adarms_cond_by_step`: implemented; captures adaptive normalization conditioning.
- `expert_selected_attention_by_step`: implemented; captures selected expert attention maps by denoising step.
- More dense VLM KV layer selection near the discovered band: layers `7,8,9,10,12,14` or all VLM KV layers for focused captures.
- Per-call replay metadata: implemented in `capture_metadata` except full model id/dtype when not available from the model object.

### Medium Priority

- Full model id and dtype in each call file if the policy object exposes them reliably.
- VLM selected attention maps beyond the final VLM layer if attention routing becomes central.
- Action projection input/output diagnostics around `action_out_proj` if final motor projection becomes suspect.

### Lower Priority / Only For Focused Runs

- Full VLM hidden states for every layer. Useful but expensive, and KV-cache tracing currently seems more directly causal.
- MLP intermediate activations inside selected VLM/expert layers. Useful for circuit-level work but too detailed before we know the right layers.
- Logits or token probabilities are less central here because PI0.5's relevant output is a continuous action chunk, not a text token.

## Current Best Hypothesis

Scene 4 causal tracing points to a VLM-to-expert interface band, strongest around layers `12` and `14`, with a smaller robust site around layer `8`.

The next regenerated captures should test whether expert hidden states at specific denoising steps mediate this interface-level effect.
