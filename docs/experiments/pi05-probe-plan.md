# PI0.5 Probe Plan

> Status: superseded. This plan predates the strict metadata-prior gate. Target-identity probes are now background controls, not primary evidence. Future probes should use the held-out-layout gate and focus on behavior-linked object interaction labels such as `first_moved_object` and `first_lifted_object`. See `docs/experiments/pi05-consolidated-findings.md`.

## Scope for the next analysis batch

Focus on two experiments only:

1. **E1: Language routing**
2. **E2: Target identity encoding**

Everything else stays deferred until we understand the basic behavioral and representational structure of the canonical 480-episode dataset.

## Probe target labels

### Within-benchmark probes

Use **multi-class object identity** as the label.

#### `LIBERO_OBJECT`

4-way classification:

- `alphabet_soup`
- `cream_cheese`
- `butter`
- `milk`

#### `Scene 1`

4-way classification:

- `alphabet_soup`
- `cream_cheese`
- `ketchup`
- `tomato_sauce`

### Cross-benchmark transfer probes

Use only the **overlapping classes**:

- `alphabet_soup`
- `cream_cheese`

This becomes a 2-way classification problem for transfer analysis.

## What not to do

- Do **not** train one target-identity probe per task. Within a benchmark, task name and object label are effectively the same variable, so a per-task target probe is degenerate.
- Do **not** start with one giant all-data 6-way probe across both benchmarks as the primary analysis. The benchmark and class labels are partially confounded (`butter`, `milk`, `ketchup`, `tomato_sauce` only appear in one benchmark each), which muddies interpretation.

## Probe families to test

The capture dataset supports all of these from the same saved tensors.

### Family A: pooled summaries

These are cheap, robust, and good for the first pass.

1. all-token mean pool of `prefix_final_hidden_state`
2. text-token mean pool of `prefix_final_hidden_state`
3. image-token mean pool of `prefix_final_hidden_state`

Expert-side analogs:

4. final-step expert hidden mean-pooled across chunk tokens
5. per-flow-step expert hidden mean-pooled across chunk tokens

### Family B: stride pooling

Useful middle ground between pooling everything and probing every token.

Recommended first version:

- non-overlapping mean pools over fixed windows of prefix tokens
- e.g. stride / block sizes of `8`, `16`, or `32`

This gives a coarse spatial heatmap while keeping the number of probes manageable.

### Family C: per-token probes

Primary localization analysis.

- One probe per token index per captured VLM layer.
- Input = hidden state at token position `p`.
- This yields a heatmap of decodability over token positions.

This is the right analysis for questions like:

- does target identity live in image tokens, text tokens, or both?
- where should we look for causal interventions later?

### Family D: per-dimension probes

Secondary / exploratory.

- One probe per hidden dimension per layer.
- Input = one hidden dimension across all prefix tokens.

This is more channel-centric and less directly tied to the current spatial research questions, but it is still useful as an exploratory analysis later.

## Recommendation on probe order

Start with:

1. pooled probes (Family A)
2. stride pooling (Family B)
3. per-token probes (Family C)

Only then, if useful:

4. per-dimension probes (Family D)

Reason:

- pooled probes tell us quickly whether target identity is present at all
- stride pooling gives coarse localization cheaply
- per-token gives the full spatial map once the basic result is established
- per-dimension is interesting but not required for the first mechanistic result

## Train / validation / test splits

## Principle

Split by **layout**, not by individual VLM/expert calls.

Reason:

- multiple VLM/expert calls come from the same rollout and are highly correlated
- the two seeds for a single layout are also highly correlated
- splitting by call would leak near-duplicate examples across train and test

## Split unit

Within each `(benchmark, object_label)` cell, the split unit is:

- **layout id**

All episodes and all calls from the same `(benchmark, object_label, layout)` go to the same split.

Because the canonical dataset is organized as:

- `30 layouts x 2 seeds` per task

the clean split is:

- `20 layouts` train
- `5 layouts` validation
- `5 layouts` test

This yields per class:

- train: `40 episodes`
- validation: `10 episodes`
- test: `10 episodes`

with multiple VLM/expert calls per episode.

## Split design

### Within-benchmark probes

Create one split file per benchmark:

- `libero_object_splits.json`
- `scene1_splits.json`

Each split file maps layout ids to:

- train
- validation
- test

### Cross-benchmark transfer probes

Use the overlapping classes only:

- `alphabet_soup`
- `cream_cheese`

Train on all train+validation layouts from the source benchmark.
Test on the held-out test layouts from the target benchmark.

Recommended transfer analyses:

1. train on `LIBERO_OBJECT`, test on `Scene 1`
2. train on `Scene 1`, test on `LIBERO_OBJECT`

## Outputs to report for E2

For every probe family / capture point:

- test accuracy
- balanced accuracy
- per-class accuracy
- confusion matrix

For per-token and stride probes:

- heatmap of accuracy by token region / token index

## Interpretation guide

### If pooled probes are strong and transfer well

- target identity is encoded in a scene-general way

### If pooled probes are strong within benchmark but transfer poorly

- target encoding is benchmark- or scene-specific

### If per-token probes show high accuracy mainly in text tokens

- target identity may be mostly instruction-side rather than grounded in visual input

### If per-token probes show high accuracy in image tokens too

- stronger evidence of visual grounding

### If expert probes are weak while VLM probes are strong

- information is present in the VLM side but not preserved / used cleanly by the expert

## Immediate next step

Build the analysis script so we can:

1. scan rollout dirs in place
2. enumerate available `(benchmark, task, layout, seed)` cells
3. generate the fixed split files by layout
4. extract the first pooled feature set for E2
