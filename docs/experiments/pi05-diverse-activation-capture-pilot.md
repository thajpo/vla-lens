# PI0.5 Diverse Activation Capture Pilot

## Goal

Collect a much larger and more diverse PI0.5 activation dataset so probe results are not dominated by the existing `LIBERO_OBJECT` and `LIBERO_90` Scene 1 captures.

The immediate concern is simple:

> A probe may look like it decodes object identity or position, but it may just be learning benchmark/task/layout regularities.

## Storage Target

`New Volume` is mounted at:

```text
/media/j/New Volume
```

It is a USB SanDisk SSD formatted as NTFS, with about `932 GB` free before the pilot.

Future captures should write under:

```text
/media/j/New Volume/vla-lens-artifacts/pi05_diverse_captures
```

## Disk Speed Check

Sequential synthetic IO:

| Drive | Write Speed | Read Speed |
|---|---:|---:|
| root NVMe ext4 | ~1964 MB/s | cached read ~12133 MB/s |
| New Volume USB NTFS | ~384 MB/s | cached read ~12193 MB/s |

Torch-load spot check:

| File | Cold-ish Load | Cached Load |
|---|---:|---:|
| root VLM call, 66 MB | ~0.03 s | ~0.024 s |
| New Volume VLM call, 66 MB | ~0.31 s | ~0.022 s |

Interpretation:

- `New Volume` is slower than the internal NVMe, especially for first reads.
- It is still fast enough for capture. Current episodes write a few hundred MB over several seconds, while model/env runtime is the bottleneck.
- Reformatting may help somewhat, but the main speed difference is likely USB SSD vs internal NVMe, not just NTFS vs ext4.
- Do not reformat unless we later see IO bottlenecks during large analysis jobs.

## Pilot Command

```bash
uv run vla-pi05-batch-capture \
  --output-root "/media/j/New Volume/vla-lens-artifacts/pi05_diverse_captures" \
  --benchmarks libero_spatial,libero_object,libero_goal,libero_10,libero_90 \
  --tasks-per-benchmark 4 \
  --layouts 1 \
  --seeds 1000 \
  --max-episodes 20
```

## Pilot Result

The first run exposed a bug: the old capture code assumed every task had `basket_1`. That is false for drawer/stove/cabinet tasks. The capture code was fixed to use the environment's actual object-state list.

Final pilot result:

| Benchmark | Episodes | Success Rate | Mean Steps | Mean Runtime |
|---|---:|---:|---:|---:|
| `libero_spatial` | 4 | 1.0 | 94.2 | 7.4 s |
| `libero_object` | 4 | 1.0 | 130.8 | 7.8 s |
| `libero_goal` | 4 | 1.0 | 124.8 | 7.6 s |
| `libero_10` | 4 | 0.5 | 276.5 | 14.0 s |
| `libero_90` | 4 | 0.0 | 320.0 | 15.6 s |

Storage:

| Quantity | Value |
|---|---:|
| manifest episodes | 20 |
| successful capture data | 6.43 GiB |
| full pilot directory | 7.5 GiB |
| mean episode size | 329 MB |
| median episode size | 265 MB |
| min / max episode size | 151 MB / 530 MB |

Most storage is still VLM activations:

| Component | Pilot Size |
|---|---:|
| VLM | 6.52 GiB |
| Expert / flow | 0.93 GiB |
| Images | 0.01 GiB |

## 1000-Episode Run Shape

The simplest 1000-episode run is:

```text
5 benchmarks x 10 tasks x 10 layouts x 2 seeds = 1000 episodes
```

Command:

```bash
uv run vla-pi05-batch-capture \
  --output-root "/media/j/New Volume/vla-lens-artifacts/pi05_diverse_captures" \
  --benchmarks libero_spatial,libero_object,libero_goal,libero_10,libero_90 \
  --tasks-per-benchmark 10 \
  --layouts 10 \
  --seeds 1000,2000 \
  --max-episodes 1000
```

Expected storage from pilot average:

```text
1000 episodes x ~329 MB = ~321 GiB
```

This fits comfortably on `New Volume`.

Expected runtime from pilot:

```text
roughly 3-5 hours, depending on failure-heavy tasks and environment overhead
```

## Data Quality Notes

The pilot success rates are not balanced. That is not automatically bad, because failures are useful, but we should avoid a dataset where a probe can infer benchmark/task identity and therefore infer success or object layout.

For probe validity, the 1000-episode dataset should preserve:

- multiple benchmarks
- multiple tasks per benchmark
- multiple layouts per task
- multiple seeds per layout
- both successes and failures
- enough repeated task/layout structure to train probes, but enough held-out layouts to test generalization

The first 1000-episode run should be treated as broad activation collection, not final causal evidence.
