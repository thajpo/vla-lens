# VLA Lens Dataset Format

Status: active architecture contract.

## Canonical Layout

VLA Lens now treats LeRobotDataset v3 as the canonical robot-data layer.

```text
dataset-root/
  meta/
    info.json
    stats.json
    tasks.jsonl
    episodes/...parquet
  data/
    ...parquet
  videos/
    ...mp4
  vla_lens/
    overlay.json
    tables/
    arrays/
    artifacts/
```

The LeRobot layer owns observations, actions, episode/frame indexes, timestamps,
task metadata, low-dimensional robot data, and camera media. The VLA Lens
overlay owns model internals, policy-call alignment, token metadata, model-site
metadata, derived artifacts, fingerprints, and dashboard/research state.

The overlay must join back to LeRobot rows with LeRobot keys:

```text
episode_index
frame_index
timestamp
task_index
```

## Robot Fields

Canonical robot fields follow LeRobot naming:

```text
episode_index
frame_index
timestamp
task_index
action
observation.state
observation.images.<camera>
```

`action_chunks`, `generation_actions`, token streams, attention tensors, hidden
states, and probe artifacts are not robot dataset fields. They belong in the
VLA Lens overlay.

## Cutoff Policy

Standalone `.vlatrace` episode bundles are old internal storage, not the public
dataset compatibility layer. The repository may still read them while existing
capture and dashboard paths are being replaced, but new dataset-layer work must
target:

```text
LeRobot v3 robot data + vla_lens/ interpretability overlay
```

Do not add broad backwards-compatibility aliases to the core contract. If a
specific old artifact needs conversion, that should be a one-off importer or
research utility outside the canonical dataset schema.

This implementation is contract-only. It validates LeRobot-like roots and
overlay references without importing `lerobot`, Torch, LeRobot policies, or
video encoders.

## Validation

The dependency-free validator checks:

- required metadata: `meta/info.json`, `meta/stats.json`, and task metadata
  through `meta/tasks.jsonl` or `meta/tasks.parquet`
- at least one episode metadata parquet under `meta/episodes/`
- at least one low-dimensional data parquet under `data/`
- required step fields including `episode_index`, `frame_index`, `timestamp`,
  `task_index`, and `action`
- MP4 video shards under `videos/` when image features are declared
- overlay tables under `vla_lens/tables/` only reference known LeRobot
  `episode_index` and in-range `frame_index` values

Current code lives in `vla_lens.capture.lerobot_v3`.
