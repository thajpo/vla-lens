# VLA Lens Dataset Format

Status: active architecture contract.

Last updated: May 27, 2026.

## Canonical Layout

VLA Lens now treats LeRobotDataset v3 as the canonical robot-data layer.

```text
dataset-root/
  meta/
    info.json
    stats.json
    tasks.parquet
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

Standalone episode-bundle directories are not a public dataset compatibility
layer. The repository supports only:

```text
LeRobot v3 robot data + vla_lens/ interpretability overlay
```

Do not add broad backwards-compatibility aliases to the core contract. If a
specific old artifact needs conversion, keep it outside the canonical dataset
schema and do not wire it into `TraceDataset.open`.

The current implementation writes this layout for PI0.5 capture. The normal
dashboard/test stack reads and writes the file contract directly without
importing `lerobot`, Torch, LeRobot policies, or simulator packages. The capture
containers/native capture environments still carry those runtime dependencies.

## Writer Behavior

PI0.5 capture now writes:

```text
meta/info.json
meta/stats.json
meta/tasks.parquet
meta/tasks.jsonl
meta/episodes/chunk-000/file-000.parquet
data/chunk-000/file-000.parquet
videos/observation.images.<camera>/chunk-000/file-000.mp4
vla_lens/overlay.json
vla_lens/tables/episode_refs.parquet
vla_lens/episodes/episode_000000/...
```

The robot layer stores `action`, optional `observation.state`, frame indexes,
timestamps, rewards/done flags, and MP4 camera streams. The overlay stores
policy calls, token tables, action chunks, generation trajectories, context
tables/arrays, model-site tensors, artifacts, and fingerprints.

`TraceDataset.open(path)` accepts a LeRobot v3 dataset root or a directory
containing nested LeRobot v3 roots. For a LeRobot root, the dashboard can show
episodes even when
`vla_lens/` is absent; model internals simply appear unavailable. For a
top-level batch output, the opener discovers nested `meta/info.json` + `data/`
roots and presents their episodes as one dataset view. Nested roots that have a
VLA Lens overlay keep their captured `trace_id`; plain LeRobot roots without an
overlay get a stable path-derived prefix so repeated `episode_000000` IDs do
not collide.

## Validation

The dependency-free validator checks:

- required metadata: `meta/info.json`, `meta/stats.json`, and task metadata
  through `meta/tasks.parquet` or `meta/tasks.jsonl`
- at least one episode metadata parquet under `meta/episodes/`
- at least one low-dimensional data parquet under `data/`
- required step fields including `episode_index`, `frame_index`, `timestamp`,
  `task_index`, and `action`
- MP4 video shards under `videos/` when image features are declared
- overlay tables under `vla_lens/tables/` only reference known LeRobot
  `episode_index` and in-range `frame_index` values

Contract validation lives in `vla_lens.capture.lerobot_v3`; read/write storage
lives in `vla_lens.lerobot_dataset`.
