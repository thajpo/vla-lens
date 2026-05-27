# VLA Lens Architecture And Workflows

Status: active architecture contract.

Last updated: May 27, 2026.

## Dataset Layer Cutoff

The canonical dataset shape is now LeRobotDataset v3 robot data plus a VLA Lens
overlay:

```text
LeRobot v3 meta/data/videos
  + vla_lens/ tables, arrays, artifacts, fingerprints
```

LeRobot owns observations, actions, episode/frame indexes, timestamps, task
metadata, and camera media. VLA Lens owns model internals, policy-call
alignment, token metadata, probes, artifacts, and dashboard state. Standalone
episode-bundle directories are an internal overlay primitive, not a dataset
compatibility layer.

## Architecture

```mermaid
flowchart TB
    subgraph Native["Native model/environment world"]
        Env["Environment\nLIBERO, robosuite, SimplerEnv"]
        Policy["Policy / VLA\nPI0.5, OpenVLA, CogACT"]
    end

    subgraph Adapters["Adapter/importer edge"]
        EnvAdapter["EnvAdapter\nobservations, state, rewards, success"]
        ModelAdapter["ModelAdapter\nforward hooks, tokens, modules, actions"]
        Recorder["LeRobot Recorder\nlive rollout orchestration"]
    end

    subgraph Core["vla_lens core"]
        Writer["DatasetWriter\nLeRobot + overlay"]
        RobotData["LeRobot v3 robot data\nmeta, data, videos"]
        Overlay["VLA Lens overlay\nmodel sites, tokens, artifacts"]
        Dataset["Dataset view\nLeRobot + overlay"]
        Selector["ActivationQuery / FeatureView\naxis-aware slicing + cache"]
        Artifact["LensArtifact\nprobes, attribution, patch results"]
    end

    subgraph Storage["Dataset storage"]
        Meta["LeRobot meta\ninfo, stats, tasks, episodes"]
        Tables["LeRobot data parquet\nlow-dimensional robot rows"]
        Media["LeRobot videos\nMP4 camera streams"]
        Interp["vla_lens/\nmodel arrays and research artifacts"]
    end

    subgraph Tools["Research tools"]
        Probes["Probe suites"]
        Attribution["Attribution / attention maps"]
        Patching["Offline and replay patching"]
        Dashboard["Dashboard / reports"]
        Stats["Dataset stats / coverage"]
    end

    Env --> EnvAdapter --> Recorder
    Policy --> ModelAdapter --> Recorder
    Recorder --> Writer

    Writer --> RobotData
    Writer --> Overlay
    RobotData --> Meta
    RobotData --> Tables
    RobotData --> Media
    Overlay --> Interp

    RobotData --> Dataset
    Overlay --> Dataset
    Dataset --> Selector
    Dataset --> Stats
    Selector --> Probes
    Selector --> Attribution
    Selector --> Patching
    Probes --> Artifact
    Attribution --> Artifact
    Patching --> Artifact
    Artifact --> Dataset
    Dataset --> Dashboard
    Artifact --> Dashboard
```

## Workflow 1: Capture Or Import Into Dataset Roots

```mermaid
sequenceDiagram
    participant R as Researcher
    participant E as EnvAdapter
    participant M as ModelAdapter
    participant C as CaptureRunner
    participant L as LeRobot v3 Root
    participant O as vla_lens Overlay

    R->>C: rollout(policy, env, capture config)
    C->>E: reset / step / read cameras and state
    C->>M: forward policy and collect internals
    M-->>C: actions, action chunks, activations, token metadata
    E-->>C: frames, robot state, object state, reward, done
    C->>L: write robot data to meta, data, videos
    C->>O: write model internals, policy calls, tokens, artifacts
    O-->>R: reusable interpretability overlay joined to LeRobot keys
```

## Workflow 2: Dataset Browsing And Coverage

```mermaid
flowchart LR
    Root["LeRobot v3 root\n+ vla_lens/ overlay"] --> Open["Dataset view"]
    Open --> EpisodeIndex["episode_index\ntask, outcome, model, env"]
    Open --> FrameIndex["frame_index\ntime, policy_call_index"]
    Open --> ActivationIndex["model_site_index\nmodule, layer, tensor_type, axes, shape"]
    Open --> ArtifactIndex["artifact_index\nsaved probes/maps/results"]

    EpisodeIndex --> Filters["Filter episodes\nbenchmark, task, success, object"]
    ActivationIndex --> Coverage["activation_coverage"]
    EpisodeIndex --> TaskStats["stats.by_task"]
    FrameIndex --> TimelineStats["episode length / call density"]

    Filters --> Cohort["Research cohort\nsuccess/failure pairs, task subsets"]
    Coverage --> Plan["Choose feasible probes/views"]
    TaskStats --> Plan
    TimelineStats --> Plan
```

## Workflow 3: Probe Suite After Capture

```mermaid
flowchart TB
    Dataset["TraceDataset"] --> Query["ActivationQuery\nmodule pattern, layers, token kind, timesteps"]
    Dataset --> Labels["LabelSelector / row labels\nobject, success, phase, action target"]
    Query --> FeatureView["FeatureView"]
    FeatureView --> Cache{"feature cache hit?"}
    Cache -- yes --> Matrix["load cached X + rows"]
    Cache -- no --> Materialize["load arrays lazily\nreduce tokens / select timesteps"]
    Materialize --> Matrix
    Matrix --> Train["Train probes\nlogistic, ridge, linear sweep"]
    Labels --> Train
    Train --> Metrics["metrics\naccuracy, R2, margins, baselines"]
    Train --> Predictions["predictions over episode/time/layer"]
    Metrics --> Artifact["ProbeSuite LensArtifact"]
    Predictions --> Artifact
    Artifact --> Dashboard["Dashboard\nlayer x time heatmaps and timelines"]
```

## Workflow 4: Attribution Or Attention Overlay

```mermaid
flowchart LR
    Dataset["TraceDataset"] --> Episode["Select episode + timestep"]
    Episode --> Frame["Camera frame"]
    Episode --> Tokens["Token metadata\nimage patches, action tokens, text tokens"]
    Episode --> Activations["Activation / attention arrays"]

    Tokens --> Map["PatchMap\ncamera, patch row/col, pixel bounds"]
    Activations --> Score["Attribution score\nattention mass, grad x act, norm, probe attribution"]
    Score --> ScalarField["scalar per image patch"]
    Map --> Overlay["overlay heatmap on frame"]
    ScalarField --> Overlay
    Overlay --> Artifact["AttributionMap LensArtifact"]
    Artifact --> Dashboard["Dashboard\nscrub timestep/layer/head and update overlay"]
```

## Workflow 5: Flow Action And Generation Analysis

```mermaid
flowchart TB
    Overlay["vla_lens/ overlay"] --> Generation["generation_actions\nframe x generation_step x horizon x action_dim"]
    Overlay --> Chunk["action_chunks\nframe x horizon x action_dim"]
    Robot["LeRobot data"] --> Executed["action\nframe x action_dim"]
    Robot --> State["observation.state / context\noptional"]

    Generation --> Commit["commitment metric\n||A_s - A_final||"]
    Generation --> Heatmap["generation_step x horizon heatmaps"]
    Chunk --> Receding["receding-horizon view\npredicted chunks over time"]
    Executed --> Receding
    State --> Outcome["behavior context\ncontact, lift, distance, success"]

    Commit --> Artifact["ActionGeneration LensArtifact"]
    Heatmap --> Artifact
    Receding --> Artifact
    Outcome --> Artifact
    Artifact --> Dashboard["Dashboard\ncompare flow formation to executed behavior"]
```

## Workflow 6: Offline Patching And Intervention Results

```mermaid
sequenceDiagram
    participant R as Researcher
    participant D as TraceDataset
    participant S as Selector
    participant P as Patcher
    participant M as ModelAdapter
    participant A as Artifact
    participant UI as Dashboard

    R->>D: choose target failure trace and source success trace
    R->>S: define SiteSpec(module, layer, timestep, token/generation step)
    S-->>R: verify activation availability and shapes
    R->>P: run_offline_patch(target, source, site, metrics)
    P->>M: replay same observation through model with activation hook
    M-->>P: original and patched action outputs
    P->>P: compute action delta, flow delta, probe delta, optional attribution delta
    P->>A: save PatchResult LensArtifact
    A->>D: register artifact in trace/dataset index
    D->>UI: display original vs patched action chunk and metrics
```
