# VLA Lens Architecture And Workflows

## Architecture

```mermaid
flowchart TB
    subgraph Native["Native model/environment world"]
        Env["Environment\nLIBERO, robosuite, SimplerEnv"]
        Policy["Policy / VLA\nPI0.5, OpenVLA, CogACT"]
        Legacy["Existing saved captures\nPI0.5 legacy dirs, OpenVLA runs"]
    end

    subgraph Adapters["Adapter/importer edge"]
        EnvAdapter["EnvAdapter\nobservations, state, rewards, success"]
        ModelAdapter["ModelAdapter\nforward hooks, tokens, modules, actions"]
        Importer["Legacy Importer\nconvert old captures"]
        Recorder["TraceRecorder\nlive rollout orchestration"]
    end

    subgraph Core["vla_lens core"]
        Writer["TraceWriter / TraceBundle.create"]
        Bundle["TraceBundle\none episode-aligned trace"]
        Dataset["TraceDataset\nmany bundles + indexes"]
        Selector["ActivationQuery / FeatureView\naxis-aware slicing + cache"]
        Artifact["LensArtifact\nprobes, attribution, patch results"]
    end

    subgraph Storage["Trace storage"]
        Manifest["manifest.json"]
        Tables["Parquet indexes\ntimesteps, tokens, arrays, activations, artifacts"]
        Arrays["NumPy arrays now\nZarr-compatible direction later"]
        RawRefs["Raw source refs\nlegacy pt/npz/png provenance"]
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
    Legacy --> Importer --> Writer

    Writer --> Bundle
    Bundle --> Manifest
    Bundle --> Tables
    Bundle --> Arrays
    Bundle --> RawRefs

    Bundle --> Dataset
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

## Workflow 1: Capture Or Import Into Trace Bundles

```mermaid
sequenceDiagram
    participant R as Researcher
    participant E as EnvAdapter
    participant M as ModelAdapter
    participant T as TraceRecorder
    participant W as TraceWriter
    participant B as TraceBundle
    participant I as Legacy Importer

    alt live capture
        R->>T: rollout(policy, env, capture config)
        T->>E: reset / step / read cameras and state
        T->>M: forward policy and collect internals
        M-->>T: actions, action chunks, activations, token metadata
        E-->>T: frames, robot state, object state, reward, done
        T->>W: write normalized episode + model-call data
    else import existing captures
        R->>I: convert legacy PI0.5 capture root
        I->>W: normalize meta, state, actions, images, hidden means, flow actions
    end

    W->>B: create .vlatrace bundle
    B-->>R: reusable trace for analysis/dashboard
```

## Workflow 2: Dataset Browsing And Coverage

```mermaid
flowchart LR
    Root["Trace root\nmany .vlatrace bundles"] --> Open["TraceDataset.open"]
    Open --> EpisodeIndex["episode_index\ntrace, task, outcome, model, env"]
    Open --> TimestepIndex["timestep_index\ntime, phase, model_call_index"]
    Open --> ActivationIndex["activation_index\nmodule, layer, tensor_type, axes, shape"]
    Open --> ArtifactIndex["artifact_index\nsaved probes/maps/results"]

    EpisodeIndex --> Filters["Filter episodes\nbenchmark, task, success, object"]
    ActivationIndex --> Coverage["activation_coverage"]
    EpisodeIndex --> TaskStats["stats.by_task"]
    TimestepIndex --> TimelineStats["episode length / call density"]

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
    Bundle["TraceBundle"] --> Generation["generation_actions\ntimestep x generation_step x horizon x action_dim"]
    Bundle --> Chunk["action_chunks\ntimestep x horizon x action_dim"]
    Bundle --> Executed["executed_actions\ntimestep x action_dim"]
    Bundle --> State["robot/object state\noptional"]

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
