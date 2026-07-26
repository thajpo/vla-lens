# Documentation

You do not need to read every file.

## Start Here

Read these in order:

1. [Quickstart](quickstart.md) to run the project.
2. [Current state](current-state.md) to understand what works and what does not.
3. [Dataset format](dataset-format.md) when you need the storage model.

The active feature backlog lives in GitHub issues, not planning documents.

## Run Something

Open these only for the task you are performing:

- [Hardware run paths](hardware-run-paths.md): demo, capture, replay, and normal
  checks.
- [PI0.5 capture profiles](pi05-capture-profiles.md): choose what model internals
  to save and understand the storage cost.
- [PI0.5 ROCm environment](pi05-rocm-capture-env.md): understand the dedicated
  capture environment.
- [Docker](docker.md): dashboard and Linux capture containers.
- [Cloud capture](cloud-capture.md): place large capture outputs on mounted
  storage.
- [Remote GPU to local analysis](remote-gpu-local-analysis.md): capture remotely,
  copy home, and analyze locally.
- [Dashboard API](dashboard-api.md): backend routes and payload conventions.
- [Workbench frontend](workbench-frontend.md): frontend modules and development
  commands.
- [Autonomous research campaigns](autonomous-research-campaigns.md): program
  decision graphs, independently locked child studies, typed reducer-checked
  event history, numeric decision gates, audit roles, evidence retention, and
  generated result summaries.

## Understand The System

These are short durable explanations, not implementation plans:

- [Architecture workflows](vla-lens-architecture-workflows.md): how capture,
  storage, analysis, and UI fit together.
- [Probe evidence](probe-evidence.md): what probe results mean and how the UI
  preserves source evidence.
- [Interventions](interventions.md): saved intervention evidence, replay gating,
  controls, and the current scientific limitation.
- [Model, dataset, and simulator agnosticism](model-dataset-sim-agnosticity.md):
  what belongs in the generic core versus an adapter.
- [Research UI principles](research_ui_principles.md): the UI design contract.
- [Glossary](glossary.md): precise terms used by evidence and intervention code.
- [PI0.5 Lens library](library/pi05-lens.md): reusable PI0.5 analysis helpers.

## Research Record

These files preserve scientific evidence. They are not required onboarding:

- [Research log](../RESEARCH.md): canonical questions, methods, controls,
  findings, and revisit decisions.
- [PI0.5 broad-1000 probe experiments](pi05_broad_1000_probe_experiments.md):
  campaign results, including negative findings and replication guidance.
- [Probe workflow feedback](probe_evidence_workflow_feedback.md): what worked and
  what remained awkward during the probe evidence UI validation.
- [Probe hypothesis guidance](probe_hypothesis_guidance.md): how to propose and
  preflight a probe before spending compute.

## Archive

The [July 2026 system review](audits/vla-lens-system-review/README.md) is a
fixed-revision audit. It is useful when investigating why an architecture issue
exists, but it is not current run guidance.

## Documentation Rule

- Put unfinished feature work in a GitHub issue.
- Use the issue as the implementation plan by default.
- If a temporary plan file is genuinely needed, delete it before merge after
  its useful content is captured in the PR description.
- Keep Markdown only for current behavior, runbooks, durable architecture, and
  research evidence.
- Prefer short prose and bullets over tables or exhaustive AI-generated plans.

## Critical Environment Rule

PI0.5 capture and replay use the hardware wrappers:

```bash
scripts/pi05_capture.sh --backend rocm ...
scripts/pi05_batch_capture.sh --backend cuda ...
scripts/pi05_intervene.sh --backend rocm ...
```

Do not run PI0.5, LeRobot, or LIBERO execution through the normal repo `uv run`
environment.
