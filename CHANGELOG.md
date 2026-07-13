# Changelog

All notable changes to this project should be recorded here.

This project follows an explicit environment split:

- Normal repo/dev/test/server work uses `.venv` and `uv run`.
- PI0.5/LeRobot/LIBERO capture uses `.venv-pi05-*` and the backend wrapper scripts.

## [Unreleased]

### Added

- Repo hygiene baseline for GitHub issue templates, pull request validation, and branch-protection documentation.
- Discovery-artifact APIs, LensView episode inspection, probe evidence bundles,
  evidence pins, and expanded probe study diagnostics.
- Probe workflow preflight, stable trained-probe identities, stronger metadata
  baselines, object-flow targets, and geometry-campaign specifications/results.
- Typed intervention evidence contracts, preflight and saved-run APIs,
  backend-normalized target seeding, a source-aware target picker, canonical
  Interventions routing, and richer intervention provenance.
- A static system audit covering the domain model, storage/indexing, researcher
  workflow, capture/runtime, UI state, evidence semantics, and test posture.

### Changed

- Recorded that stronger metadata baselines invalidate the pooled binary probe
  candidates as intervention targets.
- Recorded the mostly negative/diagnostic geometry campaign; object-local `z`
  remains a confirmation candidate rather than a promoted intervention target.
- Consolidated the June intervention branches without rewriting published
  history, preserving backend target normalization and the richer selection UX.

### Fixed

- Restored the research-UI schema boundary and split oversized server,
  frontend, and test modules without weakening the 700-line guardrail.
- Returned all four required GitHub checks to green before merging the pending
  intervention and audit work.

## [0.1.0] - 2026-05-26

### Baseline

- Initial tracked release baseline for the VLA Lens package and documentation.
- Repository license is MIT; see `LICENSE`.
