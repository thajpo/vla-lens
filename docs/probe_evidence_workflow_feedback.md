# Probe Evidence Workflow Feedback Log

This log records concrete probe workflows exercised against the v1
`ProbeEvidenceBundle` loop. It is intentionally short and operational: each
entry should say what research question was attempted, what path was followed,
what evidence was missing or confusing, and whether the durable contract
changed.

## 2026-06-10 - Raw layer contribution probe loop

Evidence source:

- `tests/fixtures/probe_evidence/raw_layer_contribution.json`
- Backend selector coverage in `tests/probe_evidence_selectors_test.py`
- Frontend selector and route coverage in `frontend/src/types/probeEvidence.test.mjs`
- Pin persistence and reopening coverage in `tests/evidence_pins_test.py` and
  `frontend/src/pages/workbench/episodeRouteModel.test.mjs`

Question attempted:

```text
For the grasp-intent probe, which dataset moment is most strongly positive,
where did the probe read from, and what numeric contributor pushed the score?
```

Click path exercised:

```text
select dataset demo
  -> select probe-grasp-intent / run-probe-grasp-intent
  -> select top ranked moment episode-1, policy_call 3
  -> open episode microscope with dataset/lens/lens_run/ranking/call context
  -> default model inspector to action_head.layers.8.resid
  -> inspect contributor dim_42 with claim_level numeric_only
  -> pin evidence state
  -> reopen pin through #episode route
```

Observed evidence:

- Ranked moment: `episode-1`, `policy_call=3`, `ranking=top`, score `0.88`.
- Prediction evidence is present for the selected moment.
- Model locus evidence resolves to `action_head.layers.8.resid`.
- Contribution evidence resolves to `dim_42`, positive sign, `claim_level=numeric_only`.
- Pin payload preserves dataset, lens artifact, lens run, episode, policy call,
  model site, contributor, primitive kind, score, prediction, and claim level.

Missing evidence:

- This workflow does not prove a causal mechanism; it only shows a probe score,
  source locus, and numeric contribution.
- Failure-case browsing depends on labels/proxy targets and remains unavailable
  when the bundle lacks `failure_case` evidence.
- No random/control probe comparison is available in v1.

Confusing panel or copy:

- Raw activation dimensions must not be presented as semantic features. The
  contract and tests now force `claim_level=numeric_only` for this workflow.
- Raw dataset/capture schema imports still exist in bridge components, but Phase
  7 now requires each bridge to be explicitly allowlisted with a rationale.

Wished-for comparison:

- Compare this top positive moment against a control/random probe.
- Compare top positive moments against false positives once labels or proxy
  targets exist.
- Compare the same contributor across success/failure cohorts.

Contract change:

- No new evidence primitive was added for this workflow.
- Phase 6 added minimal evidence pins based on `ResearchSelectionState`.
- Phase 7 strengthened guardrails so capability-gated panels, failure-case
  selectors, missing-label copy, and raw-schema import boundaries are tested.

Residual risk:

- This is an implementation-path exercise using the canonical probe fixture and
  route/pin tests. It should be followed by a human-in-browser usability pass on
  a current research dataset before treating the workflow as UX-complete.
