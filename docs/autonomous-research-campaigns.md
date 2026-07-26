# Autonomous Research Campaigns

An autonomous research program is a decision graph, not a queue of scripts.
Each result-bearing child study is locked separately. The agent may run cheap
declared work without repeatedly asking the researcher, but it may not change a
child study's question, cohort, primary metric, threshold, controls, or
confirmation site after seeing results.

## Five Sources With Different Jobs

1. The GitHub campaign issue is a short human projection of the checked event
   state. It is never an independent source of lifecycle truth.
2. `configs/campaigns/*.yaml` holds the immutable program decision graph and
   the separately locked child contracts. A child fingerprint identifies the
   exact job an agent actually ran.
3. The campaign event directory is hash-chained and reduced by deterministic
   transition rules. It is the lifecycle authority for locks, pool access,
   budgets, attempts, audits, and terminal results. `study_advanced` is only a
   checked projection and cannot activate work.
4. Native probe, patch, counterfactual, and rollout artifacts are the evidence.
   The campaign points to them instead of copying their arrays.
5. `RESEARCH.md` records the resulting change in scientific belief. It is not a
   command diary.

Research-run records in the workbench may show progress, parent/child grouping,
and artifact links. They are lifecycle views, not the scientific protocol.

## Check A Program Plan

This runs in the normal repository environment. By default it writes nothing:

```bash
uv run python scripts/validate_research_program.py \
  configs/campaigns/rq024_controlled_scene_to_behavior.yaml

uv run python scripts/validate_research_program.py \
  configs/campaigns/rq024_controlled_scene_to_behavior.yaml --json
```

The human view leads with the question, competing explanations, study gates,
and program fingerprint. JSON is intended for an auditor agent. This is a
schema and cross-field consistency check. It does not inspect datasets, prove
that a runner works, establish that a confirmation lock predates results, or
judge whether the science is good.

Persist the exact checked snapshot when preparing a review or lock:

```bash
uv run python scripts/validate_research_program.py \
  configs/campaigns/rq024_controlled_scene_to_behavior.yaml \
  --output /path/to/study/design_check.json
```

The snapshot contains the plan, fingerprint, checker hash, Git commit, dirty
state, command, and explicit limits of the check. Snapshot writes are
create-only: identical content is accepted, but different content is never
overwritten. A later agent can pass `--expect-fingerprint sha256:...` to refuse
a changed plan. Strict loaders reject duplicate YAML or JSON keys and
noncanonical values before hashing.

The program plan contains no mutable status. Lifecycle state is reconstructed
from the typed event chain; the issue and research-run records only render that
state. Each result-bearing study gets a
separate child plan with an exact cohort, job matrix, runner, metric, controls,
budget, output root, and completion rule. That child plan—not the umbrella
program—is committed and fingerprinted before execution.

## Prepare And Lock One Child

Start FOUNDATION from
[`rq024_foundation.child.template.yaml`](../configs/campaigns/rq024_foundation.child.template.yaml).
The template is deliberately invalid until the exact candidate, exposure,
trial, environment, checkpoint, and runner files exist. This is a visible
blocker, not a placeholder that authorizes hardware work.

Check the child envelope without touching model or simulator code:

```bash
uv run python scripts/validate_research_child.py \
  /path/to/rq024_foundation.child.yaml \
  --program configs/campaigns/rq024_controlled_scene_to_behavior.yaml
```

The child binds the exact program and study, predecessor result-event hashes, family pool,
cohort, exposure log, trial table, metric, controls, uncertainty method,
checkpoint, environment, code, runner, separate seed domains, budgets, outputs,
retry behavior, and resume identity.

The child cannot contain its own fingerprint or the Git commit that contains
it; either would be a self-reference. After committing the child, separate
auditors write reports bound to its externally computed fingerprint. A lock
receipt based on
[`research_child_lock.template.yaml`](../configs/campaigns/research_child_lock.template.yaml)
binds those reports and the commit containing the child.

The parent program owns the formula, controls, required outputs, required
audits, budget ceilings, and allowed claims. The child cannot replace them. It
resolves only concrete execution choices and every study-specific lock field.
Confirmations load their actual discovery child from the event chain and must
keep its metric, useful-effect threshold, controls, inference, decision gates,
analysis code, checkpoint, and runner unchanged.

Use the strict templates for
[`audit reports`](../configs/campaigns/research_audit.template.yaml),
[`data-access records`](../configs/campaigns/research_data_access.template.yaml),
and [`trial runtime receipts`](../configs/campaigns/research_runtime_receipt.template.yaml).
Passing audit checks require immutable evidence references; a `pass` report
cannot contain a warning or failed check.

Only the full preflight may authorize execution:

```bash
uv run python scripts/validate_research_child.py \
  /path/to/rq024_foundation.child.yaml \
  --program configs/campaigns/rq024_controlled_scene_to_behavior.yaml \
  --lock-receipt /path/to/rq024_foundation.lock.yaml \
  --event-root research/rq024/events \
  --verify-files \
  --claim-output \
  --output research/rq024/foundation-start-authorization.json
```

This verifies exact input and audit bytes, Git tracking and code-tree identity,
the child lock, reducer-derived predecessors and gates, its active budget
reservation, an atomic child-fingerprint output claim, and current free space.
Append an `execution_authorized` event that cites this receipt before recording
any pool access or starting a trial. A matching abandoned claim containing only
its marker is safe to resume; a nonempty or mismatched directory fails closed.
The PI0.5 wrapper must still run its capture-runtime check at launch. A valid
umbrella plan alone never authorizes hardware execution.

## Tamper-Evident Campaign History

Every state change gets one create-only event. Concurrent agents serialize
appends, each event records the exact previous event-file hash, and the reducer
rejects illegal event order before writing. Editing an old event breaks the
chain. Matching hashes alone are insufficient: referenced child, lock,
analysis, audit, and result documents are loaded and validated when they enter
state.

```bash
uv run python scripts/research_campaign_event.py append \
  --program configs/campaigns/rq024_controlled_scene_to_behavior.yaml \
  --event-root research/rq024/events \
  --event-id rq024-program-locked-r1 \
  --event-type program_locked \
  --actor-id planner-agent \
  --subject-id rq024-controlled-scene-to-behavior \
  --subject-fingerprint sha256:... \
  --payload /path/to/event_payload.yaml

uv run python scripts/research_campaign_event.py verify \
  --program configs/campaigns/rq024_controlled_scene_to_behavior.yaml \
  --event-root research/rq024/events

uv run python scripts/research_campaign_event.py status \
  --program configs/campaigns/rq024_controlled_scene_to_behavior.yaml \
  --event-root research/rq024/events
```

`status` is the normal agent entry point. It reports whether the chain and
referenced bytes are valid, the active gate, whether hardware is authorized,
the deterministic next action, blocked studies, terminal results, and spent
budget. It never infers permission from issue prose.

The legal execution order is program lock → child preparation and parent-owned
audits → budget reservation → child lock → full preflight and
`execution_authorized` → permitted pool access → create-only trial attempts →
one typed result audit covering execution, calculation, and claim → validated
result → budget release. A result becomes a predecessor only after release.
Fabricated verdict rows and an empty-ledger `pool_accessed` or
`study_advanced` event are rejected.

## Agent Roles

- The planner defines the comparison and locks the plan.
- The executor runs only declared trials and records all failures.
- The analyst recomputes declared measurements from saved trial outputs.
- A separate auditor verifies pairing, exclusions, hashes, calculations,
  controls, and claim wording.
- The summarizer renders the result from structured measurements and the audit.

An agent may fill several roles during exploration. Confirmation should use a
separate auditor and a committed plan that existed before confirmation outputs.

## Gates

Every planned study has explicit outputs, controls, an advance rule, stop rules,
allowed conclusions, forbidden conclusions, outcome actions, a budget, and
required audits. Entry conditions are the graph's only source of truth; an
agent does not reconcile a second transition graph. The umbrella plan is not a
runnable job. Passing an audit means the experiment followed its contract; it
does not make the scientific hypothesis true.

Any change to a claim-bearing child choice creates a new child revision.
Operational retries keep the same child plan but preserve the failed attempt and
create a linked new attempt. Resume must reject outputs made under a different
child fingerprint.

Confirmation is prospective rather than magically secret. Baseline competence
may be inspected when the program explicitly permits it, but counterfactual and
internal results cannot be used for selection. The selected metric, minimum
useful effect, model site, scope, controls, analysis code, cohort hash, and first
access log are committed before confirmation outputs exist. Only a valid
`confirmed_negative` supersedes the exploratory positive. Failure to pass the
positive gate is `inconclusive` unless a separately locked, family-level
negative bound passes. `invalid`, `not_applicable`, and `inconclusive` results
never supersede a source claim.

RQ-024 uses two confirmation access phases. Frozen behavior jobs may read only
the behavior namespace first. Semantic and causal confirmation jobs are all
locked before any phase-two activation, key/value, readout, or patch output is
opened. Every confirmation read has `selection_allowed: false` and is written
to the exposure log.

## Evidence That Stays

Keep the small material needed to audit the conclusion permanently:

- plan, cohort, exclusions, commands, code and environment identity;
- checkpoint, dataset, trace, seed, and noise identities;
- trial status, failures, physical actions, behavior, predictions, and controls;
- candidate tables, measurements, uncertainty, audit issues, and verdict.

Feature matrices, full hidden states, and prefix key/value caches may be
deleted when their source captures, exact construction recipe, shape, byte
size, and hashes are retained. If the source capture is deleted, mark the run
as no longer rerunnable instead of claiming that the compact artifact is
self-contained.

Use distinct words for distinct reproducibility checks:

- `verify`: inspect schemas and hashes;
- `recompute`: rebuild measurements and the summary from saved outputs;
- `replay-readout`: apply an already fitted probe again;
- `refit`: repeat the frozen probe search and selection;
- `rerun`: execute PI0.5 or the simulator again.

## Result Summary

The short summary is generated only after the audit. Effect summaries always answer:

1. What was the question?
2. What changed, and what stayed fixed?
3. What was the primary effect in its declared unit and its interval?
4. What was the strongest control?
5. How many independent tasks and scene clusters contributed?
6. Was this discovery or prospectively held-out confirmation?
7. What failed, was missing, or was excluded?
8. What conclusion is supported, and what conclusion is forbidden?
9. Which alternative explanation remains strongest?
10. Which artifact and metric IDs reproduce each number?
11. Does the campaign advance, stop, or fork a new question?

`not_applicable` is separate from a negative result: it means structural
support was absent. FOUNDATION uses `gate_passed` or `gate_failed`, not a fake
effect verdict. Scientific effects use `exploratory_negative`,
`exploratory_positive`, `confirmed_negative`, or `confirmed_positive`;
`invalid` and `inconclusive` remain available. Execution status such as
`completed` is separate.

The analysis package is a strict numeric input, not a narrative. Start from
[`configs/campaigns/research_analysis.template.yaml`](../configs/campaigns/research_analysis.template.yaml).
Its metric intervals must exactly match the child's method, level, grouping
unit, repeat count, seed, and predeclared independent-unit counts. Its decision
values are evaluated against typed child gates. Result cards cannot supply
free-form pass/fail booleans.

Start from
[`configs/campaigns/research_result_card.template.yaml`](../configs/campaigns/research_result_card.template.yaml)
and render the checked result deterministically:

```bash
uv run python scripts/render_research_summary.py /path/to/result_card.yaml \
  --program configs/campaigns/rq024_controlled_scene_to_behavior.yaml \
  --child-plan /path/to/locked_child.yaml \
  --child-lock /path/to/child_lock.yaml \
  --audit-report /path/to/audit.json \
  --analysis-package /path/to/analysis.json \
  --authorization-receipt /path/to/start_authorization.json \
  --attempt-ledger /path/to/attempt_ledger_snapshot.json \
  --budget-record /path/to/budget_record.json \
  --event-root research/rq024/events \
  --result-event-id REPLACE_WITH_ACCEPTED_RESULT_EVENT_ID
```

The renderer binds actual program, study, child, child lock, authorization,
trial, attempt range, event-chain tip, budget, analysis, audit, artifact, and
predecessor hashes. It derives numeric gate results, verdict, and next action
from the analysis and program rather than trusting prose. Effect intervals state
method, level, family grouping, repeats, seed, and source artifact. Comparable
controls use the same unit. FOUNDATION starts from
[`research_preparation_result_card.template.yaml`](../configs/campaigns/research_preparation_result_card.template.yaml)
and reports eligible-family counts without inventing an interval.
The CLI refuses to render an authoritative summary unless the exact result card
and analysis are already bound to the named accepted `result_recorded` event.

## Execution Readiness

Readiness is established only by a child job preflight, not a word in the
program YAML. The current RQ-024 control issue names the active child and its
next authorized action. An agent should implement each small reusable capability
at the study that first needs it, test it, lock the resulting child job, and then
execute. It must not silently replace a missing method with global pooling, raw
normalized-action L2, or the old two-object pose swap.

RQ-024 is currently at FOUNDATION preparation, not model execution. The exact
missing inputs are a versioned task-object-family parser, candidate and rejection
tables, a 72-row seed-separated trial table, resolved checkpoint snapshot
receipt, machine capture-environment receipt, and runner config. Simulator
contact telemetry is also missing; the existing end-effector-distance proxy is
not contact or collision data. Agents may build and audit these inputs
autonomously inside the program budget, but may not call the template ready.

## Honest Limits

The controls prove document identity, legal ordering, declared budget use,
output freshness, and reproducibility of recorded calculations. They do not
prove that the simulator produced honest bytes, that a metric is scientifically
appropriate, or that two different agent IDs are truly independent people.
Those require recomputation and review. A local hash chain also cannot detect
deletion of its final events unless a known tip is anchored elsewhere. Anchor
every child lock and result milestone in a pushed Git commit or the campaign
issue, including event count and tip hash; otherwise call the ledger internally
tamper-evident, not permanently append-only.
