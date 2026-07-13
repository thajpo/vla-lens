# Repository Consolidation Implementation Plan

Status: approved staging plan.

Plan date: July 13, 2026.

## Goal

Return VLA Lens to one understandable, green, origin-backed line of development
before starting new feature work.

"Merge all branches" means preserve and integrate every unique line of work. It
does not mean merging detached worktrees, generated directories, or redundant
branch tips. Every integration must retain provenance and pass the relevant
checks before it reaches `master`.

## Safety Contract

- Do not force-push, amend published commits, or use destructive resets.
- Do not mix the system audit, baseline repair, and intervention product work in
  one commit or pull request.
- Do not delete a worktree or branch until its unique commits and untracked
  files have been inspected and either integrated or explicitly abandoned.
- Keep normal development in `.venv` with `uv run`; do not load PI0.5,
  LeRobot, LIBERO, Torch, or a capture environment for this task.
- Keep `master` changes reviewable through focused pull requests. A branch that
  is merely preserved on origin does not need to be merged until its scope is
  validated.
- Do not weaken regression guardrails by raising source-size limits or adding
  blanket import allowlist entries solely to make CI green.

## Verified Starting State

| Item | Verified state | Required disposition |
| --- | --- | --- |
| `master` / `origin/master` | `882eeb8`, June 18 geometry campaign report | Integration base. |
| `codex/interventions-page-target-workflow` | `4f6d974`, pushed, one commit over `master`, no PR | Preserve; replay on top of PR #7 after it merges. |
| `fm/host-vla-data-h1` | `b5e19d5`, pushed, four commits over `master`, open PR #7 | Green, review, and merge first. |
| System review audit | Nine Markdown files originally untracked | Preserve on `codex/system-review-audit` with this index and plan. |
| Codex worktrees | Three clean detached worktrees at `882eeb8` | Remove only after final verification. |
| Treehouse worktrees | Five directories whose `.git` files point to a deleted Firstmate repository | Compare filesystem contents before removal; branch tip for PR #7 is already preserved. |
| Stashes | None | No action. |
| Unreachable commits | One May 25 commit; no recent orphaned work | Confirm it is redundant before final cleanup. |

The current intervention branch completed `341` Python tests and failed two
shared guardrail tests. The changed feature commit does not touch the offending
files, and `src/vla_lens/server/probe_studies.py` has the same 1,532 lines on
`master`, PR #7, and the current branch. This makes baseline repair a separate
precondition, not an intervention-feature fix.

## Phase 1 — Preserve And Publish The System Audit

Branch and worktree:

```text
branch:   codex/system-review-audit
base:     origin/master
worktree: /home/j/Projects/vla-lens-system-audit
```

Scope:

1. Move the nine static audit documents out of the intervention checkout.
2. Add this README and consolidation plan.
3. Replace top-level wording that incorrectly describes the audit as expected
   to remain untracked. Preserve historical command logs inside subsystem
   audits as audit provenance.
4. Scan the staged content for secrets, accidental binary files, and malformed
   whitespace.
5. Commit with:

   ```text
   docs: preserve system review and consolidation backlog
   ```

6. Push the branch and open a draft documentation PR.

Validation:

```bash
git diff --cached --check
git status --short --branch
rg -n -i 'api[_-]?key|secret|password|private key' \
  docs/audits/vla-lens-system-review
```

Acceptance:

- Only `docs/audits/vla-lens-system-review/` is staged and committed.
- The original intervention worktree has no audit files left to mix into its
  feature branch.
- The audit branch exists on origin and has a reviewable PR.

## Phase 2 — Restore The Shared Green Baseline

Create an isolated branch and worktree from the latest `origin/master` after
the audit PR disposition is known:

```text
branch: codex/restore-baseline-checks
scope:  the two currently failing anti-regression contracts
```

Reproduce first:

```bash
uv run pytest \
  tests/research_ui_import_boundary_test.py::test_research_ui_raw_schema_imports_are_explicitly_allowlisted \
  tests/source_file_size_check_test.py::test_refactored_core_files_stay_under_700_lines -vv
```

Implementation:

1. Remove raw dataset-schema coupling from
   `frontend/src/components/workflows/ProbeSuitePreset.tsx` and
   `frontend/src/pages/probeDisplayCopy.ts` by routing research UI through the
   intended evidence/view-model contracts. Add an allowlist rationale only if a
   file is genuinely a narrow boundary adapter.
2. Split coherent responsibilities out of:
   - `src/vla_lens/server/probe_studies.py` (`1,532` lines),
   - `tests/vla_lens_trace_artifacts_test.py` (`924` lines),
   - `tests/fastapi_server_test.py` (`724` lines), and
   - `frontend/src/pages/EpisodesPage.tsx` (`714` lines).
3. Preserve API behavior with focused characterization tests before moving
   logic. Do not raise the 700-line cap.
4. Keep the two repairs as distinct commits if that improves review, but publish
   them through one baseline-restoration PR because they share the single
   acceptance criterion that the existing required test job is green.

Validation:

```bash
uv run pytest
uv run ruff check scripts src tests
cd frontend
npm test
npm run lint
npm run build
```

Acceptance:

- All normal-lane Python and frontend checks pass.
- No capture dependency is imported or installed.
- The source-size and UI-boundary guardrails remain equally or more strict.
- The baseline PR is merged before PR #7 is refreshed.

## Phase 3 — Refresh And Merge PR #7

PR #7 is the smaller, backend-contract-first intervention branch. It normalizes
probe and Episode Lens seeds through backend `TargetSpec` conversion and
propagates token-space identity.

Implementation:

1. Update `fm/host-vla-data-h1` from the new `origin/master` without rewriting
   published history. Prefer merging `origin/master` into the PR branch over a
   force-pushed rebase.
2. Run the full normal-lane validation matrix.
3. Verify its hosted-dataset evidence still demonstrates:
   - backend-normalized target preferred when available,
   - explicit `local_fallback` metadata when unavailable,
   - trace, policy call, model site, and token space preserved.
4. Push the refresh commit, wait for all four GitHub checks, and obtain the
   required human review.
5. Merge PR #7 through GitHub. Do not bypass branch protection for a stale
   failing check.

Acceptance:

- PR #7 reports `4/4` checks passing.
- Required review is satisfied.
- `origin/master` contains `b5e19d5`'s effective changes.

## Phase 4 — Reconcile The Rich Intervention Workflow

Do not force-rebase the already-pushed
`codex/interventions-page-target-workflow`. Preserve it as the original source
branch. Create a new integration branch from updated `origin/master`:

```text
branch: codex/interventions-target-workflow-v2
source commit to replay: 4f6d974
```

Implementation:

1. Cherry-pick `4f6d974` onto the new branch.
2. Resolve the known overlapping files deliberately. The pre-merge simulation
   found textual conflicts in:
   - `frontend/src/components/interventions/interventionLabModel.test.mjs`,
   - `frontend/src/components/interventions/interventionLabModel.ts`, and
   - `frontend/src/pages/episodes/useEpisodeLensView.ts`.
3. Preserve both contracts when resolving:
   - PR #7: backend-normalized target first, explicit local fallback, token
     space propagation;
   - rich workflow: contributor/model-locus selection source, source object
     reference, layer/feature identity, target picker, and canonical
     Interventions routing.
4. Extend tests so a normalized target retains the richer source metadata and a
   fallback remains visibly non-authoritative.
5. Run the full normal-lane validation matrix and open a new PR. Keep the
   original branch unchanged as a recovery point until the new PR merges.

Acceptance:

- No `TargetSpec` information is dropped across probe, Episode Lens, target
  picker, request construction, preflight, and saved inspected evidence.
- Old Evidence URLs still resolve to the canonical Interventions route where
  compatibility is intended.
- Full Python/frontend validation passes.
- The integration PR is reviewed and merged.

## Phase 5 — Update Operational Documentation

After both intervention lines are integrated:

1. Update `docs/current-state.md` beyond its June 6 snapshot.
2. Update `CHANGELOG.md` with the June probe/evidence/intervention work and the
   consolidation outcome.
3. Link this system review from `docs/README.md`.
4. Record the two important scientific conclusions:
   - stronger metadata baselines invalidated the pooled binary probe candidates;
   - geometry probes were mostly negative/diagnostic, with object-local z worth
     methodological confirmation but not intervention promotion.
5. Record remaining architectural work without presenting it as implemented:
   policy-call index, generic example manifest, experiment recipe, unified
   selection state, evidence lineage, and a live intervention vertical slice.

Acceptance:

- Onboarding docs describe the actual merged system.
- Backlog items are distinguishable from shipped capabilities.
- Commands preserve the normal/capture environment split.

## Phase 6 — Worktree And Branch Cleanup

Cleanup is last because directory existence is not proof that work is
redundant.

1. For each registered worktree, record path, expected commit, branch, Git
   pointer health, filesystem differences, and untracked files.
2. Compare the five broken Treehouse directories against their recorded commits
   without trusting their missing Git metadata. Recover any unique file into a
   named branch before deletion.
3. Remove the three clean detached Codex worktrees after their `882eeb8` state
   is confirmed redundant.
4. Remove merged local branches with safe `git branch -d`, never `-D` as the
   default.
5. Request a final explicit confirmation before deleting published remote
   branches. Keeping merged remote branches temporarily is safer than losing a
   recovery reference.
6. Fetch/prune only after all recovery references are recorded.

Acceptance:

```text
local master == origin/master
no relevant untracked or unstaged files
no stashes
no unique unpublished commits
only intentional active worktrees remain
all required GitHub checks on master pass
current-state and changelog describe the merged repository
```

## Stop Conditions And Rollback

- Stop before commit if staged scope contains anything outside the named phase.
- Stop before merge if required checks fail, even when the failure appears
  unrelated; first prove and repair the shared baseline.
- Stop before deleting a broken worktree if its filesystem cannot be compared
  confidently.
- If branch reconciliation changes product semantics beyond the two documented
  contracts, preserve both branches and request an owner decision instead of
  guessing.
- Roll back by reverting the focused merge/commit through normal Git history;
  never erase published history to hide a failed consolidation attempt.
