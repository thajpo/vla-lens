# Repository Consolidation Execution Record

Status: completed July 15, 2026.

Plan date: July 13, 2026.

## Goal

Return VLA Lens to one understandable, green, origin-backed line of development
before starting new feature work.

## Execution Record

The consolidation was executed on July 13, 2026 without force-pushing or
rewriting either published intervention branch:

| Phase | Result |
| --- | --- |
| System audit | Preserved on `codex/system-review-audit` in PR #8. |
| Shared baseline | PR #9 merged as `9e716cc`; 342 Python tests and all four required GitHub checks passed. |
| Backend target contract | PR #7 refreshed by merging the repaired baseline, then merged as `e5815f2`; all four required checks passed. |
| Rich intervention workflow | Successor PR #10 merged as `644c4bd`; backend-normalized targets, explicit local fallback metadata, and source-object provenance were reconciled together. The final branch passed 343 Python tests, 64 frontend tests, both linters, the production build, and all four required GitHub checks. |

The successor branch used a merge of the preserved source branch rather than a
single cherry-pick. This retained the original branch ancestry and made the
three semantic conflict resolutions explicit in merge commit `a801cdd` before
GitHub squash-merged the PR.

The configured no-mistakes Claude reviewer began returning HTTP 401 invalid
credentials during this work. Each affected PR records the run ID and the full
manual validation matrix used as fallback. GitHub's required checks were not
bypassed. The repository's one-review rule was bypassed with the repository
owner's administrator authority after the user explicitly requested merging;
automatic merge is not enabled for this repository.

The final worktree audit repaired the five stale Treehouse `.git` pointers and
then used Git itself to verify every registered checkout was clean. Seven
detached worktrees at `882eeb8` and completed branch worktrees were removed only
after that verification. The primary checkout was returned to a clean,
fast-forwarded `master`. The single unreachable commit, `71b585b` from May 25,
was an earlier version of reachable commit `d4b0a7c` with the same parent; the
reachable version preserves its dataset/API change while restoring unrelated
files removed by the earlier snapshot. After the replay work merged in PR #12,
the merged local and remote feature branches were removed and pruned.

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
