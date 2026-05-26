# Branch Protection Checklist

Status: local documentation only. No remote branch protection settings are changed by this file.

Recommended default-branch protection:

- Require pull request review before merge.
- Require status checks to pass before merge.
- Require branches to be up to date before merge when practical.
- Require these stable PR checks from `.github/workflows/pr-checks.yml`:
  - `lint`
  - `test`
  - `frontend-build`
  - `docker-dashboard`
- Keep required check names stable unless branch protection is updated at the same time.
- Do not require PI0.5/LeRobot/LIBERO hardware capture smokes on every PR; hosted CI does not prove ROCm, CUDA, MPS, or LIBERO capture readiness.
- For capture-affecting PRs, require reviewer confirmation that backend-specific wrapper checks were run outside normal `uv run`.
- Review repository license status before public distribution or package publication. This repo currently has no tracked license file.
