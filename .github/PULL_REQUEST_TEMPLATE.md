## Summary

<!-- What changed and why? -->

## Scope

- [ ] I stayed within the intended ownership boundary for this change.
- [ ] I did not mix normal repo work with PI0.5 capture runtime setup.
- [ ] I updated docs or templates when user-facing commands, validation, or workflow expectations changed.
- [ ] This PR does not make a repository license decision. If licensing/distribution changed, link a license decision issue.

## Environment Split

Normal repo/dev/test/server work uses `.venv` and `uv run`.

PI0.5/LeRobot/LIBERO capture work uses `.venv-pi05-rocm`, `.venv-pi05-cuda`, or `.venv-pi05-mps` through:

- `scripts/pi05_capture.sh --backend rocm|cuda|mps ...`
- `scripts/pi05_batch_capture.sh --backend rocm|cuda|mps ...`
- `scripts/docker_pi05_cuda.sh ...` or `scripts/docker_pi05_rocm.sh ...` for Linux capture containers

Do not validate PI0.5 capture by running plain `uv run vla-pi05-capture` in the normal repo environment.

## Validation

- [ ] `scripts/check_vla_lens.sh`
- [ ] `uv run pytest`
- [ ] `uv run ruff check scripts src tests`
- [ ] `cd frontend && npm run lint`
- [ ] `cd frontend && npm run build`
- [ ] Capture-specific smoke, if applicable: `scripts/check_pi05_env.sh --backend ...`
- [ ] Capture-specific smoke, if applicable: wrapper-based one-episode or batch capture
- [ ] Not applicable / explained below

Notes:

<!-- Mention skipped checks, capture-specific validation, or admin decisions here. -->
