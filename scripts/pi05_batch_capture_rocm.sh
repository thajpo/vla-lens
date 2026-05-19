#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PI05_ROCM_VENV:-$ROOT/.venv-pi05-rocm}"

if [[ ! -x "$VENV/bin/vla-pi05-batch-capture" ]]; then
  cat >&2 <<EOF
PI0.5 ROCm capture environment is not ready.

Expected:
  $VENV/bin/vla-pi05-batch-capture

Run:
  scripts/setup_pi05_rocm_env.sh

Do not use plain 'uv run vla-pi05-batch-capture' for PI0.5 ROCm capture;
it may resync the normal repo environment and break the capture stack.
EOF
  exit 2
fi

"$ROOT/scripts/check_pi05_rocm_env.sh" >/dev/null
exec "$VENV/bin/vla-pi05-batch-capture" "$@"
