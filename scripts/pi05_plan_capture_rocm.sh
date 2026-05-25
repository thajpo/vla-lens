#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PI05_ROCM_VENV:-$ROOT/.venv-pi05-rocm}"

if [[ ! -x "$VENV/bin/python" ]]; then
  cat >&2 <<EOF
PI0.5 ROCm capture environment is not ready.

Expected:
  $VENV/bin/python

Run:
  scripts/setup_pi05_rocm_env.sh

Do not use plain 'uv run' for PI0.5 ROCm capture; it may resync the normal
repo environment and break the capture stack.
EOF
  exit 2
fi

"$ROOT/scripts/check_pi05_env.sh" --backend rocm >/dev/null
export VLA_LENS_CAPTURE_PYTHON="${VLA_LENS_CAPTURE_PYTHON:-$VENV/bin/python}"
export VLA_LENS_CAPTURE_PYTHONPATH="${VLA_LENS_CAPTURE_PYTHONPATH:-$ROOT/src}"
export VLA_LENS_CAPTURE_DEVICE="${VLA_LENS_CAPTURE_DEVICE:-cuda}"
export VLA_LENS_CAPTURE_DTYPE="${VLA_LENS_CAPTURE_DTYPE:-bfloat16}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV/bin/python" -m vla_lens.pi05.plan_capture "$@"
