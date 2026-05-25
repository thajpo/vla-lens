#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PI05_PYTHON="${PI05_ROCM_PYTHON:-${PI05_PYTHON:-3.11}}" \
  exec "$ROOT/scripts/setup_pi05_env.sh" --backend rocm "$@"
