#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE_ROOT="${1:-$ROOT/runs/vla_lens_demo}"
FRONTEND_DIST="$ROOT/frontend/dist"
HOST="${VLA_LENS_HOST:-127.0.0.1}"
PORT="${VLA_LENS_PORT:-8080}"
BACKEND_PORT="${VLA_LENS_BACKEND_PORT:-8765}"

usage() {
  cat <<'EOF'
Usage:
  scripts/view_vla_lens.sh [TRACE_ROOT]

Serve the built VLA Lens dashboard locally from a LeRobot v3 dataset root,
a directory containing nested LeRobot roots, a trace dataset, or one .vlatrace
bundle. With no TRACE_ROOT, a synthetic demo dataset is created under
runs/vla_lens_demo if needed.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "$ROOT"

has_dataset() {
  [[ -f "$TRACE_ROOT/meta/info.json" && -d "$TRACE_ROOT/data" ]] && return 0
  find "$TRACE_ROOT" -path '*/meta/info.json' -type f | grep -q . && return 0
  find "$TRACE_ROOT" -name '*.vlatrace' -type d | grep -q .
}

if [[ $# -eq 0 ]]; then
  mkdir -p "$TRACE_ROOT"
  if ! has_dataset; then
    uv run python scripts/build_vla_lens_demo.py --out "$TRACE_ROOT" --overwrite
  fi
elif [[ ! -e "$TRACE_ROOT" ]]; then
  echo "Trace root does not exist: $TRACE_ROOT" >&2
  exit 2
fi

if [[ ! -f "$FRONTEND_DIST/index.html" ]]; then
  npm ci --prefix "$ROOT/frontend"
  npm run build --prefix "$ROOT/frontend"
fi

exec uv run python scripts/serve_vla_lens_app.py "$TRACE_ROOT" \
  --frontend-dist "$FRONTEND_DIST" \
  --host "$HOST" \
  --port "$PORT" \
  --backend-port "$BACKEND_PORT" \
  --public-url "http://127.0.0.1:$PORT"
