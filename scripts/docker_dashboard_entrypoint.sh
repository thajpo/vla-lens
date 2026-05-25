#!/usr/bin/env bash
set -euo pipefail

TRACE_ROOT="${VLA_LENS_TRACE_ROOT:-/data/vla-lens/runs/vla_lens_demo}"
FRONTEND_DIST="${VLA_LENS_FRONTEND_DIST:-/app/frontend/dist}"
HOST="${VLA_LENS_HOST:-0.0.0.0}"
PORT="${VLA_LENS_PORT:-8080}"
BACKEND_HOST="${VLA_LENS_BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${VLA_LENS_BACKEND_PORT:-8765}"
BOOTSTRAP_DEMO="${VLA_LENS_BOOTSTRAP_DEMO:-1}"
PUBLIC_URL="${VLA_LENS_PUBLIC_URL:-}"

mkdir -p "$(dirname "$TRACE_ROOT")"

if ! find "$TRACE_ROOT" -maxdepth 3 -name '*.vlatrace' -type d | grep -q .; then
  if [[ "$BOOTSTRAP_DEMO" != "1" ]]; then
    cat >&2 <<EOF
No .vlatrace bundles found under:
  $TRACE_ROOT

Set VLA_LENS_BOOTSTRAP_DEMO=1 to create a synthetic demo dataset, or mount an
existing trace dataset at VLA_LENS_TRACE_ROOT.
EOF
    exit 2
  fi
  echo "No .vlatrace bundles found under $TRACE_ROOT; building synthetic demo dataset."
  python scripts/build_vla_lens_demo.py --out "$TRACE_ROOT" --overwrite
fi

args=(
  scripts/serve_vla_lens_app.py
  "$TRACE_ROOT"
  --frontend-dist "$FRONTEND_DIST"
  --host "$HOST"
  --port "$PORT"
  --backend-host "$BACKEND_HOST"
  --backend-port "$BACKEND_PORT"
)

if [[ -n "$PUBLIC_URL" ]]; then
  args+=(--public-url "$PUBLIC_URL")
fi

exec python "${args[@]}"
