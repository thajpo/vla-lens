#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEMO_ROOT="${VLA_LENS_DEMO_ROOT:-$ROOT/runs/vla_lens_demo}"
BACKEND_HOST="${VLA_LENS_BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${VLA_LENS_BACKEND_PORT:-8765}"
FRONTEND_PORT="${VLA_LENS_FRONTEND_PORT:-5173}"

backend_pid=""
frontend_pid=""

cleanup() {
  if [[ -n "$frontend_pid" ]] && kill -0 "$frontend_pid" 2>/dev/null; then
    kill "$frontend_pid" 2>/dev/null || true
  fi
  if [[ -n "$backend_pid" ]] && kill -0 "$backend_pid" 2>/dev/null; then
    kill "$backend_pid" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

cd "$ROOT"

echo "== Build synthetic demo dataset =="
uv run python scripts/build_vla_lens_demo.py --out "$DEMO_ROOT" --overwrite

echo "== Ensure frontend dependencies =="
if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  npm ci --prefix "$ROOT/frontend"
else
  echo "frontend/node_modules already exists"
fi

echo "== Start backend =="
uv run python scripts/serve_vla_lens_dashboard.py "$DEMO_ROOT" \
  --host "$BACKEND_HOST" \
  --port "$BACKEND_PORT" &
backend_pid="$!"

for _ in $(seq 1 80); do
  if uv run python - "$BACKEND_HOST" "$BACKEND_PORT" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

host, port = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(f"http://{host}:{port}/api/dataset", timeout=0.25) as response:
    raise SystemExit(0 if response.status == 200 else 1)
PY
  then
    break
  fi
  sleep 0.25
done

echo "== Start frontend =="
(
  cd "$ROOT/frontend"
  VLA_LENS_BACKEND_URL="http://$BACKEND_HOST:$BACKEND_PORT" npm run dev -- --port "$FRONTEND_PORT"
) &
frontend_pid="$!"

cat <<EOF

VLA Lens demo is running.

Frontend:
  http://127.0.0.1:$FRONTEND_PORT/

Backend:
  http://$BACKEND_HOST:$BACKEND_PORT/api/dataset

Press Ctrl-C to stop both servers.
EOF

wait -n "$backend_pid" "$frontend_pid"
