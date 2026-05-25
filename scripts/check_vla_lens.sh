#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

echo "== Python lint =="
uv run ruff check scripts src tests

echo "== Python tests =="
uv run pytest

echo "== Frontend dependencies =="
if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  npm ci --prefix "$ROOT/frontend"
else
  echo "frontend/node_modules already exists"
fi

echo "== Frontend lint =="
npm run lint --prefix "$ROOT/frontend"

echo "== Frontend build =="
npm run build --prefix "$ROOT/frontend"

echo "VLA Lens checks passed."
