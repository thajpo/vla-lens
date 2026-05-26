#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

echo "== Python lint =="
uv run --frozen ruff check scripts src tests

echo "== Python tests =="
uv run --frozen pytest

echo "== Frontend dependencies =="
npm ci --prefix "$ROOT/frontend" --prefer-offline

echo "== Frontend lint =="
npm run lint --prefix "$ROOT/frontend"

echo "== Frontend tests =="
npm run test --prefix "$ROOT/frontend"

echo "== Frontend build =="
npm run build --prefix "$ROOT/frontend"

echo "VLA Lens checks passed."
