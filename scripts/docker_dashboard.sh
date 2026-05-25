#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${VLA_LENS_DASHBOARD_IMAGE:-vla-lens-dashboard:local}"
HOST_PORT="${VLA_LENS_DASHBOARD_PORT:-8080}"
CONTAINER_PORT=8080

usage() {
  cat <<'EOF'
Usage:
  scripts/docker_dashboard.sh [DATASET_ROOT]

Run the VLA Lens dashboard container.

Examples:
  scripts/docker_dashboard.sh
  scripts/docker_dashboard.sh runs/pi05-light-5-test
  scripts/docker_dashboard.sh /path/to/some-dataset

No DATASET_ROOT:
  mounts ./runs and creates/serves runs/vla_lens_demo if needed.

With DATASET_ROOT:
  mounts that existing LeRobot v3 dataset root, directory containing nested
  LeRobot roots, trace dataset, or .vlatrace bundle and serves it directly.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "$ROOT"

docker build -f docker/dashboard.Dockerfile -t "$IMAGE" .

if [[ $# -eq 0 ]]; then
  mkdir -p "$ROOT/runs/vla_lens_demo"
  exec docker run --rm \
    -p "$HOST_PORT:$CONTAINER_PORT" \
    -v "$ROOT/runs:/data/vla-lens/runs" \
    -e VLA_LENS_TRACE_ROOT=/data/vla-lens/runs/vla_lens_demo \
    -e VLA_LENS_BOOTSTRAP_DEMO=1 \
    -e VLA_LENS_PUBLIC_URL="http://127.0.0.1:$HOST_PORT" \
    "$IMAGE"
fi

TRACE_ROOT="$1"
if [[ ! -e "$TRACE_ROOT" ]]; then
  echo "Dataset root does not exist: $TRACE_ROOT" >&2
  exit 2
fi

TRACE_ROOT="$(realpath "$TRACE_ROOT")"
exec docker run --rm \
  -p "$HOST_PORT:$CONTAINER_PORT" \
  -v "$TRACE_ROOT:/data/vla-lens/trace-root" \
  -e VLA_LENS_TRACE_ROOT=/data/vla-lens/trace-root \
  -e VLA_LENS_BOOTSTRAP_DEMO=0 \
  -e VLA_LENS_PUBLIC_URL="http://127.0.0.1:$HOST_PORT" \
  "$IMAGE"
