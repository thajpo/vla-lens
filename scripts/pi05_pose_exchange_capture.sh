#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${PI05_BACKEND:-rocm}"

usage() {
  cat <<'EOF'
Usage:
  scripts/pi05_pose_exchange_capture.sh --backend rocm|cuda|mps|cpu \
    OUTPUT_ROOT --job CAPTURE_JOB.json [capture args...]

Without --run-capture the command only inspects the paired capture plan.
EOF
}

args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)
      BACKEND="${2:-}"
      shift 2
      ;;
    --backend=*)
      BACKEND="${1#--backend=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

case "$BACKEND" in
  rocm) VENV="${PI05_VENV:-${PI05_ROCM_VENV:-$ROOT/.venv-pi05-rocm}}" ;;
  cuda) VENV="${PI05_VENV:-${PI05_CUDA_VENV:-$ROOT/.venv-pi05-cuda}}" ;;
  mps) VENV="${PI05_VENV:-${PI05_MPS_VENV:-$ROOT/.venv-pi05-mps}}" ;;
  cpu) VENV="${PI05_VENV:-${PI05_CPU_VENV:-$ROOT/.venv-pi05-cpu}}" ;;
  *)
    echo "Unsupported PI0.5 backend: $BACKEND" >&2
    usage >&2
    exit 2
    ;;
esac

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "PI0.5 $BACKEND environment is not ready at $VENV" >&2
  exit 2
fi

"$ROOT/scripts/check_pi05_env.sh" --backend "$BACKEND" >/dev/null

case "$BACKEND" in
  rocm|cuda) defaults=(--device cuda --dtype bfloat16) ;;
  mps) defaults=(--device mps --dtype float32) ;;
  cpu) defaults=(--device cpu --dtype float32) ;;
esac

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  exec "$VENV/bin/python" -m vla_lens.pi05.pose_exchange_capture \
  "${defaults[@]}" "${args[@]}"
