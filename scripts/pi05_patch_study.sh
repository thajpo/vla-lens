#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${PI05_BACKEND:-rocm}"

usage() {
  cat <<'EOF'
Usage:
  scripts/pi05_patch_study.sh --backend rocm|cuda|mps|cpu \
    DATASET_ROOT --study STUDY.json [vla-pi05-patch-study args...]

Without --run-study the command performs a runtime-free plan inspection.
Execution requires --run-study plus explicit replay tolerances.
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
  echo "Run scripts/setup_pi05_env.sh --backend $BACKEND" >&2
  exit 2
fi

"$ROOT/scripts/check_pi05_env.sh" --backend "$BACKEND" >/dev/null

case "$BACKEND" in
  rocm|cuda) defaults=(--device cuda --dtype bfloat16) ;;
  mps) defaults=(--device mps --dtype float32) ;;
  cpu) defaults=(--device cpu --dtype float32) ;;
esac

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  exec "$VENV/bin/python" -m vla_lens.pi05.patch_study_runner \
  "${defaults[@]}" "${args[@]}"
