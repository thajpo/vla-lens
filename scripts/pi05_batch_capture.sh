#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${PI05_BACKEND:-rocm}"

usage() {
  cat <<'EOF'
Usage:
  scripts/pi05_batch_capture.sh --backend rocm|cuda|mps|cpu [vla-pi05-batch-capture args...]

This wrapper forces batch-generated capture commands to use the matching
hardware virtualenv, device, and dtype even when older YAML configs still carry
machine-local python_executable/device fields.
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
  rocm|cuda|mps|cpu) ;;
  *)
    echo "Unsupported PI0.5 backend: $BACKEND" >&2
    usage >&2
    exit 2
    ;;
esac

default_venv_for_backend() {
  case "$1" in
    rocm) echo "${PI05_ROCM_VENV:-$ROOT/.venv-pi05-rocm}" ;;
    cuda) echo "${PI05_CUDA_VENV:-$ROOT/.venv-pi05-cuda}" ;;
    mps) echo "${PI05_MPS_VENV:-$ROOT/.venv-pi05-mps}" ;;
    cpu) echo "${PI05_CPU_VENV:-$ROOT/.venv-pi05-cpu}" ;;
  esac
}

default_device_for_backend() {
  case "$1" in
    rocm|cuda) echo "cuda" ;;
    mps) echo "mps" ;;
    cpu) echo "cpu" ;;
  esac
}

default_dtype_for_backend() {
  case "$1" in
    rocm|cuda) echo "bfloat16" ;;
    mps|cpu) echo "float32" ;;
  esac
}

VENV="${PI05_VENV:-$(default_venv_for_backend "$BACKEND")}"

if [[ ! -x "$VENV/bin/vla-pi05-batch-capture" ]]; then
  cat >&2 <<EOF
PI0.5 $BACKEND batch capture environment is not ready.

Expected:
  $VENV/bin/vla-pi05-batch-capture

Run:
  scripts/setup_pi05_env.sh --backend $BACKEND

Do not use plain 'uv run vla-pi05-batch-capture' for PI0.5 capture;
it may resync the normal repo environment and break the capture stack.
EOF
  exit 2
fi

"$ROOT/scripts/check_pi05_env.sh" --backend "$BACKEND" >/dev/null

export VLA_LENS_CAPTURE_PYTHON="${VLA_LENS_CAPTURE_PYTHON:-$VENV/bin/python}"
export VLA_LENS_CAPTURE_PYTHONPATH="${VLA_LENS_CAPTURE_PYTHONPATH:-$ROOT/src}"
export VLA_LENS_CAPTURE_DEVICE="${VLA_LENS_CAPTURE_DEVICE:-$(default_device_for_backend "$BACKEND")}"
export VLA_LENS_CAPTURE_DTYPE="${VLA_LENS_CAPTURE_DTYPE:-$(default_dtype_for_backend "$BACKEND")}"

exec "$VENV/bin/vla-pi05-batch-capture" "${args[@]}"
