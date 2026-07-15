#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${PI05_BACKEND:-rocm}"

usage() {
  cat <<'EOF'
Usage:
  scripts/pi05_intervene.sh --backend rocm|cuda|mps|cpu \
    DATASET_ROOT --request REQUEST.json [vla-pi05-intervene args...]

The command measures deterministic no-op replay first. An intervention runs
only with --run-intervention and explicit --max-noop-l2/--max-noop-max-abs.

Backend defaults:
  rocm/cuda -> --device cuda --dtype bfloat16
  mps/cpu   -> --device mps|cpu --dtype float32
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

has_arg() {
  local name="$1"
  local item
  for item in "${args[@]}"; do
    if [[ "$item" == "$name" || "$item" == "$name="* ]]; then
      return 0
    fi
  done
  return 1
}

VENV="${PI05_VENV:-$(default_venv_for_backend "$BACKEND")}" # capture-specific runtime
if [[ ! -x "$VENV/bin/vla-pi05-intervene" ]]; then
  cat >&2 <<EOF
PI0.5 $BACKEND intervention environment is not ready.

Expected:
  $VENV/bin/vla-pi05-intervene

Run:
  scripts/setup_pi05_env.sh --backend $BACKEND

Do not use plain 'uv run vla-pi05-intervene'; it may resync the normal repo
environment and break the capture-specific stack.
EOF
  exit 2
fi

"$ROOT/scripts/check_pi05_env.sh" --backend "$BACKEND" >/dev/null

if ! has_arg "--device"; then
  args+=("--device" "${PI05_INTERVENTION_DEVICE:-$(default_device_for_backend "$BACKEND")}")
fi
if ! has_arg "--dtype"; then
  args+=("--dtype" "${PI05_INTERVENTION_DTYPE:-$(default_dtype_for_backend "$BACKEND")}")
fi

exec "$VENV/bin/vla-pi05-intervene" "${args[@]}"
