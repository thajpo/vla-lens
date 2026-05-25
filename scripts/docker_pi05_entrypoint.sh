#!/usr/bin/env bash
set -euo pipefail

BACKEND="${PI05_BACKEND:-cuda}"
VENV="${PI05_VENV:-/opt/vla-lens/pi05}"

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="/app/src${PYTHONPATH:+:$PYTHONPATH}"
export VLA_LENS_CAPTURE_PYTHON="${VLA_LENS_CAPTURE_PYTHON:-$VENV/bin/python}"
export VLA_LENS_CAPTURE_PYTHONPATH="${VLA_LENS_CAPTURE_PYTHONPATH:-/app/src}"

case "$BACKEND" in
  rocm|cuda)
    export VLA_LENS_CAPTURE_DEVICE="${VLA_LENS_CAPTURE_DEVICE:-cuda}"
    export VLA_LENS_CAPTURE_DTYPE="${VLA_LENS_CAPTURE_DTYPE:-bfloat16}"
    ;;
  *)
    export VLA_LENS_CAPTURE_DEVICE="${VLA_LENS_CAPTURE_DEVICE:-cpu}"
    export VLA_LENS_CAPTURE_DTYPE="${VLA_LENS_CAPTURE_DTYPE:-float32}"
    ;;
esac

if [[ $# -eq 0 ]]; then
  set -- vla-pi05-batch-capture --help
fi

case "$1" in
  batch)
    shift
    set -- vla-pi05-batch-capture "$@"
    ;;
  capture)
    shift
    set -- vla-pi05-capture "$@"
    ;;
  check)
    shift
    set -- scripts/check_pi05_env.sh --backend "$BACKEND" "$@"
    ;;
  bash|sh|python|vla-pi05-batch-capture|vla-pi05-capture)
    ;;
  *)
    set -- vla-pi05-batch-capture "$@"
    ;;
esac

exec "$@"
