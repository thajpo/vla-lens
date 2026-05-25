#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${PI05_BACKEND:-rocm}"

usage() {
  cat <<'EOF'
Usage:
  scripts/check_pi05_env.sh --backend rocm|cuda|mps|cpu

Set PI05_STRICT_DEVICE_CHECK=0 to check imports without requiring a visible GPU.
EOF
}

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
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
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

VENV="${PI05_VENV:-$(default_venv_for_backend "$BACKEND")}"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  cat >&2 <<EOF
PI0.5 $BACKEND capture environment is missing.

Expected:
  $PY

Run:
  scripts/setup_pi05_env.sh --backend $BACKEND
EOF
  exit 2
fi

PI05_BACKEND="$BACKEND" "$PY" - <<'PY'
import importlib.metadata as md
import os
import platform

backend = os.environ["PI05_BACKEND"]
strict_device = os.environ.get("PI05_STRICT_DEVICE_CHECK", "1") != "0"


def fail(message: str) -> None:
    raise SystemExit(message)


try:
    import torch
except Exception as exc:
    fail(f"torch import failed: {exc}")

if backend == "rocm":
    if "+rocm" not in torch.__version__:
        fail(f"expected ROCm torch build, found torch {torch.__version__}")
    if strict_device and not torch.cuda.is_available():
        fail("ROCm torch build is installed, but torch.cuda.is_available() is false")
elif backend == "cuda":
    if torch.version.cuda is None and "+cu" not in torch.__version__:
        fail(f"expected CUDA torch build, found torch {torch.__version__}")
    if strict_device and not torch.cuda.is_available():
        fail("CUDA torch build is installed, but torch.cuda.is_available() is false")
elif backend == "mps":
    if strict_device and platform.system() != "Darwin":
        fail("MPS backend requires macOS")
    mps = getattr(torch.backends, "mps", None)
    if strict_device and (mps is None or not mps.is_available()):
        fail("torch.backends.mps.is_available() is false")
elif backend == "cpu":
    pass
else:
    fail(f"unknown backend: {backend}")

try:
    from transformers.models.siglip import check
except Exception as exc:
    fail(f"OpenPI transformers replacement import failed: {exc}")

if not check.check_whether_transformers_replace_is_installed_correctly():
    fail("OpenPI transformers replacement check failed")

try:
    import lerobot.policies.pi05.modeling_pi05  # noqa: F401
except Exception as exc:
    fail(f"LeRobot PI0.5 import failed: {exc}")

try:
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
except Exception as exc:
    fail(f"LIBERO import failed: {exc}")

try:
    import robosuite
except Exception as exc:
    fail(f"robosuite import failed: {exc}")

if robosuite.__version__ != "1.4.0":
    fail(f"expected robosuite 1.4.0, found {robosuite.__version__}")

print(f"PI0.5 {backend} capture environment OK")
for pkg in [
    "torch",
    "torchvision",
    "lerobot",
    "transformers",
    "peft",
    "hf-libero",
    "robosuite",
]:
    print(f"  {pkg}: {md.version(pkg)}")
PY
