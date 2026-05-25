#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${PI05_BACKEND:-auto}"
PYTHON_VERSION="${PI05_PYTHON:-3.11}"
OPENPI_PATCH_DIR="${PI05_OPENPI_PATCH_DIR:-$ROOT/.cache/openpi-transformers-replace}"

usage() {
  cat <<'EOF'
Usage:
  scripts/setup_pi05_env.sh --backend rocm|cuda|mps|cpu|auto

Environment overrides:
  PI05_VENV             explicit virtualenv path
  PI05_ROCM_VENV        ROCm virtualenv path
  PI05_CUDA_VENV        CUDA virtualenv path
  PI05_MPS_VENV         Apple Silicon/MPS virtualenv path
  PI05_CPU_VENV         CPU virtualenv path
  PI05_PYTHON           Python version, default 3.11
  PI05_ROCM_INDEX_URL   PyTorch ROCm wheel index
  PI05_CUDA_INDEX_URL   PyTorch CUDA wheel index
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

detect_backend() {
  if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    echo "mps"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    echo "cuda"
  elif command -v rocminfo >/dev/null 2>&1 || [[ -x /opt/rocm/bin/rocminfo ]]; then
    echo "rocm"
  else
    echo "cpu"
  fi
}

if [[ "$BACKEND" == "auto" ]]; then
  BACKEND="$(detect_backend)"
fi

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

echo "Setting up PI0.5 capture environment:"
echo "  repo:    $ROOT"
echo "  backend: $BACKEND"
echo "  venv:    $VENV"

uv venv "$VENV" --python "$PYTHON_VERSION"

echo "Installing VLA Lens package shell and base runtime dependencies..."
uv pip install --python "$PY" --no-deps -e "$ROOT"
uv pip install --python "$PY" \
  "duckdb>=1.1,<2.0" \
  "hydra-core>=1.3,<2.0" \
  "imageio>=2.37,<3.0" \
  "imageio-ffmpeg>=0.6,<0.7" \
  "matplotlib>=3.10,<4.0" \
  "numcodecs>=0.13,<0.16" \
  "numpy==1.26.4" \
  "pandas>=2.2,<3.0" \
  "pyarrow==19.0.1" \
  "pyyaml>=6.0,<7.0" \
  "scikit-learn>=1.6,<2.0" \
  "zarr>=2.18,<3.0"

echo "Installing PyTorch stack for backend=$BACKEND..."
case "$BACKEND" in
  rocm)
    uv pip install --python "$PY" \
      torch torchvision torchaudio \
      --index-url "${PI05_ROCM_INDEX_URL:-https://download.pytorch.org/whl/rocm7.2}"
    ;;
  cuda)
    uv pip install --python "$PY" \
      torch torchvision torchaudio \
      --index-url "${PI05_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
    ;;
  mps|cpu)
    uv pip install --python "$PY" torch torchvision torchaudio
	  ;;
esac

echo "Installing native build helpers for simulator/runtime packages..."
uv pip install --python "$PY" setuptools wheel cmake ninja
export PATH="$VENV/bin:$PATH"
export CMAKE_POLICY_VERSION_MINIMUM="${CMAKE_POLICY_VERSION_MINIMUM:-3.5}"

echo "Installing LeRobot without dependency resolution to avoid replacing Torch..."
uv pip install --python "$PY" "lerobot==0.4.4" --no-deps

echo "Installing PI0.5/LIBERO runtime dependencies..."
uv pip install --python "$PY" \
  "datasets==4.8.5" \
  "diffusers==0.35.2" \
  "huggingface-hub==0.35.3" \
  "accelerate==1.13.0" \
  "setuptools" \
  "cmake" \
  "einops" \
  "opencv-python-headless==4.12.0.88" \
  "av==15.1.0" \
  "jsonlines==4.0.0" \
  "packaging" \
  "pynput==1.8.1" \
  "pyserial==3.5" \
  "wandb==0.24.2" \
  "draccus==0.10.0" \
  "gymnasium==1.3.0" \
  "rerun-sdk==0.26.2" \
  "deepdiff==8.6.2" \
  "termcolor==3.3.0" \
  "transformers==4.53.2" \
  "safetensors==0.7.0" \
  "hf-libero==0.1.3" \
  "peft==0.19.1"

echo "Forcing LIBERO-compatible robosuite..."
uv pip install --python "$PY" --no-deps --reinstall "robosuite==1.4.0"

echo "Writing non-interactive LIBERO config if needed..."
LIBERO_CONFIG_DIR="${LIBERO_CONFIG_PATH:-$HOME/.libero}"
LIBERO_CONFIG_FILE="$LIBERO_CONFIG_DIR/config.yaml"
LIBERO_DATASETS_DIR="${LIBERO_DATASETS_PATH:-$LIBERO_CONFIG_DIR/datasets}"
mkdir -p "$LIBERO_CONFIG_DIR" "$LIBERO_DATASETS_DIR"

LIBERO_BENCHMARK_ROOT="$("$PY" - <<'PY'
from pathlib import Path
import site

for root in site.getsitepackages():
    candidate = Path(root) / "libero" / "libero"
    if candidate.exists():
        print(candidate)
        break
else:
    raise SystemExit("could not locate installed libero/libero package")
PY
)"

if [[ ! -f "$LIBERO_CONFIG_FILE" || "${PI05_FORCE_LIBERO_CONFIG:-0}" == "1" ]]; then
  cat > "$LIBERO_CONFIG_FILE" <<EOF
benchmark_root: $LIBERO_BENCHMARK_ROOT
bddl_files: $LIBERO_BENCHMARK_ROOT/bddl_files
init_states: $LIBERO_BENCHMARK_ROOT/init_files
datasets: $LIBERO_DATASETS_DIR
assets: $LIBERO_BENCHMARK_ROOT/assets
EOF
fi

echo "Installing OpenPI transformers replacement patch..."
mkdir -p "$(dirname "$OPENPI_PATCH_DIR")"
if [[ ! -d "$OPENPI_PATCH_DIR/.git" ]]; then
  rm -rf "$OPENPI_PATCH_DIR"
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/Physical-Intelligence/openpi.git \
    "$OPENPI_PATCH_DIR"
  git -C "$OPENPI_PATCH_DIR" sparse-checkout set src/openpi/models_pytorch/transformers_replace
fi

SITE_PACKAGES="$("$PY" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
cp -R "$OPENPI_PATCH_DIR/src/openpi/models_pytorch/transformers_replace/"* \
  "$SITE_PACKAGES/transformers/"

echo "Verifying capture environment..."
"$ROOT/scripts/check_pi05_env.sh" --backend "$BACKEND"

cat <<EOF

Done.

Use:
  scripts/pi05_capture.sh --backend $BACKEND ...
  scripts/pi05_batch_capture.sh --backend $BACKEND ...

Backend-specific wrappers are also available when useful.
Avoid:
  uv run vla-pi05-capture ...
  uv run vla-pi05-batch-capture ...
EOF
