#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PI05_ROCM_VENV:-$ROOT/.venv-pi05-rocm}"
PYTHON_VERSION="${PI05_ROCM_PYTHON:-3.11}"
OPENPI_PATCH_DIR="${PI05_OPENPI_PATCH_DIR:-$ROOT/.cache/openpi-transformers-replace}"

echo "Setting up PI0.5 ROCm capture environment:"
echo "  repo:  $ROOT"
echo "  venv:  $VENV"

uv venv "$VENV" --python "$PYTHON_VERSION"
PY="$VENV/bin/python"

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

echo "Installing ROCm PyTorch stack..."
uv pip install --python "$PY" \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm7.2

echo "Installing LeRobot without dependency resolution to avoid replacing ROCm Torch..."
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
"$ROOT/scripts/check_pi05_rocm_env.sh"

cat <<EOF

Done.

Use:
  scripts/pi05_capture_rocm.sh ...
  scripts/pi05_batch_capture_rocm.sh ...

Avoid:
  uv run vla-pi05-capture ...
  uv run vla-pi05-batch-capture ...
EOF
