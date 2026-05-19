#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PI05_ROCM_VENV:-$ROOT/.venv-pi05-rocm}"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  cat >&2 <<EOF
PI0.5 ROCm capture environment is missing.

Expected:
  $PY

Run:
  scripts/setup_pi05_rocm_env.sh
EOF
  exit 2
fi

"$PY" - <<'PY'
import importlib.metadata as md

try:
    import torch
except Exception as exc:
    raise SystemExit(f"torch import failed: {exc}") from exc

if "+rocm" not in torch.__version__:
    raise SystemExit(f"expected ROCm torch build, found torch {torch.__version__}")

try:
    from transformers.models.siglip import check
except Exception as exc:
    raise SystemExit(f"OpenPI transformers replacement import failed: {exc}") from exc

if not check.check_whether_transformers_replace_is_installed_correctly():
    raise SystemExit("OpenPI transformers replacement check failed")

try:
    import lerobot.policies.pi05.modeling_pi05  # noqa: F401
except Exception as exc:
    raise SystemExit(f"LeRobot PI0.5 import failed: {exc}") from exc

try:
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
except Exception as exc:
    raise SystemExit(f"LIBERO import failed: {exc}") from exc

try:
    import robosuite
except Exception as exc:
    raise SystemExit(f"robosuite import failed: {exc}") from exc

if robosuite.__version__ != "1.4.0":
    raise SystemExit(f"expected robosuite 1.4.0, found {robosuite.__version__}")

print("PI0.5 ROCm capture environment OK")
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

