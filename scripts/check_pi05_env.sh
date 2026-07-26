#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${PI05_BACKEND:-rocm}"
RECEIPT=""
JSON_STDOUT=0

usage() {
  cat <<'EOF'
Usage:
  scripts/check_pi05_env.sh --backend rocm|cuda|mps|cpu [--receipt PATH] [--json]

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
    --receipt)
      RECEIPT="${2:-}"
      shift 2
      ;;
    --receipt=*)
      RECEIPT="${1#--receipt=}"
      shift
      ;;
    --json)
      JSON_STDOUT=1
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

PI05_BACKEND="$BACKEND" PI05_ENV_RECEIPT="$RECEIPT" PI05_JSON_STDOUT="$JSON_STDOUT" "$PY" - <<'PY'
import hashlib
import importlib.metadata as md
import json
import os
import platform
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from packaging.version import Version
import yaml

backend = os.environ["PI05_BACKEND"]
strict_device = os.environ.get("PI05_STRICT_DEVICE_CHECK", "1") != "0"
receipt_path = os.environ.get("PI05_ENV_RECEIPT", "").strip()
json_stdout = os.environ.get("PI05_JSON_STDOUT") == "1"


def fail(message: str) -> None:
    raise SystemExit(message)


def package_version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        fail(f"required package is not installed: {name}")


def assert_version_range(name: str, lower: str, upper: str) -> None:
    version = Version(package_version(name).split("+", 1)[0])
    if version < Version(lower) or version >= Version(upper):
        fail(f"expected {name}>={lower},<{upper}, found {package_version(name)}")


def assert_exact_version(name: str, expected: str | None) -> None:
    if not expected:
        return
    actual = package_version(name)
    if actual != expected:
        fail(f"expected {name}=={expected}, found {actual}")


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
openpi_check_path = Path(check.__file__).resolve()

try:
    import lerobot.policies.pi05.modeling_pi05  # noqa: F401
except Exception as exc:
    fail(f"LeRobot PI0.5 import failed: {exc}")

try:
    with redirect_stdout(sys.stderr):
        import libero
        from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
except Exception as exc:
    fail(f"LIBERO import failed: {exc}")

libero_config_file = Path(
    os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero"))
).expanduser() / "config.yaml"
if not libero_config_file.is_file():
    fail(
        f"LIBERO config is missing: {libero_config_file}. Run "
        f"scripts/setup_pi05_env.sh --backend {backend}"
    )
try:
    libero_config = yaml.safe_load(libero_config_file.read_text())
except Exception as exc:
    fail(f"LIBERO config could not be read: {libero_config_file}: {exc}")
if not isinstance(libero_config, dict):
    fail(f"LIBERO config must contain a path mapping: {libero_config_file}")
for key in ("benchmark_root", "bddl_files", "init_states"):
    configured_path = libero_config.get(key)
    if not configured_path or not Path(str(configured_path)).expanduser().exists():
        fail(
            f"LIBERO config has a missing {key} path: {configured_path!r}. "
            f"Repair it with PI05_FORCE_LIBERO_CONFIG=1 "
            f"scripts/setup_pi05_env.sh --backend {backend}"
        )

try:
    with redirect_stdout(sys.stderr):
        import robosuite
except Exception as exc:
    fail(f"robosuite import failed: {exc}")

if robosuite.__version__ != "1.4.0":
    fail(f"expected robosuite 1.4.0, found {robosuite.__version__}")

assert_version_range("numpy", "2.0", "2.3")
assert_version_range("pyarrow", "21.0", "25.0")
assert_exact_version("lerobot", "0.4.4")
assert_exact_version("datasets", "4.8.5")
assert_exact_version("opencv-python-headless", "4.12.0.88")
assert_exact_version("rerun-sdk", "0.26.2")
assert_exact_version("transformers", "4.53.2")
assert_exact_version("peft", "0.19.1")
assert_exact_version("hf-libero", "0.1.3")

if backend == "rocm":
    assert_exact_version("torch", os.environ.get("PI05_EXPECTED_TORCH_VERSION", "2.12.0+rocm7.2"))
    assert_exact_version(
        "torchvision",
        os.environ.get("PI05_EXPECTED_TORCHVISION_VERSION", "0.27.0+rocm7.2"),
    )
    assert_exact_version(
        "torchaudio",
        os.environ.get("PI05_EXPECTED_TORCHAUDIO_VERSION", "2.11.0+rocm7.2"),
    )
elif backend == "cuda":
    assert_exact_version("torch", os.environ.get("PI05_EXPECTED_TORCH_VERSION", "2.11.0+cu128"))
    assert_exact_version(
        "torchvision",
        os.environ.get("PI05_EXPECTED_TORCHVISION_VERSION", "0.26.0+cu128"),
    )
    assert_exact_version(
        "torchaudio",
        os.environ.get("PI05_EXPECTED_TORCHAUDIO_VERSION", "2.11.0+cu128"),
    )

package_names = [
    "torch",
    "torchvision",
    "torchaudio",
    "lerobot",
    "datasets",
    "numpy",
    "pyarrow",
    "opencv-python-headless",
    "rerun-sdk",
    "transformers",
    "peft",
    "hf-libero",
    "robosuite",
]
packages = {
    pkg: {
        "version": md.version(pkg),
        "location": str(Path(md.distribution(pkg).locate_file("")).resolve()),
    }
    for pkg in package_names
}
gpu_devices = []
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        gpu_devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory": int(properties.total_memory),
                "major": int(properties.major),
                "minor": int(properties.minor),
            }
        )

receipt = {
    "schema_version": 1,
    "kind": "vla_lens.pi05_capture_environment_receipt",
    "status": "pass",
    "backend": backend,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "runtime": {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
    },
    "packages": packages,
    "accelerator": {
        "torch_version": torch.__version__,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "rocm_runtime": getattr(torch.version, "hip", None),
        "device_available": bool(torch.cuda.is_available()),
        "devices": gpu_devices,
    },
    "openpi_transformers": {
        "replacement_check": True,
        "check_module": str(openpi_check_path),
        "check_module_sha256": "sha256:"
        + hashlib.sha256(openpi_check_path.read_bytes()).hexdigest(),
    },
    "libero": {
        "module": str(Path(libero.__file__).resolve()),
        "config_path": str(libero_config_file.resolve()),
        "config_sha256": "sha256:"
        + hashlib.sha256(libero_config_file.read_bytes()).hexdigest(),
        "paths": {
            key: str(Path(str(libero_config[key])).expanduser().resolve())
            for key in ("benchmark_root", "bddl_files", "init_states")
        },
    },
}
encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
if receipt_path:
    Path(receipt_path).write_text(encoded, encoding="utf-8")
if json_stdout:
    print(encoded, end="")
else:
    print(f"PI0.5 {backend} capture environment OK")
    for pkg in package_names:
        print(f"  {pkg}: {packages[pkg]['version']}")
    if receipt_path:
        print(f"  receipt: {receipt_path}")
PY
