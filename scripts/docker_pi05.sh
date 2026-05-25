#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="${PI05_BACKEND:-rocm}"
BUILD=1
ARGS=()

usage() {
  cat <<'EOF'
Usage:
  scripts/docker_pi05.sh --backend rocm|cuda [--no-build] [batch args...]
  scripts/docker_pi05.sh --backend rocm|cuda [--no-build] capture [single-capture args...]
  scripts/docker_pi05.sh --backend rocm|cuda [--no-build] check

Output paths:
  Relative output roots write under the container workdir. The default configs use
  runs/..., and the wrapper mounts $VLA_LENS_RUNS_DIR or ./runs at /app/runs.

  Absolute --output-root or --vlatrace-out-root values are treated as host paths,
  mounted at /capture-output, and rewritten for the command inside the container.
  Absolute output_root values inside --config YAML files are handled the same way
  unless an explicit CLI output root override is provided.
  The --vlatrace-out-root flag name is legacy; capture now writes LeRobot v3
  dataset roots plus vla_lens/ overlays.

Examples:
  scripts/docker_pi05.sh --backend rocm --config configs/pi05_light_5_test.yaml --run
  scripts/docker_pi05.sh --backend rocm --config configs/pi05_light_5_test.yaml --output-root /mnt/nvme/pi05-light-5-test --run
  scripts/docker_pi05.sh --backend rocm capture --vlatrace-out-root /mnt/nvme/smoke --episodes 1 ...

Useful environment:
  VLA_LENS_RUNS_DIR       host directory mounted at /app/runs
  VLA_LENS_HF_CACHE_DIR   host Hugging Face cache mounted in the container
  VLA_LENS_LIBERO_CACHE_DIR host LIBERO cache mounted in the container
  HF_TOKEN                forwarded when set
  PI05_STRICT_DEVICE_CHECK=1 verifies visible GPU access during `check`
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
    --no-build)
      BUILD=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      ARGS+=("$@")
      break
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

case "$BACKEND" in
  rocm|cuda) ;;
  *)
    echo "Unsupported Docker PI0.5 backend: $BACKEND" >&2
    usage >&2
    exit 2
    ;;
esac

image_for_backend() {
  case "$1" in
    rocm) echo "${VLA_LENS_PI05_ROCM_IMAGE:-vla-lens-pi05-rocm:local}" ;;
    cuda) echo "${VLA_LENS_PI05_CUDA_IMAGE:-vla-lens-pi05-cuda:local}" ;;
  esac
}

dockerfile_for_backend() {
  case "$1" in
    rocm) echo "docker/capture.rocm.Dockerfile" ;;
    cuda) echo "docker/capture.cuda.Dockerfile" ;;
  esac
}

IMAGE="${VLA_LENS_PI05_IMAGE:-$(image_for_backend "$BACKEND")}"
DOCKERFILE="$(dockerfile_for_backend "$BACKEND")"
RUNS_DIR="${VLA_LENS_RUNS_DIR:-$ROOT/runs}"
HF_CACHE_DIR="${VLA_LENS_HF_CACHE_DIR:-$HOME/.cache/huggingface}"
LIBERO_CACHE_DIR="${VLA_LENS_LIBERO_CACHE_DIR:-$HOME/.cache/libero}"
HOST_OUTPUT_ROOT=""
CONTAINER_OUTPUT_ROOT="/capture-output"
CONTAINER_INPUT_ROOT="/host-inputs"
CONTAINER_PATH_RESULT=""
CONFIG_HOST_PATH=""
SAW_OUTPUT_ROOT_FLAG=0
INPUT_MOUNT_INDEX=0
INPUT_MOUNT_FLAGS=()

container_path_for_known_host_mount() {
  local host_path="$1"
  local configs_root runs_root relative
  configs_root="$(realpath "$ROOT/configs")"
  runs_root="$(realpath "$RUNS_DIR")"

  if [[ "$host_path" == "$configs_root" ]]; then
    CONTAINER_PATH_RESULT="/app/configs"
    return 0
  fi
  if [[ "$host_path" == "$configs_root/"* ]]; then
    relative="${host_path#"$configs_root"/}"
    CONTAINER_PATH_RESULT="/app/configs/$relative"
    return 0
  fi
  if [[ "$host_path" == "$runs_root" ]]; then
    CONTAINER_PATH_RESULT="/app/runs"
    return 0
  fi
  if [[ "$host_path" == "$runs_root/"* ]]; then
    relative="${host_path#"$runs_root"/}"
    CONTAINER_PATH_RESULT="/app/runs/$relative"
    return 0
  fi
  return 1
}

mount_readonly_input_path() {
  local host_path="$1"
  local host_dir base container_dir
  host_dir="$(dirname "$host_path")"
  base="$(basename "$host_path")"
  container_dir="$CONTAINER_INPUT_ROOT/$INPUT_MOUNT_INDEX"
  INPUT_MOUNT_FLAGS+=(-v "$host_dir:$container_dir:ro")
  INPUT_MOUNT_INDEX=$((INPUT_MOUNT_INDEX + 1))
  CONTAINER_PATH_RESULT="$container_dir/$base"
}

container_input_path() {
  local value="$1"
  local host_candidate host_path
  if [[ "$value" == /* ]]; then
    if [[ ! -e "$value" ]]; then
      echo "Input path does not exist on host: $value" >&2
      exit 2
    fi
    host_path="$(realpath "$value")"
    if container_path_for_known_host_mount "$host_path"; then
      return
    fi
    mount_readonly_input_path "$host_path"
    return
  fi

  host_candidate="$ROOT/$value"
  if [[ -e "$host_candidate" ]]; then
    host_path="$(realpath "$host_candidate")"
    if container_path_for_known_host_mount "$host_path"; then
      return
    fi
    mount_readonly_input_path "$host_path"
    return
  fi

  CONTAINER_PATH_RESULT="$value"
}

remember_config_host_path() {
  local value="$1"
  local host_candidate
  if [[ "$value" == /* ]]; then
    host_candidate="$value"
  else
    host_candidate="$ROOT/$value"
  fi
  if [[ -f "$host_candidate" ]]; then
    CONFIG_HOST_PATH="$(realpath "$host_candidate")"
  fi
}

container_output_path() {
  local value="$1"
  if [[ "$value" != /* ]]; then
    CONTAINER_PATH_RESULT="$value"
    return
  fi
  local host_path="$value"
  mkdir -p "$host_path"
  host_path="$(realpath "$host_path")"
  if [[ -n "$HOST_OUTPUT_ROOT" && "$HOST_OUTPUT_ROOT" != "$host_path" ]]; then
    cat >&2 <<EOF
Only one absolute output root can be mounted per container run.
First:  $HOST_OUTPUT_ROOT
Second: $host_path
EOF
    exit 2
  fi
  HOST_OUTPUT_ROOT="$host_path"
  CONTAINER_PATH_RESULT="$CONTAINER_OUTPUT_ROOT"
}

rewrite_output_args() {
  local -n input_args=$1
  local -n output_args=$2
  local index item flag value
  output_args=()
  for ((index = 0; index < ${#input_args[@]}; index++)); do
    item="${input_args[$index]}"
    case "$item" in
      --output-root|--vlatrace-out-root)
        if (( index + 1 >= ${#input_args[@]} )); then
          echo "Missing value for $item" >&2
          exit 2
        fi
        SAW_OUTPUT_ROOT_FLAG=1
        container_output_path "${input_args[$((index + 1))]}"
        output_args+=("$item" "$CONTAINER_PATH_RESULT")
        index=$((index + 1))
        ;;
      --output-root=*|--vlatrace-out-root=*)
        SAW_OUTPUT_ROOT_FLAG=1
        flag="${item%%=*}"
        value="${item#*=}"
        container_output_path "$value"
        output_args+=("$flag=$CONTAINER_PATH_RESULT")
        ;;
      *)
        output_args+=("$item")
        ;;
    esac
  done
}

rewrite_input_args() {
  local -n input_args=$1
  local -n output_args=$2
  local index item flag value
  output_args=()
  for ((index = 0; index < ${#input_args[@]}; index++)); do
    item="${input_args[$index]}"
    case "$item" in
      --config|--episode-plan)
        if (( index + 1 >= ${#input_args[@]} )); then
          echo "Missing value for $item" >&2
          exit 2
        fi
        value="${input_args[$((index + 1))]}"
        if [[ "$item" == "--config" ]]; then
          remember_config_host_path "$value"
        fi
        container_input_path "$value"
        output_args+=("$item" "$CONTAINER_PATH_RESULT")
        index=$((index + 1))
        ;;
      --config=*|--episode-plan=*)
        flag="${item%%=*}"
        value="${item#*=}"
        if [[ "$flag" == "--config" ]]; then
          remember_config_host_path "$value"
        fi
        container_input_path "$value"
        output_args+=("$flag=$CONTAINER_PATH_RESULT")
        ;;
      *)
        output_args+=("$item")
        ;;
    esac
  done
}

read_config_output_root() {
  local config_path="$1"
  python3 - "$config_path" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

config = Path(sys.argv[1])
pattern = re.compile(r"^output_root\s*:\s*(.*)$")
for raw_line in config.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    match = pattern.match(line)
    if match is None:
        continue
    value = match.group(1).split("#", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    print(value)
    break
PY
}

append_config_output_override() {
  local output_root
  if [[ "$SAW_OUTPUT_ROOT_FLAG" == "1" || -z "$CONFIG_HOST_PATH" ]]; then
    return
  fi
  output_root="$(read_config_output_root "$CONFIG_HOST_PATH")"
  if [[ -z "$output_root" || "$output_root" != /* ]]; then
    return
  fi
  container_output_path "$output_root"
  REWRITTEN_ARGS+=("--output-root" "$CONTAINER_PATH_RESULT")
}

cd "$ROOT"
mkdir -p "$RUNS_DIR" "$HF_CACHE_DIR" "$LIBERO_CACHE_DIR"

REWRITTEN_ARGS=()
PATH_REWRITTEN_ARGS=()
rewrite_input_args ARGS PATH_REWRITTEN_ARGS
rewrite_output_args PATH_REWRITTEN_ARGS REWRITTEN_ARGS
append_config_output_override

if [[ "$BUILD" == "1" ]]; then
  docker build -f "$DOCKERFILE" -t "$IMAGE" .
fi

TTY_FLAGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_FLAGS=(-it)
elif [[ ! -t 0 ]]; then
  TTY_FLAGS=(-i)
fi

ROCM_GROUP_FLAGS=()
add_device_group() {
  local path="$1"
  local gid existing
  [[ -e "$path" ]] || return
  gid="$(stat -c "%g" "$path")"
  for existing in "${ROCM_GROUP_FLAGS[@]}"; do
    if [[ "$existing" == "$gid" ]]; then
      return
    fi
  done
  ROCM_GROUP_FLAGS+=(--group-add "$gid")
}

ACCELERATOR_FLAGS=()
case "$BACKEND" in
  rocm)
    add_device_group /dev/kfd
    for path in /dev/dri/*; do
      add_device_group "$path"
    done
    ACCELERATOR_FLAGS=(
      --device=/dev/kfd
      --device=/dev/dri
      "${ROCM_GROUP_FLAGS[@]}"
      --ipc=host
      --shm-size=16g
      --security-opt seccomp=unconfined
    )
    ;;
  cuda)
    ACCELERATOR_FLAGS=(
      --gpus all
      --ipc=host
      --shm-size=16g
    )
    ;;
esac

ENV_FLAGS=()
pass_env_if_set() {
  local name="$1"
  if [[ -n "${!name+x}" ]]; then
    ENV_FLAGS+=("-e" "$name=${!name}")
  fi
}

for name in \
  PI05_STRICT_DEVICE_CHECK \
  VLA_LENS_CAPTURE_DEVICE \
  VLA_LENS_CAPTURE_DTYPE \
  HF_TOKEN \
  HUGGING_FACE_HUB_TOKEN \
  HF_HUB_ENABLE_HF_TRANSFER \
  WANDB_API_KEY \
  WANDB_MODE \
  HIP_VISIBLE_DEVICES \
  ROCR_VISIBLE_DEVICES \
  HSA_OVERRIDE_GFX_VERSION \
  CUDA_VISIBLE_DEVICES \
  NVIDIA_DRIVER_CAPABILITIES \
  PYTORCH_HIP_ALLOC_CONF \
  PYTORCH_CUDA_ALLOC_CONF
do
  pass_env_if_set "$name"
done
if [[ "$BACKEND" == "cuda" && -z "${NVIDIA_DRIVER_CAPABILITIES+x}" ]]; then
  ENV_FLAGS+=("-e" "NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics")
fi

VOLUME_FLAGS=(
  -v "$RUNS_DIR:/app/runs"
  -v "$ROOT/configs:/app/configs:ro"
  -v "$HF_CACHE_DIR:/root/.cache/huggingface"
  -v "$LIBERO_CACHE_DIR:/root/.cache/libero"
  "${INPUT_MOUNT_FLAGS[@]}"
)

if [[ -n "$HOST_OUTPUT_ROOT" ]]; then
  VOLUME_FLAGS+=(-v "$HOST_OUTPUT_ROOT:$CONTAINER_OUTPUT_ROOT")
fi

exec docker run --rm "${TTY_FLAGS[@]}" \
  "${ACCELERATOR_FLAGS[@]}" \
  "${ENV_FLAGS[@]}" \
  "${VOLUME_FLAGS[@]}" \
  "$IMAGE" "${REWRITTEN_ARGS[@]}"
