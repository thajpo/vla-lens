#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR=""
MAX_RESTARTS="${PI05_CAPTURE_MAX_RESTARTS:-20}"
RESTART_DELAY_SECONDS="${PI05_CAPTURE_RESTART_DELAY_SECONDS:-15}"
RECYCLE_EXIT_CODE="${PI05_CAPTURE_RECYCLE_EXIT_CODE:-75}"
ATTEMPT=0

for ((i = 1; i <= $#; i++)); do
  arg="${!i}"
  if [[ "$arg" == "--log-dir" ]]; then
    next=$((i + 1))
    LOG_DIR="${!next}"
  fi
done

if [[ -z "$LOG_DIR" ]]; then
  echo "Usage: $0 --log-dir LOG_DIR -- [pi05_plan_capture_rocm.sh args...]" >&2
  exit 2
fi

ARGS=()
SEEN_SEPARATOR=0
for arg in "$@"; do
  if [[ "$SEEN_SEPARATOR" == "0" ]]; then
    if [[ "$arg" == "--" ]]; then
      SEEN_SEPARATOR=1
    fi
    continue
  fi
  ARGS+=("$arg")
done

if [[ "$SEEN_SEPARATOR" == "0" || "${#ARGS[@]}" == "0" ]]; then
  echo "Usage: $0 --log-dir LOG_DIR -- [pi05_plan_capture_rocm.sh args...]" >&2
  exit 2
fi

mkdir -p "$LOG_DIR"
SUPERVISOR_LOG="$LOG_DIR/supervisor.log"

while true; do
  ATTEMPT=$((ATTEMPT + 1))
  ATTEMPT_LOG="$LOG_DIR/plan_capture_attempt_${ATTEMPT}.log"
  {
    echo "== $(date --iso-8601=seconds) attempt=$ATTEMPT start"
    echo "cmd: scripts/pi05_plan_capture_rocm.sh ${ARGS[*]}"
  } | tee -a "$SUPERVISOR_LOG"

  set +e
  "$ROOT/scripts/pi05_plan_capture_rocm.sh" "${ARGS[@]}" >"$ATTEMPT_LOG" 2>&1
  status=$?
  set -e

  {
    echo "== $(date --iso-8601=seconds) attempt=$ATTEMPT exit=$status log=$ATTEMPT_LOG"
  } | tee -a "$SUPERVISOR_LOG"

  if [[ "$status" == "0" ]]; then
    echo "capture completed" | tee -a "$SUPERVISOR_LOG"
    exit 0
  fi

  if [[ "$status" != "$RECYCLE_EXIT_CODE" ]]; then
    echo "capture failed with non-recycle exit=$status; not restarting" | tee -a "$SUPERVISOR_LOG"
    exit "$status"
  fi

  if ((ATTEMPT >= MAX_RESTARTS)); then
    echo "capture failed after $ATTEMPT attempts; last exit=$status" | tee -a "$SUPERVISOR_LOG"
    exit "$status"
  fi

  echo "recycling after exit=$status in ${RESTART_DELAY_SECONDS}s" | tee -a "$SUPERVISOR_LOG"
  sleep "$RESTART_DELAY_SECONDS"
done
