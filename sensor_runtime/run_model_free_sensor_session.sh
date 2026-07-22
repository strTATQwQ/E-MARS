#!/usr/bin/env bash
set -euo pipefail

PROFILE=""
RESULT_DIR=""
GRANT_ID=""
while test "$#" -gt 0; do
  case "$1" in
    --profile) PROFILE="${2:-}"; shift 2 ;;
    --result-dir) RESULT_DIR="${2:-}"; shift 2 ;;
    --grant-id) GRANT_ID="${2:-}"; shift 2 ;;
    *) echo "unknown model-free session argument: $1" >&2; exit 2 ;;
  esac
done
case "$PROFILE" in
  bootstrap|soak|completion_sim|completion_sim_map) ;;
  *) echo "profile must be bootstrap, soak, completion_sim, or completion_sim_map" >&2; exit 2 ;;
esac
test -n "$RESULT_DIR"
test -n "$GRANT_ID"
test "${INTERNNAV_SENSOR_SESSION_LEASE_ACK:-0}" = 1 || {
  echo "model-free session requires the coordinator-held Isaac lease" >&2
  exit 2
}

CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export INTERNNAV_T1_CONTROL_ROOT="$CONTROL_ROOT"
export PYTHONPATH="$CONTROL_ROOT"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
exec python3 -m sensor_runtime.session --profile "$PROFILE" --result-dir "$RESULT_DIR" --grant-id "$GRANT_ID"
