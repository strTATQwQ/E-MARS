#!/usr/bin/env bash
set -eo pipefail

VARIANT="${1:?variant required}"
ATTEMPT="${2:?attempt required}"
MAX_STEP="${3:-4000}"
CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
LOG="$CONTROL_ROOT/results/internnav_t4/t4_2_r3/${VARIANT}_${ATTEMPT}_launcher.log"
mkdir -p "$(dirname "$LOG")"
nohup env INTERNVLA_T4_R3_SMOKE_MAX_STEP="$MAX_STEP" \
  bash "$CONTROL_ROOT/scripts/run_t4_2_r3_sensor_smoke.sh" \
  "$VARIANT" "$ATTEMPT" 0 >"$LOG" 2>&1 </dev/null &
echo "$!"
