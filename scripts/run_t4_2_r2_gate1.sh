#!/usr/bin/env bash
set -eo pipefail

CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
ATTEMPT="${1:-001}"
RUN_LABEL="t4_2_r2_gate1_smoke"
DATASET="$CONTROL_ROOT/episodes/t4_2_obstacle_smoke_one"
RESULT_DIR="$CONTROL_ROOT/results/internnav_t4/${RUN_LABEL}_oracle_attempt_${ATTEMPT}"
OUTER_LOG="$CONTROL_ROOT/runtime/${RUN_LABEL}_oracle_${ATTEMPT}.outer.log"
PID_FILE="$CONTROL_ROOT/runtime/${RUN_LABEL}_oracle_${ATTEMPT}.pid"

test -d "$DATASET/val_unseen"
test ! -e "$RESULT_DIR"
test ! -e "$OUTER_LOG"
rm -f "$PID_FILE"

cd "$CONTROL_ROOT"
nohup setsid env \
  INTERNVLA_T4_TRACE_COSTMAP_STAGES=1 \
  INTERNVLA_T4_RUN_LABEL="$RUN_LABEL" \
  INTERNVLA_T4_EXPECTED_COUNT=1 \
  INTERNVLA_T4_MIN_SR_OVERRIDE=0 \
  INTERNVLA_T4_MAX_STEP=4000 \
  INTERNVLA_T4_ORACLE_DATASET_ROOT="$DATASET" \
  INTERNVLA_T3_OBSTACLE_ORACLE_DATASET_ROOT="$DATASET" \
  bash scripts/run_t4_nvblox_gt_pose.sh oracle "$ATTEMPT" \
  >"$OUTER_LOG" 2>&1 </dev/null &
echo "$!" >"$PID_FILE"
cat "$PID_FILE"
