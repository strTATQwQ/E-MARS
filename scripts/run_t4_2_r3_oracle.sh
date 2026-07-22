#!/usr/bin/env bash
set -eo pipefail

SPLIT="${1:-}"
ATTEMPT="${2:-001}"
CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
MARKER="$CONTROL_ROOT/results/internnav_t4/t4_2_r3/gate2_pass.json"
test -f "$MARKER" || { echo "BLOCKED: repaired smoke Gate 2 has not passed" >&2; exit 3; }
case "$SPLIT" in
  frozen)
    DATASET="$CONTROL_ROOT/episodes/t3_obstacle_oracle_final_v2"
    EXPECTED=10
    MIN_SR=0.9
    ;;
  validation)
    DATASET="$CONTROL_ROOT/episodes/t4_2_r3_oracle_validation"
    EXPECTED=10
    MIN_SR=0.8
    ;;
  *) echo "usage: $0 {frozen|validation} [attempt]" >&2; exit 2 ;;
esac
test -d "$DATASET"
export INTERNVLA_T4_R3_ENABLE_SENSOR_BRIDGE=1 INTERNVLA_T4_R3_ENABLE_D435I=1 INTERNVLA_T4_R3_ENABLE_LIDAR=1
export INTERNVLA_T4_PRE_INFLATION_LAYER_NAME=footprint_clearing_layer
export INTERNVLA_T4_TRACE_COSTMAP_STAGES=1
export INTERNVLA_T4_EXPECTED_COUNT="$EXPECTED" INTERNVLA_T4_MIN_SR_OVERRIDE="$MIN_SR"
export INTERNVLA_T4_ORACLE_DATASET_ROOT="$DATASET" INTERNVLA_T3_OBSTACLE_ORACLE_DATASET_ROOT="$DATASET"
export INTERNVLA_T4_NAV2_PARAMS_OVERRIDE="$CONTROL_ROOT/configs/internnav_t4/t4_2_r3_fused_clearing.yaml"
export INTERNVLA_T4_NVBLOX_PARAMS_OVERRIDE="$CONTROL_ROOT/configs/internnav_t4/t4_2_r3_fused_clearing_nvblox.yaml"
export INTERNVLA_T4_GO2_USD_BUILDER="$CONTROL_ROOT/scripts/build_t4_r3_go2_usd.py"
export INTERNVLA_T4_RUNTIME_OVERLAY_BUILDER="$CONTROL_ROOT/scripts/build_t4_r3_sensor_runtime_overlay.py"
export INTERNVLA_T4_RESULT_ROOT="$CONTROL_ROOT/results/internnav_t4/t4_2_r3/gate3"
export INTERNVLA_T4_RUN_LABEL="t4_2_r3_oracle_${SPLIT}"
bash "$CONTROL_ROOT/scripts/run_t4_sensor_gate.sh" t4_2 oracle "$ATTEMPT"
