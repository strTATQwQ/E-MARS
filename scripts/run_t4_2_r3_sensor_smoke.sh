#!/usr/bin/env bash
set -eo pipefail

VARIANT="${1:-}"
ATTEMPT="${2:-001}"
ENFORCE_GATE2="${3:-0}"
case "$VARIANT" in
  d435i) ENABLE_D435I=1; ENABLE_LIDAR=0; PRE_LAYER=nvblox_layer ;;
  lidar) ENABLE_D435I=0; ENABLE_LIDAR=1; PRE_LAYER=nvblox_layer ;;
  fused) ENABLE_D435I=1; ENABLE_LIDAR=1; PRE_LAYER=nvblox_layer ;;
  fused_clearing) ENABLE_D435I=1; ENABLE_LIDAR=1; PRE_LAYER=footprint_clearing_layer ;;
  fused_clearing_diag_timeout) ENABLE_D435I=1; ENABLE_LIDAR=1; PRE_LAYER=footprint_clearing_layer ;;
  *) echo "usage: $0 {d435i|lidar|fused|fused_clearing|fused_clearing_diag_timeout} [attempt] [enforce_gate2=0|1]" >&2; exit 2 ;;
esac

CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
RESULT_ROOT="$CONTROL_ROOT/results/internnav_t4/t4_2_r3/gate1"
DATASET="$CONTROL_ROOT/episodes/t4_2_obstacle_smoke_one"
test -d "$DATASET"
test -f "$CONTROL_ROOT/configs/internnav_t4/t4_2_r3_${VARIANT}.yaml"
test -f "$CONTROL_ROOT/configs/internnav_t4/t4_2_r3_${VARIANT}_nvblox.yaml"

export INTERNVLA_T4_R3_ENABLE_SENSOR_BRIDGE=1
export INTERNVLA_T4_R3_ENABLE_D435I="$ENABLE_D435I"
export INTERNVLA_T4_R3_ENABLE_LIDAR="$ENABLE_LIDAR"
export INTERNVLA_T4_DEPTH_STRIDE_OVERRIDE=10
export INTERNVLA_T4_PRE_INFLATION_LAYER_NAME="$PRE_LAYER"
export INTERNVLA_T4_TRACE_COSTMAP_STAGES=1
export INTERNVLA_T4_EXPECTED_COUNT=1
export INTERNVLA_T4_MIN_SR_OVERRIDE=0.0
export INTERNVLA_T4_MAX_STEP="${INTERNVLA_T4_R3_SMOKE_MAX_STEP:-16000}"
export INTERNVLA_T4_ORACLE_DATASET_ROOT="$DATASET"
export INTERNVLA_T3_OBSTACLE_ORACLE_DATASET_ROOT="$DATASET"
export INTERNVLA_T4_NAV2_PARAMS_OVERRIDE="$CONTROL_ROOT/configs/internnav_t4/t4_2_r3_${VARIANT}.yaml"
export INTERNVLA_T4_NVBLOX_PARAMS_OVERRIDE="$CONTROL_ROOT/configs/internnav_t4/t4_2_r3_${VARIANT}_nvblox.yaml"
export INTERNVLA_T4_GO2_USD_BUILDER="$CONTROL_ROOT/scripts/build_t4_r3_go2_usd.py"
export INTERNVLA_T4_RUNTIME_OVERLAY_BUILDER="$CONTROL_ROOT/scripts/build_t4_r3_sensor_runtime_overlay.py"
export INTERNVLA_T4_RESULT_ROOT="$RESULT_ROOT"
export INTERNVLA_T4_RUN_LABEL="t4_2_r3_${VARIANT}_smoke"

bash "$CONTROL_ROOT/scripts/run_t4_sensor_gate.sh" t4_2 oracle "$ATTEMPT"

RESULT_DIR="$RESULT_ROOT/t4_2_r3_${VARIANT}_smoke_oracle_attempt_${ATTEMPT}"
if test "$ENFORCE_GATE2" = 1; then
  test "$VARIANT" = fused_clearing || test "$VARIANT" = fused_clearing_diag_timeout
  python3 "$CONTROL_ROOT/scripts/validate_t4_2_r3_smoke.py" \
    "$RESULT_DIR" --write-gate-marker
fi
