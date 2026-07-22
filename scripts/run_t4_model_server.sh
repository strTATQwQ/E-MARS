#!/usr/bin/env bash
set -eo pipefail

# Run on the DGX Spark. This keeps the frozen model/checkpoint loader while
# adding only T4 cache invalidation and the explicit history ablation switch.
CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
ROS_WS="${INTERNVLA_ROS_WS:-$CONTROL_ROOT/ros_ws}"
INTERNNAV_ROOT="${INTERNNAV_ROOT:-$HOME/internnav-t0/InternNav}"
MODEL_PYTHON="${INTERNVLA_MODEL_PYTHON:-$HOME/internnav-t0/venv-model/bin/python}"
RESULT_DIR="${INTERNVLA_MODEL_RESULT_DIR:-$CONTROL_ROOT/results/internnav_t4/t4_model}"
ROS_DISTRO_NAME="${INTERNVLA_ROS_DISTRO:-jazzy}"
VARIANT_CONFIG="${INTERNVLA_T4_VARIANT_CONFIG:-}"
FUNCTIONAL_MODEL="${INTERNVLA_T4_FUNCTIONAL_MODEL:-0}"
case "$FUNCTIONAL_MODEL" in 0|1) ;; *) echo "invalid functional model mode" >&2; exit 2 ;; esac
test -f "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
test -f "$ROS_WS/install/setup.bash"
test -x "$MODEL_PYTHON"
test -f "$CONTROL_ROOT/internvla_t4_recovery/internvla_t4_recovery/model_node.py"
if test "$FUNCTIONAL_MODEL" = 1; then
  test -z "$VARIANT_CONFIG"
  test "${INTERNNAV_T4_RESOURCE_LEASE_ACK:-}" = "dgx+isaac"
  test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim
  test "${INTERNNAV_SIMULATION_TARGET:-}" = isaac
  test "${INTERNVLA_BACKEND:-real}" = real
  test "${INTERNVLA_PRELOAD_MODEL:-1}" = 1
  test ! -e "$RESULT_DIR"
elif test -n "$VARIANT_CONFIG"; then
  test "${INTERNNAV_T4_RESOURCE_LEASE_ACK:-}" = "dgx+isaac"
  test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim
  test "${INTERNNAV_SIMULATION_TARGET:-}" = isaac
  test -f "$VARIANT_CONFIG"
  test ! -e "$RESULT_DIR"
fi
mkdir -p "$RESULT_DIR"
set +u
source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
source "$ROS_WS/install/setup.bash"
set -u
export INTERNNAV_ROOT
export INTERNVLA_BACKEND="${INTERNVLA_BACKEND:-real}"
export INTERNVLA_PRELOAD_MODEL="${INTERNVLA_PRELOAD_MODEL:-1}"
if test -n "$VARIANT_CONFIG"; then
  mapfile -t FIELDS < <(
    python3 "$CONTROL_ROOT/scripts/t4_ablation_runtime.py" fields \
      --config "$VARIANT_CONFIG" \
      --matrix "$CONTROL_ROOT/configs/completion_sim/ablation/frozen_matrix_v1.json"
  )
  test "${#FIELDS[@]}" -eq 11
  export INTERNVLA_T4_VARIANT_ID="${FIELDS[0]}"
  export INTERNVLA_T4_VARIANT_CONFIG_SHA256="${FIELDS[1]}"
  export INTERNVLA_T4_MATRIX_SHA256="${FIELDS[2]}"
  export INTERNVLA_T4_SYSTEM_MODE="${FIELDS[3]}"
  export INTERNVLA_T4_TRAJECTORY_MODE="${FIELDS[4]}"
  export INTERNVLA_T4_TERMINATION_MODE="${FIELDS[5]}"
  export INTERNVLA_T4_HISTORY_MODE="${FIELDS[6]}"
  export INTERNVLA_T4_RECOVERY_MODE="${FIELDS[7]}"
  export INTERNVLA_T4_VIEW_MODE="${FIELDS[8]}"
  if test "$INTERNVLA_T4_RECOVERY_MODE" = on; then
    ros2 interface show internvla_ros2_msgs/srv/RecoveryControl >/dev/null
    python3 -c "from internvla_ros2_msgs.srv import RecoveryControl"
  fi
  python3 "$CONTROL_ROOT/scripts/t4_ablation_runtime.py" validate \
    --config "$VARIANT_CONFIG" \
    --matrix "$CONTROL_ROOT/configs/completion_sim/ablation/frozen_matrix_v1.json" \
    >"$RESULT_DIR/model_variant_claim.json"
else
  export INTERNVLA_T4_VARIANT_ID=none
  export INTERNVLA_T4_VARIANT_CONFIG_SHA256=none
  export INTERNVLA_T4_HISTORY_MODE="${INTERNVLA_T4_HISTORY_MODE:-on}"
fi
export INTERNVLA_T4_RECOVERY_MODEL_AUDIT="$RESULT_DIR/model_recovery_audit.jsonl"
export PYTHONPATH="$CONTROL_ROOT/internvla_t4_recovery:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$INTERNNAV_ROOT"
exec "$MODEL_PYTHON" -m internvla_t4_recovery.model_node "$@" \
  2>&1 | tee "$RESULT_DIR/model_node.log"
