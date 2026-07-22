#!/usr/bin/env bash
set -euo pipefail

VARIANT_CONFIG="${1:-}"
ATTEMPT="${2:-001}"
if test -z "$VARIANT_CONFIG" || ! test -f "$VARIANT_CONFIG"; then
  echo "usage: $0 GENERATED_VARIANT_CONFIG [attempt]" >&2
  exit 2
fi
case "$ATTEMPT" in
  *[!A-Za-z0-9._-]*|'') echo "unsafe attempt id" >&2; exit 2 ;;
esac

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
MATRIX="$CONTROL_ROOT/configs/completion_sim/ablation/frozen_matrix_v1.json"
test -f "$MATRIX"

# A worker, strict profile, or hardware target cannot turn a generated config
# into an online run by setting one loose feature flag.  The explicit offline
# dry-run below exits before any Docker, ROS, model or resource operation.
test "${INTERNNAV_RUNTIME_POLICY:-}" = "completion_sim"
test "${INTERNNAV_SIMULATION_TARGET:-}" = "isaac"

mapfile -t FIELDS < <(
  python3 "$ROOT/t4_ablation_runtime.py" fields \
    --config "$VARIANT_CONFIG" --matrix "$MATRIX"
)
test "${#FIELDS[@]}" -eq 11
VARIANT_ID="${FIELDS[0]}"
CONFIG_SHA256="${FIELDS[1]}"
MATRIX_SHA256="${FIELDS[2]}"
SYSTEM_MODE="${FIELDS[3]}"
TRAJECTORY_MODE="${FIELDS[4]}"
TERMINATION_MODE="${FIELDS[5]}"
HISTORY_MODE="${FIELDS[6]}"
RECOVERY_MODE="${FIELDS[7]}"
VIEW_MODE="${FIELDS[8]}"
COMPLETION_PROFILE_SHA256="${FIELDS[9]}"
STRICT_PROFILE_SHA256="${FIELDS[10]}"

# The model process is owned by the DGX half of the same lease.  Requiring its
# coordinator-provided config identity prevents a history arm from silently
# reusing a model server started for another arm.
test "${INTERNVLA_T4_DGX_MODEL_CONFIG_SHA256:-}" = "$CONFIG_SHA256"

export INTERNVLA_T4_VARIANT_ID="$VARIANT_ID"
export INTERNVLA_T4_VARIANT_CONFIG="$(cd -- "$(dirname -- "$VARIANT_CONFIG")" && pwd -P)/$(basename -- "$VARIANT_CONFIG")"
export INTERNVLA_T4_VARIANT_CONFIG_SHA256="$CONFIG_SHA256"
export INTERNVLA_T4_MATRIX_SHA256="$MATRIX_SHA256"
export INTERNVLA_T4_COMPLETION_PROFILE_SHA256="$COMPLETION_PROFILE_SHA256"
export INTERNVLA_T4_STRICT_PROFILE_SHA256="$STRICT_PROFILE_SHA256"
export INTERNVLA_T4_SYSTEM_MODE="$SYSTEM_MODE"
export INTERNVLA_T4_TRAJECTORY_MODE="$TRAJECTORY_MODE"
export INTERNVLA_T4_TERMINATION_MODE="$TERMINATION_MODE"
export INTERNVLA_T4_HISTORY_MODE="$HISTORY_MODE"
export INTERNVLA_T4_REQUIRED_HISTORY_MODE="$HISTORY_MODE"
export INTERNVLA_T4_RECOVERY_MODE="$RECOVERY_MODE"
export INTERNVLA_T4_VIEW_MODE="$VIEW_MODE"
export INTERNVLA_T4_ENABLE_RECOVERY=0
test "$RECOVERY_MODE" != on || export INTERNVLA_T4_ENABLE_RECOVERY=1
export INTERNVLA_T4_USE_T4_ADAPTER=1
export INTERNVLA_T4_COMPLETION_FAST_PATH=1
export INTERNVLA_T4_RUN_LABEL="t4_6_${VARIANT_ID}"
export INTERNVLA_T4_MIN_SR_OVERRIDE=0.0
export INTERNVLA_T4_MODEL_DATASET_ROOT="${INTERNVLA_T4_MODEL_DATASET_ROOT:-$HOME/internnav-t0/episodes/pilot}"
export INTERNVLA_T4_ABLATION_DATASET_FILE="$INTERNVLA_T4_MODEL_DATASET_ROOT/val_unseen/val_unseen.json.gz"

# H1/Go2 is a semantic-RGB-only factor.  The dedicated semantic camera can
# change, while D435i depth, LiDAR, collision and mapping geometry remain on
# the frozen Go2 values loaded by run_t4_sensor_gate.sh.
if test "$VIEW_MODE" = h1_view; then
  export INTERNVLA_T4_CAMERA_HEIGHT_OVERRIDE_M=1.25
  export INTERNVLA_T4_CAMERA_PITCH_OVERRIDE_DEG=30
  export INTERNVLA_T4_CAMERA_HFOV_OVERRIDE_DEG=90
  export INTERNVLA_T4_CAMERA_VFOV_OVERRIDE_DEG=67.5
else
  unset INTERNVLA_T4_CAMERA_HEIGHT_OVERRIDE_M
  unset INTERNVLA_T4_CAMERA_PITCH_OVERRIDE_DEG
  unset INTERNVLA_T4_CAMERA_HFOV_OVERRIDE_DEG
  unset INTERNVLA_T4_CAMERA_VFOV_OVERRIDE_DEG
fi

if test "${INTERNNAV_T4_OFFLINE_DRY_RUN:-0}" = 1; then
  python3 - <<'PY'
import json, os
print(json.dumps({
    "schema_version": 1,
    "status": "OFFLINE_DRY_RUN_PASS",
    "online_execution": False,
    "resource_use": "none",
    "runtime_policy": os.environ["INTERNNAV_RUNTIME_POLICY"],
    "runtime_target": "isaac_simulation",
    "variant_id": os.environ["INTERNVLA_T4_VARIANT_ID"],
    "config_sha256": os.environ["INTERNVLA_T4_VARIANT_CONFIG_SHA256"],
    "matrix_sha256": os.environ["INTERNVLA_T4_MATRIX_SHA256"],
    "factors": {
        "system_mode": os.environ["INTERNVLA_T4_SYSTEM_MODE"],
        "trajectory_mode": os.environ["INTERNVLA_T4_TRAJECTORY_MODE"],
        "termination_mode": os.environ["INTERNVLA_T4_TERMINATION_MODE"],
        "history_mode": os.environ["INTERNVLA_T4_HISTORY_MODE"],
        "recovery_mode": os.environ["INTERNVLA_T4_RECOVERY_MODE"],
        "view_mode": os.environ["INTERNVLA_T4_VIEW_MODE"],
    },
    "functional_fast_path": {
        "map": "static_global_plus_lidar_local",
        "pose": "isaac_ground_truth_navigation_odometry",
        "nvblox": "shadow_or_deferred",
    },
    "model_host": "dgx_spark_only",
}, indent=2, sort_keys=True))
PY
  exit 0
fi

# The real launcher must sit inside the coordinator's combined DGX->Isaac
# lease.  Dry-run cannot be promoted by falling through this boundary.
test "${INTERNNAV_T4_RESOURCE_LEASE_ACK:-}" = "dgx+isaac"

exec bash "$ROOT/run_t4_sensor_gate.sh" t4_4 model "$ATTEMPT"
