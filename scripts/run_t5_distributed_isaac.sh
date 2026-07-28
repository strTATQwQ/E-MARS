#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run_t5_distributed_isaac.sh LANE MODE RESULT_ROOT DATASET_ROOT

  LANE: a | b
  MODE: oracle | model

The matching lane-a/lane-b lease must already be held.  This entrypoint owns
only its paired Isaac worker; model, Nav2, localization, mapping, watchdog and
velocity control are forbidden on x86.
EOF
  exit 64
}

[[ $# -eq 4 ]] || usage
lane="$1"
mode="$2"
result_root="$3"
dataset_root="$4"
case "$mode" in oracle|model) ;; *) usage ;; esac

case "$lane" in
  a)
    lane_name=lane_a
    gpu=0
    ros_domain_id=75
    lane_namespace=/t5/lane_a
    edge_ip=10.100.100.128
    controller_port=25137
    model_client_port=25139
    oracle_port=25140
    clock_port=25141
    container=internnav_t5_isaac_a
    cpuset="${INTERNVLA_T5_LANE_A_CPUSET:-0,2,4,6,8,10,12,14,16}"
    identity_prefix='a::'
    expected_lease=lane-a
    ipc_alias=/tmp/internnav_t5_a_ipc
    lane_lock=/tmp/internnav_t5_isaac_a_runtime.lock
    ;;
  b)
    lane_name=lane_b
    gpu=1
    ros_domain_id=76
    lane_namespace=/t5/lane_b
    edge_ip=10.100.120.122
    controller_port=25138
    model_client_port=25239
    oracle_port=25240
    clock_port=25241
    container=internnav_t5_isaac_b
    cpuset="${INTERNVLA_T5_LANE_B_CPUSET:-1,3,5,7,9,11,13,15,17}"
    identity_prefix='b::'
    expected_lease=lane-b
    ipc_alias=/tmp/internnav_t5_b_ipc
    lane_lock=/tmp/internnav_t5_isaac_b_runtime.lock
    ;;
  *) usage ;;
esac
[[ "$cpuset" =~ ^[0-9,-]+$ ]] || {
  echo "Isaac Lane CPU set contains unsafe characters" >&2
  exit 64
}
if test "$lane" = a; then
  test "$cpuset" = 0,2,4,6,8,10,12,14,16
else
  test "$cpuset" = 1,3,5,7,9,11,13,15,17
fi
if test "$mode" = oracle; then
  evaluator_port="$oracle_port"
else
  evaluator_port="$model_client_port"
fi

# A lane-specific deployment root is mandatory.  A persistent owner marker
# below prevents two lanes from accidentally sharing it.
: "${INTERNNAV_T1_CONTROL_ROOT:?lane-specific INTERNNAV_T1_CONTROL_ROOT is required}"
root="$INTERNNAV_T1_CONTROL_ROOT"
ros_ws="${INTERNVLA_ROS_WS:-$root/ros_ws}"
worker_root="${INTERNVLA_T5_ISAAC_WORKER_ROOT:-$HOME/internnav-t1-t2/runtime/t5_isaac_workers}"
lane_profile_root="$worker_root/$lane"
shared_asset_lock=/tmp/internnav_t5_isaac_shared_assets.lock
container_cuda_visible_devices=0
t5_deployment_parent=/home/song/internnav-t1-t2/.t5-deployments
t5_host_repository_root=/home/song/internnav-t1-t2
t5_isaac_python_package_root="$root/internvla_ros2"

test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = "$expected_lease"
test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim
test "${INTERNNAV_SIMULATION_TARGET:-}" = isaac
engineering_canary_sec="${INTERNNAV_T5_ENGINEERING_CANARY_SEC:-0}"
engineering_canary_timebase="${INTERNNAV_T5_ENGINEERING_CANARY_TIMEBASE:-wall}"
engineering_canary_wall_watchdog_sec="${INTERNNAV_T5_ENGINEERING_CANARY_WALL_WATCHDOG_SEC:-7200}"
engineering_evaluator_restart_timeout_sec="${INTERNNAV_T5_EVALUATOR_RESTART_TIMEOUT_SEC:-180}"
[[ "$engineering_canary_sec" =~ ^[0-9]+$ ]]
case "$engineering_canary_timebase" in wall|sim) ;; *) exit 64 ;; esac
[[ "$engineering_canary_wall_watchdog_sec" =~ ^[1-9][0-9]*$ ]]
[[ "$engineering_evaluator_restart_timeout_sec" =~ ^[1-9][0-9]*$ ]]
engineering_canary_pre_ready_nav2_timeout_limit="${INTERNNAV_T5_PRE_READY_NAV2_TIMEOUT_LIMIT:-5}"
if test "$engineering_canary_sec" != 0; then
  [[ "$engineering_canary_pre_ready_nav2_timeout_limit" =~ ^[0-9]+$ ]]
  ((engineering_canary_pre_ready_nav2_timeout_limit <= 10))
  ((engineering_canary_sec >= 30 && engineering_canary_sec <= 600))
  if test "$engineering_canary_timebase" = sim; then
    test "$engineering_canary_sec" = 600
    ((engineering_canary_wall_watchdog_sec >= 3000))
    ((engineering_canary_wall_watchdog_sec <= 8400))
    ((engineering_evaluator_restart_timeout_sec >= 60))
    ((engineering_evaluator_restart_timeout_sec <= 600))
  fi
  test "${INTERNNAV_T5_ENGINEERING_CANARY_ACK:-}" = fast-path
  # The model client emits structured success/error markers.  The current
  # oracle facade has no equivalent fail-closed application marker yet.
  test "$mode" = model
  execution_profile=engineering_canary
else
  test "$engineering_canary_timebase" = wall
  execution_profile=fixed_dataset
fi
export INTERNNAV_T5_EXECUTION_PROFILE="$execution_profile"
export INTERNNAV_T5_ENGINEERING_CANARY_TIMEBASE="$engineering_canary_timebase"
final_pilot_lane="${INTERNNAV_T5_FINAL_PILOT_LANE:-off}"
case "$final_pilot_lane" in off|a|b) ;; *) exit 64 ;; esac
export INTERNNAV_T5_FINAL_PILOT_LANE="$final_pilot_lane"
fault_injection_profile="${INTERNNAV_T5_FAULT_INJECTION_PROFILE:-off}"
case "$fault_injection_profile" in
  off) ;;
  completion_sim_minimal_v1)
    test "$execution_profile" = engineering_canary
    test "$engineering_canary_sec" = 600
    test "$engineering_canary_timebase" = sim
    test "$mode" = model
    ;;
  *) echo "unsupported T5 fault injection profile: $fault_injection_profile" >&2; exit 64 ;;
esac
export INTERNNAV_T5_FAULT_INJECTION_PROFILE="$fault_injection_profile"
nvblox_mode="${INTERNNAV_T5_NVBLOX_MODE:-off}"
case "$nvblox_mode" in off|shadow|active_local_gt) ;; *) exit 64 ;; esac
if test "$nvblox_mode" = active_local_gt; then
  test "$lane" = a
  test "$mode" = oracle
  test "$execution_profile" = fixed_dataset
fi
export INTERNNAV_T5_NVBLOX_MODE="$nvblox_mode"
# Rev-C is an explicit Lane-B observational profile, never an implicit addition
# to the normal navigation path.  The live profile has no independent probe:
# the evaluator client owns the arm/capture/transport sequence so there is one
# request owner and no request-file race.
isaac_sensor_profile="${INTERNNAV_T5_ISAAC_SENSOR_PROFILE:-baseline}"
step3_live_advisor="${INTERNNAV_T5_STEP3_LIVE_ADVISOR:-0}"
step3_direct_high_level="${INTERNNAV_T5_STEP3_DIRECT_HIGH_LEVEL:-0}"
step3_timeout_advisor="${INTERNVLA_T5_STEP3_TIMEOUT_ADVISOR:-0}"
full_rgb_capture="${INTERNVLA_T5_FULL_RGB_CAPTURE:-0}"
d435_5hz_capture="${INTERNVLA_T5_D435_5HZ_CAPTURE:-0}"
if [[ -v INTERNVLA_T5_REVC_OBSERVER_ENABLE ]]; then
  echo "Rev-C observer must be selected only by the T5 sensor profile" >&2
  exit 64
fi
case "$step3_live_advisor" in 0|1) ;; *) exit 64 ;; esac
case "$step3_direct_high_level" in 0|1) ;; *) exit 64 ;; esac
case "$step3_timeout_advisor" in 0|1) ;; *) exit 64 ;; esac
case "$full_rgb_capture" in 0|1) ;; *) exit 64 ;; esac
case "$d435_5hz_capture" in 0|1) ;; *) exit 64 ;; esac
case "$isaac_sensor_profile" in
  baseline)
    test "$step3_live_advisor" = 0
    test "$step3_direct_high_level" = 0
    test "$step3_timeout_advisor" = 0
    test "${INTERNVLA_T5_REVC_ENABLE:-0}" = 0
    export INTERNVLA_T5_REVC_ENABLE=0
    export INTERNVLA_T5_REVC_OBSERVER_ENABLE=0
    ;;
  lane_b_revc_smoke)
    test "$step3_live_advisor" = 0
    test "$step3_direct_high_level" = 0
    test "$step3_timeout_advisor" = 0
    test "$lane" = b
    test "$mode" = model
    test "$execution_profile" = engineering_canary
    test "$engineering_canary_sec" = 60
    export INTERNVLA_T5_REVC_ENABLE=1
    export INTERNVLA_T5_REVC_OBSERVER_ENABLE=1
    ;;
  lane_b_revc_fixed5_capture)
    test "$step3_live_advisor" = 0
    test "$step3_direct_high_level" = 0
    test "$step3_timeout_advisor" = 0
    test "$lane" = b
    test "$mode" = model
    test "$execution_profile" = fixed_dataset
    export INTERNVLA_T5_REVC_ENABLE=1
    export INTERNVLA_T5_REVC_OBSERVER_ENABLE=0
    ;;
  lane_b_step3_live_canary)
    test "$step3_live_advisor" = 1
    test "$step3_direct_high_level" = 0
    test "$step3_timeout_advisor" = 0
    test "$lane" = b
    test "$mode" = model
    test "$execution_profile" = fixed_dataset
    test "$fault_injection_profile" = off
    export INTERNVLA_T5_REVC_ENABLE=1
    export INTERNVLA_T5_REVC_OBSERVER_ENABLE=0
    ;;
  lane_b_step3_direct_fixed5)
    test "$step3_live_advisor" = 0
    test "$step3_direct_high_level" = 1
    test "$step3_timeout_advisor" = 0
    test "$lane" = b
    test "$mode" = model
    test "$execution_profile" = fixed_dataset
    test "$fault_injection_profile" = off
    export INTERNVLA_T5_REVC_ENABLE=1
    export INTERNVLA_T5_REVC_OBSERVER_ENABLE=0
    ;;
  lane_a_step3_timeout_advisor)
    test "$step3_live_advisor" = 0
    test "$step3_direct_high_level" = 0
    test "$step3_timeout_advisor" = 1
    test "$lane" = a
    test "$mode" = model
    test "$execution_profile" = fixed_dataset
    test "$fault_injection_profile" = off
    export INTERNVLA_T5_REVC_ENABLE=1
    export INTERNVLA_T5_REVC_OBSERVER_ENABLE=0
    ;;
  dual_lane_wp03_stop_shadow)
    test "$step3_live_advisor" = 0
    test "$step3_direct_high_level" = 0
    test "$mode" = model
    test "$execution_profile" = fixed_dataset
    test "$fault_injection_profile" = off
    export INTERNVLA_T5_REVC_ENABLE=1
    export INTERNVLA_T5_REVC_OBSERVER_ENABLE=0
    ;;
  *) echo "unsupported T5 Isaac sensor profile: $isaac_sensor_profile" >&2; exit 64 ;;
esac
export INTERNNAV_T5_ISAAC_SENSOR_PROFILE="$isaac_sensor_profile"
export INTERNVLA_T5_STEP3_TIMEOUT_ADVISOR="$step3_timeout_advisor"
export INTERNVLA_T5_FULL_RGB_CAPTURE="$full_rgb_capture"
if test "$full_rgb_capture" = 1; then
  export INTERNVLA_T5_FULL_RGB_CAPTURE_ROOT="$result_root/evaluator/full_rgb_5hz"
  test ! -e "$INTERNVLA_T5_FULL_RGB_CAPTURE_ROOT"
else
  unset INTERNVLA_T5_FULL_RGB_CAPTURE_ROOT
fi
export INTERNVLA_T5_D435_5HZ_CAPTURE="$d435_5hz_capture"
if test "$d435_5hz_capture" = 1; then
  export INTERNVLA_T5_D435_5HZ_CAPTURE_ROOT="$result_root/evaluator/d435_rgb_5hz"
  test ! -e "$INTERNVLA_T5_D435_5HZ_CAPTURE_ROOT"
else
  unset INTERNVLA_T5_D435_5HZ_CAPTURE_ROOT
fi
# The promoted navigation-fast profile is usable for functional evidence.
# Other named profiles remain diagnostic-only single-Lane ablations.
rtf_ablation_profile="${INTERNNAV_T5_RTF_ABLATION_PROFILE:-navigation_fast}"
case "$rtf_ablation_profile" in
  navigation_fast)
    # The isolated ablation improved RTF by 26.9% over 1440 rays.  Reducing
    # the full sensor cadence to 2.5 Hz was slower and weakened measured yaw,
    # so keep real RGB-D/LiDAR at 5 Hz and reduce only LiDAR angular density.
    rtf_lidar_enabled=1
    rtf_lidar_ray_count=720
    rtf_depth_stride=4
    rtf_sensor_hz=5
    rtf_rendering_interval=4
    ;;
  off|baseline)
    rtf_lidar_enabled=1
    rtf_lidar_ray_count=1440
    rtf_depth_stride=4
    rtf_sensor_hz=5
    rtf_rendering_interval=4
    ;;
  lidar_off_probe)
    rtf_lidar_enabled=0
    rtf_lidar_ray_count=0
    rtf_depth_stride=4
    rtf_sensor_hz=5
    rtf_rendering_interval=4
    ;;
  lidar_720)
    rtf_lidar_enabled=1
    rtf_lidar_ray_count=720
    rtf_depth_stride=4
    rtf_sensor_hz=5
    rtf_rendering_interval=4
    ;;
  depth_stride8)
    rtf_lidar_enabled=1
    rtf_lidar_ray_count=1440
    rtf_depth_stride=8
    rtf_sensor_hz=5
    rtf_rendering_interval=4
    ;;
  sensor_2p5hz)
    rtf_lidar_enabled=1
    rtf_lidar_ray_count=1440
    rtf_depth_stride=4
    rtf_sensor_hz=2.5
    rtf_rendering_interval=8
    ;;
  *) echo "unsupported T5 RTF ablation profile: $rtf_ablation_profile" >&2; exit 64 ;;
esac
rtf_diagnostic_enabled=0
case "$rtf_ablation_profile" in
  off|navigation_fast) ;;
  *)
    rtf_diagnostic_enabled=1
    test "$execution_profile" = engineering_canary
    test "$engineering_canary_sec" = 60
    test "$engineering_canary_timebase" = wall
    test "$mode" = model
    ;;
esac
if test "$isaac_sensor_profile" != baseline; then
  case "$rtf_ablation_profile" in off|navigation_fast) ;; *) exit 64 ;; esac
fi
export INTERNNAV_T5_RTF_ABLATION_PROFILE="$rtf_ablation_profile"
strict_extension_profile="${INTERNNAV_T5_STRICT_EXTENSION_PROFILE:-off}"
case "$strict_extension_profile" in
  off)
    test "${INTERNVLA_T4_ENABLE_STEREO_ODOMETRY:-0}" = 0
    export INTERNVLA_T4_ENABLE_STEREO_ODOMETRY=0
    ;;
  cuvslam_shadow)
    test "$lane" = a
    test "$mode" = model
    test "$execution_profile" = fixed_dataset
    test "$isaac_sensor_profile" = baseline
    case "$rtf_ablation_profile" in off|navigation_fast) ;; *) exit 64 ;; esac
    export INTERNVLA_T4_ENABLE_STEREO_ODOMETRY=1
    ;;
  *) echo "unsupported T5 strict-extension profile: $strict_extension_profile" >&2; exit 64 ;;
esac
export INTERNNAV_T5_STRICT_EXTENSION_PROFILE="$strict_extension_profile"
export INTERNVLA_T4_R3_ENABLE_LIDAR="$rtf_lidar_enabled"
export INTERNVLA_T4_R3_LIDAR_RAY_COUNT="$rtf_lidar_ray_count"
export INTERNVLA_T4_DEPTH_STRIDE_OVERRIDE="$rtf_depth_stride"
# T5 completion_sim is a navigation/semantic-planning profile, not a Go2
# dynamics claim. It integrates the bounded root twist as a 20 Hz planar pose
# with held joints, collision geometry, contact reporting, estop and
# stale-command safe-stop,
# while reducing physics and real simulated RGB-D/LiDAR cadence.  These fixed
# exports are intentionally absent from every T4 and real-Go2 entrypoint.
export INTERNVLA_GO2_MOTION_PROFILE=t5_completion_planar_root_velocity
export INTERNVLA_GO2_PHYSICS_HZ=20
export INTERNVLA_GO2_CONTROL_HZ=20
export INTERNVLA_GO2_SENSOR_HZ="$rtf_sensor_hz"
export INTERNVLA_GO2_EXPECT_PHYSICS_DT=0.05
export INTERNVLA_GO2_EXPECT_RENDERING_INTERVAL="$rtf_rendering_interval"
test "$(id -un)" = song
test -z "${ROS_DOMAIN_ID:-}" || test "$ROS_DOMAIN_ID" = "$ros_domain_id"
test -z "${CUDA_VISIBLE_DEVICES:-}" || test "$CUDA_VISIBLE_DEVICES" = "$gpu"
test -z "${ROS_NAMESPACE:-}" || test "$ROS_NAMESPACE" = "$lane_namespace"
test -d "$root"
root="$(cd "$root" && pwd -P)"
test -d "$t5_deployment_parent"
t5_deployment_parent="$(cd "$t5_deployment_parent" && pwd -P)"
test -d "$t5_host_repository_root"
t5_host_repository_root="$(cd "$t5_host_repository_root" && pwd -P)"
test "$(dirname "$t5_deployment_parent")" = "$t5_host_repository_root"
case "$root" in
  "$t5_deployment_parent"/*) ;;
  *) echo "T5 Isaac root is outside the deployment parent" >&2; exit 64 ;;
esac
root_deployment_name="${root#"$t5_deployment_parent"/}"
case "$root_deployment_name" in
  ""|.|..|*/*) echo "T5 Isaac root must be one direct deployment" >&2; exit 64 ;;
esac
test -d "$dataset_root"
dataset_root="$(cd "$dataset_root" && pwd -P)"
test -f "$dataset_root/val_unseen/val_unseen.json.gz"
dataset_episode_count="$(python3 - "$dataset_root/val_unseen/val_unseen.json.gz" <<'PY'
import gzip, json, sys
with gzip.open(sys.argv[1], "rt", encoding="utf-8") as stream:
    value = json.load(stream)
episodes = value.get("episodes") if isinstance(value, dict) else None
if not isinstance(episodes, list) or not 1 <= len(episodes) <= 20:
    raise SystemExit("T5 dataset must contain between 1 and 20 episodes")
print(len(episodes))
PY
)"
[[ "$dataset_episode_count" =~ ^[1-9][0-9]*$ ]]
# The T3 Oracle overlay briefly imports its frozen ten-episode parent before
# replacing it with this run's bounded screen dataset.  Deployments intentionally
# omit episode assets, so bind that immutable parent to the host repository
# instead of the nonexistent deployment-relative default.
if test "$mode" = oracle; then
  t3_base_oracle_dataset_root="$t5_host_repository_root/episodes/h1_nav2_oracle"
  test -f "$t3_base_oracle_dataset_root/val_unseen/val_unseen.json.gz"
  test "$(python3 - "$t3_base_oracle_dataset_root/val_unseen/val_unseen.json.gz" <<'PY'
import gzip, json, sys
with gzip.open(sys.argv[1], "rt", encoding="utf-8") as stream:
    value = json.load(stream)
episodes = value.get("episodes") if isinstance(value, dict) else None
print(len(episodes) if isinstance(episodes, list) else -1)
PY
)" = 10
  export INTERNVLA_T3_BASE_ORACLE_DATASET_ROOT="$t3_base_oracle_dataset_root"
fi
export INTERNVLA_T5_DATASET_EPISODE_COUNT="$dataset_episode_count"
export INTERNVLA_T4_EXPECTED_COUNT="$dataset_episode_count"
if test "$final_pilot_lane" != off; then
  test "$final_pilot_lane" = "$lane"
  test "$mode" = model
  test "$execution_profile" = fixed_dataset
  case "$dataset_episode_count" in 1|10) ;; *) exit 64 ;; esac
  if test "$isaac_sensor_profile" != baseline; then
    test "$isaac_sensor_profile" = dual_lane_wp03_stop_shadow
  fi
  test "$strict_extension_profile" = off
  test "$rtf_ablation_profile" = navigation_fast
fi
if test "$engineering_canary_sec" != 0; then
  test "$dataset_episode_count" = 5
fi
if test "$isaac_sensor_profile" = lane_b_revc_fixed5_capture; then
  test "$dataset_episode_count" = 5
fi
if test "$isaac_sensor_profile" = lane_b_step3_live_canary; then
  if test "$dataset_episode_count" != 1 && test "$dataset_episode_count" != 3; then
    exit 64
  fi
fi
if test "$isaac_sensor_profile" = lane_b_step3_direct_fixed5; then
  if test "$dataset_episode_count" != 1 && test "$dataset_episode_count" != 5; then
    exit 64
  fi
fi
if test "$isaac_sensor_profile" = lane_a_step3_timeout_advisor; then
  if test "$dataset_episode_count" != 1 && test "$dataset_episode_count" != 2 \
      && test "$dataset_episode_count" != 3 && test "$dataset_episode_count" != 5; then
    exit 64
  fi
fi
if test "$isaac_sensor_profile" = dual_lane_wp03_stop_shadow; then
  case "$dataset_episode_count" in 1|10) ;; *) exit 64 ;; esac
fi
if test "$nvblox_mode" = active_local_gt; then
  test "$dataset_episode_count" = 3
  export INTERNVLA_T4_MIN_SR_OVERRIDE=0.6666666666666666
elif test "$mode" = oracle; then
  export INTERNVLA_T4_MIN_SR_OVERRIDE=0.8
elif test "$dataset_episode_count" = 20; then
  export INTERNVLA_T4_MIN_SR_OVERRIDE=0.05
else
  # D0 fixed-five is a hardware-reproduction gate: completion and isolation
  # are mandatory, but its same-episode SR is reported rather than tuned.
  export INTERNVLA_T4_MIN_SR_OVERRIDE=0.0
fi
test -f "$root/scripts/run_t4_sensor_gate.sh"
test -f "$root/scripts/build_t5_isaac_remote_phase_overlay.py"
test -f "$root/scripts/audit_t5_kit_gpu_log.py"
test -f "$root/scripts/t5_clock_publisher.py"
test -f "$root/scripts/t5_clock_graph_probe.py"
test -f "$root/scripts/analyze_t5_engineering_canary.py"
test -f "$root/scripts/sample_t5_host_telemetry.py"
test -f "$root/scripts/summarize_t5_rtf_ablation.py"
test -f "$root/scripts/probe_t5_revc_snapshot_smoke.py"
test -f "$root/scripts/probe_t5_revc_fixed5_capture.py"
test -f "$root/scripts/t5_step3_timeout_advisor_node.py"
test -f "$root/scripts/t5_process_identity_audit.py"
test -f "$root/configs/internnav_t5/go2_continuous_completion_cfg.py"
test -f "$root/configs/internnav_t5/revc_four_camera_snapshot.json"
test -f "$t5_isaac_python_package_root/internvla_ros2/fault_injection.py"
test -d "$ros_ws/install"
export INTERNVLA_T5_ISAAC_PYTHON_PACKAGE_ROOT="$t5_isaac_python_package_root"

for port in "$controller_port" "$model_client_port" "$oracle_port" "$clock_port"; do
  [[ "$port" =~ ^[0-9]+$ ]] && ((port >= 1024 && port <= 65535))
done
test "$controller_port" != "$model_client_port"
test "$controller_port" != "$oracle_port"
test "$controller_port" != "$clock_port"
test "$model_client_port" != "$oracle_port"
test "$model_client_port" != "$clock_port"
test "$oracle_port" != "$clock_port"

command -v flock >/dev/null
command -v taskset >/dev/null
command -v docker >/dev/null
command -v ss >/dev/null
command -v realpath >/dev/null

run_token="$(basename "$result_root")"
[[ "$run_token" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}$ ]]
task_run_label="t5_${lane_name}_${mode}_${run_token}"
upstream_internnav_root="${INTERNNAV_ROOT:-$HOME/internnav-t0/InternNav}"
upstream_task_name="${task_run_label}_${mode}_001"
# Upstream uses task_name for both its LMDB resume store and result directory.
# A fresh run identity is mandatory: otherwise a new fixed-five can silently
# skip an old success and copy a stale result.json.
test ! -e "$upstream_internnav_root/data/sample_episodes/$upstream_task_name"
test ! -e "$upstream_internnav_root/logs/$upstream_task_name"

# The outer lease protects the remote GPU.  These host-local locks prevent a
# second entrypoint for the same lane and serialize against shared asset/cache
# mutation.  Normal lane execution takes the shared/read side of the asset
# lock, so A and B remain concurrent.
exec 8>"$lane_lock"
flock -n 8 || { echo "Isaac lane $lane is already running" >&2; exit 73; }
exec 9>"$shared_asset_lock"
flock -s -n 9 || {
  echo "shared Isaac assets are being mutated; runtime refused" >&2
  exit 73
}

test ! -e "$result_root"
mkdir -p "$(dirname "$result_root")"
mkdir "$result_root"
result_root="$(cd "$result_root" && pwd -P)"
case "$result_root/" in
  "$root"/*) ;;
  *) echo "result root must be inside the lane deployment root" >&2; exit 64 ;;
esac
mkdir -p "$result_root/logs/kit" "$result_root/pids" \
  "$result_root/health" "$result_root/runtime/tmp"
if test "$fault_injection_profile" != off; then
  mkdir -p "$result_root/fault_control"
  export INTERNNAV_T5_FAULT_CONTROL_PATH="$result_root/fault_control/state.json"
  export INTERNNAV_T5_FAULT_EVENT_PATH="$result_root/fault_control/events.jsonl"
  python3 - "$INTERNNAV_T5_FAULT_CONTROL_PATH" "$lane" \
    "$fault_injection_profile" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps({
    "schema_version": 1,
    "profile": sys.argv[3],
    "lane": sys.argv[2],
    "revision": 0,
    "observed_sim_ns": 0,
    "active": [],
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
fi

pid_ledger="$result_root/pid_ledger.jsonl"
: >"$pid_ledger"
clock_result="$result_root/clock_summary.json"
clock_live_state="$result_root/health/clock_live.json"
health_state="$result_root/health/isaac_health.json"
ipc_dir="$root/runtime/t4_ipc"
health_socket="$ipc_alias/isaac_health.sock"

# Fail closed if the two invocations were pointed at one deployment root.
mkdir -p "$root/runtime"
exec 7>"$root/runtime/.t5_isaac_lane_owner.lock"
flock -x -w 5 7
if test -f "$root/runtime/t5_isaac_lane_owner"; then
  test "$(cat "$root/runtime/t5_isaac_lane_owner")" = "$lane_name"
else
  printf '%s\n' "$lane_name" >"$root/runtime/t5_isaac_lane_owner.tmp.$$"
  mv "$root/runtime/t5_isaac_lane_owner.tmp.$$" \
    "$root/runtime/t5_isaac_lane_owner"
fi
flock -u 7
exec 7>&-

test -d "$lane_profile_root/cache/xdg"
test -d "$lane_profile_root/cache/ov"
test -d "$lane_profile_root/cache/nvidia"
test -d "$lane_profile_root/config"
test -d "$lane_profile_root/data"
test -d "$lane_profile_root/tmp"
mkdir -p "$lane_profile_root/kit/profile" \
  "$lane_profile_root/tmp/$run_token"

export ROS_DOMAIN_ID="$ros_domain_id"
export ROS_NAMESPACE="$lane_namespace"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_STATIC_PEERS="$edge_ip"
export CUDA_VISIBLE_DEVICES="$gpu"
export INTERNVLA_ISAAC_RENDER_GPU="$gpu"
export INTERNVLA_ISAAC_PHYSICS_GPU=0
export INTERNNAV_T5_LANE="$lane"
export INTERNNAV_T5_LANE_NAMESPACE="$lane_namespace"
export INTERNNAV_T5_ID_PREFIX="$identity_prefix"
export INTERNVLA_EPISODE_ID_PREFIX="$identity_prefix"
export INTERNVLA_RESET_ID_PREFIX="$identity_prefix"
export INTERNVLA_REQUEST_ID_PREFIX="$identity_prefix"
if test "$mode" = model; then
  # The phase materializes BasePathKeyEpisodeloader with the exact merged
  # evaluator config, then reverses those fresh path keys exactly once.
  export INTERNVLA_MODEL_EPISODE_ORDER_MANIFEST="$result_root/ordered_episode_manifest.json"
else
  unset INTERNVLA_MODEL_EPISODE_ORDER_MANIFEST
fi
export XDG_CACHE_HOME="$lane_profile_root/cache/xdg"
export XDG_CONFIG_HOME="$lane_profile_root/config"
export XDG_DATA_HOME="$lane_profile_root/data"
export OV_CACHE_ROOT="$lane_profile_root/cache/ov"
export NVIDIA_SHADER_CACHE_PATH="$lane_profile_root/cache/nvidia"
export CUDA_CACHE_PATH="$lane_profile_root/cache/nvidia/cuda"
export TMPDIR="$lane_profile_root/tmp/$run_token"
export OMNI_KIT_USER_CONFIG="$lane_profile_root/kit/profile"
export OMNI_KIT_LOG_PATH="$result_root/logs/kit"
export INTERNVLA_T5_FULL_MP4_ENCODING=0
export INTERNVLA_T5_VIDEO_POLICY=jsonl_keyframes_only
export INTERNVLA_SAVE_VIDEO=0
export INTERNVLA_T4_CONTAINER_NAME="$container"
export INTERNVLA_T4_IPC_ALIAS_OVERRIDE="$ipc_alias"
export INTERNVLA_T4_IPC_ALLOWED_ROOT="$root"
export INTERNVLA_T5_EXPECTED_LEASE_ACK="$expected_lease"
if test "$isaac_sensor_profile" != baseline; then
  export INTERNVLA_T5_REVC_CAMERA_CONFIG="$root/configs/internnav_t5/revc_four_camera_snapshot.json"
  export INTERNVLA_T5_REVC_SNAPSHOT_REQUEST_PATH="$result_root/evaluator/revc_snapshot.request.json"
  export INTERNVLA_T5_REVC_SNAPSHOT_ACK_PATH="$result_root/evaluator/revc_snapshot.ack.json"
else
  unset INTERNVLA_T5_REVC_CAMERA_CONFIG
  unset INTERNVLA_T5_REVC_SNAPSHOT_REQUEST_PATH
  unset INTERNVLA_T5_REVC_SNAPSHOT_ACK_PATH
fi

# Pin this shell so every host-side Isaac/Kit descendant inherits the exact
# lane CPU affinity.  The prepared container must independently have the same
# cpuset, because docker-exec does not inherit the client's affinity mask.
taskset -pc "$cpuset" "$$" >"$result_root/cpu_affinity.txt"

docker inspect -f '{{.State.Running}}' "$container" | grep -Fxq true
test "$(docker inspect -f '{{index .Config.Labels "internnav.t5.lane"}}' "$container")" = "$lane"
test "$(docker inspect -f '{{index .Config.Labels "internnav.t5.gpu"}}' "$container")" = "$gpu"
test "$(docker inspect -f '{{.HostConfig.CpusetCpus}}' "$container")" = "$cpuset"
container_pid_mode="$(docker inspect -f '{{.HostConfig.PidMode}}' "$container")"
container_ipc_mode="$(docker inspect -f '{{.HostConfig.IpcMode}}' "$container")"
test -z "$container_pid_mode"
test "$container_ipc_mode" = private
nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ' | grep -Fxq "$gpu"
host_gpu_uuid="$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
host_gpu_pci_bus_id="$(nvidia-smi -i "$gpu" --query-gpu=pci.bus_id --format=csv,noheader | tr -d '[:space:]')"
[[ "$host_gpu_uuid" =~ ^GPU-[A-Za-z0-9-]+$ ]]
[[ "$host_gpu_pci_bus_id" =~ ^[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9]$ ]]
mapfile -t container_gpu_uuids < <(
  docker exec --user admin -e CUDA_VISIBLE_DEVICES=0 "$container" \
    nvidia-smi --query-gpu=uuid --format=csv,noheader | sed 's/[[:space:]]//g'
)
test "${#container_gpu_uuids[@]}" = 1
test "${container_gpu_uuids[0]}" = "$host_gpu_uuid"
mapfile -t container_gpu_pci_bus_ids < <(
  docker exec --user admin -e CUDA_VISIBLE_DEVICES=0 "$container" \
    nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | sed 's/[[:space:]]//g'
)
test "${#container_gpu_pci_bus_ids[@]}" = 1
test "${container_gpu_pci_bus_ids[0],,}" = "${host_gpu_pci_bus_id,,}"
python3 - "$result_root/gpu_mapping.json" "$lane" "$gpu" \
  "$container_cuda_visible_devices" "$host_gpu_uuid" "$host_gpu_pci_bus_id" \
  "$container" <<'PY'
import json, sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "status": "PASS",
    "lane": sys.argv[2],
    "host_physical_gpu_index": int(sys.argv[3]),
    "host_cuda_visible_devices": sys.argv[3],
    "host_cuda_logical_zero_gpu_uuid": sys.argv[5],
    "isaac_physics_logical_gpu_index": 0,
    "container_logical_cuda_visible_devices": sys.argv[4],
    "physical_gpu_uuid": sys.argv[5],
    "container_gpu_uuid": sys.argv[5],
    "physical_gpu_pci_bus_id": sys.argv[6],
    "container_gpu_pci_bus_id": sys.argv[6],
    "container": sys.argv[7],
    "container_gpu_count": 1,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

scan_forbidden_compute() {
  local output="$1" scan_rc=0
  python3 "$root/scripts/t5_process_identity_audit.py" \
    --mode forbidden-compute --output "$output" || scan_rc=$?
  case "$scan_rc" in
    0) return 0 ;;
    73)
      echo "model/Nav2/localization/map/velocity processes are forbidden on x86" >&2
      return 73
      ;;
    *)
      printf 'forbidden compute identity audit failed: rc=%s\n' "$scan_rc" >&2
      return 74
      ;;
  esac
}

scan_lane_runtime_residuals() {
  local output="$1" scan_rc=0
  python3 "$root/scripts/t5_process_identity_audit.py" \
    --mode lane-runtime-residual --deployment-root "$root" \
    --output "$output" || scan_rc=$?
  case "$scan_rc" in
    0) return 0 ;;
    73) return 73 ;;
    *) return 74 ;;
  esac
}

scan_forbidden_compute "$result_root/forbidden_compute_prestart.txt"

record_pid_event() {
  local component="$1" pid="$2" event="$3" log_path="$4" scope="${5:-host}" pgid=""
  if test -n "$pid"; then
    if test "$scope" = container; then
      pgid="$(docker exec --user admin "$container" ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    else
      pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    fi
  fi
  python3 - "$pid_ledger" "$lane" "$component" "$pid" "$pgid" "$event" \
    "$log_path" "$scope" "$ros_domain_id" "$lane_namespace" <<'PY'
import json, os, sys, time
from pathlib import Path

row = {
    "schema_version": 1,
    "lane": sys.argv[2],
    "component": sys.argv[3],
    "pid": int(sys.argv[4]) if sys.argv[4] else None,
    "pgid": int(sys.argv[5]) if sys.argv[5] else None,
    "event": sys.argv[6],
    "log_path": sys.argv[7],
    "scope": sys.argv[8],
    "ros_domain_id": int(sys.argv[9]),
    "namespace": sys.argv[10],
    "host": os.uname().nodename,
    "wall_unix": time.time(),
}
with Path(sys.argv[1]).open("a", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps(row, sort_keys=True) + "\n")
PY
}

write_health_state() {
  local state="$1"
  python3 - "$health_state" "$lane" "$state" "$ros_domain_id" \
    "$lane_namespace" "$identity_prefix" "$health_socket" <<'PY'
import json, os, sys, time
from pathlib import Path

path = Path(sys.argv[1])
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps({
    "schema_version": 1,
    "lane": sys.argv[2],
    "state": sys.argv[3],
    "ros_domain_id": int(sys.argv[4]),
    "namespace": sys.argv[5],
    "identity_prefix": sys.argv[6],
    "endpoint": "unix://" + sys.argv[7],
    "host": os.uname().nodename,
    "wall_unix": time.time(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
}

clock_publisher_count() {
  local count="" query_rc=0
  count="$(timeout --signal=TERM --kill-after=0.5s 8s \
    docker exec --user admin --workdir /workspaces/isaac \
    -e "ROS_DOMAIN_ID=$ros_domain_id" -e "ROS_NAMESPACE=$lane_namespace" \
    -e ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST -e "ROS_STATIC_PEERS=$edge_ip" \
    -e ROS_LOCALHOST_ONLY=0 "$container" \
    timeout --signal=TERM --kill-after=0.5s 6s bash -lc '
      set -eo pipefail
      set +u
      source /opt/ros/jazzy/setup.bash
      source /workspaces/isaac/install/setup.bash
      set -u
      exec python3 "$1/scripts/t5_clock_graph_probe.py" \
        --topic /clock --discovery-sec 1.5
    ' bash "$root")" || query_rc=$?
  test "$query_rc" = 0 || return "$query_rc"
  [[ "$count" =~ ^[0-9]+$ ]] || return 65
  printf '%s\n' "$count"
}

clock_publisher_count_is() {
  local expected="$1" observed=""
  observed="$(clock_publisher_count)" || return $?
  test "$observed" = "$expected"
}

runtime_clock_authority_ready() {
  # Uniqueness is proved before the evaluator starts.  At runtime, use the
  # live publisher plus the acknowledged fresh sensor/clock state; a repeated
  # short DDS graph rediscovery can miss the publisher under concurrent Kit
  # load and must not terminate an otherwise healthy fixed-dataset run.
  leader_is_alive "$clock_pid" || return $?
  engineering_canary_fresh_sensor_clock_count >/dev/null
}

read_agent_application_log_counts() {
  local log_path="$1" agent_kind="$2"
  python3 - "$log_path" "$agent_kind" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
agent_kind = sys.argv[2]
if agent_kind == "model":
    success_marker = b"INTERNVLA_MODEL_ACTION_OK "
    error_marker = b"INTERNVLA_LOCAL_IPC_STEP_ERROR "
elif agent_kind == "oracle":
    success_marker = b"INTERNVLA_ORACLE_ACTION_OK "
    error_marker = b"INTERNVLA_ORACLE_STEP_ERROR "
else:
    raise RuntimeError("unsupported evaluator agent kind")
data = path.read_bytes()
# The evaluator writes concurrently.  Ignore a final unterminated record rather
# than treating an in-flight flush as malformed evidence.
lines = data.split(b"\n")
if lines:
    lines = lines[:-1]

markers = {
    success_marker: "success",
    error_marker: "error",
}
counts = {"success": 0, "error": 0}
dataset_counts = []
for line in lines:
    dataset_counts.extend(
        int(value) for value in re.findall(rb"\btotal_path:\s*([0-9]+)\b", line)
    )
    for marker, kind in markers.items():
        _, present, encoded = line.partition(marker)
        if not present:
            continue
        value = json.loads(encoded.decode("utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise RuntimeError("invalid %s model application marker" % kind)
        counts[kind] += 1
if len(set(dataset_counts)) > 1:
    raise RuntimeError("evaluator reported inconsistent total_path values")
dataset_count = dataset_counts[0] if dataset_counts else -1
print("%d %d %d" % (counts["success"], counts["error"], dataset_count))
PY
}

read_engineering_canary_application_log_state() {
  local log_path="$1" agent_kind="$2"
  python3 - "$log_path" "$agent_kind" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
agent_kind = sys.argv[2]
if agent_kind == "model":
    success_marker = b"INTERNVLA_MODEL_ACTION_OK "
    error_marker = b"INTERNVLA_LOCAL_IPC_STEP_ERROR "
elif agent_kind == "oracle":
    success_marker = b"INTERNVLA_ORACLE_ACTION_OK "
    error_marker = b"INTERNVLA_ORACLE_STEP_ERROR "
else:
    raise RuntimeError("unsupported evaluator agent kind")

lines = path.read_bytes().split(b"\n")[:-1]
success_count = 0
error_count = 0
consecutive_success_count = 0
nav2_timeout_warning_count = 0
nonwarning_error_count = 0
dataset_counts = []
safe_stop_timeout_errors = {
    "RuntimeError('local InternVLA step failed: 4 Nav2 goal/cmd_vel timeout')",
    "RuntimeError('local InternVLA step failed: 4 timeout waiting for step goal acceptance')",
}
for line in lines:
    dataset_counts.extend(
        int(value) for value in re.findall(rb"\btotal_path:\s*([0-9]+)\b", line)
    )
    _, success_present, success_encoded = line.partition(success_marker)
    _, error_present, error_encoded = line.partition(error_marker)
    if success_present and error_present:
        raise RuntimeError("one application log record contains success and error")
    if success_present:
        value = json.loads(success_encoded.decode("utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise RuntimeError("invalid success model application marker")
        success_count += 1
        consecutive_success_count += 1
    elif error_present:
        value = json.loads(error_encoded.decode("utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise RuntimeError("invalid error model application marker")
        error_count += 1
        consecutive_success_count = 0
        error_text = value.get("error")
        safe_stop = value.get("safe_stop") is True
        if (
            safe_stop
            and isinstance(error_text, str)
            and error_text in safe_stop_timeout_errors
        ):
            nav2_timeout_warning_count += 1
        else:
            nonwarning_error_count += 1
if len(set(dataset_counts)) > 1:
    raise RuntimeError("evaluator reported inconsistent total_path values")
dataset_count = dataset_counts[0] if dataset_counts else -1
print(
    "%d %d %d %d %d %d"
    % (
        success_count,
        error_count,
        dataset_count,
        consecutive_success_count,
        nav2_timeout_warning_count,
        nonwarning_error_count,
    )
)
PY
}

engineering_canary_fresh_sensor_clock_count() {
  python3 - "$result_root/health/sensor_frames.jsonl" "$clock_live_state" <<'PY'
import json
import math
import sys
import time
from pathlib import Path

sensor = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
clock = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
now = time.time()
clock_age = now - float(clock.get("updated_unix", 0.0))
checks = (
    sensor.get("status") == "PASS",
    sensor.get("evidence")
    == "real_sim_depth_frame_acknowledged_by_dgx_controller",
    clock.get("status") == "RUNNING",
    isinstance(clock.get("last_clock_ns"), int) and clock.get("last_clock_ns") > 0,
    isinstance(clock.get("received_step_count"), int)
    and clock.get("received_step_count") > 0,
    isinstance(clock.get("publish_count"), int) and clock.get("publish_count") > 0,
    clock.get("regression_count") == 0,
    clock.get("invalid_count") == 0,
    math.isfinite(clock_age),
    0.0 <= clock_age <= 15.0,
)
if not all(checks):
    raise SystemExit(1)
print(clock["received_step_count"])
PY
}

wait_for_engineering_canary_ready_snapshot() {
  local canary_model_state="" canary_current_clock_received=""
  local canary_action_progress_age_sec=-1
  while (( SECONDS < runtime_ready_deadline )); do
    leader_is_alive "$gate_pid" || return 1
    canary_model_state="$(read_engineering_canary_application_log_state \
      "$result_root/logs/evaluator_outer.log" "$mode")" || return $?
    read -r model_action_count model_step_error_count evaluator_total_path_count \
      canary_consecutive_action_count canary_nav2_timeout_warning_count \
      canary_nonwarning_error_count <<<"$canary_model_state"
    [[ "$model_action_count" =~ ^[0-9]+$ ]] || return 1
    [[ "$model_step_error_count" =~ ^[0-9]+$ ]] || return 1
    test "$evaluator_total_path_count" = "$dataset_episode_count" || return 1
    if test "$model_action_count" -lt "$canary_previous_action_count"; then
      echo "engineering canary readiness failed: model action count rolled back" >&2
      return 1
    fi
    if test "$model_action_count" -gt "$canary_previous_action_count"; then
      canary_previous_action_count="$model_action_count"
      canary_last_action_progress_seconds=$SECONDS
    fi
    # Pre-READY errors are retained as WARN evidence and included in the READY
    # baseline.  The two-success streak below proves recovery; only errors
    # added after READY count against the bounded canary window.
    if test "$canary_nav2_timeout_warning_count" \
        -gt "$engineering_canary_pre_ready_nav2_timeout_limit"; then
      echo "engineering canary readiness failed: pre-READY Nav2 timeout warning limit exceeded" >&2
      return 1
    fi
    if test "$canary_nonwarning_error_count" != 0; then
      echo "engineering canary readiness failed: fatal application error before READY" >&2
      return 1
    fi
    canary_current_clock_received="$(
      engineering_canary_fresh_sensor_clock_count 2>/dev/null || true
    )"
    if [[ "$canary_current_clock_received" =~ ^[1-9][0-9]*$ ]]; then
      if test "$canary_previous_clock_received" -ge 0 && \
          test "$canary_current_clock_received" \
            -gt "$canary_previous_clock_received"; then
        canary_clock_progress_observed=1
      fi
      canary_previous_clock_received="$canary_current_clock_received"
    fi
    if test "$canary_consecutive_action_count" -ge 2 && \
        test "$canary_clock_progress_observed" = 1 && \
        [[ "$canary_current_clock_received" =~ ^[1-9][0-9]*$ ]] && \
        test "$canary_last_action_progress_seconds" -ge 0; then
      canary_action_progress_age_sec=$((
        SECONDS - canary_last_action_progress_seconds
      ))
      test "$canary_action_progress_age_sec" -le 15 || {
        sleep 1
        continue
      }
      canary_sensor_action_progress_age_sec="$canary_action_progress_age_sec"
      ready_model_action_count="$model_action_count"
      ready_model_step_error_count="$model_step_error_count"
      return 0
    fi
    # An allowed safe-stop timeout reaches this path with streak zero.  Keep
    # waiting under the original readiness deadline for two new successes.
    sleep 1
  done
  echo "engineering canary readiness deadline expired before a stable READY snapshot" >&2
  return 1
}

directory_has_no_sockets() {
  local directory="$1" first="" find_rc=0
  first="$(find "$directory" -maxdepth 1 -type s -print -quit)" || find_rc=$?
  test "$find_rc" = 0 || return "$find_rc"
  test -z "$first"
}

ipc_target_is_allowed_t5_runtime() {
  local candidate="$1" deployment_parent="$2" relative deployment canonical
  case "$candidate" in /*) ;; *) return 1 ;; esac
  relative="${candidate#"$deployment_parent"/}"
  test "$relative" != "$candidate" || return 1
  deployment="${relative%/runtime/t4_ipc}"
  test "$deployment" != "$relative" || return 1
  case "$deployment" in ""|.|..|*/*) return 1 ;; esac
  test "$candidate" = "$deployment_parent/$deployment/runtime/t4_ipc" || return 1
  canonical="$(realpath -m -- "$candidate")" || return 1
  test "$canonical" = "$candidate"
}

ipc_kernel_has_no_live_socket() {
  local candidate="$1" unix_rc=0
  test -r /proc/net/unix || return 1
  grep -F -q -- "$candidate/" /proc/net/unix || unix_rc=$?
  case "$unix_rc" in
    0) return 1 ;;
    1) return 0 ;;
    *) return 1 ;;
  esac
}

ipc_old_target_has_no_usage() {
  local candidate="$1" old_root process_dir metadata resolved process_uid
  local current_uid state same_uid maps_rc
  old_root="${candidate%/runtime/t4_ipc}"
  current_uid="$(id -u)" || return 1
  if test -e "$candidate" || test -L "$candidate"; then
    test -d "$candidate" || return 1
    test ! -L "$candidate" || return 1
    directory_has_no_sockets "$candidate" || return 1
  fi
  ipc_kernel_has_no_live_socket "$candidate" || return 1
  # Native ROS/Kit children do not necessarily retain the launcher path in
  # argv.  Readable environment, cwd, open files and maps therefore count as
  # use.  Linux may report another same-UID environ as readable and still deny
  # the open, so environ is best-effort; the other live-use checks stay strict.
  for process_dir in /proc/[0-9]*; do
    process_uid="$(stat -c %u -- "$process_dir" 2>/dev/null)" || {
      test ! -d "$process_dir" || return 1
      continue
    }
    same_uid=false
    if test "$process_uid" = "$current_uid"; then
      same_uid=true
      state="$(sed -n 's/^State:[[:space:]]*\([^[:space:]]\).*/\1/p' \
        "$process_dir/status" 2>/dev/null)" || {
          test ! -d "$process_dir" || return 1
          continue
        }
      test -n "$state" || return 1
      # A zombie has no usable cwd, files, maps or environment.
      test "$state" != Z || continue
    fi
    metadata="$process_dir/cmdline"
    if ! test -r "$metadata"; then
      test "$same_uid" = false || test ! -d "$process_dir" || return 1
    else
      metadata="$(tr '\0' '\n' 2>/dev/null <"$metadata")" || {
        test "$same_uid" = false || test ! -d "$process_dir" || return 1
      }
      case "$metadata" in *"$old_root"*) return 1 ;; esac
    fi
    if test -r "$process_dir/environ"; then
      metadata="$(tr '\0' '\n' 2>/dev/null <"$process_dir/environ")" || metadata=""
      case "$metadata" in *"$old_root"*) return 1 ;; esac
    fi
    if test -r "$process_dir/maps"; then
      maps_rc=0
      grep -F -q -- "$old_root/" "$process_dir/maps" 2>/dev/null || maps_rc=$?
      case "$maps_rc" in
        0) return 1 ;;
        1) ;;
        *) test "$same_uid" = false || test ! -d "$process_dir" || return 1 ;;
      esac
    elif test "$same_uid" = true && test -d "$process_dir"; then
      return 1
    fi
    for metadata in "$process_dir/cwd" "$process_dir/root"; do
      resolved="$(readlink -- "$metadata" 2>/dev/null)" || {
        test ! -L "$metadata" || test "$same_uid" = false || return 1
        continue
      }
      case "$resolved" in "$old_root"|"$old_root"/*) return 1 ;; esac
    done
    if ! test -r "$process_dir/fd" || ! test -x "$process_dir/fd"; then
      test "$same_uid" = false || test ! -d "$process_dir" || return 1
      continue
    fi
    for metadata in "$process_dir/fd/"*; do
      resolved="$(readlink -- "$metadata" 2>/dev/null)" || {
        test ! -L "$metadata" || test "$same_uid" = false || return 1
        continue
      }
      case "$resolved" in "$old_root"|"$old_root"/*) return 1 ;; esac
    done
  done
  return 0
}

replace_ipc_alias_atomically() {
  local alias_path="$1" target="$2" expected_link="$3" expected_inode="$4"
  local temporary="${alias_path}.new.$$"
  test ! -e "$temporary" && test ! -L "$temporary" || return 1
  ln -s "$target" "$temporary" || return 1
  if ! test -L "$alias_path" || \
      ! test "$(readlink -- "$alias_path")" = "$expected_link" || \
      ! test "$(stat -c '%d:%i' -- "$alias_path")" = "$expected_inode"; then
    rm -f -- "$temporary"
    return 1
  fi
  if ! mv -T -- "$temporary" "$alias_path"; then
    rm -f -- "$temporary"
    return 1
  fi
}

reconcile_ipc_alias() {
  local alias_path="$1" target="$2" deployment_parent="$3"
  local existing_target existing_inode
  ipc_target_is_allowed_t5_runtime "$target" "$deployment_parent" || return 1
  mkdir -p "$target"
  directory_has_no_sockets "$target" || return 1
  if ! test -e "$alias_path" && ! test -L "$alias_path"; then
    ln -s "$target" "$alias_path"
    return
  fi
  test -L "$alias_path" || return 1
  existing_target="$(readlink -- "$alias_path")" || return 1
  if test "$existing_target" = "$target"; then
    return 0
  fi
  ipc_target_is_allowed_t5_runtime "$existing_target" "$deployment_parent" || return 1
  ipc_old_target_has_no_usage "$existing_target" || return 1
  existing_inode="$(stat -c '%d:%i' -- "$alias_path")" || return 1
  # Revalidate both identity and inactivity immediately before the atomic
  # rename.  The lane lock excludes peer launchers; these checks also reject
  # an independently restarted owner of the previous deployment.
  test "$(readlink -- "$alias_path")" = "$existing_target" || return 1
  ipc_old_target_has_no_usage "$existing_target" || return 1
  replace_ipc_alias_atomically \
    "$alias_path" "$target" "$existing_target" "$existing_inode"
}

capture_runtime_gpu_evidence() {
  local compute_csv="$result_root/health/nvidia_compute_apps.csv"
  local pmon_txt="$result_root/health/nvidia_pmon.txt"
  local container_csv="$result_root/health/container_nvidia_smi.csv"
  nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits >"$compute_csv" 2>&1 || true
  nvidia-smi pmon -c 1 >"$pmon_txt" 2>&1 || true
  docker exec --user admin -e CUDA_VISIBLE_DEVICES=0 "$container" \
    nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv,noheader \
    >"$container_csv" 2>&1
  python3 - "$result_root/health/runtime_gpu_evidence.json" \
    "$result_root/gpu_mapping.json" "$result_root/health/kit_gpu_ready_audit.json" \
    "$compute_csv" "$pmon_txt" "$container_csv" "$host_gpu_uuid" \
    "$host_gpu_pci_bus_id" "$gpu" <<'PY'
import hashlib, json, sys, time
from pathlib import Path

output, mapping_path, kit_path, compute_path, pmon_path, container_path = map(
    Path, sys.argv[1:7]
)
expected_uuid, expected_bus, expected_index = sys.argv[7:10]
mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
kit = json.loads(kit_path.read_text(encoding="utf-8"))
compute_text = compute_path.read_text(encoding="utf-8", errors="replace")
container_text = container_path.read_text(encoding="utf-8", errors="replace")
checks = {
    "static_mapping_pass": mapping.get("status") == "PASS",
    "kit_physical_identity_pass": kit.get("status") == "PASS",
    "host_physical_index_matches": mapping.get("host_physical_gpu_index") == int(expected_index),
    "host_uuid_matches": mapping.get("physical_gpu_uuid") == expected_uuid,
    "host_pci_bus_matches": str(mapping.get("physical_gpu_pci_bus_id", "")).lower() == expected_bus.lower(),
    "container_exposes_one_gpu": mapping.get("container_gpu_count") == 1,
    "physics_logical_zero_resolves_to_expected_uuid": (
        mapping.get("host_cuda_visible_devices") == expected_index
        and mapping.get("host_cuda_logical_zero_gpu_uuid") == expected_uuid
        and mapping.get("isaac_physics_logical_gpu_index") == 0
    ),
    "container_runtime_uuid_visible": expected_uuid in container_text,
}
process_uuid_observed = expected_uuid in compute_text
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "expected_host_physical_gpu_index": int(expected_index),
    "expected_gpu_uuid": expected_uuid,
    "expected_pci_bus_id": expected_bus,
    "physics_logical_gpu_index": 0,
    "physics_resolved_gpu_uuid": mapping.get("host_cuda_logical_zero_gpu_uuid"),
    "runtime_compute_process_uuid_observed": process_uuid_observed,
    "runtime_compute_process_note": (
        "matched expected UUID in nvidia-smi compute-apps"
        if process_uuid_observed
        else "no compute-app row was exposed; raw pmon evidence retained"
    ),
    "checks": checks,
    "raw_evidence": {
        str(path.name): {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for path in (compute_path, pmon_path, container_path)
    },
    "recorded_unix": time.time(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if payload["status"] != "PASS":
    raise SystemExit(f"runtime GPU identity evidence failed: {checks}")
PY
}

group_is_alive() {
  local pid="$1" snapshot=""
  test -n "$pid" || return 1
  snapshot="$(ps -eo pgid=)" || return 2
  awk -v expected="$pid" \
    '$1 == expected { found=1 } END { exit(found ? 0 : 1) }' <<<"$snapshot"
}

group_has_runnable_member() {
  local pid="$1" snapshot=""
  test -n "$pid" || return 1
  snapshot="$(ps -eo stat=,pgid=)" || return 2
  awk -v expected="$pid" \
    '$2 == expected && $1 !~ /^Z/ { found=1 } END { exit(found ? 0 : 1) }' \
    <<<"$snapshot"
}

leader_is_alive() {
  local pid="$1" state="" pgid="" snapshot=""
  test -n "$pid" || return 1
  snapshot="$(ps -o stat=,pgid= -p "$pid")" || return $?
  read -r state pgid <<<"$snapshot" || return 2
  [[ "$state" != Z* && "$pgid" = "$pid" ]]
}

stop_host_group() {
  local component="$1" pid="$2" log_path="$3" probe_rc=0
  test -n "$pid" || return 0
  record_pid_event "$component" "$pid" stop_requested "$log_path"
  group_has_runnable_member "$pid" || probe_rc=$?
  if test "$probe_rc" -ge 2; then
    record_pid_event "$component" "$pid" audit_error "$log_path"
    return 1
  fi
  if test "$probe_rc" = 0; then
    kill -INT -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 300); do
      probe_rc=0
      group_has_runnable_member "$pid" || probe_rc=$?
      test "$probe_rc" -lt 2 || {
        record_pid_event "$component" "$pid" audit_error "$log_path"
        return 1
      }
      test "$probe_rc" = 0 || break
      sleep 0.1
    done
  fi
  probe_rc=0
  group_has_runnable_member "$pid" || probe_rc=$?
  test "$probe_rc" -lt 2 || {
    record_pid_event "$component" "$pid" audit_error "$log_path"
    return 1
  }
  if test "$probe_rc" = 0; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 200); do
      probe_rc=0
      group_has_runnable_member "$pid" || probe_rc=$?
      test "$probe_rc" -lt 2 || {
        record_pid_event "$component" "$pid" audit_error "$log_path"
        return 1
      }
      test "$probe_rc" = 0 || break
      sleep 0.1
    done
  fi
  probe_rc=0
  group_has_runnable_member "$pid" || probe_rc=$?
  test "$probe_rc" -lt 2 || {
    record_pid_event "$component" "$pid" audit_error "$log_path"
    return 1
  }
  if test "$probe_rc" = 0; then
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
  wait "$pid" 2>/dev/null || true
  for _ in $(seq 1 50); do
    probe_rc=0
    group_is_alive "$pid" || probe_rc=$?
    test "$probe_rc" -lt 2 || {
      record_pid_event "$component" "$pid" audit_error "$log_path"
      return 1
    }
    test "$probe_rc" = 0 || break
    sleep 0.1
  done
  probe_rc=0
  group_is_alive "$pid" || probe_rc=$?
  test "$probe_rc" -lt 2 || {
    record_pid_event "$component" "$pid" audit_error "$log_path"
    return 1
  }
  if test "$probe_rc" = 0; then
    record_pid_event "$component" "$pid" residual "$log_path"
    return 1
  fi
  record_pid_event "$component" "$pid" verified_absent "$log_path"
}

clock_pid=""
clock_container_pid=""
health_pid=""
gate_pid=""
telemetry_pid=""
revc_probe_pid=""
revc_probe_reaped=0
revc_probe_rc=125
revc_probe_output=""
revc_probe_log=""
step3_timeout_advisor_pid=""
step3_timeout_advisor_container_pid=""
case "$isaac_sensor_profile" in
  lane_b_revc_smoke)
    revc_probe_output="$result_root/evaluator/revc_snapshot_smoke.json"
    revc_probe_log="$result_root/logs/revc_snapshot_smoke.log"
    ;;
  lane_b_revc_fixed5_capture)
    revc_probe_output="$result_root/evaluator/revc_fixed5_capture.json"
    revc_probe_log="$result_root/logs/revc_fixed5_capture.log"
    ;;
esac
run_rc=125
clock_rc=125
online_ready=0
lane_lock_released=false
asset_lock_fd_released=false

wait_revc_snapshot_probe() {
  test "$isaac_sensor_profile" != baseline || return 0
  test "$isaac_sensor_profile" != lane_b_step3_live_canary || return 0
  test "$isaac_sensor_profile" != lane_b_step3_direct_fixed5 || return 0
  test "$isaac_sensor_profile" != lane_a_step3_timeout_advisor || return 0
  test "$isaac_sensor_profile" != dual_lane_wp03_stop_shadow || return 0
  test -n "$revc_probe_pid"
  if test "$revc_probe_reaped" = 0; then
    revc_probe_rc=0
    wait "$revc_probe_pid" || revc_probe_rc=$?
    revc_probe_reaped=1
    record_pid_event revc_snapshot_probe "$revc_probe_pid" exited \
      "$revc_probe_log"
  fi
  test "$revc_probe_rc" = 0
  python3 - "$revc_probe_output" "$isaac_sensor_profile" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.is_symlink() or not path.is_file():
    raise SystemExit("Rev-C smoke result is not a regular file")
value = json.loads(path.read_text(encoding="utf-8"))
profile = sys.argv[2]
common = (
    value.get("schema_version") == 1,
    value.get("status") == "PASS",
    value.get("profile") == profile,
    value.get("lane") == "b",
    value.get("camera_order")
    == ["front_left", "front", "front_right", "rear"],
)
if profile == "lane_b_revc_smoke":
    specific = (
        value.get("same_render_tick") is True,
        isinstance(value.get("external_preview_max_hz"), (int, float)),
        not isinstance(value.get("external_preview_max_hz"), bool),
        0.0 < float(value.get("external_preview_max_hz", 0.0)) <= 1.0,
        isinstance(value.get("images"), list) and len(value.get("images")) == 4,
        isinstance(value.get("observer"), dict),
        value.get("observer", {}).get("status") == "CAPTURED",
        value.get("observer", {}).get("resolution") == [500, 500],
        value.get("observer", {}).get("fed_to_step3") is False,
    )
elif profile == "lane_b_revc_fixed5_capture":
    snapshots = value.get("snapshots")
    identities = [
        (
            item.get("request", {}).get("episode_id"),
            item.get("request", {}).get("reset_generation"),
            item.get("request", {}).get("sequence_id"),
        )
        for item in snapshots if isinstance(item, dict)
    ] if isinstance(snapshots, list) else []
    specific = (
        value.get("capture_count") == 5,
        value.get("unique_execution_identities") is True,
        value.get("same_render_tick_all") is True,
        len(identities) == len(set(identities)) == 5,
        all(
            item.get("same_render_tick") is True
            and isinstance(item.get("images"), list)
            and len(item["images"]) == 4
            for item in snapshots
        ) if isinstance(snapshots, list) else False,
    )
else:
    specific = (False,)
checks = common + specific
raise SystemExit(0 if all(checks) else 75)
PY
}

wait_revc_fixed5_after_evaluator() {
  test "$isaac_sensor_profile" = lane_b_revc_fixed5_capture || return 0
  local deadline=$((SECONDS + 30))
  while kill -0 "$revc_probe_pid" 2>/dev/null && ((SECONDS < deadline)); do
    sleep 0.1
  done
  if kill -0 "$revc_probe_pid" 2>/dev/null; then
    echo "Rev-C fixed-five probe did not finish with the natural evaluator" >&2
    return 75
  fi
  wait_revc_snapshot_probe
}

stop_clock() {
  test -n "$clock_pid" || return 0
  if test -n "$clock_container_pid"; then
    docker exec --user admin "$container" \
      kill -INT "$clock_container_pid" 2>/dev/null || true
    for _ in $(seq 1 300); do
      docker exec --user admin "$container" \
        kill -0 "$clock_container_pid" 2>/dev/null || break
      sleep 0.1
    done
    if docker exec --user admin "$container" \
        kill -0 "$clock_container_pid" 2>/dev/null; then
      docker exec --user admin "$container" \
        kill -TERM "$clock_container_pid" 2>/dev/null || true
      for _ in $(seq 1 100); do
        docker exec --user admin "$container" \
          kill -0 "$clock_container_pid" 2>/dev/null || break
        sleep 0.1
      done
    fi
    if docker exec --user admin "$container" \
        kill -0 "$clock_container_pid" 2>/dev/null; then
      docker exec --user admin "$container" \
        kill -KILL "$clock_container_pid" 2>/dev/null || true
    fi
  fi
  wait "$clock_pid" 2>/dev/null
  clock_rc=$?
  if test -n "$clock_container_pid" && docker exec --user admin "$container" \
      kill -0 "$clock_container_pid" 2>/dev/null; then
    record_pid_event clock_container "$clock_container_pid" residual \
      "$result_root/logs/clock.log" container
    return 1
  fi
  record_pid_event clock_client "$clock_pid" verified_absent \
    "$result_root/logs/clock.log"
  record_pid_event clock_container "$clock_container_pid" verified_absent \
    "$result_root/logs/clock.log" container
}

stop_step3_timeout_advisor() {
  test -n "$step3_timeout_advisor_pid" || return 0
  if test -n "$step3_timeout_advisor_container_pid"; then
    docker exec --user admin "$container" \
      kill -TERM "$step3_timeout_advisor_container_pid" 2>/dev/null || true
    for _ in $(seq 1 300); do
      docker exec --user admin "$container" \
        kill -0 "$step3_timeout_advisor_container_pid" 2>/dev/null || break
      sleep 0.1
    done
    if docker exec --user admin "$container" \
        kill -0 "$step3_timeout_advisor_container_pid" 2>/dev/null; then
      docker exec --user admin "$container" \
        kill -KILL "$step3_timeout_advisor_container_pid" 2>/dev/null || true
    fi
  fi
  wait "$step3_timeout_advisor_pid" 2>/dev/null || true
  if test -n "$step3_timeout_advisor_container_pid" && \
      docker exec --user admin "$container" \
        kill -0 "$step3_timeout_advisor_container_pid" 2>/dev/null; then
    record_pid_event step3_timeout_advisor_container \
      "$step3_timeout_advisor_container_pid" residual \
      "$result_root/logs/step3_timeout_advisor.log" container
    return 1
  fi
  record_pid_event step3_timeout_advisor "$step3_timeout_advisor_pid" \
    verified_absent "$result_root/logs/step3_timeout_advisor.log"
  record_pid_event step3_timeout_advisor_container \
    "$step3_timeout_advisor_container_pid" verified_absent \
    "$result_root/logs/step3_timeout_advisor.log" container
}

write_status() {
  local status="$1" observed_rc="$2" residual="$3" socket_count="$4" \
    clock_count="$5" clock_probe_exit_code="$6"
  python3 - "$result_root/isaac_status.json" "$status" "$lane" "$mode" \
    "$observed_rc" "$clock_rc" "$residual" "$socket_count" "$clock_count" \
    "$clock_probe_exit_code" "$ros_domain_id" "$lane_namespace" "$gpu" \
    "$cpuset" "$container" \
    "$identity_prefix" "$controller_port" "$model_client_port" "$oracle_port" \
    "$clock_port" "$ipc_alias" "$health_socket" "$lane_lock_released" \
    "$asset_lock_fd_released" "$root" "$lane_profile_root" <<'PY'
import json, os, sys, time
from pathlib import Path

execution_profile = os.environ["INTERNNAV_T5_EXECUTION_PROFILE"]
evaluator_exit_code = int(sys.argv[5])
fixed_dataset_completed = execution_profile == "fixed_dataset" and evaluator_exit_code == 0
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "status": sys.argv[2],
    "lane": sys.argv[3],
    "mode": sys.argv[4],
    "execution_profile": execution_profile,
    "final_pilot_lane": os.environ["INTERNNAV_T5_FINAL_PILOT_LANE"],
    "engineering_duration_timebase": os.environ[
        "INTERNNAV_T5_ENGINEERING_CANARY_TIMEBASE"
    ],
    "strict_extension_profile": os.environ["INTERNNAV_T5_STRICT_EXTENSION_PROFILE"],
    "fault_injection_profile": os.environ["INTERNNAV_T5_FAULT_INJECTION_PROFILE"],
    "stereo_odometry_enabled": os.environ["INTERNVLA_T4_ENABLE_STEREO_ODOMETRY"] == "1",
    "episode_acceptance_claimed": fixed_dataset_completed,
    "evaluation_completed_naturally": fixed_dataset_completed,
    "termination_reason": (
        (
            "coordinator_after_600_sim_second_soak"
            if os.environ["INTERNNAV_T5_ENGINEERING_CANARY_TIMEBASE"] == "sim"
            else "coordinator_after_bounded_canary"
        )
        if execution_profile == "engineering_canary"
        else ("evaluator_natural_exit" if evaluator_exit_code == 0
              else "evaluator_nonzero_exit")
    ),
    "evaluator_exit_code": evaluator_exit_code,
    "clock_exit_code": int(sys.argv[6]),
    "residual_count": int(sys.argv[7]),
    "socket_residual_count": int(sys.argv[8]),
    "clock_publishers_after_stop": int(sys.argv[9]),
    "clock_probe_exit_code": int(sys.argv[10]),
    "host": os.uname().nodename,
    "host_role": "isaac_x86_simulator_only",
    "ros_domain_id": int(sys.argv[11]),
    "namespace": sys.argv[12],
    "dds": {"automatic_discovery_range": os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"],
            "static_peer": os.environ["ROS_STATIC_PEERS"]},
    "cuda_visible_devices": sys.argv[13],
    "host_cuda_visible_devices": sys.argv[13],
    "container_logical_cuda_visible_devices": "0",
    "isaac_render_gpu_physical_index": int(os.environ["INTERNVLA_ISAAC_RENDER_GPU"]),
    "isaac_physics_gpu_visible_index": int(os.environ["INTERNVLA_ISAAC_PHYSICS_GPU"]),
    "kit_active_gpu_log_audit_required": True,
    "kit_active_gpu_log_audit": (
        json.loads((Path(sys.argv[1]).parent / "kit_gpu_audit.json").read_text(
            encoding="utf-8"
        ))
        if (Path(sys.argv[1]).parent / "kit_gpu_audit.json").is_file()
        else None
    ),
    "cpuset": sys.argv[14],
    "container": sys.argv[15],
    "identity_prefix": sys.argv[16],
    "ports": {
        "controller": int(sys.argv[17]),
        "model_client": int(sys.argv[18]),
        "oracle": int(sys.argv[19]),
        "clock": int(sys.argv[20]),
    },
    "ipc_alias": sys.argv[21],
    "health_endpoint": "unix://" + sys.argv[22],
    "lane_lock_released": sys.argv[23] == "true",
    "shared_asset_lock_fd_released": sys.argv[24] == "true",
    "deployment_root": sys.argv[25],
    "lane_profile_root": sys.argv[26],
    "sole_clock_source": True,
    "shared_assets_lock_mode": "shared_read",
    "isaac_sensor_profile": os.environ["INTERNNAV_T5_ISAAC_SENSOR_PROFILE"],
    "revc_enabled": os.environ.get("INTERNVLA_T5_REVC_ENABLE") == "1",
    "revc_observer_enabled": (
        os.environ.get("INTERNVLA_T5_REVC_OBSERVER_ENABLE") == "1"
    ),
    "model_observation_capture_enabled": (
        os.environ.get("INTERNVLA_T5_FULL_RGB_CAPTURE") == "1"
    ),
    "independent_d435_5hz_capture_enabled": (
        os.environ.get("INTERNVLA_T5_D435_5HZ_CAPTURE") == "1"
    ),
    "full_mp4_encoding_allowed": False,
    "video_policy": "jsonl_keyframes_only",
    "owners": ["isaac_sim", "go2_physics", "simulated_sensors",
               "evaluator", "clock_source"],
    "forbidden_owners": ["internvla_model", "localization", "map",
                         "nvblox", "nav2", "watchdog", "velocity_control"],
    "finished_unix": time.time(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

finalize_engineering_canary_tail_guard() {
  test "$engineering_canary_sec" != 0 || return 0
  local canary_artifact="$result_root/engineering_canary.json"
  local tail_model_progress tail_model_action_count
  local tail_model_step_error_count tail_evaluator_total_path_count
  test -f "$canary_artifact" || return 1
  tail_model_progress="$(read_agent_application_log_counts \
    "$result_root/logs/evaluator_outer.log" "$mode")" || return $?
  read -r tail_model_action_count tail_model_step_error_count \
    tail_evaluator_total_path_count <<<"$tail_model_progress"
  [[ "$tail_model_action_count" =~ ^[0-9]+$ ]] || return 1
  [[ "$tail_model_step_error_count" =~ ^[0-9]+$ ]] || return 1
  [[ "$tail_evaluator_total_path_count" =~ ^[0-9]+$ ]] || return 1
  python3 - "$canary_artifact" "$tail_model_action_count" \
    "$tail_model_step_error_count" "$tail_evaluator_total_path_count" \
    "$dataset_episode_count" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
cutoff_action = payload.get("final_model_action_count")
cutoff_errors = payload.get("final_model_step_error_count")
ready_errors = payload.get("ready_model_step_error_count")
tail_action = int(sys.argv[2])
tail_errors = int(sys.argv[3])
tail_total_path = int(sys.argv[4])
expected_total_path = int(sys.argv[5])
tail_checks = {
    "model_action_count_not_rolled_back_at_tail": (
        isinstance(cutoff_action, int)
        and not isinstance(cutoff_action, bool)
        and tail_action >= cutoff_action
    ),
    "model_step_error_count_not_rolled_back_at_tail": (
        isinstance(cutoff_errors, int)
        and not isinstance(cutoff_errors, bool)
        and tail_errors >= cutoff_errors
    ),
    "evaluator_total_path_stable_at_tail": tail_total_path
    == expected_total_path,
}
payload["tail_guard"] = {
    "after_evaluator_stop": True,
    "post_cutoff_model_step_errors_are_audit_only": True,
    "cutoff_model_action_count": cutoff_action,
    "tail_model_action_count": tail_action,
    "ready_model_step_error_count": ready_errors,
    "cutoff_model_step_error_count": cutoff_errors,
    "tail_model_step_error_count": tail_errors,
    "post_cutoff_model_step_error_count": (
        tail_errors - cutoff_errors
        if isinstance(cutoff_errors, int)
        and not isinstance(cutoff_errors, bool)
        else None
    ),
    "tail_evaluator_total_path_count": tail_total_path,
    "expected_evaluator_total_path_count": expected_total_path,
    "checks": tail_checks,
}
payload["checks"].update(tail_checks)
required_check_names = payload.get("required_check_names")
if not isinstance(required_check_names, list) or not all(
    isinstance(name, str) and name in payload["checks"]
    for name in required_check_names
):
    raise SystemExit("engineering canary required-check contract is invalid")
for name in tail_checks:
    if name not in required_check_names:
        required_check_names.append(name)
payload["required_check_names"] = required_check_names
payload["status"] = (
    "PASS" if all(payload["checks"][name] for name in required_check_names)
    else "FAIL"
)
temporary = path.with_name("." + path.name + ".tail.tmp")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
os.replace(temporary, path)
if payload["status"] != "PASS":
    raise SystemExit(75)
PY
}

finalize() {
  local shell_rc=$? residual=0 status=FAIL socket_count=0 clock_count=1
  local forbidden_scan_rc=0 runtime_scan_rc=0 clock_probe_rc=0
  local container_grep_rc=0 asset_release_rc=0 lane_release_rc=0
  trap - EXIT INT TERM HUP
  set +e
  write_health_state STOPPING || residual=$((residual + 1))
  stop_host_group evaluator "$gate_pid" "$result_root/logs/evaluator_outer.log" || \
    residual=$((residual + 1))
  stop_host_group host_telemetry "$telemetry_pid" \
    "$result_root/logs/host_telemetry.log" || residual=$((residual + 1))
  if test -n "$revc_probe_pid" && test "$revc_probe_reaped" = 0; then
    stop_host_group revc_snapshot_probe "$revc_probe_pid" \
      "$revc_probe_log" || residual=$((residual + 1))
    revc_probe_reaped=1
  fi
  stop_step3_timeout_advisor || residual=$((residual + 1))
  # Read typed application markers only after the evaluator process group is
  # absent.  This closes the final sampling-to-cleanup race without connecting
  # another client to either single-client framed TCP endpoint.
  finalize_engineering_canary_tail_guard || residual=$((residual + 1))
  python3 "$root/scripts/audit_t5_kit_gpu_log.py" \
    --log "$result_root/logs/evaluator_outer.log" \
    --expected-active-gpu "$gpu" \
    --expected-gpu-uuid "$host_gpu_uuid" \
    --expected-pci-bus-id "$host_gpu_pci_bus_id" \
    --output "$result_root/kit_gpu_audit.json" \
    >"$result_root/logs/kit_gpu_audit.log" 2>&1 || residual=$((residual + 1))
  if test "$rtf_diagnostic_enabled" = 1; then
    python3 "$root/scripts/summarize_t5_rtf_ablation.py" \
      --result-root "$result_root" \
      --output "$result_root/rtf_ablation_summary.json" || true
  fi
  stop_clock || residual=$((residual + 1))
  stop_host_group health "$health_pid" "$result_root/logs/health.log" || \
    residual=$((residual + 1))
  rm -f "$health_socket" || residual=$((residual + 1))
  if test -L "$ipc_alias" && test "$(readlink "$ipc_alias")" = "$ipc_dir"; then
    rm -f "$ipc_alias" || residual=$((residual + 1))
  else
    residual=$((residual + 1))
  fi
  docker exec --user admin "$container" bash -lc \
    'alias_path="$1"; target="$2"; if test -L "$alias_path" && test "$(readlink "$alias_path")" = "$target"; then rm -f "$alias_path"; elif test -e "$alias_path" || test -L "$alias_path"; then exit 1; fi' \
    bash "$ipc_alias" "$ipc_dir" || residual=$((residual + 1))
  if find "$ipc_dir" -maxdepth 1 -type s -print \
      >"$result_root/socket_residuals.txt"; then
    socket_count=0
    while IFS= read -r _; do socket_count=$((socket_count + 1)); done \
      <"$result_root/socket_residuals.txt"
  else
    socket_count=-1
    residual=$((residual + 1))
  fi
  test "$socket_count" = 0 || residual=$((residual + 1))
  test ! -e "$ipc_alias" && test ! -L "$ipc_alias" || residual=$((residual + 1))
  for _ in $(seq 1 50); do
    clock_probe_rc=0
    clock_count="$(clock_publisher_count 2>/dev/null)" || clock_probe_rc=$?
    test "$clock_probe_rc" = 0 || break
    test "$clock_count" = 0 && break
    sleep 0.1
  done
  if test "$clock_probe_rc" != 0; then
    clock_count=-1
    residual=$((residual + 1))
  elif test "$clock_count" != 0; then
    residual=$((residual + 1))
  fi
  test -f "$clock_result" || residual=$((residual + 1))
  if test -f "$clock_result"; then
    python3 - "$clock_result" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if value.get("status") == "PASS" else 1)
PY
    test $? = 0 || residual=$((residual + 1))
  fi
  scan_forbidden_compute "$result_root/forbidden_compute_poststop.txt" || \
    forbidden_scan_rc=$?
  test "$forbidden_scan_rc" = 0 || residual=$((residual + 1))
  scan_lane_runtime_residuals "$result_root/lane_host_process_residuals.txt" || \
    runtime_scan_rc=$?
  test "$runtime_scan_rc" = 0 || residual=$((residual + 1))
  if docker top "$container" -eo pid,pgid,args \
      >"$result_root/container_top_poststop.txt"; then
    grep -E '[t]5_clock_publisher.py|[i]nternvla_t4_sensor_bridge|[i]nternvla_t4_client|[i]nternvla_nav2_oracle_bridge' \
      "$result_root/container_top_poststop.txt" \
      >"$result_root/lane_container_process_residuals.txt" || container_grep_rc=$?
    case "$container_grep_rc" in
      0) residual=$((residual + 1)) ;;
      1) ;;
      *) residual=$((residual + 1)) ;;
    esac
  else
    : >"$result_root/lane_container_process_residuals.txt"
    residual=$((residual + 1))
  fi
  ss -H -lun "sport = :$clock_port" >"$result_root/clock_udp_residuals.txt" || \
    residual=$((residual + 1))
  test ! -s "$result_root/clock_udp_residuals.txt" || residual=$((residual + 1))
  find "$result_root" -type f -iname '*.mp4' -print \
    >"$result_root/mp4_residuals.txt" || residual=$((residual + 1))
  test ! -s "$result_root/mp4_residuals.txt" || residual=$((residual + 1))
  if test "$fault_injection_profile" != off; then
    python3 -c 'import json,sys;v=json.load(open(sys.argv[1],encoding="utf-8"));assert v.get("profile")=="completion_sim_minimal_v1" and v.get("active")==[], "fault control remained active during cleanup"' \
      "$INTERNNAV_T5_FAULT_CONTROL_PATH"
    test $? = 0 || residual=$((residual + 1))
  fi

  # Explicitly release and probe the lane-local lock.  The shared asset lock
  # cannot be globally probed because the other lane may legitimately retain
  # a shared holder; closing fd 9 proves this process released its share.
  flock -u 9 || asset_release_rc=$?
  exec 9>&- || asset_release_rc=$?
  if test "$asset_release_rc" = 0; then
    asset_lock_fd_released=true
  else
    residual=$((residual + 1))
  fi
  flock -u 8 || lane_release_rc=$?
  exec 8>&- || lane_release_rc=$?
  exec 6>"$lane_lock"
  if test "$lane_release_rc" = 0 && flock -n 6; then
    if flock -u 6; then
      lane_lock_released=true
    else
      residual=$((residual + 1))
    fi
  else
    residual=$((residual + 1))
  fi
  exec 6>&- || residual=$((residual + 1))

  test "$run_rc" = 0 && test "$residual" = 0 && status=PASS
  if ! write_health_state STOPPED; then
    residual=$((residual + 1))
    status=FAIL
  fi
  if ! write_status "$status" "$run_rc" "$residual" "$socket_count" \
      "$clock_count" "$clock_probe_rc"; then
    residual=$((residual + 1))
    status=FAIL
  fi
  final_rc="$run_rc"
  test "$final_rc" != 125 || final_rc="$shell_rc"
  if test "$status" != PASS && test "$final_rc" = 0; then final_rc=1; fi
  exit "$final_rc"
}
trap finalize EXIT
trap 'run_rc=130; exit 130' INT
trap 'run_rc=143; exit 143' TERM HUP

reconcile_ipc_alias "$ipc_alias" "$ipc_dir" "$t5_deployment_parent"
# Execute the exact same fail-closed reconciliation functions in the prepared
# private-PID container.  Keeping one implementation prevents host/container
# policy drift while still checking each namespace for live users.
{
  declare -f directory_has_no_sockets
  declare -f ipc_target_is_allowed_t5_runtime
  declare -f ipc_kernel_has_no_live_socket
  declare -f ipc_old_target_has_no_usage
  declare -f replace_ipc_alias_atomically
  declare -f reconcile_ipc_alias
  printf '%s\n' 'reconcile_ipc_alias "$1" "$2" "$3"'
} | docker exec -i --user admin "$container" bash -s -- \
  "$ipc_alias" "$ipc_dir" "$t5_deployment_parent"

python3 - "$result_root/isaac_contract.json" "$lane" "$mode" \
  "$ros_domain_id" "$lane_namespace" "$gpu" "$cpuset" "$container" \
  "$identity_prefix" "$edge_ip" "$controller_port" "$model_client_port" \
  "$oracle_port" "$clock_port" "$ipc_alias" "$health_socket" "$dataset_root" \
  "$root" "$lane_profile_root" "$run_token" "$task_run_label" \
  "$upstream_task_name" <<'PY'
import hashlib, json, os, sys, time
from pathlib import Path

dataset = Path(sys.argv[17]) / "val_unseen" / "val_unseen.json.gz"
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "status": "READY_TO_START",
    "lane": sys.argv[2],
    "mode": sys.argv[3],
    "execution_profile": os.environ["INTERNNAV_T5_EXECUTION_PROFILE"],
    "final_pilot_lane": os.environ["INTERNNAV_T5_FINAL_PILOT_LANE"],
    "engineering_interval": {
        "configured_seconds": int(os.environ.get(
            "INTERNNAV_T5_ENGINEERING_CANARY_SEC", "0"
        )),
        "duration_timebase": os.environ[
            "INTERNNAV_T5_ENGINEERING_CANARY_TIMEBASE"
        ],
        "wall_time_is_liveness_only": os.environ[
            "INTERNNAV_T5_ENGINEERING_CANARY_TIMEBASE"
        ] == "sim",
    },
    "strict_extension_profile": os.environ["INTERNNAV_T5_STRICT_EXTENSION_PROFILE"],
    "fault_injection_profile": os.environ["INTERNNAV_T5_FAULT_INJECTION_PROFILE"],
    "stereo_odometry_enabled": os.environ["INTERNVLA_T4_ENABLE_STEREO_ODOMETRY"] == "1",
    "episode_acceptance_claimed": False,
    "episode_acceptance_eligible": os.environ["INTERNNAV_T5_EXECUTION_PROFILE"]
        == "fixed_dataset",
    "host": os.uname().nodename,
    "ros_domain_id": int(sys.argv[4]),
    "namespace": sys.argv[5],
    "dds": {"automatic_discovery_range": os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"],
            "static_peer": os.environ["ROS_STATIC_PEERS"]},
    "gpu": int(sys.argv[6]),
    "cuda_visible_devices": sys.argv[6],
    "host_cuda_visible_devices": sys.argv[6],
    "container_logical_cuda_visible_devices": "0",
    "isaac_render_gpu_physical_index": int(os.environ["INTERNVLA_ISAAC_RENDER_GPU"]),
    "isaac_physics_gpu_visible_index": int(os.environ["INTERNVLA_ISAAC_PHYSICS_GPU"]),
    "kit_active_gpu_log_audit_required": True,
    "cpuset": sys.argv[7],
    "container": sys.argv[8],
    "identity": {"episode_prefix": sys.argv[9],
                 "reset_prefix": sys.argv[9],
                 "request_prefix": sys.argv[9]},
    "dgx_peer": sys.argv[10],
    "ports": {"controller": int(sys.argv[11]),
              "model_client": int(sys.argv[12]),
              "oracle": int(sys.argv[13]), "clock": int(sys.argv[14])},
    "ipc_alias": sys.argv[15],
    "health_endpoint": "unix://" + sys.argv[16],
    "dataset": {"path": str(dataset.resolve()), "bytes": dataset.stat().st_size,
                "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "model_identity_order": (
                    "materialized_upstream_manifest"
                    if sys.argv[3] == "model" else "runtime_episode"
                ),
                "model_episode_order_manifest": os.environ.get(
                    "INTERNVLA_MODEL_EPISODE_ORDER_MANIFEST"
                )},
    "deployment_root": sys.argv[18],
    "lane_profile_root": sys.argv[19],
    "run_token": sys.argv[20],
    "task_run_label": sys.argv[21],
    "upstream_task_name": sys.argv[22],
    "cache_roots": {
        "xdg": os.environ["XDG_CACHE_HOME"],
        "ov": os.environ["OV_CACHE_ROOT"],
        "shader": os.environ["NVIDIA_SHADER_CACHE_PATH"],
        "cuda": os.environ["CUDA_CACHE_PATH"],
        "tmp": os.environ["TMPDIR"],
        "kit_profile": os.environ["OMNI_KIT_USER_CONFIG"],
        "kit_log": os.environ["OMNI_KIT_LOG_PATH"],
    },
    "lease_ack": os.environ["INTERNNAV_T5_RESOURCE_LEASE_ACK"],
    "shared_asset_lock_mode": "shared_read",
    "sole_clock_source": True,
    "expected_episode_count": int(os.environ["INTERNVLA_T4_EXPECTED_COUNT"]),
    "minimum_success_rate": float(os.environ["INTERNVLA_T4_MIN_SR_OVERRIDE"]),
    "nvblox_mode": os.environ["INTERNNAV_T5_NVBLOX_MODE"],
    "isaac_sensor_profile": {
        "profile": os.environ["INTERNNAV_T5_ISAAC_SENSOR_PROFILE"],
        "revc_enabled": os.environ.get("INTERNVLA_T5_REVC_ENABLE") == "1",
        "observer_enabled": (
            os.environ.get("INTERNVLA_T5_REVC_OBSERVER_ENABLE") == "1"
        ),
        "scope": "lane_b_snapshot_evidence_only"
            if os.environ["INTERNNAV_T5_ISAAC_SENSOR_PROFILE"] != "baseline"
            else "disabled",
        "external_preview_max_hz": 1.0
            if os.environ["INTERNNAV_T5_ISAAC_SENSOR_PROFILE"] != "baseline"
            else None,
    },
    "rtf_ablation": {
        "profile": os.environ["INTERNNAV_T5_RTF_ABLATION_PROFILE"],
        "diagnostic_enabled": os.environ["INTERNNAV_T5_RTF_ABLATION_PROFILE"]
            not in {"off", "navigation_fast"},
        "diagnostic_only": os.environ["INTERNNAV_T5_RTF_ABLATION_PROFILE"]
            not in {"off", "navigation_fast"},
        "functional_evidence_allowed": os.environ[
            "INTERNNAV_T5_RTF_ABLATION_PROFILE"
        ] in {"off", "navigation_fast"},
        "lidar_enabled": os.environ["INTERNVLA_T4_R3_ENABLE_LIDAR"] == "1",
        "lidar_ray_count": int(os.environ["INTERNVLA_T4_R3_LIDAR_RAY_COUNT"]),
        "depth_stride": int(os.environ["INTERNVLA_T4_DEPTH_STRIDE_OVERRIDE"]),
        "sensor_hz": float(os.environ["INTERNVLA_GO2_SENSOR_HZ"]),
        "rendering_interval": int(os.environ["INTERNVLA_GO2_EXPECT_RENDERING_INTERVAL"]),
        "rgbd_model_resolution": [640, 480],
        "sensor_extrinsics_and_timestamps_changed": False,
        "footprint_or_motion_bounds_changed": False,
        "promotion_basis": (
            "isolated lidar_720 ablation: +26.9% RTF versus 1440 rays"
            if os.environ["INTERNNAV_T5_RTF_ABLATION_PROFILE"]
            == "navigation_fast"
            else None
        ),
    },
    "completion_sim_motion": {
        "profile": os.environ["INTERNVLA_GO2_MOTION_PROFILE"],
        "physics_hz": float(os.environ["INTERNVLA_GO2_PHYSICS_HZ"]),
        "control_hz": float(os.environ["INTERNVLA_GO2_CONTROL_HZ"]),
        "sensor_hz": float(os.environ["INTERNVLA_GO2_SENSOR_HZ"]),
        "physics_dt_sec": float(os.environ["INTERNVLA_GO2_EXPECT_PHYSICS_DT"]),
        "rendering_interval": int(os.environ["INTERNVLA_GO2_EXPECT_RENDERING_INTERVAL"]),
        "bounded_twist_integrated_as_planar_root_pose": True,
        "root_pose_write_rate_hz": 20.0,
        "z_preserved_and_roll_pitch_zeroed": True,
        "gravity_dynamics_claimed": False,
        "physx_collision_and_contact_retained": True,
        "high_fidelity_go2_dynamics_claimed": False,
        "real_go2_configuration_inherits_deviation": False,
        "deviation": "T5 completion_sim navigation-only planar motion at reduced physics and sensor cadence",
    },
    "full_mp4_encoding_allowed": False,
    "video_policy": "jsonl_keyframes_only",
    "model_observation_capture_enabled": (
        os.environ.get("INTERNVLA_T5_FULL_RGB_CAPTURE") == "1"
    ),
    "independent_d435_5hz_capture_enabled": (
        os.environ.get("INTERNVLA_T5_D435_5HZ_CAPTURE") == "1"
    ),
    "started_unix": time.time(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

if test "$rtf_diagnostic_enabled" = 1; then
  setsid python3 -u "$root/scripts/sample_t5_host_telemetry.py" \
    --gpu-index "$gpu" --output "$result_root/host_telemetry.jsonl" \
    >"$result_root/logs/host_telemetry.log" 2>&1 &
  telemetry_pid=$!
  record_pid_event host_telemetry "$telemetry_pid" started \
    "$result_root/logs/host_telemetry.log"
fi

record_pid_event lane_lock "" acquired "$lane_lock"
record_pid_event shared_assets "" acquired_shared "$shared_asset_lock"
write_health_state STARTING
setsid python3 -u - "$health_socket" "$health_state" \
  >"$result_root/logs/health.log" 2>&1 <<'PY' &
import json, os, signal, socket, sys
from pathlib import Path

endpoint = Path(sys.argv[1])
state = Path(sys.argv[2])
endpoint.unlink(missing_ok=True)
server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(str(endpoint))
os.chmod(endpoint, 0o600)
server.listen(4)
server.settimeout(0.2)
stopping = False

def stop(_signum, _frame):
    global stopping
    stopping = True

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
try:
    while not stopping:
        try:
            connection, _ = server.accept()
        except TimeoutError:
            continue
        with connection:
            try:
                payload = state.read_bytes()
            except FileNotFoundError:
                payload = json.dumps({"state": "UNKNOWN"}).encode()
            connection.sendall(payload)
finally:
    server.close()
    endpoint.unlink(missing_ok=True)
PY
health_pid=$!
record_pid_event health "$health_pid" started "$result_root/logs/health.log"
for _ in $(seq 1 100); do
  test -S "$health_socket" && break
  leader_is_alive "$health_pid" || break
  sleep 0.1
done
test -S "$health_socket"
leader_is_alive "$health_pid"

# This must be zero before our publisher starts; a pre-existing publisher in
# the lane's isolated DDS domain is cross-run contamination.
clock_publisher_count_is 0
clock_pid_file="$result_root/pids/clock.container.pid"
setsid docker exec --user admin --workdir "$root" \
  -e "ROS_DOMAIN_ID=$ros_domain_id" -e "ROS_NAMESPACE=$lane_namespace" \
  -e ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST -e "ROS_STATIC_PEERS=$edge_ip" \
  -e "CUDA_VISIBLE_DEVICES=$container_cuda_visible_devices" -e ROS_LOCALHOST_ONLY=0 \
  -e "CLOCK_PID_FILE=$clock_pid_file" -e "CLOCK_RESULT=$clock_result" \
  -e "CLOCK_LIVE_STATE=$clock_live_state" \
  "$container" bash -lc '
    set -eo pipefail
    printf "%s\n" "$$" >"$CLOCK_PID_FILE"
    set +u
    source /opt/ros/jazzy/setup.bash
    source /workspaces/isaac/install/setup.bash
    set -u
    exec python3 "$1/scripts/t5_clock_publisher.py" \
      --port "$2" --result "$CLOCK_RESULT" --state "$CLOCK_LIVE_STATE" \
      --ros-args -r __ns:="$ROS_NAMESPACE"
  ' bash "$root" "$clock_port" >"$result_root/logs/clock.log" 2>&1 &
clock_pid=$!
record_pid_event clock_client "$clock_pid" started "$result_root/logs/clock.log"
for _ in $(seq 1 200); do
  test -s "$clock_pid_file" && break
  leader_is_alive "$clock_pid" || break
  sleep 0.1
done
test -s "$clock_pid_file"
clock_container_pid="$(cat "$clock_pid_file")"
[[ "$clock_container_pid" =~ ^[1-9][0-9]*$ ]]
record_pid_event clock_container "$clock_container_pid" started \
  "$result_root/logs/clock.log" container
for _ in $(seq 1 200); do
  clock_publisher_count_is 1 && break
  leader_is_alive "$clock_pid" || break
  sleep 0.1
done
clock_publisher_count_is 1
write_health_state INFRA_READY

export INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx+isaac
export INTERNVLA_T4_COMPLETION_FAST_PATH=1
export INTERNVLA_T4_RESULT_ROOT="$result_root/evaluator"
export INTERNVLA_T4_RUN_LABEL="$task_run_label"
export INTERNVLA_T4_PHASE_OVERLAY_BUILDER="$root/scripts/build_t5_isaac_remote_phase_overlay.py"
if test "$step3_direct_high_level" = 1; then
  export INTERNVLA_T4_RUNTIME_OVERLAY_BUILDER="$root/scripts/build_t5_lane_b_step3_direct_runtime_overlay.py"
else
  export INTERNVLA_T4_RUNTIME_OVERLAY_BUILDER="$root/scripts/build_t4_r3_sensor_runtime_overlay.py"
fi
export INTERNVLA_T4_GO2_USD_BUILDER="$root/scripts/build_t4_r3_go2_usd.py"
export INTERNVLA_T4_R3_ENABLE_SENSOR_BRIDGE=1
export INTERNVLA_T4_R3_ENABLE_D435I=1
export INTERNVLA_T4_R3_ENABLE_LIDAR="$rtf_lidar_enabled"
export INTERNVLA_T4_R3_LIDAR_RAY_COUNT="$rtf_lidar_ray_count"
export INTERNVLA_T4_R3_ENABLE_RGB_IPC=0
export INTERNVLA_T4_DEPTH_STRIDE_OVERRIDE="$rtf_depth_stride"
# The dedicated overlay validates the frozen T4 base before applying the
# T5-only 20 Hz physics / 5 Hz real-sensor navigation deviation.
export INTERNVLA_T3_CONFIG="$root/configs/internnav_t5/go2_continuous_completion_cfg.py"
export INTERNVLA_T4_DGX_BIND_IP="$edge_ip"
export INTERNVLA_T4_CONTROLLER_TCP_PORT="$controller_port"
export INTERNVLA_T5_EDGE_IP="$edge_ip"
export INTERNVLA_T5_LANE_IP="$edge_ip"
export INTERNVLA_T5_MODEL_CLIENT_PORT="$model_client_port"
export INTERNVLA_T5_ORACLE_PORT="$oracle_port"
export INTERNVLA_T5_CLOCK_UDP_ENDPOINT="127.0.0.1:$clock_port"
export INTERNVLA_GO2_CONTROLLER_ENDPOINT="tcp://$edge_ip:$controller_port"
export INTERNVLA_T5_SENSOR_FRAME_READY_RECORD="$result_root/health/sensor_frames.jsonl"

if test "$mode" = oracle; then
  export INTERNVLA_T4_PHASE_OVERRIDE=continuous_oracle
  export INTERNVLA_T4_ORACLE_DATASET_ROOT="$dataset_root"
  export INTERNVLA_ORACLE_DATASET_ROOT="$dataset_root"
else
  case "$dataset_episode_count" in
    1|3)
      # Both the bounded READY canary and the natural fixed-five run use the
      # frozen five-episode source contract.  Candidate screen1/screen3 use the
      # official completion_sim bounded pilot-count override, because the
      # official canary identity is fixed at exactly five episodes.
      export INTERNVLA_T4_PHASE_OVERRIDE=pilot
      ;;
    5)
      # The original fixed-five keeps its historical official canary identity
      # and count for byte-for-byte compatible baseline behavior.
      export INTERNVLA_T4_PHASE_OVERRIDE=canary
      unset INTERNVLA_T4_EXPECTED_COUNT
      ;;
    10|20)
      export INTERNVLA_T4_PHASE_OVERRIDE=pilot
      ;;
    *)
      echo "T5 model evaluation requires 1/3 bounded pilot, 5 canary, or 10/20 pilot episodes" >&2
      exit 64
      ;;
  esac
  export INTERNVLA_T4_MODEL_DATASET_ROOT="$dataset_root"
fi

if test "$isaac_sensor_profile" = lane_b_revc_smoke; then
  revc_probe_timeout_sec="${INTERNNAV_T5_REVC_SMOKE_TIMEOUT_SEC:-600}"
  [[ "$revc_probe_timeout_sec" =~ ^[1-9][0-9]*$ ]]
  ((revc_probe_timeout_sec >= 30 && revc_probe_timeout_sec <= 900))
  test ! -e "$INTERNVLA_T5_REVC_SNAPSHOT_REQUEST_PATH"
  test ! -e "$INTERNVLA_T5_REVC_SNAPSHOT_ACK_PATH"
  test "$INTERNVLA_T4_RESULT_ROOT" = "$result_root/evaluator"
  mkdir -p "$INTERNVLA_T4_RESULT_ROOT"
  test ! -e "$INTERNVLA_T4_RESULT_ROOT/revc_snapshot_smoke.json"
  setsid python3 -u "$root/scripts/probe_t5_revc_snapshot_smoke.py" \
    --result-root "$INTERNVLA_T4_RESULT_ROOT" \
    --order-manifest "$INTERNVLA_MODEL_EPISODE_ORDER_MANIFEST" \
    --contract "$INTERNVLA_T5_REVC_CAMERA_CONFIG" \
    --request "$INTERNVLA_T5_REVC_SNAPSHOT_REQUEST_PATH" \
    --ack "$INTERNVLA_T5_REVC_SNAPSHOT_ACK_PATH" \
    --output "$INTERNVLA_T4_RESULT_ROOT/revc_snapshot_smoke.json" \
    --run-token "$run_token" \
    --timeout-sec "$revc_probe_timeout_sec" \
    >"$result_root/logs/revc_snapshot_smoke.log" 2>&1 &
  revc_probe_pid=$!
  record_pid_event revc_snapshot_probe "$revc_probe_pid" started \
    "$result_root/logs/revc_snapshot_smoke.log"
elif test "$isaac_sensor_profile" = lane_b_revc_fixed5_capture; then
  # The probe starts before READY so it can capture the first natural episode.
  # Cover the 1200 s outer READY allowance plus the 21600 s fixed-five window.
  revc_probe_timeout_sec="${INTERNNAV_T5_REVC_FIXED5_TIMEOUT_SEC:-24000}"
  [[ "$revc_probe_timeout_sec" =~ ^[1-9][0-9]*$ ]]
  ((revc_probe_timeout_sec >= 300 && revc_probe_timeout_sec <= 24000))
  test ! -e "$INTERNVLA_T5_REVC_SNAPSHOT_REQUEST_PATH"
  test ! -e "$INTERNVLA_T5_REVC_SNAPSHOT_ACK_PATH"
  test "$INTERNVLA_T4_RESULT_ROOT" = "$result_root/evaluator"
  mkdir -p "$INTERNVLA_T4_RESULT_ROOT"
  test ! -e "$revc_probe_output"
  setsid python3 -u "$root/scripts/probe_t5_revc_fixed5_capture.py" \
    --result-root "$INTERNVLA_T4_RESULT_ROOT" \
    --order-manifest "$INTERNVLA_MODEL_EPISODE_ORDER_MANIFEST" \
    --contract "$INTERNVLA_T5_REVC_CAMERA_CONFIG" \
    --request "$INTERNVLA_T5_REVC_SNAPSHOT_REQUEST_PATH" \
    --ack "$INTERNVLA_T5_REVC_SNAPSHOT_ACK_PATH" \
    --output "$revc_probe_output" \
    --run-token "$run_token" \
    --timeout-sec "$revc_probe_timeout_sec" \
    >"$revc_probe_log" 2>&1 &
  revc_probe_pid=$!
  record_pid_event revc_snapshot_probe "$revc_probe_pid" started \
    "$revc_probe_log"
elif test "$step3_timeout_advisor" = 1; then
  test ! -e "$INTERNVLA_T5_REVC_SNAPSHOT_REQUEST_PATH"
  test ! -e "$INTERNVLA_T5_REVC_SNAPSHOT_ACK_PATH"
  test "$INTERNVLA_T4_RESULT_ROOT" = "$result_root/evaluator"
  mkdir -p "$INTERNVLA_T4_RESULT_ROOT"
  advisor_python_deps="$result_root/runtime_deps/python3.12"
  advisor_pyzmq_source=/home/song/env_isaacsim/lib/python3.12/site-packages
  test -d "$advisor_pyzmq_source/zmq"
  test -d "$advisor_pyzmq_source/pyzmq.libs"
  test -d "$advisor_pyzmq_source/pyzmq-27.1.0.dist-info"
  test ! -e "$advisor_python_deps"
  mkdir -p "$advisor_python_deps"
  cp -a "$advisor_pyzmq_source/zmq" "$advisor_pyzmq_source/pyzmq.libs" \
    "$advisor_pyzmq_source/pyzmq-27.1.0.dist-info" "$advisor_python_deps/"
  step3_timeout_advisor_pid_file="$result_root/pids/step3_timeout_advisor.container.pid"
  setsid docker exec --user admin --workdir "$root" \
    -e "ROS_DOMAIN_ID=$ros_domain_id" -e "ROS_NAMESPACE=$lane_namespace" \
    -e ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST -e "ROS_STATIC_PEERS=$edge_ip" \
    -e ROS_LOCALHOST_ONLY=0 -e "PYTHONPATH=$root:$advisor_python_deps" \
    -e "STEP3_TIMEOUT_ADVISOR_PID_FILE=$step3_timeout_advisor_pid_file" \
    "$container" bash -lc '
      set -eo pipefail
      set +u
      source /opt/ros/jazzy/setup.bash
      source /workspaces/isaac/install/setup.bash
      set -u
      python3 -c "import rclpy, zmq, slow_planner, scripts.probe_t5_revc_snapshot_smoke"
      printf "%s\n" "$$" >"$STEP3_TIMEOUT_ADVISOR_PID_FILE"
      exec python3 -u "$1/scripts/t5_step3_timeout_advisor_node.py" \
        --endpoint "$2" --result-root "$3" --contract "$4" \
        --request "$5" --ack "$6" --output "$7" --deadline-sec 12
    ' bash "$root" \
    "${INTERNVLA_T5_STEP3_TIMEOUT_ENDPOINT:-tcp://$edge_ip:8200}" \
    "$INTERNVLA_T4_RESULT_ROOT" "$INTERNVLA_T5_REVC_CAMERA_CONFIG" \
    "$INTERNVLA_T5_REVC_SNAPSHOT_REQUEST_PATH" \
    "$INTERNVLA_T5_REVC_SNAPSHOT_ACK_PATH" \
    "$INTERNVLA_T4_RESULT_ROOT/step3_timeout_advice.jsonl" \
    >"$result_root/logs/step3_timeout_advisor.log" 2>&1 &
  step3_timeout_advisor_pid=$!
  record_pid_event step3_timeout_advisor "$step3_timeout_advisor_pid" started \
    "$result_root/logs/step3_timeout_advisor.log"
  for _ in $(seq 1 200); do
    test -s "$step3_timeout_advisor_pid_file" && break
    leader_is_alive "$step3_timeout_advisor_pid" || break
    sleep 0.1
  done
  test -s "$step3_timeout_advisor_pid_file"
  step3_timeout_advisor_container_pid="$(cat "$step3_timeout_advisor_pid_file")"
  [[ "$step3_timeout_advisor_container_pid" =~ ^[1-9][0-9]*$ ]]
  record_pid_event step3_timeout_advisor_container \
    "$step3_timeout_advisor_container_pid" started \
    "$result_root/logs/step3_timeout_advisor.log" container
fi

write_health_state STARTING_SIM
evaluator_log="$result_root/logs/evaluator_outer.log"
: >"$evaluator_log"
evaluator_cycle_index=1
evaluator_cycle_reaped=0
evaluator_cycle_exit_code=125

start_evaluator_cycle() {
  local attempt continuation_reset=0
  printf -v attempt '%03d' "$evaluator_cycle_index"
  if ((evaluator_cycle_index > 1)); then
    # A 600-sim-second soak may outlive the frozen five-episode evaluator.
    # Give every continuation a fresh attempt directory, upstream task name,
    # IPC token and order manifest while retaining the same frozen dataset.
    export INTERNVLA_MODEL_EPISODE_ORDER_MANIFEST="$result_root/ordered_episode_manifest_soak_${attempt}.json"
    test ! -e "$INTERNVLA_MODEL_EPISODE_ORDER_MANIFEST"
    continuation_reset=1
  fi
  INTERNVLA_T5_EVALUATOR_CONTINUATION_RESET="$continuation_reset" \
    setsid bash "$root/scripts/run_t4_sensor_gate.sh" t4_4 "$mode" "$attempt" \
    >>"$evaluator_log" 2>&1 &
  gate_pid=$!
  evaluator_cycle_reaped=0
  evaluator_cycle_exit_code=125
  record_pid_event evaluator "$gate_pid" "cycle_${attempt}_started" "$evaluator_log"
}

reap_evaluator_cycle() {
  local attempt ipc_token socket_name probe_rc=0
  test "$evaluator_cycle_reaped" = 0 || return 0
  printf -v attempt '%03d' "$evaluator_cycle_index"
  if wait "$gate_pid"; then
    evaluator_cycle_exit_code=0
  else
    evaluator_cycle_exit_code=$?
  fi
  evaluator_cycle_reaped=1
  record_pid_event evaluator "$gate_pid" "cycle_${attempt}_natural_exit" "$evaluator_log"
  test "$evaluator_cycle_exit_code" = 0 || return 1
  group_is_alive "$gate_pid" || probe_rc=$?
  test "$probe_rc" = 1
  ipc_token="$(printf '%s' "${task_run_label}_${mode}_${attempt}" | sha256sum | cut -c1-16)"
  for socket_name in agent oracle controller; do
    test ! -S "$ipc_alias/${ipc_token}_${socket_name}.sock"
  done
}

ensure_engineering_evaluator_running() {
  local completed_clock_ns completed_received_step_count
  leader_is_alive "$gate_pid" && return 0
  reap_evaluator_cycle
  # A fast wall-time canary may finish the frozen five before its READY+60 s
  # observation window closes.  A clean natural exit is not a runtime error:
  # keep the health/clock monitors alive and continue counting new errors until
  # the wall cutoff.  Sim-time soak still starts a continuation evaluator so it
  # can accumulate the full configured simulation duration.
  test "$engineering_canary_timebase" = sim || return 0
  read -r completed_clock_ns completed_received_step_count \
    < <(python3 - "$clock_live_state" <<'PY'
import json, sys, time
from pathlib import Path

path = Path(sys.argv[1])
not_before = time.time()
deadline = time.monotonic() + 15.0
previous = None
while time.monotonic() < deadline:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        time.sleep(0.6)
        continue
    stamp = value.get("last_clock_ns")
    received = value.get("received_step_count")
    updated = value.get("updated_unix")
    valid = (
        isinstance(stamp, int)
        and not isinstance(stamp, bool)
        and stamp > 0
        and isinstance(received, int)
        and not isinstance(received, bool)
        and received > 0
        and isinstance(updated, (int, float))
        and not isinstance(updated, bool)
        and updated >= not_before
        and value.get("regression_count") == 0
        and value.get("invalid_count") == 0
    )
    current = (stamp, received, updated) if valid else None
    if (
        current is not None
        and previous is not None
        and current[:2] == previous[:2]
        and current[2] > previous[2]
    ):
        print(stamp, received)
        break
    previous = current
    time.sleep(0.6)
else:
    raise SystemExit("completed evaluator clock did not quiesce after reap")
PY
)
  [[ "$completed_clock_ns" =~ ^[1-9][0-9]*$ ]]
  [[ "$completed_received_step_count" =~ ^[1-9][0-9]*$ ]]
  ((completed_received_step_count >= canary_last_received_step_count))
  ((completed_clock_ns >= canary_last_clock_ns))
  if ((completed_received_step_count > canary_last_received_step_count)); then
    canary_accumulated_sim_ns=$((
      canary_accumulated_sim_ns + completed_clock_ns - canary_last_clock_ns
    ))
  fi
  canary_last_clock_ns="$completed_clock_ns"
  canary_last_received_step_count="$completed_received_step_count"
  if ((canary_accumulated_sim_ns >= canary_target_clock_ns)); then
    return 0
  fi
  evaluator_cycle_index=$((evaluator_cycle_index + 1))
  export INTERNVLA_T5_SIM_CLOCK_START_NS="$completed_clock_ns"
  # One camera observation consumes a freshly advanced simulation step, so the
  # persistent clock receiver count is a strict upper bound on the completed
  # evaluator's local camera sequence.  Seed the next process above that bound.
  export INTERNVLA_T5_CAMERA_SOURCE_SEQUENCE_START="$completed_received_step_count"
  start_evaluator_cycle
  # The next simulator process seeds its local clock from a new epoch.  Do not
  # count the cross-process epoch jump as simulated execution time.
  canary_clock_epoch_pending=1
}

start_evaluator_cycle

# INFRA_READY means only the isolated container and sole /clock publisher are
# alive.  Do not publish READY until Kit has selected the exact physical GPU,
# a real simulated depth frame has completed a DGX controller round trip, and
# both real data planes have completed application-level round trips after the
# gate starts.  Do not open raw probe connections to the single-client model
# ingress: they queue behind the active evaluator and can consume the reset
# reconnect slot.
runtime_ready_timeout="${INTERNVLA_T5_RUNTIME_READY_TIMEOUT_SEC:-900}"
[[ "$runtime_ready_timeout" =~ ^[1-9][0-9]*$ ]]
runtime_ready_deadline=$((SECONDS + runtime_ready_timeout))
kit_ready_tmp="$result_root/runtime/tmp/kit_gpu_ready_audit.json"
kit_ready=0
application_ready=0
model_action_count=0
model_step_error_count=0
evaluator_total_path_count=-1
order_manifest_ready=0
canary_consecutive_action_count=0
canary_nav2_timeout_warning_count=0
canary_nonwarning_error_count=0
ready_model_action_count=0
ready_model_step_error_count=0
ready_safe_stop_timeout_warning_count=0
ready_fatal_error_count=0
canary_previous_clock_received=-1
canary_clock_progress_observed=0
canary_previous_action_count=0
canary_last_action_progress_seconds=-1
canary_sensor_action_progress_age_sec=-1
while (( SECONDS < runtime_ready_deadline )); do
  leader_is_alive "$gate_pid" || break
  if test "$kit_ready" = 0; then
    rm -f "$kit_ready_tmp"
    if python3 "$root/scripts/audit_t5_kit_gpu_log.py" \
        --log "$result_root/logs/evaluator_outer.log" \
        --expected-active-gpu "$gpu" \
        --expected-gpu-uuid "$host_gpu_uuid" \
        --expected-pci-bus-id "$host_gpu_pci_bus_id" \
        --output "$kit_ready_tmp" >/dev/null 2>&1; then
      mv "$kit_ready_tmp" "$result_root/health/kit_gpu_ready_audit.json"
      kit_ready=1
    fi
  fi
  model_progress="$(read_agent_application_log_counts \
    "$result_root/logs/evaluator_outer.log" "$mode")"
  read -r model_action_count model_step_error_count evaluator_total_path_count \
    <<<"$model_progress"
  [[ "$model_action_count" =~ ^[0-9]+$ ]]
  [[ "$model_step_error_count" =~ ^[0-9]+$ ]]
  [[ "$evaluator_total_path_count" =~ ^-?[0-9]+$ ]]
  if test "$engineering_canary_sec" != 0; then
    canary_model_state="$(read_engineering_canary_application_log_state \
      "$result_root/logs/evaluator_outer.log" "$mode")"
    read -r model_action_count model_step_error_count evaluator_total_path_count \
      canary_consecutive_action_count canary_nav2_timeout_warning_count \
      canary_nonwarning_error_count <<<"$canary_model_state"
    [[ "$canary_consecutive_action_count" =~ ^[0-9]+$ ]]
    [[ "$canary_nav2_timeout_warning_count" =~ ^[0-9]+$ ]]
    [[ "$canary_nonwarning_error_count" =~ ^[0-9]+$ ]]
    if test "$model_action_count" -lt "$canary_previous_action_count"; then
      echo "engineering canary readiness failed: model action count rolled back" >&2
      break
    fi
    if test "$model_action_count" -gt "$canary_previous_action_count"; then
      canary_previous_action_count="$model_action_count"
      canary_last_action_progress_seconds=$SECONDS
    fi
    # Only the two exact safe-stop Nav2 timeout signatures may be WARN before
    # READY.  Every other application error remains fatal.
    if test "$canary_nav2_timeout_warning_count" \
        -gt "$engineering_canary_pre_ready_nav2_timeout_limit"; then
      echo "engineering canary readiness failed: pre-READY Nav2 timeout warning limit exceeded" >&2
      break
    fi
    if test "$canary_nonwarning_error_count" != 0; then
      echo "engineering canary readiness failed: fatal application error before READY" >&2
      break
    fi
    canary_current_clock_received="$(
      engineering_canary_fresh_sensor_clock_count 2>/dev/null || true
    )"
    if [[ "$canary_current_clock_received" =~ ^[1-9][0-9]*$ ]]; then
      if test "$canary_previous_clock_received" -ge 0 && \
          test "$canary_current_clock_received" \
            -gt "$canary_previous_clock_received"; then
        canary_clock_progress_observed=1
      fi
      canary_previous_clock_received="$canary_current_clock_received"
    fi
  fi
  if test "$mode" = model; then
    if test -s "$INTERNVLA_MODEL_EPISODE_ORDER_MANIFEST"; then
      order_manifest_ready=1
    fi
  else
    order_manifest_ready=1
  fi
  if test "$kit_ready" = 1 && \
      test -s "$result_root/health/sensor_frames.jsonl" && \
      test "$evaluator_total_path_count" = "$dataset_episode_count" && \
      test "$order_manifest_ready" = 1 && \
      runtime_clock_authority_ready; then
    if test "$engineering_canary_sec" != 0; then
      if test "$canary_consecutive_action_count" -ge 2 && \
          test "$canary_clock_progress_observed" = 1 && \
          [[ "$canary_current_clock_received" =~ ^[1-9][0-9]*$ ]] && \
          test "$canary_last_action_progress_seconds" -ge 0 && \
          test "$((SECONDS - canary_last_action_progress_seconds))" -le 15; then
        canary_sensor_action_progress_age_sec=$((
          SECONDS - canary_last_action_progress_seconds
        ))
        ready_model_action_count="$model_action_count"
        ready_model_step_error_count="$model_step_error_count"
        application_ready=1
        break
      fi
    elif test "$model_action_count" -gt 0 && \
        test "$model_step_error_count" = 0; then
      application_ready=1
      break
    fi
  fi
  sleep 1
done
if test "$engineering_canary_sec" = 0 && test "$application_ready" = 1; then
  ready_model_action_count="$model_action_count"
  ready_model_step_error_count="$model_step_error_count"
fi
test "$kit_ready" = 1
test "$application_ready" = 1
test "$order_manifest_ready" = 1
runtime_clock_authority_ready
test -s "$result_root/health/sensor_frames.jsonl"
capture_runtime_gpu_evidence
if test "$engineering_canary_sec" != 0; then
  wait_for_engineering_canary_ready_snapshot
  python3 - "$result_root/health/pre_ready_warnings.json" \
    "$canary_nav2_timeout_warning_count" \
    "$engineering_canary_pre_ready_nav2_timeout_limit" \
    "$canary_nonwarning_error_count" "$ready_model_action_count" \
    "$ready_model_step_error_count" <<'PY'
import json
import sys
import time
from pathlib import Path

warning_count = int(sys.argv[2])
nonwarning_count = int(sys.argv[4])
payload = {
    "schema_version": 1,
    "status": "WARN" if warning_count or nonwarning_count else "PASS",
    "profile": "engineering_canary",
    "warning_kind": "pre_ready_safe_stop_nav2_timeout",
    "warning_count": warning_count,
    "warning_limit": int(sys.argv[3]),
    "nonwarning_error_count": nonwarning_count,
    "safe_stop_required": True,
    "matched_error_texts": [
        "RuntimeError('local InternVLA step failed: 4 Nav2 goal/cmd_vel timeout')",
        "RuntimeError('local InternVLA step failed: 4 timeout waiting for step goal acceptance')",
    ],
    "ready_model_action_count": int(sys.argv[5]),
    "ready_model_step_error_count": int(sys.argv[6]),
    "recorded_unix": time.time(),
}
Path(sys.argv[1]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
fi
python3 - "$result_root/health/sensor_frames.jsonl" "$identity_prefix" \
  "$edge_ip" "$controller_port" "$evaluator_port" "$mode" \
  "$result_root/health/runtime_readiness_evidence.json" \
  "$model_action_count" "$model_step_error_count" \
  "$evaluator_total_path_count" "$dataset_episode_count" \
  "${INTERNVLA_MODEL_EPISODE_ORDER_MANIFEST:-}" "$execution_profile" \
  "$canary_consecutive_action_count" "$canary_nav2_timeout_warning_count" \
  "$canary_nonwarning_error_count" \
  "$engineering_canary_pre_ready_nav2_timeout_limit" "$clock_live_state" \
  "$canary_clock_progress_observed" \
  "$canary_sensor_action_progress_age_sec" <<'PY'
import hashlib, json, math, sys, time
from pathlib import Path

sensor_path = Path(sys.argv[1])
sensor = json.loads(sensor_path.read_text(encoding="utf-8"))
mode = sys.argv[6]
engineering_canary = sys.argv[13] == "engineering_canary"
clock = json.loads(Path(sys.argv[18]).read_text(encoding="utf-8"))
now = time.time()
clock_age_seconds = now - float(clock.get("updated_unix", 0.0))
sensor_action_progress_age_seconds = int(sys.argv[20])
order_manifest_evidence = None
if mode == "model":
    order_path = Path(sys.argv[12])
    if order_path.is_symlink() or not order_path.is_file():
        raise SystemExit("model episode order manifest is not a regular file")
    order = json.loads(order_path.read_text(encoding="utf-8"))
    ordered_ids = order.get("ordered_episode_ids")
    order_manifest_pass = (
        order.get("schema_version") == 1
        and order.get("status") == "PASS"
        and order.get("dataset_episode_count") == int(sys.argv[11])
        and isinstance(ordered_ids, list)
        and len(ordered_ids) == int(sys.argv[11])
    )
    sensor_order_match = order_manifest_pass and sensor.get("episode_id") == (
        sys.argv[2] + str(ordered_ids[0])
    )
    order_manifest_evidence = {
        "path": str(order_path),
        "sha256": hashlib.sha256(order_path.read_bytes()).hexdigest(),
        "first_ordered_episode_id": str(ordered_ids[0])
        if order_manifest_pass else None,
    }
else:
    order_manifest_pass = True
    sensor_order_match = True
profile_error_policy_pass = (
    int(sys.argv[9]) == 0
    if not engineering_canary
    else (
        int(sys.argv[15]) <= int(sys.argv[17])
    )
)
canary_freshness_pass = (
    not engineering_canary
    or (
        int(sys.argv[14]) >= 2
        and 0 <= sensor_action_progress_age_seconds <= 15
        and clock.get("status") == "RUNNING"
        and isinstance(clock.get("last_clock_ns"), int)
        and clock.get("last_clock_ns") > 0
        and isinstance(clock.get("received_step_count"), int)
        and clock.get("received_step_count") > 0
        and isinstance(clock.get("publish_count"), int)
        and clock.get("publish_count") > 0
        and int(sys.argv[19]) == 1
        and clock.get("regression_count") == 0
        and clock.get("invalid_count") == 0
        and math.isfinite(clock_age_seconds)
        and 0.0 <= clock_age_seconds <= 15.0
    )
)
checks = {
    "sensor_ack_pass": sensor.get("status") == "PASS",
    "real_sensor_evidence": sensor.get("evidence") == "real_sim_depth_frame_acknowledged_by_dgx_controller",
    "lane_identity_prefix": str(sensor.get("episode_id", "")).startswith(sys.argv[2]),
    "controller_endpoint": sensor.get("controller_endpoint") == f"tcp://{sys.argv[3]}:{sys.argv[4]}",
    "controller_application_round_trip_after_gate": True,
    "agent_application_round_trip_after_gate": int(sys.argv[8]) > 0,
    "agent_readiness_errors_accepted_by_profile": profile_error_policy_pass,
    "engineering_canary_consecutive_actions_at_ready": (
        not engineering_canary or int(sys.argv[14]) >= 2
    ),
    "engineering_canary_pre_ready_nav2_timeout_warnings_bounded": (
        not engineering_canary or int(sys.argv[15]) <= int(sys.argv[17])
    ),
    "engineering_canary_pre_ready_fatal_errors_absent": (
        not engineering_canary or int(sys.argv[16]) == 0
    ),
    "engineering_canary_sensor_and_clock_fresh_at_ready": canary_freshness_pass,
    "fresh_evaluator_dataset_count": int(sys.argv[10]) == int(sys.argv[11]),
    "model_episode_order_manifest": order_manifest_pass,
    "sensor_matches_first_materialized_episode": sensor_order_match,
    "sole_clock_publisher": True,
    "kit_gpu_identity_audit": True,
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "mode": sys.argv[6],
    "controller_endpoint": f"tcp://{sys.argv[3]}:{sys.argv[4]}",
    "client_endpoint": f"tcp://{sys.argv[3]}:{sys.argv[5]}",
    "agent_action_success_marker_count": int(sys.argv[8]),
    "agent_step_error_marker_count": int(sys.argv[9]),
    "ready_consecutive_action_count": int(sys.argv[14]),
    "pre_ready_nav2_timeout_warning_count": int(sys.argv[15]),
    "pre_ready_nonwarning_error_count": int(sys.argv[16]),
    "pre_ready_nav2_timeout_warning_limit": int(sys.argv[17]),
    "sensor_freshness_basis": (
        "recent_success_action_consumed_rgbd_observation"
        if engineering_canary else None
    ),
    "sensor_action_progress_age_seconds_at_ready": (
        sensor_action_progress_age_seconds if engineering_canary else None
    ),
    "clock_state_age_seconds_at_ready": clock_age_seconds,
    "clock_progress_observed_before_ready": int(sys.argv[19]) == 1,
    "evaluator_total_path_count": int(sys.argv[10]),
    "expected_dataset_episode_count": int(sys.argv[11]),
    "model_episode_order_manifest": order_manifest_evidence,
    "sensor_frame_record": str(sensor_path),
    "checks": checks,
    "recorded_unix": time.time(),
}
Path(sys.argv[7]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if payload["status"] != "PASS":
    raise SystemExit(f"runtime readiness evidence failed: {checks}")
PY
if test "$engineering_canary_sec" != 0; then
  # Re-snapshot immediately before publishing READY.  GPU/readiness evidence
  # generation can overlap an evaluator marker.  An allowed timeout remains a
  # WARN and restarts the two-success streak under the original deadline.
  wait_for_engineering_canary_ready_snapshot
  ready_safe_stop_timeout_warning_count="$canary_nav2_timeout_warning_count"
  ready_fatal_error_count="$canary_nonwarning_error_count"
  python3 - "$result_root/health/runtime_readiness_evidence.json" \
    "$result_root/health/pre_ready_warnings.json" \
    "$model_action_count" "$model_step_error_count" \
    "$canary_consecutive_action_count" "$canary_nav2_timeout_warning_count" \
    "$canary_nonwarning_error_count" \
    "$canary_sensor_action_progress_age_sec" "$clock_live_state" <<'PY'
import json
import math
import os
import sys
import time
from pathlib import Path

readiness_path = Path(sys.argv[1])
warning_path = Path(sys.argv[2])
readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
warnings = json.loads(warning_path.read_text(encoding="utf-8"))
sensor_action_progress_age_seconds = int(sys.argv[8])
clock = json.loads(Path(sys.argv[9]).read_text(encoding="utf-8"))
clock_age_seconds = time.time() - float(clock.get("updated_unix", 0.0))
if not 0 <= sensor_action_progress_age_seconds <= 15:
    raise SystemExit("successful observation action became stale before READY publication")
if not math.isfinite(clock_age_seconds) or not 0.0 <= clock_age_seconds <= 15.0:
    raise SystemExit("clock state became stale before READY publication")
readiness.update({
    "agent_action_success_marker_count": int(sys.argv[3]),
    "agent_step_error_marker_count": int(sys.argv[4]),
    "ready_consecutive_action_count": int(sys.argv[5]),
    "pre_ready_nav2_timeout_warning_count": int(sys.argv[6]),
    "pre_ready_nonwarning_error_count": int(sys.argv[7]),
    "sensor_freshness_basis": "recent_success_action_consumed_rgbd_observation",
    "sensor_action_progress_age_seconds_at_ready": sensor_action_progress_age_seconds,
    "clock_state_age_seconds_at_ready": clock_age_seconds,
    "ready_baseline_refreshed_unix": time.time(),
})
warnings.update({
    "status": "WARN" if int(sys.argv[6]) or int(sys.argv[7]) else "PASS",
    "warning_count": int(sys.argv[6]),
    "nonwarning_error_count": int(sys.argv[7]),
    "ready_model_action_count": int(sys.argv[3]),
    "ready_model_step_error_count": int(sys.argv[4]),
    "ready_baseline_refreshed_unix": time.time(),
})
for path, payload in ((readiness_path, readiness), (warning_path, warnings)):
    temporary = path.with_name("." + path.name + ".ready.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
PY
fi
if test "$isaac_sensor_profile" = lane_b_revc_smoke; then
  wait_revc_snapshot_probe
fi
write_health_state READY
python3 - "$health_socket" "$lane" "$result_root/health/ready_probe.json" \
  "$result_root/health/runtime_readiness_evidence.json" \
  "$result_root/health/runtime_gpu_evidence.json" \
  "$ready_model_action_count" "$ready_model_step_error_count" \
  "$canary_nav2_timeout_warning_count" \
  "$canary_nonwarning_error_count" "$canary_consecutive_action_count" \
  "$execution_profile" <<'PY'
import json, socket, sys, time
from pathlib import Path

client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.settimeout(2.0)
client.connect(sys.argv[1])
chunks = []
while True:
    chunk = client.recv(65536)
    if not chunk:
        break
    chunks.append(chunk)
client.close()
value = json.loads(b"".join(chunks))
readiness = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
gpu = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
if (
    value.get("lane") != sys.argv[2]
    or value.get("state") != "READY"
    or readiness.get("status") != "PASS"
    or gpu.get("status") != "PASS"
):
    raise SystemExit("lane health endpoint or runtime evidence is not READY/PASS")
Path(sys.argv[3]).write_text(json.dumps({
    "schema_version": 2,
    "status": "PASS",
    "lane": sys.argv[2],
    "endpoint": "unix://" + sys.argv[1],
    "observed_state": value["state"],
    "runtime_readiness_evidence": sys.argv[4],
    "runtime_gpu_evidence": sys.argv[5],
    "sensor_freshness_basis": readiness.get("sensor_freshness_basis"),
    "sensor_action_progress_age_seconds_at_ready": readiness.get(
        "sensor_action_progress_age_seconds_at_ready"
    ),
    "ready_model_action_count": int(sys.argv[6]),
    "ready_model_step_error_count": int(sys.argv[7]),
    "pre_ready_nav2_timeout_warning_count": int(sys.argv[8]),
    "ready_safe_stop_timeout_warning_count": int(sys.argv[8]),
    "ready_fatal_error_count": int(sys.argv[9]),
    "ready_consecutive_action_count": int(sys.argv[10]),
    "execution_profile": sys.argv[11],
    "probed_unix": time.time(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
online_ready=1
write_health_state RUNNING

# Engineering bring-up may stop deliberately after a bounded READY interval.
# This is not an episode/evidence PASS: it proves the lane's real Kit GPU,
# simulated sensor round trip, DGX endpoints, /clock authority, and cleanup
# boundary without manufacturing SIGTERM=143 as a failure.  The surrounding
# lane lease remains held until finalize has proved every residual is zero.
if test "$engineering_canary_sec" != 0; then
  canary_started_unix="$(python3 -c 'import time; print(time.time())')"
  canary_started_seconds=$SECONDS
  canary_wall_deadline=$((SECONDS + engineering_canary_sec))
  canary_wall_watchdog_deadline=$((
    SECONDS + engineering_canary_wall_watchdog_sec
  ))
  read -r canary_started_clock_ns canary_last_received_step_count \
    < <(python3 - "$clock_live_state" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
stamp = value.get("last_clock_ns")
received = value.get("received_step_count")
if (
    not isinstance(stamp, int)
    or isinstance(stamp, bool)
    or stamp <= 0
    or not isinstance(received, int)
    or isinstance(received, bool)
    or received <= 0
):
    raise SystemExit("engineering interval has no positive simulation clock")
print(stamp, received)
PY
)
  [[ "$canary_started_clock_ns" =~ ^[1-9][0-9]*$ ]]
  [[ "$canary_last_received_step_count" =~ ^[1-9][0-9]*$ ]]
  canary_target_clock_ns=$((engineering_canary_sec * 1000000000))
  canary_last_clock_ns="$canary_started_clock_ns"
  canary_accumulated_sim_ns=0
  canary_clock_epoch_pending=0
  canary_clock_epoch_count=1
  canary_clock_progress_wall_seconds=$SECONDS
  canary_clock_restart_deadline=0
  canary_fault_restart_deadline=0
  canary_samples="$result_root/health/engineering_canary_samples.jsonl"
  : >"$canary_samples"
  fault_restart_maintenance_active() {
    test "$fault_injection_profile" = completion_sim_minimal_v1 || return 1
    python3 - "$INTERNNAV_T5_FAULT_CONTROL_PATH" "$lane" <<'PY'
import json, sys
from pathlib import Path

try:
    value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
active = value.get("active")
valid = (
    value.get("profile") == "completion_sim_minimal_v1"
    and value.get("lane") == sys.argv[2]
    and isinstance(active, list)
    and len(active) == 1
    and isinstance(active[0], dict)
    and active[0].get("kind")
    in {"model_service_restart", "dgx_ros_node_restart"}
)
raise SystemExit(0 if valid else 1)
PY
  }
  sample_canary() {
    local wall_elapsed="$((SECONDS - canary_started_seconds))"
    local current_clock_ns current_received_step_count
    local fault_restart_active=0
    local sim_elapsed_ns sim_elapsed_seconds elapsed
    local sampled_model_action_count sampled_model_step_error_count
    local sampled_evaluator_total_path_count sampled_model_progress
    local sampled_consecutive_action_count sampled_safe_stop_timeout_count
    local sampled_fatal_error_count
    ensure_engineering_evaluator_running
    read -r current_clock_ns current_received_step_count \
      < <(python3 - "$clock_live_state" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
stamp = value.get("last_clock_ns")
received = value.get("received_step_count")
if (
    not isinstance(stamp, int)
    or isinstance(stamp, bool)
    or stamp <= 0
    or not isinstance(received, int)
    or isinstance(received, bool)
    or received <= 0
):
    raise SystemExit("engineering sample has no positive simulation clock")
print(stamp, received)
PY
)
    [[ "$current_clock_ns" =~ ^[1-9][0-9]*$ ]]
    [[ "$current_received_step_count" =~ ^[1-9][0-9]*$ ]]
    if fault_restart_maintenance_active; then
      fault_restart_active=1
    fi
    ((current_received_step_count >= canary_last_received_step_count))
    if ((current_received_step_count > canary_last_received_step_count)); then
      if test "$canary_clock_epoch_pending" = 1 || \
          ((current_clock_ns < canary_last_clock_ns)); then
        canary_clock_epoch_count=$((canary_clock_epoch_count + 1))
        canary_clock_epoch_pending=0
        canary_clock_restart_deadline=0
        canary_clock_progress_wall_seconds=$SECONDS
        canary_fault_restart_deadline=0
        canary_last_clock_ns="$current_clock_ns"
        canary_last_received_step_count="$current_received_step_count"
        test "$engineering_canary_timebase" != sim || return 0
      else
        canary_accumulated_sim_ns=$((
          canary_accumulated_sim_ns + current_clock_ns - canary_last_clock_ns
        ))
      fi
      canary_clock_progress_wall_seconds=$SECONDS
      canary_fault_restart_deadline=0
      canary_last_clock_ns="$current_clock_ns"
      canary_last_received_step_count="$current_received_step_count"
    else
      if ((fault_restart_active)); then
        # The director clears active before the client reports recovery.  Keep
        # the observed bounded hold until the simulation clock advances again.
        canary_fault_restart_deadline=$((
          canary_clock_progress_wall_seconds + engineering_evaluator_restart_timeout_sec
        ))
      fi
      if test "$canary_clock_epoch_pending" = 1; then
        if test "$canary_clock_restart_deadline" = 0; then
          canary_clock_restart_deadline=$((
            SECONDS + engineering_evaluator_restart_timeout_sec
          ))
        fi
        ((SECONDS < canary_clock_restart_deadline)) || {
          echo "restarted evaluator produced no simulation step before its wall liveness deadline" >&2
          return 75
        }
      elif ((SECONDS - canary_clock_progress_wall_seconds > 60)); then
        if ((canary_fault_restart_deadline > 0)); then
          ((SECONDS <= canary_fault_restart_deadline)) || {
            echo "fault restart made no simulation progress within its wall liveness deadline" >&2
            return 75
          }
        else
          echo "simulation clock made no progress for 60 wall seconds" >&2
          return 75
        fi
      fi
      test "$engineering_canary_timebase" != sim || return 0
    fi
    sim_elapsed_ns="$canary_accumulated_sim_ns"
    printf -v sim_elapsed_seconds '%d.%09d' \
      "$((sim_elapsed_ns / 1000000000))" "$((sim_elapsed_ns % 1000000000))"
    if test "$engineering_canary_timebase" = sim; then
      elapsed="$sim_elapsed_seconds"
    else
      elapsed="$wall_elapsed"
    fi
    sampled_model_progress="$(read_engineering_canary_application_log_state \
      "$result_root/logs/evaluator_outer.log" "$mode")"
    read -r sampled_model_action_count sampled_model_step_error_count \
      sampled_evaluator_total_path_count sampled_consecutive_action_count \
      sampled_safe_stop_timeout_count sampled_fatal_error_count \
      <<<"$sampled_model_progress"
    [[ "$sampled_model_action_count" =~ ^[0-9]+$ ]]
    [[ "$sampled_model_step_error_count" =~ ^[0-9]+$ ]]
    test "$sampled_evaluator_total_path_count" = "$dataset_episode_count"
    python3 - "$health_socket" "$clock_live_state" "$lane" \
      "$canary_samples" "$elapsed" "$sampled_model_action_count" \
      "$sampled_model_step_error_count" \
      "$sampled_evaluator_total_path_count" "$ready_model_action_count" \
      "$ready_model_step_error_count" "$engineering_canary_timebase" \
      "$wall_elapsed" "$sim_elapsed_seconds" "$evaluator_cycle_index" \
      "$canary_clock_epoch_count" "$sampled_safe_stop_timeout_count" \
      "$sampled_fatal_error_count" "$ready_safe_stop_timeout_warning_count" \
      "$ready_fatal_error_count" <<'PY'
import json, math, socket, sys, time
from pathlib import Path

health_socket, clock_path, lane, samples_path = sys.argv[1:5]
client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.settimeout(2.0)
client.connect(health_socket)
chunks = []
while True:
    chunk = client.recv(65536)
    if not chunk:
        break
    chunks.append(chunk)
client.close()
health = json.loads(b"".join(chunks))
clock = json.loads(Path(clock_path).read_text(encoding="utf-8"))
now = time.time()
clock_state_age_seconds = now - float(clock.get("updated_unix", 0.0))
if health.get("lane") != lane or health.get("state") != "RUNNING":
    raise SystemExit("engineering canary health endpoint is not RUNNING")
if (
    clock.get("status") != "RUNNING"
    or not isinstance(clock.get("last_clock_ns"), int)
    or not isinstance(clock.get("received_step_count"), int)
    or not isinstance(clock.get("publish_count"), int)
    or clock.get("regression_count") != 0
    or clock.get("invalid_count") != 0
    or not math.isfinite(clock_state_age_seconds)
    or clock_state_age_seconds < 0.0
    or clock_state_age_seconds > 15.0
):
    raise SystemExit("engineering canary live clock state is stale or invalid")
row = {
    "schema_version": 1,
    "elapsed_seconds": float(sys.argv[5]),
    "sampled_unix": now,
    "health_state": health["state"],
    "lane": lane,
    "clock": clock,
    "clock_state_age_seconds": clock_state_age_seconds,
    "model_action_count": int(sys.argv[6]),
    "model_step_error_count": int(sys.argv[7]),
    "evaluator_total_path_count": int(sys.argv[8]),
    "ready_model_action_count": int(sys.argv[9]),
    "ready_model_step_error_count": int(sys.argv[10]),
    "new_model_step_error_count": int(sys.argv[7]) - int(sys.argv[10]),
    "duration_timebase": sys.argv[11],
    "elapsed_wall_seconds": int(sys.argv[12]),
    "elapsed_sim_seconds": float(sys.argv[13]),
    "evaluator_cycle_index": int(sys.argv[14]),
    "sim_clock_epoch_count": int(sys.argv[15]),
    "safe_stop_timeout_warning_count": int(sys.argv[16]),
    "fatal_error_count": int(sys.argv[17]),
    "ready_safe_stop_timeout_warning_count": int(sys.argv[18]),
    "ready_fatal_error_count": int(sys.argv[19]),
    "new_safe_stop_timeout_warning_count": int(sys.argv[16]) - int(sys.argv[18]),
    "new_fatal_error_count": int(sys.argv[17]) - int(sys.argv[19]),
}
with Path(samples_path).open("a", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps(row, sort_keys=True) + "\n")
PY
    ensure_engineering_evaluator_running
  }
  runtime_clock_authority_ready
  sample_canary
  while true; do
    if test "$engineering_canary_timebase" = sim; then
      ((canary_accumulated_sim_ns >= canary_target_clock_ns)) && break
      ((SECONDS < canary_wall_watchdog_deadline)) || {
        echo "600 sim-s soak exceeded its wall liveness watchdog" >&2
        exit 124
      }
    else
      ((SECONDS >= canary_wall_deadline)) && break
    fi
    sleep 5
    sample_canary
  done
  canary_wall_observed_sec="$((SECONDS - canary_started_seconds))"
  canary_sim_observed_ns="$canary_accumulated_sim_ns"
  printf -v canary_sim_observed_seconds '%d.%09d' \
    "$((canary_sim_observed_ns / 1000000000))" \
    "$((canary_sim_observed_ns % 1000000000))"
  if test "$engineering_canary_timebase" = sim; then
    canary_selected_observed_seconds="$canary_sim_observed_seconds"
    canary_observed_sec=$((
      (canary_sim_observed_ns + 999999999) / 1000000000
    ))
  else
    canary_selected_observed_seconds="$canary_wall_observed_sec"
    canary_observed_sec="$canary_wall_observed_sec"
  fi
  leader_is_alive "$gate_pid" || reap_evaluator_cycle
  canary_cutoff_state="$(read_engineering_canary_application_log_state \
    "$result_root/logs/evaluator_outer.log" "$mode")"
  read -r canary_cutoff_model_action_count \
    canary_cutoff_model_step_error_count canary_cutoff_evaluator_total_path_count \
    canary_cutoff_consecutive_action_count \
    canary_cutoff_nav2_timeout_warning_count \
    canary_cutoff_nonwarning_error_count <<<"$canary_cutoff_state"
  test "$canary_cutoff_evaluator_total_path_count" = "$dataset_episode_count"
  python3 - "$result_root/health/engineering_canary_cutoff.json" \
    "$canary_selected_observed_seconds" "$canary_cutoff_model_action_count" \
    "$canary_cutoff_model_step_error_count" "$ready_model_action_count" \
    "$ready_model_step_error_count" \
    "$canary_cutoff_evaluator_total_path_count" \
    "$canary_cutoff_consecutive_action_count" \
    "$canary_cutoff_nav2_timeout_warning_count" \
    "$canary_cutoff_nonwarning_error_count" \
    "$engineering_canary_timebase" "$canary_wall_observed_sec" \
    "$canary_sim_observed_seconds" "$evaluator_cycle_index" \
    "$canary_clock_epoch_count" "$ready_safe_stop_timeout_warning_count" \
    "$ready_fatal_error_count" <<'PY'
import json
import sys
import time
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "status": "RECORDED",
    "elapsed_seconds": float(sys.argv[2]),
    "cutoff_model_action_count": int(sys.argv[3]),
    "cutoff_model_step_error_count": int(sys.argv[4]),
    "ready_model_action_count": int(sys.argv[5]),
    "ready_model_step_error_count": int(sys.argv[6]),
    "new_model_step_error_count": int(sys.argv[4]) - int(sys.argv[6]),
    "evaluator_total_path_count": int(sys.argv[7]),
    "consecutive_action_count": int(sys.argv[8]),
    "nav2_timeout_warning_count": int(sys.argv[9]),
    "nonwarning_error_count": int(sys.argv[10]),
    "safe_stop_timeout_warning_count": int(sys.argv[9]),
    "fatal_error_count": int(sys.argv[10]),
    "duration_timebase": sys.argv[11],
    "elapsed_wall_seconds": int(sys.argv[12]),
    "elapsed_sim_seconds": float(sys.argv[13]),
    "evaluator_cycle_count": int(sys.argv[14]),
    "sim_clock_epoch_count": int(sys.argv[15]),
    "ready_safe_stop_timeout_warning_count": int(sys.argv[16]),
    "ready_fatal_error_count": int(sys.argv[17]),
    "new_safe_stop_timeout_warning_count": int(sys.argv[9]) - int(sys.argv[16]),
    "new_fatal_error_count": int(sys.argv[10]) - int(sys.argv[17]),
    "recorded_unix": time.time(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  leader_is_alive "$gate_pid" || reap_evaluator_cycle
  stop_host_group evaluator "$gate_pid" \
    "$result_root/logs/evaluator_outer.log"
  runtime_clock_authority_ready
  python3 "$root/scripts/analyze_t5_engineering_canary.py" \
    --samples "$canary_samples" \
    --ready "$result_root/health/ready_probe.json" \
    --cutoff "$result_root/health/engineering_canary_cutoff.json" \
    --output "$result_root/engineering_canary.json" \
    --lane "$lane" --mode "$mode" \
    --configured-sec "$engineering_canary_sec" \
    --observed-sec "$canary_observed_sec" \
    --timebase "$engineering_canary_timebase" \
    --started-unix "$canary_started_unix"
  runtime_clock_authority_ready
  record_pid_event evaluator "$gate_pid" canary_complete \
    "$result_root/logs/evaluator_outer.log"
  run_rc=0
  exit 0
fi

if test "$isaac_sensor_profile" = lane_b_revc_fixed5_capture; then
  while leader_is_alive "$gate_pid" && kill -0 "$revc_probe_pid" 2>/dev/null; do
    sleep 5
  done
  if leader_is_alive "$gate_pid" && ! kill -0 "$revc_probe_pid" 2>/dev/null; then
    set +e
    wait_revc_snapshot_probe
    revc_early_rc=$?
    set -e
    if test "$revc_early_rc" != 0; then
      echo "Rev-C fixed-five probe failed before the evaluator completed" >&2
      exit "$revc_early_rc"
    fi
  fi
fi
set +e
wait "$gate_pid"
run_rc=$?
set -e
record_pid_event evaluator "$gate_pid" exited "$result_root/logs/evaluator_outer.log"
if test "$run_rc" = 0 && \
    test "$isaac_sensor_profile" = lane_b_revc_fixed5_capture; then
  set +e
  wait_revc_fixed5_after_evaluator
  revc_wait_rc=$?
  set -e
  test "$revc_wait_rc" = 0 || run_rc="$revc_wait_rc"
fi
exit "$run_rc"
