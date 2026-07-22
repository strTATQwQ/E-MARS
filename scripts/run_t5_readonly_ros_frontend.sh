#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="${1:-$repo_root/configs/internnav_t5/lane_b_step3.yaml}"
results="${2:-$repo_root/results/t5_readonly_ros_frontend/live}"
frontend_python="${T5_FRONTEND_PYTHON:-/home/rail/ai-stack/venvs/step3-vl-10b-tf4.57.6/bin/python}"
start_d435="${T5_START_D435:-1}"
log_dir="$results/logs"
mkdir -p "$log_dir"

exec 9>/tmp/t5_readonly_ros_frontend_dgxb.lock
if ! flock -n 9; then
  echo "read-only frontend already owns /tmp/t5_readonly_ros_frontend_dgxb.lock" >&2
  exit 73
fi

if [[ -r /opt/ros/jazzy/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  set -u
fi
unitree_setup="${UNITREE_ROS_SETUP:-/home/rail/unitree_ros2_official_20260722/cyclonedds_ws/install_jazzy/local_setup.bash}"
unitree_auth="${UNITREE_DDS_AUTH_ENV:-/home/rail/.unitree_dds_auth/env.sh}"
[[ -r "$unitree_setup" ]] || { echo "missing Unitree ROS setup: $unitree_setup" >&2; exit 2; }
[[ -r "$unitree_auth" ]] || { echo "missing read-only DDS auth env: $unitree_auth" >&2; exit 2; }
# shellcheck disable=SC1090
set +u
source "$unitree_setup"
# shellcheck disable=SC1090
source "$unitree_auth"
set -u

[[ -x "$frontend_python" ]] || { echo "missing frontend Python: $frontend_python" >&2; exit 2; }
[[ -r "$config" ]] || { echo "missing frontend config: $config" >&2; exit 2; }
command -v python3 >/dev/null
command -v ros2 >/dev/null

export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export T5_LANE_B_RESULTS="$results"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

children=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${children[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done
  local deadline=$((SECONDS + 5))
  while (( SECONDS < deadline )); do
    local alive=0
    for pid in "${children[@]}"; do
      kill -0 "$pid" 2>/dev/null && alive=1
    done
    (( alive == 0 )) && break
    sleep 0.2
  done
  for pid in "${children[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

if [[ "$start_d435" == 1 ]]; then
  setsid ros2 launch realsense2_camera rs_launch.py \
    camera_namespace:=check camera_name:=d435 \
    serial_no:="'207122079555'" \
    enable_color:=true enable_depth:=true \
    rgb_camera.color_profile:=640x480x15 \
    depth_module.depth_profile:=640x480x15 \
    enable_infra:=false enable_infra1:=false enable_infra2:=false \
    enable_gyro:=false enable_accel:=false \
    align_depth.enable:=false \
    >"$log_dir/d435.log" 2>&1 &
  children+=("$!")
fi

if ! command -v ffmpeg >/dev/null; then
  echo "WARN: ffmpeg missing; Go2 H.264 preview will report decoder_unavailable" >&2
fi

setsid python3 -m slow_planner_frontend.ros_adapter --config "$config" \
  >"$log_dir/ros_adapter.log" 2>&1 &
children+=("$!")

setsid "$frontend_python" -m slow_planner_frontend --config "$config" \
  >"$log_dir/frontend.log" 2>&1 &
children+=("$!")

echo "read-only dashboard starting on http://0.0.0.0:8300/"
echo "results=$results"

while true; do
  for pid in "${children[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || exit $?
      exit 1
    fi
  done
  sleep 1
done
