#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
frontend_root="$repo_root/frontend"
config="${1:-$repo_root/configs/internnav_t5/lane_b_step3.yaml}"
domain_id="${ROS_DOMAIN_ID:-0}"
run_root="${2:-${E_MARS_SIM_FRONTEND_RUN_ROOT:-/tmp/e-mars-sim-frontend-$domain_id}}"
frontend_python="${E_MARS_FRONTEND_PYTHON:-python3}"
ros_python="${E_MARS_ROS_PYTHON:-python3}"
lock_path="${E_MARS_SIM_FRONTEND_LOCK:-/tmp/e_mars_sim_frontend_${domain_id}.lock}"
log_dir="$run_root/logs"

test -f "$frontend_root/pyproject.toml" || {
  echo "frontend submodule is not initialized; run: git submodule update --init --recursive" >&2
  exit 2
}
test -r "$config" || { echo "missing frontend config: $config" >&2; exit 2; }
command -v "$frontend_python" >/dev/null
command -v "$ros_python" >/dev/null
command -v setsid >/dev/null
command -v flock >/dev/null

mkdir -p "$log_dir"
exec 9>"$lock_path"
if ! flock -n 9; then
  echo "simulation frontend lock is already held: $lock_path" >&2
  exit 73
fi

if [[ -r /opt/ros/jazzy/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  set -u
fi
if [[ -n "${E_MARS_ROS_SETUP:-}" ]]; then
  test -r "$E_MARS_ROS_SETUP"
  set +u
  # shellcheck disable=SC1090
  source "$E_MARS_ROS_SETUP"
  set -u
fi

export PYTHONPATH="$frontend_root:$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export T5_LANE_B_RESULTS="$run_root"
export ROS_DOMAIN_ID="$domain_id"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"

children=()
cleanup() {
  local rc=$?
  trap - EXIT INT TERM HUP
  set +e
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
  exit "$rc"
}
trap cleanup EXIT INT TERM HUP

cd "$frontend_root"
setsid "$ros_python" -m slow_planner_frontend.ros_adapter --config "$config" \
  >"$log_dir/ros_adapter.log" 2>&1 &
children+=("$!")
setsid "$frontend_python" -m slow_planner_frontend --config "$config" \
  >"$log_dir/frontend.log" 2>&1 &
children+=("$!")

echo "E-MARS simulation panel: http://0.0.0.0:8300/"
echo "config=$config"
echo "run_root=$run_root"

while true; do
  for pid in "${children[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || exit $?
      exit 1
    fi
  done
  sleep 1
done
