#!/usr/bin/env bash
set -eo pipefail

RUN_DIR=${1:?run directory required}
MAX_SEQUENCES=${2:-3}
SEQUENCE_INDICES=${SEQUENCE_INDICES-}
if [[ -z "$SEQUENCE_INDICES" && "$MAX_SEQUENCES" -lt 30 ]]; then
  SEQUENCE_INDICES=0,10,20
fi
mkdir -p "$RUN_DIR"
. /opt/ros/humble/setup.bash
. /home/song/dgx-unitree/isaac_vln_benchmark/ros2_ws/install/setup.bash
set -u
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}
export GO2_ENABLE_MOTION=0

adapter_pid=""
cleanup() {
  set +e
  [[ -z "$adapter_pid" ]] || kill "$adapter_pid" 2>/dev/null || true
  pkill -TERM -f '[/]lib/isaac_vln_benchmark/go2_benchmark_adapter_node' 2>/dev/null || true
}
trap cleanup EXIT
cleanup

ros2 run isaac_vln_benchmark go2_benchmark_adapter_node \
  --ros-args -p publish_synthetic_camera_fallback:=false >"$RUN_DIR/adapter.log" 2>&1 &
adapter_pid=$!
sleep 4

args=(
  --output "$RUN_DIR/result"
  --max-sequences "$MAX_SEQUENCES"
)
if [[ -n "$SEQUENCE_INDICES" ]]; then
  args+=(--sequence-indices "$SEQUENCE_INDICES")
fi
python3 /home/song/dgx-unitree/isaac_vln_benchmark/scripts/run_rgbd_tracking_sequences_live.py "${args[@]}" \
  | tee "$RUN_DIR/runner.log"
