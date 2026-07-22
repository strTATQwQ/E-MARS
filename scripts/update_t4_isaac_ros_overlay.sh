#!/usr/bin/env bash
set -euo pipefail

# Run only inside an already-held Isaac (or combined DGX->Isaac) lease.  This
# updates the isolated T4 Jazzy source overlay and regenerates typed interfaces;
# it never touches the frozen T3 workspace or a real-Go2 target.
CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
T4_WS="${INTERNVLA_T4_ISAAC_ROS_WS:-$HOME/internnav-t4/isaac_ros_ws_45}"
CONTAINER="${INTERNVLA_T4_CONTAINER_NAME:-internnav_t4_isaac_ros}"
RESULT_DIR="${INTERNVLA_T4_OVERLAY_UPDATE_RESULT_DIR:-}"

case "${INTERNNAV_T4_RESOURCE_LEASE_ACK:-}" in
  isaac|dgx+isaac) ;;
  *) echo "Isaac resource lease acknowledgement is required" >&2; exit 2 ;;
esac
test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim
test "${INTERNNAV_SIMULATION_TARGET:-}" = isaac
test -n "$RESULT_DIR"
test ! -e "$RESULT_DIR"

CONTROL_ROOT_REAL="$(realpath "$CONTROL_ROOT")"
T4_WS_REAL="$(realpath "$T4_WS")"
case "$T4_WS_REAL" in
  "$HOME/internnav-t4/isaac_ros_ws_45") ;;
  *) echo "unsafe T4 workspace: $T4_WS_REAL" >&2; exit 2 ;;
esac
test -d "$T4_WS_REAL/src"
test "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" = true
command -v rsync >/dev/null
mkdir -p "$RESULT_DIR/logs"

packages=(
  internvla_ros2_msgs
  internvla_ros2
  internvla_nav2_adapter
  internvla_go2_controller
  internvla_t4_sensors
  internvla_t4_recovery
)
for package in "${packages[@]}"; do
  source_dir="$CONTROL_ROOT_REAL/$package"
  destination="$T4_WS_REAL/src/$package"
  test -f "$source_dir/package.xml"
  test ! -L "$source_dir"
  test ! -L "$destination"
  destination_parent="$(realpath "$(dirname "$destination")")"
  test "$destination_parent" = "$T4_WS_REAL/src"
  test "$(realpath -m "$destination")" = "$T4_WS_REAL/src/$package"
  mkdir -p "$destination"
  rsync -a --delete --exclude '__pycache__/' --exclude '*.pyc' \
    "$source_dir/" "$destination/"
done

docker exec --user admin --workdir /workspaces/isaac "$CONTAINER" bash -lc '
  set -euo pipefail
  set +u
  source /opt/ros/jazzy/setup.bash
  test ! -f install/setup.bash || source install/setup.bash
  set -u
  colcon build --symlink-install --event-handlers console_direct+ \
    --cmake-args -DBUILD_TESTING=OFF \
    --packages-up-to internvla_t4_sensors internvla_t4_recovery
' >"$RESULT_DIR/logs/colcon.log" 2>&1

docker exec --user admin --workdir /workspaces/isaac "$CONTAINER" bash -lc '
  set -euo pipefail
  set +u
  source /opt/ros/jazzy/setup.bash
  source install/setup.bash
  set -u
  ros2 interface show internvla_ros2_msgs/srv/RecoveryControl >/dev/null
  python3 -c "from internvla_ros2_msgs.srv import RecoveryControl; import internvla_t4_recovery.adapter_node, internvla_t4_sensors.client_node"
' >"$RESULT_DIR/logs/import_check.log" 2>&1

SOURCE_DIGEST="$({
  for package in "${packages[@]}"; do
    find "$CONTROL_ROOT_REAL/$package" -type f \
      ! -path '*/__pycache__/*' ! -name '*.pyc' -print0
  done
} | sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)"

python3 - "$RESULT_DIR/overlay_update_manifest.json" \
  "$SOURCE_DIGEST" "$CONTAINER" <<'PY'
import json,sys,time
from pathlib import Path
output=Path(sys.argv[1])
payload={
    "schema_version":1,
    "status":"PASS",
    "runtime_policy":"completion_sim",
    "runtime_target":"isaac_simulation",
    "resource_lease_ack":"isaac_or_combined",
    "source_digest_sha256":sys.argv[2],
    "container":sys.argv[3],
    "recovery_control_interface_generated":True,
    "strict_evidence_modified":False,
    "real_go2_targeted":False,
    "completed_unix":time.time(),
}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
