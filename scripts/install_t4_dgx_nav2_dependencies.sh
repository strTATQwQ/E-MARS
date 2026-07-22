#!/usr/bin/env bash
set -euo pipefail

# This script runs on DGX only and must be wrapped by with_resource_lease.sh.
# It deliberately accepts a sudo password only on stdin through sudo itself;
# no credential is accepted in argv, written to disk, or copied into evidence.
case "${INTERNNAV_T4_RESOURCE_LEASE_ACK:-}" in
  dgx|dgx+isaac) ;;
  *) echo "DGX resource lease acknowledgement is required" >&2; exit 2 ;;
esac
test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim
test "${INTERNNAV_SIMULATION_TARGET:-}" = isaac

ROS_DISTRO_NAME="${INTERNVLA_ROS_DISTRO:-jazzy}"
RESULT_DIR="${INTERNVLA_T4_DGX_DEPENDENCY_RESULT_DIR:-}"
test -n "$RESULT_DIR"
test ! -e "$RESULT_DIR"
mkdir -p "$RESULT_DIR"

packages=(
  "ros-$ROS_DISTRO_NAME-navigation2"
  "ros-$ROS_DISTRO_NAME-nav2-bringup"
)

run_sudo() {
  if sudo -n true 2>/dev/null; then
    sudo -n "$@"
  else
    test "${INTERNNAV_DGX_SUDO_STDIN_ACK:-}" = 1 || {
      echo "sudo needs a cached credential or INTERNNAV_DGX_SUDO_STDIN_ACK=1 with the password supplied on stdin" >&2
      exit 77
    }
    sudo -S -p '' "$@"
  fi
}

run_sudo apt-get update >"$RESULT_DIR/apt-update.log" 2>&1
run_sudo apt-get install -y --no-install-recommends "${packages[@]}" \
  >"$RESULT_DIR/apt-install.log" 2>&1

set +u
source "/opt/ros/$ROS_DISTRO_NAME/setup.bash"
set -u
ros2 pkg prefix nav2_bringup >"$RESULT_DIR/nav2_bringup.prefix"
ros2 pkg prefix nav2_collision_monitor >"$RESULT_DIR/nav2_collision_monitor.prefix"
dpkg-query -W -f='${Package}\t${Version}\n' "${packages[@]}" \
  >"$RESULT_DIR/package-versions.tsv"
python3 - "$RESULT_DIR/status.json" "$ROS_DISTRO_NAME" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "schema_version": 1,
            "status": "PASS",
            "host_role": "dgx_onboard_compute",
            "ros_distro": sys.argv[2],
            "installed": ["navigation2", "nav2_bringup"],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
