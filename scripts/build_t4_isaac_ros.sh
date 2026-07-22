#!/usr/bin/env bash
set -eo pipefail

# T4 deliberately uses the first official Isaac ROS release that supports the
# Blackwell generation in this lab.  The frozen T3 Humble workspace is not
# modified; only this isolated Jazzy workspace/container is written.
T4_WS="${INTERNVLA_T4_ISAAC_ROS_WS:-$HOME/internnav-t4/isaac_ros_ws_45}"
CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
CONTAINER="${INTERNVLA_T4_CONTAINER_NAME:-internnav_t4_isaac_ros}"
IMAGE="${INTERNVLA_T4_IMAGE:-cached_isaac_run_dev_image_local:latest}"

case "$(realpath "$T4_WS")" in
  "$HOME/internnav-t4/isaac_ros_ws_45") ;;
  *) echo "unsafe T4 workspace: $T4_WS" >&2; exit 2 ;;
esac
test -d "$T4_WS/src"
test -d "$CONTROL_ROOT/internvla_t4_sensors"
command -v isaac-ros >/dev/null
docker buildx version >/dev/null

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  ISAAC_ROS_WS="$T4_WS" isaac-ros activate --build-local --build-only \
    -c docker.run.container_name="$CONTAINER"
fi

for package in \
  internvla_ros2_msgs internvla_ros2 internvla_nav2_adapter \
  internvla_go2_controller internvla_t4_sensors internvla_t4_recovery; do
  destination="$T4_WS/src/$package"
  test ! -e "$destination" || {
    echo "refusing to overwrite existing isolated source: $destination" >&2
    exit 2
  }
  cp -a "$CONTROL_ROOT/$package" "$destination"
done

if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  test "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" = false || {
    echo "refusing to replace running container: $CONTAINER" >&2
    exit 2
  }
  docker rm "$CONTAINER" >/dev/null
fi

docker run -d \
  --name "$CONTAINER" \
  --network host \
  --ipc host \
  --pid host \
  --gpus all \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  -e ROS_LOCALHOST_ONLY=0 \
  -e USERNAME=admin \
  -e HOST_USER_UID="$(id -u)" \
  -e HOST_USER_GID="$(id -g)" \
  -v "$T4_WS:/workspaces/isaac" \
  -v "$CONTROL_ROOT:$CONTROL_ROOT" \
  --entrypoint /usr/local/bin/scripts/workspace-entrypoint.sh \
  "$IMAGE" sleep infinity >/dev/null

docker exec --user root --workdir /workspaces/isaac "$CONTAINER" bash -lc '
  set -eo pipefail
  source /opt/ros/jazzy/setup.bash
  apt-get update
  apt-cache show ros-jazzy-isaac-ros-nvblox >/dev/null
  apt-cache show ros-jazzy-isaac-ros-visual-slam >/dev/null
  apt-get install -y --no-install-recommends \
    ros-jazzy-isaac-ros-nvblox \
    ros-jazzy-isaac-ros-visual-slam \
    ros-jazzy-navigation2 \
    ros-jazzy-nav2-bringup \
    ros-jazzy-robot-localization
  # The official image already contains a Jazzy rosdep cache.  Do not refresh
  # it here: some lab networks intentionally block raw.githubusercontent.com.
  rosdep install --from-paths src --ignore-src -r -y --rosdistro jazzy
'

docker exec --user admin --workdir /workspaces/isaac "$CONTAINER" bash -lc '
  set -eo pipefail
  source /opt/ros/jazzy/setup.bash
  colcon build --symlink-install --event-handlers console_direct+ \
    --cmake-args -DBUILD_TESTING=OFF \
    --packages-up-to internvla_t4_sensors internvla_t4_recovery
'

docker exec --user admin --workdir /workspaces/isaac "$CONTAINER" bash -lc '
  set -eo pipefail
  source /opt/ros/jazzy/setup.bash
  source install/setup.bash
  nvidia-smi -L
  ros2 pkg prefix nvblox_ros
  ros2 pkg prefix nvblox_nav2
  ros2 pkg prefix isaac_ros_visual_slam
  ros2 pkg prefix internvla_t4_sensors
  ros2 pkg executables nvblox_ros | grep -F "nvblox_ros nvblox_node"
  ros2 interface show nvblox_msgs/msg/DistanceMapSlice >/dev/null
  python3 -c "import internvla_t4_sensors.sensor_bridge_node, internvla_t4_sensors.odometry_supervisor_node"
'

python3 - "$T4_WS/build_t4_manifest.json" "$CONTAINER" "$IMAGE" <<'PY'
import json, subprocess, sys, time
from pathlib import Path

output = Path(sys.argv[1])
image_id = subprocess.check_output(
    ["docker", "image", "inspect", "--format", "{{.Id}}", sys.argv[3]], text=True
).strip()
image_digest = subprocess.check_output(
    ["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", sys.argv[3]], text=True
).strip()
packages = subprocess.check_output([
    "docker", "exec", "--user", "admin", sys.argv[2], "bash", "-lc",
    "source /opt/ros/jazzy/setup.bash; source /workspaces/isaac/install/setup.bash; "
    "ros2 pkg prefix nvblox_ros; ros2 pkg prefix nvblox_nav2; "
    "ros2 pkg prefix isaac_ros_visual_slam; ros2 pkg prefix internvla_t4_sensors"
], text=True).splitlines()
payload = {
    "schema_version": 1,
    "status": "PASS",
    "built_unix": time.time(),
    "isaac_ros_release": "release-4.5",
    "ros_distro": "jazzy",
    "cuda_major": 13,
    "image": sys.argv[3],
    "image_id": image_id,
    "repo_digests": json.loads(image_digest),
    "container_name": "redacted-in-delivery",
    "package_prefixes": packages,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
