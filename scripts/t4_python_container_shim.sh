#!/usr/bin/env bash
set -eo pipefail

CONTAINER="${INTERNVLA_T4_CONTAINER_NAME:-internnav_t4_isaac_ros}"
exec docker exec \
  --user admin \
  --workdir /workspaces/isaac \
  -e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}" \
  -e "ROS_NAMESPACE=${ROS_NAMESPACE:-}" \
  -e ROS_LOCALHOST_ONLY=0 \
  -e "ROS_AUTOMATIC_DISCOVERY_RANGE=${ROS_AUTOMATIC_DISCOVERY_RANGE:-SYSTEM_DEFAULT}" \
  -e "ROS_STATIC_PEERS=${ROS_STATIC_PEERS:-}" \
  -e "INTERNNAV_RUNTIME_POLICY=${INTERNNAV_RUNTIME_POLICY:-}" \
  -e "INTERNNAV_SIMULATION_TARGET=${INTERNNAV_SIMULATION_TARGET:-}" \
  -e "INTERNNAV_T5_RESOURCE_LEASE_ACK=${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" \
  -e "INTERNNAV_T5_ID_PREFIX=${INTERNNAV_T5_ID_PREFIX:-}" \
  -e "INTERNNAV_T4_MAP_COMPANION_ACK=${INTERNNAV_T4_MAP_COMPANION_ACK:-}" \
  -e PYTHONUNBUFFERED=1 \
  "$CONTAINER" \
  bash -lc 'source /opt/ros/jazzy/setup.bash; source /workspaces/isaac/install/setup.bash; exec python3 "$@"' \
  bash "$@"
