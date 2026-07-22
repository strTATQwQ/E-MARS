#!/usr/bin/env bash
set -euo pipefail

# Starts only the Step3-first high-level mission gateway.  This script does not
# start Nav2 or any velocity/goal bridge and therefore cannot move the robot.

: "${INTERNVLA_REAL_MISSION_CONFIG_SHA256:?set the frontend control config SHA256}"
if [[ ! "${INTERNVLA_REAL_MISSION_CONFIG_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid INTERNVLA_REAL_MISSION_CONFIG_SHA256" >&2
  exit 64
fi

if [[ "${INTERNNAV_RUNTIME_POLICY:-}" == "completion_sim" ]] \
  || [[ "${INTERNNAV_SIMULATION_TARGET:-}" == "isaac" ]]; then
  echo "strict real-Go2 mission ingress refuses completion_sim/Isaac overlays" >&2
  exit 65
fi

source "${ROS_SETUP_PATH:-/opt/ros/jazzy/setup.bash}"
if [[ -n "${INTERNNAV_INSTALL_SETUP:-}" ]]; then
  source "${INTERNNAV_INSTALL_SETUP}"
fi

export INTERNVLA_REAL_MISSION_STEP3_ENDPOINT="${INTERNVLA_REAL_MISSION_STEP3_ENDPOINT:-tcp://127.0.0.1:8200}"
export INTERNVLA_REAL_MISSION_INPUT_TOPIC="${INTERNVLA_REAL_MISSION_INPUT_TOPIC:-/user_instruction}"
export INTERNVLA_REAL_MISSION_CANONICAL_TOPIC="${INTERNVLA_REAL_MISSION_CANONICAL_TOPIC:-/internvla/mission/canonical}"
export INTERNVLA_REAL_MISSION_STATUS_TOPIC="${INTERNVLA_REAL_MISSION_STATUS_TOPIC:-/internvla/mission/status}"
export INTERNVLA_REAL_MISSION_STATE_PATH="${INTERNVLA_REAL_MISSION_STATE_PATH:-/tmp/internvla_real_go2_mission_state.json}"

exec ros2 run internvla_t4_sensors internvla_real_go2_mission_gateway
