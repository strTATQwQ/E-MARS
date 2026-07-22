#!/usr/bin/env bash
set -euo pipefail

# Coordinator-only suggestion. This wrapper cannot start the F1 workload until
# the shared 01R session owns the map companion in its process/cleanup ledger.
test "$#" -eq 2 || {
  echo "usage: $0 <fresh-result-dir> <grant-id>" >&2
  exit 2
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="$1"
GRANT_ID="$2"

test "${INTERNNAV_SENSOR_SESSION_LEASE_ACK:-}" = 1 || {
  echo "t4 map smoke must run inside scripts/with_resource_lease.sh" >&2
  exit 2
}
test -n "$RESULT_DIR" && test -n "$GRANT_ID"

# These markers are deliberately absent from the ba0671d worker baseline.  00
# must implement the interface_requests in handoff.json before this check can
# pass. Failing here prevents an untracked docker-exec sidecar or a bootstrap-
# only run from being mislabeled as the F1 map smoke.
grep -Fq 'INTERNNAV_T4_MAP_COMPANION_MODULE' \
  "$ROOT/sensor_runtime/ros_inner_supervisor.py" || {
  echo "managed completion map companion interface is not integrated" >&2
  exit 78
}
grep -Fq 'completion_sim_map' "$ROOT/sensor_runtime/session.py" || {
  echo "completion_sim_map grant/result validation is not integrated" >&2
  exit 78
}

export INTERNNAV_T4_MAP_COMPANION_MODULE=t4_completion.map.companion
export INTERNNAV_T4_MAP_CONFIG_DIR="$ROOT/configs/completion_sim/map"
export INTERNNAV_T4_MAP_NVBLOX_MODE=shadow
export INTERNNAV_SIMULATION_TARGET=isaac

exec bash "$ROOT/sensor_runtime/run_model_free_sensor_session.sh" \
  --profile completion_sim_map --result-dir "$RESULT_DIR" --grant-id "$GRANT_ID"
