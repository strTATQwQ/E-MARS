#!/usr/bin/env bash
set -eo pipefail

CONTAINER="${INTERNVLA_T4_CONTAINER_NAME:-internnav_t4_isaac_ros}"
CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
exec docker exec \
  --user root \
  --workdir /workspaces/isaac \
  "$CONTAINER" \
  python3 "$CONTROL_ROOT/scripts/t4_container_process_cleanup.py" \
    --control-root "$CONTROL_ROOT"
