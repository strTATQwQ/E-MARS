#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 3 ]] || {
  echo "usage: run_t5_fast_prepare_lane_a_online.sh CODE_SHA RUN_ID RESULT_ROOT" >&2
  exit 64
}

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"

# This explicit entry is the only supported way to select the single-Lane-A
# preparation path.  The legacy three-host entry remains dual-Lane by default.
INTERNNAV_T5_FAST_PREPARE_SCOPE=lane-a \
INTERNNAV_T5_LANE_A_PREPARE_ENTRY=1 \
  exec bash "$root/coordination/run_t5_fast_prepare_online.sh" "$@"
