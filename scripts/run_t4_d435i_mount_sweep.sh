#!/usr/bin/env bash
set -euo pipefail

CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
RUNNER="$CONTROL_ROOT/scripts/run_t4_camera_ablation.sh"
STATUS="$CONTROL_ROOT/results/internnav_t4/t4_1_d435i_mount_sweep_status.json"

TRIALS=(
  "d435i_low_062_p30 032"
  "d435i_low_062_p40 033"
  "d435i_mid_085_p20 034"
  "d435i_mid_085_p30 035"
  "d435i_mid_085_p40 036"
  "d435i_high_100_p30 037"
  "d435i_h1_125_p30 038"
)

write_status() {
  local state="$1" variant="$2" attempt="$3" completed="$4"
  python3 - "$STATUS" "$state" "$variant" "$attempt" "$completed" <<'PY'
import json,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version":1,
    "status":sys.argv[2],
    "active_variant":sys.argv[3] or None,
    "active_attempt":sys.argv[4] or None,
    "completed_trial_count":int(sys.argv[5]),
    "updated_unix":time.time(),
},indent=2,sort_keys=True)+"\n")
PY
}

completed=1  # d435i_low_062_p20 attempt 031 is the frozen first trial.
write_status RUNNING "" "" "$completed"
for trial in "${TRIALS[@]}"; do
  read -r variant attempt <<<"$trial"
  result="$CONTROL_ROOT/results/internnav_t4/t4_1_camera_dev_${variant}_attempt_${attempt}"
  test ! -e "$result/phase_status.json"
  write_status RUNNING "$variant" "$attempt" "$completed"
  bash "$RUNNER" dev "$variant" "$attempt"
  completed=$((completed + 1))
done
write_status COMPLETE "" "" "$completed"
