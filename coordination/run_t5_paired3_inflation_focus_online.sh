#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_t5_paired3_inflation_focus_online.sh RUN_ID CODE_SHA PREP_RESULT_ROOT RESULT_ROOT

Run the frozen high-inflation-risk paired-3.  For each episode, Lane A runs
InternVLA-only while Lane B concurrently runs InternVLA plus bounded Step3
task-state control.  Both arms consume the same frozen paired10_a episode.
The next episode starts only after both leases have been released.
EOF
  exit 64
}

[[ $# -eq 4 ]] || usage
run_id="$1"
code_sha="$2"
prepare_relative="$3"
result_relative="$4"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
manifest_relative=configs/internnav_t5/paired3_inflation_focus_manifest.json
manifest="$root/$manifest_relative"

[[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]{7,63}$ ]] || usage
[[ "$code_sha" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$prepare_relative" =~ ^results/internnav_t5/final-pilot-prepare-[a-z0-9][a-z0-9._-]{7,95}$ ]] || usage
expected_result="results/internnav_t5/paired3-inflation-focus-${run_id}"
test "$result_relative" = "$expected_result" || usage

git_command=(git -C "$root")
if [[ -f "$root/.git" ]] && grep -Eq '^gitdir: [A-Za-z]:/' "$root/.git"; then
  command -v git.exe >/dev/null
  command -v wslpath >/dev/null
  git_command=(git.exe -C "$(wslpath -w "$root")")
fi
"${git_command[@]}" cat-file -e "$code_sha^{commit}"
"${git_command[@]}" cat-file -e "$code_sha:$manifest_relative"
test "$("${git_command[@]}" rev-parse HEAD | tr -d '\r')" = "$code_sha"
test -z "$("${git_command[@]}" status --porcelain --untracked-files=all | tr -d '\r')"

prepare_root="$root/$prepare_relative"
result_root="$root/$result_relative"
receipt="$prepare_root/final_pilot_prepare_receipt.json"
test -f "$manifest" && test ! -L "$manifest"
test -f "$receipt" && test ! -L "$receipt"
test ! -e "$result_root"

mapfile -t episodes < <(python3 - "$manifest" "$receipt" "$code_sha" <<'PY'
import json,sys
from pathlib import Path
manifest=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
receipt=json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected=["5840_1474","2613_625","6623_1657"]
assert manifest.get("status")=="FROZEN_FOR_EXECUTION"
assert manifest.get("source",{}).get("source_lane")=="a"
assert manifest.get("source",{}).get("pair_set")=="paired10_a"
assert manifest.get("source",{}).get("episode_keys")==expected
assert manifest.get("arms",{}).get("lane_a")=={
 "evaluation_arm":"internvla_only","step3_timeout_advisor":False,
 "step3_task_state_control":False}
assert manifest.get("arms",{}).get("lane_b")=={
 "evaluation_arm":"internvla_step3","step3_timeout_advisor":True,
 "step3_task_state_control":True}
assert receipt.get("status")=="PASS"
assert receipt.get("code_ref_sha")==sys.argv[3]
assert receipt.get("prepare_scope")=="dual"
assert receipt.get("prepared_lanes")==["a","b"]
assert receipt.get("split",{}).get("lanes",{}).get("a",{}).get(
 "episode_keys")==[
 "6898_1741","6292_1573","5840_1474","6842_1720","1420_364",
 "4084_1003","583_145","1803_448","6623_1657","2613_625"]
assert receipt.get("checks") and all(receipt["checks"].values())
print(*expected,sep="\n")
PY
)
test "${#episodes[@]}" = 3

umask 077
mkdir -p "$result_root/logs"
active_pids=()
terminate_active_children() {
  local deadline pid any_alive
  trap - EXIT INT TERM HUP
  for pid in "${active_pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  deadline=$((SECONDS + 60))
  while (( SECONDS < deadline )); do
    any_alive=0
    for pid in "${active_pids[@]}"; do kill -0 "$pid" 2>/dev/null && any_alive=1; done
    test "$any_alive" = 1 || break
    sleep 1
  done
  for pid in "${active_pids[@]}"; do
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  exit 130
}
trap terminate_active_children INT TERM HUP

run_lane() {
  local lane="$1" advisor="$2" task_state="$3" episode="$4"
  local lane_run_id="$5" lane_result="$6" log="$7"
  (
    unset INTERNVLA_T4_VIEW_MODE INTERNVLA_T4_HISTORY_MODE
    unset INTERNVLA_T5_TRAJECTORY_RERANK INTERNVLA_T4_PROGRESS_HORIZON_SEC
    unset INTERNVLA_T4_REFRESH_DISTANCE_M INTERNVLA_T4_REFRESH_TIME_SEC
    unset INTERNVLA_T4_TRAJECTORY_VALIDITY_SEC
    export INTERNNAV_T5_CANDIDATE_PROFILE=recovery_a
    export INTERNNAV_T5_RTF_ABLATION_PROFILE=navigation_fast
    export INTERNNAV_T5_ISAAC_SENSOR_PROFILE=dual_lane_wp03_stop_shadow
    export INTERNNAV_T5_STEP3_LIVE_ADVISOR=0
    export INTERNVLA_T5_STEP3_TIMEOUT_ADVISOR="$advisor"
    export INTERNVLA_T5_STEP3_TASK_STATE_CONTROL="$task_state"
    export INTERNVLA_T5_FULL_RGB_CAPTURE=1
    export INTERNVLA_T5_D435_5HZ_CAPTURE=1
    export INTERNVLA_T5_TERMINATION_MODE=oracle_termination
    export INTERNVLA_T5_SYSTEM2_REPLAN_POLICY=observation_bound
    export INTERNVLA_T5_SYSTEM2_QUEUE_HORIZON=0
    export INTERNVLA_T5_SYSTEM1_QUEUE_HORIZON=0
    export INTERNNAV_T5_LIVE_FRONTIER_CAPTURE=0
    export INTERNNAV_T5_STRICT_EXTENSION_PROFILE=off
    export INTERNNAV_T5_NVBLOX_MODE=off
    export INTERNNAV_T5_RUN_MODE=model
    export INTERNNAV_T5_FAULT_INJECTION_PROFILE=off
    export INTERNNAV_T5_SCREEN_EPISODE_KEY="$episode"
    export INTERNNAV_T5_PILOT_SOURCE_LANE=a
    exec bash "$root/coordination/run_t5_fast_lane_online.sh" \
      "$lane" pilot-screen1 "$lane_run_id" "$code_sha" \
      "$prepare_relative" "$lane_result"
  ) >"$log" 2>&1
}

verify_child() {
  local summary="$1" expected_lane="$2" expected_arm="$3" expected_episode="$4"
  python3 - "$summary" "$expected_lane" "$expected_arm" "$expected_episode" <<'PY'
import json,sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
checks=value.get("checks") or {}
lease=value.get("lease_release") or {}
lease_checks=lease.get("checks") or {}
binding=value.get("input_binding") or {}
assert value.get("status")=="PASS" and checks and all(checks.values())
assert value.get("lane")==sys.argv[2]
assert value.get("evaluation_arm")==sys.argv[3]
assert value.get("pair_set")=="paired10_a"
assert value.get("source_lane")=="a"
assert binding.get("execution_episode_keys")==[sys.argv[4]]
assert value.get("capture")=={
 "model_observations":True,"independent_d435_5hz":True}
assert lease.get("status")=="PASS"
assert lease_checks.get("wrapped_command_released") is True
assert lease_checks.get("wrapped_group_absent_before_release") is True
assert lease_checks.get("exact_lane_resources") is True
PY
}

child_results=()
overall_rc=0
for episode in "${episodes[@]}"; do
  token="${episode//_/-}"
  a_id="${run_id}-${token}-a"
  b_id="${run_id}-${token}-b"
  a_relative="results/internnav_t5/fast-lane-a-pilot-screen1-${a_id}"
  b_relative="results/internnav_t5/fast-lane-b-pilot-screen1-${b_id}"
  test ! -e "$root/$a_relative" && test ! -e "$root/$b_relative"
  set +e
  run_lane a 0 0 "$episode" "$a_id" "$a_relative" \
    "$result_root/logs/${episode}_internvla_only.log" & a_pid=$!
  run_lane b 1 1 "$episode" "$b_id" "$b_relative" \
    "$result_root/logs/${episode}_internvla_step3.log" & b_pid=$!
  active_pids=("$a_pid" "$b_pid")
  wait "$a_pid"; a_rc=$?
  wait "$b_pid"; b_rc=$?
  active_pids=()
  set -e
  printf '%s\n' "$a_rc" >"$result_root/${episode}_internvla_only.rc"
  printf '%s\n' "$b_rc" >"$result_root/${episode}_internvla_step3.rc"
  if test "$a_rc" != 0 || test "$b_rc" != 0; then overall_rc=75; break; fi
  verify_child "$root/$a_relative/fast_lane_final_summary.json" \
    a internvla_only "$episode" || { overall_rc=75; break; }
  verify_child "$root/$b_relative/fast_lane_final_summary.json" \
    b internvla_step3 "$episode" || { overall_rc=75; break; }
  child_results+=("$episode|$a_relative|$b_relative")
done
trap - INT TERM HUP

python3 - "$result_root/paired3_execution_summary.json" "$manifest" \
  "$run_id" "$code_sha" "$prepare_relative" "$overall_rc" \
  "${child_results[@]}" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
output,manifest_path=map(Path,sys.argv[1:3])
records=[]
for packed in sys.argv[7:]:
    episode,a,b=packed.split("|",2)
    records.append({"episode_key":episode,"internvla_only":a,"internvla_step3":b})
checks={
 "all_three_pairs_completed":len(records)==3,
 "all_children_passed_and_released":int(sys.argv[6])==0,
}
payload={
 "schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
 "stage":"t5_paired3_inflation_focus","run_id":sys.argv[3],
 "code_ref_sha":sys.argv[4],"prepare_result":sys.argv[5],
 "manifest":"configs/internnav_t5/paired3_inflation_focus_manifest.json",
 "manifest_sha256":hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
 "pairs":records,"checks":checks,
 "evidence_classification":"TARGETED_DEVELOPMENT_DIAGNOSTIC_NOT_HELD_OUT",
 "unique_episode_count":3,"execution_count":6 if all(checks.values()) else None,
 "recorded_unix":time.time(),
}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
raise SystemExit(0 if payload["status"]=="PASS" else 75)
PY
