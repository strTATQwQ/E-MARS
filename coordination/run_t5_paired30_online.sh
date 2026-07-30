#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_t5_paired30_online.sh RUN_ID CODE_SHA PREP_RESULT_ROOT RESULT_ROOT

Run the frozen paired-30 in two balanced rounds.  Lane A always owns the first
15 episodes and Lane B the second 15; the InternVLA-only and InternVLA+Step3
arms swap in round 2.  Every execution ends at the earlier of 1800 wall
seconds after READY or the evaluator's configured 8000-step (observed 8100)
limit.  A clean wall-capped failure is retained and the next pair continues.
EOF
  exit 64
}

[[ $# -eq 4 ]] || usage
run_id="$1"
code_sha="$2"
prepare_relative="$3"
result_relative="$4"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
manifest_relative=configs/internnav_t5/paired30_episode_manifest.json
manifest="$root/$manifest_relative"

[[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]{7,63}$ ]] || usage
[[ "$code_sha" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$prepare_relative" =~ ^results/internnav_t5/final-pilot-prepare-[a-z0-9][a-z0-9._-]{7,95}$ ]] || usage
expected_result="results/internnav_t5/paired30-${run_id}"
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

mapfile -t lane_a_episodes < <(python3 - "$manifest" "$receipt" "$code_sha" <<'PY'
import json,sys
from pathlib import Path
manifest=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
receipt=json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
keys=manifest.get("episode_keys")
sets=manifest.get("lane_sets") or {}
limits=manifest.get("per_execution_limits") or {}
assert manifest.get("status")=="FROZEN_FOR_EXECUTION"
assert manifest.get("episode_count")==30
assert isinstance(keys,list) and len(keys)==len(set(keys))==30
assert sets.get("a")+sets.get("b")==keys
assert len(sets["a"])==len(sets["b"])==15
assert limits=={
 "wall_seconds_after_ready":1800,
 "configured_max_step":8000,
 "observed_evaluator_step_limit":8100,
 "termination":"earlier_of_wall_or_physics_steps",
}
assert receipt.get("status")=="PASS"
assert receipt.get("code_ref_sha")==sys.argv[3]
assert receipt.get("prepare_scope")=="dual"
assert receipt.get("prepared_lanes")==["a","b"]
assert receipt.get("checks") and all(receipt["checks"].values())
print(*sets["a"],sep="\n")
PY
)
mapfile -t lane_b_episodes < <(python3 - "$manifest" <<'PY'
import json,sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(*value["lane_sets"]["b"],sep="\n")
PY
)
test "${#lane_a_episodes[@]}" = 15
test "${#lane_b_episodes[@]}" = 15

umask 077
mkdir -p "$result_root/logs"
progress="$result_root/progress.json"
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
  local lane="$1" advisor="$2" episode="$3" lane_run_id="$4"
  local lane_result="$5" log="$6"
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
    export INTERNVLA_T5_STEP3_TASK_STATE_CONTROL=0
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
    export INTERNNAV_T5_PILOT_SOURCE_LANE="$lane"
    export INTERNNAV_T5_PAIRED30_MANIFEST="$manifest_relative"
    export INTERNNAV_T5_PILOT_MAX_STEP=8000
    export INTERNNAV_T5_FAST_SCREEN_TIMEOUT_SEC=1800
    export INTERNVLA_T3_STATIC_CLEARANCE_GATE_M=0.25
    exec bash "$root/coordination/run_t5_fast_lane_online.sh" \
      "$lane" pilot-screen1 "$lane_run_id" "$code_sha" \
      "$prepare_relative" "$lane_result"
  ) >"$log" 2>&1
}

verify_release() {
  local result="$1" rc="$2" expected_lane="$3" expected_episode="$4"
  python3 - "$result" "$rc" "$expected_lane" "$expected_episode" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); rc=int(sys.argv[2])
lease=json.loads((root/"lease_release_summary.json").read_text(encoding="utf-8"))
checks=lease.get("checks") or {}
assert lease.get("status")=="PASS"
assert checks.get("wrapped_command_released") is True
assert checks.get("wrapped_group_absent_before_release") is True
assert checks.get("exact_lane_resources") is True
if rc==0:
    final=json.loads((root/"fast_lane_final_summary.json").read_text(encoding="utf-8"))
    binding=final.get("input_binding") or {}
    assert final.get("status")=="PASS"
    assert final.get("lane")==sys.argv[3]
    assert binding.get("pair_set")=="paired30"
    assert binding.get("execution_episode_keys")==[sys.argv[4]]
elif rc==124:
    # The wall cap is an expected censored navigation failure, but cleanup is
    # still required to pass before the next episode can start.
    pass
else:
    raise AssertionError(f"unexpected child rc {rc}")
PY
}

write_progress() {
  local round="$1" index="$2" status="$3" a_episode="$4" b_episode="$5"
  local a_result="$6" b_result="$7" a_rc="$8" b_rc="$9"
  local advisor_a="${10}" advisor_b="${11}"
  python3 - "$progress" "$run_id" "$code_sha" "$round" "$index" "$status" \
    "$a_episode" "$b_episode" "$a_result" "$b_result" "$a_rc" "$b_rc" \
    "$advisor_a" "$advisor_b" <<'PY'
import json,os,sys,time
from pathlib import Path
path=Path(sys.argv[1])
old=json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
rows=old.get("completed_pairs",[])
identity=[sys.argv[4],int(sys.argv[5])]
row={"round":identity[0],"index":identity[1],"status":sys.argv[6],
     "lane_a_episode":sys.argv[7],"lane_b_episode":sys.argv[8],
     "lane_a_result":sys.argv[9],"lane_b_result":sys.argv[10],
     "lane_a_rc":int(sys.argv[11]),"lane_b_rc":int(sys.argv[12]),
     "lane_a_arm":"internvla_step3" if sys.argv[13]=="1" else "internvla_only",
     "lane_b_arm":"internvla_step3" if sys.argv[14]=="1" else "internvla_only",
     "recorded_unix":time.time()}
rows=[item for item in rows if [item.get("round"),item.get("index")]!=identity]
rows.append(row)
payload={"schema_version":1,"run_id":sys.argv[2],"code_ref_sha":sys.argv[3],
         "status":"RUNNING","completed_pairs":rows,
         "completed_pair_count":len(rows),"recorded_unix":time.time()}
tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
os.replace(tmp,path)
PY
}

run_pair() {
  local round="$1" index="$2" advisor_a="$3" advisor_b="$4"
  local a_episode="${lane_a_episodes[$index]}" b_episode="${lane_b_episodes[$index]}"
  local token="${round}-$((index+1))"
  local a_id="${run_id}-${token}-a" b_id="${run_id}-${token}-b"
  local a_relative="results/internnav_t5/fast-lane-a-pilot-screen1-${a_id}"
  local b_relative="results/internnav_t5/fast-lane-b-pilot-screen1-${b_id}"
  local a_rc b_rc
  if test -f "$root/$a_relative/lease_release_summary.json" && \
      test -f "$root/$b_relative/lease_release_summary.json"; then
    a_rc="$(cat "$result_root/${round}_${index}_lane_a.rc")"
    b_rc="$(cat "$result_root/${round}_${index}_lane_b.rc")"
    verify_release "$root/$a_relative" "$a_rc" a "$a_episode"
    verify_release "$root/$b_relative" "$b_rc" b "$b_episode"
    write_progress "$round" "$index" RESUMED "$a_episode" "$b_episode" \
      "$a_relative" "$b_relative" "$a_rc" "$b_rc" "$advisor_a" "$advisor_b"
    return
  fi
  test ! -e "$root/$a_relative" && test ! -e "$root/$b_relative"
  set +e
  run_lane a "$advisor_a" "$a_episode" "$a_id" "$a_relative" \
    "$result_root/logs/${round}_${a_episode}_lane_a.log" & a_pid=$!
  run_lane b "$advisor_b" "$b_episode" "$b_id" "$b_relative" \
    "$result_root/logs/${round}_${b_episode}_lane_b.log" & b_pid=$!
  active_pids=("$a_pid" "$b_pid")
  wait "$a_pid"; a_rc=$?
  wait "$b_pid"; b_rc=$?
  active_pids=()
  set -e
  printf '%s\n' "$a_rc" >"$result_root/${round}_${index}_lane_a.rc"
  printf '%s\n' "$b_rc" >"$result_root/${round}_${index}_lane_b.rc"
  verify_release "$root/$a_relative" "$a_rc" a "$a_episode"
  verify_release "$root/$b_relative" "$b_rc" b "$b_episode"
  write_progress "$round" "$index" PASS "$a_episode" "$b_episode" \
    "$a_relative" "$b_relative" "$a_rc" "$b_rc" "$advisor_a" "$advisor_b"
}

for index in $(seq 0 14); do run_pair round1 "$index" 0 1; done
for index in $(seq 0 14); do run_pair round2 "$index" 1 0; done
trap - INT TERM HUP

python3 - "$progress" "$manifest" <<'PY'
import json,os,sys,time
from pathlib import Path
path,manifest=map(Path,sys.argv[1:3])
value=json.loads(path.read_text(encoding="utf-8"))
rows=value.get("completed_pairs",[])
value.update({
 "status":"PASS" if len(rows)==30 else "FAIL",
 "unique_episode_count":30,
 "execution_count":60 if len(rows)==30 else None,
 "manifest_sha256":__import__("hashlib").sha256(manifest.read_bytes()).hexdigest(),
 "recorded_unix":time.time(),
})
tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
os.replace(tmp,path)
raise SystemExit(0 if value["status"]=="PASS" else 75)
PY
