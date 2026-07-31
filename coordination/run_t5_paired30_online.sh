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
x86_target=song@10.100.120.123
dgx_a_target=railgun@10.100.100.128
dgx_b_target=rail@10.100.120.122

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

mapfile -t deployment_roots < <(python3 - "$receipt" "$code_sha" <<'PY'
import json,re,sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
roots=value.get("deployment_roots") or {}
checks={
 "status":value.get("status")=="PASS",
 "code":value.get("code_ref_sha")==sys.argv[2],
 "scope":value.get("prepare_scope")=="dual" and value.get("prepared_lanes")==["a","b"],
 "keys":set(roots)=={"dgx_a","dgx_b","x86_a","x86_b"},
 "paths":all(re.fullmatch(r"/[A-Za-z0-9._/-]+",str(item)) for item in roots.values()),
}
if not all(checks.values()): raise SystemExit(f"paired30 deployment binding failed: {checks}")
for key in ("dgx_a","dgx_b","x86_a","x86_b"): print(roots[key])
PY
)
dgx_a_root="${deployment_roots[0]}"
dgx_b_root="${deployment_roots[1]}"
x86_a_root="${deployment_roots[2]}"
x86_b_root="${deployment_roots[3]}"
test "${#deployment_roots[@]}" = 4

ssh_options=(-T -i "${INTERNNAV_T5_SSH_IDENTITY_FILE:-$HOME/.ssh/id_ed25519_internnav_runtime}" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2)
remote() { local target="$1"; shift; ssh "${ssh_options[@]}" "$target" "$@"; }

umask 077
mkdir -p "$result_root/logs"
progress="$result_root/progress.json"

prepare_paired30_static_maps() {
  local receipt_path="$result_root/static_map_prepare_receipt.json"
  local maps_dir="$result_root/static_maps" archive="$result_root/paired30_static_maps.tar.gz"
  local stage="/home/song/internnav-t1-t2/.t5-paired30-static-maps/${run_id}-${code_sha:0:12}"
  local destination_a="$dgx_a_root/inputs/paired30_${run_id}_static_maps"
  local destination_b="$dgx_b_root/inputs/paired30_${run_id}_static_maps"
  local map_sha remote_program remote_program_b64
  mkdir -p "$maps_dir"
  if test -f "$receipt_path"; then
    mapfile -t prepared_map < <(python3 - "$receipt_path" "$code_sha" <<'PY'
import json,re,sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
checks={
 "status":value.get("status")=="PASS",
 "code":value.get("code_ref_sha")==sys.argv[2],
 "episode_count":value.get("episode_count")==30,
 "dataset":value.get("dataset_sha256")=="b4de2a6be2c37c1ad160aa83d062acd7ad7671534df76df478ec7c495fed61f5",
 "sha":re.fullmatch(r"[0-9a-f]{64}",str(value.get("manifest_sha256",""))) is not None,
}
paths=value.get("remote_manifest_paths") or {}
checks["lanes"]=set(paths)=={"a","b"}
if not all(checks.values()): raise SystemExit(f"paired30 map receipt failed: {checks}")
print(value["manifest_sha256"]); print(paths["a"]); print(paths["b"])
PY
)
    map_sha="${prepared_map[0]}"
    lane_a_map_manifest="${prepared_map[1]}"
    lane_b_map_manifest="${prepared_map[2]}"
    remote "$dgx_a_target" "test \"\$(cat '$dgx_a_root/T5_DEPLOYMENT_REF')\" = '$code_sha'; test \"\$(sha256sum '$lane_a_map_manifest'|cut -d' ' -f1)\" = '$map_sha'"
    remote "$dgx_b_target" "test \"\$(cat '$dgx_b_root/T5_DEPLOYMENT_REF')\" = '$code_sha'; test \"\$(sha256sum '$lane_b_map_manifest'|cut -d' ' -f1)\" = '$map_sha'"
    paired30_map_manifest_sha="$map_sha"
    return
  fi

  read -r -d '' remote_program <<'REMOTE_X86' || true
set -euo pipefail
stage="$1"; deployment="$2"; code="$3"; manifest_rel="$4"
test "$(cat "$deployment/T5_DEPLOYMENT_REF")" = "$code"
test ! -e "$stage"
mkdir -p "$stage/cache" "$stage/maps"
python3 "$deployment/scripts/materialize_t5_frozen_subset.py" \
  --source-root /home/song/internnav-t0/data/InternData-N1/vln_pe/raw_data/r2r \
  --manifest "$deployment/$manifest_rel" --output-root "$stage/dataset" \
  >"$stage/materialization.json"
dataset="$stage/dataset/val_unseen/val_unseen.json.gz"
python3 "$deployment/scripts/build_t3_static_maps.py" \
  --dataset "$dataset" \
  --scene-root "$HOME/internnav-t0/InternNav/data/scene_data/mp3d_pe" \
  --output-root "$stage/cache" --minimum-required-prefix-clearance-m 0.25 \
  >"$stage/map_build.json"
cp "$stage/cache"/*.bin "$stage/maps/"
python3 "$deployment/scripts/build_t4_truth_isolated_static_manifest.py" \
  --manifest "$stage/cache/manifest.json" --dataset "$dataset" \
  --output "$stage/maps/manifest.json" >"$stage/truth_isolation.json"
python3 - "$stage/maps/manifest.json" "$deployment/$manifest_rel" <<'PY'
import hashlib,json,sys
from pathlib import Path
maps=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
contract=json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
keys=[f"{item['trajectory_id']}_{item['episode_id']}" for item in maps.get("generations",[])]
raw=maps.get("maps",{})
records=raw.values() if isinstance(raw,dict) else raw
scans={item.get("scan") for item in records if isinstance(item,dict)}
checks={
 "dataset":maps.get("dataset_sha256")==contract.get("dataset_sha256"),
 "episodes":maps.get("episode_count")==30 and keys==contract.get("episode_keys"),
 "scenes":scans=={"2azQ1b91cZZ","QUCTc6BB5sX","TbHJrupSAjP","Z6MFQCViBuw","zsNo4HB9uLZ"},
 "clearance":maps.get("required_prefix_minimum_clearance_gate_m")==0.25,
 "truth":maps.get("t4_truth_isolation",{}).get("status")=="PASS",
}
if not all(checks.values()): raise SystemExit(f"paired30 static map validation failed: {checks}")
for item in records:
 path=Path(sys.argv[1]).parent/item["file"]
 assert hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"]
PY
tar -C "$stage/maps" -czf "$stage/paired30_static_maps.tar.gz" .
sha256sum "$stage/maps/manifest.json" | cut -d' ' -f1 >"$stage/manifest.sha256"
REMOTE_X86
  remote_program_b64="$(printf '%s' "$remote_program" | base64 | tr -d '\r\n')"
  remote "$x86_target" \
    "flock -x -w 1800 /tmp/internnav_t5_isaac_shared_assets.lock bash -c \"\$(printf '%s' '$remote_program_b64'|base64 -d)\" paired30-map '$stage' '$x86_a_root' '$code_sha' '$manifest_relative'" \
    >"$result_root/logs/paired30_static_map_build.log" 2>&1
  remote "$x86_target" "cat '$stage/paired30_static_maps.tar.gz'" >"$archive"
  remote "$x86_target" "cat '$stage/manifest.sha256'" >"$result_root/paired30_static_map_manifest.sha256"
  map_sha="$(tr -d '\r\n' <"$result_root/paired30_static_map_manifest.sha256")"
  [[ "$map_sha" =~ ^[0-9a-f]{64}$ ]]
  tar -C "$maps_dir" -xzf "$archive"
  test "$(sha256sum "$maps_dir/manifest.json" | cut -d' ' -f1)" = "$map_sha"

  deploy_map() {
    local target="$1" deployment="$2" destination="$3" temporary="$3.tmp-${code_sha:0:12}"
    remote "$target" "set -euo pipefail; test \"\$(cat '$deployment/T5_DEPLOYMENT_REF')\" = '$code_sha'; test ! -e '$destination'; test ! -e '$temporary'; mkdir -p '$temporary'"
    gzip -dc "$archive" | ssh "${ssh_options[@]}" "$target" "tar -C '$temporary' -xf -"
    remote "$target" "set -euo pipefail; test \"\$(sha256sum '$temporary/manifest.json'|cut -d' ' -f1)\" = '$map_sha'; mv '$temporary' '$destination'"
  }
  deploy_map "$dgx_a_target" "$dgx_a_root" "$destination_a"
  deploy_map "$dgx_b_target" "$dgx_b_root" "$destination_b"
  lane_a_map_manifest="$destination_a/manifest.json"
  lane_b_map_manifest="$destination_b/manifest.json"
  paired30_map_manifest_sha="$map_sha"
  python3 - "$receipt_path" "$code_sha" "$map_sha" "$lane_a_map_manifest" "$lane_b_map_manifest" <<'PY'
import json,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
 "schema_version":1,"status":"PASS","code_ref_sha":sys.argv[2],
 "episode_count":30,"dataset_sha256":"b4de2a6be2c37c1ad160aa83d062acd7ad7671534df76df478ec7c495fed61f5",
 "minimum_required_prefix_clearance_m":0.25,"manifest_sha256":sys.argv[3],
 "remote_manifest_paths":{"a":sys.argv[4],"b":sys.argv[5]},
 "recorded_unix":time.time(),
},indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
}

prepare_paired30_static_maps
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
  local static_map_manifest
  if test "$lane" = a; then
    static_map_manifest="$lane_a_map_manifest"
  else
    static_map_manifest="$lane_b_map_manifest"
  fi
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
    export INTERNNAV_T5_PAIRED30_STATIC_MAP_MANIFEST_PATH="$static_map_manifest"
    export INTERNNAV_T5_PAIRED30_STATIC_MAP_MANIFEST_SHA256="$paired30_map_manifest_sha"
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
