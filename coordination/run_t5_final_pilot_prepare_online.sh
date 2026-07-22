#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_t5_final_pilot_prepare_online.sh RUN_ID CODE_SHA BASE_PREP_RESULT_ROOT RESULT_ROOT

Prepare the frozen full20 source, deterministic disjoint A10/B10 overlays and
one common five-scene static-map bundle.  This stage starts neither InternVLA
nor Isaac.  The x86 map build is serialized only by the shared asset-I/O lock;
the frozen D0 preparation and its maps are never modified.
EOF
  exit 64
}

[[ $# -eq 4 ]] || usage
run_id="$1"
code_sha="$2"
base_prepare_relative="$3"
result_relative="$4"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
x86_ip="${ISAAC_HOST:-10.100.120.123}"
x86_target="song@$x86_ip"
dgx_a_target=railgun@10.100.100.128
lane_a_cpuset=0,2,4,6,8,10,12,14,16
source_dataset_root="${INTERNNAV_T5_FINAL_PILOT_SOURCE_ROOT:-/home/song/internnav-t1-t2/episodes/t3_model_pilot_go2_clear_v2}"
expected_source_sha=f09b11004d15d579620ba2b2769dcbf863efe4cf04d9e404f071f66244463f8b

[[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]{7,95}$ ]] || usage
[[ "$code_sha" =~ ^[0-9a-f]{40}$ ]] || usage
test "$x86_ip" = 10.100.120.123
test "$source_dataset_root" = /home/song/internnav-t1-t2/episodes/t3_model_pilot_go2_clear_v2
[[ "$base_prepare_relative" =~ ^results/internnav_t5/d0-0-prepare-t5d00[0-9]{8}t[0-9]{6}$ ]] || usage
expected_result="results/internnav_t5/final-pilot-prepare-${run_id}"
test "$result_relative" = "$expected_result" || {
  echo "RESULT_ROOT must equal $expected_result" >&2
  exit 64
}

git_command=(git -C "$root")
if [[ -f "$root/.git" ]] && grep -Eq '^gitdir: [A-Za-z]:/' "$root/.git"; then
  command -v git.exe >/dev/null
  command -v wslpath >/dev/null
  git_command=(git.exe -C "$(wslpath -w "$root")")
fi
"${git_command[@]}" cat-file -e "$code_sha^{commit}"
test "$("${git_command[@]}" rev-parse HEAD | tr -d '\r')" = "$code_sha"
test -z "$("${git_command[@]}" status --porcelain --untracked-files=all | tr -d '\r')"

base_prepare="$root/$base_prepare_relative"
result_dir="$root/$result_relative"
test -d "$base_prepare"
test ! -L "$base_prepare"
for name in fast_prepare_input.json d0_prepare_summary.json d0_prepare_final_summary.json; do
  test -f "$base_prepare/$name"
  test ! -L "$base_prepare/$name"
done
test -f "$root/scripts/materialize_t5_final_pilot_split.py"
test -f "$root/scripts/build_t5_final_pilot_prepare_receipt.py"
test ! -e "$result_dir"
umask 077
mkdir -p "$result_dir/assets/source/val_unseen" "$result_dir/maps" "$result_dir/logs" "$result_dir/remote"

mapfile -t roots < <(python3 - "$base_prepare/fast_prepare_input.json" \
  "$base_prepare/d0_prepare_summary.json" \
  "$base_prepare/d0_prepare_final_summary.json" "$code_sha" <<'PY'
import hashlib,json,re,sys
from pathlib import Path
inputs,summary,final=map(Path,sys.argv[1:4])
code=sys.argv[4]
values=[json.loads(path.read_text(encoding="utf-8")) for path in (inputs,summary,final)]
inp,summ,end=values
checks={
 "input_code":inp.get("code_ref_sha")==code,
 "summary_pass":summ.get("status")=="PASS" and bool(summ.get("checks")) and all(summ["checks"].values()),
 "final_pass":end.get("status")=="PASS" and bool(end.get("checks")) and all(end["checks"].values()),
 "summary_code":summ.get("code_ref_sha")==end.get("code_ref_sha")==code,
 "summary_sealed":end.get("preparation_summary_sha256")==hashlib.sha256(summary.read_bytes()).hexdigest(),
 "same_roots":summ.get("deployment_roots")==end.get("deployment_roots"),
}
roots=end.get("deployment_roots",{})
scope_values=[(value.get("prepare_scope"),value.get("prepared_lanes")) for value in values]
lane_a_scope=all(scope=="lane-a" and lanes==["a"] for scope,lanes in scope_values)
explicit_dual=all(scope=="dual" and lanes==["a","b"] for scope,lanes in scope_values)
legacy_dual=all(scope is None and lanes is None for scope,lanes in scope_values)
checks["scope_consistent"]=lane_a_scope or explicit_dual or legacy_dual
required_root_keys=({"dgx_a","x86_a"} if lane_a_scope else
                    {"dgx_a","dgx_b","x86_a","x86_b"})
allowed_root_keys=required_root_keys|{"x86_prepare"}
checks["root_keys"]=required_root_keys.issubset(roots) and set(roots).issubset(allowed_root_keys)
safe=re.compile(r"^/[A-Za-z0-9._/-]+$")
checks["root_shape"]=all(isinstance(value,str) and safe.fullmatch(value) for value in roots.values())
if not all(checks.values()): raise SystemExit(f"base prepare binding failed: {checks}")
print("lane-a" if lane_a_scope else "dual")
for key in (("dgx_a","x86_a") if lane_a_scope else
            ("dgx_a","dgx_b","x86_a","x86_b")): print(roots[key])
PY
)
prepare_scope="${roots[0]}"
dgx_a_root="${roots[1]}"
if test "$prepare_scope" = lane-a; then
  x86_a_root="${roots[2]}"
  prepared_lanes=a
else
  dgx_b_target=rail@10.100.120.116
  dgx_b_root="${roots[2]}"
  x86_a_root="${roots[3]}"
  x86_b_root="${roots[4]}"
  prepared_lanes=a,b
fi

ssh_options=(-T -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2)
remote() { local target="$1"; shift; ssh "${ssh_options[@]}" "$target" "$@"; }
bindings=(
  "$dgx_a_target|railgun|10.100.100.128|$dgx_a_root"
  "$x86_target|song|$x86_ip|$x86_a_root"
)
if test "$prepare_scope" = dual; then
  bindings+=("$dgx_b_target|rail|10.100.120.116|$dgx_b_root")
  bindings+=("$x86_target|song|$x86_ip|$x86_b_root")
fi
for binding in "${bindings[@]}"; do
  IFS='|' read -r target user ip deployment <<<"$binding"
  remote "$target" \
    "set -euo pipefail; test \"\$(id -un)\" = '$user'; ip -4 -o addr show | grep -Fq ' $ip/'; test \"\$(cat '$deployment/T5_DEPLOYMENT_REF')\" = '$code_sha'"
done
remote "$x86_target" \
  "set -euo pipefail; test -f '$source_dataset_root/val_unseen/val_unseen.json.gz'; test \"\$(sha256sum '$source_dataset_root/val_unseen/val_unseen.json.gz'|cut -d' ' -f1)\" = '$expected_source_sha'"
remote "$x86_target" \
  "flock -s /tmp/internnav_t5_isaac_shared_assets.lock cat '$source_dataset_root/val_unseen/val_unseen.json.gz'" \
  >"$result_dir/assets/source/val_unseen/val_unseen.json.gz"
test "$(sha256sum "$result_dir/assets/source/val_unseen/val_unseen.json.gz" | cut -d' ' -f1)" = "$expected_source_sha"

python3 "$root/scripts/materialize_t5_final_pilot_split.py" \
  --source "$result_dir/assets/source/val_unseen/val_unseen.json.gz" \
  --lane-a-output-root "$result_dir/assets/lane_a" \
  --lane-b-output-root "$result_dir/assets/lane_b" \
  --audit-output "$result_dir/assets/final_pilot_split_audit.json" \
  >"$result_dir/logs/split_materialization.json"
lane_a_sha="$(sha256sum "$result_dir/assets/lane_a/val_unseen/val_unseen.json.gz" | cut -d' ' -f1)"
lane_b_sha="$(sha256sum "$result_dir/assets/lane_b/val_unseen/val_unseen.json.gz" | cut -d' ' -f1)"
[[ "$lane_a_sha" =~ ^[0-9a-f]{64}$ ]]
[[ "$lane_b_sha" =~ ^[0-9a-f]{64}$ ]]

asset_archive="$result_dir/assets/final_pilot_inputs.tar.gz"
if test "$prepare_scope" = lane-a; then
  tar -C "$result_dir/assets" -czf "$asset_archive" lane_a
else
  tar -C "$result_dir/assets" -czf "$asset_archive" source lane_a lane_b
fi
asset_archive_sha="$(sha256sum "$asset_archive" | cut -d' ' -f1)"
remote_archive="/tmp/internnav-t5-final-pilot-${run_id}-${code_sha:0:12}.tar.gz"
remote "$x86_target" "set -euo pipefail; test ! -e '$remote_archive'; : >'$remote_archive'"
ssh "${ssh_options[@]}" "$x86_target" "cat >'$remote_archive'" <"$asset_archive"
remote "$x86_target" "test \"\$(sha256sum '$remote_archive'|cut -d' ' -f1)\" = '$asset_archive_sha'"

stage_root="/home/song/internnav-t1-t2/.t5-final-pilot-assets/${run_id}-${code_sha:0:12}"
lane_a_remote="$x86_a_root/inputs/final_pilot_${run_id}_a10"
if test "$prepare_scope" = dual; then
  lane_b_remote="$x86_b_root/inputs/final_pilot_${run_id}_b10"
else
  lane_b_remote=-
fi
read -r -d '' x86_prepare_program <<'REMOTE_X86' || true
set -euo pipefail
archive="$1"; stage="$2"; deployment="$3"; code="$4"; source_sha="$5"; lane_a_sha="$6"; lane_b_sha="$7"; lane_a_dest="$8"; lane_b_dest="$9"; prepare_scope="${10}"; source_dataset_root="${11}"
case "$prepare_scope" in lane-a|dual) ;; *) exit 64 ;; esac
test "$(cat "$deployment/T5_DEPLOYMENT_REF")" = "$code"
test ! -e "$stage"; test ! -e "$lane_a_dest"
test "$prepare_scope" != dual || test ! -e "$lane_b_dest"
mkdir -p "$(dirname "$stage")" "$stage/input"
tar -C "$stage/input" -xzf "$archive"
if test "$prepare_scope" = lane-a; then
  source="$source_dataset_root/val_unseen/val_unseen.json.gz"
else
  source="$stage/input/source/val_unseen/val_unseen.json.gz"
fi
lane_a="$stage/input/lane_a/val_unseen/val_unseen.json.gz"
test "$(sha256sum "$source"|cut -d' ' -f1)" = "$source_sha"
test "$(sha256sum "$lane_a"|cut -d' ' -f1)" = "$lane_a_sha"
if test "$prepare_scope" = dual; then
  lane_b="$stage/input/lane_b/val_unseen/val_unseen.json.gz"
  test "$(sha256sum "$lane_b"|cut -d' ' -f1)" = "$lane_b_sha"
fi
cache="$stage/map_cache"
python3 "$deployment/scripts/build_t3_static_maps.py" \
  --dataset "$source" \
  --scene-root "$HOME/internnav-t0/InternNav/data/scene_data/mp3d_pe" \
  --output-root "$cache" \
  --minimum-required-prefix-clearance-m 0.30
maps="$stage/final_pilot_static_maps"
mkdir "$maps"
cp "$cache"/*.bin "$maps/"
python3 "$deployment/scripts/build_t4_truth_isolated_static_manifest.py" \
  --manifest "$cache/manifest.json" --dataset "$source" \
  --output "$maps/manifest.json"
python3 - "$maps/manifest.json" "$source_sha" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
keys=[f"{item['trajectory_id']}_{item['episode_id']}" for item in value.get("generations",[])]
raw_maps=value.get("maps",{})
map_records=raw_maps.values() if isinstance(raw_maps,dict) else raw_maps
scans={item.get("scan") for item in map_records if isinstance(item,dict)}
assert value.get("dataset_sha256")==sys.argv[2]
assert value.get("episode_count")==len(keys)==20 and len(set(keys))==20
assert scans=={"2azQ1b91cZZ","QUCTc6BB5sX","TbHJrupSAjP","Z6MFQCViBuw","zsNo4HB9uLZ"}
assert value.get("t4_truth_isolation",{}).get("status")=="PASS"
PY
tar -C "$maps" -czf "$stage/final_pilot_static_maps.tar.gz" .
mkdir -p "$(dirname "$lane_a_dest")"
mv "$stage/input/lane_a" "$lane_a_dest"
if test "$prepare_scope" = dual; then
  mkdir -p "$(dirname "$lane_b_dest")"
  mv "$stage/input/lane_b" "$lane_b_dest"
fi
python3 - "$stage/remote_prepare.json" "$source" "$lane_a_dest" "$lane_b_dest" "$maps/manifest.json" "$code" "$prepare_scope" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
source,lane_a,maps=Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[5])
scope=sys.argv[7]
lanes={"a":lane_a}
if scope=="dual": lanes["b"]=Path(sys.argv[4])
payload={"schema_version":1,"status":"PASS","code_ref_sha":sys.argv[6],"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
 "prepare_scope":scope,"prepared_lanes":sorted(lanes),
 "lane_dataset_sha256":{lane:hashlib.sha256((path/"val_unseen/val_unseen.json.gz").read_bytes()).hexdigest() for lane,path in lanes.items()},
 "lane_dataset_roots":{lane:str(path) for lane,path in lanes.items()},
 "map_manifest_sha256":hashlib.sha256(maps.read_bytes()).hexdigest(),"recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
REMOTE_X86
x86_prepare_b64="$(printf '%s' "$x86_prepare_program" | base64 | tr -d '\r\n')"
if test "$prepare_scope" = lane-a; then
  remote "$x86_target" \
    "flock -x -w 1800 /tmp/internnav_t5_isaac_shared_assets.lock taskset -c '$lane_a_cpuset' bash -c \"\$(printf '%s' '$x86_prepare_b64'|base64 -d)\" final-pilot-assets '$remote_archive' '$stage_root' '$x86_a_root' '$code_sha' '$expected_source_sha' '$lane_a_sha' '-' '$lane_a_remote' '-' lane-a '$source_dataset_root'" \
    >"$result_dir/logs/x86_asset_prepare.log" 2>&1
else
  remote "$x86_target" \
    "flock -x -w 1800 /tmp/internnav_t5_isaac_shared_assets.lock bash -c \"\$(printf '%s' '$x86_prepare_b64'|base64 -d)\" final-pilot-assets '$remote_archive' '$stage_root' '$x86_a_root' '$code_sha' '$expected_source_sha' '$lane_a_sha' '$lane_b_sha' '$lane_a_remote' '$lane_b_remote' dual '$source_dataset_root'" \
    >"$result_dir/logs/x86_asset_prepare.log" 2>&1
fi
remote "$x86_target" "cat '$stage_root/remote_prepare.json'" >"$result_dir/remote/x86_prepare.json"
remote "$x86_target" "cat '$stage_root/final_pilot_static_maps.tar.gz'" \
  >"$result_dir/maps/final_pilot_static_maps.tar.gz"
mkdir "$result_dir/maps/final_pilot_static_maps"
tar -C "$result_dir/maps/final_pilot_static_maps" -xzf \
  "$result_dir/maps/final_pilot_static_maps.tar.gz"
map_manifest_sha="$(sha256sum "$result_dir/maps/final_pilot_static_maps/manifest.json" | cut -d' ' -f1)"

deploy_map() {
  local target="$1" deployment="$2" expected_user="$3" expected_ip="$4"
  local destination="$deployment/inputs/final_pilot_${run_id}_static_maps"
  local temporary="$destination.tmp-${run_id}"
  remote "$target" \
    "set -euo pipefail; test \"\$(id -un)\" = '$expected_user'; ip -4 -o addr show | grep -Fq ' $expected_ip/'; test \"\$(cat '$deployment/T5_DEPLOYMENT_REF')\" = '$code_sha'; test ! -e '$destination'; test ! -e '$temporary'; mkdir -p '$temporary'"
  gzip -dc "$result_dir/maps/final_pilot_static_maps.tar.gz" | \
    ssh "${ssh_options[@]}" "$target" "tar -C '$temporary' -xf -"
  remote "$target" \
    "set -euo pipefail; test \"\$(sha256sum '$temporary/manifest.json'|cut -d' ' -f1)\" = '$map_manifest_sha'; mv '$temporary' '$destination'; printf '%s\n' '$destination/manifest.json'; sha256sum '$destination/manifest.json'|cut -d' ' -f1"
}
mapfile -t lane_a_map < <(deploy_map "$dgx_a_target" "$dgx_a_root" railgun 10.100.100.128)
lane_a_map_manifest="${lane_a_map[0]}"
test "${lane_a_map[1]}" = "$map_manifest_sha"
python3 - "$result_dir/remote/dgx_a_map.json" a "$code_sha" \
  "$lane_a_map_manifest" "${lane_a_map[1]}" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"schema_version":1,"status":"PASS","lane":sys.argv[2],
 "code_ref_sha":sys.argv[3],"manifest_path":sys.argv[4],"manifest_sha256":sys.argv[5]},indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
if test "$prepare_scope" = dual; then
  mapfile -t lane_b_map < <(deploy_map "$dgx_b_target" "$dgx_b_root" rail 10.100.120.116)
  lane_b_map_manifest="${lane_b_map[0]}"
  test "${lane_b_map[1]}" = "$map_manifest_sha"
  python3 - "$result_dir/remote/dgx_b_map.json" b "$code_sha" \
    "$lane_b_map_manifest" "${lane_b_map[1]}" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"schema_version":1,"status":"PASS","lane":sys.argv[2],
 "code_ref_sha":sys.argv[3],"manifest_path":sys.argv[4],"manifest_sha256":sys.argv[5]},indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
fi

receipt_args=(
  --code-sha "$code_sha"
  --base-prepare-root "$base_prepare"
  --split-audit "$result_dir/assets/final_pilot_split_audit.json"
  --map-dir "$result_dir/maps/final_pilot_static_maps"
  --map-archive "$result_dir/maps/final_pilot_static_maps.tar.gz"
  --remote-x86-receipt "$result_dir/remote/x86_prepare.json"
  --remote-lane-a-map-receipt "$result_dir/remote/dgx_a_map.json"
  --remote-lane-a-dataset-root "$lane_a_remote"
  --remote-lane-a-map-manifest "$lane_a_map_manifest"
  --output "$result_dir/final_pilot_prepare_receipt.json"
)
if test "$prepare_scope" = dual; then
  receipt_args+=(
    --remote-lane-b-map-receipt "$result_dir/remote/dgx_b_map.json"
    --remote-lane-b-dataset-root "$lane_b_remote"
    --remote-lane-b-map-manifest "$lane_b_map_manifest"
  )
fi
python3 "$root/scripts/build_t5_final_pilot_prepare_receipt.py" "${receipt_args[@]}" \
  >"$result_dir/logs/final_receipt.json"
remote "$x86_target" "rm -f -- '$remote_archive'"
python3 - "$result_dir/final_pilot_prepare_summary.json" \
  "$result_dir/final_pilot_prepare_receipt.json" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
output,receipt=map(Path,sys.argv[1:])
value=json.loads(receipt.read_text(encoding="utf-8"))
payload={"schema_version":1,"status":value.get("status"),"stage":"t5_final_pilot_prepare",
 "prepare_scope":value.get("prepare_scope"),"prepared_lanes":value.get("prepared_lanes"),
 "receipt_sha256":hashlib.sha256(receipt.read_bytes()).hexdigest(),"online_workloads_started":False,
 "shared_io_scope":"full20_split_upload_and_five_scene_static_map_build","recorded_unix":time.time()}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
