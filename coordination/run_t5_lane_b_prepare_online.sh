#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_t5_lane_b_prepare_online.sh CODE_SHA RUN_ID RESULT_ROOT

Lane-B-only exact-ref preparation for the Step3 live canary. RESULT_ROOT must
equal results/internnav_t5/step3-lane-b-prepare-RUN_ID. The command acquires
only the fail-closed lane-b lease (DGX_B plus Isaac GPU1).
EOF
  exit 64
}

[[ $# -eq 3 ]] || usage
code_sha="$1"
run_id="$2"
result_relative="$3"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
result_dir="$root/$result_relative"
expected_result="results/internnav_t5/step3-lane-b-prepare-$run_id"
[[ "$code_sha" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]{7,95}$ ]] || usage
test "$result_relative" = "$expected_result" || usage

git_command=(git -C "$root")
if [[ -f "$root/.git" ]] && grep -Eq '^gitdir: [A-Za-z]:/' "$root/.git"; then
  command -v git.exe >/dev/null
  command -v wslpath >/dev/null
  git_command=(git.exe -C "$(wslpath -w "$root")")
fi
"${git_command[@]}" cat-file -e "$code_sha^{commit}"
test "$("${git_command[@]}" rev-parse HEAD | tr -d '\r')" = "$code_sha"
test -z "$("${git_command[@]}" status --porcelain --untracked-files=all | tr -d '\r')"

if [[ "${INTERNNAV_T5_INSIDE_LANE_B_PREPARE:-0}" != 1 ]]; then
  test ! -e "$result_dir"
  umask 077
  mkdir -p "$result_dir/logs" "$result_dir/lease" "$result_dir/remote/x86" \
    "$result_dir/remote/dgx_b" "$result_dir/maps"
  bash "$root/scripts/create_source_bundle.sh" "$code_sha" \
    "$result_dir/deployment.tar.gz" 1
  sha256sum "$result_dir/deployment.tar.gz" | cut -d' ' -f1 \
    >"$result_dir/deployment_archive_sha256.txt"
  set +e
  env ISAAC_HOST=10.100.120.123 bash "$root/scripts/with_resource_lease.sh" lane-b \
    --owner codex-t5-step3 --task "t5-step3-lane-b-prepare:$run_id:$code_sha" \
    --log-dir "$result_dir/lease" --acquire-timeout 7200 \
    --cleanup-timeout 600 --kill-wait-timeout 60 -- \
    env INTERNNAV_T5_INSIDE_LANE_B_PREPARE=1 \
      INTERNNAV_T5_RESOURCE_LEASE_ACK=lane-b \
      bash "$root/coordination/run_t5_lane_b_prepare_online.sh" \
        "$code_sha" "$run_id" "$result_relative"
  rc=$?
  set -e
  python3 - "$result_dir/lease_release_summary.json" \
    "$result_dir/lease/lease_metadata.txt" "$result_dir/lease/lease_cleanup_receipt.json" \
    "$rc" <<'PY'
import json, sys, time
from pathlib import Path
output, metadata_path, cleanup_path = map(Path, sys.argv[1:4])
rc = int(sys.argv[4])
metadata = metadata_path.read_text(encoding="utf-8") if metadata_path.is_file() else ""
cleanup = json.loads(cleanup_path.read_text(encoding="utf-8")) if cleanup_path.is_file() else {}
checks = {
    "dgx_b_held": "dgx_b" in metadata,
    "isaac_gpu1_held": "isaac_gpu1" in metadata,
    "lane_a_resources_absent": "dgx_a" not in metadata and "isaac_gpu0" not in metadata,
    "released": "state=RELEASED" in metadata,
    "cleanup_pass": cleanup.get("status") == "PASS" and cleanup.get(
        "wrapped_process_group_absent_before_lock_release"
    ) is True,
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "command_exit": rc,
    "resource_profile": "lane-b",
    "checks": checks,
    "recorded_unix": time.time(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  exit "$rc"
fi

test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = lane-b
archive="$result_dir/deployment.tar.gz"
archive_sha="$(tr -d '\r\n' <"$result_dir/deployment_archive_sha256.txt")"
test "$(sha256sum "$archive" | cut -d' ' -f1)" = "$archive_sha"
tag="${run_id}-${code_sha:0:12}"
dgx_root="/home/rail/internnav-t1-t2/.t5-deployments/${tag}-lane-b"
x86_root="/home/song/internnav-t1-t2/.t5-deployments/${tag}-isaac-b"
dgx_target=rail@10.100.120.116
x86_target=song@10.100.120.123
ssh_options=(-T -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2)

remote() {
  local target="$1"; shift
  ssh "${ssh_options[@]}" "$target" "$@"
}

deploy() {
  local target="$1" destination="$2" stage
  stage="/tmp/t5-step3-${tag}-$(basename "$destination").tar.gz"
  remote "$target" "set -euo pipefail; umask 077; test ! -e '$destination'; test ! -e '$stage'; : >'$stage'"
  ssh "${ssh_options[@]}" "$target" "cat >'$stage'" <"$archive"
  remote "$target" \
    "set -euo pipefail; test \"\$(sha256sum '$stage'|cut -d' ' -f1)\" = '$archive_sha'; install -d -m 700 '$destination'; gzip -dc '$stage'|tar -C '$destination' -xf -; printf '%s\n' '$code_sha' >'$destination/T5_DEPLOYMENT_REF'; printf '%s\n' '$archive_sha' >'$destination/T5_DEPLOYMENT_ARCHIVE_SHA256'; rm -f '$stage'"
}

remote "$dgx_target" \
  "set -euo pipefail; test \"\$(id -un)\" = rail; test ! -e /tmp/internnav_dgx.quarantine; test -z \"\$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|sed '/^[[:space:]]*$/d')\"; test ! -e '$dgx_root'"
remote "$x86_target" \
  "set -euo pipefail; test \"\$(id -un)\" = song; test ! -e /tmp/internnav_isaac.quarantine; test ! -e /tmp/internnav_isaac_gpu1.quarantine; test -z \"\$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader|sed '/^[[:space:]]*$/d')\"; test ! -e '$x86_root'; if docker inspect internnav_t5_isaac_b >/dev/null 2>&1; then test \"\$(docker inspect -f '{{.State.Running}}' internnav_t5_isaac_b)\" = false; fi"

deploy "$x86_target" "$x86_root"
deploy "$dgx_target" "$dgx_root"

mapfile -t frozen < <(python3 - "$root/configs/internnav_t5/d0_run_manifest.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))["fixed_input"]
print(value["dataset_remote_path"])
print(value["dataset_file_sha256"])
print("|".join(value["episode_keys"]))
PY
)
dataset_root="${frozen[0]}"
dataset_sha="${frozen[1]}"
episode_keys="${frozen[2]}"

read -r -d '' x86_program <<'REMOTE_X86' || true
set -euo pipefail
root="$1"; dataset_root="$2"; dataset_sha="$3"; episode_keys="$4"
audit="$root/results/lane_b_prepare"
mkdir -p "$audit" "$root/inputs"
dataset="$dataset_root/val_unseen/val_unseen.json.gz"
test "$(sha256sum "$dataset"|cut -d' ' -f1)" = "$dataset_sha"
python3 - "$dataset" "$dataset_sha" "$episode_keys" "$audit/dataset_audit.json" <<'PY'
import gzip, json, sys
from pathlib import Path
with gzip.open(sys.argv[1], "rt", encoding="utf-8") as stream:
    value = json.load(stream)
episodes = value.get("episodes") if isinstance(value, dict) else None
keys = [f"{item['trajectory_id']}_{item['episode_id']}" for item in episodes or []]
expected = sys.argv[3].split("|")
checks = {"episode_count": len(keys) == 5, "episode_keys": keys == expected}
payload = {"schema_version": 1, "status": "PASS" if all(checks.values()) else "FAIL",
           "dataset_file": sys.argv[1], "dataset_sha256": sys.argv[2],
           "episode_count": len(keys), "episode_keys": keys, "checks": checks}
Path(sys.argv[4]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
raise SystemExit(0 if payload["status"] == "PASS" else 1)
PY
static_output="$root/inputs/d0_fixed5_static_maps"
internnav_root="$HOME/internnav-t0/InternNav"
cache_dir="$root/runtime/static_maps/dgx_onboard_${dataset_sha:0:16}_0p30"
test ! -e "$static_output"
mkdir -p "$root/runtime/static_maps"
if test ! -f "$cache_dir/manifest.json"; then
  test ! -e "$cache_dir"
  python3 "$root/scripts/build_t3_static_maps.py" \
    --dataset "$dataset" --scene-root "$internnav_root/data/scene_data/mp3d_pe" \
    --output-root "$cache_dir" --minimum-required-prefix-clearance-m 0.30
fi
mkdir "$static_output"
cp "$cache_dir"/*.bin "$static_output/"
python3 "$root/scripts/build_t4_truth_isolated_static_manifest.py" \
  --manifest "$cache_dir/manifest.json" --dataset "$dataset" \
  --output "$static_output/manifest.json"
python3 "$root/scripts/validate_t5_frozen_assets.py" static-map \
  --golden "$root/configs/internnav_t5/golden_bundle_manifest.json" \
  --manifest "$static_output/manifest.json" \
  --output "$audit/static_map_source_audit.json"
ownership="$audit/containers_owned"
mkdir -p "$ownership"
INTERNNAV_T1_CONTROL_ROOT=/home/song/internnav-t1-t2 \
INTERNVLA_ROS_WS=/home/song/internnav-t4/isaac_ros_ws_45 \
INTERNVLA_T5_ISAAC_WORKER_ROOT=/home/song/internnav-t1-t2/runtime/t5_isaac_workers \
INTERNVLA_T5_X86_LANE_B_ROOT="$root" \
INTERNVLA_T5_CONTAINER_OWNERSHIP_DIR="$ownership" \
INTERNNAV_T5_ISAAC_PREPARE_SCOPE=lane_b \
INTERNNAV_T5_RESOURCE_LEASE_ACK=lane-b \
INTERNNAV_RUNTIME_POLICY=completion_sim INTERNNAV_SIMULATION_TARGET=isaac \
INTERNVLA_T5_ISAAC_IP=10.100.120.123 \
INTERNVLA_T5_LANE_B_CPUSET=1,3,5,7,9,11,13,15,17 \
  bash "$root/scripts/prepare_t5_isaac_workers.sh" "$audit/isaac_worker_prepare.json"
docker stop -t 20 internnav_t5_isaac_b >/dev/null
test "$(docker inspect -f '{{.State.Running}}' internnav_t5_isaac_b)" = false
test "$(docker inspect -f '{{.State.Pid}}' internnav_t5_isaac_b)" = 0
docker inspect internnav_t5_isaac_b >"$audit/container_b_inspect.json"
python3 "$root/scripts/validate_t5_isaac_worker_spec.py" \
  /home/song/internnav-t1-t2/runtime/t5_isaac_workers/b/expected_container_spec.json \
  "$audit/container_b_inspect.json" "$audit/container_b_spec_audit.json"
python3 - "$audit/x86_prepare_summary.json" "$audit/dataset_audit.json" \
  "$audit/static_map_source_audit.json" "$audit/isaac_worker_prepare.json" \
  "$audit/container_b_spec_audit.json" <<'PY'
import json, sys, time
from pathlib import Path
values = [json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[2:]]
checks = {"dataset": values[0].get("status") == "PASS",
          "static_map": values[1].get("status") == "PASS",
          "lane_b_worker": values[2].get("status") == "PASS" and values[2].get("prepare_scope") == "lane_b",
          "container_spec": values[3].get("status") == "PASS"}
payload = {"schema_version": 1, "status": "PASS" if all(checks.values()) else "FAIL",
           "scope": "lane_b_only", "checks": checks, "recorded_unix": time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
raise SystemExit(0 if payload["status"] == "PASS" else 1)
PY
REMOTE_X86
x86_b64="$(printf '%s' "$x86_program" | base64 | tr -d '\r\n')"
remote "$x86_target" \
  "exec taskset -c 1,3,5,7,9,11,13,15,17 bash -c \"\$(printf '%s' '$x86_b64'|base64 -d)\" lane-b-x86 '$x86_root' '$dataset_root' '$dataset_sha' '$episode_keys'" \
  >"$result_dir/logs/x86_prepare.log" 2>&1
remote "$x86_target" "tar -C '$x86_root/inputs/d0_fixed5_static_maps' -czf - ." \
  >"$result_dir/maps/d0_fixed5_static_maps.tar.gz"
remote "$x86_target" "tar -C '$x86_root/results/lane_b_prepare' -czf - ." | \
  tar -C "$result_dir/remote/x86" -xzf -
map_manifest_sha="$(remote "$x86_target" "sha256sum '$x86_root/inputs/d0_fixed5_static_maps/manifest.json'|cut -d' ' -f1")"
remote "$dgx_target" "install -d -m 700 '$dgx_root/inputs/d0_fixed5_static_maps'"
gzip -dc "$result_dir/maps/d0_fixed5_static_maps.tar.gz" | \
  ssh "${ssh_options[@]}" "$dgx_target" \
    "tar -C '$dgx_root/inputs/d0_fixed5_static_maps' -xf -"
remote "$dgx_target" \
  "test \"\$(sha256sum '$dgx_root/inputs/d0_fixed5_static_maps/manifest.json'|cut -d' ' -f1)\" = '$map_manifest_sha'"

read -r -d '' dgx_program <<'REMOTE_DGX' || true
set -euo pipefail
root="$1"; map_sha="$2"
audit="$root/results/lane_b_prepare"
workspace="$root/ros_ws"
mkdir -p "$audit" "$workspace/src"
test "$(sha256sum "$root/inputs/d0_fixed5_static_maps/manifest.json"|cut -d' ' -f1)" = "$map_sha"
test -d /home/rail/ai-stack/models/Step3-VL-10B
test -f /home/rail/ai-stack/venvs/step3-vl-10b-tf4.57.6/T5_STEP3_RUNTIME_READY.json
ln -sfn "$root/internnav_t5_lane_b_msgs" \
  "$workspace/src/internnav_t5_lane_b_msgs"
"$HOME/internnav-t0/venv-model/bin/python" "$root/scripts/validate_t5_frozen_assets.py" checkpoint \
  --golden "$root/configs/internnav_t5/golden_bundle_manifest.json" \
  --content-manifest "$root/configs/internnav_t5/checkpoint_content_manifest.json" \
  --internnav-root "$HOME/internnav-t0/InternNav" \
  --output "$audit/model_inventory.json"
INTERNNAV_T1_CONTROL_ROOT="$root" INTERNVLA_ROS_WS="$workspace" \
INTERNNAV_RUNTIME_POLICY=completion_sim INTERNNAV_SIMULATION_TARGET=isaac \
INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx \
  bash "$root/scripts/build_t4_host_ros.sh" >"$audit/ros_build.log" 2>&1
INTERNNAV_T1_CONTROL_ROOT="$root" INTERNVLA_ROS_WS="$workspace" \
INTERNNAV_RUNTIME_POLICY=completion_sim INTERNNAV_SIMULATION_TARGET=isaac \
INTERNNAV_T5_LANE=b INTERNNAV_T5_LANE_NAMESPACE=/t5/lane_b \
INTERNNAV_T5_RESOURCE_LEASE_ACK=lane-b \
  bash "$root/scripts/build_t5_lane_b_step3_ros.sh" >"$audit/step3_ros_build.log" 2>&1
python3 - "$audit/dgx_prepare_summary.json" "$root" <<'PY'
import json, sys, time
from pathlib import Path
root = Path(sys.argv[2])
checks = {"deployment_ref": (root / "T5_DEPLOYMENT_REF").is_file(),
          "ros_workspace": (root / "ros_ws/install/setup.bash").is_file(),
          "private_interface": (root / "ros_ws/install/internnav_t5_lane_b_msgs").is_dir(),
          "internvla_inventory": (root / "results/lane_b_prepare/model_inventory.json").is_file()}
payload = {"schema_version": 1, "status": "PASS" if all(checks.values()) else "FAIL",
           "scope": "dgx_b_only", "checks": checks, "recorded_unix": time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
raise SystemExit(0 if payload["status"] == "PASS" else 1)
PY
REMOTE_DGX
dgx_b64="$(printf '%s' "$dgx_program" | base64 | tr -d '\r\n')"
remote "$dgx_target" \
  "exec bash -c \"\$(printf '%s' '$dgx_b64'|base64 -d)\" lane-b-dgx '$dgx_root' '$map_manifest_sha'" \
  >"$result_dir/logs/dgx_prepare.log" 2>&1
remote "$dgx_target" "tar -C '$dgx_root/results/lane_b_prepare' -czf - ." | \
  tar -C "$result_dir/remote/dgx_b" -xzf -

python3 - "$result_dir" "$code_sha" "$archive_sha" "$dgx_root" "$x86_root" \
  "$dataset_root" "$dataset_sha" "$episode_keys" "$map_manifest_sha" <<'PY'
import hashlib, json, sys, time
from pathlib import Path
result = Path(sys.argv[1])
code_sha, archive_sha, dgx_root, x86_root, dataset_root, dataset_sha, keys_raw, map_sha = sys.argv[2:]
keys = keys_raw.split("|")
dataset_path = result / "remote/x86/dataset_audit.json"
dataset_audit_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
input_payload = {
    "schema_version": 1, "status": "PASS", "authorization_mode": "LANE_B_EXACT_REF",
    "prepare_scope": "lane-b", "prepared_lanes": ["b"],
    "code_ref_sha": code_sha, "deployment_archive_sha256": archive_sha,
    "deployment_roots": {"dgx_b": dgx_root, "x86_b": x86_root},
    "dataset_root": dataset_root, "dataset_file_sha256": dataset_sha,
    "episode_count": 5, "episode_keys": keys,
}
input_path = result / "fast_prepare_input.json"
input_path.write_text(json.dumps(input_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
checks = {"lane_b_only": True, "dgx_b_prepare": json.loads(
    (result / "remote/dgx_b/dgx_prepare_summary.json").read_text())["status"] == "PASS",
    "x86_b_prepare": json.loads(
        (result / "remote/x86/x86_prepare_summary.json").read_text())["status"] == "PASS"}
summary = {
    "schema_version": 1, "status": "PASS" if all(checks.values()) else "FAIL",
    "prepare_scope": "lane-b", "prepared_lanes": ["b"],
    "code_ref_sha": code_sha, "deployment_roots": input_payload["deployment_roots"],
    "fast_prepare_input_sha256": input_sha, "dataset_audit_sha256": dataset_audit_sha,
    "dataset_file_sha256": dataset_sha, "episode_count": 5, "episode_keys": keys,
    "static_map_manifest_sha256": map_sha, "checks": checks,
    "recorded_unix": time.time(),
}
summary_path = result / "d0_prepare_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
final = {
    **summary,
    "preparation_summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
}
(result / "d0_prepare_final_summary.json").write_text(
    json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY
