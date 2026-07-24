#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || {
  echo "usage: prepare_t5_isaac_workers.sh RESULT_JSON" >&2
  exit 64
}
result_json="$1"
root="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"
ros_ws="${INTERNVLA_ROS_WS:-$HOME/internnav-t4/isaac_ros_ws_45}"
image="${INTERNVLA_T5_ISAAC_ROS_IMAGE:-cached_isaac_run_dev_image_local:latest}"
worker_root="${INTERNVLA_T5_ISAAC_WORKER_ROOT:-$root/runtime/t5_isaac_workers}"
prepare_scope="${INTERNVLA_T5_ISAAC_PREPARE_SCOPE:-dual}"
if test -n "${INTERNNAV_T5_ISAAC_PREPARE_SCOPE:-}"; then
  test ! -v INTERNVLA_T5_ISAAC_PREPARE_SCOPE || {
    echo "set only one Isaac prepare scope variable" >&2
    exit 64
  }
  prepare_scope="$INTERNNAV_T5_ISAAC_PREPARE_SCOPE"
fi
case "$prepare_scope" in
  both) prepare_scope=dual ;;
  lane-a) prepare_scope=a ;;
  lane_b|lane-b) prepare_scope=b ;;
esac
lane_a_deployment_root="${INTERNVLA_T5_X86_LANE_A_ROOT:-}"
lane_b_deployment_root="${INTERNVLA_T5_X86_LANE_B_ROOT:-}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
validator="$script_dir/validate_t5_isaac_worker_spec.py"
ownership_dir="${INTERNVLA_T5_CONTAINER_OWNERSHIP_DIR:-}"

case "$prepare_scope" in
  dual)
    : "${INTERNVLA_T5_X86_LANE_B_ROOT:?set INTERNVLA_T5_X86_LANE_B_ROOT}"
    case "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" in
      all-lanes)
        # Formal D0.0 preparation owns both DGXs and both Isaac GPUs.
        ;;
      isaac)
        # The engineering fast-prepare entry owns both Isaac GPU locks plus the
        # legacy/global Isaac lock.  This is sufficient for the host-local shared
        # assets and the two stopped worker containers; it never touches a DGX.
        ;;
      *)
        echo "dual Isaac worker preparation requires all-lanes or isaac lease" >&2
        exit 2
        ;;
    esac
    ;;
  a)
    : "${INTERNVLA_T5_X86_LANE_A_ROOT:?set INTERNVLA_T5_X86_LANE_A_ROOT}"
    test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = lane-a || {
      echo "Lane-A Isaac worker preparation requires the lane-a lease" >&2
      exit 2
    }
    # The value is evidence for a mount that must not exist.  The Lane-A path
    # neither stats nor mutates a Lane-B deployment or worker profile.
    lane_b_deployment_root="$root/.t5-deployments/RESERVED_EXTERNAL_THREAD_LANE_B"
    ;;
  b)
    : "${INTERNVLA_T5_X86_LANE_B_ROOT:?set INTERNVLA_T5_X86_LANE_B_ROOT}"
    test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = lane-b || {
      echo "Lane-B Isaac worker preparation requires the lane-b lease" >&2
      exit 2
    }
    # Lane-B-only preparation must neither inspect nor mutate Lane A.
    lane_a_deployment_root="$root/.t5-deployments/RESERVED_EXTERNAL_THREAD_LANE_A"
    ;;
  *)
    echo "Isaac prepare scope must be dual, a, or b" >&2
    exit 64
    ;;
esac
test "${INTERNNAV_RUNTIME_POLICY:-}" = completion_sim
test "${INTERNNAV_SIMULATION_TARGET:-}" = isaac
test "$(id -un)" = song
isaac_ip="${INTERNVLA_T5_ISAAC_IP:-10.100.120.123}"
[[ "$isaac_ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]
ip -4 -o addr show | grep -Fq " $isaac_ip/"
lane_a_cpuset="${INTERNVLA_T5_LANE_A_CPUSET:-0,2,4,6,8,10,12,14,16}"
lane_b_cpuset="${INTERNVLA_T5_LANE_B_CPUSET:-1,3,5,7,9,11,13,15,17}"
[[ "$lane_a_cpuset" =~ ^[0-9,-]+$ ]]
test "$isaac_ip" = 10.100.120.123
test "$lane_a_cpuset" = 0,2,4,6,8,10,12,14,16
if test "$prepare_scope" = dual || test "$prepare_scope" = b; then
  [[ "$lane_b_cpuset" =~ ^[0-9,-]+$ ]]
  test "$lane_b_cpuset" = 1,3,5,7,9,11,13,15,17
fi
test -d "$ros_ws/install"
test -d "$root"
if test "$prepare_scope" != b; then
  test -d "$lane_a_deployment_root"
  case "$lane_a_deployment_root" in "$root"/*) ;; *) exit 65 ;; esac
fi
if test "$prepare_scope" = dual; then
  test -d "$lane_b_deployment_root"
  case "$lane_b_deployment_root" in "$root"/*) ;; *) exit 65 ;; esac
  test "$lane_a_deployment_root" != "$lane_b_deployment_root"
elif test "$prepare_scope" = b; then
  test -d "$lane_b_deployment_root"
  case "$lane_b_deployment_root" in "$root"/*) ;; *) exit 65 ;; esac
fi
test -f "$validator"
docker image inspect "$image" >/dev/null
test ! -e "$result_json"
mkdir -p "$(dirname "$result_json")" "$worker_root"
test -z "$ownership_dir" || install -d -m 700 "$ownership_dir"
image_id="$(docker image inspect -f '{{.Id}}' "$image")"
image_repo_digests="$(docker image inspect -f '{{json .RepoDigests}}' "$image")"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/internnav-t5-worker-spec.XXXXXX")"
cleanup() { rm -rf -- "$scratch"; }
trap cleanup EXIT

# Container creation, recreation, and shader/cache maintenance mutate shared
# x86 state.  The caller holds either all-lanes or the host-local isaac lease;
# both exclude the two GPU lanes while this shared-assets section runs.  A
# stale running container is never taken over.
exec 8>/tmp/internnav_t5_isaac_shared_assets.lock
flock -x -w 30 8

prepare_lane() {
  local lane="$1" gpu="$2" domain="$3" cpuset="$4" container="$5"
  local lane_deployment_root="$6" other_lane_deployment_root="$7" other_lane="$8"
  local namespace static_peer identity_prefix
  if test "$lane" = a; then
    namespace=/t5/lane_a
    static_peer=10.100.100.128
    identity_prefix='a::'
  else
    namespace=/t5/lane_b
    static_peer=10.100.120.122
    identity_prefix='b::'
  fi
  local lane_root="$worker_root/$lane"
  local other_lane_root="$worker_root/$other_lane"
  local expected="$lane_root/expected_container_spec.json"
  local inspected="$scratch/${lane}_inspect.json"
  local audit="$scratch/${lane}_spec_audit.json"
  install -d -m 700 \
    "$lane_root/cache/xdg" "$lane_root/cache/ov" "$lane_root/cache/nvidia" \
    "$lane_root/config" "$lane_root/data" "$lane_root/tmp" "$lane_root/logs"
  python3 - "$expected" "$container" "$lane" "$gpu" "$domain" "$cpuset" \
    "$root" "$lane_deployment_root" "$other_lane_deployment_root" \
    "$lane_root" "$other_lane_root" "$ros_ws" "$image" "$image_id" \
    "$(id -u)" "$(id -g)" "$namespace" "$static_peer" \
    "$identity_prefix" <<'PY'
import json,sys
from pathlib import Path
keys=("container","lane","gpu","ros_domain_id","cpuset","control_root",
      "lane_deployment_root","other_lane_deployment_root","lane_profile_root",
      "other_lane_profile_root","ros_workspace","image_reference","image_id",
      "host_user_uid","host_user_gid","namespace","static_peer",
      "identity_prefix")
payload={"schema_version":1,"spec_version":1,**dict(zip(keys,sys.argv[2:]))}
for key in ("gpu","ros_domain_id","host_user_uid","host_user_gid"):
    payload[key]=int(payload[key])
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY

  create_lane_container() {
    docker create \
      --name "$container" --network host --ipc private \
      --cpuset-cpus "$cpuset" --gpus "device=$gpu" \
      --label internnav.t5.spec_version=1 \
      --label internnav.t5.role=isaac_ros_worker \
      --label "internnav.t5.lane=$lane" \
      --label "internnav.t5.gpu=$gpu" \
      --label "internnav.t5.ros_domain_id=$domain" \
      --label "internnav.t5.control_root=$root" \
      --label "internnav.t5.deployment_root=$lane_deployment_root" \
      --label "internnav.t5.worker_profile_root=$lane_root" \
      -e "ROS_DOMAIN_ID=$domain" -e ROS_LOCALHOST_ONLY=0 \
      -e "ROS_NAMESPACE=$namespace" \
      -e ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
      -e "ROS_STATIC_PEERS=$static_peer" \
      -e "INTERNNAV_T5_LANE=$lane" \
      -e "INTERNNAV_T5_ID_PREFIX=$identity_prefix" \
      -e CUDA_VISIBLE_DEVICES=0 -e "NVIDIA_VISIBLE_DEVICES=$gpu" \
      -e "XDG_CACHE_HOME=$lane_root/cache/xdg" \
      -e "XDG_CONFIG_HOME=$lane_root/config" \
      -e "XDG_DATA_HOME=$lane_root/data" \
      -e "OV_CACHE_ROOT=$lane_root/cache/ov" \
      -e "NVIDIA_SHADER_CACHE_PATH=$lane_root/cache/nvidia" \
      -e "TMPDIR=$lane_root/tmp" \
      -e USERNAME=admin -e "HOST_USER_UID=$(id -u)" -e "HOST_USER_GID=$(id -g)" \
      --mount "type=bind,src=$root,dst=$root,readonly" \
      --mount "type=bind,src=$lane_deployment_root,dst=$lane_deployment_root" \
      --mount "type=bind,src=$lane_root,dst=$lane_root" \
      --mount "type=bind,src=$ros_ws,dst=/workspaces/isaac,readonly" \
      --entrypoint /usr/local/bin/scripts/workspace-entrypoint.sh \
      "$image" sleep infinity >/dev/null
  }

  if docker container inspect "$container" >/dev/null 2>&1; then
    test "$(docker inspect -f '{{.State.Running}}' "$container")" != true
    docker inspect "$container" >"$inspected"
    if ! python3 "$validator" "$expected" "$inspected" "$audit"; then
      # Drift is repairable only while stopped and within both exclusive locks.
      docker rm "$container" >/dev/null
      create_lane_container
    fi
  else
    create_lane_container
  fi
  test -z "$ownership_dir" || : >"$ownership_dir/$container"
  docker start "$container" >/dev/null
  docker inspect "$container" >"$inspected"
  python3 "$validator" "$expected" "$inspected" "$audit"
  cp "$audit" "$lane_root/container_spec_audit.json"
  docker exec --user admin --workdir /workspaces/isaac \
    -e "ROS_DOMAIN_ID=$domain" "$container" bash -lc \
    'set -eo pipefail; source /opt/ros/jazzy/setup.bash; source install/setup.bash; ros2 pkg prefix internvla_t4_sensors >/dev/null'
}

if test "$prepare_scope" != b; then
  prepare_lane a 0 75 "$lane_a_cpuset" internnav_t5_isaac_a \
    "$lane_a_deployment_root" "$lane_b_deployment_root" b
fi
if test "$prepare_scope" = dual; then
  prepare_lane b 1 76 "$lane_b_cpuset" internnav_t5_isaac_b \
    "$lane_b_deployment_root" "$lane_a_deployment_root" a
elif test "$prepare_scope" = b; then
  prepare_lane b 1 76 "$lane_b_cpuset" internnav_t5_isaac_b \
    "$lane_b_deployment_root" "$lane_a_deployment_root" a
fi

if test "$prepare_scope" = dual; then
  python3 - "$result_json" "$image" "$image_id" "$image_repo_digests" \
    "$worker_root" "$lane_a_deployment_root" "$lane_b_deployment_root" <<'PY'
import json,subprocess,sys,time
from pathlib import Path

def inspect(name):
    return json.loads(subprocess.check_output(["docker","inspect",name],text=True))[0]

lanes={}
for lane,name,gpu,domain,cpuset in (
    ("a","internnav_t5_isaac_a",0,75,"0,2,4,6,8,10,12,14,16"),
    ("b","internnav_t5_isaac_b",1,76,"1,3,5,7,9,11,13,15,17"),
):
    value=inspect(name)
    host=value["HostConfig"]
    profile=Path(sys.argv[5])/lane
    spec_audit=json.loads((profile/'container_spec_audit.json').read_text())
    lanes[lane]={
        "container":name,"gpu":gpu,"ros_domain_id":domain,"cpuset":cpuset,
        "running":value["State"]["Running"],"pid_mode":host["PidMode"],
        "ipc_mode":host["IpcMode"],"network_mode":host["NetworkMode"],
        "container_id":value["Id"],"image_id":value["Image"],
        "deployment_root":sys.argv[6] if lane=='a' else sys.argv[7],
        "profile_root":str(profile),"spec_audit":spec_audit,
    }
checks={
    "both_running":all(v["running"] for v in lanes.values()),
    "private_pid":all(v["pid_mode"] != "host" for v in lanes.values()),
    "private_ipc":all(v["ipc_mode"] != "host" for v in lanes.values()),
    "different_cpuset":lanes["a"]["cpuset"] != lanes["b"]["cpuset"],
    "different_gpu":lanes["a"]["gpu"] != lanes["b"]["gpu"],
    "different_domain":lanes["a"]["ros_domain_id"] != lanes["b"]["ros_domain_id"],
    "exact_specs":all(v["spec_audit"]["status"] == "PASS" for v in lanes.values()),
}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
         "image":sys.argv[2],"image_id":sys.argv[3],
         "image_repo_digests":json.loads(sys.argv[4]),"worker_root":sys.argv[5],
         "lanes":lanes,"checks":checks,"shared_asset_mutation_serialized":True,
         "full_mp4_encoding_allowed_concurrently":False,"recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
raise SystemExit(0 if payload["status"]=="PASS" else 1)
PY
elif test "$prepare_scope" = a; then
  python3 - "$result_json" "$image" "$image_id" "$image_repo_digests" \
    "$worker_root" "$lane_a_deployment_root" <<'PY'
import json,subprocess,sys,time
from pathlib import Path

value=json.loads(subprocess.check_output(
    ["docker","inspect","internnav_t5_isaac_a"],text=True))[0]
host=value["HostConfig"]
profile=Path(sys.argv[5])/"a"
spec_audit=json.loads((profile/"container_spec_audit.json").read_text())
lane={
    "container":"internnav_t5_isaac_a","gpu":0,"ros_domain_id":75,
    "cpuset":"0,2,4,6,8,10,12,14,16","running":value["State"]["Running"],
    "pid_mode":host["PidMode"],"ipc_mode":host["IpcMode"],
    "network_mode":host["NetworkMode"],"container_id":value["Id"],
    "image_id":value["Image"],"deployment_root":sys.argv[6],
    "profile_root":str(profile),"spec_audit":spec_audit,
}
checks={
    "lane_a_running":lane["running"],
    "private_pid":lane["pid_mode"] != "host",
    "private_ipc":lane["ipc_mode"] != "host",
    "gpu0_exact":lane["gpu"] == 0,
    "even_cpuset_exact":lane["cpuset"] == "0,2,4,6,8,10,12,14,16",
    "domain_exact":lane["ros_domain_id"] == 75,
    "exact_spec":spec_audit["status"] == "PASS",
}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
         "image":sys.argv[2],"image_id":sys.argv[3],
         "image_repo_digests":json.loads(sys.argv[4]),"worker_root":sys.argv[5],
         "prepared_lanes":["a"],"lanes":{"a":lane},"checks":checks,
         "shared_asset_mutation_serialized":True,
         "full_mp4_encoding_allowed_concurrently":False,"recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
raise SystemExit(0 if payload["status"]=="PASS" else 1)
PY
else
  python3 - "$result_json" "$image" "$image_id" "$image_repo_digests" \
    "$worker_root" "$lane_b_deployment_root" <<'PY'
import json,subprocess,sys,time
from pathlib import Path

value=json.loads(subprocess.check_output(
    ["docker","inspect","internnav_t5_isaac_b"],text=True))[0]
host=value["HostConfig"]
profile=Path(sys.argv[5])/"b"
spec_audit=json.loads((profile/"container_spec_audit.json").read_text())
lane={
    "container":"internnav_t5_isaac_b","gpu":1,"ros_domain_id":76,
    "cpuset":"1,3,5,7,9,11,13,15,17","running":value["State"]["Running"],
    "pid_mode":host["PidMode"],"ipc_mode":host["IpcMode"],
    "network_mode":host["NetworkMode"],"container_id":value["Id"],
    "image_id":value["Image"],"deployment_root":sys.argv[6],
    "profile_root":str(profile),"spec_audit":spec_audit,
}
checks={
    "lane_b_running":lane["running"],
    "private_pid":lane["pid_mode"] != "host",
    "private_ipc":lane["ipc_mode"] != "host",
    "gpu1_exact":lane["gpu"] == 1,
    "odd_cpuset_exact":lane["cpuset"] == "1,3,5,7,9,11,13,15,17",
    "domain_exact":lane["ros_domain_id"] == 76,
    "exact_spec":spec_audit["status"] == "PASS",
}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
         "image":sys.argv[2],"image_id":sys.argv[3],
         "image_repo_digests":json.loads(sys.argv[4]),"worker_root":sys.argv[5],
         "prepare_scope":"lane_b","prepared_lanes":["b"],"lanes":{"b":lane},
         "checks":checks,"shared_asset_mutation_serialized":True,
         "full_mp4_encoding_allowed_concurrently":False,"recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
raise SystemExit(0 if payload["status"]=="PASS" else 1)
PY
fi
