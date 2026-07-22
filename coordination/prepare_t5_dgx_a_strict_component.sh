#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s visual-slam|nvblox RUN_ID results/internnav_t5/strict-component-RUN_ID\n' "${0##*/}" >&2
  exit 64
}

[[ $# -eq 3 ]] || usage
component="$1"
run_id="$2"
result_relative="$3"
case "$component" in
  visual-slam) package=ros-jazzy-isaac-ros-visual-slam ;;
  # Install the core runtime and Nav2 layer directly.  The convenience
  # metapackage also pulls semantic examples pinned to CUDA 13.0 and would
  # downgrade the DGX Spark host's CUDA 13.2 configuration.
  nvblox) package=ros-jazzy-nvblox-ros ;;
  *) usage ;;
esac
[[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]{7,63}$ ]] || usage
test "$result_relative" = "results/internnav_t5/strict-component-$run_id" || usage

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
result_dir="$root/$result_relative"
target=railgun@10.100.100.128
remote_result="/home/railgun/internnav-t1-t2/results/t5_strict_component/$run_id"

if [[ -z "${DGX_A_SUDO_PASSWORD:-}" ]]; then
  test -f "$root/.env.local" || { echo 'untracked .env.local is required' >&2; exit 78; }
  DGX_A_SUDO_PASSWORD="$(python3 - "$root/.env.local" <<'PY'
import sys
from pathlib import Path
values={}
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line=raw.strip()
    if line and not line.startswith("#") and "=" in line:
        key,value=line.split("=",1)
        values[key.strip()]=value.strip().strip('"').strip("'")
password=values.get("DGX_A_PASSWORD","")
if not password:
    raise SystemExit("DGX_A_PASSWORD is missing from .env.local")
print(password,end="")
PY
)"
fi
: "${DGX_A_SUDO_PASSWORD:?DGX_A sudo password is empty}"

if [[ "${INTERNNAV_T5_INSIDE_STRICT_COMPONENT:-0}" != 1 ]]; then
  test ! -e "$result_dir"
  mkdir -p "$result_dir"
  env INTERNNAV_T5_INSIDE_STRICT_COMPONENT=1 \
    bash "$root/scripts/with_resource_lease.sh" dgx-a \
      --owner codex-00 --task "t5-dgx-a-$component-prepare:$run_id" \
      --log-dir "$result_dir/lease" --acquire-timeout 30 \
      --cleanup-timeout 180 --kill-wait-timeout 30 -- \
    bash "$root/coordination/prepare_t5_dgx_a_strict_component.sh" \
      "$component" "$run_id" "$result_relative"
  exit $?
fi

test -d "$result_dir"
remote_program='set -euo pipefail
component="$1"; package="$2"; run_id="$3"; result="$4"
test "$(id -u)" = 0
test "$(uname -m)" = aarch64
test -f /etc/apt/sources.list.d/nvidia-isaac-ros.list
test ! -e "$result"
install -d -m 0755 "$result"
export DEBIAN_FRONTEND=noninteractive
apt-get update
packages=("$package")
if test "$component" = nvblox; then
  packages+=(ros-jazzy-nvblox-nav2)
fi
apt-get install -y --no-install-recommends "${packages[@]}"
dpkg-query -W -f="\${Package} \${Version}\\n" "${packages[@]}" >"$result/package.txt"
set +u
source /opt/ros/jazzy/setup.bash
set -u
ros2 pkg list >"$result/ros_packages.txt"
case "$component" in
 visual-slam)
   ros2 pkg prefix isaac_ros_visual_slam >"$result/prefix.txt"
   grep -Fx isaac_ros_visual_slam "$result/ros_packages.txt"
   smoke_package=isaac_ros_visual_slam
   smoke_executable=isaac_ros_visual_slam
   ;;
 nvblox)
   ros2 pkg prefix nvblox_ros >"$result/prefix.txt"
   grep -Fx nvblox_ros "$result/ros_packages.txt"
   smoke_package=nvblox_ros
   smoke_executable=nvblox_node
   ;;
 *) exit 64 ;;
esac
set +e
runuser -u railgun -- env ROS_DOMAIN_ID=91 ROS_LOCALHOST_ONLY=1 \
  timeout --signal=TERM --kill-after=5s 20s \
  bash -c "set +u; source /opt/ros/jazzy/setup.bash; exec ros2 run $smoke_package $smoke_executable" \
  >"$result/node_smoke.log" 2>&1
smoke_rc=$?
set -e
printf "%s\n" "$smoke_rc" >"$result/node_smoke_exit.txt"
# GNU timeout returns 124 only when the process stayed alive for the full
# bounded smoke.  An early clean exit is still a runtime failure here.
test "$smoke_rc" = 124
sleep 2
test -z "$(pgrep -af "/opt/ros/jazzy/lib/$smoke_package/$smoke_executable" || true)"
python3 - "$result/prepare.json" "$component" "$package" "$run_id" <<"PY"
import json,platform,subprocess,sys,time
from pathlib import Path
out=Path(sys.argv[1])
payload={
 "schema_version":1,"status":"COMPONENT_RUNTIME_READY",
 "component":sys.argv[2],"package":sys.argv[3],"run_id":sys.argv[4],
 "host":platform.node(),"architecture":platform.machine(),
 "package_version":out.with_name("package.txt").read_text(encoding="utf-8").strip(),
 "ros_prefix":out.with_name("prefix.txt").read_text(encoding="utf-8").strip(),
 "node_smoke_exit":int(out.with_name("node_smoke_exit.txt").read_text(encoding="utf-8").strip()),
 "node_smoke_full_duration":True,
 "recorded_unix":time.time(),
}
out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
chown -R railgun:railgun "$result"'

{
  printf '%s\n' "$DGX_A_SUDO_PASSWORD"
  printf '%s\n' "$remote_program"
} | ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes "$target" \
  "sudo -S -p '' bash -s -- '$component' '$package' '$run_id' '$remote_result'"

mkdir -p "$result_dir/remote"
scp -q -r -o BatchMode=yes -o StrictHostKeyChecking=yes \
  "$target:$remote_result/prepare.json" "$target:$remote_result/package.txt" \
  "$target:$remote_result/prefix.txt" "$target:$remote_result/node_smoke.log" \
  "$target:$remote_result/node_smoke_exit.txt" "$result_dir/remote/"
python3 - "$result_dir" "$component" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
root=Path(sys.argv[1]); component=sys.argv[2]
receipt=json.loads((root/"remote/prepare.json").read_text(encoding="utf-8"))
checks={
 "component":receipt.get("component")==component,
 "runtime_ready":receipt.get("status")=="COMPONENT_RUNTIME_READY",
 "package_installed":bool(receipt.get("package_version")),
 "ros_prefix":str(receipt.get("ros_prefix","")).startswith("/opt/ros/jazzy"),
 "node_smoke":receipt.get("node_smoke_exit")==124 and receipt.get("node_smoke_full_duration") is True,
}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
 "checks":checks,"component":component,
 "remote_receipt_sha256":hashlib.sha256((root/"remote/prepare.json").read_bytes()).hexdigest(),
 "recorded_unix":time.time()}
(root/"strict_component_prepare_summary.json").write_text(
 json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(payload,sort_keys=True))
raise SystemExit(0 if payload["status"]=="PASS" else 75)
PY
