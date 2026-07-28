#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

[[ $# -eq 5 ]] || { echo "remote T4 functional stage: invalid arguments" >&2; exit 64; }
role="$1"
stage="$2"
stable_root="$3"
grant_id="$4"
expected_sha="$5"

[[ "$grant_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || {
  echo "unsafe grant-id" >&2; exit 64;
}
[[ "$expected_sha" =~ ^[0-9a-f]{40}$ ]] || { echo "unsafe SHA" >&2; exit 64; }

case "$role" in
  dgx)
    expected_user=railgun
    expected_ip=" 10.100.100.128/"
    expected_root=/home/railgun/internnav-t1-t2
    expected_stage="/home/railgun/.codex-internnav-stage/t4-functional-${grant_id}"
    ;;
  isaac)
    expected_user=song
    expected_ip=" 10.100.120.111/"
    expected_root=/home/song/internnav-t1-t2
    expected_stage="/home/song/.codex-internnav-stage/t4-functional-${grant_id}"
    ;;
  *) echo "unsafe host role" >&2; exit 64 ;;
esac

test "$(id -un)" = "$expected_user"
ip -4 -o addr show | grep -Fq "$expected_ip"
test "$stable_root" = "$expected_root"
test "$stage" = "$expected_stage"
test -d "$stable_root" && test ! -L "$stable_root"
test "$(realpath "$stable_root")" = "$expected_root"
test -d "$stage" && test ! -L "$stage"
test "$(realpath "$stage")" = "$expected_stage"

receipt="$stage/${role}_deployment_receipt.json"
payload_tool="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/t4_functional_payload.py"
test -f "$payload_tool"
phase=preflight
deployment_root=""
ros_workspace=""
partial=""

write_receipt() {
  local rc="$1"
  python3 - "$receipt" "$role" "$phase" "$rc" "$grant_id" \
    "$expected_sha" "$deployment_root" "$ros_workspace" <<'PY'
import json,sys,time
from pathlib import Path
output=Path(sys.argv[1])
payload={
    "schema_version":1,
    "role":sys.argv[2],
    "phase":sys.argv[3],
    "exit_code":int(sys.argv[4]),
    "status":"PASS" if int(sys.argv[4]) == 0 and sys.argv[3] == "ready" else "FAIL",
    "grant_id":sys.argv[5],
    "ref_sha":sys.argv[6],
    "deployment_root":sys.argv[7] or None,
    "ros_workspace":sys.argv[8] or None,
    "runtime_policy":"completion_sim",
    "runtime_target":"isaac_simulation",
    "resource_lease_ack":"dgx+isaac",
    "strict_evidence_modified":False,
    "real_go2_targeted":False,
    "recorded_unix":time.time(),
}
temporary=output.with_suffix(".tmp")
temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
temporary.replace(output)
PY
}

on_exit() {
  local rc=$?
  trap - EXIT INT TERM HUP
  set +e
  if test -n "$partial" && test -d "$partial"; then
    case "$partial" in
      "$deployment_parent/"*.partial.*) rm -rf -- "$partial" ;;
      *) echo "refusing unsafe partial cleanup: $partial" >&2 ;;
    esac
  fi
  write_receipt "$rc"
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT TERM HUP

cd "$stage"
sha256sum -c SHA256SUMS
python3 - "$stage/payload_manifest.json" "$expected_sha" <<'PY'
import json,sys
value=json.load(open(sys.argv[1],encoding="utf-8"))
if value.get("ref_sha") != sys.argv[2]:
    raise SystemExit("payload manifest ref mismatch")
if value.get("status") != "FUNCTIONAL_PAYLOAD_READY":
    raise SystemExit("payload manifest is not ready")
PY

deployment_parent="$stable_root/.t4-deployments"
test ! -L "$deployment_parent"
install -d -m 700 "$deployment_parent"
test "$(realpath "$deployment_parent")" = "$stable_root/.t4-deployments"
test "$(stat -c %U "$deployment_parent")" = "$expected_user"
deployment_root="$deployment_parent/${grant_id}-${expected_sha:0:12}-${role}"
test ! -e "$deployment_root"
partial="$deployment_root.partial.$$"
mkdir -m 700 "$partial"
phase=extracting
tar -C "$partial" --no-same-owner -xf "$stage/payload.tar"
python3 "$payload_tool" verify-tree \
  --root "$partial" \
  --manifest "$stage/payload_manifest.json" \
  --expected-ref "$expected_sha"
install -m 600 "$stage/payload_manifest.json" "$partial/payload_manifest.json"
mkdir -m 700 "$partial/results" "$partial/runtime"
if test "$role" = isaac; then
  test -d "$stable_root/episodes"
  ln -s "$stable_root/episodes" "$partial/episodes"
fi
mv -- "$partial" "$deployment_root"
partial=""
phase=building

if test "$role" = dgx; then
  ros_workspace="$deployment_root/ros_ws"
  mkdir -p "$ros_workspace/src"
  INTERNNAV_T1_CONTROL_ROOT="$deployment_root" \
  INTERNVLA_ROS_WS="$ros_workspace" \
  INTERNNAV_RUNTIME_POLICY=completion_sim \
  INTERNNAV_SIMULATION_TARGET=isaac \
  INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx+isaac \
    bash "$deployment_root/scripts/build_t4_host_ros.sh" \
      >"$deployment_root/results/dgx_ros_build.log" 2>&1
else
  ros_workspace="$HOME/internnav-t4/isaac_ros_ws_45"
  INTERNNAV_T1_CONTROL_ROOT="$deployment_root" \
  INTERNVLA_T4_ISAAC_ROS_WS="$ros_workspace" \
  INTERNVLA_T4_OVERLAY_UPDATE_RESULT_DIR="$deployment_root/results/isaac_overlay_update" \
  INTERNNAV_RUNTIME_POLICY=completion_sim \
  INTERNNAV_SIMULATION_TARGET=isaac \
  INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx+isaac \
    bash "$deployment_root/scripts/update_t4_isaac_ros_overlay.sh"
fi

phase=ready
write_receipt 0
cp "$receipt" "$deployment_root/deployment_receipt.json"
trap - EXIT INT TERM HUP
exit 0
