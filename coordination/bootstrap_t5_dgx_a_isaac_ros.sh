#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s RUN_ID results/internnav_t5/strict-bootstrap-RUN_ID\n' "${0##*/}" >&2
  exit 64
}

[[ $# -eq 2 ]] || usage
run_id="$1"
result_relative="$2"
[[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]{7,63}$ ]] || usage
test "$result_relative" = "results/internnav_t5/strict-bootstrap-$run_id" || usage

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if [[ -z "${DGX_A_SUDO_PASSWORD:-}" ]]; then
  test -f "$root/.env.local" || {
    echo "untracked .env.local is required for DGX_A sudo" >&2
    exit 78
  }
  DGX_A_SUDO_PASSWORD="$(python3 - "$root/.env.local" <<'PY'
import sys
from pathlib import Path
values={}
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line=raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
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
result_dir="$root/$result_relative"
target=railgun@10.100.100.128
remote_result="/home/railgun/internnav-t1-t2/results/t5_strict_bootstrap/$run_id"

if [[ "${INTERNNAV_T5_INSIDE_DGX_A_BOOTSTRAP:-0}" != 1 ]]; then
  test ! -e "$result_dir"
  mkdir -p "$result_dir"
  env INTERNNAV_T5_INSIDE_DGX_A_BOOTSTRAP=1 \
    bash "$root/scripts/with_resource_lease.sh" dgx-a \
      --owner codex-00 --task "t5-dgx-a-isaac-ros-4.5-bootstrap:$run_id" \
      --log-dir "$result_dir/lease" --acquire-timeout 30 \
      --cleanup-timeout 180 --kill-wait-timeout 30 -- \
    bash "$root/coordination/bootstrap_t5_dgx_a_isaac_ros.sh" \
      "$run_id" "$result_relative"
  exit $?
fi

test -d "$result_dir"
remote_program='set -euo pipefail
run_id="$1"; result="$2"
docker_proxy="http://10.100.100.103:10811"
proxy_dropin="/etc/systemd/system/docker.service.d/98-internnav-bootstrap-proxy.conf"
proxy_active=0
key_tmp=""
cleanup() {
  rc=$?
  trap - EXIT
  set +e
  if test "$proxy_active" = 1; then
    rm -f -- "$proxy_dropin"
    systemctl daemon-reload
    systemctl restart docker
  fi
  test -z "$key_tmp" || rm -f -- "$key_tmp"
  exit "$rc"
}
trap cleanup EXIT
test "$(id -u)" = 0
test "$(uname -m)" = aarch64
. /etc/os-release
test "$VERSION_CODENAME" = noble
if docker ps -q | grep -q .; then
  echo "refusing to restart Docker while a container is running" >&2
  exit 75
fi
install -d -m 0755 "$(dirname "$result")"
test ! -e "$result"
install -d -m 0755 "$result"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg software-properties-common
key_tmp="$(mktemp)"
curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key | gpg --dearmor >"$key_tmp"
test -s "$key_tmp"
install -m 0644 "$key_tmp" /usr/share/keyrings/nvidia-isaac-ros.gpg
printf "%s\n" "deb [signed-by=/usr/share/keyrings/nvidia-isaac-ros.gpg] https://isaac.download.nvidia.com/isaac-ros/release-4.5 noble-fastos main" \
  >/etc/apt/sources.list.d/nvidia-isaac-ros.list
apt-get update
apt-get install -y --no-install-recommends isaac-ros-cli
getent group docker >/dev/null
usermod -aG docker railgun
systemctl daemon-reload
systemctl restart docker
timeout 300 isaac-ros init docker
docker info >"$result/docker_info.txt"
test ! -e "$proxy_dropin"
install -d -m 0755 "$(dirname "$proxy_dropin")"
printf "%s\n" "[Service]" \
  "Environment=HTTP_PROXY=$docker_proxy" \
  "Environment=HTTPS_PROXY=$docker_proxy" \
  "Environment=NO_PROXY=localhost,127.0.0.1,10.0.0.0/8" \
  >"$proxy_dropin"
systemctl daemon-reload
systemctl restart docker
proxy_active=1
docker run --rm hello-world >"$result/docker_hello.txt"
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L >"$result/docker_gpu.txt"
rm -f -- "$proxy_dropin"
systemctl daemon-reload
systemctl restart docker
proxy_active=0
test ! -e "$proxy_dropin"
python3 - "$result/bootstrap.json" "$run_id" <<"PY"
import json,os,platform,subprocess,sys,time
from pathlib import Path

def output(argv):
    return subprocess.check_output(argv,text=True).strip()

def candidate(name):
    value=output(["apt-cache","policy",name])
    for line in value.splitlines():
        if line.strip().startswith("Candidate:"):
            item=line.split(":",1)[1].strip()
            return None if item=="(none)" else item
    return None

payload={
 "schema_version":1,"status":"RUNTIME_BOOTSTRAP_PASS","run_id":sys.argv[2],
 "host":platform.node(),"architecture":platform.machine(),"release":"4.5",
 "apt_source":"noble-fastos","isaac_ros_cli":output(["which","isaac-ros"]),
 "docker_server_version":output(["docker","version","--format","{{.Server.Version}}"]),
 "docker_runtimes":output(["docker","info","--format","{{json .Runtimes}}"]),
 "gpu_probe":Path(sys.argv[1]).with_name("docker_gpu.txt").read_text(encoding="utf-8").strip(),
 "docker_pull_proxy":"temporary http://10.100.100.103:10811; removed before receipt",
 "docker_proxy_dropin_removed":True,
 "apt_candidates":{
  "isaac-ros-cli":candidate("isaac-ros-cli"),
  "ros-jazzy-isaac-ros-visual-slam":candidate("ros-jazzy-isaac-ros-visual-slam"),
  "ros-jazzy-isaac-ros-nvblox":candidate("ros-jazzy-isaac-ros-nvblox"),
 },
 "railgun_groups_after_new_login_required":output(["getent","group","docker"]),
 "recorded_unix":time.time(),
}
if not all(payload["apt_candidates"].values()) or "NVIDIA GB10" not in payload["gpu_probe"]:
    payload["status"]="RUNTIME_BOOTSTRAP_FAIL"
path=Path(sys.argv[1]); path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
raise SystemExit(0 if payload["status"]=="RUNTIME_BOOTSTRAP_PASS" else 75)
PY
chown -R railgun:railgun "$result"'

{
  printf '%s\n' "$DGX_A_SUDO_PASSWORD"
  printf '%s\n' "$remote_program"
} | ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes "$target" \
  "sudo -S -p '' bash -s -- '$run_id' '$remote_result'"

mkdir -p "$result_dir/remote"
scp -q -r -o BatchMode=yes -o StrictHostKeyChecking=yes \
  "$target:$remote_result/bootstrap.json" "$target:$remote_result/docker_info.txt" \
  "$target:$remote_result/docker_hello.txt" "$target:$remote_result/docker_gpu.txt" \
  "$result_dir/remote/"
ssh -T -o BatchMode=yes -o StrictHostKeyChecking=yes "$target" \
  "id -nG | tr ' ' '\n' | grep -Fx docker >/dev/null; docker info >/dev/null"
python3 - "$result_dir" <<'PY'
import hashlib,json,sys,time
from pathlib import Path
root=Path(sys.argv[1]); receipt=json.loads((root/"remote/bootstrap.json").read_text(encoding="utf-8"))
checks={
 "remote_bootstrap":receipt.get("status")=="RUNTIME_BOOTSTRAP_PASS",
 "release":receipt.get("release")=="4.5" and receipt.get("apt_source")=="noble-fastos",
 "packages":all((receipt.get("apt_candidates") or {}).values()),
 "gb10_container": "NVIDIA GB10" in str(receipt.get("gpu_probe") or ""),
 "temporary_proxy_removed":receipt.get("docker_proxy_dropin_removed") is True,
}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
 "checks":checks,"remote_receipt_sha256":hashlib.sha256((root/"remote/bootstrap.json").read_bytes()).hexdigest(),
 "recorded_unix":time.time()}
(root/"strict_bootstrap_summary.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
print(json.dumps(payload,sort_keys=True))
raise SystemExit(0 if payload["status"]=="PASS" else 75)
PY
