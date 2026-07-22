#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 8 ]] || {
  echo "usage: prepare_t4_parallel_lanes.sh ROOT GRANT DEPLOYMENT_REF DGX_A_ROOT ISAAC_A_ROOT DGX_B_ROOT ISAAC_B_ROOT OUTPUT" >&2
  exit 64
}
root="$(readlink -f "$1")"
grant_id="$2"
deployment_ref="$3"
dgx_a_root="$4"
isaac_a_root="$5"
dgx_b_root="$6"
isaac_b_root="$7"
output="$8"

[[ "$grant_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || exit 64
[[ "$deployment_ref" =~ ^[0-9a-f]{40}$ ]] || exit 64
case "$dgx_a_root" in /home/railgun/internnav-t1-t2/.t4-deployments/*) ;; *) exit 64 ;; esac
case "$dgx_b_root" in /home/railgun/internnav-t1-t2/.t4-deployments/*-dgx-b) ;; *) exit 64 ;; esac
case "$isaac_a_root" in /home/song/internnav-t1-t2/.t4-deployments/*) ;; *) exit 64 ;; esac
case "$isaac_b_root" in /home/song/internnav-t1-t2/.t4-deployments/*-isaac-b) ;; *) exit 64 ;; esac
case "$output" in "$root"/results/parallel/t4_functional/*) ;; *) exit 64 ;; esac
test ! -e "$output"
mkdir -p "$(dirname "$output")"

ssh_opts=(-o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2)
dgx_a=railgun@10.100.100.128
dgx_b=rail@10.100.120.116
isaac=song@10.100.120.111
ssh "${ssh_opts[@]}" "$dgx_a" \
  "test -d /home/railgun/internnav-t0 && test ! -L /home/railgun/internnav-t0; test -d '$dgx_a_root'; test ! -L '$dgx_a_root'"
ssh "${ssh_opts[@]}" "$isaac" \
  "test -d '$isaac_a_root' && test ! -L '$isaac_a_root'; test ! -e '$isaac_b_root'"
ssh "${ssh_opts[@]}" "$dgx_b" \
  "sudo -n install -d -m 755 -o rail -g rail /home/railgun; install -d -m 700 /home/railgun/.codex-internnav-stage; test ! -e '$dgx_b_root'; test -d /home/rail/internnav-t0; test ! -L /home/rail/internnav-t0; test -x /home/rail/internnav-t0/venv-model/bin/python"

# DGX-B already owns an architecture-native model environment.  Match the
# frozen source revision and checkpoint inventory instead of copying DGX-A's
# host-specific venv paths.
model_inventory_command='git -C "$HOME/internnav-t0/InternNav" rev-parse HEAD; find "$HOME/internnav-t0/InternNav/checkpoints/InternVLA-N1-DualVLN" -maxdepth 1 -type f -printf "%s %f\n" | sort | sha256sum | cut -d" " -f1'
dgx_a_model_inventory="$(ssh "${ssh_opts[@]}" "$dgx_a" "$model_inventory_command")"
dgx_b_model_inventory="$(ssh "${ssh_opts[@]}" "$dgx_b" "$model_inventory_command")"
test "$dgx_a_model_inventory" = "$dgx_b_model_inventory"

# Clone only the immutable deployment payload and ROS workspace. Historical
# results/runtime state is deliberately excluded so cleanup remains lane-local.
ssh "${ssh_opts[@]}" "$dgx_a" \
  "tar -C '$dgx_a_root' --exclude=./results --exclude=./runtime --exclude=./ros_ws -cf - ." |
  ssh "${ssh_opts[@]}" "$dgx_b" \
    "set -euo pipefail; install -d -m 700 '$dgx_b_root'; tar -C '$dgx_b_root' -xf -"
ssh "${ssh_opts[@]}" "$dgx_b" \
  "set -euo pipefail; install -d -m 700 '$dgx_b_root/results' '$dgx_b_root/runtime' '$dgx_b_root/ros_ws' '$dgx_b_root/ros_ws/src'"
ssh "${ssh_opts[@]}" "$dgx_b" \
  "set -eo pipefail; export PYTHONDONTWRITEBYTECODE=1; set +u; source /opt/ros/jazzy/setup.bash; set -u; INTERNNAV_T1_CONTROL_ROOT='$dgx_b_root' INTERNVLA_ROS_WS='$dgx_b_root/ros_ws' INTERNNAV_RUNTIME_POLICY=completion_sim INTERNNAV_SIMULATION_TARGET=isaac INTERNNAV_T4_RESOURCE_LEASE_ACK=dgx+isaac bash '$dgx_b_root/scripts/build_t4_host_ros.sh' >'$dgx_b_root/results/dgx_ros_build.log' 2>&1"
ssh "${ssh_opts[@]}" "$isaac" \
  "set -euo pipefail; install -d -m 700 '$isaac_b_root'; rsync -a --exclude=/results --exclude=/runtime '$isaac_a_root/' '$isaac_b_root/'; install -d -m 700 '$isaac_b_root/results' '$isaac_b_root/runtime'"

update_receipt='import json,sys; from pathlib import Path; p=Path(sys.argv[1]); v=json.loads(p.read_text()); v["deployment_root"]=sys.argv[2]; v["grant_id"]=sys.argv[3]; v["ros_workspace"]=sys.argv[2]+"/ros_ws" if sys.argv[4]=="dgx" else "/home/song/internnav-t4/isaac_ros_ws_45"; v["parallel_lane_clone"]="b"; p.write_text(json.dumps(v,indent=2,sort_keys=True)+"\n")'
ssh "${ssh_opts[@]}" "$dgx_b" \
  "python3 -c '$update_receipt' '$dgx_b_root/deployment_receipt.json' '$dgx_b_root' '$grant_id' dgx"
ssh "${ssh_opts[@]}" "$isaac" \
  "python3 -c '$update_receipt' '$isaac_b_root/deployment_receipt.json' '$isaac_b_root' '$grant_id' isaac"

ssh "${ssh_opts[@]}" "$dgx_b" \
  "set -euo pipefail; export PYTHONDONTWRITEBYTECODE=1; python3 '$dgx_b_root/coordination/t4_functional_payload.py' verify-tree --root '$dgx_b_root' --manifest '$dgx_b_root/payload_manifest.json' --expected-ref '$deployment_ref' >/dev/null; python3 '$dgx_b_root/coordination/t4_functional_run_contract.py' validate-deployment --role dgx --deployment-root '$dgx_b_root' --deployment-ref '$deployment_ref' >/dev/null; python3 '$dgx_b_root/coordination/t4_functional_run_contract.py' validate-dgx-workspace --deployment-root '$dgx_b_root' --workspace '$dgx_b_root/ros_ws' >/dev/null; /home/rail/internnav-t0/venv-model/bin/python -c 'import torch,internnav'"
ssh "${ssh_opts[@]}" "$isaac" \
  "set -euo pipefail; export PYTHONDONTWRITEBYTECODE=1; python3 '$isaac_b_root/coordination/t4_functional_payload.py' verify-tree --root '$isaac_b_root' --manifest '$isaac_b_root/payload_manifest.json' --expected-ref '$deployment_ref' >/dev/null; python3 '$isaac_b_root/coordination/t4_functional_run_contract.py' validate-deployment --role isaac --deployment-root '$isaac_b_root' --deployment-ref '$deployment_ref' >/dev/null"

python3 - "$output" "$grant_id" "$deployment_ref" "$dgx_a_root" "$isaac_a_root" \
  "$dgx_b_root" "$isaac_b_root" "$dgx_a_model_inventory" <<'PY'
import json,sys,time
from pathlib import Path
revision,inventory=sys.argv[8].splitlines()
payload={"schema_version":1,"status":"PASS","grant_id":sys.argv[2],
"deployment_ref_sha":sys.argv[3],"lane_a":{"dgx_root":sys.argv[4],"isaac_root":sys.argv[5],
"dgx_host":"10.100.100.128","isaac_gpu_index":0,"ros_domain_id":71,"tcp_port":24137},
"lane_b":{"dgx_root":sys.argv[6],"isaac_root":sys.argv[7],
"dgx_host":"10.100.120.116","isaac_gpu_index":1,"ros_domain_id":72,"tcp_port":24138},
"model_tree":{"model_revision":revision,"checkpoint_inventory_sha256":inventory,
"source_target_match":True,"host_native_environments":True},
"strict_evidence_modified":False,"real_go2_targeted":False,"recorded_unix":time.time()}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
PY
