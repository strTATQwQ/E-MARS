#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: recover_t5_resource_quarantine.sh <dgx-a|dgx-b|isaac-gpu0|isaac-gpu1|isaac|all> RESULT_DIR

Recovery-only entry for a persistent T5 host quarantine.  It acquires the
physical locks in the standard DGX-before-Isaac order and cannot launch an
arbitrary workload.  Only marker-scoped PGIDs/owned containers plus fixed T5
sockets, ports, and runtime locks are cleaned/audited before marker removal.
EOF
  exit 64
}

if [[ "${1:-}" == --inside ]]; then
  [[ $# -eq 3 ]] || usage
  inside=1
  scope="$2"
  result_dir="$3"
else
  [[ $# -eq 2 ]] || usage
  inside=0
  scope="$1"
  result_dir="$2"
fi

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
t5_isaac_ip="${INTERNVLA_T5_ISAAC_IP:-10.100.120.123}"
test "$t5_isaac_ip" = 10.100.120.123
case "$scope" in
  dgx-a) lease_mode=recover-dgx-a ;;
  dgx-b) lease_mode=recover-dgx-b ;;
  isaac-gpu0) lease_mode=recover-isaac-gpu0 ;;
  isaac-gpu1) lease_mode=recover-isaac-gpu1 ;;
  isaac) lease_mode=recover-isaac ;;
  all) lease_mode=recover-all ;;
  *) usage ;;
esac

case "$result_dir" in
  "$root"/results/internnav_t5/quarantine-recovery-*) ;;
  *) echo "RESULT_DIR must be a new absolute quarantine-recovery path under results/internnav_t5" >&2; exit 64 ;;
esac

if (( inside == 0 )); then
  test ! -e "$result_dir"
  umask 077
  mkdir -p "$result_dir/lease"
  exec bash "$root/scripts/with_resource_lease.sh" "$lease_mode" \
    --owner codex-00 --task "t5-quarantine-recovery:$scope" \
    --log-dir "$result_dir/lease" --acquire-timeout 30 \
    --cleanup-timeout 120 --kill-wait-timeout 30 -- \
    bash "$root/scripts/recover_t5_resource_quarantine.sh" \
      --inside "$scope" "$result_dir"
fi

test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = "$lease_mode"
test -d "$result_dir/lease"
mkdir -p "$result_dir/audits"
source "$root/scripts/t5_quarantine_common.sh"

ssh_options=(-T -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2)
remote() { local target="$1"; shift; ssh "${ssh_options[@]}" "$target" "$@"; }

declare -a specifications=()
case "$scope" in
  dgx-a)
    specifications+=("railgun@10.100.100.128|/tmp/internnav_dgx.quarantine|dgx_a")
    ;;
  dgx-b)
    specifications+=("rail@10.100.120.122|/tmp/internnav_dgx.quarantine|dgx_b")
    ;;
  isaac-gpu0)
    specifications+=("song@$t5_isaac_ip|/tmp/internnav_isaac_gpu0.quarantine|x86_gpu0")
    ;;
  isaac-gpu1)
    specifications+=("song@$t5_isaac_ip|/tmp/internnav_isaac_gpu1.quarantine|x86_gpu1")
    ;;
  isaac)
    specifications+=("song@$t5_isaac_ip|/tmp/internnav_isaac.quarantine|x86")
    specifications+=("song@$t5_isaac_ip|/tmp/internnav_isaac_gpu0.quarantine|x86_gpu0")
    specifications+=("song@$t5_isaac_ip|/tmp/internnav_isaac_gpu1.quarantine|x86_gpu1")
    ;;
  all)
    specifications+=("railgun@10.100.100.128|/tmp/internnav_dgx.quarantine|dgx_a")
    specifications+=("rail@10.100.120.122|/tmp/internnav_dgx.quarantine|dgx_b")
    specifications+=("song@$t5_isaac_ip|/tmp/internnav_isaac.quarantine|x86")
    specifications+=("song@$t5_isaac_ip|/tmp/internnav_isaac_gpu0.quarantine|x86_gpu0")
    specifications+=("song@$t5_isaac_ip|/tmp/internnav_isaac_gpu1.quarantine|x86_gpu1")
    ;;
esac

declare -a receipts=()
recovery_rc=0
for specification in "${specifications[@]}"; do
  IFS='|' read -r target marker role <<<"$specification"
  receipt="$result_dir/audits/${role}_quarantine_recovery.json"
  receipts+=("$receipt")
  if ! t5_quarantine_recover "$target" "$marker" "$role" "$receipt"; then
    recovery_rc=75
  fi
done

python3 - "$result_dir/recovery_summary.json" "$scope" "$recovery_rc" "${receipts[@]}" <<'PY'
import json, sys, time
from pathlib import Path
output, scope, observed = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
receipts=[]
checks={}
for name in sys.argv[4:]:
    path=Path(name)
    try: value=json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError,json.JSONDecodeError): value=None
    receipts.append({'path':str(path),'value':value})
    checks[path.stem]=isinstance(value,dict) and value.get('status')=='PASS' \
        and value.get('workload_started') is False
payload={'schema_version':1,'status':'PASS' if observed==0 and checks and all(checks.values()) else 'FAIL',
 'scope':scope,'recovery_only':True,'arbitrary_workload_allowed':False,
 'checks':checks,'receipts':receipts,'recorded_unix':time.time()}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
raise SystemExit(0 if payload['status']=='PASS' else 75)
PY
