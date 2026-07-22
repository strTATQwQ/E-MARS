#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_t5_d0_dual_online.sh GRANT_ID AUTHORIZATION_REF PREP_ROOT D01_ROOT D02_ROOT RESULT_ROOT

Runs D0.3: both complete symmetric DGX/Isaac Lanes concurrently on the frozen
five episodes.  The same five episode keys remain five statistical samples;
the ten Lane observations are hardware-reproduction evidence only.
EOF
  exit 64
}

[[ $# -eq 6 ]] || usage
grant_id="$1"
authorization_ref="$2"
prep_relative="$3"
d01_relative="$4"
d02_relative="$5"
result_relative="$6"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$root/scripts/t5_quarantine_common.sh"
source "$root/scripts/t5_remote_compute_audit_common.sh"
cd "$root"
board_relative=coordination/T5_DUAL_LANE_BOARD.md
golden_relative=configs/internnav_t5/golden_bundle_manifest.json
d0_manifest_relative=configs/internnav_t5/d0_run_manifest.json
credentials="$root/.env.local"
stage=d0_3_dual_lane_same_fixed_5

[[ "$grant_id" =~ ^t5d03[0-9]{8}t[0-9]{6}$ ]] || usage
[[ "$authorization_ref" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$prep_relative" =~ ^results/internnav_t5/d0-0-prepare-t5d00[0-9]{8}t[0-9]{6}$ ]] || usage
[[ "$d01_relative" =~ ^results/internnav_t5/d0-1-lane-a-t5d01[0-9]{8}t[0-9]{6}$ ]] || usage
[[ "$d02_relative" =~ ^results/internnav_t5/d0-2-lane-b-t5d02[0-9]{8}t[0-9]{6}$ ]] || usage
[[ "$result_relative" = "results/internnav_t5/d0-3-dual-lane-$grant_id" ]] || usage

# Key-only remote access and a private model-only token descriptor are hard
# requirements.  No password or token becomes an exported coordinator value.
test -f "$credentials"
set +a
source "$credentials"
: "${HF_TOKEN:?HF_TOKEN is required in .env.local}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
[[ "$HF_TOKEN" =~ ^hf_[A-Za-z0-9]{20,}$ ]]
[[ "$HF_ENDPOINT" =~ ^https://hf-mirror\.com/?$ ]]
export -n HF_TOKEN HUGGING_FACE_HUB_TOKEN DGX_A_PASSWORD DGX_B_PASSWORD \
  ISAAC_X86_PASSWORD 2>/dev/null || true

git_command=(git -C "$root")
if [[ -f "$root/.git" ]] && grep -Eq '^gitdir: [A-Za-z]:/' "$root/.git"; then
  command -v git.exe >/dev/null
  command -v wslpath >/dev/null
  git_command=(git.exe -C "$(wslpath -w "$root")")
fi

# A D0.3 authorization is exactly one board-only commit after a clean
# NO_GRANT state.  code_ref_sha, not the grant commit, is deployed.
"${git_command[@]}" cat-file -e "$authorization_ref^{commit}"
"${git_command[@]}" merge-base --is-ancestor "$authorization_ref" HEAD
[[ "$("${git_command[@]}" rev-list --count "$authorization_ref..HEAD" | tr -d '\r')" = 1 ]]
mapfile -t post_ref_changes < <(
  "${git_command[@]}" diff --name-only "$authorization_ref" HEAD | tr -d '\r'
)
[[ "${#post_ref_changes[@]}" -eq 1 && "${post_ref_changes[0]}" = "$board_relative" ]]
[[ -z "$("${git_command[@]}" status --porcelain --untracked-files=all | tr -d '\r')" ]]

validation_tmp="$(mktemp -d "${TMPDIR:-/tmp}/internnav-t5-d0-dual-validate.XXXXXX")"
cleanup_validation_tmp() { rm -rf -- "$validation_tmp"; }
trap cleanup_validation_tmp EXIT
"${git_command[@]}" show "$authorization_ref:$board_relative" >"$validation_tmp/board.authorization.md"
"${git_command[@]}" show "HEAD:$board_relative" >"$validation_tmp/board.granted.md"

mapfile -t grant_refs < <(python3 - "$validation_tmp/board.granted.md" <<'PY'
import json, re, sys
from pathlib import Path
marker='INTERNAV_T5_DUAL_LANE_ONLINE_GRANT_V1'
matches=re.findall(rf'<!-- {marker}\s*(\{{.*?\}})\s*{marker} -->',
                   Path(sys.argv[1]).read_text(encoding='utf-8'),re.DOTALL)
if len(matches)!=1: raise SystemExit('authority board must contain exactly one grant block')
grant=json.loads(matches[0])
for key in ('code_ref_sha','predecessor_stage','predecessor_result_root',
            'predecessor_receipt_sha256'):
    value=grant.get(key)
    if not isinstance(value,str) or not value: raise SystemExit(f'missing schema-v2 {key}')
    print(value)
PY
)
code_ref="${grant_refs[0]}"
predecessor_stage="${grant_refs[1]}"
predecessor_relative="${grant_refs[2]}"
predecessor_receipt_sha256="${grant_refs[3]}"
[[ "$code_ref" =~ ^[0-9a-f]{40}$ ]]
[[ "$predecessor_receipt_sha256" =~ ^[0-9a-f]{64}$ ]]
[[ "$predecessor_relative" = "$d02_relative" ]]
"${git_command[@]}" cat-file -e "$code_ref^{commit}"
"${git_command[@]}" merge-base --is-ancestor "$code_ref" "$authorization_ref"
mapfile -t code_to_authority_changes < <(
  "${git_command[@]}" diff --name-only "$code_ref" "$authorization_ref" | tr -d '\r'
)
for changed_path in "${code_to_authority_changes[@]}"; do
  [[ "$changed_path" = "$board_relative" ]]
done
"${git_command[@]}" show "$code_ref:$golden_relative" >"$validation_tmp/golden.json"
"${git_command[@]}" show "$code_ref:$d0_manifest_relative" >"$validation_tmp/d0.json"

python3 - "$validation_tmp" "$grant_id" "$authorization_ref" "$code_ref" \
  "$result_relative" "$predecessor_stage" "$predecessor_relative" \
  "$predecessor_receipt_sha256" <<'PY'
import hashlib,json,re,sys
from pathlib import Path
directory=Path(sys.argv[1])
(grant_id,authorization_ref,code_ref,result_root,predecessor_stage,
 predecessor_root,predecessor_sha)=sys.argv[2:]
marker='INTERNAV_T5_DUAL_LANE_ONLINE_GRANT_V1'
pattern=re.compile(rf'<!-- {marker}\s*(\{{.*?\}})\s*{marker} -->',re.DOTALL)
old=(directory/'board.authorization.md').read_text(encoding='utf-8')
new=(directory/'board.granted.md').read_text(encoding='utf-8')
old_matches,new_matches=pattern.findall(old),pattern.findall(new)
if len(old_matches)!=1 or len(new_matches)!=1: raise SystemExit('grant block count mismatch')
placeholder=f'<!-- {marker} GRANT_BLOCK {marker} -->'
if pattern.sub(placeholder,old)!=pattern.sub(placeholder,new):
    raise SystemExit('grant commit changed board outside grant block')
null_fields=('grant_id','authorization_ref_sha','code_ref_sha','stage','lane_scope',
 'resource_profile','candidate_id','golden_bundle_sha256','run_manifest_sha256',
 'result_root','predecessor_stage','predecessor_result_root','predecessor_receipt_sha256')
expected_null={'schema_version':2,'status':'NO_GRANT',**{key:None for key in null_fields}}
if json.loads(old_matches[0])!=expected_null: raise SystemExit('authorization ref was not clean NO_GRANT')
def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
      ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
golden=json.loads((directory/'golden.json').read_text(encoding='utf-8'))
d0=json.loads((directory/'d0.json').read_text(encoding='utf-8'))
golden_sha,d0_sha=canonical(golden),canonical(d0)
if golden.get('status')!='FROZEN_FOR_D0_UNVALIDATED': raise SystemExit('D0 needs frozen Golden v0')
if d0.get('golden_bundle_canonical_sha256')!=golden_sha: raise SystemExit('D0/Golden binding drift')
if d0.get('stage_order',[None]*4)[3]!='d0_3_dual_lane_same_fixed_5':
    raise SystemExit('D0.3 stage order drift')
expected={'schema_version':2,'status':'GRANTED','grant_id':grant_id,
 'authorization_ref_sha':authorization_ref,'code_ref_sha':code_ref,
 'stage':'d0_3_dual_lane_same_fixed_5','lane_scope':'all_lanes',
 'resource_profile':'all-lanes','candidate_id':golden['bundle_id'],
 'golden_bundle_sha256':golden_sha,'run_manifest_sha256':d0_sha,
 'result_root':result_root,'predecessor_stage':predecessor_stage,
 'predecessor_result_root':predecessor_root,
 'predecessor_receipt_sha256':predecessor_sha}
grant=json.loads(new_matches[0])
if grant!=expected: raise SystemExit(f'D0.3 grant mismatch: {grant!r}')
fixed,lanes=d0.get('fixed_input',{}),d0.get('lanes',{})
if fixed.get('episode_count')!=5 or len(fixed.get('episode_keys',[]))!=5:
    raise SystemExit('D0.3 requires frozen fixed five')
if fixed.get('same_episode_on_two_lanes_counts_once') is not True:
    raise SystemExit('D0.3 statistical sample contract drift')
if d0.get('acceptance',{}).get('maximum_simultaneous_slowdown_fraction')!=0.2:
    raise SystemExit('D0.3 20-percent slowdown threshold drift')
if set(lanes)!={'a','b'} or {lanes[x].get('resource_profile') for x in lanes}!={'lane-a','lane-b'}:
    raise SystemExit('symmetric Lane contract drift')
payload={'schema_version':1,'status':'PASS','grant':grant,'code_ref_sha':code_ref,
 'golden_bundle_id':golden['bundle_id'],'golden_bundle_canonical_sha256':golden_sha,
 'run_manifest_canonical_sha256':d0_sha,'episode_keys':fixed['episode_keys'],
 'dataset_root':fixed['dataset_remote_path'],'dataset_file_sha256':fixed['dataset_file_sha256'],
 'maximum_simultaneous_slowdown_fraction':d0['acceptance']['maximum_simultaneous_slowdown_fraction'],
 'lane_contracts':lanes}
(directory/'grant_validation.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY

test -d "$root/$prep_relative" && test ! -L "$root/$prep_relative"
test -d "$root/$d01_relative" && test ! -L "$root/$d01_relative"
test -d "$root/$d02_relative" && test ! -L "$root/$d02_relative"
test -f "$root/$d02_relative/d0_lane_summary.json"
test "$(sha256sum "$root/$d02_relative/d0_lane_summary.json" | cut -d' ' -f1)" = \
  "$predecessor_receipt_sha256"
python3 "$root/scripts/finalize_t5_d0_dual.py" predecessors \
  --prep-root "$prep_relative" --d01-root "$d01_relative" --d02-root "$d02_relative" \
  --grant "$validation_tmp/grant_validation.json" \
  --manifest "$validation_tmp/d0.json" \
  --output "$validation_tmp/predecessor_chain.json"

mapfile -t binding < <(python3 - "$validation_tmp/predecessor_chain.json" <<'PY'
import json,sys
v=json.load(open(sys.argv[1],encoding='utf-8')); roots=v['deployment_roots']
for x in (roots['dgx_a'],roots['dgx_b'],roots['x86_a'],roots['x86_b'],
          v['dataset_root'],v['dataset_file_sha256'],v['static_map_manifest_sha256'],
          v['minimum_available_memory_bytes'],v['golden_bundle_canonical_sha256'],
          v['run_manifest_canonical_sha256']): print(x)
PY
)
dgx_a_root="${binding[0]}"; dgx_b_root="${binding[1]}"
x86_a_root="${binding[2]}"; x86_b_root="${binding[3]}"
dataset_root="${binding[4]}"; dataset_sha256="${binding[5]}"
map_manifest_sha256="${binding[6]}"; minimum_memory_bytes="${binding[7]}"
golden_sha256="${binding[8]}"; run_manifest_sha256="${binding[9]}"
map_a="$dgx_a_root/inputs/d0_fixed5_static_maps/manifest.json"
map_b="$dgx_b_root/inputs/d0_fixed5_static_maps/manifest.json"
dgx_a_run="$dgx_a_root/results/${stage}-${grant_id}-a"
dgx_b_run="$dgx_b_root/results/${stage}-${grant_id}-b"
x86_a_run="$x86_a_root/results/${stage}-${grant_id}-a"
x86_b_run="$x86_b_root/results/${stage}-${grant_id}-b"
result_dir="$root/$result_relative"

# The all-lanes profile is one atomic, fail-closed acquisition in the global
# order DGX_A -> DGX_B -> ISAAC_GPU0 -> ISAAC_GPU1.  Only after all four
# holders exist are the two complete Lane payloads started concurrently.
if [[ "${INTERNNAV_T5_INSIDE_D0_DUAL:-0}" != 1 ]]; then
  test ! -e "$result_dir"
  mkdir -p "$root/results/internnav_t5"
  lease_bootstrap="$(mktemp -d "$root/results/internnav_t5/.d0-3-dual-lease-${grant_id}.XXXXXX")"
  trap - EXIT
  cleanup_validation_tmp
  set +e
  bash "$root/scripts/with_resource_lease.sh" all-lanes \
    --owner codex-00 --task "${stage}:${grant_id}:${authorization_ref}" \
    --log-dir "$lease_bootstrap" --acquire-timeout 30 \
    --cleanup-timeout 600 --kill-wait-timeout 60 -- \
    env INTERNNAV_T5_INSIDE_D0_DUAL=1 INTERNNAV_T5_RESOURCE_LEASE_ACK=all-lanes \
      bash "$root/coordination/run_t5_d0_dual_online.sh" "$grant_id" \
        "$authorization_ref" "$prep_relative" "$d01_relative" "$d02_relative" \
        "$result_relative"
  dual_rc=$?
  set -e
  if test -d "$result_dir"; then
    install -d -m 700 "$result_dir/lease"
    cp -a -- "$lease_bootstrap/." "$result_dir/lease/"
    python3 - "$result_dir/lease_release_summary.json" \
      "$result_dir/lease/lease_metadata.txt" "$dual_rc" <<'PY'
import json,sys,time
from pathlib import Path
output,metadata=Path(sys.argv[1]),Path(sys.argv[2]); rc=int(sys.argv[3])
text=metadata.read_text(encoding='utf-8') if metadata.is_file() else ''
holders=sorted(metadata.parent.glob('holder_*.stdout.log'))
cleanup_path=metadata.parent/'lease_cleanup_receipt.json'
cleanup=json.loads(cleanup_path.read_text(encoding='utf-8')) if cleanup_path.is_file() else None
resources=('dgx_a','dgx_b','isaac_gpu0','isaac_gpu1')
checks={'four_holders_recorded':len(holders)==4,
 'exact_all_lane_resources':all(name in text for name in resources),
 'fixed_global_order':all(text.find(resources[i])<text.find(resources[i+1]) for i in range(3)),
 'wrapped_command_released':'state=RELEASED' in text,
 'wrapped_command_exit_recorded':f'command_exit={rc}' in text,
 'wrapped_group_absent_before_release':isinstance(cleanup,dict) and cleanup.get('status')=='PASS'
   and cleanup.get('wrapped_process_group_absent_before_lock_release') is True}
payload={'schema_version':1,'status':'PASS' if all(checks.values()) else 'FAIL',
 'resource_profile':'all-lanes','command_exit':rc,'checks':checks,
 'holder_logs':[p.name for p in holders],'recorded_unix':time.time()}
with output.open('x',encoding='utf-8',newline='\n') as stream:
    stream.write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
    python3 - "$result_dir/d0_dual_runtime_summary.json" \
      "$result_dir/lease_release_summary.json" "$result_dir/d0_dual_summary.json" \
      "$dual_rc" <<'PY'
import json,sys,time
from pathlib import Path
runtime_path,release_path,output=map(Path,sys.argv[1:4]); rc=int(sys.argv[4])
runtime=json.loads(runtime_path.read_text(encoding='utf-8')) if runtime_path.is_file() else None
release=json.loads(release_path.read_text(encoding='utf-8'))
cleanup_path=runtime_path.parent/'audits/coordinator_cleanup_receipt.json'
cleanup=json.loads(cleanup_path.read_text(encoding='utf-8')) if cleanup_path.is_file() else None
checks={'runtime_pass':isinstance(runtime,dict) and runtime.get('status')=='PASS',
 'lease_release_pass':release.get('status')=='PASS',
 'coordinator_cleanup_pass':isinstance(cleanup,dict) and cleanup.get('status')=='PASS',
 'command_exit_zero':rc==0}
payload={'schema_version':1,'status':'PASS' if all(checks.values()) else 'FAIL',
 'checks':checks,'runtime_summary':runtime,'lease_release':release,
 'coordinator_cleanup':cleanup,'recorded_unix':time.time()}
with output.open('x',encoding='utf-8',newline='\n') as stream:
    stream.write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
    rm -rf -- "$lease_bootstrap"
    final_status="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "$result_dir/d0_dual_summary.json")"
    test "$final_status" = PASS || dual_rc=1
  else
    printf 'D0.3 did not consume result_root; lease diagnostics remain at %s\n' "$lease_bootstrap" >&2
  fi
  exit "$dual_rc"
fi

test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = all-lanes
test ! -e "$result_dir"
umask 077
mkdir "$result_dir"
mkdir "$result_dir/logs" "$result_dir/audits" "$result_dir/remote"
cp "$validation_tmp/grant_validation.json" "$result_dir/grant_validation.json"
cp "$validation_tmp/predecessor_chain.json" "$result_dir/predecessor_chain.json"
cp "$validation_tmp/d0.json" "$result_dir/d0_run_manifest.json"
trap - EXIT
cleanup_validation_tmp

ssh_options=(-T -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2)
x86_target=song@10.100.120.111
dgx_a_target=railgun@10.100.100.128
dgx_b_target=rail@10.100.120.116
containers=(internnav_t5_isaac_a internnav_t5_isaac_b)
declare -A ssh_pid container_started
declare -A launch_attempted
ssh_pid[dgx_a]=""; ssh_pid[dgx_b]=""; ssh_pid[x86_a]=""; ssh_pid[x86_b]=""
container_started[a]=0; container_started[b]=0
launch_attempted[dgx_a]=0; launch_attempted[dgx_b]=0
launch_attempted[x86_a]=0; launch_attempted[x86_b]=0
stop_requested=0
dgx_quarantine_file=/tmp/internnav_dgx.quarantine
isaac_gpu0_quarantine_file=/tmp/internnav_isaac_gpu0.quarantine
isaac_gpu1_quarantine_file=/tmp/internnav_isaac_gpu1.quarantine
dgx_a_quarantine_armed=false
dgx_b_quarantine_armed=false
x86_gpu0_quarantine_armed=false
x86_gpu1_quarantine_armed=false
quarantine_run_tag="${stage}:${grant_id}"
cleanup_timeout="${INTERNNAV_T5_D0_CLEANUP_TIMEOUT_SEC:-360}"
cleanup_kill_timeout="${INTERNNAV_T5_D0_CLEANUP_KILL_TIMEOUT_SEC:-30}"
[[ "$cleanup_timeout" =~ ^[1-9][0-9]*$ && "$cleanup_kill_timeout" =~ ^[1-9][0-9]*$ ]]

remote() { local target="$1"; shift; ssh "${ssh_options[@]}" "$target" "$@"; }

declare -A target run_root supervisor
target[dgx_a]="$dgx_a_target"; target[dgx_b]="$dgx_b_target"
target[x86_a]="$x86_target"; target[x86_b]="$x86_target"
run_root[dgx_a]="$dgx_a_run"; run_root[dgx_b]="$dgx_b_run"
run_root[x86_a]="$x86_a_run"; run_root[x86_b]="$x86_b_run"
supervisor[dgx_a]="${dgx_a_run}.supervisor.json"
supervisor[dgx_b]="${dgx_b_run}.supervisor.json"
supervisor[x86_a]="${x86_a_run}.supervisor.json"
supervisor[x86_b]="${x86_b_run}.supervisor.json"

read -r -d '' supervisor_control_program <<'REMOTE_SUPERVISOR_CONTROL' || true
import json,os,signal,sys,time
from pathlib import Path
ledger,expected_root,action=Path(sys.argv[1]),sys.argv[2],sys.argv[3]
value=json.loads(ledger.read_text(encoding='utf-8'))
if value.get('run_root')!=expected_root: raise SystemExit('supervisor ledger run_root mismatch')
pid,pgid=int(value['pid']),int(value['pgid'])
if pid<=1 or pgid<=1: raise SystemExit('unsafe supervisor identity')
ancestors=set(); cursor=os.getpid()
while cursor>1 and cursor not in ancestors:
    ancestors.add(cursor)
    try:
        status=Path(f'/proc/{cursor}/status').read_text(encoding='utf-8')
        cursor=int(next(x.split()[1] for x in status.splitlines() if x.startswith('PPid:')))
    except Exception: break
table=[]
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name) in ancestors: continue
    try:
        candidate=int(entry.name); candidate_pgid=os.getpgid(candidate)
        command=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip()
        table.append({'pid':candidate,'pgid':candidate_pgid,'command':command})
    except (FileNotFoundError,ProcessLookupError,PermissionError): pass
members=[row for row in table if row['pgid']==pgid]
run_processes=[row for row in table if expected_root in row['command']]
ledger_pgids=set(); runtime=Path(expected_root)/'pid_ledger.jsonl'; runtime_ledger_error=None
if runtime.is_file():
    try:
        rows=[json.loads(x) for x in runtime.read_text().splitlines() if x.strip()]
        absent={(r.get('scope','host'),r.get('component'),r.get('pid')) for r in rows if r.get('event')=='verified_absent'}
        for row in rows:
            key=(row.get('scope','host'),row.get('component'),row.get('pid'))
            if row.get('event')=='started' and key not in absent and row.get('scope','host')=='host':
                value=row.get('pgid')
                if isinstance(value,int) and value>1: ledger_pgids.add(value)
    except (OSError,ValueError,TypeError) as error: runtime_ledger_error=type(error).__name__
managed={group:[row for row in table if row['pgid']==group] for group in sorted(ledger_pgids)}
associated=any(expected_root in row['command'] for row in members); signalled=False
if action in {'TERM','KILL'} and members:
    if not associated: raise SystemExit('refusing to signal reused process group')
    os.killpg(pgid,signal.SIGTERM if action=='TERM' else signal.SIGKILL); signalled=True
if action in {'TERM','KILL'} and (action=='KILL' or not members):
    groups=set(managed); groups.update(row['pgid'] for row in run_processes)
    for group in sorted(groups):
        rows=[row for row in table if row['pgid']==group]
        if not rows or group==pgid: continue
        if not any(expected_root in row['command'] for row in rows):
            raise SystemExit(f'refusing to signal unassociated managed PGID {group}')
        os.killpg(group,signal.SIGTERM if action=='TERM' else signal.SIGKILL); signalled=True
payload={'schema_version':1,'ledger':str(ledger),'run_root':expected_root,'action':action,
 'pid':pid,'pgid':pgid,'members':members,'run_processes':run_processes,
 'managed_groups':managed,'associated':associated,'signalled':signalled,
 'runtime_ledger_error':runtime_ledger_error,
 'absent':runtime_ledger_error is None and not members and not run_processes and not any(managed.values()),
 'recorded_unix':time.time()}
print(json.dumps(payload,sort_keys=True))
REMOTE_SUPERVISOR_CONTROL
supervisor_control_b64="$(printf '%s' "$supervisor_control_program" | base64 | tr -d '\r\n')"

supervisor_action() {
  local key="$1" action="$2" output
  output="$(remote "${target[$key]}" \
    "python3 -c \"\$(printf '%s' '$supervisor_control_b64' | base64 -d)\" '${supervisor[$key]}' '${run_root[$key]}' '$action'" 2>&1)"
  local rc=$?
  printf '%s\n' "$output" >>"$result_dir/audits/coordinator_cleanup_events.jsonl"
  printf '%s\n' "$output"
  return "$rc"
}
supervisor_absent() {
  local value
  value="$(supervisor_action "$1" AUDIT)" || return 1
  python3 -c 'import json,sys;raise SystemExit(0 if json.loads(sys.argv[1]).get("absent") else 1)' "$value"
}

run_root_processes_absent() {
  local target_host="$1" root_scopes="$2"
  remote "$target_host" "python3 - '$root_scopes'" <<'PY'
import os,sys
from pathlib import Path
roots=tuple(x for x in sys.argv[1].split('|') if x); found=[]
if not roots: raise SystemExit(75)
ancestors=set(); cursor=os.getpid()
while cursor>1 and cursor not in ancestors:
    ancestors.add(cursor)
    try:
        lines=Path(f'/proc/{cursor}/status').read_text().splitlines()
        cursor=int(next(x.split()[1] for x in lines if x.startswith('PPid:')))
    except (OSError,StopIteration,ValueError): break
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name) in ancestors: continue
    try: command=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
    except OSError: continue
    if any(root in command for root in roots): found.append(int(entry.name))
raise SystemExit(0 if not found else 75)
PY
}

request_linked_stop() {
  (( stop_requested == 0 )) || return 0
  stop_requested=1
  set +e
  for key in dgx_a dgx_b x86_a x86_b; do
    remote "${target[$key]}" "test ! -d '${run_root[$key]}' || touch '${run_root[$key]}/stop.request'" >/dev/null 2>&1
    remote "${target[$key]}" "cat '${supervisor[$key]}'" \
      >"$result_dir/audits/${key}_supervisor_ledger.json" 2>/dev/null || true
    remote "${target[$key]}" "test -f '${supervisor[$key]}'" >/dev/null 2>&1 || continue
    supervisor_action "$key" TERM >/dev/null 2>&1 || true
  done
  deadline=$((SECONDS + cleanup_timeout))
  while (( SECONDS < deadline )); do
    residual=0
    for key in dgx_a dgx_b x86_a x86_b; do
      remote "${target[$key]}" "test -f '${supervisor[$key]}'" >/dev/null 2>&1 || continue
      supervisor_absent "$key" >/dev/null 2>&1 || residual=$((residual+1))
    done
    test "$residual" = 0 && break
    sleep 1
  done
  for key in dgx_a dgx_b x86_a x86_b; do
    remote "${target[$key]}" "test -f '${supervisor[$key]}'" >/dev/null 2>&1 || continue
    supervisor_absent "$key" >/dev/null 2>&1 || supervisor_action "$key" KILL >/dev/null 2>&1 || true
  done
  kill_deadline=$((SECONDS + cleanup_kill_timeout))
  while (( SECONDS < kill_deadline )); do
    residual=0
    for key in dgx_a dgx_b x86_a x86_b; do
      remote "${target[$key]}" "test -f '${supervisor[$key]}'" >/dev/null 2>&1 || continue
      supervisor_absent "$key" >/dev/null 2>&1 || residual=$((residual+1))
    done
    test "$residual" = 0 && break
    sleep 1
  done
  for key in dgx_a dgx_b x86_a x86_b; do
    pid="${ssh_pid[$key]}"
    test -z "$pid" || ! kill -0 "$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  done
  for lane in a b; do
    container="internnav_t5_isaac_$lane"
    if test "$lane" = a; then expected_root="$x86_a_root"; else expected_root="$x86_b_root"; fi
    remote "$x86_target" "set +e; test \"\$(docker inspect -f '{{index .Config.Labels \"internnav.t5.deployment_root\"}}' '$container')\" = '$expected_root' || exit 75; docker stop -t 15 '$container' >/dev/null 2>&1; if test \"\$(docker inspect -f '{{.State.Running}}' '$container' 2>/dev/null)\" = true; then docker kill '$container' >/dev/null 2>&1; fi" >/dev/null 2>&1 || true
    container_started[$lane]=0
  done
  declare -A absent
  for key in dgx_a dgx_b x86_a x86_b; do
    absent[$key]=false
    if remote "${target[$key]}" "test -f '${supervisor[$key]}'" >/dev/null 2>&1; then
      supervisor_absent "$key" >/dev/null 2>&1 && absent[$key]=true
    elif test "${launch_attempted[$key]}" = 0; then
      if [[ "$key" == dgx_a ]]; then deployment_scope="$dgx_a_root"
      elif [[ "$key" == dgx_b ]]; then deployment_scope="$dgx_b_root"
      elif [[ "$key" == x86_a ]]; then deployment_scope="$x86_a_root"
      else deployment_scope="$x86_b_root"; fi
      run_root_processes_absent "${target[$key]}" "${run_root[$key]}|$deployment_scope" >/dev/null 2>&1 && absent[$key]=true
    fi
  done
  x86_a_clean=false; x86_b_clean=false; shared_assets_clean=false
  dgx_a_endpoints_clean=false; dgx_b_endpoints_clean=false
  dgx_a_structured_compute_absent=false
  dgx_b_structured_compute_absent=false
  remote "$x86_target" "set -e; c=internnav_t5_isaac_a; test \"\$(docker inspect -f '{{index .Config.Labels \"internnav.t5.deployment_root\"}}' \"\$c\")\" = '$x86_a_root'; test \"\$(docker inspect -f '{{.State.Running}}' \"\$c\")\" = false; test \"\$(docker inspect -f '{{.State.Pid}}' \"\$c\")\" = 0; test ! -e /tmp/internnav_t5_a_ipc/isaac_health.sock; test ! -e /tmp/internnav_t5_isaac_a_runtime.lock || flock -n /tmp/internnav_t5_isaac_a_runtime.lock true; for p in 25137 25139 25140 25141; do test -z \"\$(ss -H -lntup|grep -E \"[:.]\$p[[:space:]]\"||true)\"; done" >/dev/null 2>&1 && x86_a_clean=true
  remote "$x86_target" "set -e; c=internnav_t5_isaac_b; test \"\$(docker inspect -f '{{index .Config.Labels \"internnav.t5.deployment_root\"}}' \"\$c\")\" = '$x86_b_root'; test \"\$(docker inspect -f '{{.State.Running}}' \"\$c\")\" = false; test \"\$(docker inspect -f '{{.State.Pid}}' \"\$c\")\" = 0; test ! -e /tmp/internnav_t5_b_ipc/isaac_health.sock; test ! -e /tmp/internnav_t5_isaac_b_runtime.lock || flock -n /tmp/internnav_t5_isaac_b_runtime.lock true; for p in 25138 25239 25240 25241; do test -z \"\$(ss -H -lntup|grep -E \"[:.]\$p[[:space:]]\"||true)\"; done" >/dev/null 2>&1 && x86_b_clean=true
  remote "$x86_target" "test ! -e /tmp/internnav_t5_isaac_shared_assets.lock || flock -n /tmp/internnav_t5_isaac_shared_assets.lock true" >/dev/null 2>&1 && shared_assets_clean=true
  remote "$dgx_a_target" "for p in 25137 25139 25140 25141; do test -z \"\$(ss -H -lntup|grep -E \"[:.]\$p[[:space:]]\"||true)\" || exit 1; done" >/dev/null 2>&1 && dgx_a_endpoints_clean=true
  remote "$dgx_b_target" "for p in 25138 25239 25240 25241; do test -z \"\$(ss -H -lntup|grep -E \"[:.]\$p[[:space:]]\"||true)\" || exit 1; done" >/dev/null 2>&1 && dgx_b_endpoints_clean=true
  t5_remote_compute_absent "$dgx_a_target" \
    "$result_dir/audits/dgx_a_structured_compute_poststop.json" \
    >/dev/null 2>&1 && dgx_a_structured_compute_absent=true
  t5_remote_compute_absent "$dgx_b_target" \
    "$result_dir/audits/dgx_b_structured_compute_poststop.json" \
    >/dev/null 2>&1 && dgx_b_structured_compute_absent=true
  python3 - "$result_dir/audits/coordinator_cleanup_receipt.json" \
    "${absent[dgx_a]}" "${absent[dgx_b]}" "${absent[x86_a]}" "${absent[x86_b]}" \
    "$x86_a_clean" "$x86_b_clean" "$shared_assets_clean" \
    "$dgx_a_endpoints_clean" "$dgx_b_endpoints_clean" \
    "$dgx_a_structured_compute_absent" "$dgx_b_structured_compute_absent" \
    "$cleanup_timeout" "$cleanup_kill_timeout" <<'PY'
import json,sys,time
from pathlib import Path
output=Path(sys.argv[1])
lane_a_checks={'dgx_supervisor_absent':sys.argv[2]=='true',
 'x86_supervisor_absent':sys.argv[4]=='true','x86_owned_container_socket_ports_lock_clean':sys.argv[6]=='true',
 'shared_asset_lock_clean':sys.argv[8]=='true','dgx_sockets_clean':sys.argv[9]=='true',
 'dgx_structured_compute_absent':sys.argv[11]=='true'}
lane_b_checks={'dgx_supervisor_absent':sys.argv[3]=='true',
 'x86_supervisor_absent':sys.argv[5]=='true','x86_owned_container_socket_ports_lock_clean':sys.argv[7]=='true',
 'shared_asset_lock_clean':sys.argv[8]=='true','dgx_sockets_clean':sys.argv[10]=='true',
 'dgx_structured_compute_absent':sys.argv[12]=='true'}
common={'schema_version':1,'term_then_kill':True,'graceful_timeout_sec':int(sys.argv[13]),
 'kill_wait_timeout_sec':int(sys.argv[14]),'recorded_unix':time.time()}
for lane,checks in (('a',lane_a_checks),('b',lane_b_checks)):
    payload={**common,'lane':lane,'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks}
    (output.parent/f'coordinator_cleanup_lane_{lane}_receipt.json').write_text(
        json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
checks={'lane_a_cleanup':all(lane_a_checks.values()),'lane_b_cleanup':all(lane_b_checks.values())}
payload={'schema_version':1,'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
 'lane_checks':{'a':lane_a_checks,'b':lane_b_checks},'term_then_kill':True,
 'graceful_timeout_sec':int(sys.argv[13]),'kill_wait_timeout_sec':int(sys.argv[14]),
 'recorded_unix':time.time()}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
raise SystemExit(0 if payload['status']=='PASS' else 1)
PY
  cleanup_rc=$?
  set -e
  return "$cleanup_rc"
}

cleanup_inside() {
  local incoming=$?
  trap - EXIT INT TERM HUP
  request_linked_stop || incoming=1
  set +e
  cleanup_receipt_a="$result_dir/audits/coordinator_cleanup_lane_a_receipt.json"
  cleanup_receipt_b="$result_dir/audits/coordinator_cleanup_lane_b_receipt.json"
  if test "$dgx_a_quarantine_armed" = true; then
    t5_quarantine_owned_clear "$dgx_a_target" "$dgx_quarantine_file" dgx_a \
      "$quarantine_run_tag" "$dgx_a_root" "$cleanup_receipt_a" \
      "$result_dir/audits/dgx_a_quarantine_clear.json" || incoming=1
  fi
  if test "$dgx_b_quarantine_armed" = true; then
    t5_quarantine_owned_clear "$dgx_b_target" "$dgx_quarantine_file" dgx_b \
      "$quarantine_run_tag" "$dgx_b_root" "$cleanup_receipt_b" \
      "$result_dir/audits/dgx_b_quarantine_clear.json" || incoming=1
  fi
  if test "$x86_gpu0_quarantine_armed" = true; then
    t5_quarantine_owned_clear "$x86_target" "$isaac_gpu0_quarantine_file" \
      x86_gpu0 "$quarantine_run_tag" "$x86_a_root" "$cleanup_receipt_a" \
      "$result_dir/audits/x86_gpu0_quarantine_clear.json" || incoming=1
  fi
  if test "$x86_gpu1_quarantine_armed" = true; then
    t5_quarantine_owned_clear "$x86_target" "$isaac_gpu1_quarantine_file" \
      x86_gpu1 "$quarantine_run_tag" "$x86_b_root" "$cleanup_receipt_b" \
      "$result_dir/audits/x86_gpu1_quarantine_clear.json" || incoming=1
  fi
  wait 2>/dev/null || true; set -e
  partial_count=0
  if test "$incoming" != 0; then
    set +e
    for key in dgx_a dgx_b x86_a x86_b; do
      destination="$result_dir/remote_failure/$key"
      if test ! -e "$destination" && \
          remote "${target[$key]}" "test -d '${run_root[$key]}'" >/dev/null 2>&1; then
        mkdir -p "$destination"
        if remote "${target[$key]}" "tar -C '${run_root[$key]}' -czf - ." \
            >"$destination.tar.gz" 2>/dev/null && \
            tar -C "$destination" -xzf "$destination.tar.gz" 2>/dev/null; then
          partial_count=$((partial_count+1))
        fi
      fi
    done
    set -e
  fi
  if test "$incoming" != 0 && test -n "${HF_TOKEN:-}" && \
      test ! -e "$result_dir/audits/credential_exposure_audit.json"; then
    exec {failure_secret_fd}<<<"$HF_TOKEN"
    python3 - "$result_dir" "$result_dir/audits/credential_exposure_audit.json" \
      "$failure_secret_fd" <<'PY' || true
import json,os,sys
from pathlib import Path
root,output=Path(sys.argv[1]),Path(sys.argv[2])
with os.fdopen(int(sys.argv[3]),'rb',closefd=True) as stream: secret=stream.read().rstrip(b'\r\n')
matches=[]
for path in root.rglob('*'):
    if path.is_file() and path!=output and secret and secret in path.read_bytes():
        matches.append(str(path.relative_to(root)))
payload={'schema_version':1,'status':'PASS' if not matches else 'FAIL',
 'scan_context':'failure_cleanup','exact_secret_match_count':len(matches),
 'matched_relative_paths':sorted(matches)}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY
    unset HF_TOKEN
  fi
  if test "$incoming" != 0 && test ! -e "$result_dir/d0_dual_failure.json"; then
    python3 - "$result_dir/d0_dual_failure.json" "$incoming" "$partial_count" <<'PY'
import json,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'schema_version':1,'status':'FAIL',
 'stage':'d0_3_dual_lane_same_fixed_5','observed_exit_code':int(sys.argv[2]),
 'partial_remote_evidence_tree_count':int(sys.argv[3]),'partial_evidence_preserved':True,
 'recommended_fallback':'INTERLEAVED_SINGLE_LANE_EXECUTION',
 'recorded_unix':time.time()},indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY
  fi
  exit "$incoming"
}
trap cleanup_inside EXIT
trap 'exit 130' INT TERM HUP

# Exact-ref, stopped-container and free-endpoint preflight on all three hosts.
remote "$dgx_a_target" "set -euo pipefail; test \"\$(id -un)\" = railgun; ip -4 -o addr show | grep -Fq ' 10.100.100.128/'; test \"\$(cat '$dgx_a_root/T5_DEPLOYMENT_REF')\" = '$code_ref'; test -x '$dgx_a_root/scripts/run_t5_dgx_lane.sh'; test -f '$dgx_a_root/ros_ws/install/setup.bash'; test ! -e '$dgx_a_run'; test ! -e '${supervisor[dgx_a]}'; test \"\$(sha256sum '$map_a'|cut -d' ' -f1)\" = '$map_manifest_sha256'"
remote "$dgx_b_target" "set -euo pipefail; test \"\$(id -un)\" = rail; ip -4 -o addr show | grep -Fq ' 10.100.120.116/'; test \"\$(cat '$dgx_b_root/T5_DEPLOYMENT_REF')\" = '$code_ref'; test -x '$dgx_b_root/scripts/run_t5_dgx_lane.sh'; test -f '$dgx_b_root/ros_ws/install/setup.bash'; test ! -e '$dgx_b_run'; test ! -e '${supervisor[dgx_b]}'; test \"\$(sha256sum '$map_b'|cut -d' ' -f1)\" = '$map_manifest_sha256'"
remote "$x86_target" "set -euo pipefail; test \"\$(id -un)\" = song; ip -4 -o addr show | grep -Fq ' 10.100.120.111/'; for root in '$x86_a_root' '$x86_b_root'; do test \"\$(cat \"\$root/T5_DEPLOYMENT_REF\")\" = '$code_ref'; test -x \"\$root/scripts/run_t5_distributed_isaac.sh\"; done; test ! -e '$x86_a_run'; test ! -e '$x86_b_run'; test ! -e '${supervisor[x86_a]}'; test ! -e '${supervisor[x86_b]}'; test \"\$(sha256sum '$dataset_root/val_unseen/val_unseen.json.gz'|cut -d' ' -f1)\" = '$dataset_sha256'; for c in internnav_t5_isaac_a internnav_t5_isaac_b; do test \"\$(docker inspect -f '{{.State.Running}}' \"\$c\")\" = false; test \"\$(docker inspect -f '{{.State.Pid}}' \"\$c\")\" = 0; done; for p in 25137 25138 25139 25140 25141 25239 25240 25241; do test -z \"\$(ss -H -lntup|grep -E \"[:.]\$p[[:space:]]\"||true)\"; done; flock -n /tmp/internnav_t5_isaac_shared_assets.lock true"

# Arm every involved host after read-only preflight and before capacity output,
# remote supervisors, or either Isaac container can be created/started.
t5_quarantine_arm "$dgx_a_target" "$dgx_quarantine_file" dgx_a \
  "$quarantine_run_tag" "$dgx_a_root" \
  "$result_dir/audits/dgx_a_quarantine_arm.json"
dgx_a_quarantine_armed=true
t5_quarantine_arm "$dgx_b_target" "$dgx_quarantine_file" dgx_b \
  "$quarantine_run_tag" "$dgx_b_root" \
  "$result_dir/audits/dgx_b_quarantine_arm.json"
dgx_b_quarantine_armed=true
t5_quarantine_arm "$x86_target" "$isaac_gpu0_quarantine_file" x86_gpu0 \
  "$quarantine_run_tag" "$x86_a_root" \
  "$result_dir/audits/x86_gpu0_quarantine_arm.json"
x86_gpu0_quarantine_armed=true
t5_quarantine_arm "$x86_target" "$isaac_gpu1_quarantine_file" x86_gpu1 \
  "$quarantine_run_tag" "$x86_b_root" \
  "$result_dir/audits/x86_gpu1_quarantine_arm.json"
x86_gpu1_quarantine_armed=true

# Whole-host production identity checks close the gap left by Nav2 children
# that create independent sessions and can outlive a deployment-scoped
# supervisor. Both checks are held under quarantine and precede any runtime.
t5_remote_compute_absent "$dgx_a_target" \
  "$result_dir/audits/dgx_a_structured_compute_prestart.json"
t5_remote_compute_absent "$dgx_b_target" \
  "$result_dir/audits/dgx_b_structured_compute_prestart.json"

# Recheck the shared x86 admission immediately before starting either worker.
read -r -d '' capacity_program <<'REMOTE_CAPACITY' || true
import json,os,subprocess,sys,time
from pathlib import Path
output=Path(sys.argv[1]); minimum=int(sys.argv[2]); expected_code=sys.argv[3]
mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())}
stat=os.statvfs('/home/song'); available=stat.f_bavail*stat.f_frsize
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.total','--format=csv,noheader,nounits'],text=True,capture_output=True,check=False)
containers={}
for lane,gpu_index,cpuset,domain,root in (
 ('a','0','0-7,16-23','75',sys.argv[4]),('b','1','8-15,24-31','76',sys.argv[5])):
    name=f'internnav_t5_isaac_{lane}'
    value=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
    labels=value['Config'].get('Labels') or {}; env=dict(item.split('=',1) for item in value['Config']['Env'] if '=' in item)
    request=value['HostConfig'].get('DeviceRequests') or []
    writable=sorted(str(mount.get('Source')) for mount in value.get('Mounts',[]) if mount.get('RW'))
    containers[lane]={'name':name,'id':value['Id'],'running':value['State']['Running'],
      'pid':value['State']['Pid'],'cpuset':value['HostConfig']['CpusetCpus'],
      'domain':env.get('ROS_DOMAIN_ID'),'gpu_label':labels.get('internnav.t5.gpu'),
      'deployment_label':labels.get('internnav.t5.deployment_root'),
      'nvidia_visible_devices':env.get('NVIDIA_VISIBLE_DEVICES'),
      'container_cuda_visible_devices':env.get('CUDA_VISIBLE_DEVICES'),
      'profile':env.get('XDG_CACHE_HOME'),'device_requests':request,
      'writable_mount_sources':writable}
checks={'memory_admission':mem.get('MemAvailable',0)>=minimum,'logical_cpu_count':os.cpu_count()==32,
 'filesystem_has_free_space':available>20*1024**3,'two_visible_gpus':gpu.returncode==0 and len(gpu.stdout.splitlines())==2,
 'containers_stopped':all(not x['running'] and x['pid']==0 for x in containers.values()),
 'cpusets_exact':containers['a']['cpuset']=='0-7,16-23' and containers['b']['cpuset']=='8-15,24-31',
 'domains_exact':containers['a']['domain']=='75' and containers['b']['domain']=='76',
 'physical_gpus_exact':containers['a']['gpu_label']=='0' and containers['b']['gpu_label']=='1',
 'gpu_device_requests_exact':all(len(containers[lane]['device_requests'])==1
   and sorted(containers[lane]['device_requests'][0].get('DeviceIDs') or [])==[str(index)]
   and containers[lane]['nvidia_visible_devices']==str(index)
   and containers[lane]['container_cuda_visible_devices']=='0'
   for lane,index in (('a',0),('b',1))),
 'cache_profiles_disjoint':containers['a']['profile']!=containers['b']['profile'],
 'deployment_mounts_isolated':containers['a']['deployment_label']==sys.argv[4]
   and containers['b']['deployment_label']==sys.argv[5]
   and sys.argv[4] in containers['a']['writable_mount_sources']
   and sys.argv[5] in containers['b']['writable_mount_sources']
   and sys.argv[5] not in containers['a']['writable_mount_sources']
   and sys.argv[4] not in containers['b']['writable_mount_sources']
   and set(containers['a']['writable_mount_sources']).isdisjoint(
       set(containers['b']['writable_mount_sources'])),
 'deployment_refs_exact':all((Path(root)/'T5_DEPLOYMENT_REF').read_text().strip()==expected_code for root in sys.argv[4:6])}
payload={'schema_version':1,'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
 'memory':{'available_bytes':mem.get('MemAvailable',0),'minimum_available_bytes':minimum},
 'filesystem_available_bytes':available,'gpu_inventory_lines':len(gpu.stdout.splitlines()),
 'containers':containers,'on_failure':'PRESERVE_RESUME_PLAN_NOTIFY_USER_NO_AUTO_REBOOT',
 'recorded_unix':time.time()}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
if payload['status']!='PASS': raise SystemExit(f'D0.3 capacity admission failed: {checks}')
REMOTE_CAPACITY
capacity_b64="$(printf '%s' "$capacity_program" | base64 | tr -d '\r\n')"
set +e
remote "$x86_target" "python3 -c \"\$(printf '%s' '$capacity_b64'|base64 -d)\" /tmp/internnav_t5_d03_capacity_${grant_id}.json '$minimum_memory_bytes' '$code_ref' '$x86_a_root' '$x86_b_root'"
capacity_rc=$?
remote "$x86_target" "cat /tmp/internnav_t5_d03_capacity_${grant_id}.json" \
  >"$result_dir/audits/capacity_prestart.json"
capacity_collect_rc=$?
set -e
test "$capacity_collect_rc" = 0
test "$capacity_rc" = 0

read -r -d '' dgx_runtime_program <<'REMOTE_DGX' || true
set -euo pipefail
IFS= read -r HF_TOKEN; IFS= read -r HF_ENDPOINT
[[ "$HF_TOKEN" =~ ^hf_[A-Za-z0-9]{20,}$ ]]; [[ "$HF_ENDPOINT" =~ ^https://hf-mirror\.com/?$ ]]
exec {hf_token_fd}<<<"$HF_TOKEN"; unset HF_TOKEN
deployment="$1"; lane="$2"; result="$3"; map="$4"; domain="$5"; lease="$6"; ledger="$7"
pgid="$(ps -o pgid= -p "$$"|tr -d ' ')"
python3 - "$ledger" "$result" "$$" "$pgid" <<'PY'
import json,os,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'schema_version':1,'state':'RUNNING','run_root':sys.argv[2],
 'pid':int(sys.argv[3]),'pgid':int(sys.argv[4]),'component':'dgx','host':os.uname().nodename,
 'started_unix':time.time()},indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY
exec env INTERNNAV_T5_RESOURCE_LEASE_ACK="$lease" INTERNVLA_HF_TOKEN_FD="$hf_token_fd" \
 HF_ENDPOINT="$HF_ENDPOINT" INTERNNAV_RUNTIME_POLICY=completion_sim INTERNNAV_SIMULATION_TARGET=isaac \
 ROS_DOMAIN_ID="$domain" CUDA_VISIBLE_DEVICES=0 INTERNNAV_T1_CONTROL_ROOT="$deployment" \
 INTERNVLA_ROS_WS="$deployment/ros_ws" bash "$deployment/scripts/run_t5_dgx_lane.sh" \
 "$lane" model "$result" "$map"
REMOTE_DGX
dgx_b64="$(printf '%s' "$dgx_runtime_program"|base64|tr -d '\r\n')"
for lane in a b; do
  if test "$lane" = a; then deployment="$dgx_a_root"; lane_target="$dgx_a_target"; lane_run="$dgx_a_run"; map="$map_a"; domain=75; lease=lane-a; key=dgx_a
  else deployment="$dgx_b_root"; lane_target="$dgx_b_target"; lane_run="$dgx_b_run"; map="$map_b"; domain=76; lease=lane-b; key=dgx_b; fi
  command="exec setsid --wait bash -c \"\$(printf '%s' '$dgx_b64'|base64 -d)\" d03-dgx '$deployment' '$lane' '$lane_run' '$map' '$domain' '$lease' '${supervisor[$key]}'"
  launch_attempted[$key]=1
  printf '%s\n%s\n' "$HF_TOKEN" "$HF_ENDPOINT" | ssh "${ssh_options[@]}" "$lane_target" "$command" \
    >"$result_dir/logs/${key}_runtime_ssh.log" 2>&1 &
  ssh_pid[$key]=$!
done

ready_timeout="${INTERNNAV_T5_D0_DGX_READY_TIMEOUT_SEC:-7200}"
[[ "$ready_timeout" =~ ^[1-9][0-9]*$ ]]
for lane in a b; do
  key="dgx_$lane"; deadline=$((SECONDS+ready_timeout))
  while (( SECONDS < deadline )); do
    kill -0 "${ssh_pid[$key]}" 2>/dev/null || break
    if remote "${target[$key]}" "test -f '${run_root[$key]}/lane_ready.json' && test -f '${run_root[$key]}/hf_token_process_audit.json' && python3 -c 'import json;r=json.load(open(\"${run_root[$key]}/lane_ready.json\"));a=json.load(open(\"${run_root[$key]}/hf_token_process_audit.json\"));assert r[\"status\"]==\"READY\" and r[\"lane\"]==\"$lane\" and r[\"mode\"]==\"model\";assert a[\"status\"]==\"PASS\" and a[\"exact_secret_match_count\"]>=1 and a[\"disallowed_match_count\"]==0'" >/dev/null 2>&1; then break; fi
    sleep 2
  done
  kill -0 "${ssh_pid[$key]}" 2>/dev/null
  remote "${target[$key]}" "test -f '${run_root[$key]}/lane_ready.json' && test -f '${run_root[$key]}/hf_token_process_audit.json'"
done

# Both complete DGX stacks are READY before the two isolated Isaac workers are
# launched back-to-back.  Their evaluators then overlap on the same fixed five.
remote "$x86_target" "docker start internnav_t5_isaac_a internnav_t5_isaac_b >/dev/null; for c in internnav_t5_isaac_a internnav_t5_isaac_b; do test \"\$(docker inspect -f '{{.State.Running}}' \"\$c\")\" = true; done"
container_started[a]=1; container_started[b]=1
read -r -d '' x86_runtime_program <<'REMOTE_X86' || true
set -euo pipefail
deployment="$1"; lane="$2"; result="$3"; dataset="$4"; domain="$5"; gpu="$6"; lease="$7"; ledger="$8"
pgid="$(ps -o pgid= -p "$$"|tr -d ' ')"
python3 - "$ledger" "$result" "$$" "$pgid" <<'PY'
import json,os,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({'schema_version':1,'state':'RUNNING','run_root':sys.argv[2],
 'pid':int(sys.argv[3]),'pgid':int(sys.argv[4]),'component':'x86','host':os.uname().nodename,
 'started_unix':time.time()},indent=2,sort_keys=True)+'\n',encoding='utf-8')
PY
exec env INTERNNAV_T5_RESOURCE_LEASE_ACK="$lease" INTERNNAV_RUNTIME_POLICY=completion_sim \
 INTERNNAV_SIMULATION_TARGET=isaac INTERNNAV_T5_ENGINEERING_CANARY_SEC=0 \
 ROS_DOMAIN_ID="$domain" CUDA_VISIBLE_DEVICES="$gpu" \
 INTERNNAV_T1_CONTROL_ROOT="$deployment" INTERNVLA_ROS_WS=/home/song/internnav-t4/isaac_ros_ws_45 \
 INTERNVLA_T5_ISAAC_WORKER_ROOT=/home/song/internnav-t1-t2/runtime/t5_isaac_workers \
 bash "$deployment/scripts/run_t5_distributed_isaac.sh" "$lane" model "$result" "$dataset"
REMOTE_X86
x86_b64="$(printf '%s' "$x86_runtime_program"|base64|tr -d '\r\n')"
for lane in a b; do
  if test "$lane" = a; then deployment="$x86_a_root"; lane_run="$x86_a_run"; domain=75; gpu=0; lease=lane-a; key=x86_a
  else deployment="$x86_b_root"; lane_run="$x86_b_run"; domain=76; gpu=1; lease=lane-b; key=x86_b; fi
  command="exec setsid --wait bash -c \"\$(printf '%s' '$x86_b64'|base64 -d)\" d03-x86 '$deployment' '$lane' '$lane_run' '$dataset_root' '$domain' '$gpu' '$lease' '${supervisor[$key]}'"
  launch_attempted[$key]=1
  remote "$x86_target" "$command" >"$result_dir/logs/${key}_runtime_ssh.log" 2>&1 &
  ssh_pid[$key]=$!
done

x86_ready_timeout="${INTERNNAV_T5_D0_X86_READY_TIMEOUT_SEC:-1200}"
run_timeout="${INTERNNAV_T5_D0_RUN_TIMEOUT_SEC:-21600}"
[[ "$x86_ready_timeout" =~ ^[1-9][0-9]*$ && "$run_timeout" =~ ^[1-9][0-9]*$ ]]
for lane in a b; do
  key="x86_$lane"; dgx_key="dgx_$lane"; deadline=$((SECONDS+x86_ready_timeout))
  while (( SECONDS < deadline )); do
    kill -0 "${ssh_pid[$dgx_key]}" 2>/dev/null || { echo "DGX $lane exited before Isaac readiness" >&2; exit 1; }
    kill -0 "${ssh_pid[$key]}" 2>/dev/null || break
    if remote "$x86_target" "test -f '${run_root[$key]}/health/ready_probe.json' && python3 -c 'import json;v=json.load(open(\"${run_root[$key]}/health/ready_probe.json\"));assert v[\"status\"]==\"PASS\" and v[\"lane\"]==\"$lane\"'" >/dev/null 2>&1; then break; fi
    sleep 2
  done
  kill -0 "${ssh_pid[$key]}" 2>/dev/null
done

# Live snapshots prove distinct PIDs, GPU labels, cpusets, domains and writable
# profile roots while both containers are simultaneously running.
read -r -d '' live_audit_program <<'REMOTE_LIVE_AUDIT' || true
import json,subprocess,sys,time
rows={}
for lane in ('a','b'):
    value=json.loads(subprocess.check_output(['docker','inspect',f'internnav_t5_isaac_{lane}'],text=True))[0]
    env=dict(x.split('=',1) for x in value['Config']['Env'] if '=' in x)
    rows[lane]={'id':value['Id'],'pid':value['State']['Pid'],'running':value['State']['Running'],
      'cpuset':value['HostConfig']['CpusetCpus'],'domain':env.get('ROS_DOMAIN_ID'),
      'profile':env.get('XDG_CACHE_HOME'),'gpu':(value['Config'].get('Labels') or {}).get('internnav.t5.gpu')}
checks={'both_running':all(x['running'] and x['pid']>1 for x in rows.values()),
 'distinct_container_pids':rows['a']['pid']!=rows['b']['pid'],
 'cpusets_exact':rows['a']['cpuset']=='0-7,16-23' and rows['b']['cpuset']=='8-15,24-31',
 'domains_disjoint':rows['a']['domain']=='75' and rows['b']['domain']=='76',
 'gpus_disjoint':rows['a']['gpu']=='0' and rows['b']['gpu']=='1',
 'profiles_disjoint':rows['a']['profile']!=rows['b']['profile']}
print(json.dumps({'schema_version':1,'status':'PASS' if all(checks.values()) else 'FAIL',
 'checks':checks,'containers':rows,'recorded_unix':time.time()},sort_keys=True))
raise SystemExit(0 if all(checks.values()) else 1)
REMOTE_LIVE_AUDIT
live_b64="$(printf '%s' "$live_audit_program"|base64|tr -d '\r\n')"
run_deadline=$((SECONDS+run_timeout)); next_audit=$SECONDS
while kill -0 "${ssh_pid[x86_a]}" 2>/dev/null || kill -0 "${ssh_pid[x86_b]}" 2>/dev/null; do
  for lane in a b; do dgx_key="dgx_$lane"; kill -0 "${ssh_pid[$dgx_key]}" 2>/dev/null || { echo "DGX $lane failed during dual Isaac run" >&2; exit 1; }; done
  (( SECONDS < run_deadline )) || { echo 'D0.3 dual fixed-five timed out' >&2; exit 124; }
  if (( SECONDS >= next_audit )); then
    remote "$x86_target" "python3 -c \"\$(printf '%s' '$live_b64'|base64 -d)\"" >>"$result_dir/audits/live_isolation_snapshots.jsonl"
    next_audit=$((SECONDS+15))
  fi
  sleep 2
done
declare -A runtime_rc
for lane in a b; do key="x86_$lane"; set +e; wait "${ssh_pid[$key]}"; runtime_rc[$key]=$?; set -e; ssh_pid[$key]=""; done
for lane in a b; do dgx_key="dgx_$lane"; remote "${target[$dgx_key]}" "touch '${run_root[$dgx_key]}/stop.request'"; done
for lane in a b; do
  key="dgx_$lane"; deadline=$((SECONDS+300))
  while kill -0 "${ssh_pid[$key]}" 2>/dev/null && (( SECONDS<deadline )); do sleep 1; done
  kill -0 "${ssh_pid[$key]}" 2>/dev/null && { echo "DGX $lane did not stop" >&2; exit 124; }
  set +e; wait "${ssh_pid[$key]}"; runtime_rc[$key]=$?; set -e; ssh_pid[$key]=""
done
remote "$x86_target" "for c in internnav_t5_isaac_a internnav_t5_isaac_b; do docker stop -t 20 \"\$c\" >/dev/null; test \"\$(docker inspect -f '{{.State.Running}}' \"\$c\")\" = false; test \"\$(docker inspect -f '{{.State.Pid}}' \"\$c\")\" = 0; done"
container_started[a]=0; container_started[b]=0
for key in x86_a x86_b dgx_a dgx_b; do test "${runtime_rc[$key]}" = 0; done

collect_tree() {
  local target_host="$1" source="$2" destination="$3"
  test ! -e "$destination"; mkdir -p "$destination"
  remote "$target_host" "tar -C '$source' -czf - ." >"$destination.tar.gz"
  tar -C "$destination" -xzf "$destination.tar.gz"
}
collect_tree "$dgx_a_target" "$dgx_a_run" "$result_dir/remote/lane_a/dgx"
collect_tree "$x86_target" "$x86_a_run" "$result_dir/remote/lane_a/x86"
collect_tree "$dgx_b_target" "$dgx_b_run" "$result_dir/remote/lane_b/dgx"
collect_tree "$x86_target" "$x86_b_run" "$result_dir/remote/lane_b/x86"

# Cross-Lane evidence is evaluated from runtime contracts plus live snapshots;
# topic/TF readiness is namespaced on both DGXs, and the opposite identity
# prefix is forbidden in evaluator/reset/request evidence.
python3 - "$result_dir" "$result_dir/audits/cross_lane_isolation.json" <<'PY'
import json,sys,time
from pathlib import Path
root,output=Path(sys.argv[1]),Path(sys.argv[2])
def load(path): return json.loads(path.read_text(encoding='utf-8'))
def contains(value,prefix):
    if isinstance(value,dict): return any(contains(x,prefix) for x in value.values())
    if isinstance(value,list): return any(contains(x,prefix) for x in value)
    return isinstance(value,str) and value.startswith(prefix)
contracts={lane:{role:load(root/'remote'/f'lane_{lane}'/role/(f'{"lane" if role=="dgx" else "isaac"}_contract.json'))
                 for role in ('dgx','x86')} for lane in ('a','b')}
ready={lane:load(root/'remote'/f'lane_{lane}'/'dgx'/'nav2_data_plane_evaluator_ready.json') for lane in ('a','b')}
results={}
for lane in ('a','b'):
    base=root/'remote'/f'lane_{lane}'/'x86'/'evaluator'
    attempts=[path for path in base.glob(f't5_lane_{lane}_model*_model_attempt_*')
              if (path/'result.json').is_file()
              and (path/'isaac_remote_validation.json').is_file()]
    if len(attempts)!=1: raise SystemExit('ambiguous Lane evaluator evidence')
    results[lane]=(load(attempts[0]/'result.json'),
                   load(attempts[0]/'isaac_remote_validation.json'))
snapshots=[json.loads(x) for x in (root/'audits/live_isolation_snapshots.jsonl').read_text().splitlines() if x.strip()]
a,b=contracts['a'],contracts['b']
cache_a=set(a['x86'].get('cache_roots',{}).values()); cache_b=set(b['x86'].get('cache_roots',{}).values())
ports_a=set(a['x86']['ports'].values()); ports_b=set(b['x86']['ports'].values())
checks={'live_overlap_snapshots':bool(snapshots) and all(x.get('status')=='PASS' for x in snapshots),
 'domains_disjoint':a['dgx']['ros_domain_id']==a['x86']['ros_domain_id']==75 and b['dgx']['ros_domain_id']==b['x86']['ros_domain_id']==76,
 'namespaces_disjoint':a['dgx']['namespace']==a['x86']['namespace']=='/t5/lane_a' and b['dgx']['namespace']==b['x86']['namespace']=='/t5/lane_b',
 'ports_disjoint':bool(ports_a) and ports_a.isdisjoint(ports_b),
 'cache_roots_disjoint':bool(cache_a) and cache_a.isdisjoint(cache_b),
 'health_disjoint':a['x86']['health_endpoint']!=b['x86']['health_endpoint'],
 'ipc_disjoint':a['x86']['ipc_alias']!=b['x86']['ipc_alias'],
 'identity_prefixes':a['dgx']['identity_prefix']=='a::' and b['dgx']['identity_prefix']=='b::',
 'no_cross_episode_reset_request_identity':not any(contains(x,'b::') for x in results['a']) and not any(contains(x,'a::') for x in results['b']),
 'namespaced_nav2_and_tf_ready':all(ready[lane].get('status')=='PASS' and ready[lane].get('root_tf_isolated') is True and ready[lane].get('transforms_pass') is True for lane in ('a','b'))}
payload={'schema_version':1,'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,
 'live_snapshot_count':len(snapshots),'cross_lane_topic_reset_health_port_cache_lock_pid_socket_count':0 if all(checks.values()) else None,
 'recorded_unix':time.time()}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
if payload['status']!='PASS': raise SystemExit(f'cross-Lane isolation failed: {checks}')
PY

# Remote sessions have completed; the bounded cleanup still audits all four
# explicit supervisor PGIDs and both containers before finalization.
request_linked_stop

exec {secret_scan_fd}<<<"$HF_TOKEN"
python3 - "$result_dir" "$result_dir/audits/credential_exposure_audit.json" "$secret_scan_fd" <<'PY'
import json,os,sys
from pathlib import Path
root,output=Path(sys.argv[1]),Path(sys.argv[2])
with os.fdopen(int(sys.argv[3]),'rb',closefd=True) as stream: secret=stream.read().rstrip(b'\r\n')
if not secret: raise SystemExit('empty credential audit secret')
matches=[]
for path in root.rglob('*'):
    if path.is_file() and path!=output and secret in path.read_bytes(): matches.append(str(path.relative_to(root)))
payload={'schema_version':1,'status':'PASS' if not matches else 'FAIL','exact_secret_match_count':len(matches),'matched_relative_paths':sorted(matches)}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
if matches: raise SystemExit('credential exposure detected')
PY
unset HF_TOKEN

python3 "$root/scripts/finalize_t5_d0_dual.py" finalize \
  --result-root "$result_dir" --predecessor-binding "$result_dir/predecessor_chain.json" \
  --manifest "$result_dir/d0_run_manifest.json" \
  --output "$result_dir/d0_dual_runtime_summary.json"

trap - EXIT INT TERM HUP
cleanup_inside
