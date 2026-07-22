#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_t5_d0_lane_online.sh STAGE GRANT_ID AUTHORIZATION_REF PREP_RESULT_ROOT RESULT_ROOT

  STAGE: a (D0.1 Lane A) | b (D0.2 Lane B)

Runs the frozen five-episode InternVLA hardware-reproduction gate for exactly
one complete DGX/Isaac lane.  D0 reports SR/OS/SPL/NE; it does not tune the
bundle and deliberately has no non-zero-SR acceptance threshold.
EOF
  exit 64
}

[[ $# -eq 5 ]] || usage
stage_selector="$1"
grant_id="$2"
authorization_ref="$3"
prep_relative="$4"
result_relative="$5"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
source "$root/scripts/t5_quarantine_common.sh"
source "$root/scripts/t5_remote_compute_audit_common.sh"
board_relative=coordination/T5_DUAL_LANE_BOARD.md
golden_relative=configs/internnav_t5/golden_bundle_manifest.json
d0_manifest_relative=configs/internnav_t5/d0_run_manifest.json
credentials="$root/.env.local"

case "$stage_selector" in
  a)
    lane=a
    stage=d0_1_lane_a_fixed_5
    lane_scope=lane_a
    resource_profile=lane-a
    grant_prefix=t5d01
    result_prefix=d0-1-lane-a
    dgx_target=railgun@10.100.100.128
    dgx_user=railgun
    dgx_ip=10.100.100.128
    ros_domain_id=75
    container=internnav_t5_isaac_a
    other_container=internnav_t5_isaac_b
    gpu=0
    cpuset=0-7,16-23
    identity_prefix='a::'
    other_identity_prefix='b::'
    own_ports='25137 25139 25140 25141'
    other_ports='25138 25239 25240 25241'
    ;;
  b)
    lane=b
    stage=d0_2_lane_b_same_fixed_5
    lane_scope=lane_b
    resource_profile=lane-b
    grant_prefix=t5d02
    result_prefix=d0-2-lane-b
    dgx_target=rail@10.100.120.116
    dgx_user=rail
    dgx_ip=10.100.120.116
    ros_domain_id=76
    container=internnav_t5_isaac_b
    other_container=internnav_t5_isaac_a
    gpu=1
    cpuset=8-15,24-31
    identity_prefix='b::'
    other_identity_prefix='a::'
    own_ports='25138 25239 25240 25241'
    other_ports='25137 25139 25140 25141'
    ;;
  *) usage ;;
esac

[[ "$grant_id" =~ ^${grant_prefix}[0-9]{8}t[0-9]{6}$ ]] || usage
[[ "$authorization_ref" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$prep_relative" =~ ^results/internnav_t5/d0-0-prepare-t5d00[0-9]{8}t[0-9]{6}$ ]] || usage
case "$result_relative" in
  results/internnav_t5/"$result_prefix"-"$grant_id") ;;
  *) echo "result root does not match stage/grant" >&2; exit 64 ;;
esac

# Credentials stay in the ignored local file.  The token is later sent as two
# stdin lines to the DGX remote shell; it is never an argument, log field,
# deployment artifact, result artifact, or remote credential file.
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

# Online authority is one clean NO_GRANT state followed by exactly one
# board-only grant commit.  The separate immutable code ref is read from the
# schema-v2 grant below.  Untracked files are forbidden as well.
"${git_command[@]}" cat-file -e "$authorization_ref^{commit}"
"${git_command[@]}" merge-base --is-ancestor "$authorization_ref" HEAD
[[ "$("${git_command[@]}" rev-list --count "$authorization_ref..HEAD" | tr -d '\r')" = 1 ]]
mapfile -t post_ref_changes < <(
  "${git_command[@]}" diff --name-only "$authorization_ref" HEAD | tr -d '\r'
)
[[ "${#post_ref_changes[@]}" -eq 1 ]]
[[ "${post_ref_changes[0]}" = "$board_relative" ]]
[[ -z "$("${git_command[@]}" status --porcelain --untracked-files=all | tr -d '\r')" ]]

validation_tmp="$(mktemp -d "${TMPDIR:-/tmp}/internnav-t5-d0-lane-validate.XXXXXX")"
cleanup_validation_tmp() { rm -rf -- "$validation_tmp"; }
trap cleanup_validation_tmp EXIT
"${git_command[@]}" show "$authorization_ref:$board_relative" >"$validation_tmp/board.authorization.md"
"${git_command[@]}" show "HEAD:$board_relative" >"$validation_tmp/board.granted.md"

# authorization_ref is the immediately preceding clean NO_GRANT state; it is
# deliberately not the deployed code.  Schema v2 names the immutable D0.0
# code ref and the predecessor receipt independently.
mapfile -t grant_refs < <(python3 - "$validation_tmp/board.granted.md" <<'PY'
import json, re, sys
from pathlib import Path
marker="INTERNAV_T5_DUAL_LANE_ONLINE_GRANT_V1"
matches=re.findall(rf"<!-- {marker}\s*(\{{.*?\}})\s*{marker} -->",
                   Path(sys.argv[1]).read_text(encoding="utf-8"),re.DOTALL)
if len(matches)!=1: raise SystemExit("the authority board must contain exactly one grant block")
grant=json.loads(matches[0])
for key in ("code_ref_sha","predecessor_stage","predecessor_result_root",
            "predecessor_receipt_sha256"):
    value=grant.get(key)
    if not isinstance(value,str) or not value: raise SystemExit(f"missing schema-v2 {key}")
    print(value)
PY
)
code_ref="${grant_refs[0]}"
predecessor_stage="${grant_refs[1]}"
predecessor_relative="${grant_refs[2]}"
predecessor_receipt_sha256="${grant_refs[3]}"
[[ "$code_ref" =~ ^[0-9a-f]{40}$ ]]
[[ "$predecessor_receipt_sha256" =~ ^[0-9a-f]{64}$ ]]
[[ "$predecessor_relative" =~ ^results/internnav_t5/[A-Za-z0-9._/-]+$ ]]
[[ "$predecessor_relative" != *'/../'* && "$predecessor_relative" != */.. ]]
"${git_command[@]}" cat-file -e "$code_ref^{commit}"
"${git_command[@]}" merge-base --is-ancestor "$code_ref" "$authorization_ref"
# Between the immutable deployment commit and each later NO_GRANT signature,
# only the authority board may change. This prevents validation of a newer
# local tree while silently deploying an older code ref under the same Golden
# identity.
mapfile -t code_to_authority_changes < <(
  "${git_command[@]}" diff --name-only "$code_ref" "$authorization_ref" | tr -d '\r'
)
for changed_path in "${code_to_authority_changes[@]}"; do
  [[ "$changed_path" = "$board_relative" ]]
done
"${git_command[@]}" show "$code_ref:$golden_relative" >"$validation_tmp/golden.json"
"${git_command[@]}" show "$code_ref:$d0_manifest_relative" >"$validation_tmp/d0.json"

python3 - "$validation_tmp" "$grant_id" "$authorization_ref" "$stage" \
  "$lane_scope" "$resource_profile" "$result_relative" "$lane" "$code_ref" \
  "$predecessor_stage" "$predecessor_relative" "$predecessor_receipt_sha256" \
  "$prep_relative" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

directory = Path(sys.argv[1])
(grant_id, authorization_ref, stage, lane_scope, resource_profile, result_root,
 lane, code_ref, predecessor_stage, predecessor_root, predecessor_sha,
 prep_root) = sys.argv[2:14]
marker = "INTERNAV_T5_DUAL_LANE_ONLINE_GRANT_V1"
pattern = re.compile(rf"<!-- {marker}\s*(\{{.*?\}})\s*{marker} -->", re.DOTALL)
old_text = (directory / "board.authorization.md").read_text(encoding="utf-8")
new_text = (directory / "board.granted.md").read_text(encoding="utf-8")
old_matches, new_matches = pattern.findall(old_text), pattern.findall(new_text)
if len(old_matches) != 1 or len(new_matches) != 1:
    raise SystemExit("the authority board must contain exactly one grant block")
placeholder = f"<!-- {marker} GRANT_BLOCK {marker} -->"
if pattern.sub(placeholder, old_text) != pattern.sub(placeholder, new_text):
    raise SystemExit("grant commit changed board content outside the grant block")
old_grant = json.loads(old_matches[0])
null_fields=("grant_id","authorization_ref_sha","code_ref_sha","stage","lane_scope",
             "resource_profile","candidate_id","golden_bundle_sha256",
             "run_manifest_sha256","result_root","predecessor_stage",
             "predecessor_result_root","predecessor_receipt_sha256")
clean_no_grant={"schema_version":2,"status":"NO_GRANT",
                **{key:None for key in null_fields}}
if old_grant != clean_no_grant:
    raise SystemExit("authorization ref was not a clean NO_GRANT state")

def canonical(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()

golden = json.loads((directory / "golden.json").read_text(encoding="utf-8"))
d0 = json.loads((directory / "d0.json").read_text(encoding="utf-8"))
golden_sha, d0_sha = canonical(golden), canonical(d0)
if golden.get("status") != "FROZEN_FOR_D0_UNVALIDATED":
    raise SystemExit("D0 requires the unmodified frozen Golden v0")
if d0.get("golden_bundle_canonical_sha256") != golden_sha:
    raise SystemExit("D0 run manifest is not bound to Golden v0")
expected = {
    "schema_version": 2,
    "status": "GRANTED",
    "grant_id": grant_id,
    "authorization_ref_sha": authorization_ref,
    "code_ref_sha": code_ref,
    "stage": stage,
    "lane_scope": lane_scope,
    "resource_profile": resource_profile,
    "candidate_id": golden["bundle_id"],
    "golden_bundle_sha256": golden_sha,
    "run_manifest_sha256": d0_sha,
    "result_root": result_root,
    "predecessor_stage": predecessor_stage,
    "predecessor_result_root": predecessor_root,
    "predecessor_receipt_sha256": predecessor_sha,
}
grant = json.loads(new_matches[0])
if grant != expected:
    raise SystemExit(f"D0 single-lane grant mismatch: {grant!r}")
if lane == "a":
    if predecessor_stage != "d0_0_online_prepare" or predecessor_root != prep_root:
        raise SystemExit("D0.1 must bind the supplied D0.0 preparation root")
else:
    if predecessor_stage != "d0_1_lane_a_fixed_5" or re.fullmatch(
        r"results/internnav_t5/d0-1-lane-a-t5d01[0-9]{8}t[0-9]{6}",
        predecessor_root,
    ) is None:
        raise SystemExit("D0.2 must bind a D0.1 Lane A result root")
fixed, lanes = d0.get("fixed_input", {}), d0.get("lanes", {})
if fixed.get("episode_count") != 5 or len(fixed.get("episode_keys", [])) != 5:
    raise SystemExit("D0 requires exactly the frozen five episodes")
if set(lanes) != {"a", "b"} or lanes[lane].get("resource_profile") != resource_profile:
    raise SystemExit("D0 lane/profile contract drifted")
validation = {
    "schema_version": 1, "status": "PASS", "grant": grant,
    "code_ref_sha": code_ref,
    "golden_bundle_id": golden["bundle_id"],
    "golden_bundle_canonical_sha256": golden_sha,
    "run_manifest_canonical_sha256": d0_sha,
    "dataset_root": fixed["dataset_remote_path"],
    "dataset_file_sha256": fixed["dataset_file_sha256"],
    "episode_keys": fixed["episode_keys"], "lane_contract": lanes[lane],
}
(directory / "grant_validation.json").write_text(
    json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

prep_dir="$root/$prep_relative"
predecessor_dir="$root/$predecessor_relative"
result_dir="$root/$result_relative"
test -d "$prep_dir"
test ! -L "$prep_dir"
test -f "$prep_dir/d0_prepare_summary.json"
test -f "$prep_dir/lease_release_summary.json"
test -f "$prep_dir/d0_prepare_final_summary.json"
test -f "$prep_dir/remote/x86/dataset_audit.json"
test -d "$predecessor_dir"
test ! -L "$predecessor_dir"
if test "$lane" = a; then
  predecessor_receipt="$predecessor_dir/d0_prepare_final_summary.json"
else
  predecessor_receipt="$predecessor_dir/d0_lane_summary.json"
fi
test -f "$predecessor_receipt"
test ! -L "$predecessor_receipt"
test "$(sha256sum "$predecessor_receipt" | cut -d' ' -f1)" = "$predecessor_receipt_sha256"

python3 - "$validation_tmp/grant_validation.json" "$predecessor_receipt" \
  "$lane" "$validation_tmp/predecessor_binding.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
grant=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
receipt_path=Path(sys.argv[2]); lane=sys.argv[3]; output=Path(sys.argv[4])
receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
g=grant['grant']
if lane=='a':
    runtime_path=receipt_path.parent/str(receipt.get('preparation_summary',''))
    release_path=receipt_path.parent/str(receipt.get('lease_release_summary',''))
    runtime=json.loads(runtime_path.read_text(encoding='utf-8')) if runtime_path.is_file() else None
    release=json.loads(release_path.read_text(encoding='utf-8')) if release_path.is_file() else None
else:
    runtime=receipt.get('runtime_summary')
    release=receipt.get('lease_release')
checks={
 'receipt_sha':hashlib.sha256(receipt_path.read_bytes()).hexdigest()==g['predecessor_receipt_sha256'],
 'receipt_status':receipt.get('status')=='PASS',
 'receipt_checks':bool(receipt.get('checks')) and all(receipt['checks'].values()),
 'runtime_present':isinstance(runtime,dict) and runtime.get('status')=='PASS',
 'lease_release':isinstance(release,dict) and release.get('status')=='PASS'
    and bool(release.get('checks')) and all(release['checks'].values()),
}
if lane=='a':
    checks.update({
      'runtime_reference_sha':runtime_path.is_file() and hashlib.sha256(runtime_path.read_bytes()).hexdigest()
          ==receipt.get('preparation_summary_sha256'),
      'lease_reference_sha':release_path.is_file() and hashlib.sha256(release_path.read_bytes()).hexdigest()
          ==receipt.get('lease_release_summary_sha256'),
    })
if isinstance(runtime,dict):
    checks.update({
      'same_code_ref':runtime.get('code_ref_sha')==g['code_ref_sha'],
      'same_golden':runtime.get('golden_bundle_canonical_sha256')==g['golden_bundle_sha256'],
      'same_run_manifest':runtime.get('run_manifest_canonical_sha256')==g['run_manifest_sha256'],
      'expected_stage':runtime.get('stage')==(
          'd0_0_online_prepare' if lane=='a' else 'd0_1_lane_a_fixed_5'),
      'expected_lane':lane=='a' or runtime.get('lane')=='a',
    })
payload={'schema_version':1,'status':'PASS' if all(checks.values()) else 'FAIL',
         'predecessor_stage':g['predecessor_stage'],
         'predecessor_result_root':g['predecessor_result_root'],
         'predecessor_receipt_sha256':g['predecessor_receipt_sha256'],
         'checks':checks}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
if payload['status']!='PASS': raise SystemExit(f'predecessor receipt failed: {checks}')
PY

# Bind the lane grant to a successful D0.0 receipt created from exactly this
# immutable code ref, Golden Bundle and D0 manifest.  D0.0 provides distinct
# x86_a/x86_b deployment roots so the persistent lane-owner marker is never
# shared between A and B.
python3 - "$validation_tmp/grant_validation.json" \
  "$prep_dir/d0_prepare_summary.json" "$prep_dir/lease_release_summary.json" \
  "$prep_dir/remote/x86/dataset_audit.json" "$prep_dir/d0_prepare_final_summary.json" \
  "$lane" "$validation_tmp/prep_binding.json" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

grant = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
prep = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
release = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
dataset = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
prep_final = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
lane, output = sys.argv[6], Path(sys.argv[7])
g = grant["grant"]
checks = {
    "prep_status": prep.get("status") == "PASS",
    "prep_stage": prep.get("stage") == "d0_0_online_prepare",
    "formal_authorization_mode": prep.get("authorization_mode")
        == "FORMAL_BOARD_GRANT_V2"
        and prep_final.get("authorization_mode") == "FORMAL_BOARD_GRANT_V2",
    "formal_board_grant_used": prep.get("board_grant_used") is True
        and prep_final.get("board_grant_used") is True,
    "formal_authorization_binding":
        re.fullmatch(r"t5d00[0-9]{8}t[0-9]{6}", str(prep.get("grant_id", "")))
            is not None
        and prep_final.get("grant_id") == prep.get("grant_id")
        and re.fullmatch(r"[0-9a-f]{40}", str(prep.get("authorization_ref_sha", "")))
            is not None
        and prep_final.get("authorization_ref_sha")
            == prep.get("authorization_ref_sha")
        and prep_final.get("code_ref_sha") == prep.get("code_ref_sha"),
    "same_code_ref": prep.get("code_ref_sha") == g["code_ref_sha"],
    "same_golden": prep.get("golden_bundle_canonical_sha256") == g["golden_bundle_sha256"],
    "same_run_manifest": prep.get("run_manifest_canonical_sha256") == g["run_manifest_sha256"],
    "same_bundle": prep.get("golden_bundle_id") == grant["golden_bundle_id"],
    "prep_checks": bool(prep.get("checks")) and all(prep["checks"].values()),
    "prepare_release": release.get("status") == "PASS"
        and int(release.get("command_exit", -1)) == 0
        and bool(release.get("checks")) and all(release["checks"].values()),
    "prepare_final": prep_final.get("status") == "PASS"
        and bool(prep_final.get("checks")) and all(prep_final["checks"].values()),
    "dataset_status": dataset.get("status") == "PASS",
    "dataset_sha": dataset.get("dataset_sha256") == grant["dataset_file_sha256"],
    "dataset_count": dataset.get("episode_count") == 5,
    "dataset_keys": dataset.get("episode_keys") == grant["episode_keys"],
}
execution = prep.get("execution", {})
checks["prepare_only"] = (
    execution.get("internvla_model_loaded") is False
    and execution.get("isaac_sim_started") is False
    and execution.get("episode_or_evaluator_started") is False
)
roots = prep.get("deployment_roots", {})
required = {"dgx_a", "dgx_b", "x86_a", "x86_b"}
checks["lane_roots_present"] = required.issubset(roots)
selected = {
    "dgx": roots.get("dgx_a" if lane == "a" else "dgx_b", ""),
    "x86": roots.get("x86_a" if lane == "a" else "x86_b", ""),
}
expected_prefix = {
    "dgx": "/home/railgun/internnav-t1-t2/.t5-deployments/" if lane == "a"
           else "/home/rail/internnav-t1-t2/.t5-deployments/",
    "x86": "/home/song/internnav-t1-t2/.t5-deployments/",
}
safe = re.compile(r"^/[A-Za-z0-9._/-]+$")
checks["lane_roots_safe"] = all(
    safe.fullmatch(value) is not None and value.startswith(expected_prefix[key])
    and "/../" not in value + "/" for key, value in selected.items()
)
dataset_file = str(dataset.get("dataset_file", ""))
checks["dataset_root"] = dataset_file == grant["dataset_root"].rstrip("/") + "/val_unseen/val_unseen.json.gz"
map_sha = prep.get("static_map_manifest_sha256", "")
checks["map_receipt"] = re.fullmatch(r"[0-9a-f]{64}", map_sha) is not None
checks["distinct_x86_lane_roots"] = roots.get("x86_a") != roots.get("x86_b")
payload = {
    "schema_version": 1, "status": "PASS" if all(checks.values()) else "FAIL",
    "checks": checks, "prep_grant_id": prep.get("grant_id"),
    "authorization_ref_sha": g["authorization_ref_sha"],
    "code_ref_sha": g["code_ref_sha"],
    "golden_bundle_canonical_sha256": g["golden_bundle_sha256"],
    "run_manifest_canonical_sha256": g["run_manifest_sha256"],
    "deployment_roots": selected, "dataset_root": grant["dataset_root"],
    "dataset_file_sha256": grant["dataset_file_sha256"],
    "static_map_manifest_sha256": map_sha,
    "source_receipts": {
        "prepare_summary_sha256": hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest(),
        "prepare_release_sha256": hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest(),
        "dataset_audit_sha256": hashlib.sha256(Path(sys.argv[4]).read_bytes()).hexdigest(),
        "prepare_final_sha256": hashlib.sha256(Path(sys.argv[5]).read_bytes()).hexdigest(),
    },
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if payload["status"] != "PASS":
    raise SystemExit(f"D0.0 preparation receipt mismatch: {checks}")
PY

python3 - "$lane" "$validation_tmp/prep_binding.json" "$predecessor_receipt" \
  "$predecessor_dir/prep_binding.json" \
  "$validation_tmp/preparation_chain_binding.json" <<'PY'
import hashlib, json, sys
from pathlib import Path

lane = sys.argv[1]
current_path, predecessor_receipt_path, predecessor_prep_path, output = map(
    Path, sys.argv[2:6]
)
current = json.loads(current_path.read_text(encoding='utf-8'))
checks = {
    'current_preparation_binding_pass': current.get('status') == 'PASS'
        and bool(current.get('checks')) and all(current['checks'].values()),
}
payload = {
    'schema_version': 1, 'lane': lane,
    'current_preparation_binding_sha256': hashlib.sha256(
        current_path.read_bytes()
    ).hexdigest(),
}
if lane == 'a':
    checks['direct_d0_0_predecessor'] = predecessor_receipt_path.name == (
        'd0_prepare_final_summary.json'
    )
else:
    predecessor_receipt = json.loads(
        predecessor_receipt_path.read_text(encoding='utf-8')
    )
    predecessor_runtime = predecessor_receipt.get('runtime_summary', {})
    predecessor_prep = json.loads(
        predecessor_prep_path.read_text(encoding='utf-8')
    ) if predecessor_prep_path.is_file() else None
    predecessor_prep_sha = (
        hashlib.sha256(predecessor_prep_path.read_bytes()).hexdigest()
        if predecessor_prep_path.is_file() else None
    )
    checks.update({
        'predecessor_prep_binding_exists': predecessor_prep is not None,
        'predecessor_runtime_binds_prep_file': predecessor_prep_sha
            == predecessor_runtime.get('prep_binding_sha256'),
        'same_d0_0_prep_grant': predecessor_prep is not None
            and predecessor_prep.get('prep_grant_id') == current.get('prep_grant_id'),
        'same_d0_0_source_receipts': predecessor_prep is not None
            and predecessor_prep.get('source_receipts') == current.get('source_receipts'),
        'same_code_golden_manifest': predecessor_prep is not None and all(
            predecessor_prep.get(key) == current.get(key) for key in (
                'code_ref_sha', 'golden_bundle_canonical_sha256',
                'run_manifest_canonical_sha256', 'dataset_file_sha256',
                'static_map_manifest_sha256',
            )
        ),
    })
    payload['predecessor_preparation_binding_sha256'] = predecessor_prep_sha
payload['checks'] = checks
payload['status'] = 'PASS' if all(checks.values()) else 'FAIL'
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
if payload['status'] != 'PASS':
    raise SystemExit(f'D0 preparation chain mismatch: {checks}')
PY

mapfile -t binding_values < <(python3 - "$validation_tmp/prep_binding.json" <<'PY'
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
for item in (
    value["deployment_roots"]["dgx"], value["deployment_roots"]["x86"],
    value["dataset_root"], value["dataset_file_sha256"],
    value["static_map_manifest_sha256"], value["golden_bundle_canonical_sha256"],
    value["run_manifest_canonical_sha256"],
): print(item)
PY
)
dgx_root="${binding_values[0]}"
x86_root="${binding_values[1]}"
dataset_root="${binding_values[2]}"
dataset_sha256="${binding_values[3]}"
map_manifest_sha256="${binding_values[4]}"
golden_sha256="${binding_values[5]}"
run_manifest_sha256="${binding_values[6]}"
map_manifest="$dgx_root/inputs/d0_fixed5_static_maps/manifest.json"
dgx_run="$dgx_root/results/${stage}-${grant_id}"
x86_run="$x86_root/results/${stage}-${grant_id}"

# Acquire only this Lane's two physical locks before atomically consuming the
# local result root.  Lease logs are staged outside it and copied after the
# remote holders have actually released their flocks.
if [[ "${INTERNNAV_T5_INSIDE_D0_LANE:-0}" != 1 ]]; then
  test ! -e "$result_dir"
  mkdir -p "$root/results/internnav_t5"
  lease_bootstrap="$(mktemp -d "$root/results/internnav_t5/.${stage}-lease-${grant_id}.XXXXXX")"
  trap - EXIT
  cleanup_validation_tmp
  set +e
  bash "$root/scripts/with_resource_lease.sh" "$resource_profile" \
    --owner codex-00 \
    --task "${stage}:${grant_id}:${authorization_ref}" \
    --log-dir "$lease_bootstrap" --acquire-timeout 30 \
    --cleanup-timeout 600 --kill-wait-timeout 60 -- \
    env INTERNNAV_T5_INSIDE_D0_LANE=1 \
      INTERNNAV_T5_RESOURCE_LEASE_ACK="$resource_profile" \
      bash "$root/coordination/run_t5_d0_lane_online.sh" \
        "$stage_selector" "$grant_id" "$authorization_ref" \
        "$prep_relative" "$result_relative"
  lane_rc=$?
  set -e
  if test -d "$result_dir"; then
    install -d -m 700 "$result_dir/lease"
    cp -a -- "$lease_bootstrap/." "$result_dir/lease/"
    python3 - "$result_dir/lease_release_summary.json" \
      "$result_dir/lease/lease_metadata.txt" "$lane_rc" "$resource_profile" <<'PY'
import json, sys, time
from pathlib import Path
output, metadata_path = Path(sys.argv[1]), Path(sys.argv[2])
command_rc, profile = int(sys.argv[3]), sys.argv[4]
text = metadata_path.read_text(encoding="utf-8") if metadata_path.is_file() else ""
holders = sorted(metadata_path.parent.glob("holder_*.stdout.log"))
cleanup_path=metadata_path.parent/"lease_cleanup_receipt.json"
cleanup=json.loads(cleanup_path.read_text(encoding="utf-8")) if cleanup_path.is_file() else None
expected = ("dgx_a", "isaac_gpu0") if profile == "lane-a" else ("dgx_b", "isaac_gpu1")
checks = {
    "two_holders_recorded": len(holders) == 2,
    "exact_resources_named": all(name in text for name in expected)
        and all(name not in text for name in ({"dgx_a","dgx_b","isaac_gpu0","isaac_gpu1"} - set(expected))),
    "wrapped_command_released": "state=RELEASED" in text,
    "wrapped_command_exit_recorded": f"command_exit={command_rc}" in text,
    "wrapped_group_absent_before_release": isinstance(cleanup,dict)
        and cleanup.get("status")=="PASS"
        and cleanup.get("wrapped_process_group_absent_before_lock_release") is True,
}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
         "resource_profile":profile,"command_exit":command_rc,"checks":checks,
         "holder_logs":[p.name for p in holders],"recorded_unix":time.time()}
with output.open("x", encoding="utf-8", newline="\n") as stream:
    stream.write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
    python3 - "$result_dir/d0_lane_runtime_summary.json" \
      "$result_dir/lease_release_summary.json" "$result_dir/d0_lane_summary.json" "$lane_rc" <<'PY'
import json, sys, time
from pathlib import Path
runtime_path, release_path, output = map(Path, sys.argv[1:4])
command_rc = int(sys.argv[4])
runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else None
release = json.loads(release_path.read_text(encoding="utf-8"))
coordinator_cleanup_path=runtime_path.parent/'audits/coordinator_cleanup_receipt.json'
coordinator_cleanup=(json.loads(coordinator_cleanup_path.read_text(encoding='utf-8'))
                     if coordinator_cleanup_path.is_file() else None)
checks={"runtime_pass":runtime is not None and runtime.get("status")=="PASS",
        "lease_release_pass":release.get("status")=="PASS",
        "coordinator_cleanup_pass":isinstance(coordinator_cleanup,dict)
            and coordinator_cleanup.get("status")=="PASS",
        "command_exit_zero":command_rc==0}
payload={"schema_version":1,"status":"PASS" if all(checks.values()) else "FAIL",
         "checks":checks,"runtime_summary":runtime,"lease_release":release,
         "coordinator_cleanup":coordinator_cleanup,
         "recorded_unix":time.time()}
with output.open("x",encoding="utf-8",newline="\n") as stream:
    stream.write(json.dumps(payload,indent=2,sort_keys=True)+"\n")
PY
    rm -rf -- "$lease_bootstrap"
    final_status="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$result_dir/d0_lane_summary.json")"
    test "$final_status" = PASS || lane_rc=1
  else
    printf 'D0 Lane did not consume result_root; lease diagnostics remain at %s\n' "$lease_bootstrap" >&2
  fi
  exit "$lane_rc"
fi

test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = "$resource_profile"
test ! -e "$result_dir"
umask 077
mkdir "$result_dir"
mkdir "$result_dir/logs" "$result_dir/remote" "$result_dir/audits"
cp "$validation_tmp/grant_validation.json" "$result_dir/grant_validation.json"
cp "$validation_tmp/prep_binding.json" "$result_dir/prep_binding.json"
cp "$validation_tmp/predecessor_binding.json" "$result_dir/predecessor_binding.json"
cp "$validation_tmp/preparation_chain_binding.json" \
  "$result_dir/preparation_chain_binding.json"
trap - EXIT
cleanup_validation_tmp

ssh_options=(-T -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=5 -o ServerAliveCountMax=2)
x86_target=song@10.100.120.111
dgx_ssh_pid=""
x86_ssh_pid=""
dgx_launch_attempted=0
x86_launch_attempted=0
container_started=0
stop_requested=0
dgx_quarantine_file=/tmp/internnav_dgx.quarantine
isaac_lane_quarantine_file="/tmp/internnav_isaac_gpu${gpu}.quarantine"
dgx_quarantine_armed=false
x86_quarantine_armed=false
quarantine_run_tag="${stage}:${grant_id}"
dgx_quarantine_role="dgx_${lane}"
x86_quarantine_role="x86_gpu${gpu}"
dgx_quarantine_roots="$dgx_root"
x86_quarantine_roots="$x86_root"
dgx_supervisor_ledger="${dgx_run}.supervisor.json"
x86_supervisor_ledger="${x86_run}.supervisor.json"
cleanup_timeout="${INTERNNAV_T5_D0_CLEANUP_TIMEOUT_SEC:-360}"
cleanup_kill_timeout="${INTERNNAV_T5_D0_CLEANUP_KILL_TIMEOUT_SEC:-30}"
[[ "$cleanup_timeout" =~ ^[1-9][0-9]*$ ]]
[[ "$cleanup_kill_timeout" =~ ^[1-9][0-9]*$ ]]

remote() {
  local target="$1"; shift
  ssh "${ssh_options[@]}" "$target" "$@"
}

assert_other_lane_quiet() {
  local port_tests="" port
  for port in $other_ports; do
    port_tests+="; test -z \"\$(ss -H -lntup | grep -E '[:.]$port[[:space:]]' || true)\""
  done
  remote "$x86_target" \
    "test \"\$(docker inspect -f '{{.State.Running}}' '$other_container')\" = false$port_tests"
}

read -r -d '' supervisor_control_program <<'REMOTE_SUPERVISOR_CONTROL' || true
import json, os, signal, sys, time
from pathlib import Path

ledger_path, expected_root, action = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
value = json.loads(ledger_path.read_text(encoding="utf-8"))
if value.get("run_root") != expected_root:
    raise SystemExit("supervisor ledger run_root mismatch")
pid, pgid = int(value["pid"]), int(value["pgid"])
if pid <= 1 or pgid <= 1:
    raise SystemExit("unsafe supervisor identity")
ancestors=set()
cursor=os.getpid()
while cursor > 1 and cursor not in ancestors:
    ancestors.add(cursor)
    try:
        status=Path(f"/proc/{cursor}/status").read_text(encoding="utf-8")
        cursor=int(next(line.split()[1] for line in status.splitlines()
                        if line.startswith("PPid:")))
    except (FileNotFoundError, PermissionError, StopIteration, ValueError):
        break
process_table=[]
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    candidate=int(entry.name)
    if candidate in ancestors:
        continue
    try:
        candidate_pgid=os.getpgid(candidate)
        command=(entry/"cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        process_table.append({"pid": candidate, "pgid":candidate_pgid, "command": command})
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
members=[row for row in process_table if row["pgid"]==pgid]
run_processes=[row for row in process_table if expected_root in row["command"]]
ledger_pgids=set()
runtime_ledger=Path(expected_root)/"pid_ledger.jsonl"
runtime_ledger_error=None
if runtime_ledger.is_file():
    try:
        rows=[json.loads(line) for line in runtime_ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
        absent_ids={(row.get("scope","host"),row.get("component"),row.get("pid"))
                    for row in rows if row.get("event")=="verified_absent"}
        for row in rows:
            key=(row.get("scope","host"),row.get("component"),row.get("pid"))
            if row.get("event")=="started" and key not in absent_ids and row.get("scope","host")=="host":
                candidate_pgid=row.get("pgid")
                if isinstance(candidate_pgid,int) and candidate_pgid>1:
                    ledger_pgids.add(candidate_pgid)
    except (OSError,ValueError,TypeError) as error:
        runtime_ledger_error=type(error).__name__
managed_groups={candidate:[row for row in process_table if row["pgid"]==candidate]
                for candidate in sorted(ledger_pgids)}
associated=any(expected_root in row["command"] for row in members)
signalled=False
if action in {"TERM", "KILL"} and members:
    if not associated:
        raise SystemExit("refusing to signal a reused/unassociated process group")
    os.killpg(pgid, signal.SIGTERM if action == "TERM" else signal.SIGKILL)
    signalled=True
if action in {"TERM", "KILL"} and (action == "KILL" or not members):
    # If the supervisor trap exceeded its budget, its runtime ledger is the
    # authority for child sessions created with setsid.  If the supervisor is
    # already absent, give those children TERM directly instead of waiting the
    # full graceful window.  Never signal a reused PGID unless at least one
    # current member is still tied to this run root.
    direct_groups=set(managed_groups)
    direct_groups.update(row["pgid"] for row in run_processes)
    for candidate in sorted(direct_groups):
        rows=[row for row in process_table if row["pgid"]==candidate]
        if not rows or candidate == pgid:
            continue
        if not any(expected_root in row["command"] for row in rows):
            raise SystemExit(f"refusing to signal unassociated managed PGID {candidate}")
        os.killpg(candidate, signal.SIGTERM if action == "TERM" else signal.SIGKILL)
        signalled=True
payload={
    "schema_version":1, "ledger":str(ledger_path), "run_root":expected_root,
    "action":action, "pid":pid, "pgid":pgid, "member_count":len(members),
    "members":members, "run_processes":run_processes,
    "managed_groups":managed_groups, "associated":associated, "signalled":signalled,
    "runtime_ledger_error":runtime_ledger_error,
    "absent":runtime_ledger_error is None and not members and not run_processes
        and not any(managed_groups.values()),
    "recorded_unix":time.time(),
}
print(json.dumps(payload, sort_keys=True))
REMOTE_SUPERVISOR_CONTROL
supervisor_control_b64="$(printf '%s' "$supervisor_control_program" | base64 | tr -d '\r\n')"

supervisor_action() {
  local target="$1" ledger="$2" run_root="$3" action="$4" output
  output="$(remote "$target" \
    "python3 -c \"\$(printf '%s' '$supervisor_control_b64' | base64 -d)\" '$ledger' '$run_root' '$action'" 2>&1)"
  local rc=$?
  printf '%s\n' "$output" >>"$result_dir/audits/coordinator_cleanup_events.jsonl"
  printf '%s\n' "$output"
  return "$rc"
}

supervisor_absent() {
  local output
  output="$(supervisor_action "$1" "$2" "$3" AUDIT)" || return 1
  python3 -c 'import json,sys; raise SystemExit(0 if json.loads(sys.argv[1]).get("absent") else 1)' "$output"
}

run_root_processes_absent() {
  local target="$1" root_scopes="$2"
  remote "$target" "python3 - '$root_scopes'" <<'PY'
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
  remote "$dgx_target" "test ! -d '$dgx_run' || touch '$dgx_run/stop.request'" >/dev/null 2>&1
  remote "$x86_target" "test ! -d '$x86_run' || touch '$x86_run/stop.request'" >/dev/null 2>&1
  remote "$dgx_target" "cat '$dgx_supervisor_ledger'" \
    >"$result_dir/audits/dgx_supervisor_ledger.json" 2>/dev/null || true
  remote "$x86_target" "cat '$x86_supervisor_ledger'" \
    >"$result_dir/audits/x86_supervisor_ledger.json" 2>/dev/null || true

  # The SSH clients are transport only.  Signal the explicit remote sessions
  # first, leaving the SSH connections alive so their cleanup traps can drain
  # children, write ledgers and close sockets before the physical locks move.
  test ! -f "$result_dir/audits/coordinator_cleanup_events.jsonl" || true
  for specification in \
    "$dgx_target|$dgx_supervisor_ledger|$dgx_run" \
    "$x86_target|$x86_supervisor_ledger|$x86_run"; do
    IFS='|' read -r target ledger run_root <<<"$specification"
    remote "$target" "test -f '$ledger'" >/dev/null 2>&1 || continue
    supervisor_action "$target" "$ledger" "$run_root" TERM >/dev/null 2>&1 || true
  done
  deadline=$((SECONDS + cleanup_timeout))
  while (( SECONDS < deadline )); do
    residual=0
    for specification in \
      "$dgx_target|$dgx_supervisor_ledger|$dgx_run" \
      "$x86_target|$x86_supervisor_ledger|$x86_run"; do
      IFS='|' read -r target ledger run_root <<<"$specification"
      remote "$target" "test -f '$ledger'" >/dev/null 2>&1 || continue
      supervisor_absent "$target" "$ledger" "$run_root" >/dev/null 2>&1 || residual=$((residual + 1))
    done
    test "$residual" = 0 && break
    sleep 1
  done
  for specification in \
    "$dgx_target|$dgx_supervisor_ledger|$dgx_run" \
    "$x86_target|$x86_supervisor_ledger|$x86_run"; do
    IFS='|' read -r target ledger run_root <<<"$specification"
    remote "$target" "test -f '$ledger'" >/dev/null 2>&1 || continue
    supervisor_absent "$target" "$ledger" "$run_root" >/dev/null 2>&1 || \
      supervisor_action "$target" "$ledger" "$run_root" KILL >/dev/null 2>&1 || true
  done
  kill_deadline=$((SECONDS + cleanup_kill_timeout))
  while (( SECONDS < kill_deadline )); do
    residual=0
    for specification in \
      "$dgx_target|$dgx_supervisor_ledger|$dgx_run" \
      "$x86_target|$x86_supervisor_ledger|$x86_run"; do
      IFS='|' read -r target ledger run_root <<<"$specification"
      remote "$target" "test -f '$ledger'" >/dev/null 2>&1 || continue
      supervisor_absent "$target" "$ledger" "$run_root" >/dev/null 2>&1 || residual=$((residual + 1))
    done
    test "$residual" = 0 && break
    sleep 1
  done

  # Only after both remote PGIDs are absent may the local transports be
  # reaped.  A stuck SSH process cannot prolong lock ownership indefinitely.
  for pid in "$x86_ssh_pid" "$dgx_ssh_pid"; do
    test -z "$pid" || ! kill -0 "$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  done
  if test "$container_started" != 0; then
    remote "$x86_target" \
      "set +e; test \"\$(docker inspect -f '{{index .Config.Labels \"internnav.t5.deployment_root\"}}' '$container')\" = '$x86_root' || exit 75; docker stop -t 15 '$container' >/dev/null 2>&1; if test \"\$(docker inspect -f '{{.State.Running}}' '$container' 2>/dev/null)\" = true; then docker kill '$container' >/dev/null 2>&1; fi; test \"\$(docker inspect -f '{{.State.Running}}' '$container')\" = false; test \"\$(docker inspect -f '{{.State.Pid}}' '$container')\" = 0" \
      >/dev/null 2>&1 || true
    container_started=0
  fi
  dgx_absent=false
  x86_absent=false
  if remote "$dgx_target" "test -f '$dgx_supervisor_ledger'" >/dev/null 2>&1; then
    supervisor_absent "$dgx_target" "$dgx_supervisor_ledger" "$dgx_run" >/dev/null 2>&1 && dgx_absent=true
  elif test "$dgx_launch_attempted" = 0; then
    run_root_processes_absent "$dgx_target" "$dgx_run|$dgx_root" >/dev/null 2>&1 && dgx_absent=true
  fi
  if remote "$x86_target" "test -f '$x86_supervisor_ledger'" >/dev/null 2>&1; then
    supervisor_absent "$x86_target" "$x86_supervisor_ledger" "$x86_run" >/dev/null 2>&1 && x86_absent=true
  elif test "$x86_launch_attempted" = 0; then
    run_root_processes_absent "$x86_target" "$x86_run|$x86_root" >/dev/null 2>&1 && x86_absent=true
  fi
  container_absent=false dgx_endpoints_clean=false x86_host_clean=false
  dgx_structured_compute_absent=false
  remote "$x86_target" \
    "test \"\$(docker inspect -f '{{index .Config.Labels \"internnav.t5.deployment_root\"}}' '$container')\" = '$x86_root'; test \"\$(docker inspect -f '{{.State.Running}}' '$container')\" = false; test \"\$(docker inspect -f '{{.State.Pid}}' '$container')\" = 0" \
    >/dev/null 2>&1 && container_absent=true
  remote "$dgx_target" \
    "for p in 25137 25138 25139 25140 25141 25239 25240 25241; do test -z \"\$(ss -H -lntup|grep -E \"[:.]\$p[[:space:]]\"||true)\" || exit 1; done" \
    >/dev/null 2>&1 && dgx_endpoints_clean=true
  t5_remote_compute_absent "$dgx_target" \
    "$result_dir/audits/dgx_structured_compute_poststop.json" \
    >/dev/null 2>&1 && dgx_structured_compute_absent=true
  remote "$x86_target" \
    "set -e; for c in internnav_t5_isaac_a internnav_t5_isaac_b; do test \"\$(docker inspect -f '{{.State.Running}}' \"\$c\")\" = false; test \"\$(docker inspect -f '{{.State.Pid}}' \"\$c\")\" = 0; done; test ! -e /tmp/internnav_t5_a_ipc/isaac_health.sock; test ! -e /tmp/internnav_t5_b_ipc/isaac_health.sock; for lock in /tmp/internnav_t5_isaac_a_runtime.lock /tmp/internnav_t5_isaac_b_runtime.lock /tmp/internnav_t5_isaac_shared_assets.lock; do test ! -e \"\$lock\" || flock -n \"\$lock\" true; done; for p in 25137 25138 25139 25140 25141 25239 25240 25241; do test -z \"\$(ss -H -lntup|grep -E \"[:.]\$p[[:space:]]\"||true)\" || exit 1; done" \
    >/dev/null 2>&1 && x86_host_clean=true
  cleanup_checks="$result_dir/audits/coordinator_cleanup_receipt.json"
  python3 - "$cleanup_checks" "$dgx_absent" "$x86_absent" "$container_absent" \
    "$dgx_endpoints_clean" "$x86_host_clean" "$dgx_structured_compute_absent" \
    "$cleanup_timeout" "$cleanup_kill_timeout" <<'PY'
import json, sys, time
from pathlib import Path
output=Path(sys.argv[1])
checks={'dgx_supervisor_pgid_absent':sys.argv[2]=='true',
        'x86_supervisor_pgid_absent':sys.argv[3]=='true',
        'isaac_container_stopped_and_owned':sys.argv[4]=='true',
        'dgx_t5_ports_absent':sys.argv[5]=='true',
        'x86_containers_sockets_ports_runtime_locks_clean':sys.argv[6]=='true',
        'dgx_structured_compute_absent':sys.argv[7]=='true'}
payload={'schema_version':1,'status':'PASS' if all(checks.values()) else 'FAIL',
 'checks':checks,'graceful_timeout_sec':int(sys.argv[8]),
 'kill_wait_timeout_sec':int(sys.argv[9]),'recorded_unix':time.time()}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
raise SystemExit(0 if payload['status']=='PASS' else 1)
PY
  cleanup_rc=$?
  set -e
  return "$cleanup_rc"
}

capture_container_inspect() {
  local phase="$1"
  local raw="$result_dir/audits/container_${phase}_inspect.json"
  remote "$x86_target" "docker inspect '$container'" >"$raw"
  python3 - "$raw" "$result_dir/audits/container_${phase}.json" "$phase" \
    "$container" "$lane" "$gpu" "$cpuset" <<'PY'
import json, sys, time
from pathlib import Path
raw, output, phase, expected_name, lane, gpu, cpuset = sys.argv[1:]
value=json.loads(Path(raw).read_text(encoding='utf-8'))[0]
checks={
    'name':value.get('Name','').lstrip('/')==expected_name,
    'lane_label':value.get('Config',{}).get('Labels',{}).get('internnav.t5.lane')==lane,
    'gpu_label':value.get('Config',{}).get('Labels',{}).get('internnav.t5.gpu')==gpu,
    'cpuset':value.get('HostConfig',{}).get('CpusetCpus')==cpuset,
    'stopped':value.get('State',{}).get('Running') is False,
    'pid_zero':int(value.get('State',{}).get('Pid',-1))==0,
}
payload={'schema_version':1,'status':'PASS' if all(checks.values()) else 'FAIL',
         'phase':phase,'container':expected_name,'container_id':value.get('Id'),
         'running':value.get('State',{}).get('Running'),'pid':value.get('State',{}).get('Pid'),
         'checks':checks,'recorded_unix':time.time()}
Path(output).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
if payload['status']!='PASS': raise SystemExit(f'container {phase} audit failed: {checks}')
PY
}

cleanup_inside() {
  local incoming=$?
  trap - EXIT INT TERM HUP
  request_linked_stop || incoming=1
  set +e
  cleanup_receipt="$result_dir/audits/coordinator_cleanup_receipt.json"
  if test "$dgx_quarantine_armed" = true; then
    t5_quarantine_owned_clear "$dgx_target" "$dgx_quarantine_file" \
      "$dgx_quarantine_role" "$quarantine_run_tag" "$dgx_quarantine_roots" \
      "$cleanup_receipt" "$result_dir/audits/dgx_quarantine_clear.json" || incoming=1
  fi
  if test "$x86_quarantine_armed" = true; then
    t5_quarantine_owned_clear "$x86_target" "$isaac_lane_quarantine_file" \
      "$x86_quarantine_role" \
      "$quarantine_run_tag" "$x86_quarantine_roots" "$cleanup_receipt" \
      "$result_dir/audits/x86_quarantine_clear.json" || incoming=1
  fi
  wait 2>/dev/null || true
  if test "$incoming" != 0; then
    capture_container_inspect failure_cleanup >/dev/null 2>&1 || true
    # Preserve partial remote evidence after linked shutdown.  These paths are
    # separate from the normal collection paths and are created at most once.
    partial_count=0
    for specification in \
      "$dgx_target|$dgx_run|$result_dir/remote_failure_dgx" \
      "$x86_target|$x86_run|$result_dir/remote_failure_x86"; do
      IFS='|' read -r target source destination <<<"$specification"
      if test ! -e "$destination" && remote "$target" "test -d '$source'" >/dev/null 2>&1; then
        mkdir "$destination"
        if remote "$target" "tar -C '$source' -czf - ." >"$destination.tar.gz" 2>/dev/null && \
            tar -C "$destination" -xzf "$destination.tar.gz" 2>/dev/null; then
          partial_count=$((partial_count + 1))
        fi
      fi
    done
    if test ! -e "$result_dir/d0_lane_failure.json"; then
      python3 - "$result_dir/d0_lane_failure.json" "$stage" "$lane" "$incoming" "$partial_count" <<'PY'
import json, sys, time
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1, "status": "FAIL", "stage": sys.argv[2],
    "lane": sys.argv[3], "observed_exit_code": int(sys.argv[4]),
    "partial_remote_evidence_tree_count": int(sys.argv[5]),
    "recorded_unix": time.time(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    fi
  fi
  exit "$incoming"
}
trap cleanup_inside EXIT
trap 'exit 130' INT TERM HUP

# Revalidate every prepared input on the resource hosts immediately before
# launch.  The other Lane must be stopped for D0.1/D0.2 single-Lane evidence.
remote "$dgx_target" \
  "set -euo pipefail; test \"\$(id -un)\" = '$dgx_user'; ip -4 -o addr show | grep -Fq ' $dgx_ip/'; test \"\$(cat '$dgx_root/T5_DEPLOYMENT_REF')\" = '$code_ref'; test -x '$dgx_root/scripts/run_t5_dgx_lane.sh'; test -f '$dgx_root/ros_ws/install/setup.bash'; test ! -e '$dgx_run'; test ! -e '$dgx_supervisor_ledger'; test \"\$(sha256sum '$map_manifest' | cut -d' ' -f1)\" = '$map_manifest_sha256'"
remote "$x86_target" \
  "set -euo pipefail; test \"\$(id -un)\" = song; ip -4 -o addr show | grep -Fq ' 10.100.120.111/'; test \"\$(cat '$x86_root/T5_DEPLOYMENT_REF')\" = '$code_ref'; test -x '$x86_root/scripts/run_t5_distributed_isaac.sh'; test ! -e '$x86_run'; test ! -e '$x86_supervisor_ledger'; test \"\$(sha256sum '$dataset_root/val_unseen/val_unseen.json.gz' | cut -d' ' -f1)\" = '$dataset_sha256'; test \"\$(docker inspect -f '{{.State.Running}}' '$container')\" = false; test \"\$(docker inspect -f '{{.State.Running}}' '$other_container')\" = false; test \"\$(docker inspect -f '{{index .Config.Labels \"internnav.t5.lane\"}}' '$container')\" = '$lane'; test \"\$(docker inspect -f '{{index .Config.Labels \"internnav.t5.gpu\"}}' '$container')\" = '$gpu'; test \"\$(docker inspect -f '{{.HostConfig.CpusetCpus}}' '$container')\" = '$cpuset'"
capture_container_inspect prestart
assert_other_lane_quiet

for port in $own_ports $other_ports; do
  remote "$x86_target" "test -z \"\$(ss -H -lntup | grep -E '[:.]$port[[:space:]]' || true)\""
done

# Persistent host quarantine closes the gap between a failed remote cleanup
# and release of the outer physical flock.  Both acknowledgements precede the
# first DGX runtime or Isaac container start.
t5_quarantine_arm "$dgx_target" "$dgx_quarantine_file" \
  "$dgx_quarantine_role" "$quarantine_run_tag" "$dgx_quarantine_roots" \
  "$result_dir/audits/dgx_quarantine_arm.json"
dgx_quarantine_armed=true
t5_quarantine_arm "$x86_target" "$isaac_lane_quarantine_file" \
  "$x86_quarantine_role" \
  "$quarantine_run_tag" "$x86_quarantine_roots" \
  "$result_dir/audits/x86_quarantine_arm.json"
x86_quarantine_armed=true

# A deployment-scoped supervisor audit is insufficient: Nav2 launch can
# create independent sessions whose argv no longer carries the current run
# root.  Prove whole-host production compute absence after quarantine is held
# and before the first DGX/model or Isaac process is launched.
t5_remote_compute_absent "$dgx_target" \
  "$result_dir/audits/dgx_structured_compute_prestart.json"

read -r -d '' dgx_runtime_program <<'REMOTE_DGX' || true
set -euo pipefail
IFS= read -r HF_TOKEN
IFS= read -r HF_ENDPOINT
[[ "$HF_TOKEN" =~ ^hf_[A-Za-z0-9]{20,}$ ]]
[[ "$HF_ENDPOINT" =~ ^https://hf-mirror\.com/?$ ]]
exec {hf_token_fd}<<<"$HF_TOKEN"
unset HF_TOKEN
deployment_root="$1"; lane="$2"; result_root="$3"; map_manifest="$4"
lease="$5"; domain="$6"; supervisor_ledger="$7"
pgid="$(ps -o pgid= -p "$$" | tr -d ' ')"
python3 - "$supervisor_ledger" "$result_root" "$$" "$pgid" dgx <<'PY'
import json, os, sys, time
from pathlib import Path
path=Path(sys.argv[1])
payload={'schema_version':1,'state':'RUNNING','run_root':sys.argv[2],
 'pid':int(sys.argv[3]),'pgid':int(sys.argv[4]),'component':sys.argv[5],
 'host':os.uname().nodename,'started_unix':time.time()}
with path.open('x',encoding='utf-8',newline='\n') as stream:
    stream.write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
exec env INTERNNAV_T5_RESOURCE_LEASE_ACK="$lease" \
  INTERNVLA_HF_TOKEN_FD="$hf_token_fd" HF_ENDPOINT="$HF_ENDPOINT" \
  INTERNNAV_RUNTIME_POLICY=completion_sim INTERNNAV_SIMULATION_TARGET=isaac \
  ROS_DOMAIN_ID="$domain" CUDA_VISIBLE_DEVICES=0 \
  INTERNNAV_T1_CONTROL_ROOT="$deployment_root" \
  INTERNVLA_ROS_WS="$deployment_root/ros_ws" \
  bash "$deployment_root/scripts/run_t5_dgx_lane.sh" \
    "$lane" model "$result_root" "$map_manifest"
REMOTE_DGX
dgx_program_b64="$(printf '%s' "$dgx_runtime_program" | base64 | tr -d '\r\n')"
dgx_command="exec setsid --wait bash -c \"\$(printf '%s' '$dgx_program_b64' | base64 -d)\" d0-dgx '$dgx_root' '$lane' '$dgx_run' '$map_manifest' '$resource_profile' '$ros_domain_id' '$dgx_supervisor_ledger'"
dgx_launch_attempted=1
printf '%s\n%s\n' "$HF_TOKEN" "$HF_ENDPOINT" | \
  ssh "${ssh_options[@]}" "$dgx_target" "$dgx_command" \
    >"$result_dir/logs/dgx_runtime_ssh.log" 2>&1 &
dgx_ssh_pid=$!

ready_timeout="${INTERNNAV_T5_D0_DGX_READY_TIMEOUT_SEC:-7200}"
[[ "$ready_timeout" =~ ^[1-9][0-9]*$ ]]
ready_deadline=$((SECONDS + ready_timeout))
dgx_ready=0
while (( SECONDS < ready_deadline )); do
  kill -0 "$dgx_ssh_pid" 2>/dev/null || break
  if remote "$dgx_target" \
      "test -f '$dgx_run/lane_ready.json' && test -f '$dgx_run/hf_token_process_audit.json' && python3 -c 'import json; r=json.load(open(\"$dgx_run/lane_ready.json\")); a=json.load(open(\"$dgx_run/hf_token_process_audit.json\")); assert r[\"status\"]==\"READY\" and r[\"lane\"]==\"$lane\" and r[\"mode\"]==\"model\"; assert a[\"status\"]==\"PASS\" and a[\"exact_secret_match_count\"]>=1 and a[\"disallowed_match_count\"]==0'" \
      >/dev/null 2>&1; then
    dgx_ready=1
    break
  fi
  sleep 2
done
test "$dgx_ready" = 1
kill -0 "$dgx_ssh_pid" 2>/dev/null
remote "$dgx_target" \
  "test -f '$dgx_run/lane_ready.json' && test -f '$dgx_run/hf_token_process_audit.json'"

# Start only the prepared worker for this Lane after the complete DGX lane is
# ready.  Both remote processes then overlap for the entire five-episode run.
container_started=1
remote "$x86_target" "docker start '$container' >/dev/null; test \"\$(docker inspect -f '{{.State.Running}}' '$container')\" = true"
read -r -d '' x86_runtime_program <<'REMOTE_X86' || true
set -euo pipefail
deployment_root="$1"; lane="$2"; result_root="$3"; dataset_root="$4"
lease="$5"; domain="$6"; gpu="$7"; supervisor_ledger="$8"
pgid="$(ps -o pgid= -p "$$" | tr -d ' ')"
python3 - "$supervisor_ledger" "$result_root" "$$" "$pgid" x86 <<'PY'
import json, os, sys, time
from pathlib import Path
path=Path(sys.argv[1])
payload={'schema_version':1,'state':'RUNNING','run_root':sys.argv[2],
 'pid':int(sys.argv[3]),'pgid':int(sys.argv[4]),'component':sys.argv[5],
 'host':os.uname().nodename,'started_unix':time.time()}
with path.open('x',encoding='utf-8',newline='\n') as stream:
    stream.write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
exec env INTERNNAV_T5_RESOURCE_LEASE_ACK="$lease" \
  INTERNNAV_RUNTIME_POLICY=completion_sim INTERNNAV_SIMULATION_TARGET=isaac \
  INTERNNAV_T5_ENGINEERING_CANARY_SEC=0 \
  ROS_DOMAIN_ID="$domain" CUDA_VISIBLE_DEVICES="$gpu" \
  INTERNNAV_T1_CONTROL_ROOT="$deployment_root" \
  INTERNVLA_ROS_WS=/home/song/internnav-t4/isaac_ros_ws_45 \
  INTERNVLA_T5_ISAAC_WORKER_ROOT=/home/song/internnav-t1-t2/runtime/t5_isaac_workers \
  bash "$deployment_root/scripts/run_t5_distributed_isaac.sh" \
    "$lane" model "$result_root" "$dataset_root"
REMOTE_X86
x86_program_b64="$(printf '%s' "$x86_runtime_program" | base64 | tr -d '\r\n')"
x86_command="exec setsid --wait bash -c \"\$(printf '%s' '$x86_program_b64' | base64 -d)\" d0-x86 '$x86_root' '$lane' '$x86_run' '$dataset_root' '$resource_profile' '$ros_domain_id' '$gpu' '$x86_supervisor_ledger'"
x86_launch_attempted=1
remote "$x86_target" "$x86_command" >"$result_dir/logs/x86_runtime_ssh.log" 2>&1 &
x86_ssh_pid=$!

x86_ready_timeout="${INTERNNAV_T5_D0_X86_READY_TIMEOUT_SEC:-1200}"
run_timeout="${INTERNNAV_T5_D0_RUN_TIMEOUT_SEC:-21600}"
[[ "$x86_ready_timeout" =~ ^[1-9][0-9]*$ ]]
[[ "$run_timeout" =~ ^[1-9][0-9]*$ ]]
x86_ready_deadline=$((SECONDS + x86_ready_timeout))
next_cross_lane_audit=$SECONDS
x86_ready=0
while (( SECONDS < x86_ready_deadline )); do
  kill -0 "$dgx_ssh_pid" 2>/dev/null || { echo "DGX lane exited before Isaac readiness" >&2; exit 1; }
  kill -0 "$x86_ssh_pid" 2>/dev/null || break
  if remote "$x86_target" \
      "test -f '$x86_run/health/ready_probe.json' && python3 -c 'import json; v=json.load(open(\"$x86_run/health/ready_probe.json\")); assert v[\"status\"]==\"PASS\" and v[\"lane\"]==\"$lane\"'" \
      >/dev/null 2>&1; then
    x86_ready=1
    break
  fi
  if (( SECONDS >= next_cross_lane_audit )); then
    assert_other_lane_quiet
    next_cross_lane_audit=$((SECONDS + 15))
  fi
  sleep 2
done
test "$x86_ready" = 1
kill -0 "$x86_ssh_pid" 2>/dev/null
remote "$x86_target" "test -f '$x86_run/health/ready_probe.json'"

run_deadline=$((SECONDS + run_timeout))
while kill -0 "$x86_ssh_pid" 2>/dev/null; do
  kill -0 "$dgx_ssh_pid" 2>/dev/null || { echo "DGX lane failed while Isaac was running" >&2; exit 1; }
  (( SECONDS < run_deadline )) || { echo "D0 fixed-five runtime timed out" >&2; exit 124; }
  if (( SECONDS >= next_cross_lane_audit )); then
    assert_other_lane_quiet
    next_cross_lane_audit=$((SECONDS + 15))
  fi
  sleep 2
done
set +e
wait "$x86_ssh_pid"; x86_rc=$?
set -e
x86_ssh_pid=""
# A normal DGX stop is authorized only after the fixed-five x86 process exits
# cleanly.  On any x86 failure, the outer fail-closed cleanup trap stops both
# remote supervisors and preserves the failure instead of minting a normal
# coordinator_stop receipt on the DGX.
test "$x86_rc" = 0
remote "$dgx_target" "touch '$dgx_run/stop.request'"
dgx_stop_deadline=$((SECONDS + 300))
while kill -0 "$dgx_ssh_pid" 2>/dev/null && (( SECONDS < dgx_stop_deadline )); do sleep 1; done
if kill -0 "$dgx_ssh_pid" 2>/dev/null; then
  echo "DGX lane did not stop within 300 seconds" >&2
  exit 124
fi
set +e
wait "$dgx_ssh_pid"; dgx_rc=$?
set -e
dgx_ssh_pid=""

# The container is infrastructure, not evidence of an active Lane.  Stop it
# before residual audits and prove the paired/other worker are both at PID 0.
remote "$x86_target" \
  "docker stop -t 20 '$container' >/dev/null; test \"\$(docker inspect -f '{{.State.Running}}' '$container')\" = false; test \"\$(docker inspect -f '{{.State.Pid}}' '$container')\" = 0"
container_started=0
capture_container_inspect poststop
assert_other_lane_quiet
test "$dgx_rc" = 0

collect_tree() {
  local target="$1" source="$2" destination="$3"
  test ! -e "$destination"
  mkdir "$destination"
  remote "$target" "tar -C '$source' -czf - ." >"$destination.tar.gz"
  tar -C "$destination" -xzf "$destination.tar.gz"
}
collect_tree "$dgx_target" "$dgx_run" "$result_dir/remote/dgx"
collect_tree "$x86_target" "$x86_run" "$result_dir/remote/x86"

# Host-side residual audit is intentionally after result collection and
# container stop.  It covers process/PID/PGID, sockets, lane runtime flock,
# health socket, the other container, and the exact physical GPU mapping.
read -r -d '' residual_program <<'REMOTE_AUDIT' || true
import json, os, subprocess, sys, time
from pathlib import Path
output, deployment_root, run_root, role, lane, container, other, ports = (
    Path(sys.argv[1]), *sys.argv[2:])
port_set={int(v) for v in ports.split(',')}
ancestors=set(); pid=os.getpid()
while pid > 1 and pid not in ancestors:
    ancestors.add(pid)
    try:
        status=Path(f'/proc/{pid}/status').read_text()
        pid=int(next(x.split()[1] for x in status.splitlines() if x.startswith('PPid:')))
    except Exception: break
processes=[]
for entry in Path('/proc').iterdir():
    if not entry.name.isdigit() or int(entry.name) in ancestors: continue
    try: command=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip()
    except (FileNotFoundError, PermissionError): continue
    if run_root in command: processes.append({'pid':int(entry.name),'command':command})
ss=subprocess.run(['ss','-H','-lntup'],text=True,capture_output=True,check=False).stdout
sockets=[line for line in ss.splitlines() if any(f':{p} ' in line for p in port_set)]
checks={'run_process_count_zero':not processes,'lane_socket_count_zero':not sockets}
containers={}
if role == 'x86':
    for key,name in (('own',container),('other',other)):
        value=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
        containers[key]={'name':name,'running':value['State']['Running'],'pid':value['State']['Pid']}
    checks['both_containers_stopped']=all(not v['running'] and v['pid']==0 for v in containers.values())
    lock='/tmp/internnav_t5_isaac_a_runtime.lock' if lane=='a' else '/tmp/internnav_t5_isaac_b_runtime.lock'
    probe=subprocess.run(['flock','-n',lock,'true'],capture_output=True)
    checks['lane_runtime_lock_free']=probe.returncode==0
payload={'schema_version':1,'status':'PASS' if all(checks.values()) else 'FAIL',
         'role':role,'lane':lane,'deployment_root':deployment_root,'run_root':run_root,
         'processes':processes,'sockets':sockets,'containers':containers,
         'checks':checks,'recorded_unix':time.time()}
output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n',encoding='utf-8')
if payload['status'] != 'PASS': raise SystemExit(f'residual audit failed: {checks}')
REMOTE_AUDIT
residual_b64="$(printf '%s' "$residual_program" | base64 | tr -d '\r\n')"
all_ports="$(printf '%s %s' "$own_ports" "$other_ports" | tr ' ' ',')"
remote "$dgx_target" \
  "python3 -c \"\$(printf '%s' '$residual_b64' | base64 -d)\" '$dgx_run/residual_host_audit.json' '$dgx_root' '$dgx_run' dgx '$lane' none none '$all_ports'"
remote "$x86_target" \
  "python3 -c \"\$(printf '%s' '$residual_b64' | base64 -d)\" '$x86_run/residual_host_audit.json' '$x86_root' '$x86_run' x86 '$lane' '$container' '$other_container' '$all_ports'"
remote "$dgx_target" "cat '$dgx_run/residual_host_audit.json'" >"$result_dir/audits/dgx_residual_host_audit.json"
remote "$x86_target" "cat '$x86_run/residual_host_audit.json'" >"$result_dir/audits/x86_residual_host_audit.json"

# Scan every collected plaintext/result artifact for the exact credential
# before producing a PASS receipt.  The secret travels on a private descriptor;
# neither argv nor the machine-readable audit contains its value or digest.
exec {secret_scan_fd}<<<"$HF_TOKEN"
python3 - "$result_dir" "$result_dir/audits/credential_exposure_audit.json" \
  "$secret_scan_fd" <<'PY'
import json, os, sys
from pathlib import Path

root, output = Path(sys.argv[1]), Path(sys.argv[2])
with os.fdopen(int(sys.argv[3]), 'rb', closefd=True) as stream:
    secret = stream.read().rstrip(b'\r\n')
if not secret:
    raise SystemExit('credential exposure audit received an empty secret')
matches = []
for path in root.rglob('*'):
    if not path.is_file() or path == output:
        continue
    try:
        if secret in path.read_bytes():
            matches.append(str(path.relative_to(root)))
    except OSError as exc:
        raise SystemExit(f'cannot scan result artifact {path}: {exc}')
payload = {
    'schema_version': 1, 'status': 'PASS' if not matches else 'FAIL',
    'exact_secret_match_count': len(matches),
    'matched_relative_paths': sorted(matches),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
if matches:
    raise SystemExit('credential exposure detected in collected evidence')
PY
unset HF_TOKEN

python3 "$root/scripts/finalize_t5_d0_dual.py" episodes \
  --dgx-root "$result_dir/remote/dgx" \
  --lane "$lane" \
  --manifest "$result_dir/grant_validation.json" \
  --order-manifest "$result_dir/remote/x86/ordered_episode_manifest.json" \
  --readiness-evidence \
    "$result_dir/remote/x86/health/runtime_readiness_evidence.json" \
  --output "$result_dir/audits/fixed_five_episode_coverage.json"

python3 - "$result_dir/d0_lane_runtime_summary.json" "$lane" "$stage" \
  "$grant_id" "$authorization_ref" "$code_ref" "$golden_sha256" "$run_manifest_sha256" \
  "$identity_prefix" "$other_identity_prefix" "$ros_domain_id" "$gpu" \
  "$cpuset" "$dgx_root" "$x86_root" "$dataset_sha256" "$map_manifest_sha256" <<'PY'
import hashlib, json, math, sys, time
from pathlib import Path
output=Path(sys.argv[1]); result=output.parent
(lane,stage,grant_id,authorization_ref,code_ref,golden_sha,run_sha,prefix,other_prefix,
 domain,gpu,cpuset,dgx_root,x86_root,dataset_sha,map_sha)=sys.argv[2:]
def load(relative): return json.loads((result/relative).read_text(encoding='utf-8'))
dgx_status=load('remote/dgx/lane_status.json')
dgx_ready=load('remote/dgx/lane_ready.json')
model_identity=load('remote/dgx/model_identity_audit.json')
token_process_audit=load('remote/dgx/hf_token_process_audit.json')
x86_status=load('remote/x86/isaac_status.json')
dgx_contract=load('remote/dgx/lane_contract.json')
x86_contract=load('remote/x86/isaac_contract.json')
gpu_mapping=load('remote/x86/gpu_mapping.json')
kit_gpu_audit=load('remote/x86/kit_gpu_audit.json')
container_prestart=load('audits/container_prestart.json')
container_poststop=load('audits/container_poststop.json')
dgx_residual=load('audits/dgx_residual_host_audit.json')
x86_residual=load('audits/x86_residual_host_audit.json')
credential_audit=load('audits/credential_exposure_audit.json')
episode_coverage=load('audits/fixed_five_episode_coverage.json')
preparation_chain=load('preparation_chain_binding.json')
attempts=[path for path in (result/'remote/x86/evaluator').glob(
              f't5_lane_{lane}_model*_model_attempt_*')
          if (path/'result.json').is_file()
          and (path/'isaac_remote_validation.json').is_file()]
if len(attempts) != 1:
    raise SystemExit('ambiguous Lane evaluator evidence')
attempt=attempts[0]
metrics_root=json.loads((attempt/'result.json').read_text(encoding='utf-8'))
metrics=metrics_root.get('val_unseen',metrics_root)
validation=json.loads((attempt/'isaac_remote_validation.json').read_text(encoding='utf-8'))
required={name:float(metrics[name]) for name in ('SR','OS','SPL','NE')}
count=int(metrics.get('Count',metrics.get('length',0)))
kit_files=[p for p in (result/'remote/x86/logs/kit').rglob('*') if p.is_file()]

def ledger_is_clean(relative):
    rows=[json.loads(line) for line in (result/relative).read_text(encoding='utf-8').splitlines()
          if line.strip()]
    started={(row.get('scope','host'),row.get('component'),row.get('pid'))
             for row in rows if row.get('event')=='started' and row.get('pid') is not None}
    absent={(row.get('scope','host'),row.get('component'),row.get('pid'))
            for row in rows if row.get('event')=='verified_absent' and row.get('pid') is not None}
    return bool(started) and started.issubset(absent)

def no_other_prefix(value):
    if isinstance(value,dict):
        return all(no_other_prefix(v) for v in value.values())
    if isinstance(value,list): return all(no_other_prefix(v) for v in value)
    return not (isinstance(value,str) and value.startswith(other_prefix))

checks={
 'fixed_five_completed':count==5 and validation.get('episode_count')==5
     and validation.get('expected_episode_count')==5,
 'exact_frozen_episode_coverage':episode_coverage.get('status')=='PASS'
     and episode_coverage.get('lane')==lane
     and bool(episode_coverage.get('checks'))
     and all(episode_coverage['checks'].values()),
 'evaluator_validation':validation.get('status')=='PASS',
 'metrics_finite':all(math.isfinite(v) for v in required.values()),
 'metrics_ranges':0.0<=required['SR']<=1.0 and 0.0<=required['OS']<=1.0
     and 0.0<=required['SPL']<=1.0 and required['NE']>=0.0,
 # D0 has intentionally no minimum-SR gate; zero is reportable evidence.
 'sr_report_only':float(validation.get('minimum_success_rate',-1))==0.0,
 'dgx_cleanup':dgx_status.get('status')=='PASS' and dgx_status.get('residual_count')==0,
 'golden_runtime_model_identity':model_identity.get('status')=='PASS'
     and model_identity.get('golden_bundle_canonical_sha256')==golden_sha
     and bool(model_identity.get('checks')) and all(model_identity['checks'].values())
     and dgx_ready.get('golden_bundle_canonical_sha256')==golden_sha
     and dgx_ready.get('runtime_weight_inventory_sha256')
         == model_identity.get('expected_runtime_weight_inventory_sha256'),
 'x86_cleanup':x86_status.get('status')=='PASS' and x86_status.get('residual_count')==0
     and x86_status.get('socket_residual_count')==0
     and x86_status.get('clock_publishers_after_stop')==0
     and x86_status.get('lane_lock_released') is True
     and x86_status.get('shared_asset_lock_fd_released') is True,
 'fixed_dataset_execution':x86_status.get('execution_profile')=='fixed_dataset'
     and x86_status.get('episode_acceptance_claimed') is True
     and x86_status.get('evaluation_completed_naturally') is True
     and x86_status.get('termination_reason')=='evaluator_natural_exit',
 'host_residual_audits':dgx_residual.get('status')=='PASS' and x86_residual.get('status')=='PASS',
 'credential_not_exposed':credential_audit.get('status')=='PASS'
      and credential_audit.get('exact_secret_match_count')==0,
 'credential_process_scope':token_process_audit.get('status')=='PASS'
      and token_process_audit.get('exact_secret_match_count',0)>=1
      and token_process_audit.get('allowed_model_match_count')
          == token_process_audit.get('exact_secret_match_count')
      and token_process_audit.get('disallowed_match_count')==0
      and token_process_audit.get('parent_match_count')==0
      and token_process_audit.get('onboard_match_count')==0
      and token_process_audit.get('evaluator_match_count')==0
      and token_process_audit.get('secret_value_recorded') is False
      and token_process_audit.get('secret_digest_recorded') is False
      and bool(token_process_audit.get('checks'))
      and all(token_process_audit['checks'].values()),
 'same_root_preparation_chain':preparation_chain.get('status')=='PASS'
     and bool(preparation_chain.get('checks'))
     and all(preparation_chain['checks'].values()),
 'pid_pgid_ledgers_clean':ledger_is_clean('remote/dgx/pid_ledger.jsonl')
     and ledger_is_clean('remote/x86/pid_ledger.jsonl'),
 'lane_identity':dgx_contract.get('lane')==lane and x86_contract.get('lane')==lane
     and dgx_contract.get('identity_prefix')==prefix
     and x86_contract.get('identity',{}).get('episode_prefix')==prefix,
 'dds_isolation':dgx_contract.get('ros_domain_id')==int(domain)
     and x86_contract.get('ros_domain_id')==int(domain)
     and dgx_contract.get('namespace')==f'/t5/lane_{lane}'
     and x86_contract.get('namespace')==f'/t5/lane_{lane}',
 'no_other_lane_identity':no_other_prefix(metrics_root) and no_other_prefix(validation),
 'gpu_mapping':gpu_mapping.get('status')=='PASS'
     and gpu_mapping.get('host_physical_gpu_index')==int(gpu)
     and gpu_mapping.get('container_gpu_count')==1,
 'kit_gpu_contract':kit_gpu_audit.get('status')=='PASS'
     and x86_contract.get('kit_active_gpu_log_audit_required') is True
     and x86_contract.get('isaac_render_gpu_physical_index')==int(gpu)
     and x86_contract.get('isaac_physics_gpu_visible_index')==0
     and x86_contract.get('cpuset')==cpuset,
 'container_lifecycle':container_prestart.get('status')=='PASS'
     and container_poststop.get('status')=='PASS'
     and container_prestart.get('container_id')==container_poststop.get('container_id')
     and container_poststop.get('running') is False and container_poststop.get('pid')==0,
 'dataset_hash':x86_contract.get('dataset',{}).get('sha256')==dataset_sha,
 'map_hash':dgx_contract.get('inputs',{}).get('static_map_manifest',{}).get('sha256')==map_sha,
}
key_files=[attempt/'result.json',attempt/'isaac_remote_validation.json',
  result/'remote/dgx/lane_status.json',result/'remote/dgx/model_identity_audit.json',
  result/'remote/dgx/hf_token_process_audit.json',
  result/'remote/dgx/client/client_summary.json',
  result/'remote/dgx/onboard/controller_summary.json',
  result/'remote/dgx/client/client_records.jsonl',
  result/'remote/dgx/onboard/controller_records.jsonl',
  result/'remote/x86/isaac_status.json',
  result/'remote/x86/gpu_mapping.json',result/'remote/x86/kit_gpu_audit.json',
  result/'audits/credential_exposure_audit.json',
  result/'audits/fixed_five_episode_coverage.json',
 result/'audits/container_prestart.json',result/'audits/container_poststop.json']
payload={
 'schema_version':1,'status':'PASS' if all(checks.values()) else 'FAIL',
 'stage':stage,'lane':lane,'grant_id':grant_id,'authorization_ref_sha':authorization_ref,
 'code_ref_sha':code_ref,
 'golden_bundle_canonical_sha256':golden_sha,'run_manifest_canonical_sha256':run_sha,
 'prep_binding_sha256':hashlib.sha256((result/'prep_binding.json').read_bytes()).hexdigest(),
 'predecessor_binding_sha256':hashlib.sha256((result/'predecessor_binding.json').read_bytes()).hexdigest(),
 'preparation_chain_binding_sha256':hashlib.sha256((result/'preparation_chain_binding.json').read_bytes()).hexdigest(),
 'checks':checks,'episode_count':count,'episode_keys_source':'frozen D0 manifest and dataset SHA',
 'episode_coverage':episode_coverage,
 'metrics':required,'acceptance':{'sr_is_report_only':True,'minimum_sr':None,
    'required_metrics':['SR','OS','SPL','NE']},
 'placement':{'dgx_complete_lane_root':dgx_root,'x86_isaac_only_root':x86_root},
 'isolation':{'ros_domain_id':int(domain),'namespace':f'/t5/lane_{lane}',
    'identity_prefix':prefix,'gpu':int(gpu),'cpuset':cpuset,
    'cross_lane_observation_count':0 if checks['no_other_lane_identity'] else None},
 'kit_log_files':[str(p.relative_to(result)) for p in kit_files],
 'key_file_sha256':{str(p.relative_to(result)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in key_files},
 'locks':{'profile':f'lane-{lane}','state_at_summary':'HELD_BY_OUTER_FAIL_CLOSED_LEASE',
          'release_evidence':'lease_release_summary.json'},
 'recorded_unix':time.time(),
}
with output.open('x',encoding='utf-8',newline='\n') as stream:
    stream.write(json.dumps(payload,indent=2,sort_keys=True)+'\n')
if payload['status']!='PASS': raise SystemExit(f'D0 lane evidence failed: {checks}')
PY

trap - EXIT INT TERM HUP
cleanup_inside
