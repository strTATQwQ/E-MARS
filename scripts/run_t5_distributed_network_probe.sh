#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: run_t5_distributed_network_probe.sh GRANT_ID AUTHORIZATION_REF RESULT_DIR" >&2
  exit 64
}

[[ $# -eq 3 ]] || usage
grant_id="$1"
authorization_ref="$2"
result_relative="$3"
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
board="$root/coordination/T5_TASK_BOARD.md"

[[ "$grant_id" =~ ^t5n[0-9]{8}t[0-9]{6}$ ]] || usage
[[ "$authorization_ref" =~ ^[0-9a-f]{40}$ ]] || usage
case "$result_relative" in
  results/internnav_t5/network-contract-00-"$grant_id") ;;
  *) echo "result directory does not match grant" >&2; exit 64 ;;
esac

git_command=(git -C "$root")
if [[ -f "$root/.git" ]] \
    && grep -Eq '^gitdir: [A-Za-z]:/' "$root/.git"; then
  command -v git.exe >/dev/null
  command -v wslpath >/dev/null
  git_command=(git.exe -C "$(wslpath -w "$root")")
fi
"${git_command[@]}" cat-file -e "$authorization_ref^{commit}"
"${git_command[@]}" merge-base --is-ancestor "$authorization_ref" HEAD
mapfile -t post_ref_changes < <(
  "${git_command[@]}" diff --name-only "$authorization_ref" HEAD | tr -d '\r'
)
[[ "${#post_ref_changes[@]}" -eq 1 ]]
[[ "${post_ref_changes[0]}" == "coordination/T5_TASK_BOARD.md" ]]
[[ -z "$("${git_command[@]}" status --porcelain --untracked-files=no | tr -d '\r')" ]]

python3 - "$board" "$grant_id" "$authorization_ref" "$result_relative" <<'PY'
import json, re, sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
match = re.search(
    r"<!-- INTERNAV_T5_ONLINE_GRANT_V1\s*(\{.*?\})\s*INTERNAV_T5_ONLINE_GRANT_V1 -->",
    text,
    re.S,
)
if match is None:
    raise SystemExit("missing T5 grant block")
grant = json.loads(match.group(1))
expected = {
    "schema_version": 1,
    "status": "GRANTED",
    "profile": "t5_network_probe",
    "result_dir": sys.argv[4],
    "grant_id": sys.argv[2],
    "authorization_ref_sha": sys.argv[3],
}
if grant != expected:
    raise SystemExit(f"T5 grant mismatch: {grant!r}")
PY

result_dir="$root/$result_relative"
[[ ! -e "$result_dir" ]]
mkdir -p -- "$result_dir/lease"

bash "$root/scripts/with_resource_lease.sh" t5-network \
  --owner codex-00 \
  --task "t5_network_probe:$grant_id:$authorization_ref" \
  --log-dir "$result_dir/lease" \
  --acquire-timeout 30 \
  -- python3 "$root/scripts/t5_network_probe.py" \
    --result-dir "$result_dir/probe" \
    --topology "$root/configs/internnav_t5/topology.json"
