#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s CODE_SHA RUN_ID results/internnav_t5/step3-prepare-RUN_ID\n' \
    "${0##*/}" >&2
  exit 64
}

[[ $# -eq 3 ]] || usage
code_sha="$1"
run_id="$2"
result_relative="$3"
[[ "$code_sha" =~ ^[0-9a-f]{40}$ ]] || usage
[[ "$run_id" =~ ^[a-z0-9][a-z0-9._-]{7,63}$ ]] || usage
test "$result_relative" = "results/internnav_t5/step3-prepare-$run_id" || usage

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
result_dir="$root/$result_relative"
target=rail@10.100.120.116
short_sha="${code_sha:0:12}"
deployment_root="/home/rail/internnav-t1-t2/.t5-deployments/t5step3-${run_id}-${short_sha}"
remote_result="/home/rail/internnav-t1-t2/results/t5_step3_prepare/$run_id"
model_path=/home/rail/ai-stack/models/Step3-VL-10B
venv_path=/home/rail/ai-stack/venvs/step3-vl-10b-tf4.57.6

if [[ "${INTERNNAV_T5_INSIDE_STEP3_PREPARE:-0}" != 1 ]]; then
  test ! -e "$result_dir"
  if git -C "$root" rev-parse --git-dir >/dev/null 2>&1; then
    git_command=(git -C "$root")
  else
    command -v git.exe >/dev/null
    root_windows="$(wslpath -w "$root")"
    git_command=(git.exe -C "$root_windows")
  fi
  "${git_command[@]}" cat-file -e "$code_sha^{commit}"
  test "$("${git_command[@]}" rev-parse "$code_sha")" = "$code_sha"
  mkdir -p "$result_dir"
  "${git_command[@]}" archive --format=tar "$code_sha" | gzip -n -9 \
    >"$result_dir/deployment.tar.gz"
  sha256sum "$result_dir/deployment.tar.gz" | cut -d' ' -f1 \
    >"$result_dir/deployment_archive_sha256.txt"
  env INTERNNAV_T5_INSIDE_STEP3_PREPARE=1 \
    bash "$root/scripts/with_resource_lease.sh" dgx-b \
      --owner codex-00 --task "t5-step3-runtime-prepare:$run_id:$code_sha" \
      --log-dir "$result_dir/lease" --acquire-timeout 30 \
      --cleanup-timeout 120 --kill-wait-timeout 30 -- \
    bash "$root/coordination/prepare_t5_step3_runtime_online.sh" \
      "$code_sha" "$run_id" "$result_relative"
  exit $?
fi

test -d "$result_dir"
archive="$result_dir/deployment.tar.gz"
archive_sha256="$(tr -d '\r\n' <"$result_dir/deployment_archive_sha256.txt")"
[[ "$archive_sha256" =~ ^[0-9a-f]{64}$ ]]
test "$(sha256sum "$archive" | cut -d' ' -f1)" = "$archive_sha256"
stage="/home/rail/.cache/internnav_t5_uploads/${run_id}-${short_sha}.tar.gz"
partial="${deployment_root}.partial"

ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$target" \
  "set -euo pipefail; test \"\$(id -un)\" = rail; test ! -e '$stage'; test ! -e '$partial'; test ! -e '$deployment_root'; install -d -m 700 /home/rail/.cache/internnav_t5_uploads /home/rail/internnav-t1-t2/.t5-deployments /home/rail/internnav-t1-t2/results/t5_step3_prepare"
scp -q -o BatchMode=yes -o StrictHostKeyChecking=yes "$archive" "$target:$stage"
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$target" \
  "set -euo pipefail; test \"\$(sha256sum '$stage'|cut -d' ' -f1)\" = '$archive_sha256'; install -d -m 700 '$partial'; gzip -dc '$stage' | tar -C '$partial' -xf -; printf '%s\n' '$code_sha' >'$partial/T5_DEPLOYMENT_REF'; printf '%s\n' '$archive_sha256' >'$partial/T5_DEPLOYMENT_ARCHIVE_SHA256'; mv '$partial' '$deployment_root'; rm -f -- '$stage'; test \"\$(cat '$deployment_root/T5_DEPLOYMENT_REF')\" = '$code_sha'"

ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$target" \
  "set -euo pipefail; test ! -e '$remote_result'; install -d -m 700 '$remote_result'; env INTERNNAV_T5_RESOURCE_LEASE_ACK=dgx-b INTERNNAV_T1_CONTROL_ROOT='$deployment_root' bash '$deployment_root/scripts/setup_t5_step3_runtime.sh' '$model_path' '$venv_path' '$remote_result/setup'"
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$target" \
  "set -euo pipefail; env INTERNNAV_T5_RESOURCE_LEASE_ACK=dgx-b INTERNNAV_T1_CONTROL_ROOT='$deployment_root' bash '$deployment_root/scripts/run_t5_step3_shadow.sh' health-only '$remote_result/health' '$model_path' '$venv_path'; for port in 8200 8300; do test -z \"\$(ss -H -lntp | grep -E \"[:.]\${port}[[:space:]]\" || true)\"; done"

mkdir -p "$result_dir/remote"
scp -q -r -o BatchMode=yes -o StrictHostKeyChecking=yes \
  "$target:$remote_result/setup" "$target:$remote_result/health" \
  "$result_dir/remote/"

python3 - "$result_dir" "$code_sha" "$deployment_root" "$remote_result" \
  "$archive_sha256" <<'PY'
import hashlib
import json
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
code_sha, deployment_root, remote_result, archive_sha256 = sys.argv[2:]

def load(relative):
    path = root / relative
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

runtime = load("remote/setup/runtime_setup.json")
health = load("remote/health/step3/health.json")
frontend = load("remote/health/frontend_state.json")
cleanup = load("remote/health/cleanup.json")
checks = {
    "archive_sha_exact": hashlib.sha256(
        root.joinpath("deployment.tar.gz").read_bytes()
    ).hexdigest() == archive_sha256,
    "runtime_ready": isinstance(runtime, dict)
    and runtime.get("status") == "RUNTIME_READY"
    and runtime.get("receipt_revalidation")
    and all(runtime["receipt_revalidation"].values()),
    "clean_bf16_model_ready": isinstance(health, dict)
    and health.get("status") == "READY"
    and health.get("checks")
    and all(health["checks"].values()),
    "frontend_readonly": isinstance(frontend, dict)
    and frontend.get("lane_id") == "b"
    and frontend.get("readonly") is True,
    "owned_cleanup": isinstance(cleanup, dict)
    and cleanup.get("status") == "PASS"
    and cleanup.get("residual_pids") == [],
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "code_ref_sha": code_sha,
    "deployment_archive_sha256": archive_sha256,
    "deployment_root": deployment_root,
    "remote_result_root": remote_result,
    "model_path": "/home/rail/ai-stack/models/Step3-VL-10B",
    "venv_path": "/home/rail/ai-stack/venvs/step3-vl-10b-tf4.57.6",
    "checks": checks,
    "recorded_unix": time.time(),
}
root.joinpath("step3_prepare_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["status"] == "PASS" else 75)
PY
