#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s results/internnav_t5/STEP3_COPY_VERIFY_DIR\n' "${0##*/}" >&2
  exit 64
}

[[ $# -eq 1 ]] || usage
result_relative="$1"
[[ "$result_relative" =~ ^results/internnav_t5/step3-copy-verify-[a-z0-9._-]{8,80}$ ]] || usage
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
result_dir="$root/$result_relative"

if [[ "${INTERNNAV_T5_INSIDE_STEP3_COPY_VERIFY:-0}" != 1 ]]; then
  test ! -e "$result_dir"
  mkdir -p -- "$result_dir"
  env INTERNNAV_T5_INSIDE_STEP3_COPY_VERIFY=1 \
    bash "$root/scripts/with_resource_lease.sh" dgx-a \
      --owner codex-00 --task t5-step3-source-hash \
      --log-dir "$result_dir/lease-a" --acquire-timeout 30 \
      --cleanup-timeout 90 --kill-wait-timeout 30 -- \
    bash "$root/scripts/with_resource_lease.sh" dgx-b \
      --owner codex-00 --task t5-step3-destination-hash \
      --log-dir "$result_dir/lease-b" --acquire-timeout 30 \
      --cleanup-timeout 90 --kill-wait-timeout 30 -- \
    bash "$root/coordination/verify_t5_step3_lan_copy.sh" "$result_relative"
  exit $?
fi

test -d "$result_dir"
test ! -e "$result_dir/copy_verification.json"
source_root=/home/railgun/ai-stack/models/Step3-VL-10B
destination_root=/home/rail/ai-stack/models/Step3-VL-10B

ssh -o BatchMode=yes -o StrictHostKeyChecking=yes railgun@10.100.100.128 \
  "set -euo pipefail; cd '$source_root'; find . -type f -print0 | sort -z | xargs -0 sha256sum" \
  >"$result_dir/source.sha256" &
source_pid=$!
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes rail@10.100.120.116 \
  "set -euo pipefail; cd '$destination_root'; find . -type f -print0 | sort -z | xargs -0 sha256sum" \
  >"$result_dir/destination.sha256" &
destination_pid=$!

source_rc=0
destination_rc=0
wait "$source_pid" || source_rc=$?
wait "$destination_pid" || destination_rc=$?

python3 - "$result_dir" "$source_rc" "$destination_rc" <<'PY'
import hashlib
import json
import sys
import time
from pathlib import Path

root = Path(sys.argv[1])
source_rc, destination_rc = int(sys.argv[2]), int(sys.argv[3])
source = root.joinpath("source.sha256").read_bytes()
destination = root.joinpath("destination.sha256").read_bytes()
rows = [line for line in source.decode("utf-8").splitlines() if line]
checks = {
    "source_hash_completed": source_rc == 0,
    "destination_hash_completed": destination_rc == 0,
    "nonempty_manifest": len(rows) > 0,
    "exact_file_and_content_match": source == destination,
}
payload = {
    "schema_version": 1,
    "status": "COPY_VERIFIED" if all(checks.values()) else "FAIL",
    "source": "railgun@10.100.100.128:/home/railgun/ai-stack/models/Step3-VL-10B",
    "destination": "rail@10.100.120.116:/home/rail/ai-stack/models/Step3-VL-10B",
    "file_count": len(rows),
    "manifest_sha256": hashlib.sha256(source).hexdigest(),
    "checks": checks,
    "verified_unix": time.time(),
}
root.joinpath("copy_verification.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, sort_keys=True))
raise SystemExit(0 if payload["status"] == "COPY_VERIFIED" else 75)
PY
