#!/usr/bin/env bash

# Shared fail-closed audit for a DGX that must not contain any model,
# navigation, localisation, mapping, watchdog, or velocity-control process.
# The caller must define root and remote().  The exact local auditor is sent
# to the host for every observation, so an old remote deployment cannot weaken
# the process identity contract.

test -n "${root:-}"
_t5_compute_auditor="$root/scripts/t5_process_identity_audit.py"
test -f "$_t5_compute_auditor"
_t5_compute_auditor_b64="$(base64 <"$_t5_compute_auditor" | tr -d '\r\n')"
_t5_compute_auditor_sha256="$(sha256sum "$_t5_compute_auditor" | cut -d' ' -f1)"

t5_remote_compute_absent() {
  local target="$1" output="$2"
  local raw="${output%.json}.remote.json"
  local stderr_log="${output%.json}.stderr.log"
  local remote_exit

  test ! -e "$output"
  test ! -e "$raw"
  test ! -e "$stderr_log"
  mkdir -p -- "$(dirname -- "$output")"

  if remote "$target" \
      "set -uo pipefail; audit=\$(mktemp /tmp/internnav-t5-compute-audit.XXXXXX.json); trap 'rm -f -- \"\$audit\"' EXIT; set +e; python3 -c \"\$(printf '%s' '$_t5_compute_auditor_b64' | base64 -d)\" --mode forbidden-compute --output \"\$audit\"; rc=\$?; set -e; test -f \"\$audit\" || exit 75; cat -- \"\$audit\"; exit \"\$rc\"" \
      >"$raw" 2>"$stderr_log"; then
    remote_exit=0
  else
    remote_exit=$?
  fi

  python3 - "$raw" "$stderr_log" "$output" "$target" "$remote_exit" \
    "$_t5_compute_auditor_sha256" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

raw_path, stderr_path, output_path = map(Path, sys.argv[1:4])
target, remote_exit, source_sha256 = sys.argv[4], int(sys.argv[5]), sys.argv[6]
audit = None
parse_error = None
try:
    audit = json.loads(raw_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    parse_error = type(error).__name__

checks = {
    "remote_exit_zero": remote_exit == 0,
    "audit_object": isinstance(audit, dict),
    "schema_version_one": isinstance(audit, dict)
        and audit.get("schema_version") == 1,
    "forbidden_compute_mode": isinstance(audit, dict)
        and audit.get("mode") == "forbidden-compute",
    "audit_status_pass": isinstance(audit, dict)
        and audit.get("status") == "PASS",
    "match_count_zero": isinstance(audit, dict)
        and audit.get("match_count") == 0,
    "errors_empty": isinstance(audit, dict)
        and audit.get("errors") == [],
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "target": target,
    "remote_exit": remote_exit,
    "auditor_sha256": source_sha256,
    "remote_audit": audit,
    "parse_error": parse_error,
    "stderr_bytes": stderr_path.stat().st_size if stderr_path.is_file() else None,
    "checks": checks,
    "recorded_unix": time.time(),
}
output = Path(output_path)
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
os.replace(temporary, output)
raise SystemExit(0 if payload["status"] == "PASS" else 75)
PY
}
