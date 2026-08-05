#!/usr/bin/env bash

# Shared T5 quarantine operations for coordinators that already hold physical
# resource flocks.  The caller must define remote TARGET COMMAND.  All marker
# mutation is performed by the frozen local Python payload transferred inline.

_t5_quarantine_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
_t5_quarantine_control="$_t5_quarantine_script_dir/t5_resource_quarantine.py"
test -f "$_t5_quarantine_control"
T5_QUARANTINE_CONTROL_B64="$(base64 <"$_t5_quarantine_control" | tr -d '\r\n')"
readonly T5_QUARANTINE_CONTROL_B64

t5_quarantine_roots_b64() {
  local roots="$1"
  test -n "$roots"
  [[ "$roots" != *$'\n'* && "$roots" != *$'\r'* && "$roots" != *"'"* ]]
  printf '%s' "$roots" | base64 | tr -d '\r\n'
}

t5_quarantine_arm() {
  local target="$1" marker="$2" role="$3" run_tag="$4" roots="$5" output="$6"
  local reason="${7:-t5_online_stage_in_progress}" roots_b64
  case "$reason" in
    t5_online_stage_in_progress|d0_prepare_in_progress) ;;
    *) return 64 ;;
  esac
  roots_b64="$(t5_quarantine_roots_b64 "$roots")"
  remote "$target" \
    "python3 -c \"\$(printf '%s' '$T5_QUARANTINE_CONTROL_B64' | base64 -d)\" arm '$marker' '$reason' '$role' '$run_tag' '$roots_b64'" \
    >"$output"
  python3 - "$output" "$role" "$run_tag" "$reason" <<'PY'
import json, sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
checks=(value.get('status')=='PASS' and value.get('action')=='ARM'
        and value.get('role')==sys.argv[2] and value.get('run_tag')==sys.argv[3]
        and value.get('reason')==sys.argv[4]
        and value.get('marker_present') is True)
raise SystemExit(0 if checks else 75)
PY
}

t5_quarantine_owned_observe() {
  local target="$1" marker="$2" role="$3" run_tag="$4" roots="$5" output="$6"
  local roots_b64
  roots_b64="$(t5_quarantine_roots_b64 "$roots")"
  remote "$target" \
    "python3 -c \"\$(printf '%s' '$T5_QUARANTINE_CONTROL_B64' | base64 -d)\" observe-owned '$marker' t5_online_stage_in_progress '$role' '$run_tag' '$roots_b64'" \
    >"$output"
  python3 - "$output" "$role" "$run_tag" <<'PY'
import json, sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
checks=(value.get('status')=='PASS' and value.get('action')=='OBSERVE_OWNED'
        and value.get('role')==sys.argv[2] and value.get('run_tag')==sys.argv[3]
        and value.get('marker_present') is True
        and value.get('workload_started') is False)
raise SystemExit(0 if checks else 75)
PY
}

t5_quarantine_owned_clear() {
  local target="$1" marker="$2" role="$3" run_tag="$4" roots="$5"
  local cleanup_receipt="$6" output="$7" roots_b64 receipt_b64
  test -f "$cleanup_receipt"
  roots_b64="$(t5_quarantine_roots_b64 "$roots")"
  receipt_b64="$(base64 <"$cleanup_receipt" | tr -d '\r\n')"
  remote "$target" \
    "python3 -c \"\$(printf '%s' '$T5_QUARANTINE_CONTROL_B64' | base64 -d)\" owned-clear '$marker' t5_online_stage_in_progress '$role' '$run_tag' '$roots_b64' '$receipt_b64'" \
    >"$output"
  python3 - "$output" "$role" "$run_tag" <<'PY'
import json, sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
checks=(value.get('status')=='PASS' and value.get('action')=='OWNED_CLEAR'
        and value.get('role')==sys.argv[2] and value.get('run_tag')==sys.argv[3]
        and value.get('marker_absent') is True)
raise SystemExit(0 if checks else 75)
PY
}

t5_quarantine_recover() {
  local target="$1" marker="$2" role="$3" output="$4"
  remote "$target" \
    "python3 -c \"\$(printf '%s' '$T5_QUARANTINE_CONTROL_B64' | base64 -d)\" recover '$marker' '$role'" \
    >"$output"
  python3 - "$output" "$role" <<'PY'
import json, sys
from pathlib import Path
value=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
checks=(value.get('status')=='PASS' and value.get('action')=='RECOVER'
        and value.get('role')==sys.argv[2] and value.get('workload_started') is False)
raise SystemExit(0 if checks else 75)
PY
}
