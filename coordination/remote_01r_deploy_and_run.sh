#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 6 ]] || { echo "remote 01R runner: invalid arguments" >&2; exit 64; }
stage="$1"
root="$2"
profile="$3"
result_rel="$4"
grant_id="$5"
expected_sha="$6"
readonly ref="refs/heads/codex/parallel-integration"

case "$profile" in bootstrap|soak|completion_sim|completion_sim_map) ;; *) echo "unsafe profile" >&2; exit 64 ;; esac
[[ "$grant_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{7,95}$ ]] || {
  echo "unsafe grant-id" >&2; exit 64;
}
[[ "$expected_sha" =~ ^[0-9a-f]{40}$ ]] || { echo "unsafe SHA" >&2; exit 64; }
[[ "$root" == "/home/song/internnav-t1-t2" ]] || { echo "unexpected root" >&2; exit 64; }
[[ "$stage" == "/home/song/.codex-internnav-stage/01r-${profile}-${grant_id}" ]] || {
  echo "unexpected stage" >&2; exit 64;
}
if [[ "$profile" == "completion_sim_map" ]]; then
  [[ "$result_rel" == "results/parallel/t4_map/online-smoke-10-${grant_id}" ]] || {
    echo "unexpected map result path" >&2; exit 64;
  }
else
  [[ "$result_rel" == "results/parallel/sensor_producer/online-${profile}-01r-${grant_id}" ]] || {
    echo "unexpected result path" >&2; exit 64;
  }
fi
[[ -d "$root" && ! -L "$root" && "$(readlink -f "$root")" == "$root" ]] || {
  echo "control root is not a real directory" >&2; exit 64;
}
[[ -d "$stage" && ! -L "$stage" && "$(readlink -f "$stage")" == "$stage" ]] || {
  echo "stage is not a real directory" >&2; exit 64;
}

result_abs="$root/$result_rel"
backup="/home/song/.codex-internnav-backups/01r-${profile}-${grant_id}"
receipt="$stage/deployment_receipt.txt"
phase="preflight"
deployment_items=(sensor_runtime scripts go2_sensor_bridge coordination t4_completion configs .git)

write_receipt() {
  local rc="$1"
  {
    printf 'schema_version=1\n'
    printf 'phase=%s\n' "$phase"
    printf 'exit_code=%s\n' "$rc"
    printf 'profile=%s\n' "$profile"
    printf 'grant_id=%s\n' "$grant_id"
    printf 'expected_ref_sha=%s\n' "$expected_sha"
    printf 'result_dir=%s\n' "$result_rel"
    printf 'backup_dir=%s\n' "$backup"
    printf 'recorded_at=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
  } >"$receipt"
}

rollback_deployment() {
  local failed="$stage/failed-deployment" item index
  [[ "$phase" == "deploying" ]] || return 0
  mkdir -p -- "$failed"
  # Move every partially deployed item aside, including paths that did not
  # exist before deployment, then restore any original item from the backup.
  for (( index=${#deployment_items[@]}-1; index>=0; index-- )); do
    item="${deployment_items[$index]}"
    if [[ -e "$root/$item" || -L "$root/$item" ]]; then
      mv -- "$root/$item" "$failed/$item"
    fi
    if [[ -e "$backup/$item" || -L "$backup/$item" ]]; then
      mv -- "$backup/$item" "$root/$item"
    fi
  done
  phase="deployment_rolled_back"
}

on_exit() {
  local rc=$?
  trap - EXIT INT TERM HUP
  set +e
  rollback_deployment
  write_receipt "$rc"
  exit "$rc"
}
trap on_exit EXIT
trap 'exit 130' INT TERM HUP

cd "$stage"
sha256sum -c SHA256SUMS
[[ ! -e "$result_abs" ]] || { echo "remote result path already exists" >&2; exit 74; }
[[ ! -e "$backup" ]] || { echo "remote backup path already exists" >&2; exit 74; }
find "$root" -maxdepth 1 -type f \
  \( -name '*.py' -o -name '*.pyc' -o -name '*.pyo' -o -name '*.so' -o -name '*.pyd' \) \
  -print -quit | grep -q . && { echo "root import shadow exists" >&2; exit 65; }

test "$(docker inspect -f '{{.State.Running}}' internnav_t4_isaac_ros)" = true
docker exec --user admin --workdir "$root" internnav_t4_isaac_ros \
  test -d "$root/scripts"

install -d -m 700 /home/song/.codex-internnav-backups
mkdir -- "$backup"
phase="deploying"
for item in "${deployment_items[@]}"; do
  if [[ -e "$root/$item" || -L "$root/$item" ]]; then
    mv -- "$root/$item" "$backup/$item"
  fi
done

tar -C "$root" -xf "$stage/payload.tar"
git -C "$root" init -q
git -C "$root" fetch --no-tags --force "$stage/integration.bundle" \
  "$ref:$ref"
git -C "$root" symbolic-ref HEAD "$ref"
actual_sha="$(git -C "$root" rev-parse "$ref")"
[[ "$actual_sha" == "$expected_sha" ]] || { echo "deployed ref mismatch" >&2; exit 65; }

PYTHONPATH="$root" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  python3 - "$root" "$expected_sha" "$profile" <<'PY'
import sys
from pathlib import Path
from sensor_runtime.session import verify_source_provenance

root, ref_sha, profile = sys.argv[1:]
evidence = verify_source_provenance(
    Path(root), ref_sha, include_map=profile == "completion_sim_map"
)
if evidence.get("verified") is not True or evidence.get("ref_sha") != ref_sha:
    raise SystemExit("source provenance did not prove the deployed ref")
print(f"SOURCE_PROVENANCE_OK paths={len(evidence['critical_paths'])} ref={ref_sha}")
PY

docker exec --user admin --workdir "$root" internnav_t4_isaac_ros \
  test -f "$root/sensor_runtime/ros_inner_supervisor.py"

phase="online_started"
write_receipt 0
set +e
if [[ "$profile" == "completion_sim_map" ]]; then
  INTERNNAV_SENSOR_SESSION_LEASE_ACK=1 \
  INTERNNAV_T1_CONTROL_ROOT="$root" \
  PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
    bash "$root/scripts/t4_map_online_smoke.sh" "$result_rel" "$grant_id"
else
  INTERNNAV_SENSOR_SESSION_LEASE_ACK=1 \
  INTERNNAV_T1_CONTROL_ROOT="$root" \
  PYTHONNOUSERSITE=1 \
  PYTHONDONTWRITEBYTECODE=1 \
    bash "$root/sensor_runtime/run_model_free_sensor_${profile}.sh" \
      "$result_rel" "$grant_id"
fi
session_rc=$?
set -e
phase="online_finished"
write_receipt "$session_rc"
exit "$session_rc"
