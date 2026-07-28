#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: finalize_t5_paired10_replay.sh results/internnav_t5/paired10-retrospective-RUN_ID

Serial post-processing for a completed two-round paired-10 run.  The paired
execution summary must be PASS and must bind all four final10 child results.
For each child this script:

  * derives the authoritative x86 run root from both input_binding.json and
    fast_lane_final_summary.json;
  * takes the existing /tmp/internnav_t5_isaac_shared_assets.lock exclusively;
  * finalizes the independent D435 stream as a fixed 5 Hz review MP4;
  * streams an uncompressed tar of the selected replay evidence into the
    child's remote/x86 directory without deleting the remote originals; and
  * builds replay/timeline.jsonl plus replay/timeline_index.json locally.

PAIRED_RESULT_ROOT may be supplied instead of the positional argument.  This
script uses the configured SSH identity only; it never reads .env.local or a
password.
EOF
  exit 64
}

if [[ $# -eq 0 && -n "${PAIRED_RESULT_ROOT:-}" ]]; then
  set -- "$PAIRED_RESULT_ROOT"
fi
[[ $# -eq 1 ]] || usage
paired_relative="$1"
[[ "$paired_relative" =~ ^results/internnav_t5/paired10-retrospective-[a-z0-9][a-z0-9._-]{7,63}$ ]] || usage

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
paired_root="$root/$paired_relative"
paired_summary="$paired_root/paired10_execution_summary.json"
x86_target="song@10.100.120.123"
shared_io_lock=/tmp/internnav_t5_isaac_shared_assets.lock
ffmpeg=/home/song/opt/keyshot-network-2026.1/keyshot_network/bin/ffmpeg
ssh_options=(
  -T
  -i "${INTERNNAV_T5_SSH_IDENTITY_FILE:-$HOME/.ssh/id_ed25519_internnav_runtime}"
  -o IdentitiesOnly=yes
  -o BatchMode=yes
  -o ConnectTimeout=8
  -o ServerAliveInterval=5
  -o ServerAliveCountMax=2
)

test -d "$paired_root"
test ! -L "$paired_root"
test -f "$paired_summary"
test ! -L "$paired_summary"
test -f "$root/scripts/finalize_t5_d435_capture.py"
test ! -L "$root/scripts/finalize_t5_d435_capture.py"
test -f "$root/scripts/build_t5_unified_replay.py"
test ! -L "$root/scripts/build_t5_unified_replay.py"
command -v ssh >/dev/null
command -v tar >/dev/null
command -v python3 >/dev/null

mapfile -t child_specs < <(python3 - "$paired_summary" "$root" <<'PY'
import json
import re
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
root = Path(sys.argv[2]).resolve()
summary = json.loads(summary_path.read_text(encoding="utf-8"))
if summary.get("status") != "PASS" or summary.get("execution_count") != 40:
    raise SystemExit("paired10 execution summary is not a complete PASS")
code_sha = summary.get("code_ref_sha")
if not isinstance(code_sha, str) or re.fullmatch(r"[0-9a-f]{40}", code_sha) is None:
    raise SystemExit("paired10 execution summary has an invalid code SHA")

expected = (
    ("round1", "lane_a", "a", "internvla_only"),
    ("round1", "lane_b", "b", "internvla_step3"),
    ("round2", "lane_a", "a", "internvla_step3"),
    ("round2", "lane_b", "b", "internvla_only"),
)
lane_results = summary.get("lane_results")
if not isinstance(lane_results, dict):
    raise SystemExit("paired10 lane_results is absent")

for round_id, lane_key, lane, expected_arm in expected:
    round_value = lane_results.get(round_id)
    if not isinstance(round_value, dict) or round_value.get("status") != "PASS":
        raise SystemExit(f"{round_id} is not PASS")
    child_relative = round_value.get(lane_key)
    if not isinstance(child_relative, str) or re.fullmatch(
        rf"results/internnav_t5/fast-lane-{lane}-final10-[a-z0-9][a-z0-9._-]{{7,95}}",
        child_relative,
    ) is None:
        raise SystemExit(f"unsafe {round_id}/{lane_key} child result")
    child = (root / child_relative).resolve()
    try:
        child.relative_to(root / "results" / "internnav_t5")
    except ValueError as error:
        raise SystemExit(f"child result escapes results root: {child}") from error
    if not child.is_dir() or child.is_symlink():
        raise SystemExit(f"child result is not a regular directory: {child}")
    fast_path = child / "fast_lane_final_summary.json"
    binding_path = child / "input_binding.json"
    if any(path.is_symlink() or not path.is_file() for path in (fast_path, binding_path)):
        raise SystemExit(f"child summary/binding is missing or unsafe: {child_relative}")
    fast = json.loads(fast_path.read_text(encoding="utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    runtime = fast.get("runtime_summary")
    if (
        fast.get("status") != "PASS"
        or not isinstance(runtime, dict)
        or runtime.get("status") != "PASS"
        or runtime.get("code_ref_sha") != code_sha
    ):
        raise SystemExit(f"child final summary is not bound to paired run: {child_relative}")
    if runtime.get("input_binding") != binding:
        raise SystemExit(f"embedded and materialized input bindings differ: {child_relative}")
    if binding.get("lane") != lane or binding.get("evaluation_arm") != expected_arm:
        raise SystemExit(f"child lane/evaluation arm drift: {child_relative}")
    deployment = (binding.get("deployment_roots") or {}).get("x86")
    remote_root = (runtime.get("raw_log_locations") or {}).get(
        "remote_x86_result_root"
    )
    safe_remote = re.compile(r"/home/song/internnav-t1-t2/\.t5-deployments/[A-Za-z0-9._/-]+")
    if not isinstance(deployment, str) or safe_remote.fullmatch(deployment) is None:
        raise SystemExit(f"unsafe x86 deployment root: {child_relative}")
    expected_remote = (
        f"{deployment}/results/t5_fast_final10_{runtime.get('run_id')}"
    )
    if remote_root != expected_remote or safe_remote.fullmatch(remote_root or "") is None:
        raise SystemExit(f"x86 raw run root does not derive from input binding: {child_relative}")
    fields = (
        f"{round_id}_{lane_key}", child_relative, str(runtime.get("run_id")), lane,
        expected_arm, deployment, remote_root, code_sha,
    )
    if any("\t" in field or "\n" in field for field in fields):
        raise SystemExit("unsafe delimiter in child specification")
    print("\t".join(fields))
PY
)
test "${#child_specs[@]}" = 4

read -r -d '' remote_archive_program <<'REMOTE' || true
set -euo pipefail
remote_root="$1"
deployment_root="$2"
code_sha="$3"
ffmpeg="$4"
lock_path="$5"

test "$(id -un)" = song
case "$deployment_root" in
  /home/song/internnav-t1-t2/.t5-deployments/*) ;;
  *) exit 64 ;;
esac
case "$remote_root" in
  "$deployment_root"/results/t5_fast_final10_*) ;;
  *) exit 64 ;;
esac
test -d "$deployment_root" && test ! -L "$deployment_root"
test "$(realpath "$deployment_root")" = "$deployment_root"
test -d "$remote_root" && test ! -L "$remote_root"
test "$(realpath "$remote_root")" = "$remote_root"
test -f "$deployment_root/T5_DEPLOYMENT_REF" && test ! -L "$deployment_root/T5_DEPLOYMENT_REF"
test "$(tr -d '\r\n' <"$deployment_root/T5_DEPLOYMENT_REF")" = "$code_sha"
finalizer="$deployment_root/scripts/finalize_t5_d435_capture.py"
test -f "$finalizer" && test ! -L "$finalizer"
test -x "$ffmpeg" && test ! -L "$ffmpeg"
test ! -L "$lock_path"

exec 9>"$lock_path"
flock -n 9

# Encoding and archive reads are forbidden while either Isaac worker is live.
for container in internnav_t5_isaac_a internnav_t5_isaac_b; do
  if docker inspect "$container" >/dev/null 2>&1; then
    test "$(docker inspect -f '{{.State.Running}}' "$container")" = false
    test "$(docker inspect -f '{{.State.Pid}}' "$container")" = 0
  fi
done

capture="$remote_root/evaluator/d435_rgb_5hz"
model_frames="$remote_root/evaluator/full_rgb_5hz"
test -d "$capture" && test ! -L "$capture"
test -d "$model_frames" && test ! -L "$model_frames"
test -z "$(find "$capture" "$model_frames" -type l -print -quit)"

video="$capture/full_d435_rgb_5hz.mp4"
sidecar="$capture/video_sidecar.json"
if test -e "$video" || test -e "$sidecar"; then
  test -f "$video" && test ! -L "$video"
  test -f "$sidecar" && test ! -L "$sidecar"
  python3 "$finalizer" --capture-root "$capture" --validate-only >/dev/null
  python3 - "$capture" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sidecar = json.loads((root / "video_sidecar.json").read_text(encoding="utf-8"))
video = root / "full_d435_rgb_5hz.mp4"
digest = hashlib.sha256()
with video.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
assert sidecar.get("status") == "COMPLETE"
assert (sidecar.get("video") or {}).get("sha256") == digest.hexdigest()
assert (sidecar.get("video") or {}).get("bytes") == video.stat().st_size
PY
else
  python3 "$finalizer" --capture-root "$capture" --ffmpeg "$ffmpeg" >/dev/null
fi

selected=(evaluator/d435_rgb_5hz evaluator/full_rgb_5hz)
for relative in \
  evaluator/revc_snapshots \
  evaluator/step3_timeout_advice.jsonl \
  evaluator/task_state; do
  candidate="$remote_root/$relative"
  if test -e "$candidate"; then
    test ! -L "$candidate"
    if test -d "$candidate"; then
      test -z "$(find "$candidate" -type l -print -quit)"
    else
      test -f "$candidate"
    fi
    selected+=("$relative")
  fi
done

# Stdout is exclusively the uncompressed tar stream.  Keep the lock for the
# full read so another archival or shader/asset I/O job cannot overlap it.
tar -C "$remote_root" -cf - "${selected[@]}"
REMOTE

postprocess_receipts=()
for specification in "${child_specs[@]}"; do
  IFS=$'\t' read -r label child_relative child_run_id lane evaluation_arm \
    deployment_root remote_root code_sha <<<"$specification"
  child="$root/$child_relative"
  destination="$child/remote/x86"
  remote_log="$child/postprocess_remote_x86.stderr.log"
  child_receipt="$child/postprocess_receipt.json"
  test ! -L "$destination"
  mkdir -p "$destination"
  test -z "$(find "$destination" -type l -print -quit)"
  test ! -L "$remote_log"
  test ! -L "$child_receipt"

  # Deliberately stream without gzip and without a local tar staging file.  If
  # either side fails (including ENOSPC), pipefail aborts and the remote source
  # remains untouched for a later retry.
  ssh "${ssh_options[@]}" "$x86_target" bash -s -- \
      "$remote_root" "$deployment_root" "$code_sha" "$ffmpeg" "$shared_io_lock" \
      <<<"$remote_archive_program" 2>"$remote_log" \
    | tar -C "$destination" --extract --file - --overwrite \
        --no-same-owner --no-same-permissions
  test -z "$(find "$destination" -type l -print -quit)"

  python3 "$root/scripts/build_t5_unified_replay.py" "$child" \
    >"$child/replay_build_stdout.json"
  test -s "$child/replay/timeline.jsonl"
  test -s "$child/replay/timeline_index.json"

  python3 - "$child" "$label" "$child_relative" "$child_run_id" "$lane" \
    "$evaluation_arm" "$deployment_root" "$remote_root" "$code_sha" \
    "$shared_io_lock" "$ffmpeg" "$child_receipt" <<'PY'
import hashlib
import json
import os
import sys
import time
from pathlib import Path

(
    child, label, child_relative, child_run_id, lane, evaluation_arm,
    deployment_root, remote_root, code_sha, lock_path, ffmpeg, output,
) = sys.argv[1:]
child_path = Path(child).resolve()
output_path = Path(output)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def inventory(relative: str) -> dict:
    path = child_path / "remote" / "x86" / relative
    if not path.exists():
        return {"path": f"remote/x86/{relative}", "present": False}
    if path.is_symlink():
        raise SystemExit(f"symlink appeared in materialized evidence: {path}")
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    if any(item.is_symlink() for item in files):
        raise SystemExit(f"symlink appeared below materialized evidence: {path}")
    return {
        "path": f"remote/x86/{relative}",
        "present": True,
        "file_count": len(files),
        "bytes": sum(item.stat().st_size for item in files),
    }

d435 = child_path / "remote/x86/evaluator/d435_rgb_5hz"
video = d435 / "full_d435_rgb_5hz.mp4"
video_sidecar_path = d435 / "video_sidecar.json"
if any(path.is_symlink() or not path.is_file() for path in (video, video_sidecar_path)):
    raise SystemExit("finalized D435 MP4/sidecar was not materialized")
video_sidecar = json.loads(video_sidecar_path.read_text(encoding="utf-8"))
video_binding = video_sidecar.get("video") or {}
if video_sidecar.get("status") != "COMPLETE":
    raise SystemExit("D435 video sidecar is not COMPLETE")
if video_binding.get("sha256") != sha256(video) or video_binding.get("bytes") != video.stat().st_size:
    raise SystemExit("materialized D435 MP4 differs from its remote sidecar")

timeline = child_path / "replay/timeline.jsonl"
index_path = child_path / "replay/timeline_index.json"
if any(path.is_symlink() or not path.is_file() for path in (timeline, index_path)):
    raise SystemExit("unified replay outputs are absent or unsafe")
index = json.loads(index_path.read_text(encoding="utf-8"))
if index.get("timeline_sha256") != sha256(timeline):
    raise SystemExit("unified replay timeline hash does not match its index")

evidence = [
    inventory("evaluator/d435_rgb_5hz"),
    inventory("evaluator/full_rgb_5hz"),
    inventory("evaluator/revc_snapshots"),
    inventory("evaluator/step3_timeout_advice.jsonl"),
    inventory("evaluator/task_state"),
]
checks = {
    "d435_5hz_mp4_complete": True,
    "model_observation_stream_present": evidence[1]["present"],
    "remote_originals_retained": True,
    "uncompressed_stream_transfer": True,
    "unified_timeline_built": index.get("timeline_sha256") == sha256(timeline),
    "time_authority_is_x86_sim_stamp": index.get("time_authority") == "x86_sim_stamp_ns",
    "wall_latency_summary_recorded": isinstance(index.get("wall_latency_summary"), dict),
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "stage": "t5_paired10_child_replay_postprocess",
    "child_label": label,
    "child_result": child_relative,
    "run_id": child_run_id,
    "lane": lane,
    "evaluation_arm": evaluation_arm,
    "code_ref_sha": code_sha,
    "source": {
        "host": "song@10.100.120.123",
        "deployment_root": deployment_root,
        "remote_x86_result_root": remote_root,
        "remote_originals_retained": True,
    },
    "shared_io": {
        "lock": lock_path,
        "scope": "finalize_then_uncompressed_tar_stream",
        "ffmpeg": ffmpeg,
        "online_encoding_allowed": False,
    },
    "capture_inventory": evidence,
    "d435_video": {
        "path": "remote/x86/evaluator/d435_rgb_5hz/full_d435_rgb_5hz.mp4",
        "sha256": sha256(video),
        "bytes": video.stat().st_size,
        "frame_count": video_binding.get("frame_count"),
        "fps": video_binding.get("fps"),
        "sidecar_sha256": sha256(video_sidecar_path),
    },
    "unified_replay": {
        "timeline": "replay/timeline.jsonl",
        "timeline_sha256": index.get("timeline_sha256"),
        "timeline_index": "replay/timeline_index.json",
        "timeline_index_sha256": sha256(index_path),
        "event_count": index.get("event_count"),
        "wall_latency_summary": index.get("wall_latency_summary"),
    },
    "checks": checks,
    "recorded_unix": time.time(),
}
temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
os.chmod(temporary, 0o600)
os.replace(temporary, output_path)
if payload["status"] != "PASS":
    raise SystemExit("child replay postprocess did not pass")
PY
  postprocess_receipts+=("$child_receipt")
done

paired_analysis="$paired_root/paired10_analysis.json"
python3 "$root/scripts/summarize_t5_paired10.py" "$paired_root" \
  --repo-root "$root" --output "$paired_analysis" \
  >"$paired_root/paired10_analysis_stdout.json"
python3 -c 'import json,sys; value=json.load(open(sys.argv[1],encoding="utf-8")); assert value.get("status")=="COMPLETE"' \
  "$paired_analysis"

paired_receipt="$paired_root/paired10_postprocess_receipt.json"
test ! -L "$paired_receipt"
python3 - "$paired_summary" "$paired_receipt" "$paired_analysis" \
  "${postprocess_receipts[@]}" <<'PY'
import hashlib
import json
import os
import sys
import time
from pathlib import Path

summary_path = Path(sys.argv[1])
output = Path(sys.argv[2])
analysis_path = Path(sys.argv[3])
receipt_paths = [Path(value) for value in sys.argv[4:]]
summary = json.loads(summary_path.read_text(encoding="utf-8"))
analysis = json.loads(analysis_path.read_text(encoding="utf-8"))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

if len(receipt_paths) != 4:
    raise SystemExit("paired postprocess requires exactly four child receipts")
children = []
for path in receipt_paths:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"unsafe child postprocess receipt: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS" or not all((value.get("checks") or {}).values()):
        raise SystemExit(f"child postprocess is not PASS: {path}")
    children.append({
        "child_label": value.get("child_label"),
        "child_result": value.get("child_result"),
        "lane": value.get("lane"),
        "evaluation_arm": value.get("evaluation_arm"),
        "receipt": path.as_posix(),
        "receipt_sha256": sha256(path),
        "d435_video": value.get("d435_video"),
        "wall_latency_summary": (value.get("unified_replay") or {}).get("wall_latency_summary"),
    })

checks = {
    "paired_execution_pass": summary.get("status") == "PASS",
    "four_child_postprocess_receipts": len(children) == 4,
    "all_children_pass": True,
    "paired_analysis_complete": analysis.get("status") == "COMPLETE",
    "two_lanes_present": {child["lane"] for child in children} == {"a", "b"},
    "two_arms_present": {child["evaluation_arm"] for child in children}
        == {"internvla_only", "internvla_step3"},
    "remote_originals_retained": True,
}
payload = {
    "schema_version": 1,
    "status": "PASS" if all(checks.values()) else "FAIL",
    "stage": "t5_paired10_replay_postprocess",
    "run_id": summary.get("run_id"),
    "code_ref_sha": summary.get("code_ref_sha"),
    "execution_count": summary.get("execution_count"),
    "unique_episode_count": summary.get("unique_episode_count"),
    "evidence_classification": summary.get("evidence_classification"),
    "paired_execution_summary_sha256": sha256(summary_path),
    "paired_analysis": {
        "path": analysis_path.as_posix(),
        "sha256": sha256(analysis_path),
        "status": analysis.get("status"),
    },
    "shared_io_lock": "/tmp/internnav_t5_isaac_shared_assets.lock",
    "transfer_format": "streamed_uncompressed_tar",
    "children": children,
    "checks": checks,
    "recorded_unix": time.time(),
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
os.chmod(temporary, 0o600)
os.replace(temporary, output)
if payload["status"] != "PASS":
    raise SystemExit("paired replay postprocess did not pass")
PY

printf 'paired10 replay postprocess PASS: %s\n' "$paired_receipt"
