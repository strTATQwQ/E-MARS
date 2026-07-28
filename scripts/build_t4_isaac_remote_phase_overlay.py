#!/usr/bin/env python3
"""Derive an Isaac sensor/evaluator-only T4 phase for DGX onboard compute."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_slice(text: str, start_token: str, end_token: str, replacement: str) -> str:
    start = text.index(start_token)
    end = text.index(end_token, start)
    return text[:start] + replacement + text[end:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    base_builder = Path(__file__).with_name("build_t4_sensor_phase_overlay.py")
    with tempfile.TemporaryDirectory(prefix="t4_isaac_remote_phase_") as temporary:
        generated = Path(temporary) / "t4_local_phase.sh"
        generated_manifest = Path(temporary) / "t4_local_phase_manifest.json"
        subprocess.run(
            [
                sys.executable,
                str(base_builder),
                "--source",
                str(source),
                "--output",
                str(generated),
                "--manifest",
                str(generated_manifest),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        text = generated.read_text(encoding="utf-8")
        inherited = json.loads(generated_manifest.read_text(encoding="utf-8"))

    remote_compute = r'''test "${INTERNNAV_T4_RESOURCE_LEASE_ACK:-}" = dgx+isaac
DGX_CONTROLLER_IP="${INTERNVLA_T4_DGX_BIND_IP:-10.100.100.128}"
DGX_CONTROLLER_PORT="${INTERNVLA_T4_CONTROLLER_TCP_PORT:-24137}"
[[ "$DGX_CONTROLLER_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]
[[ "$DGX_CONTROLLER_PORT" =~ ^[0-9]+$ ]]
((DGX_CONTROLLER_PORT >= 1024 && DGX_CONTROLLER_PORT <= 65535))
export INTERNVLA_GO2_CONTROLLER_ENDPOINT="tcp://$DGX_CONTROLLER_IP:$DGX_CONTROLLER_PORT"

# The migration boundary is fail-closed: no Nav2, map, sensor bridge, recovery,
# or velocity relay may be started on Isaac in this phase.  Only the local
# evaluator IPC bridge remains here; all ROS navigation consumers are remote.
timeout 3 bash -c "exec 3<>/dev/tcp/$DGX_CONTROLLER_IP/$DGX_CONTROLLER_PORT; exec 3>&-"

'''
    text = _replace_slice(
        text,
        "setsid ros2 launch nav2_bringup navigation_launch.py",
        'if test "$CLIENT_KIND" = oracle; then\n',
        remote_compute,
    )

    remote_readiness = r'''for _ in $(seq 1 600); do
  test -S "$IPC_SOCKET" && break
  kill -0 "$client_pid"
  sleep 0.1
done
test -S "$IPC_SOCKET"
kill -0 "$client_pid"
timeout 3 bash -c "exec 3<>/dev/tcp/$DGX_CONTROLLER_IP/$DGX_CONTROLLER_PORT; exec 3>&-"

'''
    text = _replace_slice(
        text,
        "for _ in $(seq 1 600); do\n",
        "graph_ok=0\n",
        remote_readiness,
    )
    text = text.replace(
        'export INTERNVLA_GO2_CONTROLLER_SOCKET="$CONTROLLER_SOCKET"',
        'export INTERNVLA_GO2_CONTROLLER_ENDPOINT',
        1,
    )

    remote_validation = r'''stop_group "$client_pid"; client_pid=""
if test "$CLIENT_KIND" = model; then
  test -f "$RESULT_DIR/client_summary.json"
fi
set +e
python3 - "$RESULT_DIR" "$PHASE" "$EXPECTED_COUNT" "$MIN_SR" "$EVAL_EXIT" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
phase = sys.argv[2]
expected = int(sys.argv[3])
minimum_sr = float(sys.argv[4])
evaluator_exit = int(sys.argv[5])
result = json.loads((root / "result.json").read_text(encoding="utf-8"))
metrics = result.get("val_unseen", result)
count = int(metrics.get("Count", metrics.get("length", 0)))
sr = float(metrics.get("SR", metrics.get("sr", 0.0)))
passing = evaluator_exit == 0 and count == expected and math.isfinite(sr) and sr >= minimum_sr
payload = {
    "schema_version": 1,
    "status": "PASS" if passing else "FAIL",
    "phase": phase,
    "host_role": "isaac_sim_sensor_client",
    "evaluator_exit": evaluator_exit,
    "episode_count": count,
    "expected_episode_count": expected,
    "success_rate": sr,
    "minimum_success_rate": minimum_sr,
    "controller_endpoint": "dgx_onboard_tcp",
    "local_navigation_processes_started": False,
    "local_speed_control_processes_started": False,
}
(root / "isaac_remote_validation.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(0 if passing else 1)
PY
VALIDATION_EXIT=$?
set -e
test "$VALIDATION_EXIT" = 0

'''
    text = _replace_slice(
        text,
        'stop_group "$relay_pid"; relay_pid=""\n',
        'for file in "$RESULT_DIR"/logs/*.log; do\n',
        remote_validation,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8", newline="\n")
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "source_sha256": _sha256(source),
        "inherited_overlay_sha256": inherited["output_sha256"],
        "output_sha256": _sha256(args.output),
        "isaac_role": ["simulator", "sensor_rendering", "evaluator", "local_agent_ipc"],
        "dgx_role": ["navigation", "map", "speed_control", "recovery", "model"],
        "controller_transport": "bounded_tcp_ipv4",
        "local_navigation_processes_started": False,
        "local_speed_control_processes_started": False,
    }
    args.manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
