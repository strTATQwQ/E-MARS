#!/usr/bin/env python3
"""Derive the T5 evaluator-only phase with both ROS clients on its Lane DGX."""

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


def _replace_slice(text: str, start: str, end: str, replacement: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[:start_index] + replacement + text[end_index:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    arguments = parser.parse_args()

    source = arguments.source.resolve()
    base_builder = Path(__file__).with_name(
        "build_t4_isaac_remote_phase_overlay.py"
    )
    with tempfile.TemporaryDirectory(prefix="t5_isaac_phase_") as temporary:
        inherited_output = Path(temporary) / "t4_remote.sh"
        inherited_manifest = Path(temporary) / "t4_remote_manifest.json"
        subprocess.run(
            [
                sys.executable,
                str(base_builder),
                "--source",
                str(source),
                "--output",
                str(inherited_output),
                "--manifest",
                str(inherited_manifest),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        text = inherited_output.read_text(encoding="utf-8")
        inherited = json.loads(inherited_manifest.read_text(encoding="utf-8"))

    inherited_control_roots = (
        'CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"\n'
        'T0_CONTROL_ROOT="${INTERNNAV_T0_CONTROL_ROOT:-$HOME/internnav-t0/control}"'
    )
    exact_deployment_control_roots = (
        'CONTROL_ROOT="${INTERNNAV_T1_CONTROL_ROOT:-$HOME/internnav-t1-t2}"\n'
        '# T5 deployments carry their own frozen official T0 overlay.  Never\n'
        '# fall back to the mutable shared host control tree for a Lane run.\n'
        'T0_CONTROL_ROOT="$CONTROL_ROOT"\n'
        'export INTERNNAV_T0_CONTROL_ROOT="$CONTROL_ROOT"'
    )
    if text.count(inherited_control_roots) != 1:
        raise RuntimeError("T5 phase expected one inherited control-root binding")
    text = text.replace(
        inherited_control_roots, exact_deployment_control_roots, 1
    )

    inherited_pythonpath = (
        'ISAAC_PYTHONPATH="$COMPAT_OVERLAY:$INTERNNAV_ROOT:$SCRIPT_ROOT:'
        '$CONTROL_ROOT/internvla_go2_controller"'
    )
    deployment_pythonpath = (
        ': "${INTERNVLA_T5_ISAAC_PYTHON_PACKAGE_ROOT:?T5 deployment Python '
        'package root is required}"\n'
        'test -f "$INTERNVLA_T5_ISAAC_PYTHON_PACKAGE_ROOT/internvla_ros2/'
        'fault_injection.py"\n'
        'ISAAC_PYTHONPATH="$INTERNVLA_T5_ISAAC_PYTHON_PACKAGE_ROOT:'
        '$COMPAT_OVERLAY:$INTERNNAV_ROOT:$SCRIPT_ROOT:'
        '$CONTROL_ROOT/internvla_go2_controller"'
    )
    if text.count(inherited_pythonpath) != 1:
        raise RuntimeError("T5 phase expected one inherited Isaac PYTHONPATH")
    text = text.replace(inherited_pythonpath, deployment_pythonpath, 1)

    text = text.replace(
        'test "${INTERNNAV_T4_RESOURCE_LEASE_ACK:-}" = dgx+isaac\n'
        'DGX_CONTROLLER_IP="${INTERNVLA_T4_DGX_BIND_IP:-10.100.100.128}"',
        'case "${INTERNVLA_T5_EXPECTED_LEASE_ACK:-}" in lane-a|lane-b) ;; *) exit 2 ;; esac\n'
        'test "${INTERNNAV_T5_RESOURCE_LEASE_ACK:-}" = "$INTERNVLA_T5_EXPECTED_LEASE_ACK"\n'
        'case "${INTERNVLA_T5_EXPECTED_LEASE_ACK}:${INTERNNAV_T5_ID_PREFIX:-}" in\n'
        '  lane-a:a\\:\\:|lane-b:b\\:\\:) ;;\n'
        '  *) echo "invalid T5 lane lease/identity pair" >&2; exit 2 ;;\n'
        'esac\n'
        'DGX_CONTROLLER_IP="${INTERNVLA_T4_DGX_BIND_IP:-10.100.100.128}"',
        1,
    )
    remote_client = r'''LANE_CLIENT_IP="${INTERNVLA_T5_LANE_IP:-10.100.100.128}"
LANE_MODEL_CLIENT_PORT="${INTERNVLA_T5_MODEL_CLIENT_PORT:-25139}"
LANE_ORACLE_PORT="${INTERNVLA_T5_ORACLE_PORT:-25140}"
[[ "$LANE_CLIENT_IP" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]
[[ "$LANE_MODEL_CLIENT_PORT" =~ ^[0-9]+$ ]]
[[ "$LANE_ORACLE_PORT" =~ ^[0-9]+$ ]]
if test "$CLIENT_KIND" = oracle; then
  LANE_CLIENT_PORT="$LANE_ORACLE_PORT"
  export INTERNVLA_ORACLE_ENDPOINT="tcp://$LANE_CLIENT_IP:$LANE_CLIENT_PORT"
else
  LANE_CLIENT_PORT="$LANE_MODEL_CLIENT_PORT"
  export INTERNVLA_CLIENT_ENDPOINT="tcp://$LANE_CLIENT_IP:$LANE_CLIENT_PORT"
fi
((LANE_CLIENT_PORT >= 1024 && LANE_CLIENT_PORT <= 65535))
# Do not health-probe either single-client framed data plane with a bare TCP
# connect.  The real evaluator connection is retained by its AgentClient, and
# runtime READY is proven later by a real sensor ACK plus a successful typed
# model/oracle action marker.

'''
    text = _replace_slice(
        text,
        'if test "$CLIENT_KIND" = oracle; then\n  INTERNVLA_ORACLE_SOCKET=',
        "graph_ok=0\n",
        remote_client,
    )
    text = text.replace(
        'timeout 3 bash -c "exec 3<>/dev/tcp/$DGX_CONTROLLER_IP/'
        '$DGX_CONTROLLER_PORT; exec 3>&-"\n',
        "",
    )
    materialize_order = r'''if test "$CLIENT_KIND" = model; then
  : "${INTERNVLA_MODEL_EPISODE_ORDER_MANIFEST:?model order manifest path is required}"
  test ! -e "$INTERNVLA_MODEL_EPISODE_ORDER_MANIFEST"
  PYTHONPATH="$ISAAC_PYTHONPATH" "$ISAAC_PYTHON" \
    "$SCRIPT_ROOT/materialize_t5_episode_order.py" \
    --runtime-overlay "$SCRIPT_ROOT/internnav_go2_runtime.py" \
    --config "$CONFIG" \
    --dataset-file "$DATASET_ROOT/val_unseen/val_unseen.json.gz" \
    --expected-count "$EXPECTED_COUNT" \
    --output "$INTERNVLA_MODEL_EPISODE_ORDER_MANIFEST"
  test -s "$INTERNVLA_MODEL_EPISODE_ORDER_MANIFEST"
fi

'''
    preflight = (
        'set +e\nPYTHONPATH="$ISAAC_PYTHONPATH" "$ISAAC_PYTHON" \\\n'
        '  "$SCRIPT_ROOT/preflight_go2_continuous.py" --config "$CONFIG" \\\n'
    )
    if text.count(preflight) != 1:
        raise RuntimeError("T5 phase expected one evaluator preflight launch")
    text = text.replace(preflight, materialize_order + preflight, 1)
    text = text.replace(
        '  export INTERNVLA_ORACLE_SOCKET="$ORACLE_SOCKET"\n',
        '  export INTERNVLA_ORACLE_ENDPOINT\n',
        1,
    )
    text = text.replace(
        '  export INTERNVLA_CLIENT_SOCKET="$AGENT_SOCKET"\n',
        '  export INTERNVLA_CLIENT_ENDPOINT\n',
        1,
    )
    text = text.replace(
        'stop_group "$client_pid"; client_pid=""\n'
        'if test "$CLIENT_KIND" = model; then\n'
        '  test -f "$RESULT_DIR/client_summary.json"\n'
        'fi\n',
        '# The complete DGX lane owns the client and archives its summary separately.\n',
        1,
    )
    text = text.replace(
        '"host_role": "isaac_sim_sensor_client",',
        '"host_role": "isaac_x86_simulator_only",',
        1,
    )
    text = text.replace(
        '"controller_endpoint": "dgx_onboard_tcp",',
        '"controller_endpoint": "dgx_lane_tcp",\n'
        '    "remote_client_endpoint": "dgx_lane_tcp",',
        1,
    )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.manifest.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(text, encoding="utf-8", newline="\n")
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "source_sha256": _sha256(source),
        "inherited_t4_remote_overlay_sha256": inherited["output_sha256"],
        "output_sha256": _sha256(arguments.output),
        "isaac_x86_role": [
            "isaac_sim",
            "go2_physics",
            "simulated_sensors",
            "evaluator",
            "clock_source",
        ],
        "dgx_lane_role": [
            "internvla_model",
            "model_client",
            "oracle_client",
            "navigation",
            "map",
            "speed_control",
        ],
        "local_ros_navigation_or_client_processes_started": False,
        "split_model_edge_topology_used": False,
        "t0_control_root_binding": "exact_t5_deployment_control_root",
        "real_go2_targeted": False,
    }
    arguments.manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
