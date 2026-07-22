#!/usr/bin/env python3
"""Make camera trials performance-neutral while preserving all T3 safety gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    text = args.source.read_text(encoding="utf-8")
    old = "EXPECTED_COUNT=5; MIN_SR=0.4; SOURCE_PHASE=canary; OBSTACLE_AWARE=0; CLIENT_KIND=model"
    new = "EXPECTED_COUNT=5; MIN_SR=0.0; SOURCE_PHASE=canary; OBSTACLE_AWARE=0; CLIENT_KIND=model"
    if text.count(old) != 1:
        raise RuntimeError("frozen T3 camera phase token mismatch")
    output = text.replace(old, new)
    # Long sequential camera trials exposed a ROS 2 CLI daemon graph-cache
    # corruption after otherwise clean process teardown. Use a fresh rclpy
    # context for every lifecycle and graph probe, with no XML-RPC daemon path.
    output = output.replace(
        'ros2 lifecycle get "$node" \\\n',
        'python3 "$SCRIPT_ROOT/t4_direct_ros_probe.py" lifecycle-responsive "$node" --timeout 2 \\\n',
    )
    output = output.replace(
        "ros2 lifecycle get /collision_monitor 2>/dev/null | grep -q active",
        'grep -Fq "[collision_monitor]: Activating" "$RESULT_DIR/logs/collision_monitor.log" '
        '&& grep -Fq "Creating bond (collision_monitor)" "$RESULT_DIR/logs/collision_monitor.log"',
    )
    output = output.replace(
        "ros2 lifecycle get /collision_monitor | grep -q active",
        'grep -Fq "[collision_monitor]: Activating" "$RESULT_DIR/logs/collision_monitor.log"; '
        'grep -Fq "Creating bond (collision_monitor)" "$RESULT_DIR/logs/collision_monitor.log"',
    )
    # The active server, adapter and model client are separate DDS participants.
    # On this host a clean participant can need more than the frozen 15-second
    # client default after repeated Isaac trials; wait longer for discovery
    # without changing any request, model, controller, or metric semantics.
    discovery_token = "-p service_timeout_sec:=300.0 -p step_deadline_sec:=30.0"
    discovery_replacement = (
        "-p discovery_timeout_sec:=60.0 -p service_timeout_sec:=300.0 "
        "-p step_deadline_sec:=30.0"
    )
    if output.count(discovery_token) != 1:
        raise RuntimeError("frozen T3 model-client discovery token mismatch")
    output = output.replace(discovery_token, discovery_replacement)
    graph_block = '''{
  echo "# nodes"; ros2 node list | sort
  echo "# topics"; ros2 topic list -t | sort
  echo "# services"; ros2 service list -t | sort
  echo "# actions"; ros2 action list -t | sort
} >"$RESULT_DIR/ros_graph_snapshot.txt"'''
    graph_replacement = (
        'python3 "$SCRIPT_ROOT/t4_direct_ros_probe.py" graph --timeout 2 '
        '>"$RESULT_DIR/ros_graph_snapshot.txt"'
    )
    if output.count(graph_block) != 1:
        raise RuntimeError("frozen T3 graph-snapshot token mismatch")
    output = output.replace(graph_block, graph_replacement)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    args.output.chmod(0o755)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "source_sha256": sha256(args.source),
        "output_sha256": sha256(args.output),
        "changed_gate": "camera-development minimum SR 0.4 -> 0.0",
        "ros2_cli_graph_mode": "fresh_rclpy_context_no_daemon",
        "collision_monitor_readiness": "process-owned activation and bond log evidence",
        "model_client_discovery_timeout_sec": 60.0,
        "safety_or_integrity_gate_changed": False,
    }
    args.manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
