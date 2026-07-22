#!/usr/bin/env python3
"""Fail-closed aggregate lifecycle readiness probe for the DGX onboard stack."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import rclpy
from lifecycle_msgs.srv import GetState
from rclpy.node import Node


NODES = (
    "/controller_server",
    "/smoother_server",
    "/planner_server",
    "/behavior_server",
    "/bt_navigator",
    "/collision_monitor",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--namespace", default="")
    parser.add_argument(
        "--allow-inactive",
        action="append",
        default=[],
        choices=NODES,
        help="completion-only node allowed to remain lifecycle inactive (state 2)",
    )
    args = parser.parse_args()
    if not 1.0 <= args.timeout_sec <= 30.0:
        raise SystemExit("timeout must be between 1 and 30 seconds")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    allowed_inactive = set(args.allow_inactive)
    namespace = args.namespace.rstrip("/")
    if namespace and (not namespace.startswith("/") or "//" in namespace):
        raise SystemExit("namespace must be empty or an absolute ROS namespace")

    def acceptable(name: str, value: dict[str, object]) -> bool:
        state_id = value.get("id")
        return state_id == 3 or (name in allowed_inactive and state_id == 2)

    started = time.monotonic()
    deadline = started + args.timeout_sec
    states: dict[str, dict[str, object]] = {}
    rclpy.init()
    node = Node("internnav_t4_nav2_lifecycle_probe")
    resolved_nodes = {name: f"{namespace}{name}" for name in NODES}
    clients = {
        name: node.create_client(GetState, f"{resolved}/get_state")
        for name, resolved in resolved_nodes.items()
    }
    try:
        while time.monotonic() < deadline:
            states = {}
            for name, client in clients.items():
                if not client.wait_for_service(timeout_sec=0.2):
                    states[name] = {"id": None, "label": "service_unavailable"}
                    continue
                future = client.call_async(GetState.Request())
                rclpy.spin_until_future_complete(node, future, timeout_sec=1.0)
                if not future.done() or future.result() is None:
                    states[name] = {"id": None, "label": "request_timeout"}
                    continue
                current = future.result().current_state
                states[name] = {"id": int(current.id), "label": str(current.label)}
            if set(states) == set(NODES) and all(
                acceptable(name, value) for name, value in states.items()
            ):
                break
            time.sleep(0.1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    passing = set(states) == set(NODES) and all(
        acceptable(name, value) for name, value in states.items()
    )
    inactive_deviations = sorted(
        name
        for name, value in states.items()
        if name in allowed_inactive and value.get("id") == 2
    )
    payload = {
        "schema_version": 1,
        "status": "PASS" if passing else "FAIL",
        "required_state_id": 3,
        "allowed_inactive_nodes": sorted(allowed_inactive),
        "inactive_deviations": inactive_deviations,
        "namespace": namespace,
        "resolved_nodes": resolved_nodes,
        "states": states,
        "duration_sec": time.monotonic() - started,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if passing else 2


if __name__ == "__main__":
    raise SystemExit(main())
