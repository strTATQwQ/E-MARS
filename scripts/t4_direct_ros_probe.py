#!/usr/bin/env python3
"""Daemon-free ROS graph and lifecycle probes for isolated T4 trials."""

from __future__ import annotations

import argparse
import time

import rclpy
from lifecycle_msgs.srv import GetState


def join_name(namespace: str, name: str) -> str:
    if namespace == "/":
        return f"/{name}"
    return f"{namespace.rstrip('/')}/{name}"


def lifecycle(node: object, name: str, timeout: float, active: bool) -> int:
    service = f"{name.rstrip('/')}/get_state"
    client = node.create_client(GetState, service)
    if not client.wait_for_service(timeout_sec=timeout):
        print(f"unavailable {service}")
        return 1
    future = client.call_async(GetState.Request())
    rclpy.spin_until_future_complete(node, future, timeout_sec=timeout)
    if not future.done() or future.exception() is not None:
        print(f"no_response {service}")
        return 1
    label = str(future.result().current_state.label)
    print(label)
    return 0 if not active or label == "active" else 1


def format_types(name: str, types: list[str]) -> str:
    return f"{name} [{', '.join(types)}]"


def graph(node: object, timeout: float) -> int:
    deadline = time.monotonic() + timeout
    errors: list[str] = []
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.1, deadline - time.monotonic()))
    except Exception as exc:  # A mixed Humble/Jazzy graph can reject one sample.
        errors.append(f"spin:{type(exc).__name__}:{exc}")
    getters = (
        ("nodes", node.get_node_names_and_namespaces),
        ("topics", node.get_topic_names_and_types),
        ("services", node.get_service_names_and_types),
        (
            "actions",
            getattr(node, "get_action_names_and_types", lambda: []),
        ),
    )
    for label, getter in getters:
        print(f"# {label}")
        try:
            values = sorted(getter())
            for value in values:
                if label == "nodes":
                    name, namespace = value
                    print(join_name(namespace, name))
                else:
                    name, types = value
                    print(format_types(name, types))
        except Exception as exc:
            errors.append(f"{label}:{type(exc).__name__}:{exc}")
            print(f"# unavailable: {type(exc).__name__}")
    print("# graph_errors")
    for error in errors:
        print(error)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("lifecycle-responsive", "lifecycle-active", "graph")
    )
    parser.add_argument("node_name", nargs="?")
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()
    if args.timeout <= 0.0:
        raise SystemExit("timeout must be positive")
    if args.command.startswith("lifecycle-") and not args.node_name:
        raise SystemExit("node_name is required for lifecycle probes")
    rclpy.init()
    node = rclpy.create_node(
        "internvla_t4_direct_probe",
        namespace=f"/probe_{time.monotonic_ns()}",
        enable_rosout=False,
        start_parameter_services=False,
    )
    try:
        if args.command == "graph":
            code = graph(node, args.timeout)
        else:
            code = lifecycle(
                node,
                str(args.node_name),
                args.timeout,
                active=args.command == "lifecycle-active",
            )
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
