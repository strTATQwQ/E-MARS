#!/usr/bin/env python3
"""Fail closed unless the Nav2 command/safety data plane is actually wired."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import rclpy
from rclpy.action.graph import (
    get_action_client_names_and_types_by_node,
    get_action_server_names_and_types_by_node,
)
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener


ACTIONS = {
    "navigate_to_pose": "nav2_msgs/action/NavigateToPose",
    "follow_path": "nav2_msgs/action/FollowPath",
}
ACTION_SERVER_NODES = {
    "navigate_to_pose": "bt_navigator",
    "follow_path": "controller_server",
}
TOPICS = {
    "cmd_vel_nav": {
        "type": "geometry_msgs/msg/Twist",
        "minimum_publishers": 1,
        "minimum_subscribers": 2,
    },
    "completion_sim/collision_monitor/warn_only_cmd_vel": {
        "type": "geometry_msgs/msg/Twist",
        "minimum_publishers": 1,
        "minimum_subscribers": 1,
    },
    "cmd_vel_safe": {
        "type": "geometry_msgs/msg/Twist",
        "minimum_publishers": 1,
        "minimum_subscribers": 2,
    },
    "local_costmap/published_footprint": {
        "type": "geometry_msgs/msg/PolygonStamped",
        "minimum_publishers": 1,
        "minimum_subscribers": 1,
    },
    "tf": {
        "type": "tf2_msgs/msg/TFMessage",
        "minimum_publishers": 1,
        "minimum_subscribers": 1,
    },
    "tf_static": {
        "type": "tf2_msgs/msg/TFMessage",
        "minimum_publishers": 1,
        "minimum_subscribers": 1,
    },
}


def _resolved(namespace: str, relative_name: str) -> str:
    return f"{namespace}/{relative_name}" if namespace else f"/{relative_name}"


def _endpoint_snapshot(endpoint: object) -> dict[str, str]:
    return {
        "node_name": str(endpoint.node_name),
        "node_namespace": str(endpoint.node_namespace),
        "topic_type": str(endpoint.topic_type),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--namespace", default="")
    args = parser.parse_args()
    if not 1.0 <= args.timeout_sec <= 30.0:
        raise SystemExit("timeout must be between 1 and 30 seconds")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    namespace = args.namespace.rstrip("/")
    if namespace and (not namespace.startswith("/") or "//" in namespace):
        raise SystemExit("namespace must be empty or an absolute ROS namespace")

    expected_actions = {
        _resolved(namespace, name): action_type for name, action_type in ACTIONS.items()
    }
    expected_topics = {
        _resolved(namespace, name): contract for name, contract in TOPICS.items()
    }
    adapter_namespace = namespace or "/"
    started = time.monotonic()
    deadline = started + args.timeout_sec
    last: dict[str, object] = {}

    rclpy.init()
    node_cli_args: list[str] = []
    if namespace:
        node_cli_args = [
            "--ros-args",
            "-r",
            "/tf:=tf",
            "-r",
            "/tf_static:=tf_static",
        ]
    node = Node(
        "internnav_t4_nav2_data_plane_probe",
        namespace=namespace or None,
        cli_args=node_cli_args,
        use_global_arguments=False,
    )
    root_audit_node = None
    if namespace:
        # Keep this graph-only node free of the /tf remap so querying the
        # forbidden root names cannot silently resolve back into the lane.
        root_audit_node = Node(
            "internnav_t4_root_tf_audit_probe",
            namespace="/",
            cli_args=[],
            use_global_arguments=False,
        )
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node, spin_thread=False)
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            action_servers_by_node: dict[str, dict[str, list[str]]] = {}
            for server_node in sorted(set(ACTION_SERVER_NODES.values())):
                try:
                    action_servers_by_node[server_node] = {
                        name: sorted(types)
                        for name, types in get_action_server_names_and_types_by_node(
                            node, server_node, adapter_namespace
                        )
                    }
                except Exception as exc:  # node may not yet be visible in the graph
                    action_servers_by_node[server_node] = {
                        "__graph_error__": [type(exc).__name__]
                    }
            action_servers = {
                name: types
                for snapshot in action_servers_by_node.values()
                for name, types in snapshot.items()
                if name != "__graph_error__"
            }
            try:
                action_clients = {
                    name: sorted(types)
                    for name, types in get_action_client_names_and_types_by_node(
                        node, "internvla_nav2_active_adapter", adapter_namespace
                    )
                }
            except Exception as exc:  # node may not yet be visible in the graph
                action_clients = {"__graph_error__": [type(exc).__name__]}

            topic_state: dict[str, object] = {}
            topics_pass = True
            for name, contract in expected_topics.items():
                publishers = node.get_publishers_info_by_topic(name)
                subscribers = node.get_subscriptions_info_by_topic(name)
                publisher_types = {str(item.topic_type) for item in publishers}
                subscriber_types = {str(item.topic_type) for item in subscribers}
                lane_prefix = namespace + "/" if namespace else "/"
                endpoints_in_lane = all(
                    str(item.node_namespace) == (namespace or "/")
                    or str(item.node_namespace).startswith(lane_prefix)
                    for item in [*publishers, *subscribers]
                )
                passing = (
                    len(publishers) >= int(contract["minimum_publishers"])
                    and len(subscribers) >= int(contract["minimum_subscribers"])
                    and publisher_types == {contract["type"]}
                    and subscriber_types == {contract["type"]}
                    and endpoints_in_lane
                )
                topics_pass = topics_pass and passing
                topic_state[name] = {
                    **contract,
                    "publishers": [_endpoint_snapshot(item) for item in publishers],
                    "subscribers": [_endpoint_snapshot(item) for item in subscribers],
                    "endpoints_in_lane_namespace": endpoints_in_lane,
                    "status": "PASS" if passing else "FAIL",
                }

            servers_pass = all(
                action_servers_by_node.get(
                    ACTION_SERVER_NODES[relative_name], {}
                ).get(_resolved(namespace, relative_name))
                == [action_type]
                for relative_name, action_type in ACTIONS.items()
            )
            clients_pass = all(
                action_clients.get(name) == [action_type]
                for name, action_type in expected_actions.items()
            )
            transform_state: dict[str, object] = {}
            transforms_pass = True
            for label, target, source in (
                ("map_to_odom", "map", "odom"),
                ("odom_to_base_link", "odom", "base_link"),
                ("map_to_base_link", "map", "base_link"),
            ):
                try:
                    transform = tf_buffer.lookup_transform(target, source, Time())
                    transform_state[label] = {
                        "status": "PASS",
                        "target_frame": target,
                        "source_frame": source,
                        "resolved_parent": transform.header.frame_id,
                        "resolved_child": transform.child_frame_id,
                    }
                except Exception as exc:
                    transforms_pass = False
                    transform_state[label] = {
                        "status": "FAIL",
                        "target_frame": target,
                        "source_frame": source,
                        "error": f"{type(exc).__name__}: {exc}"[:512],
                    }

            forbidden_root_topics: dict[str, object] = {}
            root_tf_isolated = True
            if namespace:
                assert root_audit_node is not None
                for root_name in ("/tf", "/tf_static"):
                    root_publishers = root_audit_node.get_publishers_info_by_topic(
                        root_name
                    )
                    root_subscribers = root_audit_node.get_subscriptions_info_by_topic(
                        root_name
                    )
                    empty = not root_publishers and not root_subscribers
                    root_tf_isolated = root_tf_isolated and empty
                    forbidden_root_topics[root_name] = {
                        "status": "PASS" if empty else "FAIL",
                        "publishers": [
                            _endpoint_snapshot(item) for item in root_publishers
                        ],
                        "subscribers": [
                            _endpoint_snapshot(item) for item in root_subscribers
                        ],
                    }
            last = {
                "action_servers": action_servers,
                "action_servers_by_node": action_servers_by_node,
                "adapter_action_clients": action_clients,
                "topics": topic_state,
                "action_servers_pass": servers_pass,
                "adapter_action_clients_pass": clients_pass,
                "topics_pass": topics_pass,
                "transforms": transform_state,
                "transforms_pass": transforms_pass,
                "forbidden_root_tf_topics": forbidden_root_topics,
                "root_tf_isolated": root_tf_isolated,
            }
            if (
                servers_pass
                and clients_pass
                and topics_pass
                and transforms_pass
                and root_tf_isolated
            ):
                break
            time.sleep(0.1)
    finally:
        if root_audit_node is not None:
            root_audit_node.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    passing = bool(
        last.get("action_servers_pass")
        and last.get("adapter_action_clients_pass")
        and last.get("topics_pass")
        and last.get("transforms_pass")
        and last.get("root_tf_isolated")
    )
    payload = {
        "schema_version": 1,
        "status": "PASS" if passing else "FAIL",
        "namespace": namespace,
        "expected_actions": expected_actions,
        "expected_topics": expected_topics,
        "duration_sec": time.monotonic() - started,
        **last,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if passing else 2


if __name__ == "__main__":
    raise SystemExit(main())
