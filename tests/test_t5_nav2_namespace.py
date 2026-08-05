from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_nav2_action_and_velocity_names_are_namespace_relative() -> None:
    adapter = (
        ROOT / "internvla_nav2_adapter/internvla_nav2_adapter/active_node.py"
    ).read_text(encoding="utf-8")
    relay = (ROOT / "t4_completion/map/warn_relay.py").read_text(
        encoding="utf-8"
    )
    assert 'NavigateToPose,\n            "navigate_to_pose"' in adapter
    assert 'FollowPath,\n            "follow_path"' in adapter
    assert 'Twist,\n            "cmd_vel_safe"' in adapter
    assert 'NavPath,\n            "plan"' in adapter
    assert 'NavigateToPose,\n            "/navigate_to_pose"' not in adapter
    assert 'self.create_publisher(Twist, "cmd_vel_safe", 10)' in relay
    assert 'self.create_subscription(Twist, "cmd_vel_nav", self.on_raw, 10)' in relay
    assert '"completion_sim/collision_monitor/warn_only_cmd_vel"' in relay


def test_collision_monitor_data_plane_is_namespace_relative() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/internnav_t5/nav2_static_lidar.yaml").read_text(
            encoding="utf-8"
        )
    )
    monitor = config["collision_monitor"]["ros__parameters"]
    assert monitor["cmd_vel_in_topic"] == "cmd_vel_nav"
    assert monitor["cmd_vel_out_topic"] == (
        "completion_sim/collision_monitor/warn_only_cmd_vel"
    )
    assert monitor["FootprintApproach"]["footprint_topic"] == (
        "local_costmap/published_footprint"
    )


def test_t5_nav2_copy_only_changes_namespaced_topics_and_gate_tolerances() -> None:
    frozen_t4 = yaml.safe_load(
        (ROOT / "configs/completion_sim/map/nav2_static_lidar.yaml").read_text(
            encoding="utf-8"
        )
    )
    t5 = yaml.safe_load(
        (ROOT / "configs/internnav_t5/nav2_static_lidar.yaml").read_text(
            encoding="utf-8"
        )
    )
    monitor = frozen_t4["collision_monitor"]["ros__parameters"]
    monitor["cmd_vel_in_topic"] = "cmd_vel_nav"
    monitor["cmd_vel_out_topic"] = (
        "completion_sim/collision_monitor/warn_only_cmd_vel"
    )
    monitor["FootprintApproach"]["footprint_topic"] = (
        "local_costmap/published_footprint"
    )
    controller = frozen_t4["controller_server"]["ros__parameters"]
    controller["general_goal_checker"]["xy_goal_tolerance"] = 0.03
    controller["general_goal_checker"]["yaw_goal_tolerance"] = 0.03
    controller["FollowPath"]["xy_goal_tolerance"] = 0.03
    frozen_t4["local_costmap"]["local_costmap"]["ros__parameters"][
        "inflation_layer"
    ]["inflation_radius"] = 0.25
    assert t5 == frozen_t4


def test_data_plane_probe_checks_servers_clients_and_topic_endpoints() -> None:
    probe = (ROOT / "scripts/check_t4_nav2_data_plane.py").read_text(
        encoding="utf-8"
    )
    for relative_name in (
        "navigate_to_pose",
        "follow_path",
        "cmd_vel_nav",
        "completion_sim/collision_monitor/warn_only_cmd_vel",
        "cmd_vel_safe",
        "local_costmap/published_footprint",
        '"tf"',
        '"tf_static"',
    ):
        assert relative_name in probe
    assert '"minimum_subscribers": 2' in probe
    assert '"internvla_nav2_active_adapter", adapter_namespace' in probe
    assert "from rclpy.action.graph import (" in probe
    assert "get_action_server_names_and_types_by_node(" in probe
    assert "get_action_client_names_and_types_by_node(" in probe
    assert "node.get_action_server_names_and_types" not in probe
    assert "node.get_action_client_names_and_types_by_node" not in probe
    assert '"navigate_to_pose": "bt_navigator"' in probe
    assert '"follow_path": "controller_server"' in probe
    assert '"action_servers_by_node": action_servers_by_node' in probe
    assert '"endpoints_in_lane_namespace"' in probe
    assert 'tf_buffer.lookup_transform(target, source, Time())' in probe
    assert '("map_to_odom", "map", "odom")' in probe
    assert '("odom_to_base_link", "odom", "base_link")' in probe
    assert '("map_to_base_link", "map", "base_link")' in probe
    assert '"internnav_t4_root_tf_audit_probe"' in probe
    assert "root_audit_node.get_publishers_info_by_topic" in probe


def test_empty_namespace_preserves_t4_resolved_names() -> None:
    probe = (ROOT / "scripts/check_t4_nav2_data_plane.py").read_text(
        encoding="utf-8"
    )
    assert 'return f"{namespace}/{relative_name}" if namespace else f"/{relative_name}"' in probe


def test_t5_direct_nodes_and_evaluators_share_namespaced_tf() -> None:
    onboard = (ROOT / "scripts/run_t4_dgx_onboard.sh").read_text(
        encoding="utf-8"
    )
    lane = (ROOT / "scripts/run_t5_dgx_lane.sh").read_text(encoding="utf-8")
    bridge = (
        ROOT / "internvla_go2_controller/internvla_go2_controller/bridge_node.py"
    ).read_text(encoding="utf-8")
    assert 'ROS_REMAP_ARGS=(-r "__ns:=$NODE_NAMESPACE" -r /tf:=tf -r /tf_static:=tf_static)' in onboard
    assert lane.count("-r /tf:=tf -r /tf_static:=tf_static") >= 2
    assert "nav2_data_plane_evaluator_ready.json" in lane
    assert 'self.create_subscription(Twist, "cmd_vel_nav"' in bridge
    assert 'self.create_subscription(Twist, "cmd_vel_safe"' in bridge
    assert 'OccupancyGrid, "local_costmap/costmap"' in bridge
    sensor_bridge = (
        ROOT / "internvla_t4_sensors/internvla_t4_sensors/sensor_bridge_node.py"
    ).read_text(encoding="utf-8")
    assert '"global_costmap/costmap"' in sensor_bridge
