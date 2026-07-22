from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")
    enable_local_step_verifiers = LaunchConfiguration("enable_local_step_verifiers")
    default_config = PathJoinSubstitution([FindPackageShare("omninav_step_scheduler"), "config", "scheduler_default.yaml"])
    common_params = [{"config_file": config_file}]
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument("enable_local_step_verifiers", default_value="true"),
            Node(package="omninav_step_scheduler", executable="mission_manager_node", name="mission_manager", parameters=common_params),
            Node(package="omninav_step_scheduler", executable="step_supervisor_node", name="step_supervisor", parameters=common_params),
            Node(package="omninav_step_scheduler", executable="step_role_router_node", name="step_role_router", parameters=common_params),
            Node(package="omninav_step_scheduler", executable="route_choice_verifier_node", name="route_choice_verifier", parameters=common_params, condition=IfCondition(enable_local_step_verifiers)),
            Node(package="omninav_step_scheduler", executable="semantic_stop_verifier_node", name="semantic_stop_verifier", parameters=common_params, condition=IfCondition(enable_local_step_verifiers)),
            Node(package="omninav_step_scheduler", executable="visible_to_stop_monitor_node", name="visible_to_stop_monitor", parameters=common_params),
            Node(package="omninav_step_scheduler", executable="route_stop_primitive_bridge_node", name="route_stop_primitive_bridge", parameters=common_params),
            Node(package="omninav_step_scheduler", executable="omninav_scheduler_node", name="omninav_scheduler", parameters=common_params),
            Node(package="omninav_step_scheduler", executable="internnav_scheduler_node", name="internnav_scheduler", parameters=common_params),
            Node(package="omninav_step_scheduler", executable="step_pending_policy_node", name="step_pending_policy", parameters=common_params),
            Node(package="omninav_step_scheduler", executable="primitive_executor_node", name="primitive_executor", parameters=common_params),
            Node(package="omninav_step_scheduler", executable="sensor_only_planner_node", name="sensor_only_planner"),
            Node(package="omninav_step_scheduler", executable="semantic_executive_node", name="semantic_executive", parameters=common_params),
            Node(package="omninav_step_scheduler", executable="semantic_recovery_bridge_node", name="semantic_recovery_bridge", parameters=common_params),
            Node(package="omninav_step_scheduler", executable="safe_cmd_mux_node", name="safe_cmd_mux", parameters=[{"config_file": config_file, "dry_run": False}]),
            Node(package="omninav_step_scheduler", executable="metrics_logger_node", name="metrics_logger", parameters=common_params),
        ]
    )
