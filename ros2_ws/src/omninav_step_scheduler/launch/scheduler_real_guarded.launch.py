import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _guarded_nodes(context):
    dry_run = LaunchConfiguration("dry_run").perform(context).lower()
    if os.environ.get("GO2_ENABLE_MOTION") != "1" or dry_run != "false":
        raise RuntimeError("Refusing real guarded launch: require GO2_ENABLE_MOTION=1 and dry_run:=false")
    print("\033[91mREAL GUARDED MODE ENABLED: still publishing only /safe_cmd_vel, not Unitree low-level topics.\033[0m")
    config_file = LaunchConfiguration("config_file")
    common_params = [{"config_file": config_file}]
    return [
        Node(package="omninav_step_scheduler", executable="mission_manager_node", name="mission_manager", parameters=common_params),
        Node(package="omninav_step_scheduler", executable="step_supervisor_node", name="step_supervisor", parameters=common_params),
        Node(package="omninav_step_scheduler", executable="omninav_scheduler_node", name="omninav_scheduler", parameters=common_params),
        Node(package="omninav_step_scheduler", executable="step_pending_policy_node", name="step_pending_policy", parameters=common_params),
        Node(package="omninav_step_scheduler", executable="primitive_executor_node", name="primitive_executor", parameters=common_params),
        Node(package="omninav_step_scheduler", executable="safe_cmd_mux_node", name="safe_cmd_mux", parameters=[{"config_file": config_file, "dry_run": False}]),
        Node(package="omninav_step_scheduler", executable="metrics_logger_node", name="metrics_logger", parameters=common_params),
    ]


def generate_launch_description():
    default_config = PathJoinSubstitution([FindPackageShare("omninav_step_scheduler"), "config", "scheduler_default.yaml"])
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument("dry_run", default_value="true"),
            OpaqueFunction(function=_guarded_nodes),
        ]
    )
