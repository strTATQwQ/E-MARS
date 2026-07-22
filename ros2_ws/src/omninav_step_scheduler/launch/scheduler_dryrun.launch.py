from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")
    default_config = PathJoinSubstitution([FindPackageShare("omninav_step_scheduler"), "config", "scheduler_default.yaml"])
    common_params = [{"config_file": config_file}]
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value=default_config),
            Node(package="omninav_step_scheduler", executable="mission_manager_node", name="mission_manager", parameters=common_params),
            Node(package="omninav_step_scheduler", executable="step_supervisor_node", name="step_supervisor", parameters=common_params),
            Node(package="omninav_step_scheduler", executable="omninav_scheduler_node", name="omninav_scheduler", parameters=common_params),
            Node(package="omninav_step_scheduler", executable="internnav_scheduler_node", name="internnav_scheduler", parameters=common_params),
            Node(package="omninav_step_scheduler", executable="step_pending_policy_node", name="step_pending_policy", parameters=common_params),
            Node(package="omninav_step_scheduler", executable="primitive_executor_node", name="primitive_executor", parameters=common_params),
            Node(package="omninav_step_scheduler", executable="safe_cmd_mux_node", name="safe_cmd_mux", parameters=[{"config_file": config_file, "dry_run": True}]),
            Node(package="omninav_step_scheduler", executable="metrics_logger_node", name="metrics_logger", parameters=common_params),
            Node(package="omninav_step_scheduler", executable="mock_safety_status_node", name="mock_safety_status"),
            Node(package="omninav_step_scheduler", executable="mock_step_client_node", name="mock_step_client"),
            Node(package="omninav_step_scheduler", executable="mock_omninav_client_node", name="mock_omninav_client"),
        ]
    )
