from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    mode = LaunchConfiguration("mode")
    tasks_path = LaunchConfiguration("tasks_path")
    scenes_path = LaunchConfiguration("scenes_path")
    output_dir = LaunchConfiguration("output_dir")
    grounded_sam_endpoint = LaunchConfiguration("grounded_sam_endpoint")
    package_share = FindPackageShare("isaac_vln_benchmark")
    default_tasks_path = PathJoinSubstitution([package_share, "configs", "tasks.yaml"])
    default_scenes_path = PathJoinSubstitution([package_share, "configs", "scenes.yaml"])
    return LaunchDescription(
        [
            DeclareLaunchArgument("mode", default_value="step_omninav_event"),
            DeclareLaunchArgument("tasks_path", default_value=default_tasks_path),
            DeclareLaunchArgument("scenes_path", default_value=default_scenes_path),
            DeclareLaunchArgument("output_dir", default_value="runs/live_episode"),
            DeclareLaunchArgument(
                "grounded_sam_endpoint",
                default_value="http://10.100.100.128:8097/detect_segment",
            ),
            Node(
                package="isaac_vln_benchmark",
                executable="episode_manager_node",
                parameters=[{"mode": mode, "tasks_path": tasks_path, "scenes_path": scenes_path}],
                output="screen",
            ),
            Node(
                package="isaac_vln_benchmark",
                executable="go2_benchmark_adapter_node",
                parameters=[{"tasks_path": tasks_path, "scenes_path": scenes_path}],
                output="screen",
            ),
            Node(package="isaac_vln_benchmark", executable="semantic_oracle_node", output="screen"),
            Node(package="isaac_vln_benchmark", executable="obstacle_controller_node", output="screen"),
            Node(
                package="isaac_vln_benchmark",
                executable="grounded_sam_perception_node",
                parameters=[{"endpoint": grounded_sam_endpoint}],
                output="screen",
            ),
            Node(
                package="isaac_vln_benchmark",
                executable="benchmark_logger_node",
                parameters=[{"output_dir": output_dir}],
                output="screen",
            ),
        ]
    )
