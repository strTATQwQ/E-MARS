from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    mode = LaunchConfiguration("mode")
    benchmark_launch = PathJoinSubstitution([FindPackageShare("isaac_vln_benchmark"), "launch", "isaac_benchmark.launch.py"])
    return LaunchDescription(
        [
            DeclareLaunchArgument("mode", default_value="step_omninav_event"),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(benchmark_launch), launch_arguments={"mode": mode}.items()),
        ]
    )
