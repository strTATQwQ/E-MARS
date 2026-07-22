from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    delay_sec = LaunchConfiguration("delay_sec")
    drop_rate = LaunchConfiguration("drop_rate")
    return LaunchDescription(
        [
            DeclareLaunchArgument("delay_sec", default_value="2.0"),
            DeclareLaunchArgument("drop_rate", default_value="0.0"),
            Node(
                package="isaac_vln_benchmark",
                executable="delay_injector_node",
                parameters=[
                    {
                        "input_topic": "/step/response_json",
                        "output_topic": "/delayed/step/response_json",
                        "delay_sec": delay_sec,
                        "drop_rate": drop_rate,
                    }
                ],
                output="screen",
            ),
            Node(
                package="isaac_vln_benchmark",
                executable="delay_injector_node",
                name="omninav_delay_injector",
                parameters=[
                    {
                        "input_topic": "/omninav/action_candidate_json",
                        "output_topic": "/delayed/omninav/action_candidate_json",
                        "delay_sec": delay_sec,
                        "drop_rate": drop_rate,
                    }
                ],
                output="screen",
            ),
        ]
    )
