from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(package="isaac_visual_debug_overlay", executable="visual_overlay_node", output="screen"),
            Node(package="isaac_visual_debug_overlay", executable="video_recorder_node", output="screen"),
        ]
    )
