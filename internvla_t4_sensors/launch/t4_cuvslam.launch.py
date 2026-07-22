"""Release-3.2 cuVSLAM stereo odometry for the calibrated Isaac Go2 rig."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    use_sim_time = LaunchConfiguration("use_sim_time")
    node_namespace = LaunchConfiguration("namespace")
    use_sim_time_parameter = ParameterValue(use_sim_time, value_type=bool)
    visual_slam = ComposableNode(
        package="isaac_ros_visual_slam",
        plugin="nvidia::isaac_ros::visual_slam::VisualSlamNode",
        name="visual_slam_node",
        namespace=node_namespace,
        parameters=[
            {
                "use_sim_time": use_sim_time_parameter,
                "num_cameras": 2,
                "min_num_images": 2,
                "multicam_mode": 1,
                "enable_imu_fusion": False,
                "enable_image_denoising": False,
                "rectified_images": True,
                "enable_ground_constraint_in_odometry": True,
                "enable_ground_constraint_in_slam": True,
                "enable_localization_n_mapping": False,
                "sync_matching_threshold_ms": 8.0,
                "image_jitter_threshold_ms": 40.0,
                "camera_optical_frames": [
                    "go2_stereo_left_optical_frame",
                    "go2_stereo_right_optical_frame",
                ],
                "base_frame": "base_link",
                "map_frame": "cuvslam_map",
                "odom_frame": "cuvslam_odom",
                # T4SensorBridge is the sole owner of the navigation TF chain.
                "publish_map_to_odom_tf": False,
                "publish_odom_to_base_tf": False,
                "override_publishing_stamp": False,
                "enable_slam_visualization": False,
                "enable_observations_view": False,
                "enable_landmarks_view": False,
                "verbosity": 1,
            }
        ],
        remappings=[
            ("visual_slam/image_0", "/go2/stereo/left/image_rect"),
            ("visual_slam/camera_info_0", "/go2/stereo/left/camera_info"),
            ("visual_slam/image_1", "/go2/stereo/right/image_rect"),
            ("visual_slam/camera_info_1", "/go2/stereo/right/camera_info"),
            # tf2 broadcasters use rooted defaults.  Resolve their outputs
            # inside the explicit Lane namespace without changing the empty
            # namespace used by frozen T4.
            ("/tf", "tf"),
            ("/tf_static", "tf_static"),
        ],
    )
    container = ComposableNodeContainer(
        name="t4_cuvslam_container",
        namespace=node_namespace,
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[visual_slam],
        # cuVSLAM's short-form TransformListener constructor creates a hidden
        # process-local node.  Process-level remaps keep that listener off the
        # root TF topics as well as the composable node itself.
        remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
        output="screen",
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description=(
                    "Use the ROS simulation clock (T5 passes true explicitly)."
                ),
            ),
            DeclareLaunchArgument(
                "namespace",
                default_value="",
                description=(
                    "Optional T5 Lane namespace; empty preserves frozen T4."
                ),
            ),
            container,
        ]
    )
