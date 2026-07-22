cmake_minimum_required(VERSION 3.8)
project(internnav_t5_lane_b_msgs)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(internvla_ros2_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/CandidateFrontier.msg"
  "srv/PrepareFrontiers.srv"
  "srv/CommitFrontier.srv"
  DEPENDENCIES builtin_interfaces internvla_ros2_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
