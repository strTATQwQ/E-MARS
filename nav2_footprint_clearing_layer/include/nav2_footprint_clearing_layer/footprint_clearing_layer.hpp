#ifndef NAV2_FOOTPRINT_CLEARING_LAYER__FOOTPRINT_CLEARING_LAYER_HPP_
#define NAV2_FOOTPRINT_CLEARING_LAYER__FOOTPRINT_CLEARING_LAYER_HPP_

#include <mutex>
#include <string>
#include <vector>

#include "nav2_costmap_2d/layer.hpp"
#include "nav2_msgs/srv/get_costmap.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/int32.hpp"
#include "std_msgs/msg/string.hpp"

namespace nav2_footprint_clearing_layer
{

class FootprintClearingLayer : public nav2_costmap_2d::Layer
{
public:
  FootprintClearingLayer();
  void onInitialize() override;
  void activate() override;
  void deactivate() override;
  void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y) override;
  void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;
  void reset() override;
  bool isClearable() override {return true;}

private:
  void onLidar(const sensor_msgs::msg::PointCloud2::SharedPtr message);
  void onDepth(const sensor_msgs::msg::PointCloud2::SharedPtr message);
  void onPoseValid(const std_msgs::msg::Bool::SharedPtr message);
  void onCollisionClear(const std_msgs::msg::Bool::SharedPtr message);
  void onResetGeneration(const std_msgs::msg::Int32::SharedPtr message);
  bool cloudHasFootprintObstacle(const sensor_msgs::msg::PointCloud2 & message) const;
  bool safeToClear(std::string & reason, rclcpp::Time now);
  void publishStatus(const std::string & reason, bool cleared, size_t cells, size_t restored);
  void serveCostmap(
    const std::shared_ptr<nav2_msgs::srv::GetCostmap::Request> request,
    std::shared_ptr<nav2_msgs::srv::GetCostmap::Response> response);

  nav2_util::LifecycleNode::SharedPtr node_shared_;
  std::mutex mutex_;
  double robot_x_{0.0};
  double robot_y_{0.0};
  double previous_x_{0.0};
  double previous_y_{0.0};
  bool have_robot_pose_{false};
  bool have_previous_{false};
  bool pose_valid_{false};
  bool collision_clear_{false};
  bool lidar_obstacle_{true};
  bool depth_obstacle_{true};
  bool require_lidar_{true};
  bool require_depth_{true};
  double footprint_radius_m_{0.32};
  double startup_edge_m_{0.0};
  double obstacle_min_height_m_{0.05};
  double obstacle_max_height_m_{1.5};
  double data_timeout_sec_{0.35};
  double tf_timeout_sec_{0.20};
  int reset_generation_{-1};
  rclcpp::Time reset_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time pose_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time collision_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time lidar_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time depth_time_{0, 0, RCL_ROS_TIME};
  uint64_t update_count_{0};
  uint64_t cleared_update_count_{0};
  uint64_t denied_update_count_{0};
  std::vector<unsigned char> last_snapshot_;
  nav2_msgs::msg::CostmapMetaData last_metadata_;
  std_msgs::msg::Header last_header_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr depth_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr pose_subscription_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr collision_subscription_;
  rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr reset_subscription_;
  rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::String>::SharedPtr status_publisher_;
  rclcpp::Service<nav2_msgs::srv::GetCostmap>::SharedPtr service_;
};

}  // namespace nav2_footprint_clearing_layer

#endif  // NAV2_FOOTPRINT_CLEARING_LAYER__FOOTPRINT_CLEARING_LAYER_HPP_
