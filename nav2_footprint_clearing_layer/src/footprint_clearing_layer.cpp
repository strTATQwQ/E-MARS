#include "nav2_footprint_clearing_layer/footprint_clearing_layer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>

#include "nav2_costmap_2d/cost_values.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "tf2/time.hpp"

namespace nav2_footprint_clearing_layer
{

FootprintClearingLayer::FootprintClearingLayer()
{
  enabled_ = true;
  current_ = false;
}

void FootprintClearingLayer::onInitialize()
{
  node_shared_ = node_.lock();
  if (!node_shared_) {
    throw std::runtime_error("FootprintClearingLayer cannot lock lifecycle node");
  }
  declareParameter("enabled", rclcpp::ParameterValue(true));
  declareParameter("require_lidar", rclcpp::ParameterValue(true));
  declareParameter("require_depth", rclcpp::ParameterValue(true));
  declareParameter("footprint_radius_m", rclcpp::ParameterValue(0.32));
  declareParameter("startup_edge_m", rclcpp::ParameterValue(0.0));
  declareParameter("obstacle_min_height_m", rclcpp::ParameterValue(0.05));
  declareParameter("obstacle_max_height_m", rclcpp::ParameterValue(1.5));
  declareParameter("data_timeout_sec", rclcpp::ParameterValue(0.35));
  declareParameter("tf_timeout_sec", rclcpp::ParameterValue(0.20));
  node_shared_->get_parameter(name_ + ".enabled", enabled_);
  node_shared_->get_parameter(name_ + ".require_lidar", require_lidar_);
  node_shared_->get_parameter(name_ + ".require_depth", require_depth_);
  node_shared_->get_parameter(name_ + ".footprint_radius_m", footprint_radius_m_);
  node_shared_->get_parameter(name_ + ".startup_edge_m", startup_edge_m_);
  node_shared_->get_parameter(name_ + ".obstacle_min_height_m", obstacle_min_height_m_);
  node_shared_->get_parameter(name_ + ".obstacle_max_height_m", obstacle_max_height_m_);
  node_shared_->get_parameter(name_ + ".data_timeout_sec", data_timeout_sec_);
  node_shared_->get_parameter(name_ + ".tf_timeout_sec", tf_timeout_sec_);
  if (!enabled_ || !(require_lidar_ || require_depth_) ||
    footprint_radius_m_ != 0.32 || startup_edge_m_ < 0.0 || startup_edge_m_ > 0.03 ||
    obstacle_min_height_m_ >= obstacle_max_height_m_ || data_timeout_sec_ <= 0.0 ||
    tf_timeout_sec_ <= 0.0)
  {
    throw std::runtime_error("invalid or unsafe footprint-clearing parameters");
  }

  auto sensor_qos = rclcpp::SensorDataQoS().keep_last(4);
  rclcpp::SubscriptionOptions options;
  options.callback_group = callback_group_;
  lidar_subscription_ = node_shared_->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/go2/lidar/points_base", sensor_qos,
    std::bind(&FootprintClearingLayer::onLidar, this, std::placeholders::_1), options);
  depth_subscription_ = node_shared_->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/go2/depth/points", sensor_qos,
    std::bind(&FootprintClearingLayer::onDepth, this, std::placeholders::_1), options);
  pose_subscription_ = node_shared_->create_subscription<std_msgs::msg::Bool>(
    "/internvla_t4/pose_valid", 10,
    std::bind(&FootprintClearingLayer::onPoseValid, this, std::placeholders::_1), options);
  collision_subscription_ = node_shared_->create_subscription<std_msgs::msg::Bool>(
    "/internvla_t4/collision_monitor_clear", 10,
    std::bind(&FootprintClearingLayer::onCollisionClear, this, std::placeholders::_1), options);
  reset_subscription_ = node_shared_->create_subscription<std_msgs::msg::Int32>(
    "/internvla_t4/map_reset_generation", 10,
    std::bind(&FootprintClearingLayer::onResetGeneration, this, std::placeholders::_1), options);
  status_publisher_ = node_shared_->create_publisher<std_msgs::msg::String>(
    name_ + "/status", rclcpp::QoS(10));
  service_ = node_shared_->create_service<nav2_msgs::srv::GetCostmap>(
    "get_" + name_,
    std::bind(
      &FootprintClearingLayer::serveCostmap, this,
      std::placeholders::_1, std::placeholders::_2),
    rmw_qos_profile_services_default, callback_group_);
  reset_time_ = node_shared_->now();
  reset();
}

void FootprintClearingLayer::activate()
{
  if (status_publisher_) {
    status_publisher_->on_activate();
  }
}

void FootprintClearingLayer::deactivate()
{
  if (status_publisher_) {
    status_publisher_->on_deactivate();
  }
}

void FootprintClearingLayer::reset()
{
  std::lock_guard<std::mutex> guard(mutex_);
  pose_valid_ = false;
  collision_clear_ = false;
  lidar_obstacle_ = true;
  depth_obstacle_ = true;
  have_previous_ = false;
  current_ = false;
  if (node_shared_) {
    reset_time_ = node_shared_->now();
  }
}

void FootprintClearingLayer::updateBounds(
  double robot_x, double robot_y, double,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  std::lock_guard<std::mutex> guard(mutex_);
  robot_x_ = robot_x;
  robot_y_ = robot_y;
  have_robot_pose_ = std::isfinite(robot_x) && std::isfinite(robot_y);
  const double radius = footprint_radius_m_ + startup_edge_m_ + 0.02;
  if (have_robot_pose_) {
    *min_x = std::min(*min_x, robot_x_ - radius);
    *min_y = std::min(*min_y, robot_y_ - radius);
    *max_x = std::max(*max_x, robot_x_ + radius);
    *max_y = std::max(*max_y, robot_y_ + radius);
  }
  if (have_previous_) {
    *min_x = std::min(*min_x, previous_x_ - radius);
    *min_y = std::min(*min_y, previous_y_ - radius);
    *max_x = std::max(*max_x, previous_x_ + radius);
    *max_y = std::max(*max_y, previous_y_ + radius);
  }
}

bool FootprintClearingLayer::cloudHasFootprintObstacle(
  const sensor_msgs::msg::PointCloud2 & message) const
{
  int x_offset = -1;
  int y_offset = -1;
  int z_offset = -1;
  for (const auto & field : message.fields) {
    if (field.name == "x") {x_offset = static_cast<int>(field.offset);}
    if (field.name == "y") {y_offset = static_cast<int>(field.offset);}
    if (field.name == "z") {z_offset = static_cast<int>(field.offset);}
  }
  if (message.is_bigendian || x_offset < 0 || y_offset < 0 || z_offset < 0 ||
    message.point_step < 12)
  {
    return true;
  }
  const size_t count = static_cast<size_t>(message.width) * message.height;
  const double radius_squared = footprint_radius_m_ * footprint_radius_m_;
  for (size_t index = 0; index < count; ++index) {
    const size_t start = index * message.point_step;
    if (start + static_cast<size_t>(std::max({x_offset, y_offset, z_offset})) + sizeof(float) >
      message.data.size())
    {
      return true;
    }
    float x;
    float y;
    float z;
    std::memcpy(&x, message.data.data() + start + x_offset, sizeof(float));
    std::memcpy(&y, message.data.data() + start + y_offset, sizeof(float));
    std::memcpy(&z, message.data.data() + start + z_offset, sizeof(float));
    if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z) &&
      z >= obstacle_min_height_m_ && z <= obstacle_max_height_m_ &&
      static_cast<double>(x) * x + static_cast<double>(y) * y <= radius_squared)
    {
      return true;
    }
  }
  return false;
}

void FootprintClearingLayer::onLidar(
  const sensor_msgs::msg::PointCloud2::SharedPtr message)
{
  std::lock_guard<std::mutex> guard(mutex_);
  lidar_obstacle_ = cloudHasFootprintObstacle(*message);
  lidar_time_ = node_shared_->now();
}

void FootprintClearingLayer::onDepth(
  const sensor_msgs::msg::PointCloud2::SharedPtr message)
{
  std::lock_guard<std::mutex> guard(mutex_);
  depth_obstacle_ = cloudHasFootprintObstacle(*message);
  depth_time_ = node_shared_->now();
}

void FootprintClearingLayer::onPoseValid(const std_msgs::msg::Bool::SharedPtr message)
{
  std::lock_guard<std::mutex> guard(mutex_);
  pose_valid_ = message->data;
  pose_time_ = node_shared_->now();
}

void FootprintClearingLayer::onCollisionClear(const std_msgs::msg::Bool::SharedPtr message)
{
  std::lock_guard<std::mutex> guard(mutex_);
  collision_clear_ = message->data;
  collision_time_ = node_shared_->now();
}

void FootprintClearingLayer::onResetGeneration(const std_msgs::msg::Int32::SharedPtr message)
{
  std::lock_guard<std::mutex> guard(mutex_);
  if (message->data <= reset_generation_) {
    return;
  }
  reset_generation_ = message->data;
  reset_time_ = node_shared_->now();
  pose_valid_ = false;
  collision_clear_ = false;
  lidar_obstacle_ = true;
  depth_obstacle_ = true;
  have_previous_ = false;
  current_ = false;
}

bool FootprintClearingLayer::safeToClear(std::string & reason, rclcpp::Time now)
{
  const auto fresh = [now, this](const rclcpp::Time & stamp) {
      return stamp > reset_time_ && (now - stamp).seconds() <= data_timeout_sec_;
    };
  if (!have_robot_pose_) {reason = "robot_pose_missing"; return false;}
  if (!pose_valid_ || !fresh(pose_time_)) {reason = "pose_invalid_or_stale"; return false;}
  if (!collision_clear_ || !fresh(collision_time_)) {
    reason = "collision_monitor_alarm_or_stale"; return false;
  }
  if (require_lidar_ && (!fresh(lidar_time_) || lidar_obstacle_)) {
    reason = lidar_obstacle_ ? "lidar_obstacle_in_footprint" : "lidar_stale"; return false;
  }
  if (require_depth_ && (!fresh(depth_time_) || depth_obstacle_)) {
    reason = depth_obstacle_ ? "depth_obstacle_in_footprint" : "depth_stale"; return false;
  }
  try {
    const auto transform = tf_->lookupTransform(
      layered_costmap_->getGlobalFrameID(), "base_link", tf2::TimePointZero);
    const rclcpp::Time transform_time(transform.header.stamp, now.get_clock_type());
    if ((now - transform_time).seconds() > tf_timeout_sec_) {
      reason = "tf_stale";
      return false;
    }
  } catch (const std::exception &) {
    reason = "tf_unavailable";
    return false;
  }
  reason = "verified";
  return true;
}

void FootprintClearingLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master_grid,
  int min_i, int min_j, int max_i, int max_j)
{
  std::lock_guard<std::mutex> guard(mutex_);
  ++update_count_;
  const auto now = node_shared_->now();
  std::string reason;
  const bool safe = safeToClear(reason, now);
  const double radius = footprint_radius_m_ + startup_edge_m_;
  const double radius_squared = radius * radius;
  size_t restored = 0;
  size_t cleared = 0;
  for (int my = min_j; my < max_j; ++my) {
    for (int mx = min_i; mx < max_i; ++mx) {
      double wx;
      double wy;
      master_grid.mapToWorld(static_cast<unsigned int>(mx), static_cast<unsigned int>(my), wx, wy);
      const bool in_current =
        have_robot_pose_ && (wx - robot_x_) * (wx - robot_x_) +
        (wy - robot_y_) * (wy - robot_y_) <= radius_squared;
      const bool in_previous =
        have_previous_ && (wx - previous_x_) * (wx - previous_x_) +
        (wy - previous_y_) * (wy - previous_y_) <= radius_squared;
      auto cost = master_grid.getCost(static_cast<unsigned int>(mx), static_cast<unsigned int>(my));
      if (in_previous && !in_current && cost == nav2_costmap_2d::FREE_SPACE) {
        master_grid.setCost(
          static_cast<unsigned int>(mx), static_cast<unsigned int>(my),
          nav2_costmap_2d::NO_INFORMATION);
        ++restored;
        cost = nav2_costmap_2d::NO_INFORMATION;
      }
      if (safe && in_current && cost == nav2_costmap_2d::NO_INFORMATION) {
        master_grid.setCost(
          static_cast<unsigned int>(mx), static_cast<unsigned int>(my),
          nav2_costmap_2d::FREE_SPACE);
        ++cleared;
      }
    }
  }
  if (safe) {
    ++cleared_update_count_;
    previous_x_ = robot_x_;
    previous_y_ = robot_y_;
    have_previous_ = true;
  } else {
    ++denied_update_count_;
  }
  current_ = safe;

  last_snapshot_.assign(
    master_grid.getCharMap(),
    master_grid.getCharMap() + master_grid.getSizeInCellsX() * master_grid.getSizeInCellsY());
  last_header_.stamp = now;
  last_header_.frame_id = layered_costmap_->getGlobalFrameID();
  last_metadata_.map_load_time = now;
  last_metadata_.update_time = now;
  last_metadata_.layer = name_;
  last_metadata_.resolution = master_grid.getResolution();
  last_metadata_.size_x = master_grid.getSizeInCellsX();
  last_metadata_.size_y = master_grid.getSizeInCellsY();
  last_metadata_.origin.position.x = master_grid.getOriginX();
  last_metadata_.origin.position.y = master_grid.getOriginY();
  last_metadata_.origin.orientation.w = 1.0;
  publishStatus(reason, safe, cleared, restored);
}

void FootprintClearingLayer::publishStatus(
  const std::string & reason, bool cleared, size_t cells, size_t restored)
{
  if (!status_publisher_ || !status_publisher_->is_activated()) {
    return;
  }
  std::ostringstream stream;
  stream << "{\"schema_version\":1,\"reset_generation\":" << reset_generation_
         << ",\"update_count\":" << update_count_
         << ",\"cleared\":" << (cleared ? "true" : "false")
         << ",\"cleared_cells\":" << cells
         << ",\"restored_previous_cells\":" << restored
         << ",\"reason\":\"" << reason << "\""
         << ",\"pose_valid\":" << (pose_valid_ ? "true" : "false")
         << ",\"collision_monitor_clear\":" << (collision_clear_ ? "true" : "false")
         << ",\"lidar_obstacle\":" << (lidar_obstacle_ ? "true" : "false")
         << ",\"depth_obstacle\":" << (depth_obstacle_ ? "true" : "false")
         << ",\"footprint_radius_m\":" << footprint_radius_m_
         << ",\"startup_edge_m\":" << startup_edge_m_ << "}";
  std_msgs::msg::String message;
  message.data = stream.str();
  status_publisher_->publish(message);
}

void FootprintClearingLayer::serveCostmap(
  const std::shared_ptr<nav2_msgs::srv::GetCostmap::Request>,
  std::shared_ptr<nav2_msgs::srv::GetCostmap::Response> response)
{
  std::lock_guard<std::mutex> guard(mutex_);
  response->map.header = last_header_;
  response->map.metadata = last_metadata_;
  response->map.data = last_snapshot_;
}

}  // namespace nav2_footprint_clearing_layer

PLUGINLIB_EXPORT_CLASS(
  nav2_footprint_clearing_layer::FootprintClearingLayer,
  nav2_costmap_2d::Layer)
