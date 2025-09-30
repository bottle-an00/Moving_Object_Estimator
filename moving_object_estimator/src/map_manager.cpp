#include "moving_object_estimator/map_manager.hpp"

void MapManager::addCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &ros_cloud) {
  CloudPtr cloud(new CloudT());
  pcl::fromROSMsg(*ros_cloud, *cloud);

  if (deque_.size() >= max_deque_size_) deque_.pop_front();
  deque_.push_back(cloud);
}

void MapManager::addObj(const geometry_msgs::msg::Point &point) {
  pcl::PointXYZ pt;
  pt.x = static_cast<float>(point.x);
  pt.y = static_cast<float>(point.y);
  pt.z = 0.0;

  if (obj_deque_.size() >= max_deque_size_) obj_deque_.pop_front();
  obj_deque_.push_back(pt);
}

std::vector<MapManager::CloudConstPtr> MapManager::snapshot() const {
  std::vector<CloudConstPtr> out;
  out.reserve(deque_.size());
  for (const auto &c : deque_) out.push_back(c);
  return out;
}

std::vector<MapManager::PointXYZ> MapManager::snapshot_obj() const {
  std::vector<PointXYZ> out;
  out.reserve(obj_deque_.size());
  for (const auto &c : obj_deque_) out.push_back(c);
  return out;
}
