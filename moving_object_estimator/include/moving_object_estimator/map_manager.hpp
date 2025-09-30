#pragma once
#include <deque>
#include <vector>
#include <memory>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <geometry_msgs/msg/point_stamped.hpp>

class MapManager {
public:
  using PointXYZ = pcl::PointXYZ;
  using CloudT = pcl::PointCloud<pcl::PointXYZ>;
  using CloudPtr = CloudT::Ptr;
  using CloudConstPtr = CloudT::ConstPtr;

  explicit MapManager(size_t max_deque_size = 10)
    : max_deque_size_(max_deque_size) {}

  void set_max_deque_size(size_t n) { max_deque_size_ = (n==0?1:n); }
  size_t max_deque_size() const { return max_deque_size_; }

  size_t size() const { return deque_.size(); }
  bool empty() const { return deque_.empty(); }

  // PointCloud2 -> PCL 변환하여 deque에 push(고정 크기 유지)
  void addCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &ros_cloud);

  void addObj(const geometry_msgs::msg::Point &point);

  // 현재 deque를 스냅샷으로 반환(SharedPtr 복사 → 가볍다)
  std::vector<CloudConstPtr> snapshot() const;
  std::vector<PointXYZ> snapshot_obj() const;

private:
  size_t max_deque_size_;
  std::deque<CloudPtr> deque_;
  std::deque<PointXYZ> obj_deque_;

};
