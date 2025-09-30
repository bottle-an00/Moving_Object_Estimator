#pragma once
#include <vector>
#include <algorithm>
#include <cmath>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>

#include "moving_object_estimator/map_manager.hpp"

class DynamicObjectDetector {
public:
  using CloudT = pcl::PointCloud<pcl::PointXYZ>;
  using CloudPtr = CloudT::Ptr;
  using CloudConstPtr = CloudT::ConstPtr;

  DynamicObjectDetector(int knn_k = 5, double wall_dist_thresh = 0.03, double voxel_leaf = 0.03)
  : knn_k_(knn_k), wall_dist_thresh_(wall_dist_thresh), voxel_leaf_(voxel_leaf),
    accum_(new CloudT()), static_obj_(new CloudT()),tree_ready_(false),tree_ready_obj_(false) {}

  void set_knn_k(int k) { knn_k_ = std::max(1, k); }
  void set_wall_dist_thresh(double d) { wall_dist_thresh_ = d; }
  void set_voxel_leaf(double v) { voxel_leaf_ = std::max(1e-3, v); }

  int knn_k() const { return knn_k_; }
  double wall_dist_thresh() const { return wall_dist_thresh_; }
  double voxel_leaf() const { return voxel_leaf_; }

  // MapManager의 deque 스냅샷을 받아 누적→다운샘플→KDTree 재빌드
  void rebuild(const std::vector<CloudConstPtr> &deque_snapshot);
  void rebuild_obj(const std::vector<pcl::PointXYZ> &deque_snapshot);

  bool ready() const { return tree_ready_ && accum_ && !accum_->empty(); }
  bool ready_obj() const { return tree_ready_obj_ && static_obj_ && !static_obj_->empty(); }

  // 입력 포인트가 벽에 속하는가?
  // true  -> wall (드롭)
  // false -> dynamic(통과)
  bool isWall(const geometry_msgs::msg::Point &p) const;
  bool isStatic(const geometry_msgs::msg::Point &p ) const;

  void addStatic(const geometry_msgs::msg::Point &p);

  std::vector<pcl::PointXYZ> get_static_vec() const;
  size_t get_static_vec_size() const;

private:
  int knn_k_;
  double wall_dist_thresh_;
  double voxel_leaf_;

  CloudPtr accum_;
  CloudPtr static_obj_;
  std::vector<pcl::PointXYZ> static_vec_;

  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_;
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree_obj_;
  bool tree_ready_;
  bool tree_ready_obj_;
};
