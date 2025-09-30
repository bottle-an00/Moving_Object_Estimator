#include "moving_object_estimator/dynamic_obstacle_detector.hpp"

void DynamicObjectDetector::rebuild(const std::vector<CloudConstPtr> &deque_snapshot) {
  CloudPtr merged(new CloudT());
  size_t total = 0;
  for (auto &c : deque_snapshot) total += c->size();
  merged->reserve(total);
  for (auto &c : deque_snapshot) *merged += *c;

  CloudPtr ds(new CloudT());
  if (!merged->empty()) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(merged);
    vg.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
    vg.filter(*ds);
  }

  accum_.swap(ds); 
  if (accum_ && !accum_->empty()) {
    kdtree_.setInputCloud(accum_);
    tree_ready_ = true;
  } else {
    tree_ready_ = false;
  }
}

void DynamicObjectDetector::rebuild_obj(const std::vector<pcl::PointXYZ> &deque_snapshot) {
  CloudPtr merged(new CloudT());
  size_t total = deque_snapshot.size();

  merged->reserve(total);
  for (auto &c : deque_snapshot) merged->push_back(c);

  CloudPtr ds(new CloudT());
  if (!merged->empty()) {
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(merged);
    vg.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
    vg.filter(*ds);
  }

  static_obj_.swap(ds); // static_obj_ <- ds
  if (static_obj_ && !static_obj_->empty()) {
    kdtree_obj_.setInputCloud(static_obj_);
    tree_ready_obj_ = true;
  } else {
    tree_ready_obj_ = false;
  }
}

bool DynamicObjectDetector::isWall(const geometry_msgs::msg::Point &p) const {
  if (!ready()) return false; 

  pcl::PointXYZ q;
  q.x = static_cast<float>(p.x);
  q.y = static_cast<float>(p.y);
  q.z = static_cast<float>(p.z);

  std::vector<int> knn_idx(knn_k_);
  std::vector<float> knn_sq(knn_k_);
  int found_knn = kdtree_.nearestKSearch(q, knn_k_, knn_idx, knn_sq);

  bool knn_check = false;
  if (found_knn > 0) {
    float min_sqr = *std::min_element(knn_sq.begin(), knn_sq.begin() + found_knn);
    double dmin = std::sqrt(min_sqr);
    knn_check = (dmin < wall_dist_thresh_);
  }

  std::vector<int> rad_idx;
  std::vector<float> rad_sq;
  int found_rad = kdtree_.radiusSearch(q, wall_dist_thresh_, rad_idx, rad_sq);

  bool radius_check = (found_rad > 0); 

  return knn_check || radius_check;
}

bool DynamicObjectDetector::isStatic(const geometry_msgs::msg::Point &p) const {
  if (!ready_obj()) return false; 

  pcl::PointXYZ q;
  q.x = static_cast<float>(p.x);
  q.y = static_cast<float>(p.y);
  q.z = static_cast<float>(p.z);

  std::vector<int> knn_idx(knn_k_);
  std::vector<float> knn_sq(knn_k_);
  int found_knn = kdtree_obj_.nearestKSearch(q, knn_k_, knn_idx, knn_sq);

  bool knn_check = false;
  if (found_knn > 0) {
    float min_sqr = *std::min_element(knn_sq.begin(), knn_sq.begin() + found_knn);
    double dmin = std::sqrt(min_sqr);
    knn_check = (dmin < wall_dist_thresh_);
  }

  std::vector<int> rad_idx;
  std::vector<float> rad_sq;
  int found_rad = kdtree_obj_.radiusSearch(q, wall_dist_thresh_, rad_idx, rad_sq);

  bool radius_check = (found_rad > 0); 

  return knn_check || radius_check;
}

void DynamicObjectDetector::addStatic(const geometry_msgs::msg::Point &p){
  pcl::PointXYZ pt;
  pt.x = static_cast<float>(p.x);
  pt.y = static_cast<float>(p.y);
  pt.z = 0.0;

  static_vec_.push_back(pt);
}

std::vector<pcl::PointXYZ> DynamicObjectDetector::get_static_vec() const {
  return static_vec_;
}

size_t DynamicObjectDetector::get_static_vec_size() const {
  return static_vec_.size();
}